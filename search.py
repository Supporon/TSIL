"""Hyper-parameter search entry point for TSIL (Optuna / TPE).

Searches lr x layers x d_model x batch_size x tau for one or more backbones and
predicates, selecting the best configuration **by validation IC** (read from
`model.fit`'s `metrics['best_score']`, i.e. the IC of the best epoch on the
validation segment). Test-set metrics are never used to pick hyper-parameters,
avoiding leakage; they are reported only for the chosen configuration.

Usage
-----
    python search.py --config_file configs/f158_csi300/config_lstm.yaml \
        --models LSTM GRU --phi_types phi_momentum --market csi300 --n_trials 50

Requires `optuna` (see requirements.txt).
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.chdir(sys.path[0])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qlib
import ruamel.yaml as YAML
from qlib.utils import init_instance_by_config
from qlib.workflow import R

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.run_utils import (
    DEFAULT_SEARCH_SPACE, TAU_VALUES, ALL_MODELS,
    inject_model_specific_params, set_market,
)


def _suggest(trial, search_space):
    """Sample one hyper-parameter point from the search space."""
    hp = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                           log=spec.get("log", False))
    return hp


def _apply(config, hp, model_type, phi_type, tau, market):
    """Build a concrete config for one trial."""
    config = copy.deepcopy(config)
    if "lr" in hp:
        config["task"]["model"]["kwargs"]["lr"] = hp["lr"]
        config["model_config"]["lr"] = hp["lr"]
    if "layers" in hp:
        config["model_config"]["e_layers"] = hp["layers"]
    if "d_model" in hp:
        config["model_config"]["d_model"] = hp["d_model"]
        config["model_config"]["d_ff"] = hp["d_model"]
    if "bs" in hp:
        config["task"]["dataset"]["kwargs"]["batch_size"] = hp["bs"]

    config["task"]["model"]["kwargs"]["model_type"] = model_type
    config["model_config"]["phi_type"] = phi_type
    config["model_config"]["tau_hat_init"] = tau
    inject_model_specific_params(config, model_type)
    set_market(config, market)
    return config


def _train_one(config, model_type):
    """Train one config; return validation IC (best epoch) for selection.

    Selection metric is the validation IC, read from fit()'s best_score, so the
    test segment never influences hyper-parameter choice.
    """
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"],
              region=config["qlib_init"]["region"])
    exp_path = os.path.join(config["logdir"], "experiments")
    os.makedirs(exp_path, exist_ok=True)
    R.set_uri(exp_path)

    with R.start(experiment_name=f"{model_type}_search",
                 recorder_name=f"trial"):
        # phi_type lives in model_config; pass it to the dataset so it only
        # extracts timestamp features for the timestamp predicate.
        config["task"]["dataset"]["kwargs"]["phi_type"] = config["model_config"].get("phi_type")
        dataset = init_instance_by_config(config["task"]["dataset"])
        model = init_instance_by_config(config["task"]["model"])
        _, metrics = model.fit(dataset)
    # best_score is the validation IC at the best epoch (see QniverseModel.fit).
    return float(metrics.get("best_score", float("nan")))


def run_search(base_config, models, phi_types, tau_values, market,
               output_dir, search_space, n_trials, seed, early_stop):
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for search.py: pip install optuna plotly")

    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    combos = [(m, p, t) for m in models for p in phi_types for t in tau_values]
    print(f"#combinations={len(combos)} (models x phi_types x tau); "
          f"{n_trials} trials each")

    for ci, (model_type, phi_type, tau) in enumerate(combos, 1):
        combo_dir = os.path.join(output_dir, f"{model_type.lower()}_{phi_type}_tau{tau}")
        os.makedirs(combo_dir, exist_ok=True)
        history = []

        def objective(trial):
            hp = _suggest(trial, search_space)
            config = _apply(base_config, hp, model_type, phi_type, tau, market)
            config["logdir"] = combo_dir
            config["task"]["model"]["kwargs"]["logdir"] = combo_dir
            config["model_config"]["ind"] = trial.number
            try:
                valid_ic = _train_one(config, model_type)
            except Exception as e:
                print(f"  trial {trial.number} failed: {e}")
                return float("-inf")
            history.append({"trial": trial.number, **hp, "valid_ic": valid_ic})
            pd.DataFrame(history).to_csv(os.path.join(combo_dir, "trial_history.csv"),
                                         index=False)
            return valid_ic

        print(f"\n[{ci}/{len(combos)}] model={model_type} phi={phi_type} tau={tau}")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed, multivariate=True),
            pruner=MedianPruner(n_startup_trials=5),
            study_name=f"{model_type}_{phi_type}_tau{tau}",
        )

        # Early stop: stop the study once no improvement for `early_stop` trials.
        state = {"best": float("-inf"), "stale": 0}

        def _es(study, trial):
            v = trial.value if trial.value is not None else float("-inf")
            if v > state["best"]:
                state["best"], state["stale"] = v, 0
            else:
                state["stale"] += 1
            if state["stale"] >= early_stop:
                study.stop()

        study.optimize(objective, n_trials=n_trials, callbacks=[_es],
                       show_progress_bar=True)

        best = study.best_trial
        row = {"model_type": model_type, "phi_type": phi_type, "tau": tau,
               "best_valid_ic": study.best_value, **best.params}
        all_rows.append(row)
        with open(os.path.join(combo_dir, "optimization_report.txt"), "w") as f:
            f.write(f"model={model_type} phi={phi_type} tau={tau}\n")
            f.write(f"best valid IC: {study.best_value:.6f}\n")
            f.write(f"best params: {best.params}\n")
        _save_visualizations(study, combo_dir)

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(output_dir, "all_best_results.csv"), index=False)
    print("\nSearch complete. Best per combination:")
    print(results.to_string(index=False))
    return results


def _save_visualizations(study, out_dir):
    """Best-effort Optuna plots; skip silently if plotly is unavailable."""
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        vdir = os.path.join(out_dir, "visualizations")
        os.makedirs(vdir, exist_ok=True)
        plot_optimization_history(study).write_html(os.path.join(vdir, "history.html"))
        plot_param_importances(study).write_html(os.path.join(vdir, "param_importances.html"))
    except Exception as e:
        print(f"  (visualization skipped: {e})")


def build_argparser():
    p = argparse.ArgumentParser(allow_abbrev=False,
                                description="TSIL hyper-parameter search (Optuna/TPE)")
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", default=["LSTM"], choices=ALL_MODELS)
    p.add_argument("--phi_types", type=str, nargs="+", default=["mse"])
    p.add_argument("--tau_values", type=float, nargs="+", default=TAU_VALUES)
    p.add_argument("--market", type=str, default="csi300", choices=["csi300", "csi500", "csi800"])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--early_stop", type=int, default=10)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--output_dir", type=str, default="output_search")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not OPTUNA_AVAILABLE:
        print("Error: optuna not installed. Run: pip install optuna plotly")
        sys.exit(1)

    with open(args.config_file) as f:
        base_config = YAML.YAML(typ="safe", pure=True).load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "search_info.json"), "w") as f:
        json.dump({"models": args.models, "phi_types": args.phi_types,
                   "tau_values": args.tau_values, "market": args.market,
                   "n_trials": args.n_trials, "search_space": DEFAULT_SEARCH_SPACE}, f, indent=2)

    run_search(base_config, args.models, args.phi_types, args.tau_values,
               args.market, args.output_dir, DEFAULT_SEARCH_SPACE,
               args.n_trials, args.seed, args.early_stop)
