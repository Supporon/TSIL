"""Paper protocol: freeze the network, then select fixed beta by validation daily IC.

No candidate prepares or evaluates the test split. The saved manifest identifies
the exact checkpoints/configurations eligible for subsequent test evaluation.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import ruamel.yaml as YAML

from src.run_utils import (DEFAULT_SEARCH_SPACE, DEFAULT_SEEDS, PENALTY_ALPHA_VALUES,
                           ALL_MODELS, RELATIONAL_MODELS, configure_run, inject_model_specific_params, set_market)
from src.temporal_predicates import TEMPORAL_PHI_TYPES, selected_predicates


def network_search_space(model_type, space=None):
    space = copy.deepcopy(space or DEFAULT_SEARCH_SPACE)
    if model_type in RELATIONAL_MODELS:
        space.pop("bs", None)
        space.setdefault("days_per_batch", {"type": "categorical", "choices": [2, 4, 8]})
    return space


def _suggest(trial, space):
    point = {}
    for key, spec in space.items():
        if spec["type"] == "categorical":
            point[key] = trial.suggest_categorical(key, spec["choices"])
        elif spec["type"] == "int":
            point[key] = trial.suggest_int(key, spec["low"], spec["high"])
        else:
            point[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
    return point


def _apply(base, hp, model_type, phi_type, alpha, market):
    config = copy.deepcopy(base)
    if model_type != base["task"]["model"]["kwargs"]["model_type"]:
        # Architecture-specific fields come from that backbone's released YAML.
        path = Path(__file__).parent / "configs" / f"f158_{market}" / f"config_{model_type.lower()}.yaml"
        target = YAML.YAML(typ="safe").load(path.read_text(encoding="utf-8"))["model_config"]
        for key in ("seq_len", "enc_in", "pred_len", "c_out", "predicate_config", "injection_mode"):
            if key in config["model_config"]:
                target[key] = config["model_config"][key]
        config["model_config"] = target
    mc = config["model_config"]
    trainer = config["task"]["model"]["kwargs"]
    trainer["model_type"] = model_type
    trainer["model_config"] = mc
    if "lr" in hp:
        trainer["lr"] = hp["lr"]
        mc["lr"] = hp["lr"]
    if "layers" in hp:
        mc["e_layers"] = hp["layers"]
    if "d_model" in hp:
        mc["d_model"] = hp["d_model"]
        mc["d_ff"] = hp["d_model"]
    if "bs" in hp:
        if model_type in RELATIONAL_MODELS:
            raise ValueError("relation backbones search days_per_batch, not stock batch_size")
        config["task"]["dataset"]["kwargs"]["batch_size"] = hp["bs"]
    if "days_per_batch" in hp:
        config["task"]["dataset"]["kwargs"]["days_per_batch"] = hp["days_per_batch"]
    mc["phi_type"], mc["penalty_alpha"] = phi_type, float(alpha)
    mc["injection_mode"] = "original" if phi_type == "mse" else base["model_config"].get("injection_mode", "tsil")
    inject_model_specific_params(config, model_type)
    set_market(config, market)
    trainer["eval_test"] = False
    trainer["plot_training"] = False
    return config


def _train_one(config, model_type):
    import qlib
    from qlib.utils import init_instance_by_config
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"], region=config["qlib_init"]["region"], kernels=1)
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])
    _, metrics = model.fit(dataset, evaluate_test=False)
    return float(metrics["best_score"])


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def run_paper_selection(base_config, models, phi_types, alpha_values, market, output_dir,
                        seeds=DEFAULT_SEEDS, network_candidates=None, train_fn=None,
                        n_trials=50, search_space=None, frozen_network_manifest=None):
    """Network selection uses seed 2025; all later stages freeze that network.

    A common main-table predicate is selected by the mean of the seed-wise
    selected validation ICs. This explicit aggregation convention fills an
    underspecified part of the manuscript; ties retain the first candidate.
    """
    train_fn = train_fn or _train_one
    phi_types = list(phi_types)
    for name in phi_types:
        selected_predicates(name)
    if not seeds or not alpha_values or not phi_types:
        raise ValueError("seeds, alpha_values and phi_types must be nonempty")
    output = Path(output_dir).resolve()
    if frozen_network_manifest is not None and frozen_network_manifest["market"] != market:
        raise ValueError("frozen network selection must come from the same market")
    manifest = {"protocol": "original-paper", "market": market, "network_seed": 2025, "test_evaluated": False,
                "predicate_selection_rule": "mean validation daily IC across seeds; exclude ones; first wins ties",
                "models": {}}

    def run(config, seed, folder):
        config["logdir"] = str(folder)
        config = configure_run(config, seed)
        config_path = Path(config["logdir"]) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as stream:
            YAML.YAML().dump(config, stream)
        score = float(train_fn(config, config["task"]["model"]["kwargs"]["model_type"]))
        if not math.isfinite(score):
            raise ValueError(f"selection requires finite validation IC: {config_path}")
        return {"valid_ic": score, "seed": int(seed), "config_path": str(config_path),
                "checkpoint": str(config_path.parent / "model.bin")}

    for model_type in models:
        model_dir = output / model_type
        network_rows = []

        def evaluate_network(hp):
            cfg = _apply(base_config, hp, model_type, "mse", 0., market)
            row = run(cfg, 2025, model_dir / "network" / str(len(network_rows)))
            row["hyperparameters"] = dict(hp)
            network_rows.append(row)
            _write_json(model_dir / "network_candidates.json", network_rows)
            return row["valid_ic"]

        if frozen_network_manifest is not None:
            network_rows.append(copy.deepcopy(frozen_network_manifest["models"][model_type]["network_selection"]))
        elif network_candidates is not None:
            for hp in network_candidates:
                evaluate_network(hp)
        else:
            import optuna
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2025))
            study.optimize(lambda trial: evaluate_network(_suggest(trial, network_search_space(model_type, search_space))),
                           n_trials=n_trials)
        if not network_rows:
            raise ValueError("network search produced no finite candidates")
        network = max(network_rows, key=lambda row: row["valid_ic"])
        frozen_hp = network["hyperparameters"]
        candidates, per_predicate = [], {}
        for name in phi_types:
            selected = []
            # Feature and Original have no beta penalty to tune.
            grid = alpha_values if base_config["model_config"].get("injection_mode", "tsil") == "tsil" and name != "mse" else [0.]
            for seed in seeds:
                rows = []
                for alpha in grid:
                    cfg = _apply(base_config, frozen_hp, model_type, name, alpha, market)
                    row = run(cfg, seed, model_dir / "predicates" / name / f"alpha{alpha}")
                    row.update(phi_type=name, penalty_alpha=float(alpha), beta=float(1 / (1 + np.exp(-alpha))))
                    rows.append(row)
                    candidates.append(row)
                    _write_json(model_dir / "predicate_candidates.json", candidates)
                selected.append(max(rows, key=lambda row: row["valid_ic"]))
            per_predicate[name] = selected
        domain = [name for name in phi_types if name in TEMPORAL_PHI_TYPES]
        best_name = max(domain, key=lambda name: np.mean([r["valid_ic"] for r in per_predicate[name]])) if domain else None
        manifest["models"][model_type] = {
            "batch_unit": "complete trading days" if model_type in RELATIONAL_MODELS else "stock-date samples",
            "network_hyperparameters": frozen_hp, "network_selection": network,
            "selected_by_predicate": per_predicate, "selected_predicate": best_name,
            "selected_runs": per_predicate[best_name] if best_name else [],
        }
        _write_json(output / "selection_manifest.json", manifest)
    return manifest


def build_argparser():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=["LSTM"])
    parser.add_argument("--phi_types", nargs="+", default=["mse", "ones", *sorted(TEMPORAL_PHI_TYPES)])
    parser.add_argument("--alpha_values", nargs="+", type=float, default=PENALTY_ALPHA_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--market", choices=["csi300", "csi800"], default="csi300")
    parser.add_argument("--injection_mode", choices=["tsil", "feature"], default="tsil")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--output_dir", default="output_search_paper")
    parser.add_argument("--frozen_network_manifest", help="reuse network choices across Original/Feature/TSIL")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    with open(args.config_file, encoding="utf-8") as stream:
        config = YAML.YAML(typ="safe").load(stream)
    config["model_config"]["injection_mode"] = args.injection_mode
    frozen = json.loads(Path(args.frozen_network_manifest).read_text(encoding="utf-8")) if args.frozen_network_manifest else None
    run_paper_selection(config, args.models, args.phi_types, args.alpha_values, args.market,
                        args.output_dir, seeds=args.seeds, n_trials=args.n_trials, frozen_network_manifest=frozen)
