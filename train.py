"""Train one fixed configuration, or evaluate a selected checkpoint on test.

Use search.py for validation-only model selection, then --checkpoint with the
saved config.yaml. Portfolio results come from Qlib's executed trades.
"""
import argparse
import copy
import json
from pathlib import Path

import ruamel.yaml as YAML
import numpy as np
import torch

from src.run_utils import (DEFAULT_SEEDS, apply_overrides, build_report,
                           configure_dataset, configure_run, portfolio_analysis)


def evaluate_checkpoint(model, dataset, checkpoint):
    """Load the already selected weights; never fit or inspect validation again."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state["model"])
    metrics, preds = model.test_epoch(dataset.prepare("test"), return_pred=True)
    info = json.loads((checkpoint.parent / "info.json").read_text(encoding="utf-8"))
    metrics.update({key: info["metric"][key] for key in ("best_epoch", "best_score")})
    return preds, metrics


def execute_backtest(preds, config):
    """Use Qlib's next-step signal execution, lot rounding and trade filters."""
    from qlib.backtest import backtest
    config = copy.deepcopy(config)
    config["strategy"]["kwargs"]["signal"] = preds["score"]
    executor = config.get("executor", {
        "class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    })
    portfolios, _ = backtest(executor=executor, strategy=config["strategy"], **config["backtest"])
    return portfolios["1day"]  # daily report and actual positions


def main(seed, config, config_file=None, checkpoint=None):
    import qlib
    from qlib.utils import init_instance_by_config

    if checkpoint is None:
        config = configure_run(config, seed)
    else:
        config = copy.deepcopy(config)
        if seed != config["task"]["model"]["kwargs"]["seed"]:
            raise ValueError("checkpoint evaluation must use its saved configuration and seed")
        config["logdir"] = str(Path(checkpoint).resolve().parent / "test_evaluation")
        config["task"]["model"]["kwargs"]["logdir"] = config["logdir"]
        configure_dataset(config)
    logdir = Path(config["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / "config.yaml").open("w", encoding="utf-8") as stream:
        YAML.YAML().dump(config, stream)
    qlib.init(provider_uri=config["qlib_init"]["provider_uri"],
              region=config["qlib_init"]["region"], kernels=1)
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])
    if checkpoint is None:
        preds, metrics = model.fit(dataset)
    else:
        preds, metrics = evaluate_checkpoint(model, dataset, checkpoint)
    if not preds.index.is_unique:
        raise ValueError("prediction index must contain each stock-date once")
    preds.to_pickle(logdir / "pred.pkl")
    daily, positions = execute_backtest(preds, config["port_analysis_config"])
    daily.to_csv(logdir / "backtest_daily.csv")
    # Preserve the actual executed holdings for audit, not a label-average proxy.
    import pickle
    with (logdir / "positions.pkl").open("wb") as stream:
        pickle.dump(positions, stream)
    analysis = portfolio_analysis(daily)
    report = build_report(metrics, analysis)
    (logdir / "backtest_report.txt").write_text(report, encoding="utf-8")
    result = {"prediction": metrics, "portfolio": analysis, "seed": int(seed),
              "source_config": str(config_file), "checkpoint": str(checkpoint) if checkpoint else None,
              "execution": "signal at t, Qlib execution at next trading step; close prices with side-specific slippage",
              "portfolio_convention": "arithmetic annualization 252; daily return std ddof=1; additive drawdown"}
    # Undefined ICIR on constant daily IC is recorded as null, not misleading zero.
    import math
    def finite(value):
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not math.isfinite(value) else value
    (logdir / "metrics.json").write_text(json.dumps(finite(result), indent=2, ensure_ascii=False,
                                                   allow_nan=False), encoding="utf-8")
    print(report)
    return result


def build_argparser():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", help="selected model.bin; requires its saved config.yaml")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--phi_type", default=None)
    parser.add_argument("--injection_mode", choices=["original", "feature", "tsil"])
    parser.add_argument("--penalty_alpha", type=float)
    parser.add_argument("--base_model")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--e_layers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--market", choices=["csi300", "csi800"])
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    with open(args.config_file, encoding="utf-8") as stream:
        config = YAML.YAML(typ="safe").load(stream)
    if args.checkpoint:
        overrides = [name for name in ("phi_type", "injection_mode", "penalty_alpha", "base_model",
                     "lr", "d_model", "e_layers", "batch_size", "market") if getattr(args, name) is not None]
        if overrides:
            raise ValueError(f"do not alter a selected checkpoint's configuration: {overrides}")
        seeds = args.seed or [config["task"]["model"]["kwargs"]["seed"]]
    else:
        config = apply_overrides(config, args)
        seeds = args.seed or DEFAULT_SEEDS
    for seed in seeds:
        main(seed, config, args.config_file, args.checkpoint)
