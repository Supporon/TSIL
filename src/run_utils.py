"""Shared helpers for the TSIL entry points (`train.py`, `search.py`).

Centralizes the defaults and the config-mutation logic that both the single-run
entry point and the hyper-parameter search entry point need, so the two stay in
sync:

- DEFAULT_SEEDS / DEFAULT_SEARCH_SPACE / PENALTY_ALPHA_VALUES: the canonical sweep values.
- apply_overrides():   map command-line flags onto a loaded YAML config.
- inject_model_specific_params(): fill in per-backbone architecture knobs.
- set_market():        retarget a config at a different A-share universe.
- build_report():      format the backtest report string.
"""
import copy
import hashlib
import json
from pathlib import Path

# Benchmarks per universe (used when --market retargets a config).
MARKET_BENCHMARKS = {
    "csi300": "SH000300",
    "csi500": "SH000905",
    "csi800": "SH000906",
}

# Default multi-seed set for the single-run entry point.
DEFAULT_SEEDS = [42, 2022, 2023, 2024, 2025]

# Default hyper-parameter search space for search.py (Optuna).
DEFAULT_SEARCH_SPACE = {
    "lr":      {"type": "categorical", "choices": [0.001, 0.0001, 0.00001]},
    "layers":  {"type": "categorical", "choices": [1, 2, 3]},
    "d_model": {"type": "categorical", "choices": [64, 128, 256]},
    "bs":      {"type": "categorical", "choices": [256, 512, 1024]},
}

# Fixed alpha grid from the paper; beta=sigmoid(alpha), never optimised by Adam.
PENALTY_ALPHA_VALUES = [-4.0, -2.0, 0.0, 2.0, 4.0]

# Backbone families.
DOMAIN_AGNOSTIC_MODELS = ["LSTM", "GRU", "TCN", "Transformer", "GAT"]
STOCK_SPECIALIZED_MODELS = ["LSR_IGRU", "MASTER", "MERA", "StockMixer"]
ALL_MODELS = DOMAIN_AGNOSTIC_MODELS + STOCK_SPECIALIZED_MODELS
RELATIONAL_MODELS = {"GAT", "MASTER", "LSR_IGRU", "LSRIGRU"}


def configure_dataset(config):
    """Pass the same predicate/injection contract to model and data pipeline."""
    mc = config["model_config"]
    kwargs = config["task"]["dataset"]["kwargs"]
    kwargs["phi_type"] = mc.get("phi_type", "mse")
    kwargs["predicate_config"] = mc.get("predicate_config", {})
    kwargs["injection_mode"] = mc.get("injection_mode", "original" if kwargs["phi_type"] == "mse" else "tsil")
    kwargs["cross_sectional"] = config["task"]["model"]["kwargs"]["model_type"] in RELATIONAL_MODELS
    kwargs.setdefault("days_per_batch", 2)
    return config


def configure_run(config, seed):
    """Copy a run, set its actual seed and keep outputs of distinct runs apart."""
    config = copy.deepcopy(config)
    mc = config["model_config"]
    mode = mc.get("injection_mode", "original" if mc.get("phi_type", "mse") == "mse" else "tsil")
    identity = {"model": mc, "trainer": {k: v for k, v in config["task"]["model"]["kwargs"].items()
                if k not in {"logdir", "model_config", "seed"}}, "seed": int(seed),
                "market": config.get("market"), "batch_size": config["task"]["dataset"]["kwargs"]["batch_size"],
                "days_per_batch": config["task"]["dataset"]["kwargs"].get("days_per_batch", 2)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    config["logdir"] = str(Path(config["logdir"]) / f"{mode}_seed{seed}_{digest}")
    model_kwargs = config["task"]["model"]["kwargs"]
    model_kwargs["logdir"] = config["logdir"]
    model_kwargs["seed"] = int(seed)
    model_kwargs["model_config"] = mc
    configure_dataset(config)
    return config


def set_market(config: dict, market: str) -> dict:
    """Retarget a loaded config at `market` (csi300 / csi500 / csi800)."""
    benchmark = MARKET_BENCHMARKS.get(market)
    if benchmark is None:
        raise ValueError(f"unknown market {market!r}; expected one of {list(MARKET_BENCHMARKS)}")
    config["market"] = market
    config["benchmark"] = benchmark
    config["data_handler_config"]["instruments"] = market
    config["port_analysis_config"]["backtest"]["benchmark"] = benchmark
    return config


def inject_model_specific_params(config: dict, model_type: str) -> dict:
    """Fill in per-backbone architecture knobs that are not in the generic YAML.

    Mirrors the per-model settings used in the paper sweeps so the same config
    template works for every backbone.
    """
    mc = config["model_config"]
    if model_type == "MASTER":
        mc["d_feat"] = 158
        mc["t_nhead"] = 4
        mc["s_nhead"] = 2
        mc["T_dropout_rate"] = 0.5
        mc["S_dropout_rate"] = 0.5
        mc["gate_input_start_index"] = 158
        mc["gate_input_end_index"] = 221
        mc["beta"] = 5
        mc["use_gate"] = False
    elif model_type == "MERA":
        mc["num_expert"] = 8
        mc["top_k"] = 4
        mc["gate_dim"] = 16
        mc["dropout"] = 0.3
    elif model_type == "StockMixer":
        mc["market_num"] = 20
        mc["scale_factor"] = 3
        mc["activation"] = "GELU"
        mc["rank_loss_alpha"] = 0.1
        mc["d_ff"] = 16
    elif model_type == "LSR_IGRU":
        mc["base_model"] = "GRU"
    return config


def apply_overrides(config: dict, args) -> dict:
    """Overlay command-line flags onto a loaded YAML config.

    Only flags that were actually passed (non-None) override the YAML; anything
    left as None keeps the value from the config file. `args` is the argparse
    Namespace from train.py.
    """
    mc = config["model_config"]

    if getattr(args, "phi_type", None) is not None:
        mc["phi_type"] = args.phi_type
    if getattr(args, "penalty_alpha", None) is not None:
        mc["penalty_alpha"] = args.penalty_alpha
    if getattr(args, "injection_mode", None) is not None:
        mc["injection_mode"] = args.injection_mode
    if getattr(args, "d_model", None) is not None:
        mc["d_model"] = args.d_model
    if getattr(args, "e_layers", None) is not None:
        mc["e_layers"] = args.e_layers
    if getattr(args, "base_model", None) is not None:
        mc["base_model"] = args.base_model

    if getattr(args, "lr", None) is not None:
        config["task"]["model"]["kwargs"]["lr"] = args.lr
        mc["lr"] = args.lr
    if getattr(args, "batch_size", None) is not None:
        config["task"]["dataset"]["kwargs"]["batch_size"] = args.batch_size
    if getattr(args, "market", None) is not None:
        set_market(config, args.market)

    # `ind` tags the report filename; default 0 for a plain single run.
    mc.setdefault("ind", 0)
    return config


def portfolio_analysis(daily):
    """Executed strategy and benchmark-relative returns, with explicit units.

    The manuscript does not fix the annualization/compounding convention. Use
    252 trading days, arithmetic annualization and additive drawdown, and retain
    the daily series so geometric/CAGR summaries can be recomputed separately.
    """
    from qlib.contrib.evaluate import risk_analysis
    gross = daily["return"]
    series = {"strategy_without_cost": gross,
              "strategy_with_cost": gross - daily["cost"],
              "excess_without_cost": gross - daily["bench"],
              "excess_with_cost": gross - daily["cost"] - daily["bench"]}
    return {f"{group}.{key}": float(value) for group, returns in series.items()
            for key, value in risk_analysis(returns, N=252, freq=None)["risk"].items()}


def build_report(metrics: dict, analysis_dict: dict) -> str:
    """Keep prediction, strategy return and excess return meanings explicit."""
    lines = ["TSIL test evaluation", "Inference forward pipeline: %.6f ms/sample" %
             metrics.get("InfT_ms_per_sample", float("nan")),
             f"Best validation epoch: {metrics.get('best_epoch')}",
             f"Best validation daily IC: {metrics.get('best_score')}"]
    for key in ("sample_count", "MSE", "MAE", "IC", "ICIR", "RankIC", "RankICIR"):
        lines.append(f"{key}: {metrics.get(key, float('nan'))}")
    lines.append("Portfolio: arithmetic AR=252*mean(r); IR=sqrt(252)*mean(r)/std(r,ddof=1)")
    lines.append("Signals formed at t are executed by Qlib at the next trading step.")
    for group in ("strategy_without_cost", "strategy_with_cost", "excess_without_cost", "excess_with_cost"):
        values = [f"{key}={analysis_dict.get(group + '.' + key, float('nan')):.6f}"
                  for key in ("annualized_return", "information_ratio", "max_drawdown")]
        lines.append(group + ": " + ", ".join(values))
    return "\n".join(lines) + "\n"
