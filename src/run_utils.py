"""Shared helpers for the TSIL entry points (`train.py`, `search.py`).

Centralizes the defaults and the config-mutation logic that both the single-run
entry point and the hyper-parameter search entry point need, so the two stay in
sync:

- DEFAULT_SEEDS / DEFAULT_SEARCH_SPACE / TAU_VALUES: the canonical sweep values.
- apply_overrides():   map command-line flags onto a loaded YAML config.
- inject_model_specific_params(): fill in per-backbone architecture knobs.
- set_market():        retarget a config at a different A-share universe.
- build_report():      format the backtest report string.
"""
from typing import Optional

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

# tau_hat_init grid (learnable I-vs-P blend init); swept alongside the space above.
TAU_VALUES = [-4.0, -2.0, 0.0, 2.0, 4.0]

# Backbone families.
DOMAIN_AGNOSTIC_MODELS = ["LSTM", "GRU", "TCN", "Transformer", "GAT", "GCN", "PatchTST"]
STOCK_SPECIALIZED_MODELS = ["LSR_IGRU", "MASTER", "MERA", "StockMixer"]
ALL_MODELS = DOMAIN_AGNOSTIC_MODELS + STOCK_SPECIALIZED_MODELS


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
    if getattr(args, "tau_hat_init", None) is not None:
        mc["tau_hat_init"] = args.tau_hat_init
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


def build_report(metrics: dict, analysis_dict: dict) -> str:
    """Format the backtest report string from ranking + portfolio metrics."""
    def g(k):
        return analysis_dict.get(k, float("nan"))

    return f"""
        **********************************
                Backtest Report
        **********************************
             InfT:       {metrics['InfT']:.6f}
        Best epoch:      {metrics['best_epoch']}
        Best score:      {metrics['best_score']:.4f}
              MSE:       {metrics['MSE']:.4f}
              MAE:       {metrics['MAE']:.4f}
               IC:       {metrics['IC']:.4f}
             ICIR:       {metrics['ICIR']:.4f}
           RankIC:       {metrics['RankIC']:.4f}
         RankICIR:       {metrics['RankICIR']:.4f}
        w/o cost AR:    {g('1day.excess_return_without_cost.annualized_return'):.4f}
        w/o cost ret mean:    {g('1day.excess_return_without_cost.mean'):.4f}
        w/o cost ret std:    {g('1day.excess_return_without_cost.std'):.4f}
        w/o cost IR:    {g('1day.excess_return_without_cost.information_ratio'):.4f}
        w/o cost MMD:    {g('1day.excess_return_without_cost.max_drawdown'):.4f}
        with cost AR:    {g('1day.excess_return_with_cost.annualized_return'):.4f}
        with cost ret mean:    {g('1day.excess_return_with_cost.mean'):.4f}
        with cost ret std:    {g('1day.excess_return_with_cost.std'):.4f}
        with cost IR:    {g('1day.excess_return_with_cost.information_ratio'):.4f}
        with cost MMD:    {g('1day.excess_return_with_cost.max_drawdown'):.4f}
        ***************************************
        """
