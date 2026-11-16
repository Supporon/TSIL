"""Reproduce small before/after counterexamples from the actual released source."""
import ast
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from src.temporal_predicates import get_temporal_phi
from src.tsil_loss import StatisticalInvariantLoss
from src.dataset_ori_onehot import _create_ts_slices

BASELINE = "5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff"


def original(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], encoding="utf-8")


def compare():
    tree = ast.parse(original("src/model_backbone_phitp.py"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightedMSELoss")
    scope = {"torch": torch, "nn": torch.nn, "TEMPORAL_PHI_TYPES": set(), "_CUSTOM_PHI_AVAILABLE": False}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "released_loss", "exec"), scope)
    pred, label = torch.tensor([1., 2.]), torch.zeros(2)
    old_loss = scope["WeightedMSELoss"]()("ones", None, None, pred, label, .5, .5)["total"].item()
    new_loss = StatisticalInvariantLoss(.5)(pred, label, {"ones": torch.ones(2, 1)})["total"].item()

    old = {"__name__": "released_predicates"}
    exec(compile(original("src/temporal_predicates.py"), "released_predicates", "exec"), old)
    torch.manual_seed(42)
    x = torch.randn(8, 20, 158)
    order = torch.tensor([3, 7, 1, 5, 0, 6, 2, 4])
    gram = old["get_cycle_I_P"](x)[1]
    permuted = old["get_cycle_I_P"](x[order])[1]
    old_order_error = (permuted - gram[order][:, order]).abs().max().item()
    market = torch.sin(2*torch.pi*torch.arange(20, dtype=torch.float64)/5).repeat(8, 1)
    q = torch.arange(40, 48, dtype=torch.float64)
    phi = get_temporal_phi("phi_cycle", market_returns=market, trading_index=q)
    permuted_phi = get_temporal_phi("phi_cycle", market_returns=market[order], trading_index=q[order])
    new_order_error = (permuted_phi - phi[order]).abs().max().item()
    # Intercept only the final Gram conversion to inspect actual descriptor width.
    old["_build_P"] = lambda phi: phi
    old_dims = {"momentum": old["get_momentum_I_P"](x)[1].shape[1],
                "volatility": old["get_volatility_I_P"](x)[1].shape[1]}
    close = torch.exp(torch.arange(41, dtype=torch.float64)[None, :]*.01)
    new_momentum = get_temporal_phi("phi_momentum", close=close)
    new_volatility = get_temporal_phi("phi_volatility", returns=torch.arange(40, dtype=torch.float64)[None, :]/1000)

    tree = ast.parse(original("src/dataset_ori_onehot.py"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_create_ts_slices")
    scope = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "released_slices", "exec"), scope)
    index = pd.MultiIndex.from_product([["A", "B"], pd.bdate_range("2024-01-01", periods=4)],
                                      names=["instrument", "datetime"])
    old_slice = scope["_create_ts_slices"](index, 3)[4]
    new_slice = _create_ts_slices(index, 3)[4]
    return {"baseline": BASELINE, "loss": {"old": old_loss, "new": new_loss, "paper_oracle": 3.0625},
            "cycle_permutation_max_error": {"old_gram": old_order_error, "new_descriptor": new_order_error},
            "descriptor_dimensions_at_T20": {"old": old_dims, "new": {"momentum": new_momentum.shape[1],
                                                                          "volatility": new_volatility.shape[1]}},
            "log_price_growth_0.01_momentum": new_momentum.tolist(),
            "first_B_history_stock_names": {"old": index[old_slice].get_level_values(0).tolist(),
                                             "new": index[new_slice].get_level_values(0).tolist()},
            "purpose": "deterministic counterexamples; no market performance claims"}


if __name__ == "__main__":
    result = compare()
    output = Path("docs/verification/before_after.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
