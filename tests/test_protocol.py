from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src import run_utils

ROOT = Path(__file__).parents[1]
CONFIGS = sorted((ROOT / "configs").glob("f158_csi*/config_*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_shipped_configs_follow_paper(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    handler = cfg["data_handler_config"]
    assert handler["label"] == ["Ref($close, -1) / $close - 1"]
    assert [p["class"] for p in handler["infer_processors"]] == ["CSZScoreNorm", "Fillna"]
    assert [p["class"] for p in handler["learn_processors"]] == ["DropnaLabel"]
    assert "tau_hat_init" not in cfg["model_config"]
    assert cfg["model_config"]["penalty_alpha"] == 0.0
    assert cfg["task"]["model"]["kwargs"]["early_stop"] == 20
    assert cfg["task"]["model"]["kwargs"]["avg_params"] is False
    ex = cfg["port_analysis_config"]["backtest"]["exchange_kwargs"]
    assert ex["deal_price"] == ["$close * 1.0001", "$close * 0.9999"]
    assert ex["trade_unit"] == 100
    assert ex["open_cost"] == 0.0005 and ex["close_cost"] == 0.0015


def test_seed_configuration_is_real_and_has_distinct_outputs(tmp_path):
    assert hasattr(run_utils, "configure_run"), "run seed is currently only a report tag"
    base = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
    base["logdir"] = str(tmp_path)
    a = run_utils.configure_run(base, seed=42)
    b = run_utils.configure_run(base, seed=2025)
    assert a["task"]["model"]["kwargs"]["seed"] == 42
    assert b["task"]["model"]["kwargs"]["seed"] == 2025
    assert a["logdir"] != b["logdir"]
    assert base["logdir"] == str(tmp_path)


def test_penalty_override_and_dataset_contract():
    cfg = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
    cfg = run_utils.apply_overrides(cfg, SimpleNamespace(phi_type="phi_momentum", penalty_alpha=2.0,
                                                        injection_mode="feature"))
    assert cfg["model_config"]["penalty_alpha"] == 2.
    assert cfg["model_config"]["injection_mode"] == "feature"
    assert hasattr(run_utils, "configure_dataset")
    run_utils.configure_dataset(cfg)
    ds = cfg["task"]["dataset"]["kwargs"]
    assert ds["phi_type"] == "phi_momentum" and ds["injection_mode"] == "feature"
    assert ds["cross_sectional"] is True  # first config is GAT


def test_relation_day_batch_size_is_kept_and_changes_run_identity(tmp_path):
    cfg = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
    cfg["logdir"] = str(tmp_path)
    cfg["task"]["dataset"]["kwargs"]["days_per_batch"] = 4
    first = run_utils.configure_run(cfg, 42)
    assert first["task"]["dataset"]["kwargs"]["days_per_batch"] == 4
    cfg["task"]["dataset"]["kwargs"]["days_per_batch"] = 8
    second = run_utils.configure_run(cfg, 42)
    assert first["logdir"] != second["logdir"]
