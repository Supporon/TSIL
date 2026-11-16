import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_portfolio_metrics_use_executed_return_and_distinguish_excess():
    from src.run_utils import portfolio_analysis, build_report
    daily = pd.DataFrame({"return": [.01, -.02, .03], "cost": [.001, .002, .001],
                          "bench": [.005, -.01, .01]})
    result = portfolio_analysis(daily)
    net = daily["return"] - daily["cost"]
    excess = net - daily["bench"]
    assert result["strategy_with_cost.annualized_return"] == pytest.approx(net.mean() * 252)
    assert result["excess_with_cost.annualized_return"] == pytest.approx(excess.mean() * 252)
    assert result["strategy_with_cost.information_ratio"] == pytest.approx(net.mean()/net.std(ddof=1)*np.sqrt(252))
    report = build_report({"InfT_ms_per_sample": .25, "best_epoch": 1, "best_score": .1}, result)
    assert "ms/sample" in report and "strategy_with_cost" in report and "excess_with_cost" in report


def test_predict_uses_only_requested_segment():
    from src.model_backbone_phitp import QniverseModel
    model = object.__new__(QniverseModel)
    sentinel = pd.DataFrame({"score": [.3]})
    calls = []
    class Dataset:
        def prepare(self, name):
            calls.append(name)
            return "prepared"
    model.test_epoch = lambda data, return_pred: ({}, sentinel)
    assert model.predict(Dataset(), segment="test") is sentinel
    assert calls == ["test"]


def test_checkpoint_evaluation_never_refits_or_reselects(tmp_path):
    from train import evaluate_checkpoint
    class Wrapper:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
        def test_epoch(self, segment, return_pred):
            assert segment == "test-data" and return_pred
            return {"IC": .2}, pd.DataFrame({"score": [.3], "label": [.1]})
    class Dataset:
        def prepare(self, name):
            assert name == "test"
            return "test-data"
    wrapper = Wrapper()
    checkpoint = tmp_path / "model.bin"
    torch.save({"model": wrapper.model.state_dict()}, checkpoint)
    (tmp_path / "info.json").write_text(json.dumps({"metric": {"best_score": .12, "best_epoch": 4}}), encoding="utf-8")
    preds, metrics = evaluate_checkpoint(wrapper, Dataset(), checkpoint)
    assert len(preds) == 1 and metrics["best_score"] == .12 and metrics["best_epoch"] == 4
