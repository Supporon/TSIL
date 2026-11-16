import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.tsil_loss import StatisticalInvariantLoss

ROOT = Path(__file__).parents[1]
NAMES = ["LSTM", "GRU", "TCN", "Transformer", "GAT", "LSR_IGRU", "MASTER", "MERA", "StockMixer"]


def model_config(name):
    filename = {"MASTER": "master", "StockMixer": "stockmixer"}.get(name, name.lower())
    cfg = yaml.safe_load((ROOT / f"configs/f158_csi300/config_{filename}.yaml").read_text(encoding="utf-8"))
    mc = cfg["model_config"]
    mc.update(d_model=8, d_ff=8, e_layers=1, dropout=0., n_heads=2,
              tau_hat_init=0., penalty_alpha=0., num_expert=2, top_k=1,
              t_nhead=2, s_nhead=2, T_dropout_rate=0., S_dropout_rate=0.)
    return mc


@pytest.mark.parametrize("name", NAMES)
def test_backbone_has_no_loss_parameter_and_can_train_one_step(name):
    module = importlib.import_module("src.models." + ("master" if name == "MASTER" else name))
    model = getattr(module, name)(SimpleNamespace(**model_config(name)))
    assert "alpha" not in dict(model.named_parameters()), "paper beta must not be a backbone parameter"
    x = torch.randn(3, 20, 158)
    pred = model(x).reshape(-1)
    assert pred.shape == (3,)
    loss = StatisticalInvariantLoss(0.5)(pred, torch.tensor([.01, -.02, .03]), {"ones": torch.ones(3, 1)})
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_daily_ic_is_not_batch_ic_and_is_order_invariant():
    from src.model_backbone_phitp import evaluate_daily
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=2), ["a", "b", "c"]],
                                    names=["datetime", "instrument"])
    # Each daily ordering is perfect; opposite daily offsets make pooled IC negative.
    frame = pd.DataFrame({"score": [101, 102, 103, 1, 2, 3],
                          "label": [1, 2, 3, 101, 102, 103]}, index=idx)
    assert frame.score.corr(frame.label) < 0
    result = evaluate_daily(frame)
    assert result["IC"] == pytest.approx(1.)
    pd.testing.assert_series_equal(pd.Series(result), pd.Series(evaluate_daily(frame.iloc[[5, 0, 3, 1, 4, 2]])))


@pytest.mark.parametrize("name", ["GAT", "MASTER", "LSR_IGRU"])
def test_relation_forward_isolated_by_date(name):
    from src.model_backbone_phitp import QniverseModel
    mc = model_config(name)
    mc["phi_type"] = "mse"
    model = QniverseModel(mc, model_type=name, seed=42, plot_training=False)
    model.model.eval()
    batch = {"data": torch.randn(6, 20, 159), "date": torch.tensor([1, 2, 1, 2, 1, 2]),
             "predicates": {}}
    first = model._forward_batch(batch)
    batch["data"][batch["date"] == 2] += 100
    second = model._forward_batch(batch)
    torch.testing.assert_close(first[batch["date"] == 1], second[batch["date"] == 1])


@pytest.mark.parametrize("name", NAMES)
def test_feature_concatenation_uses_same_raw_descriptor_at_each_time_step(name):
    from src.model_backbone_phitp import QniverseModel
    mc = model_config(name)
    mc.update(phi_type="phi_momentum", injection_mode="feature")
    model = QniverseModel(mc, model_type=name, seed=42, plot_training=False)
    captured = []
    model.model.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    phi = torch.tensor([[.1, .2, .3], [-.1, -.2, -.3]])
    batch = {"data": torch.zeros(2, 20, 159), "date": torch.ones(2, dtype=torch.long),
             "predicates": {"phi_momentum": phi}}
    model._forward_batch(batch)
    assert captured[0].shape == (2, 20, 161)
    torch.testing.assert_close(captured[0][:, :, 158:], phi[:, None, :].expand(-1, 20, -1))
    assert isinstance(model.criterion, torch.nn.MSELoss)


class TinyDataset:
    def __init__(self):
        self.index = pd.MultiIndex.from_product([["a", "b", "c"], pd.date_range("2024-01-01", periods=2)],
                                               names=["instrument", "datetime"])
        self.data = torch.randn(6, 20, 159)
        self.label = torch.tensor([.01, .02, -.01, -.02, .04, .03])
        self.calls = []

    def prepare(self, names):
        self.calls.extend(names)
        assert "test" not in names, "search must not prepare test data"
        return [self for _ in names]

    def train(self):
        pass

    def eval(self):
        pass

    def restore_index(self, index):
        return self.index[np.asarray(index)]

    def __len__(self):
        return 2

    def __iter__(self):
        for indices in (torch.arange(5), torch.tensor([5])):
            yield {"data": self.data[indices], "label": self.label[indices], "index": indices,
                   "date": indices % 2, "predicates": {"ones": torch.ones(len(indices), 1)}}


def test_training_search_path_keeps_tail_and_does_not_access_test(tmp_path):
    from src.model_backbone_phitp import QniverseModel
    mc = model_config("LSTM")
    mc.update(phi_type="ones", penalty_alpha=2.)
    model = QniverseModel(mc, model_type="LSTM", seed=42, n_epochs=2, plot_training=False, logdir=str(tmp_path))
    ds = TinyDataset()
    _, metrics = model.fit(ds, evaluate_test=False)
    assert set(ds.calls) == {"train", "valid"}
    assert model.avg_params is False
    assert (tmp_path / "model.bin").exists()
    assert np.isfinite(metrics["best_score"])
    eval_metrics, preds = model.test_epoch(ds, return_pred=True)
    assert len(preds) == 6 and eval_metrics["sample_count"] == 6
    assert eval_metrics["InfT_ms_per_sample"] >= 0


def test_missing_future_label_is_masked_only_after_forward():
    from src.model_backbone_phitp import QniverseModel
    mc = model_config("LSTM")
    mc.update(phi_type="ones", penalty_alpha=0.)
    model = QniverseModel(mc, model_type="LSTM", seed=42, plot_training=False)
    ds = TinyDataset()
    ds.label[1] = float("nan")
    forwards, losses = [], []
    model.model.register_forward_pre_hook(lambda module, args: forwards.append(len(args[0])))
    model.criterion.register_forward_pre_hook(lambda module, args: losses.append((len(args[0]), len(args[2]["ones"]))))
    _, metrics = model.train_epoch(ds)
    assert forwards == [5, 1] and losses == [(4, 4), (1, 1)]
    assert metrics["sample_count"] == 5
    metrics, pred = model.test_epoch(ds, return_pred=True)
    assert len(pred) == 6 and pred.score.notna().all()
    assert metrics["sample_count"] == 5 and metrics["prediction_count"] == 6
