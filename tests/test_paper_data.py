"""Data-path oracles for causal paper descriptors and stock-time sampling."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
NAMES = ["phi_timestamp", "phi_cycle", "phi_phase", "phi_momentum",
         "phi_volatility", "phi_return_dist", "ones"]
SMALL_WINDOWS = {"momentum_windows": [2], "volatility_windows": [3],
                 "cycle_window": 4, "phase_window": 4, "distribution_window": 4}


def cache_builder():
    assert importlib.util.find_spec("src.predicate_data"), "raw-price predicate cache is missing"
    from src.predicate_data import build_predicate_cache
    return build_predicate_cache


def prices(count=70):
    dates = pd.bdate_range("2023-10-02", periods=count, name="datetime")
    q = np.arange(count - 1)
    returns = np.stack([0.004 + 0.02 * np.sin(2 * np.pi * q / 5),
                        -0.001 + 0.03 * np.sin(2 * np.pi * q / 10)], axis=1)
    close = 100 * np.vstack([np.ones(2), np.cumprod(1 + returns, axis=0)])
    return pd.DataFrame(close, index=dates, columns=["AAA", "BBB"])


def sample_index(frame, dates=None):
    return pd.MultiIndex.from_product([frame.columns, frame.index if dates is None else dates],
                                      names=["instrument", "datetime"])


def test_slices_never_read_the_previous_stock():
    # Execute the actual standalone helper without needing Qlib's optional dependencies.
    tree = ast.parse((ROOT / "src/dataset_ori_onehot.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_create_ts_slices")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dataset_ori_onehot.py", "exec"), namespace)
    index = pd.MultiIndex.from_product([["AAA", "BBB"], pd.bdate_range("2024-01-01", periods=3)])
    slices = namespace["_create_ts_slices"](index, 3)
    assert [(s.start, s.stop) for s in slices] == [(0, 1), (0, 2), (0, 3),
                                                 (3, 4), (3, 5), (3, 6)]


def test_cache_matches_raw_price_and_simple_return_oracles():
    close = prices()
    index = sample_index(close, close.index[[39, 40, 44]])
    descriptors, eligible, q = cache_builder()(close, index, NAMES)
    np.testing.assert_array_equal(eligible, [False, True, True, False, True, True])
    np.testing.assert_array_equal(q, [39, 40, 44, 39, 40, 44])
    for row in np.flatnonzero(eligible):
        stock, date = index[row]
        p = close.loc[:date, stock].to_numpy()[-41:]
        r = p[1:] / p[:-1] - 1
        np.testing.assert_allclose(descriptors["phi_momentum"][row],
                                   [np.log(p[-1] / p[-1-k]) for k in (5, 10, 20)], atol=1e-7)
        np.testing.assert_allclose(descriptors["phi_volatility"][row],
                                   [r[-w:].std(ddof=0) for w in (10, 20, 40)], atol=1e-7)
        tail = r[-20:]
        z = (tail - tail.mean()) / (tail.std(ddof=0) + 1e-8)
        np.testing.assert_allclose(descriptors["phi_return_dist"][row],
                                   [tail.mean(), tail.std(ddof=0), (z**3).mean(), (z**4).mean()-3],
                                   atol=1e-6)
        stamp = descriptors["phi_timestamp"][row]
        assert stamp.shape == (19,)
        assert stamp[date.weekday()] == stamp[7 + date.month - 1] == 1
        assert stamp.sum() == 2
        assert descriptors["ones"][row, 0] == 1


def test_future_price_mutation_cannot_change_past_descriptors():
    close = prices()
    index = sample_index(close, close.index[40:50])
    build = cache_builder()
    before, eligible, q = build(close, index, NAMES)
    changed = close.copy()
    changed.iloc[50:] = changed.iloc[50:].to_numpy() * np.arange(2, len(changed) - 48)[:, None]
    after, eligible_after, q_after = build(changed, index, NAMES)
    np.testing.assert_array_equal(eligible, eligible_after)
    np.testing.assert_array_equal(q, q_after)
    for name in NAMES:
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize("missing", [np.nan, 0., -1., np.inf])
def test_missing_raw_history_is_never_filled_and_masks_all_modes_equally(missing):
    close = prices()
    close.iloc[20, 0] = missing
    index = sample_index(close, close.index[[40, 60, 61]])
    build = cache_builder()
    for names in (None, "ones", NAMES):
        _, eligible, _ = build(close, index, names)
        np.testing.assert_array_equal(eligible, [False, False, True, True, True, True])


def test_market_cycle_averages_only_available_returns_of_current_members():
    dates = pd.bdate_range("2024-01-01", periods=10, name="datetime")
    q = np.arange(1, 10)
    r = np.stack([.01 * np.sin(2*np.pi*q/4), .4 * np.cos(np.pi*q)], axis=1)
    close = pd.DataFrame(100*np.vstack([np.ones(2), np.cumprod(1+r, axis=0)]),
                         index=dates, columns=["AAA", "BBB"])
    members = pd.DataFrame(True, index=dates, columns=close.columns)
    members.loc[dates[5]:, "BBB"] = False
    index = sample_index(close, dates[-1:])
    descriptors, eligible, q = cache_builder()(close, index, "phi_cycle", seq_len=3,
                                               predicate_config=SMALL_WINDOWS,
                                               market_membership=members)
    assert eligible.all()
    expected = np.tile([1., 0.], (2, 1))  # q=9 and the active market's period is 4.
    np.testing.assert_allclose(descriptors["phi_cycle"], expected, atol=1e-6)


def test_missing_market_window_excludes_otherwise_valid_stock():
    close = prices()
    members = pd.DataFrame(True, index=close.index, columns=close.columns)
    members.iloc[45, :] = False
    index = sample_index(close, close.index[[44, 45, 64, 65]])
    _, eligible, _ = cache_builder()(close, index, None, market_membership=members)
    np.testing.assert_array_equal(eligible, [True, False, False, True] * 2)


@pytest.fixture
def dataset_factory(monkeypatch):
    if importlib.util.find_spec("qlib"):
        from src import dataset_ori_onehot as module
    else:
        # Isolate only the unavailable Qlib base/provider; run the actual
        # dataset source, and automatically use the real imports once installed.
        tree = ast.parse((ROOT / "src/dataset_ori_onehot.py").read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ImportFrom) and node.module.startswith("qlib"))]
        module = ModuleType("dataset_without_qlib")
        module.DatasetH = type("DatasetH", (), {"setup_data": lambda self: None})
        module.D = None
        exec(compile(tree, "dataset_ori_onehot.py", "exec"), module.__dict__)

    def make(*, mode="tsil", cross_sectional=False, batch_size=7, drop_last=False,
             predicate="phi_momentum", missing=None, index_order="datetime_first",
             learned_start=0, missing_label=None):
        close = prices(50)
        if missing is not None:
            close.iloc[missing, 0] = np.nan
        full_index = sample_index(close, close.index[learned_start:])
        values = np.array([[q + stock * 1000, 0.01] for stock in range(2)
                           for q in range(learned_start, len(close))])
        frame = pd.DataFrame(values, index=full_index,
                             columns=pd.MultiIndex.from_tuples([("feature", "x"), ("label", "y")]))
        if missing_label is not None:
            frame.loc[("AAA", close.index[missing_label]), ("label", "y")] = np.nan
        if index_order == "datetime_first":
            frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()
        original = frame.copy(deep=True)
        handler = SimpleNamespace(_infer=frame.copy(),
                                  _learn=frame.dropna(subset=[("label", "y")]).copy())

        class Provider:
            def __init__(self):
                self.calls = []

            def calendar(self, start_time=None, end_time=None, freq="day"):
                self.calls.append(("calendar", start_time, end_time))
                dates = close.index
                return dates[(dates >= pd.Timestamp(start_time) if start_time else True)
                             & (dates <= pd.Timestamp(end_time) if end_time else True)].to_numpy()

            def instruments(self, market):
                return market

            def list_instruments(self, instruments, start_time=None, end_time=None,
                                 freq="day", as_list=False):
                return {name: [(close.index[0], close.index[-1])] for name in close.columns}

            def features(self, instruments, fields, start_time=None, end_time=None, freq="day"):
                self.calls.append(("features", instruments, fields, start_time, end_time))
                assert fields == ["$close"]
                raw = close.loc[start_time:end_time, list(instruments)]
                return raw.rename_axis(columns="instrument").stack(future_stack=True).rename("$close").to_frame()

        provider = Provider()

        def parent_init(self, handler_config, segments, **kwargs):
            self.handler = handler
            self.segments = segments
            self.setup_data()

        monkeypatch.setattr(module.DatasetH, "__init__", parent_init)
        monkeypatch.setattr(module.DatasetH, "setup_data", lambda self, **kwargs: None)
        monkeypatch.setattr(module, "D", provider)
        config = {"kwargs": {"instruments": "fake", "start_time": close.index[learned_start],
                              "end_time": close.index[-1]}}
        dataset = module.MTSDatasetH(config, {"test": [close.index[40], close.index[-1]]},
                                    seq_len=20, batch_size=batch_size, shuffle=False,
                                    drop_last=drop_last, phi_type=predicate,
                                    injection_mode=mode, cross_sectional=cross_sectional)
        return dataset, provider, handler, original, close

    return make


@pytest.mark.parametrize("index_order", ["datetime_first", "instrument_first"])
def test_setup_preserves_handler_and_keeps_all_feature_history(dataset_factory, index_order):
    dataset, provider, handler, original, close = dataset_factory(index_order=index_order)
    pd.testing.assert_frame_equal(handler._learn, original)
    assert dataset._index.names == ["instrument", "datetime"]
    assert len(dataset._index) == 100
    assert len(dataset.batch_slices) == 20
    first = dataset.batch_slices[0]
    assert (first.start, first.stop) == (21, 41)
    np.testing.assert_array_equal(dataset._data[first, 0], np.arange(21, 41))
    batch = next(iter(dataset))
    assert batch["data"].shape[1] == 20
    expected = np.log(close.iloc[40, 0] / close.iloc[[35, 30, 20], 0].to_numpy())
    np.testing.assert_allclose(batch["predicates"]["phi_momentum"][0].cpu().numpy(), expected, atol=1e-7)


def test_raw_reader_extends_history_and_uses_full_provider_calendar(dataset_factory):
    dataset, provider, _, _, close = dataset_factory(learned_start=45)
    calendar_calls = [call for call in provider.calls if call[0] == "calendar"]
    assert calendar_calls and calendar_calls[0][1] is None
    read = next(call for call in provider.calls if call[0] == "features")
    assert pd.Timestamp(read[3]) == close.index[5]
    np.testing.assert_array_equal(dataset._trading_index[:5], np.arange(45, 50))
    # Raw history is long enough, but only five factor days were supplied.
    assert len(dataset.batch_slices) == 0


def test_split_purges_forward_labels_and_retains_pre_split_history(dataset_factory):
    dataset, _, _, _, close = dataset_factory()
    segment = dataset._prepare_seg((close.index[42], close.index[44]))
    dates = segment.restore_index([s.stop - 1 for s in segment.batch_slices]).get_level_values("datetime")
    assert set(dates) == set(close.index[42:44])
    assert all(dataset._index[s.start][1] < close.index[42] for s in segment.batch_slices)
    assert all(dataset._index[s.stop-1][1] <= close.index[43] for s in segment.batch_slices)
    tail = dataset._prepare_seg((close.index[48], close.index[-1]))
    assert len(tail.batch_slices) == 2


def test_eval_retains_tail_and_never_rebuilds_predicates(dataset_factory, monkeypatch):
    dataset, provider, _, _, _ = dataset_factory(drop_last=True)
    calls = len(provider.calls)
    dataset.eval()
    batches = list(dataset)
    assert [len(batch["index"]) for batch in batches] == [7, 7, 6]
    assert len(dataset) == 3
    assert all(batch["predicates"] == {} for batch in batches)
    assert len(provider.calls) == calls
    dataset.train()
    assert dataset.drop_last is True
    assert next(iter(dataset))["predicates"]


@pytest.mark.parametrize("mode,has_eval_predicates", [("original", False), ("tsil", False), ("feature", True)])
def test_modes_share_eligibility_and_obey_predicate_lifecycle(dataset_factory, mode, has_eval_predicates):
    dataset, _, _, _, _ = dataset_factory(mode=mode, missing=3)
    assert len(dataset.batch_slices) == 16
    dataset.eval()
    batches = list(dataset)
    assert sum(len(batch["index"]) for batch in batches) == 16
    assert all(bool(batch["predicates"]) == has_eval_predicates for batch in batches)


def test_cross_sectional_batches_keep_whole_days_and_default_to_two_days(dataset_factory):
    dataset, _, _, _, _ = dataset_factory(cross_sectional=True, batch_size=1)
    batches = list(dataset)
    assert len(dataset) == len(batches) == 5
    seen = []
    for batch in batches:
        dates = batch["date"].cpu().numpy()
        unique, counts = np.unique(dates, return_counts=True)
        np.testing.assert_array_equal(counts, [2, 2])
        assert len(batch["index"]) == 4
        seen.extend(unique)
    np.testing.assert_array_equal(sorted(seen), np.arange(40, 50))


def test_non_relational_model_keeps_stock_time_batches(dataset_factory):
    dataset, _, _, _, _ = dataset_factory(cross_sectional=False, batch_size=3)
    assert [len(batch["index"]) for batch in dataset] == [3, 3, 3, 3, 3, 3, 2]


def test_missing_labels_do_not_remove_historical_features(dataset_factory):
    dataset, _, handler, original, close = dataset_factory(missing_label=25)
    assert len(handler._learn) == 99
    pd.testing.assert_frame_equal(handler._infer, original)
    assert len(dataset._index) == 100
    endpoint = dataset._index.get_loc(("AAA", close.index[40]))
    window = next(slc for slc in dataset.batch_slices if slc.stop == endpoint + 1)
    np.testing.assert_array_equal(dataset._data[window, 0], np.arange(21, 41))
    assert np.isnan(dataset._label[dataset._index.get_loc(("AAA", close.index[25]))])


def test_feature_windows_cannot_bridge_missing_membership_day(dataset_factory):
    dataset, _, handler, _, close = dataset_factory()
    handler._infer = handler._infer.drop(index=(close.index[24], "AAA"))
    dataset.setup_data()
    indices = dataset.restore_index([slc.stop - 1 for slc in dataset.batch_slices])
    assert set(indices[indices.get_level_values("instrument") == "AAA"].get_level_values("datetime")) == set(close.index[44:])
    assert set(indices[indices.get_level_values("instrument") == "BBB"].get_level_values("datetime")) == set(close.index[40:])
    for slc in dataset.batch_slices:
        assert slc.stop - slc.start == 20
        np.testing.assert_array_equal(np.diff(dataset._trading_index[slc]), np.ones(19))


@pytest.mark.parametrize("change_future_price", [False, True], ids=["missing-label", "missing-next-close"])
def test_gat_predictions_do_not_depend_on_future_label_availability(dataset_factory, change_future_price):
    import torch
    import yaml
    from src.model_backbone_phitp import QniverseModel

    dataset, _, handler, _, close = dataset_factory(mode="original", cross_sectional=True)
    mc = yaml.safe_load((ROOT / "configs/f158_csi300/config_gat.yaml").read_text(encoding="utf-8"))["model_config"]
    mc.update(enc_in=1, d_model=8, d_ff=8, e_layers=1, dropout=0.)
    model = QniverseModel(mc, model_type="GAT", seed=42, logdir=None, plot_training=False)
    model.model.eval()
    date = close.index[44]

    def scores_at_date():
        segment = dataset._prepare_seg((date, close.index[45]))
        segment.eval()
        batch = next(iter(segment))
        with torch.no_grad():
            scores = model._forward_batch(batch).cpu().numpy()
        return pd.Series(scores, index=segment.restore_index(batch["index"])).sort_index()

    before = scores_at_date()
    # Only target availability changes at t. A missing t+1 close is one real
    # cause; configured DropnaLabel removes AAA@t from the learning frame.
    handler._infer.loc[(date, "AAA"), ("label", "y")] = np.nan
    handler._learn = handler._infer.dropna(subset=[("label", "y")]).copy()
    if change_future_price:
        close.iloc[45, 0] = np.nan
    dataset.setup_data()
    after = scores_at_date()

    assert after.loc[("BBB", date)] == pytest.approx(before.loc[("BBB", date)], abs=1e-8)
    pd.testing.assert_series_equal(before, after)
