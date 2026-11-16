# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data import D

from src.predicate_data import build_predicate_cache, required_history
from src.temporal_predicates import selected_predicates

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def _create_ts_slices(index, seq_len):
    """Create causal feature slices within each sorted instrument block."""
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0).size().values
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    if len(start_index_of_codes):
        start_index_of_codes[0] = 0
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, cur_loc)
            slices.append(slice(start, end))
    return np.array(slices, dtype=object)


class MTSDatasetH(DatasetH):
    """Feature sequences with separately cached, causal raw-price predicates.

    All modes use the same complete raw-history eligibility rule. TSIL yields
    descriptors during training only; Feature yields them in both modes.
    With cross_sectional=True, a batch consists of days_per_batch complete
    eligible daily cross-sections, irrespective of the stock batch_size.
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=20,
        horizon=10,
        batch_size=-1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        account=100000000,
        benchmark="SH000300",
        freq='day',
        embed='fixed',
        ts_onehot=False,
        phi_type=None,
        predicate_config=None,
        injection_mode='tsil',
        cross_sectional=False,
        days_per_batch=2,
        **kwargs,
    ):
        assert horizon > 0, "please specify horizon to avoid data leakage"
        if injection_mode not in {"original", "feature", "tsil"}:
            raise ValueError("injection_mode must be original, feature, or tsil")
        if batch_size == 0 or not isinstance(days_per_batch, int) or days_per_batch < 1:
            raise ValueError("batch_size must be nonzero and days_per_batch must be positive")
        self.handler_kwargs = handler['kwargs']
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)
        self.account = account
        self.benchmark = benchmark
        self.freq = freq
        self.embed = embed
        self.ts_onehot = ts_onehot
        self.phi_type = phi_type
        self.predicate_config = dict(predicate_config or {})
        self.injection_mode = injection_mode
        self.cross_sectional = cross_sectional
        self.days_per_batch = days_per_batch
        self._training = True
        selected_predicates(phi_type)

        self.executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        }
        self.backtest_config = {
            "start_time": segments['test'][0],
            "end_time": segments['test'][1],
            "account": self.account,
            "benchmark": self.benchmark,
            "exchange_kwargs": {
                "freq": self.freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }
        super().__init__(handler, segments, **kwargs)

    def _load_raw_history(self):
        dates = pd.DatetimeIndex(self._index.get_level_values("datetime"))
        end = pd.Timestamp(self.handler_kwargs.get("end_time") or dates.max())
        # No start_time: q is measured from the provider calendar's own origin.
        self._calendar = pd.DatetimeIndex(D.calendar(end_time=end, freq=self.freq))
        history = required_history(self.seq_len, self.predicate_config)
        first = self._calendar.searchsorted(dates.min())
        start = self._calendar[max(0, first - history)]
        universe = self.handler_kwargs["instruments"]
        if isinstance(universe, str):
            universe = D.instruments(universe)
        spans = D.list_instruments(universe, start_time=start, end_time=end,
                                   freq=self.freq, as_list=False)
        instruments = sorted(set(spans) | set(self._index.get_level_values("instrument")))
        # Fetch unprocessed prices for the whole symbol list. Membership masks
        # market returns separately and does not erase pre-membership history.
        raw = D.features(instruments, ["$close"], start_time=start, end_time=end, freq=self.freq)
        close = raw["$close"].unstack("instrument").reindex(index=self._calendar, columns=instruments)
        members = pd.DataFrame(False, index=self._calendar, columns=instruments)
        for instrument, intervals in spans.items():
            for begin, finish in intervals:
                members.loc[pd.Timestamp(begin):pd.Timestamp(finish), instrument] = True
        return close, members

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super().setup_data()
        # The handler is shared by Qlib; never swap or sort its index in place.
        # DropnaLabel is a learning processor: its future-label mask must not
        # decide which stocks participate in a relation model's input at t.
        df = self.handler._infer.reorder_levels(["instrument", "datetime"]).sort_index().copy()
        label_columns = df.columns.get_level_values(0) == "label"
        self._data = df.loc[:, ~label_columns].to_numpy(dtype=np.float32)
        self._label = df.loc[:, label_columns].to_numpy(dtype=np.float32).reshape(len(df), -1)
        if self._label.shape[1] != 1:
            raise ValueError("the paper dataset requires one next-day return label")
        self._label = self._label[:, 0]
        self._index = df.index
        self._timestamp = None

        close, membership = self._load_raw_history()
        names = self.phi_type if self.injection_mode != "original" else None
        self._predicate_cache, self._eligible, self._trading_index = build_predicate_cache(
            close, self._index, names, seq_len=self.seq_len,
            predicate_config=self.predicate_config, market_membership=membership,
        )
        self._target_dates = np.full(len(self._index), np.datetime64("NaT"), dtype="datetime64[ns]")
        known_next = (self._trading_index >= 0) & (self._trading_index + 1 < len(self._calendar))
        self._target_dates[known_next] = self._calendar.values[self._trading_index[known_next] + 1]

        self._data = np.c_[self._data, np.zeros((len(self._data), 1), dtype=np.float32)]
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)
        if self.pin_memory:
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            self.zeros = _to_tensor(self.zeros)

        # Preserve rows, but require the paper's complete consecutive-day X_t.
        # Row-count windows must not bridge gaps in an instrument's membership.
        slices = _create_ts_slices(self._index, self.seq_len)
        complete = np.array([
            slc.stop - slc.start == self.seq_len and
            self._trading_index[slc.stop - 1] - self._trading_index[slc.start] == self.seq_len - 1
            for slc in slices
        ], dtype=bool)
        self._eligible &= complete
        self.batch_slices = slices[self._eligible]
        self.daily_slices = self._daily_slices(self.batch_slices)

    def _daily_slices(self, slices):
        by_date = {}
        for slc in slices:
            date = self._index[slc.stop - 1][1]
            by_date.setdefault(date, []).append(slc)
        return [by_date[date] for date in sorted(by_date)]

    def _prepare_seg(self, slc, **kwargs):
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError("This type of input is not supported")
        start_date = pd.Timestamp(start) if start is not None else self._calendar[0]
        end_date = pd.Timestamp(stop) if stop is not None else self._calendar[-1]
        obj = copy.copy(self)
        # Qlib's Serializable can omit private data when copying an object.
        for name in ("_data", "_label", "_index", "_timestamp", "_predicate_cache",
                     "_eligible", "_trading_index", "_target_dates", "_calendar", "_training"):
            setattr(obj, name, getattr(self, name))
        obj.batch_slices = np.array([
            batch_slc for batch_slc in self.batch_slices
            if start_date <= self._index[batch_slc.stop - 1][1] <= end_date
            and self._target_dates[batch_slc.stop - 1] <= end_date.to_datetime64()
        ], dtype=object)
        obj.daily_slices = obj._daily_slices(obj.batch_slices)
        return obj

    def restore_index(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    def assign_data(self, index, vals):
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -1. :] = vals

    def clear_memory(self):
        self._data[:, -1 :] = 0

    def train(self):
        """Enable training batches and training-only predicate retrieval."""
        self.batch_size, self.drop_last, self.shuffle = self.params
        self._training = True

    def eval(self):
        """Keep every evaluation sample; retrieve only Feature descriptors."""
        self.drop_last = False
        self.shuffle = False
        self._training = False

    def _get_slices(self):
        if self.cross_sectional:
            return self.daily_slices.copy(), self.days_per_batch
        if self.batch_size < 0:
            return self.daily_slices.copy(), -self.batch_size
        return self.batch_slices.copy(), self.batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)
        for i in range(0, len(slices), batch_size):
            if self.drop_last and i + batch_size > len(slices):
                break
            slices_subset = slices[i:i + batch_size]
            if self.cross_sectional or self.batch_size < 0:
                slices_subset = [slc for daily in slices_subset for slc in daily]
            data, label, index = [], [], []
            for slc in slices_subset:
                sample = self._data[slc].clone() if self.pin_memory else self._data[slc].copy()
                if len(sample) != self.seq_len:
                    padding = self.zeros[:self.seq_len - len(sample)]
                    sample = torch.cat([padding, sample], dim=0) if self.pin_memory else np.concatenate([padding, sample])
                sample[-self.horizon:, -1:] = 0
                data.append(sample)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)
            if self.pin_memory:
                data, label = torch.stack(data), torch.stack(label)
            else:
                data, label = _to_tensor(np.stack(data)), _to_tensor(np.stack(label))
            positions = np.asarray(index, dtype=np.int64)
            predicates = {}
            if self.injection_mode == "feature" or (self.injection_mode == "tsil" and self._training):
                predicates = {name: _to_tensor(values[positions])
                              for name, values in self._predicate_cache.items()}
            yield {
                "data": data, "label": label,
                "index": torch.as_tensor(positions, dtype=torch.long, device=device),
                "date": torch.as_tensor(self._trading_index[positions], dtype=torch.long, device=device),
                "predicates": predicates, "timestamp_feature": None,
            }
