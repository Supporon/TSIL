# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import FilterCol, RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm
from qlib.data import D

from src.timefeatures import time_features

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    # assert index.is_lexsorted(), "index should be sorted"

    # number of dates for each code
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0).size().values

    # start_index for each code
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # all the [start, stop) indices of features
    # features btw [start, stop) are used to predict the `stop - 1` label
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices)

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn

import copy
import numpy as np
import torch
from collections import defaultdict
from qlib.data.dataset import DatasetH
# from qlib.data.dataset.utils import _get_date_parse_fn, _to_tensor


class MTSDatasetH(DatasetH):
    """Memory Augmented Time Series Dataset.

    The dataset adapts what each batch carries to the ``phi_type`` of the loss
    (passed through from ``model_config`` by ``train.py`` / ``search.py``):

        phi_type = None / 'mse' / 'ones' / 'phi_cycle' / 'phi_phase' /
                   'phi_momentum' / 'phi_volatility' / 'phi_return_dist' / ...
            -> calendar/timestamp features are irrelevant; they are NOT built or
               yielded. Each batch is ``{data, label, index, timestamp_feature=None}``.

        phi_type startswith 'phi_timestamp'  (the Timestamp predicate Phi_ts)
            -> calendar features ARE extracted (``timestamp_feature_get``) and
               yielded per batch so the loss can build ``P`` from them. Their
               encoding is controlled by ``embed`` / ``ts_onehot``:
                 embed='timeF'              -> continuous time features (timeenc=1)
                 embed!='timeF', ts_onehot  -> one-hot day-of-week/month/year
                 embed!='timeF', not onehot -> integer day-of-week/month/year (default)

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon (to mask historical loss for TRA)
        batch_size (int): batch size (<0 means daily batch)
        shuffle (bool): whether shuffle data
        pin_memory (bool): whether pin data to gpu memory
        drop_last (bool): whether drop last batch < batch_size
        embed (str): calendar encoding ('fixed' | 'learned' | 'timeF'); only used
            when ``phi_type`` is the timestamp predicate.
        ts_onehot (bool): one-hot the calendar features; only used when ``phi_type``
            is the timestamp predicate.
        phi_type (str): the loss predicate selector; decides whether timestamp
            features are needed (see above).
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
        account=100000000,#1000000, #
        benchmark="SH000300",
        freq='day',
        embed='fixed',
        ts_onehot=False,
        phi_type=None,
        **kwargs,
    ):
        # print(f"MTSDatasetH init kwargs: {kwargs}")  # debug output

        assert horizon > 0, "please specify `horizon` to avoid data leakage"

        self.handler_kwargs = handler['kwargs']
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)  # for train/eval switch

        self.account = account
        self.benchmark = benchmark

        self.freq = freq
        self.embed = embed
        self.ts_onehot = ts_onehot

        # Calendar/timestamp features are extracted ONLY when the loss uses the
        # timestamp predicate (phi_type starting with 'phi_timestamp'). For every
        # other phi_type (mse / ones / cycle / phase / momentum / volatility /
        # return_dist / ...) timestamps are irrelevant and are not built, carried,
        # or yielded.
        self.phi_type = phi_type
        self.use_timestamp = bool(phi_type) and ("phi_timestamp" in str(phi_type))

        # self.freq = "day"
        # self.embed = 'fixed' # 'fixed' 'learned'  'timeF'
        # self.ts_onehot = True

        self.timeenc = 0 if self.embed != 'timeF' else 1

        self.executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }
        self.backtest_config = {
            "start_time": segments['test'][0],
            "end_time": segments['test'][1],
            "account": self.account,
            "benchmark": self.benchmark, # default
            "exchange_kwargs": {
                "freq": self.freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }
        # kwargs = self.handler_kwargs
        # super().__init__(handler, segments, self.handler_kwargs, **kwargs)
        # super().__init__(handler, segments, self.handler_kwargs)
        super().__init__(handler, segments, **kwargs)

    def to_one_hot(self, x, num_class):
        return np.eye(num_class, dtype=int)[x]

    def timestamp_feature_get(self):
        freq = 'D' if self.freq == 'day' else self.freq
        df_stamp = pd.DataFrame(self._index.tolist(), columns=self._index.names)
        if self.timeenc == 0:
            if self.ts_onehot:
                dow = df_stamp['datetime'].dt.weekday  # 0–6
                dom = df_stamp['datetime'].dt.day  # 1–31
                doy = df_stamp['datetime'].dt.dayofyear  # 1–366

                dow_oh_cols = [f'dow_{i}' for i in range(7)]
                dom_oh_cols = [f'dom_{i + 1}' for i in range(31)]
                doy_oh_cols = [f'doy_{i + 1}' for i in range(366)]

                df_stamp[dow_oh_cols] = self.to_one_hot(dow, 7)
                df_stamp[dom_oh_cols] = self.to_one_hot(dom - 1, 31)  # 1–31 → 0–30
                df_stamp[doy_oh_cols] = self.to_one_hot(doy - 1, 366)  # 1–366 → 0–365
            else:
                df_stamp['dayofweek'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
                df_stamp['dayofmonth'] = df_stamp.datetime.apply(lambda row: row.day, 1)
                df_stamp['dayofyear'] = df_stamp.datetime.apply(lambda row: row.timetuple().tm_yday, 1)

                # df_stamp['weekofyear'] = df_stamp.datetime.apply(lambda row: row.isocalendar()[1], 1)
                # df_stamp['quarterofyear'] = df_stamp.datetime.apply(lambda row: (row.month-1)//3 + 1, 1)
                # df_stamp['monthofyear'] = df_stamp.datetime.apply(lambda row: row.month, 1)
                # df_stamp['year'] = df_stamp.datetime.apply(lambda row: row.year, 1)
            data_stamp = df_stamp.drop(['instrument','datetime'], axis=1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data_stamp

    def setup_data(self, handler_kwargs: dict = None, **kwargs):

        super().setup_data()

        # change index to <code, date>
        # NOTE: we will use inplace sort to reduce memory use
        # df = self.handler._data
        if 'type' not in kwargs:
            df = self.handler._learn #_infer——_data
            df.index = df.index.swaplevel()
        elif 'type' in kwargs and kwargs['type'] == 'DK_L':
            df = self.handler._learn #_learn——_data
        else:
            df = self.handler._learn
            df.index = df.index.swaplevel() # stock first

        # df = self.handler._learn
        # df.index = df.index.swaplevel()
        df.sort_index(inplace=True)
        # fea = [col for col in df.columns if col != 'label']
        fea = [col for col in df.columns if 'label' not in col]

        self._data = df[fea].squeeze().astype("float32")
        self._label = df[['label']].squeeze().astype("float32")
        self._index = df.index

        # extract calendar/timestamp features only for the timestamp predicate
        if self.use_timestamp:
            self._timestamp = self.timestamp_feature_get()  # add
        else:
            self._timestamp = None

        # add memory to feature
        self._data = np.c_[self._data, np.zeros((len(self._data), 1), dtype=np.float32)]

        # padding tensor
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)
        if self.use_timestamp:
            self.zeros_timestamp = np.zeros((self.seq_len, self._timestamp.shape[1]), dtype=np.float32)
        else:
            self.zeros_timestamp = None

        # pin memory
        if self.pin_memory:
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            self.zeros = _to_tensor(self.zeros)
            if self.use_timestamp:
                self._timestamp = _to_tensor(self._timestamp) # add
                self.zeros_timestamp = _to_tensor(self.zeros_timestamp) # add

        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

        # create daily slices
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))}
        for i, (code, date) in enumerate(act_index):
            daily_slices[date].append(self.batch_slices[i])
        self.daily_slices = list(daily_slices.values())

        # self.stock_slices = {}
        # for stock in sorted(act_index.unique(level=0)):
        #     # select all slices for the current stock
        #     mask = act_index.get_level_values(0) == stock
        #     self.stock_slices[stock] = [self.batch_slices[i] for i in np.where(mask)[0]]
        # self.stock_slices = list(self.stock_slices.values())  # convert to list

    def _prepare_seg(self, slc, **kwargs):

        fn = _get_date_parse_fn(self._index[0][1])

        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc # time split
        else:
            raise NotImplementedError(f"This type of input is not supported")
        start_date = fn(start)
        end_date = fn(stop)
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data # ordered by stock
        obj._label = self._label # ordered by stock
        obj._index = self._index # ordered by stock
        obj._timestamp = self._timestamp # add
        new_batch_slices = []
        for batch_slc in self.batch_slices:
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc) # split train/val/test by time
        obj.batch_slices = np.array(new_batch_slices)

        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices

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

    # TODO: better train/eval mode design
    def train(self):
        """enable traning mode"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""

        # self.batch_size = -1
        # self.drop_last = False
        self.drop_last = True

        self.shuffle = False

    def _get_slices(self):
        # if self.batch_size < 0:
        #     slices = self.daily_slices.copy()
        #     batch_size = -1 * self.batch_size
        # else:
        slices = self.batch_slices.copy()
        batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # get slices for this batch
            slices_subset = slices[i : i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # collect data
            data = []
            label = []
            index = []
            timestamp_feature = [] if self.use_timestamp else None
            for slc in slices_subset:
                _data = self._data[slc].clone() if self.pin_memory else self._data[slc].copy()
                if self.use_timestamp:
                    _timestamp = self._timestamp[slc].clone() if self.pin_memory else self._timestamp[slc].copy()
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                        if self.use_timestamp:
                            _timestamp = torch.cat([self.zeros_timestamp[: self.seq_len - len(_timestamp)], _timestamp], axis=0) # add
                    else:
                        _data = np.concatenate([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                        if self.use_timestamp:
                            _timestamp = np.concatenate([self.zeros_timestamp[: self.seq_len - len(_timestamp)], _timestamp], axis=0) # add
                _data[-self.horizon :, -1 :] = 0
                data.append(_data)
                if self.use_timestamp:
                    timestamp_feature.append(_timestamp) # add
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)


            # concate
            index = torch.tensor(index, device=device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
                if self.use_timestamp:
                    timestamp_feature = torch.stack(timestamp_feature)
            else:
                data = _to_tensor(np.stack(data))
                label = _to_tensor(np.stack(label))
                if self.use_timestamp:
                    timestamp_feature = _to_tensor(np.stack(timestamp_feature))
            # yield -> generator

            yield {"data": data, "label": label, "index": index, 'timestamp_feature': timestamp_feature}
