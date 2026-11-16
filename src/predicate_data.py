"""Causal raw-price descriptors, aligned to the unfiltered feature-row index."""
import numpy as np
import pandas as pd
import torch

from src.temporal_predicates import (
    DEFAULT_PREDICATE_CONFIG, descriptor_dimensions, get_temporal_phi, selected_predicates,
)


def required_history(seq_len=20, predicate_config=None):
    """Number of past returns required in every experimental mode."""
    cfg = {**DEFAULT_PREDICATE_CONFIG, **(predicate_config or {})}
    windows = [seq_len, cfg["cycle_window"], cfg["phase_window"],
               cfg["distribution_window"], *cfg["momentum_windows"], *cfg["volatility_windows"]]
    if not cfg["momentum_windows"] or not cfg["volatility_windows"]:
        raise ValueError("momentum and volatility windows must not be empty")
    if any(not isinstance(w, (int, np.integer)) or w < 1 for w in windows):
        raise ValueError("lookback windows must be positive integers")
    return max(windows)


def build_predicate_cache(close_wide_dataframe, sample_index, phi_types,
                          seq_len=20, predicate_config=None, market_membership=None):
    """Return (raw descriptors, eligibility, trading index) in sample_index order.

    close_wide_dataframe is indexed by the complete provider calendar through
    the data end; its first date defines q=0. Missing prices remain missing.
    market_membership optionally marks membership on each return's own date,
    so the market mean excludes stocks outside the universe at that date.
    Eligibility is identical for Original, Feature and TSIL, even when no
    descriptors are requested. Only sample endpoints are filtered by callers.
    """
    cfg = {**DEFAULT_PREDICATE_CONFIG, **(predicate_config or {})}
    history = required_history(seq_len, cfg)
    names = selected_predicates(phi_types or None)
    dimensions = descriptor_dimensions(seq_len, cfg)
    descriptors = {name: np.zeros((len(sample_index), dimensions[name]), dtype=np.float32)
                   for name in names}
    close = close_wide_dataframe.sort_index()
    calendar = pd.DatetimeIndex(close.index)
    q = calendar.get_indexer(pd.DatetimeIndex(sample_index.get_level_values("datetime")))
    stocks = close.columns.get_indexer(sample_index.get_level_values("instrument"))
    eligible = np.zeros(len(sample_index), dtype=bool)
    if close.empty or len(calendar) <= history:
        return descriptors, eligible, q

    values = close.to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    returns = np.full_like(values, np.nan)
    valid_pairs = valid[1:] & valid[:-1]
    np.divide(values[1:], values[:-1], out=returns[1:], where=valid_pairs)
    returns[1:] -= 1
    members = np.ones_like(valid)
    if market_membership is not None:
        members = market_membership.reindex(index=close.index, columns=close.columns,
                                            fill_value=False).fillna(False).to_numpy(dtype=bool)
    available = np.isfinite(returns) & members
    counts = available.sum(axis=1)
    market = np.full(len(calendar), np.nan)
    np.divide(np.where(available, returns, 0).sum(axis=1), counts,
              out=market, where=counts > 0)

    # Rolling counts preserve actual calendar gaps instead of filling prices.
    bad_close = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int64),
                           np.cumsum(~valid, axis=0)])
    history_ok = np.zeros_like(valid)
    history_ok[history:] = bad_close[history+1:] - bad_close[:-history-1] == 0
    cycle_window = cfg["cycle_window"]
    bad_market = np.r_[0, np.cumsum(~np.isfinite(market))]
    market_ok = np.zeros(len(calendar), dtype=bool)
    market_ok[cycle_window-1:] = bad_market[cycle_window:] - bad_market[:-cycle_window] == 0
    mapped = (q >= 0) & (stocks >= 0)
    eligible[mapped] = history_ok[q[mapped], stocks[mapped]] & market_ok[q[mapped]]

    # Bound temporary [B, history] arrays; the cache itself contains only [N,d].
    positions = np.flatnonzero(eligible)
    for start in range(0, len(positions), 4096) if names else ():
        rows = positions[start:start+4096]
        dates, instruments = q[rows], stocks[rows]
        raw_close = values[dates[:, None] - np.arange(history, -1, -1), instruments[:, None]]
        raw_returns = returns[dates[:, None] - np.arange(history-1, -1, -1), instruments[:, None]]
        market_returns = market[dates[:, None] - np.arange(cycle_window-1, -1, -1)]
        timestamps = calendar.take(dates)
        signals = {
            "close": torch.from_numpy(raw_close), "returns": torch.from_numpy(raw_returns),
            "market_returns": torch.from_numpy(market_returns),
            "trading_index": torch.from_numpy(dates.astype(np.float64)),
            "timestamp_x": torch.from_numpy(np.column_stack([timestamps.weekday, timestamps.month-1])),
        }
        for name in names:
            if name == "ones":
                descriptors[name][rows] = 1
            else:
                descriptors[name][rows] = get_temporal_phi(name, seq_len=seq_len, **cfg,
                                                          **signals).cpu().numpy()
    return descriptors, eligible, q
