"""Raw temporal descriptors from Section 3.3 of the original TSIL manuscript.

Inputs are unnormalised prices/realised returns, calendar categories and the
provider's trading-day index. Alpha158 factors are never used as a proxy.
Batch Frobenius normalisation belongs to the loss, not this module.
"""
import torch
import torch.nn.functional as F

TEMPORAL_PHI_TYPES = {
    "phi_timestamp", "phi_cycle", "phi_phase", "phi_momentum",
    "phi_volatility", "phi_return_dist",
}
DEFAULT_PREDICATE_CONFIG = {
    "momentum_windows": [5, 10, 20], "volatility_windows": [10, 20, 40],
    "cycle_window": 20, "phase_window": 20, "distribution_window": 20,
}


def selected_predicates(phi_type):
    if phi_type is None or phi_type == "mse":
        return []
    names = [phi_type] if isinstance(phi_type, str) else list(phi_type)
    if not names or len(set(names)) != len(names):
        raise ValueError("supply distinct predicates, or 'mse'")
    unknown = set(names) - TEMPORAL_PHI_TYPES - {"ones"}
    if unknown:
        raise ValueError(f"unknown predicate(s): {sorted(unknown)}; use canonical paper names")
    return names


def descriptor_dimensions(seq_len=20, predicate_config=None):
    cfg = {**DEFAULT_PREDICATE_CONFIG, **(predicate_config or {})}
    return {"ones": 1, "phi_timestamp": 19, "phi_cycle": 2,
            "phi_phase": seq_len, "phi_momentum": len(cfg["momentum_windows"]),
            "phi_volatility": len(cfg["volatility_windows"]), "phi_return_dist": 4}


def _history(x, length, name, positive=False):
    if length < 1 or x is None or x.ndim != 2 or x.shape[1] < length:
        raise ValueError(f"{name} requires [B, history] with at least {length} observations")
    x = x[:, -length:]
    if not x.is_floating_point():
        x = x.float()
    if not torch.isfinite(x).all() or (positive and not (x > 0).all()):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return x


def _trading_index(q, reference):
    if q is None or q.ndim != 1 or q.shape[0] != reference.shape[0]:
        raise ValueError("trading_index must contain one trading-day index per sample")
    q = q.to(reference)
    if not torch.isfinite(q).all() or not torch.equal(q, q.floor()):
        raise ValueError("trading_index must be finite and integer-valued")
    return q


def _dominant_period(returns):
    if returns.shape[1] < 2:
        raise ValueError("FFT window needs at least two returns")
    # Exclude DC. A flat spectrum deterministically selects k=1, one member
    # of the argmax set allowed by the paper (not evidence of a real cycle).
    spectrum = torch.fft.rfft(returns, dim=1).abs()[:, 1:]
    k = spectrum.argmax(dim=1) + 1
    return returns.shape[1] / k.to(returns.dtype)


@torch.no_grad()
def get_temporal_phi(phi_type, *, close=None, returns=None, market_returns=None,
                     trading_index=None, timestamp_x=None, seq_len=20,
                     momentum_windows=(5, 10, 20), volatility_windows=(10, 20, 40),
                     cycle_window=20, phase_window=20, distribution_window=20):
    """Return a raw [B, d_s] descriptor. Histories end at prediction day t.

    returns = realised p_t/p_(t-1)-1, never the future target;
    timestamp_x = [weekday (0..6), month (0..11)]. Window lengths are
    independent of the backbone lookback seq_len.
    """
    if phi_type == "phi_timestamp":
        if timestamp_x is None or timestamp_x.ndim != 2 or timestamp_x.shape[1] != 2:
            raise ValueError("timestamp_x must be [B, 2]: zero-based weekday and month")
        if not torch.isfinite(timestamp_x).all() or not torch.equal(timestamp_x, timestamp_x.long().to(timestamp_x)):
            raise ValueError("calendar categories must be integer-valued")
        attrs = timestamp_x.long()
        if not ((attrs >= 0).all() and (attrs[:, 0] < 7).all() and (attrs[:, 1] < 12).all()):
            raise ValueError("calendar categories outside weekday/month ranges")
        return torch.cat([F.one_hot(attrs[:, 0], 7), F.one_hot(attrs[:, 1], 12)], dim=1).float()
    if phi_type == "phi_cycle":
        r = _history(market_returns, cycle_window, "market_returns")
        q = _trading_index(trading_index, r)
        period = _dominant_period(r)
        angle = 2 * torch.pi * torch.remainder(q, period) / period
        return torch.stack([angle.sin(), angle.cos()], dim=1)
    if phi_type == "phi_phase":
        r = _history(returns, phase_window, "returns")
        q = _trading_index(trading_index, r)
        period = _dominant_period(r)[:, None]
        positions = q[:, None] - torch.arange(seq_len - 1, -1, -1, device=r.device, dtype=r.dtype)
        return torch.remainder(positions, period) / period
    if phi_type == "phi_momentum":
        if not momentum_windows or any(k < 1 for k in momentum_windows):
            raise ValueError("momentum windows must be positive")
        p = _history(close, max(momentum_windows) + 1, "close", positive=True)
        return torch.stack([torch.log(p[:, -1]) - torch.log(p[:, -1-k])
                            for k in momentum_windows], dim=1)
    if phi_type == "phi_volatility":
        if not volatility_windows or any(w < 1 for w in volatility_windows):
            raise ValueError("volatility windows must be positive")
        r = _history(returns, max(volatility_windows), "returns")
        return torch.stack([r[:, -w:].std(dim=1, unbiased=False) for w in volatility_windows], dim=1)
    if phi_type == "phi_return_dist":
        r = _history(returns, distribution_window, "returns")
        mean = r.mean(dim=1, keepdim=True)
        std = r.std(dim=1, keepdim=True, unbiased=False)
        z = (r - mean) / (std + 1e-8)
        return torch.cat([mean, std, z.pow(3).mean(dim=1, keepdim=True),
                          z.pow(4).mean(dim=1, keepdim=True) - 3], dim=1)
    raise KeyError(f"unknown temporal predicate: {phi_type}")


def get_temporal_I_P(phi_type, batch_x=None, timestamp_x=None, device=None, **raw_signals):
    """Diagnostic Gram form only; never used by the training loss.

    Reject the old Alpha158-tensor interface rather than reinterpret factors
    as prices or returns. Callers must supply named raw signals.
    """
    if batch_x is not None:
        raise ValueError("pass explicit close/returns/market_returns; Alpha158 is not a raw signal")
    phi = get_temporal_phi(phi_type, timestamp_x=timestamp_x, **raw_signals)
    if device is not None:
        phi = phi.to(device)
    phi = phi / (torch.linalg.vector_norm(phi) + 1e-8)
    return torch.eye(phi.shape[0], device=phi.device, dtype=phi.dtype), phi @ phi.T
