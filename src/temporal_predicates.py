"""
temporal_predicates.py
======================

The **six** Temporal Predicates (Φ) of the TSIL paper, gathered into a single
self-contained module.

Paper (abstract): *"we instantiate six computable, domain-grounded predicates
that capture timestamp effects, market cycles, asset phases, momentum,
volatility, and return-distribution shape"*.

Each predicate maps a mini-batch to a per-sample descriptor ``φ`` (shape
``[B, d]``), L2-normalizes it, and builds the projection matrix
``P = φ̃ φ̃ᵀ`` (shape ``[B, B]``) used by the statistical-invariant loss
``loss = τ̂·I·e² + τ·P·e²``. Every function returns ``(I, P)``.

These six were consolidated into this module (the canonical home for the paper's
temporal predicates):

    | Paper predicate            | formula                          | descriptor source             |
    | -------------------------- | -------------------------------- | ----------------------------- |
    | Φ_ts    Timestamp          | OneHot(calendar features)        | get_timestamp_I_P             |
    | Φ_cycle Market Cycle       | ⌈i / τ_batch⌉                    | get_cycle_I_P (batch FFT)     |
    | Φ_phase Asset Phase        | [⌈1/τ_i⌉, …, ⌈T/τ_i⌉]            | get_phase_I_P (per-sample FFT)|
    | Φ_mom   Momentum           | multi-horizon relative returns   | get_momentum_I_P              |
    | Φ_vol   Volatility         | multi-window rolling std         | get_volatility_I_P            |
    | Φ_dist  Return-Distribution| [μ, σ, γ, κ] over the sequence   | get_return_distribution_I_P   |

NOTE on Φ_dist: the paper's form is ``[μ, σ, γ, κ]`` of the cross-feature-averaged
sequence, implemented here in ``get_return_distribution_I_P`` (the same descriptor
that ``src/custom_phi_constructors.py`` builds for ``phi_return_dist``).

Public API
----------
- ``TEMPORAL_PHI_TYPES``            : set of phi_type keys handled here
- ``get_temporal_I_P(phi_type, batch_x, timestamp_x=None, device=None)`` : dispatcher
- the six ``get_<name>_I_P`` builders (callable directly)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers (kept local so this module has no dependency on the backbone file)
# ---------------------------------------------------------------------------

def _normalize_phi(phi: torch.Tensor) -> torch.Tensor:
    """L2-normalize the whole descriptor matrix (matches Φ̃ = Φ / ||Φ|| in the paper)."""
    return phi / (torch.linalg.norm(phi) + 1e-8)


def _build_P(phi: torch.Tensor) -> torch.Tensor:
    """P = φ̃ φ̃ᵀ."""
    return torch.mm(phi, phi.T)


def skewness(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    return torch.mean(((x - mean) / (std + 1e-8)) ** 3, dim=dim)


def torch_kurtosis(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Excess kurtosis (already minus 3)."""
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    return torch.mean(((x - mean) / (std + 1e-8)) ** 4, dim=dim) - 3


def FFT_for_Period_batch(x: torch.Tensor, k: int = 2):
    """Dominant period(s) of the *whole batch* (used by the Market-Cycle predicate)."""
    xf = torch.fft.rfft(x, dim=0)  # [.., F, ..]
    frequency_list = abs(xf).mean(1).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[0] // top_list
    return period, abs(xf).mean(-1)


def FFT_for_Period_batch_per_sample(x: torch.Tensor, k: int = 1):
    """Dominant period of *each sample* (used by the Asset-Phase predicate)."""
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)            # FFT along time -> [B, T//2+1, C]
    amplitude = abs(xf)
    amplitude_mean = amplitude.mean(-1)      # average over features -> [B, T//2+1]
    amplitude_mean[:, 0] = 0                  # drop DC component
    _, top_indices = torch.topk(amplitude_mean, k, dim=1)  # [B, k]
    seq_len = x.shape[1]
    periods = seq_len / top_indices.float()  # [B, k]
    return periods, abs(xf).mean(-1)


# ===========================================================================
# 1. Φ_ts — Timestamp predicate  (Calendar Anomalies / Seasonality)
#    φ_ts,i = [OneHot(x^(1)_{i,t}), ..., OneHot(x^(C)_{i,t})]^T   (Absolute Time)
# ===========================================================================

def get_timestamp_I_P(timestamp_x: torch.Tensor,
                      device: Optional[torch.device] = None,
                      variant: str = "last") -> Tuple[torch.Tensor, torch.Tensor]:
    """Timestamp / calendar predicate.

    Args:
        timestamp_x: calendar-feature tensor, shape [B, seq_len, n_time_feats].
        variant:
            'last' (default) -> use the calendar features of the last step;
            'dim'            -> flatten all steps row-wise (per-step features concatenated);
            'sl'             -> flatten all steps column-wise (per-feature series concatenated).
    """
    if device is None:
        device = timestamp_x.device
    batch_size = timestamp_x.shape[0]
    I = torch.eye(batch_size, device=device)

    if variant == "dim":
        phi = timestamp_x.reshape(batch_size, -1)
    elif variant == "sl":
        phi = timestamp_x.permute(0, 2, 1).contiguous().reshape(batch_size, -1)
    else:  # 'last'
        phi = timestamp_x[:, -1, :]

    phi = _normalize_phi(phi)
    return I, _build_P(phi)


# ===========================================================================
# 2. Φ_cycle — Market Cycle predicate  (Market State Cycles)
#    φ_cycle,i = ⌈i / τ_batch⌉   (Batch Length; τ_batch = dominant batch period)
# ===========================================================================

def get_cycle_I_P(batch_x: torch.Tensor,
                  device: Optional[torch.device] = None,
                  onehot: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Market-cycle predicate: position of each sample within the batch's dominant period."""
    if device is None:
        device = batch_x.device
    batch_size = batch_x.shape[0]
    I = torch.eye(batch_size, device=device)

    period_list, _ = FFT_for_Period_batch(batch_x, 1)
    p0 = period_list[0]
    if isinstance(p0, (int, float, np.integer)):
        period_value = torch.tensor(p0, device=device, dtype=torch.float32)
    else:
        period_value = (p0.to(device).float() if hasattr(p0, "to")
                        else torch.tensor(p0, device=device, dtype=torch.float32))
    period_value = torch.clamp(period_value, min=1.0)

    time_indices = torch.arange(1, batch_size + 1, device=device)
    period_indices = torch.ceil(time_indices / (period_value + 1e-8)).long()
    period_indices = torch.clamp(period_indices, min=1)

    if onehot:
        num_classes = period_indices.max().item()
        phi = F.one_hot(period_indices - 1, num_classes=num_classes).float()
    else:
        phi = period_indices.unsqueeze(dim=1).float()  # [B, 1]

    phi = _normalize_phi(phi)
    return I, _build_P(phi)


# ===========================================================================
# 3. Φ_phase — Asset Phase predicate  (Intrinsic Asset Rhythm)
#    φ_phase,i = [⌈1/τ_i⌉, ..., ⌈T/τ_i⌉]^T   (Sequence Length; τ_i = per-sample period)
# ===========================================================================

def get_phase_I_P(batch_x: torch.Tensor,
                  device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asset-phase predicate: per-sample cyclical position over the sequence."""
    if device is None:
        device = batch_x.device
    batch_size, seq_len, _ = batch_x.shape
    I = torch.eye(batch_size, device=device)

    periods, _ = FFT_for_Period_batch_per_sample(batch_x, k=1)  # [B, 1]
    time_indices = torch.arange(1, seq_len + 1, device=device)   # [T]

    periods_expanded = periods.unsqueeze(-1)                     # [B, 1, 1]
    time_indices_expanded = time_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    period_indices = torch.ceil(time_indices_expanded / (periods_expanded + 1e-8))  # [B, 1, T]
    phi = period_indices.squeeze(1)                              # [B, T]

    phi = _normalize_phi(phi)
    if phi.device != device:
        phi = phi.to(device)
    return I, _build_P(phi)


# ===========================================================================
# 4. Φ_mom — Momentum predicate  (Trend Persistence)
#    φ_mom,i = [(x_t - x_{t-k1})/(|x_{t-k1}|+ϵ), ..., (x_t - x_{t-kM})/(|x_{t-kM}|+ϵ)]
# ===========================================================================

def get_momentum_I_P(batch_x: torch.Tensor,
                     lookback_windows: list = [5, 10, 20],
                     device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-horizon momentum predicate."""
    batch_size, seq_len, _ = batch_x.shape
    if device is None:
        device = batch_x.device
    I = torch.eye(batch_size, device=device)

    momentum_features = []
    for window in lookback_windows:
        if window < seq_len:
            current = batch_x[:, -1, :]
            past = batch_x[:, -1 - window, :]
            momentum = (current - past) / (torch.abs(past) + 1e-8)
            momentum = torch.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
            momentum_features.append(momentum.mean(dim=1, keepdim=True))  # [B, 1]

    if not momentum_features:  # seq too short -> fall back to last-step mean
        momentum_features = [batch_x[:, -1, :].mean(dim=1, keepdim=True)]

    phi = torch.cat(momentum_features, dim=1)  # [B, len(windows)]
    phi = torch.nan_to_num(phi, nan=0.0)
    phi = _normalize_phi(phi)
    return I, _build_P(phi)


# ===========================================================================
# 5. Φ_vol — Volatility predicate  (Risk Regime Clustering)
#    φ_vol,i = [σ_{i,w1}, ..., σ_{i,wJ}]^T
# ===========================================================================

def get_volatility_I_P(batch_x: torch.Tensor,
                       window_sizes: list = [10, 20, 40],
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-window volatility predicate."""
    batch_size, seq_len, _ = batch_x.shape
    if device is None:
        device = batch_x.device
    I = torch.eye(batch_size, device=device)

    vol_features = []
    for window in window_sizes:
        if window <= seq_len:
            window_data = batch_x[:, -window:, :]
            volatility = window_data.std(dim=1)             # [B, feature_dim]
            vol_features.append(volatility.mean(dim=1, keepdim=True))  # [B, 1]

    if not vol_features:  # all windows longer than seq -> use full-sequence std
        vol_features = [batch_x.std(dim=1).mean(dim=1, keepdim=True)]

    phi = torch.cat(vol_features, dim=1)  # [B, len(windows)]
    phi = torch.nan_to_num(phi, nan=0.0)
    phi = _normalize_phi(phi)
    return I, _build_P(phi)


# ===========================================================================
# 6. Φ_dist — Return-Distribution predicate  (Return Generating Process)
#    φ_dist,i = [μ_i, σ_i, γ_i, κ_i]^T   (mean, std, skew, excess-kurt over the sequence)
# ===========================================================================

def get_return_distribution_I_P(batch_x: torch.Tensor,
                                device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """First four moments of the cross-feature-averaged sequence."""
    batch_size, seq_len, _ = batch_x.shape
    if device is None:
        device = batch_x.device
    I = torch.eye(batch_size, device=device)

    x = batch_x.mean(dim=2)                       # [B, seq_len]
    mean = x.mean(dim=1, keepdim=True)            # [B, 1]
    std = x.std(dim=1, keepdim=True)              # [B, 1]
    x_norm = (x - mean) / (std + 1e-8)
    skew = (x_norm ** 3).mean(dim=1, keepdim=True)
    kurt = (x_norm ** 4).mean(dim=1, keepdim=True) - 3

    skew = torch.nan_to_num(skew, nan=0.0)
    kurt = torch.nan_to_num(kurt, nan=0.0)

    phi = torch.cat([mean, std, skew, kurt], dim=1)  # [B, 4]
    phi = torch.nan_to_num(phi, nan=0.0)
    phi = _normalize_phi(phi)
    return I, _build_P(phi)


# ===========================================================================
# unified dispatcher
# ===========================================================================

# canonical keys + backward-compatible aliases that route to the six predicates
TEMPORAL_PHI_TYPES = {
    # Φ_ts
    "phi_timestamp", "phi_timestamp_last", "phi_timestamp_dim", "phi_timestamp_sl",
    # Φ_cycle
    "phi_cycle", "phi_cycle_onehot",
    # Φ_phase
    "phi_phase",
    # Φ_mom / Φ_vol / Φ_dist
    "phi_momentum", "phi_volatility", "phi_return_dist",
    # legacy aliases (route to the cycle / phase predicates)
    "phi_period", "phi_period_onehot", "phi_period_per_sample",
}


def get_temporal_I_P(phi_type: str,
                     batch_x: torch.Tensor,
                     timestamp_x: Optional[torch.Tensor] = None,
                     device: Optional[torch.device] = None,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(I, P)`` for one of the six paper temporal predicates.

    Args:
        phi_type:    one of ``TEMPORAL_PHI_TYPES``.
        batch_x:     feature tensor [B, seq_len, feature_dim].
        timestamp_x: calendar-feature tensor [B, seq_len, n_time_feats];
                     required only for the timestamp predicate.
        device:      compute device (defaults to ``batch_x.device``).
    """
    if device is None:
        device = batch_x.device

    # --- Φ_ts ---
    if phi_type.startswith("phi_timestamp"):
        if timestamp_x is None:
            raise ValueError(
                f"phi_type='{phi_type}' needs timestamp_x but none was provided.")
        variant = ("dim" if phi_type.endswith("_dim")
                   else "sl" if phi_type.endswith("_sl")
                   else "last")
        return get_timestamp_I_P(timestamp_x, device, variant)

    # --- Φ_phase --- (check 'per_sample'/'phase' before the cycle branch)
    if phi_type == "phi_phase" or "per_sample" in phi_type:
        return get_phase_I_P(batch_x, device)

    # --- Φ_cycle ---
    if phi_type in ("phi_cycle", "phi_cycle_onehot", "phi_period", "phi_period_onehot"):
        return get_cycle_I_P(batch_x, device, onehot=("onehot" in phi_type))

    # --- Φ_mom / Φ_vol / Φ_dist ---
    if phi_type == "phi_momentum":
        return get_momentum_I_P(batch_x, device=device)
    if phi_type == "phi_volatility":
        return get_volatility_I_P(batch_x, device=device)
    if phi_type == "phi_return_dist":
        return get_return_distribution_I_P(batch_x, device=device)

    raise KeyError(
        f"phi_type='{phi_type}' is not a temporal predicate. "
        f"Valid keys: {sorted(TEMPORAL_PHI_TYPES)}")


# ===========================================================================
# self-test
# ===========================================================================

if __name__ == "__main__":
    B, T, C = 32, 20, 158
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(B, T, C, device=dev)
    ts = torch.randn(B, T, 3, device=dev)

    keys = ["phi_timestamp", "phi_timestamp_dim", "phi_timestamp_sl",
            "phi_cycle", "phi_cycle_onehot", "phi_phase",
            "phi_momentum", "phi_volatility", "phi_return_dist",
            "phi_period", "phi_period_per_sample"]

    print("=" * 60)
    for k in keys:
        I, P = get_temporal_I_P(k, x, timestamp_x=ts, device=dev)
        sym = torch.allclose(P, P.T, atol=1e-5)
        print(f"{k:24s} I{tuple(I.shape)} P{tuple(P.shape)} "
              f"sym={sym} finite={torch.isfinite(P).all().item()}")
    print("=" * 60)
    print("All six temporal predicates OK.")
