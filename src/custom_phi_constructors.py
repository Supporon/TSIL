"""
Custom Phi Predicate Constructor Module

This module contains multiple phi construction methods based on stock-domain
knowledge, used to generate the P matrix and enhance the model's ability to
model latent relationships between samples.

Core role of the P matrix:
- Aggregates samples with similar characteristics during loss computation
- Constructs an inter-sample similarity matrix via P = phi @ phi.T
- The final loss is: loss = tau_hat * I * error + tau * P * error

Input data dimensions: (batch_size, sequence_length, feature_dim) = (32, 96, 158)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PhiConstructor:
    """
    Phi constructor base class, providing common helper methods
    """

    @staticmethod
    def normalize_phi(phi: torch.Tensor) -> torch.Tensor:
        """
        Apply L2 normalization to phi

        Args:
            phi: raw phi vector, shape [batch_size, d]

        Returns:
            the normalized phi vector
        """
        norm = torch.linalg.norm(phi)
        return phi / (norm + 1e-8)

    @staticmethod
    def construct_P(phi: torch.Tensor) -> torch.Tensor:
        """
        Construct the P matrix from phi

        Mathematical expression: P = phi @ phi.T

        Args:
            phi: normalized phi vector, shape [batch_size, d]

        Returns:
            P matrix, shape [batch_size, batch_size]
        """
        return torch.mm(phi, phi.T)

    @staticmethod
    def get_I(batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the identity matrix"""
        return torch.eye(batch_size, device=device)


# ============================================================================
# 1. Momentum Factor Predicate (Momentum Phi)
# ============================================================================

def get_momentum_I_P(batch_x: torch.Tensor,
                     lookback_windows: list = [5, 10, 20],
                     device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on momentum factors

    Motivation:
        The momentum effect is one of the most well-known anomalies in financial
        markets. Stocks with similar momentum characteristics tend to exhibit
        similar price movements; aggregating such samples can capture group behavior.

    Mathematical expression:
        momentum_k = (x_{t} - x_{t-k}) / x_{t-k}  (k-period momentum)
        phi = concat([momentum_5, momentum_10, momentum_20])
        phi = phi / ||phi||_2
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        lookback_windows: list of lookback windows for momentum computation
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Assume the first feature dimension is the closing price, or use the mean of all features
    # Compute momentum across multiple time windows
    momentum_features = []

    for window in lookback_windows:
        if window < seq_len:
            # Compute momentum: (current value - past value) / past value
            current = batch_x[:, -1, :]  # [batch_size, feature_dim]
            past = batch_x[:, -1-window, :]  # [batch_size, feature_dim]

            # Avoid division by zero
            momentum = (current - past) / (torch.abs(past) + 1e-8)
            momentum = torch.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)

            # Take the mean over all features as the momentum representation for this window
            momentum_avg = momentum.mean(dim=1, keepdim=True)  # [batch_size, 1]
            momentum_features.append(momentum_avg)

    # Concatenate all momentum features
    phi = torch.cat(momentum_features, dim=1)  # [batch_size, len(lookback_windows)]

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    # Construct the P matrix
    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 2. Volatility Clustering Predicate (Volatility Clustering Phi)
# ============================================================================

def get_volatility_I_P(batch_x: torch.Tensor,
                       window_sizes: list = [10, 20, 40],
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on volatility clustering

    Motivation:
        Volatility clustering is a typical feature of financial time series (the
        ARCH effect). High-volatility periods tend to be accompanied by more high
        volatility, while low-volatility periods are relatively calm. Samples with
        similar volatility characteristics should be aggregated together.

    Mathematical expression:
        vol_k = std(returns_{t-k:t})  (k-period rolling volatility)
        phi = concat([vol_10, vol_20, vol_40])
        phi = phi / ||phi||_2
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        window_sizes: list of window sizes for volatility computation
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    vol_features = []

    for window in window_sizes:
        if window <= seq_len:
            # Take the last `window` time steps
            window_data = batch_x[:, -window:, :]  # [batch_size, window, feature_dim]

            # Compute the standard deviation of each sample within the window (volatility proxy)
            volatility = window_data.std(dim=1)  # [batch_size, feature_dim]

            # Take the mean volatility over all features
            vol_avg = volatility.mean(dim=1, keepdim=True)  # [batch_size, 1]
            vol_features.append(vol_avg)

    # Concatenate
    phi = torch.cat(vol_features, dim=1)  # [batch_size, len(window_sizes)]

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 3. Return Distribution Shape Predicate (Return Distribution Phi)
# ============================================================================

def get_return_distribution_I_P(batch_x: torch.Tensor,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on return distribution shape

    Motivation:
        The higher-order moments of the return distribution (skewness, kurtosis)
        reflect market asymmetry and tail risk. Samples with similar distribution
        shapes may face similar risk characteristics.

    Mathematical expression:
        skew = E[((x - μ) / σ)^3]  (skewness)
        kurt = E[((x - μ) / σ)^4] - 3  (excess kurtosis)
        phi = [mean, std, skew, kurt]
        phi = phi / ||phi||_2
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute statistics over the time axis (taking the mean over all features)
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]

    # Mean
    mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Standard deviation
    std = x.std(dim=1, keepdim=True)  # [batch_size, 1]

    # Standardize
    x_normalized = (x - mean) / (std + 1e-8)  # [batch_size, seq_len]

    # Skewness (third moment)
    skewness = (x_normalized ** 3).mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Kurtosis (fourth moment - 3)
    kurtosis = (x_normalized ** 4).mean(dim=1, keepdim=True) - 3  # [batch_size, 1]

    # Handle NaN
    skewness = torch.nan_to_num(skewness, nan=0.0)
    kurtosis = torch.nan_to_num(kurtosis, nan=0.0)

    # Concatenate all distribution features
    phi = torch.cat([mean, std, skewness, kurtosis], dim=1)  # [batch_size, 4]

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 4. Trend Strength Predicate (Trend Strength Phi)
# ============================================================================

def get_trend_strength_I_P(batch_x: torch.Tensor,
                            device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on trend strength (similar in spirit to the ADX indicator)

    Motivation:
        In technical analysis, trend strength is an important indicator for judging
        the market state. Strongly trending markets and ranging markets require
        different trading strategies. Aggregating samples with similar trend strength
        can improve the model's adaptability.

    Mathematical expression:
        linear_fit: y = ax + b (linear regression over the time series)
        trend_direction = sign(a)  (trend direction)
        trend_strength = |a| / std(x)  (trend strength, normalized)
        r_squared = R² of the linear fit  (trend reliability)
        phi = [trend_direction, trend_strength, r_squared]
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Use the mean of all features as the proxy series
    y = batch_x.mean(dim=2)  # [batch_size, seq_len]

    # Build the time index
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

    # Linear regression: y = a*t + b
    # a = cov(t, y) / var(t)
    t_mean = t.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    cov_ty = ((t - t_mean) * (y - y_mean)).mean(dim=1, keepdim=True)
    var_t = ((t - t_mean) ** 2).mean(dim=1, keepdim=True)

    slope = cov_ty / (var_t + 1e-8)  # [batch_size, 1]

    # Trend direction
    trend_direction = torch.sign(slope)  # [batch_size, 1]

    # Trend strength (absolute value of the slope, normalized by the std of y)
    y_std = y.std(dim=1, keepdim=True)
    trend_strength = torch.abs(slope) / (y_std + 1e-8)  # [batch_size, 1]

    # Compute R² (goodness of fit)
    y_pred = slope * t + (y_mean - slope * t_mean)
    ss_res = ((y - y_pred) ** 2).sum(dim=1, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum(dim=1, keepdim=True)
    r_squared = 1 - ss_res / (ss_tot + 1e-8)  # [batch_size, 1]
    r_squared = torch.clamp(r_squared, 0, 1)

    # Concatenate
    phi = torch.cat([trend_direction, trend_strength, r_squared], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 5. Feature Correlation Structure Predicate (Feature Correlation Structure Phi)
# ============================================================================

def get_feature_correlation_I_P(batch_x: torch.Tensor,
                                 n_components: int = 10,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on feature correlation structure

    Motivation:
        Alpha158 contains a variety of factors, and different samples have different
        degrees of exposure to these factors. By capturing the correlation structure
        among features, one can identify samples with similar factor exposures.

    Mathematical expression:
        For each sample, compute the principal components of the covariance matrix of
        features along the time dimension
        phi = upper-triangular elements of the covariance matrix (or principal components)
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        n_components: number of principal components to use
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # For each sample, compute the time-dimension mean and variance of features
    feature_mean = batch_x.mean(dim=1)  # [batch_size, feature_dim]
    feature_std = batch_x.std(dim=1)  # [batch_size, feature_dim]

    # Select the statistics of the first n_components features
    n_components = min(n_components, feature_dim)

    # Combine mean and variance
    phi = torch.cat([
        feature_mean[:, :n_components],
        feature_std[:, :n_components]
    ], dim=1)  # [batch_size, 2*n_components]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 6. Price-Volume Relationship Predicate (Price-Volume Relationship Phi)
# ============================================================================

def get_price_volume_I_P(batch_x: torch.Tensor,
                          price_idx: int = 0,
                          volume_idx: int = 4,
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on the price-volume relationship

    Motivation:
        The price-volume relationship is central to technical analysis. Patterns such
        as rising price with rising volume, or price-volume divergence, reflect the
        behavioral characteristics of market participants. Samples with similar
        price-volume relationships may be in similar market states.

    Mathematical expression:
        price_change = (price_t - price_{t-1}) / price_{t-1}
        volume_change = (volume_t - volume_{t-1}) / volume_{t-1}
        pv_corr = corr(price_change, volume_change)
        phi = [pv_corr, avg_price_change, avg_volume_change]
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        price_idx: index of the price feature
        volume_idx: index of the volume feature
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Extract the price and volume series
    # Note: if the index is out of range, use the mean
    if price_idx < feature_dim and volume_idx < feature_dim:
        price = batch_x[:, :, price_idx]  # [batch_size, seq_len]
        volume = batch_x[:, :, volume_idx]  # [batch_size, seq_len]
    else:
        # Use the mean of the first half and second half of the features as a proxy
        price = batch_x[:, :, :feature_dim//2].mean(dim=2)
        volume = batch_x[:, :, feature_dim//2:].mean(dim=2)

    # Compute the rate of change
    price_change = (price[:, 1:] - price[:, :-1]) / (torch.abs(price[:, :-1]) + 1e-8)
    volume_change = (volume[:, 1:] - volume[:, :-1]) / (torch.abs(volume[:, :-1]) + 1e-8)

    # Handle NaN
    price_change = torch.nan_to_num(price_change, nan=0.0)
    volume_change = torch.nan_to_num(volume_change, nan=0.0)

    # Compute the price-volume correlation
    price_mean = price_change.mean(dim=1, keepdim=True)
    volume_mean = volume_change.mean(dim=1, keepdim=True)

    cov_pv = ((price_change - price_mean) * (volume_change - volume_mean)).mean(dim=1, keepdim=True)
    std_p = price_change.std(dim=1, keepdim=True)
    std_v = volume_change.std(dim=1, keepdim=True)

    pv_corr = cov_pv / (std_p * std_v + 1e-8)  # [batch_size, 1]

    # Average rate of change
    avg_price_change = price_change.mean(dim=1, keepdim=True)  # [batch_size, 1]
    avg_volume_change = volume_change.mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([pv_corr, avg_price_change, avg_volume_change], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 7. Multi-Scale Feature Predicate (Multi-Scale Feature Phi)
# ============================================================================

def get_multiscale_I_P(batch_x: torch.Tensor,
                        scales: list = [1, 4, 16, 32],
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on multi-scale features

    Motivation:
        Financial time series have multi-scale properties, with short-term noise,
        medium-term trends, and long-term cycles coexisting. By extracting features
        at different time scales, one can capture sample similarity across multiple
        levels.

    Mathematical expression:
        For each scale s, compute the mean and variance after downsampling
        x_s = downsample(x, factor=s)
        phi_s = [mean(x_s), std(x_s)]
        phi = concat([phi_1, phi_4, phi_16, phi_32])
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        scales: list of downsampling scales
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    scale_features = []

    for scale in scales:
        if scale <= seq_len:
            # Use average pooling for downsampling
            # First transpose to [batch_size, feature_dim, seq_len] so that avg_pool1d can be used
            x_transposed = batch_x.permute(0, 2, 1)

            # Average pooling
            pooled = F.avg_pool1d(x_transposed, kernel_size=scale, stride=scale)

            # Compute the mean and variance of the pooled series
            mean_val = pooled.mean(dim=2).mean(dim=1, keepdim=True)  # [batch_size, 1]
            std_val = pooled.std(dim=2).mean(dim=1, keepdim=True)  # [batch_size, 1]

            scale_features.append(mean_val)
            scale_features.append(std_val)

    # Concatenate
    phi = torch.cat(scale_features, dim=1)  # [batch_size, 2*len(scales)]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 8. Return Similarity Clustering Predicate (Return Similarity Clustering Phi)
# ============================================================================

def get_return_similarity_I_P(batch_x: torch.Tensor,
                               n_bins: int = 5,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on return similarity (soft clustering)

    Motivation:
        Soft-cluster samples by their return distribution so that samples with similar
        return patterns influence each other. This can capture the structural
        characteristics of the market.

    Mathematical expression:
        Compute each sample's cumulative return and intraday return distribution
        Use quantiles to map samples into different bins
        phi = one_hot(bin_index) or softmax soft assignment
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        n_bins: number of bins
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute the cumulative change (using the mean of all features)
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    cumulative_change = (x[:, -1] - x[:, 0]) / (torch.abs(x[:, 0]) + 1e-8)  # [batch_size]

    # Use a Gaussian kernel for soft assignment
    # Create n_bins center points
    centers = torch.linspace(-1, 1, n_bins, device=device)  # [n_bins]

    # Compute the distance to each center
    cumulative_change = cumulative_change.unsqueeze(1)  # [batch_size, 1]
    centers = centers.unsqueeze(0)  # [1, n_bins]

    # Gaussian-kernel soft assignment
    sigma = 0.5
    distances = -((cumulative_change - centers) ** 2) / (2 * sigma ** 2)
    phi = F.softmax(distances, dim=1)  # [batch_size, n_bins]

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 9. Temporal Pattern Predicate (Temporal Pattern Phi)
# ============================================================================

def get_temporal_pattern_I_P(batch_x: torch.Tensor,
                              n_segments: int = 4,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on temporal patterns

    Motivation:
        Split the time series into multiple segments and compare the relative change
        of each segment. Samples with similar temporal patterns (e.g., rise-then-fall,
        sustained rise, etc.) should be aggregated.

    Mathematical expression:
        Split the series into n_segments segments
        For each segment, compute: mean change, trend direction
        phi = concat([segment_1_features, ..., segment_n_features])
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        n_segments: number of segments to split into
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Use the mean of all features
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]

    segment_len = seq_len // n_segments
    segment_features = []

    for i in range(n_segments):
        start_idx = i * segment_len
        end_idx = (i + 1) * segment_len if i < n_segments - 1 else seq_len

        segment = x[:, start_idx:end_idx]  # [batch_size, segment_len]

        # Rate of change of the segment
        segment_change = (segment[:, -1] - segment[:, 0]) / (torch.abs(segment[:, 0]) + 1e-8)
        segment_change = segment_change.unsqueeze(1)  # [batch_size, 1]

        # Trend direction of the segment (using sign)
        segment_trend = torch.sign(segment_change)  # [batch_size, 1]

        segment_features.append(segment_change)
        segment_features.append(segment_trend)

    # Concatenate
    phi = torch.cat(segment_features, dim=1)  # [batch_size, 2*n_segments]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 10. Factor Exposure Predicate (Factor Exposure Phi)
# ============================================================================

def get_factor_exposure_I_P(batch_x: torch.Tensor,
                             factor_groups: Optional[list] = None,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on factor exposure

    Motivation:
        Alpha158 features can be divided into different factor groups (such as value
        factors, momentum factors, volatility factors, etc.). Compute each sample's
        degree of exposure to the different factor groups, and aggregate samples with
        similar factor structures.

    Mathematical expression:
        Split the 158 features into k factor groups
        For each factor group, compute: exposure_k = mean(features_in_group_k)
        phi = [exposure_1, ..., exposure_k]
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        factor_groups: list of factor groups, where each element is a list of feature indices
                       if None, automatically group at equal intervals
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Default grouping: split the features evenly into several groups
    if factor_groups is None:
        n_groups = 10  # Split into 10 groups by default
        group_size = feature_dim // n_groups
        factor_groups = [
            list(range(i * group_size, min((i + 1) * group_size, feature_dim)))
            for i in range(n_groups)
        ]

    factor_exposures = []

    for group_indices in factor_groups:
        if len(group_indices) > 0:
            # Extract the features of this factor group
            group_features = batch_x[:, :, group_indices]  # [batch_size, seq_len, len(group)]

            # Compute factor exposure: mean over the time and feature dimensions
            exposure = group_features.mean(dim=(1, 2), keepdim=False)  # [batch_size]
            exposure = exposure.unsqueeze(1)  # [batch_size, 1]
            factor_exposures.append(exposure)

    # Concatenate
    phi = torch.cat(factor_exposures, dim=1)  # [batch_size, n_groups]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 11. Composite Technical Indicator Predicate (Technical Indicator Phi)
# ============================================================================

def get_technical_indicator_I_P(batch_x: torch.Tensor,
                                 short_window: int = 5,
                                 long_window: int = 20,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on technical indicators (mimicking the spirit of RSI, MACD, etc.)

    Motivation:
        Technical indicators are important tools in quantitative trading. By
        constructing indicators similar to RSI and MACD, one can capture a sample's
        overbought/oversold state and trend momentum.

    Mathematical expression:
        RSI_proxy = up days / total days
        MACD_proxy = short_MA - long_MA
        BB_proxy = (price - MA) / std
        phi = [RSI_proxy, MACD_proxy, BB_proxy]
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        short_window: short-term moving-average window
        long_window: long-term moving-average window
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Use the mean of all features
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]

    # RSI proxy: compute the proportion of up days
    changes = x[:, 1:] - x[:, :-1]  # [batch_size, seq_len-1]
    up_days = (changes > 0).float().sum(dim=1, keepdim=True)
    total_days = changes.shape[1]
    rsi_proxy = up_days / total_days  # [batch_size, 1]

    # MACD proxy: short-term mean - long-term mean
    short_ma = x[:, -short_window:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    long_ma = x[:, -long_window:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    macd_proxy = short_ma - long_ma  # [batch_size, 1]

    # Bollinger Bands proxy: (current price - mean) / standard deviation
    current_price = x[:, -1:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    price_mean = x[:, -long_window:].mean(dim=1, keepdim=True)
    price_std = x[:, -long_window:].std(dim=1, keepdim=True)
    bb_proxy = (current_price - price_mean) / (price_std + 1e-8)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([rsi_proxy, macd_proxy, bb_proxy], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 12. Volatility-Return Joint Predicate (Volatility-Return Joint Phi)
# ============================================================================

def get_vol_return_joint_I_P(batch_x: torch.Tensor,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on the joint volatility-return distribution

    Motivation:
        High returns tend to come with high risk (volatility). By capturing the joint
        distribution of volatility and return, one can identify samples with similar
        risk-return characteristics.

    Mathematical expression:
        return = (x_T - x_1) / x_1
        volatility = std(x)
        sharpe_proxy = return / volatility  (Sharpe ratio proxy)
        phi = [return, volatility, sharpe_proxy]
        P = phi @ phi.T

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Use the mean of all features
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]

    # Return
    returns = (x[:, -1] - x[:, 0]) / (torch.abs(x[:, 0]) + 1e-8)
    returns = returns.unsqueeze(1)  # [batch_size, 1]

    # Volatility
    volatility = x.std(dim=1, keepdim=True)  # [batch_size, 1]

    # Sharpe ratio proxy
    sharpe = returns / (volatility + 1e-8)  # [batch_size, 1]

    # Maximum drawdown proxy
    cummax = torch.cummax(x, dim=1)[0]  # [batch_size, seq_len]
    drawdown = (cummax - x) / (cummax + 1e-8)  # [batch_size, seq_len]
    max_drawdown = drawdown.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([returns, volatility, sharpe, max_drawdown], dim=1)  # [batch_size, 4]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 13. Market Beta Predicate (Market Beta Phi) - based on inter-stock correlation
# ============================================================================

def get_market_beta_I_P(batch_x: torch.Tensor,
                         device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on market Beta

    Motivation:
        Beta measures an individual stock's sensitivity to overall market fluctuations.
        Stocks with similar Beta tend to have similar exposure to systematic risk and
        should be aggregated together.
        This embodies the systematic-risk idea of the CAPM model.

    Mathematical expression:
        market_return = mean(all_sample_returns)  (market return proxy)
        beta_i = cov(r_i, r_m) / var(r_m)  (individual-stock Beta)
        alpha_i = mean(r_i) - beta_i * mean(r_m)  (Jensen's Alpha)
        phi = [beta, alpha, residual_vol]
        P = phi @ phi.T

    Computational complexity: O(B × L × D + B²)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute each sample's return series (using the mean of all features)
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)  # [batch_size, seq_len-1]
    returns = torch.nan_to_num(returns, nan=0.0)

    # Market return: the average return across all samples
    market_return = returns.mean(dim=0, keepdim=True)  # [1, seq_len-1]
    market_return = market_return.expand(batch_size, -1)  # [batch_size, seq_len-1]

    # Compute Beta: cov(r_i, r_m) / var(r_m)
    r_mean = returns.mean(dim=1, keepdim=True)
    m_mean = market_return.mean(dim=1, keepdim=True)

    cov_rm = ((returns - r_mean) * (market_return - m_mean)).mean(dim=1, keepdim=True)
    var_m = ((market_return - m_mean) ** 2).mean(dim=1, keepdim=True)

    beta = cov_rm / (var_m + 1e-8)  # [batch_size, 1]

    # Jensen's Alpha
    alpha = r_mean - beta * m_mean  # [batch_size, 1]

    # Residual volatility (idiosyncratic risk)
    predicted_return = alpha + beta * market_return
    residual = returns - predicted_return
    residual_vol = residual.std(dim=1, keepdim=True)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([beta, alpha, residual_vol], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 14. Cross-Sample Correlation Predicate (Cross-Sample Correlation Phi) - based on inter-stock correlation
# ============================================================================

def get_cross_correlation_I_P(batch_x: torch.Tensor,
                               use_returns: bool = True,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on cross-sample correlation

    Motivation:
        Directly compute the correlation matrix among the samples within the batch and
        use it as the basis for the P matrix. This directly captures the co-movement
        between stocks, reflecting sector effects.

    Mathematical expression:
        corr_ij = corr(r_i, r_j)  (correlation coefficient of samples i and j)
        P = softmax(corr_matrix)  (normalized into a probability distribution)

    Computational complexity: O(B² × L)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        use_returns: whether to use returns to compute correlation
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Use the mean of all features
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]

    if use_returns:
        # Convert to returns
        x = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
        x = torch.nan_to_num(x, nan=0.0)

    # Standardize
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    x_normalized = (x - x_mean) / (x_std + 1e-8)  # [batch_size, seq_len-1]

    # Compute the correlation-coefficient matrix
    # corr(i,j) = (x_i @ x_j) / (len * std_i * std_j)
    # Since already standardized, simply compute the inner product divided by the length
    corr_matrix = torch.mm(x_normalized, x_normalized.T) / x_normalized.shape[1]  # [batch_size, batch_size]

    # Map the correlation coefficients to [0, 1] and normalize
    P = (corr_matrix + 1) / 2  # map correlation coefficients from [-1,1] to [0,1]
    P = P / (torch.linalg.norm(P) + 1e-8)

    return I, P


# ============================================================================
# 15. Lead-Lag Relationship Predicate (Lead-Lag Relationship Phi) - based on inter-stock correlation
# ============================================================================

def get_lead_lag_I_P(batch_x: torch.Tensor,
                      max_lag: int = 5,
                      device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on the lead-lag relationship

    Motivation:
        There is a time lag in how information propagates through the market; certain
        stocks (e.g., large caps) may lead others (e.g., small caps). By computing
        lead-lag correlations, one can capture this information-propagation effect,
        reflecting market microstructure.

    Mathematical expression:
        lag_corr_ij(k) = corr(r_i(t), r_j(t+k))  (correlation at lag k)
        optimal_lag_ij = argmax_k |lag_corr_ij(k)|  (optimal lag)
        phi_i = [max_lead_corr, max_lag_corr, net_lead_indicator]
        P = phi @ phi.T

    Computational complexity: O(B² × L × max_lag)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        max_lag: maximum number of lag periods
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute returns
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Standardize
    r_mean = returns.mean(dim=1, keepdim=True)
    r_std = returns.std(dim=1, keepdim=True)
    r_normalized = (returns - r_mean) / (r_std + 1e-8)

    # Compute lead-lag correlation with the market
    market_return = r_normalized.mean(dim=0)  # [seq_len-1]

    lead_corrs = []  # sample leads the market
    lag_corrs = []   # sample lags the market

    for lag in range(1, min(max_lag + 1, seq_len - 1)):
        # Sample leads: correlation of sample t with market t+lag
        sample_early = r_normalized[:, :-lag]  # [batch_size, seq_len-1-lag]
        market_late = market_return[lag:]  # [seq_len-1-lag]
        lead_corr = (sample_early * market_late.unsqueeze(0)).mean(dim=1)  # [batch_size]
        lead_corrs.append(lead_corr)

        # Sample lags: correlation of sample t+lag with market t
        sample_late = r_normalized[:, lag:]  # [batch_size, seq_len-1-lag]
        market_early = market_return[:-lag]  # [seq_len-1-lag]
        lag_corr = (sample_late * market_early.unsqueeze(0)).mean(dim=1)  # [batch_size]
        lag_corrs.append(lag_corr)

    # Take the maximum lead and lag correlations
    lead_corrs = torch.stack(lead_corrs, dim=1)  # [batch_size, max_lag]
    lag_corrs = torch.stack(lag_corrs, dim=1)  # [batch_size, max_lag]

    max_lead = lead_corrs.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    max_lag_val = lag_corrs.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

    # Net lead indicator: positive means a tendency to lead, negative means a tendency to lag
    net_lead = max_lead - max_lag_val  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([max_lead, max_lag_val, net_lead], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 16. Common Factor Exposure Predicate (Common Factor Exposure Phi) - based on inter-stock correlation
# ============================================================================

def get_common_factor_I_P(batch_x: torch.Tensor,
                           n_factors: int = 3,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on common factor exposure

    Motivation:
        Extract the common factors of the samples within the batch via PCA, then
        construct phi according to each sample's degree of exposure to these factors.
        Samples with similar factor exposures may belong to the same sector or be
        driven by the same macro factors. This embodies the idea of APT (Arbitrage
        Pricing Theory).

    Mathematical expression:
        X = [r_1, r_2, ..., r_B]^T  (return matrix)
        U, S, V = SVD(X)  (singular value decomposition)
        factor_loadings = U[:, :k] * S[:k]  (factor loadings)
        phi = factor_loadings
        P = phi @ phi.T

    Computational complexity: O(B × L × D + min(B, L)³)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        n_factors: number of factors to extract
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute returns
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Center
    returns_centered = returns - returns.mean(dim=0, keepdim=True)

    # SVD decomposition
    try:
        U, S, Vh = torch.linalg.svd(returns_centered, full_matrices=False)

        # Take the first n_factors factors
        n_factors = min(n_factors, min(batch_size, seq_len - 1))

        # Factor loadings: U * S
        factor_loadings = U[:, :n_factors] * S[:n_factors].unsqueeze(0)  # [batch_size, n_factors]

        phi = factor_loadings
    except:
        # Fall back to a simplified method when SVD fails
        phi = returns.mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 17. Volatility Synchronization Predicate (Volatility Synchronization Phi) - based on inter-stock correlation
# ============================================================================

def get_volatility_sync_I_P(batch_x: torch.Tensor,
                             window: int = 10,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on volatility synchronization

    Motivation:
        Volatility synchronization reflects the contagion of systematic market risk.
        When the market enters a high-volatility period, the volatility of individual
        stocks tends to rise in sync (a cross-asset version of volatility clustering).
        Samples with similar volatility synchronization may be exposed to the same
        systematic risk.

    Mathematical expression:
        vol_i(t) = rolling_std(r_i, window)  (rolling volatility)
        vol_sync_ij = corr(vol_i, vol_j)  (volatility synchronization)
        phi_i = [avg_vol_sync, vol_beta, vol_level]
        P = phi @ phi.T

    Computational complexity: O(B × L × W + B²)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        window: rolling window size
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute returns
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Compute rolling volatility
    n_windows = (seq_len - 1) // window
    if n_windows < 2:
        # Too few windows, use a simplified method
        phi = returns.std(dim=1, keepdim=True)
        phi = phi / (torch.linalg.norm(phi) + 1e-8)
        P = torch.mm(phi, phi.T)
        return I, P

    rolling_vols = []
    for i in range(n_windows):
        start_idx = i * window
        end_idx = (i + 1) * window
        window_returns = returns[:, start_idx:end_idx]
        vol = window_returns.std(dim=1)  # [batch_size]
        rolling_vols.append(vol)

    rolling_vols = torch.stack(rolling_vols, dim=1)  # [batch_size, n_windows]

    # Market volatility (the average across all samples)
    market_vol = rolling_vols.mean(dim=0, keepdim=True)  # [1, n_windows]

    # Volatility synchronization: correlation with market volatility
    vol_mean = rolling_vols.mean(dim=1, keepdim=True)
    market_vol_mean = market_vol.mean(dim=1, keepdim=True)

    cov_vm = ((rolling_vols - vol_mean) * (market_vol - market_vol_mean)).mean(dim=1, keepdim=True)
    std_v = rolling_vols.std(dim=1, keepdim=True)
    std_m = market_vol.std(dim=1, keepdim=True)

    vol_sync = cov_vm / (std_v * std_m + 1e-8)  # [batch_size, 1]

    # Volatility Beta: sensitivity of volatility to market volatility
    var_m = ((market_vol - market_vol_mean) ** 2).mean(dim=1, keepdim=True)
    vol_beta = cov_vm / (var_m + 1e-8)  # [batch_size, 1]

    # Volatility level
    vol_level = rolling_vols.mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([vol_sync, vol_beta, vol_level], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 18. Tail Risk Co-occurrence Predicate (Tail Risk Co-occurrence Phi) - based on inter-stock correlation
# ============================================================================

def get_tail_risk_I_P(batch_x: torch.Tensor,
                       quantile: float = 0.1,
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on tail risk co-occurrence

    Motivation:
        During extreme market events, correlations between assets tend to rise
        (correlation breakdown). The co-occurrence frequency of tail risk reflects
        the contagiousness of systematic risk. Samples with similar tail-risk
        characteristics should be aggregated.

    Mathematical expression:
        left_tail_i = I(r_i < quantile(r_i, q))  (left-tail indicator variable)
        right_tail_i = I(r_i > quantile(r_i, 1-q))  (right-tail indicator variable)
        tail_freq_i = mean(left_tail_i)  (tail event frequency)
        phi = [left_tail_freq, right_tail_freq, tail_asymmetry, avg_tail_return]
        P = phi @ phi.T

    Computational complexity: O(B × L × D + B × L log L)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        quantile: tail quantile threshold
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute returns
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Compute the quantile thresholds
    left_threshold = torch.quantile(returns, quantile, dim=1, keepdim=True)  # [batch_size, 1]
    right_threshold = torch.quantile(returns, 1 - quantile, dim=1, keepdim=True)  # [batch_size, 1]

    # Left-tail and right-tail events
    left_tail = (returns < left_threshold).float()  # [batch_size, seq_len-1]
    right_tail = (returns > right_threshold).float()  # [batch_size, seq_len-1]

    # Tail event frequency
    left_freq = left_tail.mean(dim=1, keepdim=True)  # [batch_size, 1]
    right_freq = right_tail.mean(dim=1, keepdim=True)  # [batch_size, 1]

    # Tail asymmetry
    tail_asymmetry = right_freq - left_freq  # [batch_size, 1]

    # Average tail return (conditional expectation)
    left_returns = returns * left_tail
    left_sum = left_returns.sum(dim=1, keepdim=True)
    left_count = left_tail.sum(dim=1, keepdim=True)
    avg_left_return = left_sum / (left_count + 1e-8)  # [batch_size, 1]

    right_returns = returns * right_tail
    right_sum = right_returns.sum(dim=1, keepdim=True)
    right_count = right_tail.sum(dim=1, keepdim=True)
    avg_right_return = right_sum / (right_count + 1e-8)  # [batch_size, 1]

    # Tail risk ratio
    tail_ratio = torch.abs(avg_left_return) / (torch.abs(avg_right_return) + 1e-8)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([left_freq, right_freq, tail_asymmetry, tail_ratio], dim=1)  # [batch_size, 4]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 19. Market Regime Sensitivity Predicate (Market Regime Sensitivity Phi) - based on inter-stock correlation
# ============================================================================

def get_market_regime_I_P(batch_x: torch.Tensor,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on market regime sensitivity

    Motivation:
        Different stocks perform very differently in bull and bear markets. By
        computing the difference in a sample's performance under different market
        regimes, one can identify defensive versus aggressive stocks.
        This embodies the idea of market timing.

    Mathematical expression:
        market_up = I(r_m > 0)  (market-up indicator)
        r_up_i = mean(r_i | market_up)  (average return in up markets)
        r_down_i = mean(r_i | !market_up)  (average return in down markets)
        upside_capture = r_up_i / r_m_up  (upside capture ratio)
        downside_capture = r_down_i / r_m_down  (downside capture ratio)
        phi = [upside_capture, downside_capture, capture_ratio]
        P = phi @ phi.T

    Computational complexity: O(B × L × D)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Compute returns
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Market return
    market_return = returns.mean(dim=0)  # [seq_len-1]

    # Market regime
    market_up = (market_return > 0).float()  # [seq_len-1]
    market_down = 1 - market_up  # [seq_len-1]

    # Performance in up markets
    up_returns = returns * market_up.unsqueeze(0)  # [batch_size, seq_len-1]
    up_sum = up_returns.sum(dim=1, keepdim=True)
    up_count = market_up.sum()
    avg_up_return = up_sum / (up_count + 1e-8)  # [batch_size, 1]

    # Performance in down markets
    down_returns = returns * market_down.unsqueeze(0)  # [batch_size, seq_len-1]
    down_sum = down_returns.sum(dim=1, keepdim=True)
    down_count = market_down.sum()
    avg_down_return = down_sum / (down_count + 1e-8)  # [batch_size, 1]

    # Market average return
    market_up_avg = (market_return * market_up).sum() / (up_count + 1e-8)
    market_down_avg = (market_return * market_down).sum() / (down_count + 1e-8)

    # Capture ratios
    upside_capture = avg_up_return / (market_up_avg + 1e-8)  # [batch_size, 1]
    downside_capture = avg_down_return / (market_down_avg + 1e-8)  # [batch_size, 1]

    # Capture ratio (>1 means defensive, <1 means aggressive)
    capture_ratio = upside_capture / (torch.abs(downside_capture) + 1e-8)  # [batch_size, 1]

    # Concatenate
    phi = torch.cat([upside_capture, downside_capture, capture_ratio], dim=1)  # [batch_size, 3]

    # Handle NaN
    phi = torch.nan_to_num(phi, nan=0.0)

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# 20. Sector Clustering Predicate (Sector Clustering Phi) - based on inter-stock correlation
# ============================================================================

def get_sector_clustering_I_P(batch_x: torch.Tensor,
                               n_clusters: int = 5,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phi construction based on sector clustering

    Motivation:
        Stocks in the same sector/industry often have similar price movements. By
        performing unsupervised clustering on the samples, one can discover the
        implicit sector structure. Uses K-Means-style soft cluster assignment.

    Mathematical expression:
        Use feature vectors for soft clustering
        dist_ik = ||feature_i - center_k||²  (distance to cluster center)
        phi = softmax(-dist / temperature)  (soft assignment probabilities)
        P = phi @ phi.T

    Computational complexity: O(B × L × D + B × K)

    Args:
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        n_clusters: number of clusters (simulating the number of sectors)
        device: compute device

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device

    I = torch.eye(batch_size, device=device)

    # Extract features: use return statistics
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]

    # Build feature vector: [mean, std, skew, autocorr]
    r_mean = returns.mean(dim=1, keepdim=True)
    r_std = returns.std(dim=1, keepdim=True)
    r_normalized = (returns - r_mean) / (r_std + 1e-8)
    r_skew = (r_normalized ** 3).mean(dim=1, keepdim=True)

    # First-order autocorrelation
    autocorr = (r_normalized[:, :-1] * r_normalized[:, 1:]).mean(dim=1, keepdim=True)

    features = torch.cat([r_mean, r_std, r_skew, autocorr], dim=1)  # [batch_size, 4]
    features = torch.nan_to_num(features, nan=0.0)

    # Simplified clustering: use uniformly distributed center points
    # In practice, K-Means iterations could be used
    feature_min = features.min(dim=0, keepdim=True)[0]
    feature_max = features.max(dim=0, keepdim=True)[0]
    feature_range = feature_max - feature_min + 1e-8

    # Normalize features
    features_normalized = (features - feature_min) / feature_range

    # Create cluster centers (uniformly distributed)
    centers = torch.zeros(n_clusters, features.shape[1], device=device)
    for k in range(n_clusters):
        centers[k] = (k + 0.5) / n_clusters

    # Compute the distance to each center
    # features_normalized: [batch_size, 4]
    # centers: [n_clusters, 4]
    distances = torch.cdist(features_normalized, centers)  # [batch_size, n_clusters]

    # Soft assignment
    temperature = 0.5
    phi = F.softmax(-distances / temperature, dim=1)  # [batch_size, n_clusters]

    # Normalize
    phi = phi / (torch.linalg.norm(phi) + 1e-8)

    P = torch.mm(phi, phi.T)

    return I, P


# ============================================================================
# Unified interface: obtain the corresponding I and P matrices based on phi_type
# ============================================================================

def get_custom_I_P(phi_type: str,
                   batch_x: torch.Tensor,
                   device: Optional[torch.device] = None,
                   **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified interface: obtain the corresponding I and P matrices based on the phi_type string

    Args:
        phi_type: phi type string, supporting the following types:
            - 'phi_momentum': momentum factor predicate
            - 'phi_volatility': volatility clustering predicate
            - 'phi_return_dist': return distribution shape predicate
            - 'phi_trend': trend strength predicate
            - 'phi_feature_corr': feature correlation structure predicate
            - 'phi_price_volume': price-volume relationship predicate
            - 'phi_multiscale': multi-scale feature predicate
            - 'phi_return_sim': return similarity clustering predicate
            - 'phi_temporal': temporal pattern predicate
            - 'phi_factor': factor exposure predicate
            - 'phi_technical': composite technical indicator predicate
            - 'phi_vol_return': volatility-return joint predicate
        batch_x: input tensor, shape [batch_size, seq_len, feature_dim]
        device: compute device
        **kwargs: other parameters

    Returns:
        I: identity matrix [batch_size, batch_size]
        P: similarity matrix [batch_size, batch_size]
    """
    if device is None:
        device = batch_x.device

    phi_type_map = {
        # Predicates based on time-series properties
        'phi_momentum': get_momentum_I_P,
        'phi_volatility': get_volatility_I_P,
        'phi_return_dist': get_return_distribution_I_P,
        'phi_trend': get_trend_strength_I_P,
        'phi_multiscale': get_multiscale_I_P,
        'phi_temporal': get_temporal_pattern_I_P,

        # Predicates based on the relatedness of financial indicators
        'phi_feature_corr': get_feature_correlation_I_P,
        'phi_price_volume': get_price_volume_I_P,
        'phi_factor': get_factor_exposure_I_P,
        'phi_technical': get_technical_indicator_I_P,
        'phi_vol_return': get_vol_return_joint_I_P,

        # Predicates based on inter-stock correlation
        'phi_return_sim': get_return_similarity_I_P,
        'phi_market_beta': get_market_beta_I_P,
        'phi_cross_corr': get_cross_correlation_I_P,
        'phi_lead_lag': get_lead_lag_I_P,
        'phi_common_factor': get_common_factor_I_P,
        'phi_vol_sync': get_volatility_sync_I_P,
        'phi_tail_risk': get_tail_risk_I_P,
        'phi_market_regime': get_market_regime_I_P,
        'phi_sector_cluster': get_sector_clustering_I_P,
    }

    if phi_type in phi_type_map:
        return phi_type_map[phi_type](batch_x, device=device, **kwargs)
    else:
        # Return the identity matrix by default
        batch_size = batch_x.shape[0]
        I = torch.eye(batch_size, device=device)
        return I, I


# ============================================================================
# Combined predicate: weighted combination of multiple phi
# ============================================================================

def get_combined_I_P(phi_types: list,
                     weights: Optional[list] = None,
                     batch_x: torch.Tensor = None,
                     device: Optional[torch.device] = None,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the P matrix by combining multiple phi predicates

    Motivation:
        A single predicate may not fully capture the complex relationships among
        samples. By combining multiple predicates, sample similarity can be
        characterized from multiple perspectives.

    Mathematical expression:
        P_combined = Σ w_i * P_i
        P_combined = P_combined / ||P_combined||_F

    Args:
        phi_types: list of phi types
        weights: list of weights; if None, equal weights are used
        batch_x: input tensor
        device: compute device
        **kwargs: other parameters

    Returns:
        I: identity matrix
        P: the combined similarity matrix
    """
    if device is None:
        device = batch_x.device

    batch_size = batch_x.shape[0]
    I = torch.eye(batch_size, device=device)

    if weights is None:
        weights = [1.0 / len(phi_types)] * len(phi_types)

    P_combined = torch.zeros(batch_size, batch_size, device=device)

    for phi_type, weight in zip(phi_types, weights):
        _, P = get_custom_I_P(phi_type, batch_x, device=device, **kwargs)
        P_combined += weight * P

    # Normalize
    P_combined = P_combined / (torch.linalg.norm(P_combined) + 1e-8)

    return I, P_combined


# ============================================================================
# Test function
# ============================================================================

def test_phi_constructors():
    """
    Test all phi constructor functions
    """
    print("=" * 60)
    print("Testing Custom Phi Constructors")
    print("=" * 60)

    # Create test data
    batch_size, seq_len, feature_dim = 32, 96, 158
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x = torch.randn(batch_size, seq_len, feature_dim, device=device)

    phi_types = [
        # Based on time-series properties
        'phi_momentum',
        'phi_volatility',
        'phi_return_dist',
        'phi_trend',
        'phi_multiscale',
        'phi_temporal',

        # Based on the relatedness of financial indicators
        'phi_feature_corr',
        'phi_price_volume',
        'phi_factor',
        'phi_technical',
        'phi_vol_return',

        # Based on inter-stock correlation
        'phi_return_sim',
        'phi_market_beta',
        'phi_cross_corr',
        'phi_lead_lag',
        'phi_common_factor',
        'phi_vol_sync',
        'phi_tail_risk',
        'phi_market_regime',
        'phi_sector_cluster',
    ]

    for phi_type in phi_types:
        try:
            I, P = get_custom_I_P(phi_type, batch_x, device=device)
            print(f"✓ {phi_type:20s} | I shape: {I.shape}, P shape: {P.shape}")

            # Verify the symmetry of the P matrix
            assert torch.allclose(P, P.T, atol=1e-5), f"{phi_type}: P is not symmetric"

        except Exception as e:
            print(f"✗ {phi_type:20s} | Error: {e}")

    # Test the combined predicate
    print("-" * 60)
    print("Testing Combined Phi")
    try:
        combined_types = ['phi_momentum', 'phi_volatility', 'phi_trend']
        I, P = get_combined_I_P(combined_types, batch_x=batch_x, device=device)
        print(f"✓ Combined ({', '.join(combined_types)}) | I shape: {I.shape}, P shape: {P.shape}")
    except Exception as e:
        print(f"✗ Combined | Error: {e}")

    print("=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_phi_constructors()
