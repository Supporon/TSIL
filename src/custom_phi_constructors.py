"""
自定义 Phi 谓词构造器模块

本模块包含多种基于股票领域知识的 phi 构造方法，用于生成 P 矩阵，
增强模型对样本间潜在关系的建模能力。

P 矩阵的核心作用：
- 将具有相似特征的样本在损失计算时进行聚合
- 通过 P = phi @ phi.T 构造样本间的相似性矩阵
- 最终损失为: loss = tau_hat * I * error + tau * P * error

输入数据维度: (batch_size, sequence_length, feature_dim) = (32, 96, 158)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PhiConstructor:
    """
    Phi 构造器基类，提供通用的辅助方法
    """
    
    @staticmethod
    def normalize_phi(phi: torch.Tensor) -> torch.Tensor:
        """
        对 phi 进行 L2 归一化
        
        Args:
            phi: 原始 phi 向量，形状为 [batch_size, d]
        
        Returns:
            归一化后的 phi 向量
        """
        norm = torch.linalg.norm(phi)
        return phi / (norm + 1e-8)
    
    @staticmethod
    def construct_P(phi: torch.Tensor) -> torch.Tensor:
        """
        根据 phi 构造 P 矩阵
        
        数学表达式: P = phi @ phi.T
        
        Args:
            phi: 归一化后的 phi 向量，形状为 [batch_size, d]
        
        Returns:
            P 矩阵，形状为 [batch_size, batch_size]
        """
        return torch.mm(phi, phi.T)
    
    @staticmethod
    def get_I(batch_size: int, device: torch.device) -> torch.Tensor:
        """获取单位矩阵"""
        return torch.eye(batch_size, device=device)


# ============================================================================
# 1. 动量因子谓词 (Momentum Phi)
# ============================================================================

def get_momentum_I_P(batch_x: torch.Tensor, 
                     lookback_windows: list = [5, 10, 20],
                     device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于动量因子的 Phi 构造
    
    动机：
        动量效应是金融市场中最著名的异象之一。具有相似动量特征的股票往往
        呈现出类似的价格走势，将这类样本聚合可以捕捉群体行为。
    
    数学表达式：
        momentum_k = (x_{t} - x_{t-k}) / x_{t-k}  (k 期动量)
        phi = concat([momentum_5, momentum_10, momentum_20])
        phi = phi / ||phi||_2
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        lookback_windows: 动量计算的回溯窗口列表
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 假设第一个特征维度是收盘价或者我们使用所有特征的均值
    # 计算多个时间窗口的动量
    momentum_features = []
    
    for window in lookback_windows:
        if window < seq_len:
            # 计算动量: (当前值 - 过去值) / 过去值
            current = batch_x[:, -1, :]  # [batch_size, feature_dim]
            past = batch_x[:, -1-window, :]  # [batch_size, feature_dim]
            
            # 避免除零
            momentum = (current - past) / (torch.abs(past) + 1e-8)
            momentum = torch.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 取所有特征的均值作为该窗口的动量表示
            momentum_avg = momentum.mean(dim=1, keepdim=True)  # [batch_size, 1]
            momentum_features.append(momentum_avg)
    
    # 拼接所有动量特征
    phi = torch.cat(momentum_features, dim=1)  # [batch_size, len(lookback_windows)]
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    # 构造 P 矩阵
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 2. 波动率聚类谓词 (Volatility Clustering Phi)
# ============================================================================

def get_volatility_I_P(batch_x: torch.Tensor,
                       window_sizes: list = [10, 20, 40],
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于波动率聚类的 Phi 构造
    
    动机：
        波动率聚类是金融时序的典型特征（ARCH效应）。高波动率时期往往伴随
        更多的高波动率，低波动率时期则相对平稳。具有相似波动率特征的样本
        应该被聚合处理。
    
    数学表达式：
        vol_k = std(returns_{t-k:t})  (k 期滚动波动率)
        phi = concat([vol_10, vol_20, vol_40])
        phi = phi / ||phi||_2
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        window_sizes: 波动率计算的窗口大小列表
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    vol_features = []
    
    for window in window_sizes:
        if window <= seq_len:
            # 取最后 window 个时间步
            window_data = batch_x[:, -window:, :]  # [batch_size, window, feature_dim]
            
            # 计算每个样本在窗口内的标准差（波动率代理）
            volatility = window_data.std(dim=1)  # [batch_size, feature_dim]
            
            # 取所有特征的均值波动率
            vol_avg = volatility.mean(dim=1, keepdim=True)  # [batch_size, 1]
            vol_features.append(vol_avg)
    
    # 拼接
    phi = torch.cat(vol_features, dim=1)  # [batch_size, len(window_sizes)]
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 3. 收益率分布形态谓词 (Return Distribution Phi)
# ============================================================================

def get_return_distribution_I_P(batch_x: torch.Tensor,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于收益率分布形态的 Phi 构造
    
    动机：
        收益率分布的高阶矩（偏度、峰度）反映了市场的非对称性和尾部风险。
        具有相似分布形态的样本可能面临相似的风险特征。
    
    数学表达式：
        skew = E[((x - μ) / σ)^3]  (偏度)
        kurt = E[((x - μ) / σ)^4] - 3  (超额峰度)
        phi = [mean, std, skew, kurt]
        phi = phi / ||phi||_2
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算时序上的统计量（对所有特征取均值）
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    # 均值
    mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 标准差
    std = x.std(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 标准化
    x_normalized = (x - mean) / (std + 1e-8)  # [batch_size, seq_len]
    
    # 偏度 (三阶矩)
    skewness = (x_normalized ** 3).mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 峰度 (四阶矩 - 3)
    kurtosis = (x_normalized ** 4).mean(dim=1, keepdim=True) - 3  # [batch_size, 1]
    
    # 处理 NaN
    skewness = torch.nan_to_num(skewness, nan=0.0)
    kurtosis = torch.nan_to_num(kurtosis, nan=0.0)
    
    # 拼接所有分布特征
    phi = torch.cat([mean, std, skewness, kurtosis], dim=1)  # [batch_size, 4]
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 4. 趋势强度谓词 (Trend Strength Phi)
# ============================================================================

def get_trend_strength_I_P(batch_x: torch.Tensor,
                            device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于趋势强度的 Phi 构造（类似 ADX 指标思想）
    
    动机：
        技术分析中，趋势强度是判断市场状态的重要指标。强趋势市场和震荡市场
        需要不同的交易策略。聚合具有相似趋势强度的样本可以提高模型的适应性。
    
    数学表达式：
        linear_fit: y = ax + b (对时序做线性回归)
        trend_direction = sign(a)  (趋势方向)
        trend_strength = |a| / std(x)  (趋势强度，归一化)
        r_squared = 线性拟合的 R²  (趋势可靠性)
        phi = [trend_direction, trend_strength, r_squared]
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 使用所有特征的均值作为代理序列
    y = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    # 构造时间索引
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
    
    # 线性回归: y = a*t + b
    # a = cov(t, y) / var(t)
    t_mean = t.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    
    cov_ty = ((t - t_mean) * (y - y_mean)).mean(dim=1, keepdim=True)
    var_t = ((t - t_mean) ** 2).mean(dim=1, keepdim=True)
    
    slope = cov_ty / (var_t + 1e-8)  # [batch_size, 1]
    
    # 趋势方向
    trend_direction = torch.sign(slope)  # [batch_size, 1]
    
    # 趋势强度（斜率的绝对值，按 y 的标准差归一化）
    y_std = y.std(dim=1, keepdim=True)
    trend_strength = torch.abs(slope) / (y_std + 1e-8)  # [batch_size, 1]
    
    # 计算 R² (拟合优度)
    y_pred = slope * t + (y_mean - slope * t_mean)
    ss_res = ((y - y_pred) ** 2).sum(dim=1, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum(dim=1, keepdim=True)
    r_squared = 1 - ss_res / (ss_tot + 1e-8)  # [batch_size, 1]
    r_squared = torch.clamp(r_squared, 0, 1)
    
    # 拼接
    phi = torch.cat([trend_direction, trend_strength, r_squared], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 5. 特征相关性结构谓词 (Feature Correlation Structure Phi)
# ============================================================================

def get_feature_correlation_I_P(batch_x: torch.Tensor,
                                 n_components: int = 10,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于特征相关性结构的 Phi 构造
    
    动机：
        Alpha158 包含多种因子，不同样本对这些因子的暴露程度不同。
        通过捕捉特征间的相关性结构，可以识别具有相似因子暴露的样本。
    
    数学表达式：
        对每个样本，计算特征在时间维度上的协方差矩阵的主成分
        phi = 协方差矩阵的上三角元素（或主成分）
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        n_components: 使用的主成分数量
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 对每个样本计算特征的时间维度均值和方差
    feature_mean = batch_x.mean(dim=1)  # [batch_size, feature_dim]
    feature_std = batch_x.std(dim=1)  # [batch_size, feature_dim]
    
    # 选取前 n_components 个特征的统计量
    n_components = min(n_components, feature_dim)
    
    # 结合均值和方差
    phi = torch.cat([
        feature_mean[:, :n_components], 
        feature_std[:, :n_components]
    ], dim=1)  # [batch_size, 2*n_components]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 6. 价量关系谓词 (Price-Volume Relationship Phi)
# ============================================================================

def get_price_volume_I_P(batch_x: torch.Tensor,
                          price_idx: int = 0,
                          volume_idx: int = 4,
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于价量关系的 Phi 构造
    
    动机：
        价量关系是技术分析的核心。量价齐升、量价背离等模式反映了市场
        参与者的行为特征。具有相似价量关系的样本可能处于相似的市场状态。
    
    数学表达式：
        price_change = (price_t - price_{t-1}) / price_{t-1}
        volume_change = (volume_t - volume_{t-1}) / volume_{t-1}
        pv_corr = corr(price_change, volume_change)
        phi = [pv_corr, avg_price_change, avg_volume_change]
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        price_idx: 价格特征的索引
        volume_idx: 成交量特征的索引
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 提取价格和成交量序列
    # 注意：如果索引超出范围，使用均值
    if price_idx < feature_dim and volume_idx < feature_dim:
        price = batch_x[:, :, price_idx]  # [batch_size, seq_len]
        volume = batch_x[:, :, volume_idx]  # [batch_size, seq_len]
    else:
        # 使用前半部分和后半部分特征的均值作为代理
        price = batch_x[:, :, :feature_dim//2].mean(dim=2)
        volume = batch_x[:, :, feature_dim//2:].mean(dim=2)
    
    # 计算变化率
    price_change = (price[:, 1:] - price[:, :-1]) / (torch.abs(price[:, :-1]) + 1e-8)
    volume_change = (volume[:, 1:] - volume[:, :-1]) / (torch.abs(volume[:, :-1]) + 1e-8)
    
    # 处理 NaN
    price_change = torch.nan_to_num(price_change, nan=0.0)
    volume_change = torch.nan_to_num(volume_change, nan=0.0)
    
    # 计算价量相关性
    price_mean = price_change.mean(dim=1, keepdim=True)
    volume_mean = volume_change.mean(dim=1, keepdim=True)
    
    cov_pv = ((price_change - price_mean) * (volume_change - volume_mean)).mean(dim=1, keepdim=True)
    std_p = price_change.std(dim=1, keepdim=True)
    std_v = volume_change.std(dim=1, keepdim=True)
    
    pv_corr = cov_pv / (std_p * std_v + 1e-8)  # [batch_size, 1]
    
    # 平均变化率
    avg_price_change = price_change.mean(dim=1, keepdim=True)  # [batch_size, 1]
    avg_volume_change = volume_change.mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([pv_corr, avg_price_change, avg_volume_change], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 7. 多尺度特征谓词 (Multi-Scale Feature Phi)
# ============================================================================

def get_multiscale_I_P(batch_x: torch.Tensor,
                        scales: list = [1, 4, 16, 32],
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于多尺度特征的 Phi 构造
    
    动机：
        金融时序具有多尺度特性，短期噪声、中期趋势、长期周期并存。
        通过在不同时间尺度上提取特征，可以捕捉样本在多个层面的相似性。
    
    数学表达式：
        对每个尺度 s，计算下采样后的均值和方差
        x_s = downsample(x, factor=s)
        phi_s = [mean(x_s), std(x_s)]
        phi = concat([phi_1, phi_4, phi_16, phi_32])
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        scales: 下采样尺度列表
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    scale_features = []
    
    for scale in scales:
        if scale <= seq_len:
            # 使用平均池化进行下采样
            # 先转置为 [batch_size, feature_dim, seq_len] 以便使用 avg_pool1d
            x_transposed = batch_x.permute(0, 2, 1)
            
            # 平均池化
            pooled = F.avg_pool1d(x_transposed, kernel_size=scale, stride=scale)
            
            # 计算池化后序列的均值和方差
            mean_val = pooled.mean(dim=2).mean(dim=1, keepdim=True)  # [batch_size, 1]
            std_val = pooled.std(dim=2).mean(dim=1, keepdim=True)  # [batch_size, 1]
            
            scale_features.append(mean_val)
            scale_features.append(std_val)
    
    # 拼接
    phi = torch.cat(scale_features, dim=1)  # [batch_size, 2*len(scales)]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 8. 收益率相似性聚类谓词 (Return Similarity Clustering Phi)
# ============================================================================

def get_return_similarity_I_P(batch_x: torch.Tensor,
                               n_bins: int = 5,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于收益率相似性的 Phi 构造（软聚类）
    
    动机：
        将样本按收益率分布进行软聚类，使得收益率模式相似的样本互相影响。
        这可以捕捉市场的结构性特征。
    
    数学表达式：
        计算每个样本的累计收益率和日内收益率分布
        使用分位数将样本映射到不同的区间
        phi = one_hot(bin_index) 或 softmax 软分配
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        n_bins: 分箱数量
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算累计变化（使用所有特征的均值）
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    cumulative_change = (x[:, -1] - x[:, 0]) / (torch.abs(x[:, 0]) + 1e-8)  # [batch_size]
    
    # 使用高斯核进行软分配
    # 创建 n_bins 个中心点
    centers = torch.linspace(-1, 1, n_bins, device=device)  # [n_bins]
    
    # 计算到每个中心的距离
    cumulative_change = cumulative_change.unsqueeze(1)  # [batch_size, 1]
    centers = centers.unsqueeze(0)  # [1, n_bins]
    
    # 高斯核软分配
    sigma = 0.5
    distances = -((cumulative_change - centers) ** 2) / (2 * sigma ** 2)
    phi = F.softmax(distances, dim=1)  # [batch_size, n_bins]
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 9. 时序模式谓词 (Temporal Pattern Phi)
# ============================================================================

def get_temporal_pattern_I_P(batch_x: torch.Tensor,
                              n_segments: int = 4,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于时序模式的 Phi 构造
    
    动机：
        将时间序列分割成多个片段，比较各片段的相对变化。
        具有相似时序模式（如先涨后跌、持续上涨等）的样本应被聚合。
    
    数学表达式：
        将序列分成 n_segments 个片段
        对每个片段计算: 均值变化、趋势方向
        phi = concat([segment_1_features, ..., segment_n_features])
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        n_segments: 分割的片段数
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 使用所有特征的均值
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    segment_len = seq_len // n_segments
    segment_features = []
    
    for i in range(n_segments):
        start_idx = i * segment_len
        end_idx = (i + 1) * segment_len if i < n_segments - 1 else seq_len
        
        segment = x[:, start_idx:end_idx]  # [batch_size, segment_len]
        
        # 片段的变化率
        segment_change = (segment[:, -1] - segment[:, 0]) / (torch.abs(segment[:, 0]) + 1e-8)
        segment_change = segment_change.unsqueeze(1)  # [batch_size, 1]
        
        # 片段的趋势方向 (用 sign)
        segment_trend = torch.sign(segment_change)  # [batch_size, 1]
        
        segment_features.append(segment_change)
        segment_features.append(segment_trend)
    
    # 拼接
    phi = torch.cat(segment_features, dim=1)  # [batch_size, 2*n_segments]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 10. 因子暴露谓词 (Factor Exposure Phi)
# ============================================================================

def get_factor_exposure_I_P(batch_x: torch.Tensor,
                             factor_groups: Optional[list] = None,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于因子暴露的 Phi 构造
    
    动机：
        Alpha158 特征可以分为不同的因子组（如价值因子、动量因子、波动因子等）。
        计算每个样本对不同因子组的暴露程度，聚合具有相似因子结构的样本。
    
    数学表达式：
        将 158 个特征分成 k 个因子组
        对每个因子组计算: exposure_k = mean(features_in_group_k)
        phi = [exposure_1, ..., exposure_k]
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        factor_groups: 因子分组列表，每个元素是一个特征索引列表
                       如果为 None，则自动按等间隔分组
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 默认分组：将特征均匀分成若干组
    if factor_groups is None:
        n_groups = 10  # 默认分成 10 组
        group_size = feature_dim // n_groups
        factor_groups = [
            list(range(i * group_size, min((i + 1) * group_size, feature_dim)))
            for i in range(n_groups)
        ]
    
    factor_exposures = []
    
    for group_indices in factor_groups:
        if len(group_indices) > 0:
            # 提取该因子组的特征
            group_features = batch_x[:, :, group_indices]  # [batch_size, seq_len, len(group)]
            
            # 计算因子暴露：时间和特征维度的均值
            exposure = group_features.mean(dim=(1, 2), keepdim=False)  # [batch_size]
            exposure = exposure.unsqueeze(1)  # [batch_size, 1]
            factor_exposures.append(exposure)
    
    # 拼接
    phi = torch.cat(factor_exposures, dim=1)  # [batch_size, n_groups]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 11. 技术指标综合谓词 (Technical Indicator Phi)
# ============================================================================

def get_technical_indicator_I_P(batch_x: torch.Tensor,
                                 short_window: int = 5,
                                 long_window: int = 20,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于技术指标的 Phi 构造（模拟 RSI、MACD 等指标的思想）
    
    动机：
        技术指标是量化交易的重要工具。通过构造类似 RSI、MACD 的指标，
        可以捕捉样本的超买超卖状态和趋势动量。
    
    数学表达式：
        RSI_proxy = 上涨天数 / 总天数
        MACD_proxy = short_MA - long_MA
        BB_proxy = (price - MA) / std
        phi = [RSI_proxy, MACD_proxy, BB_proxy]
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        short_window: 短期均线窗口
        long_window: 长期均线窗口
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 使用所有特征的均值
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    # RSI 代理：计算上涨天数占比
    changes = x[:, 1:] - x[:, :-1]  # [batch_size, seq_len-1]
    up_days = (changes > 0).float().sum(dim=1, keepdim=True)
    total_days = changes.shape[1]
    rsi_proxy = up_days / total_days  # [batch_size, 1]
    
    # MACD 代理：短期均值 - 长期均值
    short_ma = x[:, -short_window:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    long_ma = x[:, -long_window:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    macd_proxy = short_ma - long_ma  # [batch_size, 1]
    
    # 布林带代理：(当前价格 - 均值) / 标准差
    current_price = x[:, -1:].mean(dim=1, keepdim=True)  # [batch_size, 1]
    price_mean = x[:, -long_window:].mean(dim=1, keepdim=True)
    price_std = x[:, -long_window:].std(dim=1, keepdim=True)
    bb_proxy = (current_price - price_mean) / (price_std + 1e-8)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([rsi_proxy, macd_proxy, bb_proxy], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 12. 波动率-收益联合谓词 (Volatility-Return Joint Phi)
# ============================================================================

def get_vol_return_joint_I_P(batch_x: torch.Tensor,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于波动率-收益联合分布的 Phi 构造
    
    动机：
        高收益往往伴随高风险（波动率）。通过捕捉波动率和收益的联合分布，
        可以识别具有相似风险-收益特征的样本。
    
    数学表达式：
        return = (x_T - x_1) / x_1
        volatility = std(x)
        sharpe_proxy = return / volatility  (夏普比率代理)
        phi = [return, volatility, sharpe_proxy]
        P = phi @ phi.T
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 使用所有特征的均值
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    # 收益率
    returns = (x[:, -1] - x[:, 0]) / (torch.abs(x[:, 0]) + 1e-8)
    returns = returns.unsqueeze(1)  # [batch_size, 1]
    
    # 波动率
    volatility = x.std(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 夏普比率代理
    sharpe = returns / (volatility + 1e-8)  # [batch_size, 1]
    
    # 最大回撤代理
    cummax = torch.cummax(x, dim=1)[0]  # [batch_size, seq_len]
    drawdown = (cummax - x) / (cummax + 1e-8)  # [batch_size, seq_len]
    max_drawdown = drawdown.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([returns, volatility, sharpe, max_drawdown], dim=1)  # [batch_size, 4]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 13. 市场 Beta 谓词 (Market Beta Phi) - 基于股票间相关性
# ============================================================================

def get_market_beta_I_P(batch_x: torch.Tensor,
                         device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于市场 Beta 的 Phi 构造
    
    动机：
        Beta 衡量个股对市场整体波动的敏感度。具有相似 Beta 的股票
        往往对系统性风险有相似的暴露，应该被聚合处理。
        这体现了 CAPM 模型中的系统性风险思想。
    
    数学表达式：
        market_return = mean(all_sample_returns)  (市场收益率代理)
        beta_i = cov(r_i, r_m) / var(r_m)  (个股 Beta)
        alpha_i = mean(r_i) - beta_i * mean(r_m)  (Jensen's Alpha)
        phi = [beta, alpha, residual_vol]
        P = phi @ phi.T
    
    计算复杂度: O(B × L × D + B²)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算每个样本的收益率序列（使用所有特征的均值）
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)  # [batch_size, seq_len-1]
    returns = torch.nan_to_num(returns, nan=0.0)
    
    # 市场收益率：所有样本的平均收益率
    market_return = returns.mean(dim=0, keepdim=True)  # [1, seq_len-1]
    market_return = market_return.expand(batch_size, -1)  # [batch_size, seq_len-1]
    
    # 计算 Beta: cov(r_i, r_m) / var(r_m)
    r_mean = returns.mean(dim=1, keepdim=True)
    m_mean = market_return.mean(dim=1, keepdim=True)
    
    cov_rm = ((returns - r_mean) * (market_return - m_mean)).mean(dim=1, keepdim=True)
    var_m = ((market_return - m_mean) ** 2).mean(dim=1, keepdim=True)
    
    beta = cov_rm / (var_m + 1e-8)  # [batch_size, 1]
    
    # Jensen's Alpha
    alpha = r_mean - beta * m_mean  # [batch_size, 1]
    
    # 残差波动率（特异性风险）
    predicted_return = alpha + beta * market_return
    residual = returns - predicted_return
    residual_vol = residual.std(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([beta, alpha, residual_vol], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 14. 跨样本相关性谓词 (Cross-Sample Correlation Phi) - 基于股票间相关性
# ============================================================================

def get_cross_correlation_I_P(batch_x: torch.Tensor,
                               use_returns: bool = True,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于跨样本相关性的 Phi 构造
    
    动机：
        直接计算 batch 内各样本之间的相关性矩阵，作为 P 矩阵的基础。
        这直接捕捉了股票之间的联动性，体现板块效应。
    
    数学表达式：
        corr_ij = corr(r_i, r_j)  (样本 i 和 j 的相关系数)
        P = softmax(corr_matrix)  (归一化为概率分布)
    
    计算复杂度: O(B² × L)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        use_returns: 是否使用收益率计算相关性
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 使用所有特征的均值
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    
    if use_returns:
        # 转换为收益率
        x = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
        x = torch.nan_to_num(x, nan=0.0)
    
    # 标准化
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    x_normalized = (x - x_mean) / (x_std + 1e-8)  # [batch_size, seq_len-1]
    
    # 计算相关系数矩阵
    # corr(i,j) = (x_i @ x_j) / (len * std_i * std_j)
    # 由于已经标准化，直接计算内积除以长度
    corr_matrix = torch.mm(x_normalized, x_normalized.T) / x_normalized.shape[1]  # [batch_size, batch_size]
    
    # 将相关系数映射到 [0, 1] 并归一化
    P = (corr_matrix + 1) / 2  # 相关系数从 [-1,1] 映射到 [0,1]
    P = P / (torch.linalg.norm(P) + 1e-8)
    
    return I, P


# ============================================================================
# 15. 领先滞后关系谓词 (Lead-Lag Relationship Phi) - 基于股票间相关性
# ============================================================================

def get_lead_lag_I_P(batch_x: torch.Tensor,
                      max_lag: int = 5,
                      device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于领先滞后关系的 Phi 构造
    
    动机：
        信息在市场中的传导存在时间差，某些股票（如大盘股）可能领先于
        其他股票（如小盘股）。通过计算领先滞后相关性，可以捕捉这种
        信息传导效应，体现市场微观结构。
    
    数学表达式：
        lag_corr_ij(k) = corr(r_i(t), r_j(t+k))  (滞后 k 期相关性)
        optimal_lag_ij = argmax_k |lag_corr_ij(k)|  (最优滞后期)
        phi_i = [max_lead_corr, max_lag_corr, net_lead_indicator]
        P = phi @ phi.T
    
    计算复杂度: O(B² × L × max_lag)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        max_lag: 最大滞后期数
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算收益率
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 标准化
    r_mean = returns.mean(dim=1, keepdim=True)
    r_std = returns.std(dim=1, keepdim=True)
    r_normalized = (returns - r_mean) / (r_std + 1e-8)
    
    # 计算与市场的领先滞后相关性
    market_return = r_normalized.mean(dim=0)  # [seq_len-1]
    
    lead_corrs = []  # 样本领先于市场
    lag_corrs = []   # 样本滞后于市场
    
    for lag in range(1, min(max_lag + 1, seq_len - 1)):
        # 样本领先：样本 t 与 市场 t+lag 的相关性
        sample_early = r_normalized[:, :-lag]  # [batch_size, seq_len-1-lag]
        market_late = market_return[lag:]  # [seq_len-1-lag]
        lead_corr = (sample_early * market_late.unsqueeze(0)).mean(dim=1)  # [batch_size]
        lead_corrs.append(lead_corr)
        
        # 样本滞后：样本 t+lag 与 市场 t 的相关性
        sample_late = r_normalized[:, lag:]  # [batch_size, seq_len-1-lag]
        market_early = market_return[:-lag]  # [seq_len-1-lag]
        lag_corr = (sample_late * market_early.unsqueeze(0)).mean(dim=1)  # [batch_size]
        lag_corrs.append(lag_corr)
    
    # 取最大领先和滞后相关性
    lead_corrs = torch.stack(lead_corrs, dim=1)  # [batch_size, max_lag]
    lag_corrs = torch.stack(lag_corrs, dim=1)  # [batch_size, max_lag]
    
    max_lead = lead_corrs.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    max_lag_val = lag_corrs.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    
    # 净领先指标：正值表示倾向于领先，负值表示倾向于滞后
    net_lead = max_lead - max_lag_val  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([max_lead, max_lag_val, net_lead], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 16. 共同因子暴露谓词 (Common Factor Exposure Phi) - 基于股票间相关性
# ============================================================================

def get_common_factor_I_P(batch_x: torch.Tensor,
                           n_factors: int = 3,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于共同因子暴露的 Phi 构造
    
    动机：
        通过 PCA 提取 batch 内样本的共同因子，然后根据每个样本对这些
        因子的暴露程度构造 phi。具有相似因子暴露的样本可能属于同一板块
        或受相同宏观因素驱动。这体现了 APT 套利定价理论的思想。
    
    数学表达式：
        X = [r_1, r_2, ..., r_B]^T  (收益率矩阵)
        U, S, V = SVD(X)  (奇异值分解)
        factor_loadings = U[:, :k] * S[:k]  (因子载荷)
        phi = factor_loadings
        P = phi @ phi.T
    
    计算复杂度: O(B × L × D + min(B, L)³)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        n_factors: 提取的因子数量
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算收益率
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 中心化
    returns_centered = returns - returns.mean(dim=0, keepdim=True)
    
    # SVD 分解
    try:
        U, S, Vh = torch.linalg.svd(returns_centered, full_matrices=False)
        
        # 取前 n_factors 个因子
        n_factors = min(n_factors, min(batch_size, seq_len - 1))
        
        # 因子载荷：U * S
        factor_loadings = U[:, :n_factors] * S[:n_factors].unsqueeze(0)  # [batch_size, n_factors]
        
        phi = factor_loadings
    except:
        # SVD 失败时使用简化方法
        phi = returns.mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 17. 波动率同步性谓词 (Volatility Synchronization Phi) - 基于股票间相关性
# ============================================================================

def get_volatility_sync_I_P(batch_x: torch.Tensor,
                             window: int = 10,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于波动率同步性的 Phi 构造
    
    动机：
        波动率的同步性反映了市场的系统性风险传染。当市场进入高波动期，
        各股票的波动率往往同步上升（波动率聚类的跨资产版本）。
        具有相似波动率同步性的样本可能暴露于相同的系统性风险。
    
    数学表达式：
        vol_i(t) = rolling_std(r_i, window)  (滚动波动率)
        vol_sync_ij = corr(vol_i, vol_j)  (波动率同步性)
        phi_i = [avg_vol_sync, vol_beta, vol_level]
        P = phi @ phi.T
    
    计算复杂度: O(B × L × W + B²)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        window: 滚动窗口大小
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算收益率
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 计算滚动波动率
    n_windows = (seq_len - 1) // window
    if n_windows < 2:
        # 窗口数太少，使用简化方法
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
    
    # 市场波动率（所有样本的平均）
    market_vol = rolling_vols.mean(dim=0, keepdim=True)  # [1, n_windows]
    
    # 波动率同步性：与市场波动率的相关性
    vol_mean = rolling_vols.mean(dim=1, keepdim=True)
    market_vol_mean = market_vol.mean(dim=1, keepdim=True)
    
    cov_vm = ((rolling_vols - vol_mean) * (market_vol - market_vol_mean)).mean(dim=1, keepdim=True)
    std_v = rolling_vols.std(dim=1, keepdim=True)
    std_m = market_vol.std(dim=1, keepdim=True)
    
    vol_sync = cov_vm / (std_v * std_m + 1e-8)  # [batch_size, 1]
    
    # 波动率 Beta：波动率对市场波动率的敏感度
    var_m = ((market_vol - market_vol_mean) ** 2).mean(dim=1, keepdim=True)
    vol_beta = cov_vm / (var_m + 1e-8)  # [batch_size, 1]
    
    # 波动率水平
    vol_level = rolling_vols.mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([vol_sync, vol_beta, vol_level], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 18. 尾部风险共现谓词 (Tail Risk Co-occurrence Phi) - 基于股票间相关性
# ============================================================================

def get_tail_risk_I_P(batch_x: torch.Tensor,
                       quantile: float = 0.1,
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于尾部风险共现的 Phi 构造
    
    动机：
        在市场极端事件中，资产之间的相关性往往会上升（相关性破裂）。
        尾部风险的共现频率反映了系统性风险的传染性。
        具有相似尾部风险特征的样本应该被聚合。
    
    数学表达式：
        left_tail_i = I(r_i < quantile(r_i, q))  (左尾指示变量)
        right_tail_i = I(r_i > quantile(r_i, 1-q))  (右尾指示变量)
        tail_freq_i = mean(left_tail_i)  (尾部事件频率)
        phi = [left_tail_freq, right_tail_freq, tail_asymmetry, avg_tail_return]
        P = phi @ phi.T
    
    计算复杂度: O(B × L × D + B × L log L)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        quantile: 尾部分位数阈值
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算收益率
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 计算分位数阈值
    left_threshold = torch.quantile(returns, quantile, dim=1, keepdim=True)  # [batch_size, 1]
    right_threshold = torch.quantile(returns, 1 - quantile, dim=1, keepdim=True)  # [batch_size, 1]
    
    # 左尾和右尾事件
    left_tail = (returns < left_threshold).float()  # [batch_size, seq_len-1]
    right_tail = (returns > right_threshold).float()  # [batch_size, seq_len-1]
    
    # 尾部事件频率
    left_freq = left_tail.mean(dim=1, keepdim=True)  # [batch_size, 1]
    right_freq = right_tail.mean(dim=1, keepdim=True)  # [batch_size, 1]
    
    # 尾部不对称性
    tail_asymmetry = right_freq - left_freq  # [batch_size, 1]
    
    # 平均尾部收益（条件期望）
    left_returns = returns * left_tail
    left_sum = left_returns.sum(dim=1, keepdim=True)
    left_count = left_tail.sum(dim=1, keepdim=True)
    avg_left_return = left_sum / (left_count + 1e-8)  # [batch_size, 1]
    
    right_returns = returns * right_tail
    right_sum = right_returns.sum(dim=1, keepdim=True)
    right_count = right_tail.sum(dim=1, keepdim=True)
    avg_right_return = right_sum / (right_count + 1e-8)  # [batch_size, 1]
    
    # 尾部风险比率
    tail_ratio = torch.abs(avg_left_return) / (torch.abs(avg_right_return) + 1e-8)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([left_freq, right_freq, tail_asymmetry, tail_ratio], dim=1)  # [batch_size, 4]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 19. 市场状态敏感性谓词 (Market Regime Sensitivity Phi) - 基于股票间相关性
# ============================================================================

def get_market_regime_I_P(batch_x: torch.Tensor,
                           device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于市场状态敏感性的 Phi 构造
    
    动机：
        不同股票在牛市和熊市中的表现差异很大。通过计算样本在不同市场
        状态下的表现差异，可以识别防御型和进攻型股票。
        这体现了市场时机选择的思想。
    
    数学表达式：
        market_up = I(r_m > 0)  (市场上涨指示)
        r_up_i = mean(r_i | market_up)  (上涨市场中的平均收益)
        r_down_i = mean(r_i | !market_up)  (下跌市场中的平均收益)
        upside_capture = r_up_i / r_m_up  (上行捕获率)
        downside_capture = r_down_i / r_m_down  (下行捕获率)
        phi = [upside_capture, downside_capture, capture_ratio]
        P = phi @ phi.T
    
    计算复杂度: O(B × L × D)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 计算收益率
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 市场收益率
    market_return = returns.mean(dim=0)  # [seq_len-1]
    
    # 市场状态
    market_up = (market_return > 0).float()  # [seq_len-1]
    market_down = 1 - market_up  # [seq_len-1]
    
    # 上涨市场中的表现
    up_returns = returns * market_up.unsqueeze(0)  # [batch_size, seq_len-1]
    up_sum = up_returns.sum(dim=1, keepdim=True)
    up_count = market_up.sum()
    avg_up_return = up_sum / (up_count + 1e-8)  # [batch_size, 1]
    
    # 下跌市场中的表现
    down_returns = returns * market_down.unsqueeze(0)  # [batch_size, seq_len-1]
    down_sum = down_returns.sum(dim=1, keepdim=True)
    down_count = market_down.sum()
    avg_down_return = down_sum / (down_count + 1e-8)  # [batch_size, 1]
    
    # 市场平均收益
    market_up_avg = (market_return * market_up).sum() / (up_count + 1e-8)
    market_down_avg = (market_return * market_down).sum() / (down_count + 1e-8)
    
    # 捕获率
    upside_capture = avg_up_return / (market_up_avg + 1e-8)  # [batch_size, 1]
    downside_capture = avg_down_return / (market_down_avg + 1e-8)  # [batch_size, 1]
    
    # 捕获率比值（>1 表示防御型，<1 表示进攻型）
    capture_ratio = upside_capture / (torch.abs(downside_capture) + 1e-8)  # [batch_size, 1]
    
    # 拼接
    phi = torch.cat([upside_capture, downside_capture, capture_ratio], dim=1)  # [batch_size, 3]
    
    # 处理 NaN
    phi = torch.nan_to_num(phi, nan=0.0)
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 20. 板块聚类谓词 (Sector Clustering Phi) - 基于股票间相关性
# ============================================================================

def get_sector_clustering_I_P(batch_x: torch.Tensor,
                               n_clusters: int = 5,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于板块聚类的 Phi 构造
    
    动机：
        同一板块/行业的股票往往具有相似的价格走势。通过对样本进行
        无监督聚类，可以发现隐含的板块结构。使用 K-Means 风格的
        软聚类分配。
    
    数学表达式：
        使用特征向量进行软聚类
        dist_ik = ||feature_i - center_k||²  (到聚类中心的距离)
        phi = softmax(-dist / temperature)  (软分配概率)
        P = phi @ phi.T
    
    计算复杂度: O(B × L × D + B × K)
    
    Args:
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        n_clusters: 聚类数量（模拟板块数量）
        device: 计算设备
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    batch_size, seq_len, feature_dim = batch_x.shape
    if device is None:
        device = batch_x.device
    
    I = torch.eye(batch_size, device=device)
    
    # 提取特征：使用收益率的统计量
    x = batch_x.mean(dim=2)  # [batch_size, seq_len]
    returns = (x[:, 1:] - x[:, :-1]) / (torch.abs(x[:, :-1]) + 1e-8)
    returns = torch.nan_to_num(returns, nan=0.0)  # [batch_size, seq_len-1]
    
    # 构建特征向量：[mean, std, skew, autocorr]
    r_mean = returns.mean(dim=1, keepdim=True)
    r_std = returns.std(dim=1, keepdim=True)
    r_normalized = (returns - r_mean) / (r_std + 1e-8)
    r_skew = (r_normalized ** 3).mean(dim=1, keepdim=True)
    
    # 一阶自相关
    autocorr = (r_normalized[:, :-1] * r_normalized[:, 1:]).mean(dim=1, keepdim=True)
    
    features = torch.cat([r_mean, r_std, r_skew, autocorr], dim=1)  # [batch_size, 4]
    features = torch.nan_to_num(features, nan=0.0)
    
    # 简化的聚类：使用均匀分布的中心点
    # 实际应用中可以使用 K-Means 迭代
    feature_min = features.min(dim=0, keepdim=True)[0]
    feature_max = features.max(dim=0, keepdim=True)[0]
    feature_range = feature_max - feature_min + 1e-8
    
    # 归一化特征
    features_normalized = (features - feature_min) / feature_range
    
    # 创建聚类中心（均匀分布）
    centers = torch.zeros(n_clusters, features.shape[1], device=device)
    for k in range(n_clusters):
        centers[k] = (k + 0.5) / n_clusters
    
    # 计算到每个中心的距离
    # features_normalized: [batch_size, 4]
    # centers: [n_clusters, 4]
    distances = torch.cdist(features_normalized, centers)  # [batch_size, n_clusters]
    
    # 软分配
    temperature = 0.5
    phi = F.softmax(-distances / temperature, dim=1)  # [batch_size, n_clusters]
    
    # 归一化
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    
    P = torch.mm(phi, phi.T)
    
    return I, P


# ============================================================================
# 统一接口：根据 phi_type 获取对应的 I 和 P 矩阵
# ============================================================================

def get_custom_I_P(phi_type: str, 
                   batch_x: torch.Tensor, 
                   device: Optional[torch.device] = None,
                   **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一接口：根据 phi_type 字符串获取对应的 I 和 P 矩阵
    
    Args:
        phi_type: phi 类型字符串，支持以下类型：
            - 'phi_momentum': 动量因子谓词
            - 'phi_volatility': 波动率聚类谓词
            - 'phi_return_dist': 收益率分布形态谓词
            - 'phi_trend': 趋势强度谓词
            - 'phi_feature_corr': 特征相关性结构谓词
            - 'phi_price_volume': 价量关系谓词
            - 'phi_multiscale': 多尺度特征谓词
            - 'phi_return_sim': 收益率相似性聚类谓词
            - 'phi_temporal': 时序模式谓词
            - 'phi_factor': 因子暴露谓词
            - 'phi_technical': 技术指标综合谓词
            - 'phi_vol_return': 波动率-收益联合谓词
        batch_x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        device: 计算设备
        **kwargs: 其他参数
    
    Returns:
        I: 单位矩阵 [batch_size, batch_size]
        P: 相似性矩阵 [batch_size, batch_size]
    """
    if device is None:
        device = batch_x.device
    
    phi_type_map = {
        # 基于时间序列特性的谓词
        'phi_momentum': get_momentum_I_P,
        'phi_volatility': get_volatility_I_P,
        'phi_return_dist': get_return_distribution_I_P,
        'phi_trend': get_trend_strength_I_P,
        'phi_multiscale': get_multiscale_I_P,
        'phi_temporal': get_temporal_pattern_I_P,
        
        # 基于金融指标关联性的谓词
        'phi_feature_corr': get_feature_correlation_I_P,
        'phi_price_volume': get_price_volume_I_P,
        'phi_factor': get_factor_exposure_I_P,
        'phi_technical': get_technical_indicator_I_P,
        'phi_vol_return': get_vol_return_joint_I_P,
        
        # 基于股票间相关性的谓词
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
        # 默认返回单位矩阵
        batch_size = batch_x.shape[0]
        I = torch.eye(batch_size, device=device)
        return I, I


# ============================================================================
# 组合谓词：多个 phi 的加权组合
# ============================================================================

def get_combined_I_P(phi_types: list,
                     weights: Optional[list] = None,
                     batch_x: torch.Tensor = None,
                     device: Optional[torch.device] = None,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    组合多个 phi 谓词构造 P 矩阵
    
    动机：
        单一谓词可能无法完全捕捉样本间的复杂关系。
        通过组合多个谓词，可以从多个角度刻画样本相似性。
    
    数学表达式：
        P_combined = Σ w_i * P_i
        P_combined = P_combined / ||P_combined||_F
    
    Args:
        phi_types: phi 类型列表
        weights: 权重列表，如果为 None 则等权重
        batch_x: 输入张量
        device: 计算设备
        **kwargs: 其他参数
    
    Returns:
        I: 单位矩阵
        P: 组合后的相似性矩阵
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
    
    # 归一化
    P_combined = P_combined / (torch.linalg.norm(P_combined) + 1e-8)
    
    return I, P_combined


# ============================================================================
# 测试函数
# ============================================================================

def test_phi_constructors():
    """
    测试所有 phi 构造函数
    """
    print("=" * 60)
    print("Testing Custom Phi Constructors")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, feature_dim = 32, 96, 158
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x = torch.randn(batch_size, seq_len, feature_dim, device=device)
    
    phi_types = [
        # 基于时间序列特性
        'phi_momentum',
        'phi_volatility', 
        'phi_return_dist',
        'phi_trend',
        'phi_multiscale',
        'phi_temporal',
        
        # 基于金融指标关联性
        'phi_feature_corr',
        'phi_price_volume',
        'phi_factor',
        'phi_technical',
        'phi_vol_return',
        
        # 基于股票间相关性
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
            
            # 验证 P 矩阵的对称性
            assert torch.allclose(P, P.T, atol=1e-5), f"{phi_type}: P is not symmetric"
            
        except Exception as e:
            print(f"✗ {phi_type:20s} | Error: {e}")
    
    # 测试组合谓词
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
