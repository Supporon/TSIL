import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fft
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf

from tqdm import tqdm
import time

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model

from types import SimpleNamespace
import matplotlib.pyplot as plt

# vis
import plotly
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

from src.models.PDF import PDF
from src.models.PatchTST import PatchTST
from src.models.TimeMixer import TimeMixer
from src.models.TimesNet import TimesNet
from src.models.SegRNN import SegRNN
from src.models.diffusion_stock import DiffStock
from src.models.Crossformer import Crossformer
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.models.Transformer import Transformer
# from src.models.Mamba import mamba
from src.models.TCN import TCN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.LSR_IGRU import LSR_IGRU, LSRIGRU

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def skewness(x, dim=1):
    # m3 = torch.mean((x - x.mean(dim, keepdim=True))**3, dim)
    # m2 = torch.mean((x - x.mean(dim, keepdim=True))**2, dim)
    # skew = m3 / (m2 ** 1.5)
    # # 无偏修正系数（可选）
    # n = a.size(dim)
    # skew *= (n * (n - 1)) ** 0.5 / (n - 2)   # 无偏修正

    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    skew = torch.mean(((x - mean) / (std + 1e-8)) ** 3, dim=dim)

    return skew

def torch_kurtosis(x: torch.Tensor, dim: int = -1, unbiased: bool = True):
    """
    纯 PyTorch 实现峰度（超峰度，已减 3）
    返回形状与 x.mean(dim) 相同
    """
    # n = x.size(dim)
    # mean = x.mean(dim, keepdim=True)
    # var  = ((x - mean) ** 2).mean(dim, keepdim=True)
    # m4   = ((x - mean) ** 4).mean(dim, keepdim=True)
    # k0   = m4 / (var ** 2)          # 原始峰度
    # if unbiased and n > 3:          # Fisher 无偏修正
    #     k0 = (k0 - 3) * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 3

    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    kurtosis = torch.mean(((x - mean) / (std + 1e-8)) ** 4, dim=dim) - 3

    return kurtosis #(k0 - 3).squeeze(dim)    # 超峰度


def FFT_for_Period_batch(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=0)
    # find period by amplitudes
    frequency_list = abs(xf).mean(1).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[0] // top_list
    return period, abs(xf).mean(-1)#[:, top_list]

def FFT_for_Period_batch_per_sample(x, k=1):
    # x shape: [B, T, C] = [256, 60, 20]
    xf = torch.fft.rfft(x, dim=1)  # 沿时间维度进行FFT，得到 [256, 31, 20]
    # 计算幅度谱并找到主要频率
    amplitude = abs(xf)  # [256, 31, 20]
    amplitude_mean = amplitude.mean(-1)  # 沿特征维度平均，得到 [256, 31]
    amplitude_mean[:, 0] = 0  # 忽略直流分量
    # 找到每个样本的主要频率
    _, top_indices = torch.topk(amplitude_mean, k, dim=1)  # [256, k]
    # 计算周期长度
    seq_len = x.shape[1]
    periods = seq_len / top_indices.float()  # [256, k]
    return periods, abs(xf).mean(-1)#[:, top_indices]  # 返回每个样本的周期长度

def batch_acf_fft(data, n_lags, axis=1):
    """
    使用FFT方法批量计算自相关函数

    参数:
    data: 输入数据，形状为 (batch_size, timesteps, features) 或 (timesteps, features)
    n_lags: 要计算的滞后阶数
    axis: 计算自相关的时间轴

    返回:
    acf_values: 自相关值数组
    """
    # 确保数据是三维的 (batch_size, timesteps, features)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    batch_size, timesteps, n_features = data.shape
    # 预分配结果数组
    acf_values = np.zeros((batch_size, n_lags + 1, n_features))
    # 对每个特征并行处理
    for feature_idx in range(n_features):
        # 提取当前特征的所有批次数据
        feature_data = data[:, :, feature_idx]  # 形状: (batch_size, timesteps)
        # 使用列表推导式替代内部循环
        acf_results = [acf(series, nlags=n_lags, fft=True)
                       for series in feature_data]
        # 将结果存储到预分配的数组中
        acf_values[:, :, feature_idx] = np.array(acf_results)

    return acf_values

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.use_phi = loss_params.get('use_phi')

    def get_ones_I_P(self, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        # phi=1
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        # P = torch.ones((batch_size, batch_size), device=device)  # 全1矩阵
        phi = torch.ones((batch_size, 1), device=device)
        phi = phi / torch.linalg.norm(phi)  # 模一的归一化
        P = torch.mm(phi, phi.T)
        return I, P

    def get_x_I_P(self, phi_type, batch_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        # P = torch.zeros((batch_size, batch_size), device=device)
        # for m in range(batch_x.shape[-1]):
        #     temp_phi = batch_x[:, -1, m].unsqueeze(0)  # 9*9 / 90*90；在每个特征上，所有时间点的均值保持一致? 在每个时间点上，所有特征的均值保持一致？
        #     # temp_phi = batch_x[:, :, m].mean(dim=1)
        #     # temp_phi = (temp_phi / torch.linalg.norm(temp_phi)).unsqueeze(0)
        #     temp_P = torch.mm(temp_phi.T, temp_phi)
        #     P = P + temp_P

        ## phi=所有时刻的所有特征均值
        if phi_type== 'mean_dim':
            phi = batch_x[:, :, :].mean(axis=2)
        elif phi_type == 'mean_sl':
            ## phi=所有时刻均值的所有特征
            phi = batch_x[:, :, :].mean(axis=1)
        else:
            ## phi=当前时刻的所有特征
            phi = batch_x[:, -1, :]

        ## L2归一化
        # phi = phi / torch.linalg.norm(phi)
        ## 绝对值归一化
        phi = torch.abs(phi) / torch.max(torch.abs(phi))

        P = torch.mm(phi, phi.T)

        return I, P

    def get_onex_I_P(self, batch_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        ## phi = ret
        # phi = batch_x[:, -1, 1]
        ## phi = ret_mean(seq)
        # phi = batch_x[:, :, 1].mean(dim=1)
        # phi = (phi / torch.linalg.norm(phi)).unsqueeze(0)
        # P = torch.mm(phi.T, phi)

        ## phi = seq_ret
        phi = batch_x[:, :, 1]
        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return I, P

    def get_timestamp_I_P(self, phi_type, timestamp_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        # phi_type=phi_timestamp_last;phi_timestamp_dim;phi_timestamp_sl
        if phi_type == 'phi_timestamp_dim':
            # 取所有时刻点平铺(按行平铺,即按照一个seq_len的每个点其时间戳特征平铺3+3+...3)
            phi = timestamp_x.view(batch_size, -1)
        elif phi_type == 'phi_timestamp_sl':
            # 取所有时刻点平铺(按列平铺,即按照将seq_len中所有点的时间戳的第一维表示放一起60+60+60)
            phi = timestamp_x.permute(0, 2, 1).contiguous()
            phi = phi.view(batch_size, -1)
        else:
            # 取当前时刻点
            phi = timestamp_x[:, -1, :]

        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return I, P

    def get_period_I_P(self, phi_type, batch_x, device):
        """
        引入谓词,构造V和P矩阵
        """
        batch_size, seq_len, dim = batch_x.shape
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        if 'per_sample' in phi_type:
            # 获取每个样本的周期
            periods, _ = FFT_for_Period_batch_per_sample(batch_x, k=1)  # [256, 1]
            # 为每个样本创建时间索引
            batch_size, seq_len, _ = batch_x.shape
            time_indices = torch.arange(1, seq_len + 1, device=device)  # [60]
            # 为每个样本计算周期索引
            # 使用广播机制扩展维度
            periods_expanded = periods.unsqueeze(-1)  # [256, 1, 1]
            time_indices_expanded = time_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, 60]
            # 计算周期索引
            period_indices = torch.ceil(time_indices_expanded / periods_expanded)  # [256, 1, 60]
            # 重塑为最终形状 [256, 60]
            phi = period_indices.squeeze(1)  # 移除中间的维度
        else:
            period_list, _ = FFT_for_Period_batch(batch_x, 1)
            # 修复：将numpy.int64转换为tensor
            if isinstance(period_list[0], (int, float, np.integer)):
                period_value = torch.tensor(period_list[0], device=device, dtype=torch.float32)
            else:
                period_value = period_list[0].to(device).float() if hasattr(period_list[0], 'to') else torch.tensor(period_list[0], device=device, dtype=torch.float32)# 添加保护：确保周期不为0
            # 添加保护：确保周期不为0
            period_value = torch.clamp(period_value, min=1.0)
            # 创建时间索引矩阵 (seq_len, dim)
            time_indices = torch.arange(1, batch_size + 1, device=device)
            # 使用向量化操作计算周期索引
            period_indices = torch.ceil(time_indices / (period_value + 1e-8)).long()
            # 确保索引在合理范围内
            period_indices = torch.clamp(period_indices, min=1)

            if 'onehot' in phi_type:
                num_classes = period_indices.max().item()
                phi = torch.nn.functional.one_hot(period_indices - 1, num_classes=num_classes).float()
            else:
                phi = period_indices.unsqueeze(dim=1).float()

        phi = phi / torch.linalg.norm(phi)
        if phi.device != device:
            phi = phi.to(device)
        P = torch.mm(phi, phi.T)
        return I, P

    def get_autocorr_I_P(self, phi_type, batch_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        acf_results = batch_acf_fft(batch_x.cpu(), n_lags=1).reshape(batch_size, -1)
        acf_results[np.isnan(acf_results)] = 0

        phi = torch.from_numpy(acf_results).float().to(device)
        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return I, P

    def get_random_class_I_P(self, phi_type, num_class, batch_x, device):
        """
        引入谓词,构造V和P矩阵
        """
        batch_size, seq_len, dim = batch_x.shape
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        rand_label = torch.randint(low=0, high=num_class, size=(batch_size,))

        if 'onehot' in phi_type:
            phi = torch.nn.functional.one_hot(rand_label, num_classes=num_class).float().to(device)
        else:
            phi = rand_label.unsqueeze(dim=1).float().to(device)

        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return I, P

    def get_stats_I_P(self, phi_type, batch_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        """
        I = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        # phi_type=phi_timestamp_last;phi_timestamp_dim;phi_timestamp_sl
        if phi_type == 'phi_stats_mean':
            phi = batch_x.mean(dim=1)
        elif phi_type == 'phi_stats_var':
            phi = batch_x.var(dim=1)
        # elif phi_type == 'phi_stats_median':
        #     phi = torch.median(batch_x, dim=1).values
        # elif phi_type == 'phi_stats_range':
        #     # 极差
        #     phi = torch.amax(batch_x,1) - torch.amin(batch_x,1)
        # elif phi_type == 'phi_stats_min':
        #     phi = batch_x.std(dim=1)
        # elif phi_type == 'phi_stats_max':
        #     phi = batch_x.std(dim=1)
        elif phi_type == 'phi_stats_skew':
            # 偏度
            phi = skewness(batch_x, dim=1)
            phi = torch.nan_to_num(phi, nan=0.0)
        elif phi_type == 'phi_stats_kurt':
            # 峰度
            phi = torch_kurtosis(batch_x, dim=1)
            phi = torch.nan_to_num(phi, nan=0.0)
        else:
            # 默认取均值
            phi = batch_x.mean(dim=1)

        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return I, P

    def forward(self, phi_type, batch_x, timestamp_x, outputs, targets, tau_hat=None, tau=None):
        batch_size = outputs.shape[0]
        device = outputs.device
        if 'phi_x' in phi_type:
            I, P = self.get_x_I_P(phi_type, batch_x, batch_size, device)
            # I, P = self.get_onex_I_P(batch_x, batch_size, device)
        elif 'phi_timestamp' in phi_type:
            I, P = self.get_timestamp_I_P(phi_type, batch_x, batch_size, device)
        elif 'phi_period' in phi_type:
            I, P = self.get_period_I_P(phi_type, batch_x, device)
        elif 'phi_autocorr' in phi_type:
            I, P = self.get_autocorr_I_P(phi_type, batch_x, batch_size, device)
        elif 'phi_stats' in phi_type:
            I, P = self.get_stats_I_P(phi_type, batch_x, batch_size, device)
        elif 'phi_random_class' in phi_type:
            I, P = self.get_random_class_I_P(phi_type, 5, batch_x, device)
        else:
            I, P = self.get_ones_I_P(batch_size, device)
        # P = P + P_ones
        error = (outputs - targets).pow(2)
        if tau_hat is not None:
            weight_matrix = tau_hat * I + tau * P
            weighted_error = torch.matmul(weight_matrix, error)
            return {
                'total': torch.mean(weighted_error),
                'ori_mse': torch.mean(error),
                'V_loss': torch.mean(torch.matmul(tau_hat * I, error)),
                'P_loss': torch.mean(torch.matmul(tau * P, error)),
                }
            # V_loss + P_loss ≠ total , 差了一个P和V的交叉项[错误的情况下]
            # total≠ori_mse；在epoch.1-iter.1时,P未归一化时total>ori_mse,模一归一化后两者相等
            # 最小二乘形式引入phi后，使得原来error中仅考虑单个样本本身，变为了考虑整个bs上的所有样本，而各个样本的权重由P(phi)来决定
                # error1 ——> tau_hat * error1 + tau * P_21 * error2 + tau * P31 * error3
            # epoch中每个iter在更新过程中都会更新模型参数包括alpha即tau,是否合适?
                # 根据tau的变化情况(单调降),则tau_hat单调升,即趋向于I主导；（从考虑整理到逐步过渡到考虑单点?且接近原始的强收敛）
                # 若在整个训练过程中固定tau,则仍是考虑全局信息
        else:
            assert tau_hat is None, "loss type error..."

class RankMSELoss(nn.Module):
    def __init__(self, rank_weight=3.0, mse_weight=1.0):
        super(RankMSELoss, self).__init__()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight

    def forward(self, pred, label):
        mse_loss = (pred - label).pow(2).mean()
        # 对预测和真实“排序”
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0) # 做差值
        label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    
        rank_loss = F.relu(- (pred_diff * label_diff)) # 两个差值同号，顺序正确，不惩罚
        
        rank_loss = rank_loss.sum(dim=[1])
        rank_loss = rank_loss.mean()

        combined_loss = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        
        return combined_loss
    

class QniverseModel(Model):
    def __init__(
            self,
            model_config,
            model_type="WFTNet",
            lr=1e-3,
            n_epochs=500,
            early_stop=50,
            smooth_steps=5,
            max_steps_per_epoch=None,
            freeze_model=False,
            model_init_state=None,
            seed=None,
            logdir=None,
            eval_train=True,
            eval_test=False,
            avg_params=True,
            plot_training=True,  # 新增参数：是否绘制训练过程
            plot_show=False,
            **kwargs,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("Qniverse")
        self.logger.info("Qniverse Model...")

        self.model_type = model_type
        self.model = eval(model_type)(SimpleNamespace(**model_config)).to(device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=lr)

        self.model_config = model_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params
        self.loss_type = model_config['loss_type']
        self.phi_type = model_config['phi_type']
        self.plot_training = plot_training  # 新增属性
        self.plot_show = plot_show

        self.criterion = self._select_criterion()

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.global_step = -1

    def _select_criterion(self):
        if self.loss_type == 'MSE_with_weak':
            criterion = WeightedMSELoss()
        elif self.loss_type == 'RankMSELoss':
            criterion = RankMSELoss()
        elif self.loss_type == 'MSELoss':
            criterion = RankMSELoss(rank_weight=0.0, mse_weight=1.0)
        else:
            raise ValueError("can't find your loss function! please defined it!")
        return criterion


    def plot_training_curves(self, train_logs, valid_logs, best_epoch, show=False):
        """绘制训练过程曲线并保存"""
        train_logs = pd.DataFrame(train_logs).reset_index()
        valid_logs = pd.DataFrame(valid_logs).reset_index()
        epochs = range(1, len(train_logs) + 1)

        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # MSE
        ax1.plot(epochs, train_logs['MSE'], 'r-', label='Training')
        ax1.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax1.set_title('mse Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('train mse Loss')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax1_right = ax1.twinx()
        ax1_right.plot(epochs, valid_logs['MSE'], 'g-', label='Validation')
        ax1_right.set_ylabel('val mse Loss')#, color='purple'
        ax1_right.tick_params(axis='y')#, labelcolor='purple'
        ax1_right.legend(loc='upper right')

        # MAE
        ax2.plot(epochs, train_logs['MAE'], 'r-', label='Training')
        ax2.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax2.set_title('mae Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('train mae Loss')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax2_right = ax2.twinx()
        ax2_right.plot(epochs, valid_logs['MAE'], 'g-', label='Validation')
        ax2_right.set_ylabel('val mae Loss')#, color='purple'
        ax2_right.tick_params(axis='y')#, labelcolor='purple'
        ax2_right.legend(loc='upper right')

        # Correlation
        ax3.plot(epochs, train_logs['IC'], 'r-', label='Training')
        ax3.plot(epochs, valid_logs['IC'], 'g-', label='Validation')
        ax3.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax3.set_title('IC')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('IC')
        ax3.legend()
        ax3.grid(True)

        # Rank IC
        ax4.plot(epochs, train_logs['Rank_IC'], 'r-', label='Training')
        ax4.plot(epochs, valid_logs['Rank_IC'], 'g-', label='Validation')
        ax4.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax4.set_title('Rank IC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Rank IC')
        ax4.legend()
        ax4.grid(True)

        if show==True:
            plt.show()
        plt.tight_layout()
        fig_name = 'training_curves_tau_'+str(self.model_config['ind'])+'.png'
        plt.savefig(os.path.join(self.logdir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

        # 同时保存原始数据为CSV文件
        training_data = pd.DataFrame({
            'epoch': epochs,
            'train_ic': train_logs['IC'],
            'train_rank_ic': train_logs['Rank_IC'],
            'train_mse': train_logs['MSE'],
            'train_mae': train_logs['MAE'],
            'valid_ic': valid_logs['IC'],
            'valid_rank_ic': valid_logs['Rank_IC'],
            'valid_mse': valid_logs['MSE'],
            'valid_mae': valid_logs['MAE']
        })
        csv_file_name = 'training_data_tau_'+str(self.model_config['tau_hat_init'])+'.csv'
        training_data.to_csv(os.path.join(self.logdir, csv_file_name), index=False)


    def plot_training_curves_tau(self, train_logs, valid_logs, best_epoch, show=False):
        """绘制训练过程曲线并保存"""
        train_logs = pd.DataFrame(train_logs).reset_index()
        valid_logs = pd.DataFrame(valid_logs).reset_index()
        epochs = range(1, len(train_logs) + 1)

        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # loss
        ax1.tick_params(axis='y')
        ax1.plot(epochs, train_logs['Weight_loss'], 'r-', label='Total')
        ax1.plot(epochs, train_logs['V_loss'], 'g-', label='V_loss')
        # ax1.plot(epochs, train_logs['P_loss'], 'b-', label='P_loss')
        ax1.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('total&V loss')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax1_right = ax1.twinx()
        ax1_right.plot(epochs, train_logs['P_loss'], 'b-', label='P_loss')
        ax1_right.set_ylabel('P loss')#, color='purple'
        ax1_right.tick_params(axis='y')#, labelcolor='purple'
        ax1_right.legend(loc='upper right')

        # MSE
        ax2.tick_params(axis='y')#, labelcolor='orange'
        ax2.plot(epochs, train_logs['MSE'], 'r-', label='Train_mse')
        ax2.plot(epochs, train_logs['MAE'], 'b-', label='Train_mae')
        # ax2.plot(epochs, valid_logs['MSE'], 'g-', label='Val_mse')
        # ax2.plot(epochs, valid_logs['MAE'], 'o-', label='Val_mae')
        ax2.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax2.set_title('MSE/MAE')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('train loss')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax2_right = ax2.twinx()
        ax2_right.plot(epochs, valid_logs['MSE'], 'go-', label='Val_mse')
        ax2_right.plot(epochs, valid_logs['MAE'], 'mo-', label='Val_mae')
        ax2_right.set_ylabel('val loss')#, color='purple'
        ax2_right.tick_params(axis='y')#, labelcolor='purple'
        ax2_right.legend(loc='upper right')

        # Correlation
        ax3.plot(epochs, train_logs['Tau_P'], 'r-', label='Tau_P')
        ax3.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax3.set_title('Tau_P')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Tau_P')
        ax3.legend()
        ax3.grid(True)

        # Rank IC
        ax4.plot(epochs, train_logs['IC'], 'r-', label='Train_IC')
        ax4.plot(epochs, valid_logs['IC'], 'go-', label='Val_IC')
        ax4.plot(epochs, train_logs['Rank_IC'], 'b-', label='Train_RIC')
        ax4.plot(epochs, valid_logs['Rank_IC'], 'mo-', label='Val_RIC')
        ax4.axvline(best_epoch+1, linestyle='--', linewidth=2)
        ax4.set_title('Correlation')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.grid(True)

        if show==True:
            plt.show()
        plt.tight_layout()
        fig_name = 'training_curves_tau_'+str(self.model_config['ind'])+'.png'
        plt.savefig(os.path.join(self.logdir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

        # 同时保存原始数据为CSV文件
        training_data = pd.DataFrame({
            'epoch': epochs,
            'tau_P': train_logs['Tau_P'],
            'total_loss': train_logs['Weight_loss'],
            'v_loss': train_logs['V_loss'],
            'p_loss': train_logs['P_loss'],
            'train_rank_ic': train_logs['Rank_IC'],
            'train_mse': train_logs['MSE'],
            'train_mae': train_logs['MAE'],
            'valid_ic': valid_logs['IC'],
            'valid_rank_ic': valid_logs['Rank_IC'],
            'valid_mse': valid_logs['MSE'],
            'valid_mae': valid_logs['MAE']
        })
        csv_file_name = 'training_data_tau_'+str(self.model_config['tau_hat_init'])+'.csv'
        training_data.to_csv(os.path.join(self.logdir, csv_file_name), index=False)


    def train_epoch(self, data_set):

        self.model.train()

        data_set.train()

        # data_set.setup_data(type='DK_L')
        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, max_steps)

        count = 0
        total_loss = 0
        total_count = 0
        v_loss = 0
        p_loss = 0

        metrics = []

        # for batch in tqdm(data_set, total=max_steps):
        for batch in data_set: # add
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            data, label, index, timestamp = batch["data"], batch["label"], batch["index"], batch["timestamp_feature"]
            data, label, index, timestamp = data.to(device), label.to(device), index.to(device), timestamp.to(device)
            feature = data[:, :, : -1]
            
            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)
            timestamp = torch.nan_to_num(timestamp, nan=0.0)

            # feature [batch_size, seq_len, num_fea]
            pred = self.model(feature).squeeze() # B
            
            pred = torch.nan_to_num(pred, nan=0.0)

            # pred [batch_size, horizon]
            # loss = self.criterion(pred, label)
            if self.loss_type == 'MSE_with_weak':
                tau_hat = torch.sigmoid(self.model.alpha)
                tau = 1 - tau_hat
                loss_dict = self.criterion(self.phi_type, feature, timestamp, pred, label, tau_hat, tau)
                loss = loss_dict['total']
            else:
                loss = self.criterion(pred, label)
            # print(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

            if self.loss_type == 'MSE_with_weak':
                v_loss += loss_dict['V_loss'].item()
                p_loss += loss_dict['P_loss'].item()

            # 计算IC等指标
            X = np.c_[
                pred.detach().cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]
            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)
            metrics.append(evaluate(pred))

        total_loss /= total_count

        if self.loss_type == 'MSE_with_weak':
            metrics = pd.DataFrame(metrics)
            metrics = {
                "MSE": metrics.MSE.mean(),
                "MAE": metrics.MAE.mean(),
                "IC": metrics.IC.mean(),
                "Rank_IC": metrics.Rank_IC.mean(),
                'Weight_loss': total_loss,
                'V_loss': v_loss/total_count,
                'P_loss': p_loss/total_count,
                'Tau_P': torch.sigmoid(self.model.alpha).item(),  # 绘制P矩阵的权重变化
            }
        else:
            metrics = pd.DataFrame(metrics)
            metrics = {
                "MSE": metrics.MSE.mean(),
                "MAE": metrics.MAE.mean(),
                "IC": metrics.IC.mean(),
                "Rank_IC": metrics.Rank_IC.mean(),
            }

        return total_loss, metrics

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        data_set.eval()

        preds = []
        metrics = []
        total_inference_time = 0.0

        for batch in data_set:
        # for batch in tqdm(data_set):
            data, label, index = batch["data"], batch["label"], batch["index"]

            feature = data[:, :, : -1]

            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)
            with torch.no_grad():
                start_time = time.time()
                pred = self.model(feature).squeeze()
                end_time = time.time()
                pred = torch.nan_to_num(pred, nan=0.0)

            total_inference_time += end_time - start_time

            X = np.c_[
                pred.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {
            "InfT": total_inference_time / len(data_set),
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "Rank_IC": metrics.Rank_IC.mean(),
        }


        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index) # 
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
        print("preds")
        print(preds)
        return metrics, preds
    
    def fit(self, dataset, evals_result=dict()):
        
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        # train_set = dataset.prepare("train", data_key=dataset.handler.DK_R)
        # valid_set = dataset.prepare("valid", data_key=dataset.handler.DK_I)
        # test_set = dataset.prepare("test", data_key=dataset.handler.DK_I)

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.global_step = -1

        # 添加记录训练过程的列表
        # train_mses = []    # 记录训练集MSE
        # train_maes = []    # 记录训练集MAE
        # train_ics = []     # 记录训练集IC
        # train_rank_ics = [] # 记录训练集Rank IC
        # valid_ics = []     # 记录验证集IC
        # valid_rank_ics = [] # 记录验证集Rank IC
        # valid_mses = []    # 记录验证集MSE
        # valid_maes = []    # 记录验证集MAE
        train_logs = []  # 每个元素是一个 dict，含 4 个指标
        valid_logs = []

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            _, train_metrics = self.train_epoch(train_set)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                self.logger.info("\tvalid ic increased: %s" % (valid_metrics["IC"]-best_score))
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # train_ics.append(train_metrics["IC"])
            # train_rank_ics.append(train_metrics["Rank_IC"])
            # train_mses.append(train_metrics["MSE"])
            # train_maes.append(train_metrics["MAE"])
            # valid_ics.append(valid_metrics["IC"])
            # valid_rank_ics.append(valid_metrics["Rank_IC"])
            # valid_mses.append(valid_metrics["MSE"])
            # valid_maes.append(valid_metrics["MAE"])
            if self.loss_type == 'MSE_with_weak':
                train_logs.append({
                    'IC': train_metrics['IC'],
                    'Rank_IC': train_metrics['Rank_IC'],
                    'Weight_loss': train_metrics['Weight_loss'],
                    'V_loss': train_metrics['V_loss'],
                    'P_loss': train_metrics['P_loss'],
                    'MSE': train_metrics['MSE'],
                    'MAE': train_metrics['MAE'],
                    'Tau_P': train_metrics['Tau_P']
                })

                valid_logs.append({
                    'IC': valid_metrics['IC'],
                    'Rank_IC': valid_metrics['Rank_IC'],
                    'MSE': valid_metrics['MSE'],
                    'MAE': valid_metrics['MAE']
                })
            else:
                train_logs.append({
                    'IC': train_metrics['IC'],
                    'Rank_IC': train_metrics['Rank_IC'],
                    'MSE': train_metrics['MSE'],
                    'MAE': train_metrics['MAE']
                })

                valid_logs.append({
                    'IC': valid_metrics['IC'],
                    'Rank_IC': valid_metrics['Rank_IC'],
                    'MSE': valid_metrics['MSE'],
                    'MAE': valid_metrics['MAE']
                })

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])

        # 训练结束后绘制训练过程曲线
        if self.plot_training and self.logdir:
            if self.loss_type == 'MSE_with_weak':
                self.plot_training_curves_tau(train_logs, valid_logs, best_epoch, self.plot_show)
            else:
                self.plot_training_curves(train_logs, valid_logs, best_epoch, self.plot_show)

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        metrics['best_epoch'] = best_epoch
        metrics['best_score'] = best_score
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            torch.save(best_params, self.logdir + "/model.bin")

            fig_list = model_performance_graph(preds, show_notebook=False)
            fig_name = [f"{self.model_type}_cumulative_return", f"{self.model_type}_distribution_return", f"{self.model_type}_IC",
                        f"{self.model_type}_monthly_IC", f"{self.model_type}_distribution_IC", f"{self.model_type}_auto_corr"]
            for i, fig in enumerate(fig_list):
                fig: plotly.graph_objs.Figure = fig
                fig.write_html(self.logdir + f'/{fig_name[i]}.html')
            print("Vis Finished!")

            # preds = preds.swaplevel().sort_index()
            preds.to_pickle(self.logdir + "/pred.pkl")

            metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
            self.model_config['tau_hat_init'] = float(self.model_config['tau_hat_init'])
            info = {
                "config": {
                    "model_config": self.model_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

            return preds, metrics

    def predict(self, dataset, segment="test"):

        test_set = dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)
        # print("preds")
        # print(preds)

        return preds

def evaluate_ori(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label, method="pearson")
    Rank_IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC,  "Rank_IC": Rank_IC}

def evaluate(pred):
    # print(pred.head(10))
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    # 计算原始IC（皮尔逊相关系数）
    IC = score.corr(label, method="pearson")
    # 计算斯皮尔曼相关
    Rank_IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC, "Rank_IC": Rank_IC}

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q
