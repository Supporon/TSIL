"""
多模型 Phi 特征输入测试脚本 - Optuna 智能超参数优化版本

本脚本结合了:
1. train_multi_model_phi_optuna.py 的 Optuna 超参数优化框架
2. train_phi_feature_1221.py 的 phi 作为特征输入的核心功能

主要特性：
- 使用 Optuna 进行智能超参数搜索（贝叶斯优化）
- 将 phi 函数作为特征输入到模型中
- 支持多模型和多 phi 类型的测试
- 包含早停机制和可视化分析功能
- 保留命令行参数接口和配置文件加载机制
"""

import argparse
import os
import sys
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import qlib
import pandas as pd
import numpy as np
import ruamel.yaml as YAML
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
import warnings

# Optuna 相关导入
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_parallel_coordinate,
        plot_slice
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Please install with: pip install optuna optuna-dashboard plotly")

warnings.filterwarnings('ignore')
os.chdir(sys.path[0])

from basktesting import calculate_portfolio_metrics, top_k_drop_strategy

# 导入 phi 特征提取器
try:
    from src.phi_feature_extractor import (
        get_all_phi_feature_dims,
        get_phi_feature_dim,
        get_dynamic_phi_feature_dim
    )
    PHI_FEATURE_AVAILABLE = True
except ImportError:
    PHI_FEATURE_AVAILABLE = False
    print("Warning: phi_feature_extractor module not found.")


# ============================================================================
# 模型类型定义
# ============================================================================
BASE_MODEL_TYPES = ['GAT', 'LSTM', 'GRU', 'TCN', 'Transformer']
BASELINE_MODEL_TYPES = ['LSR_IGRU', 'MASTER', 'MERA', 'StockMixer']
MODEL_TYPES = BASE_MODEL_TYPES + BASELINE_MODEL_TYPES


# ============================================================================
# Phi 类型定义（作为特征输入）
# ============================================================================

# 基于时间序列特性的 phi 类型
TIMESERIES_PHI_TYPES = [
    'phi_momentum',       # 动量因子 - 3 维
    'phi_volatility',     # 波动率聚类 - 3 维
    'phi_return_dist',    # 收益率分布形态 - 4 维
    'phi_trend',          # 趋势强度 - 3 维
    'phi_multiscale',     # 多尺度特征 - 8 维
    'phi_temporal',       # 时序模式 - 8 维
]

# 基于金融指标关联性的 phi 类型
FINANCIAL_PHI_TYPES = [
    'phi_feature_corr',   # 特征相关性结构 - 20 维
    'phi_price_volume',   # 价量关系 - 3 维
    'phi_factor',         # 因子暴露 - 10 维
    'phi_technical',      # 技术指标综合 - 3 维
    'phi_vol_return',     # 波动率-收益联合 - 4 维
]

# 基于股票间相关性的 phi 类型
CROSS_SAMPLE_PHI_TYPES = [
    'phi_return_sim',     # 收益率相似性聚类 - 5 维
    'phi_market_beta',    # 市场 Beta - 3 维
    'phi_cross_corr',     # 跨样本相关性 - 1 维
    'phi_lead_lag',       # 领先滞后关系 - 3 维
    'phi_common_factor',  # 共同因子暴露 - 3 维
    'phi_vol_sync',       # 波动率同步性 - 3 维
    'phi_tail_risk',      # 尾部风险共现 - 4 维
    'phi_market_regime',  # 市场状态敏感性 - 3 维
    'phi_sector_cluster', # 板块聚类 - 5 维
]

USE_PHI_TYPES = [
    'phi_ones',
    'phi_timestamp_last_onehot',
    'phi_period', 
    'phi_period_per_sample',
    'phi_momentum',       # 动量因子谓词
    'phi_volatility',     # 波动率聚类谓词
    'phi_return_dist',    # 收益率分布形态谓词
]

# 原始 phi 类型
ORIGINAL_ONES_PHI_TYPES = ['phi_ones']
ORIGINAL_TIMESTAMP_PHI_TYPES = ['phi_timestamp_last', 'phi_timestamp_last_onehot']
ORIGINAL_PERIOD_PHI_TYPES = ['phi_period', 'phi_period_per_sample']
ORIGINAL_STATS_PHI_TYPES = ['phi_stats_mean', 'phi_stats_var', 'phi_stats_skew', 'phi_stats_kurt']

ORIGINAL_PHI_TYPES = (
    ORIGINAL_PERIOD_PHI_TYPES +
    ORIGINAL_TIMESTAMP_PHI_TYPES +
    ORIGINAL_STATS_PHI_TYPES +
    ORIGINAL_ONES_PHI_TYPES
)

# 所有 phi 类型
ALL_PHI_TYPES = USE_PHI_TYPES # TIMESERIES_PHI_TYPES + ORIGINAL_PHI_TYPES


# ============================================================================
# 超参数搜索空间定义
# ============================================================================
DEFAULT_SEARCH_SPACE = {
    'lr': {'type': 'categorical', 'choices': [0.001, 0.0001, 0.00001]},
    'layers': {'type': 'categorical', 'choices': [1, 2, 3]},
    'd_model': {'type': 'categorical', 'choices': [64, 128, 256]},
    'bs': {'type': 'categorical', 'choices': [1024]}
}


def get_available_phi_types(phi_category: str = 'all') -> list:
    """获取可用的 phi 类型列表"""
    category_map = {
        'all': ALL_PHI_TYPES,
        'timeseries': TIMESERIES_PHI_TYPES,
        'financial': FINANCIAL_PHI_TYPES,
        'cross_sample': CROSS_SAMPLE_PHI_TYPES,
        'original': ORIGINAL_PHI_TYPES,
        'original_ones': ORIGINAL_ONES_PHI_TYPES,
        'original_timestamp': ORIGINAL_TIMESTAMP_PHI_TYPES,
        'original_period': ORIGINAL_PERIOD_PHI_TYPES,
        'original_stats': ORIGINAL_STATS_PHI_TYPES,
    }
    return category_map.get(phi_category, ALL_PHI_TYPES)


def get_model_optimal_params(model_type: str, market: str = 'csi300') -> dict:
    """根据模型类型和市场返回默认最优参数配置（作为搜索起点参考）"""
    csi300_params = {
        'GAT': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64},
        'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 128},
        'TCN': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 128},
        'Transformer': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 2, 'd_model': 256},
        'LSR_IGRU': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 1, 'd_model': 128},
        'MASTER': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'MERA': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 3, 'd_model': 128},
        'StockMixer': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 1, 'd_model': 64},
    }
    
    csi800_params = {
        'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 128},
        'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64},
        'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 512, 'layers': 1, 'd_model': 128},
        'LSR_IGRU': {'seq_len': 20, 'lr': 0.0001, 'bs': 512, 'layers': 1, 'd_model': 64},
        'MASTER': {'seq_len': 20, 'lr': 0.001, 'bs': 512, 'layers': 1, 'd_model': 256},
        'MERA': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 1, 'd_model': 128},
        'StockMixer': {'seq_len': 20, 'lr': 0.001, 'bs': 256, 'layers': 1, 'd_model': 128},
    }
    
    params_map = {'csi300': csi300_params, 'csi800': csi800_params}
    market_params = params_map.get(market.lower(), csi300_params)
    
    if model_type not in market_params:
        return {'seq_len': 20, 'lr': 0.0001, 'bs': 512, 'layers': 2, 'd_model': 64}
    
    return market_params[model_type]


class PhiFeatureOptunaOptimizer:
    """Phi 特征 Optuna 超参数优化器"""
    
    def __init__(
        self,
        base_config: dict,
        model_type: str,
        phi_type: str,
        use_phi_as_feature: bool,
        market: str,
        output_dir: str,
        search_space: dict = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        seed: int = 2025,
        optimization_metric: str = 'IC',
        secondary_metrics: List[str] = None,
        early_stopping_patience: int = 10,
        sampler_type: str = 'tpe',
        pruner_type: str = 'median'
    ):
        """
        初始化优化器
        
        Args:
            base_config: 基础配置字典
            model_type: 模型类型
            phi_type: phi 类型
            use_phi_as_feature: 是否将 phi 作为特征输入
            market: 市场类型
            output_dir: 输出目录
            search_space: 超参数搜索空间
            n_trials: 优化试验次数
            n_jobs: 并行作业数
            seed: 随机种子
            optimization_metric: 主优化指标
            secondary_metrics: 次要监控指标
            early_stopping_patience: 早停耐心值
            sampler_type: 采样器类型 ('tpe', 'cmaes')
            pruner_type: 剪枝器类型 ('median', 'halving')
        """
        self.base_config = base_config
        self.model_type = model_type
        self.phi_type = phi_type
        self.use_phi_as_feature = use_phi_as_feature
        self.market = market
        self.output_dir = output_dir
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.optimization_metric = optimization_metric
        self.secondary_metrics = secondary_metrics or ['RankIC', 'ICIR', 'AR_with_cost']
        self.early_stopping_patience = early_stopping_patience
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type
        
        # 实验记录
        self.trial_history = []
        self.best_result = None
        self.best_value = float('-inf')
        self.no_improvement_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """创建采样器"""
        if self.sampler_type == 'tpe':
            return TPESampler(seed=self.seed, multivariate=True)
        elif self.sampler_type == 'cmaes':
            return CmaEsSampler(seed=self.seed)
        else:
            return TPESampler(seed=self.seed)
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """创建剪枝器"""
        if self.pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        elif self.pruner_type == 'halving':
            return SuccessiveHalvingPruner()
        else:
            return MedianPruner()
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """从试验中建议超参数"""
        hyperparams = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            elif param_type == 'int':
                hyperparams[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'float':
                hyperparams[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'loguniform':
                hyperparams[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=True
                )
        
        return hyperparams
    
    def _apply_hyperparameters(self, config: dict, hyperparams: dict) -> dict:
        """应用超参数到配置"""
        config = copy.deepcopy(config)
        
        # 应用超参数
        if 'lr' in hyperparams:
            config["task"]["model"]["kwargs"]["lr"] = hyperparams['lr']
        if 'layers' in hyperparams:
            config["model_config"]["e_layers"] = hyperparams['layers']
        if 'd_model' in hyperparams:
            config["model_config"]["d_model"] = hyperparams['d_model']
            config["model_config"]["d_ff"] = hyperparams['d_model']
        if 'bs' in hyperparams:
            config["task"]["dataset"]["kwargs"]["batch_size"] = hyperparams['bs']
        
        # 设置模型类型
        config["task"]["model"]["kwargs"]["model_type"] = self.model_type
        
        # 设置 phi 特征相关参数
        config["task"]["model"]["kwargs"]["use_phi_as_feature"] = self.use_phi_as_feature
        config["task"]["model"]["kwargs"]["phi_type"] = self.phi_type
        
        # 设置 ts_onehot 参数
        if 'onehot' in self.phi_type:
            config["task"]["dataset"]["kwargs"]["ts_onehot"] = True
        else:
            config["task"]["dataset"]["kwargs"]["ts_onehot"] = False
        
        # 设置模型特有参数
        if self.model_type == 'MASTER':
            config["model_config"]["d_feat"] = 158
            config["model_config"]["t_nhead"] = 4
            config["model_config"]["s_nhead"] = 2
            config["model_config"]["T_dropout_rate"] = 0.5
            config["model_config"]["S_dropout_rate"] = 0.5
            config["model_config"]["gate_input_start_index"] = 158
            config["model_config"]["gate_input_end_index"] = 221
            config["model_config"]["beta"] = 5
            config["model_config"]["use_gate"] = False
        elif self.model_type == 'MERA':
            config["model_config"]["num_expert"] = 8
            config["model_config"]["top_k"] = 4
            config["model_config"]["gate_dim"] = 16
            config["model_config"]["dropout"] = 0.3
        elif self.model_type == 'StockMixer':
            config["model_config"]["market_num"] = 20
            config["model_config"]["scale_factor"] = 3
            config["model_config"]["activation"] = 'GELU'
            config["model_config"]["rank_loss_alpha"] = 0.1
            config["model_config"]["d_ff"] = 16
        elif self.model_type == 'LSR_IGRU':
            config["model_config"]["base_model"] = 'GRU'
        
        config["market"] = self.market
        
        return config
    
    def _get_experiment_metrics(self, file_name: str) -> dict:
        """从实验目录获取指标"""
        folder = Path(file_name)
        file_dict = {}
        
        for file_path in folder.glob('*'):
            if file_path.is_file():
                try:
                    v = file_path.read_text(encoding='utf-8')
                    file_dict[file_path.name] = float(v.split(' ')[1])
                except Exception:
                    pass
        
        return file_dict
    
    def _run_experiment(self, config: dict, trial_number: int) -> dict:
        """运行单个实验"""
        # 初始化 qlib
        qlib.init(
            provider_uri=config["qlib_init"]["provider_uri"],
            region=config["qlib_init"]["region"],
        )
        
        # 设置实验路径
        exp_path = os.path.join(self.output_dir, "experiments")
        os.makedirs(exp_path, exist_ok=True)
        R.set_uri(exp_path)
        
        result = {}
        phi_str = self.phi_type if self.use_phi_as_feature else 'no_phi'
        
        with R.start(
            experiment_name=f"{self.model_type}_{phi_str}_PhiFeature_Optuna",
            recorder_name=f"trial_{trial_number}"
        ):
            logdir = config["task"]["model"]["kwargs"]["logdir"]
            os.makedirs(logdir, exist_ok=True)
            
            dataset = init_instance_by_config(config["task"]["dataset"])
            model = init_instance_by_config(config["task"]["model"])
            
            preds, metrics = model.fit(dataset)
            
            # 计算排名指标
            per_df = preds.reset_index()
            per_df = per_df.sort_values(['datetime', 'instrument'])
            per_df['group_id'] = per_df.groupby(['datetime', 'instrument']).cumcount()
            
            strategy_config = config['strategy_config']
            sub_df = per_df[per_df['group_id'] == 0]
            sub_df = sub_df.set_index(['datetime', 'instrument'])
            sub_df = sub_df[['score', 'label']]
            
            # Ranking Metrics
            ic = sub_df.groupby(level=0).apply(
                lambda group: group['score'].corr(group['label'])
            ).to_numpy()
            icir = ic.mean() / ic.std() if ic.std() > 0 else 0
            ic_mean = ic.mean()
            
            rankic = sub_df.groupby(level=0).apply(
                lambda group: group['score'].corr(group['label'], method='spearman')
            ).to_numpy()
            rankicir = rankic.mean() / rankic.std() if rankic.std() > 0 else 0
            rankic_mean = rankic.mean()
            
            # 回测
            r_df = top_k_drop_strategy(
                sub_df,
                top_k=strategy_config['topk'],
                n=strategy_config['hold_thre'],
                commission_rate=strategy_config['commission_rate']
            )
            portfolio_metrics, df = calculate_portfolio_metrics(r_df)
            
            # 获取分析指标
            port_analysis_config = config["port_analysis_config"]
            recorder = R.get_recorder()
            
            try:
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()
            except Exception as e:
                print(f"Warning: SignalRecord failed: {e}")
            
            try:
                sar = SigAnaRecord(recorder)
                sar.generate(label=preds[['label']])
            except Exception as e:
                print(f"Warning: SigAnaRecord failed: {e}")
            
            try:
                par = PortAnaRecord(recorder, port_analysis_config, "day")
                par.generate()
            except Exception as e:
                print(f"Warning: PortAnaRecord failed: {e}")
            
            # 获取分析结果
            analysis_dict = self._get_experiment_metrics(
                recorder.uri + '/' + recorder.experiment_id + '/' + recorder.id + '/metrics/'
            )
            
            # 获取 phi 特征维度
            phi_feature_dim = 0
            if self.use_phi_as_feature and PHI_FEATURE_AVAILABLE:
                phi_feature_dim = get_phi_feature_dim(self.phi_type)
            
            result = {
                'IC': ic_mean,
                'ICIR': icir,
                'RankIC': rankic_mean,
                'RankICIR': rankicir,
                'MSE': metrics.get('MSE', 0),
                'MAE': metrics.get('MAE', 0),
                'best_epoch': metrics.get('best_epoch', -1),
                'best_score': metrics.get('best_score', 0),
                'phi_feature_dim': phi_feature_dim,
                'AR_without_cost': analysis_dict.get('1day.excess_return_without_cost.annualized_return', 0),
                'IR_without_cost': analysis_dict.get('1day.excess_return_without_cost.information_ratio', 0),
                'MDD_without_cost': analysis_dict.get('1day.excess_return_without_cost.max_drawdown', 0),
                'AR_with_cost': analysis_dict.get('1day.excess_return_with_cost.annualized_return', 0),
                'IR_with_cost': analysis_dict.get('1day.excess_return_with_cost.information_ratio', 0),
                'MDD_with_cost': analysis_dict.get('1day.excess_return_with_cost.max_drawdown', 0)
            }
            
            # 保存回测结果
            df.to_csv(os.path.join(logdir, f'backtest_result_{phi_str}_trial_{trial_number}.csv'))
        
        return result
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna 目标函数"""
        hyperparams = self._suggest_hyperparameters(trial)
        config = self._apply_hyperparameters(self.base_config, hyperparams)
        config["task"]["model"]["kwargs"]["logdir"] = self.output_dir
        config["model_config"]["ind"] = trial.number
        
        phi_str = self.phi_type if self.use_phi_as_feature else 'no_phi'
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"Model: {self.model_type}, Phi: {phi_str}, Use Phi Feature: {self.use_phi_as_feature}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*60}")
        
        try:
            result = self._run_experiment(config, trial.number)
            
            # 记录次要指标
            for metric in self.secondary_metrics:
                if metric in result:
                    trial.set_user_attr(metric, result[metric])
            
            trial.set_user_attr('all_metrics', result)
            
            # 保存到历史记录
            trial_record = {
                'trial_number': trial.number,
                'model_type': self.model_type,
                'phi_type': self.phi_type,
                'use_phi_as_feature': self.use_phi_as_feature,
                **hyperparams,
                **result
            }
            self.trial_history.append(trial_record)
            
            objective_value = result.get(self.optimization_metric, float('-inf'))
            
            # 检查是否为最佳结果
            if objective_value > self.best_value:
                self.best_value = objective_value
                self.best_result = trial_record
                self.no_improvement_count = 0
                print(f"*** New Best! {self.optimization_metric}: {objective_value:.4f} ***")
            else:
                self.no_improvement_count += 1
            
            print(f"Result: {self.optimization_metric}={objective_value:.4f}")
            for metric in self.secondary_metrics:
                if metric in result:
                    print(f"  {metric}: {result[metric]:.4f}")
            
            self._save_trial_history()
            
            # 早停检查
            if self.no_improvement_count >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.early_stopping_patience} trials without improvement")
                trial.study.stop()
            
            return objective_value
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return float('-inf')
    
    def _save_trial_history(self):
        """保存试验历史"""
        history_df = pd.DataFrame(self.trial_history)
        history_df.to_csv(os.path.join(self.output_dir, 'trial_history.csv'), index=False)
    
    def _early_stopping_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """早停回调"""
        if self.no_improvement_count >= self.early_stopping_patience:
            study.stop()
    
    def optimize(self) -> Tuple[dict, optuna.Study]:
        """执行优化"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Please install with: pip install optuna")
        
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        phi_str = self.phi_type if self.use_phi_as_feature else 'no_phi'
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.model_type}_{phi_str}_phi_feature_optimization"
        )
        
        callbacks = [self._early_stopping_callback]
        
        print(f"\n{'#'*60}")
        print(f"Starting Optuna Optimization (Phi as Feature)")
        print(f"Model: {self.model_type}")
        print(f"Phi Type: {self.phi_type}")
        print(f"Use Phi as Feature: {self.use_phi_as_feature}")
        print(f"Market: {self.market}")
        print(f"Optimization Metric: {self.optimization_metric} (maximize)")
        print(f"Number of Trials: {self.n_trials}")
        print(f"Early Stopping Patience: {self.early_stopping_patience}")
        print(f"{'#'*60}\n")
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        self._generate_optimization_report(study)
        self._generate_visualizations(study)
        
        return self.best_result, study
    
    def _generate_optimization_report(self, study: optuna.Study):
        """生成优化报告"""
        phi_str = self.phi_type if self.use_phi_as_feature else 'no_phi'
        report = []
        report.append("=" * 80)
        report.append("Optuna Hyperparameter Optimization Report (Phi as Feature)")
        report.append("=" * 80)
        report.append("")
        report.append(f"Model Type: {self.model_type}")
        report.append(f"Phi Type: {self.phi_type}")
        report.append(f"Use Phi as Feature: {self.use_phi_as_feature}")
        report.append(f"Market: {self.market}")
        report.append(f"Optimization Metric: {self.optimization_metric}")
        report.append("")
        report.append("-" * 60)
        report.append("Search Space:")
        report.append("-" * 60)
        for param, config in self.search_space.items():
            report.append(f"  {param}: {config}")
        report.append("")
        report.append("-" * 60)
        report.append("Optimization Summary:")
        report.append("-" * 60)
        report.append(f"Total Trials: {len(study.trials)}")
        report.append(f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        report.append(f"Best {self.optimization_metric}: {study.best_value:.4f}")
        report.append("")
        report.append("-" * 60)
        report.append("Best Hyperparameters:")
        report.append("-" * 60)
        for param, value in study.best_params.items():
            report.append(f"  {param}: {value}")
        report.append("")
        report.append("-" * 60)
        report.append("Best Trial All Metrics:")
        report.append("-" * 60)
        if self.best_result:
            for key, value in self.best_result.items():
                if key not in ['trial_number']:
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.4f}")
                    else:
                        report.append(f"  {key}: {value}")
        report.append("")
        report.append("-" * 60)
        report.append("Top 5 Trials:")
        report.append("-" * 60)
        
        sorted_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value if t.value is not None else float('-inf'),
            reverse=True
        )[:5]
        
        for i, trial in enumerate(sorted_trials, 1):
            report.append(f"\n  #{i} Trial {trial.number}:")
            report.append(f"    {self.optimization_metric}: {trial.value:.4f}")
            report.append(f"    Params: {trial.params}")
            if 'all_metrics' in trial.user_attrs:
                metrics = trial.user_attrs['all_metrics']
                for metric in self.secondary_metrics:
                    if metric in metrics:
                        report.append(f"    {metric}: {metrics[metric]:.4f}")
        
        report.append("\n" + "=" * 80)
        
        report_path = os.path.join(self.output_dir, 'optimization_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nOptimization report saved to: {report_path}")
    
    def _generate_visualizations(self, study: optuna.Study):
        """生成可视化图表"""
        try:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            try:
                fig = plot_optimization_history(study)
                fig.write_html(os.path.join(vis_dir, 'optimization_history.html'))
                print(f"Saved: optimization_history.html")
            except Exception as e:
                print(f"Warning: Could not generate optimization history plot: {e}")
            
            try:
                fig = plot_param_importances(study)
                fig.write_html(os.path.join(vis_dir, 'param_importances.html'))
                print(f"Saved: param_importances.html")
            except Exception as e:
                print(f"Warning: Could not generate param importance plot: {e}")
            
            try:
                param_names = list(study.best_params.keys())
                if len(param_names) >= 2:
                    fig = plot_contour(study, params=param_names[:2])
                    fig.write_html(os.path.join(vis_dir, 'contour.html'))
                    print(f"Saved: contour.html")
            except Exception as e:
                print(f"Warning: Could not generate contour plot: {e}")
            
            try:
                fig = plot_parallel_coordinate(study)
                fig.write_html(os.path.join(vis_dir, 'parallel_coordinate.html'))
                print(f"Saved: parallel_coordinate.html")
            except Exception as e:
                print(f"Warning: Could not generate parallel coordinate plot: {e}")
            
            try:
                fig = plot_slice(study)
                fig.write_html(os.path.join(vis_dir, 'slice.html'))
                print(f"Saved: slice.html")
            except Exception as e:
                print(f"Warning: Could not generate slice plot: {e}")
            
            print(f"\nVisualizations saved to: {vis_dir}")
            
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")


class MultiModelPhiFeatureOptimizer:
    """多模型 Phi 特征 Optuna 优化器"""
    
    def __init__(
        self,
        base_config: dict,
        models: List[str],
        phi_types: List[str],
        use_phi_as_feature: bool,
        market: str,
        output_dir: str,
        search_space: dict = None,
        n_trials: int = 50,
        seed: int = 2025,
        optimization_metric: str = 'IC',
        early_stopping_patience: int = 10,
        compare_with_baseline: bool = False
    ):
        self.base_config = base_config
        self.models = models
        self.phi_types = phi_types
        self.use_phi_as_feature = use_phi_as_feature
        self.market = market
        self.output_dir = output_dir
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.seed = seed
        self.optimization_metric = optimization_metric
        self.early_stopping_patience = early_stopping_patience
        self.compare_with_baseline = compare_with_baseline
        
        self.all_results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize_all(self) -> pd.DataFrame:
        """优化所有模型和 phi 类型组合"""
        total_combinations = len(self.models) * len(self.phi_types)
        if self.compare_with_baseline:
            total_combinations += len(self.models)  # 每个模型加一个 baseline
        
        current = 0
        
        print(f"\n{'#'*80}")
        print(f"Multi-Model Phi Feature Optuna Optimization")
        print(f"{'#'*80}")
        print(f"Models: {self.models}")
        print(f"Phi Types: {self.phi_types}")
        print(f"Use Phi as Feature: {self.use_phi_as_feature}")
        print(f"Compare with Baseline: {self.compare_with_baseline}")
        print(f"Market: {self.market}")
        print(f"Total Combinations: {total_combinations}")
        print(f"Trials per Combination: {self.n_trials}")
        print(f"{'#'*80}\n")
        
        for model_type in self.models:
            # 如果需要对比基线，先运行不使用 phi 特征的实验
            if self.compare_with_baseline:
                current += 1
                print(f"\n{'='*60}")
                print(f"Optimization {current}/{total_combinations}")
                print(f"Model: {model_type}, Baseline (no phi feature)")
                print(f"{'='*60}")
                
                model_output_dir = os.path.join(
                    self.output_dir,
                    f"{model_type.lower()}_baseline"
                )
                
                config = copy.deepcopy(self.base_config)
                config["task"]["model"]["kwargs"]["logdir"] = model_output_dir
                config["logdir"] = model_output_dir
                
                optimizer = PhiFeatureOptunaOptimizer(
                    base_config=config,
                    model_type=model_type,
                    phi_type="none",
                    use_phi_as_feature=False,
                    market=self.market,
                    output_dir=model_output_dir,
                    search_space=self.search_space,
                    n_trials=self.n_trials,
                    seed=self.seed,
                    optimization_metric=self.optimization_metric,
                    early_stopping_patience=self.early_stopping_patience
                )
                
                try:
                    best_result, study = optimizer.optimize()
                    if best_result:
                        result_record = {
                            'model_type': model_type,
                            'phi_type': 'none',
                            'use_phi_as_feature': False,
                            'best_lr': best_result.get('lr'),
                            'best_layers': best_result.get('layers'),
                            'best_d_model': best_result.get('d_model'),
                            'best_bs': best_result.get('bs'),
                            **{f'best_{k}': v for k, v in best_result.items()
                               if k not in ['lr', 'layers', 'd_model', 'bs', 'trial_number', 
                                           'model_type', 'phi_type', 'use_phi_as_feature']}
                        }
                        self.all_results.append(result_record)
                except Exception as e:
                    print(f"Error optimizing baseline {model_type}: {e}")
                    self.all_results.append({
                        'model_type': model_type,
                        'phi_type': 'none',
                        'use_phi_as_feature': False,
                        'error': str(e)
                    })
            
            # 使用 phi 作为特征的实验
            if self.use_phi_as_feature:
                for phi_type in self.phi_types:
                    current += 1
                    
                    print(f"\n{'='*60}")
                    print(f"Optimization {current}/{total_combinations}")
                    print(f"Model: {model_type}, Phi: {phi_type}")
                    print(f"{'='*60}")
                    
                    model_output_dir = os.path.join(
                        self.output_dir,
                        f"{model_type.lower()}_{phi_type}"
                    )
                    
                    config = copy.deepcopy(self.base_config)
                    config["task"]["model"]["kwargs"]["logdir"] = model_output_dir
                    config["logdir"] = model_output_dir
                    
                    optimizer = PhiFeatureOptunaOptimizer(
                        base_config=config,
                        model_type=model_type,
                        phi_type=phi_type,
                        use_phi_as_feature=True,
                        market=self.market,
                        output_dir=model_output_dir,
                        search_space=self.search_space,
                        n_trials=self.n_trials,
                        seed=self.seed,
                        optimization_metric=self.optimization_metric,
                        early_stopping_patience=self.early_stopping_patience
                    )
                    
                    try:
                        best_result, study = optimizer.optimize()
                        if best_result:
                            result_record = {
                                'model_type': model_type,
                                'phi_type': phi_type,
                                'use_phi_as_feature': True,
                                'best_lr': best_result.get('lr'),
                                'best_layers': best_result.get('layers'),
                                'best_d_model': best_result.get('d_model'),
                                'best_bs': best_result.get('bs'),
                                **{f'best_{k}': v for k, v in best_result.items()
                                   if k not in ['lr', 'layers', 'd_model', 'bs', 'trial_number',
                                               'model_type', 'phi_type', 'use_phi_as_feature']}
                            }
                            self.all_results.append(result_record)
                    except Exception as e:
                        print(f"Error optimizing {model_type}/{phi_type}: {e}")
                        self.all_results.append({
                            'model_type': model_type,
                            'phi_type': phi_type,
                            'use_phi_as_feature': True,
                            'error': str(e)
                        })
        
        # 保存所有结果
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(os.path.join(self.output_dir, 'all_best_results.csv'), index=False)
        
        self._generate_summary_report(results_df)
        
        return results_df
    
    def _generate_summary_report(self, results_df: pd.DataFrame):
        """生成汇总报告"""
        report = []
        report.append("=" * 80)
        report.append("Multi-Model Phi Feature Optuna Optimization Summary")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Combinations: {len(results_df)}")
        
        # 检查 DataFrame 是否为空或缺少必要的列
        if len(results_df) == 0 or 'model_type' not in results_df.columns:
            report.append("")
            report.append("Warning: No valid results to summarize.")
            report.append("All experiments may have failed or no experiments were run.")
            report.append("\n" + "=" * 80)
            
            report_path = os.path.join(self.output_dir, 'summary_report.txt')
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            print(f"\nSummary report saved to: {report_path}")
            return
        
        if 'error' in results_df.columns:
            report.append(f"Successful: {len(results_df[~results_df['error'].notna()])}")
        report.append("")
        
        report.append("-" * 60)
        report.append("Best Results by Model")
        report.append("-" * 60)
        
        metric_col = f'best_{self.optimization_metric}'
        if metric_col in results_df.columns:
            for model in self.models:
                model_results = results_df[results_df['model_type'] == model]
                if len(model_results) > 0 and metric_col in model_results.columns:
                    valid_results = model_results[model_results[metric_col].notna()]
                    if len(valid_results) > 0:
                        best_idx = valid_results[metric_col].idxmax()
                        best = valid_results.loc[best_idx]
                        report.append(f"\n{model}:")
                        report.append(f"  Best {self.optimization_metric}: {best.get(metric_col, 'N/A'):.4f}")
                        report.append(f"  Phi Type: {best.get('phi_type')}")
                        report.append(f"  Use Phi as Feature: {best.get('use_phi_as_feature')}")
                        report.append(f"  Hyperparameters:")
                        report.append(f"    lr: {best.get('best_lr')}")
                        report.append(f"    layers: {best.get('best_layers')}")
                        report.append(f"    d_model: {best.get('best_d_model')}")
                        report.append(f"    bs: {best.get('best_bs')}")
        
        report.append("\n" + "-" * 60)
        report.append("Comparison: With vs Without Phi Features")
        report.append("-" * 60)
        
        for model in self.models:
            model_results = results_df[results_df['model_type'] == model]
            with_phi = model_results[model_results['use_phi_as_feature'] == True]
            without_phi = model_results[model_results['use_phi_as_feature'] == False]
            
            report.append(f"\n{model}:")
            if len(without_phi) > 0 and metric_col in without_phi.columns:
                baseline_val = without_phi[metric_col].values[0] if len(without_phi) > 0 else 'N/A'
                if isinstance(baseline_val, float):
                    report.append(f"  Without Phi Feature - {self.optimization_metric}: {baseline_val:.4f}")
            
            if len(with_phi) > 0 and metric_col in with_phi.columns:
                valid_with_phi = with_phi[with_phi[metric_col].notna()]
                if len(valid_with_phi) > 0:
                    best_idx = valid_with_phi[metric_col].idxmax()
                    best = valid_with_phi.loc[best_idx]
                    report.append(f"  Best With Phi Feature:")
                    report.append(f"    Phi Type: {best['phi_type']}")
                    report.append(f"    {self.optimization_metric}: {best[metric_col]:.4f}")
        
        report.append("\n" + "=" * 80)
        
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nSummary report saved to: {report_path}")


def print_phi_feature_info():
    """打印所有可用的 phi 类型及其特征维度信息"""
    print("\n" + "=" * 80)
    print("Available Phi Types (as Input Features)")
    print("=" * 80)
    
    if PHI_FEATURE_AVAILABLE:
        dims = get_all_phi_feature_dims()
    else:
        dims = {}
    
    print("\n[Timeseries Phi Types] (基于时间序列特性):")
    for phi in TIMESERIES_PHI_TYPES:
        dim = dims.get(phi, 'N/A')
        print(f"  - {phi:25s}: {dim} dimensions")
    
    print("\n[Financial Phi Types] (基于金融指标关联性):")
    for phi in FINANCIAL_PHI_TYPES:
        dim = dims.get(phi, 'N/A')
        print(f"  - {phi:25s}: {dim} dimensions")
    
    print("\n[Cross-Sample Phi Types] (基于股票间相关性):")
    for phi in CROSS_SAMPLE_PHI_TYPES:
        dim = dims.get(phi, 'N/A')
        print(f"  - {phi:25s}: {dim} dimensions")
    
    print("\n[Original Phi Types] (原始 phi 类型):")
    for phi in ORIGINAL_PHI_TYPES:
        dim = dims.get(phi, 'N/A')
        print(f"  - {phi:30s}: {dim} dimensions")
    
    total_types = len(ALL_PHI_TYPES)
    print(f"\nTotal: {total_types} phi types available")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Multi-Model Phi Feature Testing with Optuna Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phi Feature Categories:
  - timeseries  : 基于时间序列特性 (phi_momentum, phi_volatility, phi_trend, ...)
  - financial   : 基于金融指标关联性 (phi_feature_corr, phi_price_volume, ...)
  - cross_sample: 基于股票间相关性 (phi_market_beta, phi_cross_corr, ...)
  - original    : 原始 phi 类型
  - all         : 所有 phi 类型

Optuna Optimization Features:
  - Bayesian optimization with TPE sampler
  - Early stopping when no improvement
  - Visualization of optimization history
  - Parameter importance analysis

Examples:
  # 单模型优化，使用 phi 作为特征
  python train_phi_feature_optuna.py --models LSTM --phi_types phi_momentum --use_phi_as_feature --n_trials 30
  
  # 多模型优化，对比有无 phi 特征
  python train_phi_feature_optuna.py --models LSTM GRU --phi_category timeseries --use_phi_as_feature --compare_with_baseline
  
  # 查看所有可用的 phi 类型及其维度
  python train_phi_feature_optuna.py --list_phi_types
"""
    )
    
    # 基本参数
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--config_file", type=str, default="configs_phi/config_phi_feature_use.yaml",
                        help="Base config file path")
    
    # 模型选择
    parser.add_argument("--models", type=str, nargs='+', default=['LSTM'],
                        help=f"Model types to test. Available: {MODEL_TYPES}")
    
    # Phi 特征控制
    parser.add_argument("--use_phi_as_feature", action="store_true",
                        help="Use phi as input feature")
    parser.add_argument("--phi_types", type=str, nargs='+', default=None,
                        help="Specific phi types to test")
    parser.add_argument("--phi_category", type=str, default="all",
                        choices=['all', 'timeseries', 'financial', 'cross_sample',
                                 'original', 'original_ones', 'original_timestamp',
                                 'original_period', 'original_stats'],
                        help="Phi type category to test")
    
    # 对比实验
    parser.add_argument("--compare_with_baseline", action="store_true",
                        help="Also run baseline experiments without phi features")
    
    # 市场选择
    parser.add_argument("--market", type=str, default="csi300",
                        choices=['csi300', 'csi500', 'csi800'],
                        help="Market type")
    
    # Optuna 参数
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials per model/phi combination")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--optimization_metric", type=str, default="IC",
                        help="Primary metric to optimize")
    parser.add_argument("--sampler", type=str, default="tpe",
                        choices=['tpe', 'cmaes'])
    
    # 超参数搜索空间
    parser.add_argument("--lr_choices", type=float, nargs='+',
                        default=[0.001, 0.0001, 0.00001])
    parser.add_argument("--layers_choices", type=int, nargs='+',
                        default=[1, 2, 3])
    parser.add_argument("--d_model_choices", type=int, nargs='+',
                        default=[64, 128, 256])
    parser.add_argument("--bs_choices", type=int, nargs='+',
                        default=[1024])
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="output_model_phifeature_optuna_csi800/output_phi_feature_optuna")
    parser.add_argument("--seq_len", type=int, default=20)
    
    # 其他选项
    parser.add_argument("--list_phi_types", action="store_true",
                        help="List all available phi types with their dimensions and exit")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show experiment configuration without running")
    
    args = parser.parse_args()
    
    # 如果要求列出 phi 类型，打印并退出
    if args.list_phi_types:
        print_phi_feature_info()
        return
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is not installed.")
        print("Please install with: pip install optuna optuna-dashboard plotly")
        return
    
    # 确定要测试的 phi 类型
    if args.phi_types is not None:
        phi_types = args.phi_types
    else:
        phi_types = get_available_phi_types(args.phi_category)
    
    # 构建搜索空间
    search_space = {
        'lr': {'type': 'categorical', 'choices': args.lr_choices},
        'layers': {'type': 'categorical', 'choices': args.layers_choices},
        'd_model': {'type': 'categorical', 'choices': args.d_model_choices},
        'bs': {'type': 'categorical', 'choices': args.bs_choices},
    }
    
    # 加载配置
    with open(args.config_file) as f:
        yaml = YAML.YAML(typ='safe', pure=True)
        base_config = yaml.load(f)
    
    # 更新配置
    base_config["seq_len"] = args.seq_len
    base_config["model_config"]["seq_len"] = args.seq_len
    base_config["task"]["dataset"]["kwargs"]["seq_len"] = args.seq_len
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{args.output_dir}_{args.market}_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    # 计算实验数量
    n_experiments = len(args.models) * len(phi_types)
    if args.compare_with_baseline:
        n_experiments += len(args.models)
    
    # 保存配置
    exp_info = {
        'seed': args.seed,
        'models': args.models,
        'phi_types': phi_types,
        'phi_category': args.phi_category,
        'use_phi_as_feature': args.use_phi_as_feature,
        'compare_with_baseline': args.compare_with_baseline,
        'market': args.market,
        'n_trials': args.n_trials,
        'early_stopping_patience': args.early_stop,
        'optimization_metric': args.optimization_metric,
        'search_space': search_space,
        'timestamp': timestamp,
        'config_file': args.config_file,
        'total_experiments': n_experiments
    }
    with open(os.path.join(output_base, 'experiment_info.json'), 'w') as f:
        json.dump(exp_info, f, indent=2)
    
    print("=" * 80)
    print("Phi Feature Optuna Hyperparameter Optimization")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Phi Types ({len(phi_types)}): {phi_types}")
    print(f"Use Phi as Feature: {args.use_phi_as_feature}")
    print(f"Compare with Baseline: {args.compare_with_baseline}")
    print(f"Market: {args.market}")
    print(f"Trials per combination: {args.n_trials}")
    print(f"Early stopping patience: {args.early_stop}")
    print(f"Optimization metric: {args.optimization_metric}")
    print(f"Search space: {search_space}")
    print(f"Total Experiments: {n_experiments}")
    print(f"Output: {output_base}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n[Dry Run] Configuration saved. Exiting.")
        return
    
    # 创建多模型优化器
    optimizer = MultiModelPhiFeatureOptimizer(
        base_config=base_config,
        models=args.models,
        phi_types=phi_types,
        use_phi_as_feature=args.use_phi_as_feature,
        market=args.market,
        output_dir=output_base,
        search_space=search_space,
        n_trials=args.n_trials,
        seed=args.seed,
        optimization_metric=args.optimization_metric,
        early_stopping_patience=args.early_stop,
        compare_with_baseline=args.compare_with_baseline
    )
    
    # 执行优化
    results = optimizer.optimize_all()
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"Results saved to: {output_base}")
    print("=" * 80)
    
    # 打印最佳结果
    metric_col = f'best_{args.optimization_metric}'
    if len(results) > 0 and metric_col in results.columns:
        valid_results = results[results[metric_col].notna()]
        if len(valid_results) > 0:
            best_idx = valid_results[metric_col].idxmax()
            best = valid_results.loc[best_idx]
            print("\nBest Overall Result:")
            print(f"  Model: {best['model_type']}")
            print(f"  Phi Type: {best['phi_type']}")
            print(f"  Use Phi as Feature: {best['use_phi_as_feature']}")
            print(f"  {args.optimization_metric}: {best[metric_col]:.4f}")
            print(f"  Hyperparameters: lr={best['best_lr']}, layers={best['best_layers']}, "
                  f"d_model={best['best_d_model']}, bs={best['best_bs']}")


if __name__ == "__main__":
    main()
