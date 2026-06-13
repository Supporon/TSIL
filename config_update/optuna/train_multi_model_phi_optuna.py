"""
多模型phi类型测试脚本 - Optuna智能超参数优化版本
使用Optuna库实现贝叶斯优化，智能搜索最优超参数组合

特性：
- 贝叶斯优化智能搜索
- 多指标优化支持
- 早停机制
- 可视化分析
- 进度监控
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

# Optuna相关导入
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

# 尝试导入自定义 phi 构造器模块
try:
    from src.custom_phi_constructors import get_custom_I_P
    CUSTOM_PHI_AVAILABLE = True
except ImportError:
    CUSTOM_PHI_AVAILABLE = False
    print("Warning: custom_phi_constructors module not found.")


# ============================================================================
# 超参数搜索空间定义
# ============================================================================
DEFAULT_SEARCH_SPACE = {
    'lr': {'type': 'categorical', 'choices': [0.001, 0.0001, 0.00001]},
    'layers': {'type': 'int', 'low': 1, 'high': 3},
    'd_model': {'type': 'categorical', 'choices': [64, 128, 256]},
    'bs': {'type': 'categorical', 'choices': [256, 512, 1024]}
}

# 模型类型定义
ORIGINAL_MODEL_TYPES = ['GAT', 'LSTM', 'GRU', 'TCN', 'Transformer']
BASELINE_MODEL_TYPES = ['LSR_IGRU', 'MASTER', 'MERA', 'StockMixer']
MODEL_TYPES = ORIGINAL_MODEL_TYPES + BASELINE_MODEL_TYPES

# Phi 类型定义
ORIGINAL_PHI_TYPES = [
    'mse', 'phi_ones', 'phi_timestamp_last_onehot', 'phi_timestamp_last',
    'phi_period', 'phi_period_per_sample',
    'phi_stats_mean', 'phi_stats_var', 'phi_stats_skew', 'phi_stats_kurt'
]

TIMESERIES_PHI_TYPES = [
    'phi_momentum', 'phi_volatility', 'phi_return_dist',
    'phi_trend', 'phi_multiscale', 'phi_temporal',
]

FINANCIAL_PHI_TYPES = [
    'phi_feature_corr', 'phi_price_volume', 'phi_factor',
    'phi_technical', 'phi_vol_return',
]

CROSS_SAMPLE_PHI_TYPES = [
    'phi_return_sim', 'phi_market_beta', 'phi_cross_corr',
    'phi_lead_lag', 'phi_common_factor', 'phi_vol_sync',
    'phi_tail_risk', 'phi_market_regime', 'phi_sector_cluster',
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

CUSTOM_PHI_TYPES = TIMESERIES_PHI_TYPES
ALL_PHI_TYPES = USE_PHI_TYPES # CUSTOM_PHI_TYPES + ORIGINAL_PHI_TYPES
TAU_VALUES = [-4.0, -2.0, 0.0, 2.0, 4.0]


class OptunaHyperparameterOptimizer:
    """Optuna超参数优化器"""
    
    def __init__(
        self,
        base_config: dict,
        model_type: str,
        phi_type: str,
        tau: float,
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
            phi_type: phi类型
            tau: tau值
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
        self.tau = tau
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
        
        # 创建输出目录
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
        if 'tau' in hyperparams:
            config["model_config"]["tau_hat_init"] = hyperparams['tau']
        
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
        
        # 设置phi类型和tau
        config["model_config"]["phi_type"] = self.phi_type
        # config["model_config"]["tau_hat_init"] = self.tau
        config["market"] = self.market
        
        return config
    
    def _run_experiment(self, config: dict, trial_number: int) -> dict:
        """运行单个实验"""
        # 初始化qlib
        qlib.init(
            provider_uri=config["qlib_init"]["provider_uri"],
            region=config["qlib_init"]["region"],
        )
        
        # 设置实验路径
        exp_path = os.path.join(self.output_dir, "experiments")
        os.makedirs(exp_path, exist_ok=True)
        R.set_uri(exp_path)
        
        result = {}
        
        with R.start(
            experiment_name=f"{self.model_type}_{self.phi_type}_Optuna",
            recorder_name=f"trial_{trial_number}"
        ):
            logdir = config["task"]["model"]["kwargs"]["logdir"]
            os.makedirs(logdir, exist_ok=True)
            
            dataset = init_instance_by_config(config["task"]["dataset"])
            model = init_instance_by_config(config["task"]["model"])
            
            if self.model_type not in ['XGBoost', 'LightGBM']:
                preds, metrics = model.fit(dataset)
            else:
                model.fit(dataset)
                preds = model.predict(dataset)
                preds = preds.to_frame(name='score')
                ori_index = preds.index
                preds = preds.reset_index(drop=True)
                label = dataset.prepare("test")[['LABEL0']].reset_index(drop=True)
                label.rename(columns={'LABEL0': 'label'}, inplace=True)
                preds = pd.concat([preds, label], axis=1).dropna().reset_index(drop=True)
                preds.set_index(ori_index, inplace=True)
                scores = preds['score'].values
                labels = preds['label'].values
                metrics = {
                    "InfT": -1, 
                    "MSE": np.mean((labels - scores) ** 2), 
                    "MAE": np.mean(np.abs(labels - scores))
                }
            
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
            
            result = {
                'IC': ic_mean,
                'ICIR': icir,
                'RankIC': rankic_mean,
                'RankICIR': rankicir,
                'MSE': metrics.get('MSE', 0),
                'MAE': metrics.get('MAE', 0),
                'best_epoch': metrics.get('best_epoch', -1),
                'best_score': metrics.get('best_score', 0),
                'AR_without_cost': analysis_dict.get('1day.excess_return_without_cost.annualized_return', 0),
                'IR_without_cost': analysis_dict.get('1day.excess_return_without_cost.information_ratio', 0),
                'MDD_without_cost': analysis_dict.get('1day.excess_return_without_cost.max_drawdown', 0),
                'AR_with_cost': analysis_dict.get('1day.excess_return_with_cost.annualized_return', 0),
                'IR_with_cost': analysis_dict.get('1day.excess_return_with_cost.information_ratio', 0),
                'MDD_with_cost': analysis_dict.get('1day.excess_return_with_cost.max_drawdown', 0)
            }
        
        return result
    
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
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna目标函数"""
        # 建议超参数
        hyperparams = self._suggest_hyperparameters(trial)
        
        # 应用超参数
        config = self._apply_hyperparameters(self.base_config, hyperparams)
        config["task"]["model"]["kwargs"]["logdir"] = self.output_dir
        config["model_config"]["ind"] = trial.number
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*60}")
        
        try:
            # 运行实验
            result = self._run_experiment(config, trial.number)
            
            # 记录次要指标
            for metric in self.secondary_metrics:
                if metric in result:
                    trial.set_user_attr(metric, result[metric])
            
            # 记录所有指标
            trial.set_user_attr('all_metrics', result)
            
            # 保存到历史记录
            trial_record = {
                'trial_number': trial.number,
                **hyperparams,
                **result
            }
            self.trial_history.append(trial_record)
            
            # 获取优化指标
            objective_value = result.get(self.optimization_metric, float('-inf'))
            
            # 检查是否为最佳结果
            if objective_value > self.best_value:
                self.best_value = objective_value
                self.best_result = trial_record
                self.no_improvement_count = 0
                print(f"*** New Best! {self.optimization_metric}: {objective_value:.4f} ***")
            else:
                self.no_improvement_count += 1
            
            # 打印当前结果
            print(f"Result: {self.optimization_metric}={objective_value:.4f}")
            for metric in self.secondary_metrics:
                if metric in result:
                    print(f"  {metric}: {result[metric]:.4f}")
            
            # 实时保存历史
            self._save_trial_history()
            
            # 早停检查
            if self.no_improvement_count >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.early_stopping_patience} trials without improvement")
                trial.study.stop()
            
            return objective_value
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('-inf')
    
    def _save_trial_history(self):
        """保存试验历史"""
        history_df = pd.DataFrame(self.trial_history)
        history_df.to_csv(
            os.path.join(self.output_dir, 'trial_history.csv'), 
            index=False
        )
    
    def _early_stopping_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """早停回调"""
        if self.no_improvement_count >= self.early_stopping_patience:
            study.stop()
    
    def optimize(self) -> Tuple[dict, optuna.Study]:
        """执行优化"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Please install with: pip install optuna")
        
        # 创建study
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.model_type}_{self.phi_type}_optimization"
        )

        # db_url = f"sqlite:///{self.output_dir}/optuna_distributed.db" 
        # study = optuna.create_study(
        #     direction='maximize',
        #     sampler=sampler,
        #     pruner=pruner,
        #     study_name=f"{self.model_type}_{self.phi_type}_tau{self.tau}", # 确保名字唯一
        #     storage=db_url,  # <--- 关键修改：持久化存储
        #     load_if_exists=True # <--- 关键修改：允许断点续传
        # )
        
        # 添加回调
        callbacks = [self._early_stopping_callback]
        
        print(f"\n{'#'*60}")
        print(f"Starting Optuna Optimization")
        print(f"Model: {self.model_type}, Phi: {self.phi_type}, Tau: {self.tau}")
        print(f"Optimization Metric: {self.optimization_metric} (maximize)")
        print(f"Number of Trials: {self.n_trials}")
        print(f"Early Stopping Patience: {self.early_stopping_patience}")
        print(f"{'#'*60}\n")
        
        # 执行优化
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # 生成报告
        self._generate_optimization_report(study)
        
        # 生成可视化
        self._generate_visualizations(study)
        
        return self.best_result, study
    
    def _generate_optimization_report(self, study: optuna.Study):
        """生成优化报告"""
        report = []
        report.append("=" * 80)
        report.append("Optuna Hyperparameter Optimization Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Model Type: {self.model_type}")
        report.append(f"Phi Type: {self.phi_type}")
        report.append(f"Tau: {self.tau}")
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
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'optimization_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nOptimization report saved to: {report_path}")
    
    def _generate_visualizations(self, study: optuna.Study):
        """生成可视化图表"""
        try:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 优化历史图
            try:
                fig = plot_optimization_history(study)
                fig.write_html(os.path.join(vis_dir, 'optimization_history.html'))
                print(f"Saved: optimization_history.html")
            except Exception as e:
                print(f"Warning: Could not generate optimization history plot: {e}")
            
            # 参数重要性图
            try:
                fig = plot_param_importances(study)
                fig.write_html(os.path.join(vis_dir, 'param_importances.html'))
                print(f"Saved: param_importances.html")
            except Exception as e:
                print(f"Warning: Could not generate param importance plot: {e}")
            
            # 等高线图
            try:
                param_names = list(study.best_params.keys())
                if len(param_names) >= 2:
                    fig = plot_contour(study, params=param_names[:2])
                    fig.write_html(os.path.join(vis_dir, 'contour.html'))
                    print(f"Saved: contour.html")
            except Exception as e:
                print(f"Warning: Could not generate contour plot: {e}")
            
            # 平行坐标图
            try:
                fig = plot_parallel_coordinate(study)
                fig.write_html(os.path.join(vis_dir, 'parallel_coordinate.html'))
                print(f"Saved: parallel_coordinate.html")
            except Exception as e:
                print(f"Warning: Could not generate parallel coordinate plot: {e}")
            
            # 切片图
            try:
                fig = plot_slice(study)
                fig.write_html(os.path.join(vis_dir, 'slice.html'))
                print(f"Saved: slice.html")
            except Exception as e:
                print(f"Warning: Could not generate slice plot: {e}")
            
            print(f"\nVisualizations saved to: {vis_dir}")
            
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")


class MultiModelOptunaOptimizer:
    """多模型Optuna优化器"""
    
    def __init__(
        self,
        base_config: dict,
        models: List[str],
        phi_types: List[str],
        tau_values: List[float],
        market: str,
        output_dir: str,
        search_space: dict = None,
        n_trials: int = 50,
        seed: int = 2025,
        optimization_metric: str = 'IC',
        early_stopping_patience: int = 10
    ):
        self.base_config = base_config
        self.models = models
        self.phi_types = phi_types
        self.tau_values = tau_values
        self.market = market
        self.output_dir = output_dir
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.seed = seed
        self.optimization_metric = optimization_metric
        self.early_stopping_patience = early_stopping_patience
        
        self.all_results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize_all(self) -> pd.DataFrame:
        """优化所有模型组合"""
        total_combinations = len(self.models) * len(self.phi_types) * len(self.tau_values)
        current = 0
        
        print(f"\n{'#'*80}")
        print(f"Multi-Model Optuna Optimization")
        print(f"{'#'*80}")
        print(f"Models: {self.models}")
        print(f"Phi Types: {self.phi_types}")
        print(f"Tau Values: {self.tau_values}")
        print(f"Total Combinations: {total_combinations}")
        print(f"Trials per Combination: {self.n_trials}")
        print(f"{'#'*80}\n")
        
        for model_type in self.models:
            for phi_type in self.phi_types:
                for tau in self.tau_values:
                    current += 1
                    
                    print(f"\n{'='*60}")
                    print(f"Optimization {current}/{total_combinations}")
                    print(f"Model: {model_type}, Phi: {phi_type}, Tau: {tau}")
                    print(f"{'='*60}")
                    
                    # 创建模型输出目录
                    model_output_dir = os.path.join(
                        self.output_dir, 
                        f"{model_type.lower()}_{phi_type}_tau{tau}"
                    )
                    
                    # 更新配置
                    config = copy.deepcopy(self.base_config)
                    config["task"]["model"]["kwargs"]["model_type"] = model_type
                    config["task"]["model"]["kwargs"]["logdir"] = model_output_dir
                    config["logdir"] = model_output_dir
                    
                    # 创建优化器
                    optimizer = OptunaHyperparameterOptimizer(
                        base_config=config,
                        model_type=model_type,
                        phi_type=phi_type,
                        tau=tau,
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
                                'tau': tau,
                                'best_lr': best_result.get('lr'),
                                'best_layers': best_result.get('layers'),
                                'best_d_model': best_result.get('d_model'),
                                'best_bs': best_result.get('bs'),
                                **{f'best_{k}': v for k, v in best_result.items() 
                                   if k not in ['lr', 'layers', 'd_model', 'bs', 'trial_number']}
                            }
                            self.all_results.append(result_record)
                            
                    except Exception as e:
                        print(f"Error optimizing {model_type}/{phi_type}/tau={tau}: {e}")
                        self.all_results.append({
                            'model_type': model_type,
                            'phi_type': phi_type,
                            'tau': tau,
                            'error': str(e)
                        })
        
        # 保存所有结果
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(os.path.join(self.output_dir, 'all_best_results.csv'), index=False)
        
        # 生成汇总报告
        self._generate_summary_report(results_df)
        
        return results_df
    
    def _generate_summary_report(self, results_df: pd.DataFrame):
        """生成汇总报告"""
        report = []
        report.append("=" * 80)
        report.append("Multi-Model Optuna Optimization Summary")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Combinations: {len(results_df)}")
        report.append(f"Successful: {len(results_df[~results_df.get('error', pd.Series()).notna()])}")
        report.append("")
        
        # 按模型汇总最佳结果
        report.append("-" * 60)
        report.append("Best Results by Model")
        report.append("-" * 60)
        
        if f'best_{self.optimization_metric}' in results_df.columns:
            for model in self.models:
                model_results = results_df[results_df['model_type'] == model]
                if len(model_results) > 0:
                    metric_col = f'best_{self.optimization_metric}'
                    if metric_col in model_results.columns:
                        best_idx = model_results[metric_col].idxmax()
                        best = model_results.loc[best_idx]
                        report.append(f"\n{model}:")
                        report.append(f"  Best {self.optimization_metric}: {best.get(metric_col, 'N/A'):.4f}")
                        report.append(f"  Phi Type: {best.get('phi_type')}")
                        report.append(f"  Tau: {best.get('tau')}")
                        report.append(f"  Hyperparameters:")
                        report.append(f"    lr: {best.get('best_lr')}")
                        report.append(f"    layers: {best.get('best_layers')}")
                        report.append(f"    d_model: {best.get('best_d_model')}")
                        report.append(f"    bs: {best.get('best_bs')}")
        
        report.append("\n" + "=" * 80)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nSummary report saved to: {report_path}")


def get_available_phi_types(phi_category: str = 'all') -> list:
    """获取可用的 phi 类型列表"""
    category_map = {
        'all': ALL_PHI_TYPES,
        'original': ORIGINAL_PHI_TYPES,
        'custom': CUSTOM_PHI_TYPES,
        'timeseries': TIMESERIES_PHI_TYPES,
        'financial': FINANCIAL_PHI_TYPES,
        'cross_sample': CROSS_SAMPLE_PHI_TYPES,
    }
    return category_map.get(phi_category, ALL_PHI_TYPES)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Multi-Model Phi Type Testing with Optuna Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optuna Optimization Features:
  - Bayesian optimization with TPE sampler
  - Early stopping when no improvement
  - Visualization of optimization history
  - Parameter importance analysis

Examples:
  # Single model optimization
  python train_multi_model_phi_optuna.py --models LSTM --phi_types mse --n_trials 30
  
  # Multi-model optimization
  python train_multi_model_phi_optuna.py --models LSTM GRU --phi_category original
  
  # Custom search space with early stopping
  python train_multi_model_phi_optuna.py --models LSTM --n_trials 50 --early_stop 15
"""
    )
    
    # 基本参数
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--config_file", type=str, default="configs_phi/config_base_phi.yaml")
    
    # 模型选择
    parser.add_argument("--models", type=str, nargs='+', default=['LSTM'])
    
    # Phi类型选择
    parser.add_argument("--phi_types", type=str, nargs='+', default=None)
    parser.add_argument("--phi_category", type=str, default="all",
                        choices=['all', 'original', 'custom', 'timeseries', 'financial', 'cross_sample'])
    
    # Tau值
    parser.add_argument("--tau_values", type=float, nargs='+', default=[0.0]) # default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    
    # 市场
    parser.add_argument("--market", type=str, default="csi300",
                        choices=['csi300', 'csi500', 'csi800'])
    
    # Optuna参数
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
    parser.add_argument("--layers_range", type=int, nargs='+',
                         default=[1, 2, 3])
    parser.add_argument("--d_model_choices", type=int, nargs='+', 
                        default=[64, 128, 256])
    parser.add_argument("--bs_choices", type=int, nargs='+', 
                        default=[1024])
    parser.add_argument("--tau_choices", type=float, nargs='+', 
                        default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="output_model_optuna_csi800/output_optuna")
    parser.add_argument("--seq_len", type=int, default=20)
    
    # 其他
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is not installed.")
        print("Please install with: pip install optuna optuna-dashboard plotly")
        return
    
    # 确定phi类型
    if args.phi_types is not None:
        phi_types = args.phi_types
    else:
        phi_types = get_available_phi_types(args.phi_category)
    
    # 构建搜索空间
    search_space = {
        'lr': {'type': 'categorical', 'choices': args.lr_choices},
        'layers': {'type': 'categorical', 'choices': args.layers_range},
        'd_model': {'type': 'categorical', 'choices': args.d_model_choices},
        'bs': {'type': 'categorical', 'choices': args.bs_choices},
        'tau': {'type': 'categorical', 'choices': args.tau_choices},
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
    
    # 保存配置
    exp_info = {
        'seed': args.seed,
        'models': args.models,
        'phi_types': phi_types,
        'tau_values': args.tau_values,
        'market': args.market,
        'n_trials': args.n_trials,
        'early_stopping_patience': args.early_stop,
        'optimization_metric': args.optimization_metric,
        'search_space': search_space,
        'timestamp': timestamp
    }
    with open(os.path.join(output_base, 'experiment_info.json'), 'w') as f:
        json.dump(exp_info, f, indent=2)
    
    print("=" * 80)
    print("Optuna Hyperparameter Optimization")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Phi Types: {phi_types}")
    print(f"Tau Values: {args.tau_values}")
    print(f"Trials per combination: {args.n_trials}")
    print(f"Early stopping patience: {args.early_stop}")
    print(f"Optimization metric: {args.optimization_metric}")
    print(f"Search space: {search_space}")
    print(f"Output: {output_base}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n[Dry Run] Configuration saved. Exiting.")
        return
    
    # 创建多模型优化器
    optimizer = MultiModelOptunaOptimizer(
        base_config=base_config,
        models=args.models,
        phi_types=phi_types,
        tau_values=args.tau_values,
        market=args.market,
        output_dir=output_base,
        search_space=search_space,
        n_trials=args.n_trials,
        seed=args.seed,
        optimization_metric=args.optimization_metric,
        early_stopping_patience=args.early_stop
    )
    
    # 执行优化
    results = optimizer.optimize_all()
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"Results saved to: {output_base}")
    print("=" * 80)
    
    # 打印最佳结果
    if len(results) > 0 and f'best_{args.optimization_metric}' in results.columns:
        metric_col = f'best_{args.optimization_metric}'
        best_idx = results[metric_col].idxmax()
        best = results.loc[best_idx]
        print("\nBest Overall Result:")
        print(f"  Model: {best['model_type']}")
        print(f"  Phi Type: {best['phi_type']}")
        print(f"  Tau: {best['tau']}")
        print(f"  {args.optimization_metric}: {best[metric_col]:.4f}")
        print(f"  Hyperparameters: lr={best['best_lr']}, layers={best['best_layers']}, "
              f"d_model={best['best_d_model']}, bs={best['best_bs']}")


if __name__ == "__main__":
    main()
