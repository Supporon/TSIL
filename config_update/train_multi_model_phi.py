"""
多模型phi类型测试脚本
用于测试不同基模型（GAT、LSTM、GRU、TCN、Transformer）在不同phi类型上的表现

集成了 custom_phi_constructors.py 中定义的基于领域知识的 phi 谓词构造器，
包括基于时间序列特性、金融指标关联性、股票间相关性等多种 phi 类型。
"""
import argparse
import os
import sys
import copy
import json
from pathlib import Path
from datetime import datetime

import qlib
import pandas as pd
import numpy as np
import ruamel.yaml as YAML
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
import warnings

warnings.filterwarnings('ignore')
os.chdir(sys.path[0])

from basktesting import calculate_portfolio_metrics, top_k_drop_strategy

# 尝试导入自定义 phi 构造器模块以获取可用的 phi 类型
try:
    from src.custom_phi_constructors import get_custom_I_P
    CUSTOM_PHI_AVAILABLE = True
except ImportError:
    CUSTOM_PHI_AVAILABLE = False
    print("Warning: custom_phi_constructors module not found. Using default phi types only.")


# 定义要测试的模型类型
MODEL_TYPES = ['GAT', 'LSTM', 'GRU', 'TCN', 'Transformer']

# ============================================================================
# Phi 类型定义
# ============================================================================

# 原有的 phi 类型（基于 model_backbone_phix.py 中定义）
ORIGINAL_PHI_TYPES = [
    # 'mse',  # 基准 MSE 损失
    'phi_ones',  # 全1向量
    'phi_timestamp_last_onehot', 'phi_timestamp_last',
    'phi_timestamp_sl_onehot', 'phi_timestamp_sl',
    'phi_timestamp_dim_onehot', 'phi_timestamp_dim',
    'phi_period_sl_onehot', 'phi_period_per_sample_onehot',
    'phi_stats_mean', 'phi_stats_var', 'phi_stats_skew', 'phi_stats_kurt'
]

# 基于时间序列特性的 phi 类型（来自 custom_phi_constructors.py）
TIMESERIES_PHI_TYPES = [
    'phi_momentum',       # 动量因子谓词
    'phi_volatility',     # 波动率聚类谓词
    'phi_return_dist',    # 收益率分布形态谓词
    'phi_trend',          # 趋势强度谓词
    'phi_multiscale',     # 多尺度特征谓词
    'phi_temporal',       # 时序模式谓词
]

# 基于金融指标关联性的 phi 类型（来自 custom_phi_constructors.py）
FINANCIAL_PHI_TYPES = [
    'phi_feature_corr',   # 特征相关性结构谓词
    'phi_price_volume',   # 价量关系谓词
    'phi_factor',         # 因子暴露谓词
    'phi_technical',      # 技术指标综合谓词
    'phi_vol_return',     # 波动率-收益联合谓词
]

# 基于股票间相关性的 phi 类型（来自 custom_phi_constructors.py）
CROSS_SAMPLE_PHI_TYPES = [
    'phi_return_sim',     # 收益率相似性聚类谓词
    'phi_market_beta',    # 市场 Beta 谓词
    'phi_cross_corr',     # 跨样本相关性谓词
    'phi_lead_lag',       # 领先滞后关系谓词
    'phi_common_factor',  # 共同因子暴露谓词
    'phi_vol_sync',       # 波动率同步性谓词
    'phi_tail_risk',      # 尾部风险共现谓词
    'phi_market_regime',  # 市场状态敏感性谓词
    'phi_sector_cluster', # 板块聚类谓词
]

# 自定义 phi 类型（来自 custom_phi_constructors.py 的所有类型）
CUSTOM_PHI_TYPES = TIMESERIES_PHI_TYPES# + FINANCIAL_PHI_TYPES + CROSS_SAMPLE_PHI_TYPES

# 所有可用的 phi 类型
ALL_PHI_TYPES = CUSTOM_PHI_TYPES + ORIGINAL_PHI_TYPES

# 默认使用的 phi 类型（包含所有类型）
PHI_TYPES = ALL_PHI_TYPES

# 定义 tau 值
TAU_VALUES = [-4.0, -2.0, 0.0, 2.0, 4.0]


def get_available_phi_types(phi_category: str = 'all') -> list:
    """
    获取可用的 phi 类型列表
    
    Args:
        phi_category: phi 类型类别，可选值：
            - 'all': 所有 phi 类型
            - 'original': 原有的 phi 类型
            - 'custom': 自定义的 phi 类型（来自 custom_phi_constructors.py）
            - 'timeseries': 基于时间序列特性的 phi 类型
            - 'financial': 基于金融指标关联性的 phi 类型
            - 'cross_sample': 基于股票间相关性的 phi 类型
    
    Returns:
        phi 类型列表
    """
    category_map = {
        'all': ALL_PHI_TYPES,
        'original': ORIGINAL_PHI_TYPES,
        'custom': CUSTOM_PHI_TYPES,
        'timeseries': TIMESERIES_PHI_TYPES,
        'financial': FINANCIAL_PHI_TYPES,
        'cross_sample': CROSS_SAMPLE_PHI_TYPES,
    }
    return category_map.get(phi_category, ALL_PHI_TYPES)


def get_model_optimal_params(model_type: str, market: str = 'csi300') -> dict:
    """
    根据模型类型和市场返回最优参数配置
    
    Args:
        model_type: 模型类型 (GAT, LSTM, GRU, TCN, Transformer)
        market: 市场类型 (csi300, csi500, csi800)
    
    Returns:
        包含最优参数的字典
    """
    # # CSI300 最优参数-ori
    # csi300_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'LSTM': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 3, 'd_model': 64},
    #     'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 2, 'd_model': 64},
    #     'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 256}
    # }
    
    # # CSI500 最优参数-ori
    # csi500_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 2, 'd_model': 64},
    #     'LSTM': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 2, 'd_model': 64},
    #     'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 256},
    #     'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 128}
    # }
    
    # # CSI800 最优参数-ori
    # csi800_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64},
    #     'GRU': {'seq_len': 20, 'lr': 0.001, 'bs': 512, 'layers': 3, 'd_model': 128},
    #     'TCN': {'seq_len': 20, 'lr': 0.001, 'bs': 256, 'layers': 3, 'd_model': 64},
    #     'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 256}
    # }

    # CSI300 参数随意设置 - 1216
    # csi300_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 1, 'd_model': 64},
    #     'LSTM': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 3, 'd_model': 64},
    #     'GRU': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 3, 'd_model': 64},
    #     'TCN': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 2, 'd_model': 64},
    #     'Transformer': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 2, 'd_model': 64}
    # }

    # # CSI500 参数随意设置 - 1216
    # csi500_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 2, 'd_model': 128},
    #     'LSTM': {'seq_len': 20, 'lr': 0.0001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 2, 'd_model': 128}
    # }

    # # CSI800 参数随意设置 - 1216
    # csi800_params = {
    #     'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 128},
    #     'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64},
    #     'GRU': {'seq_len': 20, 'lr': 0.001, 'bs': 512, 'layers': 3, 'd_model': 128},
    #     'TCN': {'seq_len': 20, 'lr': 0.001, 'bs': 256, 'layers': 3, 'd_model': 64},
    #     'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 256}
    # }

    # CSI300 基于轮数 - 1217
    csi300_params = {
        'GAT': {'seq_len': 20, 'lr': 0.001, 'bs': 256, 'layers': 2, 'd_model': 256},
        'LSTM': {'seq_len': 20, 'lr': 0.0001, 'bs': 256, 'layers': 1, 'd_model': 128},
        'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 128},
        'TCN': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 3, 'd_model': 128},
        'Transformer': {'seq_len': 20, 'lr': 0.00001, 'bs': 512, 'layers': 1, 'd_model': 256}
    }

    # CSI500 基于轮数 - 1217
    csi500_params = {
        'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 2, 'd_model': 128},
        'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 256, 'layers': 1, 'd_model': 64},
        'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'Transformer': {'seq_len': 20, 'lr': 0.001, 'bs': 1024, 'layers': 1, 'd_model': 128}
    }

    # CSI800 基于轮数 - 1217
    csi800_params = {
        'GAT': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64},
        'LSTM': {'seq_len': 20, 'lr': 0.00001, 'bs': 512, 'layers': 2, 'd_model': 64},
        'GRU': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 1, 'd_model': 64},
        'TCN': {'seq_len': 20, 'lr': 0.00001, 'bs': 512, 'layers': 3, 'd_model': 128},  ##
        'Transformer': {'seq_len': 20, 'lr': 0.00001, 'bs': 1024, 'layers': 3, 'd_model': 64}
    }

    params_map = {
        'csi300': csi300_params,
        'csi500': csi500_params,
        'csi800': csi800_params
    }
    
    market_params = params_map.get(market.lower(), csi300_params)
    
    if model_type not in market_params:
        return {'seq_len': 20, 'lr': 0.0001, 'bs': 512, 'layers': 2, 'd_model': 64}
    
    return market_params[model_type]


def get_experiment_metrics(file_name: str) -> dict:
    """从实验目录获取指标"""
    folder = Path(file_name)
    file_dict = {}
    
    for file_path in folder.glob('*'):
        if file_path.is_file():
            try:
                v = file_path.read_text(encoding='utf-8')
                file_dict[file_path.name] = float(v.split(' ')[1])
            except Exception as e:
                print(f"跳过 {file_path.name}: {e}")
    
    return file_dict


def run_single_experiment(seed: int, config: dict, config_file: str, 
                          model_type: str, phi_type: str, tau: float, exp_idx: int) -> dict:
    """
    运行单个实验
    
    Args:
        seed: 随机种子
        config: 配置字典
        config_file: 配置文件路径
        model_type: 模型类型
        phi_type: phi类型
        tau: tau初始值
        exp_idx: 实验索引
    
    Returns:
        包含实验指标的字典
    """
    print('=' * 60)
    print(f"Model: {model_type}, Phi: {phi_type}, Tau: {tau}, Index: {exp_idx}")
    print('=' * 60)
    
    # 初始化qlib
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )
    
    # 设置实验路径
    exp_settings = config.get("experiment_settings", {
        "exp_path": config['logdir'] + "/experiments",
        "experiment_name": f"{model_type}_{phi_type}_Experiment",
        "recorder_name": f"tau_{tau}_Recorder"
    })
    
    exp_path = exp_settings["exp_path"]
    os.makedirs(exp_path, exist_ok=True)
    R.set_uri(exp_path)
    
    result = {
        'model_type': model_type,
        'phi_type': phi_type,
        'tau': tau,
        'exp_idx': exp_idx
    }
    
    # 启动实验记录器
    with R.start(
            experiment_name=exp_settings["experiment_name"],
            recorder_name=exp_settings["recorder_name"]
    ):
        logdir = config["task"]["model"]["kwargs"]["logdir"]
        
        dataset = init_instance_by_config(config["task"]["dataset"])
        model = init_instance_by_config(config["task"]["model"])
        
        if model_type not in ['XGBoost', 'LightGBM']:
            preds, metrics = model.fit(dataset)
        else:
            model.fit(dataset)
            preds = model.predict(dataset)
            preds = preds.to_frame(name='score')
            ori_index = preds.index
            preds = preds.reset_index(drop=True)
            if model_type in ['XGBoost', 'LightGBM']:
                label = dataset.prepare("test")[['LABEL0']].reset_index(drop=True)
                label.rename(columns={'LABEL0': 'label'}, inplace=True)
            else:
                label = dataset.prepare("test", col_set=["label"]).data_arr[:-1]
                label = pd.Series(label.reshape(-1, )).to_frame(name='label')
            preds = pd.concat([preds, label], axis=1).dropna().reset_index(drop=True)
            preds.set_index(ori_index, inplace=True)
            scores = preds['score'].values
            labels = preds['label'].values
            metrics = {"InfT": -1, "MSE": np.mean((labels - scores) ** 2), "MAE": np.mean(np.abs(labels - scores))}
        
        # 计算排名指标
        per_df = preds.reset_index()
        per_df = per_df.sort_values(['datetime', 'instrument'])
        per_df['group_id'] = per_df.groupby(['datetime', 'instrument']).cumcount()
        
        r_metrics = {'IC': [], 'ICIR': [], 'RankIC': [], 'RankICIR': []}
        r_df = []
        
        strategy_config = config['strategy_config']
        
        sub_df = per_df[per_df['group_id'] == 0]
        sub_df = sub_df.set_index(['datetime', 'instrument'])
        sub_df = sub_df[['score', 'label']]
        
        # Ranking Metric
        ic = sub_df.groupby(level=0).apply(lambda group: group['score'].corr(group['label'])).to_numpy()
        icir = ic.mean() / ic.std()
        ic = ic.mean()
        rankic = sub_df.groupby(level=0).apply(
            lambda group: group['score'].corr(group['label'], method='spearman')).to_numpy()
        rankicir = rankic.mean() / rankic.std()
        rankic = rankic.mean()
        r_metrics['IC'].append(ic)
        r_metrics['ICIR'].append(icir)
        r_metrics['RankIC'].append(rankic)
        r_metrics['RankICIR'].append(rankicir)
        
        r_df.append(top_k_drop_strategy(sub_df, top_k=strategy_config['topk'], n=strategy_config['hold_thre'],
                                        commission_rate=strategy_config['commission_rate']))
        
        r_df = pd.concat(r_df, ignore_index=True)
        # Portfolio Metric
        portfolio_metrics, df = calculate_portfolio_metrics(r_df)
        metrics.update(portfolio_metrics)
        rank_metrics = {key: np.mean(values) for key, values in r_metrics.items()}
        metrics.update(rank_metrics)
        
        # 记录信号和回测
        port_analysis_config = config["port_analysis_config"]
        recorder = R.get_recorder()
        
        # 使用 try-except 包装记录操作，防止因指标数据问题导致整个实验失败
        try:
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
        except Exception as e:
            print(f"Warning: SignalRecord generation failed: {e}")
        
        try:
            sar = SigAnaRecord(recorder)
            sar.generate(label=preds[['label']])
        except Exception as e:
            print(f"Warning: SigAnaRecord generation failed: {e}")
        
        try:
            par = PortAnaRecord(recorder, port_analysis_config, "day")
            par.generate()
        except Exception as e:
            print(f"Warning: PortAnaRecord generation failed: {e}")
        
        # 确保输出目录存在
        os.makedirs(logdir, exist_ok=True)
        
        df.to_csv(os.path.join(logdir, f'backtest_result_{phi_type}_tau_{tau}_ind_{exp_idx}.csv'))
        
        analysis_dict = get_experiment_metrics(
            recorder.uri + '/' + recorder.experiment_id + '/' + recorder.id + '/metrics/')
        
        # 生成报告
        report = f"""
            **********************************
                    Backtest Report
            **********************************
            Model Type:      {model_type}
            Phi Type:        {phi_type}
            Tau:             {tau}
            **********************************
                 InfT:       {metrics['InfT']:.6f}
            Best epoch:      {metrics['best_epoch']}
            Best score:      {metrics['best_score']:.4f}
                  MSE:       {metrics['MSE']:.4f}
                  MAE:       {metrics['MAE']:.4f}
                   IC:       {metrics['IC']:.4f}
                 ICIR:       {metrics['ICIR']:.4f}
               RankIC:       {metrics['RankIC']:.4f}
             RankICIR:       {metrics['RankICIR']:.4f}
            w/o cost AR:    {analysis_dict.get('1day.excess_return_without_cost.annualized_return', 0):.4f}
            w/o cost ret mean:    {analysis_dict.get('1day.excess_return_without_cost.mean', 0):.4f}
            w/o cost ret std:    {analysis_dict.get('1day.excess_return_without_cost.std', 0):.4f}
            w/o cost IR:    {analysis_dict.get('1day.excess_return_without_cost.information_ratio', 0):.4f}
            w/o cost MMD:    {analysis_dict.get('1day.excess_return_without_cost.max_drawdown', 0):.4f}
            with cost AR:    {analysis_dict.get('1day.excess_return_with_cost.annualized_return', 0):.4f}
            with cost ret mean:    {analysis_dict.get('1day.excess_return_with_cost.mean', 0):.4f}
            with cost ret std:    {analysis_dict.get('1day.excess_return_with_cost.std', 0):.4f}
            with cost IR:    {analysis_dict.get('1day.excess_return_with_cost.information_ratio', 0):.4f}
            with cost MMD:    {analysis_dict.get('1day.excess_return_with_cost.max_drawdown', 0):.4f}
            ***************************************
            """
        print(report)
        
        # 保存报告
        report_file = os.path.join(logdir, f'backtest_report_{phi_type}_tau_{tau}_ind_{exp_idx}.txt')
        with open(report_file, 'w') as file:
            file.write(report)
        print(f"Report saved to: {report_file}")
        
        # 收集结果
        result.update({
            'InfT': metrics.get('InfT', -1),
            'best_epoch': metrics.get('best_epoch', -1),
            'best_score': metrics.get('best_score', 0),
            'MSE': metrics.get('MSE', 0),
            'MAE': metrics.get('MAE', 0),
            'IC': metrics.get('IC', 0),
            'ICIR': metrics.get('ICIR', 0),
            'RankIC': metrics.get('RankIC', 0),
            'RankICIR': metrics.get('RankICIR', 0),
            'AR_without_cost': analysis_dict.get('1day.excess_return_without_cost.annualized_return', 0),
            'IR_without_cost': analysis_dict.get('1day.excess_return_without_cost.information_ratio', 0),
            'MDD_without_cost': analysis_dict.get('1day.excess_return_without_cost.max_drawdown', 0),
            'AR_with_cost': analysis_dict.get('1day.excess_return_with_cost.annualized_return', 0),
            'IR_with_cost': analysis_dict.get('1day.excess_return_with_cost.information_ratio', 0),
            'MDD_with_cost': analysis_dict.get('1day.excess_return_with_cost.max_drawdown', 0)
        })
    
    return result


def print_phi_types_info():
    """打印所有可用的 phi 类型信息"""
    print("\n" + "=" * 80)
    print("Available Phi Types")
    print("=" * 80)
    
    print("\n[Original Phi Types] (原有phi类型):")
    for phi in ORIGINAL_PHI_TYPES:
        print(f"  - {phi}")
    
    print("\n[Timeseries Phi Types] (基于时间序列特性):")
    for phi in TIMESERIES_PHI_TYPES:
        print(f"  - {phi}")
    
    print("\n[Financial Phi Types] (基于金融指标关联性):")
    for phi in FINANCIAL_PHI_TYPES:
        print(f"  - {phi}")
    
    print("\n[Cross-Sample Phi Types] (基于股票间相关性):")
    for phi in CROSS_SAMPLE_PHI_TYPES:
        print(f"  - {phi}")
    
    print(f"\nTotal: {len(ALL_PHI_TYPES)} phi types available")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Multi-Model Phi Type Testing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phi Type Categories:
  - original    : 原有的 phi 类型 (mse, phi_ones, phi_timestamp_*, phi_stats_*)
  - timeseries  : 基于时间序列特性 (phi_momentum, phi_volatility, phi_trend, ...)
  - financial   : 基于金融指标关联性 (phi_feature_corr, phi_price_volume, ...)
  - cross_sample: 基于股票间相关性 (phi_market_beta, phi_cross_corr, ...)
  - custom      : 所有自定义 phi 类型 (timeseries + financial + cross_sample)
  - all         : 所有 phi 类型 (original + custom)

Examples:
  # 测试所有模型和所有 phi 类型
  python train_multi_model_phi.py --market csi300
  
  # 仅测试 LSTM 模型和自定义 phi 类型
  python train_multi_model_phi.py --models LSTM --phi_category custom
  
  # 测试特定的 phi 类型
  python train_multi_model_phi.py --models LSTM GRU --phi_types phi_momentum phi_volatility
  
  # 查看所有可用的 phi 类型
  python train_multi_model_phi.py --list_phi_types
"""
    )
    
    # 基本参数
    parser.add_argument("--seed", type=int, default=2025, 
                        help="Random seed (default: 2025)")
    parser.add_argument("--config_file", type=str, default="configs_phi/config_base_phi.yaml",
                        help="Base config file path")
    
    # 模型选择
    parser.add_argument("--models", type=str, nargs='+', default=MODEL_TYPES,
                        help=f"Model types to test. Available: {MODEL_TYPES}")
    
    # Phi 类型选择
    parser.add_argument("--phi_types", type=str, nargs='+', default=None,
                        help="Specific phi types to test. If not specified, uses --phi_category")
    parser.add_argument("--phi_category", type=str, default="all",
                        choices=['all', 'original', 'custom', 'timeseries', 'financial', 'cross_sample'],
                        help="Phi type category to test (default: all)")
    
    # Tau 值
    parser.add_argument("--tau_values", type=float, nargs='+', default=TAU_VALUES,
                        help=f"Tau values to test (default: {TAU_VALUES})")
    
    # 市场选择
    parser.add_argument("--market", type=str, default="csi300",
                        choices=['csi300', 'csi500', 'csi800'],
                        help="Market type (default: csi300)")
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="output_phi_multi_model",
                        help="Output directory prefix")
    
    # 其他选项
    parser.add_argument("--list_phi_types", action="store_true",
                        help="List all available phi types and exit")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show experiment configuration without running")
    
    args = parser.parse_args()

    # args.models = 'LSTM'
    # args.phi_category = 'all'

    # 如果要求列出 phi 类型，打印并退出
    if args.list_phi_types:
        print_phi_types_info()
        return
    
    # 确定要测试的 phi 类型
    if args.phi_types is not None:
        phi_types_to_test = args.phi_types
    else:
        phi_types_to_test = get_available_phi_types(args.phi_category)
    
    # 加载基础配置
    with open(args.config_file) as f:
        yaml = YAML.YAML(typ='safe', pure=True)
        base_config = yaml.load(f)

    print(base_config)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{args.output_dir}_{args.market}_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    # 保存实验配置
    exp_info = {
        'seed': args.seed,
        'models': args.models,
        'phi_types': phi_types_to_test,
        'phi_category': args.phi_category,
        'tau_values': args.tau_values,
        'market': args.market,
        'timestamp': timestamp,
        'config_file': args.config_file,
        'total_experiments': len(args.models) * len(phi_types_to_test) * len(args.tau_values)
    }
    with open(f'{output_base}/experiment_info.json', 'w') as f:
        json.dump(exp_info, f, indent=2)
    
    # 打印实验配置
    print("=" * 80)
    print(f"Multi-Model Phi Type Testing")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Phi Category: {args.phi_category}")
    print(f"Phi Types ({len(phi_types_to_test)}): {phi_types_to_test}")
    print(f"Tau Values: {args.tau_values}")
    print(f"Market: {args.market}")
    print(f"Total Experiments: {exp_info['total_experiments']}")
    print(f"Output Directory: {output_base}")
    print("=" * 80)
    
    # Dry run 模式：仅显示配置而不运行
    if args.dry_run:
        print("\n[Dry Run Mode] Experiment configuration saved. Exiting without running.")
        return
    
    # 收集所有结果
    all_results = []
    exp_idx = 0
    
    # 遍历所有模型
    for model_type in args.models:
        # 获取模型最优参数
        optimal_params = get_model_optimal_params(model_type, args.market)
        
        print(f"\n{'#' * 60}")
        print(f"Testing Model: {model_type}")
        print(f"Optimal Params: {optimal_params}")
        print(f"{'#' * 60}")
        
        # 为每个模型创建输出目录
        model_output_dir = f"{output_base}/{model_type.lower()}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 遍历phi类型
        for phi_type in phi_types_to_test:
            # 遍历tau值
            for tau in args.tau_values:
                exp_idx += 1
                
                # 深拷贝配置
                config = copy.deepcopy(base_config)
                
                # 更新模型类型
                config["task"]["model"]["kwargs"]["model_type"] = model_type
                
                # 更新输出目录
                config["logdir"] = model_output_dir
                config["task"]["model"]["kwargs"]["logdir"] = model_output_dir
                
                # 应用最优参数
                config["seq_len"] = optimal_params['seq_len']
                config["model_config"]["seq_len"] = optimal_params['seq_len']
                config["task"]["dataset"]["kwargs"]["seq_len"] = optimal_params['seq_len']
                config["task"]["model"]["kwargs"]["lr"] = optimal_params['lr']
                config["task"]["dataset"]["kwargs"]["batch_size"] = optimal_params['bs']
                config["model_config"]["e_layers"] = optimal_params['layers']
                config["model_config"]["d_model"] = optimal_params['d_model']
                config["model_config"]["d_ff"] = optimal_params['d_model']
                
                # 设置phi类型和tau
                config["model_config"]["phi_type"] = phi_type
                config["model_config"]["tau_hat_init"] = tau
                config["model_config"]["ind"] = exp_idx
                
                # 更新市场设置
                config["market"] = args.market
                
                # try:
                result = run_single_experiment(
                    seed=args.seed,
                    config=config,
                    config_file=args.config_file,
                    model_type=model_type,
                    phi_type=phi_type,
                    tau=tau,
                    exp_idx=exp_idx
                )
                all_results.append(result)
                # except Exception as e:
                #     print(f"Error in experiment {exp_idx}: {e}")
                #     all_results.append({
                #         'model_type': model_type,
                #         'phi_type': phi_type,
                #         'tau': tau,
                #         'exp_idx': exp_idx,
                #         'error': str(e)
                #     })
    
    # 保存所有结果到CSV（追加模式）
    results_df = pd.DataFrame(all_results)
    results_csv_path = f'{output_base}/all_results.csv'
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.exists(results_csv_path)
    results_df.to_csv(results_csv_path, mode='a', header=not file_exists, index=False)
    print(f"\nAll results appended to: {results_csv_path}")
    
    # 生成汇总报告（追加模式）
    summary_report = generate_summary_report(results_df)
    summary_path = f'{output_base}/summary_report.txt'
    with open(summary_path, 'a') as f:
        f.write(summary_report)
    print(f"Summary report appended to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print(f"Total experiments: {exp_idx}")
    print(f"Results saved to: {output_base}/")
    print("=" * 80)


def generate_summary_report(results_df: pd.DataFrame) -> str:
    """生成汇总报告"""
    report = []
    report.append("=" * 80)
    report.append("Multi-Model Phi Type Testing Summary Report")
    report.append("=" * 80 + "\n")
    
    # 检查是否有错误列
    if 'error' in results_df.columns:
        successful = results_df[results_df['error'].isna()]
        failed = results_df[~results_df['error'].isna()]
        report.append(f"Total experiments: {len(results_df)}")
        report.append(f"Successful: {len(successful)}")
        report.append(f"Failed: {len(failed)}")
    else:
        successful = results_df
        report.append(f"Total experiments: {len(results_df)}")
    
    if len(successful) == 0:
        report.append("\nNo successful experiments to summarize.")
        return "\n".join(report)
    
    report.append("\n" + "-" * 60)
    report.append("Best Results by Model Type (based on IC)")
    report.append("-" * 60)
    
    for model_type in successful['model_type'].unique():
        model_results = successful[successful['model_type'] == model_type]
        if 'IC' in model_results.columns:
            best_idx = model_results['IC'].idxmax()
            best = model_results.loc[best_idx]
            report.append(f"\n{model_type}:")
            report.append(f"  Best Phi Type: {best['phi_type']}")
            report.append(f"  Best Tau: {best['tau']}")
            report.append(f"  IC: {best.get('IC', 'N/A'):.4f}")
            report.append(f"  RankIC: {best.get('RankIC', 'N/A'):.4f}")
    
    report.append("\n" + "-" * 60)
    report.append("Best Results by Phi Type (based on IC)")
    report.append("-" * 60)
    
    for phi_type in successful['phi_type'].unique():
        phi_results = successful[successful['phi_type'] == phi_type]
        if 'IC' in phi_results.columns:
            best_idx = phi_results['IC'].idxmax()
            best = phi_results.loc[best_idx]
            report.append(f"\n{phi_type}:")
            report.append(f"  Best Model: {best['model_type']}")
            report.append(f"  Best Tau: {best['tau']}")
            report.append(f"  IC: {best.get('IC', 'N/A'):.4f}")
            report.append(f"  RankIC: {best.get('RankIC', 'N/A'):.4f}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
