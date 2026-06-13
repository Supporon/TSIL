import argparse

import qlib
import logging
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
import pandas as pd
import numpy as np
import ruamel.yaml as YAML
from tqdm import tqdm
from qlib.utils import init_instance_by_config
import warnings
warnings.filterwarnings('ignore')
import os,sys
os.chdir(sys.path[0])
from pathlib import Path
from basktesting import calculate_portfolio_metrics, top_k_drop_strategy
# 屏蔽标准错误输出
# sys.stderr = open(os.devnull, 'w')

# python train.py --config_file configs/config_patchtst.yaml

def get_experiment_metrics(file_name, recorder=None):
    # Portfolio metrics are written to MLflow asynchronously (`async_log`), so
    # globbing the metrics/ directory (or calling list_metrics()) at report time
    # can race the flush and miss keys. The port_analysis_1day.pkl artifact is
    # saved synchronously by PortAnaRecord, so we read it directly and rebuild
    # the flat "1day.<group>.<metric>" keys the report expects.
    if recorder is not None:
        try:
            pa = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
            flat = {f"1day.{g}.{m}": float(v) for (g, m), v in pa["risk"].items()}
            if flat:
                return flat
        except Exception as e:
            print(f"port_analysis pkl unavailable, falling back to metric files: {e}")

    folder = Path(file_name)  # 换成你的路径
    file_dict = {}

    for file_path in folder.glob('*'):  # 如需递归：folder.rglob('*')
        if file_path.is_file():  # 只处理文件
            try:
                v = file_path.read_text(encoding='utf-8')
                file_dict[file_path.name] = float(v.split(' ')[1])
            except Exception as e:
                print(f"跳过 {file_path.name}: {e}")

    return file_dict

def main(seed, config, config_file="configs/config_patchtst.yaml"):
# def main(seed, config_file="configs/config_patchtst.yaml"):

    # with open(config_file) as f:
    #     yaml = YAML.YAML(typ='safe', pure=True)
    #     config = yaml.load(f)

    print('---'*10)
    print(config)
    print('---' * 10)

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )

    # 确保有实验设置
    exp_settings = config.get("experiment_settings", {
        "exp_path": config['logdir'] + "/experiments",
        "experiment_name": "test_Experiment",
        "recorder_name": "test_Recorder"
    })

    # 确保实验路径存在
    exp_path = exp_settings["exp_path"]
    os.makedirs(exp_path, exist_ok=True)
    R.set_uri(exp_path)

    # 启动实验记录器
    with R.start(
        experiment_name=exp_settings["experiment_name"],
        recorder_name=exp_settings["recorder_name"]
    ):
        # recorder = R.get_recorder()
        #
        # params = {
        #     "seed": seed,
        #     "config_file": config_file,
        #     "model_type": config["task"]["model"]["kwargs"].get("model_type", "unknown"),
        # }
        # recorder.log_params(**params)
        # 记录配置和参数
        # recorder.log_params(dict(seed=seed, config_file=config_file,model_type=config["task"]["model"]["kwargs"].get("model_type", "unknown")))

        logdir = config["task"]["model"]["kwargs"]["logdir"]
        model_type = config["task"]["model"]["kwargs"]["model_type"]

        dataset = init_instance_by_config(config["task"]["dataset"])
        model = init_instance_by_config(config["task"]["model"])


        if model_type not in ['XGBoost', 'LightGBM'] and 'ts' in config_file:
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
                label = pd.Series(label.reshape(-1,)).to_frame(name='label')
            # preds = pd.concat([preds, dataset.prepare("test")[['label']]], axis=1, join='inner')
            preds = pd.concat([preds, label], axis=1).dropna().reset_index(drop=True)
            preds.set_index(ori_index, inplace=True)
            scores = preds['score'].values
            labels = preds['label'].values
            metrics = {"InfT": -1, "MSE": np.mean((labels - scores) ** 2), "MAE": np.mean(np.abs(labels - scores))}
            print(preds)


        per_df = preds.reset_index()
        per_df = per_df.sort_values(['datetime', 'instrument'])
        per_df['group_id'] = per_df.groupby(['datetime', 'instrument']).cumcount()
        # assert (per_df.groupby(['datetime', 'instrument'])['group_id'].max() == 19).all(), "Error"

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
        rankic = sub_df.groupby(level=0).apply(lambda group: group['score'].corr(group['label'], method='spearman')).to_numpy()
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

        # 记录所有指标到实验系统
        # recorder.log_metrics(metrics)
        # prediction
        port_analysis_config = config["port_analysis_config"]
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate(label = preds[['label']])

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        df.to_csv(f'./{logdir}/backtest_result.csv')
        analysis_dict = get_experiment_metrics(recorder.uri+'/'+recorder.experiment_id+'/'+recorder.id+'/metrics/', recorder)

        report = f"""
            **********************************
                    Bachtest Report
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
            w/o cost AR:    {analysis_dict['1day.excess_return_without_cost.annualized_return']:.4f}
            w/o cost ret mean:    {analysis_dict['1day.excess_return_without_cost.mean']:.4f}
            w/o cost ret std:    {analysis_dict['1day.excess_return_without_cost.std']:.4f}
            w/o cost IR:    {analysis_dict['1day.excess_return_without_cost.information_ratio']:.4f}
            w/o cost MMD:    {analysis_dict['1day.excess_return_without_cost.max_drawdown']:.4f}
            with cost AR:    {analysis_dict['1day.excess_return_with_cost.annualized_return']:.4f}
            with cost ret mean:    {analysis_dict['1day.excess_return_with_cost.mean']:.4f}
            with cost ret std:    {analysis_dict['1day.excess_return_with_cost.std']:.4f}
            with cost IR:    {analysis_dict['1day.excess_return_with_cost.information_ratio']:.4f}
            with cost MMD:    {analysis_dict['1day.excess_return_with_cost.max_drawdown']:.4f}
            ***************************************
            """
        print(report)

        # ARR: {metrics['ARR']: .4f}
        # AVol: {metrics['AVol']: .4f}
        # MDD: {metrics['MDD']: .4f}
        # ASR: {metrics['ASR']: .4f}
        # IR: {metrics['IR']: .4f}

        # with open(f'./{logdir}/backtest_report.txt', 'w') as file:
        #     file.write(report)
        with open(f'./{logdir}/backtest_report_tau'+str(config["model_config"]["ind"])+'.txt', 'w') as file:
            file.write(report)

        # # analyze graphs
        # pred_df = recorder.load_object("pred.pkl")
        # report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        # positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        # analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        #
        # print(analysis_position.report_graph(report_normal_df))
        # print(analysis_position.risk_analysis_graph(analysis_df, report_normal_df))
        # label_df = dataset.prepare("test", col_set="label")
        # label_df.columns = ["label"]
        # pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
        # print(analysis_position.score_ic_graph(pred_label))
        # print(analysis_model.model_performance_graph(pred_label))

if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/f158_ts/config_lstm_ts_phi_param.yaml", help="config file") #
    args = parser.parse_args()
    # config_patchtst_test
    # config_tcn_ts_trans
    with open(args.config_file) as f:
        yaml = YAML.YAML(typ='safe', pure=True)
        config = yaml.load(f)

    config["model_config"]["ind"] = 0
    config["model_config"]["tau_hat_init"] = 2
    config["task"]["model"]["kwargs"]["lr"] = 0.001
    config["task"]["dataset"]["kwargs"]["batch_size"] = 1024
    config["model_config"]["e_layers"] = 2
    config["model_config"]["d_model"] = 32
    #
    main(args.seed, config, args.config_file)
    # main(**vars(args))

    # config_patchtst_test
    # config_patchtst_ts_trans
    # config_tcn_ts_trans
    # config_lstm_ts_trans