"""Single-run training entry point for TSIL.

Runs one backbone on one market through train -> predict -> backtest -> report.

Usage
-----
All parameters from the YAML:
    python train.py --config_file configs/f158_ts_2025_csi300/config_lstm_ts_phi.yaml

Override YAML fields from the command line (any omitted flag keeps the YAML value):
    python train.py --config_file configs/f158_ts_2025_csi300/config_lstm_ts_phi.yaml \
        --loss_type MSE_with_weak --phi_type phi_momentum --base_model LSTM \
        --tau_hat_init 2.0 --seed 42

Modes (selected by --loss_type / --phi_type):
    ori : --loss_type MSELoss
    phi : --loss_type MSE_with_weak --phi_type <key>   (e.g. phi_momentum)

--seed accepts multiple values; each is run in turn. Default seeds: 42 2022 2023 2024 2025.
"""
import argparse
import os
import sys

os.chdir(sys.path[0])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qlib
import ruamel.yaml as YAML
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from basktesting import calculate_portfolio_metrics, top_k_drop_strategy
from src.run_utils import DEFAULT_SEEDS, apply_overrides, build_report


def get_experiment_metrics(file_name, recorder=None):
    """Read the portfolio analysis metrics for the report.

    Portfolio metrics are written to MLflow asynchronously (`async_log`), so
    globbing the metrics/ directory at report time can race the flush and miss
    keys. The port_analysis_1day.pkl artifact is saved synchronously by
    PortAnaRecord, so we read it directly and rebuild the flat
    "1day.<group>.<metric>" keys the report expects.
    """
    from pathlib import Path

    if recorder is not None:
        try:
            pa = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
            flat = {f"1day.{g}.{m}": float(v) for (g, m), v in pa["risk"].items()}
            if flat:
                return flat
        except Exception as e:
            print(f"port_analysis pkl unavailable, falling back to metric files: {e}")

    folder = Path(file_name)
    file_dict = {}
    for file_path in folder.glob("*"):
        if file_path.is_file():
            try:
                v = file_path.read_text(encoding="utf-8")
                file_dict[file_path.name] = float(v.split(" ")[1])
            except Exception as e:
                print(f"skip {file_path.name}: {e}")
    return file_dict


def main(seed, config, config_file):
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )

    exp_settings = config.get("experiment_settings", {
        "exp_path": config["logdir"] + "/experiments",
        "experiment_name": "test_Experiment",
        "recorder_name": "test_Recorder",
    })
    exp_path = exp_settings["exp_path"]
    os.makedirs(exp_path, exist_ok=True)
    R.set_uri(exp_path)

    with R.start(
        experiment_name=exp_settings["experiment_name"],
        recorder_name=exp_settings["recorder_name"],
    ):
        logdir = config["task"]["model"]["kwargs"]["logdir"]
        model_type = config["task"]["model"]["kwargs"]["model_type"]

        dataset = init_instance_by_config(config["task"]["dataset"])
        model = init_instance_by_config(config["task"]["model"])

        # All shipped backbones are neural QniverseModel: fit() returns (preds, metrics).
        # The tree-model branch is kept only for completeness.
        if model_type not in ["XGBoost", "LightGBM"]:
            preds, metrics = model.fit(dataset)
        else:
            model.fit(dataset)
            preds = model.predict(dataset)
            preds = preds.to_frame(name="score")
            ori_index = preds.index
            preds = preds.reset_index(drop=True)
            label = dataset.prepare("test", col_set=["label"]).data_arr[:-1]
            label = pd.Series(label.reshape(-1, )).to_frame(name="label")
            preds = pd.concat([preds, label], axis=1).dropna().reset_index(drop=True)
            preds.set_index(ori_index, inplace=True)
            scores = preds["score"].values
            labels = preds["label"].values
            metrics = {"InfT": -1, "MSE": np.mean((labels - scores) ** 2),
                       "MAE": np.mean(np.abs(labels - scores))}

        per_df = preds.reset_index()
        per_df = per_df.sort_values(["datetime", "instrument"])
        per_df["group_id"] = per_df.groupby(["datetime", "instrument"]).cumcount()

        r_metrics = {"IC": [], "ICIR": [], "RankIC": [], "RankICIR": []}
        strategy_config = config["strategy_config"]

        sub_df = per_df[per_df["group_id"] == 0]
        sub_df = sub_df.set_index(["datetime", "instrument"])
        sub_df = sub_df[["score", "label"]]

        # Ranking metrics
        ic = sub_df.groupby(level=0).apply(
            lambda group: group["score"].corr(group["label"])).to_numpy()
        icir = ic.mean() / ic.std()
        ic = ic.mean()
        rankic = sub_df.groupby(level=0).apply(
            lambda group: group["score"].corr(group["label"], method="spearman")).to_numpy()
        rankicir = rankic.mean() / rankic.std()
        rankic = rankic.mean()
        r_metrics["IC"].append(ic)
        r_metrics["ICIR"].append(icir)
        r_metrics["RankIC"].append(rankic)
        r_metrics["RankICIR"].append(rankicir)

        r_df = [top_k_drop_strategy(sub_df, top_k=strategy_config["topk"],
                                    n=strategy_config["hold_thre"],
                                    commission_rate=strategy_config["commission_rate"])]
        r_df = pd.concat(r_df, ignore_index=True)
        portfolio_metrics, df = calculate_portfolio_metrics(r_df)
        metrics.update(portfolio_metrics)
        metrics.update({key: np.mean(values) for key, values in r_metrics.items()})

        # Qlib signal / portfolio analysis records (source of the report's AR/IR).
        port_analysis_config = config["port_analysis_config"]
        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate(label=preds[["label"]])
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        df.to_csv(f"./{logdir}/backtest_result_seed{seed}.csv")
        analysis_dict = get_experiment_metrics(
            recorder.uri + "/" + recorder.experiment_id + "/" + recorder.id + "/metrics/",
            recorder)

        report = build_report(metrics, analysis_dict)
        print(report)
        tag = f"{config['model_config'].get('phi_type', 'mse')}_tau{config['model_config']['ind']}_seed{seed}"
        with open(f"./{logdir}/backtest_report_{tag}.txt", "w") as f:
            f.write(report)
    return metrics


def build_argparser():
    p = argparse.ArgumentParser(allow_abbrev=False, description="TSIL single-run training")
    p.add_argument("--config_file", type=str, required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="one or more random seeds (default: 42 2022 2023 2024 2025)")
    # YAML overrides (None => keep the value in the config file)
    p.add_argument("--loss_type", type=str, default=None,
                   choices=["MSELoss", "MSE_with_weak", "RankMSELoss"],
                   help="MSELoss = Original, MSE_with_weak = TSIL")
    p.add_argument("--phi_type", type=str, default=None, help="predicate key, e.g. phi_momentum")
    p.add_argument("--base_model", type=str, default=None, help="base RNN for LSR_IGRU")
    p.add_argument("--tau_hat_init", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--e_layers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--market", type=str, default=None, choices=["csi300", "csi500", "csi800"])
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    with open(args.config_file) as f:
        config = YAML.YAML(typ="safe", pure=True).load(f)

    config = apply_overrides(config, args)

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        print("=" * 60)
        print(f"[run] config={args.config_file} seed={seed} "
              f"loss={config['model_config']['loss_type']} "
              f"phi={config['model_config'].get('phi_type')}")
        print("=" * 60)
        main(seed, config, args.config_file)
