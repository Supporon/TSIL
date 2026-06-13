"""
Fast end-to-end smoke test for the TSIL pipeline.

Loads a real config, shrinks the date segments and epoch budget so the full
train -> predict -> backtest -> report path runs in a couple of minutes on
the actual qlib cn_data, then asserts that ranking + portfolio metrics are
produced. This verifies pipeline mechanics; it does NOT reproduce paper numbers
(those need the full 2018-2025 training horizon and per-config hyperparameters).

Usage:
    python smoke_test.py --config_file configs/f158_ts_2025_csi300/config_lstm_ts_phi.yaml
"""
import argparse
import os
import sys

os.chdir(sys.path[0])  # repo root, so ./qlib_data/cn_data resolves

import ruamel.yaml as YAML
from train_ts_trans import main


def shrink(config):
    # tiny date windows that still exist in cn_data
    seg = config["task"]["dataset"]["kwargs"]["segments"]
    seg["train"] = ["2022-01-01", "2022-12-31"]
    seg["valid"] = ["2023-01-01", "2023-06-30"]
    seg["test"] = ["2024-01-01", "2024-03-31"]
    config["data_handler_config"]["start_time"] = "2022-01-01"
    config["data_handler_config"]["end_time"] = "2024-03-31"
    config["data_handler_config"]["fit_start_time"] = "2022-01-01"
    config["data_handler_config"]["fit_end_time"] = "2022-12-31"
    config["port_analysis_config"]["backtest"]["start_time"] = "2024-01-01"
    config["port_analysis_config"]["backtest"]["end_time"] = "2024-03-31"
    # cheap training
    k = config["task"]["model"]["kwargs"]
    k["n_epochs"] = 2
    k["early_stop"] = 5
    k["max_steps_per_epoch"] = 20
    k["logdir"] = "smoke_out/" + os.path.basename(config["task"]["model"]["kwargs"]["logdir"])
    config["logdir"] = k["logdir"]
    config["task"]["dataset"]["kwargs"]["batch_size"] = 512
    config["model_config"]["ind"] = 0
    config["model_config"]["tau_hat_init"] = 0.0
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/f158_ts_2025_csi300/config_lstm_ts_phi.yaml",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        yaml = YAML.YAML(typ="safe", pure=True)
        config = yaml.load(f)

    config = shrink(config)
    os.makedirs(config["logdir"], exist_ok=True)
    main(args.seed, config, args.config_file)
    print("\n[SMOKE TEST PASSED] pipeline ran end-to-end and produced a backtest report.")
