"""Smoke test for a single phi_type predicate (used to verify integration)."""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import ruamel.yaml as YAML
from smoke_test import shrink
from train_ts_trans import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--phi_type", default="phi_momentum")
    ap.add_argument("--config_file", default="configs/f158_ts_2025_csi300/config_lstm_ts_phi.yaml")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    with open(args.config_file) as f:
        config = YAML.YAML(typ="safe", pure=True).load(f)
    config = shrink(config)
    config["model_config"]["phi_type"] = args.phi_type
    config["logdir"] = "smoke_out/" + os.path.basename(config["task"]["model"]["kwargs"]["logdir"]) + "_" + args.phi_type
    config["task"]["model"]["kwargs"]["logdir"] = config["logdir"]
    os.makedirs(config["logdir"], exist_ok=True)
    main(args.seed, config, args.config_file)
    print("\n[SMOKE TEST PASSED] phi_type=%s ran end-to-end." % args.phi_type)
