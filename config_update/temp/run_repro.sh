#!/usr/bin/env bash
# Reproduction driver for the TSIL reference implementation.
#
# Runs the canonical Original-vs-TSIL comparison for the domain-agnostic
# backbones on a chosen market universe, using the shipped per-model configs.
#
# Usage:
#   bash run_repro.sh csi300            # all 7 backbones, ori + phi, on CSI300
#   bash run_repro.sh csi800 lstm gru   # only LSTM and GRU on CSI800
#
# Notes:
#   * Requires ./qlib_data/cn_data to resolve (see README §4). Symlink your
#     qlib cn_data there if needed.
#   * Uses the python on PATH; activate the project env first
#     (e.g. `conda activate <env>` with the deps in requirements.txt).
#   * train_ts_trans.py applies a small fixed set of run-time overrides in its
#     __main__ block (d_model=32, e_layers=2, lr=1e-3, batch_size=1024,
#     tau_hat_init=2). To reproduce the paper's per-model hyperparameters
#     instead, edit/remove that block — see README §8.
set -euo pipefail

MARKET="${1:-csi300}"; shift || true
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(lstm gru tcn transformer gat gcn patchtst)
fi

CFG_DIR="configs/f158_ts_2025_${MARKET}"
PY="${PYTHON:-python}"

for m in "${MODELS[@]}"; do
  for variant in ori phi; do
    cfg="${CFG_DIR}/config_${m}_ts_${variant}.yaml"
    if [ ! -f "$cfg" ]; then
      echo "[skip] missing config: $cfg"
      continue
    fi
    echo "=================================================================="
    echo "[run] model=${m} variant=${variant} market=${MARKET}"
    echo "      config=${cfg}"
    echo "=================================================================="
    "$PY" train_ts_trans.py --config_file "$cfg"
  done
done

echo "All runs finished. See each config's logdir for backtest_report_tau*.txt."
