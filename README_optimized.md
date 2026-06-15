# TSIL: Temporal Statistical Invariant Learning for Stock Prediction

TSIL provides a **predicate-level temporal knowledge interface** and a training framework that turns this interface into **statistical-invariant regularization** for stock prediction. Its purpose is to decouple heterogeneous temporal knowledge from specific architectures, feature pipelines, and hand-designed penalties, so knowledge such as timestamp, market cycle, asset phase, momentum, volatility, and return-distribution structure can be represented by unified computable **Temporal Predicates** and injected on the training side of arbitrary backbone predictors.

Given a mini-batch, each Temporal Predicate maps the input sequence and, when needed, calendar information to a finite-dimensional descriptor `Phi`. TSIL builds a structural similarity matrix `P = Phi_tilde Phi_tilde^T` and regularizes prediction residuals through a quadratic statistical-invariant term. The mechanism is **backbone-compatible, output-level, and training-only**: it changes the optimization objective, not the inference graph, inference-time inputs, or inference-time parameters.

This repository is the reference implementation accompanying the paper. It contains the canonical training/search entry points, the TSIL loss backend, the Qlib Alpha158 dataset loader, nine configured backbones, and experiment configs for CSI 300 and CSI 800.

---

## 1. Method overview

For a mini-batch of size `B`, let `y_hat` be the prediction vector, `y` the label vector, and `e = y_hat - y` the residual. The plain baseline minimizes per-sample MSE.

TSIL augments the task loss with a predicate-induced invariant penalty:

```text
L_TSIL = L_task + sum_s beta_s * || (1/B) * Phi_tilde_s^T (y_hat - y) ||_2^2
```

In the current implementation, one predicate is selected by `phi_type`, and the loss is realized as a quadratic residual form:

```text
M = tau_hat * I + tau * P
L = mean( e * (M e) )
```

where:

- `I` is the identity matrix and gives the ordinary pointwise MSE component.
- `P = Phi_tilde Phi_tilde^T` couples samples that share temporal or statistical structure.
- `tau_hat = sigmoid(model.alpha)` is learnable and initialized by `tau_hat_init`; `tau = 1 - tau_hat`.
- `phi_type = mse` bypasses `P` and uses plain MSE.
- `phi_type = ones` uses the all-ones control predicate `Phi_ones`.
- `phi_type = <temporal key>` uses a Temporal Predicate such as `phi_momentum` or `phi_volatility`.

The loss switch is controlled solely by `phi_type`; there is no separate `loss_type` field. `train.py` and `search.py` pass `phi_type` into both the model and the dataset. Timestamp features are built only for `phi_timestamp*`.

---

## 2. Repository layout

```text
TSIL_git/
├── train.py                        # single-run entry: one config, one phi_type, one or more seeds
├── search.py                       # Optuna/TPE hyper-parameter search by validation IC
├── basktesting.py                  # Top-K Dropout backtest + portfolio metrics
├── requirements.txt
├── LICENSE                         # GPL-3.0
├── src/
│   ├── model_backbone_phitp.py     # QniverseModel + TSIL loss, phi_type-driven
│   ├── temporal_predicates.py      # six paper Temporal Predicates
│   ├── custom_phi_constructors.py  # exploratory predicate constructors
│   ├── run_utils.py                # shared defaults + config overrides
│   ├── dataset_ori_onehot.py       # MTSDatasetH time-series loader, timestamp-aware
│   ├── timefeatures.py             # timestamp feature construction
│   └── models/                     # nine source-level backbones in this archive
│       ├── LSTM.py  GRU.py  TCN.py  Transformer.py  GAT.py
│       └── LSR_IGRU.py  master.py  MERA.py  StockMixer.py
├── configs/
│   ├── f158_csi300/                # CSI 300, 9 config_<model>.yaml files
│   └── f158_csi800/                # CSI 800, 9 config_<model>.yaml files
```

The configured backbones are:

```text
LSTM, GRU, TCN, Transformer, GAT, LSR_IGRU, MASTER, MERA, StockMixer
```

---

## 3. Installation

```bash
# Python 3.10-3.12 is recommended.
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Package | Version |
|---|---:|---|---:|
| numpy | 1.26.4 | pyqlib | 0.9.6 |
| pandas | 2.2.3 | scipy | 1.14.1 |
| scikit-learn | 1.5.2 | statsmodels | 0.14.2 |
| torch | 2.3.1 | torchvision | 0.18.1 |
| torchaudio | 2.3.1 | matplotlib | 3.10.0 |
| plotly | 5.24.1 | ruamel.yaml | 0.18.14 |
| tqdm | 4.66.5 | optuna | 4.9.0 |

A CUDA-capable GPU is recommended. The code falls back to CPU automatically, but full CSI 300/CSI 800 experiments are slow on CPU.

---

## 4. Data preparation

The pipeline uses Qlib China A-share data with the Alpha158 feature handler. Place the Qlib binary data at the path expected by the configs:

```text
TSIL_git/qlib_data/cn_data/        # provider_uri: "./qlib_data/cn_data"
```

You can obtain Qlib China data through Qlib's downloader or use your own updated dump:

```bash
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data/cn_data --region cn
```

For exact paper reproduction, the data must cover:

| Segment | Range |
|---|---|
| train | 2018-01-01 to 2022-12-31 |
| valid | 2023-01-01 to 2023-12-31 |
| test | 2024-01-01 to 2025-08-31 |

The label in the shipped configs is the two-day forward return:

```text
Ref($close, -2) / Ref($close, -1) - 1
```

Feature processing uses `RobustZScoreNorm` and `Fillna`; label processing uses `DropnaLabel` and `CSRankNorm`.

---

## 5. Running experiments

### 5.1 Single run with `train.py`

Use `--config_file` to choose the backbone and market. Use `--phi_type` to choose the loss mode.

```bash
# Original baseline: plain MSE
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type mse --market csi300 --seed 42

# TSIL with the momentum predicate
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type phi_momentum --tau_hat_init 2.0 --market csi300 --seed 42

# Phi_ones ablation
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type ones --market csi300 --seed 42
```

Available command-line overrides:

```text
--phi_type --base_model --tau_hat_init --lr --d_model --e_layers --batch_size --market --seed
```

`--seed` accepts one or more integers. The default is:

```text
42 2022 2023 2024 2025
```

Each market directory contains one config per backbone. The same config becomes Original, `Phi_ones`, or TSIL by changing `--phi_type`.

### 5.2 Hyper-parameter search with `search.py`

`search.py` searches learning rate, layers, hidden dimension, batch size, and `tau_hat_init` with Optuna/TPE. Selection is by validation IC returned from `QniverseModel.fit`; the test segment is not used to select hyper-parameters.

```bash
python search.py --config_file configs/f158_csi300/config_lstm.yaml \
  --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer \
  --phi_types phi_momentum --market csi300 --n_trials 50
```

Default search space:

| Parameter | Values |
|---|---|
| `lr` | 0.001, 0.0001, 0.00001 |
| `layers` | 1, 2, 3 |
| `d_model` | 64, 128, 256 |
| `bs` | 256, 512, 1024 |
| `tau_hat_init` | -4.0, -2.0, 0.0, 2.0, 4.0 |

Outputs include `trial_history.csv`, `optimization_report.txt`, Optuna visualizations, and a top-level `all_best_results.csv`.

---

## 6. Temporal predicates (`phi_type`)

The predicate that builds `P = Phi_tilde Phi_tilde^T` is selected by `model_config.phi_type` and dispatched in `WeightedMSELoss.forward`.

| `phi_type` | Meaning | Training form |
|---|---|---|
| `mse` | no predicate | Original plain-MSE baseline |
| `ones` | all-ones descriptor | control predicate / generic invariant ablation |
| `phi_timestamp` / `phi_timestamp_dim` / `phi_timestamp_sl` | calendar timestamp | TSIL |
| `phi_cycle` / `phi_cycle_onehot` | market cycle | TSIL |
| `phi_phase` | asset phase | TSIL |
| `phi_momentum` | multi-horizon momentum | TSIL |
| `phi_volatility` | multi-window volatility | TSIL |
| `phi_return_dist` | return-distribution moments | TSIL |

---

## 7. Backtest protocol

After prediction, signals are evaluated with a daily-rebalanced Top-K Dropout strategy through `basktesting.py` and Qlib's `PortAnaRecord`.

Default strategy settings:

| Setting | Value |
|---|---|
| `topk` | 50 |
| `n_drop` / `hold_thre` | 5 |
| rebalance | daily |
| buy commission | 0.05% |
| sell stamp duty | 0.15% |
| slippage | 0.01% |
| limit threshold | 0.095 |
| backtest window | 2024-01-01 to 2025-08-31 |

Reported ranking metrics include IC, ICIR, RankIC, and RankICIR. Portfolio metrics include annualized return, information ratio, and maximum drawdown with and without cost.

`backtest_report_*.txt` reads AR/IR/MMD from Qlib's `PortAnaRecord`. `backtest_result_seed*.csv` stores the self-contained Top-K-Dropout return series computed in `basktesting.py`. Use one metric source consistently when comparing runs.

---

## 8. Reproducing the paper comparisons

The paper-level comparison has three forms:

1. **Original**: `--phi_type mse`, plain MSE.
2. **Generic predicate control**: `--phi_type ones`, weighted loss using `Phi_ones`.
3. **TSIL-Enhanced**: `--phi_type <temporal key>`, weighted loss using a Temporal Predicate.

The loss-based reproduction in this release should be run by sweeping the same config across the eight modes below:

```text
mse
ones
phi_timestamp
phi_cycle
phi_phase
phi_momentum
phi_volatility
phi_return_dist
```

### 8.1 Train sweep: 9 backbones × 2 markets × 8 modes

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash -c '
for market in csi300 csi800; do
  for cfg in configs/f158_${market}/config_*.yaml; do
    for phi in mse ones phi_timestamp phi_cycle phi_phase phi_momentum phi_volatility phi_return_dist; do
      python train.py \
        --config_file "$cfg" \
        --market "$market" \
        --phi_type "$phi" \
        --seed 42
    done
  done
done
' > tsil_train_9x2x8_seed42.out 2>&1 &
```

This command checks the full train/backtest path for all configured backbones and all paper predicates. For a cheap smoke test, reduce `n_epochs` and `max_steps_per_epoch` in the YAML configs before running; these fields are not exposed as CLI flags in the current `train.py`.

### 8.2 Search sweep

```bash
CUDA_VISIBLE_DEVICES=0 nohup python search.py \
  --config_file configs/f158_csi300/config_lstm.yaml \
  --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer \
  --phi_types mse ones phi_timestamp phi_cycle phi_phase phi_momentum phi_volatility phi_return_dist \
  --tau_values -4.0 -2.0 0.0 2.0 4.0 \
  --market csi300 \
  --n_trials 50 \
  --early_stop 10 \
  --output_dir output_search_csi300 \
  > search_csi300_9x8.out 2>&1 &
```

Repeat with `--market csi800 --output_dir output_search_csi800` for CSI 800.

### 8.3 Notes on exact paper reproduction

- The current release contains Original and loss-based TSIL. The feature-concatenation ablation, where predicate descriptors are fed as extra input features, is not included.
- `train.py` uses the hyper-parameters in each YAML unless overridden by CLI flags.
- `search.py` injects model-specific parameters for MASTER, MERA, StockMixer, and LSR_IGRU.
- Early stopping and search selection currently use validation IC. If the paper protocol selects by validation RankICIR, modify the `valid_metrics["IC"]` criterion in `QniverseModel.fit` before exact reproduction.
- MASTER uses `use_gate: False` in the shipped configs because the Alpha158 handler does not provide the 63 additional market columns required by its market-feature gate.

---

## 9. Adding a new predicate

A predicate constructor maps a mini-batch to `(I, P)`:

```python
def get_<name>_I_P(batch_x, device=None, **kwargs):
    # batch_x: [batch_size, seq_len, feature_dim]
    # return I, P: [batch_size, batch_size]
```

Implementation contract:

1. Build a per-sample descriptor `phi` with shape `[batch_size, d]`.
2. Sanitize NaN/Inf values with `torch.nan_to_num`.
3. L2-normalize `phi`.
4. Return `I = torch.eye(batch_size, device=device)` and `P = phi @ phi.T`.

Wiring steps:

1. Register the constructor in `get_custom_I_P` inside `src/custom_phi_constructors.py`.
2. Add the key to `_CUSTOM_PHI_TYPES` in `src/model_backbone_phitp.py`.
3. Run with `--phi_type phi_<name>`.

---

## 10. Code provenance and acknowledgements

The overall experiment framework follows implementation patterns from:

- https://github.com/TongjiFinLab/FinTSB/tree/main/src
- https://github.com/microsoft/qlib/tree/main/examples/benchmarks

The model implementations are extracted or adapted from the corresponding open-source model repositories where applicable. When redistributing or extending this repository, check the licenses of the original projects as well as the GPL-3.0 license included here.

---

## License

Released under the GNU General Public License v3.0. See `LICENSE`.
