# TSIL: Temporal Statistical Invariant Learning for Stock Prediction

TSIL provides a **predicate-level temporal knowledge interface** and a training framework that turns this interface into **statistical-invariant regularization** for stock prediction. Its purpose is to decouple heterogeneous temporal knowledge from specific architectures, feature pipelines, and hand-designed penalties, so knowledge such as timestamp, market cycle, asset phase, momentum, volatility, and return-distribution structure can be represented by unified computable **Temporal Predicates** and injected on the training side of arbitrary backbone predictors.

Given a mini-batch, each Temporal Predicate maps causal raw signals and, when needed, calendar information to a finite-dimensional descriptor `Phi`. The structural similarity matrix `P = Phi_tilde Phi_tilde^T` defines the statistical-invariant penalty conceptually; training computes descriptor-residual projections directly without materializing `P`. The mechanism is **backbone-compatible, output-level, and training-only**: it changes the optimization objective, not the inference graph, inference-time inputs, or inference-time parameters.

This repository is the reference implementation accompanying the paper. It contains the canonical training/search entry points, the TSIL loss backend, the Qlib Alpha158 dataset loader, nine configured backbones, and experiment configs for CSI 300 and CSI 800.

---

## 1. Method overview

For a mini-batch of size `B`, let `y_hat` be the prediction vector, `y` the label vector, and `e = y_hat - y` the residual. The plain baseline minimizes per-sample MSE.

TSIL augments the task loss with a predicate-induced invariant penalty:

```text
L_TSIL = L_task + sum_s beta_s * || (1/B) * Phi_tilde_s^T (y_hat - y) ||_2^2
```

### Objective and descriptors

For residual `e = prediction - raw_next_day_return` and a batch of `B` labeled
stock-date samples:

```python
loss = mean(e ** 2)
for name, phi in descriptors.items():
    normalized = phi.detach() / (phi.detach().norm() + 1e-8)
    loss += beta[name] * ((normalized.T @ e) / B).square().sum()
```

`beta = sigmoid(penalty_alpha)` is fixed during a run; alpha is selected from
`[-4,-2,0,2,4]` using validation daily IC. The backbone has no trainable loss
coefficient. Training never materializes a `B x B` predicate Gram matrix.

| Canonical key | Causal raw descriptor | Default dimension |
|---|---|---:|
| `phi_timestamp` | Fixed weekday/month OneHot | 19 |
| `phi_cycle` | Market-return time FFT, non-DC period, global-day sin/cos | 2 |
| `phi_phase` | Asset-return time FFT, global trading-day modulo phase | 20 |
| `phi_momentum` | `log(close_t/close_(t-k))`, k=5,10,20 | 3 |
| `phi_volatility` | Population std of realized returns, w=10,20,40 | 3 |
| `phi_return_dist` | 20-day mean, population std, skew, excess kurtosis | 4 |
| `ones` | Constant control; excluded from main-table predicate selection | 1 |

The approved windows are in `src/temporal_predicates.py`. Raw Qlib `$close` data
are loaded independently of the 20-day Alpha158 input; 40 return observations
require 41 price observations. No standardized factor is treated as a price.
Historical membership masks determine the equal-weight market return. Calendar
origin, timestamp categories and market aggregation are explicit implementation
conventions because the manuscript does not fix them completely.

Original, Feature and TSIL use the same causal history eligibility. Feature
repeats the same raw descriptor along the input time axis; TSIL uses it only in
the training loss. Full consecutive feature windows are required. Labels are
`Ref($close,-1)/$close-1`, unranked; Alpha158 inputs use daily CSZScoreNorm.
Labels extending beyond a split endpoint are purged. Missing future labels
never remove stocks before a relationship-model forward pass: only supervised
losses and metrics mask those labels afterward.

---

## 2. Repository layout

```text
TSIL/
├── train.py                        # fixed runs and selected-checkpoint evaluation
├── search.py                       # network search and fixed-beta selection by validation IC
├── basktesting.py                  # historical backtest implementation
├── requirements.txt
├── LICENSE                         # GPL-3.0
├── src/
│   ├── model_backbone_phitp.py     # QniverseModel and training/evaluation loop
│   ├── tsil_loss.py                # fixed-coefficient additive TSIL objective
│   ├── temporal_predicates.py      # six causal raw Temporal Predicates
│   ├── predicate_data.py           # raw-price descriptor cache and common eligibility
│   ├── custom_phi_constructors.py  # historical exploratory constructors
│   ├── run_utils.py                # shared defaults and config overrides
│   ├── dataset_ori_onehot.py       # MTSDatasetH time-series loader
│   ├── timefeatures.py             # historical timestamp implementation
│   └── models/
│       ├── LSTM.py  GRU.py  TCN.py  Transformer.py  GAT.py
│       └── LSR_IGRU.py  master.py  MERA.py  StockMixer.py
├── scripts/
│   ├── compare_baseline.py
│   └── verify_synthetic_pipeline.py
└── configs/
    ├── f158_csi300/                # CSI 300, 9 config_<model>.yaml files
    └── f158_csi800/                # CSI 800, 9 config_<model>.yaml files
```

The configured backbones are:

```text
LSTM, GRU, TCN, Transformer, GAT, LSR_IGRU, MASTER, MERA, StockMixer
```

---

## 3. Installation

Use Python 3.10–3.12 and a fresh environment:

```bash
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
| einops | 0.8.2 | joblib | 1.4.2 |

A CUDA-capable GPU is recommended. The code falls back to CPU automatically, but full CSI 300/CSI 800 experiments are slow on CPU. Joblib is pinned to 1.4.2 for compatibility with Qlib 0.9.6's `ParallelExt` backend API.

---

## 4. Data preparation

The pipeline uses Qlib China A-share data with the Alpha158 feature handler. Place the Qlib binary data at the path expected by the configs:

```text
TSIL/qlib_data/cn_data/        # provider_uri: "./qlib_data/cn_data"
```

Supply a Qlib provider at `qlib_data/cn_data`, or edit `qlib_init.provider_uri`.
It must contain the historical CSI300/CSI800 membership, trading calendar,
OHLCV, close, factor and benchmark series. No real-market dataset is bundled here.

| Segment | Range |
|---|---|
| train | 2018-01-01 to 2022-12-31 |
| valid | 2023-01-01 to 2023-12-31 |
| test | 2024-01-01 to 2025-08-31 |

Preserve the provider's earlier calendar/history so global phases and long windows are reproducible. The label in the shipped configs is the unranked next-day return:

```text
Ref($close, -1) / $close - 1
```

Feature processing uses daily `CSZScoreNorm` and `Fillna`; supervised label processing uses `DropnaLabel`. Eligibility for a relationship-model forward pass does not depend on future-label availability.

---

## 5. Running experiments

### 5.1 Single run with `train.py`

Use `--config_file` to choose the backbone and market, `--phi_type` to select the predicate, and `--injection_mode` to select how it is used.

```bash
# Original baseline: plain MSE
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type mse --injection_mode original --market csi300 --seed 42

# TSIL with the momentum predicate and fixed beta = sigmoid(2)
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type phi_momentum --injection_mode tsil --penalty_alpha 2 \
  --market csi300 --seed 42

# Phi_ones control
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type ones --injection_mode tsil --market csi300 --seed 42

# Feature concatenation with the same raw descriptor
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
  --phi_type phi_momentum --injection_mode feature --market csi300 --seed 42
```

Available command-line overrides:

```text
--phi_type --injection_mode --penalty_alpha --base_model --lr --d_model --e_layers --batch_size --market --seed
```

`--seed` accepts one or more integers. Without an override, the configured seed is used; the fallback seed set is `42 2022 2023 2024 2025`. Each run has a separate output path containing its seed and configuration digest.

### 5.2 Hyper-parameter search with `search.py`

`search.py` first selects the network using Original/seed 2025 and validation daily IC, then freezes that network while selecting each seed/predicate's fixed beta. Search does not prepare or evaluate the test split.

```bash
python search.py --config_file configs/f158_csi300/config_lstm.yaml --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer --market csi300 --injection_mode tsil --n_trials 50 --output_dir output_search_csi300_tsil
```

Defaults include Original (`mse`), the ones control, all six temporal predicates
and seeds `42 2022 2023 2024 2025`. Network selection uses Original/seed 2025,
then freezes the network while selecting each seed/predicate's fixed beta.
For GAT, MASTER and LSR-IGRU, the approved batch search is **2/4/8 complete
trading days**; each day is forwarded independently before joining residuals.
Other models search 256/512/1024 stock-date samples. Different runs have separate
output paths containing the actual seed and a configuration digest.

Default search space:

| Parameter | Values |
|---|---|
| `lr` | 0.001, 0.0001, 0.00001 |
| `layers` | 1, 2, 3 |
| `d_model` | 64, 128, 256 |
| `bs` (non-relationship models) | 256, 512, 1024 |
| `days_per_batch` (GAT, MASTER, LSR-IGRU) | 2, 4, 8 |
| `penalty_alpha` (after freezing the network) | -4.0, -2.0, 0.0, 2.0, 4.0 |

Feature must reuse those exact network choices:

```bash
python search.py --config_file configs/f158_csi300/config_lstm.yaml --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer --market csi300 --injection_mode feature --frozen_network_manifest output_search_csi300_tsil/selection_manifest.json --output_dir output_search_csi300_feature
```

Repeat with `csi800` and separate output directories. A manifest records all
selected checkpoints/configs, candidate validation ICs, and a common main-table
predicate selected by mean validation daily IC over the five seeds (first wins
ties). This aggregation rule is a disclosed convention. `ones` is never a
candidate for that main-table choice. Search does not prepare or evaluate test.

Outputs include per-model `network_candidates.json`, `predicate_candidates.json`, and a top-level `selection_manifest.json`.

### 5.3 Final evaluation

For each selected run in the manifest, use its saved paths:

```bash
python train.py --config_file PATH_TO_SELECTED/config.yaml --checkpoint PATH_TO_SELECTED/model.bin
```

The checkpoint path evaluates only test; it does not refit or reselect weights.
Use `selected_by_predicate` for complete supplementary comparisons and
`selected_runs` for the common main-table predicate. Aggregate each model/form/
predicate's five final test runs as mean and standard deviation; never choose
which result to report by looking at test scores.

Outputs include checkpoint, prediction pickle, resolved YAML, daily executed
portfolio returns, actual holdings, JSON metrics and a text report. IC/RankIC
are calculated across stocks per day, then averaged over days; their IR uses
population daily std (`ddof=0`). Undefined correlations/IR are reported as
undefined, not replaced with zero. Inference is synchronized and reported in
**ms/sample**, excluding data loading and descriptor-cache construction.

---

## 6. Temporal predicates (`phi_type`)

`model_config.phi_type` selects the causal raw descriptor. `StatisticalInvariantLoss` in `src/tsil_loss.py` applies the fixed-coefficient TSIL objective when `injection_mode=tsil`.

| `phi_type` | Meaning | Training form |
|---|---|---|
| `mse` | no predicate | Original plain-MSE baseline |
| `ones` | all-ones descriptor | control predicate / generic invariant ablation |
| `phi_timestamp` | calendar timestamp | TSIL or Feature |
| `phi_cycle` | market cycle | TSIL or Feature |
| `phi_phase` | asset phase | TSIL or Feature |
| `phi_momentum` | multi-horizon momentum | TSIL or Feature |
| `phi_volatility` | multi-window volatility | TSIL or Feature |
| `phi_return_dist` | return-distribution moments | TSIL or Feature |

Use the canonical keys above; legacy aliases such as `phi_timestamp_dim`, `phi_timestamp_sl`, and `phi_cycle_onehot` are not supported by the repaired entry points.

---

## 7. Backtest protocol

After prediction, `train.py` evaluates signals with Qlib's executed Top-K Dropout trades.

| Setting | Value |
|---|---|
| `topk` | 50 |
| `n_drop` | 5 |
| rebalance | daily |
| buy commission | 0.05% |
| sell stamp duty | 0.15% |
| close-price slippage | +0.01% buy / -0.01% sell |
| trade unit / minimum fee | 100 shares / 5 |
| limit threshold | 0.095 |
| backtest window | 2024-01-01 to 2025-08-31 |

Qlib executes scores at the next trading step. Backtest settings are Top-50,
Drop-5, buy/sell fees 0.05%/0.15%, close-price slippage +0.01%/-0.01%, trade unit
100 and minimum fee 5. Factor data are required for Qlib lot conversion. The
retained `limit_threshold=0.095` is Qlib's uniform limit proxy; it is not a
verified board/ST-specific Chinese price-limit calendar. Suspended stocks are
filtered by the exchange. Reports distinguish absolute strategy returns from
benchmark-relative excess returns, with explicitly chosen arithmetic 252-day
annualization, daily std `ddof=1` and additive drawdown. A score's prediction
target t→t+1 is not substituted for the return of an executable trade at t+1.

`backtest_daily.csv` stores executed daily returns; `backtest_report.txt` and `metrics.json` report ranking and portfolio metrics from that execution path. The historical `basktesting.py` is not used by the repaired entry points.

---

## 8. Reproducing the paper comparisons

The paper-level comparison has three forms:

1. **Original**: `--phi_type mse --injection_mode original`, plain MSE.
2. **Feature**: `--phi_type <temporal key> --injection_mode feature`, the raw descriptor repeated along the input time axis and concatenated to the backbone features.
3. **TSIL-Enhanced**: `--phi_type <temporal key> --injection_mode tsil`, the descriptor used only in the training loss.

`--phi_type ones --injection_mode tsil` is a separate generic invariant control, excluded from main-table predicate selection.

### 8.1 Fixed-run sweep: 9 backbones × 2 markets × 8 modes

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash -c '
for market in csi300 csi800; do
  for cfg in configs/f158_${market}/config_*.yaml; do
    for phi in mse ones phi_timestamp phi_cycle phi_phase phi_momentum phi_volatility phi_return_dist; do
      mode=tsil
      if [ "$phi" = mse ]; then mode=original; fi
      python train.py \
        --config_file "$cfg" \
        --market "$market" \
        --phi_type "$phi" \
        --injection_mode "$mode" \
        --seed 42
    done
  done
done
' > tsil_train_9x2x8_seed42.out 2>&1 &
```

This is a fixed-configuration execution sweep. For paper comparisons, use the validation-selection workflow below. For a cheap smoke run, reduce `n_epochs` and `max_steps_per_epoch` in the YAML configs; these fields are not exposed as CLI flags.

### 8.2 Search sweep

```bash
CUDA_VISIBLE_DEVICES=0 nohup python search.py \
  --config_file configs/f158_csi300/config_lstm.yaml \
  --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer \
  --phi_types mse ones phi_timestamp phi_cycle phi_phase phi_momentum phi_volatility phi_return_dist \
  --alpha_values -4.0 -2.0 0.0 2.0 4.0 \
  --injection_mode tsil \
  --market csi300 \
  --n_trials 50 \
  --output_dir output_search_csi300_tsil \
  > search_csi300_9x8.out 2>&1 &
```

Repeat with `--market csi800 --output_dir output_search_csi800_tsil` for CSI 800. For Feature comparisons, reuse the resulting network manifest with `--injection_mode feature --frozen_network_manifest PATH_TO_SELECTION_MANIFEST` and a separate output directory. Evaluate the saved selected checkpoints as described in Section 5.3.

### 8.3 Notes on exact paper reproduction

- Original, Feature and TSIL share the same causal history eligibility and frozen network choices.
- `train.py` uses the hyper-parameters in each YAML unless overridden by CLI flags.
- `search.py` loads the released configuration for each backbone and applies the selected network parameters.
- Early stopping and search selection use validation daily IC; test results never select hyper-parameters, predicates or seeds.
- Aggregate the five selected final test runs as mean and standard deviation.
- MASTER uses `use_gate: False` in the shipped configs because the Alpha158 handler does not provide the 63 additional market columns required by its market-feature gate.
- The historical paper tables have not been reproduced by this code update. Synthetic integration checks establish software behavior, not real-market performance.

---

## 9. Adding a new predicate

A predicate constructor returns a raw descriptor with shape `[B, d_s]` from causal raw prices, realized returns, or calendar information. Training-side normalization and residual projection belong to `StatisticalInvariantLoss`; a constructor does not build `(I, P)` matrices.

Implementation contract:

1. Use observations available through prediction day `t`; never use the future target or reinterpret normalized Alpha158 factors as raw prices.
2. Return a finite raw descriptor with one row per sample and a known dimension.
3. Define the required lookback and preserve common causal history eligibility across Original, Feature and TSIL.
4. Let the loss detach and Frobenius-normalize the descriptor before projecting residuals.

Wiring steps:

1. Register the canonical key in `TEMPORAL_PHI_TYPES` and its dimension in `descriptor_dimensions`, in `src/temporal_predicates.py`.
2. Implement its raw descriptor in `get_temporal_phi` and supply its inputs/history through `src/predicate_data.py`.
3. Run with `--phi_type phi_<name>` and the appropriate `--injection_mode`.

Legacy `src/custom_phi_constructors.py`, `src/timefeatures.py` and
`basktesting.py` remain available as historical code. The repaired entry points
do not use them for paper predicates, timestamp encoding or portfolio returns.
Old `tau_hat_init`, legacy predicate aliases and checkpoints containing
trainable `alpha` are not interchangeable with this repaired method: retrain.

---

## 10. Code provenance and acknowledgements

The overall experiment framework follows implementation patterns from:

- [FinTSB](https://github.com/TongjiFinLab/FinTSB/tree/main/src)
- [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)

The model implementations are extracted or adapted from the corresponding open-source model repositories where applicable. When redistributing or extending this repository, check the licenses of the original projects as well as the GPL-3.0 license included here.

---

## License

GNU General Public License v3.0; see `LICENSE`.
