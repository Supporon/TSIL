# TSIL: Temporal Statistical Invariant Learning for Stock Prediction

TSIL is a **training-time regularization framework** for stock return forecasting.
On top of any backbone predictor, it injects domain temporal knowledge through
**temporal predicates** Φ and enforces **statistical-invariant constraints** built
from the structural similarity matrix `P = Φ Φᵀ`. This steers the model to respect
batch-level temporal/statistical structure instead of fitting each sample in
isolation, improving the rank quality and tradability of the predicted signal.

This repository is the reference implementation accompanying the paper. It is a
cleaned, reproducible subset of the research codebase containing only the canonical
pipeline (training entrypoints, the TSIL backend, the dataset loader, the nine
backbones used in the paper, and the experiment configs).

---

## 1. Method overview

For a mini-batch of size `B`, let `ŷ` be predictions and `y` labels. The standard
objective is the per-sample MSE. TSIL augments it with an invariant penalty:

```
L_TSIL = L_task + Σ_s β_s · || (1/B) · Φ̃_sᵀ (ŷ − y) ||²₂
```

where each predicate `Φ_s` maps the batch to a descriptor, `Φ̃_s` is its L2-normalized
form, and `P_s = Φ_s Φ_sᵀ` is the resulting structural similarity matrix. In the
implementation the loss is realized in weighted least-squares form:

```
weight_matrix = tau_hat · I + tau · P
L = mean( weight_matrix · (ŷ − y)² )
```

- `I` is the identity (the ordinary per-sample MSE term).
- `P = Φ Φᵀ` couples samples that share temporal/statistical structure.
- `tau_hat = sigmoid(model.alpha)` is a **learnable** scalar, `tau = 1 − tau_hat`.
  The model adaptively trades off pointwise convergence (`I`) against the
  invariant constraint (`P`) over training.

The predicate used to build `P` is selected **solely** by the `phi_type` config
field (there is no separate `loss_type`; see §6). Each backbone ships **one config
per market** (`config_<model>.yaml`, default `phi_type: mse`, the plain-MSE
baseline). The loss mode is then chosen **at the command line** via `--phi_type`,
which `train.py`/`search.py` overlay onto the config — no separate per-variant
files:
- `--phi_type mse`  → plain MSE (the Original baseline; the config default).
- `--phi_type ones` → weighted MSE with the all-ones control predicate `Φ_ones`.
- `--phi_type phi_momentum` (or any temporal key) → weighted MSE with a paper
  temporal predicate.

The full set of temporal predicates lives in `src/temporal_predicates.py` and
`src/custom_phi_constructors.py`, selected by setting `--phi_type` to e.g.
`phi_momentum`, `phi_volatility`, `phi_return_dist`, … (full list in §6).

The whole mechanism is **backbone-agnostic and output-level**: predicates read the
batch input/labels and act on the prediction error, so the same TSIL loss wraps
every backbone unchanged.

---

## 2. Repository layout

```
release/
├── train.py                        # single-run entry: one backbone/market, loss via --phi_type
├── search.py                       # hyper-parameter search entry (Optuna/TPE, validation-IC)
├── basktesting.py                  # Top-K Dropout backtest + portfolio metrics
├── requirements.txt
├── LICENSE                         # GPL-3.0
├── src/
│   ├── model_backbone_phitp.py     # QniverseModel + TSIL loss (WeightedMSELoss), phi_type-driven
│   ├── temporal_predicates.py      # the paper's six temporal predicates (Φ_ts/cycle/phase/mom/vol/dist)
│   ├── custom_phi_constructors.py  # extra domain temporal predicates (momentum, vol, …)
│   ├── run_utils.py                # shared defaults + config overrides for both entries
│   ├── dataset_ori_onehot.py       # MTSDatasetH time-series dataset loader (phi_type-aware)
│   ├── timefeatures.py             # timestamp feature construction
│   └── models/                     # 11 backbone implementations
│       ├── LSTM.py  GRU.py  TCN.py  Transformer.py  GAT.py  # domain-agnostic
│       └── LSR_IGRU.py  MASTER.py  MERA.py  StockMixer.py   # stock-specialized
├── configs/
│   ├── f158_csi300/                # CSI 300, one config per backbone (9 models, loss via --phi_type)
│   └── f158_csi800/                # CSI 800, one config per backbone (9 models, loss via --phi_type)
```

> The two entry points share their defaults and config-mutation logic through
> `src/run_utils.py` (`DEFAULT_SEEDS`, `DEFAULT_SEARCH_SPACE`, `TAU_VALUES`,
> `apply_overrides`, `inject_model_specific_params`).

---

## 3. Installation

```bash
# Python 3.10–3.12; create a fresh env, then:
pip install -r requirements.txt
```

Core dependencies (`requirements.txt`):

| Package      | Version  |        | Package      | Version  |
|--------------|----------|--------|--------------|----------|
| numpy        | 1.26.4   |        | pyqlib       | 0.9.6    |
| pandas       | 2.2.3    |        | scipy        | 1.14.1   |
| scikit-learn | 1.5.2    |        | statsmodels  | 0.14.2   |
| torch        | 2.3.1    |        | matplotlib   | 3.10.0   |
| torchvision  | 0.18.1   |        | plotly       | 5.24.1   |
| torchaudio   | 2.3.1    |        | ruamel.yaml  | 0.18.14  |
|              |          |        | tqdm         | 4.66.5   |

`scipy` and `statsmodels` are required at import time by the loss module
(`statsmodels.tsa.stattools.acf` powers the autocorrelation predicate); the
visualization/serialization deps (`matplotlib`, `plotly`, `ruamel.yaml`,
`tqdm`) are imported by the training/report path.

This release was verified end-to-end on Python 3.12 with torch 2.3.1 (CUDA
12.1). A CUDA-capable GPU is recommended; the code falls back to CPU
automatically.

---

## 4. Data preparation

The pipeline runs on [Qlib](https://github.com/microsoft/qlib) China A-share data
with the **Alpha158** feature handler (158 features). Place the Qlib binary data so
that the provider URI in the configs resolves:

```
release/qlib_data/cn_data/        # provider_uri: "./qlib_data/cn_data"
```

Obtain `cn_data` via Qlib's official downloader (`python -m qlib.run.get_data
qlib_data --target_dir ./qlib_data/cn_data --region cn`) or your own dump. If
your data lives elsewhere, a symlink is the simplest wiring:

```bash
ln -s /path/to/your/qlib_data ./qlib_data   # so ./qlib_data/cn_data resolves
```

Quick check that the data resolves and covers the experiment window:

```bash
head -1 qlib_data/cn_data/calendars/day.txt   # should be ≤ 2018-01-01
tail -1 qlib_data/cn_data/calendars/day.txt   # should be ≥ 2025-08-31
ls qlib_data/cn_data/instruments/             # expect csi300.txt, csi800.txt, …
```

> The public Qlib `cn_data` dump typically ends before the paper's
> 2024–2025 test window. To reproduce the paper exactly you need a dump that
> spans 2018-01-01 → 2025-08-31 (CSI 300 / CSI 800 membership included).

Features and labels are produced inside the config via the Alpha158 handler; the
label is the two-day forward return `Ref($close, -2) / Ref($close, -1) - 1`.
Feature processing: `RobustZScoreNorm` (clip outliers) + `Fillna`; label
processing: `DropnaLabel` + `CSRankNorm`.

**Chronological split** (set per config, identical across markets):

| Segment | Range                       |
|---------|-----------------------------|
| train   | 2018-01-01 → 2022-12-31     |
| valid   | 2023-01-01 → 2023-12-31     |
| test    | 2024-01-01 → 2025-08-31     |

---

## 5. Running experiments

There are two entry points: `train.py` for a single run and `search.py` for a
hyper-parameter search.

### `train.py` — single run

Pick the backbone with `--config_file` (one config per backbone per market) and
the loss mode with `--phi_type`:

```bash
# Original baseline (plain MSE — the config default)
python train.py --config_file configs/f158_csi300/config_lstm.yaml --seed 42

# TSIL-Enhanced with a temporal predicate
python train.py --config_file configs/f158_csi300/config_lstm.yaml \
    --phi_type phi_momentum --tau_hat_init 2.0 --seed 42
```

Available overrides: `--phi_type --base_model --tau_hat_init --lr
--d_model --e_layers --batch_size --market --seed`.

- **Mode selection (`--phi_type`).** The loss mode is driven entirely by
  `phi_type` (there is no `loss_type` flag):
  `mse` → *Original* (plain MSE); `ones` → the `Φ_ones` weighted control;
  `<temporal key>` (e.g. `phi_momentum`) → *TSIL-Enhanced* (invariant penalty).
  Any key from §6 is valid.
- **Multiple seeds.** `--seed` accepts a list and runs each in turn; the default
  is `42 2022 2023 2024 2025` (`DEFAULT_SEEDS` in `src/run_utils.py`). One report
  is written per seed. Pass a single value (`--seed 2025`) for a one-off run.
- **Market.** `--market csi800` (or `csi300`) retargets the universe,
  benchmark, and instrument set in one flag.

Each backbone ships **one config per market** (`config_<model>.yaml`,
default `phi_type: mse`). The same config becomes any mode via `--phi_type`, so
there are no separate `*_phi.yaml`/`*_tp.yaml` files. All CSI 300 and CSI 800
configs use the 2-day label `Ref($close, -2)/Ref($close, -1) - 1` for a strictly
controlled Original-vs-TSIL comparison.

### `search.py` — hyper-parameter search

Searches `lr × layers × d_model × batch_size × tau` with Optuna (TPE Bayesian
search) for one or more backbones and predicates, **selecting the best
configuration by validation IC** (the IC of the best epoch on the validation
segment, read from `model.fit`). The test segment is never used to pick
hyper-parameters.

```bash
python search.py --config_file configs/f158_csi300/config_lstm.yaml \
    --models LSTM GRU --phi_types phi_momentum --market csi300 --n_trials 50
```

Default search space (`DEFAULT_SEARCH_SPACE` / `TAU_VALUES` in
`src/run_utils.py`):

| Param   | Values                          |
|---------|---------------------------------|
| lr      | 0.001, 0.0001, 0.00001          |
| layers  | 1 – 3                           |
| d_model | 64, 128, 256                    |
| bs      | 256, 512, 1024                  |
| tau     | -4.0, -2.0, 0.0, 2.0, 4.0       |

`--early_stop N` stops a study after `N` trials without improvement. Outputs per
model/predicate/tau combination: `trial_history.csv`, `optimization_report.txt`,
optimization plots under `visualizations/`, plus a top-level `all_best_results.csv`.

### Stock-specialized baselines (LSR-IGRU, MASTER, MERA, StockMixer)

The same `train.py` runs the baseline backbones; their architecture-specific
parameters live in each baseline config's `model_config` block (they are **not**
auto-injected by `train.py` — only `search.py` calls
`inject_model_specific_params`, so the YAML must carry them). MASTER honours
`use_gate: False` to skip its market-feature gate when running on the plain
Alpha158 handler (which has no market columns):

```bash
python train.py --config_file configs/f158_csi300/config_master.yaml --seed 2025
```

Each market dir ships its own baseline configs (`configs/f158_csi800/config_master.yaml` for CSI 800); `--market` can still retarget any config.

Outputs (training curves, `backtest_result_seed*.csv`, `backtest_report_*.txt`)
are written under the `logdir` set in each config.

---

## 6. Temporal predicates (`phi_type`)

The predicate that builds `P = Φ Φᵀ` is chosen by the `phi_type` field in
`model_config` and dispatched inside `WeightedMSELoss.forward`
(`src/model_backbone_phitp.py`). `phi_type` is the **only** loss switch:

| `phi_type`        | Loss                                    | Config family |
|-------------------|-----------------------------------------|---------------|
| `mse`             | plain per-sample MSE (no `P`)           | `*_ori.yaml`  |
| `ones`            | weighted MSE with the `Φ_ones` control  | `*_phi.yaml`  |
| a temporal key    | weighted MSE with that predicate's `P`  | `*_tp.yaml`   |

**(a) The paper's six temporal predicates** — `src/temporal_predicates.py`
(dispatched via `get_temporal_I_P`):

| Paper predicate | `phi_type` key | Descriptor |
|-----------------|----------------|------------|
| Timestamp (Φ_ts)     | `phi_timestamp` / `phi_timestamp_dim` / `phi_timestamp_sl` | one-hot calendar features (needs the timestamp-aware dataset, see below) |
| Market Cycle (Φ_cycle) | `phi_cycle` / `phi_cycle_onehot` | position within the batch's dominant FFT period |
| Asset Phase (Φ_phase)  | `phi_phase`        | per-sample cyclical position over the sequence |
| Momentum (Φ_mom)     | `phi_momentum`     | multi-horizon relative returns (k = 5/10/20) |
| Volatility (Φ_vol)   | `phi_volatility`   | multi-window rolling std (w = 10/20/40) |
| Distribution (Φ_dist)| `phi_return_dist`  | first four moments [mean, std, skew, kurt] |

Legacy aliases `phi_period` / `phi_period_onehot` / `phi_period_per_sample` also
route to the cycle/phase predicates.

**(b) Extra exploratory predicates** — `src/custom_phi_constructors.py`
(`phi_momentum`, `phi_volatility`, `phi_return_dist` are shared with table (a)):
`phi_trend`, `phi_multiscale`, `phi_temporal`, `phi_feature_corr`,
`phi_price_volume`, `phi_factor`, `phi_technical`, `phi_vol_return`,
`phi_return_sim`, `phi_market_beta`, `phi_cross_corr`, `phi_lead_lag`,
`phi_common_factor`, `phi_vol_sync`, `phi_tail_risk`, `phi_market_regime`,
`phi_sector_cluster`. Run `python src/custom_phi_constructors.py` for a self-test
of all of them, or `python src/temporal_predicates.py` for the six paper predicates.

**Dataset note (timestamp predicate).** Only `phi_timestamp*` needs calendar
features; the dataset (`src/dataset_ori_onehot.py`) builds them **only** when the
`phi_type` it receives starts with `phi_timestamp` (encoding controlled by its
`embed` / `ts_onehot` kwargs). For every other `phi_type` no timestamp features are
built or carried. `train.py` / `search.py` pass `phi_type` through to the dataset
automatically.

The shipped `*_phi.yaml` configs use `phi_type: ones` (the `Φ_ones` control) and
`*_tp.yaml` use `phi_type: phi_momentum`; **to reproduce the paper's per-backbone
temporal-predicate results, start from `*_tp.yaml` or pass `--phi_type` with a key
from table (a)**.

### Sweeping predicates across backbones

To compare predicates across backbones, drive `train.py` over the keys you want
(`--phi_type`), or use `search.py --phi_types <k1> <k2> …` to also tune
hyper-parameters per predicate.

---

## How to add a predicate

A predicate is a function that maps a mini-batch to a per-sample descriptor
`φ ∈ ℝ^{B×d}`; the loss then couples samples through `P = φ̃ φ̃ᵀ`. Adding one is two
small steps.

**Contract.** Implement a constructor with this exact signature in
`src/custom_phi_constructors.py`:

```python
def get_<name>_I_P(batch_x, device=None, **kwargs):
    # batch_x: [batch_size, seq_len, feature_dim]
    # returns (I, P), both [batch_size, batch_size] tensors
```

Requirements:

- Compute a per-sample descriptor `phi` of shape `[batch_size, d]` (any `d ≥ 1`).
- L2-normalize over the whole batch and form `P`: `phi = phi / (torch.linalg.norm(phi) + 1e-8)`, `P = phi @ phi.T`.
- Return the identity for the pointwise term: `I = torch.eye(batch_size, device=device)`.
- Guard against NaN/Inf with `torch.nan_to_num(...)` — features can contain gaps.

**Wiring (two edits).**

1. Register the key in `get_custom_I_P`'s `phi_type_map` (same file):
   `'phi_<name>': get_<name>_I_P`.
2. Add `'phi_<name>'` to the `_CUSTOM_PHI_TYPES` set in
   `src/model_backbone_phitp.py`. The dispatch branch
   (`phi_type in _CUSTOM_PHI_TYPES`) in `WeightedMSELoss.forward` then routes to
   it automatically.

After that, `--phi_type phi_<name>` works in both `train.py` and `search.py`.

**Example** — a predicate that groups stocks by recent trading-volume trend
(sign of the last-vs-first volume change over the window):

```python
def get_volume_trend_I_P(batch_x, device=None, **kwargs):
    batch_size = batch_x.shape[0]
    if device is None:
        device = batch_x.device
    I = torch.eye(batch_size, device=device)
    # use the cross-feature mean as a stand-in for the traded volume series
    series = batch_x.mean(dim=2)                       # [B, seq_len]
    trend = (series[:, -1] - series[:, 0]).unsqueeze(1)  # [B, 1]
    phi = torch.nan_to_num(trend, nan=0.0)
    phi = phi / (torch.linalg.norm(phi) + 1e-8)
    P = phi @ phi.T
    return I, P
```

Register `'phi_volume_trend': get_volume_trend_I_P` in `phi_type_map`, add
`'phi_volume_trend'` to `_CUSTOM_PHI_TYPES`, then run
`python train.py --config_file <cfg> --phi_type phi_volume_trend`.

---

## 7. Backtest protocol

After prediction, signals are evaluated with a daily-rebalanced **Top-K Dropout**
strategy (`basktesting.py` + Qlib's `PortAnaRecord`):

- `topk = 50`, `n_drop = 5`, daily rebalancing.
- A-share frictions: 0.05% buy commission, 0.15% sell stamp duty, 0.01% slippage,
  100-share lots; suspended / limit-hit stocks excluded (`limit_threshold: 0.095`).
- Backtest window: 2024-01-01 → 2025-08-31 (the test segment).

Reported metrics:

- **Ranking quality:** IC, ICIR, RankIC, RankICIR.
- **Portfolio:** AR (annualized return) and IR (information ratio), with and without
  transaction cost, plus max drawdown.

The AR/IR/MMD figures in `backtest_report_*.txt` are read from Qlib's
`PortAnaRecord` (excess return over the benchmark). `basktesting.py` also computes
a self-contained, commission-aware Top-K-Dropout return series
(`calculate_portfolio_metrics`, annualized with 252) saved to
`backtest_result_seed*.csv`; use one source consistently when comparing runs.

---

## 8. Reproducing the paper comparisons

For each backbone and market, the paper compares two modeling forms:

1. **Original** — `--phi_type mse` (plain MSE), i.e. the `*_ori.yaml` configs.
2. **TSIL-Enhanced** — `--phi_type <temporal key>` (invariant penalty), i.e. the
   `*_tp.yaml` configs. `*_phi.yaml` (`phi_type: ones`) is the `Φ_ones` ablation.

Run the same backbone/market with each `phi_type` and compare the
`backtest_report_*.txt` outputs.

> The feature-concatenated variant reported in the paper (predicate descriptor
> fed as an extra input feature rather than as a loss term) is **not** part of
> this release: its `PhiFeatureModel` / `phi_feature_extractor` modules are not
> included here. This repo covers the Original and loss-based TSIL forms.

### Reproduction notes (read before a full run)

The shipped configs are tuned per backbone, but a few defaults differ from the
paper's full protocol. None block training; adjust them to match the paper:

- **Predicate selection.** `*_phi.yaml` use `phi_type: ones` (the `Φ_ones`
  control) and `*_tp.yaml` use `phi_type: phi_momentum`. For the paper's
  per-backbone best predicate, start from `*_tp.yaml` or pass `--phi_type` with
  the corresponding key in §6(a) (e.g. `phi_momentum` for GAT, `phi_volatility`
  for LSTM on CSI 300).
- **CSI 300 label horizon.** Some CSI 300 `*_ori.yaml` use a 3-day label
  (`Ref($close, -4)`); set them to the 2-day label `Ref($close, -2)` to match
  the `*_phi.yaml` and the paper.
- **Hyper-parameters.** `train.py` honours each config's own hyper-parameters
  (no hidden overrides); pass `--lr/--d_model/--e_layers/--batch_size` to vary
  them, or use `search.py` to tune them by validation IC.
- **Seeds.** `train.py` runs the seed set `42 2022 2023 2024 2025` by default;
  the paper reports mean ± std over multiple seeds. Pass `--seed` to change them.
- **Model selection.** Early stopping in `QniverseModel.fit` tracks validation
  **IC** (Pearson) with the patience set by `early_stop` in each config; this is
  also the metric `search.py` selects on. The paper describes selection on
  validation RankICIR; change the `valid_metrics["IC"]` criterion in `fit()` to
  match that exactly.
- **MASTER input.** `src/models/master.py` reads market features at input indices
  `[158:221]` (Qlib `Alpha158` + 63 market columns) **when `use_gate: True`**. The
  shipped baseline config feeds 158-feature `Alpha158` with `use_gate: False`, and
  the model now honours that flag by skipping the market-feature gate (all 158
  features pass straight through). To enable the gate, provide the 221-feature
  market-augmented input and set `use_gate: True`.

## License

Released under the GNU General Public License v3.0 (see `LICENSE`).
