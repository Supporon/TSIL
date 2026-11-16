# TSIL: original-paper method alignment

This branch repairs the implementation released at commit
`5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff` to follow the original TSIL manuscript.
See [the Chinese change report](docs/PAPER_ALIGNMENT_REPORT_ZH.md) for evidence,
before/after comparisons, explicit conventions and remaining reproduction limits.
The historical paper tables have **not** been reproduced by this repair.

## Objective and descriptors

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

## Installation and data

Use Python 3.10–3.12 and a fresh environment:

```bash
pip install -r requirements.txt
pip install pytest==8.4.2
```

Verification used Python 3.12.14, PyTorch 2.3.1 CPU and Qlib 0.9.6. The manuscript
states PyTorch 1.13.1 / Qlib 0.8.0; this is not a verification of that older
environment. Joblib is pinned to 1.4.2 because Qlib 0.9.6's ParallelExt fails
against the newer backend API. Full tested versions are recorded in
`docs/verification/environment.txt`.

Supply a Qlib provider at `qlib_data/cn_data`, or edit `qlib_init.provider_uri`.
It must contain the historical CSI300/CSI800 membership, trading calendar,
OHLCV, close, factor and benchmark series. The original paper dates remain:
train 2018–2022, validation 2023, test January 2024–August 2025. Preserve the
provider's earlier calendar/history so global phases and long windows are
reproducible. No real-market dataset is bundled here.

## Select configurations without evaluating test

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

Feature must reuse those exact network choices:

```bash
python search.py --config_file configs/f158_csi300/config_lstm.yaml --models LSTM GRU TCN Transformer GAT LSR_IGRU MASTER MERA StockMixer --market csi300 --injection_mode feature --frozen_network_manifest output_search_csi300_tsil/selection_manifest.json --output_dir output_search_csi300_feature
```

Repeat with `csi800` and separate output directories. A manifest records all
selected checkpoints/configs, candidate validation ICs, and a common main-table
predicate selected by mean validation daily IC over the five seeds (first wins
ties). This aggregation rule is a disclosed convention. `ones` is never a
candidate for that main-table choice. Search does not prepare or evaluate test.

## Final evaluation and fixed runs

For each selected run in the manifest, use its saved paths:

```bash
python train.py --config_file PATH_TO_SELECTED/config.yaml --checkpoint PATH_TO_SELECTED/model.bin
```

The checkpoint path evaluates only test; it does not refit or reselect weights.
Use `selected_by_predicate` for complete supplementary comparisons and
`selected_runs` for the common main-table predicate. Aggregate each model/form/
predicate's five final test runs as mean and standard deviation; never choose
which result to report by looking at test scores.

A fixed run, without a search, is also supported:

```bash
python train.py --config_file configs/f158_csi300/config_lstm.yaml --phi_type phi_momentum --injection_mode tsil --penalty_alpha 2 --seed 42
```

Outputs include checkpoint, prediction pickle, resolved YAML, daily executed
portfolio returns, actual holdings, JSON metrics and a text report. IC/RankIC
are calculated across stocks per day, then averaged over days; their IR uses
population daily std (`ddof=0`). Undefined correlations/IR are reported as
undefined, not replaced with zero. Inference is synchronized and reported in
**ms/sample**, excluding data loading and descriptor-cache construction.

Qlib executes scores at the next trading step. Backtest settings are Top-50,
Drop-5, buy/sell fees 0.05%/0.15%, close-price slippage +0.01%/-0.01%, trade unit
100 and minimum fee 5. Factor data are required for Qlib lot conversion. The
retained `limit_threshold=0.095` is Qlib's uniform limit proxy; it is not a
verified board/ST-specific Chinese price-limit calendar. Suspended stocks are
filtered by the exchange. Reports distinguish absolute strategy returns from
benchmark-relative excess returns, with explicitly chosen arithmetic 252-day
annualization, daily std `ddof=1` and additive drawdown. A score's prediction
target t→t+1 is not substituted for the return of an executable trade at t+1.

## Verification

```bash
python -m pytest tests -q
python -m compileall -q src train.py search.py scripts
python -m scripts.compare_baseline
python -m scripts.verify_synthetic_pipeline
```

The final command generates a local synthetic Qlib provider, runs actual
Alpha158/three-form training/checkpoint selection/Top-50 backtests, and verifies
labels, cost, slippage quotes, lot sizes and suspension/limit-threshold filters.
It is a software integration check, not evidence of TSIL's market performance.

Legacy `src/custom_phi_constructors.py`, `src/timefeatures.py` and
`basktesting.py` remain available as historical code. The repaired entry points
do not use them for paper predicates, timestamp encoding or portfolio returns.
Old `tau_hat_init`, legacy predicate aliases and checkpoints containing
trainable `alpha` are not interchangeable with this repaired method: retrain.

## License

GNU General Public License v3.0; see `LICENSE`.
