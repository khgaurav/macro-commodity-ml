# Gold Volatility Prediction Using Commodity and Macroeconomic Indicators

Forecasting gold's realized volatility 5 and 22 trading days ahead from daily
commodity, financial-market, and macroeconomic data. Three models — **Lasso**,
**XGBoost**, and an **LSTM** — share one feature set, one sample, one split, and
one evaluation, and are measured against two benchmarks: a persistence forecast
and **HAR**, the standard linear benchmark for realized volatility.

**Why volatility and not price.** The project began by predicting the next day's
gold *return*. All three models landed at R² ≈ 0 with a direction classifier at
ROC-AUC ≈ 0.51, below a majority-class baseline. That is a result about market
efficiency, not a modelling failure, and it is reported as one in
[`../price_prediction/`](../price_prediction/). Volatility clusters — calm
follows calm, turbulence follows turbulence — so the same pipeline was pointed at
it instead. Gold's daily log returns autocorrelate at −0.012 at lag 1; log
volatility autocorrelates at 0.317, and its 22-day mean at 0.994.

## Results

Held-out test window, 14 March 2023 – 24 July 2026 (807 rows at h=5, 805 at
h=22), never seen in training or tuning. QLIKE is the primary metric — lower is
better, zero is perfect — and it penalizes under-forecasting a spike more than
over-forecasting calm, which is what a risk application needs.

| h | Model | RMSE | R²(log) | QLIKE | Dir. acc. |
|---|---|---|---|---|---|
| 5 | Persistence | 0.00531 | 0.207 | 0.515 | 50.3% |
| 5 | HAR | 0.00463 | 0.355 | 0.393 | 64.6% |
| 5 | Lasso (tuned) | 0.00420 | 0.473 | 0.309 | 70.0% |
| 5 | **XGBoost (tuned)** | **0.00420** | **0.485** | **0.305** | 70.0% |
| 5 | LSTM (tuned) | 0.00445 | 0.464 | 0.347 | 69.5% |
| 22 | Persistence | 0.00403 | 0.152 | 0.370 | 54.2% |
| 22 | HAR | 0.00348 | 0.341 | 0.285 | 64.7% |
| 22 | **Lasso (tuned)** | **0.00307** | **0.521** | **0.216** | **77.5%** |
| 22 | XGBoost (tuned) | 0.00320 | 0.489 | 0.235 | 72.5% |
| 22 | LSTM (tuned) | 0.00326 | 0.471 | 0.252 | 72.5% |

Directional accuracy is the share of days the model correctly calls volatility
rising or falling against the trailing estimate.

### Significance

Diebold-Mariano on QLIKE loss differentials, Newey-West with h−1 lags because
consecutive forecast windows overlap. Negative favours the model.

| h | Model vs HAR | DM | p | Sig. |
|---|---|---|---|---|
| 5 | Lasso | −3.19 | 0.0014 | yes |
| 5 | XGBoost | −3.31 | 0.0009 | yes |
| 5 | LSTM | −2.18 | 0.0296 | yes |
| 22 | Lasso | −3.30 | 0.0010 | yes |
| 22 | XGBoost | −2.26 | 0.0239 | yes |
| 22 | LSTM | −1.59 | 0.1125 | **no** |

All three beat persistence at the 1% level at both horizons.

### What actually mattered

**The volatility estimator, not the model.** Re-running everything on a Parkinson
target instead of Yang-Zhang removes the significant edge over HAR for every
model — p rises to 0.087 (XGBoost), 0.090 (Lasso), 0.211 (LSTM). Accounting for
the overnight gap is the single highest-impact decision here, and it is a
measurement choice made before any model sees the data.

**Implied volatility, not realized history.** QLIKE at h=5, every row retuned
from scratch:

| Feature set | # | Lasso | XGBoost |
|---|---|---|---|
| HAR only | 3 | 0.393 | 0.393 |
| Macro / cross-asset only | 21 | 0.496 | 0.361 |
| HAR + macro | 24 | 0.398 | 0.343 |
| HAR + implied vol | 8 | **0.303** | 0.313 |
| Full | 29 | 0.309 | **0.305** |

Eight features including GVZ beat twenty-four without it. Importance measures
agree: XGBoost's two GVZ features carry 0.366 of total gain against 0.237 for all
three HAR components, and `gvz_log` is Lasso's largest coefficient by a factor of
five. This overturns the previous iteration's conclusion that volatility's own
recent history was the strongest predictor — that was true of the feature set
available at the time.

### Walk-forward and position sizing

One fixed split is not enough, so a walk-forward retrained on all resolved labels
and stepped through 237 trading days, 31 July 2025 – 29 July 2026. Level-scale
RMSE: Lasso 0.00822, XGBoost 0.00840, LSTM 0.00946, against 0.01127 for a 20-day
trailing average.

The forecast cannot pick direction, but it can size a position — position = risk
target ÷ forecast volatility. Targeting 10% annualized risk on a $1,000 stake,
leverage capped at 3×, 2bp on turnover:

| Strategy | Final | Return | Sharpe | Max DD |
|---|---|---|---|---|
| Gold buy & hold | $1,097 | +9.7% | 0.43 | −27.5% |
| Vol-target (naive 20d) | $1,091 | +9.1% | 0.70 | −14.9% |
| Vol-target (XGBoost) | **$1,186** | **+18.6%** | **1.26** | −16.2% |
| Vol-target (Lasso) | $1,164 | +16.4% | 1.14 | −16.1% |
| Vol-target (LSTM) | $1,155 | +15.5% | 0.90 | −18.2% |

The gain comes from holding less through the March 2026 drawdown, not from
leveraging up during the rally.

## Data

Assembled programmatically — no ready-made dataset covers this combination.

- **Yahoo Finance** — gold futures as full OHLC + volume, crude oil, silver,
  copper, platinum, US Dollar Index, S&P 500, EUR/USD, VIX, and the two implied
  volatility indices GVZ (gold) and OVX (crude)
- **FRED** — 10-year breakeven inflation, US 2- and 10-year Treasury yields

`data/daily_commodity_market_data_cleaned.csv` holds **5,029 rows**, 22 January
2004 – 30 July 2026. The volatility models use **4,309 rows from 27 June 2008**,
for two reasons: GVZ does not exist before mid-2008, and 34–47% of 2004–2007
trading days report `High == Low` against 5.7% in 2008 — thin pre-Globex
electronic coverage, so the early range series measures only 0.21–0.36× close-to-
close volatility rather than the 0.78× it reaches later. Training on that period
would fit a mapping to a systematically understated target. The full history is
kept in the CSV.

Macro series are forward-filled, the only imputation, because those values
persist between releases. Rows with a missing *price* are dropped instead —
carrying a stale close forward invents a zero return that propagates into the
volatility estimate.

### Target

Log of mean daily **Yang-Zhang** volatility over t+1…t+h:

```
σ_t = sqrt( ln(O_t / C_{t-1})² + RS_t )
RS_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
```

Yang-Zhang adds the overnight gap to the Rogers-Satchell intraday term, which
matters because gold trades nearly around the clock. Against close-to-close
volatility it recovers 0.78× where Parkinson recovers 0.68×. The Parkinson series
is retained under a `park_` prefix so the choice can be ablated. The rolling mean
is shifted by h so no same-day information enters the label; the log transform
tames a skew of 6.4 and guarantees non-negative forecasts.

Because forecasts are made in logs, exponentiating returns the conditional median
and biases every level-scale figure low — by 7.1% for HAR here. Duan's smearing
estimator, computed from training residuals only, corrects this and is applied to
the benchmarks too.

### Features (29)

- **HAR (3)** — log volatility and its 5- and 22-day rolling means. Fitted alone
  by least squares, these also form the HAR benchmark.
- **Macro / cross-asset (21)** — gold's 1/5/20-day log returns, 5- and 20-day
  rolling return std, close position in the daily range, volume change, relative
  volume; 1-day returns for crude, silver, copper, platinum, DXY, S&P 500,
  EUR/USD; VIX change, changes in breakeven inflation and the 2y/10y yields, and
  the 2s10s slope with its change.
- **Implied volatility (5)** — log GVZ, its change, its 5-day mean, log OVX, and
  the variance risk premium (implied − realized).

Almost all are differences or ratios; the yield-curve slope is the only level. No
selection step runs before training — 29 features against ~3,200 training rows is
not wide enough to need one, and selecting on the full sample would leak the test
period.

### Split

Chronological, earliest 80% for training, and training stops h days *before* the
test window opens. Without that purge the final training labels are averaged over
a window reaching into the test period. The same purge separates the LSTM's
early-stopping slice from the rows still being fitted.

## Layout

```text
volatility_prediction/
├── notebooks/
│   ├── lasso_gold_prediction.ipynb     # Lasso volatility model
│   ├── xgboost_gold_prediction.ipynb   # XGBoost volatility model
│   └── lstm_gold_prediction.ipynb      # LSTM volatility model
├── scripts/
│   ├── create_dataset.py               # downloads raw data (Yahoo Finance + FRED)
│   └── process_dataset.py              # cleaning + feature engineering
├── app/app.py                          # Streamlit position-sizing demo
├── data/                               # raw and cleaned CSVs
├── results/                            # metrics, predictions, ablations, backtests
├── report_figures/                     # every figure, written by the notebooks
├── Dockerfile, docker-compose.yml      # container for the demo app
└── requirements.txt
```

Each notebook is self-contained and imports nothing from the others. All three
follow the same twelve-section structure — benchmarks, tuning, DM tests, feature
importance, both ablations, a direction classifier, and the walk-forward backtest
— on identical feature lists, filters, splits, and metric definitions. That is
what makes them comparable.

## Running it

Paths in the notebooks are relative to `notebooks/`, so start Jupyter there. The
scripts and the app resolve their own paths and run from anywhere.

```bash
cd volatility_prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

jupyter lab notebooks/       # run any notebook top to bottom
streamlit run app/app.py     # or: docker compose up --build
```

The cleaned CSV is committed, so nothing needs downloading. To rebuild it:

```bash
python scripts/create_dataset.py    # → data/daily_commodity_market_data.csv
python scripts/process_dataset.py   # → data/daily_commodity_market_data_cleaned.csv
```

Both sources update continuously, so a rebuild gives a later end date and the
numbers above shift slightly.

The app is deployed at
[gold-predict.gauravkh.co.in](https://gold-predict.gauravkh.co.in). It trains
only on data preceding a date you pick, forecasts the next session's volatility,
and converts that into a position size for a chosen risk target.

## Limitations

R² near 0.5 leaves half the variance unexplained, and every model under-predicts
the largest spikes — the early-2026 excursion above 6% daily volatility is met
with a forecast near 2%. This is not a tail-risk model; any limit derived from it
needs an explicit buffer.

The LSTM's failure to beat the simpler models looks like a data property rather
than an architecture one: a 22-day moving average of log volatility autocorrelates
at 0.994, so the multi-scale dependence a recurrent layer would have to discover
is already pre-computed and handed to every model in three columns. The
comparison is also not effort-matched — XGBoost got 40 search configurations to
the LSTM's five.

Evaluation rests on one 80/20 split plus one 12-month walk-forward on one asset.
Daily bars force a range-based volatility proxy where the literature standard is
intraday realized variance; five-minute bars would remove the estimator question
entirely. The backtest is stylized, with no bid-ask spread or margin mechanics.

## Team

Gaurav Harish · Luke Sanders · Omar Gomaa

Academic coursework. Nothing here is financial or investment advice.
