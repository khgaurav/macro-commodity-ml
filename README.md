# Forecasting Gold with Commodity and Macroeconomic Indicators

Two related projects built on the same daily dataset — gold prices, other
commodities, currencies, equity-market data, and macroeconomic series drawn from
Yahoo Finance and the Federal Reserve Economic Data (FRED) database.

They are kept side by side because the second one exists as a direct consequence
of what the first one found.

| | Target | Verdict |
|---|---|---|
| [`price_prediction/`](price_prediction/) | Next trading day's gold log return | Not predictable from this data — R² ≈ 0, directional accuracy ~56% at best |
| [`volatility_prediction/`](volatility_prediction/) | Gold's near-term realized volatility (5- and 22-day) | Predictable — volatility clusters, and the models beat both a persistence baseline and HAR |

## [price_prediction/](price_prediction/)

The original project. Lasso, XGBoost, and an LSTM are trained to predict
`tomorrow_gold_log_return` from 21 engineered features, split chronologically and
evaluated on a held-out 2023–2026 test period.

All three land at roughly R² ≈ 0 against a zero-return baseline. That is a
genuine result about market efficiency rather than a modelling failure, and it is
reported as one. See [`price_prediction/README.md`](price_prediction/README.md)
and [`price_prediction/docs/Iteration_4_Report.pdf`](price_prediction/docs/Iteration_4_Report.pdf).

## [volatility_prediction/](volatility_prediction/)

The follow-on project. Returns are unpredictable, but their *magnitude* is not —
calm follows calm and turbulence follows turbulence — so the same three model
families were pointed at realized volatility instead.

The dataset gains HAR components, implied-volatility features (CBOE GVZ and OVX),
and a Yang-Zhang volatility estimator that accounts for the overnight gap. Every
model is measured against a persistence baseline and against HAR, the standard
benchmark for realized-volatility forecasting, with Diebold-Mariano significance
tests, feature and estimator ablations, a direction classifier, and a walk-forward
backtest.

It also ships a small Streamlit demo (`volatility_prediction/app/app.py`) that
turns a volatility forecast into a position size for a chosen risk target. See
[`volatility_prediction/README.md`](volatility_prediction/README.md).

## Layout

Both projects use the same internal structure and each is self-contained — its
own `requirements.txt`, its own data, and paths relative to its own root.

```text
macro-commodity-ml/
├── price_prediction/
│   ├── notebooks/      # Lasso, LSTM, XGBoost, comparison, classification metrics
│   ├── scripts/        # dataset build, preprocessing, XGBoost run, ablation test
│   ├── data/           # raw and cleaned CSVs
│   ├── results/        # exported metrics, predictions, feature importances
│   ├── docs/           # iteration reports
│   ├── README.md
│   └── requirements.txt
└── volatility_prediction/
    ├── notebooks/      # Lasso, XGBoost, LSTM volatility models
    ├── scripts/        # dataset build, preprocessing
    ├── app/            # Streamlit position-sizing demo
    ├── data/           # raw and cleaned CSVs
    ├── results/        # exported metrics, predictions, ablations, backtests
    ├── report_figures/ # every figure, written by the notebooks
    ├── Dockerfile      # container for the demo app
    ├── docker-compose.yml
    ├── README.md
    └── requirements.txt
```

## Getting Started

Pick a project and follow its README. Each has its own dependency set, so create a
separate virtual environment per project:

```bash
cd volatility_prediction        # or price_prediction
python -m venv .venv
source .venv/bin/activate       # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The cleaned dataset is committed in both projects, so the notebooks run without
rebuilding it. Because the upstream sources update over time, regenerating the
data produces a later end date than the version used in the reported experiments.

## Team Members

- Gaurav Harish
- Luke Sanders
- Omar Gomaa

## Disclaimer

Academic coursework. Nothing here is financial or investment advice.
