# Gold Return Prediction Using Commodity and Macroeconomic Indicators

## Overview

This project investigates whether historical commodity, financial-market, and macroeconomic data can be used to predict the next trading day's gold return.

The dataset was created by combining daily market data from Yahoo Finance with macroeconomic data from the Federal Reserve Economic Data (FRED) database. After preprocessing and feature engineering, the project compares machine learning models that approach the problem in different ways:

- **Lasso**, a linear baseline whose L1 penalty performs feature selection
- **XGBoost**, which learns nonlinear relationships from engineered features
- **Long Short-Term Memory (LSTM)**, which learns temporal patterns from sequences of historical observations

The prediction target is:

```text
tomorrow_gold_log_return
```

This represents the logarithmic return of gold from the current trading day to the next trading day.

## Team Members

- Gaurav Harish
- Luke Sanders
- Omar Gomaa

## Research Questions

This project aims to answer the following questions:

1. Which machine learning model predicts next-day gold returns most accurately?
2. Does macroeconomic data improve prediction performance compared with gold-price history alone?
3. Which features contribute the most to model predictions?
4. Do other commodities, such as silver, crude oil, copper, and platinum, improve gold-return forecasts?
5. How does model performance change after hyperparameter tuning?

## Dataset

### Sources

The dataset is generated using:

- **Yahoo Finance** for gold, commodities, currencies, equity-market data, volatility, and trading volume
- **Federal Reserve Economic Data (FRED)** for inflation expectations and U.S. Treasury yields

### Size

After preprocessing and feature engineering, the dataset contains:

- **5,016 daily observations**
- **21 predictor features**
- **1 target variable**
- Date range: **January 22, 2004 through July 13, 2026**

### Market and Macroeconomic Inputs

The source data includes:

- Gold closing price, high, low, and volume
- Crude oil
- Silver
- Copper
- Platinum
- U.S. Dollar Index
- S&P 500
- EUR/USD
- VIX Index
- 10-year breakeven inflation
- U.S. 2-year Treasury yield
- U.S. 10-year Treasury yield

### Engineered Features

The preprocessing pipeline creates features including:

- Gold 1-day, 5-day, and 20-day log returns
- Gold 5-day and 20-day rolling volatility
- Gold closing position within the daily high-low range
- Gold volume change
- Gold relative volume
- Daily returns for other commodities and financial assets
- Daily VIX change
- Changes in inflation expectations and Treasury yields
- The 10-year minus 2-year Treasury yield-curve slope
- Daily change in the yield-curve slope

## Repository Structure

```text
macro-commodity-ml/
├── notebooks/
│   ├── lasso_gold_prediction.ipynb
│   ├── lstm_gold_prediction.ipynb
│   ├── xgboost_gold_prediction.ipynb
│   ├── compare_gold_models.ipynb
│   └── classification_metrics_gold_models.ipynb
├── scripts/
│   ├── create_dataset.py
│   ├── process_dataset.py
│   ├── feature_ablation_test.py
│   └── xgboost_gold_prediction.py
├── data/
│   ├── daily_commodity_market_data.csv
│   └── daily_commodity_market_data_cleaned.csv
├── results/
│   └── (metrics, predictions, tuning-results, and feature-importance CSVs for every model)
├── docs/
│   ├── Iteration #02 _ Team Topic.pdf
│   └── Iteration_4_Report.pdf
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/khgaurav/macro-commodity-ml.git
cd macro-commodity-ml
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not yet available, install the main packages directly:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter yfinance
```

## Usage

### Generate the raw dataset

```bash
python scripts/create_dataset.py
```

This downloads the required Yahoo Finance and FRED data and saves:

```text
data/daily_commodity_market_data.csv
```

Because the data sources update over time, rerunning this script may produce a dataset with a later end date than the version used in the submitted experiments.

### Preprocess the dataset

```bash
python scripts/process_dataset.py
```

This performs data cleaning and feature engineering and saves:

```text
data/daily_commodity_market_data_cleaned.csv
```

### Run the Lasso model

Open:

```text
notebooks/lasso_gold_prediction.ipynb
```

Run the notebook cells in order to load the processed data, create chronological splits, scale the features, fit a baseline Lasso model, tune its regularization strength (alpha), and evaluate it on the test set. It saves the following to the `results/` folder:

```text
results/lasso_metrics.csv
results/lasso_predictions.csv
results/lasso_feature_importance.csv
```

### Run the LSTM model

Open:

```text
notebooks/lstm_gold_prediction.ipynb
```

Run the notebook cells in order to load the processed data, create chronological splits, scale the variables, build historical sequences, train the LSTM, tune its hyperparameters, evaluate it on unseen data, and generate visualizations.

### Run the XGBoost model

```bash
python scripts/xgboost_gold_prediction.py
```

This loads the processed data, creates the same chronological splits, trains a
baseline XGBoost model and a hyperparameter-tuned model (selected by a
randomized search scored with time-series cross-validation on the combined
training and validation data, then refit on that data before testing),
evaluates them against a zero-return baseline on the test period, and generates
visualizations. It saves the following to the `results/` folder:

```text
results/xgboost_metrics.csv
results/xgboost_predictions.csv
results/xgboost_tuning_results.csv
```

XGBoost is a tree-based model and is scale-invariant, so the features and target
are used on their natural scale without the standardization step used for the LSTM.

## Modeling Approach

### Chronological Splitting

The data is split chronologically rather than randomly because this is a time-series forecasting problem. Earlier observations are used for training, later observations for validation, and the newest observations for final testing.

| Dataset | Observations | Period |
|---|---:|---|
| Training | 3,511 | January 22, 2004 – July 13, 2020 |
| Validation | 752 | July 14, 2020 – July 10, 2023 |
| Test | 753 | July 11, 2023 – July 13, 2026 |

### Scaling

Predictor and target scalers are fitted using only the training data. The fitted scalers are then applied to the validation and test sets to prevent data leakage.

### Lasso

Lasso uses the engineered features directly (after scaling) and relies on its L1 penalty to shrink less useful predictors' coefficients to exactly zero, acting as a built-in feature-selection step.

### LSTM

The LSTM model receives sequences of historical feature vectors and predicts the next-day gold log return. Its main hyperparameters include:

### XGBoost

XGBoost uses the engineered features as tabular inputs and learns nonlinear interactions among gold history, other commodities, market indicators, and macroeconomic variables.

## Evaluation Metrics

Because the project is a regression problem, the primary evaluation metrics are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Directional accuracy

Classification metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrices are not primary metrics for the continuous-return prediction task.

## Results

All models are evaluated on the same held-out test period (753 trading days, July 11, 2023 – July 13, 2026):

| Model | MAE | MSE | RMSE | R² | Directional Accuracy |
|---|---:|---:|---:|---:|---:|
| Zero-Return Baseline | 0.009195 | 0.000175 | 0.013236 | -0.005592 | — |
| Baseline Lasso | 0.009173 | 0.000174 | 0.013195 | 0.000677 | 51.66% |
| Tuned Lasso | 0.009163 | 0.000174 | 0.013197 | 0.000323 | 53.25% |
| Baseline LSTM | 0.009158 | 0.000174 | 0.013195 | 0.000614 | 55.78% |
| Tuned LSTM | 0.009123 | 0.000174 | 0.013183 | 0.002461 | 56.44% |
| Baseline XGBoost | 0.009201 | 0.000174 | 0.013206 | -0.000923 | 54.18% |
| Tuned XGBoost | 0.009191 | 0.000175 | 0.013233 | -0.005104 | 52.72% |

Tuned LSTM is the best-performing model on every metric, though the margin over the Zero-Return Baseline is small across all three model families — see `docs/Iteration_4_Report.pdf` for full discussion and interpretation.

## Visualizations

The final project will include visualizations such as:

- Training and validation loss
- Actual versus predicted gold returns
- Prediction residuals
- Model-performance comparison
- Feature importance
- Feature-group or ablation comparisons

Generated figures and their repository locations will be documented here when available.

## Feature Importance and Interpretation

Lasso's coefficients are a direct measure of feature importance: the L1 penalty shrinks less-informative features' coefficients to exactly zero, so the non-zero coefficients that remain are the model's selected features.

XGBoost feature importance can be obtained directly from the trained tree model.

Because an LSTM does not provide native tree-style feature importance, its inputs may be interpreted using methods such as:

- Permutation importance
- Feature-group ablation
- SHAP analysis

The final report will discuss which gold-history, commodity, financial-market, and macroeconomic features contribute most to prediction performance.

## Reproducibility

Random seeds are set where supported to make model training more reproducible. However, minor numerical differences may still occur across operating systems, TensorFlow versions, processors, and hardware configurations.

For exact comparison, all reported models should use:

- The same processed dataset
- The same chronological split
- The same test period
- The same evaluation definitions

## Limitations

Potential limitations include:

- Financial returns are noisy and difficult to forecast consistently
- Yahoo Finance continuous futures data may contain contract-roll effects
- Some source series have different market calendars and publication schedules
- Macroeconomic series may be forward-filled between release dates
- Historical relationships may change during different market regimes
- Strong performance on historical data does not guarantee profitable real-world trading
- Transaction costs, spreads, slippage, and execution constraints are not modeled

## Future Improvements

Possible extensions include:

<!-- - Additional economic indicators
- Alternative sequence lengths
- Walk-forward validation
- Attention-based neural networks
- Transformer time-series models
- More extensive hyperparameter optimization
- Regime-specific models
- Prediction intervals and uncertainty estimates
- Trading simulations that include transaction costs -->

## Data Sources

- Yahoo Finance
- Federal Reserve Economic Data (FRED)

## Disclaimer

This project is for academic and educational purposes only. It does not provide financial or investment advice.
