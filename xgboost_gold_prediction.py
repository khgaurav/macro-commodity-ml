import os

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "daily_commodity_market_data_cleaned.csv")
TARGET = "vol_target"
HAR_FEATURES = ["har_daily", "har_weekly", "har_monthly"]
# HAR features plus the full macro / cross-asset feature set. A few of these
# alone add noise; only the whole set together gives a small measured uplift.
MACRO_FEATURES = [
    "gold_return_1d", "gold_return_5d", "gold_return_20d",
    "gold_volatility_5d", "gold_volatility_20d", "gold_close_pos",
    "gold_volume_change", "gold_relative_volume",
    "crude_oil_return_1d", "silver_return_1d", "copper_return_1d",
    "platinum_return_1d", "us_dollar_index_return_1d", "sp_500_return_1d",
    "eur_usd_return_1d", "vix_change_1d", "breakeven_inflation_10y_diff_bps",
    "us_2_year_treasury_yields_diff_bps", "us_10_year_treasury_yields_diff_bps",
    "two_ten_slope", "two_ten_slope_change_bps",
]
XGB_FEATURES = HAR_FEATURES + MACRO_FEATURES
TEST_FRACTION = 0.20  # most recent 20% of days held out for testing
SEED = 42


def evaluate(name, y_true_log, y_pred_log):
    """Print RMSE, R2, and QLIKE on the original volatility scale."""
    y_true = np.exp(np.asarray(y_true_log))
    y_pred = np.exp(np.asarray(y_pred_log))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # QLIKE loss on variance (lower is better; 0 is a perfect forecast)
    ratio = y_true ** 2 / y_pred ** 2
    qlike = np.mean(ratio - np.log(ratio) - 1)

    print(f"{name:<16} RMSE={rmse:.6f}  R2={r2:+.4f}  QLIKE={qlike:.5f}")


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()
    # Drop the High == Low rows kept in the CSV for the LSTM; the volatility
    # features/target are undefined there.
    data = data.dropna(subset=XGB_FEATURES + [TARGET])

    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data.iloc[:split], data.iloc[split:]
    y_train, y_test = train[TARGET], test[TARGET]

    print(f"train={len(train)}  test={len(test)}  "
          f"test period {test.index.min().date()} -> {test.index.max().date()}\n")

    # Persistence baseline: next-week volatility equals the past-week volatility
    evaluate("Persistence", y_test, test["har_weekly"])

    # HAR benchmark: linear regression on the three HAR components
    har = LinearRegression().fit(train[HAR_FEATURES], y_train)
    evaluate("HAR (linear)", y_test, har.predict(test[HAR_FEATURES]))

    # XGBoost on the HAR features plus VIX and cross-asset features
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=SEED,
    )
    model.fit(train[XGB_FEATURES], y_train)
    evaluate("XGBoost", y_test, model.predict(test[XGB_FEATURES]))


if __name__ == "__main__":
    main()
