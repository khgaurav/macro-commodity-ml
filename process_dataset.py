"""Clean the raw market data and engineer features for the LSTM and XGBoost models."""

import os

import numpy as np
import pandas as pd

HORIZONS = (5, 22)  # volatility forecast horizons, in trading days
PARK_HORIZONS = (5,)  # Parkinson is only needed for the h=5 estimator ablation


def log_ret(s, n=1):
    r = s / s.shift(n)
    return np.log(r.mask(r <= 0))


def parkinson_vol(high, low):
    """Daily volatility from the high-low range. Ignores the overnight gap."""
    return np.log(high / low) / (2.0 * np.sqrt(np.log(2.0)))


def yang_zhang_vol(open_, high, low, close):
    """Daily volatility: squared overnight return plus Rogers-Satchell intraday variance."""
    rs_var = (np.log(high / close) * np.log(high / open_)
              + np.log(low / close) * np.log(low / open_))
    overnight = np.log(open_ / close.shift(1))
    return np.sqrt((overnight ** 2 + rs_var.clip(lower=0)).clip(lower=0))


def har_block(vol, prefix, horizons):
    """HAR components and forward targets, in logs."""
    vol = vol.mask(vol <= 0)
    out = {}
    out[f"{prefix}har_daily"] = np.log(vol)
    out[f"{prefix}har_weekly"] = np.log(vol.rolling(5, min_periods=3).mean())
    out[f"{prefix}har_monthly"] = np.log(vol.rolling(22, min_periods=15).mean())
    for h in horizons:
        # Value at t covers days t+1 .. t+h, so no same-day information leaks in
        fwd = vol.rolling(h, min_periods=max(2, h // 3)).mean().shift(-h)
        out[f"{prefix}vol_target_{h}d"] = np.log(fwd)
        # Trailing mean over the same window: known at t, used as the persistence baseline
        out[f"{prefix}vol_current_{h}d"] = np.log(
            vol.rolling(h, min_periods=max(2, h // 3)).mean())
    return out


def main():
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "daily_commodity_market_data.csv")
    output_path = os.path.join(base, "daily_commodity_market_data_cleaned.csv")

    if not os.path.exists(input_path):
        return

    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date").sort_index()

    required = [
        "Gold", "Gold Volume", "Gold Open", "Gold High", "Gold Low", "Crude Oil",
        "Silver", "Copper", "Platinum", "US Dollar Index", "S&P 500", "EUR/USD",
        "VIX Index", "10-Year Breakeven Inflation", "US 2-Year Treasury Yields",
        "US 10-Year Treasury Yields"
    ]
    # Kept out of `required` so requiring them does not delete the pre-2008 LSTM rows
    implied = ["GVZ Index", "OVX Index"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    present_implied = [c for c in implied if c in df.columns]
    df[required + present_implied] = df[required + present_implied].apply(
        pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].drop_duplicates()

    # Preprocessing: ffill macro columns, drop missing market rows
    macro = ["10-Year Breakeven Inflation", "US 2-Year Treasury Yields",
             "US 10-Year Treasury Yields"]
    df[macro] = df[macro].ffill()
    df = df.dropna(subset=[c for c in required if c not in macro])

    features = pd.DataFrame(index=df.index)

    # --- Return-prediction target and the 21 macro / cross-asset features ---
    features["tomorrow_gold_log_return"] = log_ret(df["Gold"]).shift(-1)

    features["gold_return_1d"] = log_ret(df["Gold"])
    features["gold_return_5d"] = log_ret(df["Gold"], 5)
    features["gold_return_20d"] = log_ret(df["Gold"], 20)
    features["gold_volatility_5d"] = features["gold_return_1d"].rolling(5).std()
    features["gold_volatility_20d"] = features["gold_return_1d"].rolling(20).std()

    denom = df["Gold High"] - df["Gold Low"]
    features["gold_close_pos"] = ((df["Gold"] - df["Gold Low"]) / denom.replace(0, np.nan)).fillna(0.5).clip(0, 1)
    features["gold_volume_change"] = np.log((df["Gold Volume"] + 1) / (df["Gold Volume"].shift(1) + 1))
    features["gold_relative_volume"] = df["Gold Volume"] / df["Gold Volume"].rolling(20).mean()

    assets = {
        "Crude Oil": "crude_oil_return_1d", "Silver": "silver_return_1d",
        "Copper": "copper_return_1d", "Platinum": "platinum_return_1d",
        "US Dollar Index": "us_dollar_index_return_1d", "S&P 500": "sp_500_return_1d",
        "EUR/USD": "eur_usd_return_1d"
    }
    for col, name in assets.items():
        features[name] = log_ret(df[col])

    features["vix_change_1d"] = df["VIX Index"].diff()

    features["breakeven_inflation_10y_diff_bps"] = df["10-Year Breakeven Inflation"].diff() * 100
    features["us_2_year_treasury_yields_diff_bps"] = df["US 2-Year Treasury Yields"].diff() * 100
    features["us_10_year_treasury_yields_diff_bps"] = df["US 10-Year Treasury Yields"].diff() * 100

    two_ten_slope = df["US 10-Year Treasury Yields"] - df["US 2-Year Treasury Yields"]
    features["two_ten_slope"] = two_ten_slope
    features["two_ten_slope_change_bps"] = two_ten_slope.diff() * 100

    # --- Volatility targets and HAR features ---
    valid = (df["Gold High"] > df["Gold Low"]) & (df["Gold Open"] > 0) & (df["Gold"] > 0)

    yz = yang_zhang_vol(df["Gold Open"], df["Gold High"], df["Gold Low"], df["Gold"]).where(valid)
    park = parkinson_vol(df["Gold High"], df["Gold Low"]).where(valid)

    for name, value in har_block(yz, "", HORIZONS).items():
        features[name] = value
    # Parkinson equivalents, kept only so the estimator choice can be ablated
    for name, value in har_block(park, "park_", PARK_HORIZONS).items():
        features[name] = value

    # --- Implied-volatility block ---
    if "GVZ Index" in df.columns:
        gvz = df["GVZ Index"].ffill(limit=5)
        features["gvz_log"] = np.log(gvz.mask(gvz <= 0))
        features["gvz_change"] = log_ret(gvz)
        features["gvz_5d"] = np.log(gvz.rolling(5).mean().mask(lambda s: s <= 0))
        # Variance risk premium: implied (annualised %) rebased to a daily sigma, minus realised
        features["vrp"] = np.log(gvz.mask(gvz <= 0) / 100 / np.sqrt(252)) - features["har_weekly"]
    if "OVX Index" in df.columns:
        ovx = df["OVX Index"].ffill(limit=5)
        features["ovx_log"] = np.log(ovx.mask(ovx <= 0))

    features = features.replace([np.inf, -np.inf], np.nan)

    # Forward-looking targets are excluded from the gate so the newest day survives
    vol_prefixes = ("har_", "park_", "vol_", "gvz_", "ovx_", "vrp")
    lstm_columns = [c for c in features.columns
                    if not c.startswith(vol_prefixes)
                    and c != "tomorrow_gold_log_return"]
    df_clean = features.dropna(subset=lstm_columns).copy()
    df_clean.to_csv(output_path, index=True)

    print(f"Saved {len(df_clean)} rows to: {output_path}")


if __name__ == "__main__":
    main()
