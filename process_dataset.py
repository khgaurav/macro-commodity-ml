import os
import pandas as pd
import numpy as np

def log_ret(s, n=1):
    r = s / s.shift(n)
    return np.log(r.mask(r <= 0))

def main():
    input_path = os.path.join(os.path.dirname(__file__), "daily_commodity_market_data.csv")
    output_path = os.path.join(os.path.dirname(__file__), "daily_commodity_market_data_cleaned.csv")
    
    if not os.path.exists(input_path):
        return
        
    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date").sort_index()
    
    required = [
        "Gold", "Gold Volume", "Gold High", "Gold Low", "Crude Oil", "Silver", "Copper", 
        "Platinum", "US Dollar Index", "S&P 500", "EUR/USD", "VIX Index",
        "10-Year Breakeven Inflation", "US 2-Year Treasury Yields", "US 10-Year Treasury Yields"
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
        
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].drop_duplicates()
    
    # Preprocessing: ffill macro columns, drop missing market rows
    macro = ["10-Year Breakeven Inflation", "US 2-Year Treasury Yields", "US 10-Year Treasury Yields"]
    df[macro] = df[macro].ffill()
    df = df.dropna(subset=[c for c in required if c not in macro])
    
    # Build calculated features DataFrame
    features = pd.DataFrame(index=df.index)
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
    
    df_clean = features.replace([np.inf, -np.inf], np.nan).dropna().copy()
    df_clean.to_csv(output_path, index=True)
    print(f"Saved {len(df_clean)} rows to: {output_path}")

if __name__ == "__main__":
    main()
