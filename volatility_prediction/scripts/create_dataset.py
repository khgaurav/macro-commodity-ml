import datetime
import os
import pandas as pd
import yfinance as yf

# Configuration
START_DATE = "2003-12-01"
OUTPUT_FILE = os.path.join("..", "data", "daily_commodity_market_data.csv")

# Mappings
YAHOO_TICKERS = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "US Dollar Index": "DX-Y.NYB",
    "S&P 500": "^GSPC",
    "EUR/USD": "EURUSD=X",
    "VIX Index": "^VIX",
    # Implied volatility indices (gold and crude oil), both start mid-2008
    "GVZ Index": "^GVZ",
    "OVX Index": "^OVX"
}

FRED_TICKERS = {
    "10-Year Breakeven Inflation": "T10YIE",
    "US 2-Year Treasury Yields": "DGS2",
    "US 10-Year Treasury Yields": "DGS10"
}

def download_yahoo_data(tickers, start_date, end_date):
    """Download daily prices and volume from Yahoo Finance."""
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, multi_level_index=False)
            if not df.empty:
                print(f"Dataset name: {name}")
                print("Dataset source: Yahoo Finance")
                data[name] = df['Close']
                # Special extraction for Gold Volume and full OHLC.
                if name == "Gold":
                    data["Gold Volume"] = df['Volume']
                    data["Gold Open"] = df['Open']
                    data["Gold High"] = df['High']
                    data["Gold Low"] = df['Low']
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data

def download_fred_data(series_dict, start_date, end_date):
    """Download series from Federal Reserve Economic Data (FRED)."""
    data = {}
    for name, series_id in series_dict.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url)
            
            # Find date column
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            if not date_col:
                print(f"Date column not found in FRED CSV for {series_id}")
                continue
                
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            
            # Filter dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df_filtered = df.loc[start_dt:end_dt]
            
            if not df_filtered.empty:
                print(f"Dataset name: {name}")
                print("Dataset source: Federal Reserve Economic Data (FRED)")
                data[name] = df_filtered[series_id]
        except Exception as e:
            print(f"Error downloading FRED series {series_id}: {e}")
    return data

def merge_and_clean_data(yahoo_data, fred_data, start_date, end_date):
    """Merge Yahoo and FRED datasets without any filling or cleaning."""
    # Set up unified business day index
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    merged_df = pd.DataFrame(index=all_dates)
    
    # Merge Yahoo Finance
    for name, series in yahoo_data.items():
        series = series.groupby(series.index).last()
        merged_df = merged_df.join(series.rename(name), how='left')
        
    # Merge FRED data
    for name, series in fred_data.items():
        series = series.groupby(series.index).last()
        merged_df = merged_df.join(series.rename(name), how='left')
        
    return merged_df

def main():
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    # Fetch Data
    yahoo_data = download_yahoo_data(YAHOO_TICKERS, START_DATE, end_date)
    fred_data = download_fred_data(FRED_TICKERS, START_DATE, end_date)
    
    # Merge Data
    final_df = merge_and_clean_data(yahoo_data, fred_data, START_DATE, end_date)
    
    # Save output
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    final_df.index.name = "Date"
    final_df.to_csv(output_path)

if __name__ == "__main__":
    main()
