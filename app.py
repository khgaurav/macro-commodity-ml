import os
import subprocess
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

CSV = "daily_commodity_market_data_cleaned.csv"

FEATURES = [
    "har_daily", "har_weekly", "har_monthly",
    "gold_return_1d", "gold_return_5d", "gold_return_20d",
    "gold_volatility_5d", "gold_volatility_20d", "gold_close_pos",
    "gold_volume_change", "gold_relative_volume",
    "crude_oil_return_1d", "silver_return_1d", "copper_return_1d",
    "platinum_return_1d", "us_dollar_index_return_1d", "sp_500_return_1d",
    "eur_usd_return_1d", "vix_change_1d", "breakeven_inflation_10y_diff_bps",
    "us_2_year_treasury_yields_diff_bps", "us_10_year_treasury_yields_diff_bps",
    "two_ten_slope", "two_ten_slope_change_bps",
    "gvz_log", "gvz_change", "gvz_5d", "vrp", "ovx_log",
]

PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.7,
              colsample_bytree=0.5, min_child_weight=20, reg_lambda=5.0,
              random_state=42, n_jobs=4)

RISK = {"Careful": 0.05, "Normal": 0.10, "Aggressive": 0.15}


@st.cache_data(ttl=21600, show_spinner="Getting the latest prices")
def refresh():
    try:
        for script in ("create_dataset.py", "process_dataset.py"):
            subprocess.run([sys.executable, script], check=True,
                           capture_output=True, timeout=900)
        return True
    except Exception:
        return False


@st.cache_data
def load(stamp):
    df = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date").sort_index()
    df = df.loc["2008-06-03":].copy()
    df["next_vol_log"] = df["har_daily"].shift(-1)
    df["actual"] = np.exp(df["next_vol_log"])
    df["rough_guess"] = df["gold_return_1d"].rolling(20).std()
    return df.dropna(subset=FEATURES)


@st.cache_resource(show_spinner="Training")
def train(cutoff, stamp):
    df = load(stamp).dropna(subset=["next_vol_log"])
    past = df.loc[df.index < pd.Timestamp.fromordinal(cutoff)]
    model = xgb.XGBRegressor(**PARAMS).fit(past[FEATURES], past["next_vol_log"])
    resid = past["next_vol_log"] - model.predict(past[FEATURES])
    return model, float(np.log(np.mean(np.exp(resid)))), past


def size(vol, target):
    return float(np.clip(target / np.sqrt(252) / vol, 0, 3))


st.set_page_config(page_title="Gold position sizer", page_icon="🪙")
st.title("Gold position sizer")
st.write("Works out how much gold to hold based on how choppy the model reckons "
         "tomorrow will be. Calm forecast means hold more, rough forecast means hold "
         "less, so the risk you carry stays about the same either way.")

got_new = refresh()
stamp = os.path.getmtime(CSV)
data = load(stamp)
dates = data.index

amount = st.sidebar.number_input("Amount to invest", 100, 1_000_000, 1000, 100)
level = st.sidebar.radio("How much risk you want", list(RISK), index=1)
target = RISK[level]
picked = st.sidebar.date_input("Date", dates[-1].date(),
                               min_value=dates[2000].date(), max_value=dates[-1].date())
st.sidebar.caption("The model only sees data from before this date.")
if st.sidebar.button("Fetch new data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
st.sidebar.caption(f"Prices run to {dates[-1].date()}."
                   + ("" if got_new else " Could not reach the data sources just now, "
                                          "so this is the last good copy."))

day = dates[dates.searchsorted(pd.Timestamp(picked), side="right") - 1]
model, bias, past = train(day.toordinal(), stamp)
row = data.loc[[day]]

forecast = float(np.exp(model.predict(row[FEATURES])[0] + bias))
weight = size(forecast, target)
gold = weight * amount
swing = gold * forecast
seen = np.exp(model.predict(past.tail(504)[FEATURES]) + bias)
rougher = float((seen < forecast).mean())

st.subheader(f"Sizing for the session after {day.date()}")
a, b, c, d = st.columns(4)
a.metric("Forecast move", f"{forecast:.2%}", "per day")
b.metric("Put in gold", f"${gold:,.0f}", f"{weight:.0%} of your money")
c.metric("Keep in cash", f"${amount - gold:,.0f}", f"{1 - weight:.0%}")
d.metric("Typical daily swing", f"${swing:,.0f}", "up or down")

if weight > 1:
    st.info("Very calm. The rule would borrow to hold more gold than you put in.")
elif weight < 0.4:
    st.warning("Rough patch coming. Most of your money stays in cash.")

st.write(f"The model reckons gold moves about {forecast:.2%} tomorrow, which is roughly "
         f"{forecast * np.sqrt(252):.0%} a year. That is rougher than {rougher:.0%} of "
         f"the days it has seen in the last two years.")
st.write(f"You asked for around {target:.0%} a year of risk. Holding {weight:.0%} of "
         f"your money in something that moves {forecast:.2%} a day gets you there. "
         f"Double the forecast and the position roughly halves.")

st.write("**Same day, other risk settings**")
st.dataframe(pd.DataFrame([
    {"Setting": k, "Risk a year": f"{v:.0%}",
     "In gold": f"${size(forecast, v) * amount:,.0f}",
     "In cash": f"${amount - size(forecast, v) * amount:,.0f}"}
    for k, v in RISK.items()]), hide_index=True, use_container_width=True)

actual, rough = float(row["actual"].iloc[0]), float(row["rough_guess"].iloc[0])
if not np.isnan(actual):
    st.caption(f"Gold went on to move {actual:.2%}. A plain 20 day average, which "
               f"costs nothing to work out, would have guessed {rough:.2%}.")

tail = data.loc[:day].tail(120)
lines = pd.DataFrame({"Date": tail.index,
                      "forecast": np.exp(model.predict(tail[FEATURES]) + bias),
                      "actual": tail["actual"].values}).melt(
    "Date", var_name="line", value_name="move")
st.altair_chart(alt.Chart(lines).mark_line().encode(
    x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title=None)),
    y=alt.Y("move:Q", axis=alt.Axis(format="%", title="daily move"),
            scale=alt.Scale(zero=False)),
    color=alt.Color("line:N", legend=alt.Legend(title=None)),
    tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), "line:N",
             alt.Tooltip("move:Q", format=".2%")],
).properties(height=300).interactive(), use_container_width=True)
st.caption("Scroll to zoom and drag to pan. The forecast follows the level and the "
           "turns but does not chase one day spikes. Course project, not investment "
           "advice. It sizes risk and does not try to guess which way the price goes.")
