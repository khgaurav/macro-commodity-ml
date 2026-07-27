 
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
 
SEED = 42
TARGET = "tomorrow_gold_log_return"
ALPHA = 0.000215  # same tuned alpha selected in lasso_gold_prediction.ipynb
 
DATA_PATH = "daily_commodity_market_data_cleaned.csv"
RESULTS_DIR = Path("results")
 
np.random.seed(SEED)
 
# ---------------------------------------------------------------------------
# 1. Load data and define feature groups
# ---------------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = data.sort_index()
 
GOLD_FEATURES = [
    "gold_return_1d", "gold_return_5d", "gold_return_20d",
    "gold_volatility_5d", "gold_volatility_20d", "gold_close_pos",
    "gold_volume_change", "gold_relative_volume",
]
 
COMMODITY_FEATURES = [
    "crude_oil_return_1d", "silver_return_1d",
    "copper_return_1d", "platinum_return_1d",
]
 
MACRO_FEATURES = [
    "us_dollar_index_return_1d", "sp_500_return_1d", "eur_usd_return_1d",
    "vix_change_1d", "breakeven_inflation_10y_diff_bps",
    "us_2_year_treasury_yields_diff_bps", "us_10_year_treasury_yields_diff_bps",
    "two_ten_slope", "two_ten_slope_change_bps",
]
 
ALL_FEATURES = GOLD_FEATURES + COMMODITY_FEATURES + MACRO_FEATURES
assert set(ALL_FEATURES) == set(data.columns) - {TARGET}, \
    "Feature groups don't reconstruct the full 21-column set — check for a typo."
 
FEATURE_SETS = {
    "Gold-Only":              GOLD_FEATURES,
    "Gold + Commodities":     GOLD_FEATURES + COMMODITY_FEATURES,
    "Gold + Macro":           GOLD_FEATURES + MACRO_FEATURES,
    "All 21 Features":        ALL_FEATURES,
}
 
# ---------------------------------------------------------------------------
# 2. Chronological 70/15/15 split (identical convention to every other
#    notebook in this project)
# ---------------------------------------------------------------------------
train_end = int(len(data) * 0.70)
validation_end = int(len(data) * 0.85)
 
y_train = data[TARGET].iloc[:train_end]
y_validation = data[TARGET].iloc[train_end:validation_end]
 
print(f"Training rows:   {train_end}")
print(f"Validation rows: {validation_end - train_end}")
print()
 
# ---------------------------------------------------------------------------
# 3. Train Lasso (fixed, already-tuned alpha) on each feature set, evaluate
#    on the validation set
# ---------------------------------------------------------------------------
rows = []
 
for feature_set_name, columns in FEATURE_SETS.items():
    X_train = data[columns].iloc[:train_end]
    X_validation = data[columns].iloc[train_end:validation_end]
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
 
    model = Lasso(alpha=ALPHA, random_state=SEED)
    model.fit(X_train_scaled, y_train)
 
    predictions = model.predict(X_validation_scaled)
 
    mae = mean_absolute_error(y_validation, predictions)
    mse = mean_squared_error(y_validation, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_validation, predictions)
    directional_accuracy = np.mean(
        np.sign(predictions) == np.sign(y_validation.to_numpy())
    )
    non_zero_coefficients = int(np.sum(model.coef_ != 0))
 
    rows.append({
        "Feature_Set": feature_set_name,
        "Number_of_Features": len(columns),
        "Non_Zero_Coefficients": non_zero_coefficients,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
    })
 
    print(f"{feature_set_name:<20} ({len(columns):>2} features)  "
          f"RMSE={rmse:.6f}  R2={r2:.6f}  DirAcc={directional_accuracy:.4f}")
 
results = pd.DataFrame(rows).sort_values(by="RMSE").reset_index(drop=True)
 
# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIR / "research_question_ablation.csv"
results.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")