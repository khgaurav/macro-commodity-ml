import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "daily_commodity_market_data_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
TARGET_COLUMN = "tomorrow_gold_log_return"
SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Baseline model: a high estimator ceiling paired with early stopping lets it pick its own tree count from the validation set.
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

# Tuned model: hyperparameters are chosen by time-series cross-validation on the training+validation data, which is far more robust on this noisy target than trusting a single validation split. n_estimators is searched directly (no early stopping) so every CV fold and the final refit use the same tree count.
N_SPLITS = 5
N_TUNING_CONFIGS = 50
TUNING_SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10, 20, 50],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "n_estimators": [100, 200, 300, 500, 800],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0, 5.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
}

random.seed(SEED)
np.random.seed(SEED)


def load_data():
    """Load the cleaned commodity market dataset, indexed by date."""
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    print("Shape:", df.shape)
    print("Start date:", df.index.min())
    print("End date:", df.index.max())
    print("Missing values:", df.isna().sum().sum())
    print("Target present:", TARGET_COLUMN in df.columns)
    print()

    return df


def chronological_split(df):
    """Split the data chronologically into training, validation, and test sets."""
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    train_end = int(len(df) * TRAIN_RATIO)
    validation_end = int(len(df) * (TRAIN_RATIO + VALIDATION_RATIO))

    X_train = X.iloc[:train_end]
    X_validation = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_validation = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Testing rows:", len(X_test))

    print("\nTraining period:")
    print(X_train.index.min(), "to", X_train.index.max())
    print("Testing period:")
    print(X_test.index.min(), "to", X_test.index.max())
    print()

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_columns,
    )


def build_model(early_stopping=False, **params):
    """Create an XGBRegressor, optionally early-stopping on a validation set."""
    settings = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )
    if early_stopping:
        settings["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    return xgb.XGBRegressor(**settings, **params)


def build_baseline_model():
    """XGBoost with sensible default hyperparameters, the reference to beat."""
    return build_model(
        early_stopping=True,
        n_estimators=N_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def evaluate_predictions(y_actual, y_pred, model_name, include_directional=True):
    """Compute and print regression metrics; return them as a dict row."""
    mae = mean_absolute_error(
        y_actual,
        y_pred,
    )

    mse = mean_squared_error(
        y_actual,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_actual,
        y_pred,
    )

    # Directional accuracy is meaningless for the all-zero baseline, so skip it
    if include_directional:
        directional_accuracy = np.mean(
            np.sign(y_actual) == np.sign(y_pred)
        )
    else:
        directional_accuracy = np.nan

    print(f"{model_name} Test Results")
    print(f"MAE: {mae:.8f}")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"R²: {r2:.4f}")
    if include_directional:
        print(
            f"Directional Accuracy: "
            f"{directional_accuracy:.4%}"
        )
    print()

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
    }


def run_tuning(X_dev, y_dev):
    """Rank random configs by time-series cross-validated RMSE on the dev set."""
    rng = random.Random(SEED)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    tuning_configs = {}
    tuning_results = []

    print("Running randomized search with time-series cross-validation")
    print("=" * 70)

    for config_id in range(1, N_TUNING_CONFIGS + 1):
        config = {
            name: rng.choice(values)
            for name, values in TUNING_SPACE.items()
        }

        # Score the config as the mean RMSE across expanding-window folds
        fold_rmses = []
        for train_index, validation_index in splitter.split(X_dev):
            model = build_model(**config)
            model.fit(
                X_dev.iloc[train_index],
                y_dev.iloc[train_index],
                verbose=False,
            )
            fold_prediction = model.predict(X_dev.iloc[validation_index])
            fold_rmses.append(
                np.sqrt(mean_squared_error(
                    y_dev.iloc[validation_index],
                    fold_prediction,
                ))
            )

        mean_cv_rmse = float(np.mean(fold_rmses))
        tuning_configs[config_id] = config
        tuning_results.append({
            "config_id": config_id,
            **config,
            "mean_cv_rmse": mean_cv_rmse,
        })

        print(f"Config {config_id:>2}: mean CV RMSE {mean_cv_rmse:.8f}")

    tuning_results_df = pd.DataFrame(tuning_results).sort_values(
        "mean_cv_rmse"
    ).reset_index(drop=True)

    best_config_id = int(tuning_results_df.iloc[0]["config_id"])
    best_config = tuning_configs[best_config_id]

    print()
    return best_config, tuning_results_df


def plot_validation_curve(model, model_name):
    """Plot a model's validation RMSE across boosting rounds (early stopping)."""
    evals_result = model.evals_result()
    validation_rmse = evals_result["validation_0"]["rmse"]

    plt.figure(figsize=(10, 5))

    plt.plot(
        validation_rmse,
        label="Validation RMSE"
    )
    plt.axvline(
        model.best_iteration,
        color="red",
        linestyle="--",
        label="Best Iteration"
    )

    plt.xlabel("Boosting Round")
    plt.ylabel("Root Mean Squared Error")
    plt.title(f"{model_name} Validation RMSE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_columns):
    """Plot XGBoost gain-based feature importance."""
    importances = model.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(10, 8))

    plt.barh(
        np.array(feature_columns)[order],
        importances[order]
    )

    plt.xlabel("Importance")
    plt.title("XGBoost: Feature Importance")

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(test_dates, y_actual, y_pred):
    """Plot actual versus predicted gold returns over the test period."""
    plt.figure(figsize=(14, 6))

    plt.plot(
        test_dates,
        y_actual,
        label="Actual",
        linewidth=1
    )
    plt.plot(
        test_dates,
        y_pred,
        label="Predicted",
        linewidth=1
    )

    plt.xlabel("Date")
    plt.ylabel("Gold Log Return")
    plt.title("Tuned XGBoost: Actual vs. Predicted Test Returns")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_residuals(test_dates, y_actual, y_pred):
    """Plot prediction residuals over the test period."""
    residuals = y_actual - y_pred

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_dates,
        residuals,
        linewidth=1
    )
    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.title("Tuned XGBoost: Prediction Residuals Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_results(
    metric_rows,
    test_dates,
    y_actual,
    baseline_prediction,
    tuned_prediction,
    tuning_results_df
):
    """Write metrics, predictions, and tuning results to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = RESULTS_DIR / "xgboost_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    predictions_df = pd.DataFrame({
        "Date": test_dates,
        "Actual_Return": y_actual,
        "Baseline_XGBoost_Prediction": baseline_prediction,
        "Tuned_XGBoost_Prediction": tuned_prediction
    })
    predictions_path = RESULTS_DIR / "xgboost_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    tuning_path = RESULTS_DIR / "xgboost_tuning_results.csv"
    tuning_results_df.to_csv(tuning_path, index=False)
    print(f"Saved tuning results to: {tuning_path}")


def main():
    """Train and evaluate the XGBoost gold-return models, then save results."""
    df = load_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_columns,
    ) = chronological_split(df)

    # XGBoost is scale-invariant (tree splits depend only on feature ordering),
    # so unlike the LSTM pipeline no feature or target scaling is applied.
    test_dates = X_test.index
    y_test_actual = y_test.to_numpy()

    # Naive baseline: predict no change in gold for every test observation
    zero_prediction = np.zeros(len(y_test_actual))
    zero_metrics = evaluate_predictions(
        y_test_actual,
        zero_prediction,
        "Zero-Return Baseline",
        include_directional=False,
    )

    # Baseline XGBoost with sensible default hyperparameters
    baseline_model = build_baseline_model()
    baseline_model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    baseline_prediction = baseline_model.predict(X_test)
    baseline_metrics = evaluate_predictions(
        y_test_actual,
        baseline_prediction,
        "Baseline XGBoost",
    )

    # Tuned XGBoost: hyperparameters chosen by time-series cross-validation on
    # the combined training+validation data for a robust, less overfit choice.
    X_dev = pd.concat([X_train, X_validation])
    y_dev = pd.concat([y_train, y_validation])

    best_config, tuning_results_df = run_tuning(X_dev, y_dev)
    print("Best configuration (time-series CV):")
    print(best_config)
    print()

    # Refit on train+validation so the most recent, most regime-relevant
    # observations are included before scoring the held-out test set.
    tuned_model = build_model(**best_config)
    tuned_model.fit(
        X_dev,
        y_dev,
        verbose=False,
    )
    tuned_prediction = tuned_model.predict(X_test)
    tuned_metrics = evaluate_predictions(
        y_test_actual,
        tuned_prediction,
        "Tuned XGBoost",
    )

    # Visualize training behavior, feature importance, and test predictions
    plot_validation_curve(baseline_model, "Baseline XGBoost")
    plot_feature_importance(tuned_model, feature_columns)
    plot_actual_vs_predicted(test_dates, y_test_actual, tuned_prediction)
    plot_residuals(test_dates, y_test_actual, tuned_prediction)

    save_results(
        [zero_metrics, baseline_metrics, tuned_metrics],
        test_dates,
        y_test_actual,
        baseline_prediction,
        tuned_prediction,
        tuning_results_df,
    )


if __name__ == "__main__":
    main()
