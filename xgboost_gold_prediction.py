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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "daily_commodity_market_data_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"

TARGET_COLUMN = "tomorrow_gold_log_return"
SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Baseline XGBoost uses early stopping against the chronological validation set.
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

# Tuned XGBoost uses time-series cross-validation on the training set only.
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_data():
    """Load and validate the cleaned commodity-market dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place daily_commodity_market_data_cleaned.csv beside this script."
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    if df.index.has_duplicates:
        raise ValueError("The dataset contains duplicate dates.")

    missing_values = int(df.isna().sum().sum())
    if missing_values:
        raise ValueError(
            f"The cleaned dataset still contains {missing_values} missing values."
        )

    print("Shape:", df.shape)
    print("Start date:", df.index.min())
    print("End date:", df.index.max())
    print("Missing values:", missing_values)
    print("Duplicate dates:", int(df.index.duplicated().sum()))
    print("Target present:", TARGET_COLUMN in df.columns)
    print()

    return df


def chronological_split(df):
    """Split data chronologically into 70% train, 15% validation, and 15% test."""
    feature_columns = [
        column for column in df.columns
        if column != TARGET_COLUMN
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    train_end = int(len(df) * TRAIN_RATIO)
    validation_end = int(
        len(df) * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    X_train = X.iloc[:train_end]
    X_validation = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_validation = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    split_summary = pd.DataFrame({
        "Split": ["Training", "Validation", "Test"],
        "Rows": [
            len(X_train),
            len(X_validation),
            len(X_test),
        ],
        "Start_Date": [
            X_train.index.min(),
            X_validation.index.min(),
            X_test.index.min(),
        ],
        "End_Date": [
            X_train.index.max(),
            X_validation.index.max(),
            X_test.index.max(),
        ],
    })

    print(split_summary.to_string(index=False))
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


# ---------------------------------------------------------------------------
# Model construction and evaluation
# ---------------------------------------------------------------------------

def build_model(early_stopping=False, **params):
    """Create an XGBRegressor with reproducible project-wide settings."""
    settings = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": SEED,
    }

    if early_stopping:
        settings["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    settings.update(params)
    return xgb.XGBRegressor(**settings)


def build_baseline_model():
    """Create the untuned XGBoost reference model."""
    return build_model(
        early_stopping=True,
        n_estimators=N_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def calculate_metrics(
    y_actual,
    y_pred,
    model_name,
    include_directional=True,
):
    """Return MAE, MSE, RMSE, R², and directional accuracy."""
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    if include_directional:
        directional_accuracy = float(
            np.mean(np.sign(y_actual) == np.sign(y_pred))
        )
    else:
        directional_accuracy = np.nan

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
    }


def print_metrics(metric_row, dataset_name="Test"):
    """Print one metric row in a readable format."""
    print(f"{metric_row['Model']} {dataset_name} Results")
    print(f"MAE: {metric_row['MAE']:.8f}")
    print(f"MSE: {metric_row['MSE']:.8f}")
    print(f"RMSE: {metric_row['RMSE']:.8f}")
    print(f"R²: {metric_row['R2']:.6f}")

    directional_accuracy = metric_row["Directional_Accuracy"]
    if not pd.isna(directional_accuracy):
        print(
            "Directional Accuracy: "
            f"{directional_accuracy:.4%}"
        )

    print()


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def sample_unique_tuning_configs():
    """Draw reproducible, unique random configurations."""
    rng = random.Random(SEED)
    unique_configs = []
    seen = set()

    while len(unique_configs) < N_TUNING_CONFIGS:
        config = {
            name: rng.choice(values)
            for name, values in TUNING_SPACE.items()
        }

        signature = tuple(
            (name, config[name])
            for name in sorted(config)
        )

        if signature not in seen:
            seen.add(signature)
            unique_configs.append(config)

    return unique_configs


def run_tuning(X_train, y_train):
    """
    Tune XGBoost with expanding-window time-series cross-validation.

    Only the training set is used here. The validation set remains available
    for feature-set selection, and the test set remains untouched.
    """
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    tuning_configs = sample_unique_tuning_configs()
    tuning_results = []

    print("Running randomized XGBoost tuning")
    print("Selection metric: mean time-series CV RMSE")
    print("=" * 72)

    for config_id, config in enumerate(tuning_configs, start=1):
        fold_rmses = []

        for train_index, validation_index in splitter.split(X_train):
            model = build_model(**config)

            model.fit(
                X_train.iloc[train_index],
                y_train.iloc[train_index],
                verbose=False,
            )

            fold_prediction = model.predict(
                X_train.iloc[validation_index]
            )

            fold_rmse = np.sqrt(
                mean_squared_error(
                    y_train.iloc[validation_index],
                    fold_prediction,
                )
            )

            fold_rmses.append(fold_rmse)

        tuning_results.append({
            "Config_ID": config_id,
            **config,
            "Mean_CV_RMSE": float(np.mean(fold_rmses)),
            "Std_CV_RMSE": float(np.std(fold_rmses)),
        })

        print(
            f"Config {config_id:>2}: "
            f"mean CV RMSE = {np.mean(fold_rmses):.8f}"
        )

    tuning_results_df = pd.DataFrame(
        tuning_results
    ).sort_values(
        by=["Mean_CV_RMSE", "Std_CV_RMSE"],
        ascending=True,
    ).reset_index(drop=True)

    best_row = tuning_results_df.iloc[0]

    parameter_names = list(TUNING_SPACE.keys())
    best_config = {
        parameter: best_row[parameter]
        for parameter in parameter_names
    }

    # Restore integer types that pandas may convert to floats.
    for integer_parameter in [
        "max_depth",
        "min_child_weight",
        "n_estimators",
    ]:
        best_config[integer_parameter] = int(
            best_config[integer_parameter]
        )

    print("\nBest configuration:")
    for parameter, value in best_config.items():
        print(f"- {parameter}: {value}")
    print(
        "Best mean CV RMSE:",
        f"{best_row['Mean_CV_RMSE']:.8f}",
    )
    print()

    return best_config, tuning_results_df


# ---------------------------------------------------------------------------
# Feature importance and feature-set analysis
# ---------------------------------------------------------------------------

def create_feature_importance_table(model, feature_columns):
    """Return gain-based XGBoost feature importance as a sorted table."""
    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    feature_importance_df["Rank"] = (
        np.arange(1, len(feature_importance_df) + 1)
    )

    return feature_importance_df[
        ["Rank", "Feature", "Importance"]
    ]


def compare_feature_sets(
    X_train,
    y_train,
    X_validation,
    y_validation,
    best_config,
    feature_importance_df,
):
    """
    Compare all features, top 10, and top 5 on the validation set.

    Feature rankings are learned from the training set only. The validation
    set is then used to select the final feature set without touching the test
    set.
    """
    ranked_features = feature_importance_df["Feature"].tolist()

    feature_sets = {
        "All Features": ranked_features,
        "Top 10 Features": ranked_features[:10],
        "Top 5 Features": ranked_features[:5],
    }

    comparison_rows = []

    print("Feature-Set Validation Comparison")
    print("=" * 72)

    for feature_set_name, selected_features in feature_sets.items():
        model = build_model(**best_config)

        model.fit(
            X_train[selected_features],
            y_train,
            verbose=False,
        )

        validation_prediction = model.predict(
            X_validation[selected_features]
        )

        metric_row = calculate_metrics(
            y_validation,
            validation_prediction,
            feature_set_name,
        )

        comparison_rows.append({
            "Feature_Set": feature_set_name,
            "Number_of_Features": len(selected_features),
            "Selected_Features": "|".join(selected_features),
            "MAE": metric_row["MAE"],
            "MSE": metric_row["MSE"],
            "RMSE": metric_row["RMSE"],
            "R2": metric_row["R2"],
            "Directional_Accuracy": metric_row[
                "Directional_Accuracy"
            ],
        })

        print(
            f"{feature_set_name:<16} | "
            f"features={len(selected_features):>2} | "
            f"MAE={metric_row['MAE']:.8f} | "
            f"RMSE={metric_row['RMSE']:.8f} | "
            f"R²={metric_row['R2']:.6f} | "
            f"Direction={metric_row['Directional_Accuracy']:.4%}"
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
    ).reset_index(drop=True)

    best_feature_set_name = comparison_df.iloc[0]["Feature_Set"]
    best_feature_set = comparison_df.iloc[0][
        "Selected_Features"
    ].split("|")

    print("\nSelected feature set:", best_feature_set_name)
    print("Selected feature count:", len(best_feature_set))
    print()

    return (
        comparison_df,
        best_feature_set_name,
        best_feature_set,
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_validation_curve(model, model_name):
    """Plot baseline validation RMSE across boosting rounds."""
    evals_result = model.evals_result()
    validation_rmse = evals_result["validation_0"]["rmse"]

    plt.figure(figsize=(10, 5))
    plt.plot(
        validation_rmse,
        label="Validation RMSE",
    )
    plt.axvline(
        model.best_iteration,
        linestyle="--",
        label="Best Iteration",
    )

    plt.xlabel("Boosting Round")
    plt.ylabel("RMSE")
    plt.title(f"{model_name} Validation RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance_df):
    """Plot gain-based importance for all available predictors."""
    plot_data = feature_importance_df.sort_values(
        by="Importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        plot_data["Feature"],
        plot_data["Importance"],
    )

    plt.xlabel("Gain-Based Importance")
    plt.ylabel("Feature")
    plt.title("Tuned XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()


def plot_feature_set_comparison(feature_set_comparison_df):
    """Plot validation RMSE for each tested feature set."""
    plot_data = feature_set_comparison_df.sort_values(
        by="RMSE",
        ascending=True,
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        plot_data["Feature_Set"],
        plot_data["RMSE"],
    )

    plt.xlabel("Feature Set")
    plt.ylabel("Validation RMSE")
    plt.title("XGBoost Feature-Set Comparison")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(test_dates, y_actual, y_pred):
    """Plot actual and tuned XGBoost predictions over the test period."""
    plt.figure(figsize=(14, 6))

    plt.plot(
        test_dates,
        y_actual,
        label="Actual Return",
        linewidth=1,
    )
    plt.plot(
        test_dates,
        y_pred,
        label="Tuned XGBoost Prediction",
        linewidth=1,
    )

    plt.xlabel("Date")
    plt.ylabel("Gold Log Return")
    plt.title("Tuned XGBoost: Actual vs. Predicted Test Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(test_dates, y_actual, y_pred):
    """Plot tuned XGBoost residuals over the test period."""
    residuals = np.asarray(y_actual) - np.asarray(y_pred)

    plt.figure(figsize=(14, 5))
    plt.plot(
        test_dates,
        residuals,
        linewidth=1,
    )
    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.title("Tuned XGBoost Residuals Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def save_results(
    metric_rows,
    test_dates,
    y_actual,
    baseline_prediction,
    tuned_prediction,
    tuning_results_df,
    feature_importance_df,
    feature_set_comparison_df,
    selected_feature_set_name,
    selected_features,
    best_config,
):
    """Save all XGBoost outputs required by the project."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = RESULTS_DIR / "xgboost_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    predictions_df = pd.DataFrame({
        "Date": test_dates,
        "Actual_Return": np.asarray(y_actual),
        "Baseline_XGBoost_Prediction": np.asarray(
            baseline_prediction
        ),
        "Tuned_XGBoost_Prediction": np.asarray(
            tuned_prediction
        ),
    })
    predictions_path = RESULTS_DIR / "xgboost_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    tuning_path = RESULTS_DIR / "xgboost_tuning_results.csv"
    tuning_results_df.to_csv(tuning_path, index=False)
    print(f"Saved tuning results to: {tuning_path}")

    importance_path = (
        RESULTS_DIR / "xgboost_feature_importance.csv"
    )
    feature_importance_df.to_csv(
        importance_path,
        index=False,
    )
    print(f"Saved feature importance to: {importance_path}")

    feature_set_path = (
        RESULTS_DIR / "xgboost_feature_set_comparison.csv"
    )
    feature_set_comparison_df.to_csv(
        feature_set_path,
        index=False,
    )
    print(f"Saved feature-set comparison to: {feature_set_path}")

    selected_features_df = pd.DataFrame({
        "Feature_Set": [selected_feature_set_name] * len(
            selected_features
        ),
        "Feature": selected_features,
    })
    selected_features_path = (
        RESULTS_DIR / "xgboost_selected_features.csv"
    )
    selected_features_df.to_csv(
        selected_features_path,
        index=False,
    )
    print(f"Saved selected features to: {selected_features_path}")

    best_config_df = pd.DataFrame([
        {
            **best_config,
            "Selected_Feature_Set": selected_feature_set_name,
            "Selected_Feature_Count": len(selected_features),
        }
    ])
    best_config_path = (
        RESULTS_DIR / "xgboost_best_configuration.csv"
    )
    best_config_df.to_csv(
        best_config_path,
        index=False,
    )
    print(f"Saved best configuration to: {best_config_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    """Train, tune, evaluate, interpret, and export XGBoost results."""
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

    test_dates = X_test.index
    y_test_actual = y_test.to_numpy()

    # Zero-return test benchmark.
    zero_prediction = np.zeros(len(y_test_actual))
    zero_metrics = calculate_metrics(
        y_test_actual,
        zero_prediction,
        "Zero-Return Baseline",
        include_directional=False,
    )
    print_metrics(zero_metrics)

    # Baseline XGBoost, fitted on training data and early-stopped on validation.
    baseline_model = build_baseline_model()
    baseline_model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    baseline_prediction = baseline_model.predict(X_test)
    baseline_metrics = calculate_metrics(
        y_test_actual,
        baseline_prediction,
        "Baseline XGBoost",
    )
    print_metrics(baseline_metrics)

    # Tune hyperparameters on the training set only.
    best_config, tuning_results_df = run_tuning(
        X_train,
        y_train,
    )

    # Fit the tuned all-feature model on training data only so feature ranking
    # does not use validation or test outcomes.
    tuned_training_model = build_model(**best_config)
    tuned_training_model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    feature_importance_df = create_feature_importance_table(
        tuned_training_model,
        feature_columns,
    )

    (
        feature_set_comparison_df,
        selected_feature_set_name,
        selected_features,
    ) = compare_feature_sets(
        X_train,
        y_train,
        X_validation,
        y_validation,
        best_config,
        feature_importance_df,
    )

    # Refit the final selected model on train + validation data.
    X_dev = pd.concat([X_train, X_validation])
    y_dev = pd.concat([y_train, y_validation])

    tuned_model = build_model(**best_config)
    tuned_model.fit(
        X_dev[selected_features],
        y_dev,
        verbose=False,
    )

    tuned_prediction = tuned_model.predict(
        X_test[selected_features]
    )

    tuned_metrics = calculate_metrics(
        y_test_actual,
        tuned_prediction,
        "Tuned XGBoost",
    )
    print_metrics(tuned_metrics)

    print("Final selected feature set:", selected_feature_set_name)
    print("Final selected features:")
    for feature in selected_features:
        print("-", feature)
    print()

    # Visualizations.
    plot_validation_curve(
        baseline_model,
        "Baseline XGBoost",
    )
    plot_feature_importance(feature_importance_df)
    plot_feature_set_comparison(feature_set_comparison_df)
    plot_actual_vs_predicted(
        test_dates,
        y_test_actual,
        tuned_prediction,
    )
    plot_residuals(
        test_dates,
        y_test_actual,
        tuned_prediction,
    )

    # Project deliverables.
    save_results(
        metric_rows=[
            zero_metrics,
            baseline_metrics,
            tuned_metrics,
        ],
        test_dates=test_dates,
        y_actual=y_test_actual,
        baseline_prediction=baseline_prediction,
        tuned_prediction=tuned_prediction,
        tuning_results_df=tuning_results_df,
        feature_importance_df=feature_importance_df,
        feature_set_comparison_df=feature_set_comparison_df,
        selected_feature_set_name=selected_feature_set_name,
        selected_features=selected_features,
        best_config=best_config,
    )


if __name__ == "__main__":
    main()