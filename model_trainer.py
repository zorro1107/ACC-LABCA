"""
model_trainer.py
----------------
Trains and evaluates multiple Machine Learning algorithms for CPU utilization forecasting:
1. Naive Persistence Baseline (y_hat_t = CPU_t)
2. Linear Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor

Computes MAE, MSE, RMSE, and R2 across chronologically split validation and test sets,
plots comparative diagnostic charts, and persists the top-performing model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for environments without a display
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class NaiveBaselineModel:
    """
    Baseline predictor that assumes future CPU utilization will simply
    equal the current observation (cpu_usage).
    """
    def __init__(self, current_cpu_col="cpu_usage"):
        self.current_cpu_col = current_cpu_col

    def fit(self, X, y):
        # No fitting needed for persistence model
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and self.current_cpu_col in X.columns:
            return X[self.current_cpu_col].values
        # Fallback if first column is current cpu
        return X.iloc[:, 0].values


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """
    Calculates standard regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def train_and_compare_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_cols,
    save_model_path="best_cpu_predictor.joblib",
    plot_output_path="evaluation_results.png"
):
    """
    Trains multiple models, evaluates them on validation and test splits,
    saves the best model, and generates visualization charts.
    """
    models = {
        "Naive Persistence Baseline": NaiveBaselineModel(),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
    }

    results = []
    trained_models = {}
    test_predictions = {}

    print("\n" + "=" * 70)
    print(" MODEL TRAINING AND EVALUATION PIPELINE ")
    print("=" * 70)

    for name, model in models.items():
        print(f"[*] Training: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on Validation set
        val_preds = model.predict(X_val)
        val_metrics = evaluate_model(y_val, val_preds, name)

        # Evaluate on Test set (held-out final chronological period)
        test_preds = model.predict(X_test)
        test_predictions[name] = test_preds
        test_metrics = evaluate_model(y_test, test_preds, name)

        results.append({
            "Model": name,
            "Val MAE": val_metrics["MAE"],
            "Val RMSE": val_metrics["RMSE"],
            "Val R2": val_metrics["R2"],
            "Test MAE": test_metrics["MAE"],
            "Test MSE": test_metrics["MSE"],
            "Test RMSE": test_metrics["RMSE"],
            "Test R2": test_metrics["R2"],
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS TABLE ")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Identify best model based on lowest Test RMSE (excluding Naive Baseline for saving)
    ml_results = results_df[results_df["Model"] != "Naive Persistence Baseline"]
    best_row = ml_results.loc[ml_results["Test RMSE"].idxmin()]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    print(f"\n[+] Best Model Selected: {best_model_name}")
    print(f"    - Test MAE:  {best_row['Test MAE']} %")
    print(f"    - Test RMSE: {best_row['Test RMSE']} %")
    print(f"    - Test R2:   {best_row['Test R2']}")

    # Save the best model along with metadata and feature list
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_cols": feature_cols,
        "test_metrics": best_row.to_dict(),
    }
    joblib.dump(artifact, save_model_path)
    print(f"[+] Serialized model and metadata saved to: {save_model_path}")

    # Also save standard pickle files for easy sharing
    import pickle
    pkl_model_path = "cpu_utilization_model.pkl"
    pkl_bundle_path = "cpu_utilization_model_bundle.pkl"
    with open(pkl_model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(pkl_bundle_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_model_name,
            "feature_names": feature_cols,
            "test_metrics": best_row.to_dict()
        }, f)
    print(f"[+] Standalone pickle model saved to: {pkl_model_path}")
    print(f"[+] Complete pickle bundle saved to:  {pkl_bundle_path}")

    # Plot Diagnostics & Comparisons
    _plot_diagnostics(
        y_test=y_test,
        predictions=test_predictions,
        best_model_name=best_model_name,
        best_model=best_model,
        feature_cols=feature_cols,
        output_path=plot_output_path
    )

    return results_df, best_model_name


def _plot_diagnostics(
    y_test,
    predictions,
    best_model_name,
    best_model,
    feature_cols,
    output_path
):
    """
    Creates comprehensive visual plots:
    1. Actual vs. Predicted time-series snippet
    2. Residual error distribution
    3. Feature Importances (if applicable)
    4. Model comparison bar chart (RMSE)
    """
    plt.figure(figsize=(16, 12))

    # Plot 1: Time Series Forecast Comparison (first 120 test steps)
    plt.subplot(2, 2, 1)
    slice_len = min(120, len(y_test))
    indices = np.arange(slice_len)
    actual_slice = y_test.iloc[:slice_len].values
    plt.plot(indices, actual_slice, label="Actual CPU Usage (%)", color="black", linewidth=2.0)
    plt.plot(indices, predictions["Naive Persistence Baseline"][:slice_len],
             label="Naive Baseline", color="gray", linestyle="--", alpha=0.7)
    plt.plot(indices, predictions[best_model_name][:slice_len],
             label=f"{best_model_name} (Best)", color="royalblue", linewidth=1.8)

    plt.title("Actual vs. Predicted CPU Utilization (Test Set Snippet)", fontsize=12, fontweight="bold")
    plt.xlabel("Time Step (Intervals)", fontsize=10)
    plt.ylabel("CPU Utilization (%)", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Plot 2: Residual Distribution for Best Model
    plt.subplot(2, 2, 2)
    residuals = y_test.values - predictions[best_model_name]
    plt.hist(residuals, bins=35, color="steelblue", edgecolor="black", alpha=0.85)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title(f"Residual Distribution ({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Prediction Error (Actual - Predicted %)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Feature Importances
    plt.subplot(2, 2, 3)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:]  # top 10 features
        plt.barh(range(len(indices)), importances[indices], color="teal", align="center")
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.title(f"Top 10 Feature Importances ({best_model_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=10)
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_)
        indices = np.argsort(coefs)[-10:]
        plt.barh(range(len(indices)), coefs[indices], color="teal", align="center")
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.title(f"Top 10 Feature Coefficients ({best_model_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Absolute Coefficient Value", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 4: Model Comparison (Test RMSE)
    plt.subplot(2, 2, 4)
    model_names = list(predictions.keys())
    rmses = [np.sqrt(mean_squared_error(y_test, predictions[m])) for m in model_names]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"][:len(model_names)]
    bars = plt.bar(model_names, rmses, color=colors, edgecolor="black", width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.title("Model Performance Comparison (Test RMSE - Lower is Better)", fontsize=12, fontweight="bold")
    plt.ylabel("Test RMSE (%)", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Diagnostic evaluation plot saved to: {output_path}")
