"""
main.py
-------
Person 1: Machine Learning Model for Predicting CPU Utilization
Complete End-to-End Execution Pipeline.

Workflow:
1. Data Generation: Synthetic dataset generation simulating real OS telemetry.
2. Feature Engineering: Lag features, rolling statistics, rate-of-change, and system indicators.
3. Chronological Dataset Split: 70% Train, 15% Validation, 15% Test (zero look-ahead bias).
4. Model Training & Comparison:
   - Naive Persistence Baseline
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Performance Evaluation: MAE, MSE, RMSE, R2 comparison table & diagnostic plots.
6. Scenario inference demonstration and optional live telemetry inference.
"""

import argparse
import os
import pandas as pd
from synthetic_data_generator import generate_synthetic_cpu_data
from feature_pipeline import engineer_features, chronological_train_val_test_split
from model_trainer import train_and_compare_models
from predictor import CPUPredictor
from forecast_config import (
    MODEL_PATH,
    PREDICTION_HORIZON_SAMPLES,
    PREDICTION_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
)


def run_pipeline(
    mode: str = "synthetic",
    num_samples: int = 5000,
    collect_seconds: int = 30,
    csv_file: str = "cpu_data_synthetic.csv",
    live_demo: bool = False,
):
    print("=" * 80)
    print(" PERSON 1: MACHINE LEARNING MODEL FOR PREDICTING CPU UTILIZATION ")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 1: DATA GENERATION / COLLECTION
    # -------------------------------------------------------------
    print("\n>>> STAGE 1: DATA GENERATION / COLLECTION")
    if mode == "synthetic":
        print(f"[*] Generating synthetic system performance dataset ({num_samples} samples)...")
        raw_df = generate_synthetic_cpu_data(
            num_samples=num_samples,
            interval_seconds=SAMPLE_INTERVAL_SECONDS,
            random_seed=42,
            output_filepath=csv_file
        )
    elif mode == "real":
        from data_collector import collect_system_metrics

        print(f"[*] Collecting real telemetry using psutil for {collect_seconds} seconds...")
        csv_file = "cpu_data_real.csv"
        raw_df = collect_system_metrics(
            duration_seconds=collect_seconds,
            interval_seconds=SAMPLE_INTERVAL_SECONDS,
            output_filepath=csv_file
        )
    elif mode == "existing":
        print(f"[*] Loading existing dataset from {csv_file}...")
        raw_df = pd.read_csv(csv_file)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[+] Dataset Loaded: {len(raw_df)} rows, columns={list(raw_df.columns)}")
    print("\nDataset Snapshot:")
    print(raw_df.head(5))

    # -------------------------------------------------------------
    # STAGE 2: FEATURE ENGINEERING
    # -------------------------------------------------------------
    print("\n>>> STAGE 2: FEATURE ENGINEERING & PREPROCESSING")
    featured_df, feature_cols, target_col = engineer_features(
        raw_df,
        target_horizon=PREDICTION_HORIZON_SAMPLES,
        dropna=True
    )
    print(f"[+] Features Engineered ({len(feature_cols)} features):")
    for idx, feat in enumerate(feature_cols, 1):
        print(f"    {idx:02d}. {feat}")
    print(
        f"[+] Target Variable: '{target_col}' "
        f"({PREDICTION_HORIZON_SECONDS} seconds / {PREDICTION_HORIZON_SAMPLES} samples ahead)"
    )

    # Chronological Split (70% Train, 15% Validation, 15% Test)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_train_val_test_split(
        featured_df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=0.70,
        val_ratio=0.15,
        gap_steps=PREDICTION_HORIZON_SAMPLES
    )

    # -------------------------------------------------------------
    # STAGE 3: MODEL TRAINING & EVALUATION
    # -------------------------------------------------------------
    print("\n>>> STAGE 3: MODEL TRAINING, TUNING & EVALUATION")
    model_artifact_path = MODEL_PATH
    plot_path = "evaluation_results.png"

    results_df, best_model_name = train_and_compare_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_cols=feature_cols,
        save_model_path=model_artifact_path,
        plot_output_path=plot_path
    )

    # -------------------------------------------------------------
    # STAGE 4: MODEL INFERENCE DEMONSTRATIONS
    # -------------------------------------------------------------
    print("\n>>> STAGE 4: MODEL INFERENCE DEMONSTRATION")
    predictor = CPUPredictor(model_artifact_path)

    # Test Case 1: Specific Scenario from user request:
    # "For example, suppose the three latest CPU readings are:
    #  Previous reading: 35%
    #  Current reading minus one interval: 42%
    #  Most recent reading: 48%"
    print("\n[Scenario Test: Escalating CPU Workload]")
    test_pattern = [28.0, 29.0, 30.0, 31.0, 32.0, 35.0, 38.0, 40.0, 42.0, 45.0, 48.0]
    predicted_val = predictor.predict_from_recent_history(test_pattern)
    print(f"Input History : {test_pattern}")
    print(f"Latest CPU    : {test_pattern[-1]}%")
    print(f" Predicted CPU : {predicted_val:.2f}% (Horizon: five minutes)")

    if live_demo:
        print("\n[Live System Telemetry Test]")
        predictor.live_predict()

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY! ")
    print(f" Best Model: {best_model_name}")
    print(f" Saved Model Artifact: {os.path.abspath(model_artifact_path)}")
    print(f" Generated Visualization: {os.path.abspath(plot_path)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Utilization Prediction ML Pipeline")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "real", "existing"],
                        help="Data mode: 'synthetic' (default), 'real' (psutil capture), or 'existing'")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples to generate")
    parser.add_argument("--collect-seconds", type=int, default=30, help="Seconds to record if mode='real'")
    parser.add_argument("--live-demo", action="store_true", help="Also run optional live telemetry inference")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        num_samples=args.samples,
        collect_seconds=args.collect_seconds,
        live_demo=args.live_demo,
    )
