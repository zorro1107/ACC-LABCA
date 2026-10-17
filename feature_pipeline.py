"""
feature_pipeline.py
-------------------
Transforms raw telemetry measurements into machine-learning features
tailored for time-series CPU utilization forecasting.

Feature Categories Engineered:
1. Lag Features: Historical observations (e.g. CPU at t-1, t-2, t-3)
2. Rolling Statistics: Short & medium-term moving averages and rolling standard deviations
3. Trend & Momentum: First and second-order differences (rate of change)
4. System Pressure Indicators: Memory usage, active process count, disk I/O
5. Time Context Features: Hour, minute, cyclic encodings
6. Target Variable: Future CPU utilization (t + prediction_horizon)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from forecast_config import PREDICTION_HORIZON_SAMPLES


def engineer_features(
    df: pd.DataFrame,
    target_horizon: int = PREDICTION_HORIZON_SAMPLES,
    cpu_col: str = "cpu_usage",
    dropna: bool = True
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Constructs all predictive features from raw telemetry data.

    Parameters:
        df: Raw input DataFrame containing timestamps and metrics.
        target_horizon: Prediction steps ahead (60 steps = five minutes at five-second sampling).
        cpu_col: Name of the CPU utilization column.
        dropna: Whether to remove rows containing NaNs introduced by rolling/lags.

    Returns:
        (df_featured, feature_column_names, target_column_name)
    """
    if target_horizon < 1:
        raise ValueError("target_horizon must be at least one sample")

    data = df.copy()

    # Ensure timestamp is datetime if present
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["hour"] = data["timestamp"].dt.hour
        data["minute"] = data["timestamp"].dt.minute
        # Cyclical encoding of time
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)

    # 1. Autoregressive Lag Features
    # CPU usage at t-1, t-2, t-3, t-5
    for lag in [1, 2, 3, 5]:
        data[f"cpu_lag_{lag}"] = data[cpu_col].shift(lag)

    # 2. Rolling Statistics (Moving Averages and Volatility)
    for window in [3, 5, 10]:
        data[f"cpu_roll_mean_{window}"] = data[cpu_col].shift(1).rolling(window=window).mean()
        data[f"cpu_roll_std_{window}"] = data[cpu_col].shift(1).rolling(window=window).std()

    # 3. Rate of Change / Momentum
    # Current CPU minus previous CPU (t - (t-1)), and velocity (t-1 minus t-2)
    data["cpu_diff_1"] = data[cpu_col] - data[f"cpu_lag_1"]
    data["cpu_diff_2"] = data[f"cpu_lag_1"] - data[f"cpu_lag_2"]
    data["cpu_acceleration"] = data["cpu_diff_1"] - data["cpu_diff_2"]

    # 4. Contextual System Telemetry Lags (if columns exist)
    if "memory_usage" in data.columns:
        data["mem_lag_1"] = data["memory_usage"].shift(1)
        data["mem_diff_1"] = data["memory_usage"] - data["mem_lag_1"]

    if "active_processes" in data.columns:
        data["proc_lag_1"] = data["active_processes"].shift(1)
        data["proc_diff_1"] = data["active_processes"] - data["proc_lag_1"]

    if "disk_io_mb_s" in data.columns:
        data["disk_roll_mean_3"] = data["disk_io_mb_s"].shift(1).rolling(window=3).mean()

    # 5. Define Target Variable: Future CPU at t + target_horizon
    target_col = f"target_cpu_future_{target_horizon}"
    data[target_col] = data[cpu_col].shift(-target_horizon)

    # Exclude non-feature columns
    excluded_cols = {"timestamp", target_col}
    feature_cols = [c for c in data.columns if c not in excluded_cols]

    if dropna:
        data = data.dropna().reset_index(drop=True)

    return data, feature_cols, target_col


def chronological_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    gap_steps: int = 0
):
    """
    Performs a strict chronological split (Time-Series Split) without shuffling
    to prevent look-ahead bias and data leakage.

    Split:
        - Train: First [0% .. 70%]
        - Validation: Middle [70% .. 85%]
        - Test: Final [85% .. 100%]
    """
    if gap_steps < 0:
        raise ValueError("gap_steps cannot be negative")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_stop = train_end - gap_steps
    val_stop = val_end - gap_steps
    if train_stop <= 0 or val_stop <= train_end or val_end >= n:
        raise ValueError("Dataset is too small for the requested chronological split and gap")

    X = df[feature_cols]
    y = df[target_col]

    X_train, y_train = X.iloc[:train_stop], y.iloc[:train_stop]
    X_val, y_val = X.iloc[train_end:val_stop], y.iloc[train_end:val_stop]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"[i] Dataset Split Summary (Total rows: {n}):")
    print(f"    - Training Set:   {len(X_train)} samples ({train_ratio*100:.0f}%)")
    print(f"    - Validation Set: {len(X_val)} samples ({val_ratio*100:.0f}%)")
    print(f"    - Test Set:       {len(X_test)} samples ({(1.0 - train_ratio - val_ratio)*100:.0f}%)")
    if gap_steps:
        print(f"    - Purged boundary rows: {gap_steps} after training and validation")

    return X_train, y_train, X_val, y_val, X_test, y_test
