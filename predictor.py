"""
predictor.py
------------
Inference engine that loads the trained CPU utilization forecasting model
and generates future predictions.

Supports two operational modes:
1. Scenario / Manual Mode: Pass historical readings (e.g. 35%, 42%, 48%)
2. Live System Inference Mode: Samples real-time metrics using psutil, constructs
   feature vector on the fly, and predicts future CPU load.
"""

import math
import pickle
import time
from collections.abc import Mapping
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from feature_pipeline import engineer_features
from forecast_config import (
    MIN_HISTORY_SAMPLES,
    MODEL_PATH,
    PREDICTION_HORIZON_SAMPLES,
    PREDICTION_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
)


class CPUPredictor:
    def __init__(self, model_artifact_path: Union[str, Path] = MODEL_PATH):
        """
        Loads the trained model artifact containing model, metadata, and required features.
        """
        artifact_path = Path(model_artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}. "
                "Train it with `python main.py --mode synthetic --samples 5000`."
            )

        try:
            artifact = joblib.load(artifact_path)
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            ValueError,
            IndexError,
            KeyError,
            TypeError,
            AttributeError,
            ImportError,
        ) as exc:
            raise RuntimeError(f"Could not load model artifact {artifact_path}: it is corrupt or incompatible") from exc

        if not isinstance(artifact, Mapping):
            raise ValueError(f"Incompatible model artifact {artifact_path}: expected a metadata dictionary")
        required_keys = {
            "model",
            "model_name",
            "feature_cols",
            "test_metrics",
            "sampling_interval_seconds",
            "prediction_horizon_seconds",
            "prediction_horizon_samples",
        }
        missing_keys = required_keys.difference(artifact)
        if missing_keys:
            raise ValueError(
                f"Incompatible model artifact {artifact_path}: missing metadata {sorted(missing_keys)}"
            )
        expected_metadata = {
            "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "prediction_horizon_seconds": PREDICTION_HORIZON_SECONDS,
            "prediction_horizon_samples": PREDICTION_HORIZON_SAMPLES,
        }
        if any(type(artifact[key]) is not int for key in expected_metadata):
            raise ValueError(f"Incompatible model artifact {artifact_path}: invalid sampling/horizon metadata")
        mismatches = {
            key: artifact[key]
            for key, expected in expected_metadata.items()
            if artifact[key] != expected
        }
        if mismatches:
            raise ValueError(
                f"Incompatible model artifact {artifact_path}: expected sampling/horizon metadata "
                f"{expected_metadata}, got {mismatches}"
            )

        feature_cols = artifact["feature_cols"]
        if (
            not isinstance(feature_cols, (list, tuple))
            or not feature_cols
            or any(not isinstance(column, str) for column in feature_cols)
        ):
            raise ValueError(f"Incompatible model artifact {artifact_path}: invalid feature_cols")
        if not callable(getattr(artifact["model"], "predict", None)):
            raise ValueError(f"Incompatible model artifact {artifact_path}: model has no predict method")
        model_feature_count = getattr(artifact["model"], "n_features_in_", len(feature_cols))
        if model_feature_count != len(feature_cols):
            raise ValueError(f"Incompatible model artifact {artifact_path}: feature count does not match model")
        model_feature_names = getattr(artifact["model"], "feature_names_in_", None)
        if model_feature_names is not None and list(model_feature_names) != list(feature_cols):
            raise ValueError(f"Incompatible model artifact {artifact_path}: feature names do not match model")
        if not isinstance(artifact["model_name"], str) or not isinstance(artifact["test_metrics"], Mapping):
            raise ValueError(f"Incompatible model artifact {artifact_path}: invalid model metadata")

        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.feature_cols = list(feature_cols)
        self.test_metrics = artifact["test_metrics"]
        self.model_artifact_path = artifact_path
        print(f"[+] Loaded '{self.model_name}' from {artifact_path}")

    def predict_from_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Runs model inference on a DataFrame that matches the trained feature set.
        """
        # Ensure all required features are present
        missing = [col for col in self.feature_cols if col not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = feature_df[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.asarray(preds, dtype=float)
        if preds.ndim != 1 or preds.size != len(feature_df) or not np.isfinite(preds).all():
            raise RuntimeError("Model returned an invalid or non-finite prediction")
        return np.clip(preds, 0.0, 100.0)

    def predict_from_recent_history(
        self,
        cpu_history: Sequence[float],
        memory_history: Optional[List[float]] = None,
        process_history: Optional[List[int]] = None,
        disk_history: Optional[List[float]] = None
    ) -> float:
        """
        Takes at least 11 CPU readings at five-second intervals and forecasts
        CPU utilization five minutes after the latest reading. When no other
        telemetry is provided, memory, process count, and disk rate default to
        50%, 100, and 1 MB/s respectively.

        Example:
            cpu_history = [28, 29, 30, 31, 32, 35, 38, 40, 42, 45, 48]
        """
        cpu_history = _validate_history(cpu_history)

        n = len(cpu_history)
        if memory_history is None:
            memory_history = [50.0] * n
        elif len(memory_history) < n:
            memory_history = [memory_history[0]] * (n - len(memory_history)) + list(memory_history)

        if process_history is None:
            process_history = [100] * n
        elif len(process_history) < n:
            process_history = [process_history[0]] * (n - len(process_history)) + list(process_history)

        if disk_history is None:
            disk_history = [1.0] * n
        elif len(disk_history) < n:
            disk_history = [disk_history[0]] * (n - len(disk_history)) + list(disk_history)

        # Build mini dataframe to compute identical features
        raw_df = pd.DataFrame({
            "timestamp": pd.date_range(
                "2000-01-01",
                periods=n,
                freq=f"{SAMPLE_INTERVAL_SECONDS}s",
            ),
            "cpu_usage": cpu_history,
            "memory_usage": memory_history,
            "active_processes": process_history,
            "disk_io_mb_s": disk_history
        })

        featured_df, _, _ = engineer_features(
            raw_df,
            target_horizon=PREDICTION_HORIZON_SAMPLES,
            dropna=False,
        )

        # Take the most recent row
        latest_row = featured_df.iloc[-1:]

        pred = self.predict_from_features(latest_row)[0]
        return round(float(pred), 2)

    def live_predict(
        self,
        sampling_steps: int = MIN_HISTORY_SAMPLES,
        interval_seconds: int = SAMPLE_INTERVAL_SECONDS,
    ) -> float:
        """
        Monitors live OS metrics using psutil for a short warmup window,
        builds features, and forecasts future CPU utilization.
        """
        import psutil

        if sampling_steps < MIN_HISTORY_SAMPLES:
            raise ValueError(f"sampling_steps must be at least {MIN_HISTORY_SAMPLES}")
        if interval_seconds != SAMPLE_INTERVAL_SECONDS:
            raise ValueError(
                f"interval_seconds must be {SAMPLE_INTERVAL_SECONDS} to match the trained model"
            )

        print(f"\n[*] Sampling {sampling_steps} real-time system observations...")
        cpu_readings = []
        mem_readings = []
        proc_readings = []
        disk_readings = []

        psutil.cpu_percent(interval=None)
        prev_disk = psutil.disk_io_counters()

        for step in range(sampling_steps):
            time.sleep(interval_seconds)
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            procs = len(psutil.pids())

            curr_disk = psutil.disk_io_counters()
            disk_mb = 0.0
            if curr_disk and prev_disk:
                diff = (curr_disk.read_bytes - prev_disk.read_bytes) + (curr_disk.write_bytes - prev_disk.write_bytes)
                disk_mb = diff / (1024 * 1024 * interval_seconds)
            prev_disk = curr_disk

            cpu_readings.append(cpu)
            mem_readings.append(mem)
            proc_readings.append(procs)
            disk_readings.append(disk_mb)

            print(f"    Step {step+1}/{sampling_steps}: Current CPU={cpu:5.1f}%, RAM={mem:5.1f}%, Procs={procs}")

        prediction = self.predict_from_recent_history(
            cpu_history=cpu_readings,
            memory_history=mem_readings,
            process_history=proc_readings,
            disk_history=disk_readings
        )
        print(
            f"\n[+] Live Forecast: Predicted CPU Utilization in "
            f"{PREDICTION_HORIZON_SECONDS} seconds is: {prediction:.2f}%"
        )
        current_cpu = cpu_readings[-1]
        delta = prediction - current_cpu
        trend = "INCREASING" if delta > 1.0 else ("DECREASING" if delta < -1.0 else "STABLE")
        print(f"    Current: {current_cpu:.1f}% -> Forecast: {prediction:.1f}% (Trend: {trend}, Delta: {delta:+.2f}%)")

        return prediction


def _validate_history(history: Sequence[float]) -> List[float]:
    if isinstance(history, (str, bytes, Mapping)):
        raise TypeError("history must be a sequence of numeric CPU percentages")
    try:
        readings = list(history)
    except TypeError as exc:
        raise TypeError("history must be a sequence of numeric CPU percentages") from exc

    if not readings:
        raise ValueError("history cannot be empty")
    if len(readings) < MIN_HISTORY_SAMPLES:
        raise ValueError(
            f"history must contain at least {MIN_HISTORY_SAMPLES} readings "
            f"at {SAMPLE_INTERVAL_SECONDS}-second intervals"
        )

    validated = []
    for index, reading in enumerate(readings):
        if isinstance(reading, (str, bytes, bool)) or not isinstance(reading, Real):
            raise ValueError(f"history[{index}] must be a numeric CPU percentage")
        try:
            value = float(reading)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"history[{index}] must be a finite numeric CPU percentage") from exc
        if not math.isfinite(value):
            raise ValueError(f"history[{index}] must be finite")
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"history[{index}] must be between 0 and 100")
        validated.append(value)
    return validated


@lru_cache(maxsize=1)
def _default_predictor() -> CPUPredictor:
    return CPUPredictor(MODEL_PATH)


def predict_workload(history: Sequence[float]) -> float:
    """Forecast CPU utilization five minutes ahead from five-second readings."""
    validated_history = _validate_history(history)
    return float(_default_predictor().predict_from_recent_history(validated_history))


if __name__ == "__main__":
    predictor = CPUPredictor()

    # Demonstration matching user request:
    # "For example, suppose the three latest CPU readings are:
    #  Previous reading: 35%
    #  Current reading minus one interval: 42%
    #  Most recent reading: 48%"
    print("\n" + "=" * 60)
    print(" DEMO: SCENARIO INFERENCE FROM USER SPECIFICATION ")
    print("=" * 60)
    sample_readings = [30.0, 32.0, 35.0, 42.0, 48.0, 46.0, 44.0, 43.0, 42.0, 41.0, 40.0]
    print(f"Historical CPU readings: {sample_readings}")
    forecast = predictor.predict_from_recent_history(sample_readings)
    print(f"[>] Predicted CPU Usage in Five Minutes: {forecast:.2f}%\n")
