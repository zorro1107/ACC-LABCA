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

import time
import joblib
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class CPUPredictor:
    def __init__(self, model_artifact_path: str = "best_cpu_predictor.joblib"):
        """
        Loads the trained model artifact containing model, metadata, and required features.
        """
        print(f"[*] Loading model artifact from {model_artifact_path}...")
        artifact = joblib.load(model_artifact_path)
        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.feature_cols = artifact["feature_cols"]
        self.test_metrics = artifact["test_metrics"]
        print(f"[+] Loaded '{self.model_name}' (Test RMSE: {self.test_metrics.get('Test RMSE', 'N/A')}%)")

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
        return np.clip(preds, 0.0, 100.0)

    def predict_from_recent_history(
        self,
        cpu_history: List[float],
        memory_history: Optional[List[float]] = None,
        process_history: Optional[List[int]] = None,
        disk_history: Optional[List[float]] = None
    ) -> float:
        """
        Takes recent history readings (at least 6-10 steps for rolling features),
        constructs the full feature vector, and returns the next future CPU forecast.

        Example:
            cpu_history = [28, 30, 32, 35, 42, 48]
        """
        if len(cpu_history) < 6:
            # Pad if needed
            pad_val = cpu_history[0]
            cpu_history = [pad_val] * (6 - len(cpu_history)) + list(cpu_history)

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
            "timestamp": pd.date_range("2026-10-17 12:00:00", periods=n, freq="5s"),
            "cpu_usage": cpu_history,
            "memory_usage": memory_history,
            "active_processes": process_history,
            "disk_io_mb_s": disk_history
        })

        from feature_pipeline import engineer_features
        featured_df, _, _ = engineer_features(raw_df, target_horizon=1, dropna=False)

        # Take the most recent row
        latest_row = featured_df.iloc[-1:]

        # Fill any remaining NaNs in rolling features with available values
        latest_row = latest_row.bfill().ffill().fillna(0)

        pred = self.predict_from_features(latest_row)[0]
        return round(float(pred), 2)

    def live_predict(self, sampling_steps: int = 6, interval_seconds: int = 1) -> float:
        """
        Monitors live OS metrics using psutil for a short warmup window,
        builds features, and forecasts future CPU utilization.
        """
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
        print(f"\n[+] Live Forecast: Predicted CPU Utilization for next interval is: {prediction:.2f}%")
        current_cpu = cpu_readings[-1]
        delta = prediction - current_cpu
        trend = "INCREASING" if delta > 1.0 else ("DECREASING" if delta < -1.0 else "STABLE")
        print(f"    Current: {current_cpu:.1f}% -> Forecast: {prediction:.1f}% (Trend: {trend}, Delta: {delta:+.2f}%)")

        return prediction


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
    sample_readings = [30.0, 32.0, 35.0, 42.0, 48.0]
    print(f"Historical CPU readings: {sample_readings}")
    forecast = predictor.predict_from_recent_history(sample_readings)
    print(f"[>] Predicted Next-Interval CPU Usage: {forecast:.2f}%\n")
