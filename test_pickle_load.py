"""
test_pickle_load.py
-------------------
Example script demonstrating how a recipient can load and use
the trained CPU Utilization Prediction model pickle file.
"""

import pickle
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# OPTION 1: Loading the standalone scikit-learn model
# -------------------------------------------------------------
print("[-] Loading standalone model: cpu_utilization_model.pkl")
with open("cpu_utilization_model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"[+] Loaded Model Type: {type(model).__name__}")
print(f"[+] Trees in forest: {len(model.estimators_)}")

# -------------------------------------------------------------
# OPTION 2: Loading the complete bundle (Model + Metadata + Feature Names)
# -------------------------------------------------------------
print("\n[-] Loading bundle: cpu_utilization_model_bundle.pkl")
with open("cpu_utilization_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

bundled_model = bundle["model"]
feature_names = bundle["feature_names"]
metrics = bundle["test_metrics"]

print(f"[+] Model Name    : {bundle['model_name']}")
print(f"[+] Test RMSE     : {metrics.get('Test RMSE')}%")
print(f"[+] Test R2       : {metrics.get('Test R2')}")
print(f"[+] Total Features: {len(feature_names)}")
print(f"[+] Feature List  : {feature_names[:8]} ...")

# -------------------------------------------------------------
# Running a Test Inference
# -------------------------------------------------------------
# Create sample feature vector matching the 26 engineered features
dummy_sample = pd.DataFrame([{col: 0.0 for col in feature_names}])
# Set recent CPU and lag values (e.g. 35%, 42%, 48%)
dummy_sample["cpu_usage"] = 48.0
dummy_sample["cpu_lag_1"] = 42.0
dummy_sample["cpu_lag_2"] = 35.0
dummy_sample["cpu_roll_mean_3"] = (48.0 + 42.0 + 35.0) / 3.0
dummy_sample["memory_usage"] = 65.0
dummy_sample["active_processes"] = 110

predicted_cpu = bundled_model.predict(dummy_sample)[0]
print(f"\n[+] Sample Prediction Result: {predicted_cpu:.2f}% CPU utilization")
