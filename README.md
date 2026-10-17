# CPU Utilization Forecasting Machine Learning System (Person 1)

This project implements an end-to-end Machine Learning pipeline that forecasts future CPU utilization from system telemetry. It models both real-world operating system performance and synthetic workloads with spikes, autocorrelated noise, and periodic trends.

---

## 📌 Architecture & Workflow

The system is organized into four core stages:

```
+---------------------------------------------------------------------------------------+
|                                1. DATA GENERATION & INGESTION                        |
|                                                                                       |
|   A. Real Monitoring (psutil)                 B. Synthetic Workload Simulation        |
|      - CPU %, RAM %, Procs, Disk I/O             - CPUt = clip(Bt + St + eps_t, 0, 100) |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                2. TIME-SERIES FEATURE PIPELINE                        |
|                                                                                       |
|   - Autoregressive Lags: cpu_lag_1, cpu_lag_2, cpu_lag_3, cpu_lag_5                   |
|   - Moving Averages: cpu_roll_mean_3, cpu_roll_mean_5, cpu_roll_mean_10               |
|   - Volatility / Std Dev: cpu_roll_std_3, cpu_roll_std_5                              |
|   - Rate of Change / Acceleration: cpu_diff_1, cpu_diff_2, acceleration               |
|   - System Context: RAM lags, Process changes, Disk I/O moving averages               |
|   - Chronological Split: 70% Train, 15% Validation, 15% Test (No data leakage)        |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                3. MODEL TRAINING & COMPARISON                         |
|                                                                                       |
|   - Naive Persistence Baseline: Predicts CPUt+1 = CPUt                                |
|   - Linear Regression                                                                 |
|   - Random Forest Regressor                                                           |
|   - Gradient Boosting Regressor                                                       |
|   - Metrics: MAE, MSE, RMSE, R² Score                                                 |
|   - Artifact Serialization: best_cpu_predictor.joblib                                 |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                4. INFERENCE ENGINE (PREDICTOR)                        |
|                                                                                       |
|   - Scenario-based Inference: E.g., recent window [30%, 35%, 42%, 48%]                |
|   - Live System Telemetry Inference: Real-time sampling and forecasting               |
+---------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

| File | Purpose |
| :--- | :--- |
| [`synthetic_data_generator.py`](./synthetic_data_generator.py) | Generates synthetic time series using realistic load patterns, spikes, and AR(1) noise. |
| [`data_collector.py`](./data_collector.py) | Samples live system telemetry (CPU, RAM, active processes, disk I/O) using `psutil`. |
| [`feature_pipeline.py`](./feature_pipeline.py) | Constructs lag features, rolling windows, rates of change, and chronological splits with a forecast-horizon gap. |
| [`model_trainer.py`](./model_trainer.py) | Trains ML algorithms, compares them to the persistence baseline, saves the model, and plots diagnostic charts. |
| [`predictor.py`](./predictor.py) | Provides the stable offline prediction function and optional live telemetry inference. |
| [`forecast_config.py`](./forecast_config.py) | Defines the shared five-second sampling interval and five-minute (60-sample) forecast horizon. |
| `ml/model.joblib` | Canonical trained model artifact used by `predict_workload`. |
| [`greenscale_scheduler.py`](./greenscale_scheduler.py) | Compares reactive, predictive, and GreenScale carbon-aware scheduling decisions using configurable simulation assumptions. |
| `cpu_utilization_model.pkl` | Standalone Python pickle file containing the trained `RandomForestRegressor` for sharing. |
| `cpu_utilization_model_bundle.pkl` | Complete Python pickle bundle (model, 26 feature names, evaluation metrics). |
| [`test_pickle_load.py`](./test_pickle_load.py) | Ready-to-run script showing how to load and use the legacy `.pkl` files. |
| `evaluation_results.png` | Comprehensive 4-panel diagnostic visualization comparing models, residuals, and feature importances. |
| [`tests/test_predictor.py`](./tests/test_predictor.py) | Automated tests for the prediction interface, horizon, input validation, and artifacts. |
| [`tests/test_greenscale_scheduler.py`](./tests/test_greenscale_scheduler.py) | Automated tests for scheduling constraints, prediction fallback, and estimated energy/emissions. |

---

## 🚀 How to Run

### 1. Install Dependencies

From this directory, create and activate a virtual environment, then install the application dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

The application dependencies are NumPy, pandas, scikit-learn, joblib, Matplotlib (training plots), and psutil (optional live telemetry). Offline prediction does not import psutil.

### 2. Train the Five-Minute Model

The synthetic generator produces one CPU sample every five seconds. The training target is therefore 60 samples (300 seconds) ahead. Training uses chronological 70%/15%/15% train/validation/test partitions, purges 60 rows at train and validation boundaries to avoid target overlap, and selects the model using validation RMSE. The test partition is reserved for final evaluation.

```bash
python main.py --mode synthetic --samples 5000
```

This writes the canonical model to `ml/model.joblib` (relative to the project, regardless of the current working directory), plus the existing dataset, diagnostic plot, and legacy pickle artifacts.

### 3. Public Prediction Interface

Exact signature:

```python
predict_workload(history: Sequence[float]) -> float
```

Pass at least 11 recent CPU utilization readings, oldest first, sampled at five-second intervals. Each reading must be a finite numeric percentage in the inclusive range 0–100. The returned Python `float` is the predicted CPU utilization percentage five minutes after the latest reading. Since this interface accepts CPU history only, the model's other telemetry features default to memory usage `50%`, active processes `100`, and disk I/O `1 MB/s`.

```python
from predictor import predict_workload

history = [28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 42.0, 44.0, 45.0, 46.0, 48.0]
prediction: float = predict_workload(history)
print(f"CPU utilization in five minutes: {prediction:.2f}%")
```

### 4. Prediction Smoke Test

Run after training. This checks the public function and output type without requiring live telemetry:

```bash
python -c "from predictor import predict_workload; history = [28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 42.0, 44.0, 45.0, 46.0, 48.0]; prediction = predict_workload(history); print(f'{prediction:.2f}% ({type(prediction).__name__})')"
```

### 5. Automated Tests

```bash
python -m pytest -q
```

### 6. GreenScale Decision Engine API

`greenscale_scheduler.py` is a provider-independent simulation; it does not call Azure or scale replicas. Pass the same pending jobs, CPU history, time slots, and assumptions to `compare_strategies` to get comparable results for `reactive`, `predictive`, and `greenscale`. The result maps each strategy to decisions (including a reason for every scheduled or deferred job), the effective strategy used, and aggregate metrics suitable for UI display.

Urgent jobs start immediately, even if they exceed configured CPU capacity. Flexible jobs are placed only in a configured start slot where estimated capacity is available and the job finishes by its deadline; GreenScale selects the feasible slot with the lowest configured carbon intensity, while reactive and predictive choose the earliest feasible slot. Prediction errors or out-of-range predictions switch predictive and GreenScale to reactive behavior and are reported in each result.

```python
from greenscale_scheduler import CarbonSlot, Job, SchedulerConfig, compare_strategies

jobs = [
    Job("urgent-api", cpu_cores=1, duration_minutes=10, kind="urgent"),
    Job("nightly-report", cpu_cores=1, duration_minutes=20, kind="flexible", deadline_minutes=60),
]
history = [28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 42.0, 44.0, 45.0, 46.0, 48.0]
carbon_slots = [
    CarbonSlot(start_minutes=0, carbon_intensity_gco2e_per_kwh=450),
    CarbonSlot(start_minutes=30, carbon_intensity_gco2e_per_kwh=180),
]
results = compare_strategies(
    jobs,
    history,
    carbon_slots,
    SchedulerConfig(replicas=2, cpu_cores_per_replica=2, watts_per_cpu_core=50),
)

for strategy, result in results.items():
    print(strategy, result.metrics)
    for decision in result.decisions:
        print(decision.job_id, decision.status, decision.reason)
```

`carbon_slots` are candidate job start times in minutes from now and must include minute zero. The selected slot's carbon intensity is assumed constant for that job's full duration, and the latest or predicted CPU percentage is held constant as background load across candidate slots. Energy estimates use `cpu_cores × watts_per_cpu_core × duration_hours`; emissions are `energy_kWh × carbon_intensity_gCO2e_per_kWh`. These are simulated assumptions and estimated job-only values—not Azure measurements, whole-cluster energy, or a guarantee of actual emissions savings. Returned frozen dataclasses can be serialized for UI integration (for example, with `dataclasses.asdict`).

### 7. Optional Real-Telemetry Training

Collects real metrics at the same five-second sampling interval. Use a sufficiently long duration to provide training, validation, and test data for a five-minute horizon:

```bash
python main.py --mode real --collect-seconds 3600
```

### 8. Optional Live Inference Demo

```bash
python main.py --mode synthetic --samples 5000 --live-demo
```

### 9. Direct Data Collection

```bash
python data_collector.py
```

---

## 📊 Mathematical Details

### 1. Synthetic Data Formulation
$$CPU_t = \operatorname{clip}\left(B_t + \epsilon_t + S_t, 0, 100\right)$$
Where:
- $B_t$: Baseline periodic workload ($28 + 15\sin(2\pi t / 600) + 7\sin(2\pi t / 120)$)
- $S_t$: Intensive workload spikes decaying exponentially ($\sum h \cdot e^{-d/\tau}$)
- $\epsilon_t$: Autoregressive AR(1) noise ($\epsilon_t = 0.65\epsilon_{t-1} + w_t$, $w_t \sim \mathcal{N}(0, 3^2)$)

### 2. Feature Engineering & Target
$$X_t = [CPU_t, CPU_{t-1}, CPU_{t-2}, CPU_{t-3}, \text{MA}_3, \text{MA}_5, \sigma_3, \Delta CPU_t, \dots]$$
$$y_t = CPU_{t+60} \quad \text{(300 seconds at five-second sampling)}$$

### 3. Evaluation Metrics
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
