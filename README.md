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
| [`synthetic_data_generator.py`](file:///synthetic_data_generator.py) | Generates synthetic time series using realistic load patterns, spikes, and AR(1) noise. |
| [`data_collector.py`](file:///data_collector.py) | Samples live system telemetry (CPU, RAM, active processes, disk I/O) using `psutil`. |
| [`feature_pipeline.py`](file:///feature_pipeline.py) | Constructs lag features, rolling windows, rates of change, and enforces chronological splits. |
| [`model_trainer.py`](file:///model_trainer.py) | Trains ML algorithms, compares them to the persistence baseline, saves model, and plots diagnostic charts. |
| [`predictor.py`](file:///predictor.py) | Inference engine for both arbitrary historical windows and live real-time OS metric forecasting. |
| `cpu_utilization_model.pkl` | Standalone Python pickle file containing the trained `RandomForestRegressor` for sharing. |
| `cpu_utilization_model_bundle.pkl` | Complete Python pickle bundle (model, 26 feature names, evaluation metrics). |
| [`test_pickle_load.py`](file:///test_pickle_load.py) | Ready-to-run script showing how recipient can load and use the `.pkl` files. |
| `evaluation_results.png` | Comprehensive 4-panel diagnostic visualization comparing models, residuals, and feature importances. |
| `best_cpu_predictor.joblib` | Serialized model artifact ready for production deployment. |

---

## 🚀 How to Run

### 1. End-to-End Pipeline (Synthetic Data)
```bash
python main.py --mode synthetic --samples 5000
```

### 2. End-to-End Pipeline with Real Telemetry Capture
Collects live metrics from your machine for 30 seconds before training and evaluating:
```bash
python main.py --mode real --collect-seconds 30
```

### 3. Real-Time Inference or Scenario Testing
```bash
python predictor.py
```

### 4. Direct Data Collection
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
$$y_t = CPU_{t+1}$$

### 3. Evaluation Metrics
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
