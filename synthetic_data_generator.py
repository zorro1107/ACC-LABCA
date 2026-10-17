"""
synthetic_data_generator.py
---------------------------
Generates a realistic synthetic time-series dataset of system performance metrics
for CPU utilization forecasting.

Mathematical Formulation:
    CPU_t = clip(B_t + S_t + eps_t, 0, 100)
where:
    - B_t: Baseline workload with smooth sinusoidal periodic cycles (daily/task patterns)
    - S_t: Occasional heavy burst/spike events representing intensive tasks
    - eps_t: Stochastic noise (AR(1) process for temporal autocorrelation)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_cpu_data(
    num_samples: int = 5000,
    interval_seconds: int = 5,
    random_seed: int = 42,
    output_filepath: str = "cpu_data_synthetic.csv",
) -> pd.DataFrame:
    """
    Generates synthetic system telemetry data mimicking real server workloads.

    Parameters:
        num_samples: Number of chronological records to simulate.
        interval_seconds: Time interval between samples (in seconds).
        random_seed: Seed for reproducible synthetic generation.
        output_filepath: Destination path for the CSV file.

    Returns:
        pd.DataFrame containing the generated dataset.
    """
    np.random.seed(random_seed)

    # 1. Generate Timestamps
    start_time = datetime(2026, 10, 17, 8, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(num_samples)]

    # 2. Baseline Workload B_t (Combines smooth underlying trends + multiple periodic components)
    t = np.arange(num_samples)
    # Diurnal/periodic load cycle (period ~ 600 steps = 50 mins)
    cycle_1 = 15.0 * np.sin(2 * np.pi * t / 600)
    # Faster workload cycle (period ~ 120 steps = 10 mins)
    cycle_2 = 7.0 * np.sin(2 * np.pi * t / 120)
    base_load = 28.0 + cycle_1 + cycle_2  # Center around ~28%

    # 3. Workload Spikes S_t (Poisson-like spike events that decay over several steps)
    spikes = np.zeros(num_samples)
    spike_prob = 0.015  # 1.5% chance per interval of an intensive job trigger
    i = 0
    while i < num_samples:
        if np.random.rand() < spike_prob:
            spike_height = np.random.uniform(25.0, 50.0)
            duration = np.random.randint(3, 10)  # lasts 3-9 intervals
            for d in range(duration):
                if i + d < num_samples:
                    # Exponential or linear decay of spike
                    decay = np.exp(-d / 2.5)
                    spikes[i + d] += spike_height * decay
            i += duration
        else:
            i += 1

    # 4. Correlated Random Noise eps_t using an AR(1) process
    noise = np.zeros(num_samples)
    phi = 0.65  # Autoregressive coefficient
    white_noise = np.random.normal(0, 3.0, size=num_samples)
    for step in range(1, num_samples):
        noise[step] = phi * noise[step - 1] + white_noise[step]

    # 5. Composite CPU Utilization clamped in [0, 100]
    cpu_usage = np.clip(base_load + spikes + noise, 2.0, 99.5)

    # 6. Correlated System Metrics:
    # Memory usage (%) typically tracks longer-term load and lags CPU slightly
    mem_baseline = 40.0
    mem_noise = np.random.normal(0, 0.8, size=num_samples)
    # Cumulative rolling influence
    smoothed_cpu = pd.Series(cpu_usage).rolling(window=20, min_periods=1).mean().values
    memory_usage = np.clip(mem_baseline + 0.35 * smoothed_cpu + mem_noise, 15.0, 95.0)

    # Active Process Count: positively correlated with CPU bursts and spikes
    active_proc_base = 75
    proc_noise = np.random.randint(-4, 5, size=num_samples)
    active_processes = np.clip(
        (active_proc_base + 0.6 * cpu_usage + 0.5 * spikes + proc_noise).astype(int),
        40,
        250
    )

    # Disk I/O (MB/s): Spikes strongly when CPU spikes (e.g. data processing tasks)
    disk_io = np.clip(
        0.08 * cpu_usage + 0.3 * spikes + np.random.exponential(scale=1.5, size=num_samples),
        0.1,
        120.0
    )

    # Construct DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "cpu_usage": np.round(cpu_usage, 2),
        "memory_usage": np.round(memory_usage, 2),
        "active_processes": active_processes,
        "disk_io_mb_s": np.round(disk_io, 2),
    })

    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"[+] Synthetic dataset successfully generated: {output_filepath} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    df_synthetic = generate_synthetic_cpu_data()
    print(df_synthetic.head(10))
    print("\nDataset Summary Statistics:")
    print(df_synthetic.describe())
