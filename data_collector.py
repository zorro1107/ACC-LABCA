"""
data_collector.py
-----------------
Collects real-time operating system performance telemetry using the `psutil` library.
Records:
    - Timestamp
    - CPU Usage (%)
    - Memory Usage (%)
    - Active Process Count
    - Disk I/O Rate (Read + Write MB/s)
"""

import time
import os
import psutil
import pandas as pd
from datetime import datetime


def collect_system_metrics(
    duration_seconds: int = 60,
    interval_seconds: int = 1,
    output_filepath: str = "cpu_data_real.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Monitors and logs real system performance metrics at fixed time intervals.

    Parameters:
        duration_seconds: Total time duration for data collection.
        interval_seconds: Polling frequency in seconds (default: 1s).
        output_filepath: Destination CSV path.
        verbose: Whether to print each recorded row to stdout.

    Returns:
        pd.DataFrame containing collected real-world telemetry.
    """
    print(f"[*] Starting system metric collection for {duration_seconds} seconds...")
    print(f"[*] Polling interval: {interval_seconds}s | Output: {output_filepath}")

    records = []
    start_time = time.time()
    end_time = start_time + duration_seconds

    # Prime psutil CPU calculation
    psutil.cpu_percent(interval=None)
    prev_disk = psutil.disk_io_counters()
    prev_time = time.time()

    try:
        sample_idx = 1
        while time.time() < end_time:
            time.sleep(interval_seconds)

            current_time = time.time()
            elapsed_interval = current_time - prev_time
            prev_time = current_time

            # 1. CPU Usage (%)
            cpu_usage = psutil.cpu_percent(interval=None)

            # 2. Memory Usage (%)
            mem_info = psutil.virtual_memory()
            memory_usage = mem_info.percent

            # 3. Active Process Count
            active_processes = len(psutil.pids())

            # 4. Disk I/O Rate (MB/s)
            curr_disk = psutil.disk_io_counters()
            if curr_disk and prev_disk and elapsed_interval > 0:
                bytes_diff = (curr_disk.read_bytes - prev_disk.read_bytes) + \
                             (curr_disk.write_bytes - prev_disk.write_bytes)
                disk_io_mb_s = round(bytes_diff / (1024 * 1024 * elapsed_interval), 2)
            else:
                disk_io_mb_s = 0.0
            prev_disk = curr_disk

            # 5. Timestamp
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            row = {
                "timestamp": ts_str,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "active_processes": active_processes,
                "disk_io_mb_s": disk_io_mb_s,
            }
            records.append(row)

            if verbose:
                print(f"[{sample_idx:03d}] {ts_str} | CPU: {cpu_usage:5.1f}% | "
                      f"RAM: {memory_usage:5.1f}% | Procs: {active_processes:3d} | "
                      f"Disk: {disk_io_mb_s:6.2f} MB/s")

            sample_idx += 1

    except KeyboardInterrupt:
        print("\n[!] Data collection manually interrupted by user.")

    df = pd.DataFrame(records)

    # Save to CSV
    if not df.empty and output_filepath:
        # If file exists, append without header; otherwise write with header
        if os.path.exists(output_filepath):
            df.to_csv(output_filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(output_filepath, index=False)
        print(f"[+] Saved {len(df)} records to {output_filepath}")

    return df


if __name__ == "__main__":
    # Collect a short sample (e.g. 10 seconds) when executed directly
    df_real = collect_system_metrics(duration_seconds=10, interval_seconds=1)
    print("\nSample of collected data:")
    print(df_real)
