"""Streamlit dashboard for simulated GreenScale scheduling."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from numbers import Real

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from forecast_config import (
    MIN_HISTORY_SAMPLES,
    PREDICTION_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
)
from greenscale_scheduler import (
    CarbonSlot,
    Job,
    SchedulerConfig,
    StrategyResult,
    compare_strategies,
)

PRESET_SEEDS = {
    "Normal Demand": 101,
    "Traffic Spike": 202,
    "Low Demand": 303,
}
STRATEGY_LABELS = {
    "reactive": "Reactive",
    "predictive": "Predictive",
    "greenscale": "GreenScale",
}


def generate_cpu_history(preset: str, sample_count: int = 60) -> list[float]:
    """Create repeatable, valid CPU samples for one workload preset."""
    if preset not in PRESET_SEEDS:
        raise ValueError(f"Unknown workload preset: {preset}")
    if type(sample_count) is not int or sample_count < MIN_HISTORY_SAMPLES:
        raise ValueError(f"sample_count must be at least {MIN_HISTORY_SAMPLES}")

    rng = random.Random(PRESET_SEEDS[preset])
    readings: list[float] = []
    spike_start = sample_count - max(12, sample_count // 5)
    for index in range(sample_count):
        cycle = math.sin(2 * math.pi * index / 36)
        if preset == "Normal Demand":
            value = 38 + 10 * cycle + rng.gauss(0, 2)
        elif preset == "Traffic Spike":
            spike = 38 + rng.gauss(0, 5) if index >= spike_start else 0
            value = 32 + 7 * cycle + spike + rng.gauss(0, 2)
        else:
            value = 12 + 5 * cycle + rng.gauss(0, 1.5)
        readings.append(round(min(100.0, max(0.0, value)), 2))
    return readings


def validate_cpu_history(values: Sequence[object]) -> list[float]:
    """Return validated CPU percentages suitable for the five-second predictor."""
    if isinstance(values, (str, bytes)) or len(values) < MIN_HISTORY_SAMPLES:
        raise ValueError(
            f"CPU history must contain at least {MIN_HISTORY_SAMPLES} readings"
        )

    history: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"CPU history reading {index + 1} must be numeric")
        reading = float(value)
        if not math.isfinite(reading) or not 0 <= reading <= 100:
            raise ValueError(
                f"CPU history reading {index + 1} must be between 0 and 100"
            )
        history.append(reading)
    return history


def jobs_from_records(records: Sequence[Mapping[str, object]]) -> list[Job]:
    """Convert editable table rows into validated scheduler jobs."""
    jobs: list[Job] = []
    for row_number, record in enumerate(records, start=1):
        if all(_is_blank(value) for value in record.values()):
            continue

        job_id = record.get("job_id")
        kind = record.get("kind")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError(f"Job row {row_number}: enter a job ID")
        if kind not in ("urgent", "flexible"):
            raise ValueError(f"Job row {row_number}: choose urgent or flexible")

        deadline = _optional_number(record.get("deadline_minutes"), "deadline_minutes")
        if kind == "flexible" and deadline is None:
            raise ValueError(
                f"Job row {row_number}: deadline_minutes is required for flexible jobs"
            )
        jobs.append(
            Job(
                job_id=job_id.strip(),
                cpu_cores=_required_number(record.get("cpu_cores"), "cpu_cores"),
                duration_minutes=_required_number(
                    record.get("duration_minutes"), "duration_minutes"
                ),
                kind=kind,
                deadline_minutes=deadline,
            )
        )
    return jobs


def carbon_slots_from_records(
    records: Sequence[Mapping[str, object]],
) -> list[CarbonSlot]:
    """Convert editable table rows into scheduler carbon slots."""
    slots: list[CarbonSlot] = []
    for row_number, record in enumerate(records, start=1):
        if all(_is_blank(value) for value in record.values()):
            continue
        slots.append(
            CarbonSlot(
                start_minutes=_required_number(
                    record.get("start_minutes"), "start_minutes"
                ),
                carbon_intensity_gco2e_per_kwh=_required_number(
                    record.get("carbon_intensity_gco2e_per_kwh"),
                    "carbon_intensity_gco2e_per_kwh",
                ),
            )
        )
    if not slots:
        raise ValueError(
            "Add at least one carbon slot, including a slot at minute zero"
        )
    return slots


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    return False


def _required_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _optional_number(value: object, field_name: str) -> float | None:
    if _is_blank(value):
        return None
    return _required_number(value, field_name)


def _render_cpu_chart(
    history: Sequence[float],
    prediction: float | None,
) -> None:
    sample_times = [
        (index - len(history) + 1) * SAMPLE_INTERVAL_SECONDS
        for index in range(len(history))
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sample_times,
            y=history,
            mode="lines+markers",
            name="CPU history",
            line={"color": "#2e8b57"},
        )
    )
    if prediction is not None:
        figure.add_trace(
            go.Scatter(
                x=[sample_times[-1], PREDICTION_HORIZON_SECONDS],
                y=[history[-1], prediction],
                mode="lines+markers",
                name="Five-minute forecast",
                line={"color": "#e07a32", "dash": "dash"},
            )
        )
    figure.update_layout(
        title="CPU history and model forecast",
        xaxis_title="Seconds from latest reading",
        yaxis_title="CPU utilization (%)",
        yaxis={"range": [0, 100]},
        legend={"orientation": "h"},
        margin={"t": 55, "b": 35},
    )
    st.plotly_chart(figure, width="stretch")


def _render_strategy_comparison(results: Mapping[str, StrategyResult]) -> None:
    st.subheader("Strategy comparison")
    columns = st.columns(3)
    for column, strategy in zip(columns, STRATEGY_LABELS):
        result = results[strategy]
        metrics = result.metrics
        with column:
            st.markdown(f"#### {STRATEGY_LABELS[strategy]}")
            st.caption(
                f"Effective strategy: **{STRATEGY_LABELS[result.effective_strategy]}**"
            )
            if result.fallback_reason:
                st.warning(f"Prediction fallback: {result.fallback_reason}")
            st.metric("Scheduled jobs", metrics.scheduled_jobs)
            st.metric("Deferred jobs", metrics.deferred_jobs)
            st.metric("Estimated energy", f"{metrics.total_energy_kwh:.4f} kWh")
            st.metric(
                "Estimated emissions", f"{metrics.total_emissions_gco2e:.2f} g CO₂e"
            )
            st.metric("CPU load for capacity check", f"{result.base_cpu_percent:.2f}%")

            decisions = pd.DataFrame(
                [
                    {
                        "Job": decision.job_id,
                        "Status": decision.status.title(),
                        "Start (min)": decision.start_minutes,
                        "End (min)": decision.end_minutes,
                        "Energy (kWh)": decision.energy_kwh,
                        "Emissions (g CO₂e)": decision.emissions_gco2e,
                        "Explanation": decision.reason,
                    }
                    for decision in result.decisions
                ]
            )
            if decisions.empty:
                st.info("No jobs configured.")
            else:
                st.dataframe(decisions, hide_index=True, width="stretch")


def _comparison_figure(results: Mapping[str, StrategyResult]) -> go.Figure:
    strategies = list(STRATEGY_LABELS)
    labels = [STRATEGY_LABELS[strategy] for strategy in strategies]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Estimated energy (kWh)", "Estimated emissions (g CO₂e)"),
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[results[strategy].metrics.total_energy_kwh for strategy in strategies],
            name="Energy (kWh)",
            marker_color="#4c956c",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[
                results[strategy].metrics.total_emissions_gco2e
                for strategy in strategies
            ],
            name="Emissions (g CO₂e)",
            marker_color="#e07a32",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title="Estimated workload-only energy and emissions",
        showlegend=False,
        margin={"t": 75, "b": 35},
    )
    return figure


def _results_csv(results: Mapping[str, StrategyResult]) -> str:
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        metrics = result.metrics
        rows.append(
            {
                "row_type": "strategy_metrics",
                "strategy": strategy,
                "effective_strategy": result.effective_strategy,
                "fallback_reason": result.fallback_reason or "",
                "job_id": "",
                "status": "",
                "start_minutes": "",
                "end_minutes": "",
                "energy_kwh": "",
                "emissions_gco2e": "",
                "reason": "",
                "scheduled_jobs": metrics.scheduled_jobs,
                "deferred_jobs": metrics.deferred_jobs,
                "total_energy_kwh": metrics.total_energy_kwh,
                "total_emissions_gco2e": metrics.total_emissions_gco2e,
            }
        )
        for decision in result.decisions:
            rows.append(
                {
                    "row_type": "job_decision",
                    "strategy": strategy,
                    "effective_strategy": result.effective_strategy,
                    "fallback_reason": result.fallback_reason or "",
                    "job_id": decision.job_id,
                    "status": decision.status,
                    "start_minutes": decision.start_minutes,
                    "end_minutes": decision.end_minutes,
                    "energy_kwh": decision.energy_kwh,
                    "emissions_gco2e": decision.emissions_gco2e,
                    "reason": decision.reason,
                    "scheduled_jobs": metrics.scheduled_jobs,
                    "deferred_jobs": metrics.deferred_jobs,
                    "total_energy_kwh": metrics.total_energy_kwh,
                    "total_emissions_gco2e": metrics.total_emissions_gco2e,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False)


def main() -> None:
    st.set_page_config(
        page_title="GreenScale | Carbon-Aware Scheduler", page_icon="🌱", layout="wide"
    )
    st.title("🌱 GreenScale")
    st.subheader("Carbon-Aware Predictive Cloud Manager")
    st.warning(
        "Simulation dashboard only. It does not control Azure or other cloud "
        "infrastructure. Energy and emissions are estimates for scheduled job CPU, "
        "not measured cloud consumption."
    )
    st.markdown(
        "Compare three scheduling policies using the same workload and simulated "
        "carbon inputs: **Reactive** uses current CPU load, **Predictive** uses the "
        "five-minute CPU forecast to assess capacity, and **GreenScale** combines "
        "the forecast with the lowest-carbon feasible start slot. Urgent jobs always "
        "run immediately according to the scheduler."
    )

    with st.sidebar:
        st.header("Workload")
        preset = st.selectbox("Demand preset", list(PRESET_SEEDS))
        history_base_key = f"cpu_history_base_{preset}"
        history_widget_key = f"cpu_history_editor_{preset}"
        if history_base_key not in st.session_state:
            st.session_state[history_base_key] = pd.DataFrame(
                {"cpu_percent": generate_cpu_history(preset)}
            )
        if st.button("Reset history to preset"):
            st.session_state[history_base_key] = pd.DataFrame(
                {"cpu_percent": generate_cpu_history(preset)}
            )
            st.session_state.pop(history_widget_key, None)

        st.caption(
            f"Samples are {SAMPLE_INTERVAL_SECONDS} seconds apart; at least "
            f"{MIN_HISTORY_SAMPLES} readings are required."
        )
        history_frame = st.data_editor(
            st.session_state[history_base_key],
            key=history_widget_key,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "cpu_percent": st.column_config.NumberColumn(
                    "CPU utilization (%)", min_value=0.0, max_value=100.0, step=0.1
                )
            },
        )

        st.header("Infrastructure assumptions")
        replicas = st.number_input("Replica count", min_value=1, value=2, step=1)
        cpu_cores_per_replica = st.number_input(
            "CPU cores per replica", min_value=0.1, value=2.0, step=0.5
        )
        watts_per_cpu_core = st.number_input(
            "Watts per CPU core", min_value=0.1, value=50.0, step=5.0
        )
        st.caption("These are simulated capacity and power assumptions.")

    st.header("Jobs")
    st.caption("Add, edit, or remove jobs. Flexible jobs require a deadline.")
    default_jobs = pd.DataFrame(
        [
            {
                "job_id": "urgent-api",
                "cpu_cores": 1.0,
                "duration_minutes": 10.0,
                "kind": "urgent",
                "deadline_minutes": None,
            },
            {
                "job_id": "batch-report",
                "cpu_cores": 1.0,
                "duration_minutes": 20.0,
                "kind": "flexible",
                "deadline_minutes": 60.0,
            },
        ]
    )
    job_records = st.data_editor(
        default_jobs,
        key="jobs_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "job_id": st.column_config.TextColumn("Job ID"),
            "cpu_cores": st.column_config.NumberColumn(
                "CPU cores", min_value=0.1, step=0.1
            ),
            "duration_minutes": st.column_config.NumberColumn(
                "Duration (minutes)", min_value=0.1, step=1.0
            ),
            "kind": st.column_config.SelectboxColumn(
                "Job type", options=["urgent", "flexible"], required=True
            ),
            "deadline_minutes": st.column_config.NumberColumn(
                "Deadline (minutes; flexible required)", min_value=0.0, step=5.0
            ),
        },
    )

    st.header("Carbon-intensity schedule")
    st.caption("Simulated carbon intensity inputs; include a slot at minute zero.")
    default_slots = pd.DataFrame(
        [
            {"start_minutes": 0.0, "carbon_intensity_gco2e_per_kwh": 450.0},
            {"start_minutes": 15.0, "carbon_intensity_gco2e_per_kwh": 100.0},
            {"start_minutes": 30.0, "carbon_intensity_gco2e_per_kwh": 700.0},
        ]
    )
    slot_records = st.data_editor(
        default_slots,
        key="carbon_slots_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "start_minutes": st.column_config.NumberColumn(
                "Start time (minutes from now)", min_value=0.0, step=5.0
            ),
            "carbon_intensity_gco2e_per_kwh": st.column_config.NumberColumn(
                "Carbon intensity (gCO₂e/kWh)", min_value=0.0, step=10.0
            ),
        },
    )

    st.caption(
        "The selected slot intensity is assumed constant over each job. Capacity "
        "uses the current or forecast CPU percentage as a constant background load."
    )

    try:
        cpu_history = validate_cpu_history(history_frame["cpu_percent"].tolist())
        jobs = jobs_from_records(job_records.to_dict(orient="records"))
        carbon_slots = carbon_slots_from_records(slot_records.to_dict(orient="records"))
        config = SchedulerConfig(
            replicas=int(replicas),
            cpu_cores_per_replica=float(cpu_cores_per_replica),
            watts_per_cpu_core=float(watts_per_cpu_core),
        )
        capacity = config.cpu_capacity_cores
        oversized_urgent = [
            job for job in jobs if job.kind == "urgent" and job.cpu_cores > capacity
        ]
        for job in oversized_urgent:
            st.warning(
                f"Urgent job **{job.job_id}** requests {job.cpu_cores:g} CPU cores, "
                f"above configured capacity of {capacity:g}. The scheduler still "
                "starts urgent jobs immediately, so capacity may be exceeded."
            )

        results = compare_strategies(jobs, cpu_history, carbon_slots, config)
    except (TypeError, ValueError, OverflowError) as exc:
        st.error(f"Cannot run this simulation: {exc}")
        return

    for result in results.values():
        for decision in result.decisions:
            if (
                decision.status == "scheduled"
                and "capacity is exceeded" in decision.reason
            ):
                st.warning(
                    f"Capacity outcome for **{decision.job_id}** under "
                    f"{STRATEGY_LABELS[result.strategy]}: {decision.reason}"
                )
    for strategy in ("predictive", "greenscale"):
        fallback = results[strategy].fallback_reason
        if fallback:
            st.warning(
                f"{STRATEGY_LABELS[strategy]} is using reactive fallback: {fallback}"
            )

    st.header("CPU history and prediction")
    forecast = (
        results["predictive"].base_cpu_percent
        if results["predictive"].effective_strategy == "predictive"
        else None
    )
    _render_cpu_chart(cpu_history, forecast)
    if forecast is None:
        st.info("No model forecast is available; the chart shows CPU history only.")
    else:
        st.caption(f"Predicted CPU in five minutes: {forecast:.2f}%")

    _render_strategy_comparison(results)
    st.plotly_chart(_comparison_figure(results), width="stretch")
    st.caption(
        "All energy and emissions values are estimated job-only values, not actual "
        "Azure measurements or guaranteed savings."
    )
    st.download_button(
        "Download decisions and aggregate metrics (CSV)",
        data=_results_csv(results),
        file_name="greenscale_simulation.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
