import csv
import io
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import predictor as predictor_module
from app import (
    _results_csv,
    carbon_slots_from_records,
    generate_cpu_history,
    jobs_from_records,
    validate_cpu_history,
)
from forecast_config import MIN_HISTORY_SAMPLES
from greenscale_scheduler import CarbonSlot, Job, SchedulerConfig, compare_strategies


@pytest.mark.parametrize("preset", ["Normal Demand", "Traffic Spike", "Low Demand"])
def test_generated_presets_are_valid_and_repeatable(preset):
    history = generate_cpu_history(preset)

    assert history == generate_cpu_history(preset)
    assert len(history) >= MIN_HISTORY_SAMPLES
    assert all(0 <= value <= 100 for value in history)


def test_presets_represent_distinct_workload_levels():
    normal = generate_cpu_history("Normal Demand")
    spike = generate_cpu_history("Traffic Spike")
    low = generate_cpu_history("Low Demand")

    assert max(spike) > max(normal)
    assert sum(low) / len(low) < sum(normal) / len(normal)


def test_history_validation_rejects_too_few_or_out_of_range_readings():
    with pytest.raises(ValueError, match="at least"):
        validate_cpu_history([50.0] * (MIN_HISTORY_SAMPLES - 1))
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_cpu_history([50.0] * (MIN_HISTORY_SAMPLES - 1) + [101.0])


def test_editable_rows_create_jobs_and_carbon_slots_with_scheduler_field_names():
    jobs = jobs_from_records(
        [
            {
                "job_id": "urgent",
                "cpu_cores": 1,
                "duration_minutes": 5,
                "kind": "urgent",
                "deadline_minutes": None,
            },
            {
                "job_id": "batch",
                "cpu_cores": 2,
                "duration_minutes": 15,
                "kind": "flexible",
                "deadline_minutes": 60,
            },
            {
                "job_id": None,
                "cpu_cores": None,
                "duration_minutes": None,
                "kind": None,
                "deadline_minutes": None,
            },
        ]
    )
    slots = carbon_slots_from_records(
        [
            {"start_minutes": 0, "carbon_intensity_gco2e_per_kwh": 400},
            {"start_minutes": 20, "carbon_intensity_gco2e_per_kwh": 100},
        ]
    )

    assert [(job.job_id, job.kind, job.deadline_minutes) for job in jobs] == [
        ("urgent", "urgent", None),
        ("batch", "flexible", 60.0),
    ]
    assert slots[1].start_minutes == 20.0
    assert slots[1].carbon_intensity_gco2e_per_kwh == 100.0


def test_flexible_job_requires_deadline():
    with pytest.raises(ValueError, match="deadline_minutes is required"):
        jobs_from_records(
            [
                {
                    "job_id": "batch",
                    "cpu_cores": 1,
                    "duration_minutes": 10,
                    "kind": "flexible",
                    "deadline_minutes": None,
                }
            ]
        )


def test_csv_export_contains_strategy_metrics_and_job_decisions():
    results = compare_strategies(
        [Job("urgent", 1, 10, "urgent")],
        [30.0] * MIN_HISTORY_SAMPLES,
        [CarbonSlot(0, 400)],
        SchedulerConfig(),
        prediction_fn=lambda _history: 30.0,
    )

    rows = list(csv.DictReader(io.StringIO(_results_csv(results))))

    assert sum(row["row_type"] == "strategy_metrics" for row in rows) == 3
    decision_rows = [row for row in rows if row["row_type"] == "job_decision"]
    assert len(decision_rows) == 3
    assert all(row["job_id"] == "urgent" for row in decision_rows)
    assert all(row["status"] == "scheduled" for row in decision_rows)
    assert all(row["total_energy_kwh"] for row in rows)


def test_dashboard_displays_reactive_fallback_when_model_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        predictor_module, "MODEL_PATH", tmp_path / "missing-model.joblib"
    )
    predictor_module._default_predictor.cache_clear()
    try:
        app = AppTest.from_file(
            Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
        ).run()
        assert not app.exception
        assert any(
            "reactive fallback" in warning.value.lower() for warning in app.warning
        )
        assert any("No model forecast is available" in info.value for info in app.info)
    finally:
        predictor_module._default_predictor.cache_clear()
