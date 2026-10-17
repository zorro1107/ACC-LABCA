import math

import pytest

from greenscale_scheduler import (
    CarbonSlot,
    Job,
    SchedulerConfig,
    compare_strategies,
)


SLOTS = (
    CarbonSlot(0, 500),
    CarbonSlot(15, 100),
    CarbonSlot(30, 700),
)
HISTORY = [20.0] * 11


def _run(jobs, prediction=20.0, config=SchedulerConfig(), slots=SLOTS):
    return compare_strategies(
        jobs,
        HISTORY,
        slots,
        config,
        prediction_fn=lambda _history: prediction,
    )


def test_urgent_job_runs_immediately_even_when_it_exceeds_capacity():
    results = _run([Job("urgent-1", 5, 10, "urgent")])

    for result in results.values():
        decision = result.decisions[0]
        assert decision.status == "scheduled"
        assert decision.start_minutes == 0
        assert "Urgent job runs immediately" in decision.reason
        assert "capacity is exceeded" in decision.reason


def test_flexible_job_uses_earliest_feasible_slot_in_non_carbon_strategies():
    results = _run([Job("batch-1", 1, 10, "flexible", deadline_minutes=40)])

    assert results["reactive"].decisions[0].start_minutes == 0
    assert results["predictive"].decisions[0].start_minutes == 0


def test_flexible_job_is_deferred_when_no_slot_meets_deadline():
    results = _run([Job("batch-1", 1, 20, "flexible", deadline_minutes=19)])

    for result in results.values():
        decision = result.decisions[0]
        assert decision.status == "deferred"
        assert decision.start_minutes is None
        assert "deadline" in decision.reason


def test_capacity_limit_defers_job_in_predictive_schedule():
    jobs = [Job("large-batch", 3, 10, "flexible", deadline_minutes=40)]
    results = _run(jobs, prediction=50.0)

    decision = results["predictive"].decisions[0]
    assert decision.status == "deferred"
    assert "capacity" in decision.reason


def test_greenscale_defers_to_low_carbon_slot_when_deadline_allows():
    jobs = [Job("batch-1", 1, 10, "flexible", deadline_minutes=40)]
    results = _run(jobs)

    assert results["predictive"].decisions[0].start_minutes == 0
    green_decision = results["greenscale"].decisions[0]
    assert green_decision.start_minutes == 15
    assert "lowest-carbon" in green_decision.reason


@pytest.mark.parametrize("invalid_prediction", [float("nan"), float("inf"), -1.0, 101.0, "20"])
def test_invalid_prediction_uses_reactive_strategy(invalid_prediction):
    results = _run(
        [Job("batch-1", 1, 10, "flexible", deadline_minutes=40)],
        prediction=invalid_prediction,
    )

    for strategy in ("predictive", "greenscale"):
        result = results[strategy]
        assert result.effective_strategy == "reactive"
        assert result.fallback_reason is not None
        assert result.decisions[0].start_minutes == results["reactive"].decisions[0].start_minutes
        assert "fallback" in result.decisions[0].reason.lower()


def test_prediction_exception_falls_back_reactively():
    def failed_prediction(_history):
        raise RuntimeError("model artifact unavailable")

    results = compare_strategies(
        [Job("batch-1", 1, 10, "flexible", deadline_minutes=40)],
        HISTORY,
        SLOTS,
        prediction_fn=failed_prediction,
    )

    assert results["predictive"].effective_strategy == "reactive"
    assert "model artifact unavailable" in results["predictive"].fallback_reason


def test_energy_and_emissions_use_configured_cpu_power_and_slot_intensity():
    job = Job("batch-1", 2, 30, "flexible", deadline_minutes=40)
    config = SchedulerConfig(
        replicas=2,
        cpu_cores_per_replica=2,
        watts_per_cpu_core=100,
    )
    results = _run([job], config=config)

    decision = results["predictive"].decisions[0]
    expected_energy = 2 * 100 * (30 / 60) / 1000
    expected_emissions = expected_energy * 500
    assert decision.energy_kwh == pytest.approx(expected_energy)
    assert decision.emissions_gco2e == pytest.approx(expected_emissions)
    assert results["predictive"].metrics.total_energy_kwh == pytest.approx(expected_energy)
    assert results["predictive"].metrics.total_emissions_gco2e == pytest.approx(expected_emissions)
    assert math.isclose(
        decision.emissions_gco2e,
        decision.energy_kwh * SLOTS[0].carbon_intensity_gco2e_per_kwh,
    )


def test_flexible_job_requires_deadline():
    with pytest.raises(ValueError, match="deadline_minutes is required"):
        _run([Job("batch-1", 1, 10, "flexible")])


def test_slots_must_include_immediate_time_for_urgent_execution():
    with pytest.raises(ValueError, match="start_minutes=0"):
        _run(
            [Job("urgent-1", 1, 10, "urgent")],
            slots=(CarbonSlot(15, 100),),
        )
