"""Carbon-aware workload scheduling estimates, independent of any cloud provider."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

from predictor import predict_workload

JobKind = Literal["urgent", "flexible"]
Strategy = Literal["reactive", "predictive", "greenscale"]
DecisionStatus = Literal["scheduled", "deferred"]


@dataclass(frozen=True)
class CarbonSlot:
    """A candidate start time and its simulated carbon intensity."""

    start_minutes: float
    carbon_intensity_gco2e_per_kwh: float


@dataclass(frozen=True)
class Job:
    """A job's CPU demand, duration, urgency, and optional deadline."""

    job_id: str
    cpu_cores: float
    duration_minutes: float
    kind: JobKind
    deadline_minutes: float | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    """Capacity and energy assumptions used consistently by every strategy."""

    replicas: int = 2
    cpu_cores_per_replica: float = 2.0
    watts_per_cpu_core: float = 50.0

    @property
    def cpu_capacity_cores(self) -> float:
        return self.replicas * self.cpu_cores_per_replica


@dataclass(frozen=True)
class JobDecision:
    job_id: str
    status: DecisionStatus
    start_minutes: float | None
    end_minutes: float | None
    energy_kwh: float
    emissions_gco2e: float
    reason: str


@dataclass(frozen=True)
class ScheduleMetrics:
    scheduled_jobs: int
    deferred_jobs: int
    total_energy_kwh: float
    total_emissions_gco2e: float


@dataclass(frozen=True)
class StrategyResult:
    """Decision details and UI-ready aggregate metrics for one strategy."""

    strategy: Strategy
    effective_strategy: Strategy
    base_cpu_percent: float
    fallback_reason: str | None
    decisions: tuple[JobDecision, ...]
    metrics: ScheduleMetrics


def compare_strategies(
    jobs: Sequence[Job],
    cpu_history: Sequence[float],
    carbon_slots: Sequence[CarbonSlot],
    config: SchedulerConfig = SchedulerConfig(),
    prediction_fn: Callable[[Sequence[float]], float] | None = None,
) -> Mapping[Strategy, StrategyResult]:
    """Schedule the same jobs under reactive, predictive, and GreenScale policies.

    Carbon intensity is held constant at the selected start slot for the full
    duration of a job. Energy estimates cover scheduled job CPU only, not idle
    capacity, other workloads, cooling, or measured cloud infrastructure.
    """
    _validate_config(config)
    validated_jobs = _validate_jobs(jobs)
    validated_slots = _validate_slots(carbon_slots)
    current_cpu = _current_cpu_percent(cpu_history)
    forecast, prediction_error = _try_predict(cpu_history, prediction_fn)

    results: dict[Strategy, StrategyResult] = {}
    for strategy in ("reactive", "predictive", "greenscale"):
        fallback_reason = prediction_error if strategy != "reactive" else None
        effective_strategy: Strategy = (
            "reactive" if fallback_reason is not None else strategy
        )
        base_cpu = current_cpu if effective_strategy == "reactive" else forecast
        assert base_cpu is not None
        results[strategy] = _schedule(
            strategy=strategy,
            effective_strategy=effective_strategy,
            fallback_reason=fallback_reason,
            base_cpu_percent=base_cpu,
            jobs=validated_jobs,
            slots=validated_slots,
            config=config,
        )
    return results


def _try_predict(
    history: Sequence[float],
    prediction_fn: Callable[[Sequence[float]], float] | None,
) -> tuple[float | None, str | None]:
    predictor = prediction_fn or predict_workload
    try:
        value = predictor(history)
    except (ArithmeticError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"CPU prediction failed ({type(exc).__name__}: {exc})"

    if isinstance(value, bool) or not isinstance(value, Real):
        return None, "CPU prediction was not a numeric percentage"
    prediction = float(value)
    if not math.isfinite(prediction) or not 0.0 <= prediction <= 100.0:
        return None, "CPU prediction was outside the valid 0–100% range"
    return prediction, None


def _schedule(
    strategy: Strategy,
    effective_strategy: Strategy,
    fallback_reason: str | None,
    base_cpu_percent: float,
    jobs: Sequence[Job],
    slots: Sequence[CarbonSlot],
    config: SchedulerConfig,
) -> StrategyResult:
    capacity = config.cpu_capacity_cores
    baseline_cores = capacity * base_cpu_percent / 100.0
    slot_now = next(slot for slot in slots if slot.start_minutes == 0)
    reservations: list[tuple[Job, float, float]] = []
    decisions_by_id: dict[str, JobDecision] = {}
    scheduling_order = sorted(jobs, key=lambda job: job.kind != "urgent")

    for job in scheduling_order:
        if job.kind == "urgent":
            start = 0.0
            end = start + job.duration_minutes
            concurrent_cores = sum(
                reserved.cpu_cores
                for reserved, reserved_start, reserved_end in reservations
                if start < reserved_end and reserved_start < end
            )
            over_capacity = baseline_cores + concurrent_cores + job.cpu_cores > capacity
            reservations.append((job, start, end))
            decision = _make_decision(job, start, slot_now, config)
            reason = (
                "Urgent job runs immediately; capacity is exceeded because urgent work "
                "cannot be deferred."
                if over_capacity
                else "Urgent job runs immediately, as required."
            )
            decisions_by_id[job.job_id] = _with_reason(decision, reason)
            continue

        candidates: list[CarbonSlot] = []
        rejected_for_deadline = False
        rejected_for_capacity = False
        for slot in slots:
            end = slot.start_minutes + job.duration_minutes
            if job.deadline_minutes is None or end > job.deadline_minutes:
                rejected_for_deadline = True
                continue
            if _fits_capacity(
                job,
                slot.start_minutes,
                end,
                reservations,
                baseline_cores,
                capacity,
            ):
                candidates.append(slot)
            else:
                rejected_for_capacity = True

        if not candidates:
            reasons: list[str] = []
            if rejected_for_deadline:
                reasons.append("no candidate slot finishes by the deadline")
            if rejected_for_capacity:
                reasons.append("available capacity is insufficient")
            if not reasons:
                reasons.append("no configured candidate slot is available")
            if fallback_reason is not None:
                reasons.append(f"reactive fallback: {fallback_reason}")
            decisions_by_id[job.job_id] = JobDecision(
                job_id=job.job_id,
                status="deferred",
                start_minutes=None,
                end_minutes=None,
                energy_kwh=0.0,
                emissions_gco2e=0.0,
                reason="Flexible job was deferred because " + " and ".join(reasons) + ".",
            )
            continue

        chosen = (
            min(candidates, key=lambda slot: (slot.carbon_intensity_gco2e_per_kwh, slot.start_minutes))
            if effective_strategy == "greenscale"
            else min(candidates, key=lambda slot: slot.start_minutes)
        )
        start = chosen.start_minutes
        end = start + job.duration_minutes
        reservations.append((job, start, end))
        decision = _make_decision(job, start, chosen, config)
        if effective_strategy == "greenscale":
            reason = (
                "Scheduled in the lowest-carbon configured slot that has sufficient "
                "capacity and meets the deadline."
            )
        else:
            reason = (
                "Scheduled in the earliest configured slot that has sufficient "
                "capacity and meets the deadline."
            )
        if fallback_reason is not None:
            reason += f" Reactive fallback: {fallback_reason}."
        decisions_by_id[job.job_id] = _with_reason(decision, reason)

    decisions = tuple(decisions_by_id[job.job_id] for job in jobs)
    scheduled = [decision for decision in decisions if decision.status == "scheduled"]
    metrics = ScheduleMetrics(
        scheduled_jobs=len(scheduled),
        deferred_jobs=len(decisions) - len(scheduled),
        total_energy_kwh=sum(decision.energy_kwh for decision in scheduled),
        total_emissions_gco2e=sum(decision.emissions_gco2e for decision in scheduled),
    )
    return StrategyResult(
        strategy=strategy,
        effective_strategy=effective_strategy,
        base_cpu_percent=base_cpu_percent,
        fallback_reason=fallback_reason,
        decisions=decisions,
        metrics=metrics,
    )


def _fits_capacity(
    job: Job,
    start: float,
    end: float,
    reservations: Sequence[tuple[Job, float, float]],
    baseline_cores: float,
    capacity: float,
) -> bool:
    concurrent_cores = sum(
        reserved.cpu_cores
        for reserved, reserved_start, reserved_end in reservations
        if start < reserved_end and reserved_start < end
    )
    return baseline_cores + concurrent_cores + job.cpu_cores <= capacity + 1e-9


def _make_decision(
    job: Job,
    start: float,
    slot: CarbonSlot,
    config: SchedulerConfig,
) -> JobDecision:
    end = start + job.duration_minutes
    energy_kwh = (
        job.cpu_cores
        * config.watts_per_cpu_core
        * (job.duration_minutes / 60.0)
        / 1000.0
    )
    emissions = energy_kwh * slot.carbon_intensity_gco2e_per_kwh
    return JobDecision(
        job_id=job.job_id,
        status="scheduled",
        start_minutes=start,
        end_minutes=end,
        energy_kwh=energy_kwh,
        emissions_gco2e=emissions,
        reason="",
    )


def _with_reason(decision: JobDecision, reason: str) -> JobDecision:
    return JobDecision(
        job_id=decision.job_id,
        status=decision.status,
        start_minutes=decision.start_minutes,
        end_minutes=decision.end_minutes,
        energy_kwh=decision.energy_kwh,
        emissions_gco2e=decision.emissions_gco2e,
        reason=reason,
    )


def _validate_config(config: SchedulerConfig) -> None:
    if type(config.replicas) is not int or config.replicas <= 0:
        raise ValueError("replicas must be a positive integer")
    _positive_finite(config.cpu_cores_per_replica, "cpu_cores_per_replica")
    _positive_finite(config.watts_per_cpu_core, "watts_per_cpu_core")


def _validate_jobs(jobs: Sequence[Job]) -> tuple[Job, ...]:
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)):
        raise TypeError("jobs must be a sequence of Job values")
    seen_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, Job):
            raise TypeError("jobs must contain only Job values")
        if not isinstance(job.job_id, str) or not job.job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if job.job_id in seen_ids:
            raise ValueError(f"duplicate job_id: {job.job_id}")
        seen_ids.add(job.job_id)
        _positive_finite(job.cpu_cores, f"{job.job_id}.cpu_cores")
        _positive_finite(job.duration_minutes, f"{job.job_id}.duration_minutes")
        if job.kind not in ("urgent", "flexible"):
            raise ValueError(f"{job.job_id}.kind must be 'urgent' or 'flexible'")
        if job.kind == "flexible" and job.deadline_minutes is None:
            raise ValueError(f"{job.job_id}.deadline_minutes is required for flexible jobs")
        if job.deadline_minutes is not None:
            _nonnegative_finite(job.deadline_minutes, f"{job.job_id}.deadline_minutes")
    return tuple(jobs)


def _validate_slots(slots: Sequence[CarbonSlot]) -> tuple[CarbonSlot, ...]:
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise TypeError("carbon_slots must be a sequence of CarbonSlot values")
    if not slots:
        raise ValueError("carbon_slots must include a slot at start_minutes=0")
    validated: list[CarbonSlot] = []
    starts: set[float] = set()
    for slot in slots:
        if not isinstance(slot, CarbonSlot):
            raise TypeError("carbon_slots must contain only CarbonSlot values")
        _nonnegative_finite(slot.start_minutes, "carbon slot start_minutes")
        _nonnegative_finite(
            slot.carbon_intensity_gco2e_per_kwh,
            "carbon_intensity_gco2e_per_kwh",
        )
        if slot.start_minutes in starts:
            raise ValueError(f"duplicate carbon slot start time: {slot.start_minutes}")
        starts.add(slot.start_minutes)
        validated.append(slot)
    if 0.0 not in starts:
        raise ValueError("carbon_slots must include a slot at start_minutes=0")
    return tuple(sorted(validated, key=lambda slot: slot.start_minutes))


def _current_cpu_percent(history: Sequence[float]) -> float:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("cpu_history must contain a current CPU percentage")
    value = history[-1]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("the latest CPU reading must be a numeric percentage")
    cpu_percent = float(value)
    if not math.isfinite(cpu_percent) or not 0.0 <= cpu_percent <= 100.0:
        raise ValueError("the latest CPU reading must be between 0 and 100")
    return cpu_percent


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _nonnegative_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
