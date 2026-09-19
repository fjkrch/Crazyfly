"""CPU-testable Crazyflie episode metrics and strict aggregation.

The aggregation contract intentionally refuses incomplete episode sets.  A
terminated or crashed episode is a valid (failed) row and stays in every
applicable denominator; a missing row makes the checkpoint/scenario cell
invalid instead of quietly improving its rates.

Latency censoring
-----------------
For descriptive fixed-horizon latency summaries, an unsuccessful attempt is
assigned its complete observation horizon: 12 seconds for episode success,
the target segment duration for a switch, and 2 seconds for gust recovery.
This is reported alongside successes-only latency and is deliberately named a
fixed-horizon censored statistic rather than an uncensored mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean, median
from typing import Iterable, Sequence

import torch

from .logic import (
    CONTROL_DT_S,
    DEFAULT_GUST_STEPS,
    DEFAULT_SWITCH_STEPS,
    EPISODE_DURATION_S,
    EPISODE_STEPS,
    RECOVERY_WINDOW_S,
    SUCCESS_DWELL_S,
)


DEFAULT_EVALUATION_EPISODES = 16
DEFAULT_EVALUATION_SEED = 101
CENSORING_RULE = "unsuccessful_attempt_assigned_fixed_horizon"
AGGREGATE_WRENCH_WORK_PROXY_LABEL = (
    "time integral of |aggregate force dot linear velocity| plus "
    "|body moment dot angular velocity|; simulator proxy, not battery energy"
)


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _optional_finite_nonnegative(value: float | None, name: str) -> float | None:
    return None if value is None else _finite_nonnegative(value, name)


def goal_error_integral_m_s(
    goal_error_m: Sequence[float] | torch.Tensor,
    *,
    control_dt_s: float = CONTROL_DT_S,
) -> float:
    """Left-rectangle integral of 3-D goal distance over control intervals."""

    values = torch.as_tensor(goal_error_m, dtype=torch.float64)
    if values.ndim != 1:
        raise ValueError("goal_error_m must be a one-dimensional per-step trace")
    if torch.any(~torch.isfinite(values)) or torch.any(values < 0.0):
        raise ValueError("goal_error_m must be finite and non-negative")
    dt = _finite_nonnegative(control_dt_s, "control_dt_s")
    if dt == 0.0:
        raise ValueError("control_dt_s must be positive")
    return float(values.sum().item() * dt)


def mean_speed_inside_target_region_m_s(
    goal_error_m: Sequence[float] | torch.Tensor,
    speed_m_s: Sequence[float] | torch.Tensor,
    *,
    target_radius_m: float = 0.20,
) -> float | None:
    """Mean speed while geometrically inside the target radius.

    This deliberately uses distance only for region membership; the speed
    threshold remains part of the stricter success tube.
    """

    error = torch.as_tensor(goal_error_m, dtype=torch.float64)
    speed = torch.as_tensor(speed_m_s, dtype=torch.float64)
    if error.ndim != 1 or speed.shape != error.shape:
        raise ValueError("goal_error_m and speed_m_s must be matching one-dimensional traces")
    radius = _finite_nonnegative(target_radius_m, "target_radius_m")
    if torch.any(~torch.isfinite(error)) or torch.any(error < 0.0):
        raise ValueError("goal_error_m must be finite and non-negative")
    if torch.any(~torch.isfinite(speed)) or torch.any(speed < 0.0):
        raise ValueError("speed_m_s must be finite and non-negative")
    selected = speed[error <= radius]
    return float(selected.mean().item()) if selected.numel() else None


def command_effort_integral(
    normalized_actions: Sequence[Sequence[float]] | torch.Tensor,
    *,
    control_dt_s: float = CONTROL_DT_S,
) -> float:
    """Integral of squared normalized aggregate-wrench commands."""

    actions = torch.as_tensor(normalized_actions, dtype=torch.float64)
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError("normalized_actions must have shape (steps, 4)")
    if torch.any(~torch.isfinite(actions)):
        raise ValueError("normalized_actions must be finite")
    dt = _finite_nonnegative(control_dt_s, "control_dt_s")
    if dt == 0.0:
        raise ValueError("control_dt_s must be positive")
    return float(actions.square().sum().item() * dt)


def command_smoothness_mean_delta(
    normalized_actions: Sequence[Sequence[float]] | torch.Tensor,
) -> float:
    """Mean L2 command change between consecutive 50 Hz decisions."""

    actions = torch.as_tensor(normalized_actions, dtype=torch.float64)
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError("normalized_actions must have shape (steps, 4)")
    if torch.any(~torch.isfinite(actions)):
        raise ValueError("normalized_actions must be finite")
    if actions.shape[0] < 2:
        return 0.0
    return float(torch.linalg.vector_norm(actions[1:] - actions[:-1], dim=-1).mean().item())


def aggregate_wrench_mechanical_work_proxy_j(
    force_n: Sequence[Sequence[float]] | torch.Tensor,
    linear_velocity_m_s: Sequence[Sequence[float]] | torch.Tensor,
    moment_n_m: Sequence[Sequence[float]] | torch.Tensor,
    angular_velocity_rad_s: Sequence[Sequence[float]] | torch.Tensor,
    *,
    control_dt_s: float = CONTROL_DT_S,
) -> float:
    """Integrate absolute aggregate-wrench power in consistent frames.

    This is a mechanical-work proxy in joules, not rotor electrical or battery
    energy.  Force and linear velocity must share a frame; moment and angular
    velocity must likewise share a frame.
    """

    force = torch.as_tensor(force_n, dtype=torch.float64)
    velocity = torch.as_tensor(linear_velocity_m_s, dtype=torch.float64)
    moment = torch.as_tensor(moment_n_m, dtype=torch.float64)
    angular_velocity = torch.as_tensor(angular_velocity_rad_s, dtype=torch.float64)
    if force.ndim != 2 or force.shape[-1] != 3:
        raise ValueError("force_n must have shape (steps, 3)")
    if velocity.shape != force.shape or moment.shape != force.shape or angular_velocity.shape != force.shape:
        raise ValueError("all aggregate-wrench work traces must have matching shape (steps, 3)")
    if any(torch.any(~torch.isfinite(value)) for value in (force, velocity, moment, angular_velocity)):
        raise ValueError("aggregate-wrench work traces must be finite")
    dt = _finite_nonnegative(control_dt_s, "control_dt_s")
    if dt == 0.0:
        raise ValueError("control_dt_s must be positive")
    force_power = torch.abs(torch.sum(force * velocity, dim=-1))
    moment_power = torch.abs(torch.sum(moment * angular_velocity, dim=-1))
    return float(torch.sum(force_power + moment_power).item() * dt)


@dataclass(frozen=True)
class SwitchOutcome:
    """Outcome for one of the three targets installed after a switch."""

    switch_index: int
    switch_step: int
    success: bool
    latency_s: float | None
    censor_time_s: float = 3.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.switch_index, bool)
            or int(self.switch_index) != self.switch_index
            or self.switch_index < 0
        ):
            raise ValueError("switch_index must be a non-negative integer")
        if (
            isinstance(self.switch_step, bool)
            or int(self.switch_step) != self.switch_step
            or self.switch_step < 0
        ):
            raise ValueError("switch_step must be a non-negative integer")
        censor = _finite_nonnegative(self.censor_time_s, "censor_time_s")
        if censor <= 0.0:
            raise ValueError("censor_time_s must be positive")
        latency = _optional_finite_nonnegative(self.latency_s, "latency_s")
        if self.success and latency is None:
            raise ValueError("a successful switch outcome requires latency_s")
        if not self.success and latency is not None:
            raise ValueError("an unsuccessful switch outcome must have latency_s=None")
        if latency is not None and (latency < SUCCESS_DWELL_S - 1.0e-9 or latency > censor + 1.0e-9):
            raise ValueError("switch latency must include the 0.50 s dwell and fit its target segment")


@dataclass(frozen=True)
class GustRecoveryOutcome:
    """Outcome for one scheduled gust, whether or not it could be applied.

    ``max_displacement_m`` is relative to position immediately before the
    disturbance.  The error integral covers the post-gust recovery window.
    """

    gust_index: int
    gust_start_step: int
    applied: bool
    stable_before_gust: bool | None
    recovered: bool
    recovery_latency_s: float | None
    max_displacement_m: float | None
    post_gust_error_integral_m_s: float | None
    terminated_before_recovery: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.gust_index, bool)
            or int(self.gust_index) != self.gust_index
            or self.gust_index < 0
        ):
            raise ValueError("gust_index must be a non-negative integer")
        if (
            isinstance(self.gust_start_step, bool)
            or int(self.gust_start_step) != self.gust_start_step
            or self.gust_start_step < 0
        ):
            raise ValueError("gust_start_step must be a non-negative integer")
        latency = _optional_finite_nonnegative(self.recovery_latency_s, "recovery_latency_s")
        _optional_finite_nonnegative(self.max_displacement_m, "max_displacement_m")
        _optional_finite_nonnegative(
            self.post_gust_error_integral_m_s, "post_gust_error_integral_m_s"
        )
        if not self.applied and self.stable_before_gust is not None:
            raise ValueError("stability immediately before gust is unknown when gust was not reached")
        if self.applied and self.stable_before_gust is None:
            raise ValueError("an applied gust must record pre-gust stability")
        if not self.applied and self.recovered:
            raise ValueError("a gust that was not applied cannot be recovered")
        if not self.applied and not self.terminated_before_recovery:
            raise ValueError("a scheduled gust may be unapplied only after an early termination")
        if not self.applied and (
            self.max_displacement_m is not None or self.post_gust_error_integral_m_s is not None
        ):
            raise ValueError("an unapplied gust cannot have displacement or post-gust error metrics")
        if self.applied and (
            self.max_displacement_m is None or self.post_gust_error_integral_m_s is None
        ):
            raise ValueError("an applied gust must retain displacement and post-gust error metrics")
        if self.recovered and latency is None:
            raise ValueError("a recovered gust requires recovery_latency_s")
        if not self.recovered and latency is not None:
            raise ValueError("an unrecovered gust must have recovery_latency_s=None")
        if latency is not None and (
            latency < SUCCESS_DWELL_S - 1.0e-9 or latency > RECOVERY_WINDOW_S + 1.0e-9
        ):
            raise ValueError("recovery latency must include the dwell and be within the 2.0 s window")
        if self.recovered and self.terminated_before_recovery:
            raise ValueError("a recovered gust cannot terminate before recovery")


@dataclass(frozen=True)
class EpisodeSummary:
    """One retained evaluation episode row."""

    scenario: str
    episode_id: int
    success: bool
    terminated: bool
    truncated: bool
    failure_reason: str | None
    time_to_first_success_s: float | None
    final_goal_error_m: float
    integrated_goal_error_m_s: float
    mean_speed_inside_target_region_m_s: float | None
    crash: bool
    out_of_bounds: bool
    invalid_state: bool
    command_effort: float
    command_smoothness: float
    aggregate_wrench_mechanical_work_proxy_j: float
    completed_steps: int = EPISODE_STEPS
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    switch_outcomes: tuple[SwitchOutcome, ...] = ()
    gust_outcomes: tuple[GustRecoveryOutcome, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario or not isinstance(self.scenario, str):
            raise ValueError("scenario must be a non-empty string")
        if (
            isinstance(self.episode_id, bool)
            or int(self.episode_id) != self.episode_id
            or self.episode_id < 0
        ):
            raise ValueError("episode_id must be a non-negative integer")
        if (
            isinstance(self.completed_steps, bool)
            or int(self.completed_steps) != self.completed_steps
            or not 0 <= self.completed_steps <= EPISODE_STEPS
        ):
            raise ValueError(f"completed_steps must be between 0 and {EPISODE_STEPS}")
        if self.terminated and self.truncated:
            raise ValueError("termination and time-limit truncation must be distinguished")
        if self.truncated and self.completed_steps != EPISODE_STEPS:
            raise ValueError("a 12 s time-limit truncation must contain all 600 control steps")
        success_time = _optional_finite_nonnegative(
            self.time_to_first_success_s, "time_to_first_success_s"
        )
        if self.success and success_time is None:
            raise ValueError("a successful episode requires time_to_first_success_s")
        if not self.success and success_time is not None:
            raise ValueError("a failed episode must use time_to_first_success_s=None")
        if success_time is not None and (
            success_time < SUCCESS_DWELL_S - 1.0e-9 or success_time > EPISODE_DURATION_S + 1.0e-9
        ):
            raise ValueError("time_to_first_success_s must include dwell and fit the 12 s horizon")
        _finite_nonnegative(self.final_goal_error_m, "final_goal_error_m")
        _finite_nonnegative(self.integrated_goal_error_m_s, "integrated_goal_error_m_s")
        _optional_finite_nonnegative(
            self.mean_speed_inside_target_region_m_s,
            "mean_speed_inside_target_region_m_s",
        )
        _finite_nonnegative(self.command_effort, "command_effort")
        _finite_nonnegative(self.command_smoothness, "command_smoothness")
        _finite_nonnegative(
            self.aggregate_wrench_mechanical_work_proxy_j,
            "aggregate_wrench_mechanical_work_proxy_j",
        )
        if (self.crash or self.out_of_bounds or self.invalid_state) and not self.terminated:
            raise ValueError("crash, out-of-bounds, and invalid state are failure terminations")
        if self.failure_reason is not None and not str(self.failure_reason).strip():
            raise ValueError("failure_reason must be non-empty when present")
        if len({outcome.switch_index for outcome in self.switch_outcomes}) != len(self.switch_outcomes):
            raise ValueError("switch outcome indices must be unique within an episode")
        if len({outcome.gust_index for outcome in self.gust_outcomes}) != len(self.gust_outcomes):
            raise ValueError("gust outcome indices must be unique within an episode")

    @property
    def time_to_success_s(self) -> float | None:
        """Compatibility alias for callers using the shorter metric name."""

        return self.time_to_first_success_s

    @property
    def mechanical_work_proxy(self) -> float:
        """Compatibility alias; serialized output keeps the explicit label."""

        return self.aggregate_wrench_mechanical_work_proxy_j

    @property
    def speed_inside_target_region_m_s(self) -> float | None:
        """Compatibility alias for the per-episode region-speed mean."""

        return self.mean_speed_inside_target_region_m_s

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible episode record."""

        return asdict(self)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(mean(present)) if present else None


def _fixed_horizon_latency(
    successes_and_times: Iterable[tuple[bool, float | None]], horizon_s: float
) -> dict[str, object]:
    pairs = list(successes_and_times)
    horizon = _finite_nonnegative(horizon_s, "horizon_s")
    if horizon <= 0.0:
        raise ValueError("horizon_s must be positive")
    observed = [float(value) for success, value in pairs if success and value is not None]
    censored = [float(value) if success and value is not None else horizon for success, value in pairs]
    return {
        "censoring_rule": CENSORING_RULE,
        "horizon_s": horizon,
        "attempt_count": len(pairs),
        "observed_success_count": len(observed),
        "censored_count": len(pairs) - len(observed),
        "mean_fixed_horizon_censored_s": float(mean(censored)) if censored else None,
        "median_fixed_horizon_censored_s": float(median(censored)) if censored else None,
        "mean_successes_only_s": float(mean(observed)) if observed else None,
    }


def _scenario_kind(scenario: str) -> str:
    lowered = scenario.lower()
    if "switch" in lowered:
        return "switch"
    if "gust" in lowered:
        return "gust"
    if "reach" in lowered:
        return "reach"
    raise ValueError(f"unrecognized Crazyflie evaluation scenario: {scenario!r}")


def _validate_episode_set(
    rows: list[EpisodeSummary], expected_episode_count: int
) -> tuple[str, str]:
    if (
        isinstance(expected_episode_count, bool)
        or int(expected_episode_count) != expected_episode_count
        or expected_episode_count <= 0
    ):
        raise ValueError("expected_episode_count must be a positive integer")
    if len(rows) != expected_episode_count:
        raise ValueError(
            f"incomplete evaluation: expected {expected_episode_count} episode rows, found {len(rows)}"
        )
    expected_ids = set(range(expected_episode_count))
    actual_ids = {row.episode_id for row in rows}
    if len(actual_ids) != len(rows):
        raise ValueError("duplicate episode_id in evaluation rows")
    if actual_ids != expected_ids:
        raise ValueError(
            f"episode IDs must be exactly 0..{expected_episode_count - 1}; got {sorted(actual_ids)}"
        )
    scenarios = {row.scenario for row in rows}
    if len(scenarios) != 1:
        raise ValueError("one checkpoint/scenario summary cannot mix scenarios")
    evaluation_seeds = {row.evaluation_seed for row in rows}
    if len(evaluation_seeds) != 1:
        raise ValueError("one checkpoint/scenario summary cannot mix evaluation seeds")
    scenario = next(iter(scenarios))
    kind = _scenario_kind(scenario)
    for row in rows:
        if kind == "reach":
            if row.switch_outcomes or row.gust_outcomes:
                raise ValueError("WaypointReach rows cannot contain switch or gust outcomes")
        elif kind == "switch":
            if row.gust_outcomes:
                raise ValueError("WaypointSwitch rows cannot contain gust outcomes")
            if len(row.switch_outcomes) != len(DEFAULT_SWITCH_STEPS):
                raise ValueError("every WaypointSwitch row must retain all three scheduled switches")
            outcomes = sorted(row.switch_outcomes, key=lambda outcome: outcome.switch_index)
            if [outcome.switch_index for outcome in outcomes] != list(range(len(DEFAULT_SWITCH_STEPS))):
                raise ValueError("switch indices must be exactly 0, 1, 2")
            if [outcome.switch_step for outcome in outcomes] != list(DEFAULT_SWITCH_STEPS):
                raise ValueError("switch outcomes do not match the frozen 3/6/9 s schedule")
            if any(not math.isclose(outcome.censor_time_s, 3.0) for outcome in outcomes):
                raise ValueError("each switched target has the frozen 3.0 s observation segment")
        else:
            if row.switch_outcomes:
                raise ValueError("GustRecovery rows cannot contain switch outcomes")
            if len(row.gust_outcomes) != len(DEFAULT_GUST_STEPS):
                raise ValueError("every GustRecovery row must retain all three scheduled gusts")
            outcomes = sorted(row.gust_outcomes, key=lambda outcome: outcome.gust_index)
            if [outcome.gust_index for outcome in outcomes] != list(range(len(DEFAULT_GUST_STEPS))):
                raise ValueError("gust indices must be exactly 0, 1, 2")
            if [outcome.gust_start_step for outcome in outcomes] != list(DEFAULT_GUST_STEPS):
                raise ValueError("gust outcomes do not match the frozen 3/6/9 s schedule")
    return scenario, kind


def _summarize_switches(rows: list[EpisodeSummary]) -> dict[str, object]:
    outcomes = [outcome for row in rows for outcome in row.switch_outcomes]
    by_switch = []
    for switch_index in range(len(DEFAULT_SWITCH_STEPS)):
        group = [outcome for outcome in outcomes if outcome.switch_index == switch_index]
        latency = _fixed_horizon_latency(
            ((outcome.success, outcome.latency_s) for outcome in group),
            group[0].censor_time_s,
        )
        by_switch.append(
            {
                "switch_index": switch_index,
                "switch_step": DEFAULT_SWITCH_STEPS[switch_index],
                "switch_time_s": DEFAULT_SWITCH_STEPS[switch_index] * CONTROL_DT_S,
                "success_count": sum(outcome.success for outcome in group),
                "success_denominator": len(group),
                "success_rate": sum(outcome.success for outcome in group) / len(group),
                "latency": latency,
            }
        )
    successes = sum(outcome.success for outcome in outcomes)
    fixed_latency = [
        float(outcome.latency_s)
        if outcome.success and outcome.latency_s is not None
        else outcome.censor_time_s
        for outcome in outcomes
    ]
    observed_latency = [
        float(outcome.latency_s)
        for outcome in outcomes
        if outcome.success and outcome.latency_s is not None
    ]
    return {
        "attempt_count": len(outcomes),
        "switch_attempt_count": len(outcomes),
        "success_count": successes,
        "switch_success_count": successes,
        "success_denominator": len(outcomes),
        "success_rate": successes / len(outcomes),
        "switch_success_rate": successes / len(outcomes),
        "censoring_rule": CENSORING_RULE,
        "mean_fixed_horizon_censored_latency_s": float(mean(fixed_latency)),
        "mean_successes_only_latency_s": float(mean(observed_latency)) if observed_latency else None,
        "by_switch": by_switch,
    }


def _summarize_gusts(rows: list[EpisodeSummary]) -> dict[str, object]:
    outcomes = [outcome for row in rows for outcome in row.gust_outcomes]

    def group_summary(group: list[GustRecoveryOutcome]) -> dict[str, object]:
        success_count = sum(outcome.recovered for outcome in group)
        latency = _fixed_horizon_latency(
            ((outcome.recovered, outcome.recovery_latency_s) for outcome in group),
            RECOVERY_WINDOW_S,
        )
        conditional = [outcome for outcome in group if outcome.stable_before_gust is True]
        conditional_success = sum(outcome.recovered for outcome in conditional)
        conditional_latency = _fixed_horizon_latency(
            ((outcome.recovered, outcome.recovery_latency_s) for outcome in conditional),
            RECOVERY_WINDOW_S,
        )
        displacement_values = [
            outcome.max_displacement_m for outcome in group if outcome.max_displacement_m is not None
        ]
        error_integrals = [
            outcome.post_gust_error_integral_m_s
            for outcome in group
            if outcome.post_gust_error_integral_m_s is not None
        ]
        return {
            "scheduled_count": len(group),
            "recovery_attempt_count": len(group),
            "applied_count": sum(outcome.applied for outcome in group),
            "not_applied_count": sum(not outcome.applied for outcome in group),
            "unconditional_recovery_success_count": success_count,
            "recovery_success_count": success_count,
            "unconditional_recovery_denominator": len(group),
            "unconditional_recovery_rate": success_count / len(group),
            "recovery_success_rate": success_count / len(group),
            "unconditional_recovery_latency": latency,
            "conditional_stable_count": len(conditional),
            "conditional_recovery_success_count": conditional_success,
            "conditional_recovery_denominator": len(conditional),
            "conditional_recovery_rate": (
                conditional_success / len(conditional) if conditional else None
            ),
            "conditional_recovery_latency": conditional_latency,
            "max_displacement_observation_count": len(displacement_values),
            "mean_max_displacement_m": (
                float(mean(displacement_values)) if displacement_values else None
            ),
            "post_gust_error_integral_observation_count": len(error_integrals),
            "mean_post_gust_error_integral_m_s": (
                float(mean(error_integrals)) if error_integrals else None
            ),
            "terminated_before_recovery_count": sum(
                outcome.terminated_before_recovery for outcome in group
            ),
        }

    result = group_summary(outcomes)
    result["by_gust"] = []
    for gust_index in range(len(DEFAULT_GUST_STEPS)):
        group = [outcome for outcome in outcomes if outcome.gust_index == gust_index]
        entry = group_summary(group)
        entry.update(
            {
                "gust_index": gust_index,
                "gust_start_step": DEFAULT_GUST_STEPS[gust_index],
                "gust_start_time_s": DEFAULT_GUST_STEPS[gust_index] * CONTROL_DT_S,
            }
        )
        result["by_gust"].append(entry)
    return result


def summarize_episodes(
    episodes: Iterable[EpisodeSummary],
    *,
    expected_episode_count: int = DEFAULT_EVALUATION_EPISODES,
) -> dict[str, object]:
    """Build a strict checkpoint/scenario summary with fixed denominators.

    The default denominator is the preregistered 16 held-out episodes.  A
    smaller integration protocol must pass its own declared expected count;
    simply supplying fewer rows without doing so is an error.
    """

    rows = sorted(list(episodes), key=lambda row: row.episode_id)
    scenario, kind = _validate_episode_set(rows, expected_episode_count)
    success_count = sum(row.success for row in rows)
    success_latency = _fixed_horizon_latency(
        ((row.success, row.time_to_first_success_s) for row in rows),
        EPISODE_DURATION_S,
    )
    result: dict[str, object] = {
        "scenario": scenario,
        "evaluation_seed": rows[0].evaluation_seed,
        "expected_episode_count": expected_episode_count,
        "episode_count": len(rows),
        "n_episodes": len(rows),
        "complete": True,
        "success_count": success_count,
        "failure_count": expected_episode_count - success_count,
        "success_denominator": expected_episode_count,
        "success_rate": success_count / expected_episode_count,
        "failure_rate": (expected_episode_count - success_count) / expected_episode_count,
        "time_to_first_success": success_latency,
        # Flat aliases make tabular exports unambiguous without dropping the
        # full censoring metadata above.
        "mean_time_to_first_success_fixed_horizon_censored_s": success_latency[
            "mean_fixed_horizon_censored_s"
        ],
        "mean_time_to_first_success_s": success_latency["mean_fixed_horizon_censored_s"],
        "mean_time_to_first_success_successes_only_s": success_latency["mean_successes_only_s"],
        "mean_final_goal_error_m": float(mean(row.final_goal_error_m for row in rows)),
        "mean_integrated_goal_error_m_s": float(
            mean(row.integrated_goal_error_m_s for row in rows)
        ),
        "mean_speed_inside_target_region_m_s": _mean_or_none(
            row.mean_speed_inside_target_region_m_s for row in rows
        ),
        "speed_inside_target_region_observation_count": sum(
            row.mean_speed_inside_target_region_m_s is not None for row in rows
        ),
        "crash_count": sum(row.crash for row in rows),
        "crash_denominator": expected_episode_count,
        "crash_rate": sum(row.crash for row in rows) / expected_episode_count,
        "out_of_bounds_count": sum(row.out_of_bounds for row in rows),
        "out_of_bounds_denominator": expected_episode_count,
        "out_of_bounds_rate": sum(row.out_of_bounds for row in rows) / expected_episode_count,
        "invalid_state_count": sum(row.invalid_state for row in rows),
        "invalid_state_denominator": expected_episode_count,
        "invalid_state_rate": sum(row.invalid_state for row in rows) / expected_episode_count,
        "termination_count": sum(row.terminated for row in rows),
        "truncation_count": sum(row.truncated for row in rows),
        "mean_command_effort": float(mean(row.command_effort for row in rows)),
        "mean_command_smoothness": float(mean(row.command_smoothness for row in rows)),
        "aggregate_wrench_mechanical_work_proxy_label": AGGREGATE_WRENCH_WORK_PROXY_LABEL,
        "mean_aggregate_wrench_mechanical_work_proxy_j": float(
            mean(row.aggregate_wrench_mechanical_work_proxy_j for row in rows)
        ),
        "switch_metrics": _summarize_switches(rows) if kind == "switch" else None,
        "gust_metrics": _summarize_gusts(rows) if kind == "gust" else None,
        "episodes": [asdict(row) for row in rows],
    }
    return result


__all__ = [
    "DEFAULT_EVALUATION_EPISODES",
    "DEFAULT_EVALUATION_SEED",
    "CENSORING_RULE",
    "AGGREGATE_WRENCH_WORK_PROXY_LABEL",
    "goal_error_integral_m_s",
    "mean_speed_inside_target_region_m_s",
    "command_effort_integral",
    "command_smoothness_mean_delta",
    "aggregate_wrench_mechanical_work_proxy_j",
    "SwitchOutcome",
    "GustRecoveryOutcome",
    "EpisodeSummary",
    "summarize_episodes",
]
