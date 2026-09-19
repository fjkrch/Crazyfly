"""Pure contracts and deterministic schedules for the command-v2 tasks.

This module has no Isaac Lab imports.  The still-air and wind variants share
the exact command stream, observation definition, safety rules, and reward;
the wind variant adds only a deterministic physical world-frame wrench.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any

import torch

from .command_logic import (
    ACTION_WIDTH,
    COMMAND_WIDTH,
    CONTROL_DT_S,
    CRAZYFLIE_HOVER_ACTION,
    FAILURE_CAUSE_NAMES,
    FAILURE_HIGH_HEIGHT,
    FAILURE_LOW_HEIGHT,
    FAILURE_NONE,
    FAILURE_NONFINITE,
    FAILURE_WORKSPACE_ESCAPE,
    HARD_MAXIMUM_HEIGHT_M,
    HARD_MINIMUM_HEIGHT_M,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HEIGHT_M,
    MINIMUM_COMMAND_HOLD_STEPS,
    MINIMUM_HEIGHT_M,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_WIDTH,
    SAFETY_MARGIN_M,
    FailureClassification,
    classify_command_failures,
    command_conditioned_observation,
    integrate_command_target,
)


COMMAND_FOLLOW_WIDE_TASK_ID = "FlyCrazyflie-CommandFollowWide-v0"
COMMAND_FOLLOW_WIDE_WIND_TASK_ID = "FlyCrazyflie-CommandFollowWideWind-v0"
COMMAND_V2_PROFILE = "command_v2"
COMMAND_WIDE_CONTRACT_VERSION = "crazyflie_command_follow_v2"
COMMAND_WIDE_SCHEDULE_STATE_KIND = "flyg1.crazyflie.command-wide-schedule.v2"

TRAINING_INTERACTION_BUDGET = 1_000_000
EPISODE_STEPS = 600
MAXIMUM_HORIZONTAL_SPEED_M_S = 1.0
MAXIMUM_VERTICAL_SPEED_M_S = 0.5
MAXIMUM_YAW_RATE_RAD_S = 1.5
SOFT_WORKSPACE_XY_M = 5.0
HARD_WORKSPACE_XY_M = 5.5

WIND_FRAME = "world"
WIND_APPLICATION_POINT = "body_center_of_mass"
WIND_REFERENCE_ARM_M = 0.046
TRAINING_WIND_SEED = 20260918
HELD_OUT_WIND_SEED = 20260919
MINIMUM_WIND_PULSE_STEPS = 10
MAXIMUM_WIND_PULSE_STEPS = 30
MINIMUM_WIND_CALM_STEPS = 25
MAXIMUM_WIND_CALM_STEPS = 75
HELD_OUT_EPISODES = 16
HELD_OUT_EPISODE_STEPS = 600
HELD_OUT_PULSE_START_STEPS = (75, 175, 275, 375, 475)
HELD_OUT_PULSE_DURATION_STEPS = 25

COMMAND_CATEGORIES = ("hover", "cardinal", "diagonal", "full_simultaneous")
WIND_CATEGORIES = ("calm", "force", "torque", "combined")


@dataclass(frozen=True)
class CommandCurriculumStage:
    start_interactions: int
    maximum_horizontal_speed_m_s: float
    maximum_vertical_speed_m_s: float
    maximum_yaw_rate_rad_s: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start_interactions, int)
            or isinstance(self.start_interactions, bool)
            or self.start_interactions < 0
        ):
            raise ValueError("stage start_interactions must be a non-negative integer")
        limits = (
            self.maximum_horizontal_speed_m_s,
            self.maximum_vertical_speed_m_s,
            self.maximum_yaw_rate_rad_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("command curriculum limits must be finite and positive")

    def payload(self) -> dict[str, int | float]:
        return {
            "start_interactions": self.start_interactions,
            "maximum_horizontal_speed_m_s": self.maximum_horizontal_speed_m_s,
            "maximum_vertical_speed_m_s": self.maximum_vertical_speed_m_s,
            "maximum_yaw_rate_rad_s": self.maximum_yaw_rate_rad_s,
        }


@dataclass(frozen=True)
class WindCurriculumStage:
    start_interactions: int
    maximum_force_to_weight_ratio: float
    maximum_torque_to_weight_arm_ratio: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start_interactions, int)
            or isinstance(self.start_interactions, bool)
            or self.start_interactions < 0
        ):
            raise ValueError("wind stage start must be a non-negative integer")
        limits = (
            self.maximum_force_to_weight_ratio,
            self.maximum_torque_to_weight_arm_ratio,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("wind curriculum limits must be finite and positive")

    def payload(self) -> dict[str, int | float]:
        return {
            "start_interactions": self.start_interactions,
            "maximum_force_to_weight_ratio": self.maximum_force_to_weight_ratio,
            "maximum_torque_to_weight_arm_ratio": self.maximum_torque_to_weight_arm_ratio,
        }


@dataclass(frozen=True)
class ScheduledCommand:
    command: tuple[float, float, float, float]
    hold_steps: int
    category: str
    stage_index: int


@dataclass(frozen=True)
class ScheduledWind:
    force_ratio_world: tuple[float, float, float]
    torque_ratio_world: tuple[float, float, float]
    hold_steps: int
    category: str
    stage_index: int


COMMAND_CURRICULUM = (
    CommandCurriculumStage(0, 0.25, 0.15, 0.4),
    CommandCurriculumStage(100_000, 0.50, 0.25, 0.8),
    CommandCurriculumStage(250_000, 0.80, 0.40, 1.2),
    CommandCurriculumStage(500_000, 1.00, 0.50, 1.5),
)

WIND_CURRICULUM = (
    WindCurriculumStage(0, 0.03, 0.02),
    WindCurriculumStage(100_000, 0.05, 0.03),
    WindCurriculumStage(250_000, 0.08, 0.05),
    WindCurriculumStage(500_000, 0.12, 0.075),
    WindCurriculumStage(750_000, 0.15, 0.10),
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def command_wide_contract_payload() -> dict[str, Any]:
    """Return the compact compatibility contract common to both v2 tasks."""

    return {
        "version": COMMAND_WIDE_CONTRACT_VERSION,
        "profile": COMMAND_V2_PROFILE,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "observation_width": OBSERVATION_WIDTH,
        "action_width": ACTION_WIDTH,
        "maximum_horizontal_speed_m_s": MAXIMUM_HORIZONTAL_SPEED_M_S,
        "maximum_vertical_speed_m_s": MAXIMUM_VERTICAL_SPEED_M_S,
        "maximum_yaw_rate_rad_s": MAXIMUM_YAW_RATE_RAD_S,
    }


def wind_contract_payload() -> dict[str, Any]:
    """Return the physical and stochastic wind contract."""

    return {
        "frame": WIND_FRAME,
        "application_point": WIND_APPLICATION_POINT,
        "isaac_wrench_composer_api": "permanent_wrench_composer.set_forces_and_torques",
        "isaac_is_global": True,
        "composer_reset_before_world_transform": True,
        "reference_arm_m": WIND_REFERENCE_ARM_M,
        "training_seed": TRAINING_WIND_SEED,
        "categories": list(WIND_CATEGORIES),
        "category_cycle": True,
        "pulse_steps_inclusive": [MINIMUM_WIND_PULSE_STEPS, MAXIMUM_WIND_PULSE_STEPS],
        "calm_steps_inclusive": [MINIMUM_WIND_CALM_STEPS, MAXIMUM_WIND_CALM_STEPS],
        "active_magnitude_fraction_inclusive": [0.25, 1.0],
        "direction": "uniform_unit_sphere",
        "curriculum": [stage.payload() for stage in WIND_CURRICULUM],
        "force_scale": "vehicle_weight_n",
        "torque_scale": "vehicle_weight_n_times_reference_arm_m",
    }


def wind_evaluation_protocol_payload() -> dict[str, Any]:
    """Return the fully predeclared deterministic held-out wind protocol."""

    return {
        "version": "crazyflie_command_v2_wind_heldout_v1",
        "seed": HELD_OUT_WIND_SEED,
        "distinct_from_training_seed": HELD_OUT_WIND_SEED != TRAINING_WIND_SEED,
        "episodes": HELD_OUT_EPISODES,
        "episode_steps": HELD_OUT_EPISODE_STEPS,
        "pulse_start_steps": list(HELD_OUT_PULSE_START_STEPS),
        "pulse_duration_steps": HELD_OUT_PULSE_DURATION_STEPS,
        "pulse_categories": ["force", "torque", "combined"],
        "force_to_weight_ratio_limit": WIND_CURRICULUM[-1].maximum_force_to_weight_ratio,
        "torque_to_weight_arm_ratio_limit": WIND_CURRICULUM[-1].maximum_torque_to_weight_arm_ratio,
        "active_magnitude_fraction_inclusive": [0.25, 1.0],
        "frame": WIND_FRAME,
        "application_point": WIND_APPLICATION_POINT,
    }


def command_wide_training_contract_payload(*, wind_enabled: bool) -> dict[str, Any]:
    if not isinstance(wind_enabled, bool):
        raise TypeError("wind_enabled must be bool")
    task_id = (
        COMMAND_FOLLOW_WIDE_WIND_TASK_ID
        if wind_enabled
        else COMMAND_FOLLOW_WIDE_TASK_ID
    )
    return {
        **command_wide_contract_payload(),
        "task_id": task_id,
        "wind_enabled": wind_enabled,
        "training_interaction_budget": TRAINING_INTERACTION_BUDGET,
        "control_dt_s": CONTROL_DT_S,
        "episode_steps": EPISODE_STEPS,
        "command_order": [
            "forward_m_s",
            "left_m_s",
            "up_m_s",
            "yaw_left_rad_s",
        ],
        "observation_order": [
            "linear_velocity_error_body_x",
            "linear_velocity_error_body_y",
            "linear_velocity_error_body_z",
            "angular_velocity_body_x",
            "angular_velocity_body_y",
            "yaw_rate_error_body_z",
            "projected_gravity_body_x",
            "projected_gravity_body_y",
            "projected_gravity_body_z",
            "target_position_error_body_x",
            "target_position_error_body_y",
            "target_position_error_body_z",
        ],
        "command_hold_steps_inclusive": [
            MINIMUM_COMMAND_HOLD_STEPS,
            MAXIMUM_COMMAND_HOLD_STEPS,
        ],
        "command_categories": list(COMMAND_CATEGORIES),
        "command_curriculum": [stage.payload() for stage in COMMAND_CURRICULUM],
        "training_schedule_clock": {
            "basis": "global_control_intervals_per_environment",
            "interaction_clock": "completed_control_intervals_times_num_envs",
            "same_interval_for_every_environment": True,
            "episode_reset_advances_command_cursor": False,
            "episode_termination_gates_command_cursor": False,
            "episode_reset_advances_training_wind_cursor": False,
            "episode_termination_gates_training_wind_cursor": False,
            "schedule_independent_of_episode_termination": True,
        },
        "safety_envelope": {
            "minimum_height_m": MINIMUM_HEIGHT_M,
            "maximum_height_m": MAXIMUM_HEIGHT_M,
            "soft_workspace_xy_m": SOFT_WORKSPACE_XY_M,
            "hard_minimum_height_m": HARD_MINIMUM_HEIGHT_M,
            "hard_maximum_height_m": HARD_MAXIMUM_HEIGHT_M,
            "hard_workspace_xy_m": HARD_WORKSPACE_XY_M,
        },
        "reward_components": [
            "linear_tracking",
            "yaw_tracking",
            "tracking_progress",
            "wrong_direction_acceleration",
            "jerk",
            "target_retention",
            "attitude_stability",
            "angular_stability",
            "control_effort",
            "action_smoothness",
            "survival",
            "failure",
        ],
        "reward_contract": {
            "step_dt_s": CONTROL_DT_S,
            "dt_scaled_weights": {
                "linear_tracking": 1.50,
                "yaw_tracking": 0.30,
                "jerk": -0.002,
                "target_retention": 0.20,
                "attitude_stability": -0.20,
                "angular_stability": -0.03,
                "control_effort": -0.01,
                "action_smoothness": -0.005,
                "survival": 0.05,
            },
            "transition_weights": {
                "tracking_progress": 0.20,
                "wrong_direction_acceleration": -0.03,
            },
            "terminal_failure_penalty": -5.0,
            "linear_error_scales_m_s": [
                MAXIMUM_HORIZONTAL_SPEED_M_S,
                MAXIMUM_HORIZONTAL_SPEED_M_S,
                MAXIMUM_VERTICAL_SPEED_M_S,
            ],
            "yaw_error_scale_rad_s": MAXIMUM_YAW_RATE_RAD_S,
            "target_error_scales_m": [0.8, 0.8, 0.4],
            "wrong_acceleration_scale_m_s2": 10.0,
            "jerk_scale_m_s3": 50.0,
            "raw_acceleration_magnitude_rewarded": False,
        },
        "wind_contract": wind_contract_payload() if wind_enabled else None,
        "wind_evaluation_protocol": (
            wind_evaluation_protocol_payload() if wind_enabled else None
        ),
    }


def command_wide_training_contract_sha256(*, wind_enabled: bool) -> str:
    return _canonical_sha256(
        command_wide_training_contract_payload(wind_enabled=wind_enabled)
    )


COMMAND_WIDE_STILL_CONTRACT_SHA256 = command_wide_training_contract_sha256(
    wind_enabled=False
)
COMMAND_WIDE_WIND_CONTRACT_SHA256 = command_wide_training_contract_sha256(
    wind_enabled=True
)
WIND_EVALUATION_PROTOCOL_SHA256 = _canonical_sha256(
    wind_evaluation_protocol_payload()
)


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def command_curriculum_stage_index(total_interactions: int) -> int:
    _require_nonnegative_int(total_interactions, "total_interactions")
    active = 0
    for index, stage in enumerate(COMMAND_CURRICULUM):
        if total_interactions >= stage.start_interactions:
            active = index
    return active


def wind_curriculum_stage_index(total_interactions: int) -> int:
    _require_nonnegative_int(total_interactions, "total_interactions")
    active = 0
    for index, stage in enumerate(WIND_CURRICULUM):
        if total_interactions >= stage.start_interactions:
            active = index
    return active


def _derived_seed(namespace: str, *values: int) -> int:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("schedule coordinates must be non-negative integers")
    encoded = ":".join((COMMAND_WIDE_CONTRACT_VERSION, namespace, *(str(value) for value in values))).encode(
        "ascii"
    )
    return int.from_bytes(sha256(encoded).digest()[:8], "little")


def sample_wide_scheduled_command(
    *, seed: int, environment_id: int, segment_index: int, total_interactions: int
) -> ScheduledCommand:
    """Sample the shared still/wind command stream without global RNG state."""

    _require_nonnegative_int(seed, "seed")
    _require_nonnegative_int(environment_id, "environment_id")
    _require_nonnegative_int(segment_index, "segment_index")
    stage_index = command_curriculum_stage_index(total_interactions)
    stage = COMMAND_CURRICULUM[stage_index]
    rng = random.Random(_derived_seed("command", seed, environment_id, segment_index))
    category = COMMAND_CATEGORIES[(seed + environment_id + segment_index) % len(COMMAND_CATEGORIES)]
    hold_steps = rng.randint(MINIMUM_COMMAND_HOLD_STEPS, MAXIMUM_COMMAND_HOLD_STEPS)

    def signed_magnitude(limit: float) -> float:
        sign = -1.0 if rng.randrange(2) == 0 else 1.0
        return sign * limit * (0.35 + 0.65 * rng.random())

    forward = left = up = yaw_left = 0.0
    if category == "cardinal":
        axis = rng.randrange(4)
        values = [0.0, 0.0, 0.0, 0.0]
        values[axis] = signed_magnitude(
            stage.maximum_horizontal_speed_m_s
            if axis < 2
            else (
                stage.maximum_vertical_speed_m_s
                if axis == 2
                else stage.maximum_yaw_rate_rad_s
            )
        )
        forward, left, up, yaw_left = values
    elif category in ("diagonal", "full_simultaneous"):
        magnitude = stage.maximum_horizontal_speed_m_s * (0.35 + 0.65 * rng.random())
        component = magnitude / math.sqrt(2.0)
        forward = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
        left = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
        if category == "full_simultaneous":
            up = signed_magnitude(stage.maximum_vertical_speed_m_s)
            yaw_left = signed_magnitude(stage.maximum_yaw_rate_rad_s)
    return ScheduledCommand(
        command=(forward, left, up, yaw_left),
        hold_steps=hold_steps,
        category=category,
        stage_index=stage_index,
    )


def _unit_sphere_direction(rng: random.Random) -> tuple[float, float, float]:
    z = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    horizontal = math.sqrt(max(0.0, 1.0 - z * z))
    return (horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), z)


def _scaled_direction(
    rng: random.Random, maximum_ratio: float
) -> tuple[float, float, float]:
    magnitude = maximum_ratio * (0.25 + 0.75 * rng.random())
    direction = _unit_sphere_direction(rng)
    return tuple(magnitude * value for value in direction)


def sample_training_wind(
    *, seed: int, environment_id: int, segment_index: int, total_interactions: int
) -> ScheduledWind:
    """Sample one deterministic calm/force/torque/combined training segment."""

    _require_nonnegative_int(seed, "seed")
    _require_nonnegative_int(environment_id, "environment_id")
    _require_nonnegative_int(segment_index, "segment_index")
    stage_index = wind_curriculum_stage_index(total_interactions)
    stage = WIND_CURRICULUM[stage_index]
    rng = random.Random(_derived_seed("training-wind", seed, environment_id, segment_index))
    category = WIND_CATEGORIES[(seed + environment_id + segment_index) % len(WIND_CATEGORIES)]
    if category == "calm":
        return ScheduledWind(
            force_ratio_world=(0.0, 0.0, 0.0),
            torque_ratio_world=(0.0, 0.0, 0.0),
            hold_steps=rng.randint(MINIMUM_WIND_CALM_STEPS, MAXIMUM_WIND_CALM_STEPS),
            category=category,
            stage_index=stage_index,
        )
    force = (
        _scaled_direction(rng, stage.maximum_force_to_weight_ratio)
        if category in ("force", "combined")
        else (0.0, 0.0, 0.0)
    )
    torque = (
        _scaled_direction(rng, stage.maximum_torque_to_weight_arm_ratio)
        if category in ("torque", "combined")
        else (0.0, 0.0, 0.0)
    )
    return ScheduledWind(
        force_ratio_world=force,
        torque_ratio_world=torque,
        hold_steps=rng.randint(MINIMUM_WIND_PULSE_STEPS, MAXIMUM_WIND_PULSE_STEPS),
        category=category,
        stage_index=stage_index,
    )


def held_out_wind_at_step(
    episode_index: int,
    step: int,
    *,
    seed: int = HELD_OUT_WIND_SEED,
) -> ScheduledWind:
    """Return the held-out wrench active at one zero-based episode step."""

    _require_nonnegative_int(episode_index, "episode_index")
    _require_nonnegative_int(step, "step")
    _require_nonnegative_int(seed, "seed")
    if episode_index >= HELD_OUT_EPISODES:
        raise ValueError("held-out episode index is outside the 16-episode protocol")
    if step >= HELD_OUT_EPISODE_STEPS:
        raise ValueError("held-out step is outside the 600-step protocol")
    active_pulse = next(
        (
            pulse_index
            for pulse_index, start in enumerate(HELD_OUT_PULSE_START_STEPS)
            if start <= step < start + HELD_OUT_PULSE_DURATION_STEPS
        ),
        None,
    )
    if active_pulse is None:
        next_boundary = min(
            (
                boundary
                for start in HELD_OUT_PULSE_START_STEPS
                for boundary in (start, start + HELD_OUT_PULSE_DURATION_STEPS)
                if boundary > step
            ),
            default=HELD_OUT_EPISODE_STEPS,
        )
        return ScheduledWind(
            force_ratio_world=(0.0, 0.0, 0.0),
            torque_ratio_world=(0.0, 0.0, 0.0),
            hold_steps=next_boundary - step,
            category="calm",
            stage_index=len(WIND_CURRICULUM) - 1,
        )
    rng = random.Random(_derived_seed("heldout-wind", seed, episode_index, active_pulse))
    pulse_categories = ("force", "torque", "combined")
    category = pulse_categories[(episode_index + active_pulse) % len(pulse_categories)]
    stage = WIND_CURRICULUM[-1]
    force = (
        _scaled_direction(rng, stage.maximum_force_to_weight_ratio)
        if category in ("force", "combined")
        else (0.0, 0.0, 0.0)
    )
    torque = (
        _scaled_direction(rng, stage.maximum_torque_to_weight_arm_ratio)
        if category in ("torque", "combined")
        else (0.0, 0.0, 0.0)
    )
    pulse_start = HELD_OUT_PULSE_START_STEPS[active_pulse]
    return ScheduledWind(
        force_ratio_world=force,
        torque_ratio_world=torque,
        hold_steps=pulse_start + HELD_OUT_PULSE_DURATION_STEPS - step,
        category=category,
        stage_index=len(WIND_CURRICULUM) - 1,
    )


def held_out_wind_script(
    episode_index: int, *, seed: int = HELD_OUT_WIND_SEED
) -> tuple[ScheduledWind, ...]:
    """Materialize the exact 600-step held-out protocol for audit/testing."""

    return tuple(
        held_out_wind_at_step(episode_index, step, seed=seed)
        for step in range(HELD_OUT_EPISODE_STEPS)
    )


def wind_wrench_from_ratios(
    force_ratio_world: tuple[float, float, float],
    torque_ratio_world: tuple[float, float, float],
    *,
    vehicle_weight_n: float,
    reference_arm_m: float = WIND_REFERENCE_ARM_M,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert dimensionless ratios to a world-frame physical wrench."""

    if len(force_ratio_world) != 3 or len(torque_ratio_world) != 3:
        raise ValueError("wind ratios must be three-vectors")
    if (
        not math.isfinite(vehicle_weight_n)
        or vehicle_weight_n <= 0.0
        or not math.isfinite(reference_arm_m)
        or reference_arm_m <= 0.0
        or any(not math.isfinite(value) for value in (*force_ratio_world, *torque_ratio_world))
    ):
        raise ValueError("wind wrench inputs must be finite and physical")
    force = tuple(vehicle_weight_n * value for value in force_ratio_world)
    torque = tuple(vehicle_weight_n * reference_arm_m * value for value in torque_ratio_world)
    return force, torque


def _require_matrix(value: torch.Tensor, width: int, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch, {width}]")


def bound_wide_command_body(command_body: torch.Tensor) -> torch.Tensor:
    """Bound commands to the command-v2 1.0/0.5/1.5 envelope."""

    _require_matrix(command_body, COMMAND_WIDTH, "command_body")
    if not bool(torch.isfinite(command_body).all()):
        raise FloatingPointError("command_body contains nonfinite values")
    bounded = command_body.clone()
    horizontal_norm = torch.linalg.vector_norm(bounded[:, :2], dim=1, keepdim=True)
    bounded[:, :2] *= torch.clamp(
        MAXIMUM_HORIZONTAL_SPEED_M_S / horizontal_norm.clamp_min(1.0e-12),
        max=1.0,
    )
    bounded[:, 2].clamp_(-MAXIMUM_VERTICAL_SPEED_M_S, MAXIMUM_VERTICAL_SPEED_M_S)
    bounded[:, 3].clamp_(-MAXIMUM_YAW_RATE_RAD_S, MAXIMUM_YAW_RATE_RAD_S)
    return bounded


def _yaw_from_quaternion_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    _require_matrix(quaternion_wxyz, 4, "quaternion_wxyz")
    w, x, y, z = quaternion_wxyz.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _body_xy_to_world(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * values[:, 0] - sine * values[:, 1],
            sine * values[:, 0] + cosine * values[:, 1],
        ),
        dim=1,
    )


def _world_xy_to_body(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * values[:, 0] + sine * values[:, 1],
            -sine * values[:, 0] + cosine * values[:, 1],
        ),
        dim=1,
    )


def apply_wide_command_safety_envelope(
    command_body: torch.Tensor,
    root_position_w: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
    command_target_position_w: torch.Tensor,
    workspace_origin_w: torch.Tensor,
    *,
    minimum_height_m: float = MINIMUM_HEIGHT_M,
    maximum_height_m: float = MAXIMUM_HEIGHT_M,
    maximum_horizontal_offset_m: float = SOFT_WORKSPACE_XY_M,
    margin_m: float = SAFETY_MARGIN_M,
) -> torch.Tensor:
    """Remove only v2 command components that point outside the soft box."""

    for value, width, name in (
        (command_body, 4, "command_body"),
        (root_position_w, 3, "root_position_w"),
        (root_quaternion_wxyz, 4, "root_quaternion_wxyz"),
        (command_target_position_w, 3, "command_target_position_w"),
        (workspace_origin_w, 3, "workspace_origin_w"),
    ):
        _require_matrix(value, width, name)
    batch = command_body.shape[0]
    inputs = (
        command_body,
        root_position_w,
        root_quaternion_wxyz,
        command_target_position_w,
        workspace_origin_w,
    )
    if any(value.shape[0] != batch for value in inputs):
        raise ValueError("safety-envelope tensors must share one batch size")
    if not all(bool(torch.isfinite(value).all()) for value in inputs):
        raise FloatingPointError("safety-envelope inputs must be finite")
    if (
        not math.isfinite(minimum_height_m)
        or not math.isfinite(maximum_height_m)
        or minimum_height_m >= maximum_height_m
        or not math.isfinite(maximum_horizontal_offset_m)
        or maximum_horizontal_offset_m <= 0.0
        or not math.isfinite(margin_m)
        or margin_m < 0.0
    ):
        raise ValueError("invalid command safety envelope")

    effective = bound_wide_command_body(command_body)
    yaw = _yaw_from_quaternion_wxyz(root_quaternion_wxyz)
    horizontal_world = _body_xy_to_world(effective[:, :2], yaw)
    low_xy = workspace_origin_w[:, :2] - maximum_horizontal_offset_m
    high_xy = workspace_origin_w[:, :2] + maximum_horizontal_offset_m
    for axis in range(2):
        high = (
            (command_target_position_w[:, axis] >= high_xy[:, axis] - margin_m)
            | (root_position_w[:, axis] >= high_xy[:, axis] - margin_m)
        ) & (horizontal_world[:, axis] > 0.0)
        low = (
            (command_target_position_w[:, axis] <= low_xy[:, axis] + margin_m)
            | (root_position_w[:, axis] <= low_xy[:, axis] + margin_m)
        ) & (horizontal_world[:, axis] < 0.0)
        horizontal_world[:, axis] = torch.where(
            high | low,
            torch.zeros_like(horizontal_world[:, axis]),
            horizontal_world[:, axis],
        )
    effective[:, :2] = _world_xy_to_body(horizontal_world, yaw)
    block_up = (
        (command_target_position_w[:, 2] >= maximum_height_m - margin_m)
        | (root_position_w[:, 2] >= maximum_height_m - margin_m)
    ) & (effective[:, 2] > 0.0)
    block_down = (
        (command_target_position_w[:, 2] <= minimum_height_m + margin_m)
        | (root_position_w[:, 2] <= minimum_height_m + margin_m)
    ) & (effective[:, 2] < 0.0)
    effective[:, 2] = torch.where(
        block_up | block_down,
        torch.zeros_like(effective[:, 2]),
        effective[:, 2],
    )
    return effective


def command_wide_reward_terms(
    tracking_error_body: torch.Tensor,
    previous_tracking_error_body: torch.Tensor,
    linear_acceleration_b: torch.Tensor,
    linear_jerk_b: torch.Tensor,
    root_angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    target_position_error_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    terminated: torch.Tensor,
    *,
    step_dt_s: float = CONTROL_DT_S,
) -> dict[str, torch.Tensor]:
    """The command-v1 reward with only the declared v2 tracking scales widened."""

    for value, width, name in (
        (tracking_error_body, 4, "tracking_error_body"),
        (previous_tracking_error_body, 4, "previous_tracking_error_body"),
        (linear_acceleration_b, 3, "linear_acceleration_b"),
        (linear_jerk_b, 3, "linear_jerk_b"),
        (root_angular_velocity_b, 3, "root_angular_velocity_b"),
        (projected_gravity_b, 3, "projected_gravity_b"),
        (target_position_error_b, 3, "target_position_error_b"),
        (actions, 4, "actions"),
        (previous_actions, 4, "previous_actions"),
    ):
        _require_matrix(value, width, name)
    if not isinstance(terminated, torch.Tensor) or terminated.ndim != 1:
        raise ValueError("terminated must have shape [batch]")
    batch = tracking_error_body.shape[0]
    values = (
        previous_tracking_error_body,
        linear_acceleration_b,
        linear_jerk_b,
        root_angular_velocity_b,
        projected_gravity_b,
        target_position_error_b,
        actions,
        previous_actions,
        terminated,
    )
    if any(value.shape[0] != batch for value in values):
        raise ValueError("reward tensors must share one batch size")
    if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")

    tracking = torch.nan_to_num(tracking_error_body, nan=10.0, posinf=10.0, neginf=-10.0)
    previous_tracking = torch.nan_to_num(
        previous_tracking_error_body, nan=10.0, posinf=10.0, neginf=-10.0
    )
    acceleration = torch.nan_to_num(linear_acceleration_b, nan=100.0, posinf=100.0, neginf=-100.0)
    jerk = torch.nan_to_num(linear_jerk_b, nan=1000.0, posinf=1000.0, neginf=-1000.0)
    angular = torch.nan_to_num(root_angular_velocity_b, nan=10.0, posinf=10.0, neginf=-10.0)
    gravity = torch.nan_to_num(projected_gravity_b, nan=1.0, posinf=1.0, neginf=-1.0)
    target_error = torch.nan_to_num(target_position_error_b, nan=10.0, posinf=10.0, neginf=-10.0)
    safe_actions = torch.nan_to_num(actions, nan=1.0, posinf=1.0, neginf=-1.0)
    safe_previous = torch.nan_to_num(previous_actions, nan=1.0, posinf=1.0, neginf=-1.0)

    linear_scale = tracking.new_tensor(
        [MAXIMUM_HORIZONTAL_SPEED_M_S, MAXIMUM_HORIZONTAL_SPEED_M_S, MAXIMUM_VERTICAL_SPEED_M_S]
    )
    linear_error_sq = ((tracking[:, :3] / linear_scale).square().sum(dim=1)).clamp(max=25.0)
    previous_linear_error_norm = torch.linalg.vector_norm(previous_tracking[:, :3] / linear_scale, dim=1)
    current_linear_error_norm = torch.sqrt(linear_error_sq)
    tracking_progress = (previous_linear_error_norm - current_linear_error_norm).clamp(-1.0, 1.0)
    previous_error_unit = previous_tracking[:, :3] / torch.linalg.vector_norm(
        previous_tracking[:, :3], dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    wrong_direction_acceleration = torch.relu(
        torch.sum(acceleration * previous_error_unit, dim=1) / 10.0
    ).clamp(max=1.0)
    normalized_jerk_squared = (torch.linalg.vector_norm(jerk, dim=1) / 50.0).square().clamp(max=25.0)
    yaw_error_sq = (tracking[:, 3] / MAXIMUM_YAW_RATE_RAD_S).square().clamp(max=25.0)
    target_scale = target_error.new_tensor([0.8, 0.8, 0.4])
    target_error_sq = ((target_error / target_scale).square().sum(dim=1)).clamp(max=25.0)
    interval = float(step_dt_s)
    components = {
        "linear_tracking": 1.50 * torch.exp(-linear_error_sq) * interval,
        "yaw_tracking": 0.30 * torch.exp(-yaw_error_sq) * interval,
        "tracking_progress": 0.20 * tracking_progress,
        "wrong_direction_acceleration": -0.03 * wrong_direction_acceleration,
        "jerk": -0.002 * normalized_jerk_squared * interval,
        "target_retention": 0.20 * torch.exp(-target_error_sq) * interval,
        "attitude_stability": -0.20 * gravity[:, :2].square().sum(dim=1).clamp(max=25.0) * interval,
        "angular_stability": -0.03 * angular[:, :2].square().sum(dim=1).clamp(max=25.0) * interval,
        "control_effort": -0.01 * (
            ((safe_actions[:, 0] - CRAZYFLIE_HOVER_ACTION) / 0.35).square()
            + (safe_actions[:, 1:] / 0.08).square().sum(dim=1)
        ).clamp(max=25.0) * interval,
        "action_smoothness": -0.005 * (
            ((safe_actions - safe_previous) / safe_actions.new_tensor([0.35, 0.08, 0.08, 0.08]))
            .square()
            .sum(dim=1)
            .clamp(max=25.0)
        ) * interval,
        "survival": torch.full_like(linear_error_sq, 0.05 * interval)
        * (~terminated.to(dtype=torch.bool)).to(dtype=linear_error_sq.dtype),
        "failure": -5.0 * terminated.to(dtype=linear_error_sq.dtype),
    }
    components["total"] = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return components


__all__ = [
    "ACTION_WIDTH",
    "COMMAND_CATEGORIES",
    "COMMAND_CURRICULUM",
    "COMMAND_FOLLOW_WIDE_TASK_ID",
    "COMMAND_FOLLOW_WIDE_WIND_TASK_ID",
    "COMMAND_V2_PROFILE",
    "COMMAND_WIDE_CONTRACT_VERSION",
    "COMMAND_WIDE_SCHEDULE_STATE_KIND",
    "COMMAND_WIDE_STILL_CONTRACT_SHA256",
    "COMMAND_WIDE_WIND_CONTRACT_SHA256",
    "COMMAND_WIDTH",
    "CONTROL_DT_S",
    "CommandCurriculumStage",
    "EPISODE_STEPS",
    "FAILURE_CAUSE_NAMES",
    "FAILURE_HIGH_HEIGHT",
    "FAILURE_LOW_HEIGHT",
    "FAILURE_NONE",
    "FAILURE_NONFINITE",
    "FAILURE_WORKSPACE_ESCAPE",
    "FailureClassification",
    "HARD_MAXIMUM_HEIGHT_M",
    "HARD_MINIMUM_HEIGHT_M",
    "HARD_WORKSPACE_XY_M",
    "HELD_OUT_EPISODES",
    "HELD_OUT_EPISODE_STEPS",
    "HELD_OUT_PULSE_DURATION_STEPS",
    "HELD_OUT_PULSE_START_STEPS",
    "HELD_OUT_WIND_SEED",
    "MAXIMUM_COMMAND_HOLD_STEPS",
    "MAXIMUM_HEIGHT_M",
    "MAXIMUM_HORIZONTAL_SPEED_M_S",
    "MAXIMUM_VERTICAL_SPEED_M_S",
    "MAXIMUM_WIND_CALM_STEPS",
    "MAXIMUM_WIND_PULSE_STEPS",
    "MAXIMUM_YAW_RATE_RAD_S",
    "MINIMUM_COMMAND_HOLD_STEPS",
    "MINIMUM_HEIGHT_M",
    "MINIMUM_WIND_CALM_STEPS",
    "MINIMUM_WIND_PULSE_STEPS",
    "OBSERVATION_CONTRACT_VERSION",
    "OBSERVATION_WIDTH",
    "SOFT_WORKSPACE_XY_M",
    "ScheduledCommand",
    "ScheduledWind",
    "TRAINING_INTERACTION_BUDGET",
    "TRAINING_WIND_SEED",
    "WIND_APPLICATION_POINT",
    "WIND_CATEGORIES",
    "WIND_CURRICULUM",
    "WIND_EVALUATION_PROTOCOL_SHA256",
    "WIND_FRAME",
    "WIND_REFERENCE_ARM_M",
    "WindCurriculumStage",
    "apply_wide_command_safety_envelope",
    "bound_wide_command_body",
    "classify_command_failures",
    "command_conditioned_observation",
    "command_curriculum_stage_index",
    "command_wide_contract_payload",
    "command_wide_reward_terms",
    "command_wide_training_contract_payload",
    "command_wide_training_contract_sha256",
    "held_out_wind_at_step",
    "held_out_wind_script",
    "integrate_command_target",
    "sample_training_wind",
    "sample_wide_scheduled_command",
    "wind_contract_payload",
    "wind_curriculum_stage_index",
    "wind_evaluation_protocol_payload",
    "wind_wrench_from_ratios",
]
