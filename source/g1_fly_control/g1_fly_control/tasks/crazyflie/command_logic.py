"""Pure command-following contracts and tensor logic for Crazyflie.

This module deliberately has no Isaac Lab imports.  It defines the one-task
velocity-command curriculum, deterministic command schedule, safety envelope,
12-value policy observation, continuous reward, and hard-failure classifier
used by both training and keyboard playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any

import torch


COMMAND_FOLLOW_TASK_ID = "FlyCrazyflie-CommandFollow-v0"
COMMAND_FOLLOW_CONTRACT_VERSION = "crazyflie_command_follow_v1"
COMMAND_TRACKING_CONTRACT_VERSION = COMMAND_FOLLOW_CONTRACT_VERSION
OBSERVATION_CONTRACT_VERSION = "command_error_12_v1"

OBSERVATION_WIDTH = 12
ACTION_WIDTH = 4
COMMAND_WIDTH = 4
CONTROL_DT_S = 0.02

MAXIMUM_HORIZONTAL_SPEED_M_S = 0.8
MAXIMUM_VERTICAL_SPEED_M_S = 0.4
MAXIMUM_YAW_RATE_RAD_S = 1.2
MINIMUM_COMMAND_HOLD_STEPS = 25
MAXIMUM_COMMAND_HOLD_STEPS = 100

MINIMUM_HEIGHT_M = 0.30
MAXIMUM_HEIGHT_M = 1.70
HARD_MINIMUM_HEIGHT_M = 0.15
HARD_MAXIMUM_HEIGHT_M = 1.90
SOFT_WORKSPACE_XY_M = 4.0
HARD_WORKSPACE_XY_M = 4.5
SAFETY_MARGIN_M = 0.02

CRAZYFLIE_HOVER_ACTION = 2.0 / 1.9 - 1.0

FAILURE_NONE = 0
FAILURE_LOW_HEIGHT = 1
FAILURE_HIGH_HEIGHT = 2
FAILURE_WORKSPACE_ESCAPE = 3
FAILURE_NONFINITE = 4
FAILURE_CAUSE_NAMES = {
    FAILURE_NONE: "none",
    FAILURE_LOW_HEIGHT: "low_height",
    FAILURE_HIGH_HEIGHT: "high_height",
    FAILURE_WORKSPACE_ESCAPE: "workspace_escape",
    FAILURE_NONFINITE: "nonfinite",
}


@dataclass(frozen=True)
class CommandCurriculumStage:
    """One inclusive-from interaction boundary of the velocity curriculum."""

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


COMMAND_CURRICULUM = (
    CommandCurriculumStage(0, 0.25, 0.15, 0.4),
    CommandCurriculumStage(100_000, 0.50, 0.25, 0.8),
    CommandCurriculumStage(250_000, 0.80, 0.40, 1.2),
)


@dataclass(frozen=True)
class ScheduledCommand:
    """A deterministic body-frame command segment."""

    command: tuple[float, float, float, float]
    hold_steps: int
    category: str
    stage_index: int


@dataclass(frozen=True)
class FailureClassification:
    terminated: torch.Tensor
    cause: torch.Tensor


def command_follow_contract_payload() -> dict[str, Any]:
    """Return the small compatibility contract stored in every checkpoint."""

    return {
        "version": COMMAND_FOLLOW_CONTRACT_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "maximum_horizontal_speed_m_s": MAXIMUM_HORIZONTAL_SPEED_M_S,
        "maximum_vertical_speed_m_s": MAXIMUM_VERTICAL_SPEED_M_S,
        "maximum_yaw_rate_rad_s": MAXIMUM_YAW_RATE_RAD_S,
    }


def command_training_contract_payload() -> dict[str, Any]:
    """Return the complete task contract used for source/result provenance."""

    return {
        **command_follow_contract_payload(),
        "task_id": COMMAND_FOLLOW_TASK_ID,
        "control_dt_s": CONTROL_DT_S,
        "episode_steps": 600,
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
        "categories": ["hover", "cardinal", "diagonal", "full_simultaneous"],
        "curriculum": [stage.payload() for stage in COMMAND_CURRICULUM],
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
    }


def command_training_contract_sha256() -> str:
    encoded = json.dumps(
        command_training_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def command_tracking_contract_payload() -> dict[str, Any]:
    """Compatibility name used by the command trainer and checkpoint loader."""

    return command_training_contract_payload()


COMMAND_TRACKING_CONTRACT_SHA256 = command_training_contract_sha256()


def curriculum_stage_index(total_interactions: int) -> int:
    if (
        not isinstance(total_interactions, int)
        or isinstance(total_interactions, bool)
        or total_interactions < 0
    ):
        raise ValueError("total_interactions must be a non-negative integer")
    active = 0
    for index, stage in enumerate(COMMAND_CURRICULUM):
        if total_interactions >= stage.start_interactions:
            active = index
    return active


def curriculum_stage(total_interactions: int) -> CommandCurriculumStage:
    return COMMAND_CURRICULUM[curriculum_stage_index(total_interactions)]


def _schedule_seed(seed: int, environment_id: int, segment_index: int) -> int:
    values = (seed, environment_id, segment_index)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("schedule seed, environment id, and segment index must be non-negative integers")
    encoded = f"{COMMAND_FOLLOW_CONTRACT_VERSION}:{seed}:{environment_id}:{segment_index}".encode(
        "ascii"
    )
    return int.from_bytes(sha256(encoded).digest()[:8], "little")


def sample_scheduled_command(
    *,
    seed: int,
    environment_id: int,
    segment_index: int,
    total_interactions: int,
) -> ScheduledCommand:
    """Sample one deterministic segment without consuming global RNG state.

    Categories cycle rather than being left to chance.  Consequently every
    four consecutive segments per environment exercise hover, one-axis,
    diagonal-horizontal, and fully simultaneous commands.
    """

    stage_index = curriculum_stage_index(total_interactions)
    stage = COMMAND_CURRICULUM[stage_index]
    rng = random.Random(_schedule_seed(seed, environment_id, segment_index))
    categories = ("hover", "cardinal", "diagonal", "full_simultaneous")
    category = categories[(seed + environment_id + segment_index) % len(categories)]
    hold_steps = rng.randint(MINIMUM_COMMAND_HOLD_STEPS, MAXIMUM_COMMAND_HOLD_STEPS)

    def signed_magnitude(limit: float) -> float:
        sign = -1.0 if rng.randrange(2) == 0 else 1.0
        return sign * limit * (0.35 + 0.65 * rng.random())

    forward = left = up = yaw_left = 0.0
    if category == "cardinal":
        axis = rng.randrange(4)
        if axis == 0:
            forward = signed_magnitude(stage.maximum_horizontal_speed_m_s)
        elif axis == 1:
            left = signed_magnitude(stage.maximum_horizontal_speed_m_s)
        elif axis == 2:
            up = signed_magnitude(stage.maximum_vertical_speed_m_s)
        else:
            yaw_left = signed_magnitude(stage.maximum_yaw_rate_rad_s)
    elif category == "diagonal":
        magnitude = stage.maximum_horizontal_speed_m_s * (0.35 + 0.65 * rng.random())
        component = magnitude / math.sqrt(2.0)
        forward = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
        left = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
    elif category == "full_simultaneous":
        magnitude = stage.maximum_horizontal_speed_m_s * (0.35 + 0.65 * rng.random())
        component = magnitude / math.sqrt(2.0)
        forward = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
        left = component * (-1.0 if rng.randrange(2) == 0 else 1.0)
        up = signed_magnitude(stage.maximum_vertical_speed_m_s)
        yaw_left = signed_magnitude(stage.maximum_yaw_rate_rad_s)

    return ScheduledCommand(
        command=(forward, left, up, yaw_left),
        hold_steps=hold_steps,
        category=category,
        stage_index=stage_index,
    )


def _require_matrix(value: torch.Tensor, width: int, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch, {width}]")


def bound_command_body(
    command_body: torch.Tensor,
    *,
    maximum_horizontal_speed_m_s: float = MAXIMUM_HORIZONTAL_SPEED_M_S,
    maximum_vertical_speed_m_s: float = MAXIMUM_VERTICAL_SPEED_M_S,
    maximum_yaw_rate_rad_s: float = MAXIMUM_YAW_RATE_RAD_S,
) -> torch.Tensor:
    """Return a finite four-axis command with a norm-bounded horizontal pair."""

    _require_matrix(command_body, COMMAND_WIDTH, "command_body")
    limits = (
        maximum_horizontal_speed_m_s,
        maximum_vertical_speed_m_s,
        maximum_yaw_rate_rad_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("command limits must be finite and positive")
    if not bool(torch.isfinite(command_body).all()):
        raise FloatingPointError("command_body contains nonfinite values")
    bounded = command_body.clone()
    horizontal_norm = torch.linalg.vector_norm(bounded[:, :2], dim=1, keepdim=True)
    scale = torch.clamp(
        maximum_horizontal_speed_m_s / horizontal_norm.clamp_min(1.0e-12),
        max=1.0,
    )
    bounded[:, :2] *= scale
    bounded[:, 2].clamp_(-maximum_vertical_speed_m_s, maximum_vertical_speed_m_s)
    bounded[:, 3].clamp_(-maximum_yaw_rate_rad_s, maximum_yaw_rate_rad_s)
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


def apply_command_safety_envelope(
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
    """Remove only command components that point out of the soft envelope."""

    _require_matrix(command_body, COMMAND_WIDTH, "command_body")
    _require_matrix(root_position_w, 3, "root_position_w")
    _require_matrix(root_quaternion_wxyz, 4, "root_quaternion_wxyz")
    _require_matrix(command_target_position_w, 3, "command_target_position_w")
    _require_matrix(workspace_origin_w, 3, "workspace_origin_w")
    batch = command_body.shape[0]
    if any(
        value.shape[0] != batch
        for value in (
            root_position_w,
            root_quaternion_wxyz,
            command_target_position_w,
            workspace_origin_w,
        )
    ):
        raise ValueError("safety-envelope tensors must share one batch size")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            command_body,
            root_position_w,
            root_quaternion_wxyz,
            command_target_position_w,
            workspace_origin_w,
        )
    ):
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

    effective = bound_command_body(command_body)
    yaw = _yaw_from_quaternion_wxyz(root_quaternion_wxyz)
    horizontal_world = _body_xy_to_world(effective[:, :2], yaw)
    low_xy = workspace_origin_w[:, :2] - maximum_horizontal_offset_m
    high_xy = workspace_origin_w[:, :2] + maximum_horizontal_offset_m
    # Protect against both a target at the boundary and a vehicle that has
    # drifted farther than its target.  Tangential/inward motion is retained.
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


def integrate_command_target(
    command_target_position_w: torch.Tensor,
    effective_command_body: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
    workspace_origin_w: torch.Tensor,
    *,
    dt_s: float = CONTROL_DT_S,
    minimum_height_m: float = MINIMUM_HEIGHT_M,
    maximum_height_m: float = MAXIMUM_HEIGHT_M,
    maximum_horizontal_offset_m: float = SOFT_WORKSPACE_XY_M,
) -> torch.Tensor:
    """Integrate the safe body-relative linear command into a bounded target."""

    _require_matrix(command_target_position_w, 3, "command_target_position_w")
    _require_matrix(effective_command_body, COMMAND_WIDTH, "effective_command_body")
    _require_matrix(root_quaternion_wxyz, 4, "root_quaternion_wxyz")
    _require_matrix(workspace_origin_w, 3, "workspace_origin_w")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if len({value.shape[0] for value in (
        command_target_position_w,
        effective_command_body,
        root_quaternion_wxyz,
        workspace_origin_w,
    )}) != 1:
        raise ValueError("target integration tensors must share one batch size")
    yaw = _yaw_from_quaternion_wxyz(root_quaternion_wxyz)
    horizontal_world = _body_xy_to_world(effective_command_body[:, :2], yaw)
    result = command_target_position_w.clone()
    result[:, :2] += horizontal_world * dt_s
    result[:, 2] += effective_command_body[:, 2] * dt_s
    low_xy = workspace_origin_w[:, :2] - maximum_horizontal_offset_m
    high_xy = workspace_origin_w[:, :2] + maximum_horizontal_offset_m
    result[:, :2] = torch.maximum(torch.minimum(result[:, :2], high_xy), low_xy)
    result[:, 2].clamp_(minimum_height_m, maximum_height_m)
    return result


def command_conditioned_observation(
    root_linear_velocity_b: torch.Tensor,
    root_angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    effective_command_body: torch.Tensor,
    target_position_error_b: torch.Tensor,
) -> torch.Tensor:
    """Build the exact 12-value command-error observation."""

    for value, width, name in (
        (root_linear_velocity_b, 3, "root_linear_velocity_b"),
        (root_angular_velocity_b, 3, "root_angular_velocity_b"),
        (projected_gravity_b, 3, "projected_gravity_b"),
        (effective_command_body, 4, "effective_command_body"),
        (target_position_error_b, 3, "target_position_error_b"),
    ):
        _require_matrix(value, width, name)
    batch_sizes = {
        value.shape[0]
        for value in (
            root_linear_velocity_b,
            root_angular_velocity_b,
            projected_gravity_b,
            effective_command_body,
            target_position_error_b,
        )
    }
    if len(batch_sizes) != 1:
        raise ValueError("observation tensors must share one batch size")
    linear_error = root_linear_velocity_b - effective_command_body[:, :3]
    angular_error = root_angular_velocity_b.clone()
    angular_error[:, 2] -= effective_command_body[:, 3]
    return torch.cat(
        (linear_error, angular_error, projected_gravity_b, target_position_error_b),
        dim=1,
    )


def command_follow_reward_terms(
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
    """Continuous command-tracking reward with explicit safety/control terms."""

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
    if any(
        value.shape[0] != batch
        for value in (
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
    ):
        raise ValueError("reward tensors must share one batch size")
    if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")

    # Nonfinite terminal rows are assigned a finite failure reward.  Sanitizing
    # each dense input prevents NaN propagation before the one-time penalty.
    tracking = torch.nan_to_num(tracking_error_body, nan=10.0, posinf=10.0, neginf=-10.0)
    previous_tracking = torch.nan_to_num(
        previous_tracking_error_body, nan=10.0, posinf=10.0, neginf=-10.0
    )
    acceleration = torch.nan_to_num(
        linear_acceleration_b, nan=100.0, posinf=100.0, neginf=-100.0
    )
    jerk = torch.nan_to_num(
        linear_jerk_b, nan=1000.0, posinf=1000.0, neginf=-1000.0
    )
    angular = torch.nan_to_num(root_angular_velocity_b, nan=10.0, posinf=10.0, neginf=-10.0)
    gravity = torch.nan_to_num(projected_gravity_b, nan=1.0, posinf=1.0, neginf=-1.0)
    target_error = torch.nan_to_num(target_position_error_b, nan=10.0, posinf=10.0, neginf=-10.0)
    safe_actions = torch.nan_to_num(actions, nan=1.0, posinf=1.0, neginf=-1.0)
    safe_previous = torch.nan_to_num(previous_actions, nan=1.0, posinf=1.0, neginf=-1.0)

    linear_scale = tracking.new_tensor(
        [MAXIMUM_HORIZONTAL_SPEED_M_S, MAXIMUM_HORIZONTAL_SPEED_M_S, MAXIMUM_VERTICAL_SPEED_M_S]
    )
    linear_error_sq = ((tracking[:, :3] / linear_scale).square().sum(dim=1)).clamp(max=25.0)
    previous_linear_error_norm = torch.linalg.vector_norm(
        previous_tracking[:, :3] / linear_scale, dim=1
    )
    current_linear_error_norm = torch.sqrt(linear_error_sq)
    tracking_progress = (
        previous_linear_error_norm - current_linear_error_norm
    ).clamp(-1.0, 1.0)
    previous_error_unit = previous_tracking[:, :3] / torch.linalg.vector_norm(
        previous_tracking[:, :3], dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    # Positive projection means acceleration points in the same direction as
    # actual-minus-command error and therefore makes tracking worse.
    wrong_direction_acceleration = torch.relu(
        torch.sum(acceleration * previous_error_unit, dim=1) / 10.0
    ).clamp(max=1.0)
    normalized_jerk_squared = (
        torch.linalg.vector_norm(jerk, dim=1) / 50.0
    ).square().clamp(max=25.0)
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
        "survival": (
            torch.full_like(linear_error_sq, 0.05 * interval)
            * (~terminated.to(dtype=torch.bool)).to(dtype=linear_error_sq.dtype)
        ),
        "failure": -5.0 * terminated.to(dtype=linear_error_sq.dtype),
    }
    components["total"] = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return components


def classify_command_failures(
    root_position_w: torch.Tensor,
    workspace_origin_w: torch.Tensor,
    finite_state: torch.Tensor,
    *,
    minimum_height_m: float = HARD_MINIMUM_HEIGHT_M,
    maximum_height_m: float = HARD_MAXIMUM_HEIGHT_M,
    workspace_xy_limit_m: float = HARD_WORKSPACE_XY_M,
) -> FailureClassification:
    """Classify one mutually exclusive hard-failure cause per environment."""

    _require_matrix(root_position_w, 3, "root_position_w")
    _require_matrix(workspace_origin_w, 3, "workspace_origin_w")
    if root_position_w.shape != workspace_origin_w.shape:
        raise ValueError("position and workspace origin shapes must match")
    if not isinstance(finite_state, torch.Tensor) or finite_state.shape != (root_position_w.shape[0],):
        raise ValueError("finite_state must have shape [batch]")
    finite = finite_state.to(dtype=torch.bool)
    safe_position = torch.nan_to_num(root_position_w, nan=0.0, posinf=0.0, neginf=0.0)
    relative_xy = safe_position[:, :2] - workspace_origin_w[:, :2]
    low = finite & (safe_position[:, 2] < minimum_height_m)
    high = finite & ~low & (safe_position[:, 2] > maximum_height_m)
    escaped = finite & ~low & ~high & (
        torch.max(torch.abs(relative_xy), dim=1).values > workspace_xy_limit_m
    )
    nonfinite = ~finite
    cause = torch.zeros(root_position_w.shape[0], dtype=torch.long, device=root_position_w.device)
    cause[low] = FAILURE_LOW_HEIGHT
    cause[high] = FAILURE_HIGH_HEIGHT
    cause[escaped] = FAILURE_WORKSPACE_ESCAPE
    cause[nonfinite] = FAILURE_NONFINITE
    return FailureClassification(terminated=cause != FAILURE_NONE, cause=cause)


__all__ = [
    "ACTION_WIDTH",
    "COMMAND_CURRICULUM",
    "COMMAND_FOLLOW_CONTRACT_VERSION",
    "COMMAND_FOLLOW_TASK_ID",
    "COMMAND_TRACKING_CONTRACT_SHA256",
    "COMMAND_TRACKING_CONTRACT_VERSION",
    "COMMAND_WIDTH",
    "CONTROL_DT_S",
    "CRAZYFLIE_HOVER_ACTION",
    "CommandCurriculumStage",
    "FAILURE_CAUSE_NAMES",
    "FAILURE_HIGH_HEIGHT",
    "FAILURE_LOW_HEIGHT",
    "FAILURE_NONE",
    "FAILURE_NONFINITE",
    "FAILURE_WORKSPACE_ESCAPE",
    "HARD_MAXIMUM_HEIGHT_M",
    "HARD_MINIMUM_HEIGHT_M",
    "HARD_WORKSPACE_XY_M",
    "MAXIMUM_COMMAND_HOLD_STEPS",
    "MAXIMUM_HEIGHT_M",
    "MAXIMUM_HORIZONTAL_SPEED_M_S",
    "MAXIMUM_VERTICAL_SPEED_M_S",
    "MAXIMUM_YAW_RATE_RAD_S",
    "MINIMUM_COMMAND_HOLD_STEPS",
    "MINIMUM_HEIGHT_M",
    "OBSERVATION_CONTRACT_VERSION",
    "OBSERVATION_WIDTH",
    "SOFT_WORKSPACE_XY_M",
    "ScheduledCommand",
    "apply_command_safety_envelope",
    "bound_command_body",
    "classify_command_failures",
    "command_conditioned_observation",
    "command_follow_contract_payload",
    "command_follow_reward_terms",
    "command_training_contract_payload",
    "command_training_contract_sha256",
    "command_tracking_contract_payload",
    "curriculum_stage",
    "curriculum_stage_index",
    "integrate_command_target",
    "sample_scheduled_command",
]
