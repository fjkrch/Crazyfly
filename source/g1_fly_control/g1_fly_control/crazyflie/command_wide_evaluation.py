"""Paired held-out protocol for the wide still-air and wind command tasks."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

import torch

from g1_fly_control.crazyflie.command_evaluation import (
    CommandRollout,
    score_command_rollout,
)
from g1_fly_control.tasks.crazyflie.command_wide_logic import (
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    command_wide_contract_payload,
    wind_evaluation_protocol_payload,
)


PROTOCOL_VERSION = "crazyflie_command_wide_evaluation_v2"
EVALUATION_SEED = 20260920
EPISODE_COUNT = 16
EPISODE_STEPS = 600
SEGMENT_STEPS = 50
SEGMENT_NAMES = (
    "initial_hover",
    "forward",
    "brake_after_forward",
    "lateral",
    "horizontal_diagonal",
    "vertical",
    "yaw",
    "full_simultaneous",
    "reverse_full_simultaneous",
    "brake_after_full",
    "reverse_horizontal",
    "final_hover",
)
TASK_IDS = (COMMAND_FOLLOW_WIDE_TASK_ID, COMMAND_FOLLOW_WIDE_WIND_TASK_ID)


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def protocol_payload(*, task: str) -> dict[str, Any]:
    """Return the exact paired command and physical-disturbance protocol."""

    if task not in TASK_IDS:
        raise ValueError(f"wide evaluation task must be one of {TASK_IDS}")
    compact = command_wide_contract_payload()
    wind_enabled = task == COMMAND_FOLLOW_WIDE_WIND_TASK_ID
    return {
        "version": PROTOCOL_VERSION,
        "task": task,
        "evaluation_seed": EVALUATION_SEED,
        "episodes": EPISODE_COUNT,
        "steps_per_episode": EPISODE_STEPS,
        "segment_steps": SEGMENT_STEPS,
        "segment_names": list(SEGMENT_NAMES),
        "command_order": [
            "forward_m_s",
            "left_m_s",
            "up_m_s",
            "yaw_left_rad_s",
        ],
        "maximum_horizontal_speed_m_s": compact[
            "maximum_horizontal_speed_m_s"
        ],
        "maximum_vertical_speed_m_s": compact[
            "maximum_vertical_speed_m_s"
        ],
        "maximum_yaw_rate_rad_s": compact["maximum_yaw_rate_rad_s"],
        "deterministic_actions": True,
        "paired_command_script_across_conditions": True,
        "wind_enabled": wind_enabled,
        "wind_protocol": (
            wind_evaluation_protocol_payload()
            if wind_enabled
            else {
                "enabled": False,
                "applied_force_world_n": [0.0, 0.0, 0.0],
                "applied_torque_world_nm": [0.0, 0.0, 0.0],
            }
        ),
        "failure_denominator": "all_16_planned_episodes_and_all_600_steps",
    }


def protocol_sha256(*, task: str) -> str:
    return _canonical_sha256(protocol_payload(task=task))


def _signed(bit: int) -> float:
    return 1.0 if bit else -1.0


def held_out_command_script(
    episode_index: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return one 600-step script spanning the full wide trained envelope."""

    if type(episode_index) is not int or not 0 <= episode_index < EPISODE_COUNT:
        raise ValueError(f"episode_index must be in [0, {EPISODE_COUNT - 1}]")
    compact = command_wide_contract_payload()
    signs = tuple(_signed((episode_index >> bit) & 1) for bit in range(4))
    band = (0.55, 0.70, 0.85, 1.0)[episode_index % 4]
    horizontal = band * float(compact["maximum_horizontal_speed_m_s"])
    vertical = band * float(compact["maximum_vertical_speed_m_s"])
    yaw = band * float(compact["maximum_yaw_rate_rad_s"])
    diagonal = horizontal / math.sqrt(2.0)
    forward_sign, left_sign, up_sign, yaw_sign = signs
    segments = (
        (0.0, 0.0, 0.0, 0.0),
        (forward_sign * horizontal, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, left_sign * horizontal, 0.0, 0.0),
        (forward_sign * diagonal, left_sign * diagonal, 0.0, 0.0),
        (0.0, 0.0, up_sign * vertical, 0.0),
        (0.0, 0.0, 0.0, yaw_sign * yaw),
        (
            forward_sign * diagonal,
            left_sign * diagonal,
            up_sign * vertical,
            yaw_sign * yaw,
        ),
        (
            -forward_sign * diagonal,
            -left_sign * diagonal,
            -up_sign * vertical,
            -yaw_sign * yaw,
        ),
        (0.0, 0.0, 0.0, 0.0),
        (-forward_sign * diagonal, left_sign * diagonal, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    commands = torch.tensor(segments, device=device, dtype=dtype).repeat_interleave(
        SEGMENT_STEPS,
        dim=0,
    )
    if commands.shape != (EPISODE_STEPS, 4):
        raise RuntimeError("wide held-out command protocol has the wrong shape")
    names = tuple(name for name in SEGMENT_NAMES for _ in range(SEGMENT_STEPS))
    return commands, names


def batched_held_out_commands(
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    scripts = [
        held_out_command_script(index, device=device, dtype=dtype)[0]
        for index in range(EPISODE_COUNT)
    ]
    return torch.stack(scripts, dim=1), tuple(
        name for name in SEGMENT_NAMES for _ in range(SEGMENT_STEPS)
    )


def score_command_rollout_v2(
    rollout: CommandRollout,
    *,
    task: str,
) -> dict[str, Any]:
    """Use the established transparent score while recording the v2 protocol."""

    result = score_command_rollout(rollout)
    result["protocol"] = protocol_payload(task=task)
    result["protocol_sha256"] = protocol_sha256(task=task)
    return result


__all__ = [
    "CommandRollout",
    "EPISODE_COUNT",
    "EPISODE_STEPS",
    "EVALUATION_SEED",
    "PROTOCOL_VERSION",
    "SEGMENT_NAMES",
    "SEGMENT_STEPS",
    "TASK_IDS",
    "batched_held_out_commands",
    "held_out_command_script",
    "protocol_payload",
    "protocol_sha256",
    "score_command_rollout_v2",
]
