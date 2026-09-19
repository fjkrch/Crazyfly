"""Deterministic held-out command protocol and transparent control score."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

import torch

from g1_fly_control.tasks.crazyflie.command_logic import (
    CONTROL_DT_S,
    CRAZYFLIE_HOVER_ACTION,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
)


PROTOCOL_VERSION = "crazyflie_command_evaluation_v1"
EVALUATION_SEED = 20260918
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


def protocol_payload() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "evaluation_seed": EVALUATION_SEED,
        "episodes": EPISODE_COUNT,
        "steps_per_episode": EPISODE_STEPS,
        "segment_steps": SEGMENT_STEPS,
        "segment_names": list(SEGMENT_NAMES),
        "command_order": ["forward_m_s", "left_m_s", "up_m_s", "yaw_left_rad_s"],
        "maximum_horizontal_speed_m_s": MAXIMUM_HORIZONTAL_SPEED_M_S,
        "maximum_vertical_speed_m_s": MAXIMUM_VERTICAL_SPEED_M_S,
        "maximum_yaw_rate_rad_s": MAXIMUM_YAW_RATE_RAD_S,
        "deterministic_actions": True,
        "failure_denominator": "all_16_planned_episodes_and_all_600_steps",
    }


def protocol_sha256() -> str:
    encoded = json.dumps(
        protocol_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _signed(bit: int) -> float:
    return 1.0 if bit else -1.0


def held_out_command_script(
    episode_index: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return one fixed 600-step four-axis command script.

    Episode bits independently flip forward, lateral, vertical, and yaw signs.
    Four deterministic magnitude bands cover the trained envelope without
    consuming a random-number stream.
    """

    if type(episode_index) is not int or not 0 <= episode_index < EPISODE_COUNT:
        raise ValueError(f"episode_index must be in [0, {EPISODE_COUNT - 1}]")
    signs = tuple(_signed((episode_index >> bit) & 1) for bit in range(4))
    band = (0.55, 0.70, 0.85, 1.0)[episode_index % 4]
    horizontal = band * MAXIMUM_HORIZONTAL_SPEED_M_S
    vertical = band * MAXIMUM_VERTICAL_SPEED_M_S
    yaw = band * MAXIMUM_YAW_RATE_RAD_S
    diagonal = horizontal / math.sqrt(2.0)
    f, l, u, y = signs
    segments = (
        (0.0, 0.0, 0.0, 0.0),
        (f * horizontal, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, l * horizontal, 0.0, 0.0),
        (f * diagonal, l * diagonal, 0.0, 0.0),
        (0.0, 0.0, u * vertical, 0.0),
        (0.0, 0.0, 0.0, y * yaw),
        (f * diagonal, l * diagonal, u * vertical, y * yaw),
        (-f * diagonal, -l * diagonal, -u * vertical, -y * yaw),
        (0.0, 0.0, 0.0, 0.0),
        (-f * diagonal, l * diagonal, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    commands = torch.tensor(segments, device=device, dtype=dtype).repeat_interleave(
        SEGMENT_STEPS, dim=0
    )
    if commands.shape != (EPISODE_STEPS, 4):
        raise RuntimeError("held-out command protocol has the wrong shape")
    names = tuple(name for name in SEGMENT_NAMES for _ in range(SEGMENT_STEPS))
    return commands, names


def batched_held_out_commands(
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return commands as ``[steps, episodes, 4]`` for vector evaluation."""

    scripts = [
        held_out_command_script(index, device=device, dtype=dtype)[0]
        for index in range(EPISODE_COUNT)
    ]
    return torch.stack(scripts, dim=1), tuple(
        name for name in SEGMENT_NAMES for _ in range(SEGMENT_STEPS)
    )


@dataclass(frozen=True)
class CommandRollout:
    """Fixed-shape measurements for the held-out score."""

    effective_commands: torch.Tensor
    linear_velocity_body: torch.Tensor
    yaw_rate_body: torch.Tensor
    actions: torch.Tensor
    positions_world: torch.Tensor
    alive: torch.Tensor
    invalid: torch.Tensor


def _validate_rollout(rollout: CommandRollout) -> tuple[int, int]:
    commands = rollout.effective_commands
    if not isinstance(commands, torch.Tensor) or commands.ndim != 3 or commands.shape[2] != 4:
        raise ValueError("effective_commands must have shape [steps, episodes, 4]")
    steps, episodes = commands.shape[:2]
    expected = {
        "linear_velocity_body": (steps, episodes, 3),
        "yaw_rate_body": (steps, episodes),
        "actions": (steps, episodes, 4),
        "positions_world": (steps, episodes, 3),
        "alive": (steps, episodes),
        "invalid": (steps, episodes),
    }
    for name, shape in expected.items():
        value = getattr(rollout, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
    if rollout.alive.dtype != torch.bool or rollout.invalid.dtype != torch.bool:
        raise ValueError("alive and invalid masks must be Boolean")
    numeric = (
        commands,
        rollout.linear_velocity_body,
        rollout.yaw_rate_body,
        rollout.actions,
        rollout.positions_world,
    )
    if not all(bool(torch.isfinite(value).all()) for value in numeric):
        raise FloatingPointError("command rollout contains nonfinite numeric data")
    if float(rollout.actions.abs().max()) > 1.0 + 1.0e-6:
        raise ValueError("command rollout contains an out-of-range action")
    return steps, episodes


def _response_metrics(rollout: CommandRollout) -> tuple[float, float, float]:
    """Return response latency, overshoot, and brake-settling seconds."""

    commands = rollout.effective_commands
    actual = torch.cat(
        (rollout.linear_velocity_body, rollout.yaw_rate_body.unsqueeze(-1)), dim=-1
    )
    steps, episodes = commands.shape[:2]
    segment = SEGMENT_STEPS if steps == EPISODE_STEPS else max(1, steps // len(SEGMENT_NAMES))
    latencies: list[float] = []
    overshoots: list[float] = []
    braking: list[float] = []
    for environment in range(episodes):
        for start in range(0, steps, segment):
            stop = min(start + segment, steps)
            command = commands[start, environment]
            magnitude_sq = float(torch.dot(command, command))
            if magnitude_sq > 1.0e-8:
                projections = (actual[start:stop, environment] @ command) / magnitude_sq
                reached = torch.nonzero(projections >= 0.5, as_tuple=False)
                latency_steps = int(reached[0, 0]) if reached.numel() else stop - start
                latencies.append(latency_steps * CONTROL_DT_S)
                overshoots.append(max(0.0, float(projections.max()) - 1.0))
            elif start > 0 and float(commands[start - 1, environment].abs().max()) > 0.0:
                speed = torch.linalg.vector_norm(
                    rollout.linear_velocity_body[start:stop, environment], dim=-1
                )
                yaw_abs = rollout.yaw_rate_body[start:stop, environment].abs()
                settled = (speed <= 0.10) & (yaw_abs <= 0.20)
                found = stop - start
                for offset in range(0, max(0, settled.numel() - 9)):
                    if bool(settled[offset : offset + 10].all()):
                        found = offset
                        break
                braking.append(found * CONTROL_DT_S)
    return (
        sum(latencies) / len(latencies) if latencies else 0.0,
        sum(overshoots) / len(overshoots) if overshoots else 0.0,
        sum(braking) / len(braking) if braking else 0.0,
    )


def score_command_rollout(rollout: CommandRollout) -> dict[str, Any]:
    """Calculate inspectable raw metrics and a bounded 0–100 score."""

    steps, episodes = _validate_rollout(rollout)
    alive = rollout.alive
    command = rollout.effective_commands
    linear_error = rollout.linear_velocity_body - command[:, :, :3]
    yaw_error = rollout.yaw_rate_body - command[:, :, 3]
    active_values = max(1, int(alive.sum()))
    linear_rmse = float(
        (linear_error.square().sum(dim=-1)[alive].sum() / active_values).sqrt()
    )
    yaw_rmse = float((yaw_error.square()[alive].sum() / active_values).sqrt())
    command_norm_sq = command[:, :, :3].square().sum(dim=-1)
    commanded = alive & (command_norm_sq > 1.0e-8)
    if bool(commanded.any()):
        projection = (
            (rollout.linear_velocity_body * command[:, :, :3]).sum(dim=-1)
            / command_norm_sq.clamp_min(1.0e-8)
        )
        wrong_direction_fraction = float((projection[commanded] < 0.0).float().mean())
        mean_command_projection_ratio = float(projection[commanded].mean())
    else:
        wrong_direction_fraction = 0.0
        mean_command_projection_ratio = 1.0
    hover = alive & (command.abs().amax(dim=-1) <= 1.0e-8)
    hover_speed_rms = float(
        (rollout.linear_velocity_body.square().sum(dim=-1)[hover].mean()).sqrt()
    ) if bool(hover.any()) else 0.0
    hover_drifts: list[float] = []
    for start in range(0, steps, SEGMENT_STEPS):
        stop = min(start + SEGMENT_STEPS, steps)
        if stop - start < 2:
            continue
        zero_command = command[start:stop].abs().amax(dim=(0, 2)) <= 1.0e-8
        for environment in torch.nonzero(zero_command, as_tuple=False).flatten().tolist():
            hover_drifts.append(
                float(
                    torch.linalg.vector_norm(
                        rollout.positions_world[stop - 1, environment]
                        - rollout.positions_world[start, environment]
                    )
                )
            )
    hover_drift_mean = sum(hover_drifts) / len(hover_drifts) if hover_drifts else 0.0
    action_reference = rollout.actions.new_tensor(
        [CRAZYFLIE_HOVER_ACTION, 0.0, 0.0, 0.0]
    )
    action_effort_rms = float(
        ((rollout.actions - action_reference).square().sum(dim=-1)[alive].mean()).sqrt()
    )
    deltas = rollout.actions[1:] - rollout.actions[:-1]
    paired_alive = alive[1:] & alive[:-1]
    action_delta_rms = float(
        (deltas.square().sum(dim=-1)[paired_alive].mean()).sqrt()
    ) if bool(paired_alive.any()) else 0.0
    response_latency_s, overshoot_ratio, brake_settling_s = _response_metrics(rollout)
    survival_fraction = float(alive.float().mean())
    invalid_count = int(rollout.invalid.sum())

    components = {
        "linear_tracking": 100.0 * math.exp(-((linear_rmse / 0.35) ** 2)),
        "yaw_tracking": 100.0 * math.exp(-((yaw_rmse / 0.50) ** 2)),
        "direction": 100.0 * max(0.0, 1.0 - wrong_direction_fraction),
        "response": 100.0 * math.exp(-(response_latency_s / 0.80)),
        "braking": 100.0 * math.exp(-(brake_settling_s / 1.00)),
        "hover": 100.0 * math.exp(
            -((hover_speed_rms / 0.15) ** 2) - ((hover_drift_mean / 0.15) ** 2)
        ),
        "safety": 100.0 * survival_fraction * (1.0 if invalid_count == 0 else 0.0),
        "effort": 100.0 * math.exp(-((action_effort_rms / 0.25) ** 2)),
        "smoothness": 100.0 * math.exp(-((action_delta_rms / 0.08) ** 2)),
    }
    weights = {
        "linear_tracking": 0.30,
        "yaw_tracking": 0.10,
        "direction": 0.10,
        "response": 0.10,
        "braking": 0.10,
        "hover": 0.10,
        "safety": 0.15,
        "effort": 0.025,
        "smoothness": 0.025,
    }
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("command-score weights must sum to one")
    total = sum(weights[name] * components[name] for name in weights)
    return {
        "score": min(100.0, max(0.0, total)),
        "component_scores": components,
        "component_weights": weights,
        "raw": {
            "steps": steps,
            "episodes": episodes,
            "linear_tracking_rmse_m_s": linear_rmse,
            "yaw_tracking_rmse_rad_s": yaw_rmse,
            "wrong_direction_fraction": wrong_direction_fraction,
            "mean_command_projection_ratio": mean_command_projection_ratio,
            "response_latency_mean_s": response_latency_s,
            "overshoot_ratio_mean": overshoot_ratio,
            "brake_settling_mean_s": brake_settling_s,
            "hover_speed_rms_m_s": hover_speed_rms,
            "hover_drift_mean_m": hover_drift_mean,
            "survival_fraction": survival_fraction,
            "invalid_state_count": invalid_count,
            "action_effort_rms": action_effort_rms,
            "action_delta_rms": action_delta_rms,
        },
        "protocol": protocol_payload(),
        "protocol_sha256": protocol_sha256(),
    }


__all__ = [
    "CommandRollout",
    "EPISODE_COUNT",
    "EPISODE_STEPS",
    "EVALUATION_SEED",
    "PROTOCOL_VERSION",
    "SEGMENT_NAMES",
    "SEGMENT_STEPS",
    "batched_held_out_commands",
    "held_out_command_script",
    "protocol_payload",
    "protocol_sha256",
    "score_command_rollout",
]
