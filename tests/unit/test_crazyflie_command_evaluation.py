from __future__ import annotations

import pytest
import torch

from g1_fly_control.crazyflie.command_evaluation import (
    CommandRollout,
    EPISODE_COUNT,
    EPISODE_STEPS,
    SEGMENT_NAMES,
    SEGMENT_STEPS,
    batched_held_out_commands,
    held_out_command_script,
    protocol_payload,
    protocol_sha256,
    score_command_rollout,
)
from g1_fly_control.tasks.crazyflie.command_logic import CRAZYFLIE_HOVER_ACTION


def _rollout(*, actual_matches_command: bool = True, invalid: bool = False) -> CommandRollout:
    commands, _ = batched_held_out_commands()
    linear = commands[:, :, :3].clone() if actual_matches_command else torch.zeros_like(commands[:, :, :3])
    yaw = commands[:, :, 3].clone() if actual_matches_command else torch.zeros_like(commands[:, :, 3])
    actions = torch.zeros_like(commands)
    actions[:, :, 0] = CRAZYFLIE_HOVER_ACTION
    positions = torch.zeros((EPISODE_STEPS, EPISODE_COUNT, 3))
    alive = torch.ones((EPISODE_STEPS, EPISODE_COUNT), dtype=torch.bool)
    invalid_mask = torch.full_like(alive, invalid)
    return CommandRollout(commands, linear, yaw, actions, positions, alive, invalid_mask)


def test_protocol_is_exact_and_hashed() -> None:
    payload = protocol_payload()
    assert payload["episodes"] == 16
    assert payload["steps_per_episode"] == 600
    assert len(payload["segment_names"]) * SEGMENT_STEPS == EPISODE_STEPS
    assert len(protocol_sha256()) == 64


def test_held_out_scripts_cover_all_axes_and_simultaneous_commands() -> None:
    commands, names = held_out_command_script(15)
    assert commands.shape == (EPISODE_STEPS, 4)
    assert len(names) == EPISODE_STEPS
    assert tuple(dict.fromkeys(names)) == SEGMENT_NAMES
    assert torch.any(commands[:, 0] != 0)
    assert torch.any(commands[:, 1] != 0)
    assert torch.any(commands[:, 2] != 0)
    assert torch.any(commands[:, 3] != 0)
    simultaneous = (commands != 0).sum(dim=1)
    assert int(simultaneous.max()) == 4
    assert float(torch.linalg.vector_norm(commands[:, :2], dim=1).max()) <= 0.800001


def test_episode_bits_provide_both_command_signs() -> None:
    commands, _ = batched_held_out_commands()
    for axis in range(4):
        assert bool((commands[:, :, axis] > 0).any())
        assert bool((commands[:, :, axis] < 0).any())


def test_perfect_tracking_scores_one_hundred() -> None:
    result = score_command_rollout(_rollout())
    assert result["score"] == pytest.approx(100.0)
    assert result["raw"]["linear_tracking_rmse_m_s"] == 0.0
    assert result["raw"]["yaw_tracking_rmse_rad_s"] == 0.0
    assert sum(result["component_weights"].values()) == pytest.approx(1.0)


def test_nontracking_scores_below_perfect() -> None:
    perfect = score_command_rollout(_rollout())
    stationary = score_command_rollout(_rollout(actual_matches_command=False))
    assert 0.0 <= stationary["score"] < perfect["score"]
    assert stationary["raw"]["response_latency_mean_s"] > 0.0
    assert stationary["raw"]["linear_tracking_rmse_m_s"] > 0.0


def test_invalid_state_zeroes_safety_component() -> None:
    result = score_command_rollout(_rollout(invalid=True))
    assert result["component_scores"]["safety"] == 0.0
    assert result["raw"]["invalid_state_count"] == EPISODE_STEPS * EPISODE_COUNT


def test_score_rejects_nonfinite_and_out_of_range_actions() -> None:
    rollout = _rollout()
    rollout.actions[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        score_command_rollout(rollout)
    rollout = _rollout()
    rollout.actions[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="out-of-range"):
        score_command_rollout(rollout)
