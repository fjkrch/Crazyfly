from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "crazyflie_keyboard", ROOT / "scripts/crazyflie_keyboard.py"
)
assert SPEC is not None and SPEC.loader is not None
teleop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teleop
SPEC.loader.exec_module(teleop)


def test_held_keys_do_not_accumulate_and_release_independently() -> None:
    state = teleop.HeldKeyState()
    state.press("w")
    state.press("W")
    state.press("a")
    assert state.pressed == frozenset({"W", "A"})
    state.release("W")
    assert state.pressed == frozenset({"A"})
    state.press("SPACE")
    assert state.pressed == frozenset()


def test_reset_and_quit_clear_movement() -> None:
    state = teleop.HeldKeyState()
    state.press("W")
    state.press("R")
    assert state.pressed == frozenset()
    assert state.consume_reset() is True
    assert state.consume_reset() is False
    state.press("A")
    state.press("ESCAPE")
    assert state.quit_requested is True
    assert state.pressed == frozenset()


def test_multikey_diagonal_is_normalized_and_vertical_is_independent() -> None:
    command = teleop.command_from_pressed({"W", "A", "I"})
    assert command.forward == pytest.approx(1.0 / math.sqrt(2.0))
    assert command.left == pytest.approx(1.0 / math.sqrt(2.0))
    assert math.hypot(command.forward, command.left) == pytest.approx(1.0)
    assert command.up == 1.0


def test_e_is_an_up_alias_and_opposites_cancel() -> None:
    assert teleop.command_from_pressed({"E"}).up == 1.0
    command = teleop.command_from_pressed({"W", "S", "A", "D", "I", "Q", "J", "L"})
    assert command.as_tuple() == (0.0, 0.0, 0.0, 0.0)


def _level_observation() -> torch.Tensor:
    value = torch.zeros((1, 12), dtype=torch.float32)
    value[:, 8] = -1.0
    return value


def _assist() -> teleop.FlightAssist:
    return teleop.FlightAssist(
        teleop.FlightAssistConfig(), torch.tensor([[0.0, 0.0, 0.5]])
    )


def test_idle_level_action_is_hover() -> None:
    assist = _assist()
    action = assist.action(_level_observation(), torch.zeros((1, 3)), 0.0)
    assert action.shape == (1, 4)
    assert float(action[0, 0]) == pytest.approx(teleop.HOVER_ACTION)
    assert torch.equal(action[0, 1:], torch.zeros(3))


@pytest.mark.parametrize(
    ("desired", "action_index", "sign"),
    [
        ((0.5, 0.0, 0.0), 2, 1),
        ((-0.5, 0.0, 0.0), 2, -1),
        ((0.0, 0.5, 0.0), 1, -1),
        ((0.0, -0.5, 0.0), 1, 1),
        ((0.0, 0.0, 0.3), 0, 1),
        ((0.0, 0.0, -0.3), 0, -1),
    ],
)
def test_direction_to_native_action_signs(
    desired: tuple[float, float, float], action_index: int, sign: int
) -> None:
    assist = _assist()
    action = assist.action(
        _level_observation(), torch.tensor([desired]), 0.0
    )
    baseline = teleop.HOVER_ACTION if action_index == 0 else 0.0
    assert sign * (float(action[0, action_index]) - baseline) > 0.0


def test_extreme_finite_input_produces_finite_bounded_action() -> None:
    assist = _assist()
    observation = torch.full((1, 12), 1.0e6)
    desired = torch.tensor([[100.0, -100.0, 100.0]])
    action = assist.action(observation, desired, 100.0)
    assert bool(torch.isfinite(action).all())
    assert float(action.abs().max()) <= 1.0
    assert float(action[:, 1:].abs().max()) <= assist.config.maximum_moment_action + 1.0e-6


def test_runtime_finite_check_rejects_nonfinite_or_empty_state() -> None:
    finite = torch.zeros((1, 3))
    assert teleop._runtime_state_is_finite(finite, torch.ones((1, 4))) is True
    assert teleop._runtime_state_is_finite(finite, torch.tensor([[float("nan")]])) is False
    assert teleop._runtime_state_is_finite(torch.empty(0)) is False
    assert teleop._runtime_state_is_finite() is False


def test_smoke_gate_reports_nonfinite_state_resets() -> None:
    failures = teleop._smoke_failures(
        {
            "forward": {"mean_forward_velocity_m_s": 0.1},
            "forward_left": {"mean_command_projection_m_s": 0.1},
            "up": {"mean_vertical_velocity_m_s": 0.1},
            "down": {"mean_vertical_velocity_m_s": -0.1},
        },
        safety_termination_count=0,
        invalid_state_reset_count=1,
        maximum_action_abs=0.5,
    )
    assert failures == ["1 nonfinite-state safety reset(s)"]


def test_yaw_keys_map_to_opposite_native_yaw_moments() -> None:
    assist = _assist()
    observation = _level_observation()
    desired_velocity = torch.zeros((1, 3))
    left = assist.action(observation, desired_velocity, 0.8)
    right = assist.action(observation, desired_velocity, -0.8)
    assert float(left[0, 3]) > 0.0
    assert float(right[0, 3]) < 0.0


def test_body_relative_target_update_and_bounds() -> None:
    assist = _assist()
    position = torch.tensor([[0.0, 0.0, 0.5]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = teleop.Command(1.0, 1.0, 1.0, 0.0)
    desired = assist.desired_velocity_body(position, identity, command, 0.02)
    assert assist.target_w[0, 0] > 0.0
    assert assist.target_w[0, 1] > 0.0
    assert assist.target_w[0, 2] > 0.5
    assert desired[0, 0] > 0.0 and desired[0, 1] > 0.0 and desired[0, 2] > 0.0
    for _ in range(20_000):
        assist.desired_velocity_body(position, identity, command, 0.02)
    assert float(assist.target_w[0, 0]) <= assist.config.maximum_horizontal_offset_m
    assert float(assist.target_w[0, 1]) <= assist.config.maximum_horizontal_offset_m
    assert float(assist.target_w[0, 2]) <= assist.config.maximum_height_m + 1.0e-6


def test_held_vertical_key_cannot_command_through_height_bounds() -> None:
    assist = _assist()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    upper = torch.tensor([[0.0, 0.0, assist.config.maximum_height_m]])
    assist.target_w.copy_(upper)
    up = assist.desired_velocity_body(
        upper, identity, teleop.Command(0.0, 0.0, 1.0, 0.0), 0.02
    )
    assert float(up[0, 2]) <= 0.0

    lower = torch.tensor([[0.0, 0.0, assist.config.minimum_height_m]])
    assist.target_w.copy_(lower)
    down = assist.desired_velocity_body(
        lower, identity, teleop.Command(0.0, 0.0, -1.0, 0.0), 0.02
    )
    assert float(down[0, 2]) >= 0.0


def test_scripted_smoke_exercises_required_commands_and_reset() -> None:
    samples = [teleop.scripted_input(step) for step in range(800)]
    assert any(reset for _, reset, _ in samples)
    keys = {held for held, _, _ in samples}
    assert frozenset({"W"}) in keys
    assert frozenset({"W", "A"}) in keys
    assert frozenset({"I"}) in keys
    assert frozenset({"Q"}) in keys
    assert frozenset() in keys
