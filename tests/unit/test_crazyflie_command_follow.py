"""CPU-only acceptance tests for the single Crazyflie command task."""

from __future__ import annotations

import math

import gymnasium as gym
import pytest
import torch

from g1_fly_control.tasks.crazyflie.command_logic import (
    COMMAND_CURRICULUM,
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_TRACKING_CONTRACT_SHA256,
    FAILURE_HIGH_HEIGHT,
    FAILURE_LOW_HEIGHT,
    FAILURE_NONFINITE,
    FAILURE_WORKSPACE_ESCAPE,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
    MINIMUM_COMMAND_HOLD_STEPS,
    apply_command_safety_envelope,
    bound_command_body,
    classify_command_failures,
    command_conditioned_observation,
    command_follow_contract_payload,
    command_follow_reward_terms,
    command_tracking_contract_payload,
    command_training_contract_sha256,
    curriculum_stage_index,
    integrate_command_target,
    sample_scheduled_command,
)
from g1_fly_control.tasks.crazyflie.registration import (
    COMMAND_ENV_ENTRY_POINT,
    REGISTERED_TASK_IDS,
    TASK_TO_CFG,
    TASK_TO_ENV_ENTRY_POINT,
    register_tasks,
)


def test_command_contract_is_self_authenticated_and_keeps_native_widths() -> None:
    assert command_training_contract_sha256() == COMMAND_TRACKING_CONTRACT_SHA256
    runtime = command_follow_contract_payload()
    assert runtime == {
        "version": "crazyflie_command_follow_v1",
        "observation_contract_version": "command_error_12_v1",
        "maximum_horizontal_speed_m_s": 0.8,
        "maximum_vertical_speed_m_s": 0.4,
        "maximum_yaw_rate_rad_s": 1.2,
    }
    full = command_tracking_contract_payload()
    assert full["task_id"] == COMMAND_FOLLOW_TASK_ID
    assert len(full["observation_order"]) == 12
    assert len(full["command_order"]) == 4
    assert full["reward_contract"]["transition_weights"] == {
        "tracking_progress": 0.20,
        "wrong_direction_acceleration": -0.03,
    }
    assert full["reward_contract"]["dt_scaled_weights"]["survival"] == 0.05
    assert full["reward_contract"]["terminal_failure_penalty"] == -5.0
    assert full["reward_contract"]["raw_acceleration_magnitude_rewarded"] is False


@pytest.mark.parametrize(
    ("interactions", "expected"),
    ((0, 0), (99_999, 0), (100_000, 1), (249_999, 1), (250_000, 2), (500_000, 2)),
)
def test_curriculum_boundaries(interactions: int, expected: int) -> None:
    assert curriculum_stage_index(interactions) == expected


def test_curriculum_has_the_exact_requested_ranges() -> None:
    assert [stage.payload() for stage in COMMAND_CURRICULUM] == [
        {
            "start_interactions": 0,
            "maximum_horizontal_speed_m_s": 0.25,
            "maximum_vertical_speed_m_s": 0.15,
            "maximum_yaw_rate_rad_s": 0.4,
        },
        {
            "start_interactions": 100_000,
            "maximum_horizontal_speed_m_s": 0.5,
            "maximum_vertical_speed_m_s": 0.25,
            "maximum_yaw_rate_rad_s": 0.8,
        },
        {
            "start_interactions": 250_000,
            "maximum_horizontal_speed_m_s": 0.8,
            "maximum_vertical_speed_m_s": 0.4,
            "maximum_yaw_rate_rad_s": 1.2,
        },
    ]


def test_schedule_is_deterministic_bounded_and_cycles_all_command_classes() -> None:
    first = [
        sample_scheduled_command(
            seed=0,
            environment_id=3,
            segment_index=index,
            total_interactions=500_000,
        )
        for index in range(8)
    ]
    second = [
        sample_scheduled_command(
            seed=0,
            environment_id=3,
            segment_index=index,
            total_interactions=500_000,
        )
        for index in range(8)
    ]
    assert first == second
    assert {value.category for value in first[:4]} == {
        "hover",
        "cardinal",
        "diagonal",
        "full_simultaneous",
    }
    for value in first:
        assert MINIMUM_COMMAND_HOLD_STEPS <= value.hold_steps <= MAXIMUM_COMMAND_HOLD_STEPS
        horizontal = math.hypot(value.command[0], value.command[1])
        assert horizontal <= MAXIMUM_HORIZONTAL_SPEED_M_S + 1.0e-9
        assert abs(value.command[2]) <= MAXIMUM_VERTICAL_SPEED_M_S
        assert abs(value.command[3]) <= MAXIMUM_YAW_RATE_RAD_S


def test_full_simultaneous_schedule_has_horizontal_vertical_and_yaw() -> None:
    values = [
        sample_scheduled_command(
            seed=0,
            environment_id=0,
            segment_index=index,
            total_interactions=500_000,
        )
        for index in range(12)
    ]
    full = [value for value in values if value.category == "full_simultaneous"]
    assert full
    for value in full:
        assert value.command[0] != 0.0
        assert value.command[1] != 0.0
        assert value.command[2] != 0.0
        assert value.command[3] != 0.0


def test_command_bound_normalizes_horizontal_without_coupling_vertical_or_yaw() -> None:
    bounded = bound_command_body(torch.tensor([[2.0, 2.0, 0.3, -1.0]]))
    assert float(torch.linalg.vector_norm(bounded[0, :2])) == pytest.approx(0.8)
    assert float(bounded[0, 2]) == pytest.approx(0.3)
    assert float(bounded[0, 3]) == pytest.approx(-1.0)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        bound_command_body(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))


def test_safety_envelope_removes_only_outward_components() -> None:
    command = torch.tensor(
        [
            [0.5, 0.2, 0.3, 0.7],
            [-0.5, 0.2, -0.3, -0.7],
        ]
    )
    position = torch.tensor([[4.0, 0.0, 1.70], [-4.0, 0.0, 0.30]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    target = position.clone()
    origin = torch.zeros((2, 3))
    effective = apply_command_safety_envelope(
        command, position, quaternion, target, origin
    )
    assert float(effective[0, 0]) == pytest.approx(0.0)
    assert float(effective[0, 1]) == pytest.approx(0.2)
    assert float(effective[0, 2]) == pytest.approx(0.0)
    assert float(effective[0, 3]) == pytest.approx(0.7)
    assert float(effective[1, 0]) == pytest.approx(0.0)
    assert float(effective[1, 1]) == pytest.approx(0.2)
    assert float(effective[1, 2]) == pytest.approx(0.0)
    assert float(effective[1, 3]) == pytest.approx(-0.7)


def test_target_integration_is_heading_relative_and_bounded() -> None:
    # +90 degree yaw maps body forward to world +Y.
    half = math.sqrt(0.5)
    quaternion = torch.tensor([[half, 0.0, 0.0, half]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    command = torch.tensor([[0.8, 0.0, 0.4, 1.2]])
    result = integrate_command_target(
        target, command, quaternion, torch.zeros((1, 3)), dt_s=0.02
    )
    assert float(result[0, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(result[0, 1]) == pytest.approx(0.016)
    assert float(result[0, 2]) == pytest.approx(1.008)


def test_observation_is_exact_command_error_contract() -> None:
    linear = torch.tensor([[0.6, -0.1, 0.2]])
    angular = torch.tensor([[0.3, -0.4, 0.8]])
    gravity = torch.tensor([[0.1, 0.2, -0.9]])
    command = torch.tensor([[0.5, 0.2, -0.1, 0.6]])
    target_error = torch.tensor([[1.0, 2.0, 3.0]])
    observation = command_conditioned_observation(
        linear, angular, gravity, command, target_error
    )
    assert observation.shape == (1, 12)
    assert torch.allclose(
        observation,
        torch.tensor(
            [[0.1, -0.3, 0.3, 0.3, -0.4, 0.2, 0.1, 0.2, -0.9, 1.0, 2.0, 3.0]]
        ),
    )


def _reward_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((1, 4)),
        torch.zeros((1, 4)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.zeros((1, 3)),
        torch.tensor([[2.0 / 1.9 - 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[2.0 / 1.9 - 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([False]),
    )


def test_continuous_reward_prefers_exact_tracking_and_is_componentized() -> None:
    inputs = _reward_inputs()
    perfect = command_follow_reward_terms(*inputs)
    bad_inputs = list(inputs)
    bad_inputs[0] = torch.tensor([[0.8, 0.8, 0.4, 1.2]])
    bad = command_follow_reward_terms(*bad_inputs)
    assert set(perfect) == {
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
        "total",
    }
    assert float(perfect["total"]) > float(bad["total"])
    assert torch.isfinite(perfect["total"]).all()


def test_nonfinite_failure_reward_is_finite_and_penalized() -> None:
    inputs = list(_reward_inputs())
    inputs[0] = torch.full((1, 4), float("nan"))
    inputs[-1] = torch.tensor([True])
    reward = command_follow_reward_terms(*inputs)
    assert bool(torch.isfinite(reward["total"]).all())
    assert float(reward["failure"]) == pytest.approx(-5.0)


def test_response_reward_uses_error_reduction_and_penalizes_wrong_acceleration() -> None:
    inputs = list(_reward_inputs())
    inputs[0] = torch.tensor([[0.4, 0.0, 0.0, 0.0]])
    inputs[1] = torch.tensor([[0.8, 0.0, 0.0, 0.0]])
    inputs[2] = torch.tensor([[-2.0, 0.0, 0.0]])
    improving = command_follow_reward_terms(*inputs)
    assert float(improving["tracking_progress"]) > 0.0
    assert float(improving["wrong_direction_acceleration"]) == pytest.approx(0.0)

    inputs[0] = torch.tensor([[0.8, 0.0, 0.0, 0.0]])
    inputs[1] = torch.tensor([[0.4, 0.0, 0.0, 0.0]])
    inputs[2] = torch.tensor([[2.0, 0.0, 0.0]])
    worsening = command_follow_reward_terms(*inputs)
    assert float(worsening["tracking_progress"]) < 0.0
    assert float(worsening["wrong_direction_acceleration"]) < 0.0


def test_reward_explicitly_prefers_stability_and_survival() -> None:
    stable_inputs = list(_reward_inputs())
    stable = command_follow_reward_terms(*stable_inputs)

    unstable_inputs = list(_reward_inputs())
    unstable_inputs[4] = torch.tensor([[2.0, -2.0, 0.0]])
    unstable_inputs[5] = torch.tensor([[0.7, -0.7, -0.1]])
    unstable = command_follow_reward_terms(*unstable_inputs)
    assert float(stable["attitude_stability"]) > float(
        unstable["attitude_stability"]
    )
    assert float(stable["angular_stability"]) > float(
        unstable["angular_stability"]
    )
    assert float(stable["total"]) > float(unstable["total"])

    failed_inputs = list(_reward_inputs())
    failed_inputs[-1] = torch.tensor([True])
    failed = command_follow_reward_terms(*failed_inputs)
    assert float(stable["survival"]) > 0.0
    assert float(stable["failure"]) == pytest.approx(0.0)
    assert float(failed["failure"]) == pytest.approx(-5.0)
    assert float(stable["total"]) > float(failed["total"])


def test_stability_jerk_survival_and_hard_failure_reward_signs() -> None:
    baseline_inputs = list(_reward_inputs())
    baseline = command_follow_reward_terms(*baseline_inputs)
    assert float(baseline["attitude_stability"]) == pytest.approx(0.0)
    assert float(baseline["angular_stability"]) == pytest.approx(0.0)
    assert float(baseline["jerk"]) == pytest.approx(0.0)
    assert float(baseline["survival"]) > 0.0
    assert float(baseline["failure"]) == pytest.approx(0.0)

    disturbed_inputs = list(baseline_inputs)
    disturbed_inputs[3] = torch.tensor([[50.0, -50.0, 0.0]])
    disturbed_inputs[4] = torch.tensor([[2.0, -3.0, 0.0]])
    disturbed_inputs[5] = torch.tensor([[0.5, -0.5, -0.7]])
    disturbed_inputs[-1] = torch.tensor([True])
    disturbed = command_follow_reward_terms(*disturbed_inputs)
    assert float(disturbed["jerk"]) < 0.0
    assert float(disturbed["attitude_stability"]) < 0.0
    assert float(disturbed["angular_stability"]) < 0.0
    assert float(disturbed["survival"]) == pytest.approx(0.0)
    assert float(disturbed["failure"]) == pytest.approx(-5.0)
    assert all(bool(torch.isfinite(value).all()) for value in disturbed.values())


def test_hard_failure_classifier_has_explicit_causes_and_nonfinite_priority() -> None:
    position = torch.tensor(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 2.0],
            [4.6, 0.0, 1.0],
            [float("nan"), 0.0, 1.0],
        ]
    )
    result = classify_command_failures(
        position, torch.zeros_like(position), torch.tensor([True, True, True, False])
    )
    assert result.terminated.tolist() == [True, True, True, True]
    assert result.cause.tolist() == [
        FAILURE_LOW_HEIGHT,
        FAILURE_HIGH_HEIGHT,
        FAILURE_WORKSPACE_ESCAPE,
        FAILURE_NONFINITE,
    ]


def test_command_task_registration_is_additive_and_idempotent() -> None:
    assert COMMAND_FOLLOW_TASK_ID in REGISTERED_TASK_IDS
    assert TASK_TO_CFG[COMMAND_FOLLOW_TASK_ID].endswith(":CommandFollowEnvCfg")
    assert TASK_TO_ENV_ENTRY_POINT[COMMAND_FOLLOW_TASK_ID] == COMMAND_ENV_ENTRY_POINT
    assert register_tasks() == REGISTERED_TASK_IDS
    assert register_tasks() == REGISTERED_TASK_IDS
    spec = gym.spec(COMMAND_FOLLOW_TASK_ID)
    assert spec.entry_point == COMMAND_ENV_ENTRY_POINT
    assert spec.kwargs["env_cfg_entry_point"].endswith(":CommandFollowEnvCfg")
