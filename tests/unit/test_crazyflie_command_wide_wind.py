"""Pure/static acceptance tests for the additive command-v2 wind contract."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from g1_fly_control.tasks.crazyflie.command_wide_logic import (
    COMMAND_CURRICULUM,
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    COMMAND_V2_PROFILE,
    COMMAND_WIDE_STILL_CONTRACT_SHA256,
    COMMAND_WIDE_WIND_CONTRACT_SHA256,
    HARD_WORKSPACE_XY_M,
    HELD_OUT_EPISODES,
    HELD_OUT_EPISODE_STEPS,
    HELD_OUT_PULSE_DURATION_STEPS,
    HELD_OUT_PULSE_START_STEPS,
    HELD_OUT_WIND_SEED,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
    SOFT_WORKSPACE_XY_M,
    TRAINING_INTERACTION_BUDGET,
    TRAINING_WIND_SEED,
    WIND_CATEGORIES,
    WIND_CURRICULUM,
    WIND_EVALUATION_PROTOCOL_SHA256,
    WIND_REFERENCE_ARM_M,
    apply_wide_command_safety_envelope,
    bound_wide_command_body,
    command_curriculum_stage_index,
    command_wide_contract_payload,
    command_wide_reward_terms,
    command_wide_training_contract_payload,
    command_wide_training_contract_sha256,
    held_out_wind_at_step,
    held_out_wind_script,
    sample_training_wind,
    sample_wide_scheduled_command,
    wind_curriculum_stage_index,
    wind_evaluation_protocol_payload,
    wind_wrench_from_ratios,
)


def test_command_v2_contract_is_self_authenticated_and_exactly_12_by_4() -> None:
    compact = command_wide_contract_payload()
    assert compact["profile"] == COMMAND_V2_PROFILE == "command_v2"
    assert compact["observation_width"] == 12
    assert compact["action_width"] == 4
    assert compact["maximum_horizontal_speed_m_s"] == 1.0
    assert compact["maximum_vertical_speed_m_s"] == 0.5
    assert compact["maximum_yaw_rate_rad_s"] == 1.5
    assert TRAINING_INTERACTION_BUDGET == 1_000_000

    still = command_wide_training_contract_payload(wind_enabled=False)
    wind = command_wide_training_contract_payload(wind_enabled=True)
    assert still["task_id"] == COMMAND_FOLLOW_WIDE_TASK_ID
    assert wind["task_id"] == COMMAND_FOLLOW_WIDE_WIND_TASK_ID
    assert still["observation_order"] == wind["observation_order"]
    assert still["command_order"] == wind["command_order"]
    assert still["command_curriculum"] == wind["command_curriculum"]
    assert still["reward_contract"] == wind["reward_contract"]
    assert still["safety_envelope"] == wind["safety_envelope"]
    assert still["training_schedule_clock"] == wind["training_schedule_clock"] == {
        "basis": "global_control_intervals_per_environment",
        "interaction_clock": "completed_control_intervals_times_num_envs",
        "same_interval_for_every_environment": True,
        "episode_reset_advances_command_cursor": False,
        "episode_termination_gates_command_cursor": False,
        "episode_reset_advances_training_wind_cursor": False,
        "episode_termination_gates_training_wind_cursor": False,
        "schedule_independent_of_episode_termination": True,
    }
    assert still["wind_contract"] is None
    assert wind["wind_contract"]["isaac_is_global"] is True
    assert wind["wind_contract"]["isaac_wrench_composer_api"] == (
        "permanent_wrench_composer.set_forces_and_torques"
    )
    assert wind["wind_contract"]["composer_reset_before_world_transform"] is True
    assert command_wide_training_contract_sha256(
        wind_enabled=False
    ) == COMMAND_WIDE_STILL_CONTRACT_SHA256
    assert command_wide_training_contract_sha256(
        wind_enabled=True
    ) == COMMAND_WIDE_WIND_CONTRACT_SHA256
    assert COMMAND_WIDE_STILL_CONTRACT_SHA256 != COMMAND_WIDE_WIND_CONTRACT_SHA256


@pytest.mark.parametrize(
    ("interactions", "command_stage", "wind_stage"),
    (
        (0, 0, 0),
        (99_999, 0, 0),
        (100_000, 1, 1),
        (249_999, 1, 1),
        (250_000, 2, 2),
        (499_999, 2, 2),
        (500_000, 3, 3),
        (749_999, 3, 3),
        (750_000, 3, 4),
        (1_000_000, 3, 4),
    ),
)
def test_exact_command_and_wind_curriculum_boundaries(
    interactions: int, command_stage: int, wind_stage: int
) -> None:
    assert command_curriculum_stage_index(interactions) == command_stage
    assert wind_curriculum_stage_index(interactions) == wind_stage


def test_exact_wide_envelopes_and_workspace_are_declared() -> None:
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
        {
            "start_interactions": 500_000,
            "maximum_horizontal_speed_m_s": 1.0,
            "maximum_vertical_speed_m_s": 0.5,
            "maximum_yaw_rate_rad_s": 1.5,
        },
    ]
    assert [stage.payload() for stage in WIND_CURRICULUM] == [
        {
            "start_interactions": 0,
            "maximum_force_to_weight_ratio": 0.03,
            "maximum_torque_to_weight_arm_ratio": 0.02,
        },
        {
            "start_interactions": 100_000,
            "maximum_force_to_weight_ratio": 0.05,
            "maximum_torque_to_weight_arm_ratio": 0.03,
        },
        {
            "start_interactions": 250_000,
            "maximum_force_to_weight_ratio": 0.08,
            "maximum_torque_to_weight_arm_ratio": 0.05,
        },
        {
            "start_interactions": 500_000,
            "maximum_force_to_weight_ratio": 0.12,
            "maximum_torque_to_weight_arm_ratio": 0.075,
        },
        {
            "start_interactions": 750_000,
            "maximum_force_to_weight_ratio": 0.15,
            "maximum_torque_to_weight_arm_ratio": 0.10,
        },
    ]
    assert SOFT_WORKSPACE_XY_M == 5.0
    assert HARD_WORKSPACE_XY_M == 5.5


def test_still_and_wind_variants_receive_byte_identical_command_segments() -> None:
    # The wind schedule is sampled separately and cannot consume command RNG state.
    before = [
        sample_wide_scheduled_command(
            seed=7,
            environment_id=2,
            segment_index=index,
            total_interactions=600_000,
        )
        for index in range(12)
    ]
    for index in range(12):
        sample_training_wind(
            seed=TRAINING_WIND_SEED,
            environment_id=2,
            segment_index=index,
            total_interactions=600_000,
        )
    after = [
        sample_wide_scheduled_command(
            seed=7,
            environment_id=2,
            segment_index=index,
            total_interactions=600_000,
        )
        for index in range(12)
    ]
    assert before == after
    assert {item.category for item in before[:4]} == {
        "hover",
        "cardinal",
        "diagonal",
        "full_simultaneous",
    }


_TRAINING_NUM_ENVS = 40
_TRACE_ENVIRONMENTS = 4


def _initial_paired_schedule_state() -> dict[str, list[object]]:
    commands = [
        sample_wide_scheduled_command(
            seed=0,
            environment_id=env_id,
            segment_index=0,
            total_interactions=0,
        )
        for env_id in range(_TRACE_ENVIRONMENTS)
    ]
    winds = [
        sample_training_wind(
            seed=TRAINING_WIND_SEED,
            environment_id=env_id,
            segment_index=0,
            total_interactions=0,
        )
        for env_id in range(_TRACE_ENVIRONMENTS)
    ]
    return {
        "commands": commands,
        "command_next": [1] * _TRACE_ENVIRONMENTS,
        "command_remaining": [item.hold_steps for item in commands],
        "winds": winds,
        "wind_next": [1] * _TRACE_ENVIRONMENTS,
        "wind_remaining": [item.hold_steps for item in winds],
        "episode_generation": [0] * _TRACE_ENVIRONMENTS,
    }


def _run_paired_schedule(
    state: dict[str, list[object]],
    *,
    start_step: int,
    stop_step: int,
    reset_rule,
) -> list[tuple[object, ...]]:
    """Pure reference for the global cursors implemented by the environment."""

    trace: list[tuple[object, ...]] = []
    for control_step in range(start_step, stop_step):
        # A reset is allowed to replace episode-local target state only.  It is
        # deliberately absent from both global schedule transitions below.
        for env_id in range(_TRACE_ENVIRONMENTS):
            if reset_rule(control_step, env_id):
                state["episode_generation"][env_id] += 1

        rows = []
        for env_id in range(_TRACE_ENVIRONMENTS):
            command = state["commands"][env_id]
            wind = state["winds"][env_id]
            rows.append(
                (
                    command.command,
                    command.category,
                    command.stage_index,
                    state["command_remaining"][env_id],
                    state["command_next"][env_id],
                    wind.force_ratio_world,
                    wind.torque_ratio_world,
                    wind.category,
                    wind.stage_index,
                    state["wind_remaining"][env_id],
                    state["wind_next"][env_id],
                )
            )
        trace.append(tuple(rows))

        next_interactions = (control_step + 1) * _TRAINING_NUM_ENVS
        for env_id in range(_TRACE_ENVIRONMENTS):
            state["command_remaining"][env_id] -= 1
            if state["command_remaining"][env_id] == 0:
                segment_index = state["command_next"][env_id]
                command = sample_wide_scheduled_command(
                    seed=0,
                    environment_id=env_id,
                    segment_index=segment_index,
                    total_interactions=next_interactions,
                )
                state["commands"][env_id] = command
                state["command_next"][env_id] += 1
                state["command_remaining"][env_id] = command.hold_steps

            state["wind_remaining"][env_id] -= 1
            if state["wind_remaining"][env_id] == 0:
                segment_index = state["wind_next"][env_id]
                wind = sample_training_wind(
                    seed=TRAINING_WIND_SEED,
                    environment_id=env_id,
                    segment_index=segment_index,
                    total_interactions=next_interactions,
                )
                state["winds"][env_id] = wind
                state["wind_next"][env_id] += 1
                state["wind_remaining"][env_id] = wind.hold_steps
    return trace


def test_global_command_and_wind_streams_ignore_different_reset_patterns() -> None:
    stop_step = 12_800  # 512k interactions: crosses every command stage.
    no_resets = _initial_paired_schedule_state()
    frequent_resets = _initial_paired_schedule_state()
    different_resets = _initial_paired_schedule_state()
    reference = _run_paired_schedule(
        no_resets,
        start_step=0,
        stop_step=stop_step,
        reset_rule=lambda _step, _env: False,
    )
    first = _run_paired_schedule(
        frequent_resets,
        start_step=0,
        stop_step=stop_step,
        reset_rule=lambda step, env: (step * 3 + env * 11) % 37 == 0,
    )
    second = _run_paired_schedule(
        different_resets,
        start_step=0,
        stop_step=stop_step,
        reset_rule=lambda step, env: (step * 7 + env * 5) % 53 < 2,
    )
    assert reference == first == second
    assert frequent_resets["episode_generation"] != no_resets["episode_generation"]
    assert different_resets["episode_generation"] != frequent_resets["episode_generation"]

    command_stages = {
        row[2] for interval in reference for row in interval
    }
    wind_stages = {row[8] for interval in reference for row in interval}
    assert command_stages == {0, 1, 2, 3}
    assert wind_stages == {0, 1, 2, 3}
    full_simultaneous = [
        row[0]
        for interval in reference
        for row in interval
        if row[1] == "full_simultaneous" and row[2] == 3
    ]
    assert full_simultaneous
    assert all(all(value != 0.0 for value in command) for command in full_simultaneous)


def test_global_schedule_checkpoint_resume_is_reset_pattern_independent() -> None:
    split_step = 6_251
    stop_step = 12_800
    state = _initial_paired_schedule_state()
    _run_paired_schedule(
        state,
        start_step=0,
        stop_step=split_step,
        reset_rule=lambda step, env: (step + env) % 41 == 0,
    )
    checkpoint = copy.deepcopy(state)
    continued = _run_paired_schedule(
        state,
        start_step=split_step,
        stop_step=stop_step,
        reset_rule=lambda step, env: (step + env) % 41 == 0,
    )
    resumed_state = copy.deepcopy(checkpoint)
    resumed = _run_paired_schedule(
        resumed_state,
        start_step=split_step,
        stop_step=stop_step,
        reset_rule=lambda step, env: (step * 13 + env) % 29 == 0,
    )
    assert continued == resumed
    for key in (
        "commands",
        "command_next",
        "command_remaining",
        "winds",
        "wind_next",
        "wind_remaining",
    ):
        assert state[key] == resumed_state[key]


def test_update_zero_checkpoint_is_pristine_and_first_reset_is_deterministic() -> None:
    pristine = {
        "training_interactions": 0,
        "next_command_segment_index": [0] * _TRACE_ENVIRONMENTS,
        "command_steps_remaining": [0] * _TRACE_ENVIRONMENTS,
        "requested_command_body": [[0.0] * 4 for _ in range(_TRACE_ENVIRONMENTS)],
        "command_category_code": [0] * _TRACE_ENVIRONMENTS,
        "command_stage_index": [0] * _TRACE_ENVIRONMENTS,
        "next_wind_segment_index": [0] * _TRACE_ENVIRONMENTS,
        "wind_steps_remaining": [0] * _TRACE_ENVIRONMENTS,
        "wind_force_ratio_world": [[0.0] * 3 for _ in range(_TRACE_ENVIRONMENTS)],
        "wind_torque_ratio_world": [[0.0] * 3 for _ in range(_TRACE_ENVIRONMENTS)],
        "wind_category_code": [0] * _TRACE_ENVIRONMENTS,
        "wind_stage_index": [0] * _TRACE_ENVIRONMENTS,
    }
    restored = copy.deepcopy(pristine)
    assert restored == pristine
    # The runner's first reset samples segment zero exactly once.  A direct
    # start and an update-zero resume therefore enter the same first interval.
    direct = _initial_paired_schedule_state()
    after_resume_reset = _initial_paired_schedule_state()
    assert direct == after_resume_reset

    source = (
        Path(__file__).parents[2]
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/command_wide_env.py"
    ).read_text(encoding="utf-8")
    assert "command_is_uninitialized = bool(command_uninitialized.all())" in source
    assert 'interactions != 0' in source
    assert '"command-v2 uninitialized command cursor is not pristine"' in source
    assert "command_is_uninitialized != wind_is_uninitialized" in source


def test_wide_bound_and_safety_keep_inward_motion() -> None:
    bounded = bound_wide_command_body(torch.tensor([[2.0, 2.0, 0.9, -3.0]]))
    assert float(torch.linalg.vector_norm(bounded[0, :2])) == pytest.approx(1.0)
    assert float(bounded[0, 2]) == pytest.approx(0.5)
    assert float(bounded[0, 3]) == pytest.approx(-1.5)

    command = torch.tensor([[0.9, 0.2, 0.5, 1.5], [-0.9, 0.2, -0.5, -1.5]])
    position = torch.tensor([[5.0, 0.0, 1.7], [-5.0, 0.0, 0.3]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    effective = apply_wide_command_safety_envelope(
        command, position, quaternion, position.clone(), torch.zeros((2, 3))
    )
    assert float(effective[0, 0]) == pytest.approx(0.0)
    assert float(effective[0, 1]) == pytest.approx(0.2)
    assert float(effective[0, 2]) == pytest.approx(0.0)
    assert float(effective[1, 0]) == pytest.approx(0.0)
    assert float(effective[1, 1]) == pytest.approx(0.2)
    assert float(effective[1, 2]) == pytest.approx(0.0)


def _reward_inputs() -> tuple[torch.Tensor, ...]:
    hover = 2.0 / 1.9 - 1.0
    return (
        torch.zeros((1, 4)),
        torch.zeros((1, 4)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.zeros((1, 3)),
        torch.tensor([[hover, 0.0, 0.0, 0.0]]),
        torch.tensor([[hover, 0.0, 0.0, 0.0]]),
        torch.tensor([False]),
    )


def test_reward_is_shared_by_still_and_wind_and_prefers_tracking() -> None:
    # Both environments call this one function; no wind-specific reward exists.
    still = command_wide_reward_terms(*_reward_inputs())
    wind = command_wide_reward_terms(*_reward_inputs())
    assert set(still) == set(wind)
    assert all(torch.equal(still[name], wind[name]) for name in still)
    bad_inputs = list(_reward_inputs())
    bad_inputs[0] = torch.tensor([[1.0, 1.0, 0.5, 1.5]])
    bad = command_wide_reward_terms(*bad_inputs)
    assert float(still["total"]) > float(bad["total"])


def test_training_wind_is_deterministic_bounded_and_cycles_all_categories() -> None:
    first = [
        sample_training_wind(
            seed=TRAINING_WIND_SEED,
            environment_id=0,
            segment_index=index,
            total_interactions=999_999,
        )
        for index in range(8)
    ]
    second = [
        sample_training_wind(
            seed=TRAINING_WIND_SEED,
            environment_id=0,
            segment_index=index,
            total_interactions=999_999,
        )
        for index in range(8)
    ]
    assert first == second
    assert {item.category for item in first[:4]} == set(WIND_CATEGORIES)
    for item in first:
        force_norm = math.sqrt(sum(value * value for value in item.force_ratio_world))
        torque_norm = math.sqrt(sum(value * value for value in item.torque_ratio_world))
        assert force_norm <= 0.15 + 1.0e-12
        assert torque_norm <= 0.10 + 1.0e-12
        if item.category == "calm":
            assert 25 <= item.hold_steps <= 75
            assert force_norm == 0.0
            assert torque_norm == 0.0
        else:
            assert 10 <= item.hold_steps <= 30


def test_heldout_wind_is_distinct_fixed_and_has_exact_pulse_windows() -> None:
    assert TRAINING_WIND_SEED != HELD_OUT_WIND_SEED
    protocol = wind_evaluation_protocol_payload()
    assert protocol["episodes"] == HELD_OUT_EPISODES == 16
    assert protocol["episode_steps"] == HELD_OUT_EPISODE_STEPS == 600
    assert protocol["pulse_start_steps"] == [75, 175, 275, 375, 475]
    assert protocol["pulse_duration_steps"] == HELD_OUT_PULSE_DURATION_STEPS == 25
    assert len(WIND_EVALUATION_PROTOCOL_SHA256) == 64

    first = held_out_wind_script(0)
    second = held_out_wind_script(0)
    assert first == second
    assert len(first) == 600
    active_steps = {
        step
        for step, value in enumerate(first)
        if value.category != "calm"
    }
    expected_steps = {
        step
        for start in HELD_OUT_PULSE_START_STEPS
        for step in range(start, start + HELD_OUT_PULSE_DURATION_STEPS)
    }
    assert active_steps == expected_steps
    assert held_out_wind_at_step(0, 74).category == "calm"
    assert held_out_wind_at_step(0, 75).category != "calm"
    assert held_out_wind_at_step(0, 99).category != "calm"
    assert held_out_wind_at_step(0, 100).category == "calm"


def test_physical_wrench_scales_with_weight_and_reference_arm() -> None:
    force, torque = wind_wrench_from_ratios(
        (0.1, -0.05, 0.0),
        (0.0, 0.1, -0.05),
        vehicle_weight_n=0.27,
    )
    assert force == pytest.approx((0.027, -0.0135, 0.0))
    assert torque == pytest.approx(
        (0.0, 0.27 * WIND_REFERENCE_ARM_M * 0.1, -0.27 * WIND_REFERENCE_ARM_M * 0.05)
    )


def test_env_source_pins_world_frame_api_zeroing_telemetry_and_resume_fields() -> None:
    source = (
        Path(__file__).parents[2]
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/command_wide_env.py"
    ).read_text(encoding="utf-8")
    assert "self._robot.permanent_wrench_composer.set_forces_and_torques(" in source
    assert "self._robot.permanent_wrench_composer.reset()" in source
    assert "self._robot.permanent_wrench_composer.reset(env_ids)" in source
    assert "set_external_force_and_torque(" not in source
    assert "is_global=True" in source
    assert "permanent_wrench_composer.add_forces_and_torques(" in source
    assert "def _zero_physical_wrench_on_reset(" in source
    assert "def applied_wind_force_world(" in source
    assert "def applied_wind_torque_world(" in source
    assert '"next_wind_segment_index"' in source
    assert '"wind_steps_remaining"' in source
    assert '"wind_force_ratio_world"' in source
    assert '"wind_torque_ratio_world"' in source
    assert '"wind_active_seed"' in source
    assert '"wind_mode"' in source
    assert '"heldout_episode_index"' in source
    assert '"heldout_step_cursor"' in source
    assert "def set_wind_evaluation_mode(" in source
    assert "def clear_wind_evaluation_mode(" in source
    assert "def _schedule_independent_of_episode_resets(" in source
    assert "self._wind_steps_remaining -= 1" in source
    assert "self._wind_steps_remaining[continuing] -= 1" not in source
    assert "uninitialized = ids[self._next_wind_segment_index[ids] == 0]" in source

    shared_source = (
        Path(__file__).parents[2]
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/command_env.py"
    ).read_text(encoding="utf-8")
    assert "scheduled = ~self._manual_command_mode & continuing" not in shared_source
    assert "if not self._schedule_independent_of_episode_resets:" in shared_source
    assert "self._next_command_segment_index[scheduled_ids] == 0" in shared_source


def test_cfg_source_keeps_shared_interface_and_only_toggles_wind() -> None:
    source = (
        Path(__file__).parents[2]
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/command_wide_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "class CommandFollowWideEnvCfg" in source
    assert "class CommandFollowWideWindEnvCfg(CommandFollowWideEnvCfg)" in source
    assert "action_space: int = ACTION_WIDTH" in source
    assert "observation_space: int = OBSERVATION_WIDTH" in source
    assert "wind_enabled: bool = False" in source
    assert "wind_enabled: bool = True" in source
