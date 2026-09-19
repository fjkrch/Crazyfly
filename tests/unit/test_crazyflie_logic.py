from dataclasses import replace

import pytest
import torch

from g1_fly_control.tasks.crazyflie.logic import (
    DEFAULT_REWARD_CONTRACT_SHA256,
    DEFAULT_TRAINING_CURRICULUM,
    DEFAULT_TRAINING_CURRICULUM_SHA256,
    EPISODE_STEPS,
    FAILURE_HIGH_HEIGHT,
    FAILURE_LOW_HEIGHT,
    FAILURE_NONE,
    FAILURE_NONFINITE,
    FAILURE_WORKSPACE_ESCAPE,
    SUCCESS_DWELL_STEPS,
    RecoveryTrackingState,
    TargetTrackingState,
    advance_success_dwell,
    advance_vector_training_interactions,
    classify_failure_causes,
    interval_reward_terms,
    mixed_scenario_codes,
    mixed_scenario_contract_payload,
    reset_training_curriculum_stage_index,
    reward_contract_payload,
    survival_first_contract_payload,
    switch_target_curriculum_payload,
    success_mask,
    training_curriculum_stage_index,
    validate_monotonic_training_interactions,
    validate_training_curriculum,
)


def test_mixed_scenario_assignment_is_seeded_concurrent_and_exactly_balanced():
    env_ids = torch.arange(4)
    rounds = [
        mixed_scenario_codes(env_ids, torch.full((4,), episode), seed=7)
        for episode in range(3)
    ]
    assert rounds[0].tolist() == [1, 2, 0, 1]
    assert len(set(rounds[0].tolist())) == 3
    per_environment = torch.stack(rounds, dim=1)
    assert all(sorted(row.tolist()) == [0, 1, 2] for row in per_environment)
    assert torch.bincount(per_environment.flatten(), minlength=3).tolist() == [4, 4, 4]


def test_mixed_scenario_contract_is_training_only_and_records_seed():
    payload = mixed_scenario_contract_payload(seed=11)
    assert payload["training_only"] is True
    assert payload["seed"] == 11
    assert payload["scenario_names_by_code"] == [
        "waypoint_reach",
        "waypoint_switch",
        "gust_recovery",
    ]
    with pytest.raises(ValueError, match="non-negative"):
        mixed_scenario_contract_payload(seed=-1)


def test_switch_followup_targets_use_full_stage_without_changing_v2_hashes():
    payload = switch_target_curriculum_payload()
    assert payload["target_0_distribution"] == "active_episode_reset_curriculum_stage"
    assert payload["targets_1_through_3_distribution"] == "final_full_curriculum_stage"
    assert payload["applies_to_scenario"] == "waypoint_switch"
    assert payload["non_switch_mixed_rows_repeat_target_0"] is True
    assert DEFAULT_REWARD_CONTRACT_SHA256 == (
        "8a17e7d55b87550c68690a080749e3accbb49321354ecc0b5cd28ef0128f9ba7"
    )
    assert DEFAULT_TRAINING_CURRICULUM_SHA256 == (
        "8ea8c1f8c52205ac028b0a493855764727b59243d45a2a4901d0596f2f91920d"
    )


def test_success_tube_is_inclusive_and_requires_finite_distance_and_speed():
    distance = torch.tensor([0.20, 0.20001, 0.10, float("nan"), -0.1])
    speed = torch.tensor([0.25, 0.10, 0.25001, 0.10, 0.10])
    assert success_mask(distance, speed).tolist() == [True, False, False, False, False]
    assert success_mask(0.20, 0.25) is True


def test_success_requires_25_continuous_steps_and_is_one_shot():
    dwell = torch.zeros(2, dtype=torch.long)
    latched = torch.zeros(2, dtype=torch.bool)
    distance = torch.tensor([0.10, 0.10])
    speed = torch.tensor([0.10, 0.10])

    for _ in range(SUCCESS_DWELL_STEPS - 1):
        update = advance_success_dwell(dwell, latched, distance, speed)
        dwell, latched = update.dwell_steps, update.success_latched
        assert not update.newly_successful.any()

    # Break continuity in only the second environment.
    update = advance_success_dwell(dwell, latched, torch.tensor([0.10, 0.21]), speed)
    dwell, latched = update.dwell_steps, update.success_latched
    assert update.newly_successful.tolist() == [True, False]
    assert dwell.tolist() == [SUCCESS_DWELL_STEPS, 0]

    # The first success remains latched and cannot emit another bonus.
    update = advance_success_dwell(dwell, latched, distance, speed)
    assert update.newly_successful.tolist() == [False, False]


def test_target_switch_resets_dwell_history_latch_and_target_timer():
    state = TargetTrackingState.create(2)
    all_ids = torch.tensor([0, 1])
    state.reset(all_ids, torch.tensor([1.0, 2.0]))
    for _ in range(SUCCESS_DWELL_STEPS):
        state.update(torch.tensor([0.10, 1.5]), torch.tensor([0.10, 0.10]))
    assert state.dwell_steps.tolist() == [SUCCESS_DWELL_STEPS, 0]
    assert state.success_latched.tolist() == [True, False]
    assert state.target_age_steps.tolist() == [SUCCESS_DWELL_STEPS, SUCCESS_DWELL_STEPS]

    # Environment zero changes target.  Resetting previous distance to the new
    # target distance prevents the switch itself from creating progress.
    state.reset(torch.tensor([0]), torch.tensor([3.0]))
    assert state.dwell_steps.tolist() == [0, 0]
    assert state.success_latched.tolist() == [False, False]
    assert state.target_age_steps.tolist() == [0, SUCCESS_DWELL_STEPS]
    assert state.first_success_after_steps.tolist() == [-1, -1]
    update = state.update(torch.tensor([3.0, 1.5]), torch.tensor([0.10, 0.10]))
    assert update.progress_m[0].item() == pytest.approx(0.0)
    assert update.target_age_steps.tolist() == [1, SUCCESS_DWELL_STEPS + 1]


def test_target_success_time_is_25_control_intervals_after_reset():
    state = TargetTrackingState.create(1)
    state.reset(torch.tensor([0]), torch.tensor([0.1]))
    latest = None
    for _ in range(SUCCESS_DWELL_STEPS):
        latest = state.update(torch.tensor([0.1]), torch.tensor([0.1]))
    assert latest is not None and latest.newly_successful.item()
    assert state.first_success_after_steps.item() == SUCCESS_DWELL_STEPS


def test_recovery_tracker_counts_dwell_only_after_gust_and_accepts_deadline_boundary():
    tracker = RecoveryTrackingState.create(1)
    tracker.begin_after_gust(torch.tensor([0]), gust_start_step=150, stable_before_gust=True)
    assert tracker.start_step.item() == 155
    assert tracker.stop_step.item() == 255

    # A state inside the tube during the gust cannot count toward recovery.
    tracker.update(torch.tensor([154]), torch.tensor([0.1]), torch.tensor([0.1]))
    assert tracker.dwell_steps.item() == 0

    latest = None
    for step in range(155, 155 + SUCCESS_DWELL_STEPS):
        latest = tracker.update(torch.tensor([step]), torch.tensor([0.1]), torch.tensor([0.1]))
    assert latest is not None and latest.newly_recovered.item()
    assert latest.recovery_latency_s.item() == pytest.approx(0.50)
    assert not latest.active.item()

    # A fresh attempt can complete on step 254, the last interval in the
    # 100-step window, and reports exactly 2.0 seconds.
    tracker.begin_after_gust(torch.tensor([0]), gust_start_step=150, stable_before_gust=False)
    for step in range(155, 230):
        tracker.update(torch.tensor([step]), torch.tensor([0.3]), torch.tensor([0.1]))
    for step in range(230, 255):
        latest = tracker.update(torch.tensor([step]), torch.tensor([0.1]), torch.tensor([0.1]))
    assert latest is not None and latest.newly_recovered.item()
    assert latest.recovery_latency_s.item() == pytest.approx(2.0)


def test_termination_before_recovery_is_a_failed_attempt():
    tracker = RecoveryTrackingState.create(1)
    tracker.begin_after_gust(torch.tensor([0]), gust_start_step=150, stable_before_gust=True)
    result = tracker.update(
        torch.tensor([160]),
        torch.tensor([0.1]),
        torch.tensor([0.1]),
        terminated=torch.tensor([True]),
    )
    assert result.newly_failed.item()
    assert not result.newly_recovered.item()
    assert not result.active.item()


def test_protocol_horizon_is_600_control_decisions():
    assert EPISODE_STEPS == 600


def test_failure_classifier_covers_live_bounds_and_safe_nonfinite_branch():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.75],
            [0.0, 0.0, 0.09],
            [0.0, 0.0, 2.01],
            [2.76, 0.0, 0.75],
            [0.0, 0.0, 0.75],
        ]
    )
    result = classify_failure_causes(
        positions,
        torch.zeros_like(positions),
        torch.tensor([True, True, True, True, False]),
        minimum_height_m=0.10,
        maximum_height_m=2.00,
        workspace_xy_limit_m=2.75,
    )
    assert result.terminated.tolist() == [False, True, True, True, True]
    assert result.cause.tolist() == [
        FAILURE_NONE,
        FAILURE_LOW_HEIGHT,
        FAILURE_HIGH_HEIGHT,
        FAILURE_WORKSPACE_ESCAPE,
        FAILURE_NONFINITE,
    ]
    assert result.nonfinite.tolist() == [False, False, False, False, True]


def test_failure_classifier_treats_nonfinite_position_as_nonfinite_even_if_mask_is_true():
    positions = torch.tensor([[float("nan"), 0.0, 0.0]])
    result = classify_failure_causes(
        positions,
        torch.zeros_like(positions),
        torch.tensor([True]),
        minimum_height_m=0.10,
        maximum_height_m=2.00,
        workspace_xy_limit_m=2.75,
    )
    assert result.terminated.item()
    assert result.cause.item() == FAILURE_NONFINITE


def test_reward_terms_are_interval_quantities_without_an_extra_dt_factor():
    hover_action = 2.0 / 1.9 - 1.0
    terms = interval_reward_terms(
        torch.tensor([0.10]),
        torch.tensor([True]),
        torch.tensor([[hover_action + 0.10, 0.0025, 0.0, 0.0]]),
        torch.tensor([[0.20, 0.005, 0.0, 0.0]]),
        torch.tensor([False]),
        **reward_contract_payload_without_version(),
    )
    assert terms["progress"].item() == pytest.approx(0.025)
    assert terms["success_bonus"].item() == pytest.approx(5.0)
    assert terms["survival"].item() == pytest.approx(0.02)
    assert terms["control_effort"].item() == pytest.approx(-0.0026)
    assert terms["action_change"].item() == pytest.approx(-0.00104)
    assert terms["total"].item() == pytest.approx(5.04136)


def test_reward_clips_progress_and_physical_sensitivity_aware_command_metrics():
    hover_action = 2.0 / 1.9 - 1.0
    terms = interval_reward_terms(
        torch.tensor([100.0, -100.0]),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor(
            [[hover_action, 0.0, 0.0, 0.0], [hover_action, 1.0, 1.0, 1.0]]
        ),
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]),
        torch.zeros(2, dtype=torch.bool),
        **reward_contract_payload_without_version(),
    )
    assert terms["progress"].tolist() == pytest.approx([0.025, -0.025])
    assert terms["control_effort"].tolist() == pytest.approx([0.0, -0.04])
    assert terms["action_change"].tolist() == pytest.approx([0.0, -0.004])


def test_failure_interval_receives_no_survival_credit():
    hover_action = 2.0 / 1.9 - 1.0
    terms = interval_reward_terms(
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([[hover_action, 0.0, 0.0, 0.0]]).expand(2, -1),
        torch.zeros(2, 4),
        torch.tensor([False, True]),
        **reward_contract_payload_without_version(),
    )
    assert terms["survival"].tolist() == pytest.approx([0.02, 0.0])
    assert terms["failure"].tolist() == pytest.approx([0.0, -20.0])
    assert terms["total"].tolist() == pytest.approx([0.02, -20.0])


def reward_contract_payload_without_version():
    payload = reward_contract_payload()
    payload.pop("version")
    return payload


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("progress_clip_m", 0.0),
        ("moment_action_reference", 0.0),
        ("control_effort_normalized_squared_cap", 0.0),
        ("action_change_normalized_squared_cap", 0.0),
        ("collective_hover_action", 1.0),
        ("failure_penalty", -1.0),
        ("progress_scale", float("nan")),
    ),
)
def test_reward_contract_rejects_invalid_v2_clips_references_caps_and_scales(
    field, value
):
    with pytest.raises(ValueError):
        reward_contract_payload(**{field: value})


def test_survival_first_reward_has_the_declared_crash_timeout_success_ordering():
    episode_steps = 600
    crash_upper_bound = (
        episode_steps * 0.5 * 0.05
        + 5.0
        + (episode_steps - 1) * 0.02
        - 20.0
    )
    safe_timeout = episode_steps * 0.02
    same_flight_with_success = safe_timeout + 5.0
    assert crash_upper_bound == pytest.approx(11.98)
    assert same_flight_with_success > safe_timeout > crash_upper_bound


@pytest.mark.parametrize(
    ("interactions", "expected_index"),
    (
        (0, 0),
        (199_999, 0),
        (200_000, 1),
        (499_999, 1),
        (500_000, 2),
        (999_999, 2),
        (1_000_000, 3),
    ),
)
def test_training_curriculum_boundaries_are_inclusive_and_exact(interactions, expected_index):
    assert training_curriculum_stage_index(interactions) == expected_index


def test_survival_v2_curriculum_stage_payloads_are_exact():
    assert [stage.payload() for stage in DEFAULT_TRAINING_CURRICULUM[:3]] == [
        {
            "name": "vertical_lift",
            "start_interactions": 0,
            "spawn_height_m": 1.0,
            "spawn_position_xy_half_range_m": 0.01,
            "spawn_position_z_half_range_m": 0.005,
            "spawn_yaw_half_range_rad": 0.02,
            "spawn_linear_velocity_half_range_mps": 0.0,
            "spawn_angular_velocity_half_range_radps": 0.0,
            "goal_xy_min_m": -0.02,
            "goal_xy_max_m": 0.02,
            "goal_z_min_m": 1.45,
            "goal_z_max_m": 1.50,
            "minimum_goal_separation_m": 0.40,
        },
        {
            "name": "near",
            "start_interactions": 200_000,
            "spawn_height_m": 0.85,
            "spawn_position_xy_half_range_m": 0.025,
            "spawn_position_z_half_range_m": 0.01,
            "spawn_yaw_half_range_rad": 0.05,
            "spawn_linear_velocity_half_range_mps": 0.02,
            "spawn_angular_velocity_half_range_radps": 0.03,
            "goal_xy_min_m": -0.50,
            "goal_xy_max_m": 0.50,
            "goal_z_min_m": 1.00,
            "goal_z_max_m": 1.50,
            "minimum_goal_separation_m": 0.50,
        },
        {
            "name": "mid",
            "start_interactions": 500_000,
            "spawn_height_m": 0.70,
            "spawn_position_xy_half_range_m": 0.05,
            "spawn_position_z_half_range_m": 0.02,
            "spawn_yaw_half_range_rad": 0.10,
            "spawn_linear_velocity_half_range_mps": 0.04,
            "spawn_angular_velocity_half_range_radps": 0.07,
            "goal_xy_min_m": -1.00,
            "goal_xy_max_m": 1.00,
            "goal_z_min_m": 0.75,
            "goal_z_max_m": 1.50,
            "minimum_goal_separation_m": 0.65,
        },
    ]


def test_training_curriculum_rejects_negative_nonintegral_and_decreasing_clocks():
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            training_curriculum_stage_index(invalid)
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_monotonic_training_interactions(50_000, 49_999)
    with pytest.raises(ValueError):
        validate_monotonic_training_interactions(0, True)
    invalid_stage = replace(DEFAULT_TRAINING_CURRICULUM[1], start_interactions=-1)
    with pytest.raises(ValueError):
        validate_training_curriculum((DEFAULT_TRAINING_CURRICULUM[0], invalid_stage))


def test_vector_step_advances_before_exact_boundary_reset_without_double_count():
    before = 199_996
    after_step = advance_vector_training_interactions(before, 4)
    assert after_step == 200_000
    assert training_curriculum_stage_index(before) == 0
    assert training_curriculum_stage_index(after_step) == 1
    # The trainer's next pre-collect synchronization is equality, not another
    # increment; only the following real vector step adds four interactions.
    assert validate_monotonic_training_interactions(after_step, 200_000) == 200_000
    assert advance_vector_training_interactions(after_step, 4) == 200_004


def test_full_curriculum_stage_is_exactly_the_legacy_full_distribution():
    assert DEFAULT_TRAINING_CURRICULUM[-1].reset_distribution_payload() == {
        "spawn_height_m": 0.50,
        "spawn_position_xy_half_range_m": 0.10,
        "spawn_position_z_half_range_m": 0.05,
        "spawn_yaw_half_range_rad": 0.25,
        "spawn_linear_velocity_half_range_mps": 0.10,
        "spawn_angular_velocity_half_range_radps": 0.20,
        "goal_xy_min_m": -2.0,
        "goal_xy_max_m": 2.0,
        "goal_z_min_m": 0.50,
        "goal_z_max_m": 1.50,
        "minimum_goal_separation_m": 0.75,
    }


def test_deterministic_or_planned_resets_bypass_training_curriculum():
    assert reset_training_curriculum_stage_index(0) == 0
    assert reset_training_curriculum_stage_index(0, deterministic_eval=True) == 3
    assert reset_training_curriculum_stage_index(0, episode_plan_installed=True) == 3


def test_survival_first_contract_hashes_and_payload_are_frozen():
    assert DEFAULT_REWARD_CONTRACT_SHA256 == (
        "8a17e7d55b87550c68690a080749e3accbb49321354ecc0b5cd28ef0128f9ba7"
    )
    assert DEFAULT_TRAINING_CURRICULUM_SHA256 == (
        "8ea8c1f8c52205ac028b0a493855764727b59243d45a2a4901d0596f2f91920d"
    )
    contract = survival_first_contract_payload()
    assert contract["reward_sha256"] == DEFAULT_REWARD_CONTRACT_SHA256
    assert contract["training_curriculum_sha256"] == DEFAULT_TRAINING_CURRICULUM_SHA256
    assert contract["reward"] == {
        "version": "crazyflie_survival_first_reward_v2",
        "progress_scale": 0.5,
        "progress_clip_m": 0.05,
        "success_bonus": 5.0,
        "survival_reward_per_interval": 0.02,
        "control_effort_scale": 0.01,
        "collective_hover_action": 2.0 / 1.9 - 1.0,
        "moment_action_reference": 0.005,
        "control_effort_normalized_squared_cap": 4.0,
        "action_change_scale": 0.001,
        "action_change_normalized_squared_cap": 4.0,
        "failure_penalty": 20.0,
    }
    assert [
        stage["start_interactions"]
        for stage in contract["training_curriculum"]["stages"]
    ] == [0, 200_000, 500_000, 1_000_000]
