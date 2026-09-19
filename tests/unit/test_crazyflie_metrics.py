import pytest
import torch

from g1_fly_control.tasks.crazyflie.metrics import (
    CENSORING_RULE,
    EpisodeSummary,
    GustRecoveryOutcome,
    SwitchOutcome,
    aggregate_wrench_mechanical_work_proxy_j,
    command_effort_integral,
    command_smoothness_mean_delta,
    goal_error_integral_m_s,
    mean_speed_inside_target_region_m_s,
    summarize_episodes,
)


def _episode(
    episode_id: int,
    *,
    scenario: str = "FlyCrazyflie-WaypointReach-v0",
    success: bool = False,
    time_to_success: float | None = None,
    terminated: bool = False,
    truncated: bool = True,
    crash: bool = False,
    out_of_bounds: bool = False,
    invalid_state: bool = False,
    completed_steps: int = 600,
    switches: tuple[SwitchOutcome, ...] = (),
    gusts: tuple[GustRecoveryOutcome, ...] = (),
) -> EpisodeSummary:
    return EpisodeSummary(
        scenario=scenario,
        episode_id=episode_id,
        success=success,
        terminated=terminated,
        truncated=truncated,
        failure_reason="crash" if crash else None,
        time_to_first_success_s=time_to_success,
        final_goal_error_m=1.0 + episode_id,
        integrated_goal_error_m_s=5.0 + episode_id,
        mean_speed_inside_target_region_m_s=0.1 if success else None,
        crash=crash,
        out_of_bounds=out_of_bounds,
        invalid_state=invalid_state,
        command_effort=2.0,
        command_smoothness=0.5,
        aggregate_wrench_mechanical_work_proxy_j=0.25,
        completed_steps=completed_steps,
        switch_outcomes=switches,
        gust_outcomes=gusts,
    )


def test_trace_metrics_use_one_control_interval_factor():
    assert goal_error_integral_m_s([1.0, 2.0, 3.0]) == pytest.approx(0.12)
    assert mean_speed_inside_target_region_m_s([0.1, 0.3, 0.2], [0.2, 9.0, 0.4]) == pytest.approx(
        0.3
    )
    actions = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert command_effort_integral(actions) == pytest.approx(0.02)
    assert command_smoothness_mean_delta(actions) == pytest.approx(1.0)

    force = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    velocity = [[2.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    moment = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    angular = [[0.0, 0.0, -3.0], [0.0, 0.0, 4.0]]
    # Absolute powers are 2+3 and 2+8 W for two 0.02 s intervals.
    assert aggregate_wrench_mechanical_work_proxy_j(force, velocity, moment, angular) == pytest.approx(
        0.30
    )


def test_episode_summary_keeps_failures_in_fixed_denominator_and_censors_at_12_seconds():
    rows = [
        _episode(0, success=True, time_to_success=2.0),
        _episode(
            1,
            terminated=True,
            truncated=False,
            crash=True,
            completed_steps=100,
        ),
    ]
    summary = summarize_episodes(rows, expected_episode_count=2)
    assert summary["episode_count"] == 2
    assert summary["success_denominator"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["crash_rate"] == pytest.approx(0.5)
    latency = summary["time_to_first_success"]
    assert latency["censoring_rule"] == CENSORING_RULE
    assert latency["censored_count"] == 1
    assert latency["mean_fixed_horizon_censored_s"] == pytest.approx(7.0)
    assert latency["mean_successes_only_s"] == pytest.approx(2.0)
    assert len(summary["episodes"]) == 2
    assert summary["episodes"][1]["crash"] is True


def test_default_heldout_denominator_is_exactly_16_episodes():
    rows = [
        _episode(
            episode_id,
            success=episode_id < 4,
            time_to_success=1.0 if episode_id < 4 else None,
        )
        for episode_id in range(16)
    ]
    summary = summarize_episodes(rows)
    assert summary["expected_episode_count"] == 16
    assert summary["success_denominator"] == 16
    assert summary["success_rate"] == pytest.approx(0.25)
    # Twelve failures remain in the latency statistic at the 12 s horizon.
    assert summary["mean_time_to_first_success_s"] == pytest.approx((4 + 12 * 12) / 16)


def test_incomplete_duplicate_or_noncontiguous_episode_sets_are_rejected():
    with pytest.raises(ValueError, match="expected 16"):
        summarize_episodes([_episode(0)])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_episodes([_episode(0), _episode(0)], expected_episode_count=2)
    with pytest.raises(ValueError, match="exactly 0..1"):
        summarize_episodes([_episode(0), _episode(2)], expected_episode_count=2)


def test_switch_metrics_use_three_fixed_per_episode_denominators():
    rows = []
    for episode_id in range(2):
        switches = (
            SwitchOutcome(0, 150, True, 0.5),
            SwitchOutcome(1, 300, episode_id == 0, 1.0 if episode_id == 0 else None),
            SwitchOutcome(2, 450, False, None),
        )
        rows.append(
            _episode(
                episode_id,
                scenario="FlyCrazyflie-WaypointSwitch-v0",
                switches=switches,
            )
        )
    metrics = summarize_episodes(rows, expected_episode_count=2)["switch_metrics"]
    assert metrics["success_denominator"] == 6
    assert metrics["success_count"] == 3
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert [entry["success_denominator"] for entry in metrics["by_switch"]] == [2, 2, 2]
    assert [entry["success_rate"] for entry in metrics["by_switch"]] == pytest.approx(
        [1.0, 0.5, 0.0]
    )


def test_gust_metrics_never_replace_unconditional_rate_with_conditional_rate():
    rows = []
    for episode_id in range(2):
        gusts = (
            GustRecoveryOutcome(
                0,
                150,
                True,
                True,
                episode_id == 0,
                0.5 if episode_id == 0 else None,
                0.4,
                0.3,
            ),
            GustRecoveryOutcome(1, 300, True, False, True, 1.0, 0.8, 0.6),
            # Episode termination before the final scheduled gust remains an
            # unconditional failed attempt but cannot enter the conditional denominator.
            GustRecoveryOutcome(2, 450, False, None, False, None, None, None, True, "early termination"),
        )
        rows.append(
            _episode(
                episode_id,
                scenario="FlyCrazyflie-GustRecovery-v0",
                terminated=True,
                truncated=False,
                completed_steps=400,
                gusts=gusts,
            )
        )
    metrics = summarize_episodes(rows, expected_episode_count=2)["gust_metrics"]
    assert metrics["unconditional_recovery_denominator"] == 6
    assert metrics["unconditional_recovery_success_count"] == 3
    assert metrics["unconditional_recovery_rate"] == pytest.approx(0.5)
    assert metrics["conditional_recovery_denominator"] == 2
    assert metrics["conditional_recovery_success_count"] == 1
    assert metrics["conditional_recovery_rate"] == pytest.approx(0.5)
    assert metrics["applied_count"] == 4
    assert metrics["not_applied_count"] == 2
    assert metrics["mean_max_displacement_m"] == pytest.approx(0.6)
    assert metrics["mean_post_gust_error_integral_m_s"] == pytest.approx(0.45)
    assert len(metrics["by_gust"]) == 3


def test_gust_summary_rejects_dropped_scheduled_attempts():
    row = _episode(
        0,
        scenario="FlyCrazyflie-GustRecovery-v0",
        gusts=(GustRecoveryOutcome(0, 150, True, True, False, None, 0.2, 0.2),),
    )
    with pytest.raises(ValueError, match="all three"):
        summarize_episodes([row], expected_episode_count=1)
