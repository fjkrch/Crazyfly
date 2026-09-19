"""Evaluation accounting for target switches and independent training seeds."""

import pytest
import torch

from g1_fly_control.evaluation.metrics import EpisodeSummary, goal_relative_progress_step, summarize_episodes


def test_goal_relative_progress_omits_target_change_and_finished_rows():
    before_distance = torch.tensor([2.0, 1.5, 1.0])
    before_goal = torch.tensor([[2.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    after_root = torch.tensor([[0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    active = torch.tensor([True, True, False])
    switched = torch.tensor([False, True, False])
    delta = goal_relative_progress_step(before_distance, before_goal, after_root, active, switched)
    assert delta.tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_seed_summary_keeps_switch_progress_separate_from_start_to_final_progress():
    base = dict(
        success=False, xy_progress_m=None, time_to_target_s=None,
        mechanical_work_proxy=10.0, excessive_impact=2.0,
        joint_limit_frequency=0.1, saturation_frequency=0.2,
    )
    rows = [
        EpisodeSummary(
            seed=0, episode_id=0, accumulated_goal_relative_progress_m=1.0,
            goal_switch_count=2, push_count=1,
            push_impulse_vector_n_s=(20.0, 0.0, 0.0), push_impulse_magnitude_n_s=20.0,
            **base,
        ),
        EpisodeSummary(
            seed=0, episode_id=1, accumulated_goal_relative_progress_m=-0.5,
            goal_switch_count=1, **base,
        ),
        EpisodeSummary(
            seed=1, episode_id=0, accumulated_goal_relative_progress_m=0.75,
            xy_progress_m=0.75, success=True, time_to_target_s=3.0,
            mechanical_work_proxy=20.0, excessive_impact=0.0,
            joint_limit_frequency=0.0, saturation_frequency=0.0,
        ),
    ]
    summary = summarize_episodes(rows)
    assert summary["n_independent_training_seeds"] == 2
    first = summary["per_seed"][0]
    assert first["mean_xy_progress_m"] is None
    assert first["mean_accumulated_goal_relative_progress_m"] == pytest.approx(0.25)
    assert first["mean_goal_switch_count"] == pytest.approx(1.5)
    assert first["mean_push_impulse_magnitude_n_s"] == pytest.approx(10.0)
    assert summary["per_seed"][1]["mean_time_to_target_s"] == pytest.approx(3.0)
    assert summary["episodes"][0]["push_impulse_vector_n_s"] == (20.0, 0.0, 0.0)


def test_push_recovery_rate_counts_eligible_pushes_not_episodes():
    base = dict(
        seed=0, success=True, xy_progress_m=1.0,
        accumulated_goal_relative_progress_m=1.0, time_to_target_s=16.0,
        mechanical_work_proxy=0.0, excessive_impact=0.0,
        joint_limit_frequency=0.0, saturation_frequency=0.0,
    )
    episodes = [
        EpisodeSummary(episode_id=0, push_count=3, recovery_success=True,
                       recovery_attempt_count=3, recovery_success_count=1,
                       recovery_time_s=1.0, **base),
        EpisodeSummary(episode_id=1, push_count=1, recovery_success=True,
                       recovery_attempt_count=1, recovery_success_count=1,
                       recovery_time_s=2.0, **base),
    ]
    seed = summarize_episodes(episodes)["per_seed"][0]
    assert seed["recovery_attempt_count"] == 4
    assert seed["recovery_success_count"] == 2
    assert seed["recovery_success_rate"] == pytest.approx(0.5)
