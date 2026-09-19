"""Pure regression tests for the additive Crazyflie balanced-v3 task core."""

from __future__ import annotations

import pytest
import torch

from g1_fly_control.tasks.crazyflie.logic import (
    BALANCED_REWARD_CONTRACT_SHA256,
    BALANCED_TRAINING_CURRICULUM,
    BALANCED_TRAINING_CURRICULUM_SHA256,
    DEFAULT_COLLECTIVE_HOVER_ACTION,
    DEFAULT_REWARD_CONTRACT_SHA256,
    DEFAULT_TRAINING_CURRICULUM,
    DEFAULT_TRAINING_CURRICULUM_SHA256,
    balanced_boundary_metric,
    balanced_interval_reward_terms,
    balanced_reward_contract_payload,
    balanced_switch_target_curriculum_payload,
    balanced_task_contract_payload,
    training_curriculum_stage_index,
)


def _balanced_terms(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    *,
    speed: torch.Tensor | None = None,
    newly_successful: torch.Tensor | None = None,
    latched: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    action_delta: torch.Tensor | None = None,
    failed: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    origins: torch.Tensor | None = None,
):
    count = previous_distance.numel()
    if speed is None:
        speed = torch.zeros(count)
    if newly_successful is None:
        newly_successful = torch.zeros(count, dtype=torch.bool)
    if latched is None:
        latched = torch.zeros(count, dtype=torch.bool)
    if actions is None:
        actions = torch.zeros(count, 4)
        actions[:, 0] = DEFAULT_COLLECTIVE_HOVER_ACTION
    if action_delta is None:
        action_delta = torch.zeros(count, 4)
    if failed is None:
        failed = torch.zeros(count, dtype=torch.bool)
    if positions is None:
        positions = torch.zeros(count, 3)
        positions[:, 2] = 1.0
    if origins is None:
        origins = torch.zeros(count, 3)
    return balanced_interval_reward_terms(
        previous_distance,
        current_distance,
        speed,
        newly_successful,
        latched,
        actions,
        action_delta,
        failed,
        positions,
        origins,
    )


def test_balanced_contract_hashes_are_additive_and_v2_hashes_remain_exact():
    assert BALANCED_REWARD_CONTRACT_SHA256 == (
        "bfbba45d2f19ddf8368da6a1318f1dba446d27e49fec9f8b4ec09abf2da611e9"
    )
    assert BALANCED_TRAINING_CURRICULUM_SHA256 == (
        "5aefd508cb3a489db630506c33392f4ad7cf40f39851b0194231a661731f3f0c"
    )
    assert DEFAULT_REWARD_CONTRACT_SHA256 == (
        "8a17e7d55b87550c68690a080749e3accbb49321354ecc0b5cd28ef0128f9ba7"
    )
    assert DEFAULT_TRAINING_CURRICULUM_SHA256 == (
        "8ea8c1f8c52205ac028b0a493855764727b59243d45a2a4901d0596f2f91920d"
    )
    contract = balanced_task_contract_payload()
    assert contract["reward_sha256"] == BALANCED_REWARD_CONTRACT_SHA256
    assert contract["training_curriculum_sha256"] == BALANCED_TRAINING_CURRICULUM_SHA256
    assert contract["reward"]["positive_terms_zero_on_failure"] is True
    assert contract["reward"]["negative_progress_retained_on_failure"] is True


def test_balanced_contract_records_every_critical_scale_and_rule():
    payload = balanced_reward_contract_payload()
    expected = {
        "progress_potential_scale_m": 2.0,
        "progress_scale": 1.0,
        "proximity_scale": 0.015,
        "proximity_distance_scale_m": 0.50,
        "dwell_reward_per_interval": 0.020,
        "braking_scale": 0.010,
        "braking_speed_reference_m_s": 0.50,
        "braking_normalized_squared_cap": 4.0,
        "success_bonus": 3.0,
        "retention_scale": 0.040,
        "retention_ramp_m": 0.50,
        "survival_reward_per_interval": 0.010,
        "control_effort_scale": 0.0025,
        "collective_effort_reference": 0.15,
        "moment_effort_reference": 0.020,
        "action_change_scale": 0.0005,
        "collective_action_change_reference": 0.10,
        "moment_action_change_reference": 0.010,
        "boundary_scale": 0.050,
        "boundary_low_onset_m": 0.30,
        "boundary_low_width_m": 0.20,
        "boundary_high_onset_m": 1.70,
        "boundary_high_width_m": 0.20,
        "boundary_xy_onset_m": 2.25,
        "boundary_xy_width_m": 0.25,
        "failure_penalty": 25.0,
        "success_distance_m": 0.20,
        "success_speed_m_s": 0.25,
        "success_dwell_steps": 25,
        "minimum_height_m": 0.10,
        "maximum_height_m": 2.00,
        "workspace_xy_limit_m": 2.75,
    }
    for key, value in expected.items():
        assert payload[key] == pytest.approx(value)
    assert payload["boundary_xy_frame"] == "environment_local"
    assert payload["boundary_z_frame"] == "world"
    assert payload["success_terminates_episode"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("progress_potential_scale_m", 0.0),
        ("proximity_distance_scale_m", 0.0),
        ("collective_effort_reference", 0.0),
        ("boundary_xy_onset_m", 2.75),
        ("boundary_low_onset_m", 0.09),
        ("success_dwell_steps", 0),
        ("failure_penalty", -1.0),
        ("braking_scale", float("nan")),
    ),
)
def test_balanced_contract_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        balanced_reward_contract_payload(**{field: value})


def test_balanced_progress_is_bounded_telescoping_and_cycle_cannot_profit():
    distances = torch.tensor(
        [1.0, 1.20, 1.18, 1.16, 1.14, 1.12, 1.10, 1.08, 1.06, 1.04, 1.02, 1.0]
    )
    progress = []
    for previous, current in zip(distances[:-1], distances[1:], strict=True):
        terms = _balanced_terms(previous[None], current[None])
        progress.append(terms["progress"].item())
    assert sum(progress) == pytest.approx(0.0, abs=2.0e-7)
    assert progress[0] < 0.0
    assert all(value > 0.0 for value in progress[1:])


def test_balanced_failure_masks_positive_terms_but_retains_negative_progress_and_costs():
    hover = DEFAULT_COLLECTIVE_HOVER_ACTION
    actions = torch.tensor(
        [[hover + 0.15, 0.02, 0.0, 0.0], [hover + 0.15, 0.02, 0.0, 0.0]]
    )
    delta = torch.tensor([[0.10, 0.01, 0.0, 0.0]]).expand(2, -1)
    positions = torch.tensor([[0.0, 0.0, 2.10], [2.80, 0.0, 1.0]])
    terms = _balanced_terms(
        torch.tensor([1.0, 0.10]),
        torch.tensor([0.10, 1.0]),
        newly_successful=torch.tensor([True, True]),
        latched=torch.tensor([True, True]),
        actions=actions,
        action_delta=delta,
        failed=torch.tensor([True, True]),
        positions=positions,
    )
    assert terms["progress"][0].item() == 0.0
    assert terms["progress"][1].item() < 0.0
    for name in ("proximity", "dwell", "success_bonus", "survival"):
        assert terms[name].tolist() == pytest.approx([0.0, 0.0])
    assert terms["retention"].tolist() == pytest.approx([0.0, -0.04])
    for name in ("control_effort", "action_change", "boundary", "failure"):
        assert torch.all(terms[name] < 0.0)
    assert torch.isfinite(terms["total"]).all()


def test_balanced_goal_hold_braking_effort_change_and_retention_scales():
    hover = DEFAULT_COLLECTIVE_HOVER_ACTION
    stable = _balanced_terms(torch.tensor([0.0]), torch.tensor([0.0]))
    assert stable["proximity"].item() == pytest.approx(0.015)
    assert stable["dwell"].item() == pytest.approx(0.020)
    assert stable["survival"].item() == pytest.approx(0.010)
    assert stable["total"].item() == pytest.approx(0.045)

    actions = torch.tensor([[hover + 0.15, 0.02, 0.0, 0.0]])
    delta = torch.tensor([[0.10, 0.01, 0.0, 0.0]])
    terms = _balanced_terms(
        torch.tensor([0.45]),
        torch.tensor([0.45]),
        speed=torch.tensor([0.25]),
        latched=torch.tensor([True]),
        actions=actions,
        action_delta=delta,
    )
    q = torch.exp(torch.tensor(-0.5 * (0.45 / 0.50) ** 2)).item()
    assert terms["braking"].item() == pytest.approx(-0.010 * q * 0.25)
    assert terms["retention"].item() == pytest.approx(-0.020)
    assert terms["control_effort"].item() == pytest.approx(-0.005)
    assert terms["action_change"].item() == pytest.approx(-0.001)


def test_balanced_boundary_uses_local_xy_absolute_z_and_caps_nonfinite_rows():
    origins = torch.tensor(
        [[10.0, -4.0, 50.0], [10.0, -4.0, 50.0], [10.0, -4.0, 50.0],
         [10.0, -4.0, 50.0], [10.0, -4.0, 50.0]]
    )
    positions = torch.tensor(
        [
            [10.0, -4.0, 1.0],
            [10.0, -4.0, 0.10],
            [10.0, -4.0, 2.00],
            [12.75, -4.0, 1.0],
            [float("nan"), -4.0, 1.0],
        ]
    )
    metric = balanced_boundary_metric(positions, origins)
    assert metric.tolist() == pytest.approx([0.0, 1.0, 2.25, 4.0, 4.0])


def test_balanced_curriculum_is_exact_progressive_and_ends_at_legacy_full_scope():
    expected = [
        ("near_3d", 0, -0.35, 0.35, 0.80, 1.20, 0.40),
        ("local_3d", 200_000, -0.75, 0.75, 0.65, 1.35, 0.50),
        ("mid_3d", 500_000, -1.25, 1.25, 0.60, 1.50, 0.65),
        ("full", 1_000_000, -2.0, 2.0, 0.50, 1.50, 0.75),
    ]
    observed = [
        (
            stage.name,
            stage.start_interactions,
            stage.goal_xy_min_m,
            stage.goal_xy_max_m,
            stage.goal_z_min_m,
            stage.goal_z_max_m,
            stage.minimum_goal_separation_m,
        )
        for stage in BALANCED_TRAINING_CURRICULUM
    ]
    assert observed == expected
    assert (
        BALANCED_TRAINING_CURRICULUM[-1].reset_distribution_payload()
        == DEFAULT_TRAINING_CURRICULUM[-1].reset_distribution_payload()
    )
    for interactions, expected_index in (
        (0, 0),
        (199_999, 0),
        (200_000, 1),
        (499_999, 1),
        (500_000, 2),
        (999_999, 2),
        (1_000_000, 3),
    ):
        assert training_curriculum_stage_index(
            interactions, BALANCED_TRAINING_CURRICULUM
        ) == expected_index


def test_balanced_switch_targets_remain_in_the_active_stage():
    payload = balanced_switch_target_curriculum_payload()
    assert payload["target_0_distribution"] == "active_episode_reset_curriculum_stage"
    assert payload["targets_1_through_3_distribution"] == (
        "active_episode_reset_curriculum_stage"
    )
    assert payload["non_switch_mixed_rows_repeat_target_0"] is True
