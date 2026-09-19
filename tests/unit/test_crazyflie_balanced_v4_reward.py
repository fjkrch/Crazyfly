"""Pure regression gates for the additive balanced-v4 Crazyflie contract."""

from __future__ import annotations

import math

import pytest
import torch

from g1_fly_control.tasks.crazyflie.logic import (
    BALANCED_REWARD_CONTRACT_SHA256,
    BALANCED_TRAINING_CURRICULUM,
    BALANCED_TRAINING_CURRICULUM_SHA256,
    BALANCED_V4_REWARD_CONTRACT_SHA256,
    BALANCED_V4_TRAINING_CURRICULUM,
    BALANCED_V4_TRAINING_CURRICULUM_SHA256,
    DEFAULT_COLLECTIVE_HOVER_ACTION,
    DEFAULT_REWARD_CONTRACT_SHA256,
    DEFAULT_TRAINING_CURRICULUM_SHA256,
    GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
    authenticated_gust_recovery_mask,
    balanced_v4_interval_reward_terms,
    balanced_v4_reward_contract_payload,
    balanced_v4_task_contract_payload,
    training_curriculum_stage_index,
)


def _v4_terms(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    authenticated: torch.Tensor | None = None,
    failed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    count = previous.numel()
    if authenticated is None:
        authenticated = torch.zeros(count, dtype=torch.bool)
    if failed is None:
        failed = torch.zeros(count, dtype=torch.bool)
    actions = torch.zeros(count, 4)
    actions[:, 0] = DEFAULT_COLLECTIVE_HOVER_ACTION
    positions = torch.zeros(count, 3)
    positions[:, 2] = 1.0
    return balanced_v4_interval_reward_terms(
        previous,
        current,
        torch.zeros(count),
        torch.zeros(count, dtype=torch.bool),
        torch.zeros(count, dtype=torch.bool),
        authenticated,
        actions,
        torch.zeros_like(actions),
        failed,
        positions,
        torch.zeros_like(positions),
    )


def test_v4_hashes_are_pinned_and_legacy_hashes_remain_exact() -> None:
    assert DEFAULT_REWARD_CONTRACT_SHA256 == (
        "8a17e7d55b87550c68690a080749e3accbb49321354ecc0b5cd28ef0128f9ba7"
    )
    assert DEFAULT_TRAINING_CURRICULUM_SHA256 == (
        "8ea8c1f8c52205ac028b0a493855764727b59243d45a2a4901d0596f2f91920d"
    )
    assert BALANCED_REWARD_CONTRACT_SHA256 == (
        "bfbba45d2f19ddf8368da6a1318f1dba446d27e49fec9f8b4ec09abf2da611e9"
    )
    assert BALANCED_TRAINING_CURRICULUM_SHA256 == (
        "5aefd508cb3a489db630506c33392f4ad7cf40f39851b0194231a661731f3f0c"
    )
    assert BALANCED_V4_REWARD_CONTRACT_SHA256 == (
        "a5b5d1ab2de1021af930cefc4ccba9e724d48adf805f6f86d91e5ac25b3aaf63"
    )
    assert BALANCED_V4_TRAINING_CURRICULUM_SHA256 == (
        "57a730237bb5adb238847574fd3325e9ac5464221a9a230d4e2589516f493876"
    )
    contract = balanced_v4_task_contract_payload()
    assert contract["reward_sha256"] == BALANCED_V4_REWARD_CONTRACT_SHA256
    assert (
        contract["training_curriculum_sha256"]
        == BALANCED_V4_TRAINING_CURRICULUM_SHA256
    )


def test_v4_contract_records_exact_goal_and_gust_provenance() -> None:
    payload = balanced_v4_reward_contract_payload()
    assert payload["version"] == "crazyflie_balanced_task_reward_v4"
    assert payload["progress_potential_scale_m"] == pytest.approx(2.0)
    assert payload["progress_scale"] == pytest.approx(4.0)
    assert payload["proximity_scale"] == pytest.approx(0.15)
    assert payload["proximity_distance_scale_m"] == pytest.approx(0.80)
    assert payload["gust_recovery_bonus"] == pytest.approx(3.0)
    assert payload["gust_submitted_impulse_abs_tol_n_s"] == pytest.approx(1.0e-6)
    assert payload["audited_robot_mass_kg"] == pytest.approx(
        0.028200002387166023
    )
    assert payload["gust_delta_velocity_m_s"] == pytest.approx(0.75)
    assert payload["gust_recovery_bonus_requires_authenticated_submitted_impulse"] is True
    assert payload["gust_recovery_bonus_one_shot_per_scheduled_gust"] is True
    assert payload["gust_recovery_bonus_zero_outside_scenario"] is True
    assert payload["gust_recovery_bonus_zero_on_failure"] is True


def test_v4_curriculum_changes_only_boundaries_and_reaches_full_at_250k() -> None:
    assert [stage.start_interactions for stage in BALANCED_V4_TRAINING_CURRICULUM] == [
        0,
        50_000,
        125_000,
        250_000,
    ]
    assert [stage.name for stage in BALANCED_V4_TRAINING_CURRICULUM] == [
        "near_3d",
        "local_3d",
        "mid_3d",
        "full",
    ]
    for old, new in zip(
        BALANCED_TRAINING_CURRICULUM,
        BALANCED_V4_TRAINING_CURRICULUM,
        strict=True,
    ):
        assert old.reset_distribution_payload() == new.reset_distribution_payload()
    for interactions, expected in (
        (0, 0),
        (49_999, 0),
        (50_000, 1),
        (124_999, 1),
        (125_000, 2),
        (249_999, 2),
        (250_000, 3),
        (500_000, 3),
    ):
        assert training_curriculum_stage_index(
            interactions, BALANCED_V4_TRAINING_CURRICULUM
        ) == expected


def test_v4_goal_signal_has_exact_scale_and_telescopes() -> None:
    terms = _v4_terms(torch.tensor([1.1]), torch.tensor([1.0]))
    expected_progress = 4.0 * (
        2.0 * math.tanh(1.1 / 2.0) - 2.0 * math.tanh(1.0 / 2.0)
    )
    expected_proximity = 0.15 * math.exp(-0.5 * (1.0 / 0.8) ** 2)
    assert terms["progress"].item() == pytest.approx(expected_progress)
    assert terms["proximity"].item() == pytest.approx(expected_proximity)

    distances = torch.tensor([1.0, 1.2, 1.1, 1.0])
    cycle_progress = [
        _v4_terms(left[None], right[None])["progress"].item()
        for left, right in zip(distances[:-1], distances[1:], strict=True)
    ]
    assert sum(cycle_progress) == pytest.approx(0.0, abs=5.0e-7)


def test_gust_recovery_authentication_rejects_missing_tampered_and_nonfinite_impulses() -> None:
    expected = torch.zeros(6, 3, 3)
    expected[:, 0, 0] = 0.02115
    expected[:, 1, 1] = 0.02115
    expected[:, 2, 0] = -0.02115
    submitted = expected.clone()
    indices = torch.tensor([0, 0, 1, 1, 4, 2])
    submitted[1, 0] = 0.0
    submitted[2, 1, 1] += 2.0 * GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S
    submitted[3, 1, 1] = float("nan")
    mask = authenticated_gust_recovery_mask(
        torch.ones(6, dtype=torch.bool), indices, submitted, expected
    )
    assert mask.tolist() == [True, False, False, False, False, True]


def test_v4_gust_bonus_is_zero_for_reach_rows_and_on_failure() -> None:
    no_event = _v4_terms(torch.tensor([1.0]), torch.tensor([0.9]))
    event = _v4_terms(
        torch.tensor([1.0]),
        torch.tensor([0.9]),
        authenticated=torch.tensor([True]),
    )
    failed = _v4_terms(
        torch.tensor([1.0]),
        torch.tensor([0.9]),
        authenticated=torch.tensor([True]),
        failed=torch.tensor([True]),
    )
    assert no_event["gust_recovery_bonus"].item() == 0.0
    assert event["gust_recovery_bonus"].item() == pytest.approx(3.0)
    assert event["total"].item() - no_event["total"].item() == pytest.approx(3.0)
    assert failed["gust_recovery_bonus"].item() == 0.0

