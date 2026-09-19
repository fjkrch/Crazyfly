from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_drone_lif_checkpoint import (  # noqa: E402
    positive_axis_scale,
    rolling_episode_outcomes,
    trailing_mean,
)


def test_trailing_mean_uses_short_prefix_then_fixed_window():
    assert trailing_mean([1.0, 2.0, 6.0, 10.0], 3) == pytest.approx(
        [1.0, 1.5, 3.0, 6.0]
    )
    with pytest.raises(ValueError, match="positive"):
        trailing_mean([1.0], 0)


def test_episode_rates_retain_exact_shared_denominator_and_overlap_success():
    rows = [
        {
            "completed_updates": 1,
            "completed_episode_count": 2,
            "time_limit_truncation_count": 1,
            "successful_episode_count": 1,
            "failure_termination_count": 1,
        },
        {
            "completed_updates": 2,
            "completed_episode_count": 3,
            "time_limit_truncation_count": 2,
            "successful_episode_count": 2,
            "failure_termination_count": 1,
        },
        {
            "completed_updates": 3,
            "completed_episode_count": 1,
            "time_limit_truncation_count": 1,
            "successful_episode_count": 0,
            "failure_termination_count": 0,
        },
    ]
    result = rolling_episode_outcomes(rows, 2)
    assert result[1] == {
        "completed_updates": 2,
        "completed_episode_denominator": 5,
        "time_limit_survival_numerator": 3,
        "target_success_episode_numerator": 3,
        "failure_termination_numerator": 2,
        "time_limit_survival_rate": pytest.approx(3 / 5),
        "target_success_episode_rate": pytest.approx(3 / 5),
        "failure_termination_rate": pytest.approx(2 / 5),
    }
    assert result[2]["completed_episode_denominator"] == 4
    assert result[2]["target_success_episode_numerator"] == 2


def test_episode_partition_disagreement_fails_closed():
    rows = [
        {
            "completed_updates": 1,
            "completed_episode_count": 2,
            "time_limit_truncation_count": 0,
            "successful_episode_count": 0,
            "failure_termination_count": 1,
        }
    ]
    with pytest.raises(ValueError, match="do not equal"):
        rolling_episode_outcomes(rows, 1)


def test_positive_axis_scale_requires_large_strictly_positive_dynamic_range():
    assert positive_axis_scale([0.001, 1.0]) == "log"
    assert positive_axis_scale([1.0, 10.0]) == "linear"
    assert positive_axis_scale([-1.0, 1000.0]) == "linear"
