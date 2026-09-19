"""Auditable per-episode metrics; independent seeds are the summary units."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch


def goal_relative_progress_step(
    before_distance: torch.Tensor,
    before_goal_xy: torch.Tensor,
    after_root_xy: torch.Tensor,
    active: torch.Tensor,
    goal_switched: torch.Tensor,
) -> torch.Tensor:
    """Measure motion toward the pre-step goal, omitting target-change intervals.

    The evaluator passes the pre-reset terminal root for finished episodes.
    A goal switch contributes zero on its transition step, avoiding a spurious
    distance jump from comparing different targets.
    """
    after_distance_to_same_goal = torch.linalg.vector_norm(before_goal_xy - after_root_xy, dim=-1)
    return torch.where(active & ~goal_switched, before_distance - after_distance_to_same_goal, 0.0)


@dataclass(frozen=True)
class EpisodeSummary:
    seed: int
    episode_id: int
    success: bool
    xy_progress_m: float | None
    accumulated_goal_relative_progress_m: float
    time_to_target_s: float | None
    mechanical_work_proxy: float
    excessive_impact: float
    joint_limit_frequency: float
    saturation_frequency: float
    goal_switch_count: int = 0
    push_count: int = 0
    push_impulse_vector_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    push_impulse_magnitude_n_s: float = 0.0
    recovery_success: bool | None = None
    recovery_time_s: float | None = None
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0


def summarize_episodes(episodes: Iterable[EpisodeSummary]) -> dict[str, object]:
    rows = list(episodes)
    if not rows:
        raise ValueError("No episode records to summarize.")
    seeds = sorted({row.seed for row in rows})
    seed_rows = []
    for seed in seeds:
        group = [row for row in rows if row.seed == seed]
        plain_progress = [row.xy_progress_m for row in group if row.xy_progress_m is not None]
        target_times = [row.time_to_target_s for row in group if row.time_to_target_s is not None]
        recovery_times = [row.recovery_time_s for row in group if row.recovery_time_s is not None]
        recovery_attempts = sum(row.recovery_attempt_count for row in group)
        recovery_completions = sum(row.recovery_success_count for row in group)
        seed_rows.append(
            {
                "seed": seed,
                "episodes": len(group),
                "success_rate": float(np.mean([row.success for row in group])),
                "mean_xy_progress_m": float(np.mean(plain_progress)) if plain_progress else None,
                "mean_accumulated_goal_relative_progress_m": float(
                    np.mean([row.accumulated_goal_relative_progress_m for row in group])
                ),
                "mean_time_to_target_s": float(np.mean(target_times)) if target_times else None,
                "mean_work_proxy": float(np.mean([row.mechanical_work_proxy for row in group])),
                "mean_excessive_impact": float(np.mean([row.excessive_impact for row in group])),
                "mean_joint_limit_frequency": float(np.mean([row.joint_limit_frequency for row in group])),
                "mean_saturation_frequency": float(np.mean([row.saturation_frequency for row in group])),
                "mean_goal_switch_count": float(np.mean([row.goal_switch_count for row in group])),
                "mean_push_count": float(np.mean([row.push_count for row in group])),
                "mean_push_impulse_magnitude_n_s": float(
                    np.mean([row.push_impulse_magnitude_n_s for row in group])
                ),
                "recovery_success_rate": recovery_completions / recovery_attempts if recovery_attempts else None,
                "mean_recovery_time_s": float(np.mean(recovery_times)) if recovery_times else None,
                "recovery_attempt_count": recovery_attempts,
                "recovery_success_count": recovery_completions,
            }
        )
    return {
        "n_episodes": len(rows),
        "n_independent_training_seeds": len(seeds),
        "per_seed": seed_rows,
        "episodes": [asdict(row) for row in rows],
    }
