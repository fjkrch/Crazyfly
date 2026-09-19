"""Pure-tensor task bookkeeping shared by Isaac terms and unit tests.

No state in this module encodes a desired torso height, upright orientation, gait
phase, or support-contact pattern.  A torso contact can be diagnostic, but never
sets termination by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GoalState:
    goal_xy_w: torch.Tensor
    previous_distance: torch.Tensor
    success_latched: torch.Tensor
    within_target_steps: torch.Tensor
    goal_id: torch.Tensor
    switch_count: torch.Tensor
    last_reward_step: int = -1
    last_new_success: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor, root_xy_w: torch.Tensor, goals_xy_w: torch.Tensor) -> None:
        self.goal_xy_w[env_ids] = goals_xy_w
        self.previous_distance[env_ids] = torch.linalg.vector_norm(goals_xy_w - root_xy_w, dim=-1)
        self.success_latched[env_ids] = False
        self.within_target_steps[env_ids] = 0
        self.goal_id[env_ids] += 1
        self.last_reward_step = -1


def initial_goal_state(num_envs: int, device: torch.device | str) -> GoalState:
    return GoalState(
        goal_xy_w=torch.zeros(num_envs, 2, device=device),
        previous_distance=torch.zeros(num_envs, device=device),
        success_latched=torch.zeros(num_envs, dtype=torch.bool, device=device),
        within_target_steps=torch.zeros(num_envs, dtype=torch.long, device=device),
        goal_id=torch.zeros(num_envs, dtype=torch.long, device=device),
        switch_count=torch.zeros(num_envs, dtype=torch.long, device=device),
        last_new_success=torch.zeros(num_envs, device=device),
    )


def xy_distance(root_xy_w: torch.Tensor, goal_xy_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(goal_xy_w - root_xy_w, dim=-1)


def update_goal_reward(
    state: GoalState,
    root_xy_w: torch.Tensor,
    *,
    success_radius: float = 0.30,
    hold_steps: int = 25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return progress numerator, one-shot sustained success, and distance."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive.")
    distance = xy_distance(root_xy_w, state.goal_xy_w)
    progress = state.previous_distance - distance
    state.within_target_steps[:] = torch.where(
        distance <= success_radius, state.within_target_steps + 1, torch.zeros_like(state.within_target_steps)
    )
    newly_successful = (state.within_target_steps >= hold_steps) & ~state.success_latched
    state.previous_distance.copy_(distance)
    state.success_latched |= newly_successful
    return progress, newly_successful.to(distance.dtype), distance


def workspace_escape(root_xy_w: torch.Tensor, env_origins_xy: torch.Tensor, *, radius: float) -> torch.Tensor:
    return torch.linalg.vector_norm(root_xy_w - env_origins_xy, dim=-1) > radius


def invalid_simulation(*tensors: torch.Tensor) -> torch.Tensor:
    if not tensors:
        raise ValueError("At least one tensor is needed to assess simulation validity.")
    return torch.stack([~torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=-1) for tensor in tensors]).any(dim=0)


def bounded_position_targets(normalized_actions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized actions to physical joint limits; never rely on config-only limits."""
    if normalized_actions.shape[-1] != lower.shape[-1] or lower.shape != upper.shape:
        raise ValueError("Action and joint-limit dimensions do not agree.")
    action = normalized_actions.clamp(-1.0, 1.0)
    return 0.5 * (action + 1.0) * (upper - lower) + lower


def default_centered_targets(
    normalized_actions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, default: torch.Tensor
) -> torch.Tensor:
    """Map -1/0/+1 to lower/default/upper without exceeding physical limits."""
    if normalized_actions.shape != default.shape or lower.shape != upper.shape or lower.shape != default.shape:
        raise ValueError("Actions, defaults, and joint limits must have matching shapes.")
    neutral = default.clamp(lower, upper)
    action = normalized_actions.clamp(-1.0, 1.0)
    return torch.where(action >= 0.0, neutral + action * (upper - neutral), neutral + action * (neutral - lower))
