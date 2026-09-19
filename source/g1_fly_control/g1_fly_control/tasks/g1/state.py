"""Isaac environment-side storage for task targets and reward history."""

from __future__ import annotations

import torch

from .logic import GoalState, initial_goal_state


def get_goal_state(env) -> GoalState:
    state = getattr(env, "_flyg1_goal_state", None)
    if state is None:
        state = initial_goal_state(env.num_envs, env.device)
        env._flyg1_goal_state = state
    return state


def sample_goals(root_xy_w: torch.Tensor, *, minimum: float = 1.0, maximum: float = 3.0) -> torch.Tensor:
    if not 0 < minimum <= maximum:
        raise ValueError("Goal distances must obey 0 < minimum <= maximum.")
    angles = 2.0 * torch.pi * torch.rand(root_xy_w.shape[0], device=root_xy_w.device)
    radii = minimum + (maximum - minimum) * torch.rand(root_xy_w.shape[0], device=root_xy_w.device)
    offsets = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=-1)
    return root_xy_w + offsets

