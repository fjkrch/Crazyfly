"""Target and disturbance events. Goal transitions reset progress bookkeeping."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .state import get_goal_state, sample_goals


def reset_goal(env, env_ids: torch.Tensor, *, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    robot: Articulation = env.scene[asset_cfg.name]
    state = get_goal_state(env)
    root_xy = robot.data.root_pos_w[env_ids, :2]
    state.reset(env_ids, root_xy, sample_goals(root_xy))


def switch_goal(env, env_ids: torch.Tensor | None, *, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    robot: Articulation = env.scene[asset_cfg.name]
    state = get_goal_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    root_xy = robot.data.root_pos_w[env_ids, :2]
    state.reset(env_ids, root_xy, sample_goals(root_xy))
    state.switch_count[env_ids] += 1
    # Persist schedule information in extras for wrappers/recorders without assuming a logger.
    env.extras["flyg1_goal_switch_count"] = state.switch_count.clone()


def start_push(
    env,
    env_ids: torch.Tensor | None,
    force_newton: float,
    duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> None:
    """Apply a timed horizontal force to the named body in world coordinates."""
    if force_newton <= 0 or duration_s <= 0:
        raise ValueError("Push force and duration must be positive.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if not isinstance(body_ids, list) or len(body_ids) != 1:
        raise ValueError("Push target must resolve to exactly one named G1 body.")
    if not hasattr(env, "_flyg1_push_remaining"):
        env._flyg1_push_remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._flyg1_push_force = torch.zeros(env.num_envs, 3, device=env.device)
        env._flyg1_push_impulse = torch.zeros(env.num_envs, 3, device=env.device)
        env._flyg1_push_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._flyg1_push_body_ids = body_ids
    angle = torch.rand(len(env_ids), device=env.device) * (2.0 * torch.pi)
    force = torch.stack((force_newton * torch.cos(angle), force_newton * torch.sin(angle), torch.zeros_like(angle)), dim=-1)
    steps = round(duration_s / env.physics_dt)
    if steps < 1 or abs(steps * env.physics_dt - duration_s) > 1e-8:
        raise ValueError("Push duration must be an integer number of physics substeps.")
    env._flyg1_push_remaining[env_ids] = steps
    env._flyg1_push_force[env_ids] = force
    env._flyg1_push_impulse[env_ids] = force * duration_s
    env._flyg1_push_count[env_ids] += 1
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=force[:, None, :],
        torques=torch.zeros_like(force[:, None, :]),
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )
    env.extras["flyg1_push_impulse_w"] = env._flyg1_push_impulse.clone()
    env.extras["flyg1_push_count"] = env._flyg1_push_count.clone()


def clear_push_for_envs(env, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_flyg1_push_remaining") or len(env_ids) == 0:
        return
    env._flyg1_push_remaining[env_ids] = 0
    env._flyg1_push_force[env_ids] = 0
    force = torch.zeros(len(env_ids), 1, 3, device=env.device)
    env.scene["robot"].permanent_wrench_composer.set_forces_and_torques(
        forces=force,
        torques=force,
        body_ids=env._flyg1_push_body_ids,
        env_ids=env_ids,
        is_global=True,
    )
