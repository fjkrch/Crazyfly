"""Free-posture reward terms. Values are rates; Isaac Lab applies step_dt once."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .logic import update_goal_reward
from .state import get_goal_state


def _goal_update_once(env, robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """RewardManager calls several terms per step; mutate distance history exactly once."""
    state = get_goal_state(env)
    step = int(getattr(env, "common_step_counter", -1))
    if state.last_reward_step != step:
        progress, success, _ = update_goal_reward(state, robot.data.root_pos_w[:, :2])
        state.last_reward_step = step
        state.last_new_success = success
        return progress, success
    # A second term in the same reward pass sees zero progress but the cached one-shot bonus.
    return torch.zeros(env.num_envs, device=env.device), state.last_new_success


def goal_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    progress, _ = _goal_update_once(env, robot)
    return progress / env.step_dt


def goal_success(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), success_radius: float = 0.30) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    # The initial implementation uses one globally declared radius. A future version
    # changing it must update both the task spec and the cached goal state behavior.
    if success_radius != 0.30:
        raise ValueError("Only the preregistered 0.30 m success radius is supported in this task version.")
    _, success = _goal_update_once(env, robot)
    return success / env.step_dt


def mechanical_work_proxy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Physics-substep sum of |applied torque × velocity|, returned as a rate."""
    interval_work = getattr(env, "_flyg1_interval_work", None)
    if interval_work is None:
        raise RuntimeError("Free-posture environment must install physics-substep work accumulator.")
    return interval_work / env.step_dt


def action_change(env) -> torch.Tensor:
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    return (action - previous).square().sum(dim=-1) / env.step_dt


def joint_limit_violation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = env.action_manager.get_term("joint_pos")._joint_ids
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    joint_pos = robot.data.joint_pos[:, joint_ids]
    below = (limits[..., 0] - joint_pos).clamp_min(0.0)
    above = (joint_pos - limits[..., 1]).clamp_min(0.0)
    return (below + above).sum(dim=-1)


def excessive_impact(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 350.0,
) -> torch.Tensor:
    """Diagnostic impact penalty; no body/contact selection creates termination."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, 0], dim=-1)
    return (force - threshold).clamp_min(0.0).sum(dim=-1)
