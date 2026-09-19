"""Policy observations with explicitly simulator-state-only target localization."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .state import get_goal_state


def target_relative_body(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """XY target in the yaw-aligned root frame, using simulator state (not onboard estimation)."""
    robot: Articulation = env.scene[asset_cfg.name]
    state = get_goal_state(env)
    delta_w = torch.cat((state.goal_xy_w - robot.data.root_pos_w[:, :2], torch.zeros(env.num_envs, 1, device=env.device)), dim=-1)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)[:, :2]


def joint_velocity_normalized(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel / robot.data.joint_vel_limits.clamp_min(1e-6)
