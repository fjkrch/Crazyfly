"""Only physical invalidity/workspace/time-limit terminations; no posture checks."""

from __future__ import annotations

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .logic import invalid_simulation, workspace_escape


def invalid_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    return invalid_simulation(robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel)


def escaped_workspace(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), radius: float = 8.0):
    robot: Articulation = env.scene[asset_cfg.name]
    return workspace_escape(robot.data.root_pos_w[:, :2], env.scene.env_origins[:, :2], radius=radius)

