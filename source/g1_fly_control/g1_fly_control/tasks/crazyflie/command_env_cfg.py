"""Configuration for the single Crazyflie command-following task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnvCfg

from .adapter import validate_upstream_contract
from .command_logic import (
    ACTION_WIDTH,
    HARD_MAXIMUM_HEIGHT_M,
    HARD_MINIMUM_HEIGHT_M,
    HARD_WORKSPACE_XY_M,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HEIGHT_M,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
    MINIMUM_COMMAND_HOLD_STEPS,
    MINIMUM_HEIGHT_M,
    OBSERVATION_WIDTH,
    SOFT_WORKSPACE_XY_M,
    COMMAND_TRACKING_CONTRACT_SHA256,
    COMMAND_TRACKING_CONTRACT_VERSION,
)


# Fail at configuration import if the installed native task drifted.  The new
# task intentionally inherits its robot, 100 Hz physics, decimation-two
# control rate, and four aggregate-wrench actions unchanged.
UPSTREAM_CONTRACT = validate_upstream_contract()


@configclass
class CommandFollowEnvCfg(QuadcopterEnvCfg):
    """One body-relative velocity/yaw command-following environment."""

    episode_length_s: float = 12.0
    action_space: int = ACTION_WIDTH
    observation_space: int = OBSERVATION_WIDTH
    state_space: int = 0
    debug_vis: bool = False

    command_tracking_contract_version: str = COMMAND_TRACKING_CONTRACT_VERSION
    command_tracking_contract_sha256: str = COMMAND_TRACKING_CONTRACT_SHA256
    command_schedule_seed: int = 0
    command_hold_min_steps: int = MINIMUM_COMMAND_HOLD_STEPS
    command_hold_max_steps: int = MAXIMUM_COMMAND_HOLD_STEPS
    maximum_horizontal_speed_m_s: float = MAXIMUM_HORIZONTAL_SPEED_M_S
    maximum_vertical_speed_m_s: float = MAXIMUM_VERTICAL_SPEED_M_S
    maximum_yaw_rate_rad_s: float = MAXIMUM_YAW_RATE_RAD_S

    spawn_height_m: float = 1.0
    minimum_height_m: float = MINIMUM_HEIGHT_M
    maximum_height_m: float = MAXIMUM_HEIGHT_M
    workspace_xy_limit_m: float = SOFT_WORKSPACE_XY_M
    hard_minimum_height_m: float = HARD_MINIMUM_HEIGHT_M
    hard_maximum_height_m: float = HARD_MAXIMUM_HEIGHT_M
    hard_workspace_xy_limit_m: float = HARD_WORKSPACE_XY_M


__all__ = ["CommandFollowEnvCfg", "UPSTREAM_CONTRACT"]
