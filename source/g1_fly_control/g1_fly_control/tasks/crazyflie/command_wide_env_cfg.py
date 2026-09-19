"""Isaac Lab configuration for the additive command-v2 Crazyflie tasks."""

from __future__ import annotations

from isaaclab.utils import configclass

from .command_env_cfg import CommandFollowEnvCfg, UPSTREAM_CONTRACT
from .command_wide_logic import (
    ACTION_WIDTH,
    COMMAND_WIDE_CONTRACT_VERSION,
    COMMAND_WIDE_STILL_CONTRACT_SHA256,
    COMMAND_WIDE_WIND_CONTRACT_SHA256,
    HARD_MAXIMUM_HEIGHT_M,
    HARD_MINIMUM_HEIGHT_M,
    HARD_WORKSPACE_XY_M,
    HELD_OUT_WIND_SEED,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HEIGHT_M,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
    MINIMUM_COMMAND_HOLD_STEPS,
    MINIMUM_HEIGHT_M,
    OBSERVATION_WIDTH,
    SOFT_WORKSPACE_XY_M,
    TRAINING_WIND_SEED,
    WIND_REFERENCE_ARM_M,
)


@configclass
class CommandFollowWideEnvCfg(CommandFollowEnvCfg):
    """Still-air command-v2 configuration with the expanded command envelope."""

    episode_length_s: float = 12.0
    action_space: int = ACTION_WIDTH
    observation_space: int = OBSERVATION_WIDTH
    state_space: int = 0
    debug_vis: bool = False

    command_tracking_contract_version: str = COMMAND_WIDE_CONTRACT_VERSION
    command_tracking_contract_sha256: str = COMMAND_WIDE_STILL_CONTRACT_SHA256
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

    wind_enabled: bool = False
    wind_schedule_seed: int = TRAINING_WIND_SEED
    held_out_wind_seed: int = HELD_OUT_WIND_SEED
    wind_reference_arm_m: float = WIND_REFERENCE_ARM_M


@configclass
class CommandFollowWideWindEnvCfg(CommandFollowWideEnvCfg):
    """Command-v2 configuration with deterministic physical wind enabled."""

    command_tracking_contract_sha256: str = COMMAND_WIDE_WIND_CONTRACT_SHA256
    wind_enabled: bool = True


__all__ = [
    "CommandFollowWideEnvCfg",
    "CommandFollowWideWindEnvCfg",
    "UPSTREAM_CONTRACT",
]
