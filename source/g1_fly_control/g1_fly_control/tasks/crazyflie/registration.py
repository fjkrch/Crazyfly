"""Idempotent Gym registration for the additive Crazyflie tasks.

Only string entry points are registered here, so importing this module remains
safe before Isaac Sim starts.  Loading either entry point invokes the pinned
upstream validation in :mod:`.env_cfg`.
"""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym

from .adapter import NATIVE_TASK_ID, validate_upstream_contract
from .command_logic import COMMAND_FOLLOW_TASK_ID
from .command_wide_logic import (
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
)

TASK_IDS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
)
MIXED_TRAINING_TASK_ID = "FlyCrazyflie-Mixed-v0"
COMMAND_TASK_IDS = (
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
)
REGISTERED_TASK_IDS = TASK_IDS + (MIXED_TRAINING_TASK_ID, *COMMAND_TASK_IDS)

ENV_ENTRY_POINT = f"{__package__}.env:CrazyflieEnv"
COMMAND_ENV_ENTRY_POINT = f"{__package__}.command_env:CommandFollowEnv"
COMMAND_WIDE_ENV_ENTRY_POINT = f"{__package__}.command_wide_env:CommandFollowWideEnv"
TASK_TO_CFG: Mapping[str, str] = {
    TASK_IDS[0]: f"{__package__}.env_cfg:BalancedWaypointReachEnvCfg",
    TASK_IDS[1]: f"{__package__}.env_cfg:BalancedWaypointSwitchEnvCfg",
    TASK_IDS[2]: f"{__package__}.env_cfg:BalancedGustRecoveryEnvCfg",
    MIXED_TRAINING_TASK_ID: f"{__package__}.env_cfg:BalancedMixedTrainingEnvCfg",
    COMMAND_FOLLOW_TASK_ID: f"{__package__}.command_env_cfg:CommandFollowEnvCfg",
    COMMAND_FOLLOW_WIDE_TASK_ID: (
        f"{__package__}.command_wide_env_cfg:CommandFollowWideEnvCfg"
    ),
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID: (
        f"{__package__}.command_wide_env_cfg:CommandFollowWideWindEnvCfg"
    ),
}
TASK_TO_ENV_ENTRY_POINT: Mapping[str, str] = {
    task_id: (
        COMMAND_ENV_ENTRY_POINT
        if task_id == COMMAND_FOLLOW_TASK_ID
        else COMMAND_WIDE_ENV_ENTRY_POINT
        if task_id in {COMMAND_FOLLOW_WIDE_TASK_ID, COMMAND_FOLLOW_WIDE_WIND_TASK_ID}
        else ENV_ENTRY_POINT
    )
    for task_id in REGISTERED_TASK_IDS
}


def register_tasks() -> tuple[str, ...]:
    """Register all project tasks, accepting only exact prior registrations."""

    if NATIVE_TASK_ID in TASK_TO_CFG:
        raise RuntimeError(f"Refusing to shadow NVIDIA's native task {NATIVE_TASK_ID!r}")

    for task_id, cfg_entry_point in TASK_TO_CFG.items():
        env_entry_point = TASK_TO_ENV_ENTRY_POINT[task_id]
        if task_id in gym.registry:
            existing = gym.spec(task_id)
            existing_cfg = existing.kwargs.get("env_cfg_entry_point")
            if existing.entry_point != env_entry_point or existing_cfg != cfg_entry_point:
                raise RuntimeError(
                    f"Gym task {task_id!r} is already registered with entry_point={existing.entry_point!r}, "
                    f"env_cfg_entry_point={existing_cfg!r}; expected {env_entry_point!r}, "
                    f"{cfg_entry_point!r}"
                )
            continue

        gym.register(
            id=task_id,
            entry_point=env_entry_point,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": cfg_entry_point},
        )
    return REGISTERED_TASK_IDS


register_tasks()


__all__ = [
    "COMMAND_ENV_ENTRY_POINT",
    "COMMAND_FOLLOW_TASK_ID",
    "COMMAND_FOLLOW_WIDE_TASK_ID",
    "COMMAND_FOLLOW_WIDE_WIND_TASK_ID",
    "COMMAND_TASK_IDS",
    "COMMAND_WIDE_ENV_ENTRY_POINT",
    "ENV_ENTRY_POINT",
    "MIXED_TRAINING_TASK_ID",
    "REGISTERED_TASK_IDS",
    "TASK_IDS",
    "TASK_TO_CFG",
    "TASK_TO_ENV_ENTRY_POINT",
    "register_tasks",
    "validate_upstream_contract",
]
