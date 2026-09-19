"""Gym registrations. Import only after AppLauncher is constructed."""

from __future__ import annotations

import gymnasium as gym

TASK_IDS = (
    "FlyG1-GoalReach-FreePosture-v0",
    "FlyG1-GoalSwitch-FreePosture-v0",
    "FlyG1-PushRecovery-FreePosture-v0",
)

_CONFIGS = {
    TASK_IDS[0]: "GoalReachFreePostureEnvCfg",
    TASK_IDS[1]: "GoalSwitchFreePostureEnvCfg",
    TASK_IDS[2]: "PushRecoveryFreePostureEnvCfg",
}

for task_id, config_name in _CONFIGS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__package__}.env_cfg:{config_name}"},
        )

