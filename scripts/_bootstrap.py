"""Make source-layout imports work before the editable package is installed."""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def simulator_source_fingerprint() -> str:
    """Identify the exact project task and diagnostic code checked by a smoke run."""
    files = [ROOT / "scripts" / "_bootstrap.py", ROOT / "scripts" / "smoke_env.py"]
    files.extend(sorted((SOURCE / "g1_fly_control" / "tasks" / "g1").glob("*.py")))
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def selected_env_cfg(task: str, num_envs: int):
    from g1_fly_control.tasks.g1.env_cfg import (
        GoalReachFreePostureEnvCfg,
        GoalSwitchFreePostureEnvCfg,
        PushRecoveryFreePostureEnvCfg,
    )

    configs = {
        "FlyG1-GoalReach-FreePosture-v0": GoalReachFreePostureEnvCfg,
        "FlyG1-GoalSwitch-FreePosture-v0": GoalSwitchFreePostureEnvCfg,
        "FlyG1-PushRecovery-FreePosture-v0": PushRecoveryFreePostureEnvCfg,
    }
    if task not in configs:
        raise ValueError(f"Unknown task {task!r}; choose one of {', '.join(configs)}")
    cfg = configs[task]()
    cfg.scene.num_envs = num_envs
    return cfg


def launch_environment(task: str, num_envs: int, *, render_mode: str | None = None):
    """Registration happens after SimulationApp starts, per Isaac Lab requirements."""
    import gymnasium as gym
    import g1_fly_control.tasks.g1.registration  # noqa: F401

    try:
        wrapped = gym.make(task, cfg=selected_env_cfg(task, num_envs), render_mode=render_mode)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    env = wrapped.unwrapped
    # ManagerBasedRLEnv returns post-reset observations on done. Capture the policy
    # observation immediately before its internal _reset_idx call for timeout value
    # bootstrapping. This wrapper is scoped to the project task instance.
    underlying = env
    original_reset_idx = underlying._reset_idx

    def reset_idx_with_terminal_capture(env_ids):
        if len(env_ids) > 0:
            from g1_fly_control.tasks.g1.events import clear_push_for_envs
            clear_push_for_envs(underlying, env_ids)
            observations = underlying.observation_manager.compute()
            underlying.flyg1_terminal_observation = observations["policy"].clone()
            underlying.flyg1_terminal_root_xy = underlying.scene["robot"].data.root_pos_w[:, :2].clone()
            underlying.flyg1_terminal_goal_xy = underlying._flyg1_goal_state.goal_xy_w.clone()
            underlying.flyg1_terminal_success = underlying._flyg1_goal_state.success_latched.clone()
        return original_reset_idx(env_ids)

    underlying._reset_idx = reset_idx_with_terminal_capture

    # Accumulate the declared mechanical-work proxy at each physics substep.
    # ManagerBasedRLEnv calls scene.update once after every sim.step and computes
    # rewards after all decimated substeps. Reset the accumulator when a new policy
    # action is processed.
    import torch

    underlying._flyg1_interval_work = torch.zeros(num_envs, device=underlying.device)
    original_process_action = underlying.action_manager.process_action
    original_scene_update = underlying.scene.update

    def process_action_with_work_reset(action):
        underlying._flyg1_interval_work.zero_()
        return original_process_action(action)

    def scene_update_with_work_accumulation(dt):
        result = original_scene_update(dt)
        robot = underlying.scene["robot"]
        underlying._flyg1_interval_work += (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=-1) * dt
        if hasattr(underlying, "_flyg1_push_remaining"):
            active = underlying._flyg1_push_remaining > 0
            underlying._flyg1_push_remaining[active] -= 1
            finished = active & (underlying._flyg1_push_remaining == 0)
            if bool(finished.any()):
                from g1_fly_control.tasks.g1.events import clear_push_for_envs
                clear_push_for_envs(underlying, finished.nonzero(as_tuple=False).squeeze(-1))
        return result

    underlying.action_manager.process_action = process_action_with_work_reset
    underlying.scene.update = scene_update_with_work_accumulation
    return env
