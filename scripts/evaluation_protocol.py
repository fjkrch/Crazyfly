"""Deterministic, evaluation-only target and disturbance schedules.

Training samples goals at a relative radius in [1, 3] metres. This protocol
uses a separate fixed set of target instances from the same distribution.
The same seed, environment index, and event ordinal give the same relative
target or world-frame push regardless of policy actions or RNG call order.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math

import torch


def _unit(seed: int, env_id: int, kind: str, ordinal: int, component: str) -> float:
    key = f"heldout_v1:{seed}:{env_id}:{kind}:{ordinal}:{component}".encode()
    return int.from_bytes(sha256(key).digest()[:8], "big") / 2**64


def target_offset(seed: int, env_id: int, ordinal: int) -> tuple[float, float]:
    radius = 1.0 + 2.0 * _unit(seed, env_id, "target", ordinal, "radius")
    angle = 2.0 * math.pi * _unit(seed, env_id, "target", ordinal, "angle")
    return radius * math.cos(angle), radius * math.sin(angle)


def push_direction(seed: int, env_id: int, ordinal: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * _unit(seed, env_id, "push", ordinal, "angle")
    return math.cos(angle), math.sin(angle)


def schedule_manifest(seed: int, episodes: int, task: str) -> dict[str, object]:
    goal_events = 4 if "GoalSwitch" in task else 1
    push_events = 3 if "PushRecovery" in task else 0
    plan = [
        {
            "environment_id": env_id,
            "target_offsets_m": [target_offset(seed, env_id, k) for k in range(goal_events)],
            "push_directions_world_xy": [push_direction(seed, env_id, k) for k in range(push_events)],
        }
        for env_id in range(episodes)
    ]
    specification = {
        "version": "heldout_v1",
        "seed": seed,
        "target_radius_m": [1.0, 3.0],
        "training_target_radius_m": [1.0, 3.0],
        "target_split": "Fixed evaluation-only instances from keyed SHA-256 draws; no training or tuning path reads this schedule.",
        "target_frame": "XY offset from robot root at each target event",
        "push_frame": "world XY, applied to torso_link",
        "push_force_n": 200.0,
        "push_duration_s": 0.10,
        "nominal_event_times_s": [5.0, 10.0, 15.0],
        "plan": plan,
    }
    specification["sha256"] = sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    return specification


class HeldOutEvents:
    """Replace selected Isaac EventManager terms for one evaluated episode/env."""

    def __init__(self, seed: int, episodes: int, task: str):
        self.seed = seed
        self.episodes = episodes
        self.task = task
        self.manifest = schedule_manifest(seed, episodes, task)
        self.events: list[list[dict[str, object]]] = [[] for _ in range(episodes)]
        self.goal_ordinals = [0] * episodes
        self.push_ordinals = [0] * episodes
        self.reset_counts = [0] * episodes

    def install(self, env) -> None:
        from g1_fly_control.tasks.g1.events import start_push
        from g1_fly_control.tasks.g1.state import get_goal_state

        def scheduled_goal(sim_env, env_ids, *, asset_cfg=None):
            print(json.dumps({"protocol_event": "target", "phase": "enter"}), flush=True)
            if asset_cfg is None:
                from isaaclab.managers import SceneEntityCfg
                asset_cfg = SceneEntityCfg("robot")
            if env_ids is None:
                env_ids = torch.arange(sim_env.num_envs, device=sim_env.device)
            ids = [int(item) for item in env_ids]
            robot = sim_env.scene[asset_cfg.name]
            root_xy = robot.data.root_pos_w[env_ids, :2]
            offsets = torch.tensor(
                [target_offset(self.seed, i, self.goal_ordinals[i]) for i in ids],
                dtype=root_xy.dtype, device=root_xy.device,
            )
            goals = root_xy + offsets
            state = get_goal_state(sim_env)
            state.reset(env_ids, root_xy, goals)
            event_time = sim_env.episode_length_buf[env_ids] * sim_env.step_dt
            for row, i in enumerate(ids):
                if self.reset_counts[i] == 1:
                    self.events[i].append({
                        "kind": "target", "ordinal": self.goal_ordinals[i],
                        "time_s": float(event_time[row]),
                        "relative_offset_xy_m": offsets[row].tolist(),
                        "goal_xy_world_m": goals[row].tolist(),
                    })
                self.goal_ordinals[i] += 1
            print(json.dumps({"protocol_event": "target", "phase": "exit", "environment_ids": ids}), flush=True)

        def scheduled_reset_goal(sim_env, env_ids, *, asset_cfg=None):
            if env_ids is None:
                env_ids = torch.arange(sim_env.num_envs, device=sim_env.device)
            for item in env_ids:
                i = int(item)
                self.reset_counts[i] += 1
                self.goal_ordinals[i] = 0
                self.push_ordinals[i] = 0
            scheduled_goal(sim_env, env_ids, asset_cfg=asset_cfg)

        def scheduled_switch_goal(sim_env, env_ids, *, asset_cfg=None):
            if env_ids is None:
                env_ids = torch.arange(sim_env.num_envs, device=sim_env.device)
            scheduled_goal(sim_env, env_ids, asset_cfg=asset_cfg)
            state = get_goal_state(sim_env)
            state.switch_count[env_ids] += 1
            sim_env.extras["flyg1_goal_switch_count"] = state.switch_count.clone()

        def scheduled_push(sim_env, env_ids, force_newton, duration_s, asset_cfg):
            print(json.dumps({"protocol_event": "push", "phase": "enter"}), flush=True)
            if env_ids is None:
                env_ids = torch.arange(sim_env.num_envs, device=sim_env.device)
            ids = [int(item) for item in env_ids]
            # Reuse task validation, duration accounting, and named-body setup.
            # Replace its random draw before the next physics substep.
            start_push(sim_env, env_ids, force_newton, duration_s, asset_cfg)
            direction = torch.tensor(
                [(*push_direction(self.seed, i, self.push_ordinals[i]), 0.0) for i in ids],
                dtype=sim_env._flyg1_push_force.dtype, device=sim_env.device,
            )
            force = direction * force_newton
            impulse = force * duration_s
            sim_env._flyg1_push_force[env_ids] = force
            sim_env._flyg1_push_impulse[env_ids] = impulse
            robot = sim_env.scene[asset_cfg.name]
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=force[:, None, :], torques=torch.zeros_like(force[:, None, :]),
                body_ids=asset_cfg.body_ids, env_ids=env_ids, is_global=True,
            )
            sim_env.extras["flyg1_push_impulse_w"] = sim_env._flyg1_push_impulse.clone()
            event_time = sim_env.episode_length_buf[env_ids] * sim_env.step_dt
            for row, i in enumerate(ids):
                if self.reset_counts[i] == 1:
                    self.events[i].append({
                        "kind": "push", "ordinal": self.push_ordinals[i],
                        "time_s": float(event_time[row]),
                        "body": "torso_link", "frame": "world",
                        "force_world_n": force[row].tolist(),
                        "duration_s": duration_s,
                        "impulse_world_n_s": impulse[row].tolist(),
                    })
                self.push_ordinals[i] += 1
            print(json.dumps({"protocol_event": "push", "phase": "exit", "environment_ids": ids}), flush=True)

        manager = env.event_manager
        term = manager.get_term_cfg("reset_goal")
        term.func = scheduled_reset_goal
        manager.set_term_cfg("reset_goal", term)
        if "GoalSwitch" in self.task:
            term = manager.get_term_cfg("switch_goal")
            term.func = scheduled_switch_goal
            manager.set_term_cfg("switch_goal", term)
        if "PushRecovery" in self.task:
            term = manager.get_term_cfg("push")
            term.func = scheduled_push
            manager.set_term_cfg("push", term)
