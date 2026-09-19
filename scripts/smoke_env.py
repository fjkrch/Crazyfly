#!/usr/bin/env python3
"""Run bounded zero/random-action stepping and fail on non-finite simulator state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from _bootstrap import simulator_source_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--progress_interval", type=int, default=100, help="Print a heartbeat every N control steps (0 disables it).")
    parser.add_argument("--output_report", type=Path, help="Write the final PASS report as JSON only after all checks pass.")
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="Use bounded random actions after 10 zero-action steps.",
    )
    parser.add_argument("--random_action_scale", type=float, default=1.0, help="Uniform action range [-scale, scale], where 0 < scale <= 1.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0.0 < args.random_action_scale <= 1.0:
        parser.error("--random_action_scale must be in (0, 1].")
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment

        env = launch_environment(args.task, args.num_envs)
        env.reset()
        if args.num_envs > 1:
            spacing = torch.pdist(env.scene.env_origins[:, :2]).min()
            if spacing < 3.5:
                raise AssertionError(f"G1 environment origins are too close: {float(spacing):.3f} m")
        start = time.monotonic()
        action_dim = env.action_manager.total_action_dim
        action = torch.zeros(args.num_envs, action_dim, device=env.device)
        resets = torch.zeros((), dtype=torch.int64, device=env.device)
        # Keep validation on the simulator device while stepping.  Calling
        # ``bool(cuda_tensor)`` several times per control step forces a CUDA
        # synchronization and makes this smoke test measure host/device stalls
        # rather than simulation throughput.  The flags retain every-step
        # coverage and are materialized once after the rollout.
        targets_in_bounds = torch.ones((), dtype=torch.bool, device=env.device)
        states_finite = torch.ones((), dtype=torch.bool, device=env.device)
        roots_separated = torch.ones((), dtype=torch.bool, device=env.device)
        minimum_target_margin = torch.full((), float("inf"), device=env.device)
        minimum_root_separation = torch.full((), float("inf"), device=env.device)
        for step in range(args.steps):
            if args.random_actions and step >= 10:
                action.uniform_(-args.random_action_scale, args.random_action_scale)
            observation, _, terminated, truncated, _ = env.step(action)
            robot = env.scene["robot"]
            term = env.action_manager.get_term("joint_pos")
            limits = robot.data.soft_joint_pos_limits[:, term._joint_ids]
            target_margin = torch.minimum(
                term.processed_actions - limits[..., 0], limits[..., 1] - term.processed_actions
            )
            targets_in_bounds = targets_in_bounds & (target_margin >= -1e-5).all()
            minimum_target_margin = torch.minimum(minimum_target_margin, target_margin.min())
            tensors = (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel)
            all_states_finite = torch.stack([torch.isfinite(value).all() for value in tensors]).all()
            states_finite = states_finite & all_states_finite
            if args.num_envs > 1:
                root_separation = torch.pdist(robot.data.root_pos_w[:, :2]).min()
                roots_separated = roots_separated & (root_separation >= 0.5)
                minimum_root_separation = torch.minimum(minimum_root_separation, root_separation)
            # ManagerBasedRLEnv performs asynchronous selected-environment resets
            # internally before returning the next observation.
            resets += (terminated | truncated).sum()
            if args.progress_interval > 0 and (step + 1) % args.progress_interval == 0:
                # Bound the queued CUDA work and localize any failed check to
                # a short interval rather than synchronizing only after 1000 steps.
                if not bool(targets_in_bounds):
                    raise AssertionError(f"A position target exceeded a limit by step {step + 1}.")
                if not bool(states_finite):
                    raise FloatingPointError(f"A non-finite G1 state occurred by step {step + 1}.")
                if not bool(roots_separated):
                    raise AssertionError(f"G1 roots approached within 0.5 m by step {step + 1}.")
                print(json.dumps({"status": "RUNNING", "completed_steps": step + 1}), flush=True)
        elapsed = time.monotonic() - start
        if not bool(targets_in_bounds):
            raise AssertionError(
                "A position target exceeded a resolved joint limit during the smoke rollout."
            )
        if not bool(states_finite):
            raise FloatingPointError("A non-finite G1 state occurred during the smoke rollout.")
        if not bool(roots_separated):
            raise AssertionError("G1 roots approached closer than 0.5 m during the smoke rollout.")
        switch_count = int(env._flyg1_goal_state.switch_count.sum())
        push_count = int(getattr(env, "_flyg1_push_count", torch.zeros(1)).sum())
        tensor_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
        )
        gpu_memory_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu_memory_used_mb = (
            int(gpu_memory_query.stdout.splitlines()[0].strip())
            if gpu_memory_query.returncode == 0
            else None
        )
        report = {
            "status": "PASS",
            "task": args.task,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "random_actions": bool(args.random_actions),
            "random_action_scale": args.random_action_scale if args.random_actions else None,
            "resets": int(resets),
            "goal_switches": switch_count,
            "pushes": push_count,
            "control_steps_per_second": args.steps * args.num_envs / elapsed,
            "control_dt": env.step_dt,
            "minimum_target_margin_rad": float(minimum_target_margin),
            "minimum_root_separation_m": (
                None if args.num_envs == 1 else float(minimum_root_separation)
            ),
            "torch_peak_allocated_mb": tensor_memory_mb,
            "gpu_memory_used_mb_system_wide": gpu_memory_used_mb,
            "simulator_source_fingerprint": simulator_source_fingerprint(),
        }
        if args.output_report is not None:
            args.output_report.parent.mkdir(parents=True, exist_ok=True)
            args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2))
        env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
