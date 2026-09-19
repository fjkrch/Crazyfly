#!/usr/bin/env python3
"""Load G1 and write an auditable resolved joint/actuator report."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from _bootstrap import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "asset_audit.json")
    parser.add_argument("--probe", action="store_true", help="Verify each named action channel with low-amplitude joint targets.")
    parser.add_argument("--probe_steps", type=int, default=5)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import torch
        from _bootstrap import launch_environment

        env = launch_environment("FlyG1-GoalReach-FreePosture-v0", 1)
        print("[AUDIT] Environment created; resetting G1...", flush=True)
        env.reset()
        print("[AUDIT] Reset complete; exporting resolved articulation...", flush=True)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        print("[AUDIT] Retrieved robot and action term", flush=True)
        actuator_data = {}
        for name, actuator in robot.actuators.items():
            print(f"[AUDIT] Reading actuator {name}", flush=True)
            actuator_data[name] = {
                "joint_names": list(getattr(actuator, "joint_names", [])),
                "effort_limit": getattr(actuator, "effort_limit", None).detach().cpu().tolist() if hasattr(getattr(actuator, "effort_limit", None), "detach") else None,
                "velocity_limit": getattr(actuator, "velocity_limit", None).detach().cpu().tolist() if hasattr(getattr(actuator, "velocity_limit", None), "detach") else None,
            }
        print("[AUDIT] Actuators read; compiling limits", flush=True)
        report = {
            "status": "PASS",
            "asset_usd_path": str(env.cfg.scene.robot.spawn.usd_path),
            "asset_identifier_sha256": sha256(str(env.cfg.scene.robot.spawn.usd_path).encode()).hexdigest(),
            "num_joints": robot.num_joints,
            "joint_names": list(robot.joint_names),
            "action_joint_names": list(action_term._joint_names),
            "soft_joint_position_limits": robot.data.soft_joint_pos_limits[0].detach().cpu().tolist(),
            "joint_velocity_limits": robot.data.joint_vel_limits[0].detach().cpu().tolist(),
            "joint_effort_limits": robot.data.joint_effort_limits[0].detach().cpu().tolist(),
            "joint_stiffness": robot.data.joint_stiffness[0].detach().cpu().tolist(),
            "joint_damping": robot.data.joint_damping[0].detach().cpu().tolist(),
            "body_names": list(robot.body_names),
            "body_masses": robot.data.default_mass[0].detach().cpu().tolist(),
            "root_free": not bool(env.cfg.scene.robot.spawn.articulation_props.fix_root_link),
            "self_collisions": bool(env.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions),
            "actuators": actuator_data,
            "physics_dt": env.cfg.sim.dt,
            "control_decimation": env.cfg.decimation,
            "control_dt": env.step_dt,
        }
        if args.probe:
            if args.probe_steps < 1:
                raise ValueError("--probe_steps must be positive.")
            probe_results = []
            joint_ids = action_term._joint_ids
            action_dim = action_term.action_dim
            for action_index, (joint_name, joint_id) in enumerate(zip(action_term._joint_names, joint_ids, strict=True)):
                env.reset()
                before = robot.data.joint_pos[0, joint_id].clone()
                action = torch.zeros(1, action_dim, device=env.device)
                action[0, action_index] = 0.05
                for _ in range(args.probe_steps):
                    env.step(action)
                lower, upper = robot.data.soft_joint_pos_limits[0, joint_id]
                default = robot.data.default_joint_pos[0, joint_id].clamp(lower, upper)
                expected = default + 0.05 * (upper - default)
                actual_target = action_term.processed_actions[0, action_index]
                if not torch.isclose(actual_target, expected, atol=1e-4, rtol=0):
                    raise AssertionError(f"Action channel {action_index} did not target {joint_name} as expected.")
                probe_results.append({
                    "action_index": action_index, "joint_name": joint_name, "joint_id": int(joint_id),
                    "target_rad": float(actual_target), "measured_delta_rad": float(robot.data.joint_pos[0, joint_id] - before),
                })
            report["probe_status"] = "PASS"
            report["probe_results"] = probe_results
        print("[AUDIT] Report assembled; writing JSON", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": report["status"], "output": str(args.output), "num_joints": report["num_joints"], "action_joints": len(report["action_joint_names"]), "control_dt": report["control_dt"], "probe_status": report.get("probe_status", "not_run")}, indent=2))
        env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
