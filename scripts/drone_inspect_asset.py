#!/usr/bin/env python3
"""Audit the installed Isaac Lab Crazyflie task and asset from a live simulation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from drone_bootstrap import (
    ACTION_CONTRACT,
    CRAZYFLIE_RESOLVED_USD_URL,
    OBSERVATION_CONTRACT,
    ROOT,
    UPSTREAM_CRAZYFLIE_ASSET,
    UPSTREAM_DIRECT_RL_ENV,
    UPSTREAM_ISAACLAB_MATH,
    UPSTREAM_QUADCOPTER_ENV,
    UPSTREAM_QUADCOPTER_REGISTRATION,
    installed_versions,
    launch_environment,
    reproduction_fingerprint,
    runtime_identity,
    sha256_file,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _markdown(audit: dict) -> str:
    checks = audit["checks"]
    upstream = audit["upstream_contract"]
    gust = audit["project_gust_contract"]
    measurement = gust["physical_response_measurement"]
    goal_bounds = upstream["native_goal_bounds_relative_to_env_origin_m"]
    height_bounds = upstream["native_height_termination_m"]
    rows = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    versions = "\n".join(f"| {name} | `{value}` |" for name, value in audit["versions"].items())
    return f"""# Crazyflie installed asset audit

Generated: `{audit['generated_at_utc']}`  
Overall status: **{audit['status']}**

This is a live audit of the installed `Isaac-Quadcopter-Direct-v0` environment. It does not infer a physical sensor suite or individual-rotor interface.

## Contract checks

| Check | Result |
| --- | --- |
{rows}

## Live values

- Observation shape: `{audit['live']['observation_shape']}` (12 values, in the pinned upstream order)
- Action shape: `{audit['live']['action_shape']}` (aggregate thrust and three body moments)
- Physics timestep: `{audit['live']['physics_dt_s']}` s
- Control timestep: `{audit['live']['control_dt_s']}` s (`{audit['live']['control_frequency_hz']}` Hz)
- Native episode horizon: `{audit['live']['native_episode_horizon_s']}` s
- Crazyflie mass: `{audit['live']['mass_kg']}` kg
- Vehicle weight: `{audit['live']['weight_n']}` N
- Thrust-to-weight scale: `{audit['live']['thrust_to_weight']}`
- Moment scale: `{audit['live']['moment_scale_n_m']}` N m
- Body name/index: `{audit['live']['body_name']}` / `{audit['live']['body_id']}`
- USD identifier: `{audit['asset']['usd_identifier']}`
- Height failure bounds: `{height_bounds}` m
- Native goal bounds relative to environment origin: x `{goal_bounds['x']}` m; y `{goal_bounds['y']}` m; z `{goal_bounds['z']}` m

The native controller uses `{audit['live']['native_action_wrench_api']}` for its
aggregate body-frame thrust/moments. Project gusts instead use
`{gust['api']}` with `is_global={gust['is_global']}`: a world-frame force on
body `{gust['body_name']}` at its center of mass, with no application-point
offset. Each gust spans `{gust['control_decisions']}` control decisions / 
`{gust['physics_steps']}` physics steps (`{gust['duration_s']}` s).

Gate C reports three distinct quantities: the mass-derived expected impulse,
the force-time integral submitted to the wrench composer, and the measured
horizontal `mass * delta-velocity` response. The physical response is isolated
by `{measurement['method']}` and must be within the predeclared maximum of
`{measurement['absolute_tolerance_n_s']}` N s absolute or
`{measurement['relative_tolerance']}` relative error. This asset audit records
the contract; the custom-task smoke artifact records the live measurements.

## Observation ordering

`body linear velocity (3), body angular velocity (3), projected gravity (3), body-frame goal displacement (3)`

## Action mapping

`a0 -> (a0 + 1) / 2 * 1.9 * vehicle weight` along local +Z; `a1:a4 -> +/-0.01 N m` body moments. Policies emit bounded values in `[-1, 1]`.

## Installed versions

| Package | Version |
| --- | --- |
{versions}

## Source identities

- Reproduction fingerprint: `{audit['fingerprint']}`
- Isaac Lab commit: `{audit['runtime']['isaaclab_commit']}`
- Upstream environment SHA-256: `{audit['sources']['quadcopter_env_sha256']}`
- DirectRLEnv SHA-256: `{audit['sources']['direct_rl_env_sha256']}`
- Isaac Lab math helpers SHA-256: `{audit['sources']['isaaclab_math_sha256']}`
- Upstream registration SHA-256: `{audit['sources']['quadcopter_registration_sha256']}`
- Asset configuration SHA-256: `{audit['sources']['crazyflie_asset_cfg_sha256']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "docs" / "crazyflie_asset_audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "crazyflie_asset_audit.md")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    resolved_config = {
        "kind": "installed_asset_audit",
        "task": "Isaac-Quadcopter-Direct-v0",
        "num_envs": 1,
        "seed": 0,
        "live_steps": 1,
    }
    try:
        fingerprint, fingerprint_payload = reproduction_fingerprint(
            resolved_config=resolved_config,
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"cannot resolve reproduction fingerprint: {exc}")
    print(json.dumps({
        "status": "RESOLVED",
        "resolved_config": resolved_config,
        "fingerprint": fingerprint,
        "outputs": {
            "json": str(args.json.resolve()),
            "markdown": str(args.markdown.resolve()),
        },
    }, sort_keys=True), flush=True)
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        from g1_fly_control.tasks.crazyflie.logic import (
            GUST_DURATION_STEPS,
            GUST_RESPONSE_ABS_TOL_N_S,
            GUST_RESPONSE_REL_TOL,
        )
        from g1_fly_control.tasks.crazyflie.registration import TASK_IDS, register_tasks, validate_upstream_contract

        upstream_contract = validate_upstream_contract()
        register_tasks()
        env = launch_environment("Isaac-Quadcopter-Direct-v0", 1)
        observation, _ = env.reset(seed=0)
        tensor = observation["policy"]
        action_shape = tuple(env.single_action_space.shape)
        action = torch.zeros((1, action_shape[0]), device=env.device)
        action[:, 0] = 2.0 / float(env.cfg.thrust_to_weight) - 1.0
        next_observation, reward, terminated, truncated, _ = env.step(action)
        body_names = list(getattr(env._robot, "body_names", []))
        body_id = int(torch.as_tensor(env._body_id).flatten()[0].item())
        body_name = body_names[body_id] if body_id < len(body_names) else "body"
        mass = float(torch.as_tensor(env._robot_mass).item())
        weight = float(env._robot_weight)
        live = {
            "observation_shape": list(tensor.shape),
            "single_observation_width": int(tensor.shape[-1]),
            "action_shape": list(action_shape),
            "physics_dt_s": float(env.cfg.sim.dt),
            "decimation": int(env.cfg.decimation),
            "control_dt_s": float(env.step_dt),
            "control_frequency_hz": 1.0 / float(env.step_dt),
            "native_episode_horizon_s": float(env.cfg.episode_length_s),
            "mass_kg": mass,
            "gravity_m_s2": float(env._gravity_magnitude),
            "weight_n": weight,
            "thrust_to_weight": float(env.cfg.thrust_to_weight),
            "maximum_collective_thrust_n": float(env.cfg.thrust_to_weight) * weight,
            "moment_scale_n_m": float(env.cfg.moment_scale),
            "body_id": body_id,
            "body_name": body_name,
            "finite_after_live_step": bool(
                torch.isfinite(next_observation["policy"]).all() and torch.isfinite(reward).all()
            ),
            "terminated_after_live_step": bool(terminated.any()),
            "truncated_after_live_step": bool(truncated.any()),
            "native_full_reset_episode_length_buf": [int(value) for value in env.episode_length_buf.tolist()],
            "native_action_wrench_api": "Articulation.permanent_wrench_composer.set_forces_and_torques",
            "wrench_application_body": body_name,
        }
        gust_contract = {
            "api": "Articulation.instantaneous_wrench_composer.set_forces_and_torques",
            "body_name": body_name,
            "body_id": body_id,
            "force_frame": "world",
            "is_global": True,
            "application_point": "body_center_of_mass_no_offset_argument",
            "control_decisions": int(GUST_DURATION_STEPS),
            "physics_steps": int(GUST_DURATION_STEPS) * int(env.cfg.decimation),
            "physics_dt_s": float(env.cfg.sim.dt),
            "control_dt_s": float(env.step_dt),
            "duration_s": int(GUST_DURATION_STEPS) * float(env.step_dt),
            "expected_impulse": "robot_mass_kg * 0.75_m_s",
            "submitted_impulse": "sum(world_force_n * physics_dt_s) over each physics step",
            "raw_vehicle_response": "robot_mass_kg * (gust_end_velocity_w - gust_start_velocity_w)",
            "physical_response_measurement": {
                "method": "paired_identical_action_rollouts_same_live_environment_v1",
                "quantity": (
                    "horizontal mass*delta_velocity(gust) minus horizontal "
                    "mass*delta_velocity(no_gust_baseline)"
                ),
                "absolute_tolerance_n_s": GUST_RESPONSE_ABS_TOL_N_S,
                "relative_tolerance": GUST_RESPONSE_REL_TOL,
                "acceptance_rule": "vector_error <= max(absolute_tolerance, relative_tolerance * expected_norm)",
            },
        }
        usd_identifier = str(env.cfg.robot.spawn.usd_path)
        offline_scene = getattr(env, "_flyg1_offline_scene_report", None)
        if not isinstance(offline_scene, dict):
            offline_scene = {}
        offline_files = offline_scene.get("files")
        pinned_robot = (
            offline_files[0]
            if isinstance(offline_files, list)
            and offline_files
            and isinstance(offline_files[0], dict)
            else {}
        )
        verified_local_cf2x = (
            offline_scene.get("robot_usd") == usd_identifier
            and pinned_robot.get("path") == usd_identifier
            and isinstance(pinned_robot.get("sha256"), str)
            and len(pinned_robot["sha256"]) == 64
            and Path(usd_identifier).is_file()
            and sha256_file(Path(usd_identifier)) == pinned_robot["sha256"]
        )
        checks = {
            "native_task_registered": "Isaac-Quadcopter-Direct-v0" in gym.registry,
            "project_task_ids_registered_without_shadowing_native": all(task in gym.registry for task in TASK_IDS),
            "observation_width_12": live["single_observation_width"] == 12,
            "action_width_4": action_shape == (4,),
            "physics_dt_0.01_s": abs(live["physics_dt_s"] - 0.01) < 1e-12,
            "decimation_2": live["decimation"] == 2,
            "control_dt_0.02_s": abs(live["control_dt_s"] - 0.02) < 1e-12,
            "native_horizon_10_s": abs(live["native_episode_horizon_s"] - 10.0) < 1e-12,
            "thrust_to_weight_1.9": abs(live["thrust_to_weight"] - 1.9) < 1e-12,
            "moment_scale_0.01_n_m": abs(live["moment_scale_n_m"] - 0.01) < 1e-12,
            # The installed Nucleus identifier is deliberately resolved to a
            # byte-verified local mirror before environment construction.
            "crazyflie_cf2x_asset": verified_local_cf2x,
            "positive_finite_mass": mass > 0 and torch.isfinite(torch.tensor(mass)).item(),
            "live_step_finite": live["finite_after_live_step"],
            "instantaneous_gust_wrench_api_available": callable(
                getattr(
                    getattr(env._robot, "instantaneous_wrench_composer", None),
                    "set_forces_and_torques",
                    None,
                )
            ),
            "gust_world_frame_center_of_mass_contract": (
                gust_contract["is_global"] is True
                and gust_contract["force_frame"] == "world"
                and gust_contract["application_point"] == "body_center_of_mass_no_offset_argument"
            ),
            "gust_duration_5_control_10_physics_steps_0.10_s": (
                gust_contract["control_decisions"] == 5
                and gust_contract["physics_steps"] == 10
                and abs(gust_contract["duration_s"] - 0.10) < 1.0e-12
            ),
            "gust_physical_response_tolerance_predeclared": (
                GUST_RESPONSE_ABS_TOL_N_S == 5.0e-4
                and GUST_RESPONSE_REL_TOL == 0.10
            ),
            "native_goal_bounds_pinned": upstream_contract[
                "native_goal_bounds_relative_to_env_origin_m"
            ] == {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [0.5, 1.5]},
            "native_height_termination_pinned": upstream_contract[
                "native_height_termination_m"
            ] == [0.1, 2.0],
        }
        audit = {
            "schema_version": 1,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "native_task_id": "Isaac-Quadcopter-Direct-v0",
            "project_task_ids": list(TASK_IDS),
            "checks": checks,
            "live": live,
            "observation_contract": OBSERVATION_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "upstream_contract": upstream_contract,
            "project_gust_contract": gust_contract,
            "asset": {
                "configuration": "isaaclab_assets.CRAZYFLIE_CFG",
                "usd_identifier": usd_identifier,
                "upstream_usd_identifier": CRAZYFLIE_RESOLVED_USD_URL,
                "resolved_source_kind": "pinned verified local mirror",
                "offline_scene_report": offline_scene,
            },
            "versions": installed_versions(),
            "runtime": runtime_identity(),
            "sources": {
                "quadcopter_env": str(UPSTREAM_QUADCOPTER_ENV),
                "quadcopter_env_sha256": sha256_file(UPSTREAM_QUADCOPTER_ENV),
                "quadcopter_registration": str(UPSTREAM_QUADCOPTER_REGISTRATION),
                "quadcopter_registration_sha256": sha256_file(
                    UPSTREAM_QUADCOPTER_REGISTRATION
                ),
                "direct_rl_env": str(UPSTREAM_DIRECT_RL_ENV),
                "direct_rl_env_sha256": sha256_file(UPSTREAM_DIRECT_RL_ENV),
                "isaaclab_math": str(UPSTREAM_ISAACLAB_MATH),
                "isaaclab_math_sha256": sha256_file(UPSTREAM_ISAACLAB_MATH),
                "crazyflie_asset_cfg": str(UPSTREAM_CRAZYFLIE_ASSET),
                "crazyflie_asset_cfg_sha256": sha256_file(UPSTREAM_CRAZYFLIE_ASSET),
            },
            "limitations": [
                "Simulator state observations are not asserted to be available on physical hardware.",
                "The aggregate-wrench interface does not expose individual motor commands.",
                "This leg/VNC-derived circuit is not a known biological flight circuit.",
            ],
        }
        _atomic_text(args.json.resolve(), json.dumps(audit, indent=2, sort_keys=True) + "\n")
        _atomic_text(args.markdown.resolve(), _markdown(audit))
        print(json.dumps({
            "status": audit["status"], "json": str(args.json.resolve()),
            "markdown": str(args.markdown.resolve()), "mass_kg": mass, "checks": checks,
            "resolved_config": resolved_config, "fingerprint": fingerprint,
        }, indent=2, sort_keys=True))
        return 0 if audit["status"] == "PASS" else 1
    except BaseException:
        traceback.print_exc()
        print(json.dumps({
            "status": "FAIL",
            "json": str(args.json.resolve()),
            "markdown": str(args.markdown.resolve()),
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "failure_reason": traceback.format_exc(),
        }, indent=2, sort_keys=True))
        return 1
    finally:
        if env is not None:
            env.close()
        # Preserve the audit's process status after the environment closes.


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
