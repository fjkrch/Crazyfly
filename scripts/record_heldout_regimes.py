#!/usr/bin/env python3
"""Run a checked heldout_v1 companion replay and record pose/contact traces.

The NPZ is a companion to an existing 16-episode evaluation JSON. Every sample
is taken before a policy action, so Isaac's automatic reset cannot replace a
terminal pose with the next episode's initial pose. There is no terminal
post-action sample. ``valid_step`` excludes all samples after each environment's
first termination/truncation. Matching initial states, event schedules, outcomes,
and total steps does not prove byte-identical trajectories.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from _bootstrap import ROOT
from evaluation_protocol import HeldOutEvents
from run_matrix import _execution_source_fingerprint


EPISODES = 16
EVALUATION_SEED = 101
MAX_CONTROL_STEPS = 1010
CONDITIONS = {
    "frozen_lif": "frozen_lif_original",
    "frozen_lif_rewired": "frozen_lif_degree_rewired",
    "gru": "gru_trainable",
    "mlp": "mlp_engineering_baseline",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_state_sha256(robot, goal_state, index: int) -> str:
    """Use precisely the rounding and JSON encoding used by evaluate.py."""
    initial = {
        "root_state_w": robot.data.root_state_w[index].tolist(),
        "joint_pos": robot.data.joint_pos[index].tolist(),
        "joint_vel": robot.data.joint_vel[index].tolist(),
        "goal_xy_w": goal_state.goal_xy_w[index].tolist(),
    }
    rounded = {key: [round(float(value), 6) for value in values] for key, values in initial.items()}
    return sha256(json.dumps(rounded, sort_keys=True).encode()).hexdigest()


def _paired_plan_sha256(schedule: HeldOutEvents, index: int) -> str:
    return sha256(json.dumps({
        "scenario_schedule_sha256": schedule.manifest["sha256"],
        "environment_plan": schedule.manifest["plan"][index],
    }, sort_keys=True).encode()).hexdigest()


def _reference(path: Path, checkpoint: Path, checkpoint_metadata: dict) -> tuple[dict, HeldOutEvents]:
    """Reject a reference that cannot identify the exact held-out replay."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "executed":
        raise ValueError("--evaluation_json must contain an executed evaluation.")
    scenario = data.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("evaluation_protocol") != "heldout_v1":
        raise ValueError("--evaluation_json must use heldout_v1.")
    if scenario.get("policy_action_mode") != "deterministic":
        raise ValueError("--evaluation_json must record deterministic policy actions.")
    task = data.get("task")
    if task not in {
        "FlyG1-GoalReach-FreePosture-v0",
        "FlyG1-GoalSwitch-FreePosture-v0",
        "FlyG1-PushRecovery-FreePosture-v0",
    } or scenario.get("task") != task:
        raise ValueError("--evaluation_json has an invalid or inconsistent G1 task.")
    if checkpoint_metadata.get("task") != "FlyG1-GoalReach-FreePosture-v0":
        raise ValueError("The checkpoint must be trained on the G1 GoalReach task.")
    if checkpoint_metadata.get("policy") not in CONDITIONS:
        raise ValueError("The checkpoint policy is not a matrix condition.")
    if type(checkpoint_metadata.get("seed")) is not int or data.get("training_seed") != checkpoint_metadata["seed"]:
        raise ValueError("Evaluation training seed differs from the checkpoint.")
    if type(data.get("evaluation_seed")) is not int or data["evaluation_seed"] != EVALUATION_SEED:
        raise ValueError(f"Evaluation seed must be {EVALUATION_SEED}.")
    if type(data.get("n_episodes")) is not int or data["n_episodes"] != EPISODES:
        raise ValueError(f"Evaluation must contain {EPISODES} episodes.")
    if data.get("ablation") is not None:
        raise ValueError("An ablated evaluation cannot be replayed by this recorder.")
    if not isinstance(data.get("checkpoint"), str) or Path(data["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Evaluation checkpoint path differs from --checkpoint.")
    if not isinstance(data.get("episodes"), list) or len(data["episodes"]) != EPISODES:
        raise ValueError("Evaluation episode rows are missing.")
    schedule = HeldOutEvents(EVALUATION_SEED, EPISODES, task)
    if json.dumps(scenario.get("schedule"), sort_keys=True) != json.dumps(schedule.manifest, sort_keys=True):
        raise ValueError("Evaluation held-out schedule differs from the current protocol.")
    for index, episode in enumerate(data["episodes"]):
        if not isinstance(episode, dict) or episode.get("episode_id") != index:
            raise ValueError(f"Evaluation episode {index} is missing or out of order.")
        if episode.get("seed") != checkpoint_metadata["seed"]:
            raise ValueError(f"Evaluation episode {index} has a different training seed.")
        if episode.get("paired_plan_sha256") != _paired_plan_sha256(schedule, index):
            raise ValueError(f"Evaluation episode {index} has a different paired plan hash.")
        if not isinstance(episode.get("initial_state_sha256"), str) or len(episode["initial_state_sha256"]) != 64:
            raise ValueError(f"Evaluation episode {index} lacks an initial-state hash.")
        if (type(episode.get("success")) is not bool or "time_to_target_s" not in episode
                or not isinstance(episode.get("schedule_events"), list)):
            raise ValueError(f"Evaluation episode {index} lacks outcome or event records.")
    execution = data.get("execution")
    if (not isinstance(execution, dict) or type(execution.get("control_steps")) is not int
            or not 0 < execution["control_steps"] <= MAX_CONTROL_STEPS):
        raise ValueError("Evaluation needs a bounded control-step count.")
    return data, schedule


def _same_values(actual, expected, *, tolerance: float = 1e-4) -> bool:
    if type(actual) in (int, float) and type(expected) in (int, float):
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=tolerance
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _same_values(a, b, tolerance=tolerance) for a, b in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_values(actual[key], expected[key], tolerance=tolerance) for key in actual
        )
    return type(actual) is type(expected) and actual == expected


def _scheduled_event(event: dict) -> dict:
    if event.get("kind") == "target":
        names = ("kind", "ordinal", "time_s", "relative_offset_xy_m")
    elif event.get("kind") == "push":
        names = ("kind", "ordinal", "time_s", "body", "frame", "force_world_n",
                 "duration_s", "impulse_world_n_s")
    else:
        raise ValueError(f"Unknown held-out event kind: {event.get('kind')!r}")
    if any(name not in event for name in names):
        raise ValueError("Held-out event is missing its controlled fields.")
    return {name: event[name] for name in names}


def _reference_checks(reference: dict, replay_events: list[list[dict]], replay_success: list[bool],
                      replay_target_times: list[float | None], control_steps: int) -> dict:
    """Compare companion-run outcomes/events; this cannot prove identical actions."""
    episodes = reference["episodes"]
    if not (len(replay_events) == len(replay_success) == len(replay_target_times) == len(episodes) == EPISODES):
        raise ValueError("Companion replay must contain 16 episode outcomes and event lists.")
    success_match = []
    target_time_match = []
    scheduled_event_match = []
    full_event_match = []
    for index, episode in enumerate(episodes):
        expected_events = episode["schedule_events"]
        actual_events = replay_events[index]
        success_match.append(replay_success[index] == episode["success"])
        target_time_match.append(_same_values(replay_target_times[index], episode["time_to_target_s"]))
        scheduled_event_match.append(_same_values(
            [_scheduled_event(event) for event in actual_events],
            [_scheduled_event(event) for event in expected_events],
        ))
        full_event_match.append(_same_values(actual_events, expected_events, tolerance=1e-3))
    steps_match = control_steps == reference["execution"]["control_steps"]
    required_match = (all(success_match) and all(target_time_match)
                      and all(scheduled_event_match) and steps_match)
    return {
        "required_match": required_match,
        "success_match_by_episode": success_match,
        "target_time_match_by_episode": target_time_match,
        "scheduled_event_match_by_episode": scheduled_event_match,
        "full_event_match_by_episode": full_event_match,
        "control_steps_match": steps_match,
        "replay_success_by_episode": replay_success,
        "replay_target_time_s_by_episode": replay_target_times,
        "replay_control_steps": control_steps,
    }


def _write_npz(path: Path, *, root_state: np.ndarray, forces: np.ndarray,
               valid: np.ndarray, body_names: list[str], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                root_state=root_state,
                contact_net_forces_w=forces,
                valid_step=valid,
                episode_id=np.arange(EPISODES, dtype=np.int32),
                body_names=np.asarray(body_names, dtype=np.str_),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output_path(output: Path, *inputs: Path | None) -> Path:
    destination = output.expanduser().resolve()
    if destination in {source for source in inputs if source is not None}:
        raise ValueError("Recording output must differ from the checkpoint, evaluation JSON, and connectome manifest.")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation_json", type=Path)
    parser.add_argument("--output", type=Path, help="Output first-episode NPZ recording.")
    parser.add_argument("--connectome_manifest", type=Path,
                        help="Real MaleCNS manifest; defaults to the checkpoint's recorded path.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.evaluation_json is None or args.output is None:
        parser.error("--checkpoint, --evaluation_json, and --output are required.")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    evaluation_json = args.evaluation_json.expanduser().resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_metadata = payload.get("metadata")
    if not isinstance(checkpoint_metadata, dict):
        parser.error("Checkpoint has no metadata.")
    reference, schedule = _reference(evaluation_json, checkpoint, checkpoint_metadata)
    manifest = args.connectome_manifest or checkpoint_metadata.get("connectome_manifest")
    if checkpoint_metadata["policy"].startswith("frozen_lif") and not manifest:
        parser.error("A frozen-LIF checkpoint needs --connectome_manifest.")
    manifest_path = Path(manifest).expanduser().resolve(strict=True) if manifest else None
    try:
        output_path = _safe_output_path(args.output, checkpoint, evaluation_json, manifest_path)
    except ValueError as exc:
        parser.error(str(exc))
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment
        from train import _make_policy
        from g1_fly_control.training import load_checkpoint
        from g1_fly_control.training.runner import RecurrentPPO, policy_observation

        env = launch_environment(reference["task"], EPISODES)
        policy, circuit = _make_policy(
            checkpoint_metadata["policy"], checkpoint_metadata["observation_dim"],
            checkpoint_metadata["action_dim"], str(manifest_path) if manifest_path else None,
            env.device, rewire_seed=checkpoint_metadata.get("rewire_seed") or 0,
        )
        if circuit is not None and circuit.checksum != checkpoint_metadata.get("connectome_checksum"):
            raise ValueError("Replay connectome differs from the checkpoint's source-ID mapping.")
        load_checkpoint(checkpoint, policy=policy, map_location=env.device)
        if circuit is not None and policy.core.frozen_checksum != checkpoint_metadata.get("frozen_core_checksum"):
            raise ValueError("Replay frozen circuit differs from the checkpoint's recorded core.")
        schedule.install(env)
        observation, _ = env.reset(seed=EVALUATION_SEED)
        observation = policy_observation(observation)
        state = policy.initial_state(EPISODES, device=env.device) if hasattr(policy, "initial_state") else None
        robot = env.scene["robot"]
        for index, episode in enumerate(reference["episodes"]):
            if _initial_state_sha256(robot, env._flyg1_goal_state, index) != episode["initial_state_sha256"]:
                raise ValueError(f"Replay initial state differs from evaluation for environment {index}.")
        sensor = env.scene["contact_forces"]
        body_names = list(sensor.body_names)
        if not body_names or len(set(body_names)) != len(body_names):
            raise ValueError("Contact sensor needs unique named bodies.")
        if tuple(sensor.data.net_forces_w.shape) != (EPISODES, len(body_names), 3):
            raise ValueError("Contact sensor data shape does not match its body names.")
        if tuple(robot.data.root_state_w.shape) != (EPISODES, 13):
            raise ValueError("Robot root-state data must have shape [16,13].")
        if math.ceil(env.cfg.episode_length_s / env.step_dt) + 10 > MAX_CONTROL_STEPS:
            raise ValueError("Task episode horizon exceeds the recorder's 1010-step bound.")
        if not math.isclose(float(env.step_dt), float(reference["scenario"]["control_dt_s"]), abs_tol=1e-9):
            raise ValueError("Replay control step differs from evaluation.")

        # Preallocate a strict bound. Invalid samples remain NaN, and arrays are
        # trimmed before saving. A done environment can auto-reset while others
        # continue; only its original episode contributes to these traces.
        root_state = np.full((MAX_CONTROL_STEPS, EPISODES, 13), np.nan, dtype=np.float32)
        forces = np.full((MAX_CONTROL_STEPS, EPISODES, len(body_names), 3), np.nan, dtype=np.float32)
        valid = np.zeros((MAX_CONTROL_STEPS, EPISODES), dtype=np.bool_)
        done = torch.zeros(EPISODES, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(done)
        elapsed = torch.zeros(EPISODES, device=env.device)
        target_times = torch.full((EPISODES,), float("nan"), device=env.device)
        control_steps = 0
        with torch.no_grad():
            while not bool(done.all()):
                if control_steps >= MAX_CONTROL_STEPS:
                    raise RuntimeError(f"First episodes did not all finish within {MAX_CONTROL_STEPS} control steps.")
                active = (~done).cpu().numpy()
                root_sample = env.scene["robot"].data.root_state_w.detach().cpu().numpy()
                force_sample = env.scene["contact_forces"].data.net_forces_w.detach().cpu().numpy()
                if not np.isfinite(root_sample[active]).all() or not np.isfinite(force_sample[active]).all():
                    raise RuntimeError(f"Nonfinite active root/contact sample at control step {control_steps}.")
                root_state[control_steps, active] = root_sample[active]
                forces[control_steps, active] = force_sample[active]
                valid[control_steps, active] = True
                output = policy.act(observation, state, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(output.action)
                observation = policy_observation(observation)
                active_tensor = ~done
                elapsed += active_tensor * env.step_dt
                step_success = env._flyg1_goal_state.success_latched | (
                    (terminated | truncated) & env.flyg1_terminal_success
                )
                newly_successful = step_success & ~success & active_tensor
                target_times[newly_successful] = elapsed[newly_successful]
                success |= step_success & active_tensor
                done |= (terminated | truncated) & active_tensor
                if output.state is not None:
                    state = RecurrentPPO._state_reset(output.state, terminated | truncated)
                control_steps += 1
                if control_steps % 50 == 0 or bool(done.all()):
                    print(json.dumps({"recording_control_steps": control_steps,
                                      "active_first_episodes": int((~done).sum())}), flush=True)

        code_files = [
            ROOT / "scripts" / name for name in (
                "record_heldout_regimes.py", "evaluate.py", "evaluation_protocol.py",
                "_bootstrap.py", "train.py",
            )
        ]
        metadata = {
            "schema_version": "heldout_regimes_v1",
            "sample_phase": "pre_action_before_first_episode_reset",
            "force_definition": "ContactSensor net_forces_w per named robot body, world frame; a net-force proxy, not a ground-pair contact label.",
            "root_state_definition": "Isaac root_state_w: world position xyz, quaternion wxyz, linear velocity xyz, angular velocity xyz.",
            "condition": CONDITIONS[checkpoint_metadata["policy"]],
            "scenario": reference["task"],
            "task": reference["task"],
            "training_seed": checkpoint_metadata["seed"],
            "evaluation_seed": EVALUATION_SEED,
            "episode_count": EPISODES,
            "control_dt_s": float(env.step_dt),
            "recorded_control_steps": control_steps,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "evaluation_json": str(evaluation_json),
            "evaluation_json_sha256": _file_sha256(evaluation_json),
            "connectome_manifest": str(manifest_path) if manifest_path else None,
            "connectome_manifest_sha256": _file_sha256(manifest_path) if manifest_path else None,
            "scenario_schedule_sha256": schedule.manifest["sha256"],
            "execution_source_fingerprint": _execution_source_fingerprint(),
            "code_sha256": {str(path.relative_to(ROOT)): _file_sha256(path) for path in code_files},
            "initial_state_sha256": [row["initial_state_sha256"] for row in reference["episodes"]],
            "paired_plan_sha256": [row["paired_plan_sha256"] for row in reference["episodes"]],
            "reference_control_steps": reference.get("execution", {}).get("control_steps"),
        }
        replay_success = [bool(value) for value in success.cpu().tolist()]
        replay_target_times = [None if math.isnan(value) else float(value)
                               for value in target_times.cpu().tolist()]
        checks = _reference_checks(reference, schedule.events, replay_success,
                                   replay_target_times, control_steps)
        metadata["reference_checks"] = checks
        metadata["replay_interpretation"] = (
            "Deterministic companion replay with matched initial states, held-out plan, and checked "
            "outcomes/events; no action hashes or bitwise trajectory equivalence are available."
        )
        trimmed_valid = valid[:control_steps]
        for index in range(EPISODES):
            count = int(trimmed_valid[:, index].sum())
            if count < 1 or not trimmed_valid[:count, index].all() or trimmed_valid[count:, index].any():
                raise RuntimeError(f"First-episode sample mask is not a true prefix for environment {index}.")
        _write_npz(output_path, root_state=root_state[:control_steps],
                   forces=forces[:control_steps], valid=trimmed_valid,
                   body_names=body_names, metadata=metadata)
        print(json.dumps({"status": "recorded" if checks["required_match"] else "reference_mismatch",
                          "output": str(output_path), "reference_checks": checks,
                          "control_steps": control_steps, "episodes": EPISODES,
                          "body_count": len(body_names)}, sort_keys=True), flush=True)
        env.close()
        if not checks["required_match"]:
            return 1
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
