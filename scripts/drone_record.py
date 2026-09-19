#!/usr/bin/env python3
"""Record one immutable held-out Crazyflie episode in bounded tensor chunks.

The recorder is deliberately single-environment and at most 600 decisions.
It is a trace/debug companion; aggregate comparison metrics still come from
``drone_evaluate.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from drone_bootstrap import launch_environment, sha256_file
from drone_evaluation_protocol import SCENARIOS, load_protocol
from drone_play import _atomic_json, inspect_checkpoint, load_policy_and_normalizer


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _controller_state_frame(state: Any, *, include_full: bool) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if state is None:
        result["controller_state_l2"] = torch.tensor(0.0)
        return result
    if isinstance(state, torch.Tensor):
        selected = state[0].detach().cpu()
        result["controller_state_l2"] = torch.linalg.vector_norm(selected.float())
        if include_full:
            result["gru_hidden_state"] = selected
        return result
    membrane = getattr(state, "membrane", None)
    synapse = getattr(state, "synapse", None)
    spikes = getattr(state, "spikes", None)
    refractory = getattr(state, "refractory", None)
    if not all(isinstance(value, torch.Tensor) for value in (membrane, synapse, spikes, refractory)):
        raise TypeError(f"Unsupported controller state type: {type(state).__name__}")
    membrane_row = membrane[0].detach().cpu()
    synapse_row = synapse[0].detach().cpu()
    spikes_row = spikes[0].detach().cpu()
    refractory_row = refractory[0].detach().cpu()
    result.update(
        {
            "controller_state_l2": torch.sqrt(
                membrane_row.float().square().sum() + synapse_row.float().square().sum()
            ),
            "lif_membrane_l2": torch.linalg.vector_norm(membrane_row.float()),
            "lif_synapse_l2": torch.linalg.vector_norm(synapse_row.float()),
            "lif_spike_fraction": spikes_row.float().mean(),
            "lif_refractory_fraction": (refractory_row > 0).float().mean(),
        }
    )
    if include_full:
        result.update(
            {
                "lif_membrane": membrane_row,
                "lif_synapse": synapse_row,
                "lif_spikes": spikes_row,
                "lif_refractory": refractory_row,
            }
        )
    return result


def _stack_frames(frames: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not frames:
        raise ValueError("cannot write an empty trace chunk")
    expected = tuple(frames[0])
    if any(tuple(frame) != expected for frame in frames[1:]):
        raise RuntimeError("trace fields changed within a chunk")
    return {key: torch.stack([frame[key] for frame in frames]) for key in expected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Enforce required values after AppLauncher's pre-parse so --help remains
    # usable without supplying execution inputs.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", choices=SCENARIOS)
    parser.add_argument(
        "--protocol",
        default="integration",
        help="integration, main, or a path to a validated evaluation manifest",
    )
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--chunk_steps", type=int, default=100)
    parser.add_argument("--include_neural_state", action="store_true")
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--output_dir", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.task is None or args.output_dir is None:
        parser.error("--checkpoint, --task, and --output_dir are required")
    if not 1 <= args.steps <= 600:
        parser.error("--steps must be between 1 and the 600-decision episode horizon")
    if not 1 <= args.chunk_steps <= args.steps:
        parser.error("--chunk_steps must be positive and no larger than --steps")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            parser.error(f"Recording output is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            parser.error(f"Refusing to overwrite non-empty recording directory: {output_dir}")
    try:
        protocol = load_protocol(args.protocol)
        plans = protocol["scenarios"][args.task]
        if not 0 <= args.episode_id < len(plans):
            raise ValueError(
                f"--episode_id must be in [0, {len(plans) - 1}] for protocol {args.protocol!r}"
            )
        payload, identity = inspect_checkpoint(checkpoint, args.expected_fingerprint)
        if payload.get("evaluation_manifest_id") != protocol["manifest_id"]:
            raise ValueError(
                "Checkpoint evaluation manifest does not match the requested recording protocol"
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    plan = plans[args.episode_id]
    resolved = {
        "kind": "heldout_trace_recording",
        "task": args.task,
        "protocol": args.protocol,
        "evaluation_manifest_id": protocol["manifest_id"],
        "evaluation_seed": protocol["evaluation_seed"],
        "episode_id": args.episode_id,
        "plan_sha256": plan["plan_sha256"],
        "requested_steps": args.steps,
        "chunk_steps": args.chunk_steps,
        "include_neural_state": args.include_neural_state,
        "deterministic_actions": True,
        **identity,
    }
    print(json.dumps({"resolved_config": resolved, "fingerprint": identity["fingerprint"]}, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    chunks: list[dict[str, Any]] = []
    app = AppLauncher(args).app
    env = None
    wrapped = None
    executed_steps = 0
    episode_finished = False
    try:
        from drone_evaluate import _root_state_and_plans
        from g1_fly_control.crazyflie.controllers import reset_controller_state
        from g1_fly_control.crazyflie.normalization import NormalizedEnv

        env = launch_environment(
            args.task,
            1,
            deterministic_evaluation=True,
            contract_profile=identity["contract_profile"],
        )
        device = torch.device(env.device)
        policy, normalizer, controller_report = load_policy_and_normalizer(
            checkpoint, payload, identity, device
        )
        wrapped = NormalizedEnv(env, normalizer, training=False)
        roots, targets, gusts = _root_state_and_plans(env, [plan])
        env.set_episode_plan(
            initial_root_state_w=roots,
            targets_w=targets,
            gust_directions_w=(
                gusts if args.task == "FlyCrazyflie-GustRecovery-v0" else None
            ),
        )
        observation, _ = wrapped.reset(seed=protocol["evaluation_seed"] + args.episode_id)
        if int(env.episode_length_buf[0]) != 0:
            raise RuntimeError("Held-out recording did not begin at episode step zero")
        policy_observation = observation["policy"]
        state = policy.initial_state(1, device=device) if hasattr(policy, "initial_state") else None
        frames: list[dict[str, torch.Tensor]] = []

        def flush_chunk() -> None:
            nonlocal frames
            if not frames:
                return
            index = len(chunks)
            start = executed_steps - len(frames)
            stop = executed_steps
            path = output_dir / f"chunk-{index:04d}-steps-{start:04d}-{stop:04d}.pt"
            data = _stack_frames(frames)
            _atomic_torch_save(
                path,
                {
                    "schema_version": 1,
                    "chunk_index": index,
                    "start_decision": start,
                    "stop_decision_exclusive": stop,
                    "fingerprint": identity["fingerprint"],
                    "plan_sha256": plan["plan_sha256"],
                    "trace": data,
                },
            )
            chunks.append(
                {
                    "chunk_index": index,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "start_decision": start,
                    "stop_decision_exclusive": stop,
                    "step_count": len(frames),
                    "fields": list(data),
                    "bytes": path.stat().st_size,
                }
            )
            frames = []

        with torch.no_grad():
            for decision in range(args.steps):
                episode_step_before = env.episode_length_buf[0].detach().cpu().clone()
                active_goal_before = env._desired_pos_w[0].detach().cpu().clone()
                output_policy = policy.act(policy_observation, state, deterministic=True)
                if output_policy.action.shape != (1, 4) or not torch.isfinite(
                    output_policy.action
                ).all():
                    raise FloatingPointError("Controller emitted a nonfinite or incorrectly shaped action")
                next_observation, reward, terminated, truncated, _ = wrapped.step(
                    output_policy.action
                )
                done = terminated | truncated
                raw_observation_after = torch.where(
                    done[:, None],
                    env.drone_terminal_observation,
                    env._compute_policy_observation(),
                )
                root_position_after = torch.where(
                    done[:, None], env.terminal_position_w, env._robot.data.root_pos_w
                )
                active_goal_after = torch.where(
                    done[:, None], env.terminal_goal_w, env._desired_pos_w
                )
                distance_after = torch.where(done, env.terminal_distance_m, env._distance)
                speed_after = torch.where(done, env.terminal_speed_mps, env._speed)
                success_count_after = torch.where(
                    done, env.terminal_success_count, env.success_count
                )
                switch_count_after = torch.where(
                    done, env.terminal_switch_count, env.switch_count
                )
                gust_count_after = torch.where(done, env.terminal_gust_count, env.gust_count)
                work_after = torch.where(
                    done, env.terminal_mechanical_work_proxy, env.mechanical_work_proxy
                )
                frame = {
                    "decision_index": torch.tensor(decision, dtype=torch.long),
                    "episode_step_before": episode_step_before,
                    "normalized_observation_before": policy_observation[0].detach().cpu(),
                    "normalized_action": output_policy.action[0].detach().cpu(),
                    "policy_mean": output_policy.mean[0].detach().cpu(),
                    "value": output_policy.value[0].detach().cpu(),
                    "reward": reward[0].detach().cpu(),
                    "terminated": terminated[0].detach().cpu(),
                    "truncated": truncated[0].detach().cpu(),
                    "raw_observation_after": raw_observation_after[0].detach().cpu(),
                    "root_position_world_m_after": root_position_after[0].detach().cpu(),
                    "active_goal_world_m_before": active_goal_before,
                    "active_goal_world_m_after": active_goal_after[0].detach().cpu(),
                    "distance_m_after": distance_after[0].detach().cpu(),
                    "speed_m_s_after": speed_after[0].detach().cpu(),
                    "success_count_after": success_count_after[0].detach().cpu(),
                    "switch_count_after": switch_count_after[0].detach().cpu(),
                    "gust_count_after": gust_count_after[0].detach().cpu(),
                    "aggregate_force_body_n": env.aggregate_force_b[0].detach().cpu(),
                    "aggregate_moment_body_n_m": env.aggregate_moment_b[0].detach().cpu(),
                    "gust_force_world_n": env.gust_vector_w[0].detach().cpu(),
                    "mechanical_work_proxy_j_after": work_after[0].detach().cpu(),
                    **_controller_state_frame(
                        output_policy.state, include_full=args.include_neural_state
                    ),
                }
                for name, value in env.reward_components.items():
                    frame[f"reward_component_{name}"] = value[0].detach().cpu()
                floating_values = (
                    value for value in frame.values() if value.is_floating_point()
                )
                if any(not bool(torch.isfinite(value).all()) for value in floating_values):
                    raise FloatingPointError("Recording encountered a nonfinite tensor")
                frames.append(frame)
                executed_steps += 1
                state = reset_controller_state(policy, output_policy.state, done)
                policy_observation = next_observation["policy"]
                if len(frames) >= args.chunk_steps or bool(done[0]):
                    flush_chunk()
                if bool(done[0]):
                    episode_finished = True
                    break
        flush_chunk()
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scientific_use": "bounded_trace_companion_not_aggregate_result",
            "resolved_config": resolved,
            "controller_report": controller_report,
            "episode_plan": plan,
            "executed_steps": executed_steps,
            "episode_finished": episode_finished,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scientific_use": manifest["scientific_use"],
                    "manifest": str(manifest_path),
                    "checkpoint": str(checkpoint),
                    "controller": identity["controller"],
                    "task": args.task,
                    "protocol": args.protocol,
                    "episode_id": args.episode_id,
                    "executed_steps": executed_steps,
                    "episode_finished": episode_finished,
                    "chunk_count": len(chunks),
                    "fingerprint": identity["fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        error = traceback.format_exc()
        traceback.print_exc()
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "failed",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "resolved_config": resolved,
                "executed_steps": executed_steps,
                "episode_finished": episode_finished,
                "chunks": chunks,
                "error": error,
            },
        )
        return 1
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        # Preserve the recorder's process status after the environment closes.


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
