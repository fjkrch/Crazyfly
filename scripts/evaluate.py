#!/usr/bin/env python3
"""Evaluate a checkpoint and emit per-episode records plus seed-level summary."""

from __future__ import annotations

import argparse
import faulthandler
from hashlib import sha256
import json
import math
from pathlib import Path
import signal
import time
import traceback

import torch

from _bootstrap import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--connectome_manifest")
    parser.add_argument("--protocol", choices=("default", "heldout_v1"), default="default")
    parser.add_argument("--progress_every", type=int, default=50,
                        help="Log evaluator control-step progress every N steps (default: 50).")
    parser.add_argument("--ablate_ids", type=Path, help="JSON list of source neuron IDs whose outgoing spikes are clamped.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation.json")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress_every must be positive.")
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.checkpoint is None:
        parser.error("--checkpoint is required.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["policy"].startswith("frozen_lif") and not args.connectome_manifest:
        parser.error("A frozen-LIF checkpoint needs --connectome_manifest.")
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment
        from train import _make_policy
        from g1_fly_control.evaluation import EpisodeSummary, summarize_episodes
        from g1_fly_control.evaluation.metrics import goal_relative_progress_step
        from g1_fly_control.training import load_checkpoint
        from g1_fly_control.training.runner import policy_observation

        env = launch_environment(args.task, args.episodes)
        policy, circuit = _make_policy(metadata["policy"], metadata["observation_dim"], metadata["action_dim"], args.connectome_manifest, env.device, rewire_seed=metadata.get("rewire_seed") or 0)
        if circuit is not None and circuit.checksum != metadata.get("connectome_checksum"):
            raise ValueError("Evaluation connectome differs from the checkpoint's source-ID mapping.")
        load_checkpoint(args.checkpoint, policy=policy, map_location=env.device)
        if circuit is not None and policy.core.frozen_checksum != metadata.get("frozen_core_checksum"):
            raise ValueError("Loaded frozen circuit differs from the checkpoint's recorded core.")
        ablated_ids = None
        ablation_mask = None
        if args.ablate_ids is not None:
            if circuit is None:
                parser.error("--ablate_ids applies only to frozen-LIF policies.")
            ablated_ids = json.loads(args.ablate_ids.read_text(encoding="utf-8"))
            if not isinstance(ablated_ids, list) or any(not isinstance(item, str) for item in ablated_ids):
                parser.error("--ablate_ids must be a JSON list of source neuron ID strings.")
            if len(set(ablated_ids)) != len(ablated_ids):
                parser.error("--ablate_ids contains duplicate source neuron IDs.")
            index_of = {neuron_id: i for i, neuron_id in enumerate(circuit.neuron_ids)}
            unknown = sorted(set(ablated_ids) - index_of.keys())
            if unknown:
                parser.error(f"--ablate_ids contains IDs absent from this circuit: {unknown[:8]}")
            ablation_mask = torch.zeros(circuit.num_neurons, dtype=torch.bool, device=env.device)
            for neuron_id in ablated_ids:
                ablation_mask[index_of[neuron_id]] = True
        schedule = None
        if args.protocol == "heldout_v1":
            from evaluation_protocol import HeldOutEvents
            schedule = HeldOutEvents(args.seed, args.episodes, args.task)
            schedule.install(env)
        observation, _ = env.reset(seed=args.seed)
        observation = policy_observation(observation)
        state = policy.initial_state(args.episodes, device=env.device) if hasattr(policy, "initial_state") else None
        task_state = env._flyg1_goal_state
        initial_state_fingerprints = []
        if schedule is not None:
            robot = env.scene["robot"]
            for index in range(args.episodes):
                initial = {
                    "root_state_w": robot.data.root_state_w[index].tolist(),
                    "joint_pos": robot.data.joint_pos[index].tolist(),
                    "joint_vel": robot.data.joint_vel[index].tolist(),
                    "goal_xy_w": task_state.goal_xy_w[index].tolist(),
                }
                rounded = {key: [round(float(value), 6) for value in values] for key, values in initial.items()}
                initial_state_fingerprints.append(sha256(json.dumps(rounded, sort_keys=True).encode()).hexdigest())
        initial_distance = task_state.previous_distance.clone()
        final_distance = initial_distance.clone()
        accumulated_goal_relative_progress = torch.zeros(args.episodes, device=env.device)
        goal_switches = torch.zeros(args.episodes, dtype=torch.long, device=env.device)
        pushes = torch.zeros_like(goal_switches)
        push_impulse_vector = torch.zeros(args.episodes, 3, device=env.device)
        push_impulse_magnitude = torch.zeros(args.episodes, device=env.device)
        work = torch.zeros(args.episodes, device=env.device)
        impact = torch.zeros_like(work)
        limit_steps = torch.zeros_like(work)
        saturation_steps = torch.zeros_like(work)
        done = torch.zeros(args.episodes, dtype=torch.bool, device=env.device)
        elapsed = torch.zeros_like(work)
        success = torch.zeros_like(done)
        control_steps = 0
        maximum_control_steps = math.ceil(env.cfg.episode_length_s / env.step_dt) + 10
        evaluation_start = time.monotonic()

        def log_progress(phase: str) -> None:
            print(json.dumps({
                "evaluation_progress": phase,
                "control_steps": control_steps,
                "elapsed_wall_s": round(time.monotonic() - evaluation_start, 3),
                "active_episodes": int((~done).sum()),
                "max_episode_length_buf": int(env.episode_length_buf.max()),
                "max_switch_count": int(env._flyg1_goal_state.switch_count.max()),
            }, sort_keys=True), flush=True)

        log_progress("after_reset")
        time_to_target = torch.full_like(work, float("nan"))
        recovery_start = torch.full_like(work, float("nan"))
        recovery_attempts = torch.zeros_like(goal_switches)
        recovery_completions = torch.zeros_like(goal_switches)
        recovery_time_sum = torch.zeros_like(work)
        with torch.no_grad():
            while not bool(done.all()):
                if control_steps >= maximum_control_steps:
                    log_progress("episode_length_bound_exceeded")
                    raise RuntimeError(
                        f"Evaluation exceeded {maximum_control_steps} control steps without all episodes ending."
                    )
                if (control_steps + 1) % args.progress_every == 0:
                    log_progress("before_step")
                robot = env.scene["robot"]
                task_state = env._flyg1_goal_state
                before_goal_xy = task_state.goal_xy_w.clone()
                before_distance = torch.linalg.vector_norm(before_goal_xy - robot.data.root_pos_w[:, :2], dim=-1)
                before_switch_count = task_state.switch_count.clone()
                before_push_count = getattr(env, "_flyg1_push_count", torch.zeros_like(pushes)).clone()
                if ablation_mask is None:
                    output = policy.act(observation, state, deterministic=True)
                else:
                    output = policy.act(observation, state, deterministic=True, ablate_outgoing=ablation_mask)
                observation, _, terminated, truncated, extras = env.step(output.action)
                control_steps += 1
                if control_steps % args.progress_every == 0:
                    log_progress("after_step")
                observation = policy_observation(observation)
                robot = env.scene["robot"]
                active = ~done
                work += env._flyg1_interval_work * active
                joint_ids = env.action_manager.get_term("joint_pos")._joint_ids
                limits = robot.data.soft_joint_pos_limits[:, joint_ids]
                joint_pos = robot.data.joint_pos[:, joint_ids]
                limit_steps += ((joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])).any(dim=-1) * active
                effort_limits = robot.data.joint_effort_limits[:, joint_ids]
                saturation_steps += (robot.data.applied_torque[:, joint_ids].abs() >= effort_limits * 0.99).any(dim=-1) * active
                sensor = env.scene["contact_forces"]
                peak_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, 0], dim=-1)
                impact += (peak_force - 350.0).clamp_min(0.0).sum(dim=-1) * env.step_dt * active
                elapsed += active * env.step_dt
                task_state = env._flyg1_goal_state
                new_switches = (task_state.switch_count - before_switch_count).clamp_min(0) * active
                goal_switches += new_switches
                after_push_count = getattr(env, "_flyg1_push_count", torch.zeros_like(pushes))
                new_pushes = (after_push_count - before_push_count).clamp_min(0) * active
                pushes += new_pushes
                if hasattr(env, "_flyg1_push_impulse"):
                    impulse = env._flyg1_push_impulse * new_pushes[:, None]
                    push_impulse_vector += impulse
                    push_impulse_magnitude += torch.linalg.vector_norm(impulse, dim=-1)
                step_success = task_state.success_latched | ((terminated | truncated) & env.flyg1_terminal_success)
                newly_successful = step_success & ~success & active
                recovery_window = (elapsed >= recovery_start) & (elapsed <= recovery_start + 3.0)
                recovered_now = newly_successful & recovery_window & (recovery_attempts > recovery_completions)
                recovery_completions += recovered_now.long()
                recovery_time_sum += torch.where(recovered_now, elapsed - recovery_start, 0.0)
                # A push starts after the current physics/reward step. Count only
                # goals that had not already succeeded when that push was applied.
                eligible_push = (new_pushes > 0) & ~step_success & active
                recovery_attempts += eligible_push.long()
                recovery_start = torch.where(eligible_push, elapsed + 0.10, recovery_start)
                time_to_target[newly_successful] = elapsed[newly_successful]
                success |= step_success & active
                newly_done = (terminated | truncated) & active
                current_root_xy = robot.data.root_pos_w[:, :2]
                terminal_root_xy = getattr(env, "flyg1_terminal_root_xy", current_root_xy)
                after_root_xy = torch.where(newly_done[:, None], terminal_root_xy, current_root_xy)
                accumulated_goal_relative_progress += goal_relative_progress_step(
                    before_distance, before_goal_xy, after_root_xy, active, new_switches > 0
                )
                if bool(newly_done.any()):
                    terminal_distance = torch.linalg.vector_norm(env.flyg1_terminal_goal_xy - env.flyg1_terminal_root_xy, dim=-1)
                    final_distance[newly_done] = terminal_distance[newly_done]
                continuing = active & ~newly_done
                final_distance[continuing] = task_state.previous_distance[continuing]
                done |= newly_done
                if output.state is not None:
                    from g1_fly_control.training.runner import RecurrentPPO
                    state = RecurrentPPO._state_reset(output.state, terminated | truncated)
        records = [EpisodeSummary(
            seed=int(metadata["seed"]), episode_id=index, success=bool(success[index]),
            xy_progress_m=(
                None if args.task == "FlyG1-GoalSwitch-FreePosture-v0"
                else float(initial_distance[index] - final_distance[index])
            ),
            accumulated_goal_relative_progress_m=float(accumulated_goal_relative_progress[index]),
            time_to_target_s=float(time_to_target[index]) if bool(success[index]) else None,
            mechanical_work_proxy=float(work[index]), excessive_impact=float(impact[index]),
            joint_limit_frequency=float(limit_steps[index] / (elapsed[index] / env.step_dt).clamp_min(1)),
            saturation_frequency=float(saturation_steps[index] / (elapsed[index] / env.step_dt).clamp_min(1)),
            goal_switch_count=int(goal_switches[index]), push_count=int(pushes[index]),
            push_impulse_vector_n_s=tuple(float(component) for component in push_impulse_vector[index]),
            push_impulse_magnitude_n_s=float(push_impulse_magnitude[index]),
            recovery_success=(
                bool(recovery_completions[index] > 0)
                if int(recovery_attempts[index]) > 0 else None
            ),
            recovery_time_s=(
                float(recovery_time_sum[index] / recovery_completions[index])
                if int(recovery_completions[index]) > 0 else None
            ),
            recovery_attempt_count=int(recovery_attempts[index]),
            recovery_success_count=int(recovery_completions[index]),
        ) for index in range(args.episodes)]
        result = summarize_episodes(records)
        if schedule is not None:
            for index, row in enumerate(result["episodes"]):
                row["schedule_events"] = schedule.events[index]
                row["initial_state_sha256"] = initial_state_fingerprints[index]
                row["paired_plan_sha256"] = sha256(json.dumps({
                    "scenario_schedule_sha256": schedule.manifest["sha256"],
                    "environment_plan": schedule.manifest["plan"][index],
                }, sort_keys=True).encode()).hexdigest()
        result.update({
            "status": "executed", "checkpoint": str(args.checkpoint), "task": args.task,
            "execution": {"control_steps": control_steps,
                          "evaluation_wall_time_s": time.monotonic() - evaluation_start,
                          "maximum_control_steps": maximum_control_steps},
            "training_seed": int(metadata["seed"]), "evaluation_seed": args.seed,
            "scenario": {
                "task": args.task,
                "reset_regime": "default_standing_pose",
                "reset_regime_collision_verified": False,
                "held_out_targets_verified": schedule is not None,
                "paired_disturbance_schedule": schedule is not None and args.task == "FlyG1-PushRecovery-FreePosture-v0",
                "evaluation_protocol": args.protocol,
                "schedule": schedule.manifest if schedule is not None else None,
                "recovery_definition": "Each push before first target success is an attempt. A new sustained target success within 3.0 s after that push's 0.10 s force ends completes the attempt. Seed-level success rate is completed attempts divided by eligible attempts; episode recovery_success means at least one completion.",
                "policy_action_mode": "deterministic",
                "goal_switch_enabled": args.task == "FlyG1-GoalSwitch-FreePosture-v0",
                "push_enabled": args.task == "FlyG1-PushRecovery-FreePosture-v0",
                "control_dt_s": env.step_dt,
                "episode_horizon_s": env.cfg.episode_length_s,
            },
            "ablation": (
                {"mode": "outgoing_spikes_clamped_zero", "source_neuron_ids": ablated_ids}
                if ablated_ids is not None else None
            ),
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        env.close()
    except BaseException:
        # Isaac teardown can itself stall after a simulator or event exception.
        # Emit the original failure before entering app.close().
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
