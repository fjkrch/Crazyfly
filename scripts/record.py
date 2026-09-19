#!/usr/bin/env python3
"""Record synchronized robot/control/circuit arrays for selected evaluation episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--connectome_manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "recording.npz")
    parser.add_argument("--video", type=Path, help="Optional MP4 aligned one frame per control step.")
    parser.add_argument("--all_neurons", action="store_true", help="Record all neurons; default limits control-rate traces to 128.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required.")
    if args.video is not None:
        args.enable_cameras = True
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["policy"].startswith("frozen_lif") and not args.connectome_manifest:
        parser.error("A frozen-LIF checkpoint needs --connectome_manifest.")
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment
        from train import _make_policy
        from g1_fly_control.evaluation import EpisodeRecorder
        from g1_fly_control.training import load_checkpoint
        from g1_fly_control.training.runner import policy_observation

        env = launch_environment(args.task, 1, render_mode="rgb_array" if args.video else None)
        policy, circuit = _make_policy(metadata["policy"], metadata["observation_dim"], metadata["action_dim"], args.connectome_manifest, env.device, rewire_seed=metadata.get("rewire_seed") or 0)
        load_checkpoint(args.checkpoint, policy=policy, map_location=env.device)
        observation, _ = env.reset()
        observation = policy_observation(observation)
        state = policy.initial_state(1, device=env.device) if hasattr(policy, "initial_state") else None
        neuron_ids = list(circuit.neuron_ids) if circuit else []
        selected_neurons = None if args.all_neurons else list(range(min(128, len(neuron_ids))))
        reward_term_names = list(env.reward_manager._term_names)
        recorder = EpisodeRecorder({"task": args.task, "checkpoint": str(args.checkpoint), "seed": metadata.get("seed"), "control_dt": env.step_dt, "circuit_checksum": metadata.get("connectome_checksum"), "reward_term_names": reward_term_names, "video_path": str(args.video) if args.video else None}, neuron_ids, selected_neurons=selected_neurons)
        writer = None
        if args.video:
            import imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=round(1.0 / env.step_dt))
        with torch.no_grad():
            for step in range(args.steps):
                actor_observation = observation
                output = policy.act(observation, state, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(output.action)
                observation = policy_observation(observation)
                robot = env.scene["robot"]
                goal_state = env._flyg1_goal_state
                signals = {"environment_id": torch.tensor(0), "episode_id": torch.tensor(0), "step_in_episode": env.episode_length_buf[0], "policy_observation": actor_observation[0], "root_state": robot.data.root_state_w[0], "joint_position": robot.data.joint_pos[0], "joint_velocity": robot.data.joint_vel[0], "action": output.action[0], "applied_torque": robot.data.applied_torque[0], "mechanical_work_proxy": env._flyg1_interval_work[0], "contact_net_forces_w": env.scene["contact_forces"].data.net_forces_w[0], "goal_xy_w": goal_state.goal_xy_w[0], "goal_id": goal_state.goal_id[0], "goal_switch_count": goal_state.switch_count[0], "reward": reward[0], "reward_term_rates": env.reward_manager._step_reward[0], "push_impulse_w": getattr(env, "_flyg1_push_impulse", torch.zeros(1, 3, device=env.device))[0]}
                if hasattr(output.state, "membrane"):
                    signals.update({"encoded_input_current": policy.encoder(actor_observation)[0], "neural_voltage": output.state.membrane[0], "neural_spikes": output.state.spikes[0], "neural_filtered_rate": output.state.synapse[0]})
                recorder.append(step * env.step_dt, **signals)
                if writer is not None:
                    writer.append_data(env.render())
                if bool((terminated | truncated).any()):
                    break
                if output.state is not None:
                    from g1_fly_control.training.runner import RecurrentPPO
                    state = RecurrentPPO._state_reset(output.state, terminated | truncated)
        result = recorder.save(args.output)
        if writer is not None:
            writer.close()
        print(result)
        env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
