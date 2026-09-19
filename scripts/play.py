#!/usr/bin/env python3
"""Load a functional checkpoint and execute deterministic actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--connectome_manifest")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["policy"].startswith("frozen_lif") and not args.connectome_manifest:
        parser.error("A frozen-LIF checkpoint needs --connectome_manifest to reconstruct its verified core.")
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment
        from train import _make_policy
        from g1_fly_control.training import load_checkpoint
        from g1_fly_control.training.runner import policy_observation

        env = launch_environment(args.task, args.num_envs)
        policy, _ = _make_policy(metadata["policy"], metadata["observation_dim"], metadata["action_dim"], args.connectome_manifest, env.device, rewire_seed=metadata.get("rewire_seed") or 0)
        load_checkpoint(args.checkpoint, policy=policy, map_location=env.device)
        observation, _ = env.reset()
        observation = policy_observation(observation)
        state = policy.initial_state(args.num_envs, device=env.device) if hasattr(policy, "initial_state") else None
        total_reward = torch.zeros(args.num_envs, device=env.device)
        with torch.no_grad():
            for _ in range(args.steps):
                output = policy.act(observation, state, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(output.action)
                observation = policy_observation(observation)
                if output.state is not None:
                    from g1_fly_control.training.runner import RecurrentPPO
                    state = RecurrentPPO._state_reset(output.state, terminated | truncated)
                total_reward += reward
        print(json.dumps({"status": "PASS", "steps": args.steps, "mean_return": float(total_reward.mean())}, indent=2))
        env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
