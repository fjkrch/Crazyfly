#!/usr/bin/env python3
"""Train the MLP baseline or frozen-LIF actor with the same free-posture task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import traceback

import torch

from _bootstrap import ROOT
from memory_watch import reset_training_cuda_peak, training_memory_snapshot


def resolve_training_budget(
    *, num_envs: int, horizon: int, max_iterations: int, interaction_budget: int | None
) -> tuple[int, int]:
    """Return PPO updates and actual environment steps, rounding a requested budget up.

    Each update collects a complete ``num_envs * horizon`` rollout. A budget
    smaller than one rollout therefore still executes one full rollout.
    """
    if num_envs < 1 or horizon < 1 or max_iterations < 1:
        raise ValueError("--num_envs, --horizon, and --max_iterations must be positive.")
    if interaction_budget is not None and interaction_budget < 1:
        raise ValueError("--interaction_budget must be positive.")
    steps_per_iteration = num_envs * horizon
    iterations = (
        (interaction_budget + steps_per_iteration - 1) // steps_per_iteration
        if interaction_budget is not None
        else max_iterations
    )
    return iterations, iterations * steps_per_iteration


def policy_parameter_counts(policy: torch.nn.Module) -> dict[str, int]:
    """Count learned weights plus fixed LIF synaptic weights, excluding graph indices."""
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    core = getattr(policy, "core", None)
    if core is not None:
        total += core.weights.numel()
    return {
        "model_total_parameters": total,
        "model_trainable_parameters": trainable,
        "model_frozen_parameters": total - trainable,
    }


def _make_policy(kind: str, observation_dim: int, action_dim: int, manifest_path: str | None, device: torch.device, *, rewire_seed: int = 0):
    from g1_fly_control.connectome import load_connectome
    from g1_fly_control.connectome.rewire import degree_preserving_rewire
    from g1_fly_control.policies import FrozenLIFActorCritic, GRUActorCritic, LIFCore, MLPActorCritic

    if kind == "mlp":
        return MLPActorCritic(observation_dim, action_dim).to(device), None
    if kind == "gru":
        return GRUActorCritic(observation_dim, action_dim).to(device), None
    if manifest_path is None:
        raise ValueError("--connectome_manifest is required for --policy frozen_lif.")
    circuit = load_connectome(manifest_path)  # rejects synthetic data by default
    edge_index, weights = circuit.edge_index, circuit.weights
    if kind == "frozen_lif_rewired":
        edge_index, weights, report = degree_preserving_rewire(edge_index, weights, seed=rewire_seed)
        if report.completed_swaps == 0:
            raise ValueError("Degree-preserving rewiring made no swaps; verify graph size/constraints.")
    constants = {key: value for key, value in circuit.manifest.neuron_model.items() if key in {
        "dt", "tau_membrane", "tau_synapse", "threshold", "reset_value", "refractory_steps", "surrogate_beta", "neural_substeps"
    }}
    core = LIFCore(circuit.num_neurons, edge_index.to(device), weights.to(device), **constants).to(device)
    index_of = {neuron_id: index for index, neuron_id in enumerate(circuit.neuron_ids)}
    policy = FrozenLIFActorCritic(
        observation_dim, action_dim, core,
        input_indices=[index_of[neuron_id] for neuron_id in circuit.manifest.input_neuron_ids],
        output_indices=[index_of[neuron_id] for neuron_id in circuit.manifest.output_neuron_ids],
    ).to(device)
    if kind == "frozen_lif_rewired":
        from dataclasses import asdict
        policy.rewire_report = asdict(report)
    return policy, circuit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="FlyG1-GoalReach-FreePosture-v0")
    parser.add_argument("--policy", choices=("mlp", "gru", "frozen_lif", "frozen_lif_rewired"), default="mlp")
    parser.add_argument("--connectome_manifest")
    parser.add_argument("--rewire_seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument(
        "--interaction_budget", type=int,
        help="Minimum environment interactions; rounds up to complete PPO rollouts and overrides --max_iterations.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--target_kl", type=float, help="Reject and roll back a PPO epoch if its post-step KL exceeds this value.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--memory_sample_every", type=int, default=100,
        help="Measure rollout/update memory on the first, last, and every Nth PPO update (default: 100).",
    )
    parser.add_argument("--rollout_progress_every", type=int, default=0,
                        help="Log before and after every Nth control step within each rollout; 0 disables it.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning_rate must be positive and finite.")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0):
        parser.error("--target_kl must be positive and finite.")
    if args.memory_sample_every < 1:
        parser.error("--memory_sample_every must be positive.")
    if args.rollout_progress_every < 0:
        parser.error("--rollout_progress_every must be nonnegative.")
    try:
        iterations, actual_interactions = resolve_training_budget(
            num_envs=args.num_envs, horizon=args.horizon,
            max_iterations=args.max_iterations, interaction_budget=args.interaction_budget,
        )
    except ValueError as exc:
        parser.error(str(exc))
    torch.manual_seed(args.seed)
    # Validate data before incurring a simulator launch for a blocked biological condition.
    if args.policy.startswith("frozen_lif"):
        if not args.connectome_manifest:
            parser.error("--connectome_manifest is required for frozen-LIF policies (synthetic manifests are rejected).")
        from g1_fly_control.connectome import load_connectome
        from g1_fly_control.connectome.schema import ConnectomeValidationError
        try:
            load_connectome(args.connectome_manifest)
        except ConnectomeValidationError as exc:
            parser.error(str(exc))
    app = AppLauncher(args).app
    try:
        from _bootstrap import launch_environment
        from g1_fly_control.training import PPOConfig, RecurrentPPO, save_checkpoint
        from g1_fly_control.training.runner import policy_observation

        launch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        memory_samples = [training_memory_snapshot(launch_device, "app_launched")]
        reset_training_cuda_peak(launch_device)
        env = launch_environment(args.task, args.num_envs)
        observation, _ = env.reset(seed=args.seed)
        observation = policy_observation(observation)
        policy, circuit = _make_policy(args.policy, observation.shape[-1], env.action_manager.total_action_dim, args.connectome_manifest, env.device, rewire_seed=args.rewire_seed)
        runner = RecurrentPPO(policy, PPOConfig(horizon=args.horizon, learning_rate=args.learning_rate,
                                              target_kl=args.target_kl))
        memory_samples.append(training_memory_snapshot(env.device, "scene_and_policy_loaded"))
        frozen_hash = policy.core.frozen_checksum if args.policy.startswith("frozen_lif") else None
        run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") / args.task / args.policy / f"seed-{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = (run_dir / "checkpoint.pt").resolve()
        history = []
        training_start = time.perf_counter()
        for iteration in range(iterations):
            measure_update = (
                iteration == 0 or iteration == iterations - 1
                or (iteration + 1) % args.memory_sample_every == 0
            )
            if measure_update:
                reset_training_cuda_peak(env.device)
            def rollout_progress(step_index: int, phase: str, details: dict[str, float | int]) -> None:
                if (step_index + 1) % args.rollout_progress_every == 0:
                    print(json.dumps({"rollout_progress": phase, "iteration": iteration,
                                      "step_in_rollout": step_index + 1,
                                      "global_control_steps": iteration * args.horizon + step_index + 1,
                                      **details},
                                     sort_keys=True), flush=True)

            rollout, _, _ = runner.collect(
                env, progress_callback=rollout_progress if args.rollout_progress_every else None
            )
            if measure_update:
                memory_samples.append(training_memory_snapshot(env.device, "rollout", iteration=iteration))
                reset_training_cuda_peak(env.device)
            metrics = runner.update(rollout)
            if measure_update:
                memory_samples.append(training_memory_snapshot(env.device, "optimizer_update", iteration=iteration))
            if frozen_hash is not None and policy.core.frozen_checksum != frozen_hash:
                raise RuntimeError("Frozen circuit changed after optimizer step.")
            history.append({"iteration": iteration, **metrics})
            print(json.dumps(history[-1], sort_keys=True))
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize(env.device)
        training_wall_time_s = time.perf_counter() - training_start
        metadata = {
            "status": "executed_matrix_training" if args.interaction_budget is not None else "executed_smoke_or_pilot",
            "task": args.task, "policy": args.policy, "seed": args.seed,
            "observation_dim": observation.shape[-1], "action_dim": env.action_manager.total_action_dim,
            "ppo_config": runner.config_dict(), "history": history,
            "num_envs": args.num_envs,
            "requested_interaction_budget": args.interaction_budget,
            "max_iterations_requested": args.max_iterations if args.interaction_budget is None else None,
            "completed_iterations": iterations,
            "actual_environment_interactions": actual_interactions,
            "training_wall_time_s": training_wall_time_s,
            "memory_sample_every_updates": args.memory_sample_every,
            "memory_samples": memory_samples,
            "memory_note": (
                "PyTorch peaks are sampled within each measured stage and omit Isaac/driver allocations. "
                "nvidia-smi readings are device-wide snapshots, not peaks. Thresholds are advisory only."
            ),
            "checkpoint_path": str(checkpoint_path),
            **policy_parameter_counts(policy),
            "connectome_manifest": str(args.connectome_manifest) if args.connectome_manifest else None,
            "connectome_checksum": circuit.checksum if circuit else None,
            "frozen_core_checksum": frozen_hash,
            "rewire_seed": args.rewire_seed if args.policy == "frozen_lif_rewired" else None,
            "rewire_report": getattr(policy, "rewire_report", None),
            "rewire_rule": (
                "Directed destination swaps preserve each neuron's in/out degree; weights and signs travel with source edges; self-loops and duplicate edges are disallowed."
                if args.policy == "frozen_lif_rewired" else None
            ),
            "note": "Checkpoint reload is functional; this is not an exact mid-episode simulator continuation.",
        }
        reset_training_cuda_peak(env.device)
        checkpoint = save_checkpoint(checkpoint_path, policy=policy, optimizer=runner.optimizer, metadata=metadata)
        memory_samples.append(training_memory_snapshot(env.device, "checkpoint_saved"))
        # The checkpoint sample only exists after serialization. Save once more
        # so checkpoint metadata and the run manifest contain the same stages;
        # the reported stage peak measures the first write, not this metadata refresh.
        checkpoint = save_checkpoint(checkpoint_path, policy=policy, optimizer=runner.optimizer, metadata=metadata)
        (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), "run_manifest": str(run_dir / 'manifest.json')}, indent=2))
        env.close()
    except BaseException:
        # Isaac teardown can stall after an exception. Emit the original failure first.
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
