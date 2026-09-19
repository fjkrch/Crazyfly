#!/usr/bin/env python3
"""Build the review-only leg/wing/combined Crazyflie extension queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from drone_bootstrap import ROOT, reproduction_fingerprint, sha256_file
from drone_evaluation_protocol import SCENARIOS
from drone_run_matrix import (
    memory_acceptance_contract_payload,
    public_config,
    validate_config,
)
from drone_train import standalone_resolved_config
from g1_fly_control.connectome import load_connectome
from g1_fly_control.crazyflie.controllers import build_controller
from g1_fly_control.crazyflie.memory import (
    GPU_LIMIT_MIB,
    MEMORY_POLICY_VERSION,
    RAM_LIMIT_PERCENT,
    RSS_GROWTH_TOLERANCE_MIB,
    SWAP_OUT_GROWTH_TOLERANCE_MIB,
)


CONTROLLERS = ("frozen_lif_original", "wing_lif", "leg_wing_lif")
SEEDS = (0, 1, 2, 3, 4)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_wing_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    config = _read_json(path)
    expected = {
        "schema_version": 1,
        "label": "wing_main",
        "task": "FlyCrazyflie-WaypointReach-v0",
        "controllers": list(CONTROLLERS),
        "seeds": list(SEEDS),
        "total_interactions": 5_000_000,
        "evaluation_protocol": "main",
        "episodes_per_scenario": 16,
        "parameter_matching_required": False,
        "fusion_contract": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
        "max_concurrent_isaac_processes": 1,
        "execution_authorized": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be exactly {value!r}")
    base_path = (ROOT / str(config.get("base_config", ""))).resolve()
    base = validate_config(base_path)
    if base["label"] != "main":
        raise ValueError("Wing extension must inherit the validated main resource/evaluation contract")
    if base["total_interactions"] != config["total_interactions"]:
        raise ValueError("Wing and baseline interaction budgets differ")
    if base["evaluation"]["episodes_per_scenario"] != config["episodes_per_scenario"]:
        raise ValueError("Wing and baseline evaluation episode counts differ")
    leg = (ROOT / str(config.get("leg_connectome_manifest", ""))).resolve()
    wing = (ROOT / str(config.get("wing_connectome_manifest", ""))).resolve()
    if leg != Path(base["_connectome_path"]):
        raise ValueError("Wing comparison leg condition must use the untouched baseline connectome")
    if not wing.is_file():
        raise ValueError(f"Wing connectome manifest does not exist: {wing}")
    if sha256_file(wing) != config.get("wing_connectome_manifest_sha256"):
        raise ValueError("Wing connectome manifest SHA-256 mismatch")
    load_connectome(leg)
    load_connectome(wing)
    config.update({
        "_config_path": str(path),
        "_base": base,
        "_base_path": str(base_path),
        "_leg_path": str(leg),
        "_wing_path": str(wing),
    })
    return config, base


def _training_namespace(
    config: dict[str, Any], base: dict[str, Any], controller: str, seed: int
) -> SimpleNamespace:
    training = base["training"]
    return SimpleNamespace(
        task=config["task"], policy=controller, seed=seed,
        num_envs=training["num_envs"], total_interactions=config["total_interactions"],
        horizon=training["horizon"], microbatch_size=training["microbatch_size"],
        ppo_epochs=training["ppo_epochs"], learning_rate=training["learning_rate"],
        gamma=training["gamma"], gae_lambda=training["gae_lambda"],
        clip_ratio=training["clip_ratio"], value_coefficient=training["value_coefficient"],
        entropy_coefficient=training["entropy_coefficient"],
        max_grad_norm=training["max_grad_norm"], target_kl=training["target_kl"],
        checkpoint_every_updates=training["checkpoint_every_updates"],
        connectome_manifest=Path(config["_leg_path"]),
        wing_connectome_manifest=Path(config["_wing_path"]),
        rewire_seed=base["rewire_seed"],
        rewire_manifest=Path(base["_rewire_manifest_path"]),
        evaluation_protocol=config["evaluation_protocol"],
        warm_start_checkpoint=None,
    )


def _training_command(args: SimpleNamespace, run_dir: Path, fingerprint: str) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / "drone_train.py"),
        "--task", args.task, "--policy", args.policy, "--seed", str(args.seed),
        "--num_envs", str(args.num_envs), "--total_interactions", str(args.total_interactions),
        "--horizon", str(args.horizon), "--microbatch_size", str(args.microbatch_size),
        "--ppo_epochs", str(args.ppo_epochs), "--learning_rate", str(args.learning_rate),
        "--gamma", str(args.gamma), "--gae_lambda", str(args.gae_lambda),
        "--clip_ratio", str(args.clip_ratio), "--value_coefficient", str(args.value_coefficient),
        "--entropy_coefficient", str(args.entropy_coefficient),
        "--max_grad_norm", str(args.max_grad_norm), "--target_kl", str(args.target_kl),
        "--checkpoint_every_updates", str(args.checkpoint_every_updates),
        "--connectome_manifest", str(args.connectome_manifest),
        "--wing_connectome_manifest", str(args.wing_connectome_manifest),
        "--rewire_seed", str(args.rewire_seed), "--rewire_manifest", str(args.rewire_manifest),
        "--evaluation_protocol", args.evaluation_protocol,
        "--run_dir", str(run_dir), "--expected_fingerprint", fingerprint, "--headless",
    ]


def build_queue(config: dict[str, Any], base: dict[str, Any], output: Path) -> dict[str, Any]:
    artifact_root = output.parent / output.stem
    reports: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        _, reports[controller] = build_controller(
            controller, observation_dim=12, action_dim=4, device="cpu",
            connectome_manifest=config["_leg_path"],
            wing_connectome_manifest=config["_wing_path"],
        )
        for seed in SEEDS:
            identifier = f"{controller}__seed-{seed}"
            run_dir = artifact_root / "jobs" / identifier
            args = _training_namespace(config, base, controller, seed)
            resolved, protocol = standalone_resolved_config(args)
            from drone_bootstrap import load_fingerprint_rewire_manifest
            rewire = load_fingerprint_rewire_manifest(
                args.rewire_manifest, expected_seed=args.rewire_seed
            )
            fingerprint, payload = reproduction_fingerprint(
                resolved_config=resolved,
                evaluation_manifest=protocol,
                connectome_manifest=(
                    args.wing_connectome_manifest if controller == "wing_lif"
                    else args.connectome_manifest
                ),
                rewired_manifest=rewire,
            )
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            evaluations = []
            for scenario in SCENARIOS:
                eval_output = artifact_root / "evaluations" / identifier / f"{scenario}.json"
                evaluations.append({
                    "scenario": scenario,
                    "episodes": config["episodes_per_scenario"],
                    "status": "pending",
                    "output": str(eval_output),
                    "command": [
                        sys.executable, str(ROOT / "scripts" / "drone_evaluate.py"),
                        "--checkpoint", str(checkpoint), "--protocol", "main",
                        "--scenario", scenario, "--headless", "--output", str(eval_output),
                        "--expected_fingerprint", fingerprint, "--training_seed", str(seed),
                        "--policy", controller,
                    ],
                })
            jobs.append({
                "id": identifier, "controller": controller, "seed": seed,
                "status": "pending", "run_dir": str(run_dir),
                "checkpoint": str(checkpoint), "total_interactions": config["total_interactions"],
                "expected_fingerprint": fingerprint, "fingerprint_payload": payload,
                "training_command": _training_command(args, run_dir, fingerprint),
                "evaluations": evaluations,
            })
    bundles = sum(len(job["evaluations"]) for job in jobs)
    episodes = bundles * config["episodes_per_scenario"]
    if len(jobs) != 15 or bundles != 45 or episodes != 720:
        raise RuntimeError("Wing dry run must contain 15 jobs, 45 bundles, and 720 episodes")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "wing_main", "status": "dry_run", "dry_run": True,
        "execution_authorized": False,
        "execution_guard": "This planner has no --execute mode; review and explicitly authorize a later runner.",
        "config_path": config["_config_path"], "base_config_path": config["_base_path"],
        "output": str(output), "artifact_root": str(artifact_root),
        "sequential_process_limit": 1,
        "resource_limits": {
            "policy_version": MEMORY_POLICY_VERSION,
            "max_concurrent_isaac_processes": 1,
            "device_gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "gpu_telemetry_required_for_cuda": True,
            "finite_numeric_telemetry_required": True,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "rss_growth_tolerance_mib": RSS_GROWTH_TOLERANCE_MIB,
            "rss_growth_disposition": "warning_only",
            "swap_out_growth_tolerance_mib": SWAP_OUT_GROWTH_TOLERANCE_MIB,
            "sustained_paging_disposition": "hard_failure",
        },
        "memory_acceptance": memory_acceptance_contract_payload(),
        "parameter_matching_required": False,
        "controller_reports": reports,
        "job_count": len(jobs), "evaluation_bundle_count": bundles,
        "predicted_evaluation_episodes": episodes,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "baseline_main_config": public_config(base),
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--verify", action="store_true", help="Rebuild and verify an existing queue without changing it")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config, base = validate_wing_config(args.config)
        output = (args.output or ROOT / "runs" / "crazyflie_wing_main_v1.json").resolve()
        queue = build_queue(config, base, output)
        if args.verify:
            if not output.is_file():
                parser.error(f"Cannot verify missing queue: {output}")
            existing = _read_json(output)
            compared_fields = (
                "schema_version", "label", "status", "dry_run", "execution_authorized",
                "execution_guard", "config_path", "base_config_path", "output",
                "artifact_root", "sequential_process_limit", "resource_limits",
                "memory_acceptance",
                "parameter_matching_required", "controller_reports", "job_count",
                "evaluation_bundle_count", "predicted_evaluation_episodes", "config",
                "baseline_main_config", "jobs",
            )
            if any(existing.get(field) != queue[field] for field in compared_fields):
                raise ValueError("Existing wing queue differs from the current validated dry run")
            queue = existing
        else:
            if output.exists():
                parser.error(f"Queue already exists and is preserved: {output}")
            _atomic_json(output, queue)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": "PASS", "queue": str(output), "jobs": queue["job_count"],
        "evaluation_bundles": queue["evaluation_bundle_count"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "execution_authorized": False,
        "existing_queue_verified": bool(args.verify),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
