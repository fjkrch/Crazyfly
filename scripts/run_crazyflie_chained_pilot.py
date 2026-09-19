#!/usr/bin/env python3
"""Dry-run or execute the audited four-phase Crazyflie actor warm-start pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from drone_bootstrap import ROOT, canonical_sha256, sha256_file
from g1_fly_control.crazyflie.checkpoint import read_checkpoint
from g1_fly_control.tasks.crazyflie.logic import survival_first_contract_payload


CONTROLLER = "frozen_lif_original"
SOURCE_UPDATES = 200
PHASE_UPDATES = 100
PHASES = (
    ("reach", "FlyCrazyflie-WaypointReach-v0"),
    ("switch", "FlyCrazyflie-WaypointSwitch-v0"),
    ("gust", "FlyCrazyflie-GustRecovery-v0"),
    ("mixed", "FlyCrazyflie-Mixed-v0"),
)
TRAINING_FIELDS = (
    "num_envs",
    "horizon",
    "microbatch_size",
    "ppo_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "value_coefficient",
    "entropy_coefficient",
    "max_grad_norm",
    "target_kl",
    "checkpoint_every_updates",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_contract(config_path: Path) -> dict[str, Any]:
    """Read the existing main config without changing its matrix budget."""

    path = config_path.expanduser().resolve()
    config = _read_json(path)
    if config.get("task") != "FlyCrazyflie-WaypointReach-v0":
        raise ValueError("Base config is not the Crazyflie Reach training config")
    if CONTROLLER not in config.get("controllers", []):
        raise ValueError(f"Base config lacks controller {CONTROLLER}")
    if config.get("survival_first_contract") != survival_first_contract_payload():
        raise ValueError("Base config reward/curriculum contract differs from installed task code")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Base config lacks a training object")
    missing = [field for field in TRAINING_FIELDS if field not in training]
    if missing:
        raise ValueError(f"Base config training is missing: {', '.join(missing)}")
    if (
        training["num_envs"] != 4
        or training["horizon"] != 100
        or training["microbatch_size"] != 4
        or training["ppo_epochs"] != 2
        or training["checkpoint_every_updates"] != 100
    ):
        raise ValueError(
            "Chained pilot requires the existing 4-env, horizon-100, two-epoch, "
            "checkpoint-100 main PPO configuration"
        )
    if training.get("num_workers") != 0 or training.get("precision") != "float32":
        raise ValueError("Chained pilot requires zero workers and float32")
    connectome = (ROOT / str(config["connectome_manifest"])).resolve()
    rewire = (ROOT / str(config["rewire_manifest"])).resolve()
    if not connectome.is_file() or not rewire.is_file():
        raise ValueError("Base config connectome or rewire artifact is missing")
    return {
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "training": {field: training[field] for field in TRAINING_FIELDS},
        "connectome_manifest": str(connectome),
        "connectome_manifest_sha256": sha256_file(connectome),
        "rewire_seed": int(config["rewire_seed"]),
        "rewire_manifest": str(rewire),
        "rewire_manifest_sha256": sha256_file(rewire),
        "survival_first_contract": config["survival_first_contract"],
    }


def validate_source_checkpoint(
    checkpoint_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact update-200 survival anchor before planning phases."""

    source = checkpoint_path.expanduser().resolve()
    payload = read_checkpoint(source, map_location="cpu", resolve_external_history=True)
    counters = payload["counters"]
    training = contract["training"]
    interactions_per_update = int(training["num_envs"]) * int(training["horizon"])
    expected_interactions = SOURCE_UPDATES * interactions_per_update
    if int(counters["completed_updates"]) != SOURCE_UPDATES:
        raise ValueError(
            f"Source checkpoint must be update {SOURCE_UPDATES}, got {counters['completed_updates']}"
        )
    if int(counters["total_interactions"]) != expected_interactions:
        raise ValueError("Source checkpoint interaction count does not match update-200 cadence")
    resolved = payload["resolved_config"]
    if resolved.get("controller") != CONTROLLER:
        raise ValueError(f"Source checkpoint controller must be {CONTROLLER}")
    for field in TRAINING_FIELDS:
        if field == "checkpoint_every_updates":
            source_value = resolved.get(field)
        else:
            source_value = resolved.get(field)
        if source_value != training[field]:
            raise ValueError(
                f"Source checkpoint PPO field {field}={source_value!r} differs from "
                f"base config {training[field]!r}"
            )
    if resolved.get("survival_first_contract") != contract["survival_first_contract"]:
        raise ValueError("Source checkpoint survival reward/curriculum contract differs")
    fingerprints = payload["fingerprints"]
    core_checksum = payload["core_checksum"]
    if not isinstance(core_checksum, str) or fingerprints.get("frozen_core") != core_checksum:
        raise ValueError("Source checkpoint lacks a valid frozen-core fingerprint")
    metadata = payload.get("metadata")
    report = metadata.get("controller_report") if isinstance(metadata, Mapping) else None
    if not isinstance(report, Mapping):
        raise ValueError("Source checkpoint lacks controller report")
    if report.get("connectome_checksum") != fingerprints.get("connectome"):
        raise ValueError("Source checkpoint connectome fingerprints disagree")
    return {
        "absolute_path": str(source),
        "sha256": sha256_file(source),
        "completed_updates": SOURCE_UPDATES,
        "total_interactions": expected_interactions,
        "task": resolved.get("task"),
        "controller": CONTROLLER,
        "policy_class": payload["policy_class"],
        "resolved_config_checksum": payload["resolved_config_checksum"],
        "task_manifest_id": payload["task_manifest_id"],
        "fingerprints": dict(fingerprints),
        "core_checksum": core_checksum,
        "history_reference_sha256": payload["history_reference"]["history_sha256"],
    }


def _training_command(
    *,
    python: Path,
    task: str,
    run_dir: Path,
    warm_start_checkpoint: Path,
    seed: int,
    contract: Mapping[str, Any],
) -> list[str]:
    training = contract["training"]
    total_interactions = PHASE_UPDATES * int(training["num_envs"]) * int(training["horizon"])
    return [
        str(python),
        str((ROOT / "scripts" / "drone_train.py").resolve()),
        "--task", task,
        "--policy", CONTROLLER,
        "--seed", str(seed),
        "--num_envs", str(training["num_envs"]),
        "--total_interactions", str(total_interactions),
        "--horizon", str(training["horizon"]),
        "--microbatch_size", str(training["microbatch_size"]),
        "--ppo_epochs", str(training["ppo_epochs"]),
        "--learning_rate", str(training["learning_rate"]),
        "--gamma", str(training["gamma"]),
        "--gae_lambda", str(training["gae_lambda"]),
        "--clip_ratio", str(training["clip_ratio"]),
        "--value_coefficient", str(training["value_coefficient"]),
        "--entropy_coefficient", str(training["entropy_coefficient"]),
        "--max_grad_norm", str(training["max_grad_norm"]),
        "--target_kl", str(training["target_kl"]),
        "--checkpoint_every_updates", str(training["checkpoint_every_updates"]),
        "--connectome_manifest", str(contract["connectome_manifest"]),
        "--rewire_seed", str(contract["rewire_seed"]),
        "--rewire_manifest", str(contract["rewire_manifest"]),
        "--run_dir", str(run_dir),
        "--warm_start_checkpoint", str(warm_start_checkpoint),
        "--headless",
    ]


def build_phase_plan(
    *,
    source_checkpoint: Mapping[str, Any],
    output_root: Path,
    python: Path,
    seed: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    runs_root = (ROOT / "runs").resolve()
    if output_root == runs_root or not output_root.is_relative_to(runs_root):
        raise ValueError(f"Output root must be a child of {runs_root}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    phase_rows = []
    warm_source = Path(str(source_checkpoint["absolute_path"])).resolve()
    interactions_per_update = (
        int(contract["training"]["num_envs"]) * int(contract["training"]["horizon"])
    )
    for index, (name, task) in enumerate(PHASES, start=1):
        run_dir = output_root / f"{index:02d}-{name}"
        final_checkpoint = run_dir / "checkpoints" / "latest.pt"
        phase_rows.append(
            {
                "index": index,
                "name": name,
                "task": task,
                "controller": CONTROLLER,
                "seed": seed,
                "num_envs": int(contract["training"]["num_envs"]),
                "horizon": int(contract["training"]["horizon"]),
                "updates": PHASE_UPDATES,
                "total_interactions": PHASE_UPDATES * interactions_per_update,
                "warm_start_checkpoint": str(warm_source),
                "run_dir": str(run_dir),
                "final_checkpoint": str(final_checkpoint),
                "log": str(output_root / "logs" / f"{index:02d}-{name}.log"),
                "command": _training_command(
                    python=python,
                    task=task,
                    run_dir=run_dir,
                    warm_start_checkpoint=warm_source,
                    seed=seed,
                    contract=contract,
                ),
                "status": "pending",
            }
        )
        warm_source = final_checkpoint
    immutable = {
        "schema_version": 1,
        "kind": "crazyflie_actor_warm_start_chained_pilot",
        "controller": CONTROLLER,
        "source_anchor": dict(source_checkpoint),
        "base_contract": dict(contract),
        "phase_updates": PHASE_UPDATES,
        "phase_count": len(phase_rows),
        "total_new_interactions": sum(row["total_interactions"] for row in phase_rows),
        "phases": [
            {key: value for key, value in row.items() if key != "status"}
            for row in phase_rows
        ],
    }
    return {
        **immutable,
        "plan_id": canonical_sha256(immutable),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "dry_run_verified",
        "phases": phase_rows,
    }


def _validate_completed_phase(phase: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(str(phase["final_checkpoint"])).resolve()
    payload = read_checkpoint(checkpoint_path, map_location="cpu", resolve_external_history=True)
    if int(payload["counters"]["completed_updates"]) != PHASE_UPDATES:
        raise RuntimeError(f"Phase {phase['name']} did not complete {PHASE_UPDATES} updates")
    if int(payload["counters"]["total_interactions"]) != int(phase["total_interactions"]):
        raise RuntimeError(f"Phase {phase['name']} interaction count is wrong")
    if payload["resolved_config"].get("task") != phase["task"]:
        raise RuntimeError(f"Phase {phase['name']} checkpoint task is wrong")
    warm = payload.get("metadata", {}).get("warm_start")
    if not isinstance(warm, Mapping):
        raise RuntimeError(f"Phase {phase['name']} checkpoint lacks warm-start audit")
    expected_source = Path(str(phase["warm_start_checkpoint"])).resolve()
    source_record = warm.get("source_checkpoint", {})
    if source_record.get("absolute_path") != str(expected_source):
        raise RuntimeError(f"Phase {phase['name']} warm-start source path is wrong")
    if source_record.get("sha256") != sha256_file(expected_source):
        raise RuntimeError(f"Phase {phase['name']} warm-start source hash is wrong")
    manifest = _read_json(Path(str(phase["run_dir"])) / "training_manifest.json")
    if manifest.get("status") != "completed" or manifest.get("warm_start") != warm:
        raise RuntimeError(f"Phase {phase['name']} manifest warm-start audit is incomplete")
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_manifest_sha256": sha256_file(
            Path(str(phase["run_dir"])) / "training_manifest.json"
        ),
        "warm_start_source_sha256": source_record["sha256"],
    }


def execute_plan(plan: dict[str, Any], plan_path: Path) -> int:
    plan["execution_status"] = "running"
    _atomic_json(plan_path, plan)
    for phase in plan["phases"]:
        source = Path(phase["warm_start_checkpoint"])
        if not source.is_file():
            phase["status"] = "blocked_missing_warm_start"
            plan["execution_status"] = "blocked"
            _atomic_json(plan_path, plan)
            return 2
        phase["status"] = "running"
        phase["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(plan_path, plan)
        log_path = Path(phase["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            completed = subprocess.run(phase["command"], stdout=log, stderr=subprocess.STDOUT)
        phase["return_code"] = completed.returncode
        phase["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if completed.returncode != 0:
            phase["status"] = "failed"
            plan["execution_status"] = "failed"
            _atomic_json(plan_path, plan)
            return completed.returncode or 1
        phase["verification"] = _validate_completed_phase(phase)
        phase["status"] = "completed"
        _atomic_json(plan_path, plan)
    plan["execution_status"] = "completed"
    _atomic_json(plan_path, plan)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--base_config", type=Path,
        default=ROOT / "configs" / "experiments" / "crazyflie_main.json",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan_output", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite_plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise ValueError(f"Python interpreter does not exist: {python}")
    contract = load_training_contract(args.base_config)
    source = validate_source_checkpoint(args.source_checkpoint, contract)
    plan = build_phase_plan(
        source_checkpoint=source,
        output_root=args.output_root,
        python=python,
        seed=args.seed,
        contract=contract,
    )
    output_root = args.output_root.expanduser().resolve()
    plan_path = (
        args.plan_output.expanduser().resolve()
        if args.plan_output is not None else output_root / "chained_phase_plan.json"
    )
    if plan_path.exists() and not args.overwrite_plan:
        raise FileExistsError(f"Refusing to overwrite plan: {plan_path}")
    _atomic_json(plan_path, plan)
    print(json.dumps({
        "status": "DRY_RUN_VERIFIED" if not args.execute else "EXECUTION_STARTING",
        "plan": str(plan_path),
        "plan_id": plan["plan_id"],
        "phase_count": plan["phase_count"],
        "phase_updates": plan["phase_updates"],
        "total_new_interactions": plan["total_new_interactions"],
        "commands": [phase["command"] for phase in plan["phases"]],
    }, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return execute_plan(plan, plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
