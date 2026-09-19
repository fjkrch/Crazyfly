#!/usr/bin/env python3
"""Prepare and execute the resumable Crazyflie command-control comparison.

This is a dedicated queue for ``FlyCrazyflie-CommandFollow-v0``.  It never
discovers or adopts legacy waypoint runs.  ``--dry_run`` writes the exact
six-job queue atomically but launches no child process; ``--execute`` consumes
that reviewed queue.  Every child has an isolated run directory and append-only
attempt log, and a checkpoint is resumed whenever ``latest.pt`` exists.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source" / "g1_fly_control"
for import_path in (SCRIPT_DIR, SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from drone_bootstrap import (  # noqa: E402
    canonical_sha256,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
    source_hashes,
)
from drone_train import command_evaluation_manifest, standalone_resolved_config  # noqa: E402
from g1_fly_control.crazyflie.controllers import build_controller  # noqa: E402
from g1_fly_control.tasks.crazyflie.command_logic import (  # noqa: E402
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_TRACKING_CONTRACT_SHA256,
    command_follow_contract_payload,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "experiments" / "crazyflie_command_seed0_500k.json"
)
ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
TASK = COMMAND_FOLLOW_TASK_ID
CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
    "gru_matched",
    "mlp_normal",
)
LIF_CONTROLLERS = CONTROLLERS[:4]
TOTAL_INTERACTIONS = 500_000
EPISODES_PER_CONTROLLER = 16
STEPS_PER_EPISODE = 600
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
QUEUE_SCHEMA_VERSION = 1
QUEUE_KIND = "crazyflie_command_comparison_queue_v1"
QUEUE_FILE_NAME = "queue.json"
PARALLEL_GATE_KIND = "crazyflie_command_exact_two_process_parallel_gate_v1"
TERMINAL_STATES = {"completed", "failed"}
PAUSE_EXIT_CODE = 3
EVALUATOR = (SCRIPT_DIR / "crazyflie_command_evaluate.py").resolve()
PARALLEL_GATE_SCRIPT = (SCRIPT_DIR / "crazyflie_command_parallel_gate.py").resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace JSON without ever exposing a partial queue."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _expected_config_keys() -> set[str]:
    return {
        "schema_version",
        "label",
        "output_root",
        "isaac_python",
        "task",
        "contract_profile",
        "controllers",
        "seed",
        "total_interactions_per_job",
        "training",
        "command_envelope",
        "evaluation",
        "connectomes",
        "rewire",
        "queue",
        "comparison_contract",
    }


def validate_config(path: Path | str) -> dict[str, Any]:
    """Validate the closed command-only six-cell declaration."""

    path = Path(path).expanduser().resolve()
    config = _read_json(path)
    if set(config) != _expected_config_keys():
        raise ValueError("Command config top-level fields differ from schema version 1")

    exact_scalars = {
        "schema_version": 1,
        "label": "crazyflie_command_seed0_500k",
        "output_root": "runs/crazyflie_command_seed0_500k",
        "isaac_python": str(ISAAC_PYTHON),
        "task": TASK,
        "contract_profile": "command_v1",
        "controllers": list(CONTROLLERS),
        "seed": 0,
        "total_interactions_per_job": TOTAL_INTERACTIONS,
    }
    for field, expected in exact_scalars.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be exactly {expected!r}")

    expected_training = {
        "num_envs": 40,
        "horizon": 100,
        "microbatch_size": 40,
        "ppo_epochs": 2,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "checkpoint_every_updates": 25,
        "precision": "float32",
        "device": "cuda:0",
    }
    if config.get("training") != expected_training:
        raise ValueError("training differs from the reviewed 40-env command_v1 PPO contract")
    interactions_per_update = expected_training["num_envs"] * expected_training["horizon"]
    if TOTAL_INTERACTIONS % interactions_per_update:
        raise ValueError("Interaction budget must divide exactly by num_envs*horizon")

    expected_envelope = {
        "maximum_horizontal_speed_mps": 0.8,
        "maximum_vertical_speed_mps": 0.4,
        "maximum_yaw_rate_radps": 1.2,
        "minimum_hold_steps": 25,
        "maximum_hold_steps": 100,
        "control_dt_s": 0.02,
        "simultaneous_axes": True,
    }
    if config.get("command_envelope") != expected_envelope:
        raise ValueError("command_envelope differs from the reviewed learned envelope")
    compact = command_follow_contract_payload()
    if (
        compact["maximum_horizontal_speed_m_s"] != expected_envelope["maximum_horizontal_speed_mps"]
        or compact["maximum_vertical_speed_m_s"] != expected_envelope["maximum_vertical_speed_mps"]
        or compact["maximum_yaw_rate_rad_s"] != expected_envelope["maximum_yaw_rate_radps"]
    ):
        raise ValueError("Runtime command contract disagrees with the queue envelope")

    expected_evaluation = {
        "protocol": "command_v1",
        "episodes_per_controller": EPISODES_PER_CONTROLLER,
        "steps_per_episode": STEPS_PER_EPISODE,
        "deterministic_actions": True,
        "activity_from_actual_controller": True,
        "device": "cuda:0",
        "script": "scripts/crazyflie_command_evaluate.py",
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("evaluation differs from the deterministic command_v1 contract")

    expected_queue = {
        "default_max_parallel": 1,
        "maximum_parallel_after_gate": 2,
        "lif_first": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
        "network_independent": True,
    }
    if config.get("queue") != expected_queue:
        raise ValueError("queue resource/order contract differs from the reviewed schema")

    expected_connectomes = {
        "leg_manifest": "data/connectome/manifest.json",
        "leg_manifest_sha256": "c59c1641847ce7e81f01778df1ee4a136124d62bc713abde750b0d1e9836832c",
        "wing_manifest": "data/connectome_wing/manifest.json",
        "wing_manifest_sha256": "8e0149a10094109326618d5b1b042175c7c4e45307c1b7210ccf0c71cc0bf5a6",
    }
    if config.get("connectomes") != expected_connectomes:
        raise ValueError("connectome identities differ from the reviewed declaration")
    expected_rewire = {
        "seed": 20260916,
        "manifest": "configs/experiments/crazyflie_rewire_seed_20260916.json",
        "manifest_sha256": "6c2a10b879d22741b0d17110d052953adc4f081e9ecac636fabdf0ba68d57b14",
    }
    if config.get("rewire") != expected_rewire:
        raise ValueError("degree-preserving rewire identity changed")
    expected_comparison = {
        "same_task_command_stream_budget_seed_and_ppo_hyperparameters": True,
        "parameter_matched_controllers": [
            "frozen_lif_original",
            "frozen_lif_degree_rewired",
            "wing_lif",
            "gru_matched",
            "mlp_normal",
        ],
        "wing_extension_parameter_matching_required": False,
        "leg_wing_fusion": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
        "leg_wing_is_not_parameter_matched": True,
    }
    if config.get("comparison_contract") != expected_comparison:
        raise ValueError("comparison parameter-matching declaration changed")

    for relative, expected_hash in (
        (expected_connectomes["leg_manifest"], expected_connectomes["leg_manifest_sha256"]),
        (expected_connectomes["wing_manifest"], expected_connectomes["wing_manifest_sha256"]),
        (expected_rewire["manifest"], expected_rewire["manifest_sha256"]),
    ):
        artifact = (ROOT / relative).resolve()
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Pinned artifact is missing or changed: {artifact}")
    if not ISAAC_PYTHON.is_file() or not os.access(ISAAC_PYTHON, os.X_OK):
        raise ValueError(f"Required Isaac Python is not executable: {ISAAC_PYTHON}")
    if not EVALUATOR.is_file():
        raise ValueError(f"Command evaluator is missing: {EVALUATOR}")

    evaluation_manifest = command_evaluation_manifest()
    protocol = evaluation_manifest.get("evaluation_protocol", {})
    if (
        evaluation_manifest.get("protocol") != "command_v1"
        or evaluation_manifest.get("task") != TASK
        or protocol.get("episodes") != EPISODES_PER_CONTROLLER
        or protocol.get("steps_per_episode") != STEPS_PER_EPISODE
    ):
        raise ValueError("Resolved command_v1 evaluation protocol differs from the reviewed config")

    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    config["_output_root"] = str((ROOT / config["output_root"]).resolve())
    config["_leg_manifest"] = str((ROOT / expected_connectomes["leg_manifest"]).resolve())
    config["_wing_manifest"] = str((ROOT / expected_connectomes["wing_manifest"]).resolve())
    config["_rewire_manifest"] = str((ROOT / expected_rewire["manifest"]).resolve())
    config["_evaluation_manifest"] = evaluation_manifest
    config["_evaluation_manifest_id"] = evaluation_manifest["manifest_id"]
    return config


def _training_args(config: Mapping[str, Any], controller: str) -> SimpleNamespace:
    training = config["training"]
    return SimpleNamespace(
        task=TASK,
        contract_profile="command_v1",
        policy=controller,
        seed=0,
        num_envs=training["num_envs"],
        total_interactions=TOTAL_INTERACTIONS,
        horizon=training["horizon"],
        microbatch_size=training["microbatch_size"],
        ppo_epochs=training["ppo_epochs"],
        learning_rate=training["learning_rate"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        clip_ratio=training["clip_ratio"],
        value_coefficient=training["value_coefficient"],
        entropy_coefficient=training["entropy_coefficient"],
        max_grad_norm=training["max_grad_norm"],
        target_kl=training["target_kl"],
        checkpoint_every_updates=training["checkpoint_every_updates"],
        connectome_manifest=Path(config["_leg_manifest"]),
        wing_connectome_manifest=Path(config["_wing_manifest"]),
        rewire_seed=config["rewire"]["seed"],
        rewire_manifest=Path(config["_rewire_manifest"]),
        evaluation_protocol="command_v1",
        warm_start_checkpoint=None,
    )


def _job_fingerprint(config: Mapping[str, Any], controller: str) -> tuple[str, dict[str, Any]]:
    args = _training_args(config, controller)
    resolved, evaluation = standalone_resolved_config(args)
    rewired = load_fingerprint_rewire_manifest(
        args.rewire_manifest,
        expected_file_sha256=config["rewire"]["manifest_sha256"],
        expected_seed=args.rewire_seed,
    )
    manifest = args.wing_connectome_manifest if controller == "wing_lif" else args.connectome_manifest
    return reproduction_fingerprint(
        resolved_config=resolved,
        evaluation_manifest=evaluation,
        connectome_manifest=manifest,
        rewired_manifest=rewired,
    )


def _training_command(
    config: Mapping[str, Any], controller: str, run_dir: Path,
    fingerprint: str, pause_file: Path,
) -> list[str]:
    training = config["training"]
    return [
        config["isaac_python"], str(ROOT / "scripts" / "drone_train.py"),
        "--task", TASK,
        "--contract_profile", "command_v1",
        "--policy", controller,
        "--seed", "0",
        "--num_envs", str(training["num_envs"]),
        "--total_interactions", str(TOTAL_INTERACTIONS),
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
        "--connectome_manifest", config["_leg_manifest"],
        "--wing_connectome_manifest", config["_wing_manifest"],
        "--rewire_seed", str(config["rewire"]["seed"]),
        "--rewire_manifest", config["_rewire_manifest"],
        "--evaluation_protocol", "command_v1",
        "--run_dir", str(run_dir),
        "--expected_fingerprint", fingerprint,
        "--pause_file", str(pause_file),
        "--device", training["device"],
        "--headless",
    ]


def _evaluation_command(
    config: Mapping[str, Any], controller: str, checkpoint: Path,
    output: Path, fingerprint: str,
) -> list[str]:
    evaluation = config["evaluation"]
    return [
        config["isaac_python"], str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--protocol", evaluation["protocol"],
        "--device", evaluation["device"],
        "--expected_fingerprint", fingerprint,
        "--training_seed", "0",
        "--policy", controller,
        "--headless",
    ]


def _controller_reports(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for controller in CONTROLLERS:
        _, report = build_controller(
            controller,
            observation_dim=12,
            action_dim=4,
            device="cpu",
            connectome_manifest=config["_leg_manifest"],
            wing_connectome_manifest=config["_wing_manifest"],
            rewire_seed=config["rewire"]["seed"],
            rewire_manifest_path=(
                config["_rewire_manifest"]
                if controller == "frozen_lif_degree_rewired" else None
            ),
        )
        reports[controller] = report
    return reports


def parallel_gate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Identity that a command-specific exact-two-process gate must prove."""

    return {
        "kind": PARALLEL_GATE_KIND,
        "task": TASK,
        "contract_profile": "command_v1",
        "config_file_sha256": config["_config_sha256"],
        "config_identity_sha256": canonical_sha256(_public_config(config)),
        "execution_source_sha256": canonical_sha256(source_hashes()),
        "queue_runner_sha256": sha256_file(Path(__file__).resolve()),
        "parallel_gate_script_sha256": sha256_file(PARALLEL_GATE_SCRIPT),
        "exact_process_count": 2,
        "controllers": ["frozen_lif_original", "frozen_lif_degree_rewired"],
        "num_envs_per_process": 40,
        "horizon_per_process": 100,
        "total_interactions_per_process": 8_000,
        "pause_after_updates": 1,
        "interactions_at_pause": 4_000,
        "completed_updates_after_sequential_resume": 2,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
    }


def default_parallel_gate_path(output_root: Path) -> Path:
    return output_root / "prelaunch" / "parallel_exact_2x40" / "parallel_gate_receipt.json"


def validate_parallel_gate_receipt(
    path: Path | str, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a current, command-only, exact-two-process gate receipt."""

    path = Path(path).expanduser().resolve()
    envelope = _read_json(path)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("Parallel gate receipt envelope differs from schema v1")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != canonical_sha256(payload):
        raise ValueError("Parallel gate receipt payload SHA-256 mismatch")
    required = {
        "schema_version", "status", "contract", "checks", "telemetry_summary",
        "evidence", "completed_utc",
    }
    if set(payload) != required or payload.get("schema_version") != 1 or payload.get("status") != "PASS":
        raise ValueError("Parallel gate receipt schema/status is invalid")
    if payload.get("contract") != parallel_gate_contract(config):
        raise ValueError("Parallel gate receipt is stale or not command-specific")
    expected_checks = {
        "exactly_two_processes": True,
        "isolated_run_and_checkpoint_directories": True,
        "clean_checkpoints_passed": True,
        "simultaneous_gpu_compute_overlap": True,
        "no_missing_or_nonfinite_telemetry": True,
        "no_sustained_paging": True,
        "sequential_resume_completed": True,
    }
    if payload.get("checks") != expected_checks:
        raise ValueError("Parallel gate receipt checks are incomplete")
    telemetry = payload.get("telemetry_summary")
    if not isinstance(telemetry, dict) or set(telemetry) != {
        "max_device_gpu_used_mib", "max_system_ram_percent",
        "simultaneous_gpu_compute_overlap_samples", "swap_out_growth_pages",
    }:
        raise ValueError("Parallel gate telemetry summary differs from schema v1")
    try:
        gpu = float(telemetry["max_device_gpu_used_mib"])
        ram = float(telemetry["max_system_ram_percent"])
        overlap = int(telemetry["simultaneous_gpu_compute_overlap_samples"])
        swap_growth = int(telemetry["swap_out_growth_pages"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Parallel gate telemetry is non-numeric") from exc
    if (
        not math.isfinite(gpu) or not math.isfinite(ram)
        or gpu >= GPU_LIMIT_MIB or ram >= RAM_LIMIT_PERCENT
        or overlap < 1 or swap_growth < 0
    ):
        raise ValueError("Parallel gate telemetry did not pass strict limits")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "controllers", "parallel_launches", "sequential_resume_order",
        "telemetry_sample_count", "artifacts",
    }:
        raise ValueError("Parallel gate evidence differs from schema v1")
    expected_controllers = ["frozen_lif_original", "frozen_lif_degree_rewired"]
    if (
        evidence.get("controllers") != expected_controllers
        or evidence.get("sequential_resume_order") != expected_controllers
        or not isinstance(evidence.get("parallel_launches"), list)
        or len(evidence["parallel_launches"]) != 2
        or evidence.get("telemetry_sample_count", 0) < overlap
    ):
        raise ValueError("Parallel gate launch/resume evidence is incomplete")
    launches = evidence["parallel_launches"]
    if [launch.get("controller") for launch in launches if isinstance(launch, dict)] != expected_controllers:
        raise ValueError("Parallel gate used the wrong controllers/order")
    for launch in launches:
        if (
            not isinstance(launch, dict)
            or launch.get("task") != TASK
            or launch.get("num_envs") != 40
            or launch.get("pause_exit_code") != PAUSE_EXIT_CODE
            or launch.get("resume_exit_code") != 0
            or launch.get("paused_updates") != 1
            or launch.get("paused_interactions") != 4_000
            or launch.get("completed_updates") != 2
            or launch.get("completed_interactions") != 8_000
        ):
            raise ValueError("Parallel gate launch evidence failed bounded resume checks")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 10:
        raise ValueError("Parallel gate receipt lacks immutable artifacts")
    artifact_paths: set[Path] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "size_bytes", "sha256"}:
            raise ValueError("Parallel gate artifact record is malformed")
        artifact_path = Path(str(artifact["path"])).resolve()
        if artifact_path in artifact_paths or not artifact_path.is_relative_to(path.parent):
            raise ValueError("Parallel gate artifact paths must be unique and gate-local")
        artifact_paths.add(artifact_path)
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact["size_bytes"]
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise ValueError(f"Parallel gate artifact is missing or changed: {artifact_path}")
    return payload


def _parallel_gate_record(
    config: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()), "status": "missing",
        "sha256": None, "payload_sha256": None,
    }
    if not path.is_file():
        return record
    payload = validate_parallel_gate_receipt(path, config)
    record.update(
        status="PASS", sha256=sha256_file(path), payload_sha256=canonical_sha256(payload)
    )
    return record


def build_queue(
    config: Mapping[str, Any], output_root: Path | None = None,
    parallel_gate_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve the exact six jobs and commands without launching a child."""

    root = Path(output_root or config["_output_root"]).resolve()
    gate_path = Path(parallel_gate_path or default_parallel_gate_path(root)).resolve()
    evaluator_sha256 = sha256_file(EVALUATOR)
    jobs: list[dict[str, Any]] = []
    for priority, controller in enumerate(CONTROLLERS):
        identifier = f"{priority + 1:02d}__{controller}__command__seed-0"
        run_dir = root / "jobs" / identifier
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        pause_file = run_dir / "pause.request"
        evaluation_output = root / "evaluations" / identifier / "command_v1.json"
        fingerprint, fingerprint_payload = _job_fingerprint(config, controller)
        jobs.append({
            "id": identifier,
            "controller": controller,
            "controller_priority": priority,
            "architecture_class": "lif" if controller in LIF_CONTROLLERS else "baseline",
            "task": TASK,
            "seed": 0,
            "contract_profile": "command_v1",
            "status": "pending",
            "training_status": "pending",
            "evaluation_status": "pending",
            "total_interactions": TOTAL_INTERACTIONS,
            "expected_updates": TOTAL_INTERACTIONS // (
                config["training"]["num_envs"] * config["training"]["horizon"]
            ),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "training_manifest": str(run_dir / "training_manifest.json"),
            "pause_file": str(pause_file),
            "evaluation_output": str(evaluation_output),
            "evaluation_script": str(EVALUATOR),
            "evaluation_script_sha256": evaluator_sha256,
            "evaluation_manifest_id": config["_evaluation_manifest_id"],
            "evaluation_manifest": config["_evaluation_manifest"],
            "command_training_contract_sha256": COMMAND_TRACKING_CONTRACT_SHA256,
            "expected_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "training_command": _training_command(
                config, controller, run_dir, fingerprint, pause_file
            ),
            "evaluation_command": _evaluation_command(
                config, controller, checkpoint, evaluation_output, fingerprint
            ),
            "attempts": [],
        })
    if len(jobs) != 6 or len({job["id"] for job in jobs}) != 6:
        raise RuntimeError("Command comparison queue must contain six unique jobs")
    reports = _controller_reports(config)
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": QUEUE_KIND,
        "label": config["label"],
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "revision": 0,
        "status": "dry_run",
        "dry_run": True,
        "config_path": config["_config_path"],
        "config_file_sha256": config["_config_sha256"],
        "config_identity_sha256": canonical_sha256(_public_config(config)),
        "queue_runner": str(Path(__file__).resolve()),
        "queue_runner_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_script": str(EVALUATOR),
        "evaluation_script_sha256": evaluator_sha256,
        "output_root": str(root),
        "queue_file": str(root / QUEUE_FILE_NAME),
        "global_pause_file": str(root / "pause.request"),
        "job_count": 6,
        "predicted_training_interactions": 6 * TOTAL_INTERACTIONS,
        "predicted_evaluation_episodes": 6 * EPISODES_PER_CONTROLLER,
        "controller_order": list(CONTROLLERS),
        "lif_first": True,
        "parallel_gate": _parallel_gate_record(config, gate_path),
        "resource_limits": {
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "default_max_parallel": 1,
            "maximum_parallel_after_gate": 2,
        },
        "parameter_matching": config["comparison_contract"],
        "controller_reports": reports,
        "counts": {"pending": 6},
        "config": _public_config(config),
        "jobs": jobs,
        "events": [{"utc": _utc_now(), "event": "verified_dry_run_created"}],
    }


def _refresh_queue_status(queue: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in queue["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    queue["counts"] = dict(sorted(counts.items()))
    if queue.get("dry_run"):
        queue["status"] = "dry_run"
    elif counts.get("running"):
        queue["status"] = "running"
    elif counts.get("paused"):
        queue["status"] = "paused"
    elif counts.get("failed"):
        queue["status"] = "failed"
    elif counts.get("completed") == queue["job_count"]:
        queue["status"] = "completed"
    elif queue.get("resource_block"):
        queue["status"] = "blocked_resource"
    else:
        queue["status"] = "pending"


def save_queue(path: Path, queue: dict[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_utc"] = _utc_now()
    _refresh_queue_status(queue)
    _atomic_json(path, queue)
    summary = {
        "schema_version": queue["schema_version"],
        "kind": queue["kind"],
        "status": queue["status"],
        "dry_run": queue["dry_run"],
        "revision": queue["revision"],
        "updated_utc": queue["updated_utc"],
        "counts": queue["counts"],
        "job_count": queue["job_count"],
        "predicted_training_interactions": queue["predicted_training_interactions"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "parallel_gate": queue["parallel_gate"],
        "resource_limits": queue["resource_limits"],
        "resource_block": queue.get("resource_block"),
        "jobs": [
            {
                "id": job["id"], "controller": job["controller"],
                "status": job["status"], "training_status": job["training_status"],
                "evaluation_status": job["evaluation_status"],
                "checkpoint": job["checkpoint"],
            }
            for job in queue["jobs"]
        ],
    }
    _atomic_json(path.with_name("queue_summary.json"), summary)


def _pid_alive(pid: Any) -> bool:
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


@contextmanager
def queue_lock(queue_path: Path) -> Iterator[None]:
    lock = queue_path.with_name("queue.lock")
    if lock.exists():
        try:
            old_pid = _read_json(lock).get("pid")
        except (OSError, ValueError, json.JSONDecodeError):
            old_pid = None
        if _pid_alive(old_pid):
            raise RuntimeError(f"Queue runner is already active with PID {old_pid}")
        os.replace(lock, lock.with_name(f"queue.lock.stale-{time.time_ns()}"))
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps({"pid": os.getpid(), "created_utc": _utc_now()}) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        try:
            owner = _read_json(lock).get("pid")
        except (OSError, ValueError, json.JSONDecodeError):
            owner = None
        if owner == os.getpid():
            lock.unlink(missing_ok=True)


def _ram_used_percent(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    fields: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"MemTotal:", "MemAvailable:"}:
            fields[parts[0]] = int(parts[1])
    total = fields.get("MemTotal:", 0)
    available = fields.get("MemAvailable:", -1)
    if total <= 0 or available < 0 or available > total:
        raise RuntimeError("Cannot read finite MemTotal/MemAvailable telemetry")
    return 100.0 * (total - available) / total


def _swap_out_pages(vmstat_path: Path = Path("/proc/vmstat")) -> int:
    for line in vmstat_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "pswpout":
            value = int(parts[1])
            if value < 0:
                break
            return value
    raise RuntimeError("Cannot read pswpout telemetry")


def resource_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi telemetry failed: {completed.stderr.strip()}")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("GPU telemetry must resolve exactly CUDA device 0")
    try:
        gpu = float(rows[0])
        ram = float(_ram_used_percent())
        swap_out = int(_swap_out_pages())
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Resource telemetry is non-numeric") from exc
    if not math.isfinite(gpu) or not math.isfinite(ram):
        raise RuntimeError("Resource telemetry is non-finite")
    return {
        "utc": _utc_now(), "gpu_index": 0,
        "gpu_used_mib": gpu, "system_ram_percent": ram,
        "swap_out_pages": swap_out,
        "gpu_limit_mib_exclusive": GPU_LIMIT_MIB,
        "ram_limit_percent_exclusive": RAM_LIMIT_PERCENT,
        "passed": gpu < GPU_LIMIT_MIB and ram < RAM_LIMIT_PERCENT,
    }


def gpu_compute_client_pids() -> set[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute telemetry failed: {completed.stderr.strip()}")
    result: set[int] = set()
    for row in completed.stdout.splitlines():
        if not row.strip():
            continue
        try:
            pid = int(row.strip())
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned a non-integer compute PID") from exc
        if pid < 1:
            raise RuntimeError("nvidia-smi returned an invalid compute PID")
        result.add(pid)
    return result


def _active_gpu_allocation_observed(active: Mapping[str, Mapping[str, Any]]) -> bool:
    expected = {
        int(handle["job"]["active_process"]["pid"])
        for handle in active.values()
    }
    return bool(expected) and expected.issubset(gpu_compute_client_pids())


def _memory_gate_passed(gate: Any) -> bool:
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        return False
    try:
        gpu = float(gate["max_device_gpu_used_mib"])
        ram = float(gate["max_system_ram_percent"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(gpu) and math.isfinite(ram) and gpu < GPU_LIMIT_MIB and ram < RAM_LIMIT_PERCENT


def valid_training(job: Mapping[str, Any]) -> bool:
    try:
        manifest = _read_json(Path(job["training_manifest"]))
        checkpoint = Path(job["checkpoint"])
        actual_hash = sha256_file(checkpoint)
        schedule = manifest["command_schedule"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "completed"
        and manifest.get("task") == TASK
        and manifest.get("contract_profile") == "command_v1"
        and manifest.get("controller") == job["controller"]
        and manifest.get("seed") == 0
        and manifest.get("num_envs") == 40
        and manifest.get("horizon") == 100
        and manifest.get("requested_interactions") == TOTAL_INTERACTIONS
        and manifest.get("environment_interactions") == TOTAL_INTERACTIONS
        and manifest.get("completed_updates") == job["expected_updates"] == 125
        and manifest.get("fingerprint") == job["expected_fingerprint"]
        and manifest.get("fingerprint_payload") == job["fingerprint_payload"]
        and manifest.get("evaluation_manifest_id") == job["evaluation_manifest_id"]
        and Path(manifest.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and manifest.get("checkpoint_sha256") == actual_hash
        and manifest.get("core_checksum_before") == manifest.get("core_checksum_after")
        and isinstance(schedule, dict)
        and schedule.get("command_training_contract_sha256")
        == job["command_training_contract_sha256"]
        and isinstance(schedule.get("state"), dict)
        and schedule["state"].get("training_interactions") == TOTAL_INTERACTIONS
        and _memory_gate_passed(manifest.get("memory_gate"))
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def valid_evaluation(job: Mapping[str, Any]) -> bool:
    try:
        value = _read_json(Path(job["evaluation_output"]))
        checkpoint = Path(job["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    episodes = value.get("episodes")
    checkpoint_record = value.get("checkpoint")
    summary = value.get("summary")
    integrity = value.get("integrity")
    quality_keys = {
        "acceleration_quality", "command_response", "flight_stability",
        "survival_not_die",
    }
    return (
        value.get("schema_version") == 1
        and value.get("analysis_kind") == "crazyflie_command_follow_heldout_v1"
        and value.get("status") == "PASS"
        and value.get("task") == TASK
        and value.get("controller") == job["controller"]
        and isinstance(checkpoint_record, dict)
        and Path(checkpoint_record.get("path", "")).resolve() == checkpoint.resolve()
        and checkpoint_record.get("sha256") == checkpoint_hash
        and checkpoint_record.get("training_seed") == 0
        and checkpoint_record.get("total_interactions") == TOTAL_INTERACTIONS
        and checkpoint_record.get("reproduction_fingerprint") == job["expected_fingerprint"]
        and checkpoint_record.get("evaluation_manifest_id") == job["evaluation_manifest_id"]
        and checkpoint_record.get("evaluation_manifest") == job["evaluation_manifest"]
        and value.get("protocol")
        == job["evaluation_manifest"].get("evaluation_protocol")
        and value.get("protocol_sha256")
        == job["evaluation_manifest"].get("evaluation_protocol_sha256")
        and value.get("episodes_requested") == EPISODES_PER_CONTROLLER
        and value.get("episodes_evaluated") == EPISODES_PER_CONTROLLER
        and value.get("steps_per_episode") == STEPS_PER_EPISODE
        and value.get("vectorized_environment_count") == EPISODES_PER_CONTROLLER
        and value.get("deterministic_actions") is True
        and value.get("policy_action_source") == "actual_trained_controller_no_assist"
        and isinstance(episodes, list) and len(episodes) == EPISODES_PER_CONTROLLER
        and all(isinstance(row, dict) and row for row in episodes)
        and _memory_gate_passed(value.get("memory_gate"))
        and isinstance(summary, dict)
        and isinstance(summary.get("score"), dict)
        and isinstance(summary.get("control_quality"), dict)
        and set(summary["control_quality"].get("component_scores_0_100", {})) == quality_keys
        and isinstance(value.get("activity"), dict)
        and isinstance(integrity, dict)
        and integrity.get("task_manifest_matched") is True
        and integrity.get("evaluation_manifest_matched") is True
        and integrity.get("source_set_matched") is True
        and integrity.get("checkpoint_completed_budget") is True
        and integrity.get("all_actions_finite_and_bounded") is True
        and integrity.get("activity_from_actual_forward") is True
        and _finite_tree(value)
    )


def _process_identity_alive(job: Mapping[str, Any]) -> bool:
    active = job.get("active_process")
    if not isinstance(active, dict) or not _pid_alive(active.get("pid")):
        return False
    try:
        raw = Path(f"/proc/{active['pid']}/cmdline").read_bytes()
    except OSError:
        return False
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    if active.get("phase") == "training":
        return "drone_train.py" in command and str(job["run_dir"]) in command
    return "crazyflie_command_evaluate.py" in command and str(job["evaluation_output"]) in command


def reconcile_queue(queue: dict[str, Any], *, retry_failed: bool) -> None:
    for job in queue["jobs"]:
        if valid_training(job):
            job["training_status"] = "completed"
            if valid_evaluation(job):
                job["evaluation_status"] = "completed"
                job["status"] = "completed"
                job.pop("active_process", None)
                continue
            job["evaluation_status"] = "pending"
        else:
            job["training_status"] = "pending"
            job["evaluation_status"] = "pending"
        if _process_identity_alive(job):
            job["status"] = "running"
            job["recovered_detached_process"] = True
            continue
        job.pop("active_process", None)
        if retry_failed or job.get("status") not in {"failed", "paused"}:
            job["status"] = "pending"
            job.pop("failure", None)


def _next_phase(job: dict[str, Any]) -> str | None:
    if not valid_training(job):
        return "training"
    job["training_status"] = "completed"
    if not valid_evaluation(job):
        return "evaluation"
    job["evaluation_status"] = "completed"
    job["status"] = "completed"
    return None


def _archive_invalid_evaluation(job: Mapping[str, Any]) -> str | None:
    output = Path(job["evaluation_output"])
    if not output.exists() or valid_evaluation(job):
        return None
    archive_dir = output.parent / "invalid_attempts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{output.stem}-{time.time_ns()}{output.suffix}"
    os.replace(output, archive)
    return str(archive)


def _start_child(job: dict[str, Any], phase: str, output_root: Path) -> dict[str, Any]:
    command = list(job[f"{phase}_command"])
    if phase == "training" and Path(job["checkpoint"]).is_file():
        command.append("--resume")
    archived_output = _archive_invalid_evaluation(job) if phase == "evaluation" else None
    attempt_index = len(job["attempts"]) + 1
    log = output_root / "logs" / f"{job['id']}__{phase}__attempt-{attempt_index:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("xb")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    record = {
        "attempt": attempt_index, "phase": phase, "pid": process.pid,
        "started_utc": _utc_now(), "command": command, "log": str(log),
        "archived_invalid_output": archived_output,
    }
    job["attempts"].append(record)
    job["active_process"] = {
        "pid": process.pid, "phase": phase,
        "started_utc": record["started_utc"], "log": str(log),
    }
    job["status"] = "running"
    job[f"{phase}_status"] = "running"
    return {"process": process, "stream": stream, "phase": phase, "job": job, "record": record}


def _finish_child(handle: dict[str, Any], exit_code: int | None) -> None:
    job = handle["job"]
    phase = handle["phase"]
    stream = handle.get("stream")
    if stream is not None:
        stream.close()
    record = handle["record"]
    record["finished_utc"] = _utc_now()
    record["exit_code"] = exit_code
    job.pop("active_process", None)
    valid = valid_training(job) if phase == "training" else valid_evaluation(job)
    if valid:
        job[f"{phase}_status"] = "completed"
        job["status"] = "completed" if phase == "evaluation" else "pending"
    elif phase == "training" and exit_code == PAUSE_EXIT_CODE:
        job["training_status"] = "paused"
        job["status"] = "paused"
    else:
        job[f"{phase}_status"] = "failed"
        job["status"] = "failed"
        job["failure"] = (
            f"{phase} attempt exited {exit_code!r} and its artifact did not pass validation"
        )


def _request_job_pause(job: Mapping[str, Any], queue_path: Path, reason: str) -> Path:
    path = Path(job["pause_file"])
    if not path.exists():
        _atomic_json(path, {
            "schema_version": 1, "status": "requested", "requested_utc": _utc_now(),
            "requesting_pid": os.getpid(), "reason": reason,
            "queue": str(queue_path), "job": job["id"],
        })
    return path


def _request_global_pause(queue: dict[str, Any], queue_path: Path, reason: str) -> Path:
    path = Path(queue["global_pause_file"])
    if not path.exists():
        _atomic_json(path, {
            "schema_version": 1, "status": "requested", "requested_utc": _utc_now(),
            "requesting_pid": os.getpid(), "reason": reason, "queue": str(queue_path),
        })
    return path


def _consume_pause_files(queue: dict[str, Any]) -> None:
    paths = [Path(queue["global_pause_file"])] + [Path(job["pause_file"]) for job in queue["jobs"]]
    for path in paths:
        if not path.exists():
            continue
        archive_dir = path.parent / "pause_requests"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{path.stem}-consumed-{time.time_ns()}{path.suffix}"
        os.replace(path, archive)
        queue["events"].append({"utc": _utc_now(), "event": "pause_request_consumed", "archive": str(archive)})


def _eligible_jobs(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = queue["jobs"]
    lif_incomplete = any(
        job["controller"] in LIF_CONTROLLERS and job["status"] != "completed"
        for job in jobs
    )
    return [
        job for job in jobs
        if job["status"] == "pending"
        and (job["controller"] in LIF_CONTROLLERS or not lif_incomplete)
    ]


def execute_queue(
    queue: dict[str, Any], queue_path: Path, *, max_parallel: int, resume: bool,
    config: Mapping[str, Any], parallel_gate_path: Path,
) -> int:
    if max_parallel not in {1, 2}:
        raise ValueError("max_parallel must be 1 or 2")
    if max_parallel == 2:
        current_gate = _parallel_gate_record(config, parallel_gate_path)
        if current_gate.get("status") != "PASS":
            raise ValueError(
                "max_parallel=2 requires a current authenticated command-specific exact 2x40 gate receipt"
            )
        queue["parallel_gate"] = current_gate
    if resume:
        _consume_pause_files(queue)
    elif Path(queue["global_pause_file"]).exists() or any(
        Path(job["pause_file"]).exists() for job in queue["jobs"]
    ):
        raise RuntimeError("Pause request exists; use --execute --resume to continue")

    reconcile_queue(queue, retry_failed=resume)
    queue["dry_run"] = False
    queue.pop("resource_block", None)
    queue.setdefault("started_utc", _utc_now())
    queue["last_runner"] = {
        "pid": os.getpid(), "started_utc": _utc_now(),
        "max_parallel": max_parallel, "resume": resume,
    }
    queue["events"].append({
        "utc": _utc_now(), "event": "runner_started", "pid": os.getpid(),
        "max_parallel": max_parallel, "resume": resume,
    })
    save_queue(queue_path, queue)

    stop_requested = False
    resource_stop = False
    active: dict[str, dict[str, Any]] = {}

    def handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        _request_global_pause(queue, queue_path, f"signal_{signum}")
        for handle in active.values():
            if handle["phase"] == "training":
                _request_job_pause(handle["job"], queue_path, f"signal_{signum}")

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handle_stop)
    if hasattr(signal, "SIGHUP"):
        previous_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)

    for job in queue["jobs"]:
        if job.get("status") == "running" and _process_identity_alive(job):
            active[job["id"]] = {
                "process": None, "stream": None,
                "phase": job["active_process"]["phase"], "job": job,
                "record": {
                    "attempt": None, "phase": job["active_process"]["phase"],
                    "pid": job["active_process"]["pid"],
                    "started_utc": job["active_process"].get("started_utc"),
                    "log": job["active_process"].get("log"),
                    "recovered_by_pid_monitor": True,
                },
            }
    swap_samples: list[int] = []
    try:
        while True:
            for identifier, handle in list(active.items()):
                process = handle["process"]
                if process is None:
                    if _process_identity_alive(handle["job"]):
                        continue
                    exit_code = None
                else:
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                _finish_child(handle, exit_code)
                active.pop(identifier)
                save_queue(queue_path, queue)

            if active and not resource_stop:
                try:
                    sample = resource_snapshot()
                    swap_samples.append(sample["swap_out_pages"])
                    swap_samples = swap_samples[-3:]
                    sustained_paging = (
                        len(swap_samples) == 3
                        and swap_samples[0] < swap_samples[1] < swap_samples[2]
                    )
                    sample["sustained_paging"] = sustained_paging
                    sample["passed"] = bool(sample["passed"] and not sustained_paging)
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                    sample = {
                        "utc": _utc_now(), "passed": False,
                        "telemetry_error": f"{type(exc).__name__}: {exc}",
                    }
                queue["last_resource_sample"] = sample
                queue.setdefault("resource_samples", []).append(sample)
                if len(queue["resource_samples"]) > 200:
                    queue["resource_samples"] = queue["resource_samples"][-200:]
                if not sample["passed"]:
                    resource_stop = True
                    stop_requested = True
                    queue["resource_block"] = sample
                    _request_global_pause(queue, queue_path, "hard_resource_gate")
                    for handle in active.values():
                        handle["record"]["resource_stop_requested"] = sample
                        if handle["phase"] == "training":
                            _request_job_pause(handle["job"], queue_path, "hard_resource_gate")
                    queue["events"].append({
                        "utc": _utc_now(), "event": "hard_resource_gate", "sample": sample,
                    })
                    save_queue(queue_path, queue)

            if stop_requested or Path(queue["global_pause_file"]).exists():
                if not active:
                    for job in queue["jobs"]:
                        if job["status"] == "pending":
                            job["status"] = "paused"
                    queue["events"].append({"utc": _utc_now(), "event": "runner_paused"})
                    save_queue(queue_path, queue)
                    return 4 if resource_stop else PAUSE_EXIT_CODE
            else:
                candidates = _eligible_jobs(queue)
                while candidates and len(active) < max_parallel:
                    if active and max_parallel == 2 and not _active_gpu_allocation_observed(active):
                        queue["last_parallel_readiness"] = {
                            "utc": _utc_now(),
                            "status": "waiting_for_existing_child_gpu_allocation",
                            "active_pids": sorted(
                                int(handle["job"]["active_process"]["pid"])
                                for handle in active.values()
                            ),
                        }
                        break
                    try:
                        snapshot = resource_snapshot()
                    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                        snapshot = {
                            "utc": _utc_now(), "passed": False,
                            "telemetry_error": f"{type(exc).__name__}: {exc}",
                        }
                    queue["last_resource_precheck"] = snapshot
                    queue.setdefault("resource_prechecks", []).append(snapshot)
                    if len(queue["resource_prechecks"]) > 100:
                        queue["resource_prechecks"] = queue["resource_prechecks"][-100:]
                    if not snapshot.get("passed"):
                        if not active:
                            queue["resource_block"] = snapshot
                            queue["events"].append({
                                "utc": _utc_now(), "event": "resource_precheck_blocked",
                                "snapshot": snapshot,
                            })
                            save_queue(queue_path, queue)
                            return 4
                        break
                    job = candidates.pop(0)
                    phase = _next_phase(job)
                    if phase is None:
                        save_queue(queue_path, queue)
                        continue
                    handle = _start_child(job, phase, Path(queue["output_root"]))
                    handle["record"]["resource_precheck"] = snapshot
                    active[job["id"]] = handle
                    save_queue(queue_path, queue)

            if not active:
                if all(job["status"] == "completed" for job in queue["jobs"]):
                    queue["finished_utc"] = _utc_now()
                    queue["events"].append({"utc": _utc_now(), "event": "runner_completed"})
                    save_queue(queue_path, queue)
                    return 0
                if not _eligible_jobs(queue):
                    queue["finished_utc"] = _utc_now()
                    queue["events"].append({"utc": _utc_now(), "event": "runner_stopped_incomplete"})
                    save_queue(queue_path, queue)
                    return 1
            time.sleep(float(queue["config"]["queue"]["resource_poll_interval_seconds"]))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _validate_loaded_queue(
    queue: dict[str, Any], config: Mapping[str, Any], queue_path: Path,
    parallel_gate_path: Path,
) -> None:
    expected_order = list(CONTROLLERS)
    if (
        queue.get("schema_version") != QUEUE_SCHEMA_VERSION
        or queue.get("kind") != QUEUE_KIND
        or queue.get("config_file_sha256") != config["_config_sha256"]
        or queue.get("config_identity_sha256") != canonical_sha256(_public_config(config))
        or Path(queue.get("queue_runner", "")).resolve() != Path(__file__).resolve()
        or queue.get("queue_runner_sha256") != sha256_file(Path(__file__).resolve())
        or Path(queue.get("evaluation_script", "")).resolve() != EVALUATOR
        or queue.get("evaluation_script_sha256") != sha256_file(EVALUATOR)
        or Path(queue.get("queue_file", "")).resolve() != queue_path.resolve()
        or queue.get("job_count") != 6
        or queue.get("predicted_training_interactions") != 3_000_000
        or queue.get("predicted_evaluation_episodes") != 96
        or queue.get("controller_order") != expected_order
        or [job.get("controller") for job in queue.get("jobs", [])] != expected_order
        or any(job.get("task") != TASK or job.get("seed") != 0 for job in queue.get("jobs", []))
    ):
        raise ValueError("Existing queue does not match the reviewed command config/order")
    stored_gate = queue.get("parallel_gate")
    if not isinstance(stored_gate, dict) or stored_gate.get("path") != str(parallel_gate_path.resolve()):
        raise ValueError("Existing queue parallel-gate path changed")
    if stored_gate.get("status") == "PASS" and stored_gate != _parallel_gate_record(
        config, parallel_gate_path
    ):
        raise ValueError("Existing queue parallel-gate receipt is stale or changed")
    for job in queue["jobs"]:
        fingerprint, payload = _job_fingerprint(config, job["controller"])
        if (
            job.get("expected_fingerprint") != fingerprint
            or job.get("fingerprint_payload") != payload
            or job.get("evaluation_manifest") != config["_evaluation_manifest"]
            or job.get("evaluation_script_sha256") != sha256_file(EVALUATOR)
            or job.get("command_training_contract_sha256") != COMMAND_TRACKING_CONTRACT_SHA256
        ):
            raise ValueError(f"Existing queue fingerprint is stale for {job.get('id')}")


def _dry_run_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "verified_dry_run_no_processes_launched",
        "queue": queue["queue_file"],
        "job_count": queue["job_count"],
        "training_interactions_per_job": TOTAL_INTERACTIONS,
        "predicted_training_interactions": queue["predicted_training_interactions"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "controller_order": queue["controller_order"],
        "parallel_gate": queue["parallel_gate"],
        "cells": [
            {
                "id": job["id"], "controller": job["controller"],
                "task": job["task"], "seed": job["seed"],
                "total_interactions": job["total_interactions"],
                "expected_updates": job["expected_updates"],
                "run_dir": job["run_dir"], "checkpoint": job["checkpoint"],
                "evaluation_output": job["evaluation_output"],
                "training_command": job["training_command"],
                "evaluation_command": job["evaluation_command"],
            }
            for job in queue["jobs"]
        ],
    }


def _status_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": queue["status"], "queue": queue["queue_file"],
        "revision": queue["revision"], "counts": queue["counts"],
        "jobs": queue["job_count"],
        "predicted_training_interactions": queue["predicted_training_interactions"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "parallel_gate": queue["parallel_gate"],
        "last_resource_precheck": queue.get("last_resource_precheck"),
        "last_resource_sample": queue.get("last_resource_sample"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true", help="Write the verified six-job queue; launch nothing")
    mode.add_argument("--execute", action="store_true", help="Execute the existing verified dry-run queue")
    mode.add_argument("--status", action="store_true", help="Read and validate queue status without mutation")
    mode.add_argument("--pause", action="store_true", help="Request clean checkpoint pauses")
    parser.add_argument("--resume", action="store_true", help="With --execute, consume pause files and retry failed/paused jobs")
    parser.add_argument("--max_parallel", type=int, choices=(1, 2))
    parser.add_argument("--parallel_gate_receipt", type=Path)
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.max_parallel is not None and not args.execute:
        parser.error("--max_parallel requires --execute")
    try:
        config = validate_config(args.config)
        output_root = Path(config["_output_root"])
        queue_path = output_root / QUEUE_FILE_NAME
        gate_path = Path(
            args.parallel_gate_receipt or default_parallel_gate_path(output_root)
        ).expanduser().resolve()
        if args.dry_run:
            if queue_path.exists():
                raise ValueError(
                    f"Queue already exists and will not be overwritten: {queue_path}"
                )
            queue = build_queue(config, output_root, gate_path)
            save_queue(queue_path, queue)
            print(json.dumps(_dry_run_payload(queue), indent=2, sort_keys=True))
            return 0
        if not queue_path.is_file():
            raise ValueError(f"Verified dry-run queue does not exist: {queue_path}")
        queue = _read_json(queue_path)
        _validate_loaded_queue(queue, config, queue_path, gate_path)
        if args.status:
            print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
            return 0
        if args.pause:
            pause_file = _request_global_pause(queue, queue_path, "explicit_cli_request")
            for job in queue["jobs"]:
                if job.get("status") == "running" and job.get("active_process", {}).get("phase") == "training":
                    _request_job_pause(job, queue_path, "explicit_cli_request")
            print(json.dumps({"status": "pause_requested", "pause_file": str(pause_file)}, indent=2))
            return 0
        max_parallel = args.max_parallel or config["queue"]["default_max_parallel"]
        with queue_lock(queue_path):
            result = execute_queue(
                queue, queue_path, max_parallel=max_parallel, resume=args.resume,
                config=config, parallel_gate_path=gate_path,
            )
        print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
        return result
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
