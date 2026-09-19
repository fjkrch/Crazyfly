#!/usr/bin/env python3
"""Prepare and run the resumable seed-0 Crazyflie neural comparison queue.

This launcher is deliberately not named ``drone_*.py``: it orchestrates the
frozen training/evaluation entrypoints without entering their reproduction
source set.  Every Isaac child has a unique run/output path, is started in its
own process session, and can be recovered after the launcher or SSH session is
lost.  Queue state is replaced atomically after every transition.
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
from typing import Any, Iterator


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
)
from drone_evaluate import _validate_scenario_part  # noqa: E402
from drone_evaluation_protocol import load_protocol  # noqa: E402
from drone_score_one_seed_comparison import score_evaluation  # noqa: E402
from drone_train import standalone_resolved_config  # noqa: E402
from g1_fly_control.crazyflie.controllers import build_controller  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "experiments" / "crazyflie_neural_comparison_seed0_500k.json"
)
ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
TASKS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
)
CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
    "gru_matched",
    "mlp_normal",
)
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
TOTAL_INTERACTIONS = 500_000
EPISODES_PER_MATCHING_TASK = 16
QUEUE_FILE_NAME = "queue.json"
QUEUE_SCHEMA_VERSION = 1
TERMINAL_JOB_STATES = {"completed", "failed"}
ACTIVITY_COLLECTOR = (SCRIPT_DIR / "crazyflie_neural_activity.py").resolve()
ACTIVITY_VALIDATOR = (SCRIPT_DIR / "summarize_neural_comparison.py").resolve()
PARALLEL_GATE_RECEIPT = (
    ROOT
    / "runs"
    / "crazyflie_neural_comparison_seed0_500k_prelaunch_20260918"
    / "parallel_exact_2x40_v3"
    / "parallel_gate_receipt.json"
).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace one JSON object without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
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


def _parallel_gate_record() -> dict[str, Any]:
    """Authenticate the optional exact 2x40 gate against the current tree."""

    record: dict[str, Any] = {
        "path": str(PARALLEL_GATE_RECEIPT),
        "status": "missing",
        "sha256": None,
        "payload_sha256": None,
    }
    if not PARALLEL_GATE_RECEIPT.is_file():
        return record
    from crazyflie_parallel_gate import validate_receipt

    payload = validate_receipt(PARALLEL_GATE_RECEIPT, config_path=DEFAULT_CONFIG)
    record.update(
        status="PASS",
        sha256=sha256_file(PARALLEL_GATE_RECEIPT),
        payload_sha256=canonical_sha256(payload),
    )
    return record


def _expected_config_keys() -> set[str]:
    return {
        "schema_version",
        "label",
        "output_root",
        "isaac_python",
        "contract_profile",
        "tasks",
        "controllers",
        "seeds",
        "total_interactions_per_job",
        "connectomes",
        "rewire",
        "training",
        "evaluation",
        "analysis",
        "queue",
        "comparison_contract",
    }


def validate_config(path: Path) -> dict[str, Any]:
    """Validate the exact reviewed 18-cell comparison declaration."""

    path = path.expanduser().resolve()
    config = _read_json(path)
    if set(config) != _expected_config_keys():
        raise ValueError("Comparison config top-level fields differ from schema v1")
    exact = {
        "schema_version": 1,
        "label": "crazyflie_neural_comparison_seed0_500k",
        "output_root": "runs/crazyflie_neural_comparison_seed0_500k",
        "isaac_python": str(ISAAC_PYTHON),
        "contract_profile": "balanced_v4",
        "tasks": list(TASKS),
        "controllers": list(CONTROLLERS),
        "seeds": [0],
        "total_interactions_per_job": TOTAL_INTERACTIONS,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be exactly {expected!r}")

    training = config.get("training")
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
        "checkpoint_every_updates": 100,
        "precision": "float32",
        "device": "cuda:0",
    }
    if training != expected_training:
        raise ValueError("training must equal the reviewed 40-env balanced-v4 PPO contract")
    if TOTAL_INTERACTIONS % (training["num_envs"] * training["horizon"]):
        raise ValueError("Interaction budget must divide exactly by num_envs*horizon")

    expected_queue = {
        "default_max_parallel": 1,
        "allowed_max_parallel": [1, 2],
        "controller_barriers": True,
        "lif_first": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
    }
    if config.get("queue") != expected_queue:
        raise ValueError("queue resource/order contract differs from the reviewed schema")
    expected_evaluation = {
        "protocol": "main",
        "manifest": "configs/experiments/crazyflie_eval_manifest_v1.json",
        "manifest_sha256": "88ec44f48c289be29bad37aff19590f703b8fd92ca38a98e8b23edfadbf2f1f0",
        "episodes_per_matching_task": EPISODES_PER_MATCHING_TASK,
        "matching_task_only": True,
        "deterministic_actions": True,
        "device": "cuda:0",
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("evaluation must be the 16-episode matching-task main protocol")
    expected_analysis = {
        "neural_activity_replay": True,
        "script": "scripts/crazyflie_neural_activity.py",
        "protocol": "main",
        "episodes_per_matching_task": EPISODES_PER_MATCHING_TASK,
    }
    if config.get("analysis") != expected_analysis:
        raise ValueError("neural activity replay declaration changed")
    expected_comparison = {
        "same_task_budget_seed_and_ppo_hyperparameters": True,
        "numerically_matched_single_core_and_baselines": [
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

    connectomes = config.get("connectomes")
    expected_connectomes = {
        "leg_manifest": "data/connectome/manifest.json",
        "leg_manifest_sha256": "c59c1641847ce7e81f01778df1ee4a136124d62bc713abde750b0d1e9836832c",
        "wing_manifest": "data/connectome_wing/manifest.json",
        "wing_manifest_sha256": "8e0149a10094109326618d5b1b042175c7c4e45307c1b7210ccf0c71cc0bf5a6",
    }
    if connectomes != expected_connectomes:
        raise ValueError("connectome identities differ from the reviewed declaration")
    rewire = config.get("rewire")
    expected_rewire = {
        "seed": 20260916,
        "manifest": "configs/experiments/crazyflie_rewire_seed_20260916.json",
        "manifest_sha256": "6c2a10b879d22741b0d17110d052953adc4f081e9ecac636fabdf0ba68d57b14",
    }
    if rewire != expected_rewire:
        raise ValueError("degree-preserving rewire identity changed")

    for relative, expected_hash in (
        (connectomes["leg_manifest"], connectomes["leg_manifest_sha256"]),
        (connectomes["wing_manifest"], connectomes["wing_manifest_sha256"]),
        (rewire["manifest"], rewire["manifest_sha256"]),
        (expected_evaluation["manifest"], expected_evaluation["manifest_sha256"]),
    ):
        artifact = (ROOT / relative).resolve()
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Pinned artifact is missing or changed: {artifact}")
    if not ISAAC_PYTHON.is_file() or not os.access(ISAAC_PYTHON, os.X_OK):
        raise ValueError(f"Required Isaac Python is not executable: {ISAAC_PYTHON}")
    protocol = load_protocol("main")
    if protocol.get("episodes_per_scenario") != EPISODES_PER_MATCHING_TASK:
        raise ValueError("Resolved main protocol does not contain 16 episodes per scenario")

    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    config["_output_root"] = str((ROOT / config["output_root"]).resolve())
    config["_leg_manifest"] = str((ROOT / connectomes["leg_manifest"]).resolve())
    config["_wing_manifest"] = str((ROOT / connectomes["wing_manifest"]).resolve())
    config["_rewire_manifest"] = str((ROOT / rewire["manifest"]).resolve())
    config["_evaluation_manifest_id"] = protocol["manifest_id"]
    return config


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _training_args(config: dict[str, Any], task: str, controller: str) -> SimpleNamespace:
    training = config["training"]
    return SimpleNamespace(
        task=task,
        contract_profile=config["contract_profile"],
        policy=controller,
        seed=0,
        num_envs=training["num_envs"],
        total_interactions=config["total_interactions_per_job"],
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
        evaluation_protocol=config["evaluation"]["protocol"],
        warm_start_checkpoint=None,
    )


def _job_fingerprint(config: dict[str, Any], task: str, controller: str) -> tuple[str, dict[str, Any]]:
    args = _training_args(config, task, controller)
    resolved, protocol = standalone_resolved_config(args)
    rewired = load_fingerprint_rewire_manifest(
        args.rewire_manifest,
        expected_file_sha256=config["rewire"]["manifest_sha256"],
        expected_seed=args.rewire_seed,
    )
    manifest = args.wing_connectome_manifest if controller == "wing_lif" else args.connectome_manifest
    return reproduction_fingerprint(
        resolved_config=resolved,
        evaluation_manifest=protocol,
        connectome_manifest=manifest,
        rewired_manifest=rewired,
    )


def _training_command(
    config: dict[str, Any], task: str, controller: str, run_dir: Path,
    fingerprint: str, pause_file: Path,
) -> list[str]:
    training = config["training"]
    return [
        config["isaac_python"],
        str(ROOT / "scripts" / "drone_train.py"),
        "--task", task,
        "--contract_profile", config["contract_profile"],
        "--policy", controller,
        "--seed", "0",
        "--num_envs", str(training["num_envs"]),
        "--total_interactions", str(config["total_interactions_per_job"]),
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
        "--evaluation_protocol", config["evaluation"]["protocol"],
        "--run_dir", str(run_dir),
        "--expected_fingerprint", fingerprint,
        "--pause_file", str(pause_file),
        "--device", training["device"],
        "--headless",
    ]


def _evaluation_command(
    config: dict[str, Any], controller: str, checkpoint: Path,
    scenario: str, output: Path, fingerprint: str,
) -> list[str]:
    return [
        config["isaac_python"],
        str(ROOT / "scripts" / "drone_evaluate.py"),
        "--checkpoint", str(checkpoint),
        "--protocol", config["evaluation"]["protocol"],
        "--scenario", scenario,
        "--headless",
        "--device", config["evaluation"]["device"],
        "--output", str(output),
        "--expected_fingerprint", fingerprint,
        "--training_seed", "0",
        "--policy", controller,
    ]


def _neural_activity_command(
    config: dict[str, Any], controller: str, checkpoint: Path,
    scenario: str, output: Path, fingerprint: str,
) -> list[str]:
    return [
        config["isaac_python"],
        str(ROOT / config["analysis"]["script"]),
        "--checkpoint", str(checkpoint),
        "--protocol", "main",
        "--scenario", scenario,
        "--output", str(output),
        "--expected_fingerprint", fingerprint,
        "--training_seed", "0",
        "--policy", controller,
        "--headless",
    ]


def _controller_reports(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
                if controller == "frozen_lif_degree_rewired"
                else None
            ),
        )
        reports[controller] = report
    return reports


def build_queue(config: dict[str, Any], output_root: Path | None = None) -> dict[str, Any]:
    """Build the immutable commands and initial state without launching Isaac."""

    root = Path(output_root or config["_output_root"]).resolve()
    pause_file = root / "pause.request"
    if not ACTIVITY_COLLECTOR.is_file():
        raise ValueError(f"Neural-activity collector is missing: {ACTIVITY_COLLECTOR}")
    if not ACTIVITY_VALIDATOR.is_file():
        raise ValueError(f"Neural-activity validator is missing: {ACTIVITY_VALIDATOR}")
    activity_collector_sha256 = sha256_file(ACTIVITY_COLLECTOR)
    activity_validator_sha256 = sha256_file(ACTIVITY_VALIDATOR)
    jobs: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        for task in TASKS:
            short_task = task.removeprefix("FlyCrazyflie-").removesuffix("-v0").lower()
            identifier = f"{controller}__{short_task}__seed-0"
            run_dir = root / "jobs" / identifier
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            evaluation_output = root / "evaluations" / identifier / f"{task}.json"
            activity_output = root / "neural_activity" / identifier / f"{task}.json"
            fingerprint, payload = _job_fingerprint(config, task, controller)
            jobs.append({
                "id": identifier,
                "controller": controller,
                "controller_priority": CONTROLLERS.index(controller),
                "task": task,
                "seed": 0,
                "contract_profile": config["contract_profile"],
                "status": "pending",
                "training_status": "pending",
                "evaluation_status": "pending",
                "neural_activity_status": "pending",
                "total_interactions": config["total_interactions_per_job"],
                "expected_updates": config["total_interactions_per_job"]
                // (config["training"]["num_envs"] * config["training"]["horizon"]),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "training_manifest": str(run_dir / "training_manifest.json"),
                "evaluation_output": str(evaluation_output),
                "neural_activity_output": str(activity_output),
                "activity_collector": str(ACTIVITY_COLLECTOR),
                "activity_collector_sha256": activity_collector_sha256,
                "activity_validator": str(ACTIVITY_VALIDATOR),
                "activity_validator_sha256": activity_validator_sha256,
                "evaluation_manifest_id": config["_evaluation_manifest_id"],
                "expected_fingerprint": fingerprint,
                "fingerprint_payload": payload,
                "training_command": _training_command(
                    config, task, controller, run_dir, fingerprint, pause_file
                ),
                "evaluation_command": _evaluation_command(
                    config, controller, checkpoint, task, evaluation_output, fingerprint
                ),
                "neural_activity_command": _neural_activity_command(
                    config, controller, checkpoint, task, activity_output, fingerprint
                ),
                "attempts": [],
            })
    if len(jobs) != 18 or len({job["id"] for job in jobs}) != 18:
        raise RuntimeError("Comparison queue must contain 18 unique jobs")
    if sum(EPISODES_PER_MATCHING_TASK for _ in jobs) != 288:
        raise RuntimeError("Comparison queue must resolve to 288 matching held-out episodes")
    reports = _controller_reports(config)
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": "crazyflie_neural_comparison_queue_v1",
        "label": config["label"],
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "revision": 0,
        "status": "prepared",
        "config_path": config["_config_path"],
        "config_file_sha256": config["_config_sha256"],
        "config_identity_sha256": canonical_sha256(_public_config(config)),
        "queue_runner": str(Path(__file__).resolve()),
        "queue_runner_sha256": sha256_file(Path(__file__).resolve()),
        "activity_collector": str(ACTIVITY_COLLECTOR),
        "activity_collector_sha256": activity_collector_sha256,
        "activity_validator": str(ACTIVITY_VALIDATOR),
        "activity_validator_sha256": activity_validator_sha256,
        "output_root": str(root),
        "queue_file": str(root / QUEUE_FILE_NAME),
        "pause_file": str(pause_file),
        "job_count": 18,
        "evaluation_bundle_count": 18,
        "predicted_evaluation_episodes": 288,
        "predicted_neural_activity_replay_episodes": 288,
        "controller_order": list(CONTROLLERS),
        "task_order": list(TASKS),
        "controller_barriers": True,
        "parallel_gate": _parallel_gate_record(),
        "resource_limits": {
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "allowed_max_parallel": [1, 2],
            "default_max_parallel": 1,
        },
        "parameter_matching": {
            "numerically_matched_single_core_and_baselines": config["comparison_contract"][
                "numerically_matched_single_core_and_baselines"
            ],
            "wing_extension_parameter_matching_required": False,
            "leg_wing_is_not_parameter_matched": True,
        },
        "controller_reports": reports,
        "counts": {"pending": 18},
        "config": _public_config(config),
        "jobs": jobs,
        "events": [{"utc": _utc_now(), "event": "queue_prepared"}],
    }


def _refresh_queue_status(queue: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in queue["jobs"]:
        status = job["status"]
        counts[status] = counts.get(status, 0) + 1
    queue["counts"] = dict(sorted(counts.items()))
    if counts.get("running"):
        queue["status"] = "running"
    elif counts.get("paused"):
        queue["status"] = "paused"
    elif counts.get("failed"):
        queue["status"] = "failed"
    elif counts.get("completed") == queue["job_count"]:
        queue["status"] = "completed"
    elif queue.get("resource_block"):
        queue["status"] = "blocked_resource"
    elif queue.get("started_utc"):
        queue["status"] = "pending"
    else:
        queue["status"] = "prepared"


def save_queue(path: Path, queue: dict[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_utc"] = _utc_now()
    _refresh_queue_status(queue)
    _atomic_json(path, queue)
    summary = {
        "schema_version": queue["schema_version"],
        "kind": queue["kind"],
        "status": queue["status"],
        "revision": queue["revision"],
        "updated_utc": queue["updated_utc"],
        "counts": queue["counts"],
        "job_count": queue["job_count"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "resource_limits": queue["resource_limits"],
        "resource_block": queue.get("resource_block"),
        "jobs": [
            {
                "id": job["id"],
                "controller": job["controller"],
                "task": job["task"],
                "status": job["status"],
                "training_status": job["training_status"],
                "evaluation_status": job["evaluation_status"],
                "neural_activity_status": job["neural_activity_status"],
                "checkpoint": job["checkpoint"],
            }
            for job in queue["jobs"]
        ],
    }
    _atomic_json(path.with_name("queue_summary.json"), summary)


def _pid_alive(pid: int) -> bool:
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
            previous = _read_json(lock)
            previous_pid = previous.get("pid")
        except (OSError, ValueError):
            previous_pid = None
        if _pid_alive(previous_pid):
            raise RuntimeError(f"Queue runner is already active with PID {previous_pid}")
        stale = lock.with_name(f"queue.lock.stale-{time.time_ns()}")
        os.replace(lock, stale)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        payload = json.dumps({"pid": os.getpid(), "created_utc": _utc_now()}) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        try:
            current = _read_json(lock)
        except (OSError, ValueError):
            current = {}
        if current.get("pid") == os.getpid():
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


def resource_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi telemetry failed: {completed.stderr.strip()}")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("GPU telemetry must resolve exactly CUDA device 0")
    try:
        gpu_used = float(rows[0])
        ram_percent = float(_ram_used_percent())
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Resource telemetry is non-numeric") from exc
    if not math.isfinite(gpu_used) or not math.isfinite(ram_percent):
        raise RuntimeError("Resource telemetry is non-finite")
    return {
        "utc": _utc_now(),
        "gpu_index": 0,
        "gpu_used_mib": gpu_used,
        "system_ram_percent": ram_percent,
        "gpu_limit_mib_exclusive": GPU_LIMIT_MIB,
        "ram_limit_percent_exclusive": RAM_LIMIT_PERCENT,
        "passed": gpu_used < GPU_LIMIT_MIB and ram_percent < RAM_LIMIT_PERCENT,
    }


def gpu_compute_client_pids() -> set[int]:
    """Return current CUDA compute PIDs, failing closed on bad telemetry."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute telemetry failed: {completed.stderr.strip()}")
    result: set[int] = set()
    for row in completed.stdout.splitlines():
        row = row.strip()
        if not row:
            continue
        try:
            pid = int(row)
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned a non-integer compute PID") from exc
        if pid < 1:
            raise RuntimeError("nvidia-smi returned an invalid compute PID")
        result.add(pid)
    return result


def _active_gpu_allocation_observed(active: dict[str, dict[str, Any]]) -> bool:
    """Require every already-launched child to appear in CUDA telemetry."""

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


def valid_training(job: dict[str, Any]) -> bool:
    try:
        manifest = _read_json(Path(job["training_manifest"]))
        checkpoint = Path(job["checkpoint"])
        recorded_checkpoint = Path(manifest["checkpoint"]).resolve()
        actual_hash = sha256_file(checkpoint)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "completed"
        and manifest.get("task") == job["task"]
        and manifest.get("contract_profile") == job["contract_profile"]
        and manifest.get("controller") == job["controller"]
        and manifest.get("seed") == job["seed"]
        and manifest.get("num_envs") == 40
        and manifest.get("horizon") == 100
        and manifest.get("requested_interactions") == job["total_interactions"]
        and manifest.get("environment_interactions") == job["total_interactions"]
        and manifest.get("completed_updates") == job["expected_updates"]
        and manifest.get("fingerprint") == job["expected_fingerprint"]
        and recorded_checkpoint == checkpoint.resolve()
        and manifest.get("checkpoint_sha256") == actual_hash
        and _memory_gate_passed(manifest.get("memory_gate"))
    )


def _strict_evaluation_value(job: dict[str, Any]) -> dict[str, Any]:
    """Load and authenticate one official matching-task evaluation artifact."""

    value = _read_json(Path(job["evaluation_output"]))
    checkpoint = Path(job["checkpoint"])
    protocol = load_protocol("main")
    _validate_scenario_part(
        value,
        scenario=job["task"],
        protocol_name="main",
        protocol=protocol,
        checkpoint=checkpoint,
        expected_fingerprint=job["expected_fingerprint"],
        expected_training_seed=job["seed"],
        expected_policy=job["controller"],
    )
    if value.get("fingerprint_payload") != job.get("fingerprint_payload"):
        raise ValueError("Evaluation fingerprint payload differs from its queued payload")
    if value.get("evaluation_manifest_id") != job.get("evaluation_manifest_id"):
        raise ValueError("Evaluation manifest ID differs from its queued manifest ID")
    # Recompute the frozen 100-point inputs as an additional finite-value and
    # task-semantics gate.  The return value is consumed by activity validation.
    score_evaluation(value, job["task"])
    return value


def valid_evaluation(job: dict[str, Any]) -> bool:
    try:
        _strict_evaluation_value(job)
    except Exception:
        return False
    return True


def valid_neural_activity(job: dict[str, Any]) -> bool:
    try:
        value = _read_json(Path(job["neural_activity_output"]))
        official = _strict_evaluation_value(job)
        checkpoint = Path(job["checkpoint"])
        protocol = load_protocol("main")
        _validate_scenario_part(
            value,
            scenario=job["task"],
            protocol_name="main",
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=job["expected_fingerprint"],
            expected_training_seed=job["seed"],
            expected_policy=job["controller"],
        )
        activity = value["neural_activity"]
        if not isinstance(activity, dict):
            return False
        expected_collector = Path(job["activity_collector"]).resolve()
        if (
            expected_collector != ACTIVITY_COLLECTOR
            or job["activity_collector_sha256"] != sha256_file(ACTIVITY_COLLECTOR)
            or Path(job["activity_validator"]).resolve() != ACTIVITY_VALIDATOR
            or job["activity_validator_sha256"] != sha256_file(ACTIVITY_VALIDATOR)
            or Path(activity.get("collector", "")).resolve() != expected_collector
            or activity.get("collector_sha256") != job["activity_collector_sha256"]
        ):
            return False
        scored = score_evaluation(official, job["task"])
        # Keep the detailed biological/engineering-unit contract in one place.
        # This validates exact deterministic replay equality, per-unit counts,
        # role summaries, denominators, core/layer shape, and collector identity.
        from summarize_neural_comparison import _activity_evidence

        _evidence, issues = _activity_evidence(job, official, scored)
        if issues:
            return False
    except Exception:
        return False
    return True


def _process_identity_alive(job: dict[str, Any]) -> bool:
    active = job.get("active_process")
    if not isinstance(active, dict) or not _pid_alive(active.get("pid")):
        return False
    try:
        raw = Path(f"/proc/{active['pid']}/cmdline").read_bytes()
    except OSError:
        return False
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    if active.get("phase") == "training":
        return "drone_train.py" in command and job["run_dir"] in command
    if active.get("phase") == "evaluation":
        return "drone_evaluate.py" in command and job["evaluation_output"] in command
    return (
        "crazyflie_neural_activity.py" in command
        and job["neural_activity_output"] in command
    )


def reconcile_queue(queue: dict[str, Any], *, retry_failed: bool) -> None:
    """Recover completed artifacts and safely retain detached live children."""

    for job in queue["jobs"]:
        if valid_training(job):
            job["training_status"] = "completed"
            if valid_evaluation(job):
                job["evaluation_status"] = "completed"
                if valid_neural_activity(job):
                    job["neural_activity_status"] = "completed"
                    job["status"] = "completed"
                    job.pop("active_process", None)
                    continue
                job["neural_activity_status"] = "pending"
            else:
                job["evaluation_status"] = "pending"
                job["neural_activity_status"] = "pending"
        else:
            job["training_status"] = "pending"
            job["evaluation_status"] = "pending"
            job["neural_activity_status"] = "pending"
        if _process_identity_alive(job):
            job["status"] = "running"
            job["recovered_detached_process"] = True
            continue
        job.pop("active_process", None)
        if retry_failed or job.get("status") not in {"failed", "paused"}:
            job["status"] = "pending"
            job.pop("failure", None)


def _archive_invalid_output(job: dict[str, Any], phase: str) -> str | None:
    output = Path(job[f"{phase}_output"])
    validator = valid_evaluation if phase == "evaluation" else valid_neural_activity
    if not output.exists() or validator(job):
        return None
    archive_dir = output.parent / "invalid_attempts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{output.stem}-{time.time_ns()}{output.suffix}"
    os.replace(output, archive)
    return str(archive)


def _next_phase(job: dict[str, Any]) -> str | None:
    if not valid_training(job):
        return "training"
    job["training_status"] = "completed"
    if not valid_evaluation(job):
        return "evaluation"
    job["evaluation_status"] = "completed"
    if not valid_neural_activity(job):
        return "neural_activity"
    job["neural_activity_status"] = "completed"
    job["status"] = "completed"
    return None


def _start_child(job: dict[str, Any], phase: str, output_root: Path) -> dict[str, Any]:
    command = list(job[f"{phase}_command"])
    if phase == "training" and Path(job["checkpoint"]).is_file():
        command.append("--resume")
    archived_output = (
        _archive_invalid_output(job, phase)
        if phase in {"evaluation", "neural_activity"}
        else None
    )
    attempt_index = len(job["attempts"]) + 1
    log = output_root / "logs" / f"{job['id']}__{phase}__attempt-{attempt_index:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("xb")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    record = {
        "attempt": attempt_index,
        "phase": phase,
        "pid": process.pid,
        "started_utc": _utc_now(),
        "command": command,
        "log": str(log),
        "archived_invalid_output": archived_output,
    }
    job["attempts"].append(record)
    job["active_process"] = {
        "pid": process.pid,
        "phase": phase,
        "started_utc": record["started_utc"],
        "log": str(log),
    }
    job["status"] = "running"
    job[f"{phase}_status"] = "running"
    return {"process": process, "stream": stream, "phase": phase, "job": job, "record": record}


def _finish_child(handle: dict[str, Any], exit_code: int | None) -> None:
    job = handle["job"]
    phase = handle["phase"]
    record = handle["record"]
    stream = handle.get("stream")
    if stream is not None:
        stream.close()
    record["finished_utc"] = _utc_now()
    record["exit_code"] = exit_code
    job.pop("active_process", None)
    validators = {
        "training": valid_training,
        "evaluation": valid_evaluation,
        "neural_activity": valid_neural_activity,
    }
    valid = validators[phase](job)
    if valid:
        job[f"{phase}_status"] = "completed"
        if phase == "neural_activity":
            job["status"] = "completed"
        else:
            job["status"] = "pending"
        return
    if phase == "training" and exit_code == 3:
        job["training_status"] = "paused"
        job["status"] = "paused"
        return
    job[f"{phase}_status"] = "failed"
    job["status"] = "failed"
    job["failure"] = (
        f"{phase} attempt exited {exit_code!r} and its artifact did not pass validation"
    )


def _current_controller(queue: dict[str, Any]) -> str | None:
    for controller in CONTROLLERS:
        if any(
            job["controller"] == controller
            and job["status"] not in TERMINAL_JOB_STATES
            for job in queue["jobs"]
        ):
            return controller
    return None


def _request_pause(queue: dict[str, Any], queue_path: Path, reason: str) -> Path:
    path = Path(queue["pause_file"])
    if not path.exists():
        _atomic_json(path, {
            "schema_version": 1,
            "status": "requested",
            "requested_utc": _utc_now(),
            "requesting_pid": os.getpid(),
            "reason": reason,
            "queue": str(queue_path),
        })
    return path


def _consume_pause(queue: dict[str, Any]) -> None:
    path = Path(queue["pause_file"])
    if path.exists():
        archive_dir = path.parent / "pause_requests"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"pause-request-consumed-{time.time_ns()}.json"
        os.replace(path, archive)
        queue["events"].append({
            "utc": _utc_now(),
            "event": "pause_request_consumed",
            "archive": str(archive),
        })


def execute_queue(
    queue: dict[str, Any], queue_path: Path, *, max_parallel: int, resume: bool,
) -> int:
    if max_parallel not in {1, 2}:
        raise ValueError("max_parallel must be 1 or 2")
    if max_parallel == 2:
        gate = _parallel_gate_record()
        if gate.get("status") != "PASS" or gate != queue.get("parallel_gate"):
            raise ValueError(
                "max_parallel=2 requires the current authenticated exact 2x40 gate receipt"
            )
    if resume:
        _consume_pause(queue)
    elif Path(queue["pause_file"]).exists():
        raise RuntimeError("Pause request exists; use --resume to consume and continue it")
    reconcile_queue(queue, retry_failed=resume)
    queue.pop("resource_block", None)
    queue.setdefault("started_utc", _utc_now())
    queue["last_runner"] = {
        "pid": os.getpid(),
        "started_utc": _utc_now(),
        "max_parallel": max_parallel,
        "resume": resume,
    }
    queue["events"].append({
        "utc": _utc_now(), "event": "runner_started", "pid": os.getpid(),
        "max_parallel": max_parallel, "resume": resume,
    })
    save_queue(queue_path, queue)

    stop_requested = False

    def handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        _request_pause(queue, queue_path, f"signal_{signum}")

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handle_stop)
    if hasattr(signal, "SIGHUP"):
        previous_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)

    active: dict[str, dict[str, Any]] = {}
    for job in queue["jobs"]:
        if job.get("status") == "running" and _process_identity_alive(job):
            active[job["id"]] = {
                "process": None,
                "stream": None,
                "phase": job["active_process"]["phase"],
                "job": job,
                "record": {
                    "attempt": None,
                    "phase": job["active_process"]["phase"],
                    "pid": job["active_process"]["pid"],
                    "started_utc": job["active_process"].get("started_utc"),
                    "log": job["active_process"].get("log"),
                    "recovered_by_pid_monitor": True,
                },
            }
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

            if stop_requested or Path(queue["pause_file"]).exists():
                if not active:
                    for job in queue["jobs"]:
                        if job["status"] == "pending":
                            job["status"] = "paused"
                    queue["events"].append({"utc": _utc_now(), "event": "runner_paused"})
                    save_queue(queue_path, queue)
                    return 3
            else:
                controller = _current_controller(queue)
                if controller is not None:
                    candidates = [
                        job for job in queue["jobs"]
                        if job["controller"] == controller and job["status"] == "pending"
                    ]
                    while candidates and len(active) < max_parallel:
                        # Do not make the second admission decision against the
                        # pre-allocation baseline.  Wait until every existing
                        # child is visible as an NVIDIA compute client, then
                        # take the device-wide resource snapshot below.
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
                        snapshot = resource_snapshot()
                        queue["last_resource_precheck"] = snapshot
                        queue.setdefault("resource_prechecks", []).append(snapshot)
                        if len(queue["resource_prechecks"]) > 100:
                            queue["resource_prechecks"] = queue["resource_prechecks"][-100:]
                        if not snapshot["passed"]:
                            if not active:
                                queue["resource_block"] = snapshot
                                queue["events"].append({
                                    "utc": _utc_now(),
                                    "event": "resource_precheck_blocked",
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
                controller = _current_controller(queue)
                if controller is None:
                    queue["finished_utc"] = _utc_now()
                    queue["events"].append({"utc": _utc_now(), "event": "runner_finished"})
                    save_queue(queue_path, queue)
                    return 0 if queue["status"] == "completed" else 1
                # Only failed/terminal cells may remain in the current phase;
                # the barrier can advance on the next loop.
                if not any(job["status"] == "pending" for job in queue["jobs"]):
                    queue["finished_utc"] = _utc_now()
                    save_queue(queue_path, queue)
                    return 1
            time.sleep(float(queue["config"]["queue"]["resource_poll_interval_seconds"]))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _validate_loaded_queue(queue: dict[str, Any], config: dict[str, Any], path: Path) -> None:
    if (
        queue.get("schema_version") != QUEUE_SCHEMA_VERSION
        or queue.get("kind") != "crazyflie_neural_comparison_queue_v1"
        or queue.get("config_file_sha256") != config["_config_sha256"]
        or queue.get("config_identity_sha256") != canonical_sha256(_public_config(config))
        or Path(queue.get("queue_runner", "")).resolve() != Path(__file__).resolve()
        or queue.get("queue_runner_sha256") != sha256_file(Path(__file__).resolve())
        or Path(queue.get("activity_collector", "")).resolve() != ACTIVITY_COLLECTOR
        or queue.get("activity_collector_sha256") != sha256_file(ACTIVITY_COLLECTOR)
        or Path(queue.get("activity_validator", "")).resolve() != ACTIVITY_VALIDATOR
        or queue.get("activity_validator_sha256") != sha256_file(ACTIVITY_VALIDATOR)
        or queue.get("parallel_gate") != _parallel_gate_record()
        or Path(queue.get("queue_file", "")).resolve() != path.resolve()
        or queue.get("job_count") != 18
        or queue.get("predicted_evaluation_episodes") != 288
        or [job.get("controller") for job in queue.get("jobs", [])]
        != [controller for controller in CONTROLLERS for _ in TASKS]
        or [job.get("task") for job in queue.get("jobs", [])]
        != [task for _ in CONTROLLERS for task in TASKS]
    ):
        raise ValueError("Existing queue does not match the reviewed config/order")
    for job in queue["jobs"]:
        current_fingerprint, current_payload = _job_fingerprint(
            config, job["task"], job["controller"]
        )
        if (
            job.get("expected_fingerprint") != current_fingerprint
            or job.get("fingerprint_payload") != current_payload
            or Path(job.get("activity_collector", "")).resolve() != ACTIVITY_COLLECTOR
            or job.get("activity_collector_sha256")
            != queue.get("activity_collector_sha256")
            or Path(job.get("activity_validator", "")).resolve() != ACTIVITY_VALIDATOR
            or job.get("activity_validator_sha256")
            != queue.get("activity_validator_sha256")
        ):
            raise ValueError(
                "Existing queue fingerprint is stale for current source/runtime: "
                f"{job.get('id')}"
            )


def _status_payload(queue: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": queue["status"],
        "queue": queue["queue_file"],
        "revision": queue["revision"],
        "counts": queue["counts"],
        "jobs": queue["job_count"],
        "evaluation_episodes": queue["predicted_evaluation_episodes"],
        "last_resource_precheck": queue.get("last_resource_precheck"),
    }


def _dry_run_payload(queue: dict[str, Any]) -> dict[str, Any]:
    """Expose every immutable cell and command without writing queue state."""

    return {
        "status": "dry_run_only_no_processes_launched",
        "job_count": queue["job_count"],
        "training_interactions_per_job": TOTAL_INTERACTIONS,
        "predicted_training_interactions": queue["job_count"] * TOTAL_INTERACTIONS,
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "predicted_neural_activity_replay_episodes": queue[
            "predicted_neural_activity_replay_episodes"
        ],
        "output_root": queue["output_root"],
        "cells": [
            {
                "id": job["id"],
                "controller": job["controller"],
                "task": job["task"],
                "run_dir": job["run_dir"],
                "checkpoint": job["checkpoint"],
                "evaluation_output": job["evaluation_output"],
                "neural_activity_output": job["neural_activity_output"],
                "training_command": job["training_command"],
                "evaluation_command": job["evaluation_command"],
                "neural_activity_command": job["neural_activity_command"],
            }
            for job in queue["jobs"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all 18 immutable cells and commands without writing or launching",
    )
    mode.add_argument("--prepare", action="store_true", help="Create the new queue; never overwrite")
    mode.add_argument("--run", action="store_true", help="Run pending cells in controller-priority order")
    mode.add_argument("--resume", action="store_true", help="Retry failed/paused cells and resume checkpoints")
    mode.add_argument("--status", action="store_true", help="Read queue state without launching Isaac or mutating it")
    mode.add_argument("--pause", action="store_true", help="Request a clean training checkpoint pause")
    parser.add_argument("--max_parallel", type=int, choices=(1, 2))
    args = parser.parse_args()
    try:
        config = validate_config(args.config)
        output_root = Path(config["_output_root"])
        queue_path = output_root / QUEUE_FILE_NAME
        if args.dry_run:
            queue = build_queue(config, output_root)
            print(json.dumps(_dry_run_payload(queue), indent=2, sort_keys=True))
            return 0
        if args.prepare:
            if output_root.exists():
                raise ValueError(
                    f"Output root already exists and will not be overwritten: {output_root}"
                )
            queue = build_queue(config, output_root)
            save_queue(queue_path, queue)
            print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
            return 0
        if not queue_path.is_file():
            raise ValueError(f"Queue does not exist; run --prepare first: {queue_path}")
        queue = _read_json(queue_path)
        _validate_loaded_queue(queue, config, queue_path)
        if args.pause:
            pause_file = _request_pause(queue, queue_path, "explicit_cli_request")
            print(json.dumps({"status": "pause_requested", "pause_file": str(pause_file)}, indent=2))
            return 0
        if args.status:
            print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
            return 0
        max_parallel = args.max_parallel or config["queue"]["default_max_parallel"]
        with queue_lock(queue_path):
            result = execute_queue(
                queue, queue_path, max_parallel=max_parallel, resume=bool(args.resume)
            )
        print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
        return result
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
