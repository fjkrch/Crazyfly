#!/usr/bin/env python3
"""Run and authenticate the exact two-process Crazyflie concurrency gate.

The gate is deliberately separate from the experiment queue.  It launches two
fixed, current-shape, 40-environment balanced-v4 trainers at the same time and
records device-wide telemetry until both pause at their first clean update.
It publishes exact sequential resume commands for the checkpoint/resume gate.
A passing receipt is only emitted when both pause boundaries validate, both
initial processes are observed as GPU compute clients simultaneously, and the
external RAM/GPU/paging limits pass.

``validate_receipt`` is CPU-only and is the public fail-closed API for a queue
launcher that wants to authorize ``max_parallel=2``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source" / "g1_fly_control"
for import_path in (SCRIPT_DIR, SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from drone_bootstrap import (  # noqa: E402
    DEFAULT_CONNECTOME,
    DEFAULT_REWIRE_MANIFEST,
    DEFAULT_WING_CONNECTOME,
    canonical_sha256,
    sha256_file,
    source_hashes,
)


ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
TRAIN_SCRIPT = ROOT / "scripts" / "drone_train.py"
QUEUE_RUNNER = ROOT / "scripts" / "crazyflie_neural_comparison_queue.py"
ACTIVITY_COLLECTOR = ROOT / "scripts" / "crazyflie_neural_activity.py"
ACTIVITY_VALIDATOR = ROOT / "scripts" / "summarize_neural_comparison.py"
DEFAULT_CONFIG = (
    ROOT / "configs" / "experiments" / "crazyflie_neural_comparison_seed0_500k.json"
)
RECEIPT_NAME = "parallel_gate_receipt.json"
GATE_KIND = "crazyflie_exact_two_process_parallel_gate_v1"
GATE_SCHEMA_VERSION = 1
GPU_INDEX = 0
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
SWAP_OUT_GROWTH_TOLERANCE_MIB = 1.0
SUSTAINED_SWAP_SAMPLE_COUNT = 4
SAMPLE_INTERVAL_SECONDS = 0.5
MINIMUM_TELEMETRY_SAMPLES = 4
TASK_SPECS = (
    ("reach", "FlyCrazyflie-WaypointReach-v0"),
    ("switch", "FlyCrazyflie-WaypointSwitch-v0"),
)
POLICY = "frozen_lif_original"
CONTRACT_PROFILE = "balanced_v4"
NUM_ENVS = 40
SEED = 0
REWIRE_SEED = 20260916
TOTAL_INTERACTIONS = 8000
HORIZON = 100
MICROBATCH_SIZE = 40
PPO_EPOCHS = 2
LEARNING_RATE = 0.0003
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.002
MAX_GRAD_NORM = 1.0
TARGET_KL = 0.05
# Match the reviewed 500k main contract.  The explicit pause at update one
# still creates an immutable update-00000001.pt boundary even though periodic
# cadence is 100 updates.
CHECKPOINT_EVERY_UPDATES = 100
PAUSE_AFTER_UPDATES = 1
INTERACTIONS_PER_UPDATE = NUM_ENVS * HORIZON
EXPECTED_TOTAL_UPDATES = TOTAL_INTERACTIONS // INTERACTIONS_PER_UPDATE
PAUSE_EXIT_CODE = 3


class GateValidationError(ValueError):
    """Raised when a concurrency-gate receipt is incomplete or stale."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateValidationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateValidationError(f"Expected a JSON object in {path}")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one durable JSON file without replacing any existing artifact."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # The destination was checked above and the gate owns its newly made
        # output directory.  link+unlink retains no-overwrite semantics even
        # if another writer races us between the check and commit.
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateValidationError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise GateValidationError(f"{location} must be a finite number")
    if minimum is not None and number < minimum:
        raise GateValidationError(f"{location} must be >= {minimum}")
    return number


def _canonical_config_identity(config_path: Path) -> tuple[str, str]:
    config = _read_json(config_path)
    return sha256_file(config_path), canonical_sha256(config)


def trainer_commands(output_dir: Path, *, resume: bool = False) -> list[dict[str, Any]]:
    """Return the immutable bounded trainer commands and isolated paths."""

    output_dir = output_dir.expanduser().resolve()
    commands: list[dict[str, Any]] = []
    for label, task in TASK_SPECS:
        run_dir = output_dir / "trainers" / label
        manifest = run_dir / "training_manifest.json"
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        numbered_checkpoint = (
            run_dir / "checkpoints" / f"update-{EXPECTED_TOTAL_UPDATES:08d}.pt"
            if resume
            else run_dir / "checkpoints" / f"update-{PAUSE_AFTER_UPDATES:08d}.pt"
        )
        log = output_dir / (f"{label}_resume.log" if resume else f"{label}_initial.log")
        command = [
            str(ISAAC_PYTHON),
            str(TRAIN_SCRIPT),
            "--task", task,
            "--contract_profile", CONTRACT_PROFILE,
            "--policy", POLICY,
            "--seed", str(SEED),
            "--num_envs", str(NUM_ENVS),
            "--total_interactions", str(TOTAL_INTERACTIONS),
            "--horizon", str(HORIZON),
            "--microbatch_size", str(MICROBATCH_SIZE),
            "--ppo_epochs", str(PPO_EPOCHS),
            "--learning_rate", str(LEARNING_RATE),
            "--gamma", str(GAMMA),
            "--gae_lambda", str(GAE_LAMBDA),
            "--clip_ratio", str(CLIP_RATIO),
            "--value_coefficient", str(VALUE_COEFFICIENT),
            "--entropy_coefficient", str(ENTROPY_COEFFICIENT),
            "--max_grad_norm", str(MAX_GRAD_NORM),
            "--target_kl", str(TARGET_KL),
            "--checkpoint_every_updates", str(CHECKPOINT_EVERY_UPDATES),
            "--connectome_manifest", str(DEFAULT_CONNECTOME.resolve()),
            "--wing_connectome_manifest", str(DEFAULT_WING_CONNECTOME.resolve()),
            "--rewire_seed", str(REWIRE_SEED),
            "--rewire_manifest", str(DEFAULT_REWIRE_MANIFEST.resolve()),
            "--evaluation_protocol", "main",
            "--run_dir", str(run_dir),
            "--device", "cuda:0",
            "--headless",
        ]
        if resume:
            command.append("--resume")
        else:
            command.extend(("--pause_after_updates", str(PAUSE_AFTER_UPDATES)))
        commands.append(
            {
                "label": label,
                "task": task,
                "stage": "resume" if resume else "initial_pause",
                "command": command,
                "run_dir": str(run_dir),
                "manifest_path": str(manifest),
                "checkpoint_path": str(checkpoint),
                "numbered_checkpoint_path": str(numbered_checkpoint),
                "log_path": str(log),
            }
        )
    return commands


def _parse_csv_rows(text: str) -> list[list[str]]:
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(text), skipinitialspace=True)
        if row
    ]


def _run_nvidia_query(query: str) -> list[list[str]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi telemetry unavailable: {exc}") from exc
    if completed.returncode:
        raise RuntimeError(
            f"nvidia-smi telemetry failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return _parse_csv_rows(completed.stdout)


def _descendant_pids(root_pid: int) -> list[int]:
    try:
        root = psutil.Process(root_pid)
        result = {root_pid, *(child.pid for child in root.children(recursive=True))}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        result = {root_pid}
    return sorted(result)


def telemetry_sample(processes: Mapping[str, subprocess.Popen[Any]]) -> dict[str, Any]:
    """Capture one device-wide resource and tracked-process sample."""

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu_rows = _run_nvidia_query(
        "--query-gpu=index,uuid,name,memory.total,memory.used"
    )
    devices: list[dict[str, Any]] = []
    for row in gpu_rows:
        if len(row) != 5:
            raise RuntimeError("Malformed nvidia-smi GPU telemetry row")
        index, uuid, name, total, used = row
        devices.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_mib": float(total),
                "used_mib": float(used),
            }
        )

    compute_rows = _run_nvidia_query(
        "--query-compute-apps=pid,gpu_uuid,used_memory"
    )
    compute_processes: list[dict[str, Any]] = []
    for row in compute_rows:
        if len(row) != 3:
            raise RuntimeError("Malformed nvidia-smi compute-process telemetry row")
        pid, uuid, used = row
        compute_processes.append(
            {"pid": int(pid), "gpu_uuid": uuid, "used_mib": float(used)}
        )

    tracked: dict[str, Any] = {}
    compute_pids = {row["pid"] for row in compute_processes}
    for label, process in processes.items():
        tree = _descendant_pids(process.pid)
        tracked[label] = {
            "root_pid": process.pid,
            "root_alive": process.poll() is None,
            "process_tree_pids": tree,
            "matched_compute_pids": sorted(compute_pids.intersection(tree)),
        }
    return {
        "timestamp_utc": _utc_now(),
        "monotonic_seconds": time.monotonic(),
        "system_ram_percent": float(memory.percent),
        "system_ram_used_mib": memory.used / (1024**2),
        "system_swap_out_mib": swap.sout / (1024**2),
        "gpu_devices": devices,
        "compute_processes": compute_processes,
        "tracked_processes": tracked,
    }


def analyze_telemetry(
    samples: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] = tuple(label for label, _ in TASK_SPECS),
    expected_root_pids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Recompute all external memory and simultaneous-overlap evidence."""

    if len(samples) < MINIMUM_TELEMETRY_SAMPLES:
        raise GateValidationError(
            f"At least {MINIMUM_TELEMETRY_SAMPLES} telemetry samples are required"
        )
    max_gpu = 0.0
    max_ram = 0.0
    swap_values: list[float] = []
    overlap_indices: list[int] = []
    compute_overlap_indices: list[int] = []
    timestamps: list[float] = []
    for index, raw in enumerate(samples):
        if not isinstance(raw, Mapping):
            raise GateValidationError(f"telemetry.samples[{index}] must be an object")
        timestamps.append(
            _finite_number(
                raw.get("monotonic_seconds"),
                f"telemetry.samples[{index}].monotonic_seconds",
                minimum=0.0,
            )
        )
        ram = _finite_number(
            raw.get("system_ram_percent"),
            f"telemetry.samples[{index}].system_ram_percent",
            minimum=0.0,
        )
        if ram > 100.0:
            raise GateValidationError(
                f"telemetry.samples[{index}].system_ram_percent must be <= 100"
            )
        max_ram = max(max_ram, ram)
        swap_values.append(
            _finite_number(
                raw.get("system_swap_out_mib"),
                f"telemetry.samples[{index}].system_swap_out_mib",
                minimum=0.0,
            )
        )
        devices = raw.get("gpu_devices")
        if not isinstance(devices, list) or not devices:
            raise GateValidationError(
                f"telemetry.samples[{index}].gpu_devices must be a non-empty list"
            )
        selected = [device for device in devices if device.get("index") == GPU_INDEX]
        if len(selected) != 1:
            raise GateValidationError(
                f"telemetry.samples[{index}] must contain GPU index {GPU_INDEX} exactly once"
            )
        max_gpu = max(
            max_gpu,
            _finite_number(
                selected[0].get("used_mib"),
                f"telemetry.samples[{index}].gpu_devices[{GPU_INDEX}].used_mib",
                minimum=0.0,
            ),
        )
        tracked = raw.get("tracked_processes")
        if not isinstance(tracked, Mapping) or set(tracked) != set(labels):
            raise GateValidationError(
                f"telemetry.samples[{index}].tracked_processes has wrong labels"
            )
        compute = raw.get("compute_processes")
        if not isinstance(compute, list):
            raise GateValidationError(
                f"telemetry.samples[{index}].compute_processes must be a list"
            )
        compute_pids: set[int] = set()
        for compute_index, process in enumerate(compute):
            if not isinstance(process, Mapping):
                raise GateValidationError(
                    f"telemetry.samples[{index}].compute_processes[{compute_index}] must be an object"
                )
            pid = process.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
                raise GateValidationError("Compute telemetry contains an invalid PID")
            compute_pids.add(pid)
            _finite_number(
                process.get("used_mib"),
                f"telemetry.samples[{index}].compute_processes[{compute_index}].used_mib",
                minimum=0.0,
            )
        alive = []
        on_gpu = []
        for label in labels:
            entry = tracked[label]
            if not isinstance(entry, Mapping):
                raise GateValidationError(
                    f"telemetry.samples[{index}].tracked_processes.{label} must be an object"
                )
            root_pid = entry.get("root_pid")
            if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid < 1:
                raise GateValidationError("Tracked telemetry contains an invalid root PID")
            if expected_root_pids is not None and root_pid != expected_root_pids.get(label):
                raise GateValidationError("Tracked telemetry root PID changed")
            tree = entry.get("process_tree_pids")
            if not isinstance(tree, list) or root_pid not in tree or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in tree
            ):
                raise GateValidationError("Tracked telemetry contains an invalid process tree")
            alive.append(entry.get("root_alive") is True)
            matched = entry.get("matched_compute_pids")
            if not isinstance(matched, list) or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
                for pid in matched
            ):
                raise GateValidationError(
                    f"telemetry.samples[{index}].tracked_processes.{label}.matched_compute_pids "
                    "must be a PID list"
                )
            if sorted(compute_pids.intersection(tree)) != matched:
                raise GateValidationError("Tracked compute PID intersection is inconsistent")
            on_gpu.append(bool(matched))
        if all(alive):
            overlap_indices.append(index)
        if all(alive) and all(on_gpu):
            compute_overlap_indices.append(index)
    if any(second < first for first, second in zip(timestamps, timestamps[1:])):
        raise GateValidationError("Telemetry monotonic timestamps moved backwards")

    paging_windows: list[dict[str, Any]] = []
    width = SUSTAINED_SWAP_SAMPLE_COUNT
    for start in range(len(swap_values) - width + 1):
        window = swap_values[start : start + width]
        if (
            all(second > first for first, second in zip(window, window[1:]))
            and window[-1] - window[0] > SWAP_OUT_GROWTH_TOLERANCE_MIB
        ):
            paging_windows.append(
                {
                    "start_sample_index": start,
                    "end_sample_index": start + width - 1,
                    "growth_mib": window[-1] - window[0],
                    "values_mib": window,
                }
            )
    return {
        "sample_count": len(samples),
        "max_device_gpu_used_mib": max_gpu,
        "max_system_ram_percent": max_ram,
        "first_swap_out_mib": swap_values[0],
        "last_swap_out_mib": swap_values[-1],
        "simultaneous_process_overlap_sample_indices": overlap_indices,
        "simultaneous_process_overlap_sample_count": len(overlap_indices),
        "simultaneous_gpu_compute_overlap_sample_indices": compute_overlap_indices,
        "simultaneous_gpu_compute_overlap_sample_count": len(compute_overlap_indices),
        "sustained_swap_out_windows": paging_windows,
        "sustained_swap_out_detected": bool(paging_windows),
        "gpu_limit_passed": max_gpu < GPU_LIMIT_MIB,
        "ram_limit_passed": max_ram < RAM_LIMIT_PERCENT,
        "simultaneous_gpu_compute_overlap_passed": bool(compute_overlap_indices),
        "no_new_sustained_swap_out_passed": not paging_windows,
    }


def _read_clean_checkpoint(path: Path) -> dict[str, Any]:
    """Load and internally validate one checkpoint on CPU."""

    from g1_fly_control.crazyflie.checkpoint import read_checkpoint

    payload = read_checkpoint(path, map_location="cpu", resolve_external_history=True)
    import torch

    def require_finite(value: Any, location: str) -> None:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise GateValidationError(f"Checkpoint tensor {location} is nonfinite")
        elif isinstance(value, Mapping):
            for key, child in value.items():
                require_finite(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                require_finite(child, f"{location}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise GateValidationError(f"Checkpoint value {location} is nonfinite")

    for field in (
        "policy_state", "optimizer_state", "scheduler_state", "normalizers",
        "recurrent_state", "history", "metadata",
    ):
        require_finite(payload.get(field), field)
    return payload


def validate_paused_training(
    manifest: Mapping[str, Any],
    *,
    task: str,
    run_dir: Path,
    checkpoint_path: Path,
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Strictly validate one clean one-update pause boundary."""

    expected = {
        "schema_version": 1,
        "status": "paused",
        "contract_profile": CONTRACT_PROFILE,
        "controller": POLICY,
        "seed": SEED,
        "requested_interactions": TOTAL_INTERACTIONS,
        "environment_interactions": INTERACTIONS_PER_UPDATE,
        "completed_updates": PAUSE_AFTER_UPDATES,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise GateValidationError(
                f"Training manifest {task} field {field!r} must equal {value!r}"
            )
    if Path(str(manifest.get("checkpoint", ""))).resolve() != (run_dir / "checkpoints" / "latest.pt"):
        raise GateValidationError(f"Training manifest {task} has the wrong latest checkpoint path")
    checkpoint_hash = sha256_file(checkpoint_path)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise GateValidationError(f"Training manifest {task} checkpoint hash mismatch")
    memory = manifest.get("memory_gate")
    if not isinstance(memory, Mapping) or memory.get("passed") is not True:
        raise GateValidationError(f"Training manifest {task} failed its memory gate")
    if memory.get("device_gpu_telemetry_complete") is not True:
        raise GateValidationError(f"Training manifest {task} lacks GPU telemetry")
    report_gpu = _finite_number(
        memory.get("max_device_gpu_used_mib"),
        f"{task}.memory_gate.max_device_gpu_used_mib",
        minimum=0.0,
    )
    report_ram = _finite_number(
        memory.get("max_system_ram_percent"),
        f"{task}.memory_gate.max_system_ram_percent",
        minimum=0.0,
    )
    if report_gpu >= GPU_LIMIT_MIB or report_ram >= RAM_LIMIT_PERCENT:
        raise GateValidationError(f"Training manifest {task} reached an exclusive memory limit")
    if memory.get("sustained_paging_detected") is not False:
        raise GateValidationError(f"Training manifest {task} detected sustained paging")
    if memory.get("failures") != []:
        raise GateValidationError(f"Training manifest {task} contains memory failures")
    payload = manifest.get("fingerprint_payload")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(payload, Mapping) or not isinstance(fingerprint, str):
        raise GateValidationError(f"Training manifest {task} lacks its fingerprint")
    if canonical_sha256(payload) != fingerprint:
        raise GateValidationError(f"Training manifest {task} fingerprint is not canonical")
    if payload.get("source_sha256") != dict(expected_source_hashes):
        raise GateValidationError(f"Training manifest {task} source hashes are stale")
    resolved = payload.get("resolved_config")
    expected_resolved = {
        "kind": "standalone_training",
        "task": task,
        "contract_profile": CONTRACT_PROFILE,
        "controller": POLICY,
        "num_envs": NUM_ENVS,
        "total_interactions": TOTAL_INTERACTIONS,
        "horizon": HORIZON,
        "microbatch_size": MICROBATCH_SIZE,
        "ppo_epochs": PPO_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
        "seed": SEED,
        "evaluation_protocol": "main",
    }
    if not isinstance(resolved, Mapping) or any(
        resolved.get(field) != value for field, value in expected_resolved.items()
    ):
        raise GateValidationError(f"Training manifest {task} resolved configuration changed")
    checkpoint = _read_clean_checkpoint(checkpoint_path)
    if checkpoint.get("resolved_config") != dict(resolved):
        raise GateValidationError(f"Checkpoint {task} resolved configuration mismatch")
    counters = checkpoint.get("counters")
    if not isinstance(counters, Mapping) or (
        counters.get("completed_updates") != PAUSE_AFTER_UPDATES
        or counters.get("total_interactions") != INTERACTIONS_PER_UPDATE
        or counters.get("resume_count") != 0
    ):
        raise GateValidationError(f"Checkpoint {task} counters are not the clean pause boundary")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("status") != "paused":
        raise GateValidationError(f"Checkpoint {task} metadata is not paused")
    core_before = metadata.get("core_checksum_before")
    if (
        not isinstance(core_before, str)
        or not core_before
        or metadata.get("core_checksum_after") != core_before
        or checkpoint.get("core_checksum") != core_before
    ):
        raise GateValidationError(f"Checkpoint {task} changed its frozen LIF core")
    if checkpoint.get("fingerprints", {}).get("reproduction") != fingerprint:
        raise GateValidationError(f"Checkpoint {task} reproduction fingerprint mismatch")
    return {
        "status": "paused",
        "fingerprint": fingerprint,
        "checkpoint_sha256": checkpoint_hash,
        "completed_updates": PAUSE_AFTER_UPDATES,
        "environment_interactions": INTERACTIONS_PER_UPDATE,
        "max_device_gpu_used_mib": report_gpu,
        "max_system_ram_percent": report_ram,
        "checkpoint_internal_validation": "PASS",
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateValidationError(f"Required artifact is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise GateValidationError(f"Required artifact is empty: {path}")
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": sha256_file(path)}


def _expected_identity(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config_file_hash, config_identity_hash = _canonical_config_identity(config_path)
    current_sources = source_hashes()
    return {
        "config_path": str(config_path),
        "config_file_sha256": config_file_hash,
        "config_canonical_sha256": config_identity_hash,
        "source_sha256": current_sources,
        "source_set_sha256": canonical_sha256(current_sources),
        "gate_script": str(Path(__file__).resolve()),
        "gate_script_sha256": sha256_file(Path(__file__).resolve()),
        "training_script": str(TRAIN_SCRIPT.resolve()),
        "training_script_sha256": sha256_file(TRAIN_SCRIPT),
        "queue_runner": str(QUEUE_RUNNER.resolve()),
        "queue_runner_sha256": sha256_file(QUEUE_RUNNER),
        "activity_collector": str(ACTIVITY_COLLECTOR.resolve()),
        "activity_collector_sha256": sha256_file(ACTIVITY_COLLECTOR),
        "activity_validator": str(ACTIVITY_VALIDATOR.resolve()),
        "activity_validator_sha256": sha256_file(ACTIVITY_VALIDATOR),
        "isaac_python": str(ISAAC_PYTHON),
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "tasks": [task for _, task in TASK_SPECS],
        "controller": POLICY,
        "contract_profile": CONTRACT_PROFILE,
        "num_envs_per_process": NUM_ENVS,
        "total_interactions_per_trial": TOTAL_INTERACTIONS,
        "horizon": HORIZON,
        "microbatch_size": MICROBATCH_SIZE,
        "ppo_epochs": PPO_EPOCHS,
        "pause_after_updates": PAUSE_AFTER_UPDATES,
        "expected_pause_exit_code": PAUSE_EXIT_CODE,
        "interactions_at_pause": INTERACTIONS_PER_UPDATE,
        "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
        "seed": SEED,
        "gpu_index": GPU_INDEX,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "minimum_telemetry_samples": MINIMUM_TELEMETRY_SAMPLES,
        "swap_out_policy": {
            "sample_count": SUSTAINED_SWAP_SAMPLE_COUNT,
            "strictly_increasing": True,
            "growth_tolerance_mib_exclusive": SWAP_OUT_GROWTH_TOLERANCE_MIB,
            "disposition": "failure",
        },
        "simultaneous_gpu_compute_presence_required": True,
        "sequential_resume_commands_published": True,
    }


def _terminate_owned_processes(processes: Mapping[str, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes.values()
    ):
        time.sleep(0.1)
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _build_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    return {"payload": copied, "payload_sha256": canonical_sha256(copied)}


def run_gate(output_dir: Path, *, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Execute the exact pair and return the newly created receipt path."""

    output_dir = output_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    runs_root = (ROOT / "runs").resolve()
    if output_dir == runs_root or not output_dir.is_relative_to(runs_root):
        raise GateValidationError(f"Gate output must be a child of {runs_root}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / RECEIPT_NAME
    command_specs = trainer_commands(output_dir)
    resume_specs = {spec["label"]: spec for spec in trainer_commands(output_dir, resume=True)}
    identity = _expected_identity(config_path)
    processes: dict[str, subprocess.Popen[Any]] = {}
    log_streams: dict[str, Any] = {}
    launches: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    aborted_for_resource_limit = False
    started_at = _utc_now()
    try:
        for spec in command_specs:
            log_path = Path(spec["log_path"])
            stream = log_path.open("xb")
            log_streams[spec["label"]] = stream
            process = subprocess.Popen(
                spec["command"],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[spec["label"]] = process
            launches.append(
                {
                    **spec,
                    "pid": process.pid,
                    "started_at_utc": _utc_now(),
                    "exit_code": None,
                }
            )

        while any(process.poll() is None for process in processes.values()):
            try:
                sample = telemetry_sample(processes)
                samples.append(sample)
                selected = [
                    device
                    for device in sample["gpu_devices"]
                    if device["index"] == GPU_INDEX
                ]
                if len(selected) != 1:
                    failures.append(f"GPU index {GPU_INDEX} telemetry is incomplete")
                    aborted_for_resource_limit = True
                    _terminate_owned_processes(processes)
                    break
                if selected[0]["used_mib"] >= GPU_LIMIT_MIB:
                    failures.append(
                        f"Device-wide GPU memory reached {selected[0]['used_mib']:.1f} MiB "
                        f"(limit < {GPU_LIMIT_MIB:.1f} MiB)"
                    )
                    aborted_for_resource_limit = True
                    _terminate_owned_processes(processes)
                    break
                if sample["system_ram_percent"] >= RAM_LIMIT_PERCENT:
                    failures.append(
                        f"System RAM reached {sample['system_ram_percent']:.2f}% "
                        f"(limit < {RAM_LIMIT_PERCENT:.2f}%)"
                    )
                    aborted_for_resource_limit = True
                    _terminate_owned_processes(processes)
                    break
                if len(samples) >= SUSTAINED_SWAP_SAMPLE_COUNT:
                    recent = samples[-SUSTAINED_SWAP_SAMPLE_COUNT:]
                    recent_swap = [item["system_swap_out_mib"] for item in recent]
                    if (
                        all(b > a for a, b in zip(recent_swap, recent_swap[1:]))
                        and recent_swap[-1] - recent_swap[0]
                        > SWAP_OUT_GROWTH_TOLERANCE_MIB
                    ):
                        failures.append(
                            "System swap-out rose continuously by "
                            f"{recent_swap[-1] - recent_swap[0]:.3f} MiB"
                        )
                        aborted_for_resource_limit = True
                        _terminate_owned_processes(processes)
                        break
            except Exception as exc:
                failures.append(f"Telemetry failure: {type(exc).__name__}: {exc}")
                aborted_for_resource_limit = True
                _terminate_owned_processes(processes)
                break
            time.sleep(SAMPLE_INTERVAL_SECONDS)
        for process in processes.values():
            if process.poll() is None:
                process.wait()
        # A post-exit sample makes the end of the overlap auditable and gives
        # short early failures a chance to meet the minimum-sample diagnostic.
        try:
            samples.append(telemetry_sample(processes))
        except Exception as exc:
            failures.append(f"Final telemetry failure: {type(exc).__name__}: {exc}")
    except BaseException as exc:
        failures.append(f"Gate execution failure: {type(exc).__name__}: {exc}")
        _terminate_owned_processes(processes)
    finally:
        for stream in log_streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

    by_label = {launch["label"]: launch for launch in launches}
    for label, process in processes.items():
        by_label[label]["exit_code"] = process.poll()
    telemetry_summary: dict[str, Any] | None = None
    try:
        telemetry_summary = analyze_telemetry(
            samples,
            expected_root_pids={label: process.pid for label, process in processes.items()},
        )
    except GateValidationError as exc:
        failures.append(str(exc))

    current_sources = identity["source_sha256"]
    checkpoints_passed = True
    artifacts_complete = len(launches) == len(TASK_SPECS)
    for launch in launches:
        manifest_path = Path(launch["manifest_path"])
        checkpoint_path = Path(launch["checkpoint_path"])
        numbered_checkpoint = Path(launch["numbered_checkpoint_path"])
        log_path = Path(launch["log_path"])
        try:
            manifest = _read_json(manifest_path)
            snapshot_path = output_dir / f"{launch['label']}_paused_manifest.json"
            _atomic_create_json(snapshot_path, manifest)
            # With the production cadence (100) no periodic update-one file
            # exists before the explicit pause save.  The trainer therefore
            # links update-00000001.pt directly from the paused latest.pt.
            # Keep and authenticate that in-run immutable archive so its
            # relative external-history references remain resolvable after a
            # later resume changes latest.pt.
            paused_checkpoint_path = numbered_checkpoint
            launch["paused_manifest_artifact"] = _artifact_record(snapshot_path)
            launch["paused_checkpoint_artifact"] = _artifact_record(
                paused_checkpoint_path
            )
            launch["numbered_checkpoint_artifact"] = _artifact_record(
                numbered_checkpoint
            )
            launch["checkpoint_validation"] = validate_paused_training(
                manifest,
                task=launch["task"],
                run_dir=Path(launch["run_dir"]),
                checkpoint_path=paused_checkpoint_path,
                expected_source_hashes=current_sources,
            )
            launch["resume_command"] = resume_specs[launch["label"]]["command"]
            launch["resume_log_path"] = resume_specs[launch["label"]]["log_path"]
        except Exception as exc:
            checkpoints_passed = False
            failures.append(f"{launch['label']} checkpoint: {exc}")
        try:
            launch["log_artifact"] = _artifact_record(log_path)
        except (GateValidationError, OSError) as exc:
            artifacts_complete = False
            failures.append(f"{launch['label']} log: {exc}")

    exact_commands = len(launches) == len(command_specs) and all(
        all(launch.get(field) == spec[field] for field in spec)
        for launch, spec in zip(launches, command_specs)
    )
    distinct_run_dirs = len({launch.get("run_dir") for launch in launches}) == 2
    checks = {
        "exact_commands": exact_commands,
        "two_processes_launched": len(processes) == 2,
        "both_clean_pause_exit_codes": len(processes) == 2
        and all(process.poll() == PAUSE_EXIT_CODE for process in processes.values()),
        "isolated_run_and_checkpoint_directories": distinct_run_dirs,
        "clean_checkpoints_passed": checkpoints_passed and len(launches) == 2,
        "artifacts_complete": artifacts_complete,
        "external_gpu_limit_passed": bool(
            telemetry_summary and telemetry_summary["gpu_limit_passed"]
        ),
        "external_ram_limit_passed": bool(
            telemetry_summary and telemetry_summary["ram_limit_passed"]
        ),
        "no_new_sustained_swap_out": bool(
            telemetry_summary
            and telemetry_summary["no_new_sustained_swap_out_passed"]
        ),
        "simultaneous_gpu_compute_overlap": bool(
            telemetry_summary
            and telemetry_summary["simultaneous_gpu_compute_overlap_passed"]
        ),
        "not_aborted_for_resource_limit": not aborted_for_resource_limit,
    }
    if not all(checks.values()) and not failures:
        failures.append("One or more concurrency-gate checks failed")
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "kind": GATE_KIND,
        "status": "PASS" if all(checks.values()) and not failures else "FAIL",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "output_dir": str(output_dir),
        "gate_contract": _gate_contract(),
        "identity": identity,
        "launches": launches,
        "telemetry": {
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "samples": samples,
            "summary": telemetry_summary,
        },
        "checks": checks,
        "failures": failures,
    }
    _atomic_create_json(receipt_path, _build_envelope(payload))
    return receipt_path


def validate_receipt(
    receipt_path: Path | str,
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Validate a gate receipt, all artifacts, and current source identity.

    The returned value is the authenticated payload.  Any change to the
    receipt, source/config files, exact commands, reports, logs, or recomputed
    resource evidence raises :class:`GateValidationError`.
    """

    receipt_path = Path(receipt_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    envelope = _read_json(receipt_path)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise GateValidationError("Receipt envelope fields differ from schema v1")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise GateValidationError("Receipt payload must be an object")
    if envelope.get("payload_sha256") != canonical_sha256(payload):
        raise GateValidationError("Receipt payload SHA-256 mismatch")
    if payload.get("schema_version") != GATE_SCHEMA_VERSION or payload.get("kind") != GATE_KIND:
        raise GateValidationError("Receipt schema/kind mismatch")
    if require_pass and payload.get("status") != "PASS":
        raise GateValidationError("Parallel receipt did not pass")
    if payload.get("gate_contract") != _gate_contract():
        raise GateValidationError("Parallel receipt gate contract changed")
    output_dir = Path(str(payload.get("output_dir", ""))).expanduser().resolve()
    if receipt_path.parent != output_dir or receipt_path.name != RECEIPT_NAME:
        raise GateValidationError("Receipt is not at the canonical path inside its output directory")
    if payload.get("identity") != _expected_identity(config_path):
        raise GateValidationError("Receipt source/config identity is stale")

    launches = payload.get("launches")
    if not isinstance(launches, list) or len(launches) != 2:
        raise GateValidationError("Receipt must contain exactly two launches")
    expected_specs = trainer_commands(output_dir)
    resume_specs = {
        spec["label"]: spec for spec in trainer_commands(output_dir, resume=True)
    }
    expected_sources = payload["identity"]["source_sha256"]
    seen_pids: set[int] = set()
    for index, (launch, expected) in enumerate(zip(launches, expected_specs)):
        if not isinstance(launch, Mapping):
            raise GateValidationError(f"launches[{index}] must be an object")
        for field in (
            "label", "task", "stage", "command", "run_dir", "manifest_path",
            "checkpoint_path", "numbered_checkpoint_path", "log_path",
        ):
            if launch.get(field) != expected[field]:
                raise GateValidationError(f"launches[{index}].{field} changed")
        pid = launch.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 or pid in seen_pids:
            raise GateValidationError(f"launches[{index}].pid must be unique and positive")
        seen_pids.add(pid)
        if launch.get("exit_code") != PAUSE_EXIT_CODE:
            raise GateValidationError(f"launches[{index}] did not use clean pause exit 3")
        numbered_checkpoint_path = Path(expected["numbered_checkpoint_path"])
        log_path = Path(expected["log_path"])
        paused_manifest_path = output_dir / f"{expected['label']}_paused_manifest.json"
        paused_checkpoint_path = numbered_checkpoint_path
        manifest_artifact = _artifact_record(paused_manifest_path)
        paused_checkpoint_artifact = _artifact_record(paused_checkpoint_path)
        numbered_checkpoint_artifact = _artifact_record(numbered_checkpoint_path)
        log_artifact = _artifact_record(log_path)
        if launch.get("paused_manifest_artifact") != manifest_artifact:
            raise GateValidationError(f"launches[{index}] paused manifest changed")
        if launch.get("paused_checkpoint_artifact") != paused_checkpoint_artifact:
            raise GateValidationError(f"launches[{index}] paused checkpoint changed")
        if launch.get("numbered_checkpoint_artifact") != numbered_checkpoint_artifact:
            raise GateValidationError(f"launches[{index}] numbered checkpoint changed")
        if launch.get("log_artifact") != log_artifact:
            raise GateValidationError(f"launches[{index}] log hash/size changed")
        validation = validate_paused_training(
            _read_json(paused_manifest_path),
            task=expected["task"],
            run_dir=Path(expected["run_dir"]),
            checkpoint_path=paused_checkpoint_path,
            expected_source_hashes=expected_sources,
        )
        if launch.get("checkpoint_validation") != validation:
            raise GateValidationError(f"launches[{index}] checkpoint validation changed")
        resume = resume_specs[expected["label"]]
        if launch.get("resume_command") != resume["command"]:
            raise GateValidationError(f"launches[{index}] resume command changed")
        if launch.get("resume_log_path") != resume["log_path"]:
            raise GateValidationError(f"launches[{index}] resume log path changed")

    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise GateValidationError("Receipt telemetry must be an object")
    if telemetry.get("sample_interval_seconds") != SAMPLE_INTERVAL_SECONDS:
        raise GateValidationError("Receipt telemetry interval changed")
    samples = telemetry.get("samples")
    if not isinstance(samples, list):
        raise GateValidationError("Receipt telemetry samples must be a list")
    summary = analyze_telemetry(
        samples,
        expected_root_pids={launch["label"]: launch["pid"] for launch in launches},
    )
    if telemetry.get("summary") != summary:
        raise GateValidationError("Receipt telemetry summary is not reproducible")
    required_checks = {
        "exact_commands": True,
        "two_processes_launched": True,
        "both_clean_pause_exit_codes": True,
        "isolated_run_and_checkpoint_directories": True,
        "clean_checkpoints_passed": True,
        "artifacts_complete": True,
        "external_gpu_limit_passed": summary["gpu_limit_passed"],
        "external_ram_limit_passed": summary["ram_limit_passed"],
        "no_new_sustained_swap_out": summary["no_new_sustained_swap_out_passed"],
        "simultaneous_gpu_compute_overlap": summary[
            "simultaneous_gpu_compute_overlap_passed"
        ],
        "not_aborted_for_resource_limit": True,
    }
    if payload.get("checks") != required_checks or not all(required_checks.values()):
        raise GateValidationError("Receipt checks are incomplete or failed")
    if payload.get("failures") != []:
        raise GateValidationError("Passing receipt contains failures")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output_dir", type=Path)
    action.add_argument(
        "--validate",
        type=Path,
        help="CPU-only validation of an existing receipt; does not launch Isaac",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if args.validate is not None:
            payload = validate_receipt(args.validate, config_path=args.config)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "receipt": str(args.validate.resolve()),
                        "payload_sha256": canonical_sha256(payload),
                    },
                    sort_keys=True,
                )
            )
            return 0
        receipt = run_gate(args.output_dir, config_path=args.config)
        payload = _read_json(receipt)["payload"]
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "receipt": str(receipt),
                    "payload_sha256": canonical_sha256(payload),
                    "failures": payload["failures"],
                },
                sort_keys=True,
            )
        )
        return 0 if payload["status"] == "PASS" else 1
    except (FileExistsError, GateValidationError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
