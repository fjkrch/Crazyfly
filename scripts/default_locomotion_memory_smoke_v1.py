#!/usr/bin/env python3
"""Run four sequential one-update memory smokes for stock G1/Go1 tasks.

The gate is intentionally independent of main execution.  It launches every
registered training task once with the reviewed 1,024-environment batch,
records device-wide NVIDIA memory, system RAM, and swap-out activity, validates
the native RSL-RL ``model_0.pt`` checkpoint, and writes the canonical PASS
receipt only when all four cells pass.  Failed attempts remain immutable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import default_locomotion_queue_v1 as queue  # noqa: E402


GATE_KIND = "isaaclab_stock_locomotion_memory_smoke_v1"
RECEIPT_NAME = "memory_smoke_receipt.json"
ATTEMPT_PREFIX = "memory_smoke_attempt_"
POLL_INTERVAL_SECONDS = 0.5
JOB_TIMEOUT_SECONDS = 1800.0
TERMINATE_GRACE_SECONDS = 20.0
KILL_GRACE_SECONDS = 10.0
GPU_LIMIT_MIB = queue.GPU_LIMIT_MIB
RAM_LIMIT_PERCENT = queue.RAM_LIMIT_PERCENT
PAGING_STREAK_LIMIT = 3
CSV_FIELDS = (
    "timestamp_utc",
    "elapsed_seconds",
    "gpu_used_mib",
    "gpu_total_mib",
    "gpu_utilization_percent",
    "system_ram_percent",
    "system_ram_used_mib",
    "system_ram_total_mib",
    "swap_used_mib",
    "swap_out_bytes",
    "paging_streak",
    "compute_process_count",
    "owned_compute_process_count",
    "child_running",
)

_SPEC_BY_KEY = {
    (str(spec["robot"]), str(spec["terrain"])): spec
    for spec in queue.TASK_SPECS
}
SMOKE_SPECS = (
    _SPEC_BY_KEY[("g1", "rough")],
    _SPEC_BY_KEY[("go1", "rough")],
    _SPEC_BY_KEY[("g1", "flat")],
    _SPEC_BY_KEY[("go1", "flat")],
)


class SmokeGateError(RuntimeError):
    """Raised when an immutable smoke gate cannot safely continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
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
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise SmokeGateError(f"required artifact is missing or empty: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": queue.sha256_file(path),
    }


def _parse_float(text: str, field: str) -> float:
    try:
        value = float(text.strip())
    except (TypeError, ValueError) as exc:
        raise SmokeGateError(f"non-numeric {field} telemetry: {text!r}") from exc
    if not math.isfinite(value):
        raise SmokeGateError(f"non-finite {field} telemetry")
    return value


def _nvidia_query(fields: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeGateError(f"NVIDIA telemetry unavailable: {exc}") from exc
    if result.returncode != 0:
        raise SmokeGateError(
            f"NVIDIA telemetry query failed with code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
        if row
    ]


def _compute_processes() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeGateError(f"NVIDIA compute-process telemetry unavailable: {exc}") from exc
    if result.returncode != 0:
        raise SmokeGateError(
            f"NVIDIA compute-process query failed with code {result.returncode}"
        )
    rows: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise SmokeGateError("malformed NVIDIA compute-process telemetry")
        pid_text, name, memory_text = (item.strip() for item in row)
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise SmokeGateError("non-numeric NVIDIA compute PID") from exc
        rows.append(
            {
                "pid": pid,
                "process_name": name,
                "used_memory_mib": _parse_float(memory_text, "compute memory"),
            }
        )
    return rows


def sample_resources() -> dict[str, Any]:
    gpu_rows = _nvidia_query("index,memory.used,memory.total,utilization.gpu")
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 4:
        raise SmokeGateError("exactly one well-formed NVIDIA GPU is required")
    index_text, used_text, total_text, utilization_text = gpu_rows[0]
    if index_text != "0":
        raise SmokeGateError(f"expected GPU index 0, got {index_text!r}")
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    sample = {
        "timestamp_utc": _utc_now(),
        "gpu_index": 0,
        "gpu_used_mib": _parse_float(used_text, "GPU used memory"),
        "gpu_total_mib": _parse_float(total_text, "GPU total memory"),
        "gpu_utilization_percent": _parse_float(
            utilization_text, "GPU utilization"
        ),
        "system_ram_percent": float(memory.percent),
        "system_ram_used_mib": memory.used / (1024**2),
        "system_ram_total_mib": memory.total / (1024**2),
        "swap_used_mib": swap.used / (1024**2),
        "swap_out_bytes": int(swap.sout),
    }
    for key in (
        "system_ram_percent",
        "system_ram_used_mib",
        "system_ram_total_mib",
        "swap_used_mib",
    ):
        if not math.isfinite(float(sample[key])):
            raise SmokeGateError(f"non-finite {key} telemetry")
    return sample


def resource_violation(sample: Mapping[str, Any], paging_streak: int) -> str | None:
    try:
        gpu = float(sample["gpu_used_mib"])
        ram = float(sample["system_ram_percent"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return "missing_or_invalid_resource_telemetry"
    if not math.isfinite(gpu) or not math.isfinite(ram):
        return "missing_or_invalid_resource_telemetry"
    if gpu >= GPU_LIMIT_MIB:
        return f"gpu_memory_limit:{gpu:.2f}>={GPU_LIMIT_MIB:.2f}MiB"
    if ram >= RAM_LIMIT_PERCENT:
        return f"system_ram_limit:{ram:.2f}>={RAM_LIMIT_PERCENT:.2f}%"
    if paging_streak >= PAGING_STREAK_LIMIT:
        return f"sustained_swap_out:{paging_streak}_samples"
    return None


def smoke_command(spec: Mapping[str, Any]) -> list[str]:
    return [
        str(queue.ISAAC_PYTHON),
        str(queue.TRAINER),
        "--task",
        str(spec["task_id"]),
        "--agent",
        "rsl_rl_cfg_entry_point",
        "--seed",
        "0",
        "--num_envs",
        str(queue.NUM_ENVS),
        "--max_iterations",
        "1",
        "--device",
        "cuda:0",
        "--logger",
        "tensorboard",
        "--run_name",
        "seed_0",
        "--headless",
    ]


def _terminate_process_group(process: subprocess.Popen[Any]) -> dict[str, Any]:
    actions: list[str] = []
    if process.poll() is not None:
        return {"actions": actions, "exit_code": process.returncode}
    for sig, label, timeout in (
        (signal.SIGINT, "SIGINT", TERMINATE_GRACE_SECONDS),
        (signal.SIGTERM, "SIGTERM", KILL_GRACE_SECONDS),
        (signal.SIGKILL, "SIGKILL", KILL_GRACE_SECONDS),
    ):
        try:
            os.killpg(process.pid, sig)
            actions.append(label)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue
    return {"actions": actions, "exit_code": process.poll()}


def _wait_for_compute_clear(process_group: int, timeout: float = 60.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = _compute_processes()
        if not rows:
            return []
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(1.0)
    return rows


def _owned_compute_processes(
    rows: list[dict[str, Any]], process_group: int
) -> list[dict[str, Any]]:
    owned: list[dict[str, Any]] = []
    for row in rows:
        try:
            if os.getpgid(int(row["pid"])) == process_group:
                owned.append(row)
        except (ProcessLookupError, PermissionError, ValueError, TypeError):
            continue
    return owned


def _require_finite(value: Any, location: str = "checkpoint") -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise SmokeGateError(f"non-finite checkpoint tensor: {location}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SmokeGateError(f"non-finite checkpoint value: {location}")


def validate_checkpoint(
    checkpoint: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("iter") != 0:
        raise SmokeGateError("native smoke checkpoint does not end at iteration 0")
    actor = payload.get("actor_state_dict")
    critic = payload.get("critic_state_dict")
    optimizer = payload.get("optimizer_state_dict")
    if (
        not isinstance(actor, Mapping)
        or not isinstance(critic, Mapping)
        or not isinstance(optimizer, Mapping)
        or "infos" not in payload
    ):
        raise SmokeGateError("native smoke checkpoint lacks actor/critic state")
    _require_finite(payload)
    actor_count = sum(
        int(value.numel()) for value in actor.values() if isinstance(value, torch.Tensor)
    )
    critic_count = sum(
        int(value.numel()) for value in critic.values() if isinstance(value, torch.Tensor)
    )
    if actor_count != spec["actor_trainable_parameter_count"]:
        raise SmokeGateError(
            f"actor parameter count {actor_count} != "
            f"{spec['actor_trainable_parameter_count']}"
        )
    if critic_count != spec["critic_parameter_count"]:
        raise SmokeGateError(
            f"critic parameter count {critic_count} != {spec['critic_parameter_count']}"
        )
    return {
        **_artifact(checkpoint),
        "iteration": 0,
        "actor_trainable_parameter_count": actor_count,
        "critic_trainable_parameter_count": critic_count,
        "all_values_finite": True,
    }


def _next_attempt_dir(gates_root: Path) -> Path:
    existing: list[int] = []
    for path in gates_root.glob(f"{ATTEMPT_PREFIX}*"):
        suffix = path.name.removeprefix(ATTEMPT_PREFIX)
        if path.is_dir() and suffix.isdigit():
            existing.append(int(suffix))
    number = max(existing, default=0) + 1
    return gates_root / f"{ATTEMPT_PREFIX}{number:03d}"


def _row_for_csv(
    sample: Mapping[str, Any], *, elapsed: float, paging_streak: int,
    compute_process_count: int, owned_compute_process_count: int,
    child_running: bool,
) -> dict[str, Any]:
    return {
        "timestamp_utc": sample["timestamp_utc"],
        "elapsed_seconds": f"{elapsed:.3f}",
        "gpu_used_mib": f"{float(sample['gpu_used_mib']):.3f}",
        "gpu_total_mib": f"{float(sample['gpu_total_mib']):.3f}",
        "gpu_utilization_percent": f"{float(sample['gpu_utilization_percent']):.3f}",
        "system_ram_percent": f"{float(sample['system_ram_percent']):.3f}",
        "system_ram_used_mib": f"{float(sample['system_ram_used_mib']):.3f}",
        "system_ram_total_mib": f"{float(sample['system_ram_total_mib']):.3f}",
        "swap_used_mib": f"{float(sample['swap_used_mib']):.3f}",
        "swap_out_bytes": int(sample["swap_out_bytes"]),
        "paging_streak": paging_streak,
        "compute_process_count": compute_process_count,
        "owned_compute_process_count": owned_compute_process_count,
        "child_running": child_running,
    }


def run_task_smoke(
    spec: Mapping[str, Any], attempt_dir: Path
) -> dict[str, Any]:
    task_label = f"{spec['robot']}_{spec['terrain']}"
    task_dir = attempt_dir / "work" / str(spec["robot"]) / str(spec["terrain"])
    task_dir.mkdir(parents=True, exist_ok=False)
    log_path = attempt_dir / "logs" / f"{task_label}.log"
    telemetry_path = attempt_dir / "telemetry" / f"{task_label}.csv"
    receipt_path = attempt_dir / "task_receipts" / f"{task_label}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    command = smoke_command(spec)
    started_at = _utc_now()
    start = time.monotonic()
    samples = 0
    max_gpu = float("-inf")
    max_ram = float("-inf")
    max_swap = float("-inf")
    paging_streak = 0
    max_paging_streak = 0
    owned_compute_sample_count = 0
    maximum_compute_process_count = 0
    observed_owned_compute_pids: set[int] = set()
    previous_swap_out: int | None = None
    failure: str | None = None
    termination: dict[str, Any] = {"actions": [], "exit_code": None}
    exit_code: int | None = None
    process: subprocess.Popen[Any] | None = None
    with log_path.open("x", encoding="utf-8", buffering=1) as log_stream, telemetry_path.open(
        "x", encoding="utf-8", newline="", buffering=1
    ) as telemetry_stream:
        writer = csv.DictWriter(telemetry_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        log_stream.write("COMMAND " + json.dumps(command) + "\n")
        log_stream.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=task_dir,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            while True:
                elapsed = time.monotonic() - start
                try:
                    sample = sample_resources()
                    compute_processes = _compute_processes()
                except SmokeGateError as exc:
                    failure = f"telemetry_failure:{exc}"
                    termination = _terminate_process_group(process)
                    break
                swap_out = int(sample["swap_out_bytes"])
                if previous_swap_out is not None and swap_out > previous_swap_out:
                    paging_streak += 1
                else:
                    paging_streak = 0
                previous_swap_out = swap_out
                max_paging_streak = max(max_paging_streak, paging_streak)
                max_gpu = max(max_gpu, float(sample["gpu_used_mib"]))
                max_ram = max(max_ram, float(sample["system_ram_percent"]))
                max_swap = max(max_swap, float(sample["swap_used_mib"]))
                owned_compute = _owned_compute_processes(
                    compute_processes, process.pid
                )
                maximum_compute_process_count = max(
                    maximum_compute_process_count, len(compute_processes)
                )
                if process.poll() is None and owned_compute:
                    owned_compute_sample_count += 1
                    observed_owned_compute_pids.update(
                        int(row["pid"]) for row in owned_compute
                    )
                samples += 1
                writer.writerow(
                    _row_for_csv(
                        sample,
                        elapsed=elapsed,
                        paging_streak=paging_streak,
                        compute_process_count=len(compute_processes),
                        owned_compute_process_count=len(owned_compute),
                        child_running=process.poll() is None,
                    )
                )
                telemetry_stream.flush()
                violation = resource_violation(sample, paging_streak)
                if violation is not None:
                    failure = violation
                    termination = _terminate_process_group(process)
                    break
                exit_code = process.poll()
                if exit_code is not None:
                    break
                if elapsed >= JOB_TIMEOUT_SECONDS:
                    failure = f"timeout:{JOB_TIMEOUT_SECONDS:.0f}s"
                    termination = _terminate_process_group(process)
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
        except (OSError, subprocess.SubprocessError) as exc:
            failure = f"launch_or_process_error:{type(exc).__name__}:{exc}"
            if process is not None:
                termination = _terminate_process_group(process)
        except KeyboardInterrupt:
            failure = "operator_interrupt"
            if process is not None:
                termination = _terminate_process_group(process)
        finally:
            if process is not None:
                if process.poll() is None:
                    termination = _terminate_process_group(process)
                exit_code = process.poll()
    lingering = _wait_for_compute_clear(process.pid) if process is not None else []
    if lingering and failure is None:
        failure = "lingering_nvidia_compute_processes"
    checkpoint_info: dict[str, Any] | None = None
    native_artifacts: dict[str, Any] | None = None
    expected_log_root = (
        task_dir / "logs" / "rsl_rl" / str(spec["default_experiment_name"])
    )
    run_directories = sorted(
        path
        for path in expected_log_root.glob("*_seed_0")
        if path.is_dir() and not path.is_symlink()
    )
    checkpoints = sorted(task_dir.rglob("model_0.pt"))
    if failure is None and exit_code != 0:
        failure = f"trainer_exit_code:{exit_code}"
    if failure is None and len(checkpoints) != 1:
        failure = f"expected_one_model_0_checkpoint:found_{len(checkpoints)}"
    if failure is None and len(run_directories) != 1:
        failure = f"expected_one_native_run_directory:found_{len(run_directories)}"
    if failure is None:
        run_directory = run_directories[0]
        env_yaml = run_directory / "params" / "env.yaml"
        agent_yaml = run_directory / "params" / "agent.yaml"
        event_files = sorted(run_directory.glob("events.out.tfevents.*"))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if (
            checkpoints[0].parent != run_directory
            or not env_yaml.is_file()
            or not agent_yaml.is_file()
            or len(event_files) != 1
            or "[INFO] Logging experiment in directory:" not in log_text
            or "Training time:" not in log_text
        ):
            failure = "native_artifact_or_log_contract_failed"
        else:
            native_artifacts = {
                "run_directory": str(run_directory.resolve()),
                "environment_yaml": _artifact(env_yaml),
                "agent_yaml": _artifact(agent_yaml),
                "tensorboard_event": _artifact(event_files[0]),
            }
    if failure is None:
        try:
            checkpoint_info = validate_checkpoint(checkpoints[0], spec)
        except (OSError, RuntimeError, SmokeGateError, ValueError) as exc:
            failure = f"checkpoint_validation:{type(exc).__name__}:{exc}"
    duration = time.monotonic() - start
    hard_resource_failure = bool(
        failure
        and (
            failure.startswith("telemetry_failure:")
            or failure.startswith("gpu_memory_limit:")
            or failure.startswith("system_ram_limit:")
            or failure.startswith("sustained_swap_out:")
            or failure.startswith("timeout:")
            or failure == "lingering_nvidia_compute_processes"
        )
    )
    result = {
        "schema_version": 1,
        "kind": "isaaclab_stock_locomotion_memory_smoke_task_v1",
        "generated_at_utc": _utc_now(),
        "started_at_utc": started_at,
        "robot": spec["robot"],
        "terrain": spec["terrain"],
        "task": spec["task_id"],
        "seed": 0,
        "num_envs": queue.NUM_ENVS,
        "max_iterations": 1,
        "expected_interactions": queue.NUM_ENVS * queue.STEPS_PER_ENV,
        "status": "passed" if failure is None else "failed",
        "passed": failure is None,
        "failure": failure,
        "hard_resource_failure": hard_resource_failure,
        "command": command,
        "working_directory": str(task_dir.resolve()),
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
        "memory": {
            "sample_count": samples,
            "poll_interval_seconds": POLL_INTERVAL_SECONDS,
            "max_device_gpu_used_mib": round(max_gpu, 3) if samples else None,
            "gpu_limit_mib_exclusive": GPU_LIMIT_MIB,
            "max_system_ram_percent": round(max_ram, 3) if samples else None,
            "ram_limit_percent_exclusive": RAM_LIMIT_PERCENT,
            "max_swap_used_mib": round(max_swap, 3) if samples else None,
            "maximum_consecutive_swap_out_samples": max_paging_streak,
            "sustained_paging_detected": max_paging_streak >= PAGING_STREAK_LIMIT,
            "owned_compute_sample_count": owned_compute_sample_count,
            "maximum_compute_process_count": maximum_compute_process_count,
            "observed_owned_compute_pids": sorted(observed_owned_compute_pids),
        },
        "termination": termination,
        "lingering_compute_processes": lingering,
        "checkpoint": checkpoint_info,
        "native_artifacts": native_artifacts,
        "log": _artifact(log_path),
        "telemetry": _artifact(telemetry_path),
    }
    _atomic_create_json(receipt_path, result)
    return {**result, "receipt": _artifact(receipt_path)}


def _validate_task_result(result: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    memory = result.get("memory")
    if (
        result.get("passed") is not True
        or result.get("status") != "passed"
        or result.get("task") != spec["task_id"]
        or result.get("num_envs") != queue.NUM_ENVS
        or result.get("max_iterations") != 1
        or not isinstance(memory, Mapping)
        or not isinstance(memory.get("sample_count"), int)
        or memory["sample_count"] < 3
        or int(memory.get("owned_compute_sample_count", 0)) < 3
        or float(memory.get("max_device_gpu_used_mib", math.inf)) >= GPU_LIMIT_MIB
        or float(memory.get("max_system_ram_percent", math.inf)) >= RAM_LIMIT_PERCENT
        or memory.get("sustained_paging_detected") is not False
        or result.get("checkpoint") is None
        or result.get("native_artifacts") is None
        or result.get("lingering_compute_processes") != []
    ):
        raise SmokeGateError(f"task smoke result failed validation: {spec['task_id']}")


def validate_canonical_receipt(
    config: Mapping[str, Any], receipt_path: Path | None = None
) -> dict[str, Any]:
    path = receipt_path or Path(config["_output_root"]) / "gates" / RECEIPT_NAME
    value = queue._read_json(path)
    if (
        value.get("schema_version") != 1
        or value.get("kind") != GATE_KIND
        or value.get("status") != "passed"
        or value.get("passed") is not True
        or value.get("config_sha256") != queue.sha256_file(Path(config["_config_path"]))
        or value.get("queue_builder_sha256") != queue.sha256_file(
            Path(queue.__file__).resolve()
        )
        or value.get("isaaclab_commit") != queue.ISAACLAB_COMMIT
        or value.get("num_envs") != queue.NUM_ENVS
        or value.get("maximum_parallel") != 1
    ):
        raise SmokeGateError("canonical memory-smoke receipt identity is invalid")
    results = value.get("task_results")
    if not isinstance(results, list) or len(results) != len(SMOKE_SPECS):
        raise SmokeGateError("canonical memory-smoke receipt lacks four task results")
    for result, spec in zip(results, SMOKE_SPECS, strict=True):
        _validate_task_result(result, spec)
    return value


def run_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    gates_root = output_root / "gates"
    canonical_path = gates_root / RECEIPT_NAME
    if canonical_path.exists():
        raise FileExistsError(f"canonical memory-smoke receipt already exists: {canonical_path}")
    gates_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(gates_root)
    attempt_dir.mkdir(parents=False, exist_ok=False)
    lock_path = gates_root / ".memory_smoke.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise SmokeGateError(f"memory smoke lock already exists: {lock_path}") from exc
    os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
    os.fsync(lock_fd)
    os.close(lock_fd)
    started_at = _utc_now()
    task_results: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    blocked_reason: str | None = None
    try:
        try:
            preflight = sample_resources()
            existing_compute = _compute_processes()
            blocked_reason = resource_violation(preflight, 0)
            if existing_compute:
                blocked_reason = "preexisting_nvidia_compute_processes"
        except SmokeGateError as exc:
            existing_compute = []
            blocked_reason = f"preflight_telemetry_failure:{exc}"
        if blocked_reason is None:
            for spec in SMOKE_SPECS:
                result = run_task_smoke(spec, attempt_dir)
                task_results.append(result)
                if result.get("hard_resource_failure") is True:
                    blocked_reason = (
                        f"hard_resource_failure:{spec['task_id']}:"
                        f"{result.get('failure')}"
                    )
                    break
        all_pass = (
            blocked_reason is None
            and len(task_results) == len(SMOKE_SPECS)
            and all(result.get("passed") is True for result in task_results)
        )
        overall = {
            "schema_version": 1,
            "kind": GATE_KIND,
            "generated_at_utc": _utc_now(),
            "started_at_utc": started_at,
            "status": "passed" if all_pass else "failed",
            "passed": all_pass,
            "blocked_reason": blocked_reason,
            "attempt_directory": str(attempt_dir.resolve()),
            "config": str(Path(config["_config_path"]).resolve()),
            "config_sha256": queue.sha256_file(Path(config["_config_path"])),
            "queue_builder": str(Path(queue.__file__).resolve()),
            "queue_builder_sha256": queue.sha256_file(Path(queue.__file__).resolve()),
            "smoke_runner": str(Path(__file__).resolve()),
            "smoke_runner_sha256": queue.sha256_file(Path(__file__).resolve()),
            "isaaclab_commit": queue.ISAACLAB_COMMIT,
            "num_envs": queue.NUM_ENVS,
            "iterations_per_task": 1,
            "maximum_parallel": 1,
            "gpu_limit_mib_exclusive": GPU_LIMIT_MIB,
            "ram_limit_percent_exclusive": RAM_LIMIT_PERCENT,
            "sustained_paging_sample_count": PAGING_STREAK_LIMIT,
            "preflight": preflight,
            "preflight_compute_processes": existing_compute,
            "task_results": task_results,
            "passed_task_count": sum(
                result.get("passed") is True for result in task_results
            ),
            "required_task_count": len(SMOKE_SPECS),
            "source_verification": config["_source_verification"],
            "preservation_verification": {
                "frozen_g1": config["_frozen_g1_verification"],
                "frozen_runs": config["_frozen_runs_verification"],
            },
        }
        attempt_receipt = attempt_dir / "overall_receipt.json"
        _atomic_create_json(attempt_receipt, overall)
        if all_pass:
            canonical = {
                **overall,
                "attempt_receipt": _artifact(attempt_receipt),
            }
            _atomic_create_json(canonical_path, canonical)
            validate_canonical_receipt(config, canonical_path)
            return canonical
        return overall
    finally:
        lock_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-task stock G1/Go1 one-update memory smoke"
    )
    parser.add_argument("--config", type=Path, default=queue.DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = queue.validate_config(args.config)
    path = Path(config["_output_root"]) / "gates" / RECEIPT_NAME
    if args.status:
        value = validate_canonical_receipt(config, path)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    value = run_gate(config)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("passed") is True else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, SmokeGateError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
