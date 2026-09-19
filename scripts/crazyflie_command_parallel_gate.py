#!/usr/bin/env python3
"""Run the bounded exact-two-process gate for the CommandFollow queue.

The gate launches original and degree-rewired frozen-LIF trainers together,
each with 40 environments.  Both pause after update one (4,000 interactions),
their checkpoints and deterministic command cursors are validated, and they
then resume sequentially to update two (8,000 interactions).  Device-wide GPU,
RAM, swap-out, and simultaneous CUDA-process evidence is sampled externally.
Only a completely passing run receives a PASS receipt.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source" / "g1_fly_control"
for import_path in (SCRIPT_DIR, SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import crazyflie_command_queue as queue_module  # noqa: E402
from drone_bootstrap import (  # noqa: E402
    canonical_sha256,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)
from drone_train import standalone_resolved_config  # noqa: E402


CONTROLLERS = ("frozen_lif_original", "frozen_lif_degree_rewired")
TOTAL_INTERACTIONS = 8_000
INTERACTIONS_PER_UPDATE = 4_000
PAUSE_AFTER_UPDATES = 1
COMPLETED_UPDATES = 2
SAMPLE_INTERVAL_SECONDS = 0.25
PAUSE_EXIT_CODE = 3
RECEIPT_NAME = "parallel_gate_receipt.json"


class GateError(RuntimeError):
    """Raised when bounded concurrency evidence is incomplete or invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite gate artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateError(f"Required gate artifact is missing/empty: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fingerprint(config: Mapping[str, Any], controller: str) -> tuple[str, dict[str, Any]]:
    training = config["training"]
    args = SimpleNamespace(
        task=queue_module.TASK,
        contract_profile="command_v1",
        policy=controller,
        seed=0,
        num_envs=40,
        total_interactions=TOTAL_INTERACTIONS,
        horizon=100,
        microbatch_size=40,
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
    resolved, evaluation = standalone_resolved_config(args)
    rewired = load_fingerprint_rewire_manifest(
        args.rewire_manifest,
        expected_file_sha256=config["rewire"]["manifest_sha256"],
        expected_seed=args.rewire_seed,
    )
    return reproduction_fingerprint(
        resolved_config=resolved,
        evaluation_manifest=evaluation,
        connectome_manifest=args.connectome_manifest,
        rewired_manifest=rewired,
    )


def _trainer_command(
    config: Mapping[str, Any], controller: str, run_dir: Path,
    fingerprint: str, *, resume: bool,
) -> list[str]:
    training = config["training"]
    command = [
        config["isaac_python"], str(ROOT / "scripts" / "drone_train.py"),
        "--task", queue_module.TASK,
        "--contract_profile", "command_v1",
        "--policy", controller,
        "--seed", "0",
        "--num_envs", "40",
        "--total_interactions", str(TOTAL_INTERACTIONS),
        "--horizon", "100",
        "--microbatch_size", "40",
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
        "--pause_file", str(run_dir / "pause.request"),
        "--device", "cuda:0",
        "--headless",
    ]
    if resume:
        command.append("--resume")
    else:
        command.extend(("--pause_after_updates", str(PAUSE_AFTER_UPDATES)))
    return command


def gate_specs(output_dir: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, controller in enumerate(CONTROLLERS, start=1):
        run_dir = output_dir / "jobs" / f"{index:02d}__{controller}"
        fingerprint, payload = _fingerprint(config, controller)
        specs.append({
            "controller": controller,
            "task": queue_module.TASK,
            "num_envs": 40,
            "run_dir": str(run_dir),
            "manifest": str(run_dir / "training_manifest.json"),
            "checkpoint": str(run_dir / "checkpoints" / "latest.pt"),
            "paused_checkpoint": str(run_dir / "checkpoints" / "update-00000001.pt"),
            "paused_manifest_snapshot": str(output_dir / f"{controller}__paused_manifest.json"),
            "pause_log": str(output_dir / "logs" / f"{controller}__parallel_pause.log"),
            "resume_log": str(output_dir / "logs" / f"{controller}__sequential_resume.log"),
            "expected_fingerprint": fingerprint,
            "fingerprint_payload": payload,
            "pause_command": _trainer_command(
                config, controller, run_dir, fingerprint, resume=False
            ),
            "resume_command": _trainer_command(
                config, controller, run_dir, fingerprint, resume=True
            ),
        })
    return specs


def _memory_gate_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    try:
        gpu = float(value["max_device_gpu_used_mib"])
        ram = float(value["max_system_ram_percent"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(gpu) and gpu < queue_module.GPU_LIMIT_MIB
        and math.isfinite(ram) and ram < queue_module.RAM_LIMIT_PERCENT
        and value.get("sustained_paging_detected") is False
        and value.get("failures") == []
    )


def _schedule_state(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    schedule = manifest.get("command_schedule")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("command_training_contract_sha256")
        != queue_module.COMMAND_TRACKING_CONTRACT_SHA256
        or not isinstance(schedule.get("state"), Mapping)
    ):
        raise GateError("Training manifest lacks the current command schedule contract/state")
    return schedule["state"]


def _validate_checkpoint_payload(
    checkpoint: Path, *, controller: str, fingerprint: str,
    expected_status: str, interactions: int, updates: int,
) -> None:
    import torch
    from g1_fly_control.crazyflie.checkpoint import read_checkpoint

    payload = read_checkpoint(
        checkpoint, map_location="cpu", resolve_external_history=True
    )
    counters = payload.get("counters")
    metadata = payload.get("metadata")
    fingerprints = payload.get("fingerprints")
    task_schedule = payload.get("rng_states", {}).get("task_schedule", {})
    command_state = task_schedule.get("command_task_schedule")
    if (
        not isinstance(counters, Mapping)
        or counters.get("total_interactions") != interactions
        or counters.get("completed_updates") != updates
        or not isinstance(metadata, Mapping)
        or metadata.get("status") != expected_status
        or metadata.get("controller") != controller
        or metadata.get("contract_profile") != "command_v1"
        or metadata.get("core_checksum_before") != metadata.get("core_checksum_after")
        or not isinstance(fingerprints, Mapping)
        or fingerprints.get("reproduction") != fingerprint
        or not isinstance(command_state, Mapping)
        or command_state.get("training_interactions") != interactions
        or command_state.get("contract_sha256")
        != queue_module.COMMAND_TRACKING_CONTRACT_SHA256
    ):
        raise GateError(f"Checkpoint boundary validation failed: {checkpoint}")

    def require_finite(value: Any, location: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise GateError(f"Nonfinite checkpoint tensor: {location}")
        elif isinstance(value, Mapping):
            for key, child in value.items():
                require_finite(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                require_finite(child, f"{location}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise GateError(f"Nonfinite checkpoint value: {location}")

    require_finite(payload, "checkpoint")


def validate_training_boundary(
    spec: Mapping[str, Any], *, paused: bool
) -> dict[str, Any]:
    manifest_path = Path(spec["manifest"])
    manifest = _read_json(manifest_path)
    checkpoint = Path(spec["paused_checkpoint"] if paused else spec["checkpoint"])
    expected_status = "paused" if paused else "completed"
    interactions = INTERACTIONS_PER_UPDATE if paused else TOTAL_INTERACTIONS
    updates = PAUSE_AFTER_UPDATES if paused else COMPLETED_UPDATES
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != expected_status
        or manifest.get("contract_profile") != "command_v1"
        or manifest.get("controller") != spec["controller"]
        or manifest.get("seed") != 0
        or manifest.get("requested_interactions") != TOTAL_INTERACTIONS
        or manifest.get("environment_interactions") != interactions
        or manifest.get("completed_updates") != updates
        or manifest.get("fingerprint") != spec["expected_fingerprint"]
        or manifest.get("fingerprint_payload") != spec["fingerprint_payload"]
        or not _memory_gate_valid(manifest.get("memory_gate"))
    ):
        raise GateError(
            f"{spec['controller']} {expected_status} manifest failed validation"
        )
    resolved = manifest.get("resolved_config")
    if (
        not isinstance(resolved, Mapping)
        or resolved.get("task") != queue_module.TASK
        or resolved.get("total_interactions") != TOTAL_INTERACTIONS
        or resolved.get("num_envs") != 40
        or resolved.get("horizon") != 100
    ):
        raise GateError("Gate manifest resolved configuration changed")
    state = _schedule_state(manifest)
    if state.get("training_interactions") != interactions or state.get("num_envs") != 40:
        raise GateError("Gate manifest command cursor has the wrong boundary")
    recorded_checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
    latest = Path(spec["checkpoint"]).resolve()
    if recorded_checkpoint != latest or manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise GateError("Gate manifest checkpoint identity mismatch")
    if not paused:
        if manifest.get("resume_count") != 1:
            raise GateError("Sequential gate resume did not increment resume_count")
        resume = manifest.get("command_schedule_resume")
        if (
            not isinstance(resume, Mapping)
            or resume.get("restored_from_checkpoint") is not True
            or resume.get("restored_before_resume_environment_reset") is not True
        ):
            raise GateError("Sequential resume did not restore command schedule before reset")
    _validate_checkpoint_payload(
        checkpoint,
        controller=spec["controller"],
        fingerprint=spec["expected_fingerprint"],
        expected_status=expected_status,
        interactions=interactions,
        updates=updates,
    )
    return {
        "status": expected_status,
        "interactions": interactions,
        "updates": updates,
        "command_schedule_state_sha256": manifest["command_schedule"]["state_sha256"],
        "memory_gate": manifest["memory_gate"],
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _telemetry(processes: Mapping[str, subprocess.Popen[Any]]) -> dict[str, Any]:
    snapshot = queue_module.resource_snapshot()
    compute_pids = sorted(queue_module.gpu_compute_client_pids())
    return {
        **snapshot,
        "monotonic_seconds": time.monotonic(),
        "compute_pids": compute_pids,
        "tracked": {
            controller: {"pid": process.pid, "alive": process.poll() is None}
            for controller, process in processes.items()
        },
    }


def _request_pauses(specs: list[dict[str, Any]], reason: str) -> None:
    for spec in specs:
        path = Path(spec["run_dir"]) / "pause.request"
        if not path.exists():
            queue_module._atomic_json(path, {
                "schema_version": 1, "status": "requested",
                "requested_utc": _utc_now(), "reason": reason,
            })


def _terminate_owned(processes: Mapping[str, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and any(p.poll() is None for p in processes.values()):
        time.sleep(0.1)
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _monitor(
    processes: Mapping[str, subprocess.Popen[Any]],
    specs: list[dict[str, Any]], samples: list[dict[str, Any]], failures: list[str],
) -> None:
    resource_pause_requested = False
    deadline = time.monotonic() + 900.0
    while any(process.poll() is None for process in processes.values()):
        if time.monotonic() >= deadline and not resource_pause_requested:
            failures.append("Bounded gate phase exceeded 900 seconds")
            resource_pause_requested = True
            _request_pauses(specs, "parallel_gate_phase_timeout")
        try:
            sample = _telemetry(processes)
            samples.append(sample)
            if not sample["passed"] and not resource_pause_requested:
                failures.append(
                    "External hard resource limit reached: "
                    f"GPU={sample['gpu_used_mib']} MiB RAM={sample['system_ram_percent']}%"
                )
                resource_pause_requested = True
                _request_pauses(specs, "parallel_gate_hard_resource_limit")
            recent = [entry["swap_out_pages"] for entry in samples[-3:]]
            if (
                len(recent) == 3 and recent[0] < recent[1] < recent[2]
                and not resource_pause_requested
            ):
                failures.append("Sustained swap-out detected by parallel gate")
                resource_pause_requested = True
                _request_pauses(specs, "parallel_gate_sustained_paging")
        except Exception as exc:
            if resource_pause_requested:
                time.sleep(SAMPLE_INTERVAL_SECONDS)
                continue
            failures.append(f"Telemetry failure: {type(exc).__name__}: {exc}")
            resource_pause_requested = True
            _request_pauses(specs, "parallel_gate_telemetry_failure")
        if resource_pause_requested and not any(p.poll() is None for p in processes.values()):
            break
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def _summary(
    samples: list[Mapping[str, Any]], processes: Mapping[str, subprocess.Popen[Any]]
) -> dict[str, Any]:
    if not samples:
        raise GateError("Parallel gate collected no external telemetry")
    pids = {process.pid for process in processes.values()}
    overlap = sum(pids.issubset(set(sample.get("compute_pids", []))) for sample in samples)
    gpu_values = [float(sample["gpu_used_mib"]) for sample in samples]
    ram_values = [float(sample["system_ram_percent"]) for sample in samples]
    swap_values = [int(sample["swap_out_pages"]) for sample in samples]
    if not all(math.isfinite(value) for value in gpu_values + ram_values):
        raise GateError("Parallel gate telemetry contains nonfinite values")
    return {
        "max_device_gpu_used_mib": max(gpu_values),
        "max_system_ram_percent": max(ram_values),
        "simultaneous_gpu_compute_overlap_samples": overlap,
        "swap_out_growth_pages": max(swap_values) - min(swap_values),
    }


def run_gate(output_dir: Path, config_path: Path) -> Path:
    config = queue_module.validate_config(config_path)
    output_dir = output_dir.expanduser().resolve()
    runs_root = (ROOT / "runs").resolve()
    if output_dir == runs_root or not output_dir.is_relative_to(runs_root):
        raise GateError(f"Gate output must be a child of {runs_root}")
    if output_dir.exists():
        raise FileExistsError(f"Gate output exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir()
    receipt_path = output_dir / RECEIPT_NAME
    specs = gate_specs(output_dir, config)
    processes: dict[str, subprocess.Popen[Any]] = {}
    streams: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    launches: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    started = _utc_now()
    try:
        for spec in specs:
            log_path = Path(spec["pause_log"])
            stream = log_path.open("xb")
            streams[spec["controller"]] = stream
            process = subprocess.Popen(
                spec["pause_command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes[spec["controller"]] = process
            launches.append({
                "controller": spec["controller"], "task": queue_module.TASK,
                "num_envs": 40, "pid": process.pid,
                "run_dir": spec["run_dir"], "pause_command": spec["pause_command"],
            })
        _monitor(processes, specs, samples, failures)
        for process in processes.values():
            process.wait()
    except BaseException as exc:
        failures.append(f"Parallel launch failure: {type(exc).__name__}: {exc}")
        _request_pauses(specs, "parallel_gate_exception")
        _terminate_owned(processes)
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

    by_controller = {row["controller"]: row for row in launches}
    for spec in specs:
        launch = by_controller.setdefault(spec["controller"], {
            "controller": spec["controller"], "task": queue_module.TASK,
            "num_envs": 40, "run_dir": spec["run_dir"],
            "pause_command": spec["pause_command"],
        })
        process = processes.get(spec["controller"])
        launch["pause_exit_code"] = process.poll() if process is not None else None
        try:
            paused_manifest = _read_json(Path(spec["manifest"]))
            _atomic_create_json(Path(spec["paused_manifest_snapshot"]), paused_manifest)
            launch["pause_validation"] = validate_training_boundary(spec, paused=True)
            launch["paused_updates"] = PAUSE_AFTER_UPDATES
            launch["paused_interactions"] = INTERACTIONS_PER_UPDATE
            artifacts.extend((
                _artifact(Path(spec["paused_manifest_snapshot"])),
                _artifact(Path(spec["paused_checkpoint"])),
                _artifact(Path(spec["pause_log"])),
            ))
        except Exception as exc:
            failures.append(f"{spec['controller']} paused boundary: {type(exc).__name__}: {exc}")

    # Resume one process at a time.  These samples remain in the receipt but
    # cannot contribute to the simultaneous-overlap count because only one
    # root process is alive.
    if not failures:
        for spec in specs:
            stream = Path(spec["resume_log"]).open("xb")
            try:
                process = subprocess.Popen(
                    spec["resume_command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
                sequential = {spec["controller"]: process}
                _monitor(sequential, [spec], samples, failures)
                process.wait()
                launch = by_controller[spec["controller"]]
                launch["resume_command"] = spec["resume_command"]
                launch["resume_exit_code"] = process.returncode
                launch["completed_updates"] = COMPLETED_UPDATES
                launch["completed_interactions"] = TOTAL_INTERACTIONS
                launch["completion_validation"] = validate_training_boundary(
                    spec, paused=False
                )
            except Exception as exc:
                failures.append(f"{spec['controller']} resume: {type(exc).__name__}: {exc}")
                if "process" in locals() and process.poll() is None:
                    _terminate_owned({spec["controller"]: process})
            finally:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            try:
                artifacts.extend((
                    _artifact(Path(spec["manifest"])),
                    _artifact(Path(spec["checkpoint"])),
                    _artifact(Path(spec["resume_log"])),
                ))
            except Exception as exc:
                failures.append(f"{spec['controller']} completion artifact: {exc}")
            if failures:
                break

    try:
        telemetry_summary = _summary(samples, processes)
    except Exception as exc:
        failures.append(f"Telemetry summary: {type(exc).__name__}: {exc}")
        telemetry_summary = {
            "max_device_gpu_used_mib": queue_module.GPU_LIMIT_MIB,
            "max_system_ram_percent": queue_module.RAM_LIMIT_PERCENT,
            "simultaneous_gpu_compute_overlap_samples": 0,
            "swap_out_growth_pages": 0,
        }
    swap_values = [sample.get("swap_out_pages") for sample in samples]
    sustained = any(
        a < b < c for a, b, c in zip(swap_values, swap_values[1:], swap_values[2:])
    ) if len(swap_values) >= 3 else False
    launches = [by_controller[controller] for controller in CONTROLLERS]
    checks = {
        "exactly_two_processes": len(processes) == 2,
        "isolated_run_and_checkpoint_directories": len({spec["run_dir"] for spec in specs}) == 2,
        "clean_checkpoints_passed": all(
            launch.get("pause_exit_code") == PAUSE_EXIT_CODE
            and isinstance(launch.get("pause_validation"), dict)
            for launch in launches
        ),
        "simultaneous_gpu_compute_overlap": telemetry_summary["simultaneous_gpu_compute_overlap_samples"] >= 1,
        "no_missing_or_nonfinite_telemetry": bool(samples)
        and math.isfinite(telemetry_summary["max_device_gpu_used_mib"])
        and math.isfinite(telemetry_summary["max_system_ram_percent"])
        and telemetry_summary["max_device_gpu_used_mib"] < queue_module.GPU_LIMIT_MIB
        and telemetry_summary["max_system_ram_percent"] < queue_module.RAM_LIMIT_PERCENT,
        "no_sustained_paging": not sustained,
        "sequential_resume_completed": all(
            launch.get("resume_exit_code") == 0
            and isinstance(launch.get("completion_validation"), dict)
            and launch.get("completed_interactions") == TOTAL_INTERACTIONS
            for launch in launches
        ),
    }
    passed = all(checks.values()) and not failures
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "contract": queue_module.parallel_gate_contract(config),
        "checks": checks,
        "telemetry_summary": telemetry_summary,
        "evidence": {
            "controllers": list(CONTROLLERS),
            "parallel_launches": launches,
            "sequential_resume_order": list(CONTROLLERS),
            "telemetry_sample_count": len(samples),
            "artifacts": artifacts,
        },
        "completed_utc": _utc_now(),
    }
    if failures:
        # Failure detail belongs inside the immutable launch records without
        # widening the closed receipt schema consumed by the queue.
        for failure in failures:
            launches[0].setdefault("gate_failures", []).append(failure)
    envelope = {"payload": payload, "payload_sha256": canonical_sha256(payload)}
    _atomic_create_json(receipt_path, envelope)
    if not passed:
        raise GateError(
            "Command parallel gate failed; receipt preserved at "
            f"{receipt_path}: {'; '.join(failures) or checks}"
        )
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=queue_module.DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = queue_module.validate_config(args.config)
        output_root = Path(config["_output_root"])
        output_dir = args.output_dir or queue_module.default_parallel_gate_path(output_root).parent
        receipt = run_gate(output_dir, args.config)
        print(json.dumps({"status": "PASS", "receipt": str(receipt)}, indent=2))
        return 0
    except (GateError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
