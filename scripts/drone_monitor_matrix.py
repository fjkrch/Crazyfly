#!/usr/bin/env python3
"""Read-only status and memory monitor for a Crazyflie matrix queue.

The default is one snapshot.  Set ``--poll_seconds`` to poll; the monitor only
reads JSON and procfs and never signals, resumes, pauses, or launches a process.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


VALID_STATUSES = ("pending", "running", "completed", "failed", "paused", "cancelled")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_queue(queue: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    if queue.get("schema_version") != 1:
        raise ValueError("queue.schema_version must be 1")
    if queue.get("label") not in {"integration", "main"}:
        raise ValueError("queue.label must be 'integration' or 'main'")
    config = queue.get("config")
    jobs = queue.get("jobs")
    if not isinstance(config, Mapping) or not isinstance(jobs, list):
        raise ValueError("queue must contain a config object and jobs list")
    controllers, seeds = config.get("controllers"), config.get("seeds")
    if not isinstance(controllers, list) or not controllers or any(not isinstance(item, str) for item in controllers):
        raise ValueError("config.controllers must be a non-empty list of names")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
    ):
        raise ValueError("config.seeds must be a non-empty integer list")
    if len(set(controllers)) != len(controllers) or len(set(seeds)) != len(seeds):
        raise ValueError("controller and seed design values must be unique")
    return list(controllers), list(seeds)


def _system_memory() -> dict[str, Any] | None:
    path = Path("/proc/meminfo")
    try:
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            rows[key] = int(value.strip().split()[0])
        total = rows["MemTotal"] * 1024
        available = rows["MemAvailable"] * 1024
    except (OSError, ValueError, KeyError, IndexError):
        return None
    used = total - available
    return {
        "source": "/proc/meminfo",
        "total_gib": round(total / 1024**3, 3),
        "used_gib": round(used / 1024**3, 3),
        "available_gib": round(available / 1024**3, 3),
        "used_percent": round(100.0 * used / total, 3) if total else None,
    }


def _proc_command(pid_dir: Path) -> list[str] | None:
    try:
        data = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return None
    if not data:
        return None
    return [part.decode("utf-8", errors="replace") for part in data.rstrip(b"\0").split(b"\0")]


def _proc_memory(pid_dir: Path) -> dict[str, float | int | None]:
    values: dict[str, int] = {}
    try:
        for line in (pid_dir / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return {
        "pid": int(pid_dir.name),
        "rss_mib": round(values["VmRSS"] / 1024, 3) if "VmRSS" in values else None,
        "peak_rss_mib": round(values["VmHWM"] / 1024, 3) if "VmHWM" in values else None,
    }


def _matching_processes(
    commands: Sequence[Sequence[str]], proc: Path = Path("/proc")
) -> list[dict[str, Any]] | None:
    if not proc.is_dir():
        return None
    expected: set[tuple[str, ...]] = set()
    for command in commands:
        if not command:
            continue
        normalized = tuple(str(part) for part in command)
        expected.add(normalized)
        if any(Path(part).name == "drone_train.py" for part in normalized):
            # The queue stores the fresh command and the runner appends this
            # one exact flag when a valid checkpoint exists.
            expected.add((*normalized, "--resume"))
    if not expected:
        return []
    matches: list[dict[str, Any]] = []
    try:
        candidates = list(proc.iterdir())
    except OSError:
        return None
    for candidate in candidates:
        if not candidate.name.isdigit():
            continue
        command = _proc_command(candidate)
        if command is not None and tuple(command) in expected:
            matches.append({**_proc_memory(candidate), "command": command})
    return sorted(matches, key=lambda row: int(row["pid"]))


def _training_observation(job: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(job.get("run_dir", ""))) / "training_manifest.json"
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "status": None,
        "environment_interactions": None,
        "completed_updates": None,
        "memory_gate": None,
        "latest_memory_sample": None,
        "read_error": None,
    }
    if not path.is_file():
        return result
    try:
        manifest = _read_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["read_error"] = str(exc)
        return result
    samples = manifest.get("memory_samples")
    result.update(
        {
            "status": manifest.get("status"),
            "environment_interactions": manifest.get("environment_interactions"),
            "requested_interactions": manifest.get("requested_interactions"),
            "completed_updates": manifest.get("completed_updates"),
            "memory_gate": manifest.get("memory_gate") if isinstance(manifest.get("memory_gate"), Mapping) else None,
            "latest_memory_sample": samples[-1] if isinstance(samples, list) and samples else None,
            "controller_report": (
                manifest.get("controller_report")
                if isinstance(manifest.get("controller_report"), Mapping)
                else None
            ),
        }
    )
    return result


def _current_stage(job: Mapping[str, Any]) -> tuple[str, Sequence[str]]:
    evaluations = job.get("evaluations")
    if isinstance(evaluations, list):
        for bundle in evaluations:
            if isinstance(bundle, Mapping) and bundle.get("status") == "running":
                command = bundle.get("command")
                return f"evaluation:{bundle.get('scenario')}", command if isinstance(command, list) else []
    command = job.get("training_command")
    return "training", command if isinstance(command, list) else []


def snapshot(path: Path) -> dict[str, Any]:
    """Return one queue snapshot without mutating queue or process state."""

    queue_path = Path(path).resolve()
    queue = _read_object(queue_path)
    controllers, seeds = _validate_queue(queue)
    jobs = queue["jobs"]
    expected = [(controller, seed) for controller in controllers for seed in seeds]
    job_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    structural_issues: list[str] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            structural_issues.append(f"jobs[{index}] is not an object")
            continue
        key = (job.get("controller"), job.get("seed"))
        if key not in expected:
            structural_issues.append(f"unexpected job identity {key!r}")
        elif key in job_index:
            structural_issues.append(f"duplicate job identity {key!r}")
        else:
            job_index[key] = job

    rows: list[dict[str, Any]] = []
    counts = Counter()
    for controller, seed in expected:
        job = job_index.get((controller, seed))
        if job is None:
            counts["failed"] += 1
            structural_issues.append(f"missing planned cell {controller}__seed-{seed}")
            rows.append(
                {
                    "id": f"{controller}__seed-{seed}",
                    "controller": controller,
                    "seed": seed,
                    "status": "failed",
                    "record_state": "missing",
                }
            )
            continue
        status = job.get("status")
        if status not in VALID_STATUSES:
            structural_issues.append(f"{job.get('id')}: invalid status {status!r}")
            status = "failed"
        counts[status] += 1
        rows.append(
            {
                "id": job.get("id"),
                "controller": controller,
                "seed": seed,
                "status": status,
                "record_state": "present",
                "evaluation_statuses": {
                    str(bundle.get("scenario")): bundle.get("status")
                    for bundle in job.get("evaluations", [])
                    if isinstance(bundle, Mapping)
                },
            }
        )

    # Running wins.  A paused cell is the useful current cell when no process
    # is active; otherwise show the next pending cell so a queued run remains
    # easy to inspect.
    current_job: Mapping[str, Any] | None = None
    for wanted in ("running", "paused", "pending"):
        current_job = next(
            (job for job in jobs if isinstance(job, Mapping) and job.get("status") == wanted),
            None,
        )
        if current_job is not None:
            break

    current: dict[str, Any] | None = None
    if current_job is not None:
        stage, active_command = _current_stage(current_job)
        all_commands: list[Sequence[str]] = []
        train_command = current_job.get("training_command")
        if isinstance(train_command, list):
            all_commands.append(train_command)
        for bundle in current_job.get("evaluations", []):
            if isinstance(bundle, Mapping) and isinstance(bundle.get("command"), list):
                all_commands.append(bundle["command"])
        current = {
            "id": current_job.get("id"),
            "controller": current_job.get("controller"),
            "seed": current_job.get("seed"),
            "status": current_job.get("status"),
            "stage": stage,
            "active_command": active_command,
            "training": _training_observation(current_job),
            "matching_processes": _matching_processes(all_commands),
        }

    status_counts = {status: counts.get(status, 0) for status in VALID_STATUSES}
    completed = status_counts["completed"]
    return {
        "schema_version": 1,
        "observed_utc": _utc_now(),
        "queue": str(queue_path),
        "label": queue.get("label"),
        "queue_status": queue.get("status"),
        "fingerprint_scope": "per_job",
        "fingerprints_source": "queue_declared_expected_fingerprint",
        "resolved_config": dict(queue["config"]),
        "config_sha256": queue.get("config_sha256"),
        "fingerprints": {
            str(job.get("id")): job.get("expected_fingerprint")
            for job in jobs
            if isinstance(job, Mapping) and isinstance(job.get("id"), str)
        },
        "planned_jobs": len(expected),
        "completed_jobs": completed,
        "completed_fraction": f"{completed}/{len(expected)}",
        "status_counts": status_counts,
        "queue_declared_counts": queue.get("counts"),
        "current_job": current,
        "system_memory_now": _system_memory(),
        "jobs": rows,
        "structural_issues": structural_issues,
        "read_only": True,
        "note": (
            "This is a point-in-time read-only observation. Stored memory-gate values are sampled "
            "training measurements; current procfs values are not continuous peaks."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--poll_seconds", "--poll-seconds", type=float, default=0.0,
        help="Polling interval; zero (the default) reads the queue once",
    )
    parser.add_argument(
        "--max_polls", "--max-polls", type=int,
        help="Optional finite poll count; requires --poll_seconds > 0",
    )
    args = parser.parse_args()
    if not args.queue.is_file():
        parser.error(f"queue does not exist or is not a file: {args.queue}")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 0:
        parser.error("--poll_seconds must be finite and non-negative")
    if args.max_polls is not None and args.max_polls < 1:
        parser.error("--max_polls must be positive")
    if args.max_polls is not None and args.poll_seconds == 0:
        parser.error("--max_polls requires --poll_seconds > 0")

    poll = 0
    try:
        while True:
            poll += 1
            try:
                report = snapshot(args.queue)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                parser.error(f"cannot inspect queue: {exc}")
            if args.poll_seconds > 0:
                print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)
            else:
                print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            if args.poll_seconds == 0 or (args.max_polls is not None and poll >= args.max_polls):
                return 1 if report["structural_issues"] else 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
