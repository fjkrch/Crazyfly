#!/usr/bin/env python3
"""Read-only JSON snapshot of a matrix queue, its logs, files, and live processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_QUEUE = Path("runs/main_matrix_malecns_v1_heldout_v3_20260913.json")


def _tail_iteration(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    # The log can be several MB and the last line may be incomplete while Isaac writes it.
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - 1024 * 1024)
        handle.seek(start)
        lines = handle.read().splitlines()
    if start and lines:
        lines.pop(0)
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            continue
        if (not isinstance(row, dict) or type(row.get("iteration")) is not int
                or row["iteration"] < 0 or "loss" not in row):
            continue
        bad = [key for key, value in row.items()
               if type(value) in (int, float) and not math.isfinite(value)]
        return {"iteration": row["iteration"], "nonfinite_metric_keys": bad}
    return None


def _processes(proc_root: Path) -> tuple[bool, list[tuple[int, list[str], Path]]]:
    if not proc_root.is_dir():
        return False, []
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode("utf-8", errors="replace") for part in
                    (entry / "cmdline").read_bytes().split(b"\0") if part]
            cwd = (entry / "cwd").resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if argv:
            found.append((int(entry.name), argv, cwd))
    return True, found


def _runner_pids(processes: list[tuple[int, list[str], Path]], queue: Path) -> list[int]:
    pids = []
    for pid, argv, cwd in processes:
        if not any(Path(arg).name == "run_matrix.py" for arg in argv[:3]):
            continue
        try:
            output = argv[argv.index("--output") + 1]
        except (ValueError, IndexError):
            continue
        if (cwd / output).resolve() == queue:
            pids.append(pid)
    return sorted(pids)


def _file_age(path: Path, now: float) -> float | None:
    try:
        return round(max(0.0, now - path.stat().st_mtime), 1)
    except OSError:
        return None


def _evaluation_files(queue: Path, job: dict[str, Any], scenarios: list[str]) -> dict[str, Any]:
    results = {}
    for scenario in scenarios:
        record = job.get("evaluations", {}).get(scenario, {})
        path = Path(record.get("result_file") or
                    queue.parent / queue.stem / "evaluations" / job["id"] / f"{scenario}.json")
        observed_status = None
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                observed_status = value.get("status") if isinstance(value, dict) else None
            except (OSError, ValueError):
                pass
        results[scenario] = {"queue_status": record.get("status", "pending"),
                             "file": str(path), "file_exists": path.is_file(),
                             "observed_file_status": observed_status,
                             "runner_verified": record.get("status") == "passed"}
    return results


def snapshot(queue: Path, *, proc_root: Path = Path("/proc"),
             now: float | None = None, stale_after_s: float = 300.0) -> dict[str, Any]:
    queue = queue.resolve()
    manifest = json.loads(queue.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    proc_available, processes = _processes(proc_root)
    runner_pids = _runner_pids(processes, queue) if proc_available else []
    config = manifest["config"]
    training = config.get("training", {})
    steps_per_update = training.get("num_envs", 16) * training.get("horizon", 32)
    scenarios = config.get("evaluation", {}).get("scenario_tasks", [config["task"]])
    active = []
    for job in manifest["jobs"]:
        if job.get("status") not in {"running", "training_complete"}:
            continue
        log = queue.parent / queue.stem / "logs" / f"{job['id']}__train.log"
        age = _file_age(log, now)
        update = _tail_iteration(log)
        expected = math.ceil(job["interaction_budget"] / steps_per_update)
        training_pids = sorted(pid for pid, argv, _ in processes
                               if argv == job.get("training_command")) if proc_available else []
        evaluation_pids = {}
        for scenario, record in job.get("evaluations", {}).items():
            if record.get("status") == "running":
                evaluation_pids[scenario] = sorted(pid for pid, argv, _ in processes
                                                   if argv == record.get("command")) if proc_available else []
        child_pids = training_pids + [pid for pids in evaluation_pids.values() for pid in pids]
        running_evaluations = [name for name, record in job.get("evaluations", {}).items()
                               if record.get("status") == "running"]
        stage_log = (queue.parent / queue.stem / "logs" /
                     f"{job['id']}__{running_evaluations[0]}__evaluate.log") if running_evaluations else (
                         log if job["status"] == "running" else None)
        stage_age = _file_age(stage_log, now) if stage_log else None
        if not proc_available:
            health = "process_check_unavailable"
        elif child_pids and (stage_age is None or stage_age < stale_after_s):
            health = "active_process"
        elif stage_age is not None and stage_age >= stale_after_s:
            health = "needs_inspection"
        elif not runner_pids and not child_pids:
            health = "needs_inspection"
        else:
            health = "transition_or_unverified"
        checkpoint = job.get("checkpoint")
        run_manifest = job.get("run_manifest")
        training_dirs = (queue.parent / queue.stem / "training").glob(
            f"*/{job['task']}/{job['policy']}/seed-{job['seed']}")
        observed_training_dirs = sorted(path for path in training_dirs if path.is_dir())
        observed_checkpoints = [str(path / "checkpoint.pt") for path in observed_training_dirs
                                if (path / "checkpoint.pt").is_file()]
        observed_run_manifests = [str(path / "manifest.json") for path in observed_training_dirs
                                  if (path / "manifest.json").is_file()]
        active.append({
            "id": job["id"], "queue_status": job["status"], "health": health,
            "health_is_terminal": False,
            "train_log": str(log), "train_log_exists": log.is_file(),
            "train_log_age_s": age,
            "active_stage_log": str(stage_log) if stage_log else None,
            "active_stage_log_age_s": stage_age,
            "active_stage_log_stale": stage_age is not None and stage_age >= stale_after_s,
            "completed_updates": min(expected, update["iteration"] + 1) if update else None,
            "expected_updates": expected,
            "completed_interactions_observed": min(expected, update["iteration"] + 1) * steps_per_update if update else None,
            "last_iteration": update["iteration"] if update else None,
            "nonfinite_metric_keys_last_update": update["nonfinite_metric_keys"] if update else None,
            "training_pids": training_pids if proc_available else None,
            "evaluation_pids": evaluation_pids if proc_available else None,
            "checkpoint": {"queue_path": checkpoint, "queue_path_exists": Path(checkpoint).is_file() if checkpoint else False,
                           "observed_paths": observed_checkpoints,
                           "runner_verified": job.get("status") in {"training_complete", "passed"}},
            "run_manifest": {"queue_path": run_manifest, "queue_path_exists": Path(run_manifest).is_file() if run_manifest else False,
                             "observed_paths": observed_run_manifests},
            "evaluations": _evaluation_files(queue, job, scenarios),
        })
    return {"queue": str(queue), "observed_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "queue_status": manifest.get("status"), "queue_counts": manifest.get("counts"),
            "process_check_available": proc_available, "runner_pids": runner_pids if proc_available else None,
            "active_jobs": active,
            "note": "Health is one read-only observation, not a terminal verdict; file existence is not result validation."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--stale_after_s", type=float, default=300.0)
    args = parser.parse_args()
    if not math.isfinite(args.stale_after_s) or args.stale_after_s <= 0:
        parser.error("--stale_after_s must be positive and finite")
    try:
        report = snapshot(args.queue, stale_after_s=args.stale_after_s)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"cannot inspect queue: {exc}")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
