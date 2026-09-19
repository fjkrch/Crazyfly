#!/usr/bin/env python3
"""Record paired held-out root/contact traces for a validated 20-job matrix.

This is a separate, sequential companion queue. The default is a CPU-only dry
run; only ``--execute`` starts Isaac Sim. A main matrix must already be fully
complete and pass ``summarize_matrix`` validation before either mode can plan
recordings. Resume skips a recording only when its contents and all recorded
checkpoint/evaluation/source identities still match the exact inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from zipfile import BadZipFile

import numpy as np

from _bootstrap import ROOT
from regime_metrics import summarize_recording
from run_matrix import _execution_source_fingerprint, _queue_lock, _run, QueueLockedError
from summarize_matrix import summarize


CONDITIONS = (
    "frozen_lif_original", "frozen_lif_degree_rewired",
    "gru_trainable", "mlp_engineering_baseline",
)
SCENARIOS = (
    "FlyG1-GoalReach-FreePosture-v0",
    "FlyG1-GoalSwitch-FreePosture-v0",
    "FlyG1-PushRecovery-FreePosture-v0",
)
BIOLOGICAL = frozenset(CONDITIONS[:2])
RECORDER_CODE_FILES = (
    "record_heldout_regimes.py", "evaluate.py", "evaluation_protocol.py",
    "_bootstrap.py", "train.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorder_code_sha256() -> dict[str, str]:
    return {f"scripts/{name}": _sha256(ROOT / "scripts" / name) for name in RECORDER_CODE_FILES}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_main_matrix(path: Path) -> tuple[dict[str, Any], str]:
    main = _read_json(path)
    config = main.get("config", {})
    evaluation = config.get("evaluation", {}) if isinstance(config, dict) else {}
    if (main.get("schema_version") != 2 or main.get("status") != "complete"
            or main.get("counts") != {"passed": 20}
            or not isinstance(config, dict)
            or config.get("conditions") != list(CONDITIONS)
            or config.get("seeds") != list(range(5))
            or not isinstance(evaluation, dict)
            or evaluation.get("protocol") != "heldout_v1"
            or evaluation.get("episodes") != 16 or evaluation.get("seed") != 101
            or evaluation.get("scenario_tasks") != list(SCENARIOS)):
        raise ValueError("Main matrix must have all 20 passed jobs in the four-condition, five-seed heldout_v1 design.")
    current_source = _execution_source_fingerprint()
    if main.get("execution_source_fingerprint") != current_source:
        raise ValueError("Main matrix execution source differs from the current code.")
    report = summarize(path)
    if (report.get("status") != "complete" or report.get("validated_jobs") != 20
            or report.get("expected_jobs") != 20 or report.get("validation_errors")
            or report.get("source_execution_fingerprint") != current_source):
        raise ValueError("Main matrix failed the independent 20-job report validator: "
                         + "; ".join(report.get("validation_errors", [])[:3]))
    return main, report["manifest_sha256"]


def _planned_jobs(main: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    connectome = main.get("prerequisites", {}).get("real_connectome", {})
    if connectome.get("status") != "PASS":
        raise ValueError("Main matrix has no passed real-connectome prerequisite.")
    connectome_path = Path(str(connectome.get("path", ""))).resolve(strict=True)
    connectome_sha = _sha256(connectome_path)
    jobs: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in range(5):
            job_id = f"{condition}__seed-{seed}"
            matches = [job for job in main["jobs"] if job.get("id") == job_id]
            if len(matches) != 1 or matches[0].get("condition") != condition or matches[0].get("seed") != seed:
                raise ValueError(f"Missing or ambiguous main job: {job_id}")
            source = matches[0]
            checkpoint = Path(source["checkpoint"]).resolve(strict=True)
            for scenario in SCENARIOS:
                evaluation_json = Path(source["evaluations"][scenario]["result_file"]).resolve(strict=True)
                row_id = f"{job_id}__{scenario}"
                jobs.append({
                    "id": row_id, "condition": condition, "training_seed": seed,
                    "evaluation_seed": 101, "scenario": scenario,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                    "evaluation_json": str(evaluation_json),
                    "evaluation_json_sha256": _sha256(evaluation_json),
                    "connectome_manifest": str(connectome_path) if condition in BIOLOGICAL else None,
                    "connectome_manifest_sha256": connectome_sha if condition in BIOLOGICAL else None,
                    "output_npz": str(output.parent / output.stem / "recordings" / job_id / f"{scenario}.npz"),
                    "output_npz_sha256": None,
                    "log": str(output.parent / output.stem / "logs" / f"{row_id}.log"),
                    "status": "ready",
                })
    return jobs


def _new_manifest(main_path: Path, main: dict[str, Any], main_sha: str, output: Path) -> dict[str, Any]:
    return {
        "schema_version": "regime_matrix_v1", "created_utc": _now(),
        "main_matrix": str(main_path), "main_matrix_sha256": main_sha,
        "execution_source_fingerprint": main["execution_source_fingerprint"],
        "recorder_code_sha256": _recorder_code_sha256(),
        "output": str(output), "jobs": _planned_jobs(main, output),
    }


def _save(path: Path, manifest: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    manifest["counts"] = counts
    if counts.get("running"):
        status = "running"
    elif counts.get("failed"):
        status = "partial_failure"
    elif counts.get("passed") == len(manifest["jobs"]) == 60:
        status = "complete"
    else:
        status = "planned"
    manifest["status"] = status
    manifest["updated_utc"] = _now()
    _atomic_json(path, manifest)


def _recording_valid(row: dict[str, Any], manifest: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate the artifact itself before treating a previous PASS as reusable."""
    path = Path(row["output_npz"])
    if not path.is_file():
        return False, "recording file is missing"
    try:
        actual_sha = _sha256(path)
        if row.get("output_npz_sha256") not in (None, actual_sha):
            return False, "recording SHA differs from the queue"
        # The analyzer verifies array dimensions, finite valid samples, prefix
        # masks, and the metadata JSON contract without launching Isaac.
        summary = summarize_recording(path, force_threshold_n=20.0)
        metadata = summary["metadata"]
        expected = {
            "schema_version": "heldout_regimes_v1",
            "condition": row["condition"], "scenario": row["scenario"],
            "task": row["scenario"], "training_seed": row["training_seed"],
            "evaluation_seed": 101, "episode_count": 16,
            "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
            "evaluation_json": row["evaluation_json"],
            "evaluation_json_sha256": row["evaluation_json_sha256"],
            "connectome_manifest": row["connectome_manifest"],
            "connectome_manifest_sha256": row["connectome_manifest_sha256"],
            "execution_source_fingerprint": manifest["execution_source_fingerprint"],
            "code_sha256": manifest["recorder_code_sha256"],
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                return False, f"recording metadata {field} differs from the queue"
        evaluation = _read_json(Path(row["evaluation_json"]))
        scenario = evaluation["scenario"]
        episodes = evaluation["episodes"]
        checks = metadata.get("reference_checks")
        if not isinstance(checks, dict) or checks.get("required_match") is not True:
            return False, "companion replay outcomes/events did not match the reference evaluation"
        for field in ("success_match_by_episode", "target_time_match_by_episode",
                      "scheduled_event_match_by_episode"):
            values = checks.get(field)
            if not isinstance(values, list) or len(values) != 16 or any(value is not True for value in values):
                return False, f"companion replay {field} is incomplete or false"
        if (checks.get("control_steps_match") is not True
                or checks.get("replay_control_steps") != evaluation.get("execution", {}).get("control_steps")
                or checks.get("replay_success_by_episode") != [item.get("success") for item in episodes]):
            return False, "companion replay control steps or success differ from evaluation"
        full_event_flags = checks.get("full_event_match_by_episode")
        if (not isinstance(full_event_flags, list) or len(full_event_flags) != 16
                or any(type(value) is not bool for value in full_event_flags)):
            return False, "companion replay full-event diagnostics are missing"
        replay_times = checks.get("replay_target_time_s_by_episode")
        if not isinstance(replay_times, list) or len(replay_times) != 16:
            return False, "companion replay target times are missing"
        for actual, episode in zip(replay_times, episodes):
            expected_time = episode.get("time_to_target_s")
            if actual is None and expected_time is None:
                continue
            if (type(actual) not in (int, float) or type(expected_time) not in (int, float)
                    or not math.isfinite(actual) or not math.isfinite(expected_time)
                    or not math.isclose(actual, expected_time, rel_tol=0.0, abs_tol=1e-4)):
                return False, "companion replay target time differs from evaluation"
        if (metadata.get("scenario_schedule_sha256") != scenario["schedule"]["sha256"]
                or metadata.get("initial_state_sha256") != [item["initial_state_sha256"] for item in episodes]
                or metadata.get("paired_plan_sha256") != [item["paired_plan_sha256"] for item in episodes]
                or summary["per_training_seed"]["episode_count"] != 16):
            return False, "recording schedule or episode pairing differs from evaluation"
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["episode_id"].tolist() != list(range(16)):
                return False, "recording episode IDs are not 0..15"
        return True, None
    except (OSError, ValueError, KeyError, TypeError, IndexError, BadZipFile) as exc:
        return False, f"recording validation failed: {exc}"


def _command(row: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "record_heldout_regimes.py"),
               "--checkpoint", row["checkpoint"], "--evaluation_json", row["evaluation_json"],
               "--output", row["output_npz"], "--headless"]
    if row["connectome_manifest"]:
        command += ["--connectome_manifest", row["connectome_manifest"]]
    return command


def _check_sources(manifest: dict[str, Any]) -> None:
    if manifest["execution_source_fingerprint"] != _execution_source_fingerprint():
        raise ValueError("Execution source changed during the regime queue.")
    if manifest["recorder_code_sha256"] != _recorder_code_sha256():
        raise ValueError("Recorder source changed during the regime queue.")
    for row in manifest["jobs"]:
        for field, sha_field in (("checkpoint", "checkpoint_sha256"),
                                 ("evaluation_json", "evaluation_json_sha256"),
                                 ("connectome_manifest", "connectome_manifest_sha256")):
            if row[field] is not None and _sha256(Path(row[field])) != row[sha_field]:
                raise ValueError(f"{row['id']}: {field} changed since planning; use a new queue.")


def _resume_manifest(output: Path, planned: dict[str, Any]) -> dict[str, Any]:
    saved = _read_json(output)
    for field in ("schema_version", "main_matrix", "main_matrix_sha256",
                  "execution_source_fingerprint", "recorder_code_sha256", "output"):
        if saved.get(field) != planned[field]:
            raise ValueError(f"Cannot resume: {field} differs from the validated main matrix or current source.")
    if len(saved.get("jobs", [])) != 60:
        raise ValueError("Cannot resume a regime queue without all 60 planned rows.")
    identity = ("id", "condition", "training_seed", "evaluation_seed", "scenario",
                "checkpoint", "checkpoint_sha256", "evaluation_json", "evaluation_json_sha256",
                "connectome_manifest", "connectome_manifest_sha256", "output_npz", "log")
    for row, expected in zip(saved["jobs"], planned["jobs"]):
        if any(row.get(field) != expected[field] for field in identity):
            raise ValueError(f"Cannot resume: job identity differs for {expected['id']}.")
        if row.get("status") not in {"ready", "running", "passed", "failed"}:
            raise ValueError(f"Cannot resume: invalid job status for {expected['id']}.")
        if row["status"] == "passed":
            stored_sha = row.get("output_npz_sha256")
            if not isinstance(stored_sha, str) or len(stored_sha) != 64:
                valid, reason = False, "passed recording has no stored SHA-256"
            else:
                valid, reason = _recording_valid(row, saved)
            if not valid:
                row.update({"status": "ready", "output_npz_sha256": None,
                            "reason": f"Previous recording is not reusable: {reason}"})
        elif row["status"] == "running":
            row.update({"status": "ready", "output_npz_sha256": None,
                        "reason": "Previous recorder exited without a validated PASS; will retry."})
    return saved


def _main_locked(args: argparse.Namespace, output: Path) -> int:
    main_path = args.main_matrix.expanduser().resolve(strict=True)
    main, main_sha = _validated_main_matrix(main_path)
    planned = _new_manifest(main_path, main, main_sha, output)
    if args.resume:
        if not output.is_file():
            raise ValueError(f"Cannot resume; regime queue does not exist: {output}")
        manifest = _resume_manifest(output, planned)
    else:
        if output.exists():
            raise ValueError(f"Output already exists: {output}; use --resume or a new output path.")
        manifest = planned
    _check_sources(manifest)
    _save(output, manifest)
    if not args.execute:
        print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                          "manifest": str(output)}, sort_keys=True))
        return 0
    attempted = 0
    for row in manifest["jobs"]:
        if row["status"] == "passed":
            continue
        if args.max_jobs is not None and attempted >= args.max_jobs:
            break
        _check_sources(manifest)
        row.update({"status": "running", "started_utc": _now(), "command": _command(row)})
        row.pop("reason", None)
        _save(output, manifest)
        run = _run(row["command"], Path(row["log"]))
        row["run"] = {**run, "finished_utc": _now()}
        valid, reason = _recording_valid(row, manifest) if run["exit_code"] == 0 else (False, "recorder exited nonzero")
        if valid:
            row.update({"status": "passed", "output_npz_sha256": _sha256(Path(row["output_npz"])),
                        "finished_utc": _now()})
        else:
            row.update({"status": "failed", "output_npz_sha256": None,
                        "reason": reason, "finished_utc": _now()})
        _save(output, manifest)
        attempted += 1
        if not valid:
            break
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "manifest": str(output)}, sort_keys=True))
    return 1 if manifest["counts"].get("failed") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main_matrix", type=Path, required=True, help="Completed v3 heldout matrix manifest JSON.")
    parser.add_argument("--output", type=Path, help="Regime queue JSON; defaults beside main matrix.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true", help="Plan or inspect only; default mode.")
    mode.add_argument("--execute", action="store_true", help="Run Isaac recorder jobs sequentially.")
    parser.add_argument("--resume", action="store_true", help="Reuse only identity-validated recordings.")
    parser.add_argument("--max_jobs", type=int, help="Stop after this many attempted recordings.")
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max_jobs must be positive.")
    if args.max_jobs is not None and not args.execute:
        parser.error("--max_jobs requires --execute.")
    main_path = args.main_matrix.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              main_path.with_name(main_path.stem + "_regimes.json"))
    if output == main_path:
        parser.error("Regime output must differ from the main matrix manifest.")
    try:
        with _queue_lock(output):
            return _main_locked(args, output)
    except (OSError, ValueError, KeyError, QueueLockedError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
