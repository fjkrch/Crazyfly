#!/usr/bin/env python3
"""Plan, execute, and resume a sequential, auditable G1 comparison queue."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator

from _bootstrap import ROOT, simulator_source_fingerprint

POLICIES = {
    "frozen_lif_original": "frozen_lif",
    "frozen_lif_degree_rewired": "frozen_lif_rewired",
    "gru_trainable": "gru",
    "mlp_engineering_baseline": "mlp",
}
BIOLOGICAL = {"frozen_lif_original", "frozen_lif_degree_rewired"}
TASKS = {
    "FlyG1-GoalReach-FreePosture-v0",
    "FlyG1-GoalSwitch-FreePosture-v0",
    "FlyG1-PushRecovery-FreePosture-v0",
}
IDLE_TIMEOUT_S = 5 * 60
WATCHDOG_POLL_S = 5.0
SHUTDOWN_GRACE_S = 10.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _execution_source_fingerprint() -> str:
    """Pin training/evaluation implementation as well as simulator task code."""
    paths = [ROOT / "scripts" / name for name in (
        "run_matrix.py", "train.py", "evaluate.py", "evaluation_protocol.py", "memory_watch.py", "_bootstrap.py",
    )]
    package = ROOT / "source" / "g1_fly_control" / "g1_fly_control"
    for component in ("connectome", "policies", "training", "evaluation"):
        paths.extend(sorted((package / component).glob("*.py")))
    paths.extend(sorted((package / "tasks" / "g1").glob("*.py")))
    digest = sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("task") not in TASKS:
        raise ValueError("Matrix config needs a known G1 task.")
    conditions, seeds = value.get("conditions"), value.get("seeds")
    if not isinstance(conditions, list) or not conditions or len(set(conditions)) != len(conditions) or set(conditions) - POLICIES.keys():
        raise ValueError("conditions must be unique known comparison conditions.")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers.")
    if type(value.get("interaction_budget")) is not int or value["interaction_budget"] <= 0:
        raise ValueError("interaction_budget must be a positive integer.")
    training, evaluation = value.get("training", {}), value.get("evaluation", {})
    if not isinstance(training, dict) or not isinstance(evaluation, dict):
        raise ValueError("training and evaluation must be objects.")
    for label, entry in (("training.num_envs", training.get("num_envs", 16)),
                         ("training.horizon", training.get("horizon", 32)),
                         ("evaluation.episodes", evaluation.get("episodes", 16))):
        if type(entry) is not int or entry <= 0:
            raise ValueError(f"{label} must be a positive integer.")
    rates = training.get("learning_rate_by_condition", {})
    if not isinstance(rates, dict) or set(rates) - set(conditions):
        raise ValueError("training.learning_rate_by_condition must map configured conditions only.")
    for condition, rate in rates.items():
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid learning rate for {condition}.")
    target_kl = training.get("target_kl")
    if target_kl is not None and (type(target_kl) not in (int, float) or not math.isfinite(target_kl) or target_kl <= 0):
        raise ValueError("training.target_kl must be positive and finite.")
    scenarios = evaluation.get("scenario_tasks", [value["task"]])
    if not isinstance(scenarios, list) or not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios) - TASKS:
        raise ValueError("evaluation.scenario_tasks must be unique known G1 tasks.")
    if type(evaluation.get("seed", 101)) is not int or evaluation.get("seed", 101) < 0:
        raise ValueError("evaluation.seed must be a nonnegative integer.")
    if evaluation.get("protocol", "default") not in {"default", "heldout_v1"}:
        raise ValueError("evaluation.protocol must be default or heldout_v1.")
    regimes = evaluation.get("reset_regimes", ["standing"])
    if regimes != ["standing"]:
        raise ValueError("Only the standing reset regime is currently implemented and validated.")
    if evaluation.get("held_out_targets") and evaluation.get("protocol") != "heldout_v1":
        raise ValueError("held_out_targets requires evaluation.protocol heldout_v1.")
    if evaluation.get("paired_disturbance_schedule") and evaluation.get("protocol") != "heldout_v1":
        raise ValueError("paired_disturbance_schedule requires evaluation.protocol heldout_v1.")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _console_result(path: Path, status: str) -> dict[str, Any] | None:
    """Find a script's final JSON result amid Isaac Sim console messages."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    decoder = json.JSONDecoder()
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].lstrip().startswith("{"):
            continue
        try:
            result, _ = decoder.raw_decode("\n".join(lines[index:]).lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("status") == status:
            return result
    return None


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop Isaac's child processes as well as the Python launcher."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=SHUTDOWN_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _run(command: list[str], log: Path) -> dict[str, Any]:
    """Run without a total-time limit, but stop a child silent for five minutes."""
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stalled = False
    with log.open("w", encoding="utf-8") as handle:
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env=child_env, start_new_session=os.name == "posix",
            )
        except OSError as exc:
            handle.write(f"Could not launch command: {exc}\n")
            exit_code = 127
        else:
            last_size = log.stat().st_size
            last_activity = time.monotonic()
            try:
                while True:
                    try:
                        exit_code = process.wait(timeout=WATCHDOG_POLL_S)
                        break
                    except subprocess.TimeoutExpired:
                        current_size = log.stat().st_size
                        if current_size != last_size:
                            last_size = current_size
                            last_activity = time.monotonic()
                        elif time.monotonic() - last_activity >= IDLE_TIMEOUT_S:
                            stalled = True
                            _stop_process_group(process)
                            exit_code = 124
                            handle.write(f"\nMatrix watchdog: no child output for {IDLE_TIMEOUT_S:g} s; stopped process group.\n")
                            handle.flush()
                            break
            except BaseException:
                _stop_process_group(process)
                raise
    return {"exit_code": exit_code, "wall_time_s": time.monotonic() - start,
            "log": str(log), "stalled": stalled, "idle_timeout_s": IDLE_TIMEOUT_S}


class QueueLockedError(RuntimeError):
    pass


@contextmanager
def _queue_lock(output: Path) -> Iterator[None]:
    """Allow only one runner to read/modify a queue at a time."""
    lock_path = output.with_name(output.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QueueLockedError(f"Matrix queue is already in use: {output}") from exc
        else:
            import msvcrt
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise QueueLockedError(f"Matrix queue is already in use: {output}") from exc
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _smoke_valid(report: Any, task: str) -> bool:
    return bool(isinstance(report, dict) and report.get("status") == "PASS" and report.get("task") == task
                and type(report.get("num_envs")) is int and report["num_envs"] >= 16
                and type(report.get("steps")) is int and report["steps"] >= 1000
                and report.get("random_actions") is True
                and type(report.get("random_action_scale")) in (int, float)
                and 0.1 <= report["random_action_scale"] <= 1.0
                and report.get("simulator_source_fingerprint") == simulator_source_fingerprint())


def _connectome(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "BLOCKED", "reason": "A real MaleCNS --connectome_manifest was not supplied."}
    try:
        from g1_fly_control.connectome import load_connectome
        circuit = load_connectome(path)  # rejects synthetic fixtures by default
    except (OSError, ValueError) as exc:
        return {"status": "BLOCKED", "path": str(path), "reason": str(exc)}
    return {"status": "PASS", "path": str(path), "checksum": circuit.checksum}


def _train_command(job: dict[str, Any], config: dict[str, Any], output: Path, connectome: Path | None) -> list[str]:
    training = config.get("training", {})
    command = [sys.executable, str(ROOT / "scripts/train.py"), "--task", config["task"], "--policy", job["policy"],
               "--num_envs", str(training.get("num_envs", 16)), "--horizon", str(training.get("horizon", 32)),
               "--interaction_budget", str(config["interaction_budget"]), "--seed", str(job["seed"]),
               "--output_dir", str(output.parent / output.stem / "training"), "--headless"]
    rate = training.get("learning_rate_by_condition", {}).get(job["condition"])
    if rate is not None:
        command += ["--learning_rate", str(rate)]
    if training.get("target_kl") is not None:
        command += ["--target_kl", str(training["target_kl"])]
    if job["condition"] in BIOLOGICAL:
        if connectome is not None:
            command += ["--connectome_manifest", str(connectome)]
        if job["condition"] == "frozen_lif_degree_rewired":
            command += ["--rewire_seed", str(job["seed"])]
    return command


def _eval_command(job: dict[str, Any], config: dict[str, Any], scenario: str, result_file: Path,
                  connectome: Path | None) -> list[str]:
    evaluation = config.get("evaluation", {})
    command = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--checkpoint", job["checkpoint"],
               "--task", scenario, "--episodes", str(evaluation.get("episodes", 16)),
               "--seed", str(evaluation.get("seed", 101)), "--output", str(result_file), "--headless"]
    if evaluation.get("protocol", "default") != "default":
        command += ["--protocol", evaluation["protocol"]]
    if job["condition"] in BIOLOGICAL and connectome is not None:
        command += ["--connectome_manifest", str(connectome)]
    return command


def _valid_episode_records(result: dict[str, Any], *, seed: int, episodes: int,
                           heldout: bool) -> bool:
    """Require the claimed evaluation count to have actual auditable rows."""
    rows = result.get("episodes")
    per_seed = result.get("per_seed")
    if not isinstance(rows, list) or len(rows) != episodes or not isinstance(per_seed, list) or len(per_seed) != 1:
        return False
    if (result.get("n_independent_training_seeds") != 1 or not isinstance(per_seed[0], dict)
            or per_seed[0].get("seed") != seed or per_seed[0].get("episodes") != episodes):
        return False
    if any(not isinstance(row, dict) or row.get("episode_id") != index or row.get("seed") != seed
           or type(row.get("success")) is not bool for index, row in enumerate(rows)):
        return False
    if heldout and any(not isinstance(row.get("schedule_events"), list)
                       or not isinstance(row.get("initial_state_sha256"), str)
                       or len(row["initial_state_sha256"]) != 64
                       or not isinstance(row.get("paired_plan_sha256"), str)
                       or len(row["paired_plan_sha256"]) != 64 for row in rows):
        return False
    return True


def _verified_training_metadata(job: dict[str, Any], manifest: dict[str, Any],
                                checkpoint_path: str | None, run_manifest_path: str | None) -> dict[str, Any] | None:
    """Verify both persisted metadata copies and a readable model checkpoint."""
    if not checkpoint_path or not run_manifest_path:
        return None
    checkpoint = Path(checkpoint_path).resolve()
    run_manifest = Path(run_manifest_path).resolve()
    if not checkpoint.is_file() or not run_manifest.is_file() or checkpoint.parent != run_manifest.parent:
        return None
    try:
        import torch
        metadata = json.loads(run_manifest.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if (not isinstance(metadata, dict) or not isinstance(payload, dict) or payload.get("format") != 1
            or not isinstance(payload.get("policy"), dict) or not payload["policy"]
            or not isinstance(payload.get("optimizer"), dict) or not isinstance(payload.get("metadata"), dict)):
        return None
    roll_size = (manifest["config"].get("training", {}).get("num_envs", 16)
                 * manifest["config"].get("training", {}).get("horizon", 32))
    budget = job["interaction_budget"]
    expected = (budget + roll_size - 1) // roll_size
    fields = ("task", "policy", "seed", "checkpoint_path", "requested_interaction_budget",
              "actual_environment_interactions", "completed_iterations", "num_envs", "connectome_checksum",
              "rewire_seed")
    saved = payload["metadata"]
    if any(metadata.get(field) != saved.get(field) for field in fields):
        return None
    if (metadata.get("status") != "executed_matrix_training" or metadata.get("task") != job["task"]
            or metadata.get("policy") != job["policy"] or metadata.get("seed") != job["seed"]
            or metadata.get("requested_interaction_budget") != budget
            or metadata.get("completed_iterations") != expected
            or metadata.get("actual_environment_interactions") != expected * roll_size
            or metadata.get("num_envs") != manifest["config"].get("training", {}).get("num_envs", 16)
            or Path(str(metadata.get("checkpoint_path", ""))).resolve() != checkpoint):
        return None
    if job["condition"] in BIOLOGICAL:
        if metadata.get("connectome_checksum") != manifest["prerequisites"]["real_connectome"].get("checksum"):
            return None
        if (job["condition"] == "frozen_lif_degree_rewired"
                and metadata.get("rewire_seed") != job["seed"]):
            return None
    return metadata


def _valid_evaluation_result(result: Any, *, job: dict[str, Any], manifest: dict[str, Any],
                             scenario: str) -> bool:
    evaluation = manifest["config"].get("evaluation", {})
    reported_scenario = result.get("scenario", {}) if isinstance(result, dict) else {}
    expected_schedule_sha = None
    if evaluation.get("protocol") == "heldout_v1":
        from evaluation_protocol import schedule_manifest
        expected_schedule_sha = schedule_manifest(
            evaluation.get("seed", 101), evaluation.get("episodes", 16), scenario
        )["sha256"]
    return bool(isinstance(result, dict) and result.get("status") == "executed"
                and Path(str(result.get("checkpoint", ""))).resolve() == Path(job["checkpoint"]).resolve()
                and result.get("task") == scenario and result.get("training_seed") == job["seed"]
                and result.get("evaluation_seed") == evaluation.get("seed", 101)
                and result.get("n_episodes") == evaluation.get("episodes", 16)
                and _valid_episode_records(
                    result, seed=job["seed"], episodes=evaluation.get("episodes", 16),
                    heldout=evaluation.get("protocol") == "heldout_v1",
                )
                and isinstance(reported_scenario, dict)
                and reported_scenario.get("evaluation_protocol") == evaluation.get("protocol", "default")
                and (expected_schedule_sha is None or
                     (reported_scenario.get("schedule") or {}).get("sha256") == expected_schedule_sha)
                and (not evaluation.get("held_out_targets") or reported_scenario.get("held_out_targets_verified") is True)
                and (not evaluation.get("paired_disturbance_schedule") or scenario != "FlyG1-PushRecovery-FreePosture-v0"
                     or reported_scenario.get("paired_disturbance_schedule") is True))


def _reconcile_resume_artifacts(manifest: dict[str, Any]) -> None:
    """Repair stale PASS states using files on disk before queue-only or execute."""
    scenarios = manifest["config"].get("evaluation", {}).get("scenario_tasks", [manifest["config"]["task"]])
    for job in manifest["jobs"]:
        if job["status"] not in {"passed", "training_complete", "failed", "running"}:
            continue
        metadata = _verified_training_metadata(job, manifest, job.get("checkpoint"), job.get("run_manifest"))
        if metadata is None:
            if job.get("checkpoint") or job["status"] in {"passed", "training_complete"}:
                job.pop("checkpoint", None)
                job.pop("run_manifest", None)
                job.pop("training_metadata", None)
                job["evaluations"] = {}
                job["status"] = "ready"
                job["reason"] = "Missing or invalid training checkpoint/metadata; training will rerun."
            continue
        if job["status"] != "passed":
            continue
        for scenario in scenarios:
            record = job["evaluations"].get(scenario, {})
            try:
                result = json.loads(Path(record["result_file"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, TypeError):
                result = None
            if record.get("status") != "passed" or not _valid_evaluation_result(
                result, job=job, manifest=manifest, scenario=scenario
            ):
                record["status"] = "needs_rerun"
                record.pop("result", None)
                job["evaluations"][scenario] = record
                job["status"] = "training_complete"
                job["reason"] = f"Missing or invalid evaluation for {scenario}; evaluation will rerun."
            else:
                record["result"] = result


def _save(path: Path, manifest: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    manifest["counts"] = counts
    if manifest.get("dry_run"):
        manifest["status"] = "dry_run"
    elif counts.get("running"):
        manifest["status"] = "running"
    elif counts.get("failed"):
        manifest["status"] = "partial_failure"
    elif counts.get("waiting_smoke_gate"):
        manifest["status"] = "waiting_smoke_gate"
    elif counts.get("ready"):
        manifest["status"] = "queued_with_blockers" if counts.get("blocked_missing_connectome") else "queued_ready"
    elif counts.get("pending") or counts.get("training_complete"):
        manifest["status"] = "partial" if counts.get("passed") or counts.get("training_complete") else "planned_not_started"
    elif counts.get("blocked_missing_connectome"):
        manifest["status"] = "partial_blocked" if counts.get("passed") else "blocked_missing_connectome"
    elif counts.get("blocked"):
        manifest["status"] = "partial_blocked" if counts.get("passed") else "blocked"
    else:
        manifest["status"] = "complete"
    manifest["updated_utc"] = _now()
    _atomic_json(path, manifest)
    summary = {"status": manifest["status"], "counts": counts, "config_hash": manifest["config_hash"],
               "execution_source_fingerprint": manifest["execution_source_fingerprint"],
               "prerequisites": manifest["prerequisites"], "jobs": [
                   {"id": job["id"], "status": job["status"], "reason": job.get("reason"),
                    "checkpoint": job.get("checkpoint"), "evaluations": {
                        scenario: {"status": record["status"], "result_file": record.get("result_file"),
                                   "per_seed": (record.get("result") or {}).get("per_seed")}
                        for scenario, record in job["evaluations"].items()}}
                   for job in manifest["jobs"]]}
    _atomic_json(path.with_name(path.stem + "_summary.json"), summary)


def _new_manifest(config: dict[str, Any], config_path: Path, output: Path, connectome: Path | None) -> dict[str, Any]:
    jobs = []
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            job = {"id": f"{condition}__seed-{seed}", "task": config["task"], "condition": condition,
                   "policy": POLICIES[condition], "seed": seed, "interaction_budget": config["interaction_budget"],
                   "status": "pending", "evaluations": {}}
            job["training_command"] = _train_command(job, config, output, connectome)
            jobs.append(job)
    return {"schema_version": 2, "created_utc": _now(), "config_path": str(config_path.resolve()),
            "config_hash": _digest(config), "execution_source_fingerprint": _execution_source_fingerprint(),
            "config": config, "jobs": jobs,
            "prerequisites": {
                "simulator_smoke": {"status": "PENDING", "requirement": "current-code, 16 environments, 1000 steps"},
                "real_connectome": _connectome(connectome),
                "useful_learning_pilot": {"status": "UNVERIFIED", "note": "This queue does not assert task learning."}},
            "evaluation_limits": [
                "Only the standing/default reset regime is currently validated; crouched/prone/supine remain pending.",
                "Protocol heldout_v1 targets and disturbances require a small simulator evaluation before full-matrix interpretation."],
            "output": str(output)}


def _smoke(manifest: dict[str, Any], report_path: Path | None) -> bool:
    gate = manifest["prerequisites"]["simulator_smoke"]
    task = manifest["config"]["task"]
    if gate.get("status") == "PASS" and _smoke_valid(gate.get("report"), task):
        return True
    if report_path is None:
        gate.update({"status": "PENDING", "reason": "Supply --smoke_report from a completed current-code diagnostic."})
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        gate.update({"status": "BLOCKED", "reason": f"Cannot read smoke report: {exc}"})
        return False
    gate.update({"status": "PASS" if _smoke_valid(report, task) else "BLOCKED", "report": report,
                 "source": str(report_path.resolve())})
    if gate["status"] != "PASS":
        gate["reason"] = "Simulator smoke must pass for this task with at least 16 environments and 1000 steps."
    return gate["status"] == "PASS"


def _run_job(job: dict[str, Any], manifest: dict[str, Any], output: Path, connectome: Path | None) -> None:
    training_metadata = _verified_training_metadata(
        job, manifest, job.get("checkpoint"), job.get("run_manifest")
    )
    if training_metadata is None:
        job.pop("checkpoint", None)
        job.pop("run_manifest", None)
        job["evaluations"] = {}
        job.update({"status": "running", "started_utc": _now()})
        _save(output, manifest)
        log = output.parent / output.stem / "logs" / f"{job['id']}__train.log"
        run = _run(job["training_command"], log)
        result = _console_result(log, "PASS")
        job["training_run"] = {**run, "result": result, "finished_utc": _now()}
        training_metadata = _verified_training_metadata(
            job, manifest, result.get("checkpoint") if result else None,
            result.get("run_manifest") if result else None,
        ) if run["exit_code"] == 0 else None
        if training_metadata is None:
            job.update({"status": "failed", "reason": "Training failed or checkpoint/metadata did not verify the requested budget and condition."})
            _save(output, manifest)
            return
        job.update({"checkpoint": str(Path(result["checkpoint"]).resolve()),
                    "run_manifest": result["run_manifest"], "status": "training_complete",
                    "training_metadata": {key: training_metadata.get(key) for key in (
                        "requested_interaction_budget", "actual_environment_interactions", "completed_iterations",
                        "training_wall_time_s", "model_total_parameters", "model_trainable_parameters",
                        "model_frozen_parameters", "connectome_checksum")}})
        _save(output, manifest)
    config = manifest["config"]
    for scenario in config.get("evaluation", {}).get("scenario_tasks", [config["task"]]):
        existing = job["evaluations"].get(scenario, {})
        if existing.get("status") == "passed":
            try:
                saved_result = json.loads(Path(existing["result_file"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, TypeError):
                saved_result = None
            if _valid_evaluation_result(saved_result, job=job, manifest=manifest, scenario=scenario):
                existing["result"] = saved_result
                continue
        result_file = output.parent / output.stem / "evaluations" / job["id"] / f"{scenario}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        command = _eval_command(job, config, scenario, result_file, connectome)
        job["evaluations"][scenario] = {"status": "running", "command": command, "result_file": str(result_file),
                                         "started_utc": _now()}
        _save(output, manifest)
        log = output.parent / output.stem / "logs" / f"{job['id']}__{scenario}__evaluate.log"
        run = _run(command, log)
        try:
            result = json.loads(result_file.read_text(encoding="utf-8")) if run["exit_code"] == 0 else None
        except (OSError, ValueError):
            result = None
        valid = _valid_evaluation_result(result, job=job, manifest=manifest, scenario=scenario)
        job["evaluations"][scenario].update({**run, "status": "passed" if valid else "failed",
                                             "result": result if valid else None, "finished_utc": _now()})
        if not valid:
            job.update({"status": "failed", "reason": f"Evaluation failed for {scenario}."})
            _save(output, manifest)
            return
        _save(output, manifest)
    job.update({"status": "passed", "finished_utc": _now()})
    job.pop("reason", None)
    _save(output, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true", help="Emit a queue without launching Isaac Sim.")
    mode.add_argument("--execute", action="store_true", help="Run eligible jobs sequentially after a verified smoke report.")
    mode.add_argument("--queue_only", action="store_true", help="Validate prerequisites and mark ready jobs; launch nothing.")
    parser.add_argument("--resume", action="store_true", help="Resume a manifest with the identical config hash.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "matrix_manifest.json")
    parser.add_argument("--connectome_manifest", type=Path, help="Authorized real MaleCNS manifest.")
    parser.add_argument("--smoke_report", type=Path, help="Required completed current-code smoke PASS JSON before training.")
    parser.add_argument("--max_jobs", type=int, help="Stop after this many eligible jobs; resume for the rest.")
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max_jobs must be positive.")
    if args.smoke_report and not (args.execute or args.queue_only):
        parser.error("--smoke_report requires --execute or --queue_only.")
    if args.queue_only and (not args.resume or args.smoke_report is None):
        parser.error("--queue_only requires --resume and --smoke_report.")
    try:
        config = _config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    output = args.output.expanduser().resolve()
    connectome = args.connectome_manifest.expanduser().resolve() if args.connectome_manifest else None
    try:
        with _queue_lock(output):
            return _main_locked(args, parser, config, output, connectome)
    except QueueLockedError as exc:
        parser.error(str(exc))


def _main_locked(args: argparse.Namespace, parser: argparse.ArgumentParser, config: dict[str, Any],
                 output: Path, connectome: Path | None) -> int:
    if output.exists() and not args.resume:
        parser.error(f"Output already exists: {output}. Use --resume or choose a new --output.")
    if args.resume:
        if not output.is_file():
            parser.error(f"Cannot resume; manifest does not exist: {output}")
        try:
            manifest = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read manifest: {exc}")
        expected_ids = [f"{condition}__seed-{seed}" for condition in config["conditions"] for seed in config["seeds"]]
        if manifest.get("schema_version") != 2 or manifest.get("config_hash") != _digest(config) or [job.get("id") for job in manifest.get("jobs", [])] != expected_ids:
            parser.error("Resume requires a version-2 manifest with identical config and jobs.")
        if manifest.get("execution_source_fingerprint") != _execution_source_fingerprint():
            parser.error("Training/evaluation source changed since queue creation; use a new output manifest.")
        previous = manifest["prerequisites"]["real_connectome"]
        current = _connectome(connectome)
        if previous.get("status") == "PASS" and current.get("checksum") != previous.get("checksum"):
            parser.error("Connectome changed since queue creation; use a new output manifest.")
        if current["status"] == "PASS":
            manifest["prerequisites"]["real_connectome"] = current
        for job in manifest["jobs"]:
            job["training_command"] = _train_command(job, config, output, connectome)
        _reconcile_resume_artifacts(manifest)
    else:
        manifest = _new_manifest(config, args.config, output, connectome)
    if args.dry_run or not (args.execute or args.queue_only):
        manifest["dry_run"] = bool(args.dry_run)
        _save(output, manifest)
        print(json.dumps({"status": manifest["status"], "jobs": len(manifest["jobs"]), "manifest": str(output)}, indent=2))
        return 0
    manifest["dry_run"] = False
    smoke_passed = _smoke(manifest, args.smoke_report)
    circuit_passed = manifest["prerequisites"]["real_connectome"]["status"] == "PASS"
    if args.queue_only:
        for job in manifest["jobs"]:
            # Inspection must never erase a checkpoint, a failed diagnostic, or
            # an in-flight job. The next --execute --resume decides what to retry.
            if job["status"] in {"passed", "training_complete", "failed", "running"}:
                continue
            if job["condition"] in BIOLOGICAL and not circuit_passed:
                job.update({"status": "blocked_missing_connectome",
                            "reason": manifest["prerequisites"]["real_connectome"]["reason"]})
            elif not smoke_passed:
                job.update({"status": "waiting_smoke_gate", "reason": "Current-code simulator smoke gate is not PASS."})
            else:
                job.update({"status": "ready"})
                job.pop("reason", None)
        _save(output, manifest)
        print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "manifest": str(output)}, indent=2))
        return 0
    if not smoke_passed:
        for job in manifest["jobs"]:
            if job["status"] != "passed":
                if job["condition"] in BIOLOGICAL and not circuit_passed:
                    job.update({"status": "blocked_missing_connectome",
                                "reason": manifest["prerequisites"]["real_connectome"]["reason"]})
                else:
                    job.update({"status": "waiting_smoke_gate",
                                "reason": "Current-code simulator smoke gate is not PASS."})
        _save(output, manifest)
        print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "manifest": str(output)}, indent=2))
        return 2
    attempted = 0
    for job in manifest["jobs"]:
        if job["status"] == "passed":
            continue
        if job["condition"] in BIOLOGICAL and not circuit_passed:
            job.update({"status": "blocked_missing_connectome",
                        "reason": manifest["prerequisites"]["real_connectome"]["reason"]})
            _save(output, manifest)
            continue
        if args.max_jobs is not None and attempted >= args.max_jobs:
            break
        job.pop("reason", None)
        _run_job(job, manifest, output, connectome)
        attempted += 1
    _save(output, manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "manifest": str(output)}, indent=2))
    return 1 if manifest["counts"].get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
