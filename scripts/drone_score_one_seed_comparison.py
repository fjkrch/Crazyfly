#!/usr/bin/env python3
"""Score the frozen one-seed, task-separated Crazyflie comparison.

This is a CPU-only, fail-closed reporter.  It never imports or starts Isaac.
It authenticates the completed queue artifacts with the same validators used
by :mod:`drone_run_matrix`, independently checks the fixed held-out design,
and then applies a result-independent 100-point score.  Missing or invalid
cells remain visible as ``N/A`` and are never replaced with zero.

The report is explicitly descriptive: one training seed cannot estimate
training-seed variance and no confidence interval or significance claim is
emitted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence
import uuid

import drone_run_matrix as matrix_runner
from drone_bootstrap import ROOT, canonical_sha256, sha256_file, source_hashes
from drone_evaluate import _validated_gust_recovery_events
from drone_evaluation_protocol import SCENARIOS, load_protocol
from g1_fly_control.crazyflie.checkpoint import (
    load_history_reference,
    read_checkpoint,
)


CONTROLLERS = tuple(matrix_runner.CONTROLLERS)
TASKS = tuple(SCENARIOS)
TRAINING_SEED = 0
INTERACTIONS_PER_JOB = 1_000_000
EPISODES_PER_CELL = 16
EXPECTED_JOB_COUNT = len(CONTROLLERS) * len(TASKS)
EXPECTED_EVALUATION_EPISODES = EXPECTED_JOB_COUNT * EPISODES_PER_CELL
EPISODE_HORIZON_S = 12.0
MINIMUM_SUCCESS_DWELL_S = 0.5
TRACKING_SCALE_M = 1.0
MAX_NORMALIZED_COMMAND_EFFORT = EPISODE_HORIZON_S * 4.0
MAX_NORMALIZED_COMMAND_DELTA_L2 = 4.0

TASK_EVENT_OPPORTUNITIES = {
    TASKS[0]: 1,
    TASKS[1]: 4,
    TASKS[2]: 3,
}
TASK_LATENCY_HORIZON_S = {
    TASKS[0]: 12.0,
    TASKS[1]: 3.0,
    TASKS[2]: 2.0,
}

# These values are frozen independently of controller outputs.  No observed
# minimum/maximum, controller rank, or best-run value enters the formula.
SCORE_CONTRACT: dict[str, Any] = {
    "version": "crazyflie_one_seed_task_score_v1",
    "scope": {
        "controllers": list(CONTROLLERS),
        "tasks": list(TASKS),
        "training_seed": TRAINING_SEED,
        "interactions_per_job": INTERACTIONS_PER_JOB,
        "heldout_protocol": "main",
        "evaluation_seed": 101,
        "episodes_per_cell": EPISODES_PER_CELL,
        "job_count": EXPECTED_JOB_COUNT,
        "evaluation_episode_count": EXPECTED_EVALUATION_EPISODES,
        "matched_task_only": True,
    },
    "weights_points": {
        "safety": 25.0,
        "task_event_completion": 25.0,
        "all_required_events": 15.0,
        "fixed_horizon_censored_latency": 5.0,
        "tracking": 20.0,
        "control_quality": 10.0,
    },
    "task_event_opportunities_per_episode": TASK_EVENT_OPPORTUNITIES,
    "latency_horizon_s": TASK_LATENCY_HORIZON_S,
    "minimum_success_dwell_s": MINIMUM_SUCCESS_DWELL_S,
    "episode_horizon_s": EPISODE_HORIZON_S,
    "tracking_exponential_scale_m": TRACKING_SCALE_M,
    "command_effort_upper_bound": MAX_NORMALIZED_COMMAND_EFFORT,
    "command_delta_l2_upper_bound": MAX_NORMALIZED_COMMAND_DELTA_L2,
    "terminated_episode_control_quality": 0.0,
    "invalid_cell_score": None,
    "overall_controller_aggregation": "unweighted_arithmetic_mean_of_three_matched_task_scores",
    "ranking_tiebreakers": [
        "higher_macro_mean_score",
        "higher_worst_task_score",
        "higher_macro_task_event_completion_rate",
        "exact_remaining_ties_are_reported_as_ties",
    ],
    "excluded_from_composite_but_reported": [
        "mean_speed_inside_target_region_m_s",
        "aggregate_wrench_mechanical_work_proxy_j",
        "gust_conditional_recovery_rate",
        "gust_max_displacement_m",
        "gust_post_gust_error_integral_m_s",
        "training_return_reward_and_loss_metrics",
    ],
    "interpretation": (
        "Descriptive one-training-seed comparison only; no training-seed "
        "confidence interval or statistical significance claim."
    ),
}
SCORE_CONTRACT_SHA256 = canonical_sha256(SCORE_CONTRACT)

TRAINING_SCALAR_METRICS = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "attempted_kl",
    "grad_norm",
    "mean_rollout_reward",
    "goal_distance_mean_m",
    "speed_mean_m_s",
    "action_mean_abs",
    "action_change_mean_l2",
    "action_saturation_fraction",
    "interactions_per_second",
    "updates_per_second",
    "accepted_epochs",
    "rejected_step",
)
TRAINING_COUNT_METRICS = (
    "completed_episode_count",
    "target_success_count",
    "successful_episode_count",
    "failure_termination_count",
    "time_limit_truncation_count",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value, sha256(payload).hexdigest()


def _resolved_config_paths(queue: Mapping[str, Any]) -> set[Path]:
    """Return the complete referenced JSON config graph, including cycles."""

    first = queue.get("config_path")
    if not isinstance(first, str) or not first:
        raise ValueError("queue config_path is missing")
    pending = [Path(first).resolve()]
    result: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in result:
            continue
        value, _ = _read_json(path)
        result.add(path)
        references: list[Any] = [
            value.get("base_config"),
            value.get("parent_config"),
            value.get("reference_config"),
        ]
        task_configs = value.get("task_configs")
        if isinstance(task_configs, Mapping):
            references.extend(task_configs.values())
        for reference in references:
            if isinstance(reference, str) and reference:
                candidate = Path(reference).expanduser()
                pending.append(
                    (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
                )
    return result


def _comparison_artifact_paths(
    queue_path: Path, queue: Mapping[str, Any]
) -> set[Path]:
    """Enumerate every mutable file directly consumed by the scorer."""

    def recorded_path(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()

    paths = {queue_path.resolve(), Path(__file__).resolve()}
    config_paths = _resolved_config_paths(queue)
    paths.update(config_paths)
    provenance_receipts: set[Path] = set()
    for config_path in config_paths:
        value, _ = _read_json(config_path)
        for declaration_name in ("learning_rate_selection", "concurrency_preflight"):
            declaration = value.get(declaration_name)
            if isinstance(declaration, Mapping):
                receipt = recorded_path(declaration.get("receipt"))
                if receipt is not None:
                    provenance_receipts.add(receipt)
    decision = queue.get("concurrency_decision")
    if isinstance(decision, Mapping):
        receipt = recorded_path(decision.get("receipt"))
        if receipt is not None:
            provenance_receipts.add(receipt)
        for entry in decision.get("reports", []):
            if isinstance(entry, Mapping):
                report_path = recorded_path(entry.get("path"))
                if report_path is not None:
                    paths.add(report_path)
    paths.update(provenance_receipts)
    for receipt in provenance_receipts:
        if not receipt.is_file():
            continue
        try:
            value, _ = _read_json(receipt)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for field in ("declaration", "selector", "selected_checkpoint"):
            dependency = recorded_path(value.get(field))
            if dependency is not None:
                paths.add(dependency)
        candidates = value.get("candidates")
        reports = value.get("reports")
        entries = [
            *(candidates if isinstance(candidates, list) else []),
            *(reports if isinstance(reports, list) else []),
        ]
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            for field in ("training_manifest", "checkpoint", "path"):
                dependency = recorded_path(entry.get(field))
                if dependency is not None:
                    paths.add(dependency)
    for job in queue.get("jobs", []):
        if not isinstance(job, Mapping):
            raise ValueError("queue contains a non-object job")
        checkpoint = Path(str(job.get("checkpoint", ""))).resolve()
        manifest = (Path(str(job.get("run_dir", ""))) / "training_manifest.json").resolve()
        paths.update((checkpoint, manifest))
        for field in ("connectome_manifest_path", "rewire_manifest_path"):
            dependency = recorded_path(job.get(field))
            if dependency is not None:
                paths.add(dependency)
        try:
            manifest_value, _ = _read_json(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest_value = {}
        reference = manifest_value.get("history_reference")
        if isinstance(reference, Mapping) and isinstance(reference.get("segments"), list):
            for segment in reference["segments"]:
                if isinstance(segment, Mapping) and isinstance(segment.get("path"), str):
                    paths.add((checkpoint.parent / segment["path"]).resolve())
        for bundle in job.get("evaluations", []):
            if not isinstance(bundle, Mapping):
                raise ValueError("queue contains a non-object evaluation bundle")
            paths.add(Path(str(bundle.get("output", ""))).resolve())
    return paths


def _capture_final_artifact_snapshot(
    queue_path: Path,
    queue: Mapping[str, Any],
    queue_sha256: str,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate all 12 cells, then freeze every byte used by the report."""

    current_queue, current_queue_sha256 = _read_json(queue_path)
    if current_queue_sha256 != queue_sha256 or current_queue != queue:
        raise ValueError("queue changed while the report was being built")
    validate_design(current_queue, queue_path=queue_path)
    by_id = {cell.get("job_id"): cell for cell in cells}
    for job in current_queue["jobs"]:
        cell = by_id.get(job["id"])
        if not isinstance(cell, Mapping):
            raise ValueError(f"report cell disappeared before snapshot: {job['id']}")
        bundle = job["evaluations"][0]
        if cell.get("status") == "valid":
            if not matrix_runner._valid_training(dict(job)):
                raise ValueError(f"training artifact changed before snapshot: {job['id']}")
            if not matrix_runner._valid_evaluation(
                dict(bundle), dict(job), EPISODES_PER_CELL
            ):
                raise ValueError(f"evaluation artifact changed before snapshot: {job['id']}")
        training = cell.get("training_evidence")
        evaluation = cell.get("evaluation_evidence")
        if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError(f"report evidence is malformed for {job['id']}")
        manifest = Path(str(training.get("manifest", "")))
        checkpoint = Path(str(job["checkpoint"]))
        output = Path(str(bundle["output"]))
        observed = (
            sha256_file(manifest) if manifest.is_file() else None,
            sha256_file(checkpoint) if checkpoint.is_file() else None,
            sha256_file(output) if output.is_file() else None,
        )
        expected = (
            training.get("manifest_sha256"),
            training.get("checkpoint_sha256"),
            evaluation.get("sha256"),
        )
        if observed != expected:
            raise ValueError(f"cell bytes changed before snapshot: {job['id']}")
    paths = _comparison_artifact_paths(queue_path, current_queue)
    files = {
        str(path): sha256_file(path) if path.is_file() else None
        for path in sorted(paths, key=str)
    }
    snapshot = {
        "schema_version": 1,
        "queue_sha256": queue_sha256,
        "source_sha256": source_hashes(),
        "files_sha256": files,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def _revalidate_report_artifacts(report: Mapping[str, Any]) -> None:
    """Fail closed if anything used by the report changed before publication."""

    snapshot = report.get("artifact_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("report lacks its final artifact snapshot")
    body = dict(snapshot)
    claimed = body.pop("snapshot_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise ValueError("report artifact snapshot identity changed")
    queue_path = Path(str(report.get("queue", ""))).resolve()
    queue, queue_sha256 = _read_json(queue_path)
    if queue_sha256 != report.get("queue_sha256") or queue_sha256 != snapshot.get(
        "queue_sha256"
    ):
        raise ValueError("queue changed before report publication")
    validate_design(queue, queue_path=queue_path)
    if source_hashes() != snapshot.get("source_sha256"):
        raise ValueError("executable source set changed before report publication")
    expected_paths = _comparison_artifact_paths(queue_path, queue)
    recorded_files = snapshot.get("files_sha256")
    if not isinstance(recorded_files, Mapping) or set(recorded_files) != {
        str(path) for path in expected_paths
    }:
        raise ValueError("report artifact path set changed before publication")
    for path in expected_paths:
        current = sha256_file(path) if path.is_file() else None
        if current != recorded_files[str(path)]:
            raise ValueError(f"report artifact changed before publication: {path}")
    cell_status = {
        cell.get("job_id"): cell.get("status")
        for cell in report.get("controller_task_table", [])
        if isinstance(cell, Mapping)
    }
    for job in queue["jobs"]:
        if cell_status.get(job["id"]) == "valid":
            if not matrix_runner._valid_training(dict(job)):
                raise ValueError(f"training authentication changed before publication: {job['id']}")
            bundle = job["evaluations"][0]
            if not matrix_runner._valid_evaluation(
                dict(bundle), dict(job), EPISODES_PER_CELL
            ):
                raise ValueError(f"evaluation authentication changed before publication: {job['id']}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_text(path: Path, text: str) -> Path:
    """Durably stage complete bytes beside their final no-clobber path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _publish_report_pair(
    json_path: Path,
    json_text: str,
    markdown_path: Path,
    markdown_text: str,
) -> None:
    """Publish both immutable outputs without overwriting or leaving a half-pair.

    ``os.link`` is the publication primitive because it atomically fails when
    another writer already owns a destination.  Both complete files are staged
    first.  If the second publication fails, the first hard link created by
    this invocation is rolled back before the error is surfaced.
    """

    destinations = (json_path.resolve(), markdown_path.resolve())
    if destinations[0] == destinations[1]:
        raise ValueError("JSON and Markdown report paths must be different")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {path}")

    staged: list[Path] = []
    published: list[tuple[Path, Path]] = []
    try:
        staged = [
            _stage_text(destinations[0], json_text),
            _stage_text(destinations[1], markdown_text),
        ]
        for temporary, destination in zip(staged, destinations, strict=True):
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing report: {destination}"
                ) from exc
            published.append((temporary, destination))
            _fsync_directory(destination.parent)
    except BaseException:
        for temporary, destination in reversed(published):
            try:
                if destination.exists() and os.path.samefile(temporary, destination):
                    destination.unlink()
                    _fsync_directory(destination.parent)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        for temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_all_numbers_finite(item) for item in value)
    return False


def _manifest_contract() -> tuple[dict[str, Any], str]:
    manifest = load_protocol("main")
    return manifest, canonical_sha256(manifest)


def _queue_status_from_jobs(
    queue: Mapping[str, Any], jobs: Sequence[Any]
) -> tuple[str, dict[str, int]]:
    statuses = [job.get("status") for job in jobs if isinstance(job, Mapping)]
    if len(statuses) != len(jobs):
        raise ValueError("queue contains a non-object job")
    if any(not isinstance(status, str) for status in statuses):
        raise ValueError("queue contains a non-string job state")
    counts = {
        status: statuses.count(status)
        for status in sorted(set(statuses))
    }
    invalid = set(statuses) - matrix_runner.VALID_STATUS
    if invalid:
        raise ValueError(f"queue contains unknown job states: {sorted(invalid, key=str)}")
    if counts.get("running"):
        status = "running"
    elif counts.get("failed"):
        status = "failed"
    elif counts.get("paused"):
        status = "paused"
    elif counts.get("completed") == len(jobs):
        status = "completed"
    elif queue.get("dry_run") is True:
        status = "dry_run"
    else:
        status = "pending"
    return status, counts


def _validate_concurrency_metadata(queue: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    decision = queue.get("concurrency_decision")
    resources = queue.get("resource_limits")
    requested = queue.get("requested_max_concurrent_isaac_processes")
    effective = queue.get("max_concurrent_isaac_processes")
    if requested != 2:
        errors.append("comparison must request at most two concurrent Isaac processes")
    if effective not in {1, 2}:
        errors.append("comparison effective Isaac concurrency must be exactly 1 or 2")
    if not isinstance(resources, Mapping):
        errors.append("comparison resource_limits must be an object")
    elif resources.get("max_concurrent_isaac_processes") != effective:
        errors.append("resource limit and effective Isaac concurrency disagree")
    if not isinstance(decision, Mapping):
        return [*errors, "comparison concurrency_decision must be an object"]
    if (
        decision.get("requested_max_concurrent_isaac_processes") != requested
        or decision.get("effective_max_concurrent_isaac_processes") != effective
    ):
        errors.append("comparison concurrency decision disagrees with queue limits")
    if effective == 2:
        aggregate = decision.get("aggregate_overlap_memory_gate")
        if (
            decision.get("status") != "paired_smoke_pass"
            or decision.get("fallback_applied") is not False
            or not isinstance(decision.get("receipt_sha256"), str)
            or not isinstance(decision.get("receipt_id"), str)
            or not isinstance(decision.get("reports"), list)
            or len(decision["reports"]) != 2
            or not isinstance(aggregate, Mapping)
            or aggregate.get("passed") is not True
        ):
            errors.append(
                "concurrency=2 lacks the authenticated passing paired-smoke decision"
            )
    elif effective == 1 and (
        decision.get("status") != "fallback_sequential"
        or decision.get("fallback_applied") is not True
    ):
        errors.append("concurrency=1 must retain the recorded sequential fallback decision")
    return errors


def validate_design(queue: Mapping[str, Any], *, queue_path: Path) -> None:
    """Validate the exact 12-job/192-episode comparison before reading results."""

    errors: list[str] = []
    queue_path = queue_path.resolve()
    config = queue.get("config")
    jobs = queue.get("jobs")
    manifest, manifest_sha256 = _manifest_contract()
    if queue.get("schema_version") != 1:
        errors.append("queue.schema_version must be 1")
    if not isinstance(config, Mapping):
        errors.append("queue.config must be an object")
        config = {}
    if not isinstance(jobs, list):
        errors.append("queue.jobs must be a list")
        jobs = []
    if queue.get("job_count") != EXPECTED_JOB_COUNT or len(jobs) != EXPECTED_JOB_COUNT:
        errors.append(f"comparison must contain exactly {EXPECTED_JOB_COUNT} jobs")
    if queue.get("evaluation_bundle_count") != EXPECTED_JOB_COUNT:
        errors.append(f"comparison must contain exactly {EXPECTED_JOB_COUNT} matched evaluation bundles")
    if queue.get("predicted_evaluation_episodes") != EXPECTED_EVALUATION_EPISODES:
        errors.append(
            f"comparison must declare exactly {EXPECTED_EVALUATION_EPISODES} evaluation episodes"
        )
    if queue.get("label") != matrix_runner.COMPARISON_LABEL:
        errors.append("queue label is not the frozen seed-0 comparison label")
    errors.extend(_validate_concurrency_metadata(queue))
    try:
        recorded_output = Path(str(queue.get("output", ""))).resolve()
        if recorded_output != queue_path:
            errors.append("queue path differs from its immutable output path")
    except (OSError, RuntimeError, TypeError, ValueError):
        errors.append("queue output path is invalid")
    if isinstance(jobs, list):
        try:
            expected_status, expected_counts = _queue_status_from_jobs(queue, jobs)
            if queue.get("status") != expected_status:
                errors.append(
                    f"queue status is inconsistent: {queue.get('status')!r} != {expected_status!r}"
                )
            if queue.get("counts") != expected_counts:
                errors.append("queue counts do not recompute from job states")
        except ValueError as exc:
            errors.append(str(exc))
    if config:
        if queue.get("config_sha256") != canonical_sha256(dict(config)):
            errors.append("queue.config_sha256 does not authenticate the embedded config")
        if config.get("label") != matrix_runner.COMPARISON_LABEL:
            errors.append("config label is not the frozen seed-0 comparison label")
        if config.get("matrix_layout") != matrix_runner.TASK_SEPARATED_COMPARISON_LAYOUT:
            errors.append("config is not the task-separated seed-0 comparison layout")
        if config.get("execution_readiness") != matrix_runner.TASK_SEPARATED_COMPARISON_READINESS:
            errors.append("config execution-readiness identity changed")
        if tuple(config.get("controllers", ())) != CONTROLLERS:
            errors.append("config controllers differ from the exact four-controller order")
        if tuple(config.get("tasks", ())) != TASKS:
            errors.append("config tasks differ from the exact three-task order")
        if config.get("seeds") != [TRAINING_SEED]:
            errors.append("config must contain only training seed 0")
        if config.get("total_interactions") != INTERACTIONS_PER_JOB:
            errors.append("config must declare exactly 1,000,000 interactions per job")
        evaluation = config.get("evaluation")
        if not isinstance(evaluation, Mapping):
            errors.append("config.evaluation must be an object")
        else:
            if evaluation.get("protocol") != "main":
                errors.append("evaluation protocol must be main")
            if evaluation.get("seed") != 101:
                errors.append("evaluation seed must be 101")
            if evaluation.get("episodes_per_scenario") != EPISODES_PER_CELL:
                errors.append("evaluation must contain 16 episodes per matched cell")
            if tuple(evaluation.get("scenarios", ())) != TASKS:
                errors.append("evaluation scenarios differ from the frozen task order")
            if evaluation.get("matched_task_only") is not True:
                errors.append("evaluation must explicitly declare matched_task_only=true")

    expected_order = [
        (task, controller, TRAINING_SEED)
        for controller in CONTROLLERS
        for task in TASKS
    ]
    expected_cells = set(expected_order)
    actual_cells: set[tuple[str, str, int]] = set()
    actual_order: list[tuple[str, str, int]] = []
    seen_ids: set[str] = set()
    run_paths: list[Path] = []
    checkpoint_paths: list[Path] = []
    evaluation_paths: list[Path] = []
    for index, job in enumerate(jobs):
        prefix = f"jobs[{index}]"
        if not isinstance(job, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        controller, task, seed = job.get("controller"), job.get("task"), job.get("seed")
        if controller not in CONTROLLERS or task not in TASKS or seed != TRAINING_SEED:
            errors.append(f"{prefix} controller/task/seed is outside the frozen design")
        else:
            actual_cells.add((task, controller, seed))
            actual_order.append((task, controller, seed))
        identifier = job.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            errors.append(f"{prefix}.id is missing or duplicated")
        else:
            seen_ids.add(identifier)
        if job.get("total_interactions") != INTERACTIONS_PER_JOB:
            errors.append(f"{prefix} does not declare exactly 1,000,000 interactions")
        if job.get("contract_profile") != "balanced_v4":
            errors.append(f"{prefix} is not a balanced-v4 training cell")
        try:
            run_path = Path(str(job.get("run_dir", ""))).resolve()
            checkpoint_path = Path(str(job.get("checkpoint", ""))).resolve()
            if checkpoint_path.parent.parent != run_path:
                errors.append(f"{prefix} checkpoint is outside its run directory")
            run_paths.append(run_path)
            checkpoint_paths.append(checkpoint_path)
        except (OSError, RuntimeError, TypeError, ValueError):
            errors.append(f"{prefix} run/checkpoint path is invalid")
        fingerprint = job.get("expected_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            errors.append(f"{prefix}.expected_fingerprint is not a lowercase SHA-256")
        bundles = job.get("evaluations")
        if not isinstance(bundles, list) or len(bundles) != 1:
            errors.append(f"{prefix} must contain exactly one matched evaluation")
            continue
        bundle = bundles[0]
        if not isinstance(bundle, Mapping):
            errors.append(f"{prefix}.evaluations[0] must be an object")
            continue
        if bundle.get("scenario") != task:
            errors.append(f"{prefix} evaluation scenario is not its training task")
        if bundle.get("protocol") != "main" or bundle.get("protocol_label") != "main":
            errors.append(f"{prefix} evaluation is not the main held-out protocol")
        if bundle.get("evaluation_seed") != 101:
            errors.append(f"{prefix} evaluation seed is not 101")
        if bundle.get("evaluation_manifest_id") != manifest["manifest_id"]:
            errors.append(f"{prefix} evaluation manifest ID differs from the frozen main manifest")
        if bundle.get("evaluation_manifest_sha256") != manifest_sha256:
            errors.append(f"{prefix} evaluation manifest SHA-256 differs from the frozen main manifest")
        try:
            evaluation_paths.append(Path(str(bundle.get("output", ""))).resolve())
        except (OSError, RuntimeError, TypeError, ValueError):
            errors.append(f"{prefix} evaluation output path is invalid")
        plans = manifest["scenarios"].get(task, []) if task in TASKS else []
        if bundle.get("expected_episode_ids") != list(range(EPISODES_PER_CELL)):
            errors.append(f"{prefix} held-out episode IDs are not exactly 0..15")
        if bundle.get("expected_plan_sha256") != [plan["plan_sha256"] for plan in plans]:
            errors.append(f"{prefix} held-out plan hashes differ from the frozen main manifest")
        if bundle.get("expected_plan_object_sha256") != [canonical_sha256(plan) for plan in plans]:
            errors.append(f"{prefix} held-out plan-object hashes differ from the frozen main manifest")
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        errors.append(f"task/controller cells are incomplete or duplicated; missing={missing}, extra={extra}")
    if len(actual_cells) != len(jobs):
        errors.append("comparison contains duplicate task/controller/seed cells")
    if actual_order != expected_order:
        errors.append("comparison jobs are not in frozen controller-first task order")
    for label, paths in (
        ("run directories", run_paths),
        ("checkpoints", checkpoint_paths),
        ("evaluation outputs", evaluation_paths),
    ):
        if len(paths) != len(set(paths)):
            errors.append(f"comparison contains duplicate {label}")

    config_path = queue.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        errors.append("queue config_path is missing")
    else:
        try:
            resolved_config = matrix_runner.validate_config(Path(config_path))
            if (
                resolved_config.get("_matrix_layout")
                != matrix_runner.TASK_SEPARATED_COMPARISON_LAYOUT
            ):
                raise ValueError("resolved config is not the comparison layout")
            matrix_runner.validate_resume_queue(
                dict(queue), resolved_config, queue_path
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(
                "current config/source/fingerprint validation failed: "
                f"{type(exc).__name__}: {exc}"
            )
    if errors:
        raise ValueError("Invalid one-seed comparison design:\n- " + "\n- ".join(errors))


def _event_count(row: Mapping[str, Any], scenario: str) -> int:
    if scenario in TASKS[:2]:
        value = row.get("target_success_event_count")
        maximum = TASK_EVENT_OPPORTUNITIES[scenario]
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError("target_success_event_count is outside the task contract")
        if type(row.get("success")) is not bool or row["success"] is not (value >= maximum):
            raise ValueError("episode success disagrees with its target-success event count")
        if scenario == TASKS[1]:
            outcomes = row.get("switch_outcomes")
            if (
                not isinstance(outcomes, Sequence)
                or isinstance(outcomes, (str, bytes, bytearray))
                or len(outcomes) != 3
            ):
                raise ValueError("Switch episode must contain three switch outcomes")
            switched = sum(
                1
                for outcome in outcomes
                if isinstance(outcome, Mapping) and outcome.get("success") is True
            )
            if any(not isinstance(outcome, Mapping) or type(outcome.get("success")) is not bool for outcome in outcomes):
                raise ValueError("Switch outcome success fields must be Boolean")
            if value not in {switched, switched + 1}:
                raise ValueError("Switch event count disagrees with retained outcomes")
        return value
    outcomes = row.get("gust_outcomes")
    if (
        not isinstance(outcomes, Sequence)
        or isinstance(outcomes, (str, bytes, bytearray))
        or len(outcomes) != 3
    ):
        raise ValueError("Gust episode must contain exactly three recovery outcomes")
    plan = row.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("Gust episode lacks its immutable held-out plan")
    recovered_flags = _validated_gust_recovery_events(dict(row), dict(plan))
    recovered = sum(recovered_flags)
    target_count = row.get("target_success_event_count")
    if type(target_count) is not int or not 0 <= target_count <= 1:
        raise ValueError("Gust target-success event count is outside the task contract")
    if type(row.get("success")) is not bool or row["success"] is not (target_count >= 1):
        raise ValueError("Gust target-tube episode success disagrees with target-success count")
    if recovered and target_count != 1:
        raise ValueError("A Gust recovery requires an authenticated target-tube dwell")
    return recovered


def _latency(summary: Mapping[str, Any], scenario: str) -> float:
    if scenario == TASKS[0]:
        return _nonnegative(
            summary.get("mean_time_to_first_success_fixed_horizon_censored_s"),
            "Reach fixed-horizon latency",
        )
    if scenario == TASKS[1]:
        switch = summary.get("switch_metrics")
        if not isinstance(switch, Mapping):
            raise ValueError("Switch summary has no switch_metrics")
        return _nonnegative(
            switch.get("mean_fixed_horizon_censored_latency_s"),
            "Switch fixed-horizon latency",
        )
    gust = summary.get("gust_metrics")
    latency = gust.get("unconditional_recovery_latency") if isinstance(gust, Mapping) else None
    if not isinstance(latency, Mapping):
        raise ValueError("Gust summary has no unconditional recovery latency")
    return _nonnegative(
        latency.get("mean_fixed_horizon_censored_s"),
        "Gust fixed-horizon recovery latency",
    )


def score_evaluation(result: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    """Recompute the raw task metrics and frozen 100-point score."""

    episodes = result.get("episodes")
    summary = result.get("summary")
    if not isinstance(episodes, list) or len(episodes) != EPISODES_PER_CELL:
        raise ValueError("Evaluation must contain exactly 16 episode rows")
    if not isinstance(summary, Mapping):
        raise ValueError("Evaluation has no summary object")
    if not _all_numbers_finite(result):
        raise ValueError("Evaluation contains a nonfinite numeric value")
    if not matrix_runner._consistent_evaluation_summary(
        dict(summary), episodes, scenario=scenario, expected=EPISODES_PER_CELL
    ):
        raise ValueError("Evaluation summary does not recompute from episode rows")

    event_counts: list[int] = []
    control_quality: list[float] = []
    for index, row in enumerate(episodes):
        if not isinstance(row, Mapping) or row.get("episode_id") != index:
            raise ValueError("Episode rows must be ordered exactly 0..15")
        for field in (
            "success",
            "terminated",
            "truncated",
            "crash",
            "out_of_bounds",
            "invalid_state",
        ):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Episode {field} field must be Boolean")
        event_counts.append(_event_count(row, scenario))
        terminated = row.get("terminated")
        effort = _nonnegative(row.get("command_effort"), "command_effort")
        smoothness = _nonnegative(row.get("command_smoothness"), "command_smoothness")
        if terminated:
            control_quality.append(0.0)
        else:
            effort_quality = _clip01(1.0 - effort / MAX_NORMALIZED_COMMAND_EFFORT)
            smoothness_quality = _clip01(
                1.0 - smoothness / MAX_NORMALIZED_COMMAND_DELTA_L2
            )
            control_quality.append(0.5 * (effort_quality + smoothness_quality))

    opportunities = TASK_EVENT_OPPORTUNITIES[scenario]
    event_success_count = sum(event_counts)
    event_denominator = EPISODES_PER_CELL * opportunities
    event_completion_rate = event_success_count / event_denominator
    event_episode_count = sum(value > 0 for value in event_counts)
    all_required_event_episode_count = sum(value == opportunities for value in event_counts)
    all_required_event_rate = all_required_event_episode_count / EPISODES_PER_CELL
    evaluator_success_count = sum(row["success"] is True for row in episodes)
    evaluator_success_rate = evaluator_success_count / EPISODES_PER_CELL
    termination_count = sum(row["terminated"] is True for row in episodes)
    safety_rate = 1.0 - termination_count / EPISODES_PER_CELL

    latency_s = _latency(summary, scenario)
    latency_horizon_s = TASK_LATENCY_HORIZON_S[scenario]
    latency_quality = _clip01(
        (latency_horizon_s - latency_s)
        / (latency_horizon_s - MINIMUM_SUCCESS_DWELL_S)
    )
    final_error_m = _nonnegative(summary.get("mean_final_goal_error_m"), "mean final error")
    integrated_error_m_s = _nonnegative(
        summary.get("mean_integrated_goal_error_m_s"), "mean integrated error"
    )
    time_average_error_m = integrated_error_m_s / EPISODE_HORIZON_S
    tracking_quality = 0.5 * (
        math.exp(-time_average_error_m / TRACKING_SCALE_M)
        + math.exp(-final_error_m / TRACKING_SCALE_M)
    )
    mean_control_quality = mean(control_quality)

    quality = {
        "safety": safety_rate,
        "task_event_completion": event_completion_rate,
        "all_required_events": all_required_event_rate,
        "fixed_horizon_censored_latency": latency_quality,
        "tracking": tracking_quality,
        "control_quality": mean_control_quality,
    }
    components = {
        name: SCORE_CONTRACT["weights_points"][name] * value
        for name, value in quality.items()
    }
    total = sum(components.values())
    if not 0.0 <= total <= 100.0 + 1.0e-9:
        raise ValueError("Composite score left its declared [0, 100] range")

    crash_count = sum(row["crash"] for row in episodes)
    out_of_bounds_count = sum(row["out_of_bounds"] for row in episodes)
    invalid_state_count = sum(row["invalid_state"] for row in episodes)
    nonterminated_count = EPISODES_PER_CELL - termination_count
    raw = {
        "episode_count": EPISODES_PER_CELL,
        "task_event_success_count": event_success_count,
        "task_event_denominator": event_denominator,
        "task_event_completion_rate": event_completion_rate,
        "event_episode_count": event_episode_count,
        "event_episode_denominator": EPISODES_PER_CELL,
        "event_episode_rate": event_episode_count / EPISODES_PER_CELL,
        "all_required_event_episode_count": all_required_event_episode_count,
        "all_required_event_episode_denominator": EPISODES_PER_CELL,
        "all_required_event_episode_rate": all_required_event_rate,
        "strict_task_complete_episode_count": all_required_event_episode_count,
        "strict_task_complete_episode_denominator": EPISODES_PER_CELL,
        "strict_task_complete_episode_rate": all_required_event_rate,
        "evaluator_target_tube_success_count": evaluator_success_count,
        "evaluator_target_tube_success_denominator": EPISODES_PER_CELL,
        "evaluator_target_tube_success_rate": evaluator_success_rate,
        "fixed_horizon_censored_latency_s": latency_s,
        "latency_horizon_s": latency_horizon_s,
        "termination_count": termination_count,
        "nonterminated_episode_count": nonterminated_count,
        "survival_denominator": EPISODES_PER_CELL,
        "survival_rate": safety_rate,
        "crash_count": crash_count,
        "crash_denominator": EPISODES_PER_CELL,
        "out_of_bounds_count": out_of_bounds_count,
        "out_of_bounds_denominator": EPISODES_PER_CELL,
        "invalid_state_count": invalid_state_count,
        "invalid_state_denominator": EPISODES_PER_CELL,
        "mean_final_goal_error_m": final_error_m,
        "mean_integrated_goal_error_m_s": integrated_error_m_s,
        "time_average_goal_error_m": time_average_error_m,
        "mean_speed_inside_target_region_m_s": summary.get(
            "mean_speed_inside_target_region_m_s"
        ),
        "speed_inside_target_region_observation_count": summary.get(
            "speed_inside_target_region_observation_count"
        ),
        "mean_command_effort": summary.get("mean_command_effort"),
        "mean_command_smoothness": summary.get("mean_command_smoothness"),
        "mean_aggregate_wrench_mechanical_work_proxy_j": summary.get(
            "mean_aggregate_wrench_mechanical_work_proxy_j"
        ),
        "aggregate_wrench_mechanical_work_proxy_label": summary.get(
            "aggregate_wrench_mechanical_work_proxy_label"
        ),
        "switch_metrics": deepcopy(summary.get("switch_metrics")),
        "gust_metrics": deepcopy(summary.get("gust_metrics")),
        "score_evidence": {
            "safety": {
                "numerator": nonterminated_count,
                "denominator": EPISODES_PER_CELL,
            },
            "task_event_completion": {
                "numerator": event_success_count,
                "denominator": event_denominator,
            },
            "all_required_events": {
                "numerator": all_required_event_episode_count,
                "denominator": EPISODES_PER_CELL,
            },
            "fixed_horizon_censored_latency": {
                "mean_s": latency_s,
                "best_possible_s": MINIMUM_SUCCESS_DWELL_S,
                "horizon_s": latency_horizon_s,
            },
            "tracking": {
                "time_average_goal_error_m": time_average_error_m,
                "final_goal_error_m": final_error_m,
                "exponential_scale_m": TRACKING_SCALE_M,
            },
            "control_quality": {
                "mean_command_effort": summary.get("mean_command_effort"),
                "command_effort_upper_bound": MAX_NORMALIZED_COMMAND_EFFORT,
                "mean_command_smoothness": summary.get("mean_command_smoothness"),
                "command_delta_l2_upper_bound": MAX_NORMALIZED_COMMAND_DELTA_L2,
            },
        },
    }
    return {
        "score": total,
        "score_components_points": components,
        "component_quality_0_to_1": quality,
        "raw_metrics": raw,
        "episode_event_counts": event_counts,
    }


def _metric_statistics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: list[float] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if value is None:
            continue
        values.append(_finite(value, f"training history row {index} {field}"))
    if not values:
        return {
            "observation_count": 0,
            "first": None,
            "last": None,
            "mean": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "observation_count": len(values),
        "first": values[0],
        "last": values[-1],
        "mean": mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summarize_training_history(
    rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("training history is not a row sequence")
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("training history is empty or malformed")
    failure_causes = {str(code): 0 for code in range(1, 5)}
    count_totals: dict[str, int] = {}
    for field in TRAINING_COUNT_METRICS:
        total = 0
        for index, row in enumerate(rows):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"training history row {index} {field} must be a non-negative integer"
                )
            total += value
        count_totals[field] = total
    for index, row in enumerate(rows):
        counts = row.get("failure_cause_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(failure_causes):
            raise ValueError(
                f"training history row {index} has malformed failure-cause counts"
            )
        for code in failure_causes:
            value = counts[code]
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"training history row {index} failure cause {code} is invalid"
                )
            failure_causes[code] += value
    first, last = rows[0], rows[-1]
    reference_summary = None
    if reference is not None:
        reference_summary = {
            "history_sha256": reference.get("history_sha256"),
            "row_count": reference.get("row_count"),
            "segments": [
                {
                    "path": segment.get("path"),
                    "sha256": segment.get("sha256"),
                    "row_count": segment.get("row_count"),
                }
                for segment in reference.get("segments", [])
                if isinstance(segment, Mapping)
            ],
        }
    return {
        "row_count": len(rows),
        "first_completed_updates": first.get("completed_updates"),
        "last_completed_updates": last.get("completed_updates"),
        "first_total_interactions": first.get("total_interactions"),
        "last_total_interactions": last.get("total_interactions"),
        "scalar_metrics": {
            field: _metric_statistics(rows, field)
            for field in TRAINING_SCALAR_METRICS
        },
        "count_totals": count_totals,
        "failure_cause_count_totals": failure_causes,
        "history_reference": reference_summary,
    }


def _training_history_summary(
    checkpoint_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    payload = read_checkpoint(
        checkpoint_path, map_location="cpu", resolve_external_history=False
    )
    reference = payload.get("history_reference")
    if reference != manifest.get("history_reference"):
        raise ValueError("checkpoint and manifest history references differ")
    if reference is None:
        rows = payload.get("history")
    else:
        rows = load_history_reference(reference, checkpoint_path=checkpoint_path)
    return _summarize_training_history(rows, reference)


def _training_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    run_dir = Path(str(job.get("run_dir", "")))
    manifest_path = run_dir / "training_manifest.json"
    evidence: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": None,
        "checkpoint": job.get("checkpoint"),
        "checkpoint_sha256": None,
        "training_metrics_summary": None,
    }
    if not manifest_path.is_file():
        return evidence, [f"missing training manifest: {manifest_path}"]
    try:
        manifest, digest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return evidence, [f"cannot read training manifest: {exc}"]
    evidence["manifest_sha256"] = digest
    checkpoint_path = Path(str(job.get("checkpoint", "")))
    if checkpoint_path.is_file():
        evidence["checkpoint_sha256"] = sha256_file(checkpoint_path)
    else:
        issues.append(f"missing checkpoint: {checkpoint_path}")
    try:
        runner_valid = matrix_runner._valid_training(dict(job))
    except Exception as exc:  # fail closed while retaining a useful table row
        runner_valid = False
        issues.append(f"queue-runner training validator raised {type(exc).__name__}: {exc}")
    if not runner_valid:
        issues.append("queue-runner training authentication rejected the artifact")
    elif checkpoint_path.is_file():
        try:
            evidence["training_metrics_summary"] = _training_history_summary(
                checkpoint_path, manifest
            )
        except Exception as exc:
            issues.append(
                "authenticated training history could not be summarized: "
                f"{type(exc).__name__}: {exc}"
            )
    expected = {
        "status": "completed",
        "task": job.get("task"),
        "controller": job.get("controller"),
        "seed": TRAINING_SEED,
        "requested_interactions": INTERACTIONS_PER_JOB,
        "environment_interactions": INTERACTIONS_PER_JOB,
        "fingerprint": job.get("expected_fingerprint"),
        "evaluation_manifest_id": job.get("evaluation_manifest_id"),
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            issues.append(
                f"training manifest {field} differs: {manifest.get(field)!r} != {expected_value!r}"
            )
    if manifest.get("warm_start") is not None:
        issues.append("training run is warm-started; every task/controller cell must start fresh")
    if not isinstance(manifest.get("memory_gate"), Mapping) or manifest["memory_gate"].get("passed") is not True:
        issues.append("training hard memory gate did not pass")
    if manifest.get("core_checksum_before") != manifest.get("core_checksum_after"):
        issues.append("frozen-core checksum changed during training")
    if checkpoint_path.is_file() and manifest.get("checkpoint_sha256") != evidence["checkpoint_sha256"]:
        issues.append("training manifest checkpoint SHA-256 differs from the checkpoint bytes")
    report = manifest.get("controller_report")
    if not isinstance(report, Mapping):
        issues.append("training manifest lacks controller_report")
        report = {}
    evidence.update(
        {
            "status": manifest.get("status"),
            "task": manifest.get("task"),
            "controller": manifest.get("controller"),
            "seed": manifest.get("seed"),
            "environment_interactions": manifest.get("environment_interactions"),
            "completed_updates": manifest.get("completed_updates"),
            "completed_episodes": manifest.get("completed_episodes"),
            "training_wall_time_s": manifest.get("training_wall_time_s"),
            "resume_count": manifest.get("resume_count"),
            "warm_start": deepcopy(manifest.get("warm_start")),
            "fingerprint": manifest.get("fingerprint"),
            "memory_gate": deepcopy(manifest.get("memory_gate")),
            "controller_report": deepcopy(report),
            "core_checksum_before": manifest.get("core_checksum_before"),
            "core_checksum_after": manifest.get("core_checksum_after"),
            "history_reference": deepcopy(manifest.get("history_reference")),
            "ppo": deepcopy(manifest.get("ppo")),
        }
    )
    try:
        if sha256_file(manifest_path) != digest:
            issues.append("training manifest changed while the report was being built")
        if (
            checkpoint_path.is_file()
            and evidence["checkpoint_sha256"] is not None
            and sha256_file(checkpoint_path) != evidence["checkpoint_sha256"]
        ):
            issues.append("training checkpoint changed while the report was being built")
    except OSError as exc:
        issues.append(f"training artifact disappeared during report construction: {exc}")
    return evidence, issues


def _evaluation_evidence(
    job: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    output_path = Path(str(bundle.get("output", "")))
    checkpoint = Path(str(job.get("checkpoint", "")))
    checkpoint_sha256_before = (
        sha256_file(checkpoint) if checkpoint.is_file() else None
    )
    evidence: dict[str, Any] = {
        "path": str(output_path),
        "sha256": None,
        "checkpoint_current_sha256": checkpoint_sha256_before,
    }
    if not output_path.is_file():
        return evidence, None, [f"missing evaluation artifact: {output_path}"]
    try:
        result, digest = _read_json(output_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return evidence, None, [f"cannot read evaluation artifact: {exc}"]
    evidence["sha256"] = digest
    try:
        runner_valid = matrix_runner._valid_evaluation(
            dict(bundle), dict(job), EPISODES_PER_CELL
        )
    except Exception as exc:
        runner_valid = False
        issues.append(f"queue-runner evaluation validator raised {type(exc).__name__}: {exc}")
    if not runner_valid:
        issues.append("queue-runner evaluation authentication rejected the artifact")
    expected = {
        "status": "completed",
        "label": "main",
        "protocol": "main",
        "scenario": job.get("task"),
        "training_seed": TRAINING_SEED,
        "controller": job.get("controller"),
        "fingerprint": job.get("expected_fingerprint"),
        "evaluation_seed": 101,
        "evaluation_manifest_id": bundle.get("evaluation_manifest_id"),
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            issues.append(
                f"evaluation {field} differs: {result.get(field)!r} != {expected_value!r}"
            )
    if checkpoint.is_file() and result.get("checkpoint_sha256") != sha256_file(checkpoint):
        issues.append("evaluation checkpoint SHA-256 differs from the trained checkpoint")
    try:
        if Path(str(result.get("checkpoint", ""))).resolve() != checkpoint.resolve():
            issues.append("evaluation checkpoint path differs from the queue checkpoint")
    except (OSError, RuntimeError):
        issues.append("evaluation checkpoint path is invalid")
    if not isinstance(result.get("memory_gate"), Mapping) or result["memory_gate"].get("passed") is not True:
        issues.append("evaluation hard memory gate did not pass")
    if not _all_numbers_finite(result):
        issues.append("evaluation contains a nonfinite numeric value")
    try:
        scored = score_evaluation(result, str(job.get("task")))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        issues.append(f"score recomputation rejected evaluation: {exc}")
        scored = None
    evidence.update(
        {
            "status": result.get("status"),
            "scenario": result.get("scenario"),
            "training_seed": result.get("training_seed"),
            "evaluation_seed": result.get("evaluation_seed"),
            "fingerprint": result.get("fingerprint"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "memory_gate": deepcopy(result.get("memory_gate")),
            "summary": deepcopy(result.get("summary")),
            "episodes": deepcopy(result.get("episodes")),
        }
    )
    try:
        if sha256_file(output_path) != digest:
            issues.append("evaluation artifact changed while the report was being built")
        if (
            checkpoint_sha256_before is not None
            and checkpoint.is_file()
            and sha256_file(checkpoint) != checkpoint_sha256_before
        ):
            issues.append("evaluation checkpoint changed while the report was being built")
    except OSError as exc:
        issues.append(f"evaluation artifact disappeared during report construction: {exc}")
    return evidence, scored, issues


def _rank_overall(rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("macro_mean_score") is not None]
    valid.sort(
        key=lambda row: (
            -row["macro_mean_score"],
            -row["worst_task_score"],
            -row["macro_task_event_completion_rate"],
        )
    )
    previous: tuple[float, float, float] | None = None
    rank = 0
    for position, row in enumerate(valid, start=1):
        key = (
            row["macro_mean_score"],
            row["worst_task_score"],
            row["macro_task_event_completion_rate"],
        )
        if key != previous:
            rank = position
            previous = key
        row["rank"] = rank


def build_report(queue_path: Path) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    queue, queue_sha256 = _read_json(queue_path)
    validate_design(queue, queue_path=queue_path)

    cells: list[dict[str, Any]] = []
    for job in queue["jobs"]:
        bundle = job["evaluations"][0]
        training, training_issues = _training_evidence(job)
        evaluation, scored, evaluation_issues = _evaluation_evidence(job, bundle)
        issues = training_issues + evaluation_issues
        queue_status_issues = []
        if job.get("status") != "completed":
            queue_status_issues.append(f"queue job status is {job.get('status')!r}, not 'completed'")
        if bundle.get("status") != "completed":
            queue_status_issues.append(
                f"queue evaluation status is {bundle.get('status')!r}, not 'completed'"
            )
        issues = queue_status_issues + issues
        valid = not issues and scored is not None
        cells.append(
            {
                "job_id": job["id"],
                "controller": job["controller"],
                "task": job["task"],
                "training_seed": job["seed"],
                "status": "valid" if valid else "invalid_or_incomplete",
                "score": scored["score"] if valid else None,
                "score_components_points": (
                    scored["score_components_points"] if valid else None
                ),
                "component_quality_0_to_1": (
                    scored["component_quality_0_to_1"] if valid else None
                ),
                "raw_metrics": scored["raw_metrics"] if valid else None,
                "episode_event_counts": scored["episode_event_counts"] if valid else None,
                "issues": issues,
                "training_evidence": training,
                "evaluation_evidence": evaluation,
            }
        )

    overall: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        controller_cells = [cell for cell in cells if cell["controller"] == controller]
        by_task = {cell["task"]: cell for cell in controller_cells}
        valid = len(controller_cells) == len(TASKS) and all(
            task in by_task and by_task[task]["score"] is not None for task in TASKS
        )
        if valid:
            scores = [float(by_task[task]["score"]) for task in TASKS]
            event_rates = [
                float(by_task[task]["raw_metrics"]["task_event_completion_rate"])
                for task in TASKS
            ]
            terminations = sum(
                int(by_task[task]["raw_metrics"]["termination_count"]) for task in TASKS
            )
            overall.append(
                {
                    "controller": controller,
                    "status": "valid",
                    "rank": None,
                    "macro_mean_score": mean(scores),
                    "worst_task_score": min(scores),
                    "macro_task_event_completion_rate": mean(event_rates),
                    "total_terminations_out_of_48": terminations,
                    "task_scores": {task: by_task[task]["score"] for task in TASKS},
                }
            )
        else:
            overall.append(
                {
                    "controller": controller,
                    "status": "invalid_or_incomplete",
                    "rank": None,
                    "macro_mean_score": None,
                    "worst_task_score": None,
                    "macro_task_event_completion_rate": None,
                    "total_terminations_out_of_48": None,
                    "task_scores": {
                        task: by_task.get(task, {}).get("score") for task in TASKS
                    },
                }
            )
    _rank_overall(overall)

    artifact_snapshot = _capture_final_artifact_snapshot(
        queue_path, queue, queue_sha256, cells
    )

    valid_cells = sum(cell["score"] is not None for cell in cells)
    validated_episodes = valid_cells * EPISODES_PER_CELL
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "status": "complete" if valid_cells == EXPECTED_JOB_COUNT else "incomplete_or_invalid",
        "scientific_status": "descriptive_one_training_seed_no_confidence_interval",
        "interpretation": (
            "This table compares seed 0 on fixed paired held-out plans. It is useful as a "
            "bounded engineering comparison, but it is not a multi-seed scientific result "
            "and does not estimate training-seed uncertainty."
        ),
        "queue": str(queue_path),
        "queue_sha256": queue_sha256,
        "queue_status": queue.get("status"),
        "execution_resources": {
            "requested_max_concurrent_isaac_processes": queue.get(
                "requested_max_concurrent_isaac_processes"
            ),
            "max_concurrent_isaac_processes": queue.get(
                "max_concurrent_isaac_processes"
            ),
            "resource_limits": deepcopy(queue.get("resource_limits")),
            "concurrency_decision": deepcopy(queue.get("concurrency_decision")),
        },
        "score_contract": deepcopy(SCORE_CONTRACT),
        "score_contract_sha256": SCORE_CONTRACT_SHA256,
        "scorer": str(Path(__file__).resolve()),
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "artifact_snapshot": artifact_snapshot,
        "design": {
            "controllers": list(CONTROLLERS),
            "tasks": list(TASKS),
            "training_seed": TRAINING_SEED,
            "jobs": EXPECTED_JOB_COUNT,
            "interactions_per_job": INTERACTIONS_PER_JOB,
            "total_training_interactions": EXPECTED_JOB_COUNT * INTERACTIONS_PER_JOB,
            "episodes_per_matched_task_cell": EPISODES_PER_CELL,
            "total_evaluation_episodes": EXPECTED_EVALUATION_EPISODES,
            "evaluation_seed": 101,
            "protocol": "main",
        },
        "completion": {
            "valid_cells": valid_cells,
            "planned_cells": EXPECTED_JOB_COUNT,
            "validated_evaluation_episodes": validated_episodes,
            "planned_evaluation_episodes": EXPECTED_EVALUATION_EPISODES,
        },
        "overall_controller_table": overall,
        "controller_task_table": cells,
        "invalid_cells": [
            {"job_id": cell["job_id"], "issues": cell["issues"]}
            for cell in cells
            if cell["score"] is None
        ],
        "confidence_intervals": None,
        "confidence_interval_reason": (
            "Only one independent training seed is present; held-out episodes are paired "
            "test cases, not replacements for independent training-seed replicates."
        ),
        "report_sha256_contract": "canonical_sha256_of_report_without_report_sha256_field",
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _short_task(task: str) -> str:
    return task.removeprefix("FlyCrazyflie-").removesuffix("-v0")


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _component_summary(cell: Mapping[str, Any]) -> str:
    components = cell.get("score_components_points")
    if not isinstance(components, Mapping):
        return "N/A"
    order = (
        "safety",
        "task_event_completion",
        "all_required_events",
        "fixed_horizon_censored_latency",
        "tracking",
        "control_quality",
    )
    return "/".join(_fmt(components.get(name), 2) for name in order)


def _memory_summary(evidence: Mapping[str, Any]) -> str:
    gate = evidence.get("memory_gate")
    if not isinstance(gate, Mapping):
        return "N/A"
    return (
        f"{_fmt(gate.get('max_device_gpu_used_mib'), 1)} MiB/"
        f"{_fmt(gate.get('max_system_ram_percent'), 1)}%"
    )


def markdown(report: Mapping[str, Any]) -> str:
    completion = report["completion"]
    execution = report.get("execution_resources") or {}
    lines = [
        "# Crazyflie one-seed, task-separated 1M comparison",
        "",
        "> **DESCRIPTIVE ONE-SEED RESULT:** seed 0 only. No training-seed confidence interval or statistical-significance claim is valid.",
        "",
        f"Status: **{report['status']}** — {completion['valid_cells']}/{completion['planned_cells']} validated cells and "
        f"{completion['validated_evaluation_episodes']}/{completion['planned_evaluation_episodes']} authenticated held-out episodes.",
        "",
        f"Score contract: `{report['score_contract_sha256']}`. Report self-hash: `{report['report_sha256']}`.",
        "",
        "Execution concurrency: "
        f"requested `{execution.get('requested_max_concurrent_isaac_processes', 'N/A')}`, "
        f"effective `{execution.get('max_concurrent_isaac_processes', 'N/A')}`; "
        f"decision `{_escape((execution.get('concurrency_decision') or {}).get('status', 'N/A'))}`.",
        "",
        "## Overall controller score",
        "",
        "| Rank | Controller | Macro score /100 | Worst task | Event completion | Terminations /48 | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["overall_controller_table"]:
        lines.append(
            f"| {row['rank'] if row['rank'] is not None else 'N/A'} | `{_escape(row['controller'])}` | "
            f"{_fmt(row['macro_mean_score'])} | {_fmt(row['worst_task_score'])} | "
            f"{_fmt(row['macro_task_event_completion_rate'])} | "
            f"{row['total_terminations_out_of_48'] if row['total_terminations_out_of_48'] is not None else 'N/A'} | "
            f"{row['status']} |"
        )
    lines += [
        "",
        "## Controller × task scorecard",
        "",
        "| Controller | Task | Score /100 | Components S/E/A/L/T/C | Events | Any-event episodes | Strict-complete episodes | Censored latency (s) | Survival | Final error (m) | Avg error (m) | Crash/OOB/invalid | Status |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cell in report["controller_task_table"]:
        raw = cell.get("raw_metrics") or {}
        failures = "; ".join(
            (
                f"{raw.get('crash_count', 'N/A')}/{raw.get('crash_denominator', 'N/A')}",
                f"{raw.get('out_of_bounds_count', 'N/A')}/"
                f"{raw.get('out_of_bounds_denominator', 'N/A')}",
                f"{raw.get('invalid_state_count', 'N/A')}/"
                f"{raw.get('invalid_state_denominator', 'N/A')}",
            )
        )
        events = (
            f"{raw.get('task_event_success_count', 'N/A')}/"
            f"{raw.get('task_event_denominator', 'N/A')}"
        )
        any_events = (
            f"{raw.get('event_episode_count', 'N/A')}/"
            f"{raw.get('event_episode_denominator', 'N/A')}"
        )
        all_events = (
            f"{raw.get('all_required_event_episode_count', 'N/A')}/"
            f"{raw.get('all_required_event_episode_denominator', 'N/A')}"
        )
        lines.append(
            f"| `{_escape(cell['controller'])}` | {_short_task(cell['task'])} | {_fmt(cell['score'])} | "
            f"{_component_summary(cell)} | {events} | {any_events} | {all_events} | "
            f"{_fmt(raw.get('fixed_horizon_censored_latency_s'))} | "
            f"{raw.get('nonterminated_episode_count', 'N/A')}/"
            f"{raw.get('survival_denominator', 'N/A')} "
            f"({_fmt(raw.get('survival_rate'))}) | "
            f"{_fmt(raw.get('mean_final_goal_error_m'))} | "
            f"{_fmt(raw.get('time_average_goal_error_m'))} | {failures} | {cell['status']} |"
        )
    lines += [
        "",
        "Components are points in fixed order: safety/event completion/strict task completion/censored latency/tracking/control quality.",
        "",
        "## Per-cell control, resource, and identity evidence",
        "",
        "| Controller | Task | Score | Effort | Smoothness | Work proxy (J) | Train GPU/RAM | Eval GPU/RAM | Actor/trainable/frozen/state | Training events; last loss | Checkpoint SHA-256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for cell in report["controller_task_table"]:
        raw = cell.get("raw_metrics") or {}
        training = cell.get("training_evidence") or {}
        evaluation = cell.get("evaluation_evidence") or {}
        controller = training.get("controller_report") or {}
        history = training.get("training_metrics_summary") or {}
        count_totals = history.get("count_totals") or {}
        scalar_metrics = history.get("scalar_metrics") or {}
        loss = scalar_metrics.get("loss") or {}
        parameters = (
            f"{controller.get('actor_trainable_parameters', 'N/A')}/"
            f"{controller.get('total_trainable_parameters', 'N/A')}/"
            f"{controller.get('frozen_parameters', 'N/A')}/"
            f"{controller.get('total_dynamic_state_per_environment', 'N/A')}"
        )
        training_metrics = (
            f"{count_totals.get('target_success_count', 'N/A')}; "
            f"{_fmt(loss.get('last'), 6)}"
        )
        checkpoint_sha = training.get("checkpoint_sha256") or "N/A"
        lines.append(
            f"| `{_escape(cell['controller'])}` | {_short_task(cell['task'])} | "
            f"{_fmt(cell.get('score'))} | {_fmt(raw.get('mean_command_effort'), 6)} | "
            f"{_fmt(raw.get('mean_command_smoothness'), 6)} | "
            f"{_fmt(raw.get('mean_aggregate_wrench_mechanical_work_proxy_j'), 6)} | "
            f"{_memory_summary(training)} | {_memory_summary(evaluation)} | {parameters} | "
            f"{training_metrics} | `{_escape(checkpoint_sha)}` |"
        )
    lines += [
        "",
        "## Score definition",
        "",
        "The fixed score is `25×safety + 25×event completion + 15×all-required-events + 5×censored-latency quality + 20×tracking quality + 10×control quality`. All qualities are bounded to [0,1], use only fixed physical/protocol constants, and never use cross-controller observed minima or maxima.",
        "",
        "Mechanical-work proxy, speed inside the target region, per-switch outcomes, unconditional/conditional gust recovery, displacement, post-gust error, training metrics, parameter counts, memory gates, fingerprints, hashes, and all episode rows remain in the JSON report but do not receive post-hoc score weights.",
        "",
    ]
    invalid = report.get("invalid_cells", [])
    if invalid:
        lines += ["## Invalid or incomplete cells", ""]
        for cell in invalid:
            lines.append(
                f"- `{_escape(cell['job_id'])}`: "
                + "; ".join(_escape(issue) for issue in cell["issues"])
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        str(report["interpretation"]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if not args.queue.is_file():
        parser.error(f"Queue does not exist: {args.queue}")
    if args.json.resolve() == args.markdown.resolve():
        parser.error("--json and --markdown must be different paths")
    try:
        report = build_report(args.queue)
        json_text = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        markdown_text = markdown(report)
        # Keep this immediately adjacent to publication: every queue/config/
        # source/checkpoint/evaluation/manifest/history byte used across all
        # 12 cells is checked a final time after report rendering.
        _revalidate_report_artifacts(report)
        _publish_report_pair(
            args.json,
            json_text,
            args.markdown,
            markdown_text,
        )
    except (FileExistsError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid_cells": report["completion"]["valid_cells"],
                "planned_cells": EXPECTED_JOB_COUNT,
                "validated_evaluation_episodes": report["completion"][
                    "validated_evaluation_episodes"
                ],
                "planned_evaluation_episodes": EXPECTED_EVALUATION_EPISODES,
                "json": str(args.json.resolve()),
                "markdown": str(args.markdown.resolve()),
                "score_contract_sha256": SCORE_CONTRACT_SHA256,
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
