#!/usr/bin/env python3
"""Build the fail-closed report for the fresh 60-cell all-fair v4 matrix.

Only revision-4 cells are admissible evidence: ten controllers, still/wind,
and independent training seeds 0, 1, and 2.  Revision-2 and revision-3 files
are preserved historical identities but their results are never merged into
this report.  Publication is refused unless all 60 training and held-out
evaluation artifacts pass provenance, budget, resource, activity, latency,
and paired-schedule checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Callable, Mapping, Sequence

import crazyflie_command_all_fair_queue_v4 as fair_queue
import crazyflie_command_combinations_report as v3_helpers
import crazyflie_command_report as common
from g1_fly_control.tasks.crazyflie.command_wide_logic import (
    HELD_OUT_WIND_SEED,
    TRAINING_WIND_SEED,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json"
)
DEFAULT_REPORT = ROOT / "docs/crazyflie_command_all_fair_report_v4.md"
CONTROLLERS = tuple(fair_queue.CONTROLLERS)
TASKS = tuple(fair_queue.TASKS)
SEEDS = tuple(fair_queue.SEEDS)
LIF_CONTROLLERS = frozenset(fair_queue.LIF_CONTROLLERS)
BASELINE_CONTROLLERS = frozenset(fair_queue.BASELINE_CONTROLLERS)
POLICY_BY_CONTROLLER = dict(fair_queue.POLICY_BY_CONTROLLER)
COMPONENTS = {
    key: tuple(value) for key, value in fair_queue.COMPONENTS_BY_CONTROLLER.items()
}
INTERACTIONS = int(fair_queue.TOTAL_INTERACTIONS)
EPISODES = int(fair_queue.EPISODES_PER_JOB)
STEPS = int(fair_queue.STEPS_PER_EPISODE)
JOB_COUNT = int(fair_queue.JOB_COUNT)
GPU_LIMIT_MIB = float(fair_queue.GPU_LIMIT_MIB)
RAM_LIMIT_PERCENT = float(fair_queue.RAM_LIMIT_PERCENT)
ANALYSIS_KIND = "crazyflie_command_follow_heldout_v2"
EXPECTED_ROLES = v3_helpers.EXPECTED_ROLES
SEGMENT_NAMES = (
    "initial_hover",
    "forward",
    "brake_after_forward",
    "lateral",
    "horizontal_diagonal",
    "vertical",
    "yaw",
    "full_simultaneous",
    "reverse_full_simultaneous",
    "brake_after_full",
    "reverse_horizontal",
    "final_hover",
)
SCORE_COMPONENTS = (
    "linear_tracking",
    "yaw_tracking",
    "direction",
    "response",
    "braking",
    "hover",
    "safety",
    "effort",
    "smoothness",
)
QUALITY_COMPONENTS = (
    "acceleration_quality",
    "command_response",
    "flight_stability",
    "survival_not_die",
)
PLOT_NAMES = {
    "score": "command_v4_score_by_condition.png",
    "wind_delta": "command_v4_wind_delta.png",
    "reward": "command_v4_training_reward.png",
    "loss": "command_v4_training_loss.png",
    "activity": "command_v4_activity_groups.png",
    "latency": "command_v4_inference_latency.png",
    "efficiency": "command_v4_efficiency.png",
    "memory": "command_v4_peak_memory.png",
}


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the revision-4 output root: {path}")


def _record_hash(
    snapshots: dict[Path, str], path: Path, expected: str | None = None
) -> str:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"authenticated input is missing: {path}")
    actual = _sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"authenticated input SHA-256 changed: {path}")
    prior = snapshots.get(path)
    if prior is not None and prior != actual:
        raise ValueError(f"conflicting SHA-256 identities for {path}")
    snapshots[path] = actual
    return actual


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _condition(task: str) -> str:
    if task == TASKS[0]:
        return "still"
    if task == TASKS[1]:
        return "wind"
    raise ValueError(f"unsupported revision-4 task: {task}")


def _capacity_tier(controller: str) -> str:
    if controller in BASELINE_CONTROLLERS:
        return "single-core-capacity engineering baseline"
    count = len(COMPONENTS[controller])
    if count == 1:
        return "one-core LIF (4,776 actor parameters)"
    if count == 2:
        return "two-core LIF (9,224 actor parameters)"
    return "three-core LIF (13,672 actor parameters; isolated tier)"


def _expected_pairs() -> list[tuple[str, int, str]]:
    return [
        (controller, seed, task)
        for controller in CONTROLLERS
        for seed in SEEDS
        for task in TASKS
    ]


@dataclass
class AllFairCell(common.V2Cell):
    seed: int = 0
    components: tuple[str, ...] = ()
    capacity_tier: str = ""
    training_wall_time_s: float | None = None
    train_memory: dict[str, Any] = field(default_factory=dict)
    eval_memory: dict[str, Any] = field(default_factory=dict)
    inference_latency: dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 0
    prior_failure_count: int = 0
    command_schedule_identity_sha256: str = ""
    wind_schedule_identity_sha256: str = ""


@dataclass
class AllFairReportData:
    config_path: Path
    queue_path: Path
    output_root: Path
    config_sha256: str
    queue_sha256: str
    cells: list[AllFairCell]
    input_hashes: dict[Path, str]
    generated_at_utc: str

    @property
    def complete(self) -> bool:
        return (
            [(cell.controller, cell.seed, cell.task) for cell in self.cells]
            == _expected_pairs()
            and len(self.cells) == JOB_COUNT == 60
            and all(cell.complete for cell in self.cells)
        )


def _validate_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        # A caller may use an exact byte copy in a test or archival location,
        # but its public identity still has to pass the queue's closed schema.
        if not config_path.is_file():
            raise ValueError(f"revision-4 config is missing: {config_path}")
    validated = fair_queue.validate_config(config_path)
    public = _public_config(validated)
    if (
        public.get("controllers") != list(CONTROLLERS)
        or public.get("seeds") != list(SEEDS)
        or public.get("tasks") != list(TASKS)
        or public.get("total_interactions_per_job") != INTERACTIONS
        or public.get("historical_v2_identity", {}).get("fair_evidence_eligible")
        is not False
        or public.get("preserved_v3_identity", {}).get("relation")
        != "preserved_not_adopted_or_mutated"
    ):
        raise ValueError("revision-4 config does not isolate the fresh all-fair matrix")
    output_root = Path(str(validated["_output_root"])).resolve()
    if output_root.name != "crazyflie_command_all_fair_seeds0_1_2_1m":
        raise ValueError("reporter accepts only the dedicated revision-4 output root")
    return validated, output_root


def _validate_controller_report(
    controller: str, report: Mapping[str, Any]
) -> None:
    expected_actor = fair_queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
    expected_critic = fair_queue.CRITIC_PARAMETERS_BY_CONTROLLER[controller]
    expected_dynamic = fair_queue.DYNAMIC_STATE_BY_CONTROLLER[controller]
    expected_frozen = fair_queue.FROZEN_PARAMETERS_BY_CONTROLLER[controller]
    exact = {
        "controller_kind": fair_queue.REPORT_KIND_BY_CONTROLLER[controller],
        "actor_trainable_parameters": expected_actor,
        "critic_trainable_parameters": expected_critic,
        "total_trainable_parameters": expected_actor + expected_critic,
        "total_dynamic_state_per_environment": expected_dynamic,
        "frozen_parameters": expected_frozen,
        "frozen_synaptic_weights": expected_frozen,
        "parameter_matching_required": (
            fair_queue.PARAMETER_MATCH_REQUIRED_BY_CONTROLLER[controller]
        ),
        "fusion_contract": fair_queue.FUSION_BY_CONTROLLER[controller],
    }
    for name, expected in exact.items():
        if report.get(name) != expected:
            raise ValueError(
                f"{controller} controller-report {name} differs from {expected!r}"
            )
    actor_match = report.get("actor_parameter_match_passed")
    expected_match = abs(expected_actor - 4_776) <= 0.05 * 4_776
    if actor_match is not expected_match:
        raise ValueError(f"{controller} actor parameter-match result drifted")
    labels = report.get("core_labels")
    per_core = report.get("per_core_checksums")
    if controller in LIF_CONTROLLERS:
        expected_labels = (
            list(COMPONENTS[controller])
            if len(COMPONENTS[controller]) > 1
            else ["primary"]
        )
        if labels != expected_labels:
            raise ValueError(f"{controller} frozen-core order differs")
        per_core = _mapping(per_core, f"{controller}.per_core_checksums")
        if set(per_core) != set(expected_labels) or any(
            not isinstance(value, str) or len(value) != 64
            for value in per_core.values()
        ):
            raise ValueError(f"{controller} per-core checksum set differs")
        aggregate = report.get("core_checksum")
        if not isinstance(aggregate, str) or len(aggregate) != 64:
            raise ValueError(f"{controller} lacks an aggregate frozen-core checksum")
        if len(expected_labels) == 1:
            expected_aggregate = per_core["primary"]
        elif controller == "leg_wing_lif":
            # This established two-core class predates the generic
            # multi-connectome policy and therefore has its own canonical
            # checksum envelope in controller_core_checksum().
            expected_aggregate = _canonical_sha256(
                {
                    "composition": fair_queue.FUSION_BY_CONTROLLER[controller],
                    "leg": per_core["leg"],
                    "wing": per_core["wing"],
                }
            )
        else:
            expected_aggregate = _canonical_sha256(
                {
                    "composition": fair_queue.FUSION_BY_CONTROLLER[controller],
                    "cores": dict(per_core),
                }
            )
        if aggregate != expected_aggregate:
            raise ValueError(
                f"{controller} aggregate checksum does not bind its per-core checksums"
            )
        for name in (
            "connectome_manifests",
            "connectome_checksums",
            "connectome_manifest_fingerprints",
            "per_core_population_indices",
        ):
            value = _mapping(report.get(name), f"{controller}.{name}")
            if set(value) != set(expected_labels):
                raise ValueError(f"{controller} {name} does not cover its cores")
    else:
        if labels != [] or per_core != {} or report.get("core_checksum") is not None:
            raise ValueError(f"{controller} engineering baseline claims a LIF core")


def _validate_source_snapshot(
    payload: Mapping[str, Any], snapshots: dict[Path, str], label: str
) -> Mapping[str, Any]:
    source = _mapping(payload.get("source_sha256"), f"{label}.source_sha256")
    if not source:
        raise ValueError(f"{label} source snapshot is empty")
    for raw_path, expected in source.items():
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError(f"{label} source snapshot contains an invalid entry")
        _record_hash(snapshots, _resolve(raw_path), expected)
    return source


def _validate_attempts(
    job: Mapping[str, Any], output_root: Path, snapshots: dict[Path, str]
) -> tuple[int, int]:
    attempts = job.get("attempts")
    failures = job.get("failure_history")
    if not isinstance(attempts, list) or len(attempts) < 2:
        raise ValueError(f"{job.get('id')} lacks immutable training/evaluation attempts")
    if not isinstance(failures, list):
        raise ValueError(f"{job.get('id')} failure_history must be a list")
    successful_phases: set[str] = set()
    for index, raw in enumerate(attempts, start=1):
        attempt = _mapping(raw, f"{job.get('id')} attempt {index}")
        phase = attempt.get("phase")
        if attempt.get("attempt") != index or phase not in {"training", "evaluation"}:
            raise ValueError(f"{job.get('id')} attempt ordering/phase differs")
        expected_command = list(job[f"{phase}_command"])
        command = attempt.get("command")
        if phase == "training" and command == [*expected_command, "--resume"]:
            pass
        elif command != expected_command:
            raise ValueError(f"{job.get('id')} attempt command differs from queue")
        for timestamp in ("started_utc", "finished_utc"):
            if not isinstance(attempt.get(timestamp), str) or not attempt[timestamp]:
                raise ValueError(f"{job.get('id')} attempt lacks {timestamp}")
        log = _resolve(str(attempt.get("log", "")))
        _require_under(log, output_root / "logs", f"{job.get('id')} attempt log")
        _record_hash(snapshots, log)
        exit_code = attempt.get("exit_code")
        if exit_code == 0 and "launch_error" not in attempt:
            successful_phases.add(str(phase))
        elif exit_code is None and (
            not isinstance(attempt.get("launch_error"), str)
            or not attempt["launch_error"]
        ):
            raise ValueError(
                f"{job.get('id')} launch failure lacks a nonempty launch_error"
            )
        elif exit_code is not None and type(exit_code) is not int:
            raise ValueError(f"{job.get('id')} attempt exit code is invalid")
        archived = attempt.get("archived_invalid_output")
        if archived is not None:
            archived_path = _resolve(str(archived))
            _require_under(
                archived_path, output_root / "evaluations",
                f"{job.get('id')} archived evaluation",
            )
            _record_hash(snapshots, archived_path)
    if successful_phases != {"training", "evaluation"}:
        raise ValueError(f"{job.get('id')} lacks successful training/evaluation attempts")
    for index, raw in enumerate(failures, start=1):
        failure = _mapping(raw, f"{job.get('id')} failure {index}")
        if (
            failure.get("index") != index
            or failure.get("phase") not in {"training", "evaluation"}
            or not isinstance(failure.get("utc"), str)
            or not isinstance(failure.get("reason"), str)
            or not failure["reason"]
            or (
                failure.get("exit_code") is not None
                and type(failure.get("exit_code")) is not int
            )
        ):
            raise ValueError(f"{job.get('id')} failure history is malformed")
    if failures and job.get("failure") != failures[-1]["reason"]:
        raise ValueError(f"{job.get('id')} retained failure reason differs from history")
    return len(attempts), len(failures)


def _validate_queue(
    queue: Mapping[str, Any],
    validated_config: Mapping[str, Any],
    *,
    config_path: Path,
    queue_path: Path,
    output_root: Path,
    snapshots: dict[Path, str],
) -> Sequence[Mapping[str, Any]]:
    mutable_queue = dict(queue)
    fair_queue._validate_loaded_queue(mutable_queue, validated_config, queue_path)
    public_config = _public_config(validated_config)
    exact = {
        "schema_version": 4,
        "kind": fair_queue.QUEUE_KIND,
        "status": "completed",
        "dry_run": False,
        "counts": {"completed": JOB_COUNT},
        "job_count": JOB_COUNT,
        "predicted_training_interactions": JOB_COUNT * INTERACTIONS,
        "predicted_evaluation_episodes": JOB_COUNT * EPISODES,
        "controller_order": list(CONTROLLERS),
        "seed_order": list(SEEDS),
        "task_order": list(TASKS),
        "task_protocols": dict(fair_queue.PROTOCOL_BY_TASK),
        "lif_job_count": 48,
        "baseline_job_count": 12,
        "lif_first": True,
        "maximum_parallel": 1,
        "failure_isolation": True,
        "retry_failed_or_paused_only_with_resume": True,
        "historical_v2_fair_evidence_eligible": False,
        "config_file_sha256": _sha256_file(config_path),
        "config_identity_sha256": _canonical_sha256(public_config),
        "config": public_config,
    }
    for name, expected in exact.items():
        if queue.get(name) != expected:
            raise ValueError(f"revision-4 queue {name} differs from {expected!r}")
    if queue.get("resource_block") not in (None, {}):
        raise ValueError("completed revision-4 queue retains a resource block")
    if _resolve(str(queue.get("config_path", ""))) != config_path:
        raise ValueError("revision-4 queue config path differs")
    if _resolve(str(queue.get("queue_file", ""))) != queue_path:
        raise ValueError("revision-4 queue file identity differs")
    if _resolve(str(queue.get("output_root", ""))) != output_root:
        raise ValueError("revision-4 queue output root differs")
    resources = _mapping(queue.get("resource_limits"), "queue.resource_limits")
    if resources != {
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "sustained_paging_sample_count": 3,
        "default_max_parallel": 1,
        "maximum_parallel": 1,
    }:
        raise ValueError("revision-4 queue resource declaration differs")
    dependencies = (
        ("queue_runner", "queue_runner_sha256", "crazyflie_command_all_fair_queue_v4.py"),
        ("v1_helper", "v1_helper_sha256", "crazyflie_command_queue.py"),
        ("v2_supervisor_helper", "v2_supervisor_helper_sha256", "crazyflie_command_queue_v2.py"),
        ("v3_validator_helper", "v3_validator_helper_sha256", "crazyflie_command_combinations_queue_v3.py"),
        ("trainer", "trainer_sha256", "drone_train.py"),
        ("evaluator", "evaluator_sha256", "crazyflie_command_evaluate.py"),
    )
    for path_field, hash_field, basename in dependencies:
        raw_path = queue.get(path_field)
        expected_hash = queue.get(hash_field)
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError(f"queue lacks authenticated {path_field}")
        path = _resolve(raw_path)
        if path.name != basename:
            raise ValueError(f"queue {path_field} has the wrong file identity")
        _record_hash(snapshots, path, expected_hash)
    if queue.get("historical_v2") != public_config["historical_v2_identity"]:
        raise ValueError("queue historical-v2 preservation identity differs")
    if queue.get("preserved_v3") != public_config["preserved_v3_identity"]:
        raise ValueError("queue preserved-v3 identity differs")
    reports = _mapping(queue.get("controller_reports"), "queue.controller_reports")
    if set(reports) != set(CONTROLLERS):
        raise ValueError("revision-4 controller-report set differs")
    for controller in CONTROLLERS:
        _validate_controller_report(
            controller, _mapping(reports[controller], f"report {controller}")
        )
    jobs_raw = queue.get("jobs")
    if not isinstance(jobs_raw, list) or len(jobs_raw) != JOB_COUNT:
        raise ValueError("revision-4 queue must contain exactly 60 jobs")
    reference_source: Mapping[str, Any] | None = None
    failure_events = [
        event
        for event in queue.get("events", [])
        if isinstance(event, Mapping) and event.get("event") == "job_failure_isolated"
    ]
    recorded_failure_count = 0
    jobs: list[Mapping[str, Any]] = []
    for index, (raw_job, expected_pair) in enumerate(
        zip(jobs_raw, _expected_pairs(), strict=True), start=1
    ):
        job = _mapping(raw_job, f"queue.jobs[{index - 1}]")
        controller, seed, task = expected_pair
        condition = _condition(task)
        exact_job = {
            "id": f"{index:03d}__{controller}__{condition}__seed-{seed}",
            "controller": controller,
            "policy": POLICY_BY_CONTROLLER[controller],
            "controller_priority": CONTROLLERS.index(controller),
            "seed_priority": SEEDS.index(seed),
            "task_priority": TASKS.index(task),
            "architecture_class": (
                "lif" if controller in LIF_CONTROLLERS else "baseline"
            ),
            "controller_components": list(COMPONENTS[controller]),
            "task": task,
            "evaluation_protocol": "command_v2",
            "seed": seed,
            "command_schedule_seed": seed,
            "paired_task_seed_key": f"{controller}__seed-{seed}",
            "contract_profile": "command_v2",
            "status": "completed",
            "training_status": "completed",
            "evaluation_status": "completed",
            "total_interactions": INTERACTIONS,
            "expected_updates": 250,
            "expected_actor_trainable_parameters": (
                fair_queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
            ),
            "expected_critic_trainable_parameters": (
                fair_queue.CRITIC_PARAMETERS_BY_CONTROLLER[controller]
            ),
            "expected_dynamic_state_per_environment": (
                fair_queue.DYNAMIC_STATE_BY_CONTROLLER[controller]
            ),
        }
        for name, expected in exact_job.items():
            if job.get(name) != expected:
                raise ValueError(f"revision-4 job {index} {name} differs")
        if "active_process" in job:
            raise ValueError(f"completed job {job['id']} retains an active process")
        run_dir = _resolve(str(job.get("run_dir", "")))
        _require_under(run_dir, output_root / "jobs", f"job {index} run_dir")
        expected_paths = {
            "training_manifest": run_dir / "training_manifest.json",
            "checkpoint": run_dir / "checkpoints/latest.pt",
            "pause_file": run_dir / "pause.request",
        }
        for name, expected_path in expected_paths.items():
            actual = _resolve(str(job.get(name, "")))
            _require_under(actual, output_root, f"job {index} {name}")
            if actual != expected_path:
                raise ValueError(f"revision-4 job {index} {name} is misplaced")
        evaluation_output = _resolve(str(job.get("evaluation_output", "")))
        _require_under(
            evaluation_output, output_root / "evaluations",
            f"job {index} evaluation_output",
        )
        payload = _mapping(job.get("fingerprint_payload"), f"job {index} fingerprint")
        fingerprint = job.get("expected_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != _canonical_sha256(payload):
            raise ValueError(f"revision-4 job {index} reproduction fingerprint is invalid")
        source = _validate_source_snapshot(payload, snapshots, f"job {index}")
        if reference_source is None:
            reference_source = source
        elif dict(source) != dict(reference_source):
            raise ValueError("revision-4 jobs do not share one source snapshot")
        evaluation_manifest = _mapping(
            job.get("evaluation_manifest"), f"job {index} evaluation manifest"
        )
        manifest_id = evaluation_manifest.get("manifest_id")
        unsigned = {
            key: value
            for key, value in evaluation_manifest.items()
            if key != "manifest_id"
        }
        if (
            not isinstance(manifest_id, str)
            or manifest_id != _canonical_sha256(unsigned)
            or job.get("evaluation_manifest_id") != manifest_id
            or evaluation_manifest.get("task") != task
            or evaluation_manifest.get("protocol") != "command_v2"
        ):
            raise ValueError(f"revision-4 job {index} evaluation identity is invalid")
        report = _mapping(reports[controller], f"controller report {controller}")
        identities = {
            "controller_report_sha256": _canonical_sha256(report),
            "expected_core_checksum": report.get("core_checksum"),
            "expected_per_core_checksums": report.get("per_core_checksums", {}),
            "expected_connectome_manifests": report.get("connectome_manifests", {}),
            "expected_connectome_checksums": report.get("connectome_checksums", {}),
            "command_training_contract_sha256": fair_queue.CONTRACT_SHA_BY_TASK[task],
        }
        for name, expected in identities.items():
            if job.get(name) != expected:
                raise ValueError(f"revision-4 job {index} {name} differs")
        _attempts, failures = _validate_attempts(job, output_root, snapshots)
        recorded_failure_count += failures
        jobs.append(job)
    if recorded_failure_count != len(failure_events):
        raise ValueError("isolated failure events do not reconcile with job histories")
    for job in jobs:
        for failure in job["failure_history"]:
            matches = [
                event
                for event in failure_events
                if event.get("job") == job["id"]
                and event.get("index") == failure["index"]
                and event.get("phase") == failure["phase"]
                and event.get("exit_code") == failure["exit_code"]
                and event.get("reason") == failure["reason"]
                and event.get("utc") == failure["utc"]
            ]
            if len(matches) != 1:
                raise ValueError(f"{job['id']} isolated failure lacks one queue event")
    return jobs


_SCHEDULE_STATE_KEYS = {
    "schema_version",
    "kind",
    "contract_sha256",
    "num_envs",
    "wind_enabled",
    "command_schedule_seed",
    "wind_schedule_seed",
    "held_out_wind_seed",
    "wind_active_seed",
    "wind_mode",
    "training_interactions",
    "next_command_segment_index",
    "command_steps_remaining",
    "requested_command_body",
    "command_category_code",
    "command_stage_index",
    "next_wind_segment_index",
    "wind_steps_remaining",
    "wind_force_ratio_world",
    "wind_torque_ratio_world",
    "applied_wind_force_world",
    "applied_wind_torque_world",
    "wind_category_code",
    "wind_stage_index",
    "heldout_episode_index",
    "heldout_step_cursor",
    "wind_evaluation_protocol_sha256",
}
_COMMAND_STATE_FIELDS = (
    "schema_version",
    "kind",
    "num_envs",
    "command_schedule_seed",
    "training_interactions",
    "next_command_segment_index",
    "command_steps_remaining",
    "requested_command_body",
    "command_category_code",
    "command_stage_index",
)
_WIND_STATE_FIELDS = (
    "schema_version",
    "kind",
    "num_envs",
    "wind_enabled",
    "wind_schedule_seed",
    "held_out_wind_seed",
    "wind_active_seed",
    "wind_mode",
    "training_interactions",
    "next_wind_segment_index",
    "wind_steps_remaining",
    "wind_force_ratio_world",
    "wind_torque_ratio_world",
    "applied_wind_force_world",
    "applied_wind_torque_world",
    "wind_category_code",
    "wind_stage_index",
    "heldout_episode_index",
    "heldout_step_cursor",
    "wind_evaluation_protocol_sha256",
)


def _vector_rows(
    value: Any, *, label: str, rows: int, width: int | None = None
) -> list[Any]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} must contain exactly {rows} environment rows")
    if width is not None:
        for index, row in enumerate(value):
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"{label}[{index}] must have width {width}")
            for column, item in enumerate(row):
                _finite(item, f"{label}[{index}][{column}]")
    return value


def _validate_schedule(
    value: Any, *, job: Mapping[str, Any]
) -> tuple[str, str]:
    schedule = _mapping(value, f"{job['id']} command_schedule")
    state = _mapping(schedule.get("state"), f"{job['id']} command schedule state")
    if set(state) != _SCHEDULE_STATE_KEYS:
        raise ValueError(f"{job['id']} command schedule state fields differ")
    if schedule.get("command_training_contract_sha256") != job.get(
        "command_training_contract_sha256"
    ):
        raise ValueError(f"{job['id']} command training contract differs")
    if schedule.get("state_sha256") != _canonical_sha256(state):
        raise ValueError(f"{job['id']} command schedule state hash does not recompute")
    wind = job["task"] == TASKS[1]
    exact = {
        "schema_version": 2,
        "kind": "flyg1.crazyflie.command-wide-schedule.v2",
        "contract_sha256": job["command_training_contract_sha256"],
        "num_envs": 40,
        "wind_enabled": wind,
        "command_schedule_seed": job["seed"],
        "wind_schedule_seed": TRAINING_WIND_SEED,
        "held_out_wind_seed": HELD_OUT_WIND_SEED,
        "wind_active_seed": TRAINING_WIND_SEED,
        "wind_mode": "training",
        "training_interactions": INTERACTIONS,
    }
    for name, expected in exact.items():
        if state.get(name) != expected:
            raise ValueError(f"{job['id']} schedule state {name} differs")
    wind_protocol = state.get("wind_evaluation_protocol_sha256")
    if not isinstance(wind_protocol, str) or len(wind_protocol) != 64:
        raise ValueError(f"{job['id']} schedule lacks the wind protocol hash")
    integer_vectors = (
        "next_command_segment_index",
        "command_steps_remaining",
        "command_category_code",
        "command_stage_index",
        "next_wind_segment_index",
        "wind_steps_remaining",
        "wind_category_code",
        "wind_stage_index",
        "heldout_episode_index",
        "heldout_step_cursor",
    )
    for name in integer_vectors:
        rows = _vector_rows(state.get(name), label=f"{job['id']} {name}", rows=40)
        if any(type(item) is not int or item < 0 for item in rows):
            raise ValueError(f"{job['id']} {name} contains an invalid cursor")
    _vector_rows(
        state.get("requested_command_body"),
        label=f"{job['id']} requested_command_body",
        rows=40,
        width=4,
    )
    for name in (
        "wind_force_ratio_world",
        "wind_torque_ratio_world",
        "applied_wind_force_world",
        "applied_wind_torque_world",
    ):
        rows = _vector_rows(
            state.get(name), label=f"{job['id']} {name}", rows=40, width=3
        )
        if not wind and any(float(item) != 0.0 for row in rows for item in row):
            raise ValueError(f"{job['id']} still-air schedule contains physical wind")
    command_identity = {name: state[name] for name in _COMMAND_STATE_FIELDS}
    wind_identity = (
        {name: state[name] for name in _WIND_STATE_FIELDS} if wind else {}
    )
    return _canonical_sha256(command_identity), (
        _canonical_sha256(wind_identity) if wind_identity else ""
    )


def _validate_activity_summary(
    summary: Mapping[str, Any],
    *,
    expected_roles: Mapping[str, int],
    expected_sample_count: int,
    label: str,
) -> None:
    sample_count = _integer(summary.get("sample_count"), f"{label}.sample_count")
    if sample_count != expected_sample_count:
        raise ValueError(
            f"{label} sample count differs from the held-out protocol "
            f"({sample_count} != {expected_sample_count})"
        )
    for name in (
        "mean_absolute_activity_per_unit",
        "rms_activity_per_unit",
        "active_fraction_per_unit",
    ):
        number = _finite(summary.get(name), f"{label}.{name}")
        if number < 0.0 or (name == "active_fraction_per_unit" and number > 1.0):
            raise ValueError(f"{label}.{name} is outside its physical range")
    roles = _mapping(summary.get("roles"), f"{label}.roles")
    if set(roles) != set(expected_roles):
        raise ValueError(f"{label} role groups differ")
    for role, expected_count in expected_roles.items():
        role_summary = _mapping(roles[role], f"{label}.{role}")
        if role_summary.get("unit_count") != expected_count:
            raise ValueError(f"{label}.{role} unit count differs")
        mean = _finite(
            role_summary.get("mean_absolute_activity_per_unit"),
            f"{label}.{role}.mean",
        )
        active = _finite(
            role_summary.get("active_fraction_per_unit"),
            f"{label}.{role}.active",
        )
        if mean < 0.0 or not 0.0 <= active <= 1.0:
            raise ValueError(f"{label}.{role} activity is outside physical bounds")


def _expected_activity_sample_counts(
    episodes: Sequence[Mapping[str, Any]], *, job_id: str
) -> tuple[int, dict[str, int]]:
    """Recompute actual-forward samples through each episode's first done.

    The evaluator records the policy forward that produces a terminal action,
    but ``alive_steps`` excludes that terminal interval.  A naturally
    terminated episode therefore contributes ``alive_steps + 1`` activity
    samples, while a full-horizon truncated episode contributes exactly
    ``alive_steps``.  Invalid intervals are inadmissible under this report's
    integrity gate and cannot be used to infer activity coverage.
    """

    if len(episodes) != EPISODES:
        raise ValueError(f"{job_id} activity episodes differ from the protocol")
    segment_counts = {name: 0 for name in SEGMENT_NAMES}
    overall_count = 0
    for index, episode in enumerate(episodes):
        row = _mapping(episode, f"{job_id} activity episode {index}")
        alive_steps = _integer(
            row.get("alive_steps"), f"{job_id} episode {index} alive_steps"
        )
        invalid_steps = _integer(
            row.get("invalid_steps"), f"{job_id} episode {index} invalid_steps"
        )
        terminated = row.get("terminated")
        truncated = row.get("truncated")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError(f"{job_id} episode {index} done flags must be booleans")
        if invalid_steps != 0:
            raise ValueError(
                f"{job_id} activity coverage cannot include invalid intervals"
            )
        activity_steps = alive_steps + int(terminated)
        if not 1 <= activity_steps <= STEPS:
            raise ValueError(
                f"{job_id} episode {index} activity coverage is outside the protocol"
            )
        if not terminated and alive_steps != STEPS:
            raise ValueError(
                f"{job_id} episode {index} non-terminated activity coverage is partial"
            )
        overall_count += activity_steps
        for segment_index, name in enumerate(SEGMENT_NAMES):
            segment_counts[name] += min(
                max(activity_steps - segment_index * 50, 0), 50
            )
    return overall_count, segment_counts


def _validate_activity(
    value: Any,
    *,
    job: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    activity = dict(_mapping(value, f"{job['id']} evaluation.activity"))
    common._validate_detailed_activity(
        activity, str(job["policy"]), require_unit_detail=True
    )
    if not fair_queue._valid_activity(job, activity):
        raise ValueError(f"{job['id']} activity provenance differs from its controller")
    per_unit = activity["per_unit"]
    role_counts: dict[str, int] = {}
    for raw in per_unit:
        unit = _mapping(raw, f"{job['id']} activity unit")
        role = str(unit["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    controller = str(job["controller"])
    if controller in LIF_CONTROLLERS:
        if activity.get("kind") != "sampled_lif_spikes":
            raise ValueError(f"{job['id']} LIF activity kind differs")
        expected_role_names: set[str] = set()
        for component in COMPONENTS[controller]:
            expected_role_names.update(EXPECTED_ROLES[component])
        if set(role_counts) != expected_role_names:
            raise ValueError(f"{job['id']} does not include every biological role")
        for unit in per_unit:
            id_prefix = str(unit["id"]).split(":", 1)[0]
            role_prefix = str(unit["role"]).split(":", 1)[0]
            if id_prefix != role_prefix or role_prefix not in COMPONENTS[controller]:
                raise ValueError(f"{job['id']} activity core provenance differs")
    else:
        if activity.get("kind") != "engineering_absolute_activations":
            raise ValueError(f"{job['id']} baseline activity kind differs")
        expected_role_names = (
            {"gru:hidden"}
            if controller == "gru_matched"
            else {"mlp:hidden_0", "mlp:hidden_1"}
        )
        if set(role_counts) != expected_role_names:
            raise ValueError(f"{job['id']} engineering activity layers differ")
    if activity.get("role_counts") != role_counts:
        raise ValueError(f"{job['id']} activity role_counts do not recompute")
    expected_overall_samples, expected_segment_samples = (
        _expected_activity_sample_counts(episodes, job_id=str(job["id"]))
    )
    overall = _mapping(activity.get("overall"), f"{job['id']} activity overall")
    _validate_activity_summary(
        overall,
        expected_roles=role_counts,
        expected_sample_count=expected_overall_samples,
        label=f"{job['id']} activity overall",
    )
    count = len(per_unit)
    expected_overall = {
        "mean_absolute_activity_per_unit": sum(
            float(unit["mean_absolute_activity"]) for unit in per_unit
        )
        / count,
        "rms_activity_per_unit": math.sqrt(
            sum(float(unit["rms_activity"]) ** 2 for unit in per_unit) / count
        ),
        "active_fraction_per_unit": sum(
            float(unit["active_fraction"]) for unit in per_unit
        )
        / count,
    }
    for name, expected in expected_overall.items():
        if not math.isclose(
            float(overall[name]), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"{job['id']} activity overall {name} does not recompute")
    overall_roles = _mapping(overall["roles"], f"{job['id']} activity roles")
    for role, expected_count in role_counts.items():
        units = [unit for unit in per_unit if unit["role"] == role]
        summary = _mapping(overall_roles[role], f"{job['id']} role {role}")
        if summary.get("unit_count") != expected_count:
            raise ValueError(f"{job['id']} role {role} count does not recompute")
        expected_mean = sum(float(unit["mean_absolute_activity"]) for unit in units) / len(units)
        expected_active = sum(float(unit["active_fraction"]) for unit in units) / len(units)
        if not math.isclose(
            float(summary["mean_absolute_activity_per_unit"]),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            float(summary["active_fraction_per_unit"]),
            expected_active,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{job['id']} role {role} summary does not recompute")
    segments = _mapping(activity.get("segments"), f"{job['id']} activity segments")
    if set(segments) != set(SEGMENT_NAMES):
        raise ValueError(f"{job['id']} activity segment groups differ")
    segment_samples = 0
    for name in SEGMENT_NAMES:
        segment = _mapping(segments[name], f"{job['id']} activity segment {name}")
        _validate_activity_summary(
            segment,
            expected_roles=role_counts,
            expected_sample_count=expected_segment_samples[name],
            label=f"{job['id']} activity segment {name}",
        )
        segment_samples += int(segment["sample_count"])
    if segment_samples != overall["sample_count"]:
        raise ValueError(f"{job['id']} activity segment sample counts do not reconcile")
    return activity


def _validate_cell(
    job: Mapping[str, Any],
    *,
    controller_report: Mapping[str, Any],
    output_root: Path,
) -> AllFairCell:
    controller = str(job["controller"])
    task = str(job["task"])
    seed = int(job["seed"])
    cell = AllFairCell(
        controller=controller,
        policy=str(job["policy"]),
        task=task,
        condition=_condition(task),
        job_id=str(job["id"]),
        seed=seed,
        parameters=dict(controller_report),
        components=COMPONENTS[controller],
        capacity_tier=_capacity_tier(controller),
        attempt_count=len(job["attempts"]),
        prior_failure_count=len(job["failure_history"]),
    )
    manifest_path = _resolve(str(job["training_manifest"]))
    checkpoint_path = _resolve(str(job["checkpoint"]))
    evaluation_path = _resolve(str(job["evaluation_output"]))
    for path, label in (
        (manifest_path, "training manifest"),
        (checkpoint_path, "checkpoint"),
        (evaluation_path, "evaluation"),
    ):
        _require_under(path, output_root, f"{cell.job_id} {label}")
        if not path.is_file():
            raise ValueError(f"{cell.job_id} {label} is missing")
    if not fair_queue.valid_training(job):
        raise ValueError(f"{cell.job_id} fails the queue's training validator")
    if not fair_queue.valid_evaluation(job):
        raise ValueError(f"{cell.job_id} fails the queue's evaluation validator")
    manifest = _read_json(manifest_path)
    evaluation = _read_json(evaluation_path)
    input_hashes: dict[Path, str] = {}
    _record_hash(input_hashes, manifest_path)
    checkpoint_sha = _record_hash(input_hashes, checkpoint_path)
    _record_hash(input_hashes, evaluation_path)
    num_envs = _integer(manifest.get("num_envs"), "training num_envs", 1)
    horizon = _integer(manifest.get("horizon"), "training horizon", 1)
    expected_updates = INTERACTIONS // (num_envs * horizon)
    exact_manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": task,
        "contract_profile": "command_v2",
        "controller": job["policy"],
        "seed": seed,
        "num_envs": 40,
        "horizon": 100,
        "requested_interactions": INTERACTIONS,
        "environment_interactions": INTERACTIONS,
        "completed_updates": expected_updates,
        "fingerprint": job["expected_fingerprint"],
        "fingerprint_payload": job["fingerprint_payload"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "checkpoint_sha256": checkpoint_sha,
    }
    for name, expected in exact_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"{cell.job_id} training manifest {name} differs")
    if _resolve(str(manifest.get("checkpoint", ""))) != checkpoint_path:
        raise ValueError(f"{cell.job_id} training checkpoint path differs")
    if manifest.get("resolved_config") != job["fingerprint_payload"]["resolved_config"]:
        raise ValueError(f"{cell.job_id} resolved training config differs")
    if _canonical_sha256(manifest.get("controller_report")) != _canonical_sha256(
        controller_report
    ):
        raise ValueError(f"{cell.job_id} training controller report differs")
    train_memory = v3_helpers._validate_memory_gate(
        manifest.get("memory_gate"), f"{cell.job_id} training memory gate"
    )
    wall_time = _finite(
        manifest.get("training_wall_time_s"), f"{cell.job_id} training_wall_time_s"
    )
    if wall_time <= 0.0:
        raise ValueError(f"{cell.job_id} training wall time must be positive")
    command_hash, wind_hash = _validate_schedule(
        manifest.get("command_schedule"), job=job
    )
    history, history_hashes = common._load_history(
        _mapping(manifest.get("history_reference"), "training history_reference"),
        checkpoint=checkpoint_path,
        expected_updates=expected_updates,
        interactions_per_update=num_envs * horizon,
        total_interactions=INTERACTIONS,
    )
    input_hashes.update(history_hashes)
    expected_core = job["expected_core_checksum"]
    expected_per_core = job["expected_per_core_checksums"]
    for name, expected in (
        ("core_checksum_before", expected_core),
        ("core_checksum_after", expected_core),
        ("per_core_checksums_before", expected_per_core),
        ("per_core_checksums_after", expected_per_core),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"{cell.job_id} frozen-core gate {name} differs")
    exact_evaluation = {
        "schema_version": 1,
        "analysis_kind": ANALYSIS_KIND,
        "status": "PASS",
        "task": task,
        "controller": job["policy"],
        "episodes_requested": EPISODES,
        "episodes_evaluated": EPISODES,
        "steps_per_episode": STEPS,
        "vectorized_environment_count": EPISODES,
        "deterministic_actions": True,
        "policy_action_source": "actual_trained_controller_no_assist",
    }
    for name, expected in exact_evaluation.items():
        if evaluation.get(name) != expected:
            raise ValueError(f"{cell.job_id} evaluation {name} differs")
    if not common._finite_tree(evaluation):
        raise ValueError(f"{cell.job_id} evaluation contains nonfinite evidence")
    episodes_raw = evaluation.get("episodes")
    if not isinstance(episodes_raw, list) or len(episodes_raw) != EPISODES or not all(
        isinstance(episode, Mapping) and episode for episode in episodes_raw
    ):
        raise ValueError(f"{cell.job_id} evaluation does not contain 16 episodes")
    episodes = list(episodes_raw)
    checkpoint = _mapping(evaluation.get("checkpoint"), "evaluation checkpoint")
    checkpoint_exact = {
        "sha256": checkpoint_sha,
        "training_seed": seed,
        "total_interactions": INTERACTIONS,
        "reproduction_fingerprint": job["expected_fingerprint"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "evaluation_manifest": job["evaluation_manifest"],
    }
    for name, expected in checkpoint_exact.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"{cell.job_id} evaluation checkpoint {name} differs")
    if _resolve(str(checkpoint.get("path", ""))) != checkpoint_path:
        raise ValueError(f"{cell.job_id} evaluation checkpoint path differs")
    evaluation_manifest = job["evaluation_manifest"]
    if (
        evaluation.get("protocol")
        != evaluation_manifest["evaluation_protocol"]
        or evaluation.get("protocol_sha256")
        != evaluation_manifest["evaluation_protocol_sha256"]
    ):
        raise ValueError(f"{cell.job_id} evaluation protocol identity differs")
    eval_memory = v3_helpers._validate_memory_gate(
        evaluation.get("memory_gate"), f"{cell.job_id} evaluation memory gate"
    )
    integrity = _mapping(evaluation.get("integrity"), "evaluation.integrity")
    for name in (
        "task_manifest_matched",
        "evaluation_manifest_matched",
        "source_set_matched",
        "checkpoint_completed_budget",
        "all_actions_finite_and_bounded",
        "activity_from_actual_forward",
        "physical_wind_telemetry_passed",
        "physical_wind_uses_terminal_actual_interval",
        "inference_latency_from_actual_forward",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"{cell.job_id} integrity check {name} did not pass")
    summary = _mapping(evaluation.get("summary"), "evaluation.summary")
    common._validate_v2_physical_wind(
        summary.get("physical_wind"), condition=cell.condition
    )
    score = _mapping(summary.get("score"), "evaluation summary score")
    quality = _mapping(
        summary.get("control_quality"), "evaluation control quality"
    )
    common._validate_score(score)
    common._validate_quality(quality)
    if (
        score.get("protocol") != evaluation_manifest["evaluation_protocol"]
        or score.get("protocol_sha256")
        != evaluation_manifest["evaluation_protocol_sha256"]
    ):
        raise ValueError(f"{cell.job_id} score protocol differs")
    if not math.isclose(
        float(quality["component_scores_0_100"]["command_response"]),
        float(score["component_scores"]["response"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{cell.job_id} quality response differs from score")
    rewards_raw = _mapping(
        summary.get("reward_component_mean_per_episode"),
        "evaluation reward components",
    )
    if set(rewards_raw) != common.REWARD_COMPONENTS:
        raise ValueError(f"{cell.job_id} reward-component set differs")
    reward_components = {
        name: _finite(value, f"reward component {name}")
        for name, value in rewards_raw.items()
    }
    reward_sums = {name: 0.0 for name in common.REWARD_COMPONENTS}
    for index, episode in enumerate(episodes):
        episode_rewards = _mapping(
            episode.get("reward_components"),
            f"{cell.job_id} episode {index} reward components",
        )
        if set(episode_rewards) != common.REWARD_COMPONENTS:
            raise ValueError(f"{cell.job_id} episode {index} reward components differ")
        for name, value in episode_rewards.items():
            reward_sums[name] += _finite(
                value, f"{cell.job_id} episode {index} reward {name}"
            )
    for name, total in reward_sums.items():
        if not common._float32_episode_mean_matches(
            reward_components[name], total / EPISODES
        ):
            raise ValueError(f"{cell.job_id} reward-component mean {name} differs")
    v3_helpers._validate_summary_counts(summary, episodes)
    activity = _validate_activity(
        evaluation.get("activity"), job=job, episodes=episodes
    )
    latency = v3_helpers._validate_inference_latency(
        evaluation.get("inference_latency")
    )
    if _canonical_sha256(evaluation.get("controller_report")) != _canonical_sha256(
        controller_report
    ):
        raise ValueError(f"{cell.job_id} evaluation controller report differs")
    cell.complete = True
    cell.reason = "verified complete"
    cell.training_manifest = manifest
    cell.evaluation = evaluation
    cell.history = history
    cell.score = dict(score)
    cell.quality = dict(quality)
    cell.reward_components = reward_components
    cell.activity = activity
    cell.core_before = manifest["core_checksum_before"]
    cell.core_after = manifest["core_checksum_after"]
    cell.training_wall_time_s = wall_time
    cell.train_memory = train_memory
    cell.eval_memory = eval_memory
    cell.inference_latency = latency
    cell.command_schedule_identity_sha256 = command_hash
    cell.wind_schedule_identity_sha256 = wind_hash
    cell.input_hashes = input_hashes
    return cell


def _validate_paired_schedules(cells: Sequence[AllFairCell]) -> None:
    for seed in SEEDS:
        selected = [cell for cell in cells if cell.seed == seed]
        command_hashes = {
            cell.command_schedule_identity_sha256 for cell in selected
        }
        if len(selected) != len(CONTROLLERS) * len(TASKS) or len(command_hashes) != 1:
            raise ValueError(
                f"seed {seed} does not share one reset-invariant command schedule"
            )
        wind_hashes = {
            cell.wind_schedule_identity_sha256
            for cell in selected
            if cell.condition == "wind"
        }
        if len(wind_hashes) != 1 or "" in wind_hashes:
            raise ValueError(f"seed {seed} wind cells do not share one wind schedule")
        by_controller = {
            (cell.controller, cell.condition): cell for cell in selected
        }
        for controller in CONTROLLERS:
            if (
                by_controller[(controller, "still")].command_schedule_identity_sha256
                != by_controller[(controller, "wind")].command_schedule_identity_sha256
            ):
                raise ValueError(
                    f"{controller} seed {seed} still/wind command schedules differ"
                )


def collect_report_data(
    config_path: Path = DEFAULT_CONFIG, queue_path: Path | None = None
) -> AllFairReportData:
    validated_config, output_root = _validate_config(config_path)
    config_path = config_path.expanduser().resolve()
    queue_path = (
        queue_path.expanduser().resolve()
        if queue_path is not None
        else (output_root / fair_queue.QUEUE_FILE_NAME).resolve()
    )
    _require_under(queue_path, output_root, "revision-4 queue")
    if not queue_path.is_file():
        raise ValueError(f"revision-4 queue is missing: {queue_path}")
    snapshots: dict[Path, str] = {}
    config_sha = _record_hash(snapshots, config_path)
    queue_sha = _record_hash(snapshots, queue_path)
    _record_hash(snapshots, Path(__file__).resolve())
    _record_hash(snapshots, Path(v3_helpers.__file__).resolve())
    _record_hash(snapshots, Path(common.__file__).resolve())
    # Historical identities are preservation receipts only.  Their result
    # payloads are never loaded into cells, metrics, tables, or plots.
    public = _public_config(validated_config)
    for identity_name in ("historical_v2_identity", "preserved_v3_identity"):
        identity = _mapping(public[identity_name], identity_name)
        for path_field, hash_field in (
            ("config", "config_sha256"),
            ("runner", "runner_sha256"),
            ("completed_queue", "completed_queue_sha256"),
            ("completed_report", "completed_report_sha256"),
        ):
            if path_field in identity:
                _record_hash(
                    snapshots,
                    _resolve(str(identity[path_field])),
                    str(identity[hash_field]),
                )
    queue = _read_json(queue_path)
    jobs = _validate_queue(
        queue,
        validated_config,
        config_path=config_path,
        queue_path=queue_path,
        output_root=output_root,
        snapshots=snapshots,
    )
    reports = _mapping(queue["controller_reports"], "queue.controller_reports")
    cells = [
        _validate_cell(
            job,
            controller_report=_mapping(
                reports[job["controller"]], f"report {job['controller']}"
            ),
            output_root=output_root,
        )
        for job in jobs
    ]
    for cell in cells:
        for path, expected in cell.input_hashes.items():
            _record_hash(snapshots, path, expected)
    _validate_paired_schedules(cells)
    data = AllFairReportData(
        config_path=config_path,
        queue_path=queue_path,
        output_root=output_root,
        config_sha256=config_sha,
        queue_sha256=queue_sha,
        cells=cells,
        input_hashes=snapshots,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not data.complete:
        raise ValueError("revision-4 report requires all 60 authenticated cells")
    return data


def _cell_map(
    data: AllFairReportData,
) -> dict[tuple[str, int, str], AllFairCell]:
    return {
        (cell.controller, cell.seed, cell.condition): cell for cell in data.cells
    }


def _score(cell: AllFairCell, key: str = "total") -> float:
    if cell.score is None:
        raise ValueError(f"verified cell {cell.job_id} lacks a score")
    if key == "total":
        return float(cell.score["score"])
    return float(cell.score["component_scores"][key])


def _quality(cell: AllFairCell, key: str) -> float:
    if cell.quality is None:
        raise ValueError(f"verified cell {cell.job_id} lacks control quality")
    return float(cell.quality["component_scores_0_100"][key])


def _raw(cell: AllFairCell, key: str) -> float:
    if cell.score is None:
        raise ValueError(f"verified cell {cell.job_id} lacks raw score metrics")
    return float(cell.score["raw"][key])


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric series")
    return statistics.mean(values), _sample_sd(values)


def _fmt(value: Any, digits: int = 3) -> str:
    return common._fmt(value, digits)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return common._markdown_table(headers, rows)


def _raw_seed_text(values: Mapping[int, float], digits: int = 3) -> str:
    if set(values) != set(SEEDS):
        raise ValueError("aggregate metric lacks all three raw seeds")
    return ", ".join(f"s{seed}={_fmt(values[seed], digits)}" for seed in SEEDS)


def _series_text(values: Mapping[int, float], digits: int = 3) -> str:
    ordered = [float(values[seed]) for seed in SEEDS]
    mean, spread = _mean_sd(ordered)
    return (
        f"{_fmt(mean, digits)} ± {_fmt(spread, digits)} "
        f"[{_raw_seed_text(values, digits)}]"
    )


def _series_for(
    data: AllFairReportData,
    controller: str,
    condition: str,
    getter: Callable[[AllFairCell], float],
) -> dict[int, float]:
    cells = _cell_map(data)
    return {
        seed: float(getter(cells[(controller, seed, condition)]))
        for seed in SEEDS
    }


def _wind_delta_series(
    data: AllFairReportData,
    controller: str,
    getter: Callable[[AllFairCell], float],
) -> dict[int, float]:
    cells = _cell_map(data)
    return {
        seed: float(getter(cells[(controller, seed, "wind")]))
        - float(getter(cells[(controller, seed, "still")]))
        for seed in SEEDS
    }


def _activity_groups(
    cell: AllFairCell,
) -> dict[str, tuple[float, float, int]]:
    if cell.activity is None:
        raise ValueError(f"verified cell {cell.job_id} lacks activity")
    units = cell.activity["per_unit"]
    if cell.controller in LIF_CONTROLLERS:
        group_names = list(cell.components)
        selector = lambda unit, group: str(unit["id"]).split(":", 1)[0] == group
    else:
        group_names = list(cell.activity["overall"]["roles"])
        selector = lambda unit, group: unit["role"] == group
    result: dict[str, tuple[float, float, int]] = {}
    for group in group_names:
        selected = [unit for unit in units if selector(unit, group)]
        if not selected:
            raise ValueError(f"verified cell {cell.job_id} lacks activity group {group}")
        result[group] = (
            sum(float(unit["mean_absolute_activity"]) for unit in selected)
            / len(selected),
            sum(float(unit["active_fraction"]) for unit in selected)
            / len(selected),
            len(selected),
        )
    return result


def _final_history(cell: AllFairCell, name: str) -> float:
    if not cell.history:
        raise ValueError(f"verified cell {cell.job_id} lacks training history")
    return float(cell.history[-1][name])


def render_markdown(
    data: AllFairReportData, plot_paths: Mapping[str, Path]
) -> str:
    cells = _cell_map(data)
    aggregate_scores = {
        (controller, condition): statistics.mean(
            _series_for(data, controller, condition, _score).values()
        )
        for controller in CONTROLLERS
        for condition in ("still", "wind")
    }
    best_pair, best_mean = max(aggregate_scores.items(), key=lambda item: item[1])
    command_hashes = {
        seed: cells[(CONTROLLERS[0], seed, "still")].command_schedule_identity_sha256
        for seed in SEEDS
    }
    wind_hashes = {
        seed: cells[(CONTROLLERS[0], seed, "wind")].wind_schedule_identity_sha256
        for seed in SEEDS
    }
    lines = [
        "# Crazyflie all-fair command-control comparison (revision 4)",
        "",
        "Report state: **COMPLETE — all 60 fresh revision-4 cells independently verified**",
        "",
        "## Executive summary",
        "",
        (
            f"The highest observed three-seed mean held-out score is {_fmt(best_mean)} "
            f"for `{best_pair[0]}` in `{best_pair[1]}`. This is a descriptive result, "
            "not a causal topology claim. Exact architecture comparisons are restricted "
            "to controllers in the same capacity tier."
        ),
        "",
        (
            "All 60 jobs trained from fresh initialization for exactly 1,000,000 "
            "interactions and contributed 16 held-out episodes of 600 steps: "
            "60,000,000 training interactions and 960 evaluation episodes in total."
        ),
        "",
        (
            "Seeds 0, 1, and 2 each passed the reset-invariant paired-command identity "
            "gate; all ten wind cells within a seed also shared one wind schedule."
        ),
        "",
        f"Generated (UTC): `{data.generated_at_utc}`  ",
        f"Config SHA-256: `{data.config_sha256}`  ",
        f"Queue snapshot SHA-256: `{data.queue_sha256}`",
        "",
        "## Evidence boundary and fairness",
        "",
        (
            "This report uses only the 60 new revision-4 cells: 10 controllers × 2 "
            "conditions × 3 independent seeds. Revision-2 and revision-3 artifacts are "
            "preserved by pinned identity only; none of their scores, episodes, histories, "
            "latencies, activity, or resource measurements are merged here."
        ),
        "",
        (
            "The one-core LIF tier compares Original, Rewired, Wing, and Optic at exactly "
            "4,776 actor parameters and 1,024 dynamic-state values per environment. The "
            "two-core LIF tier compares Leg+Wing, Leg+Optic, and Wing+Optic at exactly "
            "9,224 actor parameters and 2,048 state values. The three-core LIF has 13,672 "
            "actor parameters and 3,072 state values and is isolated. GRU (4,793 actor "
            "parameters) and MLP (4,827) are near-capacity engineering baselines for the "
            "one-core tier. Cross-tier comparisons are descriptive only."
        ),
        "",
        "Activity values are observed correlations from actual action-producing forwards; they do not establish causal neuron or connectome effects.",
        "",
        "## Per-seed 20-cell results",
    ]
    for seed in SEEDS:
        lines.extend(("", f"### Seed {seed}", ""))
        rows = []
        for controller in CONTROLLERS:
            for condition in ("still", "wind"):
                cell = cells[(controller, seed, condition)]
                rows.append(
                    (
                        controller,
                        condition,
                        _capacity_tier(controller),
                        cell.parameters["actor_trainable_parameters"],
                        cell.parameters["total_dynamic_state_per_environment"],
                        _fmt(_score(cell)),
                        _fmt(_score(cell, "linear_tracking")),
                        _fmt(_quality(cell, "acceleration_quality")),
                        _fmt(_quality(cell, "command_response")),
                        _fmt(_quality(cell, "flight_stability")),
                        _fmt(_quality(cell, "survival_not_die")),
                        _fmt(_final_history(cell, "mean_rollout_reward")),
                        _fmt(_final_history(cell, "loss"), 5),
                        _fmt(cell.inference_latency["mean_ms"], 4),
                        _fmt(cell.train_memory["max_device_gpu_used_mib"], 1),
                        _fmt(cell.eval_memory["max_device_gpu_used_mib"], 1),
                    )
                )
        lines.extend(
            _table(
                (
                    "Controller",
                    "Condition",
                    "Capacity tier",
                    "Actor params",
                    "State/env",
                    "Score",
                    "Tracking",
                    "Acceleration",
                    "Response",
                    "Stability",
                    "Survival",
                    "Final train reward",
                    "Final loss",
                    "Inference mean ms / 16-env call",
                    "Train peak VRAM MiB",
                    "Eval peak VRAM MiB",
                ),
                rows,
            )
        )
    lines.extend(("", "## Aggregate controller-condition results across three seeds", ""))
    aggregate_rows = []
    for controller in CONTROLLERS:
        for condition in ("still", "wind"):
            aggregate_rows.append(
                (
                    controller,
                    condition,
                    _capacity_tier(controller),
                    _series_text(_series_for(data, controller, condition, _score)),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _quality(cell, "acceleration_quality"),
                        )
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _quality(cell, "command_response"),
                        )
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _quality(cell, "flight_stability"),
                        )
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _quality(cell, "survival_not_die"),
                        )
                    ),
                )
            )
    lines.extend(
        _table(
            (
                "Controller",
                "Condition",
                "Capacity tier",
                "Score mean ± sample SD [raw seeds]",
                "Acceleration mean ± SD [raw]",
                "Response mean ± SD [raw]",
                "Stability mean ± SD [raw]",
                "Survival mean ± SD [raw]",
            ),
            aggregate_rows,
        )
    )
    lines.extend(("", f"![Held-out score]({plot_paths['score']})", ""))
    lines.extend(("## All held-out score subscores", ""))
    subscore_rows = []
    for controller in CONTROLLERS:
        for condition in ("still", "wind"):
            subscore_rows.append(
                (
                    controller,
                    condition,
                    *(
                        _series_text(
                            _series_for(
                                data,
                                controller,
                                condition,
                                lambda cell, key=key: _score(cell, key),
                            )
                        )
                        for key in SCORE_COMPONENTS
                    ),
                )
            )
    lines.extend(
        _table(
            ("Controller", "Condition", *SCORE_COMPONENTS), subscore_rows
        )
    )
    lines.extend(("", "## Paired wind effect (wind minus still within seed)", ""))
    wind_rows = []
    for controller in CONTROLLERS:
        wind_rows.append(
            (
                controller,
                _series_text(_wind_delta_series(data, controller, _score)),
                _series_text(
                    _wind_delta_series(
                        data, controller, lambda cell: _score(cell, "linear_tracking")
                    )
                ),
                _series_text(
                    _wind_delta_series(
                        data,
                        controller,
                        lambda cell: _quality(cell, "acceleration_quality"),
                    )
                ),
                _series_text(
                    _wind_delta_series(
                        data,
                        controller,
                        lambda cell: _quality(cell, "command_response"),
                    )
                ),
                _series_text(
                    _wind_delta_series(
                        data,
                        controller,
                        lambda cell: _quality(cell, "flight_stability"),
                    )
                ),
                _series_text(
                    _wind_delta_series(
                        data,
                        controller,
                        lambda cell: _quality(cell, "survival_not_die"),
                    )
                ),
                _series_text(
                    _wind_delta_series(
                        data,
                        controller,
                        lambda cell: _raw(cell, "linear_tracking_rmse_m_s"),
                    )
                ),
            )
        )
    lines.extend(
        _table(
            (
                "Controller",
                "Δ score mean ± SD [raw seeds]",
                "Δ tracking",
                "Δ acceleration",
                "Δ response",
                "Δ stability",
                "Δ survival",
                "Δ linear RMSE m/s",
            ),
            wind_rows,
        )
    )
    lines.extend(
        (
            "",
            "Negative score deltas mean lower performance in wind; positive RMSE deltas mean more tracking error.",
            "",
            f"![Paired wind deltas]({plot_paths['wind_delta']})",
            "",
            "## Training reward and loss",
            "",
        )
    )
    training_rows = []
    for controller in CONTROLLERS:
        for condition in ("still", "wind"):
            training_rows.append(
                (
                    controller,
                    condition,
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _final_history(cell, "mean_rollout_reward"),
                        )
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _final_history(cell, "loss"),
                        ),
                        5,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _final_history(cell, "policy_loss"),
                        ),
                        5,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: _final_history(cell, "value_loss"),
                        ),
                        5,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.training_wall_time_s),
                        ),
                        1,
                    ),
                )
            )
    lines.extend(
        _table(
            (
                "Controller",
                "Condition",
                "Final rollout reward mean ± SD [raw]",
                "Final total loss mean ± SD [raw]",
                "Final policy loss",
                "Final value loss",
                "Training wall s",
            ),
            training_rows,
        )
    )
    lines.extend(
        (
            "",
            f"![Training reward curves]({plot_paths['reward']})",
            "",
            f"![Training loss curves]({plot_paths['loss']})",
            "",
            "## Activity by authenticated biological core or engineering layer",
            "",
        )
    )
    activity_rows = []
    for controller in CONTROLLERS:
        example = cells[(controller, SEEDS[0], "still")]
        for condition in ("still", "wind"):
            for group in _activity_groups(example):
                mean_values = {
                    seed: _activity_groups(cells[(controller, seed, condition)])[group][0]
                    for seed in SEEDS
                }
                active_values = {
                    seed: _activity_groups(cells[(controller, seed, condition)])[group][1]
                    for seed in SEEDS
                }
                unit_counts = {
                    _activity_groups(cells[(controller, seed, condition)])[group][2]
                    for seed in SEEDS
                }
                if len(unit_counts) != 1:
                    raise ValueError(f"{controller}/{group} activity unit count drifted")
                activity_rows.append(
                    (
                        controller,
                        condition,
                        "biological core" if controller in LIF_CONTROLLERS else "engineering layer",
                        group,
                        unit_counts.pop(),
                        _series_text(mean_values, 5),
                        _series_text(active_values, 5),
                    )
                )
    lines.extend(
        _table(
            (
                "Controller",
                "Condition",
                "Group type",
                "Authenticated group",
                "Units",
                "Mean |activity| mean ± SD [raw]",
                "Active fraction mean ± SD [raw]",
            ),
            activity_rows,
        )
    )
    lines.extend(
        (
            "",
            "The GRU and MLP rows are engineering-unit activations, not biological neurons. Activity differences are descriptive correlations only.",
            "",
            f"![Activity groups]({plot_paths['activity']})",
            "",
            "## Capacity, latency, efficiency, and resource gates",
            "",
        )
    )
    capacity_rows = []
    for controller in CONTROLLERS:
        actor = fair_queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
        critic = fair_queue.CRITIC_PARAMETERS_BY_CONTROLLER[controller]
        dynamic = fair_queue.DYNAMIC_STATE_BY_CONTROLLER[controller]
        for condition in ("still", "wind"):
            capacity_rows.append(
                (
                    controller,
                    condition,
                    _capacity_tier(controller),
                    actor,
                    critic,
                    dynamic,
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell, actor=actor: _score(cell) / (actor / 1000.0),
                        )
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.inference_latency["mean_ms"]),
                        ),
                        4,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.inference_latency["p95_ms"]),
                        ),
                        4,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.train_memory["max_process_rss_mib"]),
                        ),
                        1,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.train_memory["max_device_gpu_used_mib"]),
                        ),
                        1,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.eval_memory["max_process_rss_mib"]),
                        ),
                        1,
                    ),
                    _series_text(
                        _series_for(
                            data,
                            controller,
                            condition,
                            lambda cell: float(cell.eval_memory["max_device_gpu_used_mib"]),
                        ),
                        1,
                    ),
                )
            )
    lines.extend(
        _table(
            (
                "Controller",
                "Condition",
                "Capacity tier",
                "Actor params",
                "Critic params",
                "Dynamic state/env",
                "Score/1k actor params mean ± SD [raw]",
                "Inference mean ms / 16-env call",
                "Inference p95 ms / 16-env call",
                "Train peak RSS MiB",
                "Train peak VRAM MiB",
                "Eval peak RSS MiB",
                "Eval peak VRAM MiB",
            ),
            capacity_rows,
        )
    )
    lines.extend(
        (
            "",
            "Latency times the exact action-producing `policy.act` path with CUDA synchronization before and after each call. Ten real warmup calls are excluded from steady-state statistics; no extra forward passes are introduced.",
            "",
            f"![Inference latency]({plot_paths['latency']})",
            "",
            f"![Descriptive score efficiency]({plot_paths['efficiency']})",
            "",
            f"![Peak RAM and VRAM]({plot_paths['memory']})",
            "",
            "## Reset-invariant paired schedule proof",
            "",
        )
    )
    lines.extend(
        _table(
            ("Seed", "Shared command-state identity SHA-256", "Shared wind-state identity SHA-256"),
            [(seed, command_hashes[seed], wind_hashes[seed]) for seed in SEEDS],
        )
    )
    lines.extend(("", "## Frozen-core before/after integrity", ""))
    integrity_rows = []
    for cell in data.cells:
        if cell.controller not in LIF_CONTROLLERS:
            continue
        assert cell.training_manifest is not None
        labels = cell.parameters["core_labels"]
        before = cell.training_manifest["per_core_checksums_before"]
        after = cell.training_manifest["per_core_checksums_after"]
        for label in labels:
            integrity_rows.append(
                (
                    cell.job_id,
                    label,
                    cell.parameters["per_core_checksums"][label],
                    before[label],
                    after[label],
                    "PASS",
                )
            )
    lines.extend(
        _table(
            (
                "Job",
                "Core",
                "Declared SHA-256",
                "Before PPO SHA-256",
                "After PPO SHA-256",
                "Gate",
            ),
            integrity_rows,
        )
    )
    lines.extend(("", "## Immutable per-job evidence and resume history", ""))
    evidence_rows = []
    for cell in data.cells:
        assert cell.training_manifest is not None
        history_reference = cell.training_manifest["history_reference"]
        evidence_rows.append(
            (
                cell.job_id,
                "PASS",
                cell.training_manifest["checkpoint_sha256"],
                history_reference["history_sha256"],
                len(cell.history),
                cell.attempt_count,
                cell.prior_failure_count,
                _fmt(_score(cell)),
            )
        )
    lines.extend(
        _table(
            (
                "Queue job",
                "Validation",
                "Checkpoint SHA-256",
                "History SHA-256",
                "History rows",
                "Attempts",
                "Prior isolated failures",
                "Held-out score",
            ),
            evidence_rows,
        )
    )
    lines.extend(("", "## Authenticated input manifest", ""))
    lines.extend(
        _table(
            ("Input path", "SHA-256"),
            [
                (str(path), digest)
                for path, digest in sorted(
                    data.input_hashes.items(), key=lambda item: str(item[0])
                )
            ],
        )
    )
    lines.extend(
        (
            "",
            "Every listed input hash was rechecked immediately before no-overwrite publication. The report does not infer missing values, does not reuse historical scores, and makes no causal claim from activity or cross-capacity differences.",
            "",
        )
    )
    return "\n".join(lines)


def _palette() -> dict[str, str]:
    # Explicit, restrained, color-vision-conscious palette. Condition is also
    # encoded by hatch/marker/line style rather than color alone.
    colors = (
        "#2F5D8A",
        "#C47A1B",
        "#4E7A51",
        "#9A5672",
        "#596A9B",
        "#9B7B32",
        "#397A78",
        "#7A5A92",
        "#6B6B6B",
        "#B05A43",
    )
    return dict(zip(CONTROLLERS, colors, strict=True))


def _aggregate_curve(
    data: AllFairReportData,
    controller: str,
    condition: str,
    field: str,
) -> tuple[list[float], list[float], list[float]]:
    selected = [
        cell
        for cell in data.cells
        if cell.controller == controller and cell.condition == condition
    ]
    if [cell.seed for cell in selected] != list(SEEDS):
        raise ValueError(f"{controller}/{condition} curve lacks all ordered seeds")
    x_rows = [[float(row["total_interactions"]) for row in cell.history] for cell in selected]
    if not x_rows or any(row != x_rows[0] for row in x_rows[1:]):
        raise ValueError(f"{controller}/{condition} training x-axis differs by seed")
    y_rows = [[float(row[field]) for row in cell.history] for cell in selected]
    if any(len(row) != len(x_rows[0]) for row in y_rows):
        raise ValueError(f"{controller}/{condition} training curve lengths differ")
    means = [statistics.mean(values) for values in zip(*y_rows, strict=True)]
    spreads = [_sample_sd(values) for values in zip(*y_rows, strict=True)]
    return x_rows[0], means, spreads


def render_plots(
    data: AllFairReportData, destinations: Mapping[str, Path]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _palette()
    x = np.arange(len(CONTROLLERS), dtype=float)
    short = {
        "original_lif": "Original",
        "rewired_lif": "Rewired",
        "wing_lif": "Wing",
        "leg_wing_lif": "Leg+Wing",
        "optic_lif": "Optic",
        "leg_optic_lif": "Leg+Optic",
        "wing_optic_lif": "Wing+Optic",
        "leg_wing_optic_lif": "Leg+Wing+Optic",
        "gru_matched": "GRU",
        "mlp_normal": "MLP",
    }
    labels = [short[controller] for controller in CONTROLLERS]

    fig, axis = plt.subplots(figsize=(15.5, 7.0), constrained_layout=True)
    for offset, condition, hatch, alpha in (
        (-0.20, "still", "", 1.0),
        (0.20, "wind", "//", 0.62),
    ):
        means = []
        spreads = []
        for controller in CONTROLLERS:
            values = list(_series_for(data, controller, condition, _score).values())
            mean, spread = _mean_sd(values)
            means.append(mean)
            spreads.append(spread)
        bars = axis.bar(
            x + offset,
            means,
            0.38,
            yerr=spreads,
            capsize=3,
            color=[colors[controller] for controller in CONTROLLERS],
            alpha=alpha,
            edgecolor="#202020",
            linewidth=0.6,
            label=condition,
        )
        for bar in bars:
            bar.set_hatch(hatch)
    axis.axvline(3.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.axvline(6.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.axvline(7.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Held-out command score (0–100)")
    axis.set_title("Fresh v4 held-out score across three seeds (error bars: sample SD)")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axis.legend(title="Condition", ncols=2)
    fig.savefig(destinations["score"], dpi=170, facecolor="white")
    plt.close(fig)

    metrics: tuple[tuple[str, Callable[[AllFairCell], float]], ...] = (
        ("Score", _score),
        ("Acceleration", lambda cell: _quality(cell, "acceleration_quality")),
        ("Response", lambda cell: _quality(cell, "command_response")),
        ("Stability", lambda cell: _quality(cell, "flight_stability")),
        ("Survival", lambda cell: _quality(cell, "survival_not_die")),
    )
    fig, axis = plt.subplots(figsize=(16.0, 7.5), constrained_layout=True)
    widths = 0.16
    metric_colors = ("#2F5D8A", "#C47A1B", "#4E7A51", "#9A5672", "#6B6B6B")
    for metric_index, ((metric, getter), color) in enumerate(
        zip(metrics, metric_colors, strict=True)
    ):
        means = []
        spreads = []
        for controller in CONTROLLERS:
            values = list(_wind_delta_series(data, controller, getter).values())
            mean, spread = _mean_sd(values)
            means.append(mean)
            spreads.append(spread)
        axis.bar(
            x + (metric_index - 2) * widths,
            means,
            widths,
            yerr=spreads,
            capsize=2,
            color=color,
            label=metric,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel("Wind minus still (score points)")
    axis.set_title("Paired wind deltas across three seeds (error bars: sample SD)")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axis.legend(ncols=5)
    fig.savefig(destinations["wind_delta"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2), sharey=True, constrained_layout=True)
    for axis, condition in zip(axes, ("still", "wind"), strict=True):
        for controller in CONTROLLERS:
            interactions, means, spreads = _aggregate_curve(
                data, controller, condition, "mean_rollout_reward"
            )
            lower = np.asarray(means) - np.asarray(spreads)
            upper = np.asarray(means) + np.asarray(spreads)
            axis.plot(
                interactions,
                means,
                color=colors[controller],
                linewidth=1.35,
                label=short[controller],
            )
            axis.fill_between(interactions, lower, upper, color=colors[controller], alpha=0.08)
        axis.set_title(condition)
        axis.set_xlabel("Training interactions")
        axis.grid(color="#D8D8D8", linewidth=0.7)
    axes[0].set_ylabel("Mean rollout reward (three-seed mean ± sample SD)")
    axes[1].legend(ncols=2, fontsize=7, loc="best")
    fig.suptitle("Fresh v4 training reward")
    fig.savefig(destinations["reward"], dpi=170, facecolor="white")
    plt.close(fig)

    loss_fields = (
        ("loss", "Total loss"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(17.0, 13.0), sharex=True, constrained_layout=True
    )
    for row_index, (field, ylabel) in enumerate(loss_fields):
        for column_index, condition in enumerate(("still", "wind")):
            axis = axes[row_index, column_index]
            for controller in CONTROLLERS:
                interactions, means, spreads = _aggregate_curve(
                    data, controller, condition, field
                )
                lower = np.asarray(means) - np.asarray(spreads)
                upper = np.asarray(means) + np.asarray(spreads)
                axis.plot(
                    interactions,
                    means,
                    color=colors[controller],
                    linewidth=1.15,
                    label=short[controller],
                )
                axis.fill_between(
                    interactions, lower, upper, color=colors[controller], alpha=0.07
                )
            axis.set_title(f"{condition}: {ylabel}")
            axis.set_ylabel(ylabel)
            axis.grid(color="#D8D8D8", linewidth=0.7)
    axes[-1, 0].set_xlabel("Training interactions")
    axes[-1, 1].set_xlabel("Training interactions")
    axes[0, 1].legend(ncols=2, fontsize=7, loc="best")
    fig.suptitle("Fresh v4 PPO losses (three-seed mean ± sample SD)")
    fig.savefig(destinations["loss"], dpi=170, facecolor="white")
    plt.close(fig)

    activity_labels: list[tuple[str, str]] = []
    for controller in CONTROLLERS:
        example = _cell_map(data)[(controller, SEEDS[0], "still")]
        activity_labels.extend((controller, group) for group in _activity_groups(example))
    activity_x = np.arange(len(activity_labels), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(18.0, 11.0), sharex=True, constrained_layout=True)
    for offset, condition, hatch, alpha in (
        (-0.19, "still", "", 1.0),
        (0.19, "wind", "//", 0.62),
    ):
        activity_means: list[float] = []
        active_means: list[float] = []
        for controller, group in activity_labels:
            values = [
                _activity_groups(_cell_map(data)[(controller, seed, condition)])[group]
                for seed in SEEDS
            ]
            activity_means.append(statistics.mean(value[0] for value in values))
            active_means.append(statistics.mean(value[1] for value in values))
        for axis, values in zip(axes, (activity_means, active_means), strict=True):
            bars = axis.bar(
                activity_x + offset,
                values,
                0.38,
                color=[colors[controller] for controller, _group in activity_labels],
                alpha=alpha,
                edgecolor="#202020",
                linewidth=0.5,
                label=condition,
            )
            for bar in bars:
                bar.set_hatch(hatch)
    axes[0].set_ylabel("Mean |activity| per unit")
    axes[1].set_ylabel("Active fraction per unit")
    axes[1].set_xticks(
        activity_x,
        [f"{short[controller]}\n{group}" for controller, group in activity_labels],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    for axis in axes:
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axes[0].legend(title="Condition", ncols=2)
    fig.suptitle("Authenticated biological-core and engineering-layer activity")
    fig.savefig(destinations["activity"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.0), sharey=False, constrained_layout=True)
    for axis, field, title in (
        (axes[0], "mean_ms", "Mean latency"),
        (axes[1], "p95_ms", "p95 latency"),
    ):
        for offset, condition, hatch, alpha in (
            (-0.19, "still", "", 1.0),
            (0.19, "wind", "//", 0.62),
        ):
            means = []
            spreads = []
            for controller in CONTROLLERS:
                values = list(
                    _series_for(
                        data,
                        controller,
                        condition,
                        lambda cell, field=field: float(cell.inference_latency[field]),
                    ).values()
                )
                mean, spread = _mean_sd(values)
                means.append(mean)
                spreads.append(spread)
            bars = axis.bar(
                x + offset,
                means,
                0.38,
                yerr=spreads,
                capsize=2,
                color=[colors[controller] for controller in CONTROLLERS],
                alpha=alpha,
                edgecolor="#202020",
                linewidth=0.5,
                label=condition,
            )
            for bar in bars:
                bar.set_hatch(hatch)
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("Milliseconds per vectorized 16-env policy call")
        axis.set_title(title)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axes[0].legend(title="Condition", ncols=2)
    fig.suptitle("Action-producing inference latency across three seeds")
    fig.savefig(destinations["latency"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13.5, 8.0), constrained_layout=True)
    condition_markers = {"still": "o", "wind": "^"}
    condition_offsets = {"still": -45.0, "wind": 45.0}
    for controller in CONTROLLERS:
        actor = fair_queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
        for condition in ("still", "wind"):
            values = [
                value / (actor / 1000.0)
                for value in _series_for(data, controller, condition, _score).values()
            ]
            mean, spread = _mean_sd(values)
            axis.errorbar(
                actor + condition_offsets[condition],
                mean,
                yerr=spread,
                marker=condition_markers[condition],
                markersize=7,
                capsize=3,
                color=colors[controller],
                markeredgecolor="#202020",
                linewidth=1.0,
            )
            axis.annotate(
                f"{short[controller]} {condition}",
                (actor + condition_offsets[condition], mean),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axvline(7_000, color="#555555", linestyle=":", linewidth=1.2)
    axis.axvline(11_500, color="#555555", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Trainable actor parameters (small horizontal offset separates conditions)")
    axis.set_ylabel("Held-out score per 1,000 actor parameters")
    axis.set_title("Descriptive capacity efficiency; vertical dividers isolate capacity tiers")
    axis.grid(color="#D8D8D8", linewidth=0.7)
    fig.savefig(destinations["efficiency"], dpi=170, facecolor="white")
    plt.close(fig)

    memory_specs = (
        ("Train peak RSS", lambda cell: float(cell.train_memory["max_process_rss_mib"])),
        ("Train peak VRAM", lambda cell: float(cell.train_memory["max_device_gpu_used_mib"])),
        ("Eval peak RSS", lambda cell: float(cell.eval_memory["max_process_rss_mib"])),
        ("Eval peak VRAM", lambda cell: float(cell.eval_memory["max_device_gpu_used_mib"])),
    )
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 12.0), constrained_layout=True)
    for axis, (title, getter) in zip(axes.flat, memory_specs, strict=True):
        for offset, condition, hatch, alpha in (
            (-0.19, "still", "", 1.0),
            (0.19, "wind", "//", 0.62),
        ):
            means = []
            spreads = []
            for controller in CONTROLLERS:
                values = list(_series_for(data, controller, condition, getter).values())
                mean, spread = _mean_sd(values)
                means.append(mean)
                spreads.append(spread)
            bars = axis.bar(
                x + offset,
                means,
                0.38,
                yerr=spreads,
                capsize=2,
                color=[colors[controller] for controller in CONTROLLERS],
                alpha=alpha,
                edgecolor="#202020",
                linewidth=0.5,
                label=condition,
            )
            for bar in bars:
                bar.set_hatch(hatch)
        axis.set_xticks(x, labels, rotation=38, ha="right", fontsize=7)
        axis.set_ylabel("MiB")
        axis.set_title(title)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axes[0, 0].legend(title="Condition", ncols=2)
    fig.suptitle("Measured peak memory across three seeds (error bars: sample SD)")
    fig.savefig(destinations["memory"], dpi=170, facecolor="white")
    plt.close(fig)


def _revalidate_inputs(snapshots: Mapping[Path, str]) -> None:
    for path, expected in snapshots.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(
                f"authenticated input changed during report generation: {path}"
            )


def _publish_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report evidence: {destination}")
    os.link(source, destination)
    directory_fd = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_report_bundle(
    data: AllFairReportData, report_path: Path
) -> dict[str, Any]:
    if not data.complete:
        raise ValueError("refusing to publish an incomplete revision-4 report")
    report_path = report_path.expanduser().resolve()
    plot_paths = {
        name: data.output_root / filename for name, filename in PLOT_NAMES.items()
    }
    destinations = [report_path, *plot_paths.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing revision-4 report evidence: "
            + ", ".join(map(str, existing))
        )
    data.output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".crazyflie-command-v4-report-", dir=data.output_root)
    )
    report_temp: Path | None = None
    published: list[Path] = []
    try:
        staged_plots = {
            name: staging / filename for name, filename in PLOT_NAMES.items()
        }
        render_plots(data, staged_plots)
        relative_plots = {
            name: Path(os.path.relpath(path, report_path.parent))
            for name, path in plot_paths.items()
        }
        markdown = render_markdown(data, relative_plots)
        handle, raw_path = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
        )
        report_temp = Path(raw_path)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        _revalidate_inputs(data.input_hashes)
        for name in PLOT_NAMES:
            _publish_no_overwrite(staged_plots[name], plot_paths[name])
            published.append(plot_paths[name])
        _publish_no_overwrite(report_temp, report_path)
        published.append(report_path)
        return {
            "status": "COMPLETE",
            "schema_version": 4,
            "verified_jobs": JOB_COUNT,
            "controllers": len(CONTROLLERS),
            "conditions": len(TASKS),
            "seeds": list(SEEDS),
            "training_interactions": JOB_COUNT * INTERACTIONS,
            "evaluation_episodes": JOB_COUNT * EPISODES,
            "historical_cells_merged": 0,
            "report": str(report_path),
            "report_sha256": _sha256_file(report_path),
            "plots": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in plot_paths.items()
            },
        }
    except BaseException:
        # A bundle is all-or-nothing from the caller's perspective. Hard links
        # created during this attempt are safe to remove because destinations
        # were proven absent immediately before staging.
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        if report_temp is not None:
            report_temp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = collect_report_data(args.config, args.queue)
        result = write_report_bundle(data, args.report)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
