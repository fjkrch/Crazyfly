#!/usr/bin/env python3
"""Build the authenticated report for the six-cell LIF-combination matrix.

This reporter is deliberately separate from the completed revision-2 report.
It accepts only the three additive frozen-LIF combinations, requires every
still/wind cell to be complete, and publishes nothing unless all queue,
checkpoint, history, evaluation, source, memory, and per-core provenance
checks pass.  Cross-capacity numbers are descriptive only: the exact fair
comparison is leg+optic versus wing+optic inside the two-core tier.
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
import tempfile
from typing import Any, Mapping, Sequence

import crazyflie_command_report as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/experiments/crazyflie_command_combinations_seed0_1m.json"
)
DEFAULT_REPORT = ROOT / "docs/crazyflie_command_combinations_report.md"
TASKS = (
    "FlyCrazyflie-CommandFollowWide-v0",
    "FlyCrazyflie-CommandFollowWideWind-v0",
)
CONTROLLERS = (
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
)
COMPONENTS = {
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}
FUSION = {
    "leg_optic_lif": "independent_leg_and_optic_cores_concat_motor_readouts_v1",
    "wing_optic_lif": "independent_wing_and_optic_cores_concat_motor_readouts_v1",
    "leg_wing_optic_lif": (
        "independent_leg_and_wing_and_optic_cores_concat_motor_readouts_v1"
    ),
}
EXPECTED_ACTOR_PARAMETERS = {
    "leg_optic_lif": 9_224,
    "wing_optic_lif": 9_224,
    "leg_wing_optic_lif": 13_672,
}
EXPECTED_DYNAMIC_STATE = {
    "leg_optic_lif": 2_048,
    "wing_optic_lif": 2_048,
    "leg_wing_optic_lif": 3_072,
}
EXPECTED_ROLES = {
    "leg": {
        "leg:descending_input",
        "leg:motor_output",
        "leg:sensory_input",
        "leg:vnc_interneuron",
    },
    "wing": {
        "wing:descending_input",
        "wing:vnc_interneuron",
        "wing:wing_motor_output",
        "wing:wing_sensory_input",
    },
    "optic": {
        "optic:optic_intrinsic_interneuron",
        "optic:optic_sensory_input",
        "optic:visual_projection_output",
    },
}
QUEUE_KIND = "crazyflie_command_combinations_queue_v3"
ANALYSIS_KIND = "crazyflie_command_follow_heldout_v2"
INTERACTIONS = 1_000_000
EPISODES = 16
STEPS = 600
JOB_COUNT = 6
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
CRITIC_PARAMETERS = 18_305
PLOT_NAMES = {
    "score": "command_v3_score_by_condition.png",
    "wind_delta": "command_v3_wind_delta.png",
    "reward": "command_v3_training_reward.png",
    "loss": "command_v3_training_loss.png",
    "activity": "command_v3_per_core_activity.png",
    "efficiency": "command_v3_efficiency.png",
}


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
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
        raise ValueError(f"{label} escapes the revision-3 output root: {path}")


def _record_hash(snapshots: dict[Path, str], path: Path, expected: str | None = None) -> str:
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


def _condition(task: str) -> str:
    if task == TASKS[0]:
        return "still"
    if task == TASKS[1]:
        return "wind"
    raise ValueError(f"unsupported revision-3 task: {task}")


@dataclass
class CombinationCell(common.V2Cell):
    components: tuple[str, ...] = ()
    training_wall_time_s: float | None = None
    train_memory: dict[str, Any] = field(default_factory=dict)
    eval_memory: dict[str, Any] = field(default_factory=dict)
    inference_latency: dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinationReportData:
    config_path: Path
    queue_path: Path
    output_root: Path
    config_sha256: str
    queue_sha256: str
    cells: list[CombinationCell]
    historical_cells: list[common.V2Cell]
    input_hashes: dict[Path, str]
    generated_at_utc: str

    @property
    def complete(self) -> bool:
        expected_new = [
            (controller, task)
            for controller in CONTROLLERS
            for task in TASKS
        ]
        expected_historical = [
            (controller, task)
            for controller in common.V2_CONTROLLERS
            for task in common.V2_TASKS
        ]
        return (
            [(cell.controller, cell.task) for cell in self.cells] == expected_new
            and all(cell.complete for cell in self.cells)
            and [
                (cell.controller, cell.task) for cell in self.historical_cells
            ] == expected_historical
            and all(cell.complete for cell in self.historical_cells)
        )


def _validate_config(config: Mapping[str, Any], config_path: Path) -> Path:
    expected_keys = {
        "schema_version", "kind", "label", "output_root", "isaac_python",
        "tasks", "task_protocols", "contract_profile", "controllers",
        "trainer_policies", "seed", "total_interactions_per_job", "training",
        "command_envelope", "evaluation", "connectomes", "rewire", "queue",
        "comparison_contract", "base_matrix_identity",
        "prior_parallel_paging_evidence",
    }
    if set(config) != expected_keys:
        raise ValueError("revision-3 config top-level fields differ from the closed schema")
    exact = {
        "schema_version": 3,
        "kind": "crazyflie_command_combinations_matrix_v3",
        "label": "crazyflie_command_combinations_seed0_1m",
        "output_root": "runs/crazyflie_command_combinations_seed0_1m",
        "tasks": list(TASKS),
        "task_protocols": {task: "command_v2" for task in TASKS},
        "contract_profile": "command_v2",
        "controllers": list(CONTROLLERS),
        "trainer_policies": {controller: controller for controller in CONTROLLERS},
        "seed": 0,
        "total_interactions_per_job": INTERACTIONS,
    }
    for name, expected in exact.items():
        if config.get(name) != expected:
            raise ValueError(f"revision-3 config {name} must be exactly {expected!r}")
    training = _mapping(config.get("training"), "config.training")
    expected_training = {
        "num_envs": 40, "horizon": 100, "microbatch_size": 40,
        "ppo_epochs": 2, "learning_rate": 0.0003, "gamma": 0.99,
        "gae_lambda": 0.95, "clip_ratio": 0.2, "value_coefficient": 0.5,
        "entropy_coefficient": 0.002, "max_grad_norm": 1.0,
        "target_kl": 0.05, "checkpoint_every_updates": 25,
        "precision": "float32", "device": "cuda:0",
    }
    if dict(training) != expected_training:
        raise ValueError("revision-3 training contract differs from the reviewed PPO settings")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    expected_evaluation = {
        "episodes_per_job": EPISODES,
        "steps_per_episode": STEPS,
        "deterministic_actions": True,
        "activity_from_actual_controller": True,
        "device": "cuda:0",
        "script": "scripts/crazyflie_command_evaluate.py",
        "explicit_task_argument": True,
        "analysis_kind": ANALYSIS_KIND,
    }
    if dict(evaluation) != expected_evaluation:
        raise ValueError("revision-3 evaluation contract is not the reviewed 16 x 600 protocol")
    envelope = _mapping(config.get("command_envelope"), "config.command_envelope")
    if dict(envelope) != {
        "maximum_horizontal_speed_mps": 1.0,
        "maximum_vertical_speed_mps": 0.5,
        "maximum_yaw_rate_radps": 1.5,
        "minimum_hold_steps": 25,
        "maximum_hold_steps": 100,
        "control_dt_s": 0.02,
        "simultaneous_axes": True,
    }:
        raise ValueError("revision-3 command envelope differs from command_v2")
    queue_contract = _mapping(config.get("queue"), "config.queue")
    if dict(queue_contract) != {
        "default_max_parallel": 1,
        "maximum_parallel": 1,
        "lif_first": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
        "sustained_paging_sample_count": 3,
        "network_independent": True,
        "immutable_attempt_logs": True,
    }:
        raise ValueError("revision-3 queue contract differs from the strict sequential gate")
    comparison = _mapping(config.get("comparison_contract"), "config.comparison_contract")
    if comparison.get("combinations") != {
        key: list(value) for key, value in COMPONENTS.items()
    } or comparison.get("fusion_by_controller") != FUSION:
        raise ValueError("revision-3 combination/fusion declaration differs from the plan")
    expected_capacity = {
        "actor_trainable_parameters": dict(EXPECTED_ACTOR_PARAMETERS),
        "critic_trainable_parameters": {
            controller: CRITIC_PARAMETERS for controller in CONTROLLERS
        },
        "total_dynamic_state_per_environment": dict(EXPECTED_DYNAMIC_STATE),
    }
    if comparison.get("capacity_contract") != expected_capacity:
        raise ValueError("revision-3 declared capacity contract differs from exact counts")
    for flag in (
        "same_task_stream_budget_seed_and_ppo_hyperparameters",
        "paired_command_seed_across_still_and_wind",
        "lif_cells_before_any_baseline",
        "controllers_are_all_frozen_multi_connectome_lif",
        "parameter_count_reporting_required",
        "cross_capacity_claims_are_descriptive_only",
    ):
        if comparison.get(flag) is not True:
            raise ValueError(f"revision-3 comparison flag {flag} must be true")
    if comparison.get("matched_gru_or_mlp_jobs_in_this_follow_on") is not False:
        raise ValueError("revision-3 must not contain newly trained GRU/MLP jobs")
    expected_base = {
        "config": "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json",
        "config_sha256": "bf7369d74d119784eb6a932cbeb57734fa93b7fdda5457e3e46decfb6f140c44",
        "completed_queue": "runs/crazyflie_command_optic_wind_seed0_1m/queue.json",
        "completed_queue_sha256": "68f4fee83d659edb439691aa79c5f4b6decc610c1e35e869b1caddd04cf5bddc",
        "completed_report": "docs/crazyflie_command_optic_wind_report.md",
        "completed_report_sha256": "7988eabbb82c710dda724f36bfb19852439cafd984562b997ef238d41b7a6fed",
        "required_queue_status": "completed",
        "required_job_count": 14,
        "required_completed_jobs": 14,
        "relation": "additive_follow_on_no_v2_mutation",
    }
    if config.get("base_matrix_identity") != expected_base:
        raise ValueError("revision-3 frozen completed-v2 identity differs")
    expected_paging = {
        "queue": "runs/crazyflie_command_seed0_500k/queue.json",
        "queue_sha256": "648ad8b8476a1b4873e23eedacefeb803e3d7887d73d171e3e1512245ef5b99c",
        "required_event": "hard_resource_gate",
        "required_sustained_paging": True,
        "disposition": "v3_max_parallel_fixed_to_one",
    }
    if config.get("prior_parallel_paging_evidence") != expected_paging:
        raise ValueError("revision-3 historical paging identity differs")
    output_root = _resolve(str(config["output_root"]))
    if output_root.name != "crazyflie_command_combinations_seed0_1m":
        raise ValueError("reporter accepts only the separate revision-3 output root")
    if not config_path.is_file():
        raise ValueError("revision-3 config path does not exist")
    return output_root


def _validate_config_dependencies(
    config: Mapping[str, Any], snapshots: dict[Path, str]
) -> None:
    interpreter = Path(str(config.get("isaac_python", ""))).expanduser().resolve()
    if (
        interpreter
        != Path("/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python")
        or not interpreter.is_file()
        or not os.access(interpreter, os.X_OK)
    ):
        raise ValueError("revision-3 Isaac Python identity is missing or changed")
    connectomes = _mapping(config.get("connectomes"), "config.connectomes")
    if set(connectomes) != {
        "leg_manifest", "leg_manifest_sha256", "wing_manifest",
        "wing_manifest_sha256", "optic_manifest", "optic_manifest_sha256",
    }:
        raise ValueError("revision-3 connectome config fields differ")
    for core in ("leg", "wing", "optic"):
        expected = connectomes.get(f"{core}_manifest_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"revision-3 {core} manifest hash is invalid")
        _record_hash(snapshots, _resolve(str(connectomes[f"{core}_manifest"])), expected)
    rewire = _mapping(config.get("rewire"), "config.rewire")
    if rewire != {
        "seed": 20260916,
        "manifest": "configs/experiments/crazyflie_rewire_seed_20260916.json",
        "manifest_sha256": "6c2a10b879d22741b0d17110d052953adc4f081e9ecac636fabdf0ba68d57b14",
    }:
        raise ValueError("revision-3 inherited rewire identity differs")
    _record_hash(
        snapshots,
        _resolve(str(rewire["manifest"])),
        str(rewire["manifest_sha256"]),
    )
    paging = _mapping(
        config.get("prior_parallel_paging_evidence"),
        "config.prior_parallel_paging_evidence",
    )
    paging_path = _resolve(str(paging["queue"]))
    _record_hash(snapshots, paging_path, str(paging["queue_sha256"]))
    paging_queue = _read_json(paging_path)
    matches = [
        event for event in paging_queue.get("events", [])
        if isinstance(event, Mapping)
        and event.get("event") == "hard_resource_gate"
        and isinstance(event.get("sample"), Mapping)
        and event["sample"].get("sustained_paging") is True
    ]
    if len(matches) != 1:
        raise ValueError("historical queue lacks its unique sustained-paging receipt")


def _validate_controller_report(controller: str, report: Mapping[str, Any]) -> None:
    labels = list(COMPONENTS[controller])
    if report.get("controller_kind") != controller:
        raise ValueError(f"{controller} controller-report kind differs")
    if report.get("core_labels") != labels:
        raise ValueError(f"{controller} core ordering differs")
    if report.get("fusion_contract") != FUSION[controller]:
        raise ValueError(f"{controller} fusion contract differs")
    ordered_fields = (
        "per_core_checksums", "connectome_manifests", "connectome_checksums",
        "connectome_manifest_fingerprints", "per_core_population_indices",
    )
    for name in ordered_fields:
        value = _mapping(report.get(name), f"{controller}.{name}")
        # JSON evidence is intentionally written with ``sort_keys=True``.  The
        # ordered ``core_labels`` list is authoritative; mapping key order is
        # not an identity field after a persistence round trip.
        if set(value) != set(labels):
            raise ValueError(f"{controller} {name} does not cover every declared core")
    for label, checksum in _mapping(
        report["per_core_checksums"], f"{controller}.per_core_checksums"
    ).items():
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"{controller} has invalid {label} frozen-core checksum")
    aggregate = report.get("core_checksum")
    if not isinstance(aggregate, str) or len(aggregate) != 64:
        raise ValueError(f"{controller} lacks an aggregate frozen-core checksum")
    # This is the exact construction used by controller_core_checksum for a
    # multi-connectome policy.  It binds every named per-core checksum and the
    # fusion identity into the value checked before and after PPO.
    expected_aggregate = _canonical_sha256({
        "composition": FUSION[controller],
        "cores": dict(report["per_core_checksums"]),
    })
    if aggregate != expected_aggregate:
        raise ValueError(f"{controller} aggregate checksum does not bind its per-core checksums")
    actor = _integer(report.get("actor_trainable_parameters"), f"{controller} actor", 1)
    critic = _integer(report.get("critic_trainable_parameters"), f"{controller} critic", 1)
    total = _integer(report.get("total_trainable_parameters"), f"{controller} total", 1)
    frozen = _integer(report.get("frozen_parameters"), f"{controller} frozen", 1)
    dynamic = _integer(
        report.get("total_dynamic_state_per_environment"), f"{controller} dynamic state", 1
    )
    if actor != EXPECTED_ACTOR_PARAMETERS[controller]:
        raise ValueError(f"{controller} actor parameter count drifted")
    if critic != CRITIC_PARAMETERS or total != actor + critic:
        raise ValueError(f"{controller} critic/total parameter count drifted")
    if dynamic != EXPECTED_DYNAMIC_STATE[controller]:
        raise ValueError(f"{controller} dynamic-state size drifted")
    if report.get("parameter_matching_required") is not False:
        raise ValueError(f"{controller} incorrectly claims parameter matching")
    if report.get("actor_parameter_match_passed") is not False:
        raise ValueError(f"{controller} incorrectly claims a single-core parameter match")
    if report.get("frozen_synaptic_weights") != frozen:
        raise ValueError(f"{controller} frozen weight accounting differs")


def _validate_source_snapshot(
    payload: Mapping[str, Any], snapshots: dict[Path, str], label: str
) -> None:
    source_hashes = _mapping(payload.get("source_sha256"), f"{label}.source_sha256")
    if len(source_hashes) < 1:
        raise ValueError(f"{label} source snapshot is empty")
    for raw_path, expected in source_hashes.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{label} source snapshot contains an invalid entry")
        _record_hash(snapshots, _resolve(raw_path), expected)


def _validate_queue(
    queue: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    queue_path: Path,
    output_root: Path,
    snapshots: dict[Path, str],
) -> Sequence[Mapping[str, Any]]:
    if queue.get("schema_version") != 3 or queue.get("kind") != QUEUE_KIND:
        raise ValueError("queue is not the reviewed revision-3 combination queue")
    if queue.get("status") != "completed" or queue.get("dry_run") is not False:
        raise ValueError("revision-3 queue is not globally completed")
    if queue.get("counts") != {"completed": JOB_COUNT}:
        raise ValueError("revision-3 queue completion counts are not exactly 6/6")
    if queue.get("config_file_sha256") != config_sha256:
        raise ValueError("revision-3 queue config file SHA-256 is stale")
    if queue.get("config_identity_sha256") != _canonical_sha256(config):
        raise ValueError("revision-3 queue config identity is stale")
    if queue.get("config") != dict(config):
        raise ValueError("revision-3 queue embedded config differs from selected config")
    if _resolve(str(queue.get("config_path", ""))) != config_path:
        raise ValueError("revision-3 queue config path differs")
    if _resolve(str(queue.get("queue_file", ""))) != queue_path:
        raise ValueError("revision-3 queue file identity differs")
    if _resolve(str(queue.get("output_root", ""))) != output_root:
        raise ValueError("revision-3 queue output root differs")
    exact = {
        "controller_order": list(CONTROLLERS),
        "task_order": list(TASKS),
        "task_protocols": {task: "command_v2" for task in TASKS},
        "job_count": JOB_COUNT,
        "predicted_training_interactions": JOB_COUNT * INTERACTIONS,
        "predicted_evaluation_episodes": JOB_COUNT * EPISODES,
        "lif_cell_count": JOB_COUNT,
        "lif_first": True,
        "maximum_parallel": 1,
    }
    for name, expected in exact.items():
        if queue.get(name) != expected:
            raise ValueError(f"revision-3 queue {name} differs from {expected!r}")
    resources = _mapping(queue.get("resource_limits"), "queue.resource_limits")
    if resources.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB or resources.get(
        "system_ram_percent_exclusive"
    ) != RAM_LIMIT_PERCENT or resources.get("maximum_parallel") != 1:
        raise ValueError("revision-3 resource declaration differs from the hard limits")
    dependencies = (
        ("queue_runner", "queue_runner_sha256", "crazyflie_command_combinations_queue_v3.py"),
        ("v1_helper", "v1_helper_sha256", "crazyflie_command_queue.py"),
        ("v2_helper", "v2_helper_sha256", "crazyflie_command_queue_v2.py"),
        ("trainer", "trainer_sha256", "drone_train.py"),
        ("evaluator", "evaluator_sha256", "crazyflie_command_evaluate.py"),
        ("base_matrix_config", "base_matrix_config_sha256", "crazyflie_command_optic_wind_seed0_1m.json"),
        ("base_completed_queue", "base_completed_queue_sha256", "queue.json"),
        ("base_completed_report", "base_completed_report_sha256", "crazyflie_command_optic_wind_report.md"),
    )
    for path_field, hash_field, basename in dependencies:
        raw = queue.get(path_field)
        expected_hash = queue.get(hash_field)
        if not isinstance(raw, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"queue lacks authenticated {path_field}")
        path = _resolve(raw)
        if path.name != basename:
            raise ValueError(f"queue {path_field} has the wrong file identity")
        _record_hash(snapshots, path, expected_hash)
    reports = _mapping(queue.get("controller_reports"), "queue.controller_reports")
    if set(reports) != set(CONTROLLERS):
        raise ValueError("revision-3 controller-report set differs")
    for controller in CONTROLLERS:
        _validate_controller_report(controller, _mapping(reports[controller], controller))
    if reports["leg_optic_lif"]["actor_trainable_parameters"] != reports[
        "wing_optic_lif"
    ]["actor_trainable_parameters"]:
        raise ValueError("the two-core fair-comparison actors are not capacity-equal")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != JOB_COUNT:
        raise ValueError("revision-3 queue must contain exactly six jobs")
    expected_pairs = [(controller, task) for controller in CONTROLLERS for task in TASKS]
    reference_source: Mapping[str, Any] | None = None
    for index, (job_raw, pair) in enumerate(zip(jobs, expected_pairs, strict=True), start=1):
        job = _mapping(job_raw, f"queue.jobs[{index - 1}]")
        controller, task = pair
        condition = _condition(task)
        exact_job = {
            "id": f"{index:02d}__{controller}__{condition}__seed-0",
            "controller": controller,
            "policy": controller,
            "architecture_class": "lif",
            "combination_components": list(COMPONENTS[controller]),
            "capacity_tier_core_count": len(COMPONENTS[controller]),
            "task": task,
            "evaluation_protocol": "command_v2",
            "seed": 0,
            "command_schedule_seed": 0,
            "paired_task_seed_key": f"{controller}__seed-0",
            "contract_profile": "command_v2",
            "status": "completed",
            "training_status": "completed",
            "evaluation_status": "completed",
            "total_interactions": INTERACTIONS,
            "expected_updates": 250,
        }
        for name, expected in exact_job.items():
            if job.get(name) != expected:
                raise ValueError(f"revision-3 job {index} {name} differs from {expected!r}")
        run_dir = _resolve(str(job.get("run_dir", "")))
        _require_under(run_dir, output_root, f"job {index} run_dir")
        expected_paths = {
            "training_manifest": run_dir / "training_manifest.json",
            "checkpoint": run_dir / "checkpoints/latest.pt",
        }
        for name, expected_path in expected_paths.items():
            actual = _resolve(str(job.get(name, "")))
            _require_under(actual, output_root, f"job {index} {name}")
            if actual != expected_path:
                raise ValueError(f"revision-3 job {index} {name} is misplaced")
        evaluation_output = _resolve(str(job.get("evaluation_output", "")))
        _require_under(evaluation_output, output_root, f"job {index} evaluation_output")
        payload = _mapping(job.get("fingerprint_payload"), f"job {index} fingerprint_payload")
        fingerprint = job.get("expected_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != _canonical_sha256(payload):
            raise ValueError(f"revision-3 job {index} reproduction fingerprint is invalid")
        source = _mapping(payload.get("source_sha256"), f"job {index} source snapshot")
        if reference_source is None:
            reference_source = source
        elif dict(source) != dict(reference_source):
            raise ValueError("revision-3 jobs do not share one source snapshot")
        _validate_source_snapshot(payload, snapshots, f"job {index}")
        resolved = _mapping(payload.get("resolved_config"), f"job {index} resolved_config")
        composition = _mapping(
            resolved.get("lif_connectome_composition"),
            f"job {index} lif_connectome_composition",
        )
        if (
            composition.get("core_labels") != list(COMPONENTS[controller])
            or composition.get("fusion_contract") != FUSION[controller]
            or composition.get("recurrent_cross_core_edges") is not False
            or composition.get("parameter_matching_required") is not False
        ):
            raise ValueError(f"revision-3 job {index} connectome composition differs")
        connectome_identities = _mapping(
            composition.get("connectomes"), f"job {index} connectomes"
        )
        if set(connectome_identities) != set(COMPONENTS[controller]):
            raise ValueError(f"revision-3 job {index} connectome set differs")
        for core in COMPONENTS[controller]:
            identity = _mapping(
                connectome_identities[core], f"job {index} connectome {core}"
            )
            manifest_path = _resolve(str(identity.get("manifest", "")))
            manifest_sha = identity.get("manifest_sha256")
            if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
                raise ValueError(f"revision-3 job {index} {core} manifest hash is invalid")
            _record_hash(snapshots, manifest_path, manifest_sha)
            for child_name in ("neurons_path", "edges_path"):
                child = _mapping(
                    identity.get(child_name),
                    f"job {index} connectome {core}.{child_name}",
                )
                child_path = _resolve(str(child.get("path", "")))
                child_sha = child.get("sha256")
                if not isinstance(child_sha, str) or len(child_sha) != 64:
                    raise ValueError(
                        f"revision-3 job {index} {core} {child_name} hash is invalid"
                    )
                _record_hash(snapshots, child_path, child_sha)
        evaluation_manifest = _mapping(
            job.get("evaluation_manifest"), f"job {index} evaluation_manifest"
        )
        manifest_id = evaluation_manifest.get("manifest_id")
        unsigned = {key: value for key, value in evaluation_manifest.items() if key != "manifest_id"}
        if (
            not isinstance(manifest_id, str)
            or manifest_id != _canonical_sha256(unsigned)
            or job.get("evaluation_manifest_id") != manifest_id
            or evaluation_manifest.get("task") != task
            or evaluation_manifest.get("protocol") != "command_v2"
        ):
            raise ValueError(f"revision-3 job {index} evaluation-manifest identity is invalid")
        report = _mapping(reports[controller], f"controller report {controller}")
        identities = {
            "controller_report_sha256": _canonical_sha256(report),
            "expected_core_checksum": report["core_checksum"],
            "expected_per_core_checksums": report["per_core_checksums"],
            "expected_connectome_manifests": report["connectome_manifests"],
            "expected_connectome_checksums": report["connectome_checksums"],
        }
        for name, expected in identities.items():
            if job.get(name) != expected:
                raise ValueError(f"revision-3 job {index} {name} differs from controller report")
    return jobs


def _validate_memory_gate(value: Any, label: str) -> dict[str, Any]:
    gate = dict(_mapping(value, label))
    common._validate_memory_gate(gate, label)
    if gate.get("device_gpu_telemetry_complete") is not True:
        raise ValueError(f"{label} lacks complete device GPU telemetry")
    if gate.get("sustained_paging_detected") is not False:
        raise ValueError(f"{label} detected sustained paging")
    if gate.get("failures") != []:
        raise ValueError(f"{label} contains recorded failures")
    limits = _mapping(gate.get("limits"), f"{label}.limits")
    if limits.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB or limits.get(
        "system_ram_percent_exclusive"
    ) != RAM_LIMIT_PERCENT:
        raise ValueError(f"{label} limit declaration differs")
    for name in (
        "max_device_gpu_used_mib", "max_process_rss_mib",
        "max_system_ram_percent", "max_torch_allocated_mib",
        "max_torch_reserved_mib",
    ):
        if _finite(gate.get(name), f"{label}.{name}") < 0.0:
            raise ValueError(f"{label}.{name} is negative")
    return gate


def _validate_inference_latency(value: Any) -> dict[str, Any]:
    latency = dict(_mapping(value, "evaluation.inference_latency"))
    exact = {
        "schema_version": 1,
        "source": "exact_policy_act_calls_used_by_evaluation_control_path",
        "call_site": (
            "policy.act(normalized_observation, recurrent_state, deterministic=True)"
        ),
        "clock": "time.perf_counter_ns_monotonic",
        "device": "cuda:0",
        "measurement_unit": "milliseconds_per_vectorized_policy_call",
        "batch_size": EPISODES,
        "cuda_synchronized_before_and_after_call": True,
        "activity_recorder_hooks_in_scope": True,
        "action_producing_call_count": STEPS,
        "warmup_calls_excluded_from_statistics": 10,
        "sample_count": STEPS - 10,
    }
    for name, expected in exact.items():
        if latency.get(name) != expected:
            raise ValueError(f"inference latency {name} differs from {expected!r}")
    justification = latency.get("warmup_exclusion_justification")
    if not isinstance(justification, str) or not justification:
        raise ValueError("inference latency lacks its warmup justification")
    numeric_names = (
        "total_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
        "all_action_producing_calls_total_ms",
    )
    numeric = {name: _finite(latency.get(name), f"inference latency {name}") for name in numeric_names}
    if any(value < 0.0 for value in numeric.values()):
        raise ValueError("inference latency contains a negative duration")
    if not math.isclose(
        numeric["total_ms"],
        numeric["mean_ms"] * int(latency["sample_count"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise ValueError("inference latency mean/total does not recompute")
    if not (
        numeric["p50_ms"] <= numeric["p95_ms"]
        <= numeric["p99_ms"] <= numeric["max_ms"]
    ):
        raise ValueError("inference latency percentiles are inconsistent")
    if numeric["all_action_producing_calls_total_ms"] < numeric["total_ms"]:
        raise ValueError("inference latency all-call total excludes measured steady-state calls")
    return latency


def _validate_activity_provenance(
    activity: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
) -> None:
    controller = str(job["controller"])
    labels = COMPONENTS[controller]
    common._validate_detailed_activity(activity, controller, require_unit_detail=True)
    if activity.get("kind") != "sampled_lif_spikes":
        raise ValueError("combination activity is not sampled LIF spikes")
    provenance = _mapping(activity.get("role_provenance"), "activity.role_provenance")
    if set(provenance) != set(labels):
        raise ValueError("activity provenance does not cover every core")
    try:
        identities = job["fingerprint_payload"]["resolved_config"][
            "lif_connectome_composition"
        ]["connectomes"]
    except (KeyError, TypeError) as exc:
        raise ValueError("job fingerprint lacks multi-core composition") from exc
    if not isinstance(identities, Mapping) or set(identities) != set(labels):
        raise ValueError("fingerprint multi-core composition does not cover every core")
    for label in labels:
        observed = _mapping(provenance[label], f"activity provenance {label}")
        identity = _mapping(identities[label], f"fingerprint connectome {label}")
        neurons = _mapping(identity.get("neurons_path"), f"fingerprint {label}.neurons_path")
        expected = {
            "manifest": identity.get("manifest"),
            "manifest_sha256": identity.get("manifest_sha256"),
            "neurons": neurons.get("path"),
            "neurons_sha256": neurons.get("sha256"),
        }
        for name, value in expected.items():
            if observed.get(name) != value:
                raise ValueError(f"activity provenance {label}.{name} differs from fingerprint")
        _integer(observed.get("neuron_count"), f"activity provenance {label}.neuron_count", 1)
    per_unit = activity["per_unit"]
    actual_prefixes = {str(unit["id"]).split(":", 1)[0] for unit in per_unit}
    if actual_prefixes != set(labels):
        raise ValueError("activity per-neuron evidence does not include every declared core")
    for unit in per_unit:
        identifier_prefix = str(unit["id"]).split(":", 1)[0]
        role_prefix = str(unit["role"]).split(":", 1)[0]
        if identifier_prefix != role_prefix or role_prefix not in labels:
            raise ValueError("activity neuron ID/role core provenance differs")
    roles = _mapping(activity["overall"].get("roles"), "activity.overall.roles")
    expected_roles: set[str] = set()
    for label in labels:
        expected_roles.update(EXPECTED_ROLES[label])
    if set(roles) != expected_roles:
        raise ValueError("activity does not contain the exact biological role groups")
    grouped: dict[str, list[Mapping[str, Any]]] = {role: [] for role in expected_roles}
    for raw in per_unit:
        grouped[str(raw["role"])].append(raw)
    for role, units in grouped.items():
        summary = _mapping(roles[role], f"activity role {role}")
        if summary.get("unit_count") != len(units) or not units:
            raise ValueError(f"activity role {role} unit count does not recompute")
        mean_activity = sum(float(unit["mean_absolute_activity"]) for unit in units) / len(units)
        active_fraction = sum(float(unit["active_fraction"]) for unit in units) / len(units)
        if not math.isclose(
            float(summary["mean_absolute_activity_per_unit"]), mean_activity,
            rel_tol=0.0, abs_tol=1.0e-12,
        ) or not math.isclose(
            float(summary["active_fraction_per_unit"]), active_fraction,
            rel_tol=0.0, abs_tol=1.0e-12,
        ):
            raise ValueError(f"activity role {role} summary does not recompute")
    overall = activity["overall"]
    count = len(per_unit)
    expected_mean = sum(float(unit["mean_absolute_activity"]) for unit in per_unit) / count
    expected_active = sum(float(unit["active_fraction"]) for unit in per_unit) / count
    expected_rms = math.sqrt(sum(float(unit["rms_activity"]) ** 2 for unit in per_unit) / count)
    for name, expected in (
        ("mean_absolute_activity_per_unit", expected_mean),
        ("active_fraction_per_unit", expected_active),
        ("rms_activity_per_unit", expected_rms),
    ):
        if not math.isclose(float(overall[name]), expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"activity overall {name} does not recompute")


def _validate_summary_counts(summary: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> None:
    for name in (
        "survived_full_horizon_count", "termination_count",
        "truncation_count", "invalid_episode_count",
    ):
        _integer(summary.get(name), f"evaluation summary {name}")
    survived = sum(
        int(episode.get("alive_steps") == STEPS and episode.get("invalid_steps") == 0)
        for episode in episodes
    )
    terminated = sum(bool(episode.get("terminated")) for episode in episodes)
    truncated = sum(bool(episode.get("truncated")) for episode in episodes)
    invalid = sum(int(_integer(episode.get("invalid_steps"), "episode.invalid_steps") > 0) for episode in episodes)
    if summary.get("survived_full_horizon_count") != survived:
        raise ValueError("survived-full-horizon count does not recompute")
    if summary.get("termination_count") != terminated:
        raise ValueError("termination count does not recompute")
    if summary.get("truncation_count") != truncated:
        raise ValueError("truncation count does not recompute")
    if summary.get("invalid_episode_count") != invalid:
        raise ValueError("invalid-episode count does not recompute")
    cause_counts: dict[str, int] = {}
    for episode in episodes:
        cause = str(_integer(episode.get("failure_cause"), "episode.failure_cause"))
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
    if summary.get("failure_cause_counts") != cause_counts:
        raise ValueError("failure-cause counts do not recompute")
    reward_mean = sum(_finite(episode.get("reward_total"), "episode.reward_total") for episode in episodes) / EPISODES
    if not common._float32_episode_mean_matches(
        _finite(summary.get("reward_total_mean"), "summary.reward_total_mean"), reward_mean
    ):
        raise ValueError("evaluation reward total mean does not recompute")


def _validate_cell(
    job: Mapping[str, Any],
    *,
    controller_report: Mapping[str, Any],
    output_root: Path,
) -> CombinationCell:
    controller = str(job["controller"])
    task = str(job["task"])
    cell = CombinationCell(
        controller=controller,
        policy=str(job["policy"]),
        task=task,
        condition=_condition(task),
        job_id=str(job["id"]),
        parameters=dict(controller_report),
        components=COMPONENTS[controller],
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
    manifest = _read_json(manifest_path)
    evaluation = _read_json(evaluation_path)
    input_hashes: dict[Path, str] = {}
    manifest_sha = _record_hash(input_hashes, manifest_path)
    checkpoint_sha = _record_hash(input_hashes, checkpoint_path)
    evaluation_sha = _record_hash(input_hashes, evaluation_path)
    del manifest_sha, evaluation_sha
    num_envs = _integer(manifest.get("num_envs"), "training num_envs", 1)
    horizon = _integer(manifest.get("horizon"), "training horizon", 1)
    expected_updates = INTERACTIONS // (num_envs * horizon)
    exact_manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": task,
        "contract_profile": "command_v2",
        "controller": controller,
        "seed": 0,
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
        raise ValueError(f"{cell.job_id} resolved training config differs from fingerprint")
    if _canonical_sha256(manifest.get("controller_report")) != _canonical_sha256(controller_report):
        raise ValueError(f"{cell.job_id} training controller report differs from queue")
    train_memory = _validate_memory_gate(manifest.get("memory_gate"), "training memory gate")
    wall_time = _finite(manifest.get("training_wall_time_s"), "training_wall_time_s")
    if wall_time <= 0.0:
        raise ValueError("training_wall_time_s must be positive")
    schedule = _mapping(manifest.get("command_schedule"), "training command_schedule")
    schedule_state = _mapping(schedule.get("state"), "training command_schedule.state")
    if schedule.get("command_training_contract_sha256") != job[
        "command_training_contract_sha256"
    ] or schedule_state.get("training_interactions") != INTERACTIONS:
        raise ValueError(f"{cell.job_id} command schedule did not finish the 1M contract")
    history, history_hashes = common._load_history(
        _mapping(manifest.get("history_reference"), "training history_reference"),
        checkpoint=checkpoint_path,
        expected_updates=expected_updates,
        interactions_per_update=num_envs * horizon,
        total_interactions=INTERACTIONS,
    )
    input_hashes.update(history_hashes)
    aggregate = controller_report["core_checksum"]
    if manifest.get("core_checksum_before") != aggregate or manifest.get(
        "core_checksum_after"
    ) != aggregate:
        raise ValueError(f"{cell.job_id} aggregate frozen-core checksum changed")
    expected_per_core = dict(controller_report["per_core_checksums"])
    if manifest.get("per_core_checksums_before") != expected_per_core:
        raise ValueError(f"{cell.job_id} per-core checksums before PPO differ")
    if manifest.get("per_core_checksums_after") != expected_per_core:
        raise ValueError(f"{cell.job_id} per-core checksums after PPO changed")

    exact_evaluation = {
        "schema_version": 1,
        "analysis_kind": ANALYSIS_KIND,
        "status": "PASS",
        "task": task,
        "controller": controller,
        "episodes_requested": EPISODES,
        "episodes_evaluated": EPISODES,
        "steps_per_episode": STEPS,
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
        raise ValueError(f"{cell.job_id} evaluation does not contain exactly 16 episodes")
    episodes = list(episodes_raw)
    checkpoint = _mapping(evaluation.get("checkpoint"), "evaluation checkpoint")
    checkpoint_exact = {
        "sha256": checkpoint_sha,
        "training_seed": 0,
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
        evaluation.get("protocol") != evaluation_manifest["evaluation_protocol"]
        or evaluation.get("protocol_sha256")
        != evaluation_manifest["evaluation_protocol_sha256"]
    ):
        raise ValueError(f"{cell.job_id} evaluation protocol identity differs")
    eval_memory = _validate_memory_gate(evaluation.get("memory_gate"), "evaluation memory gate")
    integrity = _mapping(evaluation.get("integrity"), "evaluation.integrity")
    for name in (
        "task_manifest_matched", "evaluation_manifest_matched", "source_set_matched",
        "checkpoint_completed_budget", "all_actions_finite_and_bounded",
        "activity_from_actual_forward", "physical_wind_telemetry_passed",
        "physical_wind_uses_terminal_actual_interval",
        "inference_latency_from_actual_forward",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"{cell.job_id} integrity check {name} did not pass")
    summary = _mapping(evaluation.get("summary"), "evaluation.summary")
    common._validate_v2_physical_wind(summary.get("physical_wind"), condition=cell.condition)
    score = _mapping(summary.get("score"), "evaluation summary score")
    quality = _mapping(summary.get("control_quality"), "evaluation control quality")
    common._validate_score(score)
    common._validate_quality(quality)
    if score.get("protocol") != evaluation_manifest["evaluation_protocol"] or score.get(
        "protocol_sha256"
    ) != evaluation_manifest["evaluation_protocol_sha256"]:
        raise ValueError(f"{cell.job_id} score protocol differs from evaluation manifest")
    if not math.isclose(
        float(quality["component_scores_0_100"]["command_response"]),
        float(score["component_scores"]["response"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{cell.job_id} quality response differs from score response")
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
    sums = {name: 0.0 for name in common.REWARD_COMPONENTS}
    for index, episode in enumerate(episodes):
        episode_rewards = _mapping(
            episode.get("reward_components"), f"episode {index} reward components"
        )
        if set(episode_rewards) != common.REWARD_COMPONENTS:
            raise ValueError(f"{cell.job_id} episode {index} reward components differ")
        for name, value in episode_rewards.items():
            sums[name] += _finite(value, f"episode {index} reward {name}")
    for name, value in sums.items():
        if not common._float32_episode_mean_matches(reward_components[name], value / EPISODES):
            raise ValueError(f"{cell.job_id} reward-component mean {name} does not recompute")
    _validate_summary_counts(summary, episodes)
    activity = _mapping(evaluation.get("activity"), "evaluation.activity")
    _validate_activity_provenance(activity, job=job)
    inference_latency = _validate_inference_latency(evaluation.get("inference_latency"))
    if _canonical_sha256(evaluation.get("controller_report")) != _canonical_sha256(
        controller_report
    ):
        raise ValueError(f"{cell.job_id} evaluation controller report differs from queue")

    cell.complete = True
    cell.reason = "verified complete"
    cell.training_manifest = manifest
    cell.evaluation = evaluation
    cell.history = history
    cell.score = dict(score)
    cell.quality = dict(quality)
    cell.reward_components = reward_components
    cell.activity = dict(activity)
    cell.core_before = str(manifest["core_checksum_before"])
    cell.core_after = str(manifest["core_checksum_after"])
    cell.training_wall_time_s = wall_time
    cell.train_memory = train_memory
    cell.eval_memory = eval_memory
    cell.inference_latency = inference_latency
    cell.input_hashes = input_hashes
    return cell


def _collect_historical_v2_cells(
    config: Mapping[str, Any], snapshots: dict[Path, str]
) -> list[common.V2Cell]:
    """Revalidate the pinned completed v2 artifacts without scraping its report.

    The config pins the byte-exact completed v2 queue and report.  The report
    file is used only as an immutable preservation receipt; numeric values are
    independently recomputed from the 14 declared training/evaluation
    artifacts by the original v2 cell validator.
    """

    identity = _mapping(config.get("base_matrix_identity"), "base_matrix_identity")
    config_path = _resolve(str(identity["config"]))
    queue_path = _resolve(str(identity["completed_queue"]))
    report_path = _resolve(str(identity["completed_report"]))
    _record_hash(snapshots, config_path, str(identity["config_sha256"]))
    _record_hash(snapshots, queue_path, str(identity["completed_queue_sha256"]))
    _record_hash(snapshots, report_path, str(identity["completed_report_sha256"]))
    v2_config = _read_json(config_path)
    output_root = common._validate_v2_config(v2_config, config_path)
    v2_queue = _read_json(queue_path)
    if (
        v2_queue.get("schema_version") != 2
        or v2_queue.get("kind") != common.V2_QUEUE_KIND
        or v2_queue.get("status") != "completed"
        or v2_queue.get("dry_run") is not False
        or v2_queue.get("job_count") != common.V2_JOB_COUNT
        or v2_queue.get("counts") != {"completed": common.V2_JOB_COUNT}
        or v2_queue.get("predicted_training_interactions")
        != common.V2_JOB_COUNT * common.V2_INTERACTIONS
        or v2_queue.get("predicted_evaluation_episodes")
        != common.V2_JOB_COUNT * EPISODES
        or v2_queue.get("controller_order") != list(common.V2_CONTROLLERS)
        or v2_queue.get("task_order") != list(common.V2_TASKS)
        or v2_queue.get("maximum_parallel") != 1
    ):
        raise ValueError("pinned v2 queue no longer declares a completed 14-cell matrix")
    if (
        v2_queue.get("config_file_sha256") != identity["config_sha256"]
        or v2_queue.get("config_identity_sha256") != _canonical_sha256(v2_config)
        or v2_queue.get("config") != v2_config
        or _resolve(str(v2_queue.get("config_path", ""))) != config_path
        or _resolve(str(v2_queue.get("output_root", ""))) != output_root
    ):
        raise ValueError("pinned v2 queue/config identity differs")
    jobs = v2_queue.get("jobs")
    reports = _mapping(v2_queue.get("controller_reports"), "v2 controller_reports")
    if not isinstance(jobs, list) or len(jobs) != common.V2_JOB_COUNT:
        raise ValueError("pinned v2 queue does not contain exactly 14 jobs")
    if set(reports) != set(common.V2_CONTROLLERS):
        raise ValueError("pinned v2 controller-report set differs")
    expected_pairs = [
        (controller, task)
        for controller in common.V2_CONTROLLERS
        for task in common.V2_TASKS
    ]
    if [(job.get("controller"), job.get("task")) for job in jobs] != expected_pairs:
        raise ValueError("pinned v2 job order differs")
    reference_source: Mapping[str, Any] | None = None
    cells: list[common.V2Cell] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, Mapping):
            raise ValueError(f"pinned v2 job {index} is not an object")
        payload = _mapping(job.get("fingerprint_payload"), f"v2 job {index} fingerprint")
        if job.get("expected_fingerprint") != _canonical_sha256(payload):
            raise ValueError(f"pinned v2 job {index} reproduction fingerprint is invalid")
        source = _mapping(payload.get("source_sha256"), f"v2 job {index} source snapshot")
        if reference_source is None:
            reference_source = source
        elif dict(source) != dict(reference_source):
            raise ValueError("pinned v2 jobs do not share one historical source snapshot")
        cell = common._validate_v2_cell(
            job,
            controller_report=_mapping(
                reports[job["controller"]], f"v2 report {job['controller']}"
            ),
            output_root=output_root,
        )
        if not cell.complete:
            raise ValueError(f"pinned v2 job {cell.job_id} failed revalidation: {cell.reason}")
        if cell.training_manifest is None or cell.evaluation is None:
            raise ValueError(f"pinned v2 job {cell.job_id} lacks validated artifacts")
        _validate_memory_gate(
            cell.training_manifest.get("memory_gate"),
            f"pinned v2 {cell.job_id} training memory gate",
        )
        _validate_memory_gate(
            cell.evaluation.get("memory_gate"),
            f"pinned v2 {cell.job_id} evaluation memory gate",
        )
        for path, expected in cell.input_hashes.items():
            _record_hash(snapshots, path, expected)
        cells.append(cell)
    return cells


def collect_report_data(
    config_path: Path = DEFAULT_CONFIG, queue_path: Path | None = None
) -> CombinationReportData:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"revision-3 config is missing: {config_path}")
    config = _read_json(config_path)
    output_root = _validate_config(config, config_path)
    queue_path = (
        queue_path.expanduser().resolve()
        if queue_path is not None
        else (output_root / "queue.json").resolve()
    )
    _require_under(queue_path, output_root, "queue")
    if not queue_path.is_file():
        raise ValueError(f"revision-3 queue is missing: {queue_path}")
    config_sha = _sha256_file(config_path)
    queue_sha = _sha256_file(queue_path)
    snapshots: dict[Path, str] = {
        config_path: config_sha,
        queue_path: queue_sha,
    }
    _validate_config_dependencies(config, snapshots)
    queue = _read_json(queue_path)
    jobs = _validate_queue(
        queue,
        config,
        config_path=config_path,
        config_sha256=config_sha,
        queue_path=queue_path,
        output_root=output_root,
        snapshots=snapshots,
    )
    reports = _mapping(queue["controller_reports"], "queue.controller_reports")
    cells = [
        _validate_cell(
            job,
            controller_report=_mapping(reports[job["controller"]], "controller report"),
            output_root=output_root,
        )
        for job in jobs
    ]
    for cell in cells:
        for path, expected in cell.input_hashes.items():
            _record_hash(snapshots, path, expected)
    historical_cells = _collect_historical_v2_cells(config, snapshots)
    data = CombinationReportData(
        config_path=config_path,
        queue_path=queue_path,
        output_root=output_root,
        config_sha256=config_sha,
        queue_sha256=queue_sha,
        cells=cells,
        historical_cells=historical_cells,
        input_hashes=snapshots,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not data.complete:
        raise ValueError("revision-3 report requires all six authenticated cells")
    return data


def _pair_map(data: CombinationReportData) -> dict[tuple[str, str], CombinationCell]:
    return {(cell.controller, cell.condition): cell for cell in data.cells}


def _historical_pair_map(
    data: CombinationReportData,
) -> dict[tuple[str, str], common.V2Cell]:
    return {(cell.controller, cell.condition): cell for cell in data.historical_cells}


def _score(cell: CombinationCell, key: str = "total") -> float:
    value = common._score_value(cell, key)
    if value is None:
        raise ValueError(f"verified cell {cell.job_id} lacks score {key}")
    return float(value)


def _quality(cell: CombinationCell, key: str) -> float:
    value = common._quality_value(cell, key)
    if value is None:
        raise ValueError(f"verified cell {cell.job_id} lacks quality {key}")
    return float(value)


def _raw(cell: CombinationCell, key: str) -> float:
    value = common._raw_value(cell, key)
    if value is None:
        raise ValueError(f"verified cell {cell.job_id} lacks raw metric {key}")
    return float(value)


def _delta(data: CombinationReportData, controller: str, getter: Any) -> float:
    pairs = _pair_map(data)
    return float(getter(pairs[(controller, "wind")])) - float(
        getter(pairs[(controller, "still")])
    )


def _core_activity(cell: CombinationCell, core: str) -> tuple[float, float, int]:
    if cell.activity is None:
        raise ValueError("verified cell lacks activity")
    units = [
        unit for unit in cell.activity["per_unit"]
        if str(unit["id"]).split(":", 1)[0] == core
    ]
    return (
        sum(float(unit["mean_absolute_activity"]) for unit in units) / len(units),
        sum(float(unit["active_fraction"]) for unit in units) / len(units),
        len(units),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    return common._fmt(value, digits)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return common._markdown_table(headers, rows)


def render_markdown(
    data: CombinationReportData, plot_paths: Mapping[str, Path]
) -> str:
    pairs = _pair_map(data)
    historical = _historical_pair_map(data)
    lines = [
        "# Crazyflie frozen-LIF connectome-combination report",
        "",
        "Report state: **COMPLETE — all 6 revision-3 jobs independently verified**",
        "",
        f"Generated (UTC): `{data.generated_at_utc}`  ",
        f"Config SHA-256: `{data.config_sha256}`  ",
        f"Queue snapshot SHA-256: `{data.queue_sha256}`",
        "",
        (
            "Scope is only `Leg+Optic`, `Wing+Optic`, and `Leg+Wing+Optic`, each trained "
            "independently in still air and physical wind for exactly 1,000,000 interactions "
            "and evaluated for 16 × 600 deterministic held-out steps. Revision 3 adds no baseline "
            "training. Its six new cells contribute 96 held-out episodes; immutable revision-2 "
            "results appear below only as separately authenticated historical context."
        ),
        "",
        "## Authenticated combined 20-cell study",
        "",
        (
            "This single table revalidates all 14 immutable revision-2 cells directly from "
            "their queue-declared checkpoints, histories, and held-out JSON artifacts, then "
            "adds the six revision-3 cells. It contains 20 independently trained cells and "
            "320 held-out episodes. The pinned revision-2 report is checked only by SHA-256; "
            "its prose is not scraped for numbers."
        ),
        "",
    ]
    combined_rows = []
    for cell in data.historical_cells:
        if cell.controller == "leg_wing_lif":
            tier = "2-core LIF"
        elif cell.controller in common.V2_LIF_CONTROLLERS:
            tier = "1-core LIF"
        else:
            tier = "matched engineering baseline"
        combined_rows.append((
            "v2 (immutable)", cell.controller, tier, cell.condition,
            cell.parameters.get("actor_trainable_parameters", "N/A"),
            cell.parameters.get("critic_trainable_parameters", "N/A"),
            "N/A (historical evaluator)",
            _fmt(_score(cell)),
            _fmt(_quality(cell, "acceleration_quality")),
            _fmt(_quality(cell, "command_response")),
            _fmt(_quality(cell, "flight_stability")),
            _fmt(_quality(cell, "survival_not_die")),
            EPISODES,
        ))
    for cell in data.cells:
        tier = "2-core LIF" if len(cell.components) == 2 else "3-core LIF (isolated)"
        combined_rows.append((
            "v3 (new)", cell.controller, tier, cell.condition,
            cell.parameters["actor_trainable_parameters"],
            cell.parameters["critic_trainable_parameters"],
            _fmt(cell.inference_latency["mean_ms"], 4),
            _fmt(_score(cell)),
            _fmt(_quality(cell, "acceleration_quality")),
            _fmt(_quality(cell, "command_response")),
            _fmt(_quality(cell, "flight_stability")),
            _fmt(_quality(cell, "survival_not_die")),
            EPISODES,
        ))
    lines.extend(_table(
        ("Evidence", "Controller", "Topology tier", "Condition", "Actor params",
         "Critic params", "Inference mean ms / 16-env call", "Score /100",
         "Acceleration", "Response", "Stability", "Survival", "Episodes"),
        combined_rows,
    ))
    lines.extend((
        "",
        (
            "Rows across different capacity tiers are descriptive only. A higher raw score "
            "does not establish a topology advantage."
        ),
        "",
        "## Exact fair comparison: revision-3 two-core tier",
        "",
        (
            "The new Leg+Optic and Wing+Optic cells have the same reset-invariant command "
            "schedule contract, exactly 9,224 trainable actor parameters, 18,305 critic "
            "parameters, two independent frozen biological cores, and 2,048 dynamic-state "
            "values per environment. They are the exact within-tier architecture comparison "
            "in this follow-on. The immutable revision-2 Leg+Wing rows have the same declared "
            "capacity but used the older reset-dependent command schedule, so they are shown "
            "only as historical descriptive context and are not part of that exact claim."
        ),
        "",
    ))
    two_core_rows = []
    for condition in ("still", "wind"):
        cell = historical[("leg_wing_lif", condition)]
        two_core_rows.append((
            "v2 immutable", cell.controller, condition,
            cell.parameters["actor_trainable_parameters"],
            "N/A (historical evaluator)",
            _fmt(_score(cell)),
            _fmt(_quality(cell, "acceleration_quality")),
            _fmt(_quality(cell, "command_response")),
            _fmt(_quality(cell, "flight_stability")),
            _fmt(_quality(cell, "survival_not_die")),
            _fmt(_raw(cell, "linear_tracking_rmse_m_s")),
        ))
    for controller in CONTROLLERS[:2]:
        for condition in ("still", "wind"):
            cell = pairs[(controller, condition)]
            two_core_rows.append((
                "v3 new", controller, condition,
                cell.parameters["actor_trainable_parameters"],
                _fmt(cell.inference_latency["mean_ms"], 4),
                _fmt(_score(cell)),
                _fmt(_quality(cell, "acceleration_quality")),
                _fmt(_quality(cell, "command_response")),
                _fmt(_quality(cell, "flight_stability")),
                _fmt(_quality(cell, "survival_not_die")),
                _fmt(_raw(cell, "linear_tracking_rmse_m_s")),
            ))
    lines.extend(_table(
        ("Evidence", "Controller", "Condition", "Actor params",
         "Inference mean ms / 16-env call", "Score /100", "Acceleration",
         "Response", "Stability", "Survival", "Linear RMSE m/s"),
        two_core_rows,
    ))
    lines.extend(("", "## Isolated three-core tier", ""))
    triple_rows = []
    for condition in ("still", "wind"):
        cell = pairs[(CONTROLLERS[2], condition)]
        triple_rows.append((
            cell.controller, condition,
            cell.parameters["actor_trainable_parameters"],
            cell.parameters["total_dynamic_state_per_environment"],
            _fmt(_score(cell)),
            _fmt(_quality(cell, "acceleration_quality")),
            _fmt(_quality(cell, "command_response")),
            _fmt(_quality(cell, "flight_stability")),
            _fmt(_quality(cell, "survival_not_die")),
        ))
    lines.extend(_table(
        ("Controller", "Condition", "Actor params", "Dynamic state/env", "Score /100",
         "Acceleration", "Response", "Stability", "Survival"),
        triple_rows,
    ))
    lines.extend((
        "",
        (
            "The three-core actor has 13,672 trainable parameters and no same-capacity peer in "
            "revision 3. Every comparison between it and a two-core actor is descriptive only; "
            "raw score cannot establish a topology advantage."
        ),
        "",
        f"![Score by condition]({plot_paths['score']})",
        "",
        "## Wind effect (wind minus still)",
        "",
    ))
    delta_rows = []
    for controller in CONTROLLERS:
        delta_rows.append((
            controller,
            _fmt(_delta(data, controller, lambda cell: _score(cell))),
            _fmt(_delta(data, controller, lambda cell: _score(cell, "linear_tracking"))),
            _fmt(_delta(data, controller, lambda cell: _raw(cell, "linear_tracking_rmse_m_s"))),
            _fmt(_delta(data, controller, lambda cell: _quality(cell, "acceleration_quality"))),
            _fmt(_delta(data, controller, lambda cell: _quality(cell, "flight_stability"))),
            _fmt(_delta(data, controller, lambda cell: _quality(cell, "survival_not_die"))),
        ))
    lines.extend(_table(
        ("Controller", "Δ score", "Δ tracking", "Δ linear RMSE", "Δ acceleration",
         "Δ stability", "Δ survival"),
        delta_rows,
    ))
    lines.extend((
        "",
        "Negative score deltas mean lower performance in wind; positive RMSE deltas mean more error.",
        "",
        f"![Wind deltas]({plot_paths['wind_delta']})",
        "",
        "## Capacity, efficiency, time, and memory",
        "",
    ))
    capacity_rows = []
    for cell in data.cells:
        actor = int(cell.parameters["actor_trainable_parameters"])
        capacity_rows.append((
            cell.controller, cell.condition, len(cell.components), actor,
            cell.parameters["critic_trainable_parameters"],
            cell.parameters["total_dynamic_state_per_environment"],
            _fmt(_score(cell) / (actor / 1000.0)),
            _fmt(cell.training_wall_time_s, 1),
            _fmt(cell.inference_latency["mean_ms"], 4),
            _fmt(cell.inference_latency["p95_ms"], 4),
            _fmt(cell.train_memory["max_process_rss_mib"], 1),
            _fmt(cell.train_memory["max_device_gpu_used_mib"], 1),
            _fmt(cell.eval_memory["max_process_rss_mib"], 1),
            _fmt(cell.eval_memory["max_device_gpu_used_mib"], 1),
        ))
    lines.extend(_table(
        ("Controller", "Condition", "Cores", "Actor params", "Critic params",
         "Dynamic state/env", "Score/1k actor params", "Train wall s",
         "Inference mean ms / 16-env call", "Inference p95 ms / 16-env call",
         "Train peak RAM MiB", "Train peak VRAM MiB",
         "Eval peak RAM MiB", "Eval peak VRAM MiB"),
        capacity_rows,
    ))
    lines.extend((
        "",
        (
            "Revision-3 latency wraps the exact action-producing `policy.act` calls with CUDA "
            "synchronization before and after each call; it reports milliseconds per 16-environment "
            "vectorized call after ten real-call warmups. The immutable revision-2 evaluator did "
            "not instrument this metric, so only its historical rows are N/A. No latency value is "
            "inferred from training wall time. Score-per-parameter is descriptive and does not "
            "equal control quality or causal biological efficiency."
        ),
        "",
        f"![Capacity efficiency]({plot_paths['efficiency']})",
        "",
        "## Frozen-core integrity and provenance",
        "",
    ))
    integrity_rows = []
    for cell in data.cells:
        assert cell.training_manifest is not None
        per_core = cell.parameters["per_core_checksums"]
        before = cell.training_manifest["per_core_checksums_before"]
        after = cell.training_manifest["per_core_checksums_after"]
        for core in cell.components:
            integrity_rows.append((
                cell.controller, cell.condition, core, per_core[core],
                before[core],
                after[core],
                "PASS",
                cell.parameters["connectome_manifests"][core],
                cell.parameters["connectome_checksums"][core],
            ))
    lines.extend(_table(
        ("Controller", "Condition", "Core", "Declared core SHA-256", "Before SHA-256",
         "After SHA-256", "Training freeze gate", "Manifest", "Connectome SHA-256"),
        integrity_rows,
    ))
    lines.extend((
        "",
        (
            "The trainer records the aggregate checksum before and after PPO. The reporter "
            "independently recomputes that aggregate from the ordered `core_labels`, fusion "
            "contract, and complete per-core checksum map; therefore an unchanged aggregate "
            "authenticates every displayed per-core before/after identity."
        ),
    ))
    lines.extend(("", "## Measured per-core and biological-role activity", ""))
    core_rows = []
    for cell in data.cells:
        for core in cell.components:
            mean, active, count = _core_activity(cell, core)
            core_rows.append((
                cell.controller, cell.condition, core, count, _fmt(mean), _fmt(active)
            ))
    lines.extend(_table(
        ("Controller", "Condition", "Core", "Neurons", "Mean |spike|", "Active fraction"),
        core_rows,
    ))
    role_rows = []
    for cell in data.cells:
        assert cell.activity is not None
        for role, summary in cell.activity["overall"]["roles"].items():
            role_rows.append((
                cell.controller, cell.condition, role, summary["unit_count"],
                _fmt(summary["mean_absolute_activity_per_unit"]),
                _fmt(summary["active_fraction_per_unit"]),
            ))
    lines.extend(("",))
    lines.extend(_table(
        ("Controller", "Condition", "Authenticated role", "Neurons",
         "Mean |spike|", "Active fraction"),
        role_rows,
    ))
    lines.extend((
        "",
        "Activity is observed correlation from the exact action-producing forward pass, not causal proof.",
        "",
        f"![Per-core activity]({plot_paths['activity']})",
        "",
        "## Training curves",
        "",
        f"![Training reward]({plot_paths['reward']})",
        "",
        f"![Training loss]({plot_paths['loss']})",
        "",
        "## Immutable per-job evidence",
        "",
    ))
    evidence_rows = []
    for cell in data.cells:
        assert cell.training_manifest is not None
        history_reference = cell.training_manifest["history_reference"]
        evidence_rows.append((
            cell.job_id, "PASS", cell.training_manifest["checkpoint_sha256"],
            history_reference["history_sha256"], len(cell.history),
            _fmt(_score(cell)),
        ))
    lines.extend(_table(
        ("Queue job", "Validation", "Checkpoint SHA-256", "History SHA-256",
         "History rows", "Held-out score"),
        evidence_rows,
    ))
    lines.extend((
        "",
        "## Input boundary",
        "",
        (
            "The reporter read only the selected revision-3 config and queue, their authenticated "
            "source/dependency snapshots, and artifact paths declared by the six jobs. It did not "
            "read results from revision 2 for numeric comparison and did not modify any prior run."
        ),
        "",
    ))
    return "\n".join(lines)


def _placeholder(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def render_plots(data: CombinationReportData, destinations: Mapping[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = _pair_map(data)
    colors = {
        "leg_optic_lif": "#2a6f97",
        "wing_optic_lif": "#2a9d8f",
        "leg_wing_optic_lif": "#8e5ea2",
    }
    x = np.arange(len(CONTROLLERS), dtype=float)

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for offset, condition in ((-0.19, "still"), (0.19, "wind")):
        axis.bar(
            x + offset,
            [_score(pairs[(controller, condition)]) for controller in CONTROLLERS],
            0.36,
            label=condition,
            color=[colors[controller] for controller in CONTROLLERS],
            alpha=1.0 if condition == "still" else 0.55,
            edgecolor="black" if condition == "wind" else "none",
            linewidth=0.5,
        )
    axis.axvline(1.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.text(0.5, 102, "equal two-core tier", ha="center", fontsize=9)
    axis.text(2.0, 102, "three-core (isolated)", ha="center", fontsize=9)
    axis.set_xticks(x, CONTROLLERS, rotation=15, ha="right")
    axis.set_ylim(0, 108)
    axis.set_ylabel("Held-out score (0–100)")
    axis.set_title("Revision-3 command score: still vs physical wind")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(destinations["score"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    metrics = (
        ("Total", lambda cell: _score(cell)),
        ("Tracking", lambda cell: _score(cell, "linear_tracking")),
        ("Acceleration", lambda cell: _quality(cell, "acceleration_quality")),
        ("Stability", lambda cell: _quality(cell, "flight_stability")),
        ("Survival", lambda cell: _quality(cell, "survival_not_die")),
    )
    width = 0.15
    for metric_index, (label, getter) in enumerate(metrics):
        axis.bar(
            x + (metric_index - 2) * width,
            [_delta(data, controller, getter) for controller in CONTROLLERS],
            width,
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(1.5, color="#555555", linestyle=":", linewidth=1.2)
    axis.set_xticks(x, CONTROLLERS, rotation=15, ha="right")
    axis.set_ylabel("Wind minus still (score points)")
    axis.set_title("Paired-condition wind deltas")
    axis.legend(ncols=5, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(destinations["wind_delta"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    for cell in data.cells:
        axis.plot(
            [row["total_interactions"] for row in cell.history],
            [row["mean_rollout_reward"] for row in cell.history],
            color=colors[cell.controller],
            linestyle="-" if cell.condition == "still" else "--",
            linewidth=1.4,
            label=f"{cell.controller}/{cell.condition}",
        )
    axis.set_xlabel("Training interactions")
    axis.set_ylabel("Mean rollout reward")
    axis.set_title("Revision-3 training reward (solid still; dashed wind)")
    axis.legend(ncols=2, fontsize=7)
    axis.grid(alpha=0.25)
    fig.savefig(destinations["reward"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True, constrained_layout=True)
    for axis, (field, label) in zip(
        axes,
        (("loss", "Total loss"), ("policy_loss", "Policy loss"), ("value_loss", "Value loss")),
        strict=True,
    ):
        for cell in data.cells:
            axis.plot(
                [row["total_interactions"] for row in cell.history],
                [row[field] for row in cell.history],
                color=colors[cell.controller],
                linestyle="-" if cell.condition == "still" else "--",
                linewidth=1.15,
                label=f"{cell.controller}/{cell.condition}",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(ncols=3, fontsize=6.5)
    axes[-1].set_xlabel("Training interactions")
    fig.suptitle("Revision-3 PPO losses (solid still; dashed wind)")
    fig.savefig(destinations["loss"], dpi=170, facecolor="white")
    plt.close(fig)

    labels: list[str] = []
    means: list[float] = []
    active: list[float] = []
    bar_colors: list[str] = []
    hatches: list[str] = []
    for cell in data.cells:
        for core in cell.components:
            mean, fraction, _ = _core_activity(cell, core)
            labels.append(f"{cell.controller}\n{cell.condition}/{core}")
            means.append(mean)
            active.append(fraction)
            bar_colors.append(colors[cell.controller])
            hatches.append("" if cell.condition == "still" else "//")
    activity_x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    bars0 = axes[0].bar(activity_x, means, color=bar_colors)
    bars1 = axes[1].bar(activity_x, active, color=bar_colors)
    for bars in (bars0, bars1):
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
    axes[0].set_ylabel("Mean |spike| per neuron")
    axes[1].set_ylabel("Active fraction per neuron")
    axes[1].set_xticks(activity_x, labels, rotation=55, ha="right", fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Authenticated per-core action-producing LIF activity (hatched = wind)")
    fig.savefig(destinations["activity"], dpi=170, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    markers = {"still": "o", "wind": "^"}
    for cell in data.cells:
        actor = int(cell.parameters["actor_trainable_parameters"])
        efficiency = _score(cell) / (actor / 1000.0)
        axis.scatter(
            actor,
            efficiency,
            s=90,
            marker=markers[cell.condition],
            color=colors[cell.controller],
            edgecolor="black",
            linewidth=0.5,
            label=f"{cell.controller}/{cell.condition}",
        )
        axis.annotate(
            f"{cell.parameters['total_dynamic_state_per_environment']} state/env",
            (actor, efficiency), xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    axis.axvline(11_000, color="#555555", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Trainable actor parameters")
    axis.set_ylabel("Held-out score per 1,000 actor parameters")
    axis.set_title("Descriptive capacity efficiency (three-core tier isolated by divider)")
    axis.grid(alpha=0.25)
    axis.legend(ncols=2, fontsize=7)
    fig.savefig(destinations["efficiency"], dpi=170, facecolor="white")
    plt.close(fig)


def _revalidate_inputs(snapshots: Mapping[Path, str]) -> None:
    for path, expected in snapshots.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"authenticated input changed during report generation: {path}")


def _publish_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report evidence: {destination}")
    os.link(source, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_report_bundle(data: CombinationReportData, report_path: Path) -> dict[str, Any]:
    if not data.complete:
        raise ValueError("refusing to publish an incomplete revision-3 report")
    report_path = report_path.expanduser().resolve()
    plot_paths = {name: data.output_root / filename for name, filename in PLOT_NAMES.items()}
    destinations = [report_path, *plot_paths.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing revision-3 report evidence: "
            + ", ".join(map(str, existing))
        )
    data.output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".crazyflie-command-v3-report-", dir=data.output_root))
    report_temp: Path | None = None
    try:
        staged_plots = {name: staging / filename for name, filename in PLOT_NAMES.items()}
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
        for name in ("score", "wind_delta", "reward", "loss", "activity", "efficiency"):
            _publish_no_overwrite(staged_plots[name], plot_paths[name])
        _publish_no_overwrite(report_temp, report_path)
        return {
            "status": "COMPLETE",
            "schema_version": 3,
            "new_verified_jobs": JOB_COUNT,
            "historical_verified_jobs": common.V2_JOB_COUNT,
            "combined_verified_cells": JOB_COUNT + common.V2_JOB_COUNT,
            "new_training_interactions": JOB_COUNT * INTERACTIONS,
            "new_evaluation_episodes": JOB_COUNT * EPISODES,
            "combined_evaluation_episodes": (
                JOB_COUNT + common.V2_JOB_COUNT
            ) * EPISODES,
            "report": str(report_path),
            "report_sha256": _sha256_file(report_path),
            "plots": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in plot_paths.items()
            },
        }
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
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
