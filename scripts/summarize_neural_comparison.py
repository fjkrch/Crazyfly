#!/usr/bin/env python3
"""Authenticate and summarize the six-controller Crazyflie comparison.

The reporter is CPU-only and never starts Isaac.  It requires the exact
six-controller x three-task, seed-0, 500k-interaction design and the official
16-episode matching-task main protocol.  A complete report additionally
requires the additive held-out neural-activity rerun produced by
``crazyflie_neural_activity.py``.  Missing or inconsistent evidence fails
closed.  ``--allow_incomplete`` is only for a visibly incomplete development
report: it never selects a winner or ranks partial evidence.
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
import sys
from typing import Any, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "source" / "g1_fly_control"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from crazyflie_neural_activity import (  # noqa: E402
    LEG_MANIFEST,
    WING_MANIFEST,
    summarize_engineering_units,
    summarize_lif_counts,
)
from drone_bootstrap import canonical_sha256, sha256_file, source_hashes  # noqa: E402
from drone_evaluation_protocol import SCENARIOS, load_protocol  # noqa: E402
from drone_score_one_seed_comparison import (  # noqa: E402
    score_evaluation,
    _summarize_training_history,
)
from g1_fly_control.crazyflie.checkpoint import (  # noqa: E402
    load_history_reference,
    read_checkpoint,
)


CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "wing_lif",
    "leg_wing_lif",
    "gru_matched",
    "mlp_normal",
)
TASKS = tuple(SCENARIOS)
TRAINING_SEED = 0
INTERACTIONS_PER_JOB = 500_000
EPISODES_PER_CELL = 16
EXPECTED_JOB_COUNT = len(CONTROLLERS) * len(TASKS)
EXPECTED_EVALUATION_EPISODES = EXPECTED_JOB_COUNT * EPISODES_PER_CELL
EXPECTED_TOTAL_INTERACTIONS = EXPECTED_JOB_COUNT * INTERACTIONS_PER_JOB
CONFIG_PATH = ROOT / "configs" / "experiments" / "crazyflie_neural_comparison_seed0_500k.json"
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
LIF_CONTROLLERS = set(CONTROLLERS[:4])

SCORE_CONTRACT = {
    "version": "crazyflie_six_controller_seed0_500k_score_v1",
    "source_formula": "drone_score_one_seed_comparison.score_evaluation",
    "controllers": list(CONTROLLERS),
    "tasks": list(TASKS),
    "seed": TRAINING_SEED,
    "interactions_per_job": INTERACTIONS_PER_JOB,
    "evaluation_protocol": "main",
    "evaluation_seed": 101,
    "episodes_per_matching_task": EPISODES_PER_CELL,
    "weights_points": {
        "safety": 25.0,
        "task_event_completion": 25.0,
        "all_required_events": 15.0,
        "fixed_horizon_censored_latency": 5.0,
        "tracking": 20.0,
        "control_quality": 10.0,
    },
    "winner_basis": (
        "highest unweighted macro mean official score; then highest worst-task score; "
        "then highest macro event-completion rate; exact remaining ties retained"
    ),
    "heldout_environment_reward": (
        "not emitted by the official evaluator and therefore unavailable; training rollout "
        "reward is reported separately and is not substituted into the held-out score"
    ),
    "scientific_scope": "descriptive_single_training_seed_no_seed_level_uncertainty",
}
SCORE_CONTRACT_SHA256 = canonical_sha256(SCORE_CONTRACT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value, sha256(payload).hexdigest()


def _resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact path is missing")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _memory_gate_issues(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} lacks a memory gate"]
    issues: list[str] = []
    if value.get("passed") is not True:
        issues.append(f"{label} memory gate did not pass")
    gpu = value.get("max_device_gpu_used_mib")
    ram = value.get("max_system_ram_percent")
    if (
        isinstance(gpu, bool)
        or not isinstance(gpu, (int, float))
        or not math.isfinite(float(gpu))
        or not float(gpu) < GPU_LIMIT_MIB
    ):
        issues.append(f"{label} GPU evidence is missing/nonfinite/not below {GPU_LIMIT_MIB} MiB")
    if (
        isinstance(ram, bool)
        or not isinstance(ram, (int, float))
        or not math.isfinite(float(ram))
        or not float(ram) < RAM_LIMIT_PERCENT
    ):
        issues.append(f"{label} RAM evidence is missing/nonfinite/not below {RAM_LIMIT_PERCENT}%")
    limits = value.get("limits")
    if not isinstance(limits, Mapping) or (
        limits.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB
        or limits.get("system_ram_percent_exclusive") != RAM_LIMIT_PERCENT
    ):
        issues.append(f"{label} memory limits differ from the frozen 6.8-GiB/90% contract")
    return issues


def _load_config(queue: Mapping[str, Any]) -> tuple[dict[str, Any], Path, str]:
    configured = queue.get("config_path")
    path = _resolve_path(configured) if configured is not None else CONFIG_PATH.resolve()
    if path != CONFIG_PATH.resolve():
        raise ValueError(f"Queue config must be exactly {CONFIG_PATH.resolve()}")
    config, digest = _read_json(path)
    return config, path, digest


def validate_design(queue: Mapping[str, Any], *, queue_path: Path) -> dict[str, Any]:
    """Validate the immutable 18-cell design independently of run completion."""

    errors: list[str] = []
    try:
        config, config_path, config_sha = _load_config(queue)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate comparison config: {exc}") from exc
    if queue.get("schema_version") != 1:
        errors.append("queue schema_version must be 1")
    if config.get("schema_version") != 1:
        errors.append("config schema_version must be 1")
    if config.get("label") != "crazyflie_neural_comparison_seed0_500k":
        errors.append("config label changed")
    expected_top_level_fields = {
        "schema_version",
        "label",
        "output_root",
        "isaac_python",
        "contract_profile",
        "tasks",
        "controllers",
        "seeds",
        "total_interactions_per_job",
        "connectomes",
        "rewire",
        "training",
        "evaluation",
        "analysis",
        "queue",
        "comparison_contract",
    }
    if set(config) != expected_top_level_fields:
        errors.append("config top-level closed schema changed")
    if config.get("output_root") != "runs/crazyflie_neural_comparison_seed0_500k":
        errors.append("config output_root changed")
    if config.get("isaac_python") != (
        "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
    ):
        errors.append("config Isaac interpreter changed")
    if tuple(config.get("controllers", ())) != CONTROLLERS:
        errors.append("config controller order differs from the exact six-controller contract")
    if tuple(config.get("tasks", ())) != TASKS:
        errors.append("config task order differs from the exact three-task contract")
    if config.get("seeds") != [TRAINING_SEED]:
        errors.append("config must contain seed 0 only")
    if config.get("total_interactions_per_job") != INTERACTIONS_PER_JOB:
        errors.append("config must request exactly 500,000 interactions per job")
    if config.get("contract_profile") != "balanced_v4":
        errors.append("config must use the declared balanced_v4 task contract")
    training = config.get("training")
    expected_training = {
        "num_envs": 40,
        "horizon": 100,
        "microbatch_size": 40,
        "ppo_epochs": 2,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "checkpoint_every_updates": 100,
        "precision": "float32",
        "device": "cuda:0",
    }
    if training != expected_training:
        errors.append("config training hyperparameters differ from the frozen comparison")
    expected_analysis = {
        "neural_activity_replay": True,
        "script": "scripts/crazyflie_neural_activity.py",
        "protocol": "main",
        "episodes_per_matching_task": EPISODES_PER_CELL,
    }
    if config.get("analysis") != expected_analysis:
        errors.append("config neural-activity analysis contract changed")
    expected_queue = {
        "default_max_parallel": 1,
        "allowed_max_parallel": [1, 2],
        "controller_barriers": True,
        "lif_first": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
    }
    if config.get("queue") != expected_queue:
        errors.append("config queue/resource contract changed")
    expected_comparison = {
        "same_task_budget_seed_and_ppo_hyperparameters": True,
        "numerically_matched_single_core_and_baselines": [
            "frozen_lif_original",
            "frozen_lif_degree_rewired",
            "wing_lif",
            "gru_matched",
            "mlp_normal",
        ],
        "wing_extension_parameter_matching_required": False,
        "leg_wing_fusion": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
        "leg_wing_is_not_parameter_matched": True,
    }
    if config.get("comparison_contract") != expected_comparison:
        errors.append("config comparison/parameter-matching contract changed")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        errors.append("config evaluation must be an object")
    else:
        expected_evaluation = {
            "protocol": "main",
            "manifest": "configs/experiments/crazyflie_eval_manifest_v1.json",
            "manifest_sha256": evaluation.get("manifest_sha256"),
            "episodes_per_matching_task": EPISODES_PER_CELL,
            "matching_task_only": True,
            "deterministic_actions": True,
            "device": "cuda:0",
        }
        if dict(evaluation) != expected_evaluation:
            errors.append("config evaluation settings differ from main16 matching-task evaluation")
        try:
            eval_manifest_path = _resolve_path(evaluation.get("manifest"))
            if sha256_file(eval_manifest_path) != evaluation.get("manifest_sha256"):
                errors.append("evaluation manifest file SHA-256 differs from config")
        except (OSError, TypeError, ValueError):
            errors.append("evaluation manifest path is invalid")
    for name, expected_path in (("leg_manifest", LEG_MANIFEST), ("wing_manifest", WING_MANIFEST)):
        declaration = config.get("connectomes")
        if not isinstance(declaration, Mapping):
            errors.append("config connectomes declaration is missing")
            break
        try:
            actual_path = _resolve_path(declaration.get(name))
            if actual_path != expected_path.resolve():
                errors.append(f"{name} path differs from the primary artifact")
            if sha256_file(actual_path) != declaration.get(f"{name}_sha256"):
                errors.append(f"{name} SHA-256 differs from config")
        except (OSError, TypeError, ValueError):
            errors.append(f"{name} declaration is invalid")
    rewire = config.get("rewire")
    if not isinstance(rewire, Mapping) or rewire.get("seed") != 20260916:
        errors.append("rewire seed/declaration is missing")
    else:
        try:
            rewire_path = _resolve_path(rewire.get("manifest"))
            if sha256_file(rewire_path) != rewire.get("manifest_sha256"):
                errors.append("rewire artifact SHA-256 differs from config")
        except (OSError, TypeError, ValueError):
            errors.append("rewire artifact declaration is invalid")
    queue_config_hash = queue.get("config_sha256")
    if queue_config_hash is not None and queue_config_hash not in {
        config_sha,
        canonical_sha256(config),
    }:
        errors.append("queue config SHA-256 does not authenticate the frozen config")
    if queue.get("config_file_sha256", config_sha) != config_sha:
        errors.append("queue config_file_sha256 differs from current frozen config bytes")
    if queue.get("config_identity_sha256", canonical_sha256(config)) != canonical_sha256(config):
        errors.append("queue config_identity_sha256 differs from config canonical identity")
    embedded_config = queue.get("config")
    if embedded_config is not None and embedded_config != config:
        errors.append("queue embedded config differs from current frozen config")
    if queue.get("kind", "crazyflie_neural_comparison_queue_v1") != (
        "crazyflie_neural_comparison_queue_v1"
    ):
        errors.append("queue kind changed")
    if queue.get("label", config["label"]) != config["label"]:
        errors.append("queue label differs from config")
    if tuple(queue.get("controller_order", CONTROLLERS)) != CONTROLLERS:
        errors.append("queue controller order differs from design")
    if tuple(queue.get("task_order", TASKS)) != TASKS:
        errors.append("queue task order differs from design")
    if queue.get("controller_barriers", True) is not True:
        errors.append("queue controller barriers are disabled")
    queue_resource_limits = queue.get("resource_limits")
    if queue_resource_limits is not None and queue_resource_limits != {
        "default_max_parallel": 1,
        "allowed_max_parallel": [1, 2],
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
    }:
        errors.append("queue resource limits differ from config")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        errors.append("queue jobs must be a list")
        jobs = []
    if len(jobs) != EXPECTED_JOB_COUNT or queue.get("job_count", len(jobs)) != EXPECTED_JOB_COUNT:
        errors.append(f"queue must contain exactly {EXPECTED_JOB_COUNT} jobs")
    expected_order = [
        (controller, task, TRAINING_SEED)
        for controller in CONTROLLERS
        for task in TASKS
    ]
    actual_order: list[tuple[Any, Any, Any]] = []
    ids: set[str] = set()
    run_dirs: set[Path] = set()
    artifacts: set[Path] = set()
    current_source_hashes = source_hashes()
    for index, job in enumerate(jobs):
        prefix = f"jobs[{index}]"
        if not isinstance(job, Mapping):
            errors.append(f"{prefix} is not an object")
            continue
        actual_order.append((job.get("controller"), job.get("task"), job.get("seed")))
        identifier = job.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            errors.append(f"{prefix} has a missing/duplicate ID")
        else:
            ids.add(identifier)
        if job.get("total_interactions", INTERACTIONS_PER_JOB) != INTERACTIONS_PER_JOB:
            errors.append(f"{prefix} interaction budget differs from 500,000")
        fingerprint = job.get("expected_fingerprint")
        if not _valid_sha(fingerprint):
            errors.append(f"{prefix} expected fingerprint is invalid")
        fingerprint_payload = job.get("fingerprint_payload")
        if not isinstance(fingerprint_payload, Mapping) or canonical_sha256(dict(fingerprint_payload)) != fingerprint:
            errors.append(f"{prefix} fingerprint payload does not authenticate its fingerprint")
        elif fingerprint_payload.get("source_sha256") != current_source_hashes:
            errors.append(f"{prefix} execution source hashes are stale")
        else:
            resolved = fingerprint_payload.get("resolved_config")
            if not isinstance(resolved, Mapping):
                errors.append(f"{prefix} fingerprint lacks resolved training configuration")
            else:
                resolved_expected = {
                    "task": job.get("task"),
                    "controller": job.get("controller"),
                    "seed": TRAINING_SEED,
                    "total_interactions": INTERACTIONS_PER_JOB,
                    "num_envs": 40,
                    "horizon": 100,
                    "microbatch_size": 40,
                    "ppo_epochs": 2,
                    "learning_rate": 0.0003,
                    "evaluation_protocol": "main",
                    "contract_profile": "balanced_v4",
                }
                for field, expected in resolved_expected.items():
                    if resolved.get(field) != expected:
                        errors.append(
                            f"{prefix} resolved fingerprint {field} differs from design"
                        )
        try:
            run_dir = _resolve_path(job.get("run_dir"))
            checkpoint = _resolve_path(job.get("checkpoint"))
            manifest = _resolve_path(job.get("training_manifest"))
            evaluation_output = _resolve_path(job.get("evaluation_output"))
            activity_output = _resolve_path(job.get("neural_activity_output"))
            if checkpoint.parent.parent != run_dir or manifest.parent != run_dir:
                errors.append(f"{prefix} checkpoint/manifest lies outside its run directory")
            if run_dir in run_dirs:
                errors.append(f"{prefix} duplicates a run directory")
            run_dirs.add(run_dir)
            for artifact in (checkpoint, manifest, evaluation_output, activity_output):
                if artifact in artifacts:
                    errors.append(f"{prefix} duplicates an artifact path")
                artifacts.add(artifact)
        except (OSError, RuntimeError, TypeError, ValueError):
            errors.append(f"{prefix} artifact paths are incomplete or invalid")
    if actual_order != expected_order:
        errors.append("queue jobs are not in exact controller-first/task-second order")
    if queue.get("predicted_training_interactions", EXPECTED_TOTAL_INTERACTIONS) != EXPECTED_TOTAL_INTERACTIONS:
        errors.append("queue predicted training interactions differ from 9,000,000")
    if queue.get("predicted_evaluation_episodes", EXPECTED_EVALUATION_EPISODES) != EXPECTED_EVALUATION_EPISODES:
        errors.append("queue predicted official evaluation episodes differ from 288")
    if queue.get(
        "predicted_neural_activity_replay_episodes", EXPECTED_EVALUATION_EPISODES
    ) != EXPECTED_EVALUATION_EPISODES:
        errors.append("queue predicted neural-activity replay episodes differ from 288")
    if errors:
        raise ValueError("Invalid neural comparison design:\n- " + "\n- ".join(errors))
    return {
        "config": config,
        "config_path": str(config_path),
        "config_file_sha256": config_sha,
        "config_canonical_sha256": canonical_sha256(config),
    }


def _history_rows(checkpoint: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = read_checkpoint(
        checkpoint, map_location="cpu", resolve_external_history=False
    )
    reference = payload.get("history_reference")
    if reference != manifest.get("history_reference"):
        raise ValueError("checkpoint and manifest history references differ")
    rows = (
        payload.get("history")
        if reference is None
        else load_history_reference(reference, checkpoint_path=checkpoint)
    )
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("authenticated training history is empty or malformed")
    return rows


def _metric_stats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: list[float] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"history row {index} has invalid {field}")
        values.append(float(value))
    return {
        "first": values[0],
        "last": values[-1],
        "mean": mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _validate_training_activity_summary(activity: Mapping[str, Any]) -> None:
    required = (
        "rollout_control_steps",
        "environment_count",
        "neuron_count",
        "neuron_sample_count",
        "rollout_total_sampled_spikes",
        "rollout_sampled_spike_rate_hz_per_neuron",
    )
    if any(field not in activity for field in required):
        raise ValueError("training LIF activity sample lacks required fields")
    horizon = activity["rollout_control_steps"]
    envs = activity["environment_count"]
    neurons = activity["neuron_count"]
    samples = activity["neuron_sample_count"]
    spikes = activity["rollout_total_sampled_spikes"]
    if not all(type(value) is int and value > 0 for value in (horizon, envs, neurons, samples)):
        raise ValueError("training LIF activity denominators are invalid")
    if samples != horizon * envs or neurons != 256:
        raise ValueError("training LIF activity denominator/core width changed")
    if type(spikes) is not int or not 0 <= spikes <= samples * neurons:
        raise ValueError("training LIF spike count is outside its denominator")
    expected_rate = spikes / (samples * neurons * float(activity["control_dt_s"]))
    if not math.isclose(
        float(activity["rollout_sampled_spike_rate_hz_per_neuron"]),
        expected_rate,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("training LIF sampled rate does not recompute")


def _training_activity(rows: Sequence[Mapping[str, Any]], controller: str) -> dict[str, Any] | None:
    if controller not in LIF_CONTROLLERS:
        if any(row.get("lif_activity_sampled") is True for row in rows):
            raise ValueError("engineering baseline contains biological LIF activity")
        return None
    samples: list[dict[str, Any]] = []
    for row in rows:
        sampled = row.get("lif_activity_sampled")
        value = row.get("lif_activity")
        if sampled is True:
            if not isinstance(value, Mapping):
                raise ValueError("scheduled training LIF activity is missing")
            cores = value if controller == "leg_wing_lif" else {
                "wing" if controller == "wing_lif" else "leg": value
            }
            expected_core_names = {"leg", "wing"} if controller == "leg_wing_lif" else set(cores)
            if set(cores) - {"composition"} != expected_core_names:
                raise ValueError("training LIF core composition changed")
            if controller == "leg_wing_lif" and value.get("composition") != (
                "independent_leg_and_wing_cores_concat_motor_readouts_v1"
            ):
                raise ValueError("training combined-core fusion contract changed")
            summarized: dict[str, Any] = {
                "completed_updates": row.get("completed_updates"),
                "total_interactions": row.get("total_interactions"),
                "cores": {},
            }
            for core_name in sorted(expected_core_names):
                activity = cores[core_name]
                if not isinstance(activity, Mapping):
                    raise ValueError("training LIF core activity is malformed")
                _validate_training_activity_summary(activity)
                summarized["cores"][core_name] = deepcopy(dict(activity))
            samples.append(summarized)
        elif value is not None:
            raise ValueError("unscheduled training row contains LIF activity")
    if not samples:
        raise ValueError("LIF training history has no scheduled activity samples")
    return {
        "scope": "bounded_scheduled_training_rollouts_not_heldout_roles",
        "sample_count": len(samples),
        "samples": samples,
        "last": samples[-1],
        "role_limitation": (
            "training accumulators retain aggregate core activity only; role/top-ID results "
            "come from the exact held-out neural-activity artifact"
        ),
    }


def _training_evidence(job: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    manifest_path = _resolve_path(job.get("training_manifest"))
    checkpoint = _resolve_path(job.get("checkpoint"))
    evidence: dict[str, Any] = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "manifest_sha256": None,
        "checkpoint_sha256": None,
    }
    if not manifest_path.is_file():
        return evidence, [f"missing training manifest: {manifest_path}"]
    if not checkpoint.is_file():
        return evidence, [f"missing training checkpoint: {checkpoint}"]
    try:
        manifest, digest = _read_json(manifest_path)
        checkpoint_sha = sha256_file(checkpoint)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return evidence, [f"cannot authenticate training artifacts: {exc}"]
    evidence.update({"manifest_sha256": digest, "checkpoint_sha256": checkpoint_sha})
    expected_fields = {
        "schema_version": 1,
        "status": "completed",
        "task": job.get("task"),
        "contract_profile": "balanced_v4",
        "controller": job.get("controller"),
        "seed": TRAINING_SEED,
        "num_envs": 40,
        "horizon": 100,
        "requested_interactions": INTERACTIONS_PER_JOB,
        "environment_interactions": INTERACTIONS_PER_JOB,
        "completed_updates": 125,
        "fingerprint": job.get("expected_fingerprint"),
        "evaluation_manifest_id": protocol.get("manifest_id"),
        "checkpoint_sha256": checkpoint_sha,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            issues.append(f"training {field} differs: {manifest.get(field)!r} != {expected!r}")
    try:
        if _resolve_path(manifest.get("checkpoint")) != checkpoint:
            issues.append("training manifest checkpoint path differs from queue")
    except (TypeError, ValueError):
        issues.append("training manifest checkpoint path is invalid")
    if manifest.get("warm_start") is not None:
        issues.append("training is warm-started instead of fresh")
    expected_ppo = {
        "horizon": 100,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "learning_rate": 0.0003,
        "ppo_epochs": 2,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "burn_in": 0,
    }
    if manifest.get("ppo") != expected_ppo:
        issues.append("training PPO hyperparameters differ from the frozen config")
    issues.extend(_memory_gate_issues(manifest.get("memory_gate"), label="training"))
    controller_report = manifest.get("controller_report")
    if not isinstance(controller_report, Mapping):
        issues.append("training manifest lacks controller_report")
        controller_report = {}
    if manifest.get("core_checksum_before") != manifest.get("core_checksum_after"):
        issues.append("frozen core checksum changed during training")
    if job.get("controller") in LIF_CONTROLLERS and manifest.get("core_checksum_before") is None:
        issues.append("LIF training lacks a frozen-core checksum")
    fingerprint_payload = manifest.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, Mapping) or canonical_sha256(dict(fingerprint_payload)) != manifest.get("fingerprint"):
        issues.append("training fingerprint payload does not authenticate its fingerprint")
    try:
        rows = _history_rows(checkpoint, manifest)
        history_summary = _summarize_training_history(rows, manifest.get("history_reference"))
        if history_summary.get("last_total_interactions") != INTERACTIONS_PER_JOB:
            raise ValueError("training history does not end at 500,000 interactions")
        training_activity = _training_activity(rows, str(job.get("controller")))
        completed_episode_count = sum(int(row["completed_episode_count"]) for row in rows)
        return_sum = sum(float(row["episodic_return_sum"]) for row in rows)
        curve = {
            "total_interactions": [int(row["total_interactions"]) for row in rows],
            "loss": [float(row["loss"]) for row in rows],
            "mean_rollout_reward": [float(row["mean_rollout_reward"]) for row in rows],
        }
        if any(
            len(curve[field]) != 125
            or any(not math.isfinite(value) for value in curve[field])
            for field in curve
        ):
            raise ValueError("training curve is not 125 finite authenticated updates")
        reward_summary = {
            "mean_rollout_reward": _metric_stats(rows, "mean_rollout_reward"),
            "sum_rollout_reward": sum(float(row["sum_rollout_reward"]) for row in rows),
            "completed_episode_count": completed_episode_count,
            "episodic_return_sum": return_sum,
            "episodic_return_mean_over_completed_episodes": (
                return_sum / completed_episode_count if completed_episode_count else None
            ),
            "interpretation": "on-policy training evidence; not held-out environment reward",
        }
    except Exception as exc:
        issues.append(f"training history authentication failed: {type(exc).__name__}: {exc}")
        history_summary = None
        training_activity = None
        reward_summary = None
        curve = None
    evidence.update(
        {
            "status": manifest.get("status"),
            "task": manifest.get("task"),
            "controller": manifest.get("controller"),
            "seed": manifest.get("seed"),
            "environment_interactions": manifest.get("environment_interactions"),
            "completed_updates": manifest.get("completed_updates"),
            "completed_episodes": manifest.get("completed_episodes"),
            "resume_count": manifest.get("resume_count"),
            "training_wall_time_s": manifest.get("training_wall_time_s"),
            "ppo": deepcopy(manifest.get("ppo")),
            "controller_report": deepcopy(dict(controller_report)),
            "memory_gate": deepcopy(manifest.get("memory_gate")),
            "core_checksum_before": manifest.get("core_checksum_before"),
            "core_checksum_after": manifest.get("core_checksum_after"),
            "history_reference": deepcopy(manifest.get("history_reference")),
            "history_summary": history_summary,
            "training_reward": reward_summary,
            "training_lif_activity": training_activity,
            "training_curve": curve,
        }
    )
    return evidence, issues


def _evaluation_evidence(job: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    output = _resolve_path(job.get("evaluation_output"))
    checkpoint = _resolve_path(job.get("checkpoint"))
    evidence: dict[str, Any] = {"path": str(output), "sha256": None}
    if not output.is_file():
        return evidence, None, [f"missing official evaluation: {output}"]
    try:
        value, digest = _read_json(output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return evidence, None, [f"cannot read official evaluation: {exc}"]
    evidence["sha256"] = digest
    expected = {
        "schema_version": 1,
        "status": "completed",
        "label": "main",
        "protocol": "main",
        "scenario": job.get("task"),
        "training_seed": TRAINING_SEED,
        "controller": job.get("controller"),
        "fingerprint": job.get("expected_fingerprint"),
        "evaluation_seed": protocol.get("evaluation_seed"),
        "evaluation_manifest_id": protocol.get("manifest_id"),
        "deterministic_actions": True,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            issues.append(f"evaluation {field} differs: {value.get(field)!r} != {expected_value!r}")
    try:
        if _resolve_path(value.get("checkpoint")) != checkpoint:
            issues.append("evaluation checkpoint path differs from queue")
    except (TypeError, ValueError):
        issues.append("evaluation checkpoint path is invalid")
    issues.extend(_memory_gate_issues(value.get("memory_gate"), label="evaluation"))
    episodes = value.get("episodes")
    expected_plans = protocol.get("scenarios", {}).get(job.get("task"), [])
    if not isinstance(episodes, list) or len(episodes) != EPISODES_PER_CELL:
        issues.append("evaluation must contain exactly 16 episodes")
    elif [row.get("plan") for row in episodes if isinstance(row, Mapping)] != expected_plans:
        issues.append("evaluation episodes differ from immutable main plans")
    try:
        scored = score_evaluation(value, str(job.get("task")))
    except Exception as exc:
        issues.append(f"official score recomputation failed: {type(exc).__name__}: {exc}")
        scored = None
    evidence.update(
        {
            "summary": deepcopy(value.get("summary")),
            "episodes": deepcopy(episodes),
            "memory_gate": deepcopy(value.get("memory_gate")),
            "heldout_environment_reward": None,
            "heldout_environment_reward_status": (
                "unavailable_official_evaluator_does_not_emit_reward"
            ),
        }
    )
    return evidence, scored, issues


def _recompute_activity_payload(activity: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = 10
    cores: dict[str, Any] = {}
    raw_cores = activity.get("cores")
    if not isinstance(raw_cores, Mapping):
        raise ValueError("activity cores must be an object")
    for name, core in raw_cores.items():
        if name not in {"leg", "wing"} or not isinstance(core, Mapping):
            raise ValueError("activity has an unknown/malformed biological core")
        rows = core.get("per_neuron")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("activity core lacks full per-neuron counts")
        if [row.get("index") for row in rows] != list(range(len(rows))):
            raise ValueError("activity per-neuron indices are not contiguous")
        counts = [row.get("sampled_spike_count") for row in rows]
        cores[name] = summarize_lif_counts(
            counts,
            active_control_decisions=int(activity["active_control_decisions"]),
            manifest_path=LEG_MANIFEST if name == "leg" else WING_MANIFEST,
            top_count=expected_top,
        )
    engineering: dict[str, Any] = {}
    raw_layers = activity.get("engineering_layers")
    if not isinstance(raw_layers, Mapping):
        raise ValueError("activity engineering_layers must be an object")
    for name, layer in raw_layers.items():
        if not isinstance(name, str) or not isinstance(layer, Mapping):
            raise ValueError("activity engineering layer is malformed")
        rows = layer.get("per_unit")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("activity engineering layer lacks full per-unit accumulators")
        if [row.get("index") for row in rows] != list(range(len(rows))):
            raise ValueError("activity per-unit indices are not contiguous")
        engineering[name] = summarize_engineering_units(
            [row.get("absolute_activation_sum") for row in rows],
            [row.get("squared_activation_sum") for row in rows],
            [row.get("active_sample_count") for row in rows],
            observation_count=int(activity["active_control_decisions"]),
            layer=name,
            top_count=expected_top,
        )
    recomputed = dict(activity)
    recomputed["cores"] = cores
    recomputed["engineering_layers"] = engineering
    return recomputed


def _activity_evidence(
    job: Mapping[str, Any],
    official: Mapping[str, Any],
    scored: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    output = _resolve_path(job.get("neural_activity_output"))
    evidence: dict[str, Any] = {"path": str(output), "sha256": None, "activity": None}
    if not output.is_file():
        return evidence, [f"missing held-out neural activity artifact: {output}"]
    try:
        value, digest = _read_json(output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return evidence, [f"cannot read held-out neural activity artifact: {exc}"]
    evidence["sha256"] = digest
    identity_fields = (
        "status",
        "label",
        "protocol",
        "scenario",
        "training_seed",
        "controller",
        "fingerprint",
        "evaluation_seed",
        "evaluation_manifest_id",
        "checkpoint",
        "checkpoint_sha256",
        "deterministic_actions",
    )
    for field in identity_fields:
        if value.get(field) != official.get(field):
            issues.append(f"activity rerun {field} differs from official evaluation")
    if canonical_sha256(value.get("episodes")) != canonical_sha256(official.get("episodes")):
        issues.append("activity rerun episode rows differ from official deterministic evaluation")
    if canonical_sha256(value.get("summary")) != canonical_sha256(official.get("summary")):
        issues.append("activity rerun summary differs from official deterministic evaluation")
    issues.extend(_memory_gate_issues(value.get("memory_gate"), label="activity rerun"))
    try:
        rescored = score_evaluation(value, str(job.get("task")))
        if scored is None or canonical_sha256(rescored) != canonical_sha256(scored):
            issues.append("activity rerun score differs from official score")
    except Exception as exc:
        issues.append(f"activity rerun score cannot be recomputed: {type(exc).__name__}: {exc}")
    activity = value.get("neural_activity")
    if not isinstance(activity, Mapping):
        issues.append("activity rerun lacks neural_activity")
        activity = {}
    expected_activity_identity = {
        "schema_version": 1,
        "status": "completed",
        "analysis_kind": "crazyflie_heldout_neural_activity_v1",
        "controller": job.get("controller"),
        "scenario": job.get("task"),
        "training_seed": TRAINING_SEED,
        "evaluation_seed": official.get("evaluation_seed"),
        "evaluation_manifest_id": official.get("evaluation_manifest_id"),
        "checkpoint": official.get("checkpoint"),
        "checkpoint_sha256": official.get("checkpoint_sha256"),
        "episode_ids": list(range(EPISODES_PER_CELL)),
        "plan_sha256": [row.get("plan_sha256") for row in official.get("episodes", [])],
    }
    for field, expected in expected_activity_identity.items():
        if activity.get(field) != expected:
            issues.append(f"neural activity {field} differs from its evaluation")
    collector = Path(str(activity.get("collector", ""))).resolve()
    if collector != (ROOT / "scripts" / "crazyflie_neural_activity.py").resolve():
        issues.append("neural activity collector path is not the project collector")
    elif not collector.is_file() or activity.get("collector_sha256") != sha256_file(collector):
        issues.append("neural activity collector SHA-256 differs from current source")
    completed_steps = sum(
        int(row["completed_steps"])
        for row in official.get("episodes", [])
        if isinstance(row, Mapping) and type(row.get("completed_steps")) is int
    )
    if activity.get("active_control_decisions") != completed_steps:
        issues.append("neural activity denominator differs from official completed steps")
    try:
        recomputed = _recompute_activity_payload(activity)
        if canonical_sha256(recomputed) != canonical_sha256(dict(activity)):
            issues.append("neural role/unit summaries do not recompute from full counts")
    except Exception as exc:
        issues.append(f"neural activity recomputation failed: {type(exc).__name__}: {exc}")
    cores = activity.get("cores") if isinstance(activity, Mapping) else None
    layers = activity.get("engineering_layers") if isinstance(activity, Mapping) else None
    controller = job.get("controller")
    expected_cores = (
        {"leg", "wing"}
        if controller == "leg_wing_lif"
        else {"wing"}
        if controller == "wing_lif"
        else {"leg"}
        if controller in {"frozen_lif_original", "frozen_lif_degree_rewired"}
        else set()
    )
    if not isinstance(cores, Mapping) or set(cores) != expected_cores:
        issues.append("neural activity biological-core set differs from controller architecture")
    if controller == "gru_matched" and (not isinstance(layers, Mapping) or set(layers) != {"gru_hidden"}):
        issues.append("GRU activity must contain exactly gru_hidden engineering units")
    if controller == "mlp_normal" and (
        not isinstance(layers, Mapping)
        or not layers
        or any(not str(name).startswith("mlp_hidden_") for name in layers)
    ):
        issues.append("MLP activity must contain its engineering hidden layers")
    if controller in LIF_CONTROLLERS and layers not in ({}, None):
        issues.append("LIF activity unexpectedly assigns engineering layers")
    evidence["activity"] = deepcopy(dict(activity))
    evidence["memory_gate"] = deepcopy(value.get("memory_gate"))
    return evidence, issues


def _rewire_evidence(cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the frozen degree-preserving graph, not just its declaration."""

    from g1_fly_control.connectome import load_connectome
    from g1_fly_control.crazyflie.controllers import load_rewire_manifest

    declaration = config["rewire"]
    path = _resolve_path(declaration["manifest"])
    circuit = load_connectome(LEG_MANIFEST)
    _, _, manifest = load_rewire_manifest(
        path,
        circuit,
        expected_seed=int(declaration["seed"]),
        expected_file_sha256=str(declaration["manifest_sha256"]),
    )
    invariants = manifest["invariants"]
    if not isinstance(invariants, Mapping) or not all(value is True for value in invariants.values()):
        raise ValueError("rewire invariants did not all pass")
    reports = [
        cell.get("training_evidence", {}).get("controller_report", {})
        for cell in cells
        if cell.get("controller") == "frozen_lif_degree_rewired"
        and cell.get("status") == "valid"
    ]
    for report in reports:
        if (
            report.get("rewire_seed") != declaration["seed"]
            or report.get("rewire_manifest_path") != str(path)
            or report.get("rewire_manifest_file_sha256") != declaration["manifest_sha256"]
            or report.get("rewire_manifest_checksum") != manifest["checksum"]
            or report.get("rewire_manifest") != manifest
        ):
            raise ValueError("trained rewired controller provenance differs from frozen artifact")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "seed": manifest["seed"],
        "requested_swaps": manifest["requested_swaps"],
        "completed_swaps": manifest["completed_swaps"],
        "algorithm": manifest["algorithm"],
        "content_checksum": manifest["checksum"],
        "source_connectome_checksum": manifest["source_connectome_checksum"],
        "invariants": deepcopy(dict(invariants)),
        "validated_training_cells": len(reports),
    }


def _rank_complete(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["macro_mean_score"]),
            -float(row["worst_task_score"]),
            -float(row["macro_event_completion_rate"]),
            str(row["controller"]),
        ),
    )
    previous: tuple[float, float, float] | None = None
    rank = 0
    for position, row in enumerate(ordered, start=1):
        key = (
            float(row["macro_mean_score"]),
            float(row["worst_task_score"]),
            float(row["macro_event_completion_rate"]),
        )
        if key != previous:
            rank = position
            previous = key
        row["rank"] = rank
    winners = [row["controller"] for row in ordered if row["rank"] == 1]
    return {
        "basis": SCORE_CONTRACT["winner_basis"],
        "controllers": winners,
        "tie": len(winners) > 1,
    }


def _activation_comparison(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    wing_vs_thoracic: list[dict[str, Any]] = []
    for cell in cells:
        activity = cell.get("activity_evidence", {}).get("activity")
        if not isinstance(activity, Mapping):
            continue
        for core_name, core in (activity.get("cores") or {}).items():
            for role, role_row in core.get("roles", {}).items():
                rows.append(
                    {
                        "controller": cell["controller"],
                        "task": cell["task"],
                        "core": core_name,
                        "role": role,
                        "neuron_count": role_row["neuron_count"],
                        "sampled_spike_rate_hz_per_neuron": role_row[
                            "sampled_spike_rate_hz_per_neuron"
                        ],
                        "ever_active_neuron_fraction": role_row[
                            "ever_active_neuron_fraction"
                        ],
                        "dead_neuron_fraction": role_row["dead_neuron_fraction"],
                        "saturated_neuron_fraction": role_row[
                            "saturated_neuron_fraction"
                        ],
                        "top_neurons": deepcopy(role_row["top_neurons"]),
                    }
                )
            if core_name == "wing":
                roles = core["roles"]
                sensory = roles["wing_sensory_input"]
                motor = roles["wing_motor_output"]
                thoracic = roles["vnc_interneuron"]
                wing_neurons = int(sensory["neuron_count"]) + int(motor["neuron_count"])
                wing_spikes = int(sensory["sampled_spike_count"]) + int(
                    motor["sampled_spike_count"]
                )
                decisions = int(core["active_control_decisions"])
                wing_rate = wing_spikes / (decisions * 0.02 * wing_neurons)
                thoracic_rate = float(thoracic["sampled_spike_rate_hz_per_neuron"])
                wing_vs_thoracic.append(
                    {
                        "controller": cell["controller"],
                        "task": cell["task"],
                        "wing_definition": (
                            "wing_sensory_input_plus_wing_motor_output; descending_input excluded"
                        ),
                        "wing_neuron_count": wing_neurons,
                        "thoracic_neuron_count": int(thoracic["neuron_count"]),
                        "wing_aggregate_hz_per_neuron": wing_rate,
                        "thoracic_intrinsic_hz_per_neuron": thoracic_rate,
                        "wing_minus_thoracic_hz_per_neuron": wing_rate
                        - thoracic_rate,
                        "wing_to_thoracic_rate_ratio": (
                            wing_rate / thoracic_rate if thoracic_rate > 0 else None
                        ),
                        "more_active_population": (
                            "wing"
                            if wing_rate > thoracic_rate
                            else "thoracic"
                            if thoracic_rate > wing_rate
                            else "tie"
                        ),
                    }
                )
        if cell["controller"] == "leg_wing_lif" and activity.get("cores"):
            leg_role = activity["cores"]["leg"]["roles"]["vnc_interneuron"]
            wing_role = activity["cores"]["wing"]["roles"]["vnc_interneuron"]
            leg_rate = float(leg_role["sampled_spike_rate_hz_per_neuron"])
            wing_rate = float(wing_role["sampled_spike_rate_hz_per_neuron"])
            combined.append(
                {
                    "task": cell["task"],
                    "leg_thoracic_intrinsic_hz_per_neuron": leg_rate,
                    "wing_thoracic_intrinsic_hz_per_neuron": wing_rate,
                    "wing_minus_leg_hz_per_neuron": wing_rate - leg_rate,
                    "wing_to_leg_rate_ratio": (
                        wing_rate / leg_rate if leg_rate > 0 else None
                    ),
                    "more_active_core": (
                        "wing" if wing_rate > leg_rate else "leg" if leg_rate > wing_rate else "tie"
                    ),
                }
            )
    engineering: list[dict[str, Any]] = []
    for cell in cells:
        activity = cell.get("activity_evidence", {}).get("activity")
        if not isinstance(activity, Mapping):
            continue
        for layer_name, layer in (activity.get("engineering_layers") or {}).items():
            engineering.append(
                {
                    "controller": cell["controller"],
                    "task": cell["task"],
                    "layer": layer_name,
                    "unit_count": layer["unit_count"],
                    "mean_absolute_activation_per_unit": layer[
                        "mean_absolute_activation_per_unit"
                    ],
                    "rms_activation_per_unit": layer["rms_activation_per_unit"],
                    "mean_active_fraction_per_unit": layer[
                        "mean_active_fraction_per_unit"
                    ],
                    "top_units": deepcopy(layer["top_units"]),
                    "biological_role": None,
                }
            )
    return {
        "lif_role_rows": rows,
        "wing_vs_thoracic_rows": wing_vs_thoracic,
        "combined_leg_vs_wing_thoracic": combined,
        "engineering_unit_rows": engineering,
        "cross_family_warning": (
            "LIF sampled hertz and GRU/MLP numeric activations have different semantics and "
            "must not be ranked as one common firing-rate scale"
        ),
    }


def _report_artifact_paths(report: Mapping[str, Any]) -> set[Path]:
    """Enumerate every mutable input byte consumed by the report."""

    paths = {
        Path(str(report["queue"])).resolve(),
        Path(str(report["config"]["config_path"])).resolve(),
        Path(__file__).resolve(),
        (ROOT / "scripts" / "crazyflie_neural_activity.py").resolve(),
        LEG_MANIFEST.resolve(),
        WING_MANIFEST.resolve(),
    }
    rewire_path = report["rewire_provenance"].get("path")
    if isinstance(rewire_path, str) and rewire_path:
        paths.add(Path(rewire_path).resolve())
    config = report["config"]["config"]
    try:
        paths.add(_resolve_path(config["evaluation"]["manifest"]))
    except (KeyError, TypeError, ValueError):
        pass
    for manifest_path in (LEG_MANIFEST, WING_MANIFEST):
        try:
            manifest, _ = _read_json(manifest_path)
            paths.add((manifest_path.parent / manifest["neurons_path"]).resolve())
            paths.add((manifest_path.parent / manifest["edges_path"]).resolve())
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    cells = report.get("controller_task_table", [])
    # Completed training evidence does not duplicate the large fingerprint
    # payload. Source paths are therefore taken from the queue payload.
    try:
        queue, _ = _read_json(Path(str(report["queue"])))
        jobs = queue.get("jobs", [])
        if jobs:
            for source_path in jobs[0].get("fingerprint_payload", {}).get(
                "source_sha256", {}
            ):
                paths.add(_resolve_path(source_path))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    for cell in cells:
        training = cell.get("training_evidence", {})
        evaluation = cell.get("evaluation_evidence", {})
        activity = cell.get("activity_evidence", {})
        for value in (
            training.get("manifest"),
            training.get("checkpoint"),
            evaluation.get("path"),
            activity.get("path"),
        ):
            try:
                paths.add(_resolve_path(value))
            except (TypeError, ValueError):
                pass
        reference = training.get("history_reference")
        checkpoint_value = training.get("checkpoint")
        if isinstance(reference, Mapping) and isinstance(checkpoint_value, str):
            checkpoint = Path(checkpoint_value).resolve()
            for segment in reference.get("segments", []):
                if isinstance(segment, Mapping) and isinstance(segment.get("path"), str):
                    paths.add((checkpoint.parent / segment["path"]).resolve())
    return paths


def _capture_artifact_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(_report_artifact_paths(report), key=str):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            missing.append(str(path))
    if report.get("status") == "complete" and missing:
        raise ValueError(
            "Complete report artifact snapshot has missing files: " + ", ".join(missing)
        )
    snapshot = {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "files": files,
        "missing_at_snapshot": missing,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def _revalidate_artifact_snapshot(snapshot: Mapping[str, Any]) -> None:
    for row in snapshot.get("files", []):
        path = Path(str(row["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"Report input changed after snapshot: {path}")


def build_report(queue_path: Path, *, allow_incomplete: bool) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    queue, queue_sha = _read_json(queue_path)
    design = validate_design(queue, queue_path=queue_path)
    protocol = load_protocol("main")
    cells: list[dict[str, Any]] = []
    for job in queue["jobs"]:
        training, training_issues = _training_evidence(job, protocol)
        expected_controller_report = (queue.get("controller_reports") or {}).get(
            job.get("controller")
        )
        if not isinstance(expected_controller_report, Mapping):
            training_issues.append("queue lacks the preflight controller report")
        elif training.get("controller_report") != expected_controller_report:
            training_issues.append(
                "trained controller report differs from the authenticated CPU preflight"
            )
        evaluation, scored, evaluation_issues = _evaluation_evidence(job, protocol)
        official_value = {}
        if Path(evaluation["path"]).is_file():
            try:
                official_value, _ = _read_json(Path(evaluation["path"]))
            except Exception:
                pass
        activity, activity_issues = _activity_evidence(job, official_value, scored)
        issues = []
        if job.get("status") != "completed":
            issues.append(f"queue job status is {job.get('status')!r}, not 'completed'")
        issues.extend(training_issues + evaluation_issues + activity_issues)
        valid = not issues and scored is not None
        cells.append(
            {
                "job_id": job["id"],
                "controller": job["controller"],
                "task": job["task"],
                "seed": job["seed"],
                "status": "valid" if valid else "invalid_or_incomplete",
                "issues": issues,
                "score": scored.get("score") if valid else None,
                "score_components_points": (
                    deepcopy(scored.get("score_components_points")) if valid else None
                ),
                "raw_metrics": deepcopy(scored.get("raw_metrics")) if valid else None,
                "training_evidence": training,
                "evaluation_evidence": evaluation,
                "activity_evidence": activity,
            }
        )
    valid_cells = sum(cell["status"] == "valid" for cell in cells)
    complete = valid_cells == EXPECTED_JOB_COUNT
    if not complete and not allow_incomplete:
        details = [
            f"{cell['job_id']}: " + "; ".join(cell["issues"])
            for cell in cells
            if cell["status"] != "valid"
        ]
        raise ValueError(
            "Comparison is incomplete/invalid; rerun with --allow_incomplete only for a "
            "development report:\n- " + "\n- ".join(details)
        )
    overall: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        selected = [cell for cell in cells if cell["controller"] == controller]
        if complete:
            scores = [float(cell["score"]) for cell in selected]
            event_rates = [
                float(cell["raw_metrics"]["task_event_completion_rate"])
                for cell in selected
            ]
            reward_last = [
                float(
                    cell["training_evidence"]["training_reward"]
                    ["mean_rollout_reward"]["last"]
                )
                for cell in selected
            ]
            overall.append(
                {
                    "controller": controller,
                    "status": "valid",
                    "rank": None,
                    "macro_mean_score": mean(scores),
                    "worst_task_score": min(scores),
                    "macro_event_completion_rate": mean(event_rates),
                    "macro_last_training_rollout_reward": mean(reward_last),
                    "task_scores": {cell["task"]: cell["score"] for cell in selected},
                }
            )
        else:
            overall.append(
                {
                    "controller": controller,
                    "status": "not_ranked_incomplete_experiment",
                    "rank": None,
                    "macro_mean_score": None,
                    "worst_task_score": None,
                    "macro_event_completion_rate": None,
                    "macro_last_training_rollout_reward": None,
                    "task_scores": {cell["task"]: cell["score"] for cell in selected},
                }
            )
    winner = _rank_complete(overall) if complete else None
    try:
        rewire = _rewire_evidence(cells, design["config"])
    except Exception as exc:
        if not allow_incomplete:
            raise ValueError(f"Rewire provenance validation failed: {exc}") from exc
        rewire = {"status": "invalid_or_incomplete", "error": f"{type(exc).__name__}: {exc}"}
        complete = False
        winner = None
        for row in overall:
            row.update(
                {
                    "status": "not_ranked_incomplete_experiment",
                    "rank": None,
                    "macro_mean_score": None,
                    "worst_task_score": None,
                    "macro_event_completion_rate": None,
                    "macro_last_training_rollout_reward": None,
                }
            )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "status": "complete" if complete else "incomplete_or_invalid_no_winner",
        "allow_incomplete": allow_incomplete,
        "scientific_status": "descriptive_one_training_seed_no_confidence_interval",
        "interpretation": (
            "All held-out task scores use paired immutable main-protocol plans, but only one "
            "independent training seed is present. This is an engineering comparison, not a "
            "training-seed confidence interval or significance result."
        ),
        "queue": str(queue_path),
        "queue_sha256": queue_sha,
        "queue_status": queue.get("status"),
        "config": design,
        "design": {
            "controllers": list(CONTROLLERS),
            "tasks": list(TASKS),
            "seed": TRAINING_SEED,
            "jobs": EXPECTED_JOB_COUNT,
            "interactions_per_job": INTERACTIONS_PER_JOB,
            "total_training_interactions": EXPECTED_TOTAL_INTERACTIONS,
            "episodes_per_matching_task": EPISODES_PER_CELL,
            "official_evaluation_episodes": EXPECTED_EVALUATION_EPISODES,
            "activity_rerun_episodes": EXPECTED_EVALUATION_EPISODES,
            "parameter_matching_note": (
                "single-core LIF, rewired LIF, wing LIF, GRU, and MLP are numerically "
                "matched within the controller gate; combined leg+wing is an explicitly "
                "unmatched extension and its exact count is reported"
            ),
        },
        "completion": {
            "valid_cells": valid_cells,
            "planned_cells": EXPECTED_JOB_COUNT,
            "authenticated_official_episodes": valid_cells * EPISODES_PER_CELL,
            "planned_official_episodes": EXPECTED_EVALUATION_EPISODES,
            "authenticated_activity_rerun_episodes": valid_cells * EPISODES_PER_CELL,
            "planned_activity_rerun_episodes": EXPECTED_EVALUATION_EPISODES,
        },
        "score_contract": deepcopy(SCORE_CONTRACT),
        "score_contract_sha256": SCORE_CONTRACT_SHA256,
        "heldout_reward_status": (
            "unavailable; official evaluation artifacts do not emit environment reward; "
            "training rollout reward is shown separately without substituting it for held-out score"
        ),
        "winner": winner,
        "overall_controller_table": overall,
        "controller_task_table": cells,
        "activation_analysis": _activation_comparison(cells),
        "rewire_provenance": rewire,
        "invalid_cells": [
            {"job_id": cell["job_id"], "issues": cell["issues"]}
            for cell in cells
            if cell["status"] != "valid"
        ],
        "reporter": str(Path(__file__).resolve()),
        "reporter_sha256": sha256_file(Path(__file__).resolve()),
        "activity_collector": str((ROOT / "scripts" / "crazyflie_neural_activity.py").resolve()),
        "activity_collector_sha256": sha256_file(ROOT / "scripts" / "crazyflie_neural_activity.py"),
        "training_curves_plot": None,
        "report_sha256_contract": "canonical_sha256_of_report_without_report_sha256_field",
    }
    report["artifact_snapshot"] = _capture_artifact_snapshot(report)
    return report


def plot_training_curves(report: Mapping[str, Any], output: Path) -> None:
    """Render all available authenticated histories as task-faceted curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "frozen_lif_original": "#1f77b4",
        "frozen_lif_degree_rewired": "#9467bd",
        "wing_lif": "#17becf",
        "leg_wing_lif": "#2ca02c",
        "gru_matched": "#ff7f0e",
        "mlp_normal": "#d62728",
    }
    short = {
        "frozen_lif_original": "Leg LIF",
        "frozen_lif_degree_rewired": "Rewired leg LIF",
        "wing_lif": "Wing LIF",
        "leg_wing_lif": "Leg+wing LIF",
        "gru_matched": "GRU",
        "mlp_normal": "MLP",
    }
    task_names = {
        TASKS[0]: "Waypoint Reach",
        TASKS[1]: "Waypoint Switch",
        TASKS[2]: "Gust Recovery",
    }
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    for row_index, task in enumerate(TASKS):
        for column, (metric, title) in enumerate(
            (("loss", "PPO total loss"), ("mean_rollout_reward", "Mean rollout reward"))
        ):
            axis = axes[row_index, column]
            available = 0
            for cell in report["controller_task_table"]:
                if cell["task"] != task:
                    continue
                curve = cell.get("training_evidence", {}).get("training_curve")
                if not isinstance(curve, Mapping):
                    continue
                x = [float(value) / 1000.0 for value in curve["total_interactions"]]
                y = curve[metric]
                axis.plot(
                    x,
                    y,
                    color=colors[cell["controller"]],
                    linewidth=1.5,
                    alpha=0.9,
                    label=short[cell["controller"]],
                )
                available += 1
            axis.set_title(f"{task_names[task]} — {title}", loc="left", fontweight="bold")
            axis.grid(True, alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_ylabel(title)
            if row_index == 2:
                axis.set_xlabel("Training interactions (thousands)")
            if available == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No authenticated history available",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    status = report["status"]
    figure.suptitle(
        "Crazyflie 500k seed-0 controller comparison — authenticated training curves\n"
        f"Status: {status}; missing cells are omitted and never imputed",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    try:
        figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _short_task(task: str) -> str:
    return task.removeprefix("FlyCrazyflie-").removesuffix("-v0")


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report: Mapping[str, Any], *, plot_path: Path, markdown_path: Path) -> str:
    completion = report["completion"]
    plot_link = os.path.relpath(plot_path.resolve(), markdown_path.resolve().parent)
    lines = [
        "# Crazyflie neural-controller comparison — seed 0, 500k per task",
        "",
        f"Status: **{report['status']}** — {completion['valid_cells']}/{completion['planned_cells']} authenticated controller×task cells.",
        "",
        "> This is a descriptive one-training-seed result. It does not estimate training-seed uncertainty or statistical significance.",
        "",
        f"![Authenticated loss and reward curves]({plot_link})",
        "",
        f"Plot SHA-256: `{report['training_curves_plot']['sha256']}`.",
        "",
        "## Overall held-out score",
        "",
        "| Rank | Controller | Macro score /100 | Worst task | Event completion | Last training reward (macro) | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["overall_controller_table"]:
        lines.append(
            f"| {row['rank'] if row['rank'] is not None else 'N/A'} | `{row['controller']}` | "
            f"{_fmt(row['macro_mean_score'])} | {_fmt(row['worst_task_score'])} | "
            f"{_fmt(row['macro_event_completion_rate'])} | "
            f"{_fmt(row['macro_last_training_rollout_reward'], 6)} | {row['status']} |"
        )
    lines += [
        "",
        "Held-out environment reward is **not available** because the official evaluator does not emit it. The reward column above is the final on-policy training rollout reward averaged across tasks; it is shown for diagnosis and is not used to pick the winner.",
        "",
        "## Controller × task result",
        "",
        "| Controller | Task | Score /100 | Events | Survival | Final error (m) | Avg error (m) | Last train reward | Status |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for cell in report["controller_task_table"]:
        raw = cell.get("raw_metrics") or {}
        reward = (
            cell.get("training_evidence", {})
            .get("training_reward", {})
            .get("mean_rollout_reward", {})
            .get("last")
        )
        lines.append(
            f"| `{cell['controller']}` | {_short_task(cell['task'])} | {_fmt(cell['score'])} | "
            f"{raw.get('task_event_success_count', 'N/A')}/{raw.get('task_event_denominator', 'N/A')} | "
            f"{raw.get('nonterminated_episode_count', 'N/A')}/{raw.get('survival_denominator', 'N/A')} | "
            f"{_fmt(raw.get('mean_final_goal_error_m'))} | {_fmt(raw.get('time_average_goal_error_m'))} | "
            f"{_fmt(reward, 6)} | {cell['status']} |"
        )
    lines += [
        "",
        "## Model size and resource evidence",
        "",
        "| Controller | Task | Actor trainable | Total trainable | Frozen | Model total | Dynamic state/env | Train GPU MiB | Train RAM % | Eval GPU MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in report["controller_task_table"]:
        training = cell["training_evidence"]
        controller = training.get("controller_report") or {}
        train_memory = training.get("memory_gate") or {}
        eval_memory = cell["evaluation_evidence"].get("memory_gate") or {}
        lines.append(
            f"| `{cell['controller']}` | {_short_task(cell['task'])} | "
            f"{controller.get('actor_trainable_parameters', 'N/A')} | "
            f"{controller.get('total_trainable_parameters', 'N/A')} | "
            f"{controller.get('frozen_parameters', 'N/A')} | "
            f"{controller.get('model_total_parameters', 'N/A')} | "
            f"{controller.get('total_dynamic_state_per_environment', 'N/A')} | "
            f"{_fmt(train_memory.get('max_device_gpu_used_mib'), 1)} | "
            f"{_fmt(train_memory.get('max_system_ram_percent'), 1)} | "
            f"{_fmt(eval_memory.get('max_device_gpu_used_mib'), 1)} |"
        )
    lines += [
        "",
        "The combined leg+wing LIF is intentionally an unmatched extension. Exact parameter counts are shown instead of implying it has the same capacity as the single-core/GRU/MLP comparison set.",
        "",
        "## Biological LIF activity by role",
        "",
        "Rates are sampled spikes after the final internal neural substep, divided by the 20 ms control interval and number of neurons in the role. They are not a count of every internal-substep spike.",
        "",
        "| Controller | Task | Core | Role | Neurons | Hz/neuron | Dead | Saturated | Top neuron IDs (Hz) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["activation_analysis"]["lif_role_rows"]:
        top = ", ".join(
            f"{item['id']} ({_fmt(item['sampled_spike_rate_hz'], 2)})"
            for item in row["top_neurons"][:3]
        )
        lines.append(
            f"| `{row['controller']}` | {_short_task(row['task'])} | {row['core']} | "
            f"{row['role']} | {row['neuron_count']} | "
            f"{_fmt(row['sampled_spike_rate_hz_per_neuron'], 4)} | "
            f"{_fmt(row['dead_neuron_fraction'], 3)} | "
            f"{_fmt(row['saturated_neuron_fraction'], 3)} | {top} |"
        )
    lines += [
        "",
        "## Wing aggregate vs thoracic intrinsic activity",
        "",
        "Wing aggregate means wing sensory plus wing motor cells; descending inputs are shown separately above and are excluded from this comparison.",
        "",
        "| Controller | Task | Wing cells | Thoracic cells | Wing Hz/neuron | Thoracic Hz/neuron | Difference | Ratio | More active |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["activation_analysis"]["wing_vs_thoracic_rows"]:
        lines.append(
            f"| `{row['controller']}` | {_short_task(row['task'])} | "
            f"{row['wing_neuron_count']} | {row['thoracic_neuron_count']} | "
            f"{_fmt(row['wing_aggregate_hz_per_neuron'], 4)} | "
            f"{_fmt(row['thoracic_intrinsic_hz_per_neuron'], 4)} | "
            f"{_fmt(row['wing_minus_thoracic_hz_per_neuron'], 4)} | "
            f"{_fmt(row['wing_to_thoracic_rate_ratio'], 3)} | "
            f"{row['more_active_population']} |"
        )
    lines += [
        "",
        "## Combined leg vs wing thoracic activity",
        "",
        "| Task | Leg thoracic Hz/neuron | Wing thoracic Hz/neuron | Difference | Ratio wing/leg | More active |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["activation_analysis"]["combined_leg_vs_wing_thoracic"]:
        lines.append(
            f"| {_short_task(row['task'])} | "
            f"{_fmt(row['leg_thoracic_intrinsic_hz_per_neuron'], 4)} | "
            f"{_fmt(row['wing_thoracic_intrinsic_hz_per_neuron'], 4)} | "
            f"{_fmt(row['wing_minus_leg_hz_per_neuron'], 4)} | "
            f"{_fmt(row['wing_to_leg_rate_ratio'], 3)} | {row['more_active_core']} |"
        )
    lines += [
        "",
        "## GRU/MLP engineering-unit activity",
        "",
        "These values are not biological firing rates and have no wing/leg/thoracic labels.",
        "",
        "| Controller | Task | Layer | Units | Mean | RMS | Active fraction | Top unit indices |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["activation_analysis"]["engineering_unit_rows"]:
        top = ", ".join(str(item["index"]) for item in row["top_units"][:5])
        lines.append(
            f"| `{row['controller']}` | {_short_task(row['task'])} | {row['layer']} | "
            f"{row['unit_count']} | {_fmt(row['mean_absolute_activation_per_unit'], 5)} | "
            f"{_fmt(row['rms_activation_per_unit'], 5)} | "
            f"{_fmt(row['mean_active_fraction_per_unit'], 3)} | {top} |"
        )
    rewire = report["rewire_provenance"]
    lines += [
        "",
        "## Rewired LIF provenance",
        "",
        f"Frozen artifact: `{rewire.get('path', 'N/A')}`; seed `{rewire.get('seed', 'N/A')}`; completed swaps `{rewire.get('completed_swaps', 'N/A')}`; checksum `{rewire.get('content_checksum', 'N/A')}`.",
        "",
        "All declared invariants are revalidated from the graph: directed in/out degree, per-source weight/sign multiset, global weight multiset, no self-loops, no duplicates, source indices, and changed topology.",
        "",
    ]
    invalid = report.get("invalid_cells") or []
    if invalid:
        lines += ["## Invalid or incomplete cells", ""]
        for cell in invalid:
            lines.append(
                f"- `{_escape(cell['job_id'])}`: "
                + "; ".join(_escape(issue) for issue in cell["issues"])
            )
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "- A controller winner exists only when all 18 cells, 288 official held-out episodes, and 288 activity-rerun episodes authenticate. Partial reports have no ranking or winner.",
        "- LIF sampled Hz, GRU hidden-state magnitude, and MLP hidden-layer magnitude use different definitions; cross-family activation magnitude is not a valid biological comparison.",
        "- Seed 0 alone cannot support confidence intervals over training seeds.",
        "- Training reward and held-out task score answer different questions; low PPO loss or high dense training reward does not prove task completion.",
        "",
        f"Score contract SHA-256: `{report['score_contract_sha256']}`. Report SHA-256: `{report['report_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Publish a visibly incomplete development report with no ranking/winner",
    )
    args = parser.parse_args()
    outputs = {args.json.resolve(), args.markdown.resolve(), args.plot.resolve()}
    if len(outputs) != 3:
        parser.error("--json, --markdown, and --plot must be distinct paths")
    try:
        report = build_report(args.queue, allow_incomplete=args.allow_incomplete)
        plot_training_curves(report, args.plot.resolve())
        report["training_curves_plot"] = {
            "path": str(args.plot.resolve()),
            "sha256": sha256_file(args.plot.resolve()),
            "status": report["status"],
            "cells_with_authenticated_curves": sum(
                isinstance(cell.get("training_evidence", {}).get("training_curve"), Mapping)
                for cell in report["controller_task_table"]
            ),
        }
        report["report_sha256"] = canonical_sha256(report)
        json_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        markdown_text = markdown(
            report,
            plot_path=args.plot.resolve(),
            markdown_path=args.markdown.resolve(),
        )
        # Keep the immutable check adjacent to publication. Training may be
        # running concurrently with an --allow_incomplete development report;
        # any input that changed after parsing invalidates this attempt.
        _revalidate_artifact_snapshot(report["artifact_snapshot"])
        _atomic_text(args.json.resolve(), json_text)
        _atomic_text(args.markdown.resolve(), markdown_text)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid_cells": report["completion"]["valid_cells"],
                "planned_cells": EXPECTED_JOB_COUNT,
                "winner": report["winner"],
                "json": str(args.json.resolve()),
                "markdown": str(args.markdown.resolve()),
                "plot": str(args.plot.resolve()),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
