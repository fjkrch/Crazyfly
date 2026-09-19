#!/usr/bin/env python3
"""Validate a completed matrix and summarize comparisons over training seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "success_rate",
    "mean_xy_progress_m",
    "mean_accumulated_goal_relative_progress_m",
    "mean_time_to_target_s",
    "mean_work_proxy",
    "mean_excessive_impact",
    "mean_joint_limit_frequency",
    "mean_saturation_frequency",
    "mean_goal_switch_count",
    "mean_push_count",
    "mean_push_impulse_magnitude_n_s",
    "recovery_success_rate",
    "mean_recovery_time_s",
)
BASELINE = "frozen_lif_original"
RESOURCE_METRICS = (
    "model_total_parameters", "model_trainable_parameters", "model_frozen_parameters",
    "actual_environment_interactions", "training_wall_time_s",
    "max_sampled_gpu_used_mib", "max_sampled_ram_used_gib", "max_sampled_ram_percent",
    "max_sampled_process_rss_mib", "max_stage_torch_cuda_peak_allocated_mib",
)


def _digest_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, _digest_bytes(data)


def _number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _matches(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    number = _number(actual)
    return number is not None and math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-8)


def _validate_metric_rows(result: dict[str, Any], scenario: str) -> str | None:
    """Reject absent/nonfinite metrics and seed summaries inconsistent with episodes."""
    episodes = result["episodes"]
    seed_row = result["per_seed"][0]
    goal_switch = "GoalSwitch" in scenario
    mean_fields = {
        "mean_accumulated_goal_relative_progress_m": "accumulated_goal_relative_progress_m",
        "mean_work_proxy": "mechanical_work_proxy",
        "mean_excessive_impact": "excessive_impact",
        "mean_joint_limit_frequency": "joint_limit_frequency",
        "mean_saturation_frequency": "saturation_frequency",
        "mean_goal_switch_count": "goal_switch_count",
        "mean_push_count": "push_count",
        "mean_push_impulse_magnitude_n_s": "push_impulse_magnitude_n_s",
    }
    numeric_fields = (
        "accumulated_goal_relative_progress_m", "mechanical_work_proxy", "excessive_impact",
        "joint_limit_frequency", "saturation_frequency", "push_impulse_magnitude_n_s",
    )
    for index, episode in enumerate(episodes):
        for field in ("xy_progress_m", "time_to_target_s", "recovery_success", "recovery_time_s"):
            if field not in episode:
                return f"episode {index}: {field} is missing"
        for field in numeric_fields:
            value = _number(episode.get(field))
            if value is None:
                return f"episode {index}: {field} must be finite and numeric"
            if field != "accumulated_goal_relative_progress_m" and value < 0:
                return f"episode {index}: {field} must be nonnegative"
            if field in {"joint_limit_frequency", "saturation_frequency"} and value > 1:
                return f"episode {index}: {field} must be at most one"
        for field in ("goal_switch_count", "push_count", "recovery_attempt_count", "recovery_success_count"):
            if not _nonnegative_integer(episode.get(field)):
                return f"episode {index}: {field} must be a nonnegative integer"
        attempts = episode["recovery_attempt_count"]
        completions = episode["recovery_success_count"]
        if completions > attempts or attempts > episode["push_count"]:
            return f"episode {index}: recovery counts exceed pushes or attempts"
        vector = episode.get("push_impulse_vector_n_s")
        if not isinstance(vector, (list, tuple)) or len(vector) != 3 or any(_number(x) is None for x in vector):
            return f"episode {index}: push_impulse_vector_n_s needs three finite numbers"
        progress = episode["xy_progress_m"]
        if (goal_switch and progress is not None) or (not goal_switch and _number(progress) is None):
            return f"episode {index}: xy_progress_m must be finite outside GoalSwitch and null for GoalSwitch"
        success = episode["success"]
        target_time = _number(episode.get("time_to_target_s"))
        if (success and (target_time is None or target_time < 0)) or (not success and episode.get("time_to_target_s") is not None):
            return f"episode {index}: time_to_target_s must match target success"
        recovery_success = episode["recovery_success"]
        if ((attempts == 0 and recovery_success is not None)
                or (attempts > 0 and (type(recovery_success) is not bool
                                      or recovery_success != (completions > 0)))):
            return f"episode {index}: recovery_success must match eligible attempts and completions"
        recovery_time = _number(episode.get("recovery_time_s"))
        if ((completions == 0 and episode.get("recovery_time_s") is not None)
                or (completions > 0 and (recovery_time is None or recovery_time < 0))):
            return f"episode {index}: recovery_time_s must match completed recoveries"

    expected: dict[str, float | None] = {
        "success_rate": mean(float(episode["success"]) for episode in episodes),
        "mean_xy_progress_m": None if goal_switch else mean(episode["xy_progress_m"] for episode in episodes),
        "mean_time_to_target_s": (
            mean(episode["time_to_target_s"] for episode in episodes if episode["success"])
            if any(episode["success"] for episode in episodes) else None
        ),
        "recovery_attempt_count": float(sum(episode["recovery_attempt_count"] for episode in episodes)),
        "recovery_success_count": float(sum(episode["recovery_success_count"] for episode in episodes)),
    }
    for summary_field, episode_field in mean_fields.items():
        expected[summary_field] = mean(episode[episode_field] for episode in episodes)
    attempts = int(expected["recovery_attempt_count"])
    completions = int(expected["recovery_success_count"])
    expected["recovery_success_rate"] = completions / attempts if attempts else None
    expected["mean_recovery_time_s"] = (
        mean(episode["recovery_time_s"] for episode in episodes if episode["recovery_time_s"] is not None)
        if completions else None
    )
    for field, value in expected.items():
        if field not in seed_row or not _matches(seed_row[field], value):
            return f"per_seed {field} is missing, nonfinite, or inconsistent with episodes"
    for field in ("recovery_attempt_count", "recovery_success_count"):
        if not _nonnegative_integer(seed_row[field]):
            return f"per_seed {field} must be a nonnegative integer"
    return None


def _training_resources(job: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Read one saved training manifest and verify its identity, budget and measured costs."""
    manifest_path = job.get("run_manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        return None, "missing training run_manifest path"
    try:
        saved, saved_sha = _read_json(Path(manifest_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"cannot read training run_manifest: {exc}"
    checkpoint = job.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        return None, "missing training checkpoint"
    if (saved.get("status") != "executed_matrix_training" or saved.get("task") != job.get("task")
            or saved.get("policy") != job.get("policy") or saved.get("seed") != job.get("seed")
            or Path(str(saved.get("checkpoint_path", ""))).resolve() != Path(checkpoint).resolve()
            or Path(manifest_path).resolve().parent != Path(checkpoint).resolve().parent):
        return None, "training run_manifest identity differs from queue job"
    training = config.get("training", {})
    num_envs, horizon = training.get("num_envs", 16), training.get("horizon", 32)
    budget = config.get("interaction_budget")
    if (not _nonnegative_integer(budget) or budget == 0
            or not _nonnegative_integer(num_envs) or num_envs == 0
            or not _nonnegative_integer(horizon) or horizon == 0
            or job.get("interaction_budget") != budget
            or saved.get("requested_interaction_budget") != budget
            or saved.get("num_envs") != num_envs
            or not isinstance(saved.get("ppo_config"), dict)
            or saved["ppo_config"].get("horizon") != horizon):
        return None, "training run_manifest budget or rollout configuration differs from matrix"
    iterations, interactions = saved.get("completed_iterations"), saved.get("actual_environment_interactions")
    expected_iterations = (budget + num_envs * horizon - 1) // (num_envs * horizon)
    if (type(iterations) is not int or iterations != expected_iterations
            or type(interactions) is not int or interactions != iterations * num_envs * horizon):
        return None, "training interaction count does not match the requested budget"
    history = saved.get("history")
    if not isinstance(history, list) or len(history) != iterations:
        return None, "training history length does not match completed iterations"
    update_metrics = (
        "loss", "policy_loss", "value_loss", "entropy", "grad_norm",
        "approx_kl", "attempted_kl", "rejected_step", "accepted_epochs",
    )
    for index, row in enumerate(history):
        if not isinstance(row, dict) or type(row.get("iteration")) is not int or row["iteration"] != index:
            return None, f"training history iteration {index} is missing or out of order"
        if any(_number(row.get(metric)) is None for metric in update_metrics):
            return None, f"training history iteration {index} has a missing or nonfinite metric"
    total, trainable, frozen = (saved.get(field) for field in (
        "model_total_parameters", "model_trainable_parameters", "model_frozen_parameters"
    ))
    if (not all(_nonnegative_integer(value) for value in (total, trainable, frozen))
            or total == 0 or trainable == 0 or trainable + frozen != total):
        return None, "training parameter counts are missing or inconsistent"
    wall_time = _number(saved.get("training_wall_time_s"))
    if wall_time is None or wall_time <= 0:
        return None, "training wall time must be positive and finite"
    queue_copy = job.get("training_metadata")
    if queue_copy is not None:
        if not isinstance(queue_copy, dict) or any(
            not _matches(queue_copy.get(field), saved.get(field))
            for field in (
                "requested_interaction_budget", "actual_environment_interactions", "completed_iterations",
                "model_total_parameters", "model_trainable_parameters", "model_frozen_parameters",
                "training_wall_time_s",
            )
        ):
            return None, "queue training_metadata differs from the saved run_manifest"
    samples = saved.get("memory_samples")
    if not isinstance(samples, list) or not samples:
        return None, "training run_manifest has no memory samples"
    gpu_used: list[float] = []
    ram_used: list[float] = []
    ram_percent: list[float] = []
    process_rss: list[float] = []
    torch_peaks: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("stage"), str) or not sample["stage"]:
            return None, f"memory sample {index} has no stage"
        ram_values = {field: _number(sample.get(field)) for field in (
            "ram_used_gib", "ram_total_gib", "ram_percent", "process_rss_mib"
        )}
        if (any(value is None for value in ram_values.values())
                or ram_values["ram_total_gib"] <= 0
                or not 0 <= ram_values["ram_used_gib"] <= ram_values["ram_total_gib"]
                or not 0 <= ram_values["ram_percent"] <= 100
                or ram_values["process_rss_mib"] < 0):
            return None, f"memory sample {index} has invalid RAM telemetry"
        ram_used.append(ram_values["ram_used_gib"])
        ram_percent.append(ram_values["ram_percent"])
        process_rss.append(ram_values["process_rss_mib"])
        peak = _number(sample.get("torch_cuda_peak_allocated_mib"))
        if peak is not None:
            if peak < 0:
                return None, f"memory sample {index} has invalid PyTorch CUDA peak"
            torch_peaks.append(peak)
        devices = sample.get("gpu_devices")
        if not isinstance(devices, list):
            return None, f"memory sample {index} has invalid GPU device list"
        if sample.get("gpu_telemetry") == "ok" and not devices:
            return None, f"memory sample {index} claims GPU telemetry without a device"
        for device in devices:
            if not isinstance(device, dict):
                return None, f"memory sample {index} has invalid GPU device telemetry"
            used, capacity = _number(device.get("gpu_used_mib")), _number(device.get("gpu_total_mib"))
            if used is None or capacity is None or capacity <= 0 or not 0 <= used <= capacity:
                return None, f"memory sample {index} has invalid device-wide GPU usage"
            gpu_used.append(used)
    if not gpu_used:
        return None, "training run_manifest has no measured device-wide GPU usage"
    return {
        "seed": job["seed"], "run_manifest": str(Path(manifest_path).resolve()),
        "run_manifest_sha256": saved_sha, "checkpoint": str(Path(checkpoint).resolve()),
        "model_total_parameters": total, "model_trainable_parameters": trainable,
        "model_frozen_parameters": frozen, "actual_environment_interactions": interactions,
        "training_wall_time_s": wall_time, "memory_sample_count": len(samples),
        "gpu_device_sample_count": len(gpu_used),
        "max_sampled_gpu_used_mib": max(gpu_used),
        "max_sampled_ram_used_gib": max(ram_used),
        "max_sampled_ram_percent": max(ram_percent),
        "max_sampled_process_rss_mib": max(process_rss),
        "max_stage_torch_cuda_peak_allocated_mib": max(torch_peaks) if torch_peaks else None,
    }, None


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {"n_training_seeds": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None}


def summarize(path: Path) -> dict[str, Any]:
    manifest, manifest_sha = _read_json(path)
    config = manifest.get("config", {})
    conditions = config.get("conditions", [])
    seeds = config.get("seeds", [])
    evaluation = config.get("evaluation", {})
    scenarios = evaluation.get("scenario_tasks", [config.get("task")])
    if not conditions or not seeds or not scenarios or not isinstance(manifest.get("jobs"), list):
        raise ValueError("Matrix manifest has no complete condition/seed/scenario design.")
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    jobs = manifest["jobs"]
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []
    rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    resource_rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    provenance: dict[str, str] = {}
    training_provenance: dict[str, str] = {}
    schedule_hashes: dict[str, set[str]] = defaultdict(set)
    pairing_hashes: dict[str, dict[int, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    complete_jobs = 0
    for job in jobs:
        condition, seed = job.get("condition"), job.get("seed")
        key = (condition, seed)
        if key not in expected or key in seen or job.get("id") != f"{condition}__seed-{seed}":
            errors.append(f"Unexpected or duplicate job identity: {job.get('id')!r}")
            continue
        seen.add(key)
        if job.get("status") != "passed":
            continue
        job_results: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            record = job.get("evaluations", {}).get(scenario, {})
            if record.get("status") != "passed" or not record.get("result_file"):
                errors.append(f"{job['id']}: missing passed evaluation for {scenario}")
                break
            result_path = Path(record["result_file"])
            try:
                result, result_sha = _read_json(result_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{job['id']}: cannot read {scenario}: {exc}")
                break
            scenario_data = result.get("scenario", {})
            per_seed = result.get("per_seed", [])
            if (result.get("status") != "executed" or result.get("task") != scenario
                    or result.get("training_seed") != seed
                    or result.get("evaluation_seed") != evaluation.get("seed", 101)
                    or result.get("n_episodes") != evaluation.get("episodes", 16)
                    or result.get("checkpoint") != job.get("checkpoint")
                    or not isinstance(per_seed, list) or len(per_seed) != 1
                    or not isinstance(per_seed[0], dict) or per_seed[0].get("seed") != seed
                    or per_seed[0].get("episodes") != result.get("n_episodes")
                    or result.get("n_independent_training_seeds") != 1
                    or scenario_data.get("evaluation_protocol") != evaluation.get("protocol", "default")):
                errors.append(f"{job['id']}: evaluation metadata mismatch for {scenario}")
                break
            episodes = result.get("episodes", [])
            if (not isinstance(episodes, list) or len(episodes) != result["n_episodes"]
                    or any(not isinstance(episode, dict) or episode.get("episode_id") != index
                           or episode.get("seed") != seed or type(episode.get("success")) is not bool
                           for index, episode in enumerate(episodes))):
                errors.append(f"{job['id']}: episode count mismatch for {scenario}")
                break
            metric_error = _validate_metric_rows(result, scenario)
            if metric_error:
                errors.append(f"{job['id']}: {metric_error} for {scenario}")
                break
            if evaluation.get("protocol") == "heldout_v1":
                schedule_sha = (scenario_data.get("schedule") or {}).get("sha256")
                if not isinstance(schedule_sha, str) or len(schedule_sha) != 64:
                    errors.append(f"{job['id']}: missing schedule hash for {scenario}")
                    break
                if any(not episode.get("paired_plan_sha256") or not episode.get("initial_state_sha256")
                       for episode in episodes):
                    errors.append(f"{job['id']}: missing episode pairing hashes for {scenario}")
                    break
            job_results[scenario] = {"result": result, "sha256": result_sha, "path": str(result_path)}
        if len(job_results) != len(scenarios):
            continue
        resources, resource_error = _training_resources(job, config)
        if resource_error:
            errors.append(f"{job['id']}: {resource_error}")
            continue
        complete_jobs += 1
        resource_rows[condition][seed] = resources
        training_provenance[job["id"]] = resources["run_manifest_sha256"]
        for scenario, item in job_results.items():
            result = item["result"]
            rows[scenario][condition][seed] = result["per_seed"][0]
            provenance[f"{job['id']}::{scenario}"] = item["sha256"]
            if evaluation.get("protocol") == "heldout_v1":
                schedule_hashes[scenario].add(result["scenario"]["schedule"]["sha256"])
                for index, episode in enumerate(result["episodes"]):
                    pairing_hashes[scenario][index]["plan"].add(episode["paired_plan_sha256"])
                    pairing_hashes[scenario][index]["initial_state"].add(episode["initial_state_sha256"])
    if seen != expected:
        errors.append(f"Missing {len(expected - seen)} design jobs from manifest.")
    pairing = {}
    for scenario in scenarios:
        schedule_ok = len(schedule_hashes[scenario]) <= 1
        plan_ok = all(len(item["plan"]) <= 1 for item in pairing_hashes[scenario].values())
        initial_ok = all(len(item["initial_state"]) <= 1 for item in pairing_hashes[scenario].values())
        pairing[scenario] = {"schedule_shared": schedule_ok, "episode_plan_shared": plan_ok,
                             "initial_state_shared": initial_ok,
                             "schedule_sha256": next(iter(schedule_hashes[scenario]), None)}
        if not (schedule_ok and plan_ok and initial_ok):
            errors.append(f"{scenario}: paired evaluation schedule or initial state differs across jobs.")
    summaries: dict[str, Any] = {}
    differences: dict[str, Any] = {}
    for scenario in scenarios:
        summaries[scenario] = {}
        differences[scenario] = {}
        for condition in conditions:
            seed_rows = rows[scenario][condition]
            summaries[scenario][condition] = {
                "training_seeds": sorted(seed_rows),
                "metrics": {metric: _stats([value for seed in sorted(seed_rows)
                                            if (value := _number(seed_rows[seed].get(metric))) is not None])
                            for metric in METRICS},
            }
            if condition == BASELINE or BASELINE not in conditions:
                continue
            paired = sorted(set(rows[scenario][BASELINE]) & set(seed_rows))
            differences[scenario][condition] = {
                "reference": BASELINE, "paired_training_seeds": paired,
                "metrics": {metric: _stats([
                    target - source for seed in paired
                    if (target := _number(seed_rows[seed].get(metric))) is not None
                    and (source := _number(rows[scenario][BASELINE][seed].get(metric))) is not None
                ]) for metric in METRICS},
            }
    resources_by_condition = {
        condition: {
            "training_seeds": sorted(resource_rows[condition]),
            "per_seed": {str(seed): resource_rows[condition][seed] for seed in sorted(resource_rows[condition])},
            "metrics": {
                metric: _stats([
                    value for seed in sorted(resource_rows[condition])
                    if (value := _number(resource_rows[condition][seed].get(metric))) is not None
                ]) for metric in RESOURCE_METRICS
            },
        }
        for condition in conditions
    }
    return {
        "status": "complete" if complete_jobs == len(expected) and not errors else "incomplete",
        "manifest": str(path.resolve()), "manifest_sha256": manifest_sha,
        "analysis_source_sha256": _digest_bytes(Path(__file__).read_bytes()),
        "source_execution_fingerprint": manifest.get("execution_source_fingerprint"),
        "expected_jobs": len(expected), "validated_jobs": complete_jobs,
        "missing_or_unfinished_jobs": [job.get("id") for job in jobs if job.get("status") != "passed"],
        "validation_errors": errors, "evaluation_result_sha256": provenance,
        "training_manifest_sha256": training_provenance,
        "training_resources": {
            "measurement_note": (
                "max_sampled_gpu_used_mib and max_sampled_ram_percent are the largest observed "
                "device-wide GPU and system RAM snapshots within each training job, not continuous peaks. "
                "max_stage_torch_cuda_peak_allocated_mib is a PyTorch stage peak and omits Isaac/driver allocations. "
                "Condition means and sample SD use independent training seeds."
            ),
            "by_condition": resources_by_condition,
        },
        "pairing": pairing, "by_scenario": summaries, "paired_differences_vs_original": differences,
        "interpretation": "Means and sample SD use independent training seeds. Episodes are not independent replicates; this is descriptive, not a significance test.",
    }


def _cell(stats: dict[str, Any] | None) -> str:
    if stats is None:
        return "—"
    value = f"{stats['mean']:.3f}"
    return f"{value} ± {stats['sample_sd']:.3f}" if stats["sample_sd"] is not None else value


def _resource_cell(stats: dict[str, Any] | None, *, digits: int = 1) -> str:
    if stats is None:
        return "—"
    value = f"{stats['mean']:,.{digits}f}"
    return f"{value} ± {stats['sample_sd']:,.{digits}f}" if stats["sample_sd"] is not None else value


def markdown(report: dict[str, Any]) -> str:
    lines = ["# G1 matrix comparison", "", f"Status: **{report['status']}**; validated jobs: "
             f"{report['validated_jobs']}/{report['expected_jobs']}.", "",
             "Values are means ± sample SD across independent training seeds. A dash means the metric was unavailable.", ""]
    for scenario, conditions in report["by_scenario"].items():
        lines += [f"## {scenario}", "", "| Condition | Seeds | Success rate | XY progress (m) | Goal-relative progress (m) | Time to target (s) | Recovery rate | Recovery time (s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for condition, entry in conditions.items():
            metrics = entry["metrics"]
            lines.append(f"| {condition} | {len(entry['training_seeds'])} | {_cell(metrics['success_rate'])} | "
                         f"{_cell(metrics['mean_xy_progress_m'])} | {_cell(metrics['mean_accumulated_goal_relative_progress_m'])} | "
                         f"{_cell(metrics['mean_time_to_target_s'])} | {_cell(metrics['recovery_success_rate'])} | "
                         f"{_cell(metrics['mean_recovery_time_s'])} |")
        lines += ["", "| Condition | Work proxy | Excessive impact | Joint-limit frequency | Saturation frequency | Goal switches | Pushes | Push impulse (N·s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for condition, entry in conditions.items():
            metrics = entry["metrics"]
            lines.append(f"| {condition} | {_cell(metrics['mean_work_proxy'])} | "
                         f"{_cell(metrics['mean_excessive_impact'])} | "
                         f"{_cell(metrics['mean_joint_limit_frequency'])} | "
                         f"{_cell(metrics['mean_saturation_frequency'])} | "
                         f"{_cell(metrics['mean_goal_switch_count'])} | "
                         f"{_cell(metrics['mean_push_count'])} | "
                         f"{_cell(metrics['mean_push_impulse_magnitude_n_s'])} |")
        lines.append("")
    resources = report["training_resources"]
    lines += ["## Training cost and memory", "", resources["measurement_note"], "",
              "| Condition | Seeds | Total params | Trainable | Frozen | Interactions | Train time (s) | Max sampled GPU used (MiB) | Max sampled RAM used (GiB) | Max sampled RAM (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition, entry in resources["by_condition"].items():
        metrics = entry["metrics"]
        lines.append(f"| {condition} | {len(entry['training_seeds'])} | "
                     f"{_resource_cell(metrics['model_total_parameters'], digits=0)} | "
                     f"{_resource_cell(metrics['model_trainable_parameters'], digits=0)} | "
                     f"{_resource_cell(metrics['model_frozen_parameters'], digits=0)} | "
                     f"{_resource_cell(metrics['actual_environment_interactions'], digits=0)} | "
                     f"{_resource_cell(metrics['training_wall_time_s'])} | "
                     f"{_resource_cell(metrics['max_sampled_gpu_used_mib'])} | "
                     f"{_resource_cell(metrics['max_sampled_ram_used_gib'], digits=2)} | "
                     f"{_resource_cell(metrics['max_sampled_ram_percent'])} |")
    lines.append("")
    if report["validation_errors"]:
        lines += ["## Validation errors", ""] + [f"- {error}" for error in report["validation_errors"]] + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "validated_jobs": report["validated_jobs"],
                      "expected_jobs": report["expected_jobs"], "validation_errors": report["validation_errors"]}, indent=2))
    return 1 if report["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
