#!/usr/bin/env python3
"""Build complete JSON and Markdown reports from a Crazyflie matrix queue.

This module is deliberately CPU-only.  It reads the queue, training manifests,
and evaluation JSON files; it never imports Isaac Lab or loads a checkpoint.
Pending, failed, and missing cells remain visible in every report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

import drone_run_matrix as matrix_runner


VALID_JOB_STATUSES = ("pending", "running", "completed", "failed", "paused", "cancelled")
BASELINE_CONTROLLER = "frozen_lif_original"

# Values are stored under stable flat names in the comparison report.  Each
# tuple after the name is an accepted path in an evaluator's ``summary``.
METRIC_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "success_rate": (("success_rate",),),
    "failure_rate": (("failure_rate",),),
    "mean_time_to_first_success_fixed_horizon_censored_s": (
        ("mean_time_to_first_success_fixed_horizon_censored_s",),
        ("mean_time_to_first_success_s",),
        ("time_to_first_success", "mean_fixed_horizon_censored_s"),
    ),
    "mean_final_goal_error_m": (("mean_final_goal_error_m",),),
    "mean_integrated_goal_error_m_s": (("mean_integrated_goal_error_m_s",),),
    "mean_speed_inside_target_region_m_s": (("mean_speed_inside_target_region_m_s",),),
    "crash_rate": (("crash_rate",),),
    "out_of_bounds_rate": (("out_of_bounds_rate",),),
    "invalid_state_rate": (("invalid_state_rate",),),
    "mean_command_effort": (("mean_command_effort",),),
    "mean_command_smoothness": (("mean_command_smoothness",),),
    "mean_aggregate_wrench_mechanical_work_proxy_j": (
        ("mean_aggregate_wrench_mechanical_work_proxy_j",),
    ),
    "switch_success_rate": (
        ("switch_metrics", "success_rate"),
        ("switch_metrics", "switch_success_rate"),
    ),
    "switch_mean_fixed_horizon_censored_latency_s": (
        ("switch_metrics", "mean_fixed_horizon_censored_latency_s"),
    ),
    "gust_unconditional_recovery_rate": (
        ("gust_metrics", "unconditional_recovery_rate"),
        ("gust_metrics", "recovery_success_rate"),
    ),
    "gust_conditional_recovery_rate": (
        ("gust_metrics", "conditional_recovery_rate"),
    ),
    "gust_unconditional_mean_fixed_horizon_censored_latency_s": (
        ("gust_metrics", "unconditional_recovery_latency", "mean_fixed_horizon_censored_s"),
    ),
    "gust_conditional_mean_fixed_horizon_censored_latency_s": (
        ("gust_metrics", "conditional_recovery_latency", "mean_fixed_horizon_censored_s"),
    ),
    "gust_mean_max_displacement_m": (("gust_metrics", "mean_max_displacement_m"),),
    "gust_mean_post_gust_error_integral_m_s": (
        ("gust_metrics", "mean_post_gust_error_integral_m_s"),
    ),
}

CORE_METRICS = (
    "success_rate",
    "mean_time_to_first_success_fixed_horizon_censored_s",
    "mean_final_goal_error_m",
    "mean_integrated_goal_error_m_s",
    "crash_rate",
    "out_of_bounds_rate",
    "invalid_state_rate",
    "mean_command_effort",
    "mean_command_smoothness",
    "mean_aggregate_wrench_mechanical_work_proxy_j",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value, sha256(data).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _metric_values(summary: Mapping[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, alternatives in METRIC_PATHS.items():
        value = None
        for path in alternatives:
            candidate = _nested(summary, path)
            if candidate is not None:
                value = _finite_number(candidate)
                break
        values[name] = value
    return values


def _sample_stats(values_by_seed: Mapping[int, float | None], expected_seeds: Sequence[int]) -> dict[str, Any]:
    present = [float(values_by_seed[seed]) for seed in expected_seeds if values_by_seed.get(seed) is not None]
    return {
        "per_seed": {str(seed): values_by_seed.get(seed) for seed in expected_seeds},
        "n_training_seeds": len(present),
        "expected_training_seeds": len(expected_seeds),
        "complete": len(present) == len(expected_seeds),
        "mean": mean(present) if present else None,
        "median": median(present) if present else None,
        "std": stdev(present) if len(present) > 1 else None,
        "sample_std": stdev(present) if len(present) > 1 else None,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _paired_bootstrap_ci(
    differences: Sequence[float], *, samples: int, seed: int
) -> dict[str, Any]:
    if len(differences) < 2:
        raise ValueError("Paired bootstrap needs at least two paired seeds")
    generator = random.Random(seed)
    count = len(differences)
    bootstrap_means = sorted(
        mean(differences[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "confidence_level": 0.95,
        "lower": _quantile(bootstrap_means, 0.025),
        "upper": _quantile(bootstrap_means, 0.975),
        "resamples": samples,
        "method": "paired_training_seed_percentile_bootstrap_of_mean_difference",
    }


def _bootstrap_seed(base: int, scenario: str, controller: str, metric: str) -> int:
    material = f"{base}|{scenario}|{controller}|{metric}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_design(queue: Mapping[str, Any]) -> tuple[list[str], list[int], list[str], int, list[str]]:
    errors: list[str] = []
    if queue.get("schema_version") != 1:
        errors.append("queue.schema_version must be 1")
    label = queue.get("label")
    if label not in {"integration", "main"}:
        errors.append("queue.label must be 'integration' or 'main'")
    config = queue.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Queue has no config object")
    expected_memory_acceptance = matrix_runner.memory_acceptance_contract_payload()
    if config.get("memory_acceptance") != expected_memory_acceptance:
        errors.append(
            "config.memory_acceptance does not match crazyflie_memory_acceptance_v2"
        )
    resource_limits = queue.get("resource_limits")
    if not isinstance(resource_limits, Mapping):
        errors.append("queue.resource_limits must be an object")
    else:
        required_limits = {
            "policy_version": expected_memory_acceptance["policy_version"],
            "device_gpu_used_mib_exclusive": expected_memory_acceptance[
                "device_gpu_used_mib_exclusive"
            ],
            "system_ram_percent_exclusive": expected_memory_acceptance[
                "system_ram_percent_exclusive"
            ],
            "rss_growth_tolerance_mib": expected_memory_acceptance[
                "rss_growth_tolerance_mib"
            ],
            "rss_growth_disposition": "warning_only",
            "swap_out_growth_tolerance_mib": expected_memory_acceptance[
                "swap_out_growth_tolerance_mib"
            ],
            "sustained_paging_disposition": "hard_failure",
            "gpu_telemetry_required_for_cuda": True,
            "finite_numeric_telemetry_required": True,
        }
        for field, expected in required_limits.items():
            if resource_limits.get(field) != expected:
                errors.append(
                    f"queue.resource_limits.{field} must be {expected!r}, "
                    f"found {resource_limits.get(field)!r}"
                )
    controllers = config.get("controllers")
    seeds = config.get("seeds")
    evaluation = config.get("evaluation")
    if not isinstance(controllers, list) or not controllers or any(not isinstance(item, str) or not item for item in controllers):
        raise ValueError("config.controllers must be a non-empty list of names")
    if len(set(controllers)) != len(controllers):
        raise ValueError("config.controllers contains duplicates")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in seeds)
    ):
        raise ValueError("config.seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("config.seeds contains duplicates")
    if not isinstance(evaluation, Mapping):
        raise ValueError("config.evaluation must be an object")
    scenarios = evaluation.get("scenarios")
    episodes = evaluation.get("episodes_per_scenario")
    if not isinstance(scenarios, list) or not scenarios or any(not isinstance(item, str) or not item for item in scenarios):
        raise ValueError("config.evaluation.scenarios must be a non-empty list")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("config.evaluation.scenarios contains duplicates")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("config.evaluation.episodes_per_scenario must be a positive integer")
    expected_controllers = list(matrix_runner.CONTROLLERS)
    expected_scenarios = list(matrix_runner.SCENARIOS)
    expected_seeds = [0, 1, 2, 3, 4] if label == "main" else [0]
    expected_episodes = 16 if label == "main" else 2
    if controllers != expected_controllers:
        errors.append("design must contain the exact four controllers in declared order")
    if scenarios != expected_scenarios:
        errors.append("design must contain the exact three scenarios in declared order")
    if seeds != expected_seeds or episodes != expected_episodes:
        errors.append(
            f"{label} design must use seeds {expected_seeds} and {expected_episodes} episodes per scenario"
        )
    survival_present = "survival_first_contract" in config
    balanced_present = "balanced_task_contract" in config
    if survival_present == balanced_present:
        errors.append("config must contain exactly one versioned Crazyflie task contract")
    elif balanced_present:
        if config.get("contract_profile") != matrix_runner.CONTRACT_PROFILE_BALANCED_V3:
            errors.append("config balanced contract profile changed")
        if config.get("task") != matrix_runner.BALANCED_TASK:
            errors.append("config task differs from the balanced-v3 mixed training task")
        if config.get("balanced_task_contract") != matrix_runner.balanced_task_contract_payload():
            errors.append("config balanced-v3 reward/reset curriculum contract changed")
    else:
        if config.get("task") != matrix_runner.SURVIVAL_TASK:
            errors.append("config task differs from the historical survival-v2 training task")
        if config.get("survival_first_contract") != matrix_runner.survival_first_contract_payload():
            errors.append("config survival-first reward/reset curriculum contract changed")
    expected_total = 5_000_000 if label == "main" else 50
    if config.get("total_interactions") != expected_total:
        errors.append(f"{label} design must use exactly {expected_total:,} interactions per job")
    expected_training = (
        {
            "num_envs": 4,
            "horizon": 100,
            "ppo_epochs": 2,
            "microbatch_size": 4,
            "learning_rate": 3.0e-5,
        }
        if label == "main"
        else {
            "num_envs": 1,
            "horizon": 25,
            "ppo_epochs": 1,
            "microbatch_size": 1,
            "learning_rate": 1.0e-4,
        }
    )
    training = config.get("training")
    if not isinstance(training, Mapping):
        errors.append("config.training must be an object")
    else:
        for field, expected_value in expected_training.items():
            if training.get(field) != expected_value:
                errors.append(
                    f"{label} design training.{field} must be {expected_value}, "
                    f"found {training.get(field)!r}"
                )
    planned_jobs = len(expected_controllers) * len(expected_seeds)
    planned_bundles = planned_jobs * len(expected_scenarios)
    planned_episodes = planned_bundles * expected_episodes
    for key, expected in (
        ("job_count", planned_jobs),
        ("evaluation_bundle_count", planned_bundles),
        ("predicted_evaluation_episodes", planned_episodes),
        ("sequential_process_limit", 1),
    ):
        if queue.get(key) != expected:
            errors.append(f"queue.{key} must be {expected}, found {queue.get(key)!r}")
    if queue.get("config_sha256") != matrix_runner.canonical_sha256(dict(config)):
        errors.append("queue config_sha256 does not match its embedded config")
    return list(controllers), list(seeds), list(scenarios), episodes, errors


def _memory_warning_messages(gate: Any) -> list[str]:
    if not isinstance(gate, Mapping):
        return []
    warnings = gate.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [message for message in warnings if isinstance(message, str) and message]


def _training_details(job: Mapping[str, Any], *, require_complete: bool) -> tuple[dict[str, Any], list[str]]:
    path = Path(str(job.get("run_dir", ""))) / "training_manifest.json"
    details: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": None,
        "status": None,
        "environment_interactions": None,
        "training_wall_time_s": None,
        "parameter_counts": None,
        "memory_gate": None,
        "memory_warning_count": 0,
        "memory_warnings": [],
        "core_checksum_before": None,
        "core_checksum_after": None,
        "history_reference": None,
    }
    errors: list[str] = []
    if not path.is_file():
        if require_complete:
            errors.append(f"missing training manifest: {path}")
        return details, errors
    try:
        value, digest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read training manifest {path}: {exc}")
        return details, errors
    report = value.get("controller_report")
    memory_gate = value.get("memory_gate")
    details.update(
        {
            "sha256": digest,
            "status": value.get("status"),
            "environment_interactions": value.get("environment_interactions"),
            "training_wall_time_s": value.get("training_wall_time_s"),
            "parameter_counts": report if isinstance(report, Mapping) else None,
            "memory_gate": memory_gate if isinstance(memory_gate, Mapping) else None,
            "memory_warning_count": len(_memory_warning_messages(memory_gate)),
            "memory_warnings": _memory_warning_messages(memory_gate),
            "core_checksum_before": value.get("core_checksum_before"),
            "core_checksum_after": value.get("core_checksum_after"),
            "history_reference": (
                value.get("history_reference")
                if isinstance(value.get("history_reference"), Mapping)
                else None
            ),
            "checkpoint": value.get("checkpoint"),
            "fingerprint": value.get("fingerprint"),
        }
    )
    if require_complete:
        if not matrix_runner._valid_training(dict(job)):
            errors.append(
                "training artifact failed the queue runner's strict checkpoint, fingerprint, "
                "budget, manifest, SHA-256, or memory validation"
            )
        if value.get("status") != "completed":
            errors.append(f"training manifest status is {value.get('status')!r}, not 'completed'")
        for key in ("controller", "seed", "fingerprint"):
            expected = job.get({"controller": "controller", "seed": "seed", "fingerprint": "expected_fingerprint"}[key])
            if value.get(key) != expected:
                errors.append(f"training manifest {key} differs from queue")
        if value.get("environment_interactions") != job.get("total_interactions"):
            errors.append("training manifest interaction count differs from the planned exact budget")
        checkpoint = job.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            errors.append(f"missing checkpoint: {checkpoint!r}")
        elif value.get("checkpoint") and Path(str(value["checkpoint"])).resolve() != Path(checkpoint).resolve():
            errors.append("training manifest checkpoint differs from queue")
        if not isinstance(report, Mapping):
            errors.append("training manifest has no controller_report")
        else:
            for key in (
                "actor_trainable_parameters",
                "critic_trainable_parameters",
                "total_trainable_parameters",
                "frozen_parameters",
                "total_dynamic_state_per_environment",
            ):
                if isinstance(report.get(key), bool) or not isinstance(report.get(key), int) or report[key] < 0:
                    errors.append(f"controller_report.{key} is missing or invalid")
        if not isinstance(memory_gate, Mapping) or memory_gate.get("passed") is not True:
            errors.append("training memory gate is missing or did not pass")
        before, after = value.get("core_checksum_before"), value.get("core_checksum_after")
        if before is not None and before != after:
            errors.append("frozen core checksum changed during training")
    return details, errors


def _evaluation_result(
    bundle: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    expected_episodes: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    output = bundle.get("output")
    if not isinstance(output, str) or not output:
        return None, ["evaluation bundle has no output path"]
    path = Path(output)
    if not path.is_file():
        return None, [f"missing evaluation result: {path}"]
    try:
        result, digest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"cannot read evaluation result {path}: {exc}"]
    scenario = bundle.get("scenario")
    checks = {
        "status": (result.get("status"), "completed"),
        "scenario": (result.get("scenario"), scenario),
        "training_seed": (result.get("training_seed"), job.get("seed")),
        "fingerprint": (result.get("fingerprint"), job.get("expected_fingerprint")),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"evaluation {key} differs from queue: {actual!r} != {expected!r}")
    if not matrix_runner._valid_evaluation(dict(bundle), dict(job), expected_episodes):
        errors.append(
            "evaluation artifact failed the queue runner's strict checkpoint, protocol, plan, "
            "fingerprint, SHA-256, episode-row, or recomputed-summary validation"
        )
    if result.get("controller") is not None and result.get("controller") != job.get("controller"):
        errors.append("evaluation controller differs from queue")
    checkpoint = result.get("checkpoint")
    if checkpoint is not None and Path(str(checkpoint)).resolve() != Path(str(job.get("checkpoint"))).resolve():
        errors.append("evaluation checkpoint differs from queue")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        errors.append(
            f"evaluation has {len(episodes) if isinstance(episodes, list) else 'invalid'} episodes; "
            f"expected {expected_episodes}"
        )
        episodes = []
    else:
        ids = [row.get("episode_id") if isinstance(row, Mapping) else None for row in episodes]
        if ids != list(range(expected_episodes)):
            errors.append("evaluation episode IDs are not exactly the planned ordered range")
        for index, row in enumerate(episodes):
            if not isinstance(row, Mapping) or not isinstance(row.get("success"), bool):
                errors.append(f"evaluation episode {index} lacks a Boolean success outcome")
                break
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        # Accept the aggregation itself at top level for forward/backward
        # compatibility, while still requiring the core aggregate fields.
        summary = result if any(key in result for key in CORE_METRICS) else None
    if not isinstance(summary, Mapping):
        errors.append("evaluation has no summary object")
        summary = {}
    values = _metric_values(summary)
    for metric in CORE_METRICS:
        if metric == "mean_speed_inside_target_region_m_s":
            continue
        if values[metric] is None:
            errors.append(f"evaluation summary metric {metric} is missing or nonfinite")
    if summary.get("success_denominator") != expected_episodes:
        errors.append("evaluation summary does not retain the fixed success denominator")
    if summary.get("expected_episode_count") != expected_episodes:
        errors.append("evaluation summary expected_episode_count differs from the protocol")
    if summary.get("episode_count") != expected_episodes:
        errors.append("evaluation summary episode_count differs from the protocol")
    scenario_text = str(scenario).lower()
    if "switch" in scenario_text:
        switch = summary.get("switch_metrics")
        if not isinstance(switch, Mapping):
            errors.append("switch evaluation has no switch_metrics summary")
        else:
            for metric in ("switch_success_rate", "switch_mean_fixed_horizon_censored_latency_s"):
                if values[metric] is None:
                    errors.append(f"switch evaluation metric {metric} is missing or nonfinite")
    if "gust" in scenario_text:
        gust = summary.get("gust_metrics")
        if not isinstance(gust, Mapping):
            errors.append("gust evaluation has no gust_metrics summary")
        else:
            for metric in (
                "gust_unconditional_recovery_rate",
                "gust_unconditional_mean_fixed_horizon_censored_latency_s",
            ):
                if values[metric] is None:
                    errors.append(f"gust evaluation metric {metric} is missing or nonfinite")
            conditional_denominator = gust.get("conditional_recovery_denominator")
            if (
                isinstance(conditional_denominator, int)
                and conditional_denominator > 0
                and values["gust_conditional_recovery_rate"] is None
            ):
                errors.append("conditional gust recovery rate is missing despite eligible gusts")
    if episodes and values["success_rate"] is not None:
        observed = sum(bool(row["success"]) for row in episodes) / expected_episodes
        if not math.isclose(values["success_rate"], observed, rel_tol=1e-9, abs_tol=1e-9):
            errors.append("evaluation summary success_rate disagrees with episode rows")
    memory_gate = result.get("memory_gate")
    memory_warnings = _memory_warning_messages(memory_gate)
    return {
        "path": str(path),
        "sha256": digest,
        "result": result,
        "summary": dict(summary),
        "metrics": values,
        "episode_count": len(episodes),
        "memory_gate": dict(memory_gate) if isinstance(memory_gate, Mapping) else None,
        "memory_warning_count": len(memory_warnings),
        "memory_warnings": memory_warnings,
    }, errors


def _resource_projection(training: Mapping[str, Any]) -> dict[str, Any]:
    counts = training.get("parameter_counts")
    memory = training.get("memory_gate")
    return {
        "environment_interactions": training.get("environment_interactions"),
        "training_wall_time_s": training.get("training_wall_time_s"),
        "actor_trainable_parameters": counts.get("actor_trainable_parameters") if isinstance(counts, Mapping) else None,
        "critic_trainable_parameters": counts.get("critic_trainable_parameters") if isinstance(counts, Mapping) else None,
        "total_trainable_parameters": counts.get("total_trainable_parameters") if isinstance(counts, Mapping) else None,
        "frozen_parameters": counts.get("frozen_parameters") if isinstance(counts, Mapping) else None,
        "model_total_parameters": counts.get("model_total_parameters") if isinstance(counts, Mapping) else None,
        "total_dynamic_state_per_environment": counts.get("total_dynamic_state_per_environment") if isinstance(counts, Mapping) else None,
        "actor_parameter_deviation_fraction": counts.get("actor_parameter_deviation_fraction") if isinstance(counts, Mapping) else None,
        "max_system_ram_percent": memory.get("max_system_ram_percent") if isinstance(memory, Mapping) else None,
        "max_device_gpu_used_mib": memory.get("max_device_gpu_used_mib") if isinstance(memory, Mapping) else None,
        "max_process_rss_mib": memory.get("max_process_rss_mib") if isinstance(memory, Mapping) else None,
        "max_torch_allocated_mib": memory.get("max_torch_allocated_mib") if isinstance(memory, Mapping) else None,
        "max_torch_reserved_mib": memory.get("max_torch_reserved_mib") if isinstance(memory, Mapping) else None,
        "memory_gate_passed": memory.get("passed") if isinstance(memory, Mapping) else None,
        "memory_policy_version": memory.get("policy_version") if isinstance(memory, Mapping) else None,
        "memory_warning_count": len(_memory_warning_messages(memory)),
        "memory_warnings": _memory_warning_messages(memory),
    }


def _comparison(
    *,
    label: str,
    controllers: Sequence[str],
    seeds: Sequence[int],
    scenarios: Sequence[str],
    metric_rows: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float | None]]]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    differences: dict[str, Any] = {}
    for scenario in scenarios:
        by_scenario[scenario] = {}
        for controller in controllers:
            seed_rows = metric_rows.get(scenario, {}).get(controller, {})
            by_scenario[scenario][controller] = {
                "completed_training_seeds": sorted(seed_rows),
                "expected_training_seeds": list(seeds),
                "per_seed": {
                    str(seed): dict(seed_rows[seed]) if seed in seed_rows else None for seed in seeds
                },
                "metrics": {
                    metric: _sample_stats(
                        {seed: seed_rows.get(seed, {}).get(metric) for seed in seeds}, seeds
                    )
                    for metric in METRIC_PATHS
                },
            }
        differences[scenario] = {}
        baseline_rows = metric_rows.get(scenario, {}).get(BASELINE_CONTROLLER, {})
        for controller in controllers:
            if controller == BASELINE_CONTROLLER:
                continue
            target_rows = metric_rows.get(scenario, {}).get(controller, {})
            metrics: dict[str, Any] = {}
            for metric in METRIC_PATHS:
                paired: dict[int, float] = {}
                for seed in seeds:
                    baseline = baseline_rows.get(seed, {}).get(metric)
                    target = target_rows.get(seed, {}).get(metric)
                    if baseline is not None and target is not None:
                        paired[seed] = float(target) - float(baseline)
                stats = _sample_stats(paired, seeds)
                if not paired:
                    ci, reason = None, "metric_not_applicable_or_unavailable"
                elif label != "main":
                    ci, reason = None, "integration_matrix_is_not_an_inferential_comparison"
                elif set(paired) != set(seeds):
                    ci, reason = None, "complete_paired_main_seed_set_required"
                elif len(paired) < 2:
                    ci, reason = None, "at_least_two_paired_seeds_required"
                else:
                    ci = _paired_bootstrap_ci(
                        [paired[seed] for seed in seeds],
                        samples=bootstrap_samples,
                        seed=_bootstrap_seed(bootstrap_seed, scenario, controller, metric),
                    )
                    reason = None
                stats["bootstrap_95_ci"] = ci
                stats["bootstrap_unavailable_reason"] = reason
                metrics[metric] = stats
            differences[scenario][controller] = {
                "reference_controller": BASELINE_CONTROLLER,
                "difference_direction": f"{controller} minus {BASELINE_CONTROLLER}",
                "metrics": metrics,
            }
    return {
        "baseline_controller": BASELINE_CONTROLLER,
        "by_scenario": by_scenario,
        "paired_differences_vs_original": differences,
        "bootstrap": {
            "confidence_level": 0.95,
            "resamples": bootstrap_samples,
            "base_seed": bootstrap_seed,
            "availability_rule": "complete five-seed main pairing and at least two finite pairs",
        },
    }


def summarize(
    path: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260916,
) -> dict[str, Any]:
    """Read one queue and return a complete, failure-preserving report."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    queue_path = Path(path).resolve()
    queue, queue_sha = _read_json(queue_path)
    controllers, seeds, scenarios, episodes_per_scenario, design_errors = _validate_design(queue)
    current_identity_errors: list[str] = []
    try:
        configured_output = Path(str(queue.get("output", ""))).resolve()
        if configured_output != queue_path:
            raise ValueError(
                f"queue.output resolves to {configured_output}, not the summarized queue {queue_path}"
            )
        config_path = Path(str(queue.get("config_path", ""))).resolve()
        current_config = matrix_runner.validate_config(config_path)
        matrix_runner.validate_resume_queue(queue, current_config, queue_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        current_identity_errors.append(
            "queue is incompatible with current source/config/runtime fingerprint: " + str(exc)
        )
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Queue jobs must be a list")

    expected_keys = [(controller, seed) for controller in controllers for seed in seeds]
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    validation_errors = [*design_errors, *current_identity_errors]
    duplicate_keys: set[tuple[str, int]] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            validation_errors.append(f"jobs[{index}] is not an object")
            continue
        key = (job.get("controller"), job.get("seed"))
        if key not in expected_keys:
            validation_errors.append(f"unexpected matrix job identity at jobs[{index}]: {key!r}")
            continue
        if key in indexed:
            duplicate_keys.add(key)
            validation_errors.append(f"duplicate matrix job identity: {key!r}")
            continue
        indexed[key] = job
    declared_count = queue.get("job_count")
    if declared_count != len(expected_keys):
        validation_errors.append(
            f"queue.job_count is {declared_count!r}; design requires {len(expected_keys)}"
        )

    execution_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    checkpoint_scenario_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    metric_rows: dict[str, dict[str, dict[int, dict[str, float | None]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    validated_jobs = 0
    validated_bundles = 0
    queue_completed_jobs = 0
    completed_bundles = 0
    result_hashes: dict[str, str] = {}
    training_hashes: dict[str, str] = {}
    memory_warning_records: list[dict[str, Any]] = []

    for controller, seed in expected_keys:
        job = indexed.get((controller, seed))
        planned_id = f"{controller}__seed-{seed}"
        if job is None:
            issue = "planned controller/seed cell is missing from queue.jobs"
            validation_errors.append(f"{planned_id}: {issue}")
            execution_rows.append(
                {
                    "id": planned_id,
                    "controller": controller,
                    "seed": seed,
                    "status": "failed",
                    "record_state": "missing",
                    "issue": issue,
                    "evaluations": {
                        scenario: {"status": "failed", "record_state": "missing"}
                        for scenario in scenarios
                    },
                }
            )
            seed_rows.append(
                {
                    "id": planned_id,
                    "controller": controller,
                    "seed": seed,
                    "status": "failed",
                    "record_state": "missing",
                    "training": None,
                    "resources": None,
                    "evaluations": {scenario: None for scenario in scenarios},
                    "issues": [issue],
                }
            )
            checkpoint_scenario_rows.extend(
                {
                    "job_id": planned_id,
                    "controller": controller,
                    "training_seed": seed,
                    "scenario": scenario,
                    "status": "failed",
                    "record_state": "missing",
                    "expected_episode_count": episodes_per_scenario,
                    "metrics": None,
                }
                for scenario in scenarios
            )
            continue

        row_errors: list[str] = []
        source_status = job.get("status")
        status = source_status if source_status in VALID_JOB_STATUSES else "failed"
        if source_status not in VALID_JOB_STATUSES:
            row_errors.append(f"unknown queue job status {source_status!r}")
        if job.get("id") != planned_id:
            row_errors.append(f"job id is {job.get('id')!r}; expected {planned_id!r}")
        if job.get("total_interactions") != queue.get("config", {}).get("total_interactions"):
            row_errors.append("job interaction budget differs from config")
        if source_status == "completed":
            queue_completed_jobs += 1

        training, training_errors = _training_details(job, require_complete=source_status == "completed")
        row_errors.extend(training_errors)
        for message in training.get("memory_warnings", []):
            memory_warning_records.append({
                "scope": "training",
                "job_id": planned_id,
                "controller": controller,
                "training_seed": seed,
                "scenario": None,
                "message": message,
            })
        if training.get("sha256"):
            training_hashes[planned_id] = str(training["sha256"])

        bundle_list = job.get("evaluations")
        bundle_index: dict[str, Mapping[str, Any]] = {}
        if not isinstance(bundle_list, list):
            row_errors.append("job.evaluations is not a list")
            bundle_list = []
        for bundle in bundle_list:
            if not isinstance(bundle, Mapping) or bundle.get("scenario") not in scenarios:
                row_errors.append(f"unexpected evaluation bundle: {bundle!r}")
                continue
            scenario = str(bundle["scenario"])
            if scenario in bundle_index:
                row_errors.append(f"duplicate evaluation bundle for {scenario}")
                continue
            bundle_index[scenario] = bundle

        evaluation_cells: dict[str, Any] = {}
        execution_evaluations: dict[str, Any] = {}
        valid_job_bundles = 0
        for scenario in scenarios:
            bundle = bundle_index.get(scenario)
            if bundle is None:
                issue = f"missing evaluation bundle for {scenario}"
                row_errors.append(issue)
                execution_evaluations[scenario] = {
                    "status": "failed", "record_state": "missing", "expected_episodes": episodes_per_scenario
                }
                evaluation_cells[scenario] = None
                checkpoint_scenario_rows.append(
                    {
                        "job_id": planned_id,
                        "controller": controller,
                        "training_seed": seed,
                        "scenario": scenario,
                        "status": "failed",
                        "record_state": "missing",
                        "expected_episode_count": episodes_per_scenario,
                        "metrics": None,
                    }
                )
                continue
            bundle_source_status = bundle.get("status")
            bundle_status = bundle_source_status if bundle_source_status in VALID_JOB_STATUSES else "failed"
            if bundle_source_status not in VALID_JOB_STATUSES:
                row_errors.append(f"{scenario}: unknown evaluation status {bundle_source_status!r}")
            output = bundle.get("output")
            output_exists = isinstance(output, str) and Path(output).is_file()
            execution_evaluations[scenario] = {
                "status": bundle_status,
                "record_state": "present",
                "output": output,
                "output_exists": output_exists,
                "expected_episodes": episodes_per_scenario,
            }
            if bundle_source_status == "completed":
                completed_bundles += 1
                parsed, errors = _evaluation_result(
                    bundle, job, expected_episodes=episodes_per_scenario
                )
                if parsed is not None:
                    for message in parsed.get("memory_warnings", []):
                        memory_warning_records.append({
                            "scope": "evaluation",
                            "job_id": planned_id,
                            "controller": controller,
                            "training_seed": seed,
                            "scenario": scenario,
                            "message": message,
                        })
                if errors:
                    row_errors.extend(f"{scenario}: {error}" for error in errors)
                    evaluation_cells[scenario] = {
                        "status": "invalid_completed_artifact",
                        "output": output,
                        "metrics": parsed.get("metrics") if parsed else None,
                        "memory_warning_count": (
                            parsed.get("memory_warning_count", 0) if parsed else 0
                        ),
                        "memory_warnings": parsed.get("memory_warnings", []) if parsed else [],
                        "issues": errors,
                    }
                    checkpoint_scenario_rows.append(
                        {
                            "job_id": planned_id,
                            "controller": controller,
                            "training_seed": seed,
                            "scenario": scenario,
                            "status": "failed",
                            "record_state": "invalid_completed_artifact",
                            "expected_episode_count": episodes_per_scenario,
                            "output": output,
                            "metrics": parsed.get("metrics") if parsed else None,
                            "memory_warning_count": (
                                parsed.get("memory_warning_count", 0) if parsed else 0
                            ),
                            "memory_warnings": parsed.get("memory_warnings", []) if parsed else [],
                            "issues": errors,
                        }
                    )
                elif parsed is not None:
                    validated_bundles += 1
                    valid_job_bundles += 1
                    result_hashes[f"{planned_id}::{scenario}"] = parsed["sha256"]
                    metric_rows[scenario][controller][seed] = parsed["metrics"]
                    evaluation_cells[scenario] = {
                        "status": "completed",
                        "output": parsed["path"],
                        "sha256": parsed["sha256"],
                        "episode_count": parsed["episode_count"],
                        "metrics": parsed["metrics"],
                        "memory_warning_count": parsed["memory_warning_count"],
                        "memory_warnings": parsed["memory_warnings"],
                    }
                    checkpoint_scenario_rows.append(
                        {
                            "job_id": planned_id,
                            "controller": controller,
                            "training_seed": seed,
                            "scenario": scenario,
                            "status": "completed",
                            "record_state": "validated",
                            "expected_episode_count": episodes_per_scenario,
                            "episode_count": parsed["episode_count"],
                            "output": parsed["path"],
                            "sha256": parsed["sha256"],
                            "metrics": parsed["metrics"],
                            "memory_warning_count": parsed["memory_warning_count"],
                            "memory_warnings": parsed["memory_warnings"],
                        }
                    )
                    for episode in parsed["result"]["episodes"]:
                        episode_rows.append(
                            {
                                "job_id": planned_id,
                                "controller": controller,
                                "training_seed": seed,
                                "scenario": scenario,
                                "evaluation_result": parsed["path"],
                                **dict(episode),
                            }
                        )
            else:
                evaluation_cells[scenario] = {
                    "status": bundle_status,
                    "output": output,
                    "output_exists": output_exists,
                    "metrics": None,
                }
                checkpoint_scenario_rows.append(
                    {
                        "job_id": planned_id,
                        "controller": controller,
                        "training_seed": seed,
                        "scenario": scenario,
                        "status": bundle_status,
                        "record_state": "present",
                        "expected_episode_count": episodes_per_scenario,
                        "output": output,
                        "output_exists": output_exists,
                        "metrics": None,
                    }
                )

        job_is_valid = (
            source_status == "completed"
            and not row_errors
            and training.get("status") == "completed"
            and valid_job_bundles == len(scenarios)
        )
        if job_is_valid:
            validated_jobs += 1
        validation_errors.extend(f"{planned_id}: {error}" for error in row_errors)
        execution_rows.append(
            {
                "id": planned_id,
                "controller": controller,
                "seed": seed,
                "status": status,
                "source_status": source_status,
                "record_state": "duplicate" if (controller, seed) in duplicate_keys else "present",
                "run_dir": job.get("run_dir"),
                "checkpoint": job.get("checkpoint"),
                "checkpoint_exists": isinstance(job.get("checkpoint"), str) and Path(job["checkpoint"]).is_file(),
                "evaluations": execution_evaluations,
                "issues": row_errors,
            }
        )
        seed_rows.append(
            {
                "id": planned_id,
                "controller": controller,
                "seed": seed,
                "status": status,
                "validated_complete": job_is_valid,
                "expected_fingerprint": job.get("expected_fingerprint"),
                "training": training,
                "resources": _resource_projection(training),
                "evaluations": evaluation_cells,
                "issues": row_errors,
            }
        )

    planned_jobs = len(expected_keys)
    planned_bundles = planned_jobs * len(scenarios)
    planned_episodes = planned_bundles * episodes_per_scenario
    structural_complete = len(indexed) == planned_jobs and not duplicate_keys
    complete = (
        structural_complete
        and validated_jobs == planned_jobs
        and validated_bundles == planned_bundles
        and not validation_errors
    )
    label = str(queue.get("label"))
    comparison = _comparison(
        label=label,
        controllers=controllers,
        seeds=seeds,
        scenarios=scenarios,
        metric_rows=metric_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    computed_counts = Counter(row["status"] for row in execution_rows)
    status_counts = {status: computed_counts.get(status, 0) for status in VALID_JOB_STATUSES}
    status = "complete" if complete else "incomplete"
    return {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "status": status,
        "is_final": bool(label == "main" and complete),
        "label": label,
        "report_scope": (
            "integration_only_not_scientific" if label == "integration" else "main_matrix"
        ),
        "scientific_comparison_complete": bool(label == "main" and complete),
        "queue": str(queue_path),
        "queue_sha256": queue_sha,
        "fingerprint_scope": "per_job",
        "resolved_config": dict(queue.get("config", {})),
        "config_sha256": queue.get("config_sha256"),
        "fingerprints": {
            str(job.get("id")): job.get("expected_fingerprint")
            for job in jobs
            if isinstance(job, Mapping) and isinstance(job.get("id"), str)
        },
        "queue_status": queue.get("status"),
        "queue_declared_counts": queue.get("counts"),
        "completion": {
            "planned_jobs": planned_jobs,
            "queue_completed_jobs": queue_completed_jobs,
            "validated_completed_jobs": validated_jobs,
            "completed_fraction": f"{validated_jobs}/{planned_jobs}",
            "status_counts": status_counts,
            "planned_evaluation_bundles": planned_bundles,
            "queue_completed_evaluation_bundles": completed_bundles,
            "validated_evaluation_bundles": validated_bundles,
            "planned_evaluation_episodes_fixed_denominator": planned_episodes,
            "validated_evaluation_episodes": validated_bundles * episodes_per_scenario,
            "unvalidated_or_unfinished_evaluation_episodes": (
                planned_episodes - validated_bundles * episodes_per_scenario
            ),
        },
        "design": {
            "controllers": controllers,
            "training_seeds": seeds,
            "scenarios": scenarios,
            "episodes_per_scenario": episodes_per_scenario,
            "total_interactions_per_job": queue.get("config", {}).get("total_interactions"),
        },
        "execution_status_table": execution_rows,
        "seed_table": seed_rows,
        "checkpoint_scenario_table": checkpoint_scenario_rows,
        "episode_table": episode_rows,
        "controller_comparison": comparison,
        "training_manifest_sha256": training_hashes,
        "evaluation_result_sha256": result_hashes,
        "memory_warnings": {
            "policy_version": matrix_runner.MEMORY_POLICY_VERSION,
            "disposition": "warning_only",
            "count": len(memory_warning_records),
            "messages": [record["message"] for record in memory_warning_records],
            "records": memory_warning_records,
        },
        "missing_or_unfinished_jobs": [
            row["id"] for row in execution_rows if row["status"] != "completed" or row.get("issues")
        ],
        "validation_errors": validation_errors,
        "interpretation": (
            "Statistics use independent training seeds; episodes within a seed are not independent replicates. "
            "Failed, pending, cancelled, paused, and missing cells are retained in the execution table and never "
            "backfilled. Paired intervals compare same-numbered training seeds and are emitted only for a complete "
            "main five-seed pairing. Memory RSS-growth warnings remain visible but do not invalidate a v2 gate; "
            "GPU/RAM caps, missing CUDA telemetry, sustained paging, and nonfinite telemetry remain hard failures."
        ),
    }


def _format_number(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report: Mapping[str, Any]) -> str:
    """Render the machine report without dropping incomplete matrix cells."""

    label = str(report["label"])
    completion = report["completion"]
    title = "Crazyflie integration matrix report" if label == "integration" else "Crazyflie main matrix report"
    lines = [f"# {title}", ""]
    if label == "integration":
        lines += [
            "> **INTEGRATION ONLY:** this is an orchestration/checkpoint/evaluation check, not a scientific result.",
            "",
        ]
    if report["status"] != "complete":
        lines += [
            f"> **INCOMPLETE — not final:** {completion['validated_completed_jobs']}/{completion['planned_jobs']} "
            "planned jobs have fully validated artifacts. Missing and failed cells remain below.",
            "",
        ]
    else:
        lines += [
            f"Status: **complete** ({completion['validated_completed_jobs']}/{completion['planned_jobs']} jobs).",
            "",
        ]
    lines += [
        f"Queue status: `{_escape(report.get('queue_status'))}`. Fixed evaluation denominator: "
        f"{completion['planned_evaluation_episodes_fixed_denominator']} episodes; validated: "
        f"{completion['validated_evaluation_episodes']}.",
        "",
    ]
    memory_warnings = report.get("memory_warnings", {})
    warning_records = (
        memory_warnings.get("records", [])
        if isinstance(memory_warnings, Mapping)
        else []
    )
    lines += [
        "## Memory warnings (warning-only)",
        "",
        f"Count: **{len(warning_records)}** under "
        f"`{_escape(memory_warnings.get('policy_version'))}`. "
        "These warnings do not hide or relax hard memory failures.",
        "",
    ]
    if warning_records:
        lines.extend(
            f"- `{_escape(record.get('job_id'))}` / "
            f"`{_escape(record.get('scope'))}`"
            + (
                f" / `{_escape(record.get('scenario'))}`"
                if record.get("scenario") is not None
                else ""
            )
            + f": {_escape(record.get('message'))}"
            for record in warning_records
        )
        lines.append("")
    else:
        lines += ["No RSS-growth warnings were recorded.", ""]
    lines += [
        "## Execution status (all planned jobs)",
        "",
        "| Job | Controller | Seed | Status | Checkpoint | Evaluation status by scenario | Issues |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in report["execution_status_table"]:
        evaluations = ", ".join(
            f"{scenario}: {cell['status']}" for scenario, cell in row["evaluations"].items()
        )
        issues = "; ".join(row.get("issues", [])) or "—"
        checkpoint = "present" if row.get("checkpoint_exists") else "missing"
        lines.append(
            f"| {_escape(row['id'])} | {_escape(row['controller'])} | {row['seed']} | "
            f"{_escape(row['status'])} | {checkpoint} | {_escape(evaluations)} | {_escape(issues)} |"
        )

    lines += [
        "",
        "## Seed table",
        "",
        "| Controller | Seed | Status | Validated | Actor trainable | Critic trainable | Total trainable | Frozen | Dynamic state/env | RAM max (%) | GPU max (MiB) | Memory warnings |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_table"]:
        resources = row.get("resources") or {}
        lines.append(
            f"| {_escape(row['controller'])} | {row['seed']} | {_escape(row['status'])} | "
            f"{'yes' if row.get('validated_complete') else 'no'} | "
            f"{_format_number(resources.get('actor_trainable_parameters'), 0)} | "
            f"{_format_number(resources.get('critic_trainable_parameters'), 0)} | "
            f"{_format_number(resources.get('total_trainable_parameters'), 0)} | "
            f"{_format_number(resources.get('frozen_parameters'), 0)} | "
            f"{_format_number(resources.get('total_dynamic_state_per_environment'), 0)} | "
            f"{_format_number(resources.get('max_system_ram_percent'), 2)} | "
            f"{_format_number(resources.get('max_device_gpu_used_mib'), 1)} | "
            f"{int(resources.get('memory_warning_count', 0) or 0)} |"
        )

    lines += [
        "",
        "## Checkpoint/scenario summaries",
        "",
        "| Controller | Seed | Scenario | Status | Episodes | Success rate | Final error (m) | Integrated error (m·s) | Crash rate | OOB rate |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["checkpoint_scenario_table"]:
        metrics = row.get("metrics") or {}
        lines.append(
            f"| {_escape(row['controller'])} | {row['training_seed']} | {_escape(row['scenario'])} | "
            f"{_escape(row['status'])} | {row.get('episode_count', '—')} | "
            f"{_format_number(metrics.get('success_rate'))} | "
            f"{_format_number(metrics.get('mean_final_goal_error_m'))} | "
            f"{_format_number(metrics.get('mean_integrated_goal_error_m_s'))} | "
            f"{_format_number(metrics.get('crash_rate'))} | "
            f"{_format_number(metrics.get('out_of_bounds_rate'))} |"
        )
    lines += [
        "",
        f"The machine-readable report retains {len(report['episode_table'])} validated episode rows. "
        "No synthetic row is created for an unfinished or failed evaluation; its fixed planned denominator remains visible above.",
        "",
    ]

    comparison = report["controller_comparison"]
    lines += [
        "",
        "## Controller comparison",
        "",
        "Values are per-training-seed checkpoint/scenario aggregates. Mean, median, and sample standard "
        "deviation never substitute episodes for independent seeds.",
        "",
    ]
    for scenario, controllers in comparison["by_scenario"].items():
        lines += [
            f"### {_escape(scenario)}",
            "",
            "| Controller | Metric | Per-seed values | n | Mean | Median | Sample SD |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for controller, entry in controllers.items():
            for metric, stats in entry["metrics"].items():
                values = ", ".join(f"s{seed}={_format_number(value)}" for seed, value in stats["per_seed"].items())
                lines.append(
                    f"| {_escape(controller)} | {_escape(metric)} | {_escape(values)} | "
                    f"{stats['n_training_seeds']} | {_format_number(stats['mean'])} | "
                    f"{_format_number(stats['median'])} | {_format_number(stats['sample_std'])} |"
                )
        lines += [
            "",
            f"#### Paired differences vs `{BASELINE_CONTROLLER}`",
            "",
            "| Controller | Metric | Per-seed difference | Mean difference | Paired bootstrap 95% CI |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for controller, entry in comparison["paired_differences_vs_original"][scenario].items():
            for metric, stats in entry["metrics"].items():
                values = ", ".join(f"s{seed}={_format_number(value)}" for seed, value in stats["per_seed"].items())
                ci = stats["bootstrap_95_ci"]
                ci_text = (
                    f"[{_format_number(ci['lower'])}, {_format_number(ci['upper'])}]"
                    if ci is not None
                    else f"unavailable: {stats['bootstrap_unavailable_reason']}"
                )
                lines.append(
                    f"| {_escape(controller)} | {_escape(metric)} | {_escape(values)} | "
                    f"{_format_number(stats['mean'])} | {_escape(ci_text)} |"
                )
        lines.append("")

    if report["validation_errors"]:
        lines += ["## Validation errors", ""]
        lines.extend(f"- {_escape(error)}" for error in report["validation_errors"])
        lines.append("")
    lines += ["## Interpretation", "", str(report["interpretation"]), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", "--manifest", dest="queue", type=Path, required=True)
    parser.add_argument(
        "--output", "--json", dest="json_output", type=Path,
        help="Machine-readable report (default: QUEUE_STEM_report.json)",
    )
    parser.add_argument(
        "--markdown", type=Path,
        help="Human-readable report (default: QUEUE_STEM_report.md)",
    )
    parser.add_argument("--bootstrap_samples", "--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", "--bootstrap-seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap_samples must be at least 100")
    if not args.queue.is_file():
        parser.error(f"queue does not exist or is not a file: {args.queue}")
    json_output = args.json_output or args.queue.with_name(args.queue.stem + "_report.json")
    markdown_output = args.markdown or args.queue.with_name(args.queue.stem + "_report.md")
    if json_output.resolve() == args.queue.resolve() or markdown_output.resolve() == args.queue.resolve():
        parser.error("report outputs must not overwrite the queue")
    if json_output.resolve() == markdown_output.resolve():
        parser.error("JSON and Markdown outputs must be different files")
    try:
        report = summarize(
            args.queue,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    _atomic_text(json_output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_text(markdown_output, markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "label": report["label"],
                "is_final": report["is_final"],
                "validated_jobs": report["completion"]["validated_completed_jobs"],
                "planned_jobs": report["completion"]["planned_jobs"],
                "validation_error_count": len(report["validation_errors"]),
                "failure_reasons": report["validation_errors"],
                "memory_warning_count": report["memory_warnings"]["count"],
                "memory_warning_messages": report["memory_warnings"]["messages"],
                "json": str(json_output.resolve()),
                "markdown": str(markdown_output.resolve()),
                "fingerprint_scope": report["fingerprint_scope"],
                "resolved_config": report["resolved_config"],
                "config_sha256": report["config_sha256"],
                "fingerprints": report["fingerprints"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if report["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
