#!/usr/bin/env python3
"""Fail closed unless the frozen-LIF balanced-v3 flight proof is complete.

This verifier is intentionally read-only and CPU-only.  It starts no Isaac
application and writes no receipt: the main launcher revalidates the source,
checkpoint, training manifest, and immutable evaluation parts every time.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any

from drone_bootstrap import (
    ROOT,
    canonical_sha256,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)
from drone_evaluation_protocol import SCENARIOS, load_protocol
from drone_evaluate import (
    _merge_scenario_parts,
    _scenario_part_paths,
    _validated_gust_recovery_events,
)
from drone_train import standalone_resolved_config
from g1_fly_control.crazyflie.checkpoint import read_checkpoint
from g1_fly_control.crazyflie.controllers import build_controller
from g1_fly_control.crazyflie.memory import (
    GPU_LIMIT_MIB,
    MEMORY_POLICY_VERSION,
    RAM_LIMIT_PERCENT,
    assess as assess_memory,
)


PROOF_RUN_DIR = ROOT / "runs" / "crazyflie-balanced-v3-lif-proof"
PROOF_CHECKPOINT = PROOF_RUN_DIR / "checkpoints" / "latest.pt"
PROOF_TRAINING_MANIFEST = PROOF_RUN_DIR / "training_manifest.json"
PROOF_EVALUATION = PROOF_RUN_DIR / "evaluation.json"
MAIN_CONFIG = ROOT / "configs" / "experiments" / "crazyflie_balanced_v3_main.json"

EXPECTED_CONTROLLER = "frozen_lif_original"
EXPECTED_CONTROLLER_KIND = "frozen_lif"
EXPECTED_PROFILE = "balanced_v3"
EXPECTED_TASK = "FlyCrazyflie-Mixed-v0"
EXPECTED_SEED = 0
EXPECTED_INTERACTIONS = 500_000
EXPECTED_NUM_ENVS = 4
EXPECTED_HORIZON = 100
EXPECTED_INTERACTIONS_PER_UPDATE = EXPECTED_NUM_ENVS * EXPECTED_HORIZON
EXPECTED_UPDATES = EXPECTED_INTERACTIONS // EXPECTED_INTERACTIONS_PER_UPDATE
EXPECTED_PROTOCOL = "lif_proof"
EXPECTED_EPISODES_PER_SCENARIO = 5
EXPECTED_MEMORY_POLICY_VERSION = MEMORY_POLICY_VERSION


class ProofGateError(ValueError):
    """The empirical proof is absent, stale, malformed, or unsuccessful."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofGateError(message)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofGateError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProofGateError(f"{label} must be a JSON object: {path}")
    return value


def _expected_resolved_config(main_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recreate the exact standalone command configuration from the main PPO settings."""

    training = main_config["training"]
    _require(main_config.get("label") == "main", "Proof gate requires the fixed main config")
    _require(main_config.get("task") == EXPECTED_TASK, "Main config task is not balanced Mixed")
    _require(
        main_config.get("_contract_profile") == EXPECTED_PROFILE,
        "Main config does not select balanced_v3",
    )
    _require(
        training.get("num_envs") == EXPECTED_NUM_ENVS
        and training.get("horizon") == EXPECTED_HORIZON,
        "Main config no longer uses the required 4 x 100 rollout shape",
    )
    args = SimpleNamespace(
        evaluation_protocol=EXPECTED_PROTOCOL,
        task=EXPECTED_TASK,
        contract_profile=EXPECTED_PROFILE,
        policy=EXPECTED_CONTROLLER,
        seed=EXPECTED_SEED,
        num_envs=EXPECTED_NUM_ENVS,
        total_interactions=EXPECTED_INTERACTIONS,
        horizon=EXPECTED_HORIZON,
        microbatch_size=training["microbatch_size"],
        ppo_epochs=training["ppo_epochs"],
        learning_rate=training["learning_rate"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        clip_ratio=training["clip_ratio"],
        value_coefficient=training["value_coefficient"],
        entropy_coefficient=training["entropy_coefficient"],
        max_grad_norm=training["max_grad_norm"],
        target_kl=training["target_kl"],
        checkpoint_every_updates=training["checkpoint_every_updates"],
        rewire_seed=main_config["rewire_seed"],
        rewire_manifest=Path(main_config["_rewire_manifest_path"]).resolve(),
        warm_start_checkpoint=None,
    )
    resolved, protocol = standalone_resolved_config(args)
    _require(
        "survival_first_contract" not in resolved
        and isinstance(resolved.get("balanced_task_contract"), dict),
        "Expected proof configuration does not contain exactly the balanced contract",
    )
    return resolved, protocol


def _expected_controller_identity(main_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Construct the original controller on CPU to obtain its immutable identity."""

    policy, report = build_controller(
        EXPECTED_CONTROLLER,
        observation_dim=12,
        action_dim=4,
        device="cpu",
        connectome_manifest=Path(main_config["_connectome_path"]).resolve(),
        rewire_seed=int(main_config["rewire_seed"]),
    )
    policy_class = f"{type(policy).__module__}.{type(policy).__qualname__}"
    del policy
    _require(
        report.get("controller_kind") == EXPECTED_CONTROLLER_KIND,
        "CPU controller identity is not the original frozen LIF",
    )
    return policy_class, report


def _expected_curriculum_snapshot(expected_config: dict[str, Any]) -> dict[str, Any]:
    contract = expected_config["balanced_task_contract"]
    stages = contract["training_curriculum"]["stages"]
    active_index = max(
        index
        for index, stage in enumerate(stages)
        if int(stage["start_interactions"]) <= EXPECTED_INTERACTIONS
    )
    active = stages[active_index]
    return {
        "training_interactions": EXPECTED_INTERACTIONS,
        "active_stage_index": active_index,
        "active_stage_name": active["name"],
        "active_stage": active,
    }


def _require_finite_history_value(value: Any, *, location: str) -> None:
    """Recursively reject non-JSON or nonfinite values in proof history."""

    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        _require(math.isfinite(value), f"Proof history contains a nonfinite float at {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(
                isinstance(key, str),
                f"Proof history contains a non-string object key at {location}",
            )
            _require_finite_history_value(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_history_value(item, location=f"{location}[{index}]")
        return
    raise ProofGateError(
        f"Proof history contains unsupported {type(value).__name__} evidence at {location}"
    )


def _validate_training_history_rows(history: list[Any]) -> None:
    """Require finite rows and explicit zero nonfinite-state terminations."""

    for index, row in enumerate(history):
        _require(isinstance(row, dict), f"Proof history row {index} is not an object")
        expected_update = index + 1
        expected_interactions = expected_update * EXPECTED_INTERACTIONS_PER_UPDATE
        _require(
            type(row.get("completed_updates")) is int
            and type(row.get("total_interactions")) is int
            and row["completed_updates"] == expected_update
            and row["total_interactions"] == expected_interactions,
            f"Proof history row {index} breaks exact update/interaction continuity",
        )
        counts = row.get("failure_cause_counts")
        _require(
            isinstance(counts, dict) and set(counts) == {"1", "2", "3", "4"},
            f"Proof history row {index} lacks exact failure-cause counts",
        )
        _require(
            all(type(counts[code]) is int and counts[code] >= 0 for code in counts),
            f"Proof history row {index} has malformed failure-cause counts",
        )
        _require(
            counts["4"] == 0,
            f"Proof history row {index} records a nonfinite-state termination",
        )
        _require_finite_history_value(row, location=f"history[{index}]")


def _validate_training_memory(samples: Any, gate: Any) -> dict[str, Any]:
    _require(
        isinstance(samples, list)
        and bool(samples)
        and all(isinstance(sample, dict) for sample in samples),
        "Training memory samples are missing or malformed",
    )
    stages = {sample.get("stage") for sample in samples}
    required = {
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    }
    _require(required.issubset(stages), f"Training memory stages are incomplete: {sorted(stages)}")
    try:
        recomputed = assess_memory(samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofGateError(f"Training memory evidence cannot be assessed: {exc}") from exc
    _require(isinstance(gate, dict), "Training memory gate is missing")
    _require(
        canonical_sha256(gate) == canonical_sha256(recomputed),
        "Training memory gate does not recompute exactly",
    )
    _require(recomputed.get("passed") is True, "Training memory gate failed")
    _require(
        recomputed.get("policy_version") == EXPECTED_MEMORY_POLICY_VERSION,
        "Training memory gate is not policy v2",
    )
    warnings = recomputed.get("warnings")
    _require(
        isinstance(warnings, list)
        and all(isinstance(warning, str) and bool(warning) for warning in warnings),
        "Training memory warnings are malformed",
    )
    limits = recomputed.get("limits")
    _require(
        isinstance(limits, dict)
        and limits.get("gpu_used_mib_exclusive") == GPU_LIMIT_MIB
        and limits.get("system_ram_percent_exclusive") == RAM_LIMIT_PERCENT
        and limits.get("rss_growth_disposition") == "warning_only",
        "Training RSS growth disposition is not warning-only",
    )
    _require(
        recomputed.get("device_gpu_telemetry_complete") is True,
        "Training GPU telemetry is incomplete",
    )
    _require(
        float(recomputed.get("max_device_gpu_used_mib", math.inf)) < GPU_LIMIT_MIB,
        f"Training GPU memory is not below {GPU_LIMIT_MIB:.1f} MiB",
    )
    _require(
        float(recomputed.get("max_system_ram_percent", math.inf)) < RAM_LIMIT_PERCENT,
        f"Training RAM is not below {RAM_LIMIT_PERCENT:.1f}%",
    )
    _require(
        recomputed.get("sustained_paging_detected") is False,
        "Training paging gate failed",
    )
    return recomputed


def _validate_checkpoint_and_manifest(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    payload: dict[str, Any],
    manifest: dict[str, Any],
    expected_config: dict[str, Any],
    expected_policy_class: str,
    expected_controller_report: dict[str, Any],
    current_fingerprint: str,
    current_fingerprint_payload: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the exact 500k checkpoint and its completed training manifest."""

    _require(payload.get("policy_class") == expected_policy_class, "Proof policy class changed")
    _require(payload.get("resolved_config") == expected_config, "Proof resolved config is not exact")
    _require(payload.get("interactions_per_update") == EXPECTED_INTERACTIONS_PER_UPDATE,
             "Proof interactions-per-update is not exactly 400")
    counters = payload.get("counters")
    _require(isinstance(counters, dict), "Proof checkpoint counters are missing")
    _require(
        counters.get("total_interactions") == EXPECTED_INTERACTIONS
        and counters.get("completed_updates") == EXPECTED_UPDATES,
        "Proof checkpoint is not exactly 500,000 interactions / 1,250 updates",
    )
    history = payload.get("history")
    _require(isinstance(history, list) and len(history) == EXPECTED_UPDATES,
             "Proof checkpoint does not authenticate all 1,250 history rows")
    _require(
        history[-1].get("completed_updates") == EXPECTED_UPDATES
        and history[-1].get("total_interactions") == EXPECTED_INTERACTIONS,
        "Proof history does not end at the exact checkpoint boundary",
    )
    _validate_training_history_rows(history)

    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "Proof checkpoint metadata is missing")
    for field, expected in (
        ("status", "completed"),
        ("contract_profile", EXPECTED_PROFILE),
        ("controller", EXPECTED_CONTROLLER),
        ("seed", EXPECTED_SEED),
        ("requested_interactions", EXPECTED_INTERACTIONS),
        ("interactions_per_update", EXPECTED_INTERACTIONS_PER_UPDATE),
    ):
        _require(metadata.get(field) == expected, f"Checkpoint metadata {field} is not exact")
    _require(
        metadata.get("controller_report") == expected_controller_report,
        "Checkpoint controller identity differs from the current original frozen LIF",
    )
    expected_core = expected_controller_report["core_checksum"]
    _require(
        metadata.get("core_checksum_before") == expected_core
        and metadata.get("core_checksum_after") == expected_core
        and payload.get("core_checksum") == expected_core,
        "Frozen LIF core changed during proof training",
    )
    expected_curriculum = _expected_curriculum_snapshot(expected_config)
    _require(
        metadata.get("training_curriculum") == expected_curriculum,
        "Checkpoint curriculum clock/stage is not exact at 500,000 interactions",
    )
    mixed_snapshot = metadata.get("mixed_scenario")
    _require(
        isinstance(mixed_snapshot, dict)
        and mixed_snapshot.get("contract") == expected_config["mixed_scenario_contract"],
        "Checkpoint mixed-scenario schedule evidence is missing or stale",
    )

    fingerprints = payload.get("fingerprints")
    expected_fingerprints = {
        "reproduction": current_fingerprint,
        "source_set": canonical_sha256(current_fingerprint_payload["source_sha256"]),
        "connectome": expected_controller_report["connectome_checksum"],
        "frozen_core": expected_core,
        "rewire_manifest": current_fingerprint_payload["rewired_manifest_sha256"],
    }
    _require(fingerprints == expected_fingerprints, "Checkpoint fingerprints are stale or incomplete")
    _require(
        payload.get("evaluation_manifest_id") == protocol["manifest_id"],
        "Checkpoint evaluation manifest is not the lif_proof protocol",
    )

    required_manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": EXPECTED_TASK,
        "contract_profile": EXPECTED_PROFILE,
        "controller": EXPECTED_CONTROLLER,
        "seed": EXPECTED_SEED,
        "num_envs": EXPECTED_NUM_ENVS,
        "horizon": EXPECTED_HORIZON,
        "requested_interactions": EXPECTED_INTERACTIONS,
        "environment_interactions": EXPECTED_INTERACTIONS,
        "completed_updates": EXPECTED_UPDATES,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "fingerprint": current_fingerprint,
        "fingerprint_payload": current_fingerprint_payload,
        "resolved_config": expected_config,
        "evaluation_manifest_id": protocol["manifest_id"],
        "controller_report": expected_controller_report,
        "core_checksum_before": expected_core,
        "core_checksum_after": expected_core,
        "training_curriculum": expected_curriculum,
        "history_reference": payload.get("history_reference"),
    }
    for field, expected in required_manifest.items():
        _require(manifest.get(field) == expected, f"Training manifest field {field} is not exact")
    _require(
        manifest.get("task_manifest_id") == payload.get("task_manifest_id"),
        "Training manifest/checkpoint task identity differs",
    )
    _require(
        manifest.get("mixed_scenario") == mixed_snapshot,
        "Training manifest/checkpoint mixed-scenario evidence differs",
    )
    _require(
        canonical_sha256(metadata.get("memory_samples"))
        == canonical_sha256(manifest.get("memory_samples")),
        "Training manifest/checkpoint memory samples differ",
    )
    memory_gate = _validate_training_memory(
        manifest.get("memory_samples"), manifest.get("memory_gate")
    )
    return memory_gate


def _validated_part_paths(evaluation_path: Path, evaluation: dict[str, Any]) -> dict[str, Path]:
    attempt_id = evaluation.get("scenario_attempt_id")
    _require(isinstance(attempt_id, str) and bool(attempt_id),
             "Merged evaluation lacks a scenario attempt identity")
    try:
        expected_paths = _scenario_part_paths(evaluation_path, attempt_id)
    except ValueError as exc:
        raise ProofGateError(f"Invalid scenario attempt identity: {exc}") from exc
    declared = evaluation.get("scenario_part_artifacts")
    _require(
        isinstance(declared, dict)
        and len(declared) == len(SCENARIOS)
        and set(declared) == set(SCENARIOS),
        "Merged evaluation does not declare exactly the three scenario parts",
    )
    for scenario, expected_path in expected_paths.items():
        artifact = declared.get(scenario)
        _require(isinstance(artifact, dict), f"Missing part declaration for {scenario}")
        _require(
            artifact.get("path") == str(expected_path.resolve()),
            f"Evaluation part path is not the immutable attempt path for {scenario}",
        )
        _require(expected_path.is_file(), f"Evaluation part is missing for {scenario}")
        _require(
            artifact.get("sha256") == sha256_file(expected_path),
            f"Evaluation part SHA-256 changed for {scenario}",
        )
    return expected_paths


def _episode_has_proof_event(scenario: str, row: dict[str, Any]) -> bool:
    """Derive the user-selected event score from authenticated episode fields."""

    target_count = row.get("target_success_event_count")
    maximum_target_successes = 4 if scenario == SCENARIOS[1] else 1
    _require(
        type(target_count) is int and 0 <= target_count <= maximum_target_successes,
        f"Episode target-success event count is invalid for {scenario}",
    )
    strict_success = row.get("success")
    _require(type(strict_success) is bool, f"Episode success is not Boolean for {scenario}")
    _require(
        strict_success is (target_count >= maximum_target_successes),
        f"Strict episode success disagrees with target-success events for {scenario}",
    )
    if scenario == SCENARIOS[0]:
        return target_count >= 1
    if scenario == SCENARIOS[1]:
        outcomes = row.get("switch_outcomes")
        _require(
            isinstance(outcomes, list)
            and len(outcomes) == 3
            and all(
                isinstance(outcome, dict)
                and type(outcome.get("success")) is bool
                for outcome in outcomes
            ),
            f"Switch event outcomes are malformed for {scenario}",
        )
        switched_successes = sum(outcome["success"] for outcome in outcomes)
        _require(
            target_count in {switched_successes, switched_successes + 1},
            f"Switch event outcomes disagree with target-success count for {scenario}",
        )
        return target_count >= 1
    if scenario == SCENARIOS[2]:
        try:
            recovered = _validated_gust_recovery_events(row, row.get("plan"))
        except ValueError as exc:
            raise ProofGateError(
                f"Gust recovery/impulse evidence is malformed for {scenario}: {exc}"
            ) from exc
        return any(recovered)
    raise ProofGateError(f"Unknown proof scenario: {scenario}")


def _validate_evaluation(
    *,
    evaluation_path: Path,
    evaluation: dict[str, Any],
    rebuilt: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    current_fingerprint: str,
    current_fingerprint_payload: dict[str, Any],
    expected_controller_report: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Require an exact rebuilt five-episode proof and real Boolean successes."""

    rebuilt_for_compare = deepcopy(rebuilt)
    rebuilt_for_compare["generated_at_utc"] = evaluation.get("generated_at_utc")
    _require(
        canonical_sha256(evaluation) == canonical_sha256(rebuilt_for_compare),
        "Published evaluation differs from its strictly rebuilt immutable parts",
    )
    for field, expected in (
        ("schema_version", 1),
        ("status", "completed"),
        ("label", "lif_proof"),
        ("protocol", EXPECTED_PROTOCOL),
        ("scenario", "all"),
        ("total_episodes", len(SCENARIOS) * EXPECTED_EPISODES_PER_SCENARIO),
        ("evaluation_seed", protocol["evaluation_seed"]),
        ("evaluation_manifest_id", protocol["manifest_id"]),
        ("training_seed", EXPECTED_SEED),
        ("controller", EXPECTED_CONTROLLER),
        ("deterministic_actions", True),
        ("checkpoint", str(checkpoint.resolve())),
        ("checkpoint_sha256", checkpoint_sha256),
        ("fingerprint", current_fingerprint),
        ("fingerprint_payload", current_fingerprint_payload),
        ("controller_report", expected_controller_report),
    ):
        _require(evaluation.get(field) == expected, f"Evaluation field {field} is not exact")

    aggregate_memory = evaluation.get("memory_gate")
    _require(
        isinstance(aggregate_memory, dict)
        and aggregate_memory.get("policy_version") == EXPECTED_MEMORY_POLICY_VERSION
        and aggregate_memory.get("passed") is True
        and aggregate_memory.get("device_gpu_telemetry_complete") is True
        and type(aggregate_memory.get("monotonic_process_growth_detected")) is bool
        and aggregate_memory.get("sustained_paging_detected") is False,
        "Aggregate evaluation memory gate did not pass",
    )
    aggregate_warnings = aggregate_memory.get("warnings")
    _require(
        isinstance(aggregate_warnings, list)
        and all(isinstance(warning, str) and bool(warning) for warning in aggregate_warnings),
        "Aggregate evaluation memory warnings are malformed",
    )
    aggregate_limits = aggregate_memory.get("limits")
    _require(
        isinstance(aggregate_limits, dict)
        and aggregate_limits.get("gpu_used_mib_exclusive") == GPU_LIMIT_MIB
        and aggregate_limits.get("system_ram_percent_exclusive") == RAM_LIMIT_PERCENT
        and aggregate_limits.get("rss_growth_disposition") == "warning_only",
        "Aggregate evaluation RSS growth disposition is not warning-only",
    )
    for field, limit in (
        ("max_device_gpu_used_mib", GPU_LIMIT_MIB),
        ("max_system_ram_percent", RAM_LIMIT_PERCENT),
    ):
        value = aggregate_memory.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) < limit,
            f"Aggregate evaluation {field} is not finite and below its limit",
        )
    aggregate_rss = aggregate_memory.get("max_process_rss_mib")
    _require(
        isinstance(aggregate_rss, (int, float))
        and not isinstance(aggregate_rss, bool)
        and math.isfinite(float(aggregate_rss))
        and float(aggregate_rss) >= 0.0,
        "Aggregate evaluation RSS is missing or nonfinite",
    )
    scenario_gates = aggregate_memory.get("scenario_gates")
    _require(
        isinstance(scenario_gates, dict)
        and len(scenario_gates) == len(SCENARIOS)
        and set(scenario_gates) == set(SCENARIOS),
        "Aggregate evaluation lacks exactly three scenario memory gates",
    )
    for scenario in SCENARIOS:
        gate = scenario_gates[scenario]
        _require(
            isinstance(gate, dict)
            and gate.get("policy_version") == EXPECTED_MEMORY_POLICY_VERSION
            and gate.get("passed") is True
            and gate.get("device_gpu_telemetry_complete") is True
            and gate.get("sustained_paging_detected") is False,
            f"Evaluation RAM/VRAM/paging gate failed for {scenario}",
        )
        warnings = gate.get("warnings")
        limits = gate.get("limits")
        _require(
            isinstance(warnings, list)
            and all(isinstance(warning, str) and bool(warning) for warning in warnings)
            and isinstance(limits, dict)
            and limits.get("gpu_used_mib_exclusive") == GPU_LIMIT_MIB
            and limits.get("system_ram_percent_exclusive") == RAM_LIMIT_PERCENT
            and limits.get("rss_growth_disposition") == "warning_only",
            f"Evaluation memory policy v2 metadata is invalid for {scenario}",
        )
        for field, limit in (
            ("max_device_gpu_used_mib", GPU_LIMIT_MIB),
            ("max_system_ram_percent", RAM_LIMIT_PERCENT),
        ):
            value = gate.get(field)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) < limit,
                f"Evaluation {field} is invalid for {scenario}",
            )
    expected_aggregate_warnings = [
        f"{scenario}: {warning}"
        for scenario in SCENARIOS
        for warning in scenario_gates[scenario]["warnings"]
    ]
    expected_aggregate_failures = [
        f"{scenario}: {failure}"
        for scenario in SCENARIOS
        for failure in scenario_gates[scenario].get("failures", [])
    ]
    _require(
        aggregate_warnings == expected_aggregate_warnings
        and aggregate_memory.get("failures") == expected_aggregate_failures
        and aggregate_memory["monotonic_process_growth_detected"]
        is any(
            gate.get("monotonic_process_growth_detected") is True
            for gate in scenario_gates.values()
        )
        and aggregate_memory["sustained_paging_detected"]
        is any(
            gate.get("sustained_paging_detected") is True
            for gate in scenario_gates.values()
        ),
        "Aggregate evaluation memory metadata disagrees with scenario gates",
    )
    for field in (
        "max_device_gpu_used_mib",
        "max_system_ram_percent",
        "max_process_rss_mib",
    ):
        _require(
            aggregate_memory.get(field)
            == max(gate[field] for gate in scenario_gates.values()),
            f"Aggregate evaluation {field} disagrees with scenario gates",
        )
    results = evaluation.get("scenario_results")
    _require(
        isinstance(results, dict)
        and len(results) == len(SCENARIOS)
        and set(results) == set(SCENARIOS),
        "Evaluation does not contain exactly the three scenarios",
    )
    scenario_scores: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        result = results[scenario]
        _require(isinstance(result, dict), f"Evaluation result is malformed for {scenario}")
        episodes = result.get("episodes")
        summary = result.get("summary")
        _require(
            isinstance(episodes, list) and len(episodes) == EXPECTED_EPISODES_PER_SCENARIO,
            f"Evaluation episode count is not exactly five for {scenario}",
        )
        _require(isinstance(summary, dict), f"Evaluation summary is missing for {scenario}")
        event_flags = []
        for row in episodes:
            _require(isinstance(row, dict), f"Non-object episode row for {scenario}")
            _require(row.get("invalid_state") is False,
                     f"Evaluation recorded a nonfinite/invalid state for {scenario}")
            event_flags.append(_episode_has_proof_event(scenario, row))
        strict_success_count = sum(row["success"] is True for row in episodes)
        _require(
            summary.get("success_count") == strict_success_count,
            f"Success aggregate does not equal Boolean episode rows for {scenario}",
        )
        event_episode_count = sum(event_flags)
        _require(
            event_episode_count >= 1,
            f"LIF proof has zero held-out event-success episodes for {scenario}",
        )
        _require(
            summary.get("invalid_state_count") == 0,
            f"Evaluation invalid-state aggregate is nonzero for {scenario}",
        )
        scored_episodes = []
        for row, event_success in zip(episodes, event_flags, strict=True):
            scored_episodes.append({
                **deepcopy(row),
                "proof_event_success": event_success,
            })
        scenario_scores[scenario] = {
            "event_score": f"{event_episode_count}/{EXPECTED_EPISODES_PER_SCENARIO}",
            "event_episode_count": event_episode_count,
            "event_episode_rate": event_episode_count / EXPECTED_EPISODES_PER_SCENARIO,
            "episode_count": EXPECTED_EPISODES_PER_SCENARIO,
            "minimum_event_episode_count": 1,
            "gate_passed": event_episode_count >= 1,
            "strict_full_episode_success_count": strict_success_count,
            "strict_full_episode_success_rate": (
                strict_success_count / EXPECTED_EPISODES_PER_SCENARIO
            ),
            "summary": deepcopy(summary),
            "episodes": scored_episodes,
        }
    return scenario_scores, aggregate_memory


def verify_lif_proof(
    *,
    checkpoint: Path = PROOF_CHECKPOINT,
    training_manifest: Path = PROOF_TRAINING_MANIFEST,
    evaluation: Path = PROOF_EVALUATION,
    main_config: Path = MAIN_CONFIG,
    validated_main_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and verify every proof artifact without starting Isaac or writing files."""

    checkpoint = checkpoint.resolve()
    training_manifest = training_manifest.resolve()
    evaluation = evaluation.resolve()
    main_config = main_config.resolve()
    for path, label in (
        (checkpoint, "proof checkpoint"),
        (training_manifest, "proof training manifest"),
        (evaluation, "proof evaluation"),
        (main_config, "balanced main config"),
    ):
        _require(path.is_file(), f"Missing {label}: {path}")

    if validated_main_config is None:
        # Local import keeps the verifier importable from drone_run_matrix
        # without creating a runner <-> verifier module cycle.  The standalone
        # shell gate still performs the same closed config validation here.
        from drone_run_matrix import validate_config

        config = validate_config(main_config)
    else:
        config = validated_main_config
        _require(isinstance(config, dict), "Prevalidated main config is not a mapping")
        configured_path = config.get("_config_path")
        _require(
            isinstance(configured_path, str)
            and Path(configured_path).resolve() == main_config,
            "Prevalidated main config path differs from the requested proof config",
        )
    expected_config, protocol = _expected_resolved_config(config)
    _require(protocol == load_protocol(EXPECTED_PROTOCOL),
             "Resolved proof protocol differs from the frozen lif_proof manifest")
    expected_policy_class, controller_report = _expected_controller_identity(config)
    rewire_manifest = load_fingerprint_rewire_manifest(
        config["_rewire_manifest_path"],
        expected_file_sha256=config["_rewire_manifest_sha256"],
        expected_seed=config["rewire_seed"],
    )
    current_fingerprint, current_payload = reproduction_fingerprint(
        resolved_config=expected_config,
        evaluation_manifest=protocol,
        connectome_manifest=config["_connectome_path"],
        rewired_manifest=rewire_manifest,
    )

    checkpoint_sha = sha256_file(checkpoint)
    try:
        payload = read_checkpoint(
            checkpoint, map_location="cpu", resolve_external_history=True
        )
    except Exception as exc:
        raise ProofGateError(f"Proof checkpoint validation failed: {exc}") from exc
    manifest = _read_json_object(training_manifest, label="proof training manifest")
    training_memory = _validate_checkpoint_and_manifest(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        payload=payload,
        manifest=manifest,
        expected_config=expected_config,
        expected_policy_class=expected_policy_class,
        expected_controller_report=controller_report,
        current_fingerprint=current_fingerprint,
        current_fingerprint_payload=current_payload,
        protocol=protocol,
    )
    _require(checkpoint_sha == sha256_file(checkpoint),
             "Proof checkpoint changed while it was being validated")

    merged = _read_json_object(evaluation, label="proof evaluation")
    part_paths = _validated_part_paths(evaluation, merged)
    rebuilt = _merge_scenario_parts(
        part_paths,
        protocol_name=EXPECTED_PROTOCOL,
        protocol=protocol,
        checkpoint=checkpoint,
        expected_fingerprint=current_fingerprint,
        expected_training_seed=EXPECTED_SEED,
        expected_policy=EXPECTED_CONTROLLER,
    )
    scenario_scores, evaluation_memory = _validate_evaluation(
        evaluation_path=evaluation,
        evaluation=merged,
        rebuilt=rebuilt,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        current_fingerprint=current_fingerprint,
        current_fingerprint_payload=current_payload,
        expected_controller_report=controller_report,
        protocol=protocol,
    )
    _require(checkpoint_sha == sha256_file(checkpoint),
             "Proof checkpoint changed while evaluation evidence was being validated")
    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "balanced_v3_original_lif_empirical_flight_proof",
        "read_only": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "fingerprint": current_fingerprint,
        "controller": EXPECTED_CONTROLLER,
        "contract_profile": EXPECTED_PROFILE,
        "task": EXPECTED_TASK,
        "seed": EXPECTED_SEED,
        "environment_interactions": EXPECTED_INTERACTIONS,
        "completed_updates": EXPECTED_UPDATES,
        # Keep the legacy key for downstream readers, but name the strict
        # full-episode metric explicitly so it cannot be confused with the
        # event-based proof gate below.
        "success_counts": {
            scenario: score["strict_full_episode_success_count"]
            for scenario, score in scenario_scores.items()
        },
        "strict_full_episode_success_counts": {
            scenario: score["strict_full_episode_success_count"]
            for scenario, score in scenario_scores.items()
        },
        "event_episode_counts": {
            scenario: score["event_episode_count"]
            for scenario, score in scenario_scores.items()
        },
        "scenario_scores": scenario_scores,
        "training_memory": {
            "policy_version": training_memory["policy_version"],
            "max_device_gpu_used_mib": training_memory["max_device_gpu_used_mib"],
            "max_system_ram_percent": training_memory["max_system_ram_percent"],
            "max_process_rss_mib": training_memory["max_process_rss_mib"],
            "monotonic_process_growth_detected": training_memory[
                "monotonic_process_growth_detected"
            ],
            "warnings": training_memory["warnings"],
            "sustained_paging_detected": training_memory["sustained_paging_detected"],
        },
        "evaluation_memory": {
            "policy_version": evaluation_memory["policy_version"],
            "max_device_gpu_used_mib": evaluation_memory["max_device_gpu_used_mib"],
            "max_system_ram_percent": evaluation_memory["max_system_ram_percent"],
            "max_process_rss_mib": evaluation_memory["max_process_rss_mib"],
            "monotonic_process_growth_detected": evaluation_memory[
                "monotonic_process_growth_detected"
            ],
            "warnings": evaluation_memory["warnings"],
            "sustained_paging_detected": evaluation_memory[
                "sustained_paging_detected"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = verify_lif_proof()
    except BaseException:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "gate": "balanced_v3_original_lif_empirical_flight_proof",
            "read_only": True,
            "error": traceback.format_exc(),
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
