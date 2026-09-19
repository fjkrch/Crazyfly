#!/usr/bin/env python3
"""Select the predeclared balanced-v4 Reach original-LIF learning rate.

This command is deliberately CPU-only and fail-closed.  It authenticates both
completed 40k screen runs, applies the frozen lexicographic ordering, and
writes one immutable selection receipt.  It never launches Isaac and it never
uses held-out evaluation results.
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
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from g1_fly_control.crazyflie.checkpoint import (  # noqa: E402
    load_history_reference,
    read_checkpoint,
)
from g1_fly_control.crazyflie.memory import assess as assess_memory  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "crazyflie_balanced_v4_lif_reach_lr_screen.json"
)
EXPECTED_CONFIG_SHA256 = "cbe1585867f0e7413a403bd7173438cab7956e51a3208fd82966c9ae8b68fc73"
EXPECTED_TASK = "FlyCrazyflie-WaypointReach-v0"
EXPECTED_PROFILE = "balanced_v4"
EXPECTED_CONTROLLER = "frozen_lif_original"
EXPECTED_CONTROLLER_KIND = "frozen_lif"
EXPECTED_SEED = 20260917
EXPECTED_NUM_ENVS = 4
EXPECTED_INTERACTIONS = 40_000
EXPECTED_HORIZON = 100
EXPECTED_UPDATES = 100
EXPECTED_INTERACTIONS_PER_UPDATE = 400
EXPECTED_MEMORY_POLICY = "crazyflie_memory_acceptance_v2"
EXPECTED_LRS = (1.0e-4, 3.0e-4)
EXPECTED_RUN_DIRS = (
    "runs/crazyflie-balanced-v4-lif-reach-lr-screen-seed20260917-lr-1e-4",
    "runs/crazyflie-balanced-v4-lif-reach-lr-screen-seed20260917-lr-3e-4",
)
EXPECTED_OUTPUT = "runs/crazyflie-balanced-v4-lif-reach-lr-screen-selection.json"
COMMON_TRAINING = {
    "num_envs": EXPECTED_NUM_ENVS,
    "total_interactions": EXPECTED_INTERACTIONS,
    "horizon": EXPECTED_HORIZON,
    "expected_updates": EXPECTED_UPDATES,
    "microbatch_size": 4,
    "ppo_epochs": 2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.002,
    "max_grad_norm": 1.0,
    "target_kl": 0.05,
    "checkpoint_every_updates": 25,
    "evaluation_protocol": "integration",
}


class SelectionGateError(ValueError):
    """The screen declaration or either candidate is absent or invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionGateError(message)


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionGateError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionGateError(f"{label} must be a JSON object: {path}")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    actual = set(value)
    _require(
        actual == keys,
        f"{label} fields differ: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}",
    )


def _validate_declaration(config_path: Path) -> dict[str, Any]:
    source = config_path.expanduser().resolve()
    _require(source == DEFAULT_CONFIG.resolve(), "Only the frozen Reach LR screen config is accepted")
    _require(source.is_file(), f"LR screen config is missing: {source}")
    actual_sha = _sha256_file(source)
    _require(
        actual_sha == EXPECTED_CONFIG_SHA256,
        f"LR screen config SHA-256 changed: expected={EXPECTED_CONFIG_SHA256}, actual={actual_sha}",
    )
    config = _read_json_object(source, label="LR screen config")
    _require(config.get("schema_version") == 1, "LR screen config schema_version is not 1")
    _require(
        config.get("kind") == "crazyflie_balanced_v4_original_lif_reach_lr_screen",
        "LR screen config kind differs",
    )
    _require(config.get("status") == "predeclared_before_results", "Config is not predeclared")
    _require(config.get("task") == EXPECTED_TASK, "Config task differs")
    _require(config.get("contract_profile") == EXPECTED_PROFILE, "Config profile differs")
    _require(config.get("controller") == EXPECTED_CONTROLLER, "Config controller differs")
    _require(config.get("training_seed") == EXPECTED_SEED, "Config seed differs")
    _require(config.get("screen_is_warm_start_source") is False, "Screens must not warm-start proof runs")
    _require(
        config.get("held_out_evaluation_allowed_for_selection") is False,
        "Held-out evaluation must be excluded from LR selection",
    )
    _require(config.get("common_training") == COMMON_TRAINING, "Common PPO settings differ")
    candidates = config.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 2, "Exactly two candidates are required")
    for index, (candidate, learning_rate, run_dir) in enumerate(
        zip(candidates, EXPECTED_LRS, EXPECTED_RUN_DIRS, strict=True)
    ):
        _require(isinstance(candidate, dict), f"Candidate {index} is not an object")
        _require_exact_keys(
            candidate,
            {"candidate_id", "learning_rate", "run_dir"},
            label=f"candidate {index}",
        )
        _require(candidate["learning_rate"] == learning_rate, f"Candidate {index} LR differs")
        _require(candidate["run_dir"] == run_dir, f"Candidate {index} run_dir differs")
    _require(config.get("selection_output") == EXPECTED_OUTPUT, "Selection output path differs")

    expected_order = [
        (1, "hard_gates", "must_pass"),
        (2, "training_target_success_event_count", "higher"),
        (3, "final_goal_distance_over_completed_pilot_episodes_m", "lower"),
        (4, "cumulative_progress_reward", "higher"),
        (5, "learning_rate", "lower"),
    ]
    order = config.get("selection_order")
    _require(isinstance(order, list) and len(order) == len(expected_order), "Selection order differs")
    for row, (rank, criterion, direction) in zip(order, expected_order, strict=True):
        _require(isinstance(row, dict), f"Selection criterion {rank} is not an object")
        _require(row.get("rank") == rank, f"Selection criterion rank {rank} differs")
        _require(row.get("criterion") == criterion, f"Selection criterion {rank} differs")
        _require(row.get("direction") == direction, f"Selection direction {rank} differs")
    return config


def _require_finite_tree(value: Any, *, location: str) -> None:
    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        _require(math.isfinite(value), f"Nonfinite float at {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), f"Non-string key at {location}")
            _require_finite_tree(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, location=f"{location}[{index}]")
        return
    raise SelectionGateError(f"Unsupported {type(value).__name__} at {location}")


def _resolve_recorded_path(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is missing")
    return Path(value).expanduser().resolve()


def _validate_source_fingerprint(payload: Mapping[str, Any], *, candidate_id: str) -> None:
    sources = payload.get("source_sha256")
    _require(isinstance(sources, dict) and bool(sources), f"{candidate_id}: source hashes are missing")
    for recorded_path, expected_sha in sources.items():
        _require(
            isinstance(recorded_path, str)
            and bool(recorded_path)
            and isinstance(expected_sha, str)
            and len(expected_sha) == 64,
            f"{candidate_id}: malformed source hash entry",
        )
        path = Path(recorded_path)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        _require(path.is_file(), f"{candidate_id}: fingerprinted source is missing: {path}")
        actual = _sha256_file(path)
        _require(
            actual == expected_sha,
            f"{candidate_id}: fingerprinted source changed: {path}",
        )


def _validate_memory(
    samples: Any,
    recorded_gate: Any,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    _require(
        isinstance(samples, list) and bool(samples) and all(isinstance(row, dict) for row in samples),
        f"{candidate_id}: memory samples are missing or malformed",
    )
    required_stages = {
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    }
    stages = {sample.get("stage") for sample in samples}
    _require(
        required_stages.issubset(stages),
        f"{candidate_id}: memory sampling stages are incomplete: {sorted(stages)}",
    )
    try:
        recomputed = assess_memory(samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionGateError(f"{candidate_id}: memory evidence is invalid: {exc}") from exc
    _require(isinstance(recorded_gate, dict), f"{candidate_id}: memory gate is missing")
    _require(recorded_gate == recomputed, f"{candidate_id}: memory gate does not recompute exactly")
    _require(recomputed.get("policy_version") == EXPECTED_MEMORY_POLICY, f"{candidate_id}: memory policy differs")
    _require(recomputed.get("passed") is True, f"{candidate_id}: memory hard gate failed")
    _require(recomputed.get("failures") == [], f"{candidate_id}: memory failures are nonempty")
    return recomputed


def _history_metrics(rows: list[dict[str, Any]], *, candidate_id: str) -> dict[str, Any]:
    success_events = 0
    completed_episodes = 0
    successful_episodes = 0
    final_distance_sum = 0.0
    cumulative_progress = 0.0
    failure_counts = {str(code): 0 for code in range(1, 5)}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{candidate_id}: history row {index} is not an object")
        _require_finite_tree(row, location=f"{candidate_id}.history[{index}]")
        expected_update = index + 1
        expected_interactions = expected_update * EXPECTED_INTERACTIONS_PER_UPDATE
        _require(
            row.get("completed_updates") == expected_update
            and type(row.get("completed_updates")) is int
            and row.get("total_interactions") == expected_interactions
            and type(row.get("total_interactions")) is int,
            f"{candidate_id}: history row {index} breaks exact update/interaction continuity",
        )
        count = row.get("completed_episode_count")
        successes = row.get("target_success_count")
        strict_successes = row.get("successful_episode_count")
        failure_terminations = row.get("failure_termination_count")
        truncations = row.get("time_limit_truncation_count")
        for name, value in (
            ("completed_episode_count", count),
            ("target_success_count", successes),
            ("successful_episode_count", strict_successes),
            ("failure_termination_count", failure_terminations),
            ("time_limit_truncation_count", truncations),
        ):
            _require(
                type(value) is int and value >= 0,
                f"{candidate_id}: history row {index} has invalid {name}",
            )
        counts = row.get("failure_cause_counts")
        _require(
            isinstance(counts, dict) and set(counts) == set(failure_counts),
            f"{candidate_id}: history row {index} lacks the exact failure-cause schema",
        )
        _require(
            all(type(counts[code]) is int and counts[code] >= 0 for code in failure_counts),
            f"{candidate_id}: history row {index} has invalid failure counts",
        )
        _require(
            sum(counts.values()) == failure_terminations,
            f"{candidate_id}: history row {index} failure totals disagree",
        )
        _require(
            count == failure_terminations + truncations,
            f"{candidate_id}: history row {index} completed episode totals disagree",
        )
        for code in failure_counts:
            failure_counts[code] += counts[code]
        success_events += successes
        successful_episodes += strict_successes
        completed_episodes += count
        final_distance = row.get("final_goal_distance_mean_m")
        components = row.get("episodic_reward_component_means")
        if count:
            _require(
                type(final_distance) is float and math.isfinite(final_distance) and final_distance >= 0.0,
                f"{candidate_id}: history row {index} lacks finite completed-episode distance",
            )
            _require(
                isinstance(components, dict)
                and type(components.get("progress")) is float
                and math.isfinite(components["progress"]),
                f"{candidate_id}: history row {index} lacks finite completed-episode progress",
            )
            final_distance_sum += final_distance * count
            cumulative_progress += components["progress"] * count
        else:
            _require(final_distance is None, f"{candidate_id}: zero-episode row has a final distance")
            _require(components == {}, f"{candidate_id}: zero-episode row has episode reward components")

    _require(len(rows) == EXPECTED_UPDATES, f"{candidate_id}: history does not contain 100 rows")
    _require(completed_episodes > 0, f"{candidate_id}: no completed pilot episode exists")
    _require(failure_counts["4"] == 0, f"{candidate_id}: nonfinite failure count is nonzero")
    physical_failures = sum(failure_counts[code] for code in ("1", "2", "3"))
    _require(physical_failures == 0, f"{candidate_id}: physical/crash failure count is nonzero")
    return {
        "training_target_success_event_count": success_events,
        "successful_episode_count": successful_episodes,
        "completed_episode_count": completed_episodes,
        "final_goal_distance_over_completed_pilot_episodes_m": (
            final_distance_sum / completed_episodes
        ),
        "cumulative_progress_reward": cumulative_progress,
        "failure_cause_counts": failure_counts,
        "physical_failure_count": physical_failures,
        "nonfinite_failure_count": failure_counts["4"],
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = read_checkpoint(
            path, map_location="cpu", resolve_external_history=False
        )
    except Exception as exc:
        raise SelectionGateError(f"Cannot validate checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionGateError(f"Checkpoint payload is not an object: {path}")
    return payload


def _expected_resolved_fields(learning_rate: float) -> dict[str, Any]:
    fields = dict(COMMON_TRAINING)
    fields.pop("expected_updates")
    # ``standalone_resolved_config`` deliberately represents the default
    # integration protocol by omitting the optional evaluation_protocol key.
    # The declaration still names it so commands are unambiguous.
    fields.pop("evaluation_protocol")
    return {
        "kind": "standalone_training",
        "task": EXPECTED_TASK,
        "contract_profile": EXPECTED_PROFILE,
        "controller": EXPECTED_CONTROLLER,
        "seed": EXPECTED_SEED,
        **fields,
        "learning_rate": learning_rate,
    }


def _validate_candidate(
    declaration: dict[str, Any],
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = declaration["candidate_id"]
    learning_rate = declaration["learning_rate"]
    run_dir = (ROOT / declaration["run_dir"]).resolve()
    runs_root = (ROOT / "runs").resolve()
    _require(run_dir.is_relative_to(runs_root), f"{candidate_id}: run directory escapes runs/")
    _require(run_dir.is_dir(), f"{candidate_id}: run directory is missing: {run_dir}")
    manifest_path = run_dir / "training_manifest.json"
    manifest = _read_json_object(manifest_path, label=f"{candidate_id} training manifest")
    expected_manifest_fields = {
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
    }
    for field, expected in expected_manifest_fields.items():
        _require(manifest.get(field) == expected, f"{candidate_id}: manifest {field} differs")
    _require(
        type(manifest.get("completed_episodes")) is int and manifest["completed_episodes"] > 0,
        f"{candidate_id}: manifest completed_episodes is absent or zero",
    )

    checkpoint = run_dir / "checkpoints" / "latest.pt"
    _require(checkpoint.is_file(), f"{candidate_id}: latest checkpoint is missing")
    _require(
        _resolve_recorded_path(manifest.get("checkpoint"), label=f"{candidate_id} checkpoint")
        == checkpoint.resolve(),
        f"{candidate_id}: manifest points to another checkpoint",
    )
    checkpoint_sha = _sha256_file(checkpoint)
    _require(
        manifest.get("checkpoint_sha256") == checkpoint_sha,
        f"{candidate_id}: checkpoint SHA-256 differs",
    )

    resolved = manifest.get("resolved_config")
    _require(isinstance(resolved, dict), f"{candidate_id}: resolved config is missing")
    for field, expected in _expected_resolved_fields(learning_rate).items():
        _require(resolved.get(field) == expected, f"{candidate_id}: resolved config {field} differs")
    expected_resolved_keys = set(_expected_resolved_fields(learning_rate)) | {
        "rewire_seed",
        "rewire_manifest",
        "rewire_manifest_file_sha256",
        "rollout_rng_contract",
        "memory_acceptance",
        "balanced_v4_task_contract",
    }
    _require_exact_keys(
        resolved, expected_resolved_keys, label=f"{candidate_id} resolved config"
    )
    _require(
        "evaluation_protocol" not in resolved,
        f"{candidate_id}: default integration protocol must use omission semantics",
    )
    rewire_manifest = _resolve_recorded_path(
        resolved.get("rewire_manifest"), label=f"{candidate_id} rewire manifest"
    )
    _require(rewire_manifest.is_file(), f"{candidate_id}: rewire manifest is missing")
    _require(
        resolved.get("rewire_manifest_file_sha256") == _sha256_file(rewire_manifest),
        f"{candidate_id}: rewire manifest file hash differs",
    )
    _require(
        type(resolved.get("rewire_seed")) is int and resolved["rewire_seed"] >= 0,
        f"{candidate_id}: rewire seed is invalid",
    )
    _require(
        isinstance(resolved.get("rollout_rng_contract"), dict)
        and bool(resolved["rollout_rng_contract"]),
        f"{candidate_id}: rollout RNG contract is missing",
    )
    _require(
        isinstance(resolved.get("memory_acceptance"), dict)
        and resolved["memory_acceptance"].get("policy_version") == EXPECTED_MEMORY_POLICY,
        f"{candidate_id}: memory acceptance contract differs",
    )
    _require(
        isinstance(resolved.get("balanced_v4_task_contract"), dict),
        f"{candidate_id}: balanced_v4 task contract is missing",
    )
    _require(
        "balanced_task_contract" not in resolved and "survival_first_contract" not in resolved,
        f"{candidate_id}: another task profile leaked into the resolved config",
    )
    v4_contract = resolved["balanced_v4_task_contract"]
    for value_field, hash_field in (
        ("reward", "reward_sha256"),
        ("training_curriculum", "training_curriculum_sha256"),
    ):
        _require(
            isinstance(v4_contract.get(value_field), dict)
            and v4_contract.get(hash_field) == _canonical_sha256(v4_contract[value_field]),
            f"{candidate_id}: v4 {value_field} hash is missing or invalid",
        )

    fingerprint_payload = manifest.get("fingerprint_payload")
    fingerprint = manifest.get("fingerprint")
    _require(isinstance(fingerprint_payload, dict), f"{candidate_id}: fingerprint payload is missing")
    _require(
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and fingerprint == _canonical_sha256(fingerprint_payload),
        f"{candidate_id}: reproduction fingerprint does not authenticate its payload",
    )
    _require(
        fingerprint_payload.get("resolved_config") == resolved,
        f"{candidate_id}: fingerprint resolved config differs from the manifest",
    )
    _validate_source_fingerprint(fingerprint_payload, candidate_id=candidate_id)

    controller_report = manifest.get("controller_report")
    _require(isinstance(controller_report, dict), f"{candidate_id}: controller report is missing")
    _require(
        controller_report.get("controller_kind") == EXPECTED_CONTROLLER_KIND,
        f"{candidate_id}: controller is not frozen LIF",
    )
    core = controller_report.get("core_checksum")
    _require(isinstance(core, str) and len(core) == 64, f"{candidate_id}: frozen core checksum is missing")
    _require(
        manifest.get("core_checksum_before") == core
        and manifest.get("core_checksum_after") == core,
        f"{candidate_id}: frozen core changed during training",
    )

    payload = checkpoint_loader(checkpoint)
    metadata = payload.get("metadata")
    counters = payload.get("counters")
    fingerprints = payload.get("fingerprints")
    _require(isinstance(metadata, dict), f"{candidate_id}: checkpoint metadata is missing")
    _require(isinstance(counters, dict), f"{candidate_id}: checkpoint counters are missing")
    _require(isinstance(fingerprints, dict), f"{candidate_id}: checkpoint fingerprints are missing")
    _require(payload.get("resolved_config") == resolved, f"{candidate_id}: checkpoint config differs")
    _require(payload.get("interactions_per_update") == EXPECTED_INTERACTIONS_PER_UPDATE,
             f"{candidate_id}: checkpoint rollout shape differs")
    _require(
        counters.get("completed_updates") == EXPECTED_UPDATES
        and counters.get("total_interactions") == EXPECTED_INTERACTIONS
        and counters.get("completed_episodes") == manifest["completed_episodes"],
        f"{candidate_id}: checkpoint counters differ",
    )
    for field, expected in (
        ("status", "completed"),
        ("contract_profile", EXPECTED_PROFILE),
        ("controller", EXPECTED_CONTROLLER),
        ("seed", EXPECTED_SEED),
        ("requested_interactions", EXPECTED_INTERACTIONS),
        ("interactions_per_update", EXPECTED_INTERACTIONS_PER_UPDATE),
        ("controller_report", controller_report),
        ("core_checksum_before", core),
        ("core_checksum_after", core),
        ("memory_samples", manifest.get("memory_samples")),
        ("training_curriculum", manifest.get("training_curriculum")),
    ):
        _require(metadata.get(field) == expected, f"{candidate_id}: checkpoint metadata {field} differs")
    _require(payload.get("core_checksum") == core, f"{candidate_id}: checkpoint frozen core differs")
    _require(
        fingerprints.get("reproduction") == fingerprint
        and fingerprints.get("source_set")
        == _canonical_sha256(fingerprint_payload["source_sha256"])
        and fingerprints.get("frozen_core") == core,
        f"{candidate_id}: checkpoint fingerprints differ",
    )
    for identity in ("task_manifest_id", "evaluation_manifest_id"):
        _require(
            manifest.get(identity) == payload.get(identity),
            f"{candidate_id}: {identity} differs between manifest and checkpoint",
        )

    history_reference = manifest.get("history_reference")
    _require(
        isinstance(history_reference, dict)
        and history_reference == payload.get("history_reference")
        and payload.get("history") == [],
        f"{candidate_id}: immutable history reference differs",
    )
    try:
        rows = load_history_reference(history_reference, checkpoint_path=checkpoint)
    except Exception as exc:
        raise SelectionGateError(f"{candidate_id}: history authentication failed: {exc}") from exc
    metrics = _history_metrics(rows, candidate_id=candidate_id)
    _require(
        metrics["completed_episode_count"] == manifest["completed_episodes"],
        f"{candidate_id}: history completed episode count differs from manifest",
    )

    memory_gate = _validate_memory(
        manifest.get("memory_samples"), manifest.get("memory_gate"), candidate_id=candidate_id
    )
    evidence = {
        "candidate_id": candidate_id,
        "learning_rate": learning_rate,
        "run_dir": str(run_dir),
        "training_manifest": str(manifest_path.resolve()),
        "training_manifest_sha256": _sha256_file(manifest_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "reproduction_fingerprint": fingerprint,
        "source_set_sha256": fingerprints["source_set"],
        "frozen_core_sha256": core,
        "history_sha256": history_reference.get("history_sha256"),
        "history_segment_sha256": [segment["sha256"] for segment in history_reference["segments"]],
        "memory_gate": memory_gate,
        "metrics": metrics,
        "hard_gates_passed": True,
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return evidence


def _normalized_candidate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(value))
    normalized["learning_rate"] = "<candidate>"
    return normalized


def _select_winner(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(len(candidates) == 2, "Exactly two validated candidates are required")
    # Both candidates already passed every hard gate.  Python tuple ordering
    # implements the predeclared directions without a data-dependent rule.
    def key(item: dict[str, Any]) -> tuple[float | int, ...]:
        metrics = item["metrics"]
        return (
            -metrics["training_target_success_event_count"],
            metrics["final_goal_distance_over_completed_pilot_episodes_m"],
            -metrics["cumulative_progress_reward"],
            item["learning_rate"],
        )

    ranked = sorted(candidates, key=key)
    winner = ranked[0]
    ranking = [
        {
            "rank": index,
            "candidate_id": item["candidate_id"],
            "learning_rate": item["learning_rate"],
            "training_target_success_event_count": item["metrics"]["training_target_success_event_count"],
            "final_goal_distance_over_completed_pilot_episodes_m": item["metrics"][
                "final_goal_distance_over_completed_pilot_episodes_m"
            ],
            "cumulative_progress_reward": item["metrics"]["cumulative_progress_reward"],
        }
        for index, item in enumerate(ranked, start=1)
    ]
    return winner, ranking


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    runs_root = (ROOT / "runs").resolve()
    _require(destination.is_relative_to(runs_root), "Selection output must be inside runs/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable selection: {destination}")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def select_learning_rate(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_path: str | Path | None = None,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = _load_checkpoint,
) -> dict[str, Any]:
    config_source = Path(config_path).expanduser().resolve()
    config = _validate_declaration(config_source)
    configured_output = (ROOT / config["selection_output"]).resolve()
    destination = configured_output if output_path is None else Path(output_path).expanduser().resolve()
    _require(destination == configured_output, "Output path must equal the predeclared immutable path")
    _require(not destination.exists() and not destination.is_symlink(),
             f"Selection output already exists: {destination}")

    evidence = [
        _validate_candidate(candidate, checkpoint_loader=checkpoint_loader)
        for candidate in config["candidates"]
    ]
    # Apart from the LR itself, the resolved training configuration and the
    # executable/runtime fingerprint payload must be identical across screens.
    manifests = [
        _read_json_object(Path(item["training_manifest"]), label="training manifest")
        for item in evidence
    ]
    normalized_resolved = [
        _normalized_candidate_config(manifest["resolved_config"]) for manifest in manifests
    ]
    _require(normalized_resolved[0] == normalized_resolved[1],
             "Candidate resolved configs differ beyond learning_rate")
    normalized_fingerprints = []
    for manifest in manifests:
        payload = deepcopy(manifest["fingerprint_payload"])
        payload["resolved_config"] = _normalized_candidate_config(payload["resolved_config"])
        normalized_fingerprints.append(payload)
    _require(normalized_fingerprints[0] == normalized_fingerprints[1],
             "Candidate fingerprints differ beyond learning_rate")
    _require(manifests[0]["controller_report"] == manifests[1]["controller_report"],
             "Candidate controller identities differ")

    winner, ranking = _select_winner(evidence)
    receipt_body = {
        "schema_version": 1,
        "kind": "crazyflie_balanced_v4_original_lif_reach_lr_selection",
        "status": "selected",
        "task": EXPECTED_TASK,
        "contract_profile": EXPECTED_PROFILE,
        "controller": EXPECTED_CONTROLLER,
        "training_seed": EXPECTED_SEED,
        "screen_interactions_per_candidate": EXPECTED_INTERACTIONS,
        "screen_updates_per_candidate": EXPECTED_UPDATES,
        "declaration": str(config_source),
        "declaration_sha256": _sha256_file(config_source),
        "selector": str(Path(__file__).resolve()),
        "selector_sha256": _sha256_file(Path(__file__).resolve()),
        "selection_order": config["selection_order"],
        "held_out_evaluation_used": False,
        "warm_start_authorized": False,
        "all_hard_gates_passed": True,
        "candidates": evidence,
        "ranking": ranking,
        "selected_candidate_id": winner["candidate_id"],
        "selected_learning_rate": winner["learning_rate"],
        "selected_run_dir": winner["run_dir"],
        "selected_checkpoint": winner["checkpoint"],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
    }
    receipt = {
        **receipt_body,
        "selection_id": _canonical_sha256(receipt_body),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    _write_immutable_json(destination, receipt)
    return {**receipt, "selection_output": str(destination), "selection_output_sha256": _sha256_file(destination)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = select_learning_rate(args.config, output_path=args.output)
    except (SelectionGateError, FileExistsError, OSError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
