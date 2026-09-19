#!/usr/bin/env python3
"""Read-only, fail-closed verifier for the balanced-v4 Reach LIF proof.

The canonical proof is deliberately narrower than the historical mixed-task
gate: it authenticates one fresh original frozen-LIF Reach run, one real
checkpoint/resume transition, and exactly five held-out Reach episodes.  This
program never starts Isaac and never writes a receipt.  Its own SHA-256 is
reported so the verifier version used for a decision remains explicit without
changing the training/evaluation reproduction fingerprint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from drone_bootstrap import (
    DEFAULT_CONNECTOME,
    DEFAULT_REWIRE_MANIFEST,
    ROOT,
    canonical_sha256,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)
from drone_evaluation_protocol import load_protocol
from drone_evaluate import _validate_scenario_part
from drone_train import standalone_resolved_config
from g1_fly_control.crazyflie.checkpoint import read_checkpoint
from g1_fly_control.crazyflie.controllers import build_controller
from g1_fly_control.crazyflie.memory import (
    GPU_LIMIT_MIB,
    MEMORY_POLICY_VERSION,
    RAM_LIMIT_PERCENT,
    assess as assess_memory,
)


SCRIPT_PATH = Path(__file__).resolve()
SELECTION_PATH = (
    ROOT / "runs" / "crazyflie-balanced-v4-lif-reach-lr-screen-selection.json"
)
PROOF_RUN_DIR = ROOT / "runs" / "crazyflie-balanced-v4-lif-reach-proof-seed5"
PROOF_CHECKPOINT = PROOF_RUN_DIR / "checkpoints" / "latest.pt"
PROOF_TRAINING_MANIFEST = PROOF_RUN_DIR / "training_manifest.json"
PROOF_EVALUATION = PROOF_RUN_DIR / "evaluation-reach-lif-proof.json"

EXPECTED_TASK = "FlyCrazyflie-WaypointReach-v0"
EXPECTED_PROFILE = "balanced_v4"
EXPECTED_CONTROLLER = "frozen_lif_original"
EXPECTED_CONTROLLER_KIND = "frozen_lif"
EXPECTED_SEED = 5
EXPECTED_NUM_ENVS = 4
EXPECTED_HORIZON = 100
EXPECTED_INTERACTIONS_PER_UPDATE = EXPECTED_NUM_ENVS * EXPECTED_HORIZON
EXPECTED_INTERACTIONS = 500_000
EXPECTED_UPDATES = EXPECTED_INTERACTIONS // EXPECTED_INTERACTIONS_PER_UPDATE
EXPECTED_RESUME_COUNT = 1
EXPECTED_PAUSE_UPDATE = 100
EXPECTED_PROTOCOL = "lif_proof"
EXPECTED_EVALUATION_SEED = 101
EXPECTED_EPISODES = 5
EXPECTED_CHECKPOINT_EVERY_UPDATES = 100
EXPECTED_PPO = {
    "burn_in": 0,
    "horizon": EXPECTED_HORIZON,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.002,
    "ppo_epochs": 2,
    "max_grad_norm": 1.0,
    "target_kl": 0.05,
}

_RESUME_SEGMENT_RE = re.compile(r"-resume-(\d{4})-")


class ReachProofGateError(ValueError):
    """The canonical Reach proof is absent, stale, malformed, or unsuccessful."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReachProofGateError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ReachProofGateError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReachProofGateError(f"{label} must be a JSON object: {path}")
    return value


def _require_finite_json(value: Any, *, location: str) -> None:
    """Reject nonfinite or non-JSON evidence recursively."""

    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        _require(math.isfinite(value), f"Nonfinite float at {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), f"Non-string object key at {location}")
            _require_finite_json(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, location=f"{location}[{index}]")
        return
    raise ReachProofGateError(
        f"Unsupported {type(value).__name__} evidence at {location}"
    )


def _require_finite_tensors(value: Any, *, location: str) -> None:
    """Reject NaN/Inf from tensor-bearing checkpoint state."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            _require(
                bool(torch.isfinite(value).all().item()),
                f"Checkpoint contains a nonfinite tensor at {location}",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tensors(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_tensors(item, location=f"{location}[{index}]")
        return
    if type(value) is float:
        _require(math.isfinite(value), f"Checkpoint contains a nonfinite float at {location}")


def _validate_selection(selection_path: Path) -> dict[str, Any]:
    """Recompute both screen gates and the predeclared winner."""

    try:
        import drone_select_reach_lif_lr as selector
    except Exception as exc:  # pragma: no cover - damaged installation
        raise ReachProofGateError(f"Cannot import the frozen LR selector: {exc}") from exc

    selection_path = selection_path.resolve()
    expected_path = (ROOT / selector.EXPECTED_OUTPUT).resolve()
    _require(
        selection_path == expected_path,
        f"Selection path must be the predeclared path {expected_path}",
    )
    _require(selection_path.is_file(), f"Missing LR selection receipt: {selection_path}")
    selection_sha_start = sha256_file(selection_path)
    receipt = _read_json_object(selection_path, label="LR selection receipt")
    _require_finite_json(receipt, location="selection")

    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "task",
        "contract_profile",
        "controller",
        "training_seed",
        "screen_interactions_per_candidate",
        "screen_updates_per_candidate",
        "declaration",
        "declaration_sha256",
        "selector",
        "selector_sha256",
        "selection_order",
        "held_out_evaluation_used",
        "warm_start_authorized",
        "all_hard_gates_passed",
        "candidates",
        "ranking",
        "selected_candidate_id",
        "selected_learning_rate",
        "selected_run_dir",
        "selected_checkpoint",
        "selected_checkpoint_sha256",
        "selection_id",
        "created_at_utc",
    }
    _require(set(receipt) == expected_keys, "LR selection receipt fields differ")
    for field, expected in (
        ("schema_version", 1),
        ("kind", "crazyflie_balanced_v4_original_lif_reach_lr_selection"),
        ("status", "selected"),
        ("task", EXPECTED_TASK),
        ("contract_profile", EXPECTED_PROFILE),
        ("controller", EXPECTED_CONTROLLER),
        ("training_seed", selector.EXPECTED_SEED),
        ("screen_interactions_per_candidate", selector.EXPECTED_INTERACTIONS),
        ("screen_updates_per_candidate", selector.EXPECTED_UPDATES),
        ("held_out_evaluation_used", False),
        ("warm_start_authorized", False),
        ("all_hard_gates_passed", True),
    ):
        _require(receipt.get(field) == expected, f"LR selection field {field} differs")

    declaration = Path(receipt["declaration"]).resolve()
    selector_path = Path(receipt["selector"]).resolve()
    _require(
        declaration == selector.DEFAULT_CONFIG.resolve()
        and receipt["declaration_sha256"] == sha256_file(declaration),
        "LR selection declaration path/hash changed",
    )
    _require(
        selector_path == Path(selector.__file__).resolve()
        and receipt["selector_sha256"] == sha256_file(selector_path),
        "LR selector path/hash changed",
    )
    config = selector._validate_declaration(declaration)
    _require(
        receipt["selection_order"] == config["selection_order"],
        "LR selection ordering differs from its predeclaration",
    )

    try:
        evidence = [
            selector._validate_candidate(
                row, checkpoint_loader=selector._load_checkpoint
            )
            for row in config["candidates"]
        ]
        winner, ranking = selector._select_winner(evidence)
    except Exception as exc:
        raise ReachProofGateError(f"LR screen evidence no longer validates: {exc}") from exc
    _require(receipt["candidates"] == evidence, "LR candidate evidence does not recompute")
    _require(receipt["ranking"] == ranking, "LR ranking does not recompute")
    for field, expected in (
        ("selected_candidate_id", winner["candidate_id"]),
        ("selected_learning_rate", winner["learning_rate"]),
        ("selected_run_dir", winner["run_dir"]),
        ("selected_checkpoint", winner["checkpoint"]),
        ("selected_checkpoint_sha256", winner["checkpoint_sha256"]),
    ):
        _require(receipt[field] == expected, f"LR winner field {field} does not recompute")
    body = {
        key: deepcopy(value)
        for key, value in receipt.items()
        if key not in {"selection_id", "created_at_utc"}
    }
    _require(
        receipt["selection_id"] == canonical_sha256(body),
        "LR selection_id does not authenticate its receipt body",
    )
    learning_rate = receipt["selected_learning_rate"]
    _require(
        type(learning_rate) is float and learning_rate in selector.EXPECTED_LRS,
        "Selected learning rate is not one of the two predeclared candidates",
    )
    _require(
        selection_sha_start == sha256_file(selection_path),
        "LR selection receipt changed during validation",
    )
    return {
        "learning_rate": learning_rate,
        "selection_id": receipt["selection_id"],
        "selection_sha256": selection_sha_start,
        "selection_path": str(selection_path),
        "selected_candidate_id": receipt["selected_candidate_id"],
    }


def _expected_resolved_config(learning_rate: float) -> tuple[dict[str, Any], dict[str, Any]]:
    args = SimpleNamespace(
        evaluation_protocol=EXPECTED_PROTOCOL,
        task=EXPECTED_TASK,
        contract_profile=EXPECTED_PROFILE,
        policy=EXPECTED_CONTROLLER,
        seed=EXPECTED_SEED,
        num_envs=EXPECTED_NUM_ENVS,
        total_interactions=EXPECTED_INTERACTIONS,
        horizon=EXPECTED_HORIZON,
        microbatch_size=EXPECTED_NUM_ENVS,
        ppo_epochs=2,
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        max_grad_norm=1.0,
        target_kl=0.05,
        checkpoint_every_updates=EXPECTED_CHECKPOINT_EVERY_UPDATES,
        rewire_seed=20260916,
        rewire_manifest=DEFAULT_REWIRE_MANIFEST.resolve(),
        warm_start_checkpoint=None,
    )
    resolved, protocol = standalone_resolved_config(args)
    _require(set(key for key in resolved if key.endswith("_task_contract")) == {
        "balanced_v4_task_contract"
    }, "Expected config does not contain exactly the balanced-v4 task contract")
    _require("warm_start_source" not in resolved, "Canonical proof must be a fresh run")
    return resolved, protocol


def _profile_hashes(expected_config: dict[str, Any]) -> dict[str, str]:
    contract = expected_config.get("balanced_v4_task_contract")
    _require(isinstance(contract, dict), "Balanced-v4 task contract is missing")
    reward = contract.get("reward")
    curriculum = contract.get("training_curriculum")
    switch = contract.get("switch_targets")
    _require(
        isinstance(reward, dict) and isinstance(curriculum, dict) and isinstance(switch, dict),
        "Balanced-v4 reward/curriculum/switch contracts are malformed",
    )
    reward_sha = canonical_sha256(reward)
    curriculum_sha = canonical_sha256(curriculum)
    switch_sha = canonical_sha256(switch)
    _require(contract.get("reward_sha256") == reward_sha, "Balanced-v4 reward hash differs")
    _require(
        contract.get("training_curriculum_sha256") == curriculum_sha,
        "Balanced-v4 curriculum hash differs",
    )
    if "switch_targets_sha256" in contract:
        _require(
            contract["switch_targets_sha256"] == switch_sha,
            "Balanced-v4 switch-target hash differs",
        )
    starts = [stage.get("start_interactions") for stage in curriculum.get("stages", [])]
    _require(
        starts == [0, 50_000, 125_000, 250_000],
        "Balanced-v4 curriculum boundaries differ from the predeclared proof",
    )
    for label, payload in (("reward", reward), ("curriculum", curriculum), ("switch", switch)):
        version = payload.get("version")
        _require(isinstance(version, str) and bool(version), f"Balanced-v4 {label} version is missing")
    return {
        "task_contract_sha256": canonical_sha256(contract),
        "reward_sha256": reward_sha,
        "training_curriculum_sha256": curriculum_sha,
        "switch_targets_sha256": switch_sha,
        "reward_version_sha256": canonical_sha256(reward["version"]),
        "training_curriculum_version_sha256": canonical_sha256(curriculum["version"]),
        "switch_targets_version_sha256": canonical_sha256(switch["version"]),
    }


def _expected_curriculum_snapshot(expected_config: dict[str, Any], interactions: int) -> dict[str, Any]:
    stages = expected_config["balanced_v4_task_contract"]["training_curriculum"]["stages"]
    active_index = max(
        index
        for index, stage in enumerate(stages)
        if int(stage["start_interactions"]) <= interactions
    )
    active = stages[active_index]
    return {
        "training_interactions": interactions,
        "active_stage_index": active_index,
        "active_stage_name": active["name"],
        "active_stage": active,
    }


def _validate_training_history(
    history: list[Any], expected_config: dict[str, Any]
) -> dict[str, int]:
    _require(len(history) == EXPECTED_UPDATES, "Proof history must contain exactly 1,250 rows")
    completed_episodes = 0
    success_events = 0
    strict_successes = 0
    truncations = 0
    ppo_rejected_updates = 0
    for index, row in enumerate(history):
        _require(isinstance(row, dict), f"Training history row {index} is not an object")
        _require_finite_json(row, location=f"history[{index}]")
        update = index + 1
        interactions = update * EXPECTED_INTERACTIONS_PER_UPDATE
        _require(
            type(row.get("completed_updates")) is int
            and row["completed_updates"] == update
            and type(row.get("total_interactions")) is int
            and row["total_interactions"] == interactions
            and type(row.get("rollout_start_interactions")) is int
            and row["rollout_start_interactions"] == interactions - EXPECTED_INTERACTIONS_PER_UPDATE
            and row.get("training_curriculum_interactions") == interactions,
            f"Training history row {index} breaks exact update/interaction continuity",
        )
        expected_stage = _expected_curriculum_snapshot(expected_config, interactions)
        _require(
            row.get("active_training_curriculum_stage_index")
            == expected_stage["active_stage_index"]
            and row.get("active_training_curriculum_stage_name")
            == expected_stage["active_stage_name"],
            f"Training history row {index} has the wrong balanced-v4 curriculum stage",
        )
        counts = row.get("failure_cause_counts")
        _require(
            isinstance(counts, dict) and set(counts) == {"1", "2", "3", "4"},
            f"Training history row {index} lacks exact failure-cause counts",
        )
        _require(
            all(type(counts[code]) is int and counts[code] == 0 for code in counts),
            f"Training history row {index} records a crash/OOB/nonfinite failure",
        )
        episode_count = row.get("completed_episode_count")
        event_count = row.get("target_success_count")
        strict_count = row.get("successful_episode_count")
        failure_count = row.get("failure_termination_count")
        truncated_count = row.get("time_limit_truncation_count")
        for name, value in (
            ("completed_episode_count", episode_count),
            ("target_success_count", event_count),
            ("successful_episode_count", strict_count),
            ("failure_termination_count", failure_count),
            ("time_limit_truncation_count", truncated_count),
        ):
            _require(
                type(value) is int and value >= 0,
                f"Training history row {index} has invalid {name}",
            )
        _require(
            failure_count == 0
            and episode_count == truncated_count
            and strict_count <= event_count <= episode_count,
            f"Training history row {index} episode aggregates disagree",
        )
        rejected_step = row.get("rejected_step")
        _require(
            type(rejected_step) is bool,
            f"Training history row {index} has invalid rejected_step evidence",
        )
        # A rejected step is the expected, safely rolled-back outcome of the
        # configured PPO KL guard.  The execution plan does not define it as a
        # crash/nonfinite hard-gate failure, so retain it as explicit evidence
        # rather than silently turning the safety mechanism into a new gate.
        ppo_rejected_updates += int(rejected_step)
        completed_episodes += episode_count
        success_events += event_count
        strict_successes += strict_count
        truncations += truncated_count
        _require(
            row.get("completed_episodes") == completed_episodes,
            f"Training history row {index} cumulative episode count disagrees",
        )
    return {
        "completed_episodes": completed_episodes,
        "target_success_events": success_events,
        "strict_successes": strict_successes,
        "time_limit_truncations": truncations,
        "ppo_rejected_updates": ppo_rejected_updates,
        "physical_failures": 0,
        "nonfinite_failures": 0,
    }


def _resume_split_from_history_reference(reference: Any) -> int:
    _require(isinstance(reference, dict), "Proof history reference is missing")
    segments = reference.get("segments")
    _require(isinstance(segments, list) and len(segments) >= 2, "Proof requires pre/post-resume history segments")
    resume_ids: list[int] = []
    for index, segment in enumerate(segments):
        _require(isinstance(segment, dict), f"History segment {index} is malformed")
        path = segment.get("path")
        _require(isinstance(path, str), f"History segment {index} path is missing")
        match = _RESUME_SEGMENT_RE.search(path)
        _require(match is not None, f"History segment {index} lacks a resume generation")
        resume_ids.append(int(match.group(1)))
    _require(set(resume_ids) == {0, 1}, "Proof history must contain exactly resume generations 0 and 1")
    first_resume_one = resume_ids.index(1)
    _require(
        all(value == 0 for value in resume_ids[:first_resume_one])
        and all(value == 1 for value in resume_ids[first_resume_one:]),
        "Proof history resume generations are interleaved",
    )
    prior = segments[first_resume_one - 1]
    following = segments[first_resume_one]
    split_update = prior.get("last_completed_updates")
    _require(
        type(split_update) is int
        and split_update == EXPECTED_PAUSE_UPDATE
        and following.get("first_completed_updates") == split_update + 1
        and prior.get("last_total_interactions")
        == split_update * EXPECTED_INTERACTIONS_PER_UPDATE
        and following.get("first_total_interactions")
        == (split_update + 1) * EXPECTED_INTERACTIONS_PER_UPDATE,
        "Proof history does not have the exact contiguous update-100 resume boundary",
    )
    return split_update


def _option_value(command: Any, option: str) -> str | None:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    positions = [index for index, item in enumerate(command) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return command[positions[0] + 1]


def _validate_resume_boundary(
    *, checkpoint: Path, history_reference: dict[str, Any], current_fingerprint: str
) -> dict[str, Any]:
    split_update = _resume_split_from_history_reference(history_reference)
    boundary = checkpoint.parent / f"update-{split_update:08d}.pt"
    _require(boundary.is_file(), f"Missing immutable pre-resume checkpoint: {boundary}")
    boundary_sha_start = sha256_file(boundary)
    try:
        payload = read_checkpoint(boundary, map_location="cpu", resolve_external_history=False)
    except Exception as exc:
        raise ReachProofGateError(f"Pre-resume checkpoint is invalid: {exc}") from exc
    counters = payload.get("counters")
    _require(isinstance(counters, dict), "Pre-resume checkpoint counters are missing")
    _require(
        counters.get("completed_updates") == split_update
        and counters.get("total_interactions")
        == split_update * EXPECTED_INTERACTIONS_PER_UPDATE
        and counters.get("resume_count") == 0
        and counters.get("resume_reset") is False,
        "Pre-resume checkpoint is not a fresh-run boundary",
    )
    command = payload.get("command")
    pause_value = _option_value(command, "--pause_after_updates")
    _require(
        pause_value is not None
        and pause_value.isdigit()
        and int(pause_value) == split_update
        and "--resume" not in command,
        "Pre-resume checkpoint does not authenticate the bounded pause",
    )
    _require(
        payload.get("fingerprints", {}).get("reproduction") == current_fingerprint,
        "Pre-resume checkpoint fingerprint differs",
    )
    boundary_reference = payload.get("history_reference")
    _require(
        isinstance(boundary_reference, dict)
        and boundary_reference.get("row_count") == split_update
        and boundary_reference.get("last_completed_updates") == split_update
        and boundary_reference.get("last_total_interactions")
        == split_update * EXPECTED_INTERACTIONS_PER_UPDATE,
        "Pre-resume checkpoint history boundary differs",
    )
    _require(
        boundary_sha_start == sha256_file(boundary),
        "Pre-resume checkpoint changed during validation",
    )
    return {
        "split_update": split_update,
        "split_interactions": split_update * EXPECTED_INTERACTIONS_PER_UPDATE,
        "checkpoint": str(boundary.resolve()),
        "checkpoint_sha256": boundary_sha_start,
    }


def _validate_memory(samples: Any, gate: Any, *, label: str) -> dict[str, Any]:
    _require(
        isinstance(samples, list)
        and bool(samples)
        and all(isinstance(sample, dict) for sample in samples),
        f"{label} memory samples are missing or malformed",
    )
    _require_finite_json(samples, location=f"{label}.memory_samples")
    stages = [sample.get("stage") for sample in samples]
    required = {
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    }
    _require(required.issubset(stages), f"{label} memory stages are incomplete: {stages}")
    _require(
        stages.count("environment_loaded") >= 2 and stages.count("controller_loaded") >= 2,
        f"{label} memory evidence does not show two trainer processes",
    )
    try:
        recomputed = assess_memory(samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReachProofGateError(f"{label} memory evidence cannot be assessed: {exc}") from exc
    _require(isinstance(gate, dict), f"{label} memory gate is missing")
    _require(canonical_sha256(gate) == canonical_sha256(recomputed), f"{label} memory gate does not recompute")
    limits = recomputed.get("limits")
    _require(
        recomputed.get("policy_version") == MEMORY_POLICY_VERSION
        and recomputed.get("passed") is True
        and recomputed.get("failures") == []
        and recomputed.get("device_gpu_telemetry_complete") is True
        and recomputed.get("sustained_paging_detected") is False
        and isinstance(limits, dict)
        and limits.get("gpu_used_mib_exclusive") == GPU_LIMIT_MIB
        and limits.get("system_ram_percent_exclusive") == RAM_LIMIT_PERCENT
        and limits.get("rss_growth_disposition") == "warning_only"
        and float(recomputed.get("max_device_gpu_used_mib", math.inf)) < GPU_LIMIT_MIB
        and float(recomputed.get("max_system_ram_percent", math.inf)) < RAM_LIMIT_PERCENT,
        f"{label} RAM/VRAM/paging gate failed",
    )
    warnings = recomputed.get("warnings")
    _require(
        isinstance(warnings, list)
        and all(isinstance(item, str) and bool(item) for item in warnings),
        f"{label} memory warnings are malformed",
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
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    _require_finite_json(manifest, location="training_manifest")
    _require(payload.get("policy_class") == expected_policy_class, "Proof policy class changed")
    _require(payload.get("resolved_config") == expected_config, "Checkpoint resolved config is not exact")
    _require(
        payload.get("resolved_config_checksum") == canonical_sha256(expected_config),
        "Checkpoint resolved-config hash differs",
    )
    _require(
        payload.get("interactions_per_update") == EXPECTED_INTERACTIONS_PER_UPDATE,
        "Checkpoint interactions per update is not exactly 400",
    )
    counters = payload.get("counters")
    _require(isinstance(counters, dict), "Checkpoint counters are missing")
    _require(
        counters.get("completed_updates") == EXPECTED_UPDATES
        and counters.get("total_interactions") == EXPECTED_INTERACTIONS
        and counters.get("resume_count") == EXPECTED_RESUME_COUNT
        and counters.get("resume_reset") is True,
        "Checkpoint is not exactly 500k/1,250 updates with one real resume",
    )
    history = payload.get("history")
    _require(isinstance(history, list), "Resolved checkpoint history is missing")
    history_summary = _validate_training_history(history, expected_config)
    _require(
        counters.get("completed_episodes") == history_summary["completed_episodes"],
        "Checkpoint/history completed-episode totals differ",
    )

    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "Checkpoint metadata is missing")
    final_curriculum = _expected_curriculum_snapshot(expected_config, EXPECTED_INTERACTIONS)
    expected_core = expected_controller_report.get("core_checksum")
    _require(isinstance(expected_core, str) and len(expected_core) == 64, "Current LIF core checksum is invalid")
    for field, expected in (
        ("status", "completed"),
        ("contract_profile", EXPECTED_PROFILE),
        ("controller", EXPECTED_CONTROLLER),
        ("seed", EXPECTED_SEED),
        ("requested_interactions", EXPECTED_INTERACTIONS),
        ("interactions_per_update", EXPECTED_INTERACTIONS_PER_UPDATE),
        ("controller_report", expected_controller_report),
        ("core_checksum_before", expected_core),
        ("core_checksum_after", expected_core),
        ("resume_reset", True),
        ("training_curriculum", final_curriculum),
        ("mixed_scenario", None),
        ("mixed_schedule_resume", None),
        ("warm_start", None),
    ):
        _require(metadata.get(field) == expected, f"Checkpoint metadata field {field} differs")
    _require(payload.get("core_checksum") == expected_core, "Frozen LIF core changed during proof training")

    controller_rng = metadata.get("controller_rng_initialization")
    rollout_rng = metadata.get("rollout_rng_initialization")
    _require(
        isinstance(controller_rng, dict)
        and controller_rng.get("seed") == EXPECTED_SEED
        and controller_rng.get("controller_construction_reseed_applied") is True,
        "Checkpoint controller RNG evidence is malformed",
    )
    _require(
        isinstance(rollout_rng, dict)
        and rollout_rng.get("seed") == EXPECTED_SEED
        and rollout_rng.get("fresh_reseed_applied") is False
        and rollout_rng.get("checkpoint_rng_preserved") is True,
        "Checkpoint does not authenticate restored rollout RNG state",
    )
    command = payload.get("command")
    _require(
        isinstance(command, list)
        and command.count("--resume") == 1
        and "--warm_start_checkpoint" not in command,
        "Final checkpoint command does not authenticate exactly one resume",
    )

    expected_fingerprints = {
        "reproduction": current_fingerprint,
        "source_set": canonical_sha256(current_fingerprint_payload["source_sha256"]),
        "connectome": expected_controller_report["connectome_checksum"],
        "frozen_core": expected_core,
        "rewire_manifest": current_fingerprint_payload["rewired_manifest_sha256"],
    }
    _require(payload.get("fingerprints") == expected_fingerprints, "Checkpoint fingerprints differ")
    _require(
        payload.get("evaluation_manifest_id") == protocol["manifest_id"],
        "Checkpoint evaluation protocol hash differs",
    )
    task_manifest_payload = {
        "task": EXPECTED_TASK,
        "episode_steps": 600,
        "control_dt_s": 0.02,
        "observation_width": 12,
        "action_width": 4,
        "contract_profile": EXPECTED_PROFILE,
        "balanced_v4_task_contract": expected_config["balanced_v4_task_contract"],
        "mixed_scenario_contract": None,
    }
    expected_task_manifest_id = canonical_sha256(task_manifest_payload)
    _require(
        payload.get("task_manifest_id") == expected_task_manifest_id,
        "Checkpoint task/profile manifest hash differs",
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
        "completed_episodes": history_summary["completed_episodes"],
        "resume_count": EXPECTED_RESUME_COUNT,
        "resume_reset": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "fingerprint": current_fingerprint,
        "fingerprint_payload": current_fingerprint_payload,
        "resolved_config": expected_config,
        "task_manifest_id": expected_task_manifest_id,
        "evaluation_manifest_id": protocol["manifest_id"],
        "controller_report": expected_controller_report,
        "controller_rng_initialization": controller_rng,
        "rollout_rng_initialization": rollout_rng,
        "warm_start": None,
        "training_curriculum": final_curriculum,
        "mixed_scenario": None,
        "mixed_schedule_resume": None,
        "core_checksum_before": expected_core,
        "core_checksum_after": expected_core,
    }
    for field, expected in required_manifest.items():
        _require(manifest.get(field) == expected, f"Training manifest field {field} differs")
    _require(manifest.get("command") == command, "Manifest/checkpoint commands differ")
    expected_ppo = {**EXPECTED_PPO, "learning_rate": expected_config["learning_rate"]}
    _require(manifest.get("ppo") == expected_ppo, "Training manifest PPO settings differ")
    history_reference = payload.get("history_reference")
    _require(
        isinstance(history_reference, dict)
        and manifest.get("history_reference") == history_reference
        and history_reference.get("row_count") == EXPECTED_UPDATES
        and history_reference.get("last_completed_updates") == EXPECTED_UPDATES
        and history_reference.get("last_total_interactions") == EXPECTED_INTERACTIONS,
        "Checkpoint/manifest history hashes or boundaries differ",
    )
    _require(
        canonical_sha256(metadata.get("memory_samples"))
        == canonical_sha256(manifest.get("memory_samples")),
        "Checkpoint/manifest training memory samples differ",
    )
    memory = _validate_memory(
        manifest.get("memory_samples"), manifest.get("memory_gate"), label="Training"
    )
    resume = _validate_resume_boundary(
        checkpoint=checkpoint,
        history_reference=history_reference,
        current_fingerprint=current_fingerprint,
    )
    for state_field in (
        "policy_state",
        "optimizer_state",
        "scheduler_state",
        "normalizers",
        "recurrent_state",
    ):
        _require_finite_tensors(payload.get(state_field), location=f"checkpoint.{state_field}")
    return memory, history_summary, resume


def _validate_reach_outcomes(evaluation: dict[str, Any]) -> dict[str, int]:
    """Apply the empirical Reach gate after schema/summary authentication."""

    episodes = evaluation.get("episodes")
    summary = evaluation.get("summary")
    _require(isinstance(episodes, list) and len(episodes) == EXPECTED_EPISODES, "Reach proof needs exactly five episodes")
    _require(isinstance(summary, dict), "Reach evaluation summary is missing")
    event_episodes = 0
    strict_successes = 0
    for index, row in enumerate(episodes):
        _require(isinstance(row, dict), f"Reach episode {index} is not an object")
        target_count = row.get("target_success_event_count")
        _require(
            type(target_count) is int and target_count in {0, 1},
            f"Reach episode {index} has an invalid target-success count",
        )
        _require(
            type(row.get("success")) is bool and row["success"] is (target_count == 1),
            f"Reach episode {index} strict success disagrees with its event",
        )
        _require(
            row.get("crash") is False
            and row.get("out_of_bounds") is False
            and row.get("invalid_state") is False
            and row.get("terminated") is False
            and row.get("truncated") is True
            and row.get("completed_steps") == 600
            and row.get("failure_reason") is None,
            f"Reach episode {index} crashed, escaped, became invalid, or did not complete 600 steps",
        )
        if target_count:
            _require(
                isinstance(row.get("time_to_first_success_s"), (int, float))
                and not isinstance(row.get("time_to_first_success_s"), bool)
                and math.isfinite(float(row["time_to_first_success_s"])),
                f"Reach episode {index} success lacks finite timing evidence",
            )
            event_episodes += 1
        strict_successes += int(row["success"])
    _require(event_episodes >= 1, "Reach proof has zero genuine held-out target-success events")
    for field, expected in (
        ("episode_count", EXPECTED_EPISODES),
        ("expected_episode_count", EXPECTED_EPISODES),
        ("success_count", strict_successes),
        ("crash_count", 0),
        ("out_of_bounds_count", 0),
        ("invalid_state_count", 0),
        ("termination_count", 0),
        ("truncation_count", EXPECTED_EPISODES),
    ):
        _require(summary.get(field) == expected, f"Reach summary field {field} differs")
    return {
        "event_episode_count": event_episodes,
        "strict_success_count": strict_successes,
        "crash_count": 0,
        "out_of_bounds_count": 0,
        "invalid_state_count": 0,
    }


def _validate_evaluation(
    *,
    evaluation_path: Path,
    evaluation: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    current_fingerprint: str,
    current_fingerprint_payload: dict[str, Any],
    expected_controller_report: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    _require_finite_json(evaluation, location="evaluation")
    try:
        _validate_scenario_part(
            evaluation,
            scenario=EXPECTED_TASK,
            protocol_name=EXPECTED_PROTOCOL,
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=current_fingerprint,
            expected_training_seed=EXPECTED_SEED,
            expected_policy=EXPECTED_CONTROLLER,
        )
    except ValueError as exc:
        raise ReachProofGateError(f"Reach evaluation schema/summary validation failed: {exc}") from exc
    for field, expected in (
        ("schema_version", 1),
        ("status", "completed"),
        ("label", "lif_proof"),
        ("protocol", EXPECTED_PROTOCOL),
        ("scenario", EXPECTED_TASK),
        ("evaluation_seed", EXPECTED_EVALUATION_SEED),
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
        _require(evaluation.get(field) == expected, f"Reach evaluation field {field} differs")
    outcomes = _validate_reach_outcomes(evaluation)
    memory = evaluation.get("memory_gate")
    _require(isinstance(memory, dict) and memory.get("passed") is True, "Evaluation memory gate failed")
    _require(
        float(memory.get("max_device_gpu_used_mib", math.inf)) < GPU_LIMIT_MIB
        and float(memory.get("max_system_ram_percent", math.inf)) < RAM_LIMIT_PERCENT
        and memory.get("sustained_paging_detected") is False,
        "Evaluation RAM/VRAM/paging limits failed",
    )
    _require(checkpoint_sha256 == sha256_file(checkpoint), "Checkpoint changed during evaluation validation")
    return outcomes, memory


def verify_reach_lif_proof(
    *,
    selection: Path = SELECTION_PATH,
    checkpoint: Path = PROOF_CHECKPOINT,
    training_manifest: Path = PROOF_TRAINING_MANIFEST,
    evaluation: Path = PROOF_EVALUATION,
) -> dict[str, Any]:
    """Verify all proof evidence without starting Isaac or writing files."""

    verifier_sha_start = sha256_file(SCRIPT_PATH)
    selection = selection.resolve()
    checkpoint = checkpoint.resolve()
    training_manifest = training_manifest.resolve()
    evaluation = evaluation.resolve()
    for path, label in (
        (selection, "LR selection receipt"),
        (checkpoint, "proof checkpoint"),
        (training_manifest, "proof training manifest"),
        (evaluation, "proof Reach evaluation"),
    ):
        _require(path.is_file(), f"Missing {label}: {path}")
    _require(
        checkpoint == training_manifest.parent / "checkpoints" / "latest.pt"
        and evaluation.parent == training_manifest.parent,
        "Proof checkpoint, manifest, and evaluation do not share one run directory",
    )

    selected = _validate_selection(selection)
    expected_config, protocol = _expected_resolved_config(selected["learning_rate"])
    _require(protocol == load_protocol(EXPECTED_PROTOCOL), "Held-out lif_proof protocol changed")
    hashes = _profile_hashes(expected_config)

    policy, controller_report = build_controller(
        EXPECTED_CONTROLLER,
        observation_dim=12,
        action_dim=4,
        device="cpu",
        connectome_manifest=DEFAULT_CONNECTOME.resolve(),
        rewire_seed=20260916,
    )
    expected_policy_class = f"{type(policy).__module__}.{type(policy).__qualname__}"
    del policy
    _require(
        controller_report.get("controller_kind") == EXPECTED_CONTROLLER_KIND,
        "Current controller is not the original frozen LIF",
    )
    rewire_manifest = load_fingerprint_rewire_manifest(
        DEFAULT_REWIRE_MANIFEST.resolve(), expected_seed=20260916
    )
    fingerprint, fingerprint_payload = reproduction_fingerprint(
        resolved_config=expected_config,
        evaluation_manifest=protocol,
        connectome_manifest=DEFAULT_CONNECTOME.resolve(),
        rewired_manifest=rewire_manifest,
    )

    checkpoint_sha_start = sha256_file(checkpoint)
    manifest_sha_start = sha256_file(training_manifest)
    try:
        payload = read_checkpoint(
            checkpoint, map_location="cpu", resolve_external_history=True
        )
    except Exception as exc:
        raise ReachProofGateError(f"Proof checkpoint validation failed: {exc}") from exc
    manifest = _read_json_object(training_manifest, label="proof training manifest")
    training_memory, history_summary, resume = _validate_checkpoint_and_manifest(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha_start,
        payload=payload,
        manifest=manifest,
        expected_config=expected_config,
        expected_policy_class=expected_policy_class,
        expected_controller_report=controller_report,
        current_fingerprint=fingerprint,
        current_fingerprint_payload=fingerprint_payload,
        protocol=protocol,
    )
    _require(checkpoint_sha_start == sha256_file(checkpoint), "Checkpoint changed during training validation")
    _require(
        manifest_sha_start == sha256_file(training_manifest),
        "Training manifest changed during validation",
    )

    evaluation_sha_start = sha256_file(evaluation)
    evaluation_payload = _read_json_object(evaluation, label="proof Reach evaluation")
    outcomes, evaluation_memory = _validate_evaluation(
        evaluation_path=evaluation,
        evaluation=evaluation_payload,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha_start,
        current_fingerprint=fingerprint,
        current_fingerprint_payload=fingerprint_payload,
        expected_controller_report=controller_report,
        protocol=protocol,
    )
    _require(evaluation_sha_start == sha256_file(evaluation), "Evaluation changed during validation")
    verifier_sha_end = sha256_file(SCRIPT_PATH)
    _require(verifier_sha_start == verifier_sha_end, "Verifier changed while it was running")
    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "balanced_v4_original_lif_reach_empirical_proof",
        "read_only": True,
        "verifier": str(SCRIPT_PATH),
        "verifier_sha256": verifier_sha_end,
        "selection": selected,
        "task": EXPECTED_TASK,
        "contract_profile": EXPECTED_PROFILE,
        "controller": EXPECTED_CONTROLLER,
        "training_seed": EXPECTED_SEED,
        "evaluation_seed": EXPECTED_EVALUATION_SEED,
        "learning_rate": selected["learning_rate"],
        "environment_interactions": EXPECTED_INTERACTIONS,
        "completed_updates": EXPECTED_UPDATES,
        "resume_count": EXPECTED_RESUME_COUNT,
        "resume_boundary": resume,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_start,
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": manifest_sha_start,
        "history_reference_sha256": payload["history_reference"]["history_sha256"],
        "history_segment_sha256": [
            segment["sha256"] for segment in payload["history_reference"]["segments"]
        ],
        "evaluation": str(evaluation),
        "evaluation_sha256": evaluation_sha_start,
        "reproduction_fingerprint": fingerprint,
        "resolved_config_sha256": canonical_sha256(expected_config),
        "evaluation_manifest_id": protocol["manifest_id"],
        "profile_hashes": hashes,
        "training": history_summary,
        "held_out_reach": {
            **outcomes,
            "episode_count": EXPECTED_EPISODES,
            "event_score": f"{outcomes['event_episode_count']}/{EXPECTED_EPISODES}",
        },
        "training_memory": {
            key: training_memory[key]
            for key in (
                "policy_version",
                "max_device_gpu_used_mib",
                "max_system_ram_percent",
                "max_process_rss_mib",
                "monotonic_process_growth_detected",
                "sustained_paging_detected",
                "warnings",
            )
        },
        "evaluation_memory": {
            key: evaluation_memory[key]
            for key in (
                "policy_version",
                "max_device_gpu_used_mib",
                "max_system_ram_percent",
                "max_process_rss_mib",
                "monotonic_process_growth_detected",
                "sustained_paging_detected",
                "warnings",
            )
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--checkpoint", type=Path, default=PROOF_CHECKPOINT)
    parser.add_argument("--training_manifest", type=Path, default=PROOF_TRAINING_MANIFEST)
    parser.add_argument("--evaluation", type=Path, default=PROOF_EVALUATION)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = verify_reach_lif_proof(
            selection=args.selection,
            checkpoint=args.checkpoint,
            training_manifest=args.training_manifest,
            evaluation=args.evaluation,
        )
    except BaseException:
        try:
            verifier_sha = sha256_file(SCRIPT_PATH)
        except OSError:
            verifier_sha = None
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "gate": "balanced_v4_original_lif_reach_empirical_proof",
            "read_only": True,
            "verifier": str(SCRIPT_PATH),
            "verifier_sha256": verifier_sha,
            "error": traceback.format_exc(),
        }
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
