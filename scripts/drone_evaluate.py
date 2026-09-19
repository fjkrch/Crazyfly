#!/usr/bin/env python3
"""Evaluate one Crazyflie checkpoint on immutable held-out episode plans."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any
import uuid

import torch

from drone_bootstrap import (
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_SURVIVAL_V2,
    DEFAULT_REWIRE_MANIFEST,
    ROOT,
    canonical_sha256,
    launch_environment,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)
from drone_evaluation_protocol import SCENARIOS, load_protocol
from g1_fly_control.crazyflie.memory import MEMORY_POLICY_VERSION
from g1_fly_control.tasks.crazyflie.logic import (
    AUDITED_CRAZYFLIE_MASS_KG,
    CRAZYFLIE_MASS_ABS_TOL_KG,
    GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
)


POLICIES = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "gru_matched",
    "mlp_normal",
    "wing_lif",
    "leg_wing_lif",
)
POLICY_INFERENCE_DEVICE = torch.device("cuda:0")
POLICY_INFERENCE_BACKEND = "cuda_graph_action_only_v1"
POLICY_INFERENCE_PRECISION = "float32"
POLICY_GRAPH_CONFIG_CONTRACT = {
    "graph_api": "torch.cuda.CUDAGraph",
    "capture_after_checkpoint_load": True,
    "fixed_batch": True,
    "actor_path": "deterministic_mean_tanh",
    "parity_gate": "bitwise_action_and_recurrent_state",
    "parity_probe_steps": 3,
    "failure_mode": "fail_closed",
}
POLICY_BRIDGE_CONTRACT = {
    "gpu_to_cpu_policy_tensor_transfers_per_decision": 0,
    "cpu_to_gpu_policy_tensor_transfers_per_decision": 0,
    "observation_device": "cuda:0",
    "recurrent_state_device": "cuda:0",
    "reset_mask_device": "cuda:0",
    "action_device": "cuda:0",
    "reset_timing": "before_next_policy_action",
}

_LAST_MEMORY_SAMPLES: list[dict[str, Any]] = []
_LAST_MEMORY_GATE: dict[str, Any] | None = None

_SCENARIO_PART_NAMES = {
    "FlyCrazyflie-WaypointReach-v0": "01_waypoint_reach.json",
    "FlyCrazyflie-WaypointSwitch-v0": "02_waypoint_switch.json",
    "FlyCrazyflie-GustRecovery-v0": "03_gust_recovery.json",
}

_CONTRACT_PROFILE_FIELDS = {
    "survival_first_contract": CONTRACT_PROFILE_SURVIVAL_V2,
    "balanced_task_contract": CONTRACT_PROFILE_BALANCED_V3,
    "balanced_v4_task_contract": CONTRACT_PROFILE_BALANCED_V4,
}


def _contract_profile_from_resolved_config(resolved_config: Any) -> str:
    """Infer the one environment contract recorded by a checkpoint/config.

    Standalone training checkpoints record their contract at the top level,
    while matrix checkpoints retain the complete matrix config below
    ``resolved_config["matrix"]``. Treat both layouts identically and fail
    closed if the provenance is absent, ambiguous, malformed, or duplicated
    with different payloads.
    """

    if not isinstance(resolved_config, dict):
        raise ValueError("Checkpoint lacks a resolved configuration")
    scopes = [resolved_config]
    matrix_config = resolved_config.get("matrix")
    if matrix_config is not None:
        if not isinstance(matrix_config, dict):
            raise ValueError("Checkpoint matrix configuration is not a JSON object")
        scopes.append(matrix_config)
    declarations = {
        field: [scope[field] for scope in scopes if field in scope]
        for field in _CONTRACT_PROFILE_FIELDS
    }
    present = [field for field, values in declarations.items() if values]
    if len(present) != 1:
        raise ValueError(
            "Checkpoint resolved configuration must contain exactly one of "
            "survival_first_contract, balanced_task_contract, or "
            "balanced_v4_task_contract"
        )
    field = present[0]
    values = declarations[field]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Checkpoint {field} must be a JSON object")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Checkpoint has conflicting duplicate {field} payloads")
    return _CONTRACT_PROFILE_FIELDS[field]

# These fields must be byte-for-byte equivalent as canonical JSON across the
# three isolated Isaac children. ``generated_at_utc`` is intentionally absent:
# the merged bundle receives its own parent-process generation time.
_SHARED_PART_FIELDS = (
    "schema_version",
    "status",
    "label",
    "protocol",
    "evaluation_seed",
    "evaluation_manifest_id",
    "checkpoint",
    "checkpoint_sha256",
    "training_seed",
    "controller",
    "fingerprint",
    "fingerprint_payload",
    "controller_report",
    "deterministic_actions",
    "simulation_device",
    "simulation_device_type",
    "policy_inference_device",
    "policy_inference_device_type",
    "policy_inference_backend",
    "policy_inference_precision",
    "policy_inference_graph",
    "policy_bridge",
    "failure_denominator_rule",
    "censoring_rule",
    "integrated_error_censoring_rule",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Durably publish JSON while preserving any prior artifact as evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                prior = path.read_bytes()
                history = path.parent / f"{path.name}.history"
                history.mkdir(parents=True, exist_ok=True)
                archive = history / (
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
                    + uuid.uuid4().hex
                    + ".json"
                )
                with archive.open("xb") as stream:
                    stream.write(prior)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _new_scenario_attempt_id() -> str:
    """Return a collision-resistant, sortable identity for one parent attempt."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def _scenario_part_paths(output: Path, attempt_id: str) -> dict[str, Path]:
    """Return immutable artifact paths scoped to one isolated parent attempt."""

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    if not attempt_id or any(character not in allowed for character in attempt_id):
        raise ValueError("Scenario attempt identity contains an unsupported character")
    part_directory = output.resolve().parent / f"{output.name}.parts" / attempt_id
    return {
        scenario: part_directory / _SCENARIO_PART_NAMES[scenario]
        for scenario in SCENARIOS
    }


def _expected_policy_graph_report(batch_size: int) -> dict[str, Any]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("CUDA-Graph policy batch size must be a positive integer")
    return {
        **POLICY_GRAPH_CONFIG_CONTRACT,
        "fixed_batch_size": batch_size,
        "capture_count": 1,
        "bitwise_parity_verified": True,
    }


def _evaluation_batch_size(plan_count: int) -> int:
    """Choose the largest validated CUDA-Graph batch that exactly divides a protocol."""

    if type(plan_count) is not int or plan_count < 1:
        raise ValueError("Evaluation plan count must be a positive integer")
    return max(
        candidate
        for candidate in range(1, min(4, plan_count) + 1)
        if plan_count % candidate == 0
    )


def _finite_gust_vectors(value: Any, *, field: str) -> list[list[float]]:
    """Return one exact three-event world-vector matrix or fail closed."""

    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three gust vectors")
    result: list[list[float]] = []
    for event_index, vector in enumerate(value):
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"{field}[{event_index}] must be a three-vector")
        if any(
            type(component) not in {int, float}
            or not math.isfinite(float(component))
            for component in vector
        ):
            raise ValueError(f"{field}[{event_index}] must be finite numeric evidence")
        result.append([float(component) for component in vector])
    return result


def _validated_gust_recovery_events(
    row: dict[str, Any], expected_plan: dict[str, Any]
) -> tuple[bool, bool, bool]:
    """Authenticate applied impulse before exposing per-gust recovery flags.

    A Boolean ``recovered`` claim is proof evidence only when the corresponding
    disturbance was actually submitted at the frozen mass-normalized magnitude
    and direction.  This helper is CPU-only and is reused by both scenario-part
    validation and the final proof scorer.
    """

    if not isinstance(row, dict) or not isinstance(expected_plan, dict):
        raise ValueError("Gust episode/plan evidence must be JSON objects")
    mass = row.get("robot_mass_kg")
    if (
        type(mass) not in {int, float}
        or not math.isfinite(float(mass))
        or float(mass) <= 0.0
        or not math.isclose(
            float(mass),
            AUDITED_CRAZYFLIE_MASS_KG,
            rel_tol=0.0,
            abs_tol=CRAZYFLIE_MASS_ABS_TOL_KG,
        )
    ):
        raise ValueError("Gust episode robot mass differs from the audited cf2x mass")

    gust_plans = expected_plan.get("gusts")
    outcomes = row.get("gust_outcomes")
    if not isinstance(gust_plans, list) or len(gust_plans) != 3:
        raise ValueError("Gust plan must contain exactly three frozen disturbances")
    if not isinstance(outcomes, list) or len(outcomes) != 3:
        raise ValueError("Gust episode must contain exactly three recovery outcomes")
    indexed_outcomes: dict[int, dict[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict) or type(outcome.get("gust_index")) is not int:
            raise ValueError("Gust recovery outcomes require integer event indices")
        event_index = outcome["gust_index"]
        if event_index in indexed_outcomes or event_index not in range(3):
            raise ValueError("Gust recovery outcome indices must be exactly 0, 1, 2")
        indexed_outcomes[event_index] = outcome
    if set(indexed_outcomes) != {0, 1, 2}:
        raise ValueError("Gust recovery outcome indices must be exactly 0, 1, 2")

    applied_vectors = _finite_gust_vectors(
        row.get("gust_applied_impulse_w_n_s"), field="gust_applied_impulse_w_n_s"
    )
    reported_expected_vectors = _finite_gust_vectors(
        row.get("gust_expected_impulse_w_n_s"), field="gust_expected_impulse_w_n_s"
    )
    reported_max_error = row.get("gust_impulse_max_abs_error_n_s")
    if (
        type(reported_max_error) not in {int, float}
        or not math.isfinite(float(reported_max_error))
        or float(reported_max_error) < 0.0
    ):
        raise ValueError("Gust submitted-impulse max error must be finite and non-negative")

    recovered_flags: list[bool] = []
    applied_errors: list[float] = []
    for event_index, gust in enumerate(gust_plans):
        if not isinstance(gust, dict):
            raise ValueError("Gust plan event must be a JSON object")
        direction = gust.get("direction_world_xy")
        delta_v = gust.get("desired_mass_normalized_delta_velocity_m_s")
        if (
            gust.get("event_index") != event_index + 1
            or not isinstance(direction, list)
            or len(direction) != 2
            or any(
                type(component) not in {int, float}
                or not math.isfinite(float(component))
                for component in direction
            )
            or type(delta_v) not in {int, float}
            or not math.isfinite(float(delta_v))
            or float(delta_v) <= 0.0
        ):
            raise ValueError("Gust plan direction/magnitude evidence is malformed")
        expected_vector = [
            float(mass) * float(delta_v) * float(direction[0]),
            float(mass) * float(delta_v) * float(direction[1]),
            0.0,
        ]
        if any(
            not math.isclose(
                reported,
                expected,
                rel_tol=0.0,
                abs_tol=GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
            )
            for reported, expected in zip(
                reported_expected_vectors[event_index], expected_vector, strict=True
            )
        ):
            raise ValueError("Recorded expected gust impulse disagrees with plan and audited mass")

        outcome = indexed_outcomes[event_index]
        if type(outcome.get("applied")) is not bool or type(outcome.get("recovered")) is not bool:
            raise ValueError("Gust applied/recovered flags must be Boolean")
        applied_vector = applied_vectors[event_index]
        vector_was_applied = math.sqrt(sum(component * component for component in applied_vector)) > 0.0
        if outcome["applied"] is not vector_was_applied:
            raise ValueError("Gust applied flag disagrees with submitted impulse")
        if outcome["recovered"] and not vector_was_applied:
            raise ValueError("An unapplied gust cannot provide recovery proof")
        if vector_was_applied:
            submitted_error = max(
                abs(applied - expected)
                for applied, expected in zip(applied_vector, expected_vector, strict=True)
            )
            applied_errors.append(max(
                abs(applied - expected)
                for applied, expected in zip(
                    applied_vector,
                    reported_expected_vectors[event_index],
                    strict=True,
                )
            ))
            if submitted_error > GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S:
                raise ValueError("Submitted gust impulse differs from the frozen expected impulse")
        recovered_flags.append(outcome["recovered"])

    recomputed_max_error = max(applied_errors, default=0.0)
    if not math.isclose(
        float(reported_max_error), recomputed_max_error, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("Reported gust impulse max error does not recompute exactly")
    return tuple(recovered_flags)  # type: ignore[return-value]


def _validate_memory_evidence(samples: Any, gate: Any) -> None:
    from g1_fly_control.crazyflie.memory import (
        GPU_LIMIT_MIB,
        RAM_LIMIT_PERCENT,
        assess,
    )

    if not isinstance(samples, list) or not samples or not all(
        isinstance(sample, dict) for sample in samples
    ):
        raise ValueError("Evaluation memory samples are missing or malformed")
    stages = [sample.get("stage") for sample in samples]
    required = {"environment_loaded", "graph_captured", "steady_state", "evaluation_end"}
    if not required.issubset(stages):
        raise ValueError(f"Evaluation memory stages are incomplete: {stages}")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("Evaluation memory gate did not pass")
    try:
        recomputed = assess(samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation memory samples cannot be assessed: {exc}") from exc
    if canonical_sha256(gate) != canonical_sha256(recomputed):
        raise ValueError("Evaluation memory assessment does not recompute exactly")
    limits = gate.get("limits")
    warnings = gate.get("warnings")
    if (
        gate.get("policy_version") != MEMORY_POLICY_VERSION
        or not isinstance(warnings, list)
        or any(not isinstance(warning, str) or not warning for warning in warnings)
        or gate.get("device_gpu_telemetry_complete") is not True
        or gate.get("sustained_paging_detected") is not False
        or not isinstance(limits, dict)
        or limits.get("gpu_used_mib_exclusive") != GPU_LIMIT_MIB
        or limits.get("system_ram_percent_exclusive") != RAM_LIMIT_PERCENT
        or limits.get("rss_growth_disposition") != "warning_only"
        or not isinstance(gate.get("max_device_gpu_used_mib"), (int, float))
        or isinstance(gate.get("max_device_gpu_used_mib"), bool)
        or not math.isfinite(float(gate["max_device_gpu_used_mib"]))
        or float(gate["max_device_gpu_used_mib"]) >= GPU_LIMIT_MIB
        or not isinstance(gate.get("max_system_ram_percent"), (int, float))
        or isinstance(gate.get("max_system_ram_percent"), bool)
        or not math.isfinite(float(gate["max_system_ram_percent"]))
        or float(gate["max_system_ram_percent"]) >= RAM_LIMIT_PERCENT
    ):
        raise ValueError("Evaluation memory policy/telemetry/limit contract changed")


def _scenario_child_command(
    args: argparse.Namespace,
    scenario: str,
    part_path: Path,
) -> list[str]:
    """Build one fresh-process scenario command without forwarding parent mode."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--protocol",
        str(args.protocol),
        "--scenario",
        scenario,
        "--output",
        str(part_path.resolve()),
    ]
    if args.expected_fingerprint is not None:
        command.extend(("--expected_fingerprint", str(args.expected_fingerprint)))
    if args.training_seed is not None:
        command.extend(("--training_seed", str(args.training_seed)))
    if args.policy is not None:
        command.extend(("--policy", str(args.policy)))
    if bool(getattr(args, "headless", False)):
        command.append("--headless")
    device = getattr(args, "device", None)
    if device is not None:
        command.extend(("--device", str(device)))
    return command


def _consistent_scenario_summary(
    summary: Any,
    episodes: list[dict[str, Any]],
    *,
    scenario: str,
    expected: int,
) -> bool:
    """Recompute every aggregate from episode rows before accepting a part."""

    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
        return False
    if canonical_sha256(summary["episodes"]) != canonical_sha256(episodes):
        return False
    try:
        from g1_fly_control.tasks.crazyflie.metrics import (
            EpisodeSummary,
            GustRecoveryOutcome,
            SwitchOutcome,
            summarize_episodes,
        )

        field_names = set(EpisodeSummary.__dataclass_fields__)
        records = []
        for row in episodes:
            values = {name: row[name] for name in field_names}
            values["switch_outcomes"] = tuple(
                SwitchOutcome(**item) for item in values["switch_outcomes"]
            )
            values["gust_outcomes"] = tuple(
                GustRecoveryOutcome(**item) for item in values["gust_outcomes"]
            )
            records.append(EpisodeSummary(**values))
        recomputed = summarize_episodes(records, expected_episode_count=expected)
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
        return False
    actual_aggregates = dict(summary)
    recomputed_aggregates = dict(recomputed)
    actual_aggregates.pop("episodes", None)
    recomputed_aggregates.pop("episodes", None)
    return (
        summary.get("scenario") == scenario
        and summary.get("evaluation_seed") == episodes[0].get("evaluation_seed")
        and summary.get("expected_episode_count") == expected
        and summary.get("episode_count") == expected
        and summary.get("n_episodes") == expected
        and summary.get("complete") is True
        and canonical_sha256(actual_aggregates) == canonical_sha256(recomputed_aggregates)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value, _ = _read_json_object_and_sha256(path)
    return value


def _read_json_object_and_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read scenario part artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Scenario part artifact must be a JSON object: {path}")
    return value, sha256(payload).hexdigest()


def _validate_scenario_part(
    value: dict[str, Any],
    *,
    scenario: str,
    protocol_name: str,
    protocol: dict[str, Any],
    checkpoint: Path,
    expected_fingerprint: str | None,
    expected_training_seed: int | None,
    expected_policy: str | None,
) -> None:
    """Strictly validate a single-scenario child artifact before merging."""

    plans = protocol["scenarios"][scenario]
    expected_count = len(plans)
    required_fields = set(_SHARED_PART_FIELDS) | {
        "generated_at_utc", "scenario", "episodes", "summary", "memory_samples", "memory_gate"
    }
    missing = sorted(required_fields - value.keys())
    if missing:
        raise ValueError(f"Scenario part {scenario} is missing fields: {missing}")
    if value["schema_version"] != 1 or value["status"] != "completed":
        raise ValueError(f"Scenario part {scenario} is not a completed schema-v1 artifact")
    if value["scenario"] != scenario:
        raise ValueError(f"Scenario part identity mismatch for {scenario}")
    if value["label"] != protocol["label"] or value["protocol"] != protocol_name:
        raise ValueError(f"Scenario part protocol label/name mismatch for {scenario}")
    if (
        value["evaluation_seed"] != protocol["evaluation_seed"]
        or value["evaluation_manifest_id"] != protocol["manifest_id"]
    ):
        raise ValueError(f"Scenario part evaluation manifest mismatch for {scenario}")
    if Path(value["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"Scenario part checkpoint path mismatch for {scenario}")
    if value["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError(f"Scenario part checkpoint SHA-256 mismatch for {scenario}")
    if expected_fingerprint is not None and value["fingerprint"] != expected_fingerprint:
        raise ValueError(f"Scenario part reproduction fingerprint mismatch for {scenario}")
    fingerprint = value["fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(f"Scenario part has an invalid reproduction fingerprint for {scenario}")
    if expected_training_seed is not None and value["training_seed"] != expected_training_seed:
        raise ValueError(f"Scenario part training seed mismatch for {scenario}")
    if type(value["training_seed"]) is not int:
        raise ValueError(f"Scenario part has an invalid training seed for {scenario}")
    if expected_policy is not None and value["controller"] != expected_policy:
        raise ValueError(f"Scenario part controller mismatch for {scenario}")
    if value["controller"] not in POLICIES:
        raise ValueError(f"Scenario part has an unknown controller for {scenario}")
    if not isinstance(value["fingerprint_payload"], dict) or not isinstance(
        value["controller_report"], dict
    ):
        raise ValueError(f"Scenario part lacks fingerprint/controller provenance for {scenario}")
    if not isinstance(value["fingerprint_payload"].get("resolved_config"), dict):
        raise ValueError(f"Scenario part lacks resolved fingerprint configuration for {scenario}")
    if canonical_sha256(value["fingerprint_payload"]) != fingerprint:
        raise ValueError(f"Scenario part fingerprint payload mismatch for {scenario}")
    expected_batch_size = _evaluation_batch_size(expected_count)
    if (
        value["deterministic_actions"] is not True
        or value["simulation_device"] != str(POLICY_INFERENCE_DEVICE)
        or value["simulation_device_type"] != "cuda"
        or value["policy_inference_device"] != str(POLICY_INFERENCE_DEVICE)
        or value["policy_inference_device_type"] != "cuda"
        or value["policy_inference_backend"] != POLICY_INFERENCE_BACKEND
        or value["policy_inference_precision"] != POLICY_INFERENCE_PRECISION
        or value["policy_inference_graph"]
        != _expected_policy_graph_report(expected_batch_size)
        or value["policy_bridge"] != POLICY_BRIDGE_CONTRACT
        or value["failure_denominator_rule"] != "Every planned episode remains in the denominator"
        or value["censoring_rule"]
        != "Unsuccessful episode/attempt assigned its fixed observation horizon"
        or value["integrated_error_censoring_rule"]
        != (
            "For early termination, carry the final observed goal error through the fixed episode "
            "or post-gust recovery window"
        )
    ):
        raise ValueError(f"Scenario part execution contract mismatch for {scenario}")
    if not isinstance(value["generated_at_utc"], str) or not value["generated_at_utc"]:
        raise ValueError(f"Scenario part lacks a generation timestamp for {scenario}")
    try:
        _validate_memory_evidence(value["memory_samples"], value["memory_gate"])
    except ValueError as exc:
        raise ValueError(f"Scenario part memory evidence is invalid for {scenario}: {exc}") from exc

    episodes = value["episodes"]
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise ValueError(
            f"Scenario part {scenario} has {len(episodes) if isinstance(episodes, list) else 'invalid'} "
            f"episodes; expected {expected_count}"
        )
    for expected_plan, row in zip(plans, episodes, strict=True):
        if not isinstance(row, dict):
            raise ValueError(f"Scenario part {scenario} contains a non-object episode row")
        target_success_event_count = row.get("target_success_event_count")
        maximum_target_successes = 4 if scenario == SCENARIOS[1] else 1
        if (
            row.get("scenario") != scenario
            or row.get("episode_id") != expected_plan["episode_id"]
            or row.get("evaluation_seed") != protocol["evaluation_seed"]
            or row.get("plan_sha256") != expected_plan["plan_sha256"]
            or row.get("plan") != expected_plan
            or type(target_success_event_count) is not int
            or not 0 <= target_success_event_count <= maximum_target_successes
            or type(row.get("success")) is not bool
            or row["success"] is not (
                target_success_event_count >= maximum_target_successes
            )
        ):
            raise ValueError(
                f"Scenario part {scenario} episode {expected_plan['episode_id']} plan/provenance mismatch"
            )
        if scenario == SCENARIOS[1]:
            outcomes = row.get("switch_outcomes")
            if (
                not isinstance(outcomes, list)
                or any(
                    not isinstance(outcome, dict)
                    or type(outcome.get("success")) is not bool
                    for outcome in outcomes
                )
            ):
                raise ValueError(
                    f"Scenario part {scenario} episode {expected_plan['episode_id']} "
                    "has malformed switch outcomes"
                )
            switched_successes = sum(outcome["success"] for outcome in outcomes)
            if target_success_event_count not in {
                switched_successes,
                switched_successes + 1,
            }:
                raise ValueError(
                    f"Scenario part {scenario} episode {expected_plan['episode_id']} "
                    "target-success count disagrees with switch outcomes"
                )
        elif scenario == SCENARIOS[2]:
            try:
                _validated_gust_recovery_events(row, expected_plan)
            except ValueError as exc:
                raise ValueError(
                    f"Scenario part {scenario} episode {expected_plan['episode_id']} "
                    f"has invalid gust impulse/recovery evidence: {exc}"
                ) from exc
    if not _consistent_scenario_summary(
        value["summary"], episodes, scenario=scenario, expected=expected_count
    ):
        raise ValueError(f"Scenario part summary does not recompute exactly for {scenario}")


def _merge_scenario_parts(
    part_paths: dict[str, Path],
    *,
    protocol_name: str,
    protocol: dict[str, Any],
    checkpoint: Path,
    expected_fingerprint: str | None,
    expected_training_seed: int | None,
    expected_policy: str | None,
) -> dict[str, Any]:
    """Load, strictly cross-check, and merge three isolated child artifacts."""

    parts: dict[str, dict[str, Any]] = {}
    part_sha256: dict[str, str] = {}
    reference: dict[str, Any] | None = None
    part_parents = {path.resolve().parent for path in part_paths.values()}
    if len(part_parents) != 1:
        raise ValueError("Scenario parts do not share one immutable attempt directory")
    attempt_directory = next(iter(part_parents))
    for scenario in SCENARIOS:
        part_path = part_paths[scenario]
        value, validated_sha256 = _read_json_object_and_sha256(part_path)
        _validate_scenario_part(
            value,
            scenario=scenario,
            protocol_name=protocol_name,
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=expected_fingerprint,
            expected_training_seed=expected_training_seed,
            expected_policy=expected_policy,
        )
        if reference is None:
            reference = value
        else:
            for field in _SHARED_PART_FIELDS:
                if canonical_sha256(value[field]) != canonical_sha256(reference[field]):
                    raise ValueError(
                        f"Scenario parts disagree on shared field {field!r}: {scenario}"
                    )
        parts[scenario] = value
        part_sha256[scenario] = validated_sha256
    if reference is None:
        raise ValueError("No scenario part artifacts were provided")
    if reference["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("Checkpoint changed while isolated scenario parts were running")

    base = {field: reference[field] for field in _SHARED_PART_FIELDS}
    base["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = {
        scenario: {
            "summary": parts[scenario]["summary"],
            "episodes": parts[scenario]["episodes"],
            "memory_samples": parts[scenario]["memory_samples"],
            "memory_gate": parts[scenario]["memory_gate"],
        }
        for scenario in SCENARIOS
    }
    scenario_gates = {
        scenario: parts[scenario]["memory_gate"] for scenario in SCENARIOS
    }
    policy_versions = {
        gate.get("policy_version") for gate in scenario_gates.values()
    }
    if policy_versions != {MEMORY_POLICY_VERSION}:
        raise ValueError("Scenario memory gates do not share memory policy v2")
    limits = [gate.get("limits") for gate in scenario_gates.values()]
    if not limits or any(limit != limits[0] for limit in limits[1:]):
        raise ValueError("Scenario memory gates do not share identical policy limits")
    aggregate_memory_gate = {
        "policy_version": MEMORY_POLICY_VERSION,
        "passed": all(gate["passed"] is True for gate in scenario_gates.values()),
        "failures": [
            f"{scenario}: {failure}"
            for scenario, gate in scenario_gates.items()
            for failure in gate.get("failures", [])
        ],
        "warnings": [
            f"{scenario}: {warning}"
            for scenario, gate in scenario_gates.items()
            for warning in gate.get("warnings", [])
        ],
        "device_gpu_telemetry_complete": all(
            gate["device_gpu_telemetry_complete"] is True
            for gate in scenario_gates.values()
        ),
        "monotonic_process_growth_detected": any(
            gate["monotonic_process_growth_detected"] is True
            for gate in scenario_gates.values()
        ),
        "sustained_paging_detected": any(
            gate["sustained_paging_detected"] is True
            for gate in scenario_gates.values()
        ),
        "max_system_ram_percent": max(
            gate["max_system_ram_percent"] for gate in scenario_gates.values()
        ),
        "max_device_gpu_used_mib": max(
            gate["max_device_gpu_used_mib"] for gate in scenario_gates.values()
        ),
        "max_process_rss_mib": max(
            gate["max_process_rss_mib"] for gate in scenario_gates.values()
        ),
        "max_torch_allocated_mib": max(
            gate["max_torch_allocated_mib"] for gate in scenario_gates.values()
        ),
        "max_torch_reserved_mib": max(
            gate["max_torch_reserved_mib"] for gate in scenario_gates.values()
        ),
        "limits": limits[0],
        "scenario_gates": scenario_gates,
    }
    episodes = [row for scenario in SCENARIOS for row in results[scenario]["episodes"]]
    return {
        **base,
        "scenario": "all",
        "episodes": episodes,
        "scenario_results": results,
        "total_episodes": len(episodes),
        "memory_gate": aggregate_memory_gate,
        "scenario_attempt_id": attempt_directory.name,
        "scenario_attempt_directory": str(attempt_directory),
        "scenario_part_artifacts": {
            scenario: {
                "path": str(part_paths[scenario].resolve()),
                # This digest covers the exact bytes parsed and validated
                # above, avoiding a read-then-rehash TOCTOU window.
                "sha256": part_sha256[scenario],
            }
            for scenario in SCENARIOS
        },
    }


def _run_all_scenarios(args: argparse.Namespace) -> int:
    """Evaluate each scenario in a fresh, sequential Isaac process."""

    attempt_id = _new_scenario_attempt_id()
    part_paths = _scenario_part_paths(args.output, attempt_id)
    attempt_directory = next(iter(part_paths.values())).parent
    completed_parts: list[str] = []
    active_scenario: str | None = None
    phase = "initialize_attempt"
    resolved_config: dict[str, Any] | None = None
    fingerprint: str | None = args.expected_fingerprint
    try:
        phase = "resolve_checkpoint_identity"
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        resolved_config = checkpoint_payload.get("resolved_config")
        fingerprint = checkpoint_payload.get("fingerprints", {}).get("reproduction")
        if not isinstance(resolved_config, dict) or not isinstance(fingerprint, str):
            raise ValueError("Checkpoint lacks resolved configuration or reproduction fingerprint")
        if args.expected_fingerprint and args.expected_fingerprint != fingerprint:
            raise ValueError("Checkpoint fingerprint does not match --expected_fingerprint")
        # Even the standalone §14 command (which omits the optional CLI
        # value) forwards and validates one explicit immutable identity.
        args.expected_fingerprint = fingerprint
        print(json.dumps({
            "status": "RESOLVED",
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "outputs": {
                "merged": str(args.output),
                "scenario_attempt_directory": str(attempt_directory.resolve()),
            },
        }, sort_keys=True), flush=True)
        # Attempts are append-only evidence. A collision fails rather than
        # deleting or replacing any prior child artifact.
        attempt_directory.mkdir(parents=True, exist_ok=False)
        protocol = load_protocol(args.protocol)
        for active_scenario in SCENARIOS:
            phase = "run_scenario_child"
            part_path = part_paths[active_scenario]
            if part_path.exists():
                raise RuntimeError(f"Refusing to overwrite scenario evidence: {part_path}")
            command = _scenario_child_command(args, active_scenario, part_path)
            print(json.dumps({
                "status": "STARTING_SCENARIO_CHILD",
                "scenario": active_scenario,
                "output": str(part_path.resolve()),
                "command": command,
            }, sort_keys=True), flush=True)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Scenario child {active_scenario} exited with code {completed.returncode}"
                )
            value = _read_json_object(part_path)
            _validate_scenario_part(
                value,
                scenario=active_scenario,
                protocol_name=args.protocol,
                protocol=protocol,
                checkpoint=args.checkpoint,
                expected_fingerprint=args.expected_fingerprint,
                expected_training_seed=args.training_seed,
                expected_policy=args.policy,
            )
            fingerprint = value["fingerprint"]
            part_resolved_config = value["fingerprint_payload"].get("resolved_config")
            if part_resolved_config != resolved_config:
                raise ValueError(
                    f"Scenario child {active_scenario} resolved configuration differs from checkpoint"
                )
            part_path.chmod(0o444)
            completed_parts.append(active_scenario)

        active_scenario = None
        phase = "merge_scenario_parts"
        result = _merge_scenario_parts(
            part_paths,
            protocol_name=args.protocol,
            protocol=protocol,
            checkpoint=args.checkpoint,
            expected_fingerprint=args.expected_fingerprint,
            expected_training_seed=args.training_seed,
            expected_policy=args.policy,
        )
        phase = "publish_merged_artifact"
        _atomic_json(args.output, result)
        print(json.dumps({
            "status": "PASS",
            "output": str(args.output),
            "checkpoint": str(args.checkpoint),
            "controller": result["controller"],
            "training_seed": result["training_seed"],
            "scenarios": list(SCENARIOS),
            "episodes": len(result["episodes"]),
            "fingerprint": result["fingerprint"],
            "resolved_config": result["fingerprint_payload"]["resolved_config"],
            "scenario_attempt_id": attempt_id,
            "scenario_part_artifacts": result["scenario_part_artifacts"],
        }, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        failure = {
            "schema_version": 1,
            "status": "failed",
            "checkpoint": str(args.checkpoint),
            "protocol": args.protocol,
            "scenario": "all",
            "failed_scenario": active_scenario,
            "failed_phase": phase,
            "completed_scenarios": completed_parts,
            "scenario_attempt_id": attempt_id,
            "scenario_attempt_directory": str(attempt_directory.resolve()),
            "scenario_part_artifacts": {
                scenario: str(path.resolve()) for scenario, path in part_paths.items()
            },
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "error": traceback.format_exc(),
        }
        _atomic_json(args.output, failure)
        print(json.dumps({
            "status": "FAIL",
            "output": str(args.output),
            "checkpoint": str(args.checkpoint),
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "failure_reason": failure["error"],
        }, indent=2, sort_keys=True))
        return 1


def _condition_from_payload(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    configured = metadata.get("controller")
    if configured in POLICIES:
        return configured
    resolved = payload.get("resolved_config", {})
    configured = resolved.get("controller")
    if configured in POLICIES:
        return configured
    canonical = metadata.get("controller_report", {}).get("controller_kind")
    reverse = {
        "frozen_lif": "frozen_lif_original",
        "frozen_lif_rewired": "frozen_lif_degree_rewired",
        "gru": "gru_matched",
        "mlp": "mlp_normal",
        "wing_lif": "wing_lif",
        "leg_wing_lif": "leg_wing_lif",
    }
    if canonical in reverse:
        return reverse[canonical]
    raise ValueError("Checkpoint does not identify one of the four declared controllers")


def _root_state_and_plans(env: Any, plans: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(env.device)
    origins = env._terrain.env_origins[: len(plans)]
    roots = torch.zeros((len(plans), 13), dtype=torch.float32, device=device)
    roots[:, 3] = 1.0
    target_count = 4 if env.scenario == "waypoint_switch" else 1
    targets = torch.zeros((len(plans), target_count, 3), dtype=torch.float32, device=device)
    gusts = torch.zeros((len(plans), 3, 2), dtype=torch.float32, device=device)
    for index, plan in enumerate(plans):
        initial = plan["initial_state"]
        position = torch.tensor(initial["position_relative_to_env_origin_m"], device=device)
        roots[index, :3] = position + origins[index]
        roots[index, 2] = position[2]
        yaw = float(initial["yaw_rad"])
        roots[index, 3] = math.cos(yaw / 2.0)
        roots[index, 6] = math.sin(yaw / 2.0)
        roots[index, 7:10] = torch.tensor(initial["linear_velocity_world_m_s"], device=device)
        angular_b = torch.tensor(initial["angular_velocity_body_rad_s"], device=device)
        # Root velocity uses world coordinates; with yaw-only attitude this is
        # the exact body-to-world rotation for angular velocity.
        roots[index, 10] = math.cos(yaw) * angular_b[0] - math.sin(yaw) * angular_b[1]
        roots[index, 11] = math.sin(yaw) * angular_b[0] + math.cos(yaw) * angular_b[1]
        roots[index, 12] = angular_b[2]
        for target_index, target in enumerate(plan["targets_relative_to_env_origin_m"][:target_count]):
            value = torch.tensor(target, device=device)
            value[:2] += origins[index, :2]
            targets[index, target_index] = value
        for event_index, gust in enumerate(plan["gusts"]):
            gusts[index, event_index] = torch.tensor(gust["direction_world_xy"], device=device)
    return roots, targets, gusts


def _failure_name(code: int) -> str | None:
    from g1_fly_control.tasks.crazyflie.env import FAILURE_CAUSE_NAMES
    value = FAILURE_CAUSE_NAMES.get(code, f"unknown_failure_{code}")
    return None if value == "none" else value


def _update_success_timing(
    *,
    current_success_count: torch.Tensor,
    current_switch_count: torch.Tensor,
    previous_success_count: torch.Tensor,
    active: torch.Tensor,
    decision: int,
    first_success_step: torch.Tensor,
    switch_success_step: torch.Tensor,
    switch_event_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update batched success timing without reading CUDA scalars on the host.

    A target success is an increment of ``success_count`` while the planned
    episode is still active.  This mirrors the former scalar loop exactly:
    only the current switch index is recorded when an increment is observed,
    and the first recorded time for each event is immutable.
    """

    if switch_event_ids.shape != (switch_success_step.shape[1],):
        raise ValueError("switch_event_ids must contain one entry per tracked switch")
    incremented = active & (current_success_count > previous_success_count)
    decision_steps = torch.full_like(first_success_step, decision)
    first_success_step = torch.where(
        incremented & (first_success_step < 0), decision_steps, first_success_step
    )
    event_hit = incremented[:, None] & (
        current_switch_count[:, None] - 1 == switch_event_ids[None, :]
    )
    switch_success_step = torch.where(
        event_hit & (switch_success_step < 0),
        decision_steps[:, None].expand_as(switch_success_step),
        switch_success_step,
    )
    previous_success_count = torch.where(
        active, current_success_count, previous_success_count
    )
    return first_success_step, switch_success_step, previous_success_count


def _validated_policy_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid evaluation device {value!r}") from exc
    if device != POLICY_INFERENCE_DEVICE:
        raise ValueError(
            f"Primary evaluation requires exactly {POLICY_INFERENCE_DEVICE}, got {device}"
        )
    return device


def _initial_policy_state(
    policy: torch.nn.Module, batch_size: int, device: torch.device
) -> Any:
    if hasattr(policy, "initial_state"):
        return policy.initial_state(batch_size, device=device)
    return None


def _deterministic_actor_step(
    policy: torch.nn.Module,
    observation: torch.Tensor,
    state: Any,
    reset_before_policy: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """Run only the actor math used by deterministic ``policy.act``.

    Evaluation does not consume the stochastic log probability or critic.
    Omitting those unused branches avoids the distribution constructor's CUDA
    validation synchronization while leaving actor action/state math exactly
    unchanged for all four declared controllers.
    """

    from g1_fly_control.crazyflie.controllers import reset_controller_state
    from g1_fly_control.policies.actor_critic import FrozenLIFActorCritic, MLPActorCritic
    from g1_fly_control.policies.gru import GRUActorCritic

    if observation.ndim != 2 or observation.shape[1] != 12:
        raise ValueError("Evaluation observation must have shape [batch, 12]")
    if reset_before_policy.shape != (observation.shape[0],):
        raise ValueError("Evaluation reset mask must contain one row per environment")
    state = reset_controller_state(policy, state, reset_before_policy)
    if isinstance(policy, FrozenLIFActorCritic):
        mean, next_state = policy._mean_and_state(observation, state)
    elif isinstance(policy, GRUActorCritic):
        mean, next_state = policy._mean_and_state(observation, state)
    elif isinstance(policy, MLPActorCritic):
        if state is not None:
            raise TypeError("Feed-forward evaluation state must be None")
        next_state = None
        mean = policy._mean(observation)
    else:
        raise TypeError(f"Unsupported evaluation policy class: {type(policy).__name__}")
    if mean.shape != (observation.shape[0], 4):
        raise FloatingPointError("Policy produced an incorrectly shaped evaluation mean")
    return torch.tanh(mean), next_state


def _state_tensors(state: Any) -> tuple[torch.Tensor, ...]:
    from g1_fly_control.policies.lif_core import LIFState

    if isinstance(state, LIFState):
        return state.membrane, state.spikes, state.synapse, state.refractory
    if isinstance(state, torch.Tensor):
        return (state,)
    if state is None:
        return ()
    raise TypeError(f"Unsupported recurrent state class: {type(state).__name__}")


def _state_from_tensors(template: Any, tensors: tuple[torch.Tensor, ...]) -> Any:
    from g1_fly_control.policies.lif_core import LIFState

    if isinstance(template, LIFState):
        if len(tensors) != 4:
            raise RuntimeError("Frozen-LIF CUDA graph requires four state tensors")
        return LIFState(*tensors)
    if isinstance(template, torch.Tensor):
        if len(tensors) != 1:
            raise RuntimeError("GRU CUDA graph requires one state tensor")
        return tensors[0]
    if template is None and not tensors:
        return None
    raise TypeError("Captured policy state does not match its template")


class _CudaGraphPolicyRunner:
    """Fixed-batch, fail-closed CUDA-Graph deterministic actor runner."""

    def __init__(
        self,
        policy: torch.nn.Module,
        batch_size: int,
        *,
        device: str | torch.device = POLICY_INFERENCE_DEVICE,
    ) -> None:
        self.device = _validated_policy_device(device)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("CUDA-Graph policy batch size must be a positive integer")
        if policy.training:
            raise RuntimeError("CUDA-Graph policy capture requires policy.eval()")
        model_devices = {
            tensor.device
            for tensor in (*tuple(policy.parameters()), *tuple(policy.buffers()))
        }
        if model_devices != {self.device}:
            raise RuntimeError(
                f"Every policy parameter/buffer must reside on {self.device}; got {model_devices}"
            )
        if not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"):
            raise RuntimeError("Primary policy inference requires torch.cuda.CUDAGraph")
        self.policy = policy
        self.batch_size = batch_size
        self._static_observation = torch.zeros(
            batch_size, 12, dtype=torch.float32, device=self.device
        )
        self._static_reset = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self._state_template = _initial_policy_state(policy, batch_size, self.device)
        self._state_buffers = tuple(
            tensor.clone() for tensor in _state_tensors(self._state_template)
        )
        self._action = torch.empty(batch_size, 4, dtype=torch.float32, device=self.device)
        self._finite = torch.empty(batch_size, dtype=torch.bool, device=self.device)
        self._graph: Any = None
        self._captured = False
        self._parity_verified = False
        try:
            self._capture()
            self._verify_bitwise_parity()
        except Exception as exc:
            raise RuntimeError(
                "Fail-closed CUDA-Graph policy capture/parity gate failed"
            ) from exc

    def _current_state(self) -> Any:
        return _state_from_tensors(self._state_template, self._state_buffers)

    def reset(self) -> None:
        self._static_observation.zero_()
        self._static_reset.zero_()
        for tensor in self._state_buffers:
            tensor.zero_()

    def _capture(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            warm_state = _initial_policy_state(self.policy, self.batch_size, self.device)
            for _ in range(3):
                _, warm_state = _deterministic_actor_step(
                    self.policy,
                    self._static_observation,
                    warm_state,
                    self._static_reset,
                )
        current_stream.wait_stream(warmup_stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            action, next_state = _deterministic_actor_step(
                self.policy,
                self._static_observation,
                self._current_state(),
                self._static_reset,
            )
            finite = torch.isfinite(action).all(dim=-1)
            next_tensors = _state_tensors(next_state)
            if len(next_tensors) != len(self._state_buffers):
                raise RuntimeError("Captured recurrent state structure changed")
            for state_tensor in next_tensors:
                if state_tensor.shape[0] != self.batch_size:
                    raise RuntimeError("Captured recurrent state lost its batch dimension")
                finite = finite & torch.isfinite(state_tensor).reshape(
                    self.batch_size, -1
                ).all(dim=-1)
            self._action.copy_(
                torch.where(finite[:, None], action, torch.zeros_like(action))
            )
            self._finite.copy_(finite)
            for destination, source in zip(
                self._state_buffers, next_tensors, strict=True
            ):
                destination.copy_(source)
        torch.cuda.synchronize(self.device)
        self._captured = True
        self.reset()

    def step(
        self, observation: torch.Tensor, reset_before_policy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._captured:
            raise RuntimeError("CUDA policy graph was not captured")
        if (
            observation.shape != (self.batch_size, 12)
            or observation.dtype != torch.float32
            or observation.device != self.device
        ):
            raise ValueError(
                f"Policy observation must be float32 [{self.batch_size}, 12] on {self.device}"
            )
        if (
            reset_before_policy.shape != (self.batch_size,)
            or reset_before_policy.dtype != torch.bool
            or reset_before_policy.device != self.device
        ):
            raise ValueError(
                f"Policy reset mask must be bool [{self.batch_size}] on {self.device}"
            )
        self._static_observation.copy_(observation)
        self._static_reset.copy_(reset_before_policy)
        self._graph.replay()
        return self._action, self._finite

    def _verify_bitwise_parity(self) -> None:
        reference_state = _initial_policy_state(
            self.policy, self.batch_size, self.device
        )
        base = torch.linspace(
            -0.75,
            0.75,
            steps=self.batch_size * 12,
            dtype=torch.float32,
            device=self.device,
        ).reshape(self.batch_size, 12)
        alternating = torch.arange(self.batch_size, device=self.device).remainder(2).eq(0)
        probes = (
            (base, torch.zeros_like(alternating)),
            (-base, alternating),
            (base * 0.5, torch.zeros_like(alternating)),
        )
        self.reset()
        with torch.no_grad():
            for observation, reset_mask in probes:
                reference_action, reference_state = _deterministic_actor_step(
                    self.policy, observation, reference_state, reset_mask
                )
                captured_action, captured_finite = self.step(observation, reset_mask)
                torch.cuda.synchronize(self.device)
                if not bool(captured_finite.all().item()):
                    raise FloatingPointError("CUDA-Graph parity probe produced a nonfinite action")
                if not torch.equal(captured_action, reference_action):
                    raise RuntimeError("CUDA-Graph action is not bit-for-bit equal to eager CUDA")
                captured_state = self._current_state()
                reference_tensors = _state_tensors(reference_state)
                captured_tensors = _state_tensors(captured_state)
                if len(reference_tensors) != len(captured_tensors) or any(
                    not torch.equal(reference, captured)
                    for reference, captured in zip(
                        reference_tensors, captured_tensors, strict=True
                    )
                ):
                    raise RuntimeError(
                        "CUDA-Graph recurrent state is not bit-for-bit equal to eager CUDA"
                    )
        self.reset()
        torch.cuda.synchronize(self.device)
        self._parity_verified = True

    @property
    def report(self) -> dict[str, Any]:
        if not self._captured or not self._parity_verified:
            raise RuntimeError("CUDA-Graph policy runner has not passed its parity gate")
        return _expected_policy_graph_report(self.batch_size)


def _evaluate_batch(
    env: Any,
    normalized_env: Any,
    policy_runner: _CudaGraphPolicyRunner,
    plans: list[dict[str, Any]],
    episode_offset: int,
    scenario: str,
    evaluation_seed: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    from g1_fly_control.tasks.crazyflie.logic import CONTROL_DT_S, DEFAULT_GUST_STEPS, DEFAULT_SWITCH_STEPS
    from g1_fly_control.tasks.crazyflie.metrics import (
        EpisodeSummary,
        GustRecoveryOutcome,
        SwitchOutcome,
        command_effort_integral,
        command_smoothness_mean_delta,
        goal_error_integral_m_s,
        mean_speed_inside_target_region_m_s,
    )

    roots, targets, gusts = _root_state_and_plans(env, plans)
    env.set_episode_plan(
        initial_root_state_w=roots,
        targets_w=targets,
        gust_directions_w=gusts if scenario == "FlyCrazyflie-GustRecovery-v0" else None,
    )
    observation, _ = normalized_env.reset(seed=evaluation_seed + episode_offset)
    observation = observation["policy"]
    if int(env.episode_length_buf.abs().max()) != 0:
        raise RuntimeError("Deterministic evaluation did not start every episode at step zero")
    batch = len(plans)
    environment_device = torch.device(env.device)
    if batch != policy_runner.batch_size or environment_device != policy_runner.device:
        raise RuntimeError("Evaluation batch/environment no longer matches the captured policy graph")
    policy_runner.reset()
    reset_before_policy = torch.zeros(batch, dtype=torch.bool, device=environment_device)
    done = torch.zeros(batch, dtype=torch.bool, device=env.device)
    terminated_final = torch.zeros_like(done)
    truncated_final = torch.zeros_like(done)
    failure_final = torch.zeros(batch, dtype=torch.long, device=env.device)
    success_count_final = torch.zeros(batch, dtype=torch.long, device=env.device)
    work_final = torch.zeros(batch, device=env.device)
    previous_success_count = torch.zeros(batch, dtype=torch.long, device=env.device)
    first_success_step = torch.full((batch,), -1, dtype=torch.long, device=env.device)
    switch_success_step = torch.full((batch, 3), -1, dtype=torch.long, device=env.device)
    switch_event_ids = torch.arange(3, dtype=torch.long, device=env.device)
    completed_steps_final = torch.zeros(batch, dtype=torch.long, device=env.device)
    # [distance, speed, action x4, world position x3].  At the validated four
    # evaluation environments this fixed buffer is only 86,400 bytes.
    trace_buffer = torch.zeros((600, batch, 9), dtype=torch.float32, device=env.device)
    actions_finite = torch.ones((), dtype=torch.bool, device=env.device)
    gust_applied_final = torch.zeros(batch, 3, 3, device=env.device)
    gust_expected_final = torch.zeros(batch, 3, 3, device=env.device)
    gust_stable_final = torch.zeros(batch, 3, dtype=torch.bool, device=env.device)
    gust_recovered_final = torch.zeros(batch, 3, dtype=torch.bool, device=env.device)
    gust_latency_final = torch.full((batch, 3), float("nan"), device=env.device)

    # DirectRLEnv automatically resets terminated environments on subsequent
    # steps.  Once every fixed episode in this batch has terminated, continuing
    # to decision 600 therefore simulates unrelated reset episodes and can be
    # dramatically slower than the planned evaluation itself.  Poll one scalar
    # only at the validated 25-decision PPO horizon cadence: this keeps
    # synchronization bounded (at most 24 polls for a full-horizon batch) while
    # limiting wasted post-terminal simulation to 24 decisions.  Episode
    # metrics retain their exact terminal step; fixed-horizon censoring is
    # applied during CPU post-process.
    completion_poll_interval = 25

    with torch.no_grad():
        for decision in range(1, 601):
            step_action, finite_now = policy_runner.step(observation, reset_before_policy)
            active = ~done
            # The scalar check covers the full policy batch whenever at least
            # one planned episode is active, matching the original strict
            # check.  Extra fixed-horizon steps after all plans finish do not
            # introduce a new failure mode.
            actions_finite &= ~active.any() | finite_now.all()
            # Invalid active actions still fail the evaluation after the one
            # final host transfer. Replacing only nonfinite values keeps
            # them out of the simulator; finite actions remain unchanged.
            next_observation, _, terminated, truncated, _ = normalized_env.step(step_action)
            observation = next_observation["policy"]
            reset_before_policy = terminated | truncated
            newly_done = active & (terminated | truncated)
            current_distance = torch.where(newly_done, env.terminal_distance_m, env._distance)
            current_speed = torch.where(newly_done, env.terminal_speed_mps, env._speed)
            current_position = torch.where(
                newly_done[:, None], env.terminal_position_w, env._robot.data.root_pos_w
            )
            current_success_count = torch.where(
                newly_done, env.terminal_success_count, env.success_count
            )
            current_switch_count = torch.where(
                newly_done, env.terminal_switch_count, env.switch_count
            )
            trace_buffer[decision - 1, :, 0] = current_distance
            trace_buffer[decision - 1, :, 1] = current_speed
            trace_buffer[decision - 1, :, 2:6] = step_action
            trace_buffer[decision - 1, :, 6:9] = current_position
            completed_steps_final += active.long()
            first_success_step, switch_success_step, previous_success_count = _update_success_timing(
                current_success_count=current_success_count,
                current_switch_count=current_switch_count,
                previous_success_count=previous_success_count,
                active=active,
                decision=decision,
                first_success_step=first_success_step,
                switch_success_step=switch_success_step,
                switch_event_ids=switch_event_ids,
            )
            terminated_final = torch.where(newly_done, terminated, terminated_final)
            truncated_final = torch.where(newly_done, truncated, truncated_final)
            failure_final = torch.where(
                newly_done, env.terminal_failure_cause, failure_final
            )
            success_count_final = torch.where(
                newly_done, env.terminal_success_count, success_count_final
            )
            work_final = torch.where(
                newly_done, env.terminal_mechanical_work_proxy, work_final
            )
            terminal_event_mask = newly_done[:, None, None]
            gust_applied_final = torch.where(
                terminal_event_mask,
                env.terminal_gust_event_applied_impulse_w,
                gust_applied_final,
            )
            gust_expected_final = torch.where(
                terminal_event_mask,
                env.terminal_gust_event_expected_impulse_w,
                gust_expected_final,
            )
            terminal_metric_mask = newly_done[:, None]
            gust_stable_final = torch.where(
                terminal_metric_mask,
                env.terminal_gust_event_stable_before,
                gust_stable_final,
            )
            gust_recovered_final = torch.where(
                terminal_metric_mask,
                env.terminal_gust_event_recovered,
                gust_recovered_final,
            )
            gust_latency_final = torch.where(
                terminal_metric_mask,
                env.terminal_gust_event_recovery_latency_s,
                gust_latency_final,
            )
            done |= newly_done
            if decision % completion_poll_interval == 0 and bool(done.all().item()):
                break

    # A single post-loop host transfer contains every trace and terminal
    # value.  Counts and steps are bounded by 600 and exactly representable as
    # float32; they are converted back to int/bool during CPU post-processing.
    final_snapshot = torch.cat((
        done[:, None].float(),                         # 0
        terminated_final[:, None].float(),             # 1
        truncated_final[:, None].float(),              # 2
        failure_final[:, None].float(),                # 3
        success_count_final[:, None].float(),          # 4
        work_final[:, None].float(),                   # 5
        completed_steps_final[:, None].float(),        # 6
        first_success_step[:, None].float(),           # 7
        switch_success_step.float(),                   # 8:11
        gust_applied_final.reshape(batch, 9).float(),  # 11:20
        gust_expected_final.reshape(batch, 9).float(), # 20:29
        gust_stable_final.float(),                     # 29:32
        gust_recovered_final.float(),                  # 32:35
        gust_latency_final.float(),                    # 35:38
        actions_finite.expand(batch, 1).float(),       # 38
    ), dim=1)
    trace_element_count = 600 * 9
    host_payload = torch.cat((
        trace_buffer.permute(1, 0, 2).reshape(batch, trace_element_count),
        final_snapshot,
    ), dim=1).detach().cpu()
    traces = host_payload[:, :trace_element_count].reshape(batch, 600, 9)
    final = host_payload[:, trace_element_count:]
    if not bool(final[:, 38].bool().all()):
        raise FloatingPointError("Policy produced a nonfinite evaluation action")
    if not bool(final[:, 0].bool().all()):
        raise RuntimeError("Evaluation exceeded the frozen 600-decision episode horizon")

    records = []
    details: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        completed_steps = int(final[index, 6].item())
        episode_trace = traces[index, :completed_steps]
        distances = episode_trace[:, 0]
        speeds = episode_trace[:, 1]
        actions = episode_trace[:, 2:6]
        positions = episode_trace[:, 6:9]
        failure_code = int(final[index, 3].item())
        failure_reason = _failure_name(failure_code)
        first_step = int(final[index, 7].item())
        first_success_time = first_step * CONTROL_DT_S if first_step >= 0 else None
        switch_times = []
        for event_index, start in enumerate(DEFAULT_SWITCH_STEPS):
            event_step = int(final[index, 8 + event_index].item())
            switch_times.append(
                (event_step - start) * CONTROL_DT_S if event_step >= 0 else None
            )
        switch_outcomes = ()
        if scenario == "FlyCrazyflie-WaypointSwitch-v0":
            switch_outcomes = tuple(
                SwitchOutcome(
                    switch_index=event_index,
                    switch_step=start,
                    success=switch_times[event_index] is not None,
                    latency_s=switch_times[event_index],
                    censor_time_s=3.0,
                    failure_reason=(
                        None if switch_times[event_index] is not None
                        else failure_reason or "target_segment_ended_without_success"
                    ),
                )
                for event_index, start in enumerate(DEFAULT_SWITCH_STEPS)
            )
        gust_outcomes = ()
        if scenario == "FlyCrazyflie-GustRecovery-v0":
            outcomes = []
            gust_applied = final[index, 11:20].reshape(3, 3)
            gust_stable = final[index, 29:32]
            gust_recovered = final[index, 32:35]
            gust_latency = final[index, 35:38]
            for event_index, start in enumerate(DEFAULT_GUST_STEPS):
                applied_vector = gust_applied[event_index]
                applied = bool(torch.linalg.vector_norm(applied_vector) > 0)
                recovered = bool(gust_recovered[event_index]) if applied else False
                latency_value = float(gust_latency[event_index]) if recovered else None
                if applied:
                    # Traces are zero-indexed while schedule steps are decision
                    # counts.  Keep an applied event measurable even if the
                    # vehicle terminates during the gust itself.
                    reference_index = min(max(start - 1, 0), positions.shape[0] - 1)
                    reference = positions[reference_index]
                    stop = min(start + 5 + 100, positions.shape[0])
                    stop = max(stop, reference_index + 1)
                    max_displacement = float(
                        torch.linalg.vector_norm(
                            positions[reference_index:stop] - reference, dim=-1
                        ).max()
                    )
                    error_start = min(start + 5, distances.shape[0])
                    error_stop = min(start + 5 + 100, distances.shape[0])
                    observed_error = distances[error_start:error_stop]
                    # Censor an early failure at the last observed goal error
                    # through the fixed 2 s recovery window.  An empty trace
                    # therefore cannot look like zero tracking error.
                    fallback_error = distances[-1]
                    observed_sum = observed_error.sum() if observed_error.numel() else fallback_error * 0
                    missing_steps = 100 - observed_error.numel()
                    post_error = float(
                        (observed_sum + fallback_error * missing_steps) * CONTROL_DT_S
                    )
                else:
                    max_displacement = None
                    post_error = None
                termination_step = completed_steps if bool(final[index, 1]) else None
                recovery_deadline_step = start + 5 + 100
                outcomes.append(GustRecoveryOutcome(
                    gust_index=event_index,
                    gust_start_step=start,
                    applied=applied,
                    stable_before_gust=bool(gust_stable[event_index]) if applied else None,
                    recovered=recovered,
                    recovery_latency_s=latency_value,
                    max_displacement_m=max_displacement,
                    post_gust_error_integral_m_s=post_error,
                    # A future scheduled gust that was never reached remains
                    # in the unconditional denominator.  For an applied gust,
                    # only a termination inside its own recovery window gets
                    # this event-specific flag.
                    terminated_before_recovery=bool(
                        termination_step is not None
                        and not recovered
                        and (not applied or termination_step <= recovery_deadline_step)
                    ),
                    failure_reason=(None if recovered else failure_reason or "recovery_window_expired_or_not_reached"),
                ))
            gust_outcomes = tuple(outcomes)
        required_successes = 4 if scenario == "FlyCrazyflie-WaypointSwitch-v0" else 1
        success = int(final[index, 4].item()) >= required_successes
        crash = failure_code == 1
        out_of_bounds = failure_code in {2, 3}
        invalid = failure_code == 4
        # Keep early failures comparable on time-integrated error by carrying
        # the final observed error through the fixed 600-step horizon.
        if completed_steps < 600:
            distance_for_integral = torch.cat(
                (distances, distances[-1].repeat(600 - completed_steps))
            )
        else:
            distance_for_integral = distances
        record = EpisodeSummary(
            scenario=scenario,
            episode_id=episode_offset + index,
            success=success,
            terminated=bool(final[index, 1]),
            truncated=bool(final[index, 2]),
            failure_reason=failure_reason if not success else None,
            time_to_first_success_s=first_success_time if success else None,
            final_goal_error_m=float(distances[-1]),
            integrated_goal_error_m_s=goal_error_integral_m_s(distance_for_integral),
            mean_speed_inside_target_region_m_s=mean_speed_inside_target_region_m_s(distances, speeds),
            crash=crash,
            out_of_bounds=out_of_bounds,
            invalid_state=invalid,
            command_effort=command_effort_integral(actions),
            command_smoothness=command_smoothness_mean_delta(actions),
            aggregate_wrench_mechanical_work_proxy_j=float(final[index, 5]),
            completed_steps=completed_steps,
            evaluation_seed=evaluation_seed,
            switch_outcomes=switch_outcomes,
            gust_outcomes=gust_outcomes,
        )
        records.append(record)
        gust_applied = final[index, 11:20].reshape(3, 3)
        gust_expected = final[index, 20:29].reshape(3, 3)
        details.append({
            "plan": plan,
            "plan_sha256": plan["plan_sha256"],
            # Preserve the simulator's target-latched event count separately
            # from strict full-episode success.  The proof verifier derives
            # its event score from this authenticated count (or, for Gust,
            # from the retained per-gust recovery outcomes).
            "target_success_event_count": int(final[index, 4].item()),
            "robot_mass_kg": (
                float(env.robot_mass_kg)
                if scenario == "FlyCrazyflie-GustRecovery-v0" else None
            ),
            "gust_applied_impulse_w_n_s": gust_applied.tolist(),
            "gust_expected_impulse_w_n_s": gust_expected.tolist(),
            "gust_impulse_max_abs_error_n_s": (
                max(
                    (
                        float(
                            (
                                gust_applied[event_index]
                                - gust_expected[event_index]
                            ).abs().max()
                        )
                        for event_index in range(3)
                        if bool(torch.linalg.vector_norm(gust_applied[event_index]) > 0)
                    ),
                    default=0.0,
                )
                if scenario == "FlyCrazyflie-GustRecovery-v0" else None
            ),
        })
    return records, details


def _evaluate_scenario(
    scenario: str,
    plans: list[dict[str, Any]],
    policy: torch.nn.Module,
    normalizer: Any,
    evaluation_seed: int,
    policy_device: torch.device,
    *,
    contract_profile: str,
) -> dict[str, Any]:
    global _LAST_MEMORY_GATE, _LAST_MEMORY_SAMPLES

    from g1_fly_control.crazyflie.memory import assess, reset_cuda_peak, snapshot
    from g1_fly_control.crazyflie.normalization import NormalizedEnv
    from g1_fly_control.tasks.crazyflie.metrics import summarize_episodes

    all_records = []
    all_details = []
    memory_samples: list[dict[str, Any]] = []
    memory_gate: dict[str, Any] | None = None

    def record_memory(stage: str, *, step: int | None = None) -> None:
        nonlocal memory_gate
        global _LAST_MEMORY_GATE, _LAST_MEMORY_SAMPLES

        memory_samples.append(snapshot(stage, policy_device, step=step))
        memory_gate = assess(memory_samples)
        _LAST_MEMORY_SAMPLES = list(memory_samples)
        _LAST_MEMORY_GATE = dict(memory_gate)
        if not memory_gate["passed"]:
            raise RuntimeError(
                "Evaluation memory gate failed: " + "; ".join(memory_gate["failures"])
            )

    batch_size = _evaluation_batch_size(len(plans))
    reset_cuda_peak(policy_device)
    env = launch_environment(
        scenario,
        batch_size,
        deterministic_evaluation=True,
        device=policy_device,
        contract_profile=contract_profile,
    )
    environment_device = torch.device(env.device)
    if environment_device != POLICY_INFERENCE_DEVICE:
        env.close()
        raise RuntimeError(
            f"Crazyflie evaluation environment must reside on {POLICY_INFERENCE_DEVICE}"
        )
    if policy_device != POLICY_INFERENCE_DEVICE or next(policy.parameters()).device != policy_device:
        env.close()
        raise RuntimeError("Policy and Isaac environment are not co-located on cuda:0")
    wrapped: Any | None = None
    try:
        record_memory("environment_loaded")
        wrapped = NormalizedEnv(env, normalizer, training=False)
        policy_runner = _CudaGraphPolicyRunner(
            policy, batch_size, device=policy_device
        )
        record_memory("graph_captured")
        for offset in range(0, len(plans), batch_size):
            selected = plans[offset : offset + batch_size]
            records, details = _evaluate_batch(
                env,
                wrapped,
                policy_runner,
                selected,
                offset,
                scenario,
                evaluation_seed,
            )
            all_records.extend(records)
            all_details.extend(details)
            record_memory("steady_state", step=offset + batch_size)
        record_memory("evaluation_end", step=len(plans))
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()
    if memory_gate is None:
        raise RuntimeError("Evaluation did not produce a memory assessment")
    _validate_memory_evidence(memory_samples, memory_gate)
    summary = summarize_episodes(all_records, expected_episode_count=len(plans))
    for row, detail in zip(summary["episodes"], all_details, strict=True):
        row.update(detail)
    return {
        "summary": summary,
        "episodes": summary["episodes"],
        "policy_inference_graph": policy_runner.report,
        "memory_samples": memory_samples,
        "memory_gate": memory_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protocol", default="integration")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--scenario", choices=SCENARIOS)
    scenario.add_argument("--all_scenarios", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--training_seed", type=int)
    parser.add_argument("--policy", choices=POLICIES)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.output is None or (args.scenario is None and not args.all_scenarios):
        parser.error("--checkpoint, --output, and exactly one of --scenario/--all_scenarios are required")
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    # AppLauncher/Kit cannot reliably tear down one task and construct another
    # inside a single Python process.  The all-scenarios parent therefore does
    # no Isaac startup at all: it blocks on three fresh single-scenario
    # children and strictly merges their immutable JSON artifacts.
    if args.all_scenarios:
        return _run_all_scenarios(args)

    try:
        requested_device = _validated_policy_device(args.device or "cuda:0")
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("Crazyflie evaluation requires CUDA for the Isaac environment")
    try:
        protocol = load_protocol(args.protocol)
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        controller = _condition_from_payload(raw)
        if args.policy and args.policy != controller:
            raise ValueError(f"--policy {args.policy} does not match checkpoint controller {controller}")
        training_seed = int(raw.get("metadata", {}).get("seed", raw.get("resolved_config", {}).get("seed", -1)))
        if args.training_seed is not None and args.training_seed != training_seed:
            raise ValueError("--training_seed does not match the checkpoint")
        saved_fingerprint = raw.get("fingerprints", {}).get("reproduction")
        requested_fingerprint = args.expected_fingerprint or saved_fingerprint
        if not requested_fingerprint or saved_fingerprint != requested_fingerprint:
            raise ValueError("Checkpoint reproduction fingerprint does not match the requested evaluation")
        report = raw.get("metadata", {}).get("controller_report", {})
        connectome = report.get("connectome_manifest")
        rewire_seed = report.get("rewire_seed") or 20260916
        resolved_config = raw.get("resolved_config")
        if not isinstance(resolved_config, dict):
            raise ValueError("Checkpoint lacks a resolved configuration")
        contract_profile = _contract_profile_from_resolved_config(resolved_config)
        wing_extension = resolved_config.get("wing_extension")
        wing_manifest_path = None
        if controller in {"wing_lif", "leg_wing_lif"}:
            if not isinstance(wing_extension, dict):
                raise ValueError("Wing checkpoint lacks its wing-extension provenance")
            wing_identity = wing_extension.get("wing_connectome")
            if not isinstance(wing_identity, dict) or not isinstance(wing_identity.get("manifest"), str):
                raise ValueError("Wing checkpoint lacks its wing connectome manifest path")
            wing_manifest_path = Path(wing_identity["manifest"]).resolve()
        matrix_config = resolved_config.get("matrix")
        if isinstance(matrix_config, dict):
            rewire_path = (ROOT / matrix_config["rewire_manifest"]).resolve()
            expected_rewire_file_sha256 = matrix_config.get("rewire_manifest_sha256")
            rewire_seed = int(matrix_config["rewire_seed"])
        else:
            rewire_path = Path(resolved_config.get("rewire_manifest", DEFAULT_REWIRE_MANIFEST)).resolve()
            expected_rewire_file_sha256 = resolved_config.get("rewire_manifest_file_sha256")
            rewire_seed = int(resolved_config.get("rewire_seed", rewire_seed))
        rewire_manifest = load_fingerprint_rewire_manifest(
            rewire_path,
            expected_file_sha256=expected_rewire_file_sha256,
            expected_seed=rewire_seed,
        )
        current_fingerprint, current_fingerprint_payload = reproduction_fingerprint(
            resolved_config=resolved_config,
            evaluation_manifest=protocol,
            connectome_manifest=connectome,
            rewired_manifest=rewire_manifest,
        )
        if current_fingerprint != saved_fingerprint:
            raise ValueError(
                "Current source/config/runtime fingerprint differs from the checkpoint; "
                f"checkpoint={saved_fingerprint}, current={current_fingerprint}"
            )
        expected = current_fingerprint
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    print(json.dumps({
        "status": "RESOLVED",
        "resolved_config": resolved_config,
        "fingerprint": expected,
        "outputs": {"evaluation": str(args.output.resolve())},
    }, sort_keys=True), flush=True)

    app = AppLauncher(args).app
    try:
        from g1_fly_control.crazyflie.checkpoint import checkpoint_sha256, load_checkpoint
        from g1_fly_control.crazyflie.controllers import build_controller
        from g1_fly_control.crazyflie.normalization import RunningMeanVariance

        simulation_device = requested_device
        policy_device = requested_device
        policy, controller_report = build_controller(
            controller,
            observation_dim=12,
            action_dim=4,
            device=policy_device,
            connectome_manifest=connectome,
            wing_connectome_manifest=wing_manifest_path,
            rewire_seed=rewire_seed,
            rewire_manifest_path=(rewire_path if controller == "frozen_lif_degree_rewired" else None),
        )
        loaded = load_checkpoint(
            args.checkpoint,
            policy=policy,
            map_location=policy_device,
            expected_fingerprints={
                "reproduction": expected,
                "source_set": canonical_sha256(current_fingerprint_payload["source_sha256"]),
                "connectome": controller_report["connectome_checksum"],
                "frozen_core": controller_report["core_checksum"],
                "rewire_manifest": current_fingerprint_payload["rewired_manifest_sha256"],
            },
            expected_evaluation_manifest_id=protocol["manifest_id"],
            materialize_external_history=False,
        )
        if loaded["tainted"]:
            raise RuntimeError("Primary evaluation cannot use a tainted checkpoint")
        normalizer = RunningMeanVariance.create(12, device=simulation_device)
        normalizer.load_state_dict(loaded["normalizers"]["observation"])
        policy.eval()
        selected_scenarios = SCENARIOS if args.all_scenarios else (args.scenario,)
        results = {}
        for task in selected_scenarios:
            results[task] = _evaluate_scenario(
                task,
                protocol["scenarios"][task],
                policy,
                normalizer,
                protocol["evaluation_seed"],
                policy_device,
                contract_profile=contract_profile,
            )
        graph_reports = [results[task]["policy_inference_graph"] for task in selected_scenarios]
        if not graph_reports or any(report != graph_reports[0] for report in graph_reports[1:]):
            raise RuntimeError("Scenario CUDA-Graph execution contracts differ")
        if len(selected_scenarios) != 1:
            raise RuntimeError("Isaac child evaluation must contain exactly one scenario")
        scenario_memory_samples = results[selected_scenarios[0]]["memory_samples"]
        scenario_memory_gate = results[selected_scenarios[0]]["memory_gate"]
        base = {
            "schema_version": 1,
            "status": "completed",
            "label": protocol["label"],
            "protocol": args.protocol,
            "evaluation_seed": protocol["evaluation_seed"],
            "evaluation_manifest_id": protocol["manifest_id"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "training_seed": training_seed,
            "controller": controller,
            "fingerprint": expected,
            "fingerprint_payload": current_fingerprint_payload,
            "controller_report": controller_report,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "deterministic_actions": True,
            "simulation_device": str(simulation_device),
            "simulation_device_type": simulation_device.type,
            "policy_inference_device": str(policy_device),
            "policy_inference_device_type": policy_device.type,
            "policy_inference_backend": POLICY_INFERENCE_BACKEND,
            "policy_inference_precision": POLICY_INFERENCE_PRECISION,
            "policy_inference_graph": graph_reports[0],
            "policy_bridge": dict(POLICY_BRIDGE_CONTRACT),
            "memory_samples": scenario_memory_samples,
            "memory_gate": scenario_memory_gate,
            "failure_denominator_rule": "Every planned episode remains in the denominator",
            "censoring_rule": "Unsuccessful episode/attempt assigned its fixed observation horizon",
            "integrated_error_censoring_rule": (
                "For early termination, carry the final observed goal error through the fixed episode "
                "or post-gust recovery window"
            ),
        }
        if len(selected_scenarios) == 1:
            task = selected_scenarios[0]
            result = {
                **base,
                "scenario": task,
                "episodes": results[task]["episodes"],
                "summary": results[task]["summary"],
            }
        else:
            result = {
                **base,
                "scenario": "all",
                "episodes": [row for task in selected_scenarios for row in results[task]["episodes"]],
                "scenario_results": results,
                "total_episodes": sum(len(results[task]["episodes"]) for task in selected_scenarios),
            }
        _atomic_json(args.output.resolve(), result)
        print(json.dumps({
            "status": "PASS",
            "output": str(args.output.resolve()),
            "checkpoint": str(args.checkpoint),
            "controller": controller,
            "training_seed": training_seed,
            "scenarios": list(selected_scenarios),
            "episodes": len(result["episodes"]),
            "fingerprint": expected,
            "resolved_config": resolved_config,
        }, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        failure = {
            "schema_version": 1,
            "status": "failed",
            "checkpoint": str(args.checkpoint),
            "protocol": args.protocol,
            "scenario": "all" if args.all_scenarios else args.scenario,
            "memory_samples": _LAST_MEMORY_SAMPLES,
            "memory_gate": _LAST_MEMORY_GATE,
            "resolved_config": locals().get("resolved_config"),
            "fingerprint": locals().get("expected"),
            "error": traceback.format_exc(),
        }
        _atomic_json(args.output.resolve(), failure)
        print(json.dumps({
            "status": "FAIL",
            "output": str(args.output.resolve()),
            "checkpoint": str(args.checkpoint),
            "resolved_config": failure["resolved_config"],
            "fingerprint": failure["fingerprint"],
            "failure_reason": failure["error"],
        }, indent=2, sort_keys=True))
        return 1
    finally:
        # Scenario environments close inside _evaluate_scenario.  Avoid Kit's
        # status-masking native shutdown; the entrypoint exits explicitly.
        pass


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
