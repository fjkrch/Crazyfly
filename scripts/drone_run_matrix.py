#!/usr/bin/env python3
"""Create, dry-run, execute, and resume the sequential Crazyflie queue."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

from drone_bootstrap import (
    DEFAULT_CONNECTOME,
    ROOT,
    canonical_sha256,
    reproduction_fingerprint,
    rollout_rng_contract,
    sha256_file,
    source_hashes,
)
from drone_evaluation_protocol import SCENARIOS, load_protocol, validate_manifest
from g1_fly_control.crazyflie.memory import (
    GPU_LIMIT_MIB,
    MEMORY_POLICY_VERSION,
    RAM_LIMIT_PERCENT,
    RSS_GROWTH_TOLERANCE_MIB,
    SWAP_OUT_GROWTH_TOLERANCE_MIB,
    assess as assess_memory,
)
from g1_fly_control.crazyflie.stabilization import stabilization_contract_payload
from g1_fly_control.tasks.crazyflie.logic import (
    balanced_switch_target_curriculum_payload,
    balanced_task_contract_payload,
    balanced_v4_switch_target_curriculum_payload,
    balanced_v4_task_contract_payload,
    mixed_scenario_contract_payload,
    survival_first_contract_payload,
    switch_target_curriculum_payload,
)


CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "gru_matched",
    "mlp_normal",
)
TASK_SEPARATED_LAYOUT = "task_separated_v1"
TASK_SEPARATED_CELL_LAYOUT = "task_separated_v1_cell"
TASK_SEPARATED_SCHEMA_DRY_RUN = (
    "schema_dry_run_only_balanced_v3_not_launch_authorized"
)
TASK_SEPARATED_COMPARISON_LAYOUT = "task_separated_v1_comparison"
TASK_SEPARATED_COMPARISON_BASE_LAYOUT = "task_separated_v1_comparison_base"
TASK_SEPARATED_COMPARISON_CELL_LAYOUT = "task_separated_v1_comparison_cell"
TASK_SEPARATED_COMPARISON_READINESS = (
    "authorized_balanced_v4_seed0_1m_comparison_v1"
)
COMPARISON_LABEL = "comparison"
COMPARISON_TOTAL_INTERACTIONS = 1_000_000
COMPARISON_LEARNING_RATE = 3.0e-4
COMPARISON_SELECTION_RECEIPT = (
    ROOT / "runs" / "crazyflie-balanced-v4-lif-reach-lr-screen-selection.json"
)
COMPARISON_SELECTION_RECEIPT_SHA256 = (
    "bf0d5ef556be508822cc0ebc1919a41ac3e706bddeb173a9bbe0d6737c207489"
)
COMPARISON_SELECTION_ID = (
    "aff5081631f78881e39bebc5bb09137550c9a33e0bb094d1c14c37a1c5af4d8b"
)
COMPARISON_CONCURRENCY_RECEIPT = (
    ROOT / "runs" / "crazyflie-balanced-v4-seed0-comparison-paired-smoke-v1.json"
)
COMPARISON_CONCURRENCY_RECEIPT_KIND = (
    "crazyflie_balanced_v4_seed0_comparison_paired_smoke_v1"
)
COMPARISON_PAIRED_SMOKE_REPORTS = (
    ROOT / "runs" / "crazyflie-balanced-v4-comparison-paired-smoke-slot0.json",
    ROOT / "runs" / "crazyflie-balanced-v4-comparison-paired-smoke-slot1.json",
)
SURVIVAL_TASK = "FlyCrazyflie-WaypointReach-v0"
BALANCED_TASK = "FlyCrazyflie-Mixed-v0"
CONTRACT_PROFILE_SURVIVAL_V2 = "survival_v2"
CONTRACT_PROFILE_BALANCED_V3 = "balanced_v3"
CONTRACT_PROFILE_BALANCED_V4 = "balanced_v4"
TASK = SURVIVAL_TASK
VALID_STATUS = {"pending", "running", "completed", "failed", "paused", "cancelled"}
POLICY_INFERENCE_DEVICE = "cuda:0"
POLICY_INFERENCE_BACKEND = "cuda_graph_action_only_v1"
POLICY_INFERENCE_PRECISION = "float32"
POLICY_GRAPH_BASE_CONTRACT = {
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


def memory_acceptance_contract_payload() -> dict[str, Any]:
    """Return the canonical v2 hard/warning memory contract."""

    return {
        "policy_version": MEMORY_POLICY_VERSION,
        "device_gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "gpu_telemetry_required_for_cuda": True,
        "finite_numeric_telemetry_required": True,
        "sustained_paging_disposition": "hard_failure",
        "rss_growth_tolerance_mib": RSS_GROWTH_TOLERANCE_MIB,
        "rss_growth_disposition": "warning_only",
        "swap_out_growth_tolerance_mib": SWAP_OUT_GROWTH_TOLERANCE_MIB,
    }


def _policy_graph_config(batch_size: int) -> dict[str, Any]:
    return {**POLICY_GRAPH_BASE_CONTRACT, "fixed_batch_size": batch_size}


def _policy_graph_artifact(batch_size: int) -> dict[str, Any]:
    return {
        **_policy_graph_config(batch_size),
        "capture_count": 1,
        "bitwise_parity_verified": True,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _copy_resolved_base_config(base: dict[str, Any]) -> dict[str, Any]:
    """Copy a validated config without aliasing its mutable public values."""

    return json.loads(json.dumps(public_config(base)))


def _resolve_config_reference(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must name a JSON config")
    return (ROOT / value).resolve()


def _resolve_recorded_artifact(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must name an artifact")
    source = Path(value).expanduser()
    return (source if source.is_absolute() else ROOT / source).resolve()


def _validate_comparison_learning_rate_selection(
    declaration: Any,
) -> dict[str, Any]:
    """Authenticate the immutable v4 Reach LR selection receipt.

    The selection predates this additive queue implementation, so its recorded
    source-set hash is historical evidence rather than a demand that the
    current source tree still have the same bytes.  The receipt itself, its
    self-authenticating selection id, both candidate evidence records, and all
    referenced immutable manifests/checkpoints remain hash checked here.
    """

    expected_fields = {
        "receipt",
        "receipt_sha256",
        "selection_id",
        "selected_candidate_id",
        "selected_learning_rate",
        "application",
    }
    if not isinstance(declaration, dict) or set(declaration) != expected_fields:
        raise ValueError(
            "comparison learning_rate_selection must contain exactly "
            f"{sorted(expected_fields)}"
        )
    expected_declaration = {
        "receipt": str(COMPARISON_SELECTION_RECEIPT.relative_to(ROOT)),
        "receipt_sha256": COMPARISON_SELECTION_RECEIPT_SHA256,
        "selection_id": COMPARISON_SELECTION_ID,
        "selected_candidate_id": "lr-3e-4",
        "selected_learning_rate": COMPARISON_LEARNING_RATE,
        "application": (
            "common_learning_rate_all_four_controllers_all_three_tasks_"
            "seed0_comparison_only"
        ),
    }
    if declaration != expected_declaration:
        raise ValueError("comparison learning_rate_selection declaration changed")

    receipt_path = _resolve_recorded_artifact(
        declaration["receipt"], field="learning_rate_selection.receipt"
    )
    if receipt_path != COMPARISON_SELECTION_RECEIPT.resolve() or not receipt_path.is_file():
        raise ValueError(f"Comparison LR selection receipt is missing: {receipt_path}")
    if sha256_file(receipt_path) != declaration["receipt_sha256"]:
        raise ValueError("Comparison LR selection receipt SHA-256 changed")
    receipt = _read_json(receipt_path)
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"selection_id", "created_at_utc"}
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "crazyflie_balanced_v4_original_lif_reach_lr_selection"
        or receipt.get("status") != "selected"
        or receipt.get("contract_profile") != CONTRACT_PROFILE_BALANCED_V4
        or receipt.get("task") != SURVIVAL_TASK
        or receipt.get("controller") != "frozen_lif_original"
        or receipt.get("held_out_evaluation_used") is not False
        or receipt.get("warm_start_authorized") is not False
        or receipt.get("all_hard_gates_passed") is not True
        or receipt.get("selection_id") != declaration["selection_id"]
        or canonical_sha256(body) != declaration["selection_id"]
        or receipt.get("selected_candidate_id")
        != declaration["selected_candidate_id"]
        or receipt.get("selected_learning_rate")
        != declaration["selected_learning_rate"]
    ):
        raise ValueError("Comparison LR selection receipt identity or result changed")

    candidates = receipt.get("candidates")
    ranking = receipt.get("ranking")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 2
        or not all(isinstance(item, dict) for item in candidates)
        or not isinstance(ranking, list)
        or len(ranking) != 2
        or not all(isinstance(item, dict) for item in ranking)
    ):
        raise ValueError("Comparison LR selection receipt candidate/ranking evidence is malformed")
    by_id = {item.get("candidate_id"): item for item in candidates}
    if set(by_id) != {"lr-1e-4", "lr-3e-4"}:
        raise ValueError("Comparison LR selection receipt candidates changed")
    if (
        ranking[0].get("rank") != 1
        or ranking[0].get("candidate_id") != declaration["selected_candidate_id"]
        or ranking[0].get("learning_rate") != declaration["selected_learning_rate"]
    ):
        raise ValueError("Comparison LR selection winner/ranking changed")

    for candidate_id, evidence in by_id.items():
        evidence_sha = evidence.get("evidence_sha256")
        evidence_body = dict(evidence)
        evidence_body.pop("evidence_sha256", None)
        if (
            not isinstance(evidence_sha, str)
            or canonical_sha256(evidence_body) != evidence_sha
            or evidence.get("hard_gates_passed") is not True
        ):
            raise ValueError(f"Comparison LR evidence is invalid for {candidate_id}")
        for path_field, hash_field in (
            ("training_manifest", "training_manifest_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            artifact = _resolve_recorded_artifact(
                evidence.get(path_field), field=f"{candidate_id}.{path_field}"
            )
            expected_sha = evidence.get(hash_field)
            if (
                not artifact.is_file()
                or not isinstance(expected_sha, str)
                or sha256_file(artifact) != expected_sha
            ):
                raise ValueError(
                    f"Comparison LR evidence artifact changed for {candidate_id}: {artifact}"
                )
    selected = by_id[declaration["selected_candidate_id"]]
    if (
        selected.get("learning_rate") != declaration["selected_learning_rate"]
        or receipt.get("selected_checkpoint_sha256")
        != selected.get("checkpoint_sha256")
    ):
        raise ValueError("Comparison LR selected candidate evidence changed")

    for path_field, hash_field in (
        ("declaration", "declaration_sha256"),
        ("selector", "selector_sha256"),
        ("selected_checkpoint", "selected_checkpoint_sha256"),
    ):
        artifact = _resolve_recorded_artifact(
            receipt.get(path_field), field=f"selection_receipt.{path_field}"
        )
        expected_sha = receipt.get(hash_field)
        if (
            not artifact.is_file()
            or not isinstance(expected_sha, str)
            or sha256_file(artifact) != expected_sha
        ):
            raise ValueError(f"Comparison LR receipt dependency changed: {artifact}")
    return receipt


def _comparison_training_contract() -> dict[str, Any]:
    return {
        "num_envs": 4,
        "horizon": 100,
        "ppo_epochs": 2,
        "microbatch_size": 4,
        "num_workers": 0,
        "precision": "float32",
        "learning_rate": COMPARISON_LEARNING_RATE,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "checkpoint_every_updates": 100,
    }


def _comparison_concurrency_preflight_contract() -> dict[str, Any]:
    return {
        "receipt": str(COMPARISON_CONCURRENCY_RECEIPT.relative_to(ROOT)),
        "requested_max_concurrent_isaac_processes": 2,
        "fallback_max_concurrent_isaac_processes": 1,
        "missing_or_invalid_disposition": "record_fallback_to_sequential",
        "paired_smoke_task": SURVIVAL_TASK,
        "paired_smoke_controller": "frozen_lif_original",
        "paired_smoke_seeds": [0, 1],
        "paired_smoke_num_envs_per_process": 4,
        "paired_smoke_steps": 1000,
        "paired_smoke_requires_ppo_update": True,
    }


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not an ISO timestamp") from exc
    if result.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return result


def _validate_current_source_hashes(payload: Any, *, label: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} fingerprint payload is missing")
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{label} source fingerprint is missing")
    for recorded_path, expected_sha in sources.items():
        if (
            not isinstance(recorded_path, str)
            or not recorded_path
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ValueError(f"{label} contains a malformed source hash")
        source = Path(recorded_path)
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError(f"{label} fingerprinted source changed: {source}")
    current_sources = source_hashes()
    if sources != current_sources:
        recorded_paths = set(sources)
        current_paths = set(current_sources)
        missing = sorted(current_paths - recorded_paths)
        extra = sorted(recorded_paths - current_paths)
        changed = sorted(
            path
            for path in recorded_paths & current_paths
            if sources[path] != current_sources[path]
        )
        raise ValueError(
            f"{label} does not authenticate the complete current executable "
            f"source set: missing={missing}, extra={extra}, changed={changed}"
        )
    return canonical_sha256(sources)


def _validate_paired_smoke_report(entry: Any, *, expected_slot: int) -> dict[str, Any]:
    expected_fields = {"slot", "seed", "path", "sha256"}
    if not isinstance(entry, dict) or set(entry) != expected_fields:
        raise ValueError(
            f"Paired-smoke report {expected_slot} must contain exactly "
            f"{sorted(expected_fields)}"
        )
    expected_path = COMPARISON_PAIRED_SMOKE_REPORTS[expected_slot].resolve()
    expected_seed = expected_slot
    path = _resolve_recorded_artifact(
        entry.get("path"), field=f"paired_smoke.reports[{expected_slot}].path"
    )
    if (
        entry.get("slot") != expected_slot
        or entry.get("seed") != expected_seed
        or path != expected_path
        or not path.is_file()
        or not isinstance(entry.get("sha256"), str)
        or sha256_file(path) != entry["sha256"]
    ):
        raise ValueError(f"Paired-smoke report identity changed for slot {expected_slot}")
    report = _read_json(path)
    fingerprint_payload = report.get("fingerprint_payload")
    fingerprint = report.get("fingerprint")
    source_set_sha256 = _validate_current_source_hashes(
        fingerprint_payload, label=f"paired-smoke slot {expected_slot}"
    )
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or not isinstance(fingerprint_payload, dict)
        or canonical_sha256(fingerprint_payload) != fingerprint
    ):
        raise ValueError(f"Paired-smoke fingerprint is invalid for slot {expected_slot}")
    resolved = fingerprint_payload.get("resolved_config")
    policy_report = report.get("policy_report")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("task") != SURVIVAL_TASK
        or report.get("contract_profile") != CONTRACT_PROFILE_BALANCED_V4
        or report.get("policy") != "frozen_lif_original"
        or report.get("seed") != expected_seed
        or report.get("num_envs") != 4
        or report.get("steps") != 1000
        or report.get("ppo_update") is not True
        or report.get("full_gate_c_acceptance_passed") is not True
        or report.get("observations_finite") is not True
        or report.get("invalid_state_count") != 0
        or not isinstance(report.get("ppo_metrics"), dict)
        or not isinstance(resolved, dict)
        or resolved.get("contract_profile") != CONTRACT_PROFILE_BALANCED_V4
        or not isinstance(policy_report, dict)
        or policy_report.get("controller_kind") != "frozen_lif"
        or policy_report.get("actor_parameter_match_passed") is not True
        or report.get("frozen_core_checksum_before")
        != report.get("frozen_core_checksum_after")
        or report.get("frozen_core_checksum_before")
        != policy_report.get("core_checksum")
    ):
        raise ValueError(f"Paired-smoke hard gate failed for slot {expected_slot}")
    ppo_metrics = report["ppo_metrics"]
    if ppo_metrics.get("all_finite") is not True or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for key, value in ppo_metrics.items()
        if key != "all_finite"
    ):
        raise ValueError(f"Paired-smoke PPO metrics are nonfinite for slot {expected_slot}")
    samples = report.get("memory_samples")
    recorded_gate = report.get("memory_gate")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Paired-smoke memory samples are missing for slot {expected_slot}")
    try:
        recomputed_gate = assess_memory(samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Paired-smoke memory evidence is invalid for slot {expected_slot}: {exc}"
        ) from exc
    if (
        recorded_gate != recomputed_gate
        or recomputed_gate.get("passed") is not True
        or recomputed_gate.get("failures") != []
        or recomputed_gate.get("sustained_paging_detected") is not False
        or recomputed_gate.get("max_device_gpu_used_mib", math.inf) >= GPU_LIMIT_MIB
        or recomputed_gate.get("max_system_ram_percent", math.inf) >= RAM_LIMIT_PERCENT
    ):
        raise ValueError(f"Paired-smoke memory hard gate failed for slot {expected_slot}")
    timed_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Paired-smoke sample {expected_slot}:{index} is malformed")
        timed_samples.append(
            (
                _timestamp(
                    sample.get("timestamp_utc"),
                    field=f"paired_smoke[{expected_slot}].memory_samples[{index}]",
                ),
                sample,
            )
        )
    return {
        "slot": expected_slot,
        "seed": expected_seed,
        "path": str(path),
        "sha256": entry["sha256"],
        "fingerprint": fingerprint,
        "source_set_sha256": source_set_sha256,
        "memory_gate": recomputed_gate,
        "timed_samples": timed_samples,
    }


def _paired_smoke_overlap_evidence(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reports) != 2:
        raise ValueError("Exactly two paired-smoke reports are required")
    starts = [min(when for when, _ in report["timed_samples"]) for report in reports]
    ends = [max(when for when, _ in report["timed_samples"]) for report in reports]
    overlap_start = max(starts)
    overlap_end = min(ends)
    duration = (overlap_end - overlap_start).total_seconds()
    if duration < 1.0:
        raise ValueError("Paired-smoke reports do not prove concurrent execution")
    overlap_by_slot: list[list[dict[str, Any]]] = []
    for report in reports:
        overlap = [
            sample
            for when, sample in report["timed_samples"]
            if overlap_start <= when <= overlap_end
            and sample.get("stage") == "steady_state"
        ]
        if len(overlap) < 2:
            raise ValueError(
                f"Paired-smoke slot {report['slot']} lacks two overlapping steady-state samples"
            )
        overlap_by_slot.append(overlap)
    overlap_samples = [sample for group in overlap_by_slot for sample in group]
    gpu_values: list[float] = []
    ram_values: list[float] = []
    swap_out_values: list[float] = []
    for sample in overlap_samples:
        devices = sample.get("gpu_devices")
        if not isinstance(devices, list) or not devices:
            raise ValueError("Paired-smoke overlap lacks device-wide GPU telemetry")
        try:
            gpu_values.append(sum(float(device["used_mib"]) for device in devices))
            ram_values.append(float(sample["system_ram_percent"]))
            swap_out_values.append(float(sample["system_swap_out_mib"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Paired-smoke overlap telemetry is malformed") from exc
    if not all(math.isfinite(value) for value in gpu_values + ram_values + swap_out_values):
        raise ValueError("Paired-smoke overlap telemetry is nonfinite")
    max_gpu = max(gpu_values)
    max_ram = max(ram_values)
    swap_growth = max(swap_out_values) - min(swap_out_values)
    if max_gpu >= GPU_LIMIT_MIB or max_ram >= RAM_LIMIT_PERCENT:
        raise ValueError("Paired-smoke aggregate RAM/VRAM hard gate failed")
    if swap_growth > SWAP_OUT_GROWTH_TOLERANCE_MIB:
        raise ValueError("Paired-smoke aggregate sustained-paging gate failed")
    return {
        "overlap_start_utc": overlap_start.isoformat(),
        "overlap_end_utc": overlap_end.isoformat(),
        "overlap_duration_s": duration,
        "steady_state_samples_per_slot": [len(group) for group in overlap_by_slot],
        "max_device_gpu_used_mib": max_gpu,
        "max_system_ram_percent": max_ram,
        "swap_out_growth_mib": swap_growth,
        "gpu_limit_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "passed": True,
    }


def _validate_comparison_concurrency_receipt(
    declaration: Any,
) -> dict[str, Any]:
    if declaration != _comparison_concurrency_preflight_contract():
        raise ValueError("Comparison concurrency_preflight declaration changed")
    receipt_path = _resolve_recorded_artifact(
        declaration["receipt"], field="concurrency_preflight.receipt"
    )
    if receipt_path != COMPARISON_CONCURRENCY_RECEIPT.resolve() or not receipt_path.is_file():
        raise FileNotFoundError(f"Paired-smoke receipt is missing: {receipt_path}")
    receipt = _read_json(receipt_path)
    expected_fields = {
        "schema_version",
        "kind",
        "status",
        "created_at_utc",
        "requested_max_concurrent_isaac_processes",
        "effective_max_concurrent_isaac_processes",
        "reports",
        "receipt_id",
    }
    if set(receipt) != expected_fields:
        raise ValueError(
            f"Paired-smoke receipt must contain exactly {sorted(expected_fields)}"
        )
    body = dict(receipt)
    receipt_id = body.pop("receipt_id", None)
    _timestamp(receipt.get("created_at_utc"), field="paired_smoke.created_at_utc")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != COMPARISON_CONCURRENCY_RECEIPT_KIND
        or receipt.get("status") != "PASS"
        or receipt.get("requested_max_concurrent_isaac_processes") != 2
        or receipt.get("effective_max_concurrent_isaac_processes") != 2
        or not isinstance(receipt_id, str)
        or canonical_sha256(body) != receipt_id
    ):
        raise ValueError("Paired-smoke receipt identity or self-hash changed")
    entries = receipt.get("reports")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("Paired-smoke receipt must contain exactly two reports")
    reports = [
        _validate_paired_smoke_report(entry, expected_slot=index)
        for index, entry in enumerate(entries)
    ]
    source_sets = {report["source_set_sha256"] for report in reports}
    if len(source_sets) != 1:
        raise ValueError("Paired-smoke reports use different executable source sets")
    overlap = _paired_smoke_overlap_evidence(reports)
    return {
        "status": "paired_smoke_pass",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 2,
        "fallback_applied": False,
        "fallback_reason": None,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_id": receipt_id,
        "source_set_sha256": next(iter(source_sets)),
        "reports": [
            {
                key: report[key]
                for key in ("slot", "seed", "path", "sha256", "fingerprint")
            }
            for report in reports
        ],
        "aggregate_overlap_memory_gate": overlap,
    }


def comparison_concurrency_decision(config: dict[str, Any]) -> dict[str, Any]:
    """Return an authenticated concurrency-2 decision or recorded fallback."""

    declaration = config.get("concurrency_preflight")
    try:
        return _validate_comparison_concurrency_receipt(declaration)
    except Exception as exc:
        receipt_path = COMPARISON_CONCURRENCY_RECEIPT.resolve()
        return {
            "status": "fallback_sequential",
            "requested_max_concurrent_isaac_processes": 2,
            "effective_max_concurrent_isaac_processes": 1,
            "fallback_applied": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "receipt": str(receipt_path),
            "receipt_sha256": (
                sha256_file(receipt_path) if receipt_path.is_file() else None
            ),
            "receipt_id": None,
            "source_set_sha256": None,
            "reports": [],
            "aggregate_overlap_memory_gate": None,
        }


def _validate_task_separated_comparison_base(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the compact, non-executable common v4 comparison config."""

    expected_fields = {
        "schema_version",
        "label",
        "matrix_layout",
        "execution_readiness",
        "reference_config",
        "controllers",
        "seeds",
        "total_interactions",
        "contract_profile",
        "learning_rate_selection",
        "concurrency_preflight",
        "training",
        "evaluation",
    }
    if set(declaration) != expected_fields:
        raise ValueError(
            "Comparison base config must contain exactly "
            f"{sorted(expected_fields)}"
        )
    if (
        declaration.get("schema_version") != 1
        or declaration.get("label") != COMPARISON_LABEL
        or declaration.get("matrix_layout")
        != TASK_SEPARATED_COMPARISON_BASE_LAYOUT
        or declaration.get("execution_readiness")
        != TASK_SEPARATED_COMPARISON_READINESS
        or declaration.get("contract_profile") != CONTRACT_PROFILE_BALANCED_V4
        or tuple(declaration.get("controllers", ())) != CONTROLLERS
        or declaration.get("seeds") != [0]
        or declaration.get("total_interactions") != COMPARISON_TOTAL_INTERACTIONS
        or declaration.get("training") != _comparison_training_contract()
        or declaration.get("concurrency_preflight")
        != _comparison_concurrency_preflight_contract()
    ):
        raise ValueError("Comparison base identity, shape, or training contract changed")
    reference_path = _resolve_config_reference(
        declaration.get("reference_config"), field="reference_config"
    )
    expected_reference = (
        ROOT / "configs" / "experiments" / "crazyflie_balanced_v3_main.json"
    ).resolve()
    if reference_path != expected_reference:
        raise ValueError("Comparison base must reference the reviewed balanced-v3 main config")
    reference = validate_config(reference_path)
    if (
        reference.get("label") != "main"
        or reference.get("task") != BALANCED_TASK
        or reference.get("_contract_profile") != CONTRACT_PROFILE_BALANCED_V3
    ):
        raise ValueError("Comparison reference config is not the reviewed balanced-v3 main config")
    if declaration.get("evaluation") != reference["evaluation"]:
        raise ValueError("Comparison evaluation must reuse the exact 16-episode main protocol")
    selection = _validate_comparison_learning_rate_selection(
        declaration.get("learning_rate_selection")
    )

    config = _copy_resolved_base_config(reference)
    config.pop("task", None)
    config.pop("balanced_task_contract", None)
    config.update(
        {
            "label": COMPARISON_LABEL,
            "matrix_layout": TASK_SEPARATED_COMPARISON_BASE_LAYOUT,
            "execution_readiness": TASK_SEPARATED_COMPARISON_READINESS,
            "reference_config": declaration["reference_config"],
            "controllers": list(CONTROLLERS),
            "seeds": [0],
            "total_interactions": COMPARISON_TOTAL_INTERACTIONS,
            "contract_profile": CONTRACT_PROFILE_BALANCED_V4,
            "learning_rate_selection": dict(declaration["learning_rate_selection"]),
            "concurrency_preflight": dict(declaration["concurrency_preflight"]),
            "training": dict(declaration["training"]),
            "evaluation": json.loads(json.dumps(declaration["evaluation"])),
            "balanced_v4_task_contract": balanced_v4_task_contract_payload(),
        }
    )
    for key, value in reference.items():
        if key.startswith("_"):
            config[key] = value
    config["_config_path"] = str(path)
    config["_matrix_layout"] = TASK_SEPARATED_COMPARISON_BASE_LAYOUT
    config["_reference_config_path"] = str(reference_path)
    config["_contract_profile"] = CONTRACT_PROFILE_BALANCED_V4
    config["_learning_rate_selection_receipt"] = selection
    return config


def _validate_task_separated_comparison_cell(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "label",
        "matrix_layout",
        "execution_readiness",
        "base_config",
        "parent_config",
        "task",
    }
    if set(declaration) != expected_fields:
        raise ValueError(
            "Comparison cell config must contain exactly "
            f"{sorted(expected_fields)}"
        )
    if (
        declaration.get("schema_version") != 1
        or declaration.get("label") != COMPARISON_LABEL
        or declaration.get("matrix_layout")
        != TASK_SEPARATED_COMPARISON_CELL_LAYOUT
        or declaration.get("execution_readiness")
        != TASK_SEPARATED_COMPARISON_READINESS
        or declaration.get("task") not in SCENARIOS
    ):
        raise ValueError("Comparison cell identity, readiness, or task changed")
    base_path = _resolve_config_reference(
        declaration.get("base_config"), field="base_config"
    )
    if base_path == path:
        raise ValueError("Comparison cell base_config cannot reference itself")
    base = validate_config(base_path)
    if base.get("_matrix_layout") != TASK_SEPARATED_COMPARISON_BASE_LAYOUT:
        raise ValueError("Comparison cell requires the reviewed comparison base config")
    parent_path = _resolve_config_reference(
        declaration.get("parent_config"), field="parent_config"
    )
    if not parent_path.is_file() or parent_path == path:
        raise ValueError("Comparison cell parent_config is missing or self-referential")
    parent = _read_json(parent_path)
    task = declaration["task"]
    if (
        parent.get("matrix_layout") != TASK_SEPARATED_COMPARISON_LAYOUT
        or parent.get("execution_readiness") != TASK_SEPARATED_COMPARISON_READINESS
        or parent.get("base_config") != declaration["base_config"]
        or not isinstance(parent.get("task_configs"), dict)
        or _resolve_config_reference(
            parent["task_configs"].get(task), field=f"task_configs[{task}]"
        )
        != path
    ):
        raise ValueError("Comparison cell parent binding is invalid")

    config = _copy_resolved_base_config(base)
    config.update(
        {
            "matrix_layout": TASK_SEPARATED_COMPARISON_CELL_LAYOUT,
            "base_config": declaration["base_config"],
            "parent_config": declaration["parent_config"],
            "task": task,
        }
    )
    for key, value in base.items():
        if key.startswith("_"):
            config[key] = value
    config["_config_path"] = str(path)
    config["_matrix_layout"] = TASK_SEPARATED_COMPARISON_CELL_LAYOUT
    config["_parent_config_path"] = str(parent_path)
    config["_base_config_path"] = str(base_path)
    return config


def _validate_task_separated_comparison_umbrella(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "label",
        "matrix_layout",
        "execution_readiness",
        "base_config",
        "tasks",
        "task_configs",
        "controllers",
        "seeds",
        "total_interactions",
    }
    if set(declaration) != expected_fields:
        raise ValueError(
            "Comparison umbrella config must contain exactly "
            f"{sorted(expected_fields)}"
        )
    if (
        declaration.get("schema_version") != 1
        or declaration.get("label") != COMPARISON_LABEL
        or declaration.get("matrix_layout") != TASK_SEPARATED_COMPARISON_LAYOUT
        or declaration.get("execution_readiness")
        != TASK_SEPARATED_COMPARISON_READINESS
        or tuple(declaration.get("tasks", ())) != SCENARIOS
        or tuple(declaration.get("controllers", ())) != CONTROLLERS
        or declaration.get("seeds") != [0]
        or declaration.get("total_interactions") != COMPARISON_TOTAL_INTERACTIONS
    ):
        raise ValueError("Comparison umbrella identity or exact 12-job design changed")
    base_path = _resolve_config_reference(
        declaration.get("base_config"), field="base_config"
    )
    if base_path == path:
        raise ValueError("Comparison umbrella base_config cannot reference itself")
    base = validate_config(base_path)
    if base.get("_matrix_layout") != TASK_SEPARATED_COMPARISON_BASE_LAYOUT:
        raise ValueError("Comparison umbrella requires the reviewed comparison base config")
    for field in ("controllers", "seeds", "total_interactions"):
        if declaration[field] != base[field]:
            raise ValueError(f"Comparison umbrella {field} differs from base config")

    task_config_settings = declaration.get("task_configs")
    if (
        not isinstance(task_config_settings, dict)
        or set(task_config_settings) != set(SCENARIOS)
    ):
        raise ValueError("Comparison task_configs must map every and only declared task")
    task_configs: dict[str, dict[str, Any]] = {}
    for task in SCENARIOS:
        cell_path = _resolve_config_reference(
            task_config_settings[task], field=f"task_configs[{task}]"
        )
        cell = validate_config(cell_path)
        if (
            cell.get("_matrix_layout") != TASK_SEPARATED_COMPARISON_CELL_LAYOUT
            or cell.get("task") != task
            or cell.get("_parent_config_path") != str(path)
            or cell.get("_base_config_path") != str(base_path)
        ):
            raise ValueError(f"Comparison cell binding is invalid for {task}")
        for field in (
            "label",
            "controllers",
            "seeds",
            "total_interactions",
            "training",
            "evaluation",
            "learning_rate_selection",
            "concurrency_preflight",
            "balanced_v4_task_contract",
        ):
            if cell[field] != base[field]:
                raise ValueError(f"Comparison {task} cell differs in {field}")
        task_configs[task] = cell

    config = _copy_resolved_base_config(base)
    config.update(
        {
            "matrix_layout": TASK_SEPARATED_COMPARISON_LAYOUT,
            "base_config": declaration["base_config"],
            "tasks": list(SCENARIOS),
            "task_configs": dict(task_config_settings),
        }
    )
    config["evaluation"]["matched_task_only"] = True
    for key, value in base.items():
        if key.startswith("_"):
            config[key] = value
    config["_config_path"] = str(path)
    config["_matrix_layout"] = TASK_SEPARATED_COMPARISON_LAYOUT
    config["_base_config_path"] = str(base_path)
    config["_task_configs"] = task_configs
    return config


def _validate_task_separated_cell(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one immutable scalar-task config used by ``drone_train.py``.

    The trainer intentionally accepts a scalar task.  Small checked-in cell
    configs keep that execution contract intact while the umbrella matrix
    enumerates all three independently initialized tasks.
    """

    expected_fields = {
        "schema_version",
        "label",
        "matrix_layout",
        "execution_readiness",
        "base_config",
        "parent_config",
        "task",
    }
    if set(declaration) != expected_fields:
        raise ValueError(
            "Task-separated cell config must contain exactly "
            f"{sorted(expected_fields)}"
        )
    task = declaration.get("task")
    if declaration.get("execution_readiness") != TASK_SEPARATED_SCHEMA_DRY_RUN:
        raise ValueError("Task-separated cell is not the reviewed schema-dry-run version")
    if task not in SCENARIOS:
        raise ValueError(f"Task-separated cell task must be one of {list(SCENARIOS)}")
    base_path = _resolve_config_reference(
        declaration.get("base_config"), field="base_config"
    )
    if base_path == path:
        raise ValueError("Task-separated cell base_config cannot reference itself")
    base = validate_config(base_path)
    if base.get("_matrix_layout") in {
        TASK_SEPARATED_LAYOUT,
        TASK_SEPARATED_CELL_LAYOUT,
    }:
        raise ValueError("Task-separated cell base_config must be a historical scalar matrix")
    if declaration.get("label") != base["label"]:
        raise ValueError("Task-separated cell label differs from base_config")
    if (
        base.get("task") != BALANCED_TASK
        or base.get("_contract_profile") != CONTRACT_PROFILE_BALANCED_V3
    ):
        raise ValueError("Task-separated cells require a validated balanced-v3 base config")
    parent_path = _resolve_config_reference(
        declaration.get("parent_config"), field="parent_config"
    )
    if not parent_path.is_file():
        raise ValueError(f"Task-separated parent config does not exist: {parent_path}")

    config = _copy_resolved_base_config(base)
    config["task"] = task
    config["matrix_layout"] = TASK_SEPARATED_CELL_LAYOUT
    config["execution_readiness"] = TASK_SEPARATED_SCHEMA_DRY_RUN
    config["base_config"] = declaration["base_config"]
    config["parent_config"] = declaration["parent_config"]
    for key, value in base.items():
        if key.startswith("_"):
            config[key] = value
    config["_config_path"] = str(path)
    config["_matrix_layout"] = TASK_SEPARATED_CELL_LAYOUT
    config["_parent_config_path"] = str(parent_path)
    config["_base_config_path"] = str(base_path)
    return config


def _validate_task_separated_umbrella(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    """Validate and resolve the three-task task-separated-v1 matrix."""

    expected_fields = {
        "schema_version",
        "label",
        "matrix_layout",
        "execution_readiness",
        "base_config",
        "tasks",
        "task_configs",
        "controllers",
        "seeds",
        "total_interactions",
    }
    if set(declaration) != expected_fields:
        raise ValueError(
            "Task-separated umbrella config must contain exactly "
            f"{sorted(expected_fields)}"
        )
    label = declaration.get("label")
    if label not in {"integration", "main"}:
        raise ValueError("Task-separated label must be integration or main")
    if declaration.get("execution_readiness") != TASK_SEPARATED_SCHEMA_DRY_RUN:
        raise ValueError("Task-separated umbrella is not the reviewed schema-dry-run version")
    if tuple(declaration.get("tasks", ())) != SCENARIOS:
        raise ValueError(f"tasks must be exactly {list(SCENARIOS)} in the declared order")
    if tuple(declaration.get("controllers", ())) != CONTROLLERS:
        raise ValueError(f"controllers must be exactly {list(CONTROLLERS)} in the declared order")
    expected_seeds = [0] if label == "integration" else [0, 1, 2, 3, 4]
    if declaration.get("seeds") != expected_seeds:
        raise ValueError(f"{label} seeds must be exactly {expected_seeds}")
    expected_total = 50 if label == "integration" else 5_000_000
    if declaration.get("total_interactions") != expected_total:
        raise ValueError(
            f"{label} jobs must contain exactly {expected_total:,} interactions"
        )

    base_path = _resolve_config_reference(
        declaration.get("base_config"), field="base_config"
    )
    if base_path == path:
        raise ValueError("Task-separated base_config cannot reference itself")
    base = validate_config(base_path)
    if base.get("_matrix_layout") in {
        TASK_SEPARATED_LAYOUT,
        TASK_SEPARATED_CELL_LAYOUT,
    }:
        raise ValueError("Task-separated base_config must be a historical scalar matrix")
    if (
        base.get("label") != label
        or base.get("task") != BALANCED_TASK
        or base.get("_contract_profile") != CONTRACT_PROFILE_BALANCED_V3
    ):
        raise ValueError("Task-separated umbrella requires the matching balanced-v3 base config")
    for field in ("controllers", "seeds", "total_interactions"):
        if declaration[field] != base[field]:
            raise ValueError(f"Task-separated {field} differs from base_config")

    task_config_settings = declaration.get("task_configs")
    if not isinstance(task_config_settings, dict) or set(task_config_settings) != set(SCENARIOS):
        raise ValueError("task_configs must map every and only declared task")
    task_configs: dict[str, dict[str, Any]] = {}
    for task in SCENARIOS:
        cell_path = _resolve_config_reference(
            task_config_settings[task], field=f"task_configs[{task}]"
        )
        if cell_path == path:
            raise ValueError("Task-separated task_configs cannot reference the umbrella itself")
        cell = validate_config(cell_path)
        if (
            cell.get("_matrix_layout") != TASK_SEPARATED_CELL_LAYOUT
            or cell.get("task") != task
            or cell.get("_parent_config_path") != str(path)
            or cell.get("_base_config_path") != str(base_path)
        ):
            raise ValueError(f"Task-separated cell binding is invalid for {task}")
        for field in ("label", "controllers", "seeds", "total_interactions", "training"):
            if cell[field] != base[field]:
                raise ValueError(f"Task-separated {task} cell differs in {field}")
        if cell["evaluation"] != base["evaluation"]:
            raise ValueError(f"Task-separated {task} cell differs in evaluation protocol")
        task_configs[task] = cell

    config = _copy_resolved_base_config(base)
    config.pop("task", None)
    config.update({
        "matrix_layout": TASK_SEPARATED_LAYOUT,
        "execution_readiness": TASK_SEPARATED_SCHEMA_DRY_RUN,
        "base_config": declaration["base_config"],
        "tasks": list(SCENARIOS),
        "task_configs": dict(task_config_settings),
        "controllers": list(CONTROLLERS),
        "seeds": list(expected_seeds),
        "total_interactions": expected_total,
    })
    config["evaluation"]["matched_task_only"] = True
    for key, value in base.items():
        if key.startswith("_"):
            config[key] = value
    config["_config_path"] = str(path)
    config["_matrix_layout"] = TASK_SEPARATED_LAYOUT
    config["_base_config_path"] = str(base_path)
    config["_task_configs"] = task_configs
    return config


def validate_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _read_json(path)
    matrix_layout = config.get("matrix_layout")
    if matrix_layout == TASK_SEPARATED_COMPARISON_BASE_LAYOUT:
        return _validate_task_separated_comparison_base(path, config)
    if matrix_layout == TASK_SEPARATED_COMPARISON_CELL_LAYOUT:
        return _validate_task_separated_comparison_cell(path, config)
    if matrix_layout == TASK_SEPARATED_COMPARISON_LAYOUT:
        return _validate_task_separated_comparison_umbrella(path, config)
    if config.get("schema_version") != 1 or config.get("label") not in {"integration", "main"}:
        raise ValueError("Matrix config needs schema_version 1 and label integration or main")
    if matrix_layout == TASK_SEPARATED_LAYOUT:
        return _validate_task_separated_umbrella(path, config)
    if matrix_layout == TASK_SEPARATED_CELL_LAYOUT:
        return _validate_task_separated_cell(path, config)
    if matrix_layout is not None:
        raise ValueError(f"Unknown matrix_layout: {matrix_layout!r}")
    memory_acceptance = memory_acceptance_contract_payload()
    declared_memory_acceptance = config.get("memory_acceptance")
    if (
        "memory_acceptance" in config
        and declared_memory_acceptance != memory_acceptance
    ):
        raise ValueError(
            f"Matrix memory_acceptance must match {MEMORY_POLICY_VERSION}"
        )
    # Source JSON files predate the additive v2 contract.  Resolve it
    # canonically so it enters config hashes and per-job fingerprints while a
    # conflicting declaration still fails closed above.
    config["memory_acceptance"] = memory_acceptance
    survival_present = "survival_first_contract" in config
    balanced_present = "balanced_task_contract" in config
    if survival_present == balanced_present:
        raise ValueError(
            "Matrix config must contain exactly one of survival_first_contract "
            "or balanced_task_contract"
        )
    if balanced_present:
        contract_profile = CONTRACT_PROFILE_BALANCED_V3
        expected_task = BALANCED_TASK
        if config.get("balanced_task_contract") != balanced_task_contract_payload():
            raise ValueError(
                "Matrix balanced_task_contract does not match the frozen balanced-v3 reward/reset contract"
            )
    else:
        contract_profile = CONTRACT_PROFILE_SURVIVAL_V2
        expected_task = SURVIVAL_TASK
        if config.get("survival_first_contract") != survival_first_contract_payload():
            raise ValueError(
                "Matrix survival_first_contract does not match the frozen reward/reset curriculum"
            )
    declared_profile = config.get("contract_profile", contract_profile)
    if declared_profile != contract_profile:
        raise ValueError(
            "Matrix contract_profile disagrees with its one selected task contract"
        )
    if config.get("task") != expected_task:
        raise ValueError(
            f"{contract_profile} Crazyflie training task must be {expected_task}"
        )
    if tuple(config.get("controllers", ())) != CONTROLLERS:
        raise ValueError(f"controllers must be exactly {list(CONTROLLERS)} in the declared order")
    seeds = config.get("seeds")
    expected_seeds = [0] if config["label"] == "integration" else [0, 1, 2, 3, 4]
    if seeds != expected_seeds:
        raise ValueError(f"{config['label']} seeds must be exactly {expected_seeds}")
    total = config.get("total_interactions")
    if type(total) is not int or total <= 0:
        raise ValueError("total_interactions must be a positive integer")
    expected_total = 50 if config["label"] == "integration" else 5_000_000
    if total != expected_total:
        raise ValueError(
            f"{config['label']} jobs must contain exactly {expected_total:,} interactions"
        )
    if config.get("rollout_rng_contract") != rollout_rng_contract():
        raise ValueError("Matrix rollout_rng_contract does not match the frozen training RNG contract")
    if config.get("stabilization_and_residual_contract") != stabilization_contract_payload():
        raise ValueError(
            "Matrix stabilization_and_residual_contract does not match the frozen "
            "shared controller contract"
        )
    training = config.get("training")
    evaluation = config.get("evaluation")
    if not isinstance(training, dict) or not isinstance(evaluation, dict):
        raise ValueError("training and evaluation must be objects")
    for field in ("num_envs", "horizon", "ppo_epochs", "microbatch_size", "checkpoint_every_updates"):
        if type(training.get(field)) is not int or training[field] < 1:
            raise ValueError(f"training.{field} must be a positive integer")
    expected_resources = (
        {"num_envs": 1, "horizon": 25, "ppo_epochs": 1, "microbatch_size": 1}
        if config["label"] == "integration"
        else {"num_envs": 4, "horizon": 100, "ppo_epochs": 2, "microbatch_size": 4}
    )
    for field, expected_value in expected_resources.items():
        if training[field] != expected_value:
            raise ValueError(
                f"{config['label']} training.{field} must be exactly {expected_value}"
            )
    rollout_interactions = training["num_envs"] * training["horizon"]
    if total % rollout_interactions:
        raise ValueError(
            "total_interactions must be divisible by num_envs*horizon so the declared budget is exact"
        )
    if training.get("num_workers") != 0 or training.get("precision") != "float32":
        raise ValueError("Version 1 requires num_workers=0 and float32")
    if training["microbatch_size"] != training["num_envs"]:
        raise ValueError(
            "The validated recurrent PPO path uses the full vector sequence batch, so "
            "training.microbatch_size must equal training.num_envs"
        )
    positive_float_fields = (
        "learning_rate", "clip_ratio", "value_coefficient", "max_grad_norm", "target_kl"
    )
    nonnegative_float_fields = ("entropy_coefficient",)
    unit_interval_fields = ("gamma", "gae_lambda")
    for field in positive_float_fields:
        value = training.get(field)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{field} must be positive and finite")
    expected_learning_rate = 1.0e-4 if config["label"] == "integration" else 3.0e-5
    if training["learning_rate"] != expected_learning_rate:
        raise ValueError(
            f"{config['label']} training.learning_rate must be exactly "
            f"{expected_learning_rate}"
        )
    if training["clip_ratio"] >= 1:
        raise ValueError("training.clip_ratio must be below 1")
    for field in nonnegative_float_fields:
        value = training.get(field)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{field} must be non-negative and finite")
    for field in unit_interval_fields:
        value = training.get(field)
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"training.{field} must be in (0, 1]")
    scenarios = evaluation.get("scenarios")
    if tuple(scenarios or ()) != SCENARIOS:
        raise ValueError(f"evaluation.scenarios must be exactly {list(SCENARIOS)}")
    protocol = load_protocol(str(evaluation.get("protocol")))
    configured_evaluation_manifest = evaluation.get("manifest")
    if configured_evaluation_manifest is not None:
        if not isinstance(configured_evaluation_manifest, str) or not configured_evaluation_manifest.strip():
            raise ValueError("evaluation.manifest must name a JSON artifact when configured")
        evaluation_manifest_path = (ROOT / configured_evaluation_manifest).resolve()
        try:
            configured_protocol = validate_manifest(
                _read_json(evaluation_manifest_path),
                expected_episodes=protocol["episodes_per_scenario"],
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Configured evaluation manifest is invalid: {exc}") from exc
        if canonical_sha256(configured_protocol) != canonical_sha256(protocol):
            raise ValueError("Configured evaluation manifest differs from the selected protocol")
    else:
        evaluation_manifest_path = None
    episodes = evaluation.get("episodes_per_scenario")
    expected_episodes = 2 if config["label"] == "integration" else 16
    if episodes != expected_episodes or protocol["episodes_per_scenario"] != expected_episodes:
        raise ValueError(f"{config['label']} needs {expected_episodes} episodes per scenario")
    if evaluation.get("seed") != 101 or not evaluation.get("deterministic_actions"):
        raise ValueError("Evaluation must use seed 101 and deterministic actions")
    expected_policy_batch = min(4, expected_episodes)
    if evaluation.get("policy_inference_device") != POLICY_INFERENCE_DEVICE:
        raise ValueError("Version 1 evaluation policy inference device must be cuda:0")
    if evaluation.get("policy_inference_backend") != POLICY_INFERENCE_BACKEND:
        raise ValueError("Version 1 evaluation requires the CUDA-Graph action-only backend")
    if evaluation.get("policy_inference_precision") != POLICY_INFERENCE_PRECISION:
        raise ValueError("Version 1 evaluation policy inference must use float32")
    if evaluation.get("policy_inference_graph") != _policy_graph_config(expected_policy_batch):
        raise ValueError("Version 1 evaluation CUDA-Graph contract changed")
    if evaluation.get("policy_bridge") != POLICY_BRIDGE_CONTRACT:
        raise ValueError("Version 1 evaluation zero-host-transfer contract changed")
    connectome = (ROOT / config.get("connectome_manifest", "")).resolve()
    if not connectome.is_file():
        raise ValueError(f"Connectome manifest does not exist: {connectome}")
    rewire_setting = config.get("rewire_manifest")
    if not isinstance(rewire_setting, str) or not rewire_setting.strip():
        raise ValueError("rewire_manifest must name the frozen JSON artifact")
    rewire = (ROOT / rewire_setting).resolve()
    if not rewire.is_file():
        raise ValueError(f"Frozen rewire manifest does not exist: {rewire}")
    declared_rewire_sha256 = config.get("rewire_manifest_sha256")
    actual_rewire_sha256 = sha256_file(rewire)
    if declared_rewire_sha256 != actual_rewire_sha256:
        raise ValueError(
            "Frozen rewire file checksum mismatch: "
            f"expected={declared_rewire_sha256!r}, actual={actual_rewire_sha256!r}"
        )
    rewire_seed = config.get("rewire_seed")
    if type(rewire_seed) is not int or rewire_seed < 0:
        raise ValueError("rewire_seed must be a non-negative integer")
    try:
        # This validates the JSON object checksum, source-connectome identity,
        # exact graph tensors, configured seed, and every declared graph
        # invariant.  Merely matching the file hash is not sufficient.
        from g1_fly_control.connectome import load_connectome
        from g1_fly_control.crazyflie.controllers import load_rewire_manifest

        circuit = load_connectome(connectome)
        _, _, rewire_object = load_rewire_manifest(
            rewire,
            circuit,
            expected_seed=rewire_seed,
            expected_file_sha256=declared_rewire_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Frozen rewire manifest failed content validation: {exc}") from exc
    config["_config_path"] = str(path)
    config["_contract_profile"] = contract_profile
    config["_connectome_path"] = str(connectome)
    config["_rewire_manifest_path"] = str(rewire)
    config["_rewire_manifest_sha256"] = actual_rewire_sha256
    config["_rewire_manifest"] = rewire_object
    config["_evaluation_manifest"] = protocol
    config["_evaluation_manifest_path"] = (
        str(evaluation_manifest_path) if evaluation_manifest_path is not None else None
    )
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def resolved_job_config(
    config: dict[str, Any], controller: str, seed: int
) -> dict[str, Any]:
    """Build the exact per-cell config shared by dry-run and train entrypoints."""

    resolved = {
        "matrix": public_config(config),
        "controller": controller,
        "seed": seed,
        "task": config["task"],
        "contract_profile": config["_contract_profile"],
        "memory_acceptance": memory_acceptance_contract_payload(),
    }
    if config["task"] == BALANCED_TASK:
        resolved["mixed_scenario_contract"] = mixed_scenario_contract_payload(
            seed=seed
        )
        resolved["switch_target_curriculum"] = (
            balanced_switch_target_curriculum_payload()
        )
    elif config["task"] == "FlyCrazyflie-WaypointSwitch-v0":
        if config["_contract_profile"] == CONTRACT_PROFILE_BALANCED_V4:
            resolved["switch_target_curriculum"] = (
                balanced_v4_switch_target_curriculum_payload()
            )
        elif config["_contract_profile"] == CONTRACT_PROFILE_BALANCED_V3:
            resolved["switch_target_curriculum"] = (
                balanced_switch_target_curriculum_payload()
            )
        else:
            resolved["switch_target_curriculum"] = (
                switch_target_curriculum_payload()
            )
    return resolved


def job_fingerprint(config: dict[str, Any], controller: str, seed: int) -> tuple[str, dict[str, Any]]:
    resolved = resolved_job_config(config, controller, seed)
    return reproduction_fingerprint(
        resolved_config=resolved,
        evaluation_manifest=config["_evaluation_manifest"],
        connectome_manifest=config["_connectome_path"],
        rewired_manifest=config["_rewire_manifest"],
    )


def _training_command(
    config: dict[str, Any], controller: str, seed: int, run_dir: Path, fingerprint: str, pause_file: Path
) -> list[str]:
    training = config["training"]
    command = [
        sys.executable, str(ROOT / "scripts" / "drone_train.py"),
        "--task", config["task"],
        "--contract_profile", config["_contract_profile"],
        "--policy", controller,
        "--seed", str(seed),
        "--num_envs", str(training["num_envs"]),
        "--total_interactions", str(config["total_interactions"]),
        "--horizon", str(training["horizon"]),
        "--ppo_epochs", str(training["ppo_epochs"]),
        "--microbatch_size", str(training["microbatch_size"]),
        "--learning_rate", str(training["learning_rate"]),
        "--gamma", str(training["gamma"]),
        "--gae_lambda", str(training["gae_lambda"]),
        "--clip_ratio", str(training["clip_ratio"]),
        "--value_coefficient", str(training["value_coefficient"]),
        "--entropy_coefficient", str(training["entropy_coefficient"]),
        "--max_grad_norm", str(training["max_grad_norm"]),
        "--target_kl", str(training["target_kl"]),
        "--checkpoint_every_updates", str(training["checkpoint_every_updates"]),
        "--connectome_manifest", config["_connectome_path"],
        "--rewire_seed", str(config["rewire_seed"]),
        "--rewire_manifest", config["_rewire_manifest_path"],
        "--run_dir", str(run_dir),
        "--matrix_config", config["_config_path"],
        "--expected_fingerprint", fingerprint,
        "--pause_file", str(pause_file),
    ]
    if config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_CELL_LAYOUT:
        command.extend(("--device", POLICY_INFERENCE_DEVICE))
    command.append("--headless")
    return command


def _evaluation_command(
    config: dict[str, Any], controller: str, seed: int, checkpoint: Path,
    scenario: str, output: Path, fingerprint: str,
) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / "drone_evaluate.py"),
        "--checkpoint", str(checkpoint),
        "--protocol", config["evaluation"]["protocol"],
        "--scenario", scenario,
        "--headless",
        "--device", config["evaluation"]["policy_inference_device"],
        "--output", str(output),
        "--expected_fingerprint", fingerprint,
        "--training_seed", str(seed),
        "--policy", controller,
    ]


def build_queue(config: dict[str, Any], output: Path) -> dict[str, Any]:
    artifact_root = output.parent / output.stem
    pause_file = artifact_root / "pause.request"
    jobs: list[dict[str, Any]] = []
    evaluation_bundle_count = 0
    if config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT:
        # The user explicitly prioritized both LIF conditions.  Keep all three
        # original-LIF tasks first, then rewired LIF, GRU, and MLP.
        matrix_cells = (
            (task, controller, seed, config["_task_configs"][task])
            for controller in CONTROLLERS
            for task in config["tasks"]
            for seed in config["seeds"]
        )
    elif config.get("_matrix_layout") == TASK_SEPARATED_LAYOUT:
        matrix_cells = (
            (task, controller, seed, config["_task_configs"][task])
            for task in config["tasks"]
            for controller in CONTROLLERS
            for seed in config["seeds"]
        )
    else:
        matrix_cells = (
            (config["task"], controller, seed, config)
            for controller in CONTROLLERS
            for seed in config["seeds"]
        )
    for task, controller, seed, cell_config in matrix_cells:
        identifier = (
            f"{task}__{controller}__seed-{seed}"
            if config.get("_matrix_layout")
            in {TASK_SEPARATED_LAYOUT, TASK_SEPARATED_COMPARISON_LAYOUT}
            else f"{controller}__seed-{seed}"
        )
        run_dir = artifact_root / "jobs" / identifier
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        fingerprint, payload = job_fingerprint(cell_config, controller, seed)
        evaluations: list[dict[str, Any]] = []
        evaluation_scenarios = (
            (task,)
            if config.get("_matrix_layout")
            in {TASK_SEPARATED_LAYOUT, TASK_SEPARATED_COMPARISON_LAYOUT}
            else SCENARIOS
        )
        for scenario in evaluation_scenarios:
            evaluation_bundle_count += 1
            evaluation_output = (
                artifact_root / "evaluations" / identifier / f"{scenario}.json"
            )
            plans = cell_config["_evaluation_manifest"]["scenarios"][scenario]
            evaluations.append({
                "scenario": scenario,
                "status": "pending",
                "output": str(evaluation_output),
                "protocol": cell_config["evaluation"]["protocol"],
                "protocol_label": cell_config["_evaluation_manifest"]["label"],
                "evaluation_manifest_id": cell_config["_evaluation_manifest"]["manifest_id"],
                "evaluation_manifest_sha256": canonical_sha256(
                    cell_config["_evaluation_manifest"]
                ),
                "evaluation_seed": cell_config["_evaluation_manifest"]["evaluation_seed"],
                "expected_episode_ids": [plan["episode_id"] for plan in plans],
                "expected_plan_sha256": [plan["plan_sha256"] for plan in plans],
                "expected_plan_object_sha256": [
                    canonical_sha256(plan) for plan in plans
                ],
                "command": _evaluation_command(
                    cell_config,
                    controller,
                    seed,
                    checkpoint,
                    scenario,
                    evaluation_output,
                    fingerprint,
                ),
            })
        jobs.append({
            "id": identifier,
            "controller": controller,
            "seed": seed,
            "task": task,
            "contract_profile": cell_config["_contract_profile"],
            "status": "pending",
            "total_interactions": cell_config["total_interactions"],
            "training_resources": {
                "num_envs": cell_config["training"]["num_envs"],
                "horizon": cell_config["training"]["horizon"],
                "ppo_epochs": cell_config["training"]["ppo_epochs"],
                "microbatch_size": cell_config["training"]["microbatch_size"],
                "num_workers": cell_config["training"]["num_workers"],
                "precision": cell_config["training"]["precision"],
            },
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "expected_fingerprint": fingerprint,
            "evaluation_manifest_id": cell_config["_evaluation_manifest"]["manifest_id"],
            "connectome_manifest_path": cell_config["_connectome_path"],
            "rewire_seed": cell_config["rewire_seed"],
            "rewire_manifest_path": cell_config["_rewire_manifest_path"],
            "rewire_manifest_sha256": cell_config["_rewire_manifest_sha256"],
            "fingerprint_payload": payload,
            "training_command": _training_command(
                cell_config, controller, seed, run_dir, fingerprint, pause_file
            ),
            "evaluations": evaluations,
        })
    task_count = (
        len(config["tasks"])
        if config.get("_matrix_layout")
        in {TASK_SEPARATED_LAYOUT, TASK_SEPARATED_COMPARISON_LAYOUT}
        else 1
    )
    expected_jobs = task_count * len(CONTROLLERS) * len(config["seeds"])
    if len(jobs) != expected_jobs or len({job["id"] for job in jobs}) != expected_jobs:
        raise RuntimeError("Duplicate or missing task/controller/seed matrix cell")
    output_paths = [job["run_dir"] for job in jobs]
    output_paths.extend(bundle["output"] for job in jobs for bundle in job["evaluations"])
    if len(output_paths) != len(set(output_paths)):
        raise RuntimeError("Duplicate matrix output path")
    episodes = evaluation_bundle_count * config["evaluation"]["episodes_per_scenario"]
    comparison_layout = (
        config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT
    )
    concurrency_decision = (
        comparison_concurrency_decision(config)
        if comparison_layout
        else {
            "status": "fixed_sequential",
            "requested_max_concurrent_isaac_processes": 1,
            "effective_max_concurrent_isaac_processes": 1,
            "fallback_applied": False,
            "fallback_reason": None,
        }
    )
    effective_concurrency = concurrency_decision[
        "effective_max_concurrent_isaac_processes"
    ]
    resource_limits = {
        "policy_version": MEMORY_POLICY_VERSION,
        "max_concurrent_isaac_processes": effective_concurrency,
        "device_gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "device_gpu_measurement": "maximum sampled nvidia-smi usage; telemetry required for CUDA",
        "gpu_telemetry_required_for_cuda": True,
        "finite_numeric_telemetry_required": True,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "rss_growth_tolerance_mib": RSS_GROWTH_TOLERANCE_MIB,
        "rss_growth_disposition": "warning_only",
        "swap_out_growth_tolerance_mib": SWAP_OUT_GROWTH_TOLERANCE_MIB,
        "sustained_paging_disposition": "hard_failure",
        "steady_state_window_samples": 4,
    }
    if config.get("_matrix_layout") == TASK_SEPARATED_LAYOUT:
        resource_limits["gpu_compute_utilization_target_percent"] = {
            "minimum_inclusive": 50.0,
            "maximum_inclusive": 90.0,
            "above_maximum_disposition": "scale_down_and_retest_before_long_run",
        }
    elif comparison_layout:
        resource_limits["gpu_compute_utilization_policy"] = {
            "maximum_inclusive": 100.0,
            "disposition": "telemetry_only_not_an_acceptance_gate",
        }
    queue = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "label": config["label"],
        "status": "dry_run",
        "config_path": config["_config_path"],
        "config_sha256": canonical_sha256(public_config(config)),
        "output": str(output),
        "artifact_root": str(artifact_root),
        "pause_file": str(pause_file),
        "resource_limits": resource_limits,
        "job_count": len(jobs),
        "evaluation_bundle_count": evaluation_bundle_count,
        "predicted_evaluation_episodes": episodes,
        "jobs": jobs,
        "config": public_config(config),
    }
    if comparison_layout:
        queue.update(
            {
                "requested_max_concurrent_isaac_processes": 2,
                "max_concurrent_isaac_processes": effective_concurrency,
                "concurrency_decision": concurrency_decision,
            }
        )
    else:
        queue["sequential_process_limit"] = 1
    if config.get("_matrix_layout") == TASK_SEPARATED_LAYOUT:
        expected_shape = (
            (12, 12, 24) if config["label"] == "integration" else (60, 60, 960)
        )
        if (len(jobs), evaluation_bundle_count, episodes) != expected_shape:
            raise RuntimeError(
                f"Task-separated {config['label']} matrix must resolve to "
                f"{expected_shape[0]} jobs, {expected_shape[1]} matched-task bundles, "
                f"and {expected_shape[2]} episodes"
            )
    elif config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT:
        expected_shape = (12, 12, 192)
        if (len(jobs), evaluation_bundle_count, episodes) != expected_shape:
            raise RuntimeError(
                "Task-separated seed-0 comparison must resolve to 12 jobs, "
                "12 matched-task bundles, and 192 episodes"
            )
    elif config["label"] == "main":
        if len(jobs) != 20 or evaluation_bundle_count != 60 or episodes != 960:
            raise RuntimeError("Main matrix must resolve to 20 jobs, 60 bundles, and 960 episodes")
    return queue


def validate_resume_queue(queue: dict[str, Any], config: dict[str, Any], output: Path) -> None:
    """Reject a queue whose immutable commands or fingerprints are stale."""

    fresh = build_queue(config, output)
    queue_fields = (
        "schema_version",
        "label",
        "config_path",
        "config_sha256",
        "output",
        "artifact_root",
        "pause_file",
        "resource_limits",
        "job_count",
        "evaluation_bundle_count",
        "predicted_evaluation_episodes",
        "config",
    )
    if config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT:
        queue_fields += (
            "requested_max_concurrent_isaac_processes",
            "max_concurrent_isaac_processes",
            "concurrency_decision",
        )
    else:
        queue_fields += ("sequential_process_limit",)
    if any(queue.get(field) != fresh[field] for field in queue_fields):
        raise ValueError("Resume queue metadata differs from the current resolved matrix")
    current_jobs = queue.get("jobs")
    if not isinstance(current_jobs, list) or len(current_jobs) != len(fresh["jobs"]):
        raise ValueError("Resume queue has missing or duplicate controller/seed cells")
    job_fields = (
        "id",
        "controller",
        "seed",
        "task",
        "contract_profile",
        "total_interactions",
        "training_resources",
        "run_dir",
        "checkpoint",
        "expected_fingerprint",
        "evaluation_manifest_id",
        "connectome_manifest_path",
        "rewire_seed",
        "rewire_manifest_path",
        "rewire_manifest_sha256",
        "fingerprint_payload",
        "training_command",
    )
    bundle_fields = (
        "scenario",
        "output",
        "protocol",
        "protocol_label",
        "evaluation_manifest_id",
        "evaluation_manifest_sha256",
        "evaluation_seed",
        "expected_episode_ids",
        "expected_plan_sha256",
        "expected_plan_object_sha256",
        "command",
    )
    for current, expected_job in zip(current_jobs, fresh["jobs"], strict=True):
        if not isinstance(current, dict) or any(
            current.get(field) != expected_job[field] for field in job_fields
        ):
            raise ValueError("Resume queue job command or reproduction fingerprint is stale")
        current_bundles = current.get("evaluations")
        expected_bundles = expected_job["evaluations"]
        if not isinstance(current_bundles, list) or len(current_bundles) != len(expected_bundles):
            raise ValueError("Resume queue evaluation bundle set is incomplete")
        for current_bundle, expected_bundle in zip(current_bundles, expected_bundles, strict=True):
            if not isinstance(current_bundle, dict) or any(
                current_bundle.get(field) != expected_bundle[field] for field in bundle_fields
            ):
                raise ValueError("Resume queue evaluation command or protocol identity is stale")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_pause_request(queue: dict[str, Any], queue_path: Path) -> tuple[Path, dict[str, Any], bool]:
    """Persist an idempotent pause request without modifying the live queue."""

    try:
        pause_file = Path(queue["pause_file"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Queue does not contain a valid pause_file") from exc
    pause_file.parent.mkdir(parents=True, exist_ok=True)
    if pause_file.exists():
        try:
            existing = _read_json(pause_file)
        except (OSError, ValueError):
            existing = {
                "schema_version": 1,
                "status": "requested",
                "legacy_or_malformed_request_sha256": sha256_file(pause_file),
            }
        return pause_file, existing, False
    request = {
        "schema_version": 1,
        "status": "requested",
        "requested_utc": _utc_now(),
        "requesting_pid": os.getpid(),
        "queue": str(queue_path.resolve()),
        "config_sha256": queue.get("config_sha256"),
    }
    _atomic_json(pause_file, request)
    return pause_file, request, True


def consume_pause_request(queue: dict[str, Any], queue_path: Path) -> dict[str, Any]:
    """Clear and archive a pause request while recording an explicit resume."""

    pause_file = Path(queue["pause_file"])
    event: dict[str, Any] = {
        "resumed_utc": _utc_now(),
        "resuming_pid": os.getpid(),
        "queue": str(queue_path.resolve()),
        "pause_request_consumed": False,
        "pause_file": str(pause_file),
    }
    if pause_file.exists():
        request_sha256 = sha256_file(pause_file)
        try:
            request_payload: dict[str, Any] | None = _read_json(pause_file)
        except (OSError, ValueError):
            request_payload = None
        archive_dir = pause_file.parent / "pause_requests"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"pause-request-consumed-{time.time_ns()}.json"
        os.replace(pause_file, archive)
        directory_fd = os.open(archive_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        event.update({
            "pause_request_consumed": True,
            "request_sha256": request_sha256,
            "request_payload": request_payload,
            "request_archive": str(archive),
        })
    resumed_jobs = []
    for job in queue["jobs"]:
        if job.get("status") == "paused":
            resumed_jobs.append(job["id"])
            job["status"] = "pending"
            if "pause_reason" in job:
                job["previous_pause_reason"] = job.pop("pause_reason")
    event["resumed_jobs"] = resumed_jobs
    queue.setdefault("resume_history", []).append(event)
    queue["last_resume"] = event
    return event


def _refresh_status(queue: dict[str, Any]) -> None:
    counts = Counter(job["status"] for job in queue["jobs"])
    invalid = set(counts) - VALID_STATUS
    if invalid:
        raise RuntimeError(f"Unknown job states: {sorted(invalid)}")
    queue["counts"] = dict(sorted(counts.items()))
    if counts.get("running"):
        queue["status"] = "running"
    elif counts.get("failed"):
        queue["status"] = "failed"
    elif counts.get("paused"):
        queue["status"] = "paused"
    elif counts.get("completed") == len(queue["jobs"]):
        queue["status"] = "completed"
    elif queue.get("dry_run", False):
        queue["status"] = "dry_run"
    else:
        queue["status"] = "pending"


def save_queue(path: Path, queue: dict[str, Any]) -> None:
    _refresh_status(queue)
    queue["updated_utc"] = _utc_now()
    _atomic_json(path, queue)
    summary = {
        "schema_version": queue["schema_version"],
        "label": queue["label"],
        "status": queue["status"],
        "counts": queue["counts"],
        "job_count": queue["job_count"],
        "evaluation_bundle_count": queue["evaluation_bundle_count"],
        "predicted_evaluation_episodes": queue["predicted_evaluation_episodes"],
        "max_concurrent_isaac_processes": queue.get(
            "max_concurrent_isaac_processes",
            queue.get("sequential_process_limit", 1),
        ),
        "concurrency_decision": queue.get("concurrency_decision"),
        "jobs": [{
            "id": job["id"], "controller": job["controller"], "seed": job["seed"],
            "status": job["status"], "checkpoint": job["checkpoint"],
            "evaluations": {item["scenario"]: item["status"] for item in job["evaluations"]},
        } for job in queue["jobs"]],
    }
    _atomic_json(path.with_name(path.stem + "_summary.json"), summary)


@contextmanager
def queue_lock(path: Path) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"Queue lock exists; another runner may be active: {lock}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} utc={_utc_now()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


_ACTIVE_CHILDREN_LOCK = threading.Lock()
_ACTIVE_CHILDREN: set[subprocess.Popen[str]] = set()


def _terminate_active_children() -> None:
    """Terminate every child process group owned by this matrix parent."""

    with _ACTIVE_CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN)
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30.0
    for process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _run(command: list[str], log: Path) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log = log.with_name(f"{log.stem}__attempt-{time.time_ns()}{log.suffix}")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        with _ACTIVE_CHILDREN_LOCK:
            _ACTIVE_CHILDREN.add(process)
        try:
            exit_code = process.wait()
        except BaseException:
            # The child owns a separate process group.  Never release queue or
            # machine locks while an interrupted Isaac subprocess can remain
            # alive and collide with a resume.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise
        finally:
            with _ACTIVE_CHILDREN_LOCK:
                _ACTIVE_CHILDREN.discard(process)
    return {"exit_code": exit_code, "wall_time_s": time.monotonic() - started, "log": str(log)}


@lru_cache(maxsize=16)
def _cached_controller_report(
    controller: str,
    connectome_manifest_path: str,
    rewire_seed: int,
    rewire_manifest_path: str,
) -> dict[str, Any]:
    """Reconstruct the immutable controller report without starting Isaac."""

    from g1_fly_control.crazyflie.controllers import build_controller

    _, report = build_controller(
        controller,
        observation_dim=12,
        action_dim=4,
        device="cpu",
        connectome_manifest=connectome_manifest_path,
        rewire_seed=rewire_seed,
        rewire_manifest_path=(
            rewire_manifest_path
            if controller == "frozen_lif_degree_rewired"
            else None
        ),
    )
    return report


def _expected_controller_report(job: dict[str, Any]) -> dict[str, Any]:
    return _cached_controller_report(
        str(job["controller"]),
        str(Path(job["connectome_manifest_path"]).resolve()),
        int(job["rewire_seed"]),
        str(Path(job["rewire_manifest_path"]).resolve()),
    )


def _read_training_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Validate checkpoint structure without materializing external history."""

    from g1_fly_control.crazyflie.checkpoint import read_checkpoint

    return read_checkpoint(
        checkpoint, map_location="cpu", resolve_external_history=False
    )


def _valid_memory_gate(
    samples: Any, gate: Any, *, required_stages: set[str]
) -> bool:
    if not isinstance(samples, list) or not samples or not all(
        isinstance(sample, dict) for sample in samples
    ):
        return False
    stages = {sample.get("stage") for sample in samples}
    if not required_stages.issubset(stages):
        return False
    try:
        recomputed = assess_memory(samples)
        max_gpu = float(recomputed["max_device_gpu_used_mib"])
        max_ram = float(recomputed["max_system_ram_percent"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    limits = recomputed.get("limits")
    warnings = recomputed.get("warnings")
    failures = recomputed.get("failures")
    return (
        isinstance(gate, dict)
        and canonical_sha256(gate) == canonical_sha256(recomputed)
        and recomputed.get("policy_version") == MEMORY_POLICY_VERSION
        and recomputed.get("passed") is True
        and failures == []
        and isinstance(warnings, list)
        and all(isinstance(message, str) and message for message in warnings)
        and recomputed.get("device_gpu_telemetry_complete") is True
        and recomputed.get("sustained_paging_detected") is False
        and math.isfinite(max_gpu)
        and math.isfinite(max_ram)
        and max_gpu < GPU_LIMIT_MIB
        and max_ram < RAM_LIMIT_PERCENT
        and isinstance(limits, dict)
        and limits.get("system_ram_percent_exclusive") == RAM_LIMIT_PERCENT
        and limits.get("gpu_used_mib_exclusive") == GPU_LIMIT_MIB
        and limits.get("rss_growth_disposition") == "warning_only"
        and limits.get("gpu_telemetry_required_for_cuda") is True
    )


def _valid_training_memory(samples: Any, gate: Any) -> bool:
    return _valid_memory_gate(
        samples,
        gate,
        required_stages={
            "environment_loaded",
            "controller_loaded",
            "rollout",
            "optimizer_update",
            "training_complete",
        },
    )


def _valid_training(job: dict[str, Any]) -> bool:
    checkpoint = Path(job["checkpoint"])
    manifest = Path(job["run_dir"]) / "training_manifest.json"
    try:
        metadata = _read_json(manifest)
        checkpoint_sha256_before = sha256_file(checkpoint)
        checkpoint_payload = _read_training_checkpoint(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_sha256_before != checkpoint_sha256:
            return False
        recorded_checkpoint = Path(metadata["checkpoint"]).resolve()
        expected_report = _expected_controller_report(job)
        checkpoint_metadata = checkpoint_payload["metadata"]
        checkpoint_fingerprints = checkpoint_payload["fingerprints"]
        if not isinstance(checkpoint_metadata, dict) or not isinstance(
            checkpoint_fingerprints, dict
        ):
            return False
        expected_core = expected_report["core_checksum"]
    # Checkpoints are external binary artifacts.  Any deserialization or
    # closed-schema validation failure rejects the artifact; queue validation
    # must never crash on corrupt bytes.
    except Exception:
        return False
    checkpoint_history_valid = _valid_checkpoint_history(
        checkpoint, metadata, job, payload=checkpoint_payload
    )
    expected_updates = job["total_interactions"] // (
        job["training_resources"]["num_envs"] * job["training_resources"]["horizon"]
    )
    return (
        checkpoint.is_file()
        and metadata.get("status") == "completed"
        and metadata.get("task") == job["task"]
        and metadata.get("contract_profile") == job["contract_profile"]
        and metadata.get("controller") == job["controller"]
        and metadata.get("seed") == job["seed"]
        and metadata.get("requested_interactions") == job["total_interactions"]
        and metadata.get("environment_interactions") == job["total_interactions"]
        and metadata.get("completed_updates") == expected_updates
        and metadata.get("fingerprint") == job["expected_fingerprint"]
        and isinstance(metadata.get("fingerprint_payload"), dict)
        and metadata["fingerprint_payload"] == job["fingerprint_payload"]
        and canonical_sha256(metadata["fingerprint_payload"]) == metadata.get("fingerprint")
        and metadata.get("evaluation_manifest_id") == job["evaluation_manifest_id"]
        and metadata.get("controller_report") == expected_report
        and checkpoint_metadata.get("controller_report") == expected_report
        and metadata.get("core_checksum_before") == expected_core
        and metadata.get("core_checksum_after") == expected_core
        and checkpoint_payload.get("core_checksum") == expected_core
        and checkpoint_metadata.get("core_checksum_before") == expected_core
        and checkpoint_metadata.get("core_checksum_after") == expected_core
        and checkpoint_fingerprints.get("frozen_core") == expected_core
        and _valid_training_memory(
            metadata.get("memory_samples"), metadata.get("memory_gate")
        )
        and recorded_checkpoint == checkpoint.resolve()
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and checkpoint_history_valid
    )


def _history_row_has_no_nonfinite_failure(row: Any) -> bool:
    """Reject any observed task failure-cause code 4 in completed training."""

    if not isinstance(row, dict):
        return False
    counts = row.get("failure_cause_counts")
    if not isinstance(counts, dict):
        return False
    expected_codes = {"1", "2", "3", "4"}
    if set(counts) != expected_codes:
        return False
    if any(type(counts[code]) is not int or counts[code] < 0 for code in expected_codes):
        return False
    return counts["4"] == 0


def _external_history_has_no_nonfinite_failure(
    checkpoint: Path, reference: dict[str, Any]
) -> bool:
    """Stream authenticated JSONL a second time to inspect failure code 4."""

    run_root = checkpoint.parent.parent.resolve()
    for segment in reference.get("segments", []):
        if not isinstance(segment, dict) or not isinstance(segment.get("path"), str):
            return False
        path = (checkpoint.parent / segment["path"]).resolve()
        if not path.is_relative_to(run_root):
            return False
        digest = sha256()
        byte_count = 0
        try:
            with path.open("rb") as stream:
                for raw_line in stream:
                    byte_count += len(raw_line)
                    digest.update(raw_line)
                    if not raw_line.endswith(b"\n"):
                        return False
                    row = json.loads(raw_line)
                    if not _history_row_has_no_nonfinite_failure(row):
                        return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            byte_count != segment.get("byte_count")
            or digest.hexdigest() != segment.get("sha256")
        ):
            return False
    return True


def _valid_checkpoint_history(
    checkpoint: Path,
    metadata: dict[str, Any],
    job: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Verify inline history or stream every immutable segment fail-closed."""

    try:
        if payload is None:
            payload = _read_training_checkpoint(checkpoint)
        counters = payload["counters"]
        history = payload["history"]
        reference = payload["history_reference"]
        expected_updates = metadata["completed_updates"]
        expected_interactions_per_update = (
            job["training_resources"]["num_envs"] * job["training_resources"]["horizon"]
        )
        common_valid = (
            payload["fingerprints"].get("reproduction") == job["expected_fingerprint"]
            and counters["completed_updates"] == expected_updates
            and counters["total_interactions"] == job["total_interactions"]
            and payload["interactions_per_update"] == expected_interactions_per_update
            and isinstance(history, list)
            and reference == metadata.get("history_reference")
        )
        if not common_valid:
            return False
        if reference is None:
            # Inline checkpoints remain structurally supported.  Current
            # fingerprints additionally require explicit per-code task failure
            # evidence on every row.
            return (
                len(history) == expected_updates
                and all(_history_row_has_no_nonfinite_failure(row) for row in history)
            )

        # Compact checkpoints store no embedded rows.  Rebuild the exact
        # cursor while streaming one canonical JSONL row at a time; comparing
        # the complete cursor authenticates its aggregate history digest,
        # segment digests, byte counts, row bounds, and final counters.
        if history:
            return False
        from g1_fly_control.crazyflie.checkpoint import (
            build_history_reference_from_segments,
        )

        streamed_reference = build_history_reference_from_segments(
            reference["segments"],
            checkpoint_path=checkpoint,
            interactions_per_update=expected_interactions_per_update,
        )
        return (
            streamed_reference == reference
            and _external_history_has_no_nonfinite_failure(checkpoint, reference)
        )
    except Exception:
        return False


def _training_artifact_status(job: dict[str, Any]) -> str | None:
    """Read the atomic training status even if Isaac masks a process code."""

    try:
        status = _read_json(Path(job["run_dir"]) / "training_manifest.json").get("status")
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return status if isinstance(status, str) else None


def _consistent_evaluation_summary(
    summary: Any,
    episodes: list[dict[str, Any]],
    *,
    scenario: str,
    expected: int,
) -> bool:
    """Recompute the complete metrics object from episode rows.

    Evaluation rows carry three provenance-only fields (the held-out plan and
    its checksums) in addition to the strict ``EpisodeSummary`` schema.  The
    recomputation deliberately strips only those fields and then requires all
    aggregate fields to match exactly after canonical JSON serialization.
    """

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
        and summary.get("expected_episode_count") == expected
        and summary.get("episode_count") == expected
        and summary.get("n_episodes") == expected
        and summary.get("complete") is True
        and canonical_sha256(actual_aggregates) == canonical_sha256(recomputed_aggregates)
    )


def _valid_evaluation_memory(samples: Any, gate: Any) -> bool:
    return _valid_memory_gate(
        samples,
        gate,
        required_stages={
            "environment_loaded",
            "graph_captured",
            "steady_state",
            "evaluation_end",
        },
    )


def _valid_evaluation(bundle: dict[str, Any], job: dict[str, Any], expected: int) -> bool:
    try:
        value = _read_json(Path(bundle["output"]))
        checkpoint = Path(job["checkpoint"])
        checkpoint_sha256 = sha256_file(checkpoint)
        recorded_checkpoint = Path(value["checkpoint"]).resolve()
        episodes = value["episodes"]
        if not isinstance(episodes, list) or not all(isinstance(row, dict) for row in episodes):
            return False
        episode_ids = [row.get("episode_id") for row in episodes]
        expected_ids = bundle["expected_episode_ids"]
        if (
            any(type(episode_id) is not int for episode_id in episode_ids)
            or episode_ids != expected_ids
            or len(set(episode_ids)) != expected
        ):
            return False
        expected_plan_hashes = bundle["expected_plan_sha256"]
        expected_plan_object_hashes = bundle["expected_plan_object_sha256"]
        for index, row in enumerate(episodes):
            plan = row.get("plan")
            if (
                row.get("scenario") != bundle["scenario"]
                or row.get("evaluation_seed") != bundle["evaluation_seed"]
                or not isinstance(plan, dict)
                or plan.get("episode_id") != expected_ids[index]
                or row.get("plan_sha256") != expected_plan_hashes[index]
                or plan.get("plan_sha256") != expected_plan_hashes[index]
                or canonical_sha256(plan) != expected_plan_object_hashes[index]
            ):
                return False
            if bundle["scenario"] == SCENARIOS[2]:
                # A summary hash alone cannot prove that the scheduled world-
                # frame disturbance was actually submitted.  Reuse the exact
                # evaluator-side mass/vector/error validator before accepting
                # a Gust bundle as resumable evidence.
                from drone_evaluate import _validated_gust_recovery_events

                _validated_gust_recovery_events(row, plan)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return (
        value.get("status") == "completed"
        and value.get("label") == bundle["protocol_label"]
        and value.get("protocol") == bundle["protocol"]
        and value.get("scenario") == bundle["scenario"]
        and value.get("training_seed") == job["seed"]
        and value.get("controller") == job["controller"]
        and value.get("fingerprint") == job["expected_fingerprint"]
        and isinstance(value.get("fingerprint_payload"), dict)
        and canonical_sha256(value["fingerprint_payload"]) == value.get("fingerprint")
        and value.get("evaluation_seed") == bundle["evaluation_seed"]
        and value.get("evaluation_manifest_id") == bundle["evaluation_manifest_id"]
        and value.get("deterministic_actions") is True
        and value.get("simulation_device") == POLICY_INFERENCE_DEVICE
        and value.get("simulation_device_type") == "cuda"
        and value.get("policy_inference_device") == POLICY_INFERENCE_DEVICE
        and value.get("policy_inference_device_type") == "cuda"
        and value.get("policy_inference_backend") == POLICY_INFERENCE_BACKEND
        and value.get("policy_inference_precision") == POLICY_INFERENCE_PRECISION
        and value.get("policy_inference_graph") == _policy_graph_artifact(min(4, expected))
        and value.get("policy_bridge") == POLICY_BRIDGE_CONTRACT
        and _valid_evaluation_memory(value.get("memory_samples"), value.get("memory_gate"))
        and recorded_checkpoint == checkpoint.resolve()
        and value.get("checkpoint_sha256") == checkpoint_sha256
        and len(episodes) == expected
        and _consistent_evaluation_summary(
            value.get("summary"), episodes, scenario=bundle["scenario"], expected=expected
        )
    )


class _VerifiedMainProof:
    """Opaque in-process evidence that the read-only empirical gate passed."""

    __slots__ = ("report",)

    def __init__(self, report: dict[str, Any]) -> None:
        if (
            not isinstance(report, dict)
            or report.get("status") != "PASS"
            or report.get("gate")
            != "balanced_v3_original_lif_empirical_flight_proof"
            or report.get("environment_interactions") != 500_000
            or report.get("completed_updates") != 1_250
        ):
            raise ValueError("Main empirical proof report is not the exact passing gate")
        self.report = report


def _verify_main_lif_proof(
    config: dict[str, Any], config_path: Path
) -> _VerifiedMainProof:
    """Run the CPU/read-only proof verifier without a module import cycle."""

    # drone_verify_lif_proof deliberately has no top-level import of this
    # runner.  Standalone verifier execution imports validate_config lazily;
    # this path supplies the config that this runner already validated.
    from drone_verify_lif_proof import verify_lif_proof

    report = verify_lif_proof(
        main_config=config_path.resolve(),
        validated_main_config=config,
    )
    return _VerifiedMainProof(report)


def _save_queue_threadsafe(
    queue_lock_object: threading.RLock,
    queue_path: Path,
    queue: dict[str, Any],
) -> None:
    with queue_lock_object:
        save_queue(queue_path, queue)


def _comparison_worker(
    queue: dict[str, Any],
    queue_path: Path,
    job: dict[str, Any],
    *,
    state_lock: threading.RLock,
    stop_submitting: threading.Event,
) -> str:
    """Run one isolated comparison job while synchronizing queue mutations."""

    artifact_root = Path(queue["artifact_root"])
    pause_file = Path(queue["pause_file"])
    try:
        with state_lock:
            if pause_file.exists() or stop_submitting.is_set():
                job["status"] = "paused"
                job["pause_reason"] = f"Pause request present at {pause_file}"
                stop_submitting.set()
                save_queue(queue_path, queue)
                return "paused"
            job["status"] = "running"
            job.pop("failure", None)
            job["started_utc"] = _utc_now()
            save_queue(queue_path, queue)

        if not _valid_training(job):
            command = list(job["training_command"])
            if Path(job["checkpoint"]).is_file():
                command.append("--resume")
            run = _run(
                command,
                artifact_root / "logs" / f"{job['id']}__train.log",
            )
            with state_lock:
                job["training_run"] = run
                save_queue(queue_path, queue)
            if run["exit_code"] == 3:
                with state_lock:
                    job["status"] = "paused"
                    job["pause_reason"] = (
                        f"Training acknowledged pause request at {pause_file}"
                    )
                    stop_submitting.set()
                    save_queue(queue_path, queue)
                return "paused"
            if run["exit_code"] != 0 or not _valid_training(job):
                with state_lock:
                    job["status"] = "failed"
                    job["failure"] = (
                        "Training command failed or artifact verification rejected it"
                    )
                    job["finished_utc"] = _utc_now()
                    save_queue(queue_path, queue)
                return "failed"

        job_failed = False
        for bundle in job["evaluations"]:
            with state_lock:
                if pause_file.exists() or stop_submitting.is_set():
                    job["status"] = "paused"
                    job["pause_reason"] = f"Pause request present at {pause_file}"
                    stop_submitting.set()
                    save_queue(queue_path, queue)
                    return "paused"
            if _valid_evaluation(
                bundle,
                job,
                queue["config"]["evaluation"]["episodes_per_scenario"],
            ):
                with state_lock:
                    bundle["status"] = "completed"
                    save_queue(queue_path, queue)
                continue
            with state_lock:
                bundle["status"] = "running"
                save_queue(queue_path, queue)
            run = _run(
                bundle["command"],
                artifact_root
                / "logs"
                / f"{job['id']}__{bundle['scenario']}__evaluate.log",
            )
            valid = (
                run["exit_code"] == 0
                and _valid_evaluation(
                    bundle,
                    job,
                    queue["config"]["evaluation"]["episodes_per_scenario"],
                )
            )
            with state_lock:
                bundle["run"] = run
                if not valid:
                    bundle["status"] = "failed"
                    job["status"] = "failed"
                    job["failure"] = f"Evaluation failed for {bundle['scenario']}"
                    job_failed = True
                else:
                    bundle["status"] = "completed"
                save_queue(queue_path, queue)
            if pause_file.exists():
                with state_lock:
                    job["status"] = "paused"
                    job["pause_reason"] = f"Pause request present at {pause_file}"
                    stop_submitting.set()
                    save_queue(queue_path, queue)
                return "paused"

        with state_lock:
            job["finished_utc"] = _utc_now()
            if job_failed:
                save_queue(queue_path, queue)
                return "failed"
            job["status"] = "completed"
            job.pop("failure", None)
            save_queue(queue_path, queue)
        return "completed"
    except Exception as exc:
        with state_lock:
            job["status"] = "failed"
            job["failure"] = (
                f"Concurrent comparison worker error: {type(exc).__name__}: {exc}"
            )
            job["finished_utc"] = _utc_now()
            save_queue(queue_path, queue)
        return "failed"


def _revalidate_completed_jobs(
    queue: dict[str, Any], queue_path: Path
) -> list[dict[str, Any]]:
    """Return unfinished jobs after rejecting stale completed artifacts."""

    unfinished: list[dict[str, Any]] = []
    expected_episodes = queue["config"]["evaluation"]["episodes_per_scenario"]
    for job in queue["jobs"]:
        if job["status"] == "completed":
            training_valid = _valid_training(job)
            evaluation_validity = {
                bundle["scenario"]: _valid_evaluation(
                    bundle, job, expected_episodes
                )
                for bundle in job["evaluations"]
            }
            if training_valid and all(evaluation_validity.values()):
                continue
            job["status"] = "pending"
            job["artifact_revalidation"] = {
                "checked_utc": _utc_now(),
                "training_valid": training_valid,
                "evaluations": evaluation_validity,
                "result": "rejected_stale_or_incomplete_artifact",
            }
            for bundle in job["evaluations"]:
                bundle["status"] = (
                    "completed"
                    if evaluation_validity[bundle["scenario"]]
                    else "pending"
                )
            save_queue(queue_path, queue)
        unfinished.append(job)
    return unfinished


def _execute_comparison_concurrent(
    queue: dict[str, Any],
    queue_path: Path,
    *,
    max_jobs: int | None,
) -> int:
    """Execute the authenticated comparison queue with at most two workers."""

    recorded = queue.get("concurrency_decision")
    current = comparison_concurrency_decision(queue.get("config", {}))
    if (
        not isinstance(recorded, dict)
        or recorded != current
        or recorded.get("status") != "paired_smoke_pass"
        or recorded.get("effective_max_concurrent_isaac_processes") != 2
        or queue.get("requested_max_concurrent_isaac_processes") != 2
        or queue.get("max_concurrent_isaac_processes") != 2
        or queue.get("resource_limits", {}).get("max_concurrent_isaac_processes")
        != 2
    ):
        raise ValueError(
            "Concurrent comparison execution requires the current authenticated "
            "paired-smoke receipt"
        )
    queue["dry_run"] = False
    unfinished = _revalidate_completed_jobs(queue, queue_path)
    # Do not begin GRU/MLP until all six LIF task cells have reached a terminal
    # state in this call.  Within each phase, submission order remains the
    # immutable controller-first queue order.
    phases = [
        [
            job
            for job in unfinished
            if job["controller"]
            in {"frozen_lif_original", "frozen_lif_degree_rewired"}
        ],
        [
            job
            for job in unfinished
            if job["controller"] not in {
                "frozen_lif_original",
                "frozen_lif_degree_rewired",
            }
        ],
    ]
    state_lock = threading.RLock()
    stop_submitting = threading.Event()
    submitted = 0
    had_failures = False
    paused = False

    def can_submit() -> bool:
        return (
            not stop_submitting.is_set()
            and (max_jobs is None or submitted < max_jobs)
        )

    executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="crazyflie-comparison"
    )
    submitted_futures: set[Future[str]] = set()
    try:
        for phase in phases:
            if not can_submit():
                break
            pending = iter(phase)
            active: dict[Future[str], dict[str, Any]] = {}
            exhausted = False
            while active or (not exhausted and can_submit()):
                while len(active) < 2 and not exhausted and can_submit():
                    try:
                        job = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    future = executor.submit(
                        _comparison_worker,
                        queue,
                        queue_path,
                        job,
                        state_lock=state_lock,
                        stop_submitting=stop_submitting,
                    )
                    active[future] = job
                    submitted_futures.add(future)
                    submitted += 1
                if not active:
                    break
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result = future.result()
                    had_failures = had_failures or result == "failed"
                    paused = paused or result == "paused"
                if stop_submitting.is_set():
                    # Already-running siblings are allowed to reach their
                    # own clean pause/checkpoint boundary.  Launch no more.
                    for future in tuple(active):
                        result = future.result()
                        had_failures = had_failures or result == "failed"
                        paused = paused or result == "paused"
                        active.pop(future)
                    break
            if paused or stop_submitting.is_set():
                break
    except BaseException:
        # ThreadPoolExecutor.__exit__ waits for worker threads before control
        # reaches an outer exception handler.  Isaac workers can run for
        # hours, so terminate their independently owned process groups before
        # asking the executor to join them.  Pending, not-yet-started futures
        # are cancelled before the blocking shutdown as well.
        stop_submitting.set()
        _terminate_active_children()
        for future in submitted_futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    save_queue(queue_path, queue)
    if paused:
        return 3
    return 0 if not had_failures and queue["status"] in {"completed", "pending"} else 1


def execute(
    queue: dict[str, Any],
    queue_path: Path,
    *,
    max_jobs: int | None = None,
    main_proof: _VerifiedMainProof | None = None,
) -> int:
    # Keep this check before even the first in-memory status mutation.  The CLI
    # verifies before acquiring/creating a queue lock; direct Python callers
    # likewise cannot execute a main queue without the opaque verified token.
    if queue.get("label") == "main" and not isinstance(main_proof, _VerifiedMainProof):
        raise ValueError(
            "Main execution requires the exact passing balanced-v3 original-LIF proof gate"
        )
    if queue.get("label") == COMPARISON_LABEL:
        recorded_decision = queue.get("concurrency_decision")
        current_decision = comparison_concurrency_decision(queue.get("config", {}))
        if recorded_decision != current_decision:
            raise ValueError(
                "Comparison concurrency decision or paired-smoke receipt changed "
                "after the reviewed dry run"
            )
        effective = queue.get("max_concurrent_isaac_processes")
        if (
            effective
            != current_decision.get("effective_max_concurrent_isaac_processes")
            or queue.get("resource_limits", {}).get(
                "max_concurrent_isaac_processes"
            )
            != effective
            or effective not in {1, 2}
        ):
            raise ValueError("Comparison effective concurrency metadata is invalid")
        if effective == 2:
            return _execute_comparison_concurrent(
                queue, queue_path, max_jobs=max_jobs
            )
    queue["dry_run"] = False
    artifact_root = Path(queue["artifact_root"])
    pause_file = Path(queue["pause_file"])
    completed_this_call = 0
    had_failures = False
    for job in queue["jobs"]:
        if job["status"] == "completed":
            expected_episodes = queue["config"]["evaluation"]["episodes_per_scenario"]
            training_valid = _valid_training(job)
            evaluation_validity = {
                bundle["scenario"]: _valid_evaluation(bundle, job, expected_episodes)
                for bundle in job["evaluations"]
            }
            if training_valid and all(evaluation_validity.values()):
                continue
            job["status"] = "pending"
            job["artifact_revalidation"] = {
                "checked_utc": _utc_now(),
                "training_valid": training_valid,
                "evaluations": evaluation_validity,
                "result": "rejected_stale_or_incomplete_artifact",
            }
            for bundle in job["evaluations"]:
                bundle["status"] = (
                    "completed" if evaluation_validity[bundle["scenario"]] else "pending"
                )
            save_queue(queue_path, queue)
        if max_jobs is not None and completed_this_call >= max_jobs:
            break
        if pause_file.exists():
            job["status"] = "paused"
            job["pause_reason"] = f"Pause request present at {pause_file}"
            save_queue(queue_path, queue)
            return 3
        job["status"] = "running"
        job.pop("failure", None)
        job["started_utc"] = _utc_now()
        save_queue(queue_path, queue)
        if not _valid_training(job):
            command = list(job["training_command"])
            if Path(job["checkpoint"]).is_file():
                command.append("--resume")
            run = _run(command, artifact_root / "logs" / f"{job['id']}__train.log")
            job["training_run"] = run
            save_queue(queue_path, queue)
            # The drone entrypoints preserve status with os._exit, so only the
            # current child's explicit pause code can classify this attempt as
            # paused.  Never let a stale paused manifest mask a failed or
            # unexpectedly zero-exit resume.
            if run["exit_code"] == 3:
                job["status"] = "paused"
                save_queue(queue_path, queue)
                return 3
            if run["exit_code"] != 0 or not _valid_training(job):
                job["status"] = "failed"
                job["failure"] = "Training command failed or artifact verification rejected it"
                job["finished_utc"] = _utc_now()
                had_failures = True
                completed_this_call += 1
                save_queue(queue_path, queue)
                # A failed cell must remain explicit, but it must not strand
                # independent later controller/seed cells in pending state.
                continue
        job_failed = False
        for bundle in job["evaluations"]:
            if pause_file.exists():
                job["status"] = "paused"
                job["pause_reason"] = f"Pause request present at {pause_file}"
                save_queue(queue_path, queue)
                return 3
            if _valid_evaluation(bundle, job, queue["config"]["evaluation"]["episodes_per_scenario"]):
                bundle["status"] = "completed"
                continue
            bundle["status"] = "running"
            save_queue(queue_path, queue)
            run = _run(
                bundle["command"],
                artifact_root / "logs" / f"{job['id']}__{bundle['scenario']}__evaluate.log",
            )
            bundle["run"] = run
            if run["exit_code"] != 0 or not _valid_evaluation(
                bundle, job, queue["config"]["evaluation"]["episodes_per_scenario"]
            ):
                bundle["status"] = "failed"
                job["status"] = "failed"
                job["failure"] = f"Evaluation failed for {bundle['scenario']}"
                job_failed = True
                had_failures = True
                save_queue(queue_path, queue)
                # Scenarios and later jobs are independent evidence cells.
                # Continue collecting them while preserving the failed state.
                continue
            bundle["status"] = "completed"
            save_queue(queue_path, queue)
            if pause_file.exists():
                job["status"] = "paused"
                job["pause_reason"] = f"Pause request present at {pause_file}"
                save_queue(queue_path, queue)
                return 3
        if job_failed:
            job["finished_utc"] = _utc_now()
            completed_this_call += 1
            save_queue(queue_path, queue)
            continue
        job["status"] = "completed"
        job.pop("failure", None)
        job["finished_utc"] = _utc_now()
        completed_this_call += 1
        save_queue(queue_path, queue)
    save_queue(queue_path, queue)
    return 0 if not had_failures and queue["status"] in {"completed", "pending"} else 1


def print_dry_run(queue: dict[str, Any]) -> None:
    print(
        f"{queue['label']} matrix: {queue['job_count']} training jobs, "
        f"{queue['evaluation_bundle_count']} checkpoint/scenario bundles, "
        f"{queue['predicted_evaluation_episodes']} evaluation episodes"
    )
    print("resolved_config=" + json.dumps(queue["config"], sort_keys=True))
    print("resource_limits=" + json.dumps(queue["resource_limits"], sort_keys=True))
    for index, job in enumerate(queue["jobs"], start=1):
        print(f"[{index:02d}/{queue['job_count']:02d}] {job['id']}")
        print(f"  task={job['task']} seed={job['seed']} output={job['run_dir']}")
        print(f"  fingerprint={job['expected_fingerprint']}")
        print(f"  train: {shlex.join(job['training_command'])}")
        for bundle in job["evaluations"]:
            print(f"  evaluate[{bundle['scenario']}]: {shlex.join(bundle['command'])}")


def _cli_identity(queue: dict[str, Any]) -> dict[str, Any]:
    """Return the queue-scoped config and ordered per-job fingerprints."""

    return {
        "fingerprint_scope": "per_job",
        "resolved_config": queue["config"],
        "config_sha256": queue["config_sha256"],
        "fingerprints": {
            job["id"]: job["expected_fingerprint"] for job in queue["jobs"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry_run", action="store_true", help="Write and print the queue without starting Isaac")
    modes.add_argument("--execute", action="store_true", help="Execute jobs sequentially")
    modes.add_argument("--request_pause", action="store_true", help="Create the queue pause request and launch nothing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max_jobs", type=int)
    parser.add_argument("--authorize_main", action="store_true", help="Required in addition to --execute for a main matrix")
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max_jobs must be positive")
    try:
        config = validate_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT:
        default_queue_name = (
            "crazyflie_task_separated_v1_balanced_v4_seed0_comparison_v1.json"
        )
    elif config.get("_matrix_layout") == TASK_SEPARATED_LAYOUT:
        default_queue_name = (
            "crazyflie_task_separated_v1_balanced_v3_schema_dryrun_"
            f"{config['label']}.json"
        )
    else:
        default_queue_name = f"crazyflie_{config['label']}_v1.json"
    output = (args.output or ROOT / "runs" / default_queue_name).resolve()
    if config.get("_matrix_layout") in {
        TASK_SEPARATED_COMPARISON_BASE_LAYOUT,
        TASK_SEPARATED_COMPARISON_CELL_LAYOUT,
    }:
        parser.error("Comparison base/cell configs are not queues; use the umbrella config")
    if args.request_pause:
        if not output.is_file():
            parser.error(f"Cannot pause a missing queue: {output}")
        queue = _read_json(output)
        if queue.get("config_sha256") != canonical_sha256(public_config(config)):
            parser.error("Pause config differs from the queue config")
        try:
            validate_resume_queue(queue, config, output)
            pause_file, request, created = write_pause_request(queue, output)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            "status": "pause_requested" if created else "pause_already_requested",
            "pause_file": str(pause_file),
            "request": request,
            "queue": str(output),
            **_cli_identity(queue),
        }, indent=2, sort_keys=True))
        return 0
    if (
        args.execute
        and config.get("_matrix_layout") == TASK_SEPARATED_LAYOUT
        and config.get("execution_readiness") == TASK_SEPARATED_SCHEMA_DRY_RUN
    ):
        parser.error(
            "Task-separated balanced-v3 configs are schema-dry-run only; "
            "launch requires the separately reviewed/refrozen successor protocol"
        )
    if args.execute and config["label"] == "main" and not args.authorize_main:
        parser.error("Main execution is locked; explicit user authorization and --authorize_main are required")
    if args.execute and config["label"] == "main" and (not args.resume or not output.is_file()):
        parser.error(
            "Main execution requires --resume and an existing reviewed dry-run queue; "
            "execution cannot create an unreviewed main queue"
        )
    if (
        args.execute
        and config.get("_matrix_layout") == TASK_SEPARATED_COMPARISON_LAYOUT
        and (not args.resume or not output.is_file())
    ):
        parser.error(
            "Comparison execution requires --resume and its existing reviewed dry-run "
            "queue; execution cannot create an unreviewed comparison queue"
        )
    main_proof: _VerifiedMainProof | None = None
    if args.execute and config["label"] == "main":
        # This is intentionally before queue_lock(): even creation of the lock
        # file, pause consumption, resume bookkeeping, or queue status changes
        # is forbidden until the empirical proof has been revalidated.
        try:
            main_proof = _verify_main_lif_proof(config, args.config.resolve())
        except Exception as exc:
            parser.error(f"Main empirical proof gate failed: {exc}")
        print(json.dumps(main_proof.report, indent=2, sort_keys=True))
    try:
        with queue_lock(output):
            if args.resume:
                if not output.is_file():
                    parser.error(f"Cannot resume missing queue: {output}")
                queue = _read_json(output)
                expected = canonical_sha256(public_config(config))
                if queue.get("config_sha256") != expected:
                    parser.error("Resume config differs from the queue config")
                validate_resume_queue(queue, config, output)
            else:
                if output.exists():
                    parser.error(
                        f"Queue already exists: {output}; preserve it and use --resume to validate or continue"
                    )
                queue = build_queue(config, output)
            if args.dry_run or not args.execute:
                preview = build_queue(config, output) if args.resume else queue
                if not args.resume:
                    preview["dry_run"] = True
                    save_queue(output, preview)
                print_dry_run(preview)
                print(json.dumps({
                    "status": "PASS", "queue": str(output), "jobs": preview["job_count"],
                    "evaluation_bundles": preview["evaluation_bundle_count"],
                    "predicted_evaluation_episodes": preview["predicted_evaluation_episodes"],
                    "existing_queue_preserved": bool(args.resume),
                    **_cli_identity(preview),
                }, indent=2, sort_keys=True))
                return 0
            # Different queue files have different locks.  This additional
            # machine-wide Crazyflie lock prevents two matrix runners from
            # launching Isaac concurrently on the same workstation.  Acquire
            # it before consuming a pause request or mutating paused jobs, so
            # a lock collision preserves the user's pause intent.
            with queue_lock(ROOT / "runs" / ".crazyflie_isaac"):
                if args.resume:
                    resume_event = consume_pause_request(queue, output)
                    save_queue(output, queue)
                    print(json.dumps(
                        {"status": "resume_recorded", **resume_event}, indent=2, sort_keys=True
                    ))
                exit_code = execute(
                    queue,
                    output,
                    max_jobs=args.max_jobs,
                    main_proof=main_proof,
                )
                print(json.dumps({
                    "status": queue["status"],
                    "exit_code": exit_code,
                    "queue": str(output),
                    "failures": {
                        job["id"]: job.get("failure")
                        for job in queue["jobs"]
                        if job.get("status") == "failed"
                    },
                    **_cli_identity(queue),
                }, indent=2, sort_keys=True))
                return exit_code
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
