#!/usr/bin/env python3
"""Train one matched Crazyflie controller with resumable recurrent PPO."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from drone_bootstrap import (
    DEFAULT_CONTRACT_PROFILE,
    DEFAULT_CONNECTOME,
    DEFAULT_OPTIC_CONNECTOME,
    DEFAULT_WING_CONNECTOME,
    DEFAULT_REWIRE_MANIFEST,
    ROOT,
    canonical_sha256,
    launch_environment,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    rollout_rng_contract,
    sha256_file,
)
from drone_evaluation_protocol import load_protocol
from g1_fly_control.tasks.crazyflie.command_logic import COMMAND_FOLLOW_TASK_ID
from g1_fly_control.tasks.crazyflie.logic import (
    balanced_switch_target_curriculum_payload,
    balanced_task_contract_payload,
    balanced_v4_switch_target_curriculum_payload,
    balanced_v4_task_contract_payload,
    mixed_scenario_codes,
    mixed_scenario_contract_payload,
    survival_first_contract_payload,
    switch_target_curriculum_payload,
)
from g1_fly_control.crazyflie.memory import (
    GPU_LIMIT_MIB,
    MEMORY_POLICY_VERSION,
    RAM_LIMIT_PERCENT,
    RSS_GROWTH_TOLERANCE_MIB,
    SWAP_OUT_GROWTH_TOLERANCE_MIB,
    assess as assess_memory,
)


POLICIES = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
    "gru_matched",
    "mlp_normal",
    "wing_lif",
    "leg_wing_lif",
    "optic_lif",
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
)
POLICY_ALIASES = {"mlp": "mlp_normal"}
COMBINATION_LIF_CORE_LABELS: dict[str, tuple[str, ...]] = {
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}
COMBINATION_LIF_FUSION_CONTRACTS = {
    "leg_optic_lif": "independent_leg_and_optic_cores_concat_motor_readouts_v1",
    "wing_optic_lif": "independent_wing_and_optic_cores_concat_motor_readouts_v1",
    "leg_wing_optic_lif": (
        "independent_leg_and_wing_and_optic_cores_concat_motor_readouts_v1"
    ),
}
TASK = "FlyCrazyflie-WaypointReach-v0"
MIXED_TASK = "FlyCrazyflie-Mixed-v0"
COMMAND_TASK = COMMAND_FOLLOW_TASK_ID
COMMAND_WIDE_TASK = "FlyCrazyflie-CommandFollowWide-v0"
COMMAND_WIDE_WIND_TASK = "FlyCrazyflie-CommandFollowWideWind-v0"
COMMAND_V2_TASKS = (COMMAND_WIDE_TASK, COMMAND_WIDE_WIND_TASK)
COMMAND_TASKS = (COMMAND_TASK, *COMMAND_V2_TASKS)
TRAINING_TASKS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
    MIXED_TASK,
    *COMMAND_TASKS,
)
COMMAND_CONTRACT_PROFILE = "command_v1"
COMMAND_V2_CONTRACT_PROFILE = "command_v2"
CONTRACT_PROFILES = (
    "survival_v2",
    "balanced_v3",
    "balanced_v4",
    COMMAND_CONTRACT_PROFILE,
    COMMAND_V2_CONTRACT_PROFILE,
)
COMMAND_EVALUATION_PROTOCOL = "command_v1"
COMMAND_V2_EVALUATION_PROTOCOL = "command_v2"
EVALUATION_PROTOCOLS = (
    "integration",
    "lif_proof",
    "main",
    COMMAND_EVALUATION_PROTOCOL,
    COMMAND_V2_EVALUATION_PROTOCOL,
)
STANDALONE_HORIZON = 100
STANDALONE_LEARNING_RATE = 3.0e-5
MIXED_TASK_SCHEDULE_STATE_KIND = "flyg1.crazyflie.mixed-task-schedule.v1"


def memory_acceptance_contract_payload() -> dict[str, Any]:
    """Return the canonical hard/warning memory policy embedded in fingerprints."""

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


def _nonfinite_failure_count(episode_metrics: Mapping[str, Any]) -> int:
    """Validate and return the task's nonfinite terminal count (cause code 4)."""

    counts = episode_metrics.get("failure_cause_counts")
    if not isinstance(counts, Mapping):
        raise RuntimeError("training metrics lack failure_cause_counts")
    value = counts.get("4")
    if type(value) is not int or value < 0:
        raise RuntimeError("training failure_cause_counts['4'] must be a non-negative integer")
    return value


def _raise_for_nonfinite_failure_count(count: int, *, completed_updates: int) -> None:
    if type(count) is not int or count < 0:
        raise RuntimeError("nonfinite failure count must be a non-negative integer")
    if count:
        raise RuntimeError(
            "Training observed nonfinite terminal failure cause code 4: "
            f"count={count} at update {completed_updates}"
        )


def _memory_gate_hard_failure(gate: Mapping[str, Any], *, stage: str) -> None:
    """Fail on v2 hard violations while deliberately allowing RSS warnings."""

    if gate.get("policy_version") != MEMORY_POLICY_VERSION:
        raise RuntimeError(
            f"Memory gate at {stage} did not use {MEMORY_POLICY_VERSION}"
        )
    warnings = gate.get("warnings")
    failures = gate.get("failures")
    if not isinstance(warnings, list) or not all(
        isinstance(message, str) and message for message in warnings
    ):
        raise RuntimeError(f"Memory gate at {stage} has malformed warnings")
    if not isinstance(failures, list) or not all(
        isinstance(message, str) and message for message in failures
    ):
        raise RuntimeError(f"Memory gate at {stage} has malformed failures")
    if gate.get("passed") is not True:
        detail = "; ".join(failures) or "gate did not explicitly pass"
        raise RuntimeError(f"Memory gate failed at {stage}: {detail}")


def _failure_runtime_evidence(
    *,
    counters: Any,
    history_reference: dict[str, Any] | None,
    pending_history_rows: int,
    memory_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the evidence block persisted for every failed training attempt."""

    recomputed_memory_gate: dict[str, Any] | None = None
    memory_gate_recomputation_error: str | None = None
    if memory_samples:
        try:
            recomputed_memory_gate = assess_memory(memory_samples)
        except (KeyError, TypeError, ValueError) as exc:
            memory_gate_recomputation_error = f"{type(exc).__name__}: {exc}"
    as_dict = getattr(counters, "as_dict", None)
    counter_payload = as_dict() if callable(as_dict) else None
    return {
        "counters": counter_payload,
        "history_reference": history_reference,
        "pending_uncommitted_history_rows": pending_history_rows,
        "memory_samples": memory_samples,
        "memory_gate": recomputed_memory_gate,
        "recomputed_memory_gate": recomputed_memory_gate,
        "memory_gate_recomputation_error": memory_gate_recomputation_error,
    }


def _command_contracts(
    task: str = COMMAND_TASK,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return and cross-check the compact and complete command contracts."""

    if task == COMMAND_TASK:
        from g1_fly_control.tasks.crazyflie.command_logic import (
            COMMAND_TRACKING_CONTRACT_SHA256,
            command_follow_contract_payload,
            command_training_contract_payload,
        )

        compact = command_follow_contract_payload()
        complete = command_training_contract_payload()
        contract_sha256 = COMMAND_TRACKING_CONTRACT_SHA256
    elif task in COMMAND_V2_TASKS:
        from g1_fly_control.tasks.crazyflie.command_wide_logic import (
            command_wide_contract_payload,
            command_wide_training_contract_payload,
            command_wide_training_contract_sha256,
        )

        wind_enabled = task == COMMAND_WIDE_WIND_TASK
        compact = command_wide_contract_payload()
        complete = command_wide_training_contract_payload(
            wind_enabled=wind_enabled
        )
        contract_sha256 = command_wide_training_contract_sha256(
            wind_enabled=wind_enabled
        )
    else:
        raise ValueError(f"Task {task!r} is not a command-follow task")
    if canonical_sha256(complete) != contract_sha256:
        raise RuntimeError("Command training contract SHA-256 is internally inconsistent")
    for key, value in compact.items():
        if complete.get(key) != value:
            raise RuntimeError(
                f"Command compatibility field {key!r} differs from the training contract"
            )
    return compact, complete, contract_sha256


def _validate_task_contract_selection(
    task: str,
    contract_profile: str,
    evaluation_protocol: str | None = None,
) -> None:
    """Keep the command task isolated from legacy waypoint contracts."""

    if task == COMMAND_TASK:
        if contract_profile != COMMAND_CONTRACT_PROFILE:
            raise ValueError(
                f"{COMMAND_TASK} requires contract_profile={COMMAND_CONTRACT_PROFILE}"
            )
        if (
            evaluation_protocol is not None
            and evaluation_protocol != COMMAND_EVALUATION_PROTOCOL
        ):
            raise ValueError(
                f"{COMMAND_TASK} requires evaluation_protocol={COMMAND_EVALUATION_PROTOCOL}"
            )
    elif task in COMMAND_V2_TASKS:
        if contract_profile != COMMAND_V2_CONTRACT_PROFILE:
            raise ValueError(
                f"{task} requires contract_profile={COMMAND_V2_CONTRACT_PROFILE}"
            )
        if (
            evaluation_protocol is not None
            and evaluation_protocol != COMMAND_V2_EVALUATION_PROTOCOL
        ):
            raise ValueError(
                f"{task} requires evaluation_protocol={COMMAND_V2_EVALUATION_PROTOCOL}"
            )
    else:
        if contract_profile == COMMAND_CONTRACT_PROFILE:
            raise ValueError(
                f"contract_profile={COMMAND_CONTRACT_PROFILE} is valid only for {COMMAND_TASK}"
            )
        if evaluation_protocol == COMMAND_EVALUATION_PROTOCOL:
            raise ValueError(
                f"evaluation_protocol={COMMAND_EVALUATION_PROTOCOL} is valid only for {COMMAND_TASK}"
            )
        if contract_profile == COMMAND_V2_CONTRACT_PROFILE:
            raise ValueError(
                "contract_profile=command_v2 is valid only for "
                + ", ".join(COMMAND_V2_TASKS)
            )
        if evaluation_protocol == COMMAND_V2_EVALUATION_PROTOCOL:
            raise ValueError(
                "evaluation_protocol=command_v2 is valid only for "
                + ", ".join(COMMAND_V2_TASKS)
            )


def command_evaluation_manifest(task: str = COMMAND_TASK) -> dict[str, Any]:
    """Return the deterministic held-out command-protocol identity."""

    if task == COMMAND_TASK:
        from g1_fly_control.crazyflie.command_evaluation import (
            protocol_payload,
            protocol_sha256,
        )

        protocol_name = COMMAND_EVALUATION_PROTOCOL
        evaluation = protocol_payload()
        evaluation_sha256 = protocol_sha256()
    elif task in COMMAND_V2_TASKS:
        from g1_fly_control.crazyflie.command_wide_evaluation import (
            protocol_payload,
            protocol_sha256,
        )

        protocol_name = COMMAND_V2_EVALUATION_PROTOCOL
        evaluation = protocol_payload(task=task)
        evaluation_sha256 = protocol_sha256(task=task)
    else:
        raise ValueError(f"Task {task!r} is not a command-follow task")
    compact, complete, contract_sha256 = _command_contracts(task)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol_name,
        "task": task,
        "evaluation_protocol": evaluation,
        "evaluation_protocol_sha256": evaluation_sha256,
        "command_follow_contract": compact,
        "command_training_contract_sha256": contract_sha256,
        "command_training_contract": complete,
    }
    manifest["manifest_id"] = canonical_sha256(manifest)
    return manifest


def _evaluation_manifest_for_task(task: str, protocol: str) -> dict[str, Any]:
    if task in COMMAND_TASKS:
        expected_profile = (
            COMMAND_CONTRACT_PROFILE
            if task == COMMAND_TASK
            else COMMAND_V2_CONTRACT_PROFILE
        )
        _validate_task_contract_selection(
            task, expected_profile, protocol
        )
        return command_evaluation_manifest(task)
    if protocol in {COMMAND_EVALUATION_PROTOCOL, COMMAND_V2_EVALUATION_PROTOCOL}:
        raise ValueError(
            f"evaluation_protocol={protocol} is valid only for command-follow tasks"
        )
    return load_protocol(protocol)


def _task_contract_for_profile(
    profile: str,
    *,
    task: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the one versioned task contract selected for this run."""

    if profile == "survival_v2":
        return "survival_first_contract", survival_first_contract_payload()
    if profile == "balanced_v3":
        return "balanced_task_contract", balanced_task_contract_payload()
    if profile == "balanced_v4":
        return "balanced_v4_task_contract", balanced_v4_task_contract_payload()
    if profile == COMMAND_CONTRACT_PROFILE:
        return "command_training_contract", _command_contracts(COMMAND_TASK)[1]
    if profile == COMMAND_V2_CONTRACT_PROFILE:
        if task not in COMMAND_V2_TASKS:
            raise ValueError("command_v2 task identity is required for its task contract")
        return "command_training_contract", _command_contracts(task)[1]
    raise ValueError(
        f"Unknown Crazyflie contract profile {profile!r}; choose one of {CONTRACT_PROFILES}"
    )


def _switch_contract_for_profile(profile: str) -> dict[str, object]:
    if profile == "survival_v2":
        return switch_target_curriculum_payload()
    if profile == "balanced_v3":
        return balanced_switch_target_curriculum_payload()
    if profile == "balanced_v4":
        return balanced_v4_switch_target_curriculum_payload()
    raise ValueError(
        f"Unknown Crazyflie contract profile {profile!r}; choose one of {CONTRACT_PROFILES}"
    )


def _rollout_rng_contract() -> dict[str, Any]:
    """Return the static, fingerprinted contract for training random streams."""

    return rollout_rng_contract()


def _seed_training_streams(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("training RNG seed must be an integer in [0, 2**32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_controller_rng(seed: int) -> dict[str, Any]:
    """Reseed after Isaac startup so controller weights depend only on the job seed."""

    _seed_training_streams(seed)
    contract = _rollout_rng_contract()
    return {
        "scheme": contract["scheme"],
        "seed": seed,
        "timing": contract["controller_construction_timing"],
        "streams": contract["controller_construction_streams"],
        "controller_construction_reseed_applied": True,
    }


def _initialize_rollout_rng(seed: int, *, resume_restored: bool) -> dict[str, Any]:
    """Initialize fresh rollout RNGs without disturbing a resumed checkpoint.

    Policy architectures consume different numbers of random values while
    their parameters are constructed.  A fresh run therefore reseeds every
    training stream only after all architecture-dependent construction is
    complete.  A resume has already restored its checkpoint RNG state and is
    deliberately a no-op here.
    """

    if type(seed) is not int or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("rollout RNG seed must be an integer in [0, 2**32 - 1]")
    report: dict[str, Any] = {
        **_rollout_rng_contract(),
        "seed": seed,
        "fresh_reseed_applied": not resume_restored,
        "checkpoint_rng_preserved": bool(resume_restored),
    }
    if resume_restored:
        return report
    _seed_training_streams(seed)
    return report


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _prepare_run_directory(run_dir: Path, *, resume: bool) -> Path | None:
    """Create a run directory or recoverably archive a pre-checkpoint attempt."""

    run_dir = run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    if run_dir == runs_root or not run_dir.is_relative_to(runs_root):
        raise ValueError(f"Run directory must be a child of {runs_root}: {run_dir}")
    latest = run_dir / "checkpoints" / "latest.pt"
    if resume:
        if not latest.is_file():
            raise ValueError(f"--resume requested but checkpoint does not exist: {latest}")
        return None
    if latest.exists():
        raise ValueError(f"Run already has a checkpoint: {latest}; use --resume or another directory")

    archived: Path | None = None
    if run_dir.exists() and any(run_dir.iterdir()):
        # A fresh invocation with no latest.pt can only be an attempt that
        # stopped before committing its first restart boundary.  Preserve the
        # complete directory under a sibling archive; never delete or rewrite
        # its failure report, temporary files, or diagnostics.
        archive_root = run_dir.parent / "failed_attempt_archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / f"{run_dir.name}-{time.time_ns()}"
        os.replace(run_dir, archived)
        for directory in (run_dir.parent, archive_root):
            descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return archived


def _gradient_norm(module: torch.nn.Module) -> float | None:
    gradients = [parameter.grad.detach().norm().square() for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        return None
    return float(torch.stack(gradients).sum().sqrt())


def _connectome_artifact_identity(manifest: Path) -> dict[str, Any]:
    """Return fail-closed hashes for a manifest and both derived graph files."""

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    identity: dict[str, Any] = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    for field in ("neurons_path", "edges_path"):
        relative = raw.get(field)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Connectome manifest lacks {field}: {manifest}")
        child = (manifest.parent / relative).resolve()
        identity[field] = {"path": str(child), "sha256": sha256_file(child)}
    return identity


def _connectome_manifest_for_label(
    args: argparse.Namespace, label: str
) -> Path:
    """Resolve one authenticated biological core from the CLI namespace."""

    if label == "leg":
        return args.connectome_manifest
    if label == "wing":
        return args.wing_connectome_manifest
    if label == "optic":
        return getattr(args, "optic_connectome_manifest", DEFAULT_OPTIC_CONNECTOME)
    raise ValueError(f"Unknown Crazyflie connectome label {label!r}")


def _primary_connectome_manifest(args: argparse.Namespace) -> Path:
    """Return the primary artifact used by the legacy fingerprint envelope.

    Every additional core is independently hashed inside ``resolved_config``;
    this selection preserves the historical single-primary fingerprint shape.
    """

    labels = COMBINATION_LIF_CORE_LABELS.get(args.policy)
    if labels:
        return _connectome_manifest_for_label(args, labels[0])
    if args.policy == "wing_lif":
        return args.wing_connectome_manifest
    if args.policy == "optic_lif":
        return getattr(args, "optic_connectome_manifest", DEFAULT_OPTIC_CONNECTOME)
    return args.connectome_manifest


def _validate_combination_controller_report(
    args: argparse.Namespace, report: Mapping[str, Any]
) -> None:
    """Fail closed if a built multi-core controller differs from its CLI identity."""

    expected_labels = COMBINATION_LIF_CORE_LABELS.get(args.policy)
    if expected_labels is None:
        return
    if report.get("controller_kind") != args.policy:
        raise RuntimeError("Combination controller report kind differs from the request")
    if tuple(report.get("core_labels", ())) != expected_labels:
        raise RuntimeError("Combination controller core ordering differs from its contract")
    if report.get("fusion_contract") != COMBINATION_LIF_FUSION_CONTRACTS[args.policy]:
        raise RuntimeError("Combination controller fusion contract differs from its contract")
    for field in (
        "connectome_manifests",
        "connectome_checksums",
        "per_core_checksums",
    ):
        values = report.get(field)
        if not isinstance(values, Mapping) or tuple(values) != expected_labels:
            raise RuntimeError(
                f"Combination controller {field} must preserve exact ordered core labels"
            )
    manifests = report["connectome_manifests"]
    for label in expected_labels:
        expected_path = _connectome_manifest_for_label(args, label).resolve()
        reported_path = Path(manifests[label]).expanduser().resolve()
        if reported_path != expected_path:
            raise RuntimeError(
                f"Combination controller {label} manifest differs from the CLI artifact"
            )
        checksum = report["per_core_checksums"][label]
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise RuntimeError(
                f"Combination controller {label} frozen-core checksum is malformed"
            )
        connectome_checksum = report["connectome_checksums"][label]
        if not isinstance(connectome_checksum, str) or len(connectome_checksum) != 64:
            raise RuntimeError(
                f"Combination controller {label} connectome checksum is malformed"
            )


def _summarize_lif_activity(
    policy: torch.nn.Module, accumulators: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one history payload from every observed frozen core."""

    if not accumulators:
        raise RuntimeError("LIF activity requires at least one named core")
    summaries: dict[str, Any] = {}
    for label, accumulator in accumulators.items():
        if not isinstance(label, str) or not label:
            raise RuntimeError("LIF activity core labels must be non-empty strings")
        summary = accumulator.summary()
        if not isinstance(summary, dict) or not summary:
            raise RuntimeError(f"LIF activity summary for {label!r} is malformed")
        summaries[label] = summary
    if len(summaries) == 1:
        return next(iter(summaries.values()))
    fusion_contract = getattr(policy, "fusion_contract", None)
    if not isinstance(fusion_contract, str) or not fusion_contract:
        raise RuntimeError("Multi-core LIF activity lacks its fusion contract")
    return {"composition": fusion_contract, **summaries}


def _summed_attribute(env: Any, name: str, default: int = 0) -> int:
    value = getattr(env, name, None)
    if value is None:
        return default
    return int(torch.as_tensor(value).sum().item())


def _nonnegative_integer_vector(
    value: Any,
    *,
    name: str,
    width: int,
) -> torch.Tensor:
    """Normalize a closed-schema schedule vector without accepting coercions."""

    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Mixed task schedule {name} is not a tensor-like vector") from exc
    if tensor.ndim != 1 or tensor.numel() != width:
        raise RuntimeError(f"Mixed task schedule {name} must have width {width}")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise RuntimeError(f"Mixed task schedule {name} must use an integer dtype")
    tensor = tensor.to(torch.long)
    if bool((tensor < 0).any()):
        raise RuntimeError(f"Mixed task schedule {name} must be non-negative")
    return tensor


def _expected_mixed_scenario_starts(
    episode_indices: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Reconstruct exact cumulative start counts without replaying all episodes."""

    width = int(episode_indices.numel())
    environment_ids = torch.arange(width, dtype=torch.long)
    complete_cycles = episode_indices.div(3, rounding_mode="floor")
    counts = torch.full(
        (3,), int(complete_cycles.sum().item()), dtype=torch.long
    )
    remainders = torch.remainder(episode_indices, 3)
    for offset in range(2):
        active = remainders > offset
        if bool(active.any()):
            codes = torch.remainder(environment_ids[active] + seed + offset, 3)
            counts += torch.bincount(codes, minlength=3)
    return counts


def _validate_mixed_task_schedule_state(
    value: Any,
    *,
    seed: int,
    num_envs: int,
) -> dict[str, Any]:
    """Validate and canonicalize the restart-critical Mixed schedule state."""

    if not isinstance(value, Mapping):
        raise RuntimeError("Mixed task schedule checkpoint state is not a mapping")
    required = {
        "schema_version",
        "kind",
        "contract",
        "num_envs",
        "per_environment_episode_index",
        "scenario_episode_starts",
        "next_scenario_codes",
        "total_episode_starts",
    }
    if set(value) != required:
        raise RuntimeError(
            "Mixed task schedule checkpoint state differs from the closed schema"
        )
    if value["schema_version"] != 1 or value["kind"] != MIXED_TASK_SCHEDULE_STATE_KIND:
        raise RuntimeError("Mixed task schedule checkpoint state has an incompatible version")
    if value["contract"] != mixed_scenario_contract_payload(seed=seed):
        raise RuntimeError("Mixed task schedule checkpoint contract differs from this run")
    if (
        not isinstance(num_envs, int)
        or isinstance(num_envs, bool)
        or num_envs < 1
        or value["num_envs"] != num_envs
    ):
        raise RuntimeError("Mixed task schedule checkpoint num_envs differs from this run")

    episode_indices = _nonnegative_integer_vector(
        value["per_environment_episode_index"],
        name="per_environment_episode_index",
        width=num_envs,
    )
    starts = _nonnegative_integer_vector(
        value["scenario_episode_starts"],
        name="scenario_episode_starts",
        width=3,
    )
    next_codes = _nonnegative_integer_vector(
        value["next_scenario_codes"],
        name="next_scenario_codes",
        width=num_envs,
    )
    expected_starts = _expected_mixed_scenario_starts(episode_indices, seed=seed)
    if not torch.equal(starts, expected_starts):
        raise RuntimeError(
            "Mixed task schedule cumulative starts disagree with its per-environment indices"
        )
    expected_next = mixed_scenario_codes(
        torch.arange(num_envs, dtype=torch.long), episode_indices, seed=seed
    )
    if not torch.equal(next_codes, expected_next):
        raise RuntimeError(
            "Mixed task schedule next scenario codes disagree with its deterministic sequence"
        )
    total = value["total_episode_starts"]
    if (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or total != int(episode_indices.sum().item())
        or total != int(starts.sum().item())
    ):
        raise RuntimeError("Mixed task schedule total episode starts is inconsistent")
    return {
        "schema_version": 1,
        "kind": MIXED_TASK_SCHEDULE_STATE_KIND,
        "contract": mixed_scenario_contract_payload(seed=seed),
        "num_envs": num_envs,
        "per_environment_episode_index": episode_indices.tolist(),
        "scenario_episode_starts": starts.tolist(),
        "next_scenario_codes": next_codes.tolist(),
        "total_episode_starts": total,
    }


def _capture_mixed_task_schedule_state(
    env: Any,
    *,
    task: str,
    seed: int,
) -> dict[str, Any] | None:
    """Capture all state needed to continue the Mixed round-robin after reset."""

    if task != MIXED_TASK:
        return None
    num_envs = getattr(env, "num_envs", None)
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise RuntimeError("Mixed environment has an invalid num_envs value")
    episode_indices = _nonnegative_integer_vector(
        getattr(env, "mixed_episode_index", None),
        name="per_environment_episode_index",
        width=num_envs,
    )
    starts = _nonnegative_integer_vector(
        getattr(env, "mixed_scenario_episode_starts", None),
        name="scenario_episode_starts",
        width=3,
    )
    next_codes = mixed_scenario_codes(
        torch.arange(num_envs, dtype=torch.long), episode_indices, seed=seed
    )
    state = {
        "schema_version": 1,
        "kind": MIXED_TASK_SCHEDULE_STATE_KIND,
        "contract": mixed_scenario_contract_payload(seed=seed),
        "num_envs": num_envs,
        "per_environment_episode_index": episode_indices.tolist(),
        "scenario_episode_starts": starts.tolist(),
        "next_scenario_codes": next_codes.tolist(),
        "total_episode_starts": int(starts.sum().item()),
    }
    return _validate_mixed_task_schedule_state(
        state, seed=seed, num_envs=num_envs
    )


def _restore_mixed_task_schedule_state(
    env: Any,
    *,
    task: str,
    seed: int,
    state: Any,
) -> dict[str, Any] | None:
    """Restore Mixed scheduling before the deliberate full reset on resume."""

    if task != MIXED_TASK:
        if state is not None:
            raise RuntimeError("A fixed task checkpoint unexpectedly contains Mixed schedule state")
        return None
    num_envs = getattr(env, "num_envs", None)
    normalized = _validate_mixed_task_schedule_state(
        state, seed=seed, num_envs=num_envs
    )
    episode_indices = getattr(env, "mixed_episode_index", None)
    starts = getattr(env, "mixed_scenario_episode_starts", None)
    if not isinstance(episode_indices, torch.Tensor) or not isinstance(starts, torch.Tensor):
        raise RuntimeError("Mixed environment lacks mutable schedule tensors")
    if (
        episode_indices.shape != (num_envs,)
        or episode_indices.dtype == torch.bool
        or episode_indices.is_floating_point()
        or episode_indices.is_complex()
        or starts.shape != (3,)
        or starts.dtype == torch.bool
        or starts.is_floating_point()
        or starts.is_complex()
    ):
        raise RuntimeError("Mixed environment schedule tensors have an incompatible layout")
    episode_indices.copy_(
        torch.tensor(
            normalized["per_environment_episode_index"],
            dtype=episode_indices.dtype,
            device=episode_indices.device,
        )
    )
    starts.copy_(
        torch.tensor(
            normalized["scenario_episode_starts"],
            dtype=starts.dtype,
            device=starts.device,
        )
    )
    if _capture_mixed_task_schedule_state(env, task=task, seed=seed) != normalized:
        raise RuntimeError("Mixed task schedule did not round-trip after checkpoint restore")
    return {
        "schema_version": 1,
        "restored_from_checkpoint": True,
        "restored_before_resume_environment_reset": True,
        "checkpoint_state": normalized,
    }


def _mixed_scenario_snapshot(env: Any, *, task: str, seed: int) -> dict[str, Any] | None:
    """Return a JSON-safe audit of the training-only mixed scenario schedule."""

    if task != MIXED_TASK:
        return None
    extras = getattr(env, "extras", {})

    def required(name: str) -> Any:
        value = getattr(env, name, None)
        if value is None and isinstance(extras, dict):
            value = extras.get(name)
        if value is None:
            raise RuntimeError(f"Mixed environment lacks required audit field {name}")
        return value

    contract = mixed_scenario_contract_payload(seed=seed)
    names = tuple(str(name) for name in required("mixed_scenario_names_by_code"))
    expected_names = tuple(contract["scenario_names_by_code"])
    if names != expected_names:
        raise RuntimeError(
            f"Mixed scenario names {names!r} differ from contract {expected_names!r}"
        )

    required("mixed_episode_index")
    required("mixed_scenario_episode_starts")
    schedule_state = _capture_mixed_task_schedule_state(
        env, task=task, seed=seed
    )
    assert schedule_state is not None
    starts_tensor = torch.as_tensor(required("mixed_scenario_episode_starts")).detach().cpu()
    episode_codes = torch.as_tensor(required("episode_scenario_code")).detach().cpu().to(torch.long)
    terminal_codes = torch.as_tensor(required("terminal_scenario_code")).detach().cpu().to(torch.long)
    terminal_mask = torch.as_tensor(required("terminal_mask")).detach().cpu().to(torch.bool)
    if starts_tensor.numel() != len(names):
        raise RuntimeError("Mixed scenario episode-start counters have the wrong width")
    if episode_codes.shape != terminal_codes.shape or episode_codes.shape != terminal_mask.shape:
        raise RuntimeError("Mixed scenario code and terminal-mask buffers disagree")

    starts = [int(value) for value in starts_tensor.to(torch.long).tolist()]
    total_starts = sum(starts)
    current_counts = torch.bincount(episode_codes, minlength=len(names))[:len(names)]
    completed_counts = torch.bincount(
        terminal_codes[terminal_mask], minlength=len(names)
    )[:len(names)]
    return {
        "contract": contract,
        "task_schedule_state": schedule_state,
        "scenario_names_by_code": list(names),
        "episode_starts_by_scenario": dict(zip(names, starts, strict=True)),
        "total_episode_starts": total_starts,
        "episode_start_proportions": {
            name: (count / total_starts if total_starts else 0.0)
            for name, count in zip(names, starts, strict=True)
        },
        "current_environment_counts_by_scenario": {
            name: int(count)
            for name, count in zip(names, current_counts.tolist(), strict=True)
        },
        "last_step_terminal_counts_by_scenario": {
            name: int(count)
            for name, count in zip(names, completed_counts.tolist(), strict=True)
        },
        "episode_scenario_codes": [int(value) for value in episode_codes.tolist()],
        "terminal_scenario_codes": [int(value) for value in terminal_codes.tolist()],
        "terminal_mask": [bool(value) for value in terminal_mask.tolist()],
    }


def _schedule_json_value(value: Any, *, path: str = "state") -> Any:
    """Canonicalize scheduler state without accepting opaque Python objects."""

    if isinstance(value, torch.Tensor):
        return _schedule_json_value(value.detach().cpu().tolist(), path=path)
    if isinstance(value, np.ndarray):
        return _schedule_json_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _schedule_json_value(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"Command schedule {path} contains a nonfinite value")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(
                    f"Command schedule {path} must use non-empty string keys"
                )
            result[key] = _schedule_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _schedule_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeError(
        f"Command schedule {path} contains unsupported type {type(value).__name__}"
    )


def _validate_command_runtime_contract(env: Any, *, task: str) -> None:
    if task not in COMMAND_TASKS:
        return
    runtime = _schedule_json_value(
        getattr(env, "command_tracking_contract", None),
        path="command_tracking_contract",
    )
    expected = _command_contracts(task)[1]
    if runtime != expected:
        raise RuntimeError(
            "Command environment runtime contract differs from its selected task contract"
        )


def _capture_command_task_schedule_state(
    env: Any,
    *,
    task: str,
) -> dict[str, Any] | None:
    """Capture the exact deterministic command cursor at a clean boundary."""

    if task not in COMMAND_TASKS:
        return None
    capture = getattr(env, "command_schedule_state_dict", None)
    if not callable(capture):
        raise RuntimeError(
            f"{task} lacks command_schedule_state_dict()"
        )
    state = _schedule_json_value(capture(), path="command_schedule_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Command schedule state must be a non-empty mapping")
    contract_sha256 = _command_contracts(task)[2]
    if state.get("contract_sha256") != contract_sha256:
        raise RuntimeError("Command schedule state uses a different task contract")
    if state.get("num_envs") != getattr(env, "num_envs", None):
        raise RuntimeError("Command schedule state environment count differs")
    if state.get("training_interactions") != getattr(
        env, "training_interactions", None
    ):
        raise RuntimeError("Command schedule state training clock differs")
    return state


def _restore_command_task_schedule_state(
    env: Any,
    *,
    task: str,
    state: Any,
) -> dict[str, Any] | None:
    """Restore and round-trip the command cursor before the resume reset."""

    if task not in COMMAND_TASKS:
        if state is not None:
            raise RuntimeError(
                "A non-command checkpoint unexpectedly contains command schedule state"
            )
        return None
    normalized = _schedule_json_value(state, path="command_schedule_state")
    if not isinstance(normalized, dict) or not normalized:
        raise RuntimeError("Command checkpoint lacks its schedule state")
    restore = getattr(env, "load_command_schedule_state_dict", None)
    if not callable(restore):
        raise RuntimeError(
            f"{task} lacks load_command_schedule_state_dict()"
        )
    restore(normalized)
    round_trip = _capture_command_task_schedule_state(env, task=task)
    if round_trip != normalized:
        raise RuntimeError("Command schedule state did not round-trip after restore")
    return {
        "schema_version": 1,
        "restored_from_checkpoint": True,
        "restored_before_resume_environment_reset": True,
        "state_sha256": canonical_sha256(normalized),
        "checkpoint_state": normalized,
    }


def _command_schedule_snapshot(
    env: Any,
    *,
    task: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if task not in COMMAND_TASKS:
        return None
    if state is None:
        state = _capture_command_task_schedule_state(env, task=task)
    assert state is not None
    compact, _, contract_sha256 = _command_contracts(task)
    return {
        "command_follow_contract": compact,
        "command_training_contract_sha256": contract_sha256,
        "state_sha256": canonical_sha256(state),
        "state": state,
    }


def _command_schedule_failure_evidence(
    env: Any | None,
    *,
    task: str,
) -> dict[str, Any] | None:
    if task not in COMMAND_TASKS:
        return None
    if env is None:
        return {"capture_error": "environment_not_constructed"}
    try:
        return _command_schedule_snapshot(env, task=task)
    except Exception as exc:  # preserve the original training failure
        return {"capture_error": f"{type(exc).__name__}: {exc}"}


def _set_environment_training_interactions(env: Any, total_interactions: int) -> None:
    """Set the rollout-boundary curriculum counter on the raw drone task."""

    if type(total_interactions) is not int or total_interactions < 0:
        raise ValueError("training interactions must be a non-negative integer")
    setter = getattr(env, "set_training_interactions", None)
    if not callable(setter):
        raise RuntimeError(
            "Crazyflie training environment lacks set_training_interactions; "
            "the versioned task curriculum cannot be selected"
        )
    setter(total_interactions)


def _environment_curriculum_snapshot(
    env: Any,
    *,
    expected_interactions: int,
) -> dict[str, Any]:
    """Fail closed if the task's per-step curriculum clock drifted from PPO work."""

    actual_interactions = getattr(env, "training_interactions", None)
    stage_index = getattr(env, "active_training_curriculum_stage_index", None)
    stage_payload = getattr(env, "active_training_curriculum_stage_payload", None)
    if actual_interactions != expected_interactions:
        raise RuntimeError(
            "Crazyflie curriculum clock disagrees with completed environment interactions: "
            f"expected {expected_interactions}, observed {actual_interactions!r}"
        )
    if type(stage_index) is not int or stage_index < 0 or not isinstance(stage_payload, dict):
        raise RuntimeError("Crazyflie environment returned malformed curriculum stage state")
    stage_name = stage_payload.get("name")
    if stage_name is None and getattr(env, "command_tracking_contract", None) is not None:
        stage_name = f"command_stage_{stage_index}"
    if not isinstance(stage_name, str) or not stage_name:
        raise RuntimeError("Crazyflie environment curriculum stage lacks a stable name")
    return {
        "training_interactions": actual_interactions,
        "active_stage_index": stage_index,
        "active_stage_name": stage_name,
        "active_stage": stage_payload,
    }


def _live_task_telemetry(env: Any, *, task: str) -> dict[str, float]:
    """Return task-appropriate live gauges without inventing goal semantics."""

    robot = getattr(env, "_robot", None)
    data = getattr(robot, "data", None)
    velocity = getattr(data, "root_lin_vel_w", None)
    if not isinstance(velocity, torch.Tensor) or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise RuntimeError("Crazyflie environment lacks [num_envs, 3] world velocity")
    if not bool(torch.isfinite(velocity).all()):
        raise RuntimeError("Crazyflie live velocity telemetry is nonfinite")
    result = {
        "speed_mean_m_s": float(
            torch.linalg.vector_norm(velocity, dim=-1).mean().item()
        )
    }
    if task not in COMMAND_TASKS:
        desired = getattr(env, "_desired_pos_w", None)
        position = getattr(data, "root_pos_w", None)
        if (
            not isinstance(desired, torch.Tensor)
            or not isinstance(position, torch.Tensor)
            or desired.shape != position.shape
            or desired.ndim != 2
            or desired.shape[1] != 3
        ):
            raise RuntimeError("Waypoint task lacks aligned desired/current positions")
        distance = torch.linalg.vector_norm(desired - position, dim=-1)
        if not bool(torch.isfinite(distance).all()):
            raise RuntimeError("Waypoint goal-distance telemetry is nonfinite")
        result["goal_distance_mean_m"] = float(distance.mean().item())
    return result


def standalone_resolved_config(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact fingerprinted config for a non-matrix training job.

    This is public so additive, dry-run-only experiment planners can produce
    commands and fingerprints that the training entrypoint will independently
    reproduce before Isaac starts.
    """

    # Dry-run-only extension planners may pass a minimal Namespace. Keep their
    # implicit identity aligned with the project's active versioned profile.
    contract_profile = getattr(args, "contract_profile", DEFAULT_CONTRACT_PROFILE)
    evaluation_protocol = getattr(args, "evaluation_protocol", "integration")
    _validate_task_contract_selection(
        args.task,
        contract_profile,
        evaluation_protocol,
    )
    protocol = _evaluation_manifest_for_task(args.task, evaluation_protocol)
    contract_field, task_contract = _task_contract_for_profile(
        contract_profile,
        task=args.task,
    )
    resolved: dict[str, Any] = {
        "kind": "standalone_training",
        "task": args.task,
        "contract_profile": contract_profile,
        "controller": args.policy,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "total_interactions": args.total_interactions,
        "horizon": args.horizon,
        "microbatch_size": args.microbatch_size,
        "ppo_epochs": args.ppo_epochs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "value_coefficient": args.value_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "checkpoint_every_updates": args.checkpoint_every_updates,
        "rewire_seed": args.rewire_seed,
        "rewire_manifest": str(args.rewire_manifest),
        "rewire_manifest_file_sha256": sha256_file(args.rewire_manifest),
        "rollout_rng_contract": _rollout_rng_contract(),
        "memory_acceptance": memory_acceptance_contract_payload(),
        contract_field: task_contract,
    }
    if evaluation_protocol != "integration":
        resolved["evaluation_protocol"] = evaluation_protocol
    if args.task in COMMAND_TASKS:
        compact, _, contract_sha256 = _command_contracts(args.task)
        resolved["command_follow_contract"] = compact
        resolved["command_training_contract_sha256"] = contract_sha256
    if args.policy in {"wing_lif", "leg_wing_lif"}:
        resolved["wing_extension"] = {
            "label": "wing",
            "condition": args.policy,
            "leg_connectome": _connectome_artifact_identity(args.connectome_manifest),
            "wing_connectome": _connectome_artifact_identity(args.wing_connectome_manifest),
            "fusion_contract": (
                "independent_leg_and_wing_cores_concat_motor_readouts_v1"
                if args.policy == "leg_wing_lif"
                else None
            ),
            "parameter_matching_required": False,
        }
    if args.policy == "optic_lif":
        resolved["optic_connectome"] = _connectome_artifact_identity(
            getattr(args, "optic_connectome_manifest", DEFAULT_OPTIC_CONNECTOME)
        )
    combination_labels = COMBINATION_LIF_CORE_LABELS.get(args.policy)
    if combination_labels is not None:
        resolved["lif_connectome_composition"] = {
            "schema_version": 1,
            "core_labels": list(combination_labels),
            "connectomes": {
                label: _connectome_artifact_identity(
                    _connectome_manifest_for_label(args, label)
                )
                for label in combination_labels
            },
            "fusion_contract": COMBINATION_LIF_FUSION_CONTRACTS[args.policy],
            "recurrent_cross_core_edges": False,
            "parameter_matching_required": False,
        }
    if args.task == MIXED_TASK:
        resolved["mixed_scenario_contract"] = mixed_scenario_contract_payload(
            seed=args.seed
        )
    if args.task in {"FlyCrazyflie-WaypointSwitch-v0", MIXED_TASK}:
        resolved["switch_target_curriculum"] = _switch_contract_for_profile(
            contract_profile
        )
    if args.warm_start_checkpoint is not None:
        resolved["warm_start_source"] = {
            "absolute_path": str(args.warm_start_checkpoint),
            "sha256": sha256_file(args.warm_start_checkpoint),
            "mode": "actor_trainable_state_only_new_run",
        }
    return resolved, protocol


def _resolved_training_config(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    if args.matrix_config:
        from drone_run_matrix import (
            job_fingerprint,
            resolved_job_config,
            validate_config,
        )

        config = validate_config(args.matrix_config)
        if config.get("rollout_rng_contract") != _rollout_rng_contract():
            raise ValueError("Matrix rollout RNG contract does not match the training entrypoint")
        if (
            config["task"] != args.task
            or config["_contract_profile"] != args.contract_profile
            or args.policy not in config["controllers"]
            or args.seed not in config["seeds"]
            or config["total_interactions"] != args.total_interactions
            or config["training"]["num_envs"] != args.num_envs
            or config["training"]["horizon"] != args.horizon
            or config["training"]["microbatch_size"] != args.microbatch_size
            or config["training"]["ppo_epochs"] != args.ppo_epochs
            or config["training"]["learning_rate"] != args.learning_rate
            or config["training"]["gamma"] != args.gamma
            or config["training"]["gae_lambda"] != args.gae_lambda
            or config["training"]["clip_ratio"] != args.clip_ratio
            or config["training"]["value_coefficient"] != args.value_coefficient
            or config["training"]["entropy_coefficient"] != args.entropy_coefficient
            or config["training"]["max_grad_norm"] != args.max_grad_norm
            or config["training"]["target_kl"] != args.target_kl
            or config["training"]["checkpoint_every_updates"] != args.checkpoint_every_updates
            or config["rewire_seed"] != args.rewire_seed
        ):
            raise ValueError("CLI training cell does not match --matrix_config")
        expected_connectome = (ROOT / config["connectome_manifest"]).resolve()
        if args.connectome_manifest.resolve() != expected_connectome:
            raise ValueError("CLI connectome manifest does not match --matrix_config")
        expected_rewire = (ROOT / config["rewire_manifest"]).resolve()
        if args.rewire_manifest.resolve() != expected_rewire:
            raise ValueError("CLI rewire manifest does not match --matrix_config")
        fingerprint, payload = job_fingerprint(config, args.policy, args.seed)
        resolved = resolved_job_config(config, args.policy, args.seed)
        protocol = config["_evaluation_manifest"]
    else:
        resolved, protocol = standalone_resolved_config(args)
        rewire_manifest = load_fingerprint_rewire_manifest(
            args.rewire_manifest,
            expected_seed=args.rewire_seed,
        )
        fingerprint, payload = reproduction_fingerprint(
            resolved_config=resolved,
            evaluation_manifest=protocol,
            connectome_manifest=_primary_connectome_manifest(args),
            rewired_manifest=rewire_manifest,
        )
    if args.expected_fingerprint and fingerprint != args.expected_fingerprint:
        raise ValueError(
            f"Expected reproduction fingerprint {args.expected_fingerprint}, resolved {fingerprint}; source/config changed"
        )
    return resolved, fingerprint, payload, protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--contract_profile",
        choices=CONTRACT_PROFILES,
        default=DEFAULT_CONTRACT_PROFILE,
        help=(
            "Versioned reward/reset profile. balanced_v3 is the active default; "
            "balanced_v4 is an explicit bounded pilot; survival_v2 remains "
            "available only for compatibility/replay; command_v1 is exclusive "
            "to FlyCrazyflie-CommandFollow-v0 and command_v2 to the two Wide tasks."
        ),
    )
    parser.add_argument("--policy", choices=POLICIES + tuple(POLICY_ALIASES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--total_interactions", type=int)
    parser.add_argument(
        "--horizon", type=int, default=STANDALONE_HORIZON,
        help="Recurrent rollout length (standalone tuned default: 100 control decisions)",
    )
    parser.add_argument("--microbatch_size", type=int)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument(
        "--learning_rate", type=float, default=STANDALONE_LEARNING_RATE,
        help="Adam learning rate (standalone tuned default: 3e-5)",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--value_coefficient", type=float, default=0.5)
    parser.add_argument("--entropy_coefficient", type=float, default=0.002)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--checkpoint_every_updates", type=int, default=100)
    parser.add_argument("--connectome_manifest", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument(
        "--wing_connectome_manifest", type=Path, default=DEFAULT_WING_CONNECTOME
    )
    parser.add_argument(
        "--optic_connectome_manifest", type=Path, default=DEFAULT_OPTIC_CONNECTOME
    )
    parser.add_argument("--rewire_seed", type=int, default=20260916)
    parser.add_argument("--rewire_manifest", type=Path, default=DEFAULT_REWIRE_MANIFEST)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--matrix_config", type=Path)
    parser.add_argument(
        "--evaluation_protocol",
        choices=EVALUATION_PROTOCOLS,
        default="integration",
        help="Fingerprint the fixed held-out protocol used after standalone training",
    )
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--pause_file", type=Path)
    parser.add_argument("--pause_after_updates", type=int, help="Test-only bounded pause at a clean checkpoint boundary")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm_start_checkpoint",
        type=Path,
        help=(
            "Start a new run by loading only compatible actor trainable state; "
            "critic, optimizer, scheduler, RNG, recurrent state, counters, and history reset"
        ),
    )
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.policy is None or args.total_interactions is None or args.run_dir is None:
        parser.error("--policy, --total_interactions, and --run_dir are required")
    args.policy = POLICY_ALIASES.get(args.policy, args.policy)
    if args.microbatch_size is None:
        args.microbatch_size = args.num_envs
    if args.task not in TRAINING_TASKS:
        parser.error(f"Crazyflie training task must be one of {list(TRAINING_TASKS)}")
    try:
        _validate_task_contract_selection(
            args.task,
            args.contract_profile,
            args.evaluation_protocol,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.contract_profile == "balanced_v4" and args.task == MIXED_TASK:
        parser.error("balanced_v4 supports only task-separated Reach, Switch, or Gust training")
    if args.resume and args.warm_start_checkpoint is not None:
        parser.error("--resume and --warm_start_checkpoint are mutually exclusive")
    if args.matrix_config is not None and args.warm_start_checkpoint is not None:
        parser.error("Matrix jobs cannot use actor warm-start; use the dedicated chained-pilot runner")
    integer_fields = (
        args.num_envs,
        args.total_interactions,
        args.horizon,
        args.microbatch_size,
        args.ppo_epochs,
        args.checkpoint_every_updates,
    )
    if any(value < 1 for value in integer_fields):
        parser.error("environment count, budget, horizon, epochs, and checkpoint cadence must be positive")
    if args.microbatch_size != args.num_envs:
        parser.error(
            "This validated recurrent PPO path updates one full vector sequence batch; "
            "--microbatch_size must equal --num_envs"
        )
    if (
        not 0 <= args.seed <= np.iinfo(np.uint32).max
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error(
            "seed must be in [0, 2**32 - 1] and learning rate must be positive and finite"
        )
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("gamma must be in (0, 1] and GAE lambda must be in [0, 1]")
    if not 0.0 < args.clip_ratio < 1.0:
        parser.error("clip ratio must be in (0, 1)")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (args.value_coefficient, args.max_grad_norm)
    ):
        parser.error("value coefficient and max gradient norm must be positive and finite")
    if not math.isfinite(args.entropy_coefficient) or args.entropy_coefficient < 0.0:
        parser.error("entropy coefficient must be finite and nonnegative")
    if not math.isfinite(args.target_kl) or args.target_kl <= 0:
        parser.error("target KL must be positive and finite")
    interactions_per_update = args.num_envs * args.horizon
    if args.total_interactions % interactions_per_update:
        parser.error("--total_interactions must be exactly divisible by num_envs*horizon")
    total_updates = args.total_interactions // interactions_per_update
    if args.pause_after_updates is not None and args.pause_after_updates < 1:
        parser.error("--pause_after_updates must be positive")
    args.connectome_manifest = args.connectome_manifest.resolve()
    if not args.connectome_manifest.is_file():
        parser.error(f"Connectome manifest does not exist: {args.connectome_manifest}")
    args.wing_connectome_manifest = args.wing_connectome_manifest.resolve()
    wing_policies = {
        "wing_lif",
        "leg_wing_lif",
        "wing_optic_lif",
        "leg_wing_optic_lif",
    }
    if args.policy in wing_policies and not args.wing_connectome_manifest.is_file():
        parser.error(f"Wing connectome manifest does not exist: {args.wing_connectome_manifest}")
    args.optic_connectome_manifest = args.optic_connectome_manifest.resolve()
    optic_policies = {
        "optic_lif",
        "leg_optic_lif",
        "wing_optic_lif",
        "leg_wing_optic_lif",
    }
    if args.policy in optic_policies and not args.optic_connectome_manifest.is_file():
        parser.error(f"Optic connectome manifest does not exist: {args.optic_connectome_manifest}")
    args.rewire_manifest = args.rewire_manifest.resolve()
    if not args.rewire_manifest.is_file():
        parser.error(f"Rewire manifest does not exist: {args.rewire_manifest}")
    if args.warm_start_checkpoint is not None:
        args.warm_start_checkpoint = args.warm_start_checkpoint.resolve()
        if not args.warm_start_checkpoint.is_file():
            parser.error(f"Warm-start checkpoint does not exist: {args.warm_start_checkpoint}")
    args.run_dir = args.run_dir.resolve()
    latest_checkpoint = args.run_dir / "checkpoints" / "latest.pt"
    try:
        resolved_config, fingerprint, fingerprint_payload, evaluation_manifest = _resolved_training_config(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": "RESOLVED",
        "resolved_config": resolved_config,
        "fingerprint": fingerprint,
        "outputs": {
            "run_dir": str(args.run_dir),
            "checkpoint": str(latest_checkpoint),
            "training_manifest": str(args.run_dir / "training_manifest.json"),
        },
    }, sort_keys=True), flush=True)

    try:
        archived_attempt = _prepare_run_directory(args.run_dir, resume=args.resume)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if archived_attempt is not None:
        print(json.dumps({
            "status": "incomplete_pre_checkpoint_attempt_archived",
            "archive": str(archived_attempt),
            "new_run_dir": str(args.run_dir),
        }, sort_keys=True), flush=True)

    # Seed Isaac/application construction itself.  A second phase below makes
    # controller initialization independent of any draws consumed at startup.
    _seed_training_streams(args.seed)
    app = AppLauncher(args).app
    env = None
    normalized_env = None
    training_start = time.monotonic()
    # Keep only rows not yet committed to immutable JSONL.  Retaining every
    # nested metric dictionary for a 12,500-update main job caused genuine
    # O(updates) RSS growth even though checkpoints themselves were compact.
    history: list[dict[str, Any]] = []
    history_segments: list[dict[str, Any]] = []
    history_committed_rows = 0
    history_reference: dict[str, Any] | None = None
    memory_samples: list[dict[str, Any]] = []
    last_memory_gate: dict[str, Any] | None = None
    emitted_memory_warnings: set[str] = set()
    counters = None
    controller_report: dict[str, Any] | None = None
    controller_rng_initialization: dict[str, Any] | None = None
    rollout_rng_initialization: dict[str, Any] | None = None
    warm_start_report: dict[str, Any] | None = None
    mixed_schedule_resume: dict[str, Any] | None = None
    command_schedule_resume: dict[str, Any] | None = None
    try:
        from g1_fly_control.crazyflie.checkpoint import (
            ResumeCounters,
            append_validated_history_record_inplace,
            build_history_reference_from_segments,
            capture_rng_states,
            checkpoint_sha256,
            load_checkpoint,
            save_checkpoint_boundary,
            warm_start_actor_from_checkpoint,
            write_history_segment,
        )
        from g1_fly_control.crazyflie.controllers import (
            build_controller,
            controller_core_checksum,
            controller_core_checksums,
            named_lif_cores,
            named_lif_encoders,
            verify_frozen_core,
        )
        from g1_fly_control.crazyflie.lif_activity import (
            RolloutLIFActivityAccumulator,
            scheduled_measurement_due,
        )
        from g1_fly_control.crazyflie.memory import (
            commit_checkpoint_then_release_heap,
            reset_cuda_peak,
            snapshot,
            snapshot_after_heap_release,
        )
        from g1_fly_control.crazyflie.normalization import NormalizedEnv, RunningMeanVariance
        from g1_fly_control.training import PPOConfig, RecurrentPPO

        env = launch_environment(
            args.task,
            args.num_envs,
            deterministic_evaluation=False,
            mixed_scenario_seed=(args.seed if args.task == MIXED_TASK else None),
            command_schedule_seed=(args.seed if args.task in COMMAND_TASKS else None),
            contract_profile=args.contract_profile,
        )
        _validate_command_runtime_contract(env, task=args.task)
        device = torch.device(env.device)

        def assess_current_memory(
            stage: str, *, raise_on_hard_failure: bool = True
        ) -> dict[str, Any]:
            """Assess every sample immediately and emit each warning visibly once."""

            nonlocal last_memory_gate
            last_memory_gate = assess_memory(memory_samples)
            new_warnings = [
                message
                for message in last_memory_gate["warnings"]
                if message not in emitted_memory_warnings
            ]
            if new_warnings:
                emitted_memory_warnings.update(new_warnings)
                print(json.dumps({
                    "status": "MEMORY_WARNING",
                    "policy_version": MEMORY_POLICY_VERSION,
                    "stage": stage,
                    "warnings": new_warnings,
                }, sort_keys=True), flush=True)
            if raise_on_hard_failure:
                _memory_gate_hard_failure(last_memory_gate, stage=stage)
            return last_memory_gate

        reset_cuda_peak(device)
        memory_samples.append(snapshot("environment_loaded", device))
        assess_current_memory("environment_loaded")
        controller_rng_initialization = _initialize_controller_rng(args.seed)
        policy, controller_report = build_controller(
            args.policy,
            observation_dim=12,
            action_dim=4,
            device=device,
            connectome_manifest=args.connectome_manifest,
            wing_connectome_manifest=args.wing_connectome_manifest,
            optic_connectome_manifest=args.optic_connectome_manifest,
            rewire_seed=args.rewire_seed,
            rewire_manifest_path=(
                args.rewire_manifest if args.policy == "frozen_lif_degree_rewired" else None
            ),
        )
        _validate_combination_controller_report(args, controller_report)
        if (
            controller_report["parameter_matching_required"]
            and not controller_report["actor_parameter_match_passed"]
        ):
            raise RuntimeError("Controller actor parameter matching gate failed")
        core_checksum_before = controller_core_checksum(policy)
        per_core_checksums_before = controller_core_checksums(policy)
        lif_encoder_modules = named_lif_encoders(policy)
        runner = RecurrentPPO(
            policy,
            PPOConfig(
                horizon=args.horizon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_ratio=args.clip_ratio,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                learning_rate=args.learning_rate,
                ppo_epochs=args.ppo_epochs,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
            ),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(runner.optimizer, lr_lambda=lambda _: 1.0)
        normalizer = RunningMeanVariance.create(12, device=device)
        normalized_env = NormalizedEnv(
            env,
            normalizer,
            training=True,
            task=args.task,
        )
        counters = ResumeCounters(0, 0)
        reward_normalizer_state = {"enabled": False, "scale": 1.0, "reason": "identity shared across conditions"}
        fingerprint_map = {
            "reproduction": fingerprint,
            "source_set": canonical_sha256(fingerprint_payload["source_sha256"]),
            "connectome": controller_report["connectome_checksum"],
            "frozen_core": controller_report["core_checksum"],
            "rewire_manifest": fingerprint_payload["rewired_manifest_sha256"],
        }
        contract_field, task_contract = _task_contract_for_profile(
            args.contract_profile,
            task=args.task,
        )
        task_manifest_payload: dict[str, Any] = {
            "task": args.task, "episode_steps": 600, "control_dt_s": 0.02,
            "observation_width": 12, "action_width": 4,
            "contract_profile": args.contract_profile,
            contract_field: task_contract,
        }
        if args.task not in COMMAND_TASKS:
            # Preserve the established legacy-task manifest shape while keeping
            # command-task provenance free of an inapplicable scenario field.
            task_manifest_payload["mixed_scenario_contract"] = (
                mixed_scenario_contract_payload(seed=args.seed)
                if args.task == MIXED_TASK else None
            )
        if args.task in COMMAND_TASKS:
            compact, _, contract_sha256 = _command_contracts(args.task)
            task_manifest_payload["command_follow_contract"] = compact
            task_manifest_payload["command_training_contract_sha256"] = (
                contract_sha256
            )
        if args.task in {"FlyCrazyflie-WaypointSwitch-v0", MIXED_TASK}:
            task_manifest_payload["switch_target_curriculum"] = (
                _switch_contract_for_profile(args.contract_profile)
            )
        task_manifest_id = canonical_sha256(task_manifest_payload)
        evaluation_manifest_id = evaluation_manifest["manifest_id"]
        if args.task == MIXED_TASK:
            mixed_schedule_resume = {
                "schema_version": 1,
                "restored_from_checkpoint": False,
                "restored_before_resume_environment_reset": False,
                "checkpoint_state": None,
            }
        if args.task in COMMAND_TASKS:
            command_schedule_resume = {
                "schema_version": 1,
                "restored_from_checkpoint": False,
                "restored_before_resume_environment_reset": False,
                "state_sha256": None,
                "checkpoint_state": None,
            }
        if args.warm_start_checkpoint is not None:
            if runner.optimizer.state:
                raise RuntimeError("Fresh warm-start optimizer unexpectedly contains state")
            warm_start_report = warm_start_actor_from_checkpoint(
                args.warm_start_checkpoint,
                policy=policy,
                expected_controller=args.policy,
                expected_connectome_checksum=controller_report["connectome_checksum"],
                expected_frozen_core_checksum=core_checksum_before,
                map_location="cpu",
            )
            warm_start_report["target_context"] = {
                "task": args.task,
                "controller": args.policy,
                "reproduction_fingerprint": fingerprint,
                "resolved_config_sha256": canonical_sha256(resolved_config),
                "critic_state_preserved_from_fresh_initialization": True,
                "optimizer_state_entry_count": len(runner.optimizer.state),
                "scheduler_state_keys_reset": sorted(scheduler.state_dict()),
                "observation_normalizer_reset": True,
                "reward_normalizer_reset": True,
                "counters_reset": counters.as_dict(),
                "history_row_count_reset": len(history),
                "recurrent_state_reset": True,
                "source_rng_state_loaded": False,
            }
        if args.resume:
            loaded = load_checkpoint(
                latest_checkpoint,
                policy=policy,
                optimizer=runner.optimizer,
                scheduler=scheduler,
                map_location=device,
                expected_fingerprints=fingerprint_map,
                expected_config=resolved_config,
                expected_task_manifest_id=task_manifest_id,
                expected_evaluation_manifest_id=evaluation_manifest_id,
                for_resume=True,
                restore_rng=True,
                reset_environments_on_resume=True,
                materialize_external_history=False,
            )
            counters = ResumeCounters.from_value(loaded["resume_counters"])
            history_reference = loaded.get("history_reference")
            if not isinstance(history_reference, dict):
                raise RuntimeError("Resume checkpoint lacks its external history reference")
            history_segments = [dict(segment) for segment in history_reference["segments"]]
            history_committed_rows = int(history_reference["row_count"])
            if (
                history_committed_rows != counters.completed_updates
                or history_reference["last_completed_updates"] != counters.completed_updates
                or history_reference["last_total_interactions"] != counters.total_interactions
                or loaded.get("history") != []
            ):
                raise RuntimeError("Resume checkpoint history cursor is not at the saved boundary")
            normalizer.load_state_dict(loaded["normalizers"]["observation"])
            prior_memory_samples = loaded.get("metadata", {}).get("memory_samples", [])
            if not isinstance(prior_memory_samples, list) or not all(
                isinstance(sample, dict) for sample in prior_memory_samples
            ):
                raise RuntimeError("Checkpoint memory sample history is malformed")
            # Memory failures are sticky across practical restarts.  A resume
            # cannot discard the measurements that justified an earlier gate.
            memory_samples = [*prior_memory_samples, *memory_samples]
            restored_caller_state = loaded.get("restored_caller_rng_states")
            if not isinstance(restored_caller_state, Mapping):
                raise RuntimeError("Resume checkpoint lacks restored caller-owned RNG state")
            restored_task_schedule = restored_caller_state.get("task_schedule")
            if not isinstance(restored_task_schedule, Mapping):
                raise RuntimeError("Resume checkpoint task schedule state is malformed")
            mixed_schedule_resume = _restore_mixed_task_schedule_state(
                env,
                task=args.task,
                seed=args.seed,
                state=restored_task_schedule.get("mixed_task_schedule"),
            )
            command_schedule_resume = _restore_command_task_schedule_state(
                env,
                task=args.task,
                state=restored_task_schedule.get("command_task_schedule"),
            )
            # DirectRLEnv simulator state is deliberately restarted with its
            # recurrent state; counted work remains exact and resume_reset is visible.
            runner._observation = None
            runner._state = None
            runner._reset_before = None
        memory_samples.append(snapshot("controller_loaded", device))
        assess_current_memory("controller_loaded")
        if counters.total_interactions > args.total_interactions:
            raise RuntimeError("Checkpoint has more interactions than the requested exact budget")
        if counters.total_interactions % interactions_per_update:
            raise RuntimeError("Checkpoint interaction counter is not on the configured rollout boundary")

        # Fresh controller construction consumes an architecture-dependent
        # number of random draws.  Reset all rollout streams only after that
        # construction.  On resume, load_checkpoint already restored the exact
        # saved streams, so this helper is intentionally a no-op.
        rollout_rng_initialization = _initialize_rollout_rng(
            args.seed,
            resume_restored=args.resume,
        )
        if warm_start_report is not None:
            if not rollout_rng_initialization["fresh_reseed_applied"]:
                raise RuntimeError("Actor warm-start must use freshly reset rollout RNG streams")
            warm_start_report["target_context"]["rollout_rng_reset"] = dict(
                rollout_rng_initialization
            )
        # Set the restored/fresh boundary before the initial restart
        # checkpoint. RecurrentPPO's first collect performs the explicit reset,
        # so a resumed environment cannot accidentally reset under stage zero.
        _set_environment_training_interactions(env, counters.total_interactions)
        _environment_curriculum_snapshot(
            env,
            expected_interactions=counters.total_interactions,
        )

        def _commit_boundary(*, status: str) -> Path:
            nonlocal history_committed_rows, history_reference, history_segments
            assert counters is not None
            curriculum_snapshot = _environment_curriculum_snapshot(
                env,
                expected_interactions=counters.total_interactions,
            )
            command_schedule_state = _capture_command_task_schedule_state(
                env,
                task=args.task,
            )
            command_schedule_snapshot = _command_schedule_snapshot(
                env,
                task=args.task,
                state=command_schedule_state,
            )
            metadata = {
                "status": status,
                "contract_profile": args.contract_profile,
                "controller": args.policy,
                "seed": args.seed,
                "run_dir": str(args.run_dir),
                "requested_interactions": args.total_interactions,
                "interactions_per_update": interactions_per_update,
                "controller_report": controller_report,
                "core_checksum_before": core_checksum_before,
                "core_checksum_after": controller_core_checksum(policy),
                "per_core_checksums_before": per_core_checksums_before,
                "per_core_checksums_after": controller_core_checksums(policy),
                "memory_samples": memory_samples,
                "elapsed_wall_time_s": time.monotonic() - training_start,
                "resume_reset": counters.resume_reset,
                "controller_rng_initialization": controller_rng_initialization,
                "rollout_rng_initialization": rollout_rng_initialization,
                "training_curriculum": curriculum_snapshot,
                "mixed_scenario": _mixed_scenario_snapshot(
                    env, task=args.task, seed=args.seed
                ),
                "mixed_schedule_resume": mixed_schedule_resume,
                "command_schedule": command_schedule_snapshot,
                "command_schedule_resume": command_schedule_resume,
                "warm_start": warm_start_report,
            }
            task_schedule_rng_state: dict[str, Any] = {
                "episode_length_buf": env.episode_length_buf.detach().cpu(),
                "training_curriculum": curriculum_snapshot,
                "mixed_task_schedule": _capture_mixed_task_schedule_state(
                    env, task=args.task, seed=args.seed
                ),
                "command_task_schedule": command_schedule_state,
            }
            if args.task not in COMMAND_TASKS:
                task_schedule_rng_state["desired_position"] = getattr(
                    env, "_desired_pos_w", torch.empty(0)
                ).detach().cpu()
            rng_states = capture_rng_states(
                environment={
                    "seed": args.seed,
                    "episode_length_buf": env.episode_length_buf.detach().cpu(),
                },
                task_schedule=task_schedule_rng_state,
                include_cuda=torch.cuda.is_available(),
            )
            numbered = args.run_dir / "checkpoints" / f"update-{counters.completed_updates:08d}.pt"
            if history:
                start_row = history_committed_rows + 1
                end_row = history_committed_rows + len(history)
                segment_path = (
                    args.run_dir / "history"
                    / (
                        f"rows-{start_row:08d}-{end_row:08d}"
                        f"-resume-{counters.resume_count:04d}-{time.time_ns()}.jsonl"
                    )
                )
                segment = write_history_segment(
                    segment_path,
                    history,
                    reference_directory=numbered.parent,
                )
                history_segments.append(segment)
                history_committed_rows = end_row
                history.clear()
            history_reference = build_history_reference_from_segments(
                history_segments,
                checkpoint_path=numbered,
                interactions_per_update=interactions_per_update,
            )
            if history_committed_rows != counters.completed_updates:
                raise RuntimeError(
                    "Committed history rows do not align with completed PPO updates"
                )
            checkpoint_kwargs = {
                "policy": policy,
                "optimizer": runner.optimizer,
                "scheduler": scheduler,
                "observation_normalizer_state": normalizer.state_dict(),
                "reward_normalizer_state": reward_normalizer_state,
                "counters": counters,
                "recurrent_state": runner._state,
                "rng_states": rng_states,
                "resolved_config": resolved_config,
                "command": sys.argv,
                "task_manifest_id": task_manifest_id,
                "evaluation_manifest_id": evaluation_manifest_id,
                "fingerprints": fingerprint_map,
                "history": history,
                "history_reference": history_reference,
                "metadata": metadata,
                "interactions_per_update": interactions_per_update,
            }
            # latest.pt is the authoritative restart boundary.  Commit it
            # first; if the process stops before the archival snapshot, a
            # resume recreates the missing numbered file from this boundary.
            save_checkpoint_boundary(latest_checkpoint, numbered, **checkpoint_kwargs)
            return latest_checkpoint

        def save_boundary(*, status: str) -> Path:
            # Let the heavy commit frame (metadata, RNG snapshot, and
            # serialization arguments) unwind before collecting and trimming.
            return commit_checkpoint_then_release_heap(
                lambda: _commit_boundary(status=status)
            )

        save_boundary(
            status=(
                "resumed" if args.resume
                else "warm_started" if warm_start_report is not None
                else "initialized"
            )
        )

        while counters.completed_updates < total_updates:
            update_started = time.monotonic()
            next_update = counters.completed_updates + 1
            rollout_start_interactions = counters.total_interactions
            measure_memory = scheduled_measurement_due(
                next_update,
                total_updates=total_updates,
                checkpoint_every_updates=args.checkpoint_every_updates,
            )
            if measure_memory:
                reset_cuda_peak(device)
            lif_activity: dict[str, Any] | None = None
            lif_cores = named_lif_cores(policy)
            lif_activity_sampled = bool(lif_cores) and measure_memory
            if lif_activity_sampled:
                from contextlib import ExitStack

                activity_accumulators = {
                    label: RolloutLIFActivityAccumulator(
                        core,
                        horizon=args.horizon,
                        num_envs=args.num_envs,
                        control_dt_s=0.02,
                    )
                    for label, core in lif_cores
                }
                with ExitStack() as activity_context:
                    for accumulator in activity_accumulators.values():
                        activity_context.enter_context(accumulator)
                    # Confirm the persisted rollout boundary before collection.
                    # The raw task advances this clock by num_envs on every
                    # real step, before any same-step automatic reset.
                    _set_environment_training_interactions(env, rollout_start_interactions)
                    rollout, state, _ = runner.collect(normalized_env)
                # Validate before PPO consumes the rollout.  The summary makes
                # exactly one bounded device-to-host transfer and fails closed
                # on nonfinite/nonbinary state or unexpected call boundaries.
                lif_activity = _summarize_lif_activity(
                    policy, activity_accumulators
                )
            else:
                _set_environment_training_interactions(env, rollout_start_interactions)
                rollout, state, _ = runner.collect(normalized_env)
            rollout_curriculum = _environment_curriculum_snapshot(
                env,
                expected_interactions=rollout_start_interactions + interactions_per_update,
            )
            if measure_memory:
                # Compare like-for-like live state.  Checkpoint serialization
                # from the preceding boundary can leave unreachable glibc
                # arenas cached even though no Python object retains them.
                memory_samples.append(
                    snapshot_after_heap_release("rollout", device, step=next_update)
                )
                # A hard cap, missing CUDA telemetry, nonfinite telemetry, or
                # sustained paging stops before an optimizer step can allocate
                # more memory.  The latest.pt boundary remains the last clean,
                # resumable checkpoint and the exception path persists this
                # sample plus its recomputed gate.
                assess_current_memory("rollout")
            metrics = runner.update(rollout)
            scheduler.step()
            verify_frozen_core(policy, core_checksum_before)
            done = rollout.terminated | rollout.truncated
            episode_metrics = normalized_env.consume_training_metrics()
            nonfinite_failure_count = _nonfinite_failure_count(episode_metrics)
            rollout_completed_episodes = int(done.sum().item())
            if episode_metrics["completed_episode_count"] != rollout_completed_episodes:
                raise RuntimeError("Completed-episode instrumentation disagrees with rollout done masks")
            counters = counters.advanced(
                rollout_interactions=interactions_per_update,
                completed_episodes=rollout_completed_episodes,
            )
            action_deltas = rollout.actions[1:] - rollout.actions[:-1]
            row: dict[str, Any] = {
                "completed_updates": counters.completed_updates,
                "total_interactions": counters.total_interactions,
                "rollout_start_interactions": rollout_start_interactions,
                "training_curriculum_interactions": rollout_curriculum["training_interactions"],
                "active_training_curriculum_stage_index": rollout_curriculum["active_stage_index"],
                "active_training_curriculum_stage_name": rollout_curriculum["active_stage_name"],
                "completed_episodes": counters.completed_episodes,
                "mean_rollout_reward": float(rollout.rewards.mean()),
                "sum_rollout_reward": float(rollout.rewards.sum()),
                "action_mean": float(rollout.actions.mean()),
                "action_mean_abs": float(rollout.actions.abs().mean()),
                "action_saturation_fraction": float((rollout.actions.abs() >= 0.99).float().mean()),
                "action_change_mean_l2": float(torch.linalg.vector_norm(action_deltas, dim=-1).mean()) if action_deltas.numel() else 0.0,
                "updates_per_second": 1.0 / max(time.monotonic() - update_started, 1e-9),
                "interactions_per_second": interactions_per_update / max(time.monotonic() - update_started, 1e-9),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                **episode_metrics,
                **metrics,
            }
            row.update(_live_task_telemetry(env, task=args.task))
            if hasattr(policy, "encoder"):
                row["encoder_gradient_norm"] = _gradient_norm(policy.encoder)
                if hasattr(policy, "wing_encoder"):
                    row["wing_encoder_gradient_norm"] = _gradient_norm(policy.wing_encoder)
                if lif_encoder_modules:
                    per_core_encoder_gradients = {
                        label: _gradient_norm(encoder)
                        for label, encoder in lif_encoder_modules
                    }
                    row["per_core_encoder_gradient_norms"] = (
                        per_core_encoder_gradients
                    )
                    for label, value in per_core_encoder_gradients.items():
                        if label != lif_encoder_modules[0][0]:
                            row[f"{label}_encoder_gradient_norm"] = value
                row["decoder_gradient_norm"] = _gradient_norm(policy.decoder)
                row["lif_activity_sampled"] = lif_activity_sampled
                row["lif_activity"] = lif_activity
            mixed_scenario_snapshot = _mixed_scenario_snapshot(
                env, task=args.task, seed=args.seed
            )
            if mixed_scenario_snapshot is not None:
                row["mixed_scenario"] = mixed_scenario_snapshot
            components = getattr(env, "reward_components", None)
            if isinstance(components, dict):
                row["last_step_reward_components"] = {
                    key: float(torch.as_tensor(value).mean()) for key, value in components.items()
                }
            append_validated_history_record_inplace(history, row)
            if measure_memory:
                memory_samples.append(
                    snapshot_after_heap_release(
                        "optimizer_update", device, step=counters.completed_updates
                    )
                )
                optimizer_memory_gate = assess_current_memory(
                    "optimizer_update", raise_on_hard_failure=False
                )
            else:
                optimizer_memory_gate = None
            print(json.dumps(row, sort_keys=True), flush=True)
            if nonfinite_failure_count:
                save_boundary(status="failed_nonfinite_state")
                _raise_for_nonfinite_failure_count(
                    nonfinite_failure_count,
                    completed_updates=counters.completed_updates,
                )
            # The sample was assessed immediately above.  Save the failed
            # completed-update boundary before propagating the hard failure.
            if optimizer_memory_gate is not None and optimizer_memory_gate["passed"] is not True:
                save_boundary(status="failed_memory_gate")
                _memory_gate_hard_failure(
                    optimizer_memory_gate, stage="optimizer_update"
                )
            if counters.completed_updates % args.checkpoint_every_updates == 0:
                save_boundary(status="running")
            pause_requested = args.pause_file is not None and args.pause_file.exists()
            bounded_pause = (
                args.pause_after_updates is not None
                and counters.completed_updates >= args.pause_after_updates
                and counters.completed_updates < total_updates
            )
            if pause_requested or bounded_pause:
                save_boundary(status="paused")
                memory_gate = assess_memory(memory_samples)
                manifest = {
                    "schema_version": 1,
                    "status": "paused",
                    "contract_profile": args.contract_profile,
                    "controller": args.policy,
                    "seed": args.seed,
                    "environment_interactions": counters.total_interactions,
                    "requested_interactions": args.total_interactions,
                    "completed_updates": counters.completed_updates,
                    "resume_reset": counters.resume_reset,
                    "checkpoint": str(latest_checkpoint),
                    "checkpoint_sha256": checkpoint_sha256(latest_checkpoint),
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                    "resolved_config": resolved_config,
                    "controller_report": controller_report,
                    "controller_rng_initialization": controller_rng_initialization,
                    "rollout_rng_initialization": rollout_rng_initialization,
                    "warm_start": warm_start_report,
                    "training_curriculum": _environment_curriculum_snapshot(
                        env,
                        expected_interactions=counters.total_interactions,
                    ),
                    "mixed_scenario": _mixed_scenario_snapshot(
                        env, task=args.task, seed=args.seed
                    ),
                    "mixed_schedule_resume": mixed_schedule_resume,
                    "command_schedule": _command_schedule_snapshot(
                        env, task=args.task
                    ),
                    "command_schedule_resume": command_schedule_resume,
                    "memory_gate": memory_gate,
                    "history_reference": history_reference,
                    "core_checksum_before": core_checksum_before,
                    "core_checksum_after": controller_core_checksum(policy),
                    "per_core_checksums_before": per_core_checksums_before,
                    "per_core_checksums_after": controller_core_checksums(policy),
                }
                _atomic_json(args.run_dir / "training_manifest.json", manifest)
                print(json.dumps(manifest, indent=2, sort_keys=True))
                return 3

        memory_samples.append(snapshot("training_complete", device, step=counters.completed_updates))
        try:
            memory_gate = assess_current_memory("training_complete")
        except RuntimeError:
            save_boundary(status="failed_memory_gate")
            raise
        save_boundary(status="completed")
        verify_frozen_core(policy, core_checksum_before)
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "task": args.task,
            "contract_profile": args.contract_profile,
            "controller": args.policy,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "horizon": args.horizon,
            "requested_interactions": args.total_interactions,
            "environment_interactions": counters.total_interactions,
            "completed_updates": counters.completed_updates,
            "completed_episodes": counters.completed_episodes,
            "resume_count": counters.resume_count,
            "resume_reset": counters.resume_reset,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_sha256": checkpoint_sha256(latest_checkpoint),
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "resolved_config": resolved_config,
            "task_manifest_id": task_manifest_id,
            "evaluation_manifest_id": evaluation_manifest_id,
            "controller_report": controller_report,
            "controller_rng_initialization": controller_rng_initialization,
            "rollout_rng_initialization": rollout_rng_initialization,
            "warm_start": warm_start_report,
            "training_curriculum": _environment_curriculum_snapshot(
                env,
                expected_interactions=counters.total_interactions,
            ),
            "mixed_scenario": _mixed_scenario_snapshot(
                env, task=args.task, seed=args.seed
            ),
            "mixed_schedule_resume": mixed_schedule_resume,
            "command_schedule": _command_schedule_snapshot(
                env, task=args.task
            ),
            "command_schedule_resume": command_schedule_resume,
            "core_checksum_before": core_checksum_before,
            "core_checksum_after": controller_core_checksum(policy),
            "per_core_checksums_before": per_core_checksums_before,
            "per_core_checksums_after": controller_core_checksums(policy),
            "ppo": runner.config_dict(),
            "observation_normalizer": normalizer.state_dict(),
            "reward_normalizer": reward_normalizer_state,
            "history_reference": history_reference,
            "memory_samples": memory_samples,
            "memory_gate": memory_gate,
            "training_wall_time_s": time.monotonic() - training_start,
            "command": sys.argv,
        }
        # Tensor-valued normalizer entries are converted for JSON only; the
        # exact tensors remain in the checkpoint.
        manifest["observation_normalizer"] = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in manifest["observation_normalizer"].items()
        }
        _atomic_json(args.run_dir / "training_manifest.json", manifest)
        print(json.dumps({
            "status": "PASS",
            "run_dir": str(args.run_dir),
            "checkpoint": str(latest_checkpoint),
            "training_manifest": str(args.run_dir / "training_manifest.json"),
            "environment_interactions": counters.total_interactions,
            "completed_updates": counters.completed_updates,
            "training_curriculum": manifest["training_curriculum"],
            "fingerprint": fingerprint,
            "resolved_config": resolved_config,
            "parameter_counts": controller_report,
            "memory_gate": memory_gate,
        }, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        failure_runtime_evidence = _failure_runtime_evidence(
            counters=counters,
            history_reference=history_reference,
            pending_history_rows=len(history),
            memory_samples=memory_samples,
        )
        failure = {
            "schema_version": 1,
            "status": "failed",
            "task": args.task,
            "controller": args.policy,
            "seed": args.seed,
            "requested_interactions": args.total_interactions,
            "environment_interactions": counters.total_interactions if counters is not None else 0,
            "checkpoint": str(latest_checkpoint) if latest_checkpoint.is_file() else None,
            "checkpoint_is_last_clean_boundary": latest_checkpoint.is_file(),
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "resolved_config": resolved_config,
            "controller_report": controller_report,
            "controller_rng_initialization": controller_rng_initialization,
            "rollout_rng_initialization": rollout_rng_initialization,
            "warm_start": warm_start_report,
            "mixed_scenario": (
                _mixed_scenario_snapshot(env, task=args.task, seed=args.seed)
                if env is not None else None
            ),
            "mixed_schedule_resume": mixed_schedule_resume,
            "command_schedule": _command_schedule_failure_evidence(
                env, task=args.task
            ),
            "command_schedule_resume": command_schedule_resume,
            **failure_runtime_evidence,
            "error": traceback.format_exc(),
        }
        manifest_path = args.run_dir / "training_manifest.json"
        if args.resume and manifest_path.exists():
            # A rejected/corrupt resume must not erase the last valid paused
            # or completed run record.  Retain each failed attempt separately.
            failure_path = (
                args.run_dir
                / "resume_failures"
                / f"resume-failure-{time.time_ns()}.json"
            )
        else:
            failure_path = manifest_path
        failure["failure_artifact"] = str(failure_path)
        failure["preserved_prior_training_manifest"] = bool(
            args.resume and manifest_path.exists()
        )
        _atomic_json(failure_path, failure)
        print(json.dumps({
            "status": "FAIL",
            "run_dir": str(args.run_dir),
            "failure_artifact": str(failure_path),
            "checkpoint": failure["checkpoint"],
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "failure_reason": failure["error"],
        }, indent=2, sort_keys=True))
        return 1
    finally:
        if normalized_env is not None:
            normalized_env.close()
            env = None
        elif env is not None:
            env.close()
        # Isaac Sim 5.1's native framework shutdown terminates the process
        # with status 0.  The environment is already closed above; the
        # entrypoint flushes output and uses os._exit to preserve our status.


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
