#!/usr/bin/env python3
"""Evaluate one trained command-follow controller on the fixed 16x600 protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from typing import Any, Callable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from g1_fly_control.crazyflie.command_evaluation import (  # noqa: E402
    CommandRollout,
    EPISODE_COUNT,
    EPISODE_STEPS,
    SEGMENT_NAMES,
    SEGMENT_STEPS,
    batched_held_out_commands as batched_v1_held_out_commands,
    protocol_payload as protocol_v1_payload,
    protocol_sha256 as protocol_v1_sha256,
    score_command_rollout as score_command_rollout_v1,
)
from g1_fly_control.crazyflie.command_wide_evaluation import (  # noqa: E402
    batched_held_out_commands as batched_v2_held_out_commands,
    protocol_payload as protocol_v2_payload,
    protocol_sha256 as protocol_v2_sha256,
    score_command_rollout_v2,
)
from g1_fly_control.crazyflie.trained_keyboard import (  # noqa: E402
    SUPPORTED_CONTROLLER_KINDS,
    TASK_ID,
    TASK_IDS,
    WIDE_TASK_ID,
    WIDE_WIND_TASK_ID,
    CommandCheckpointSpec,
    TrainedActivityRecorder,
    initial_policy_state,
    inspect_command_checkpoint,
    load_command_controller,
)


CONTRACT_PROFILE_V1 = "command_v1"
CONTRACT_PROFILE_V2 = "command_v2"
EVALUATION_PROTOCOL_V1 = "command_v1"
EVALUATION_PROTOCOL_V2 = "command_v2"
ACTIVITY_THRESHOLD = 1.0e-6
INFERENCE_LATENCY_WARMUP_CALLS = 10
_LAST_MEMORY_SAMPLES: list[dict[str, Any]] = []
_LAST_MEMORY_GATE: dict[str, Any] | None = None


class AuthenticatedInferenceLatency:
    """Measure the exact policy calls used by the evaluation control path.

    CUDA work is asynchronous, so a CUDA evaluation synchronizes immediately
    before and after each ``policy.act`` call.  The first few calls are retained
    in the all-call total but excluded from steady-state statistics because they
    can include one-time lazy CUDA/kernel initialization.  No extra benchmark
    forwards are executed, which keeps the policy state and evaluation protocol
    unchanged.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        batch_size: int,
        warmup_calls: int = INFERENCE_LATENCY_WARMUP_CALLS,
        clock_ns: Callable[[], int] | None = None,
        cuda_synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(warmup_calls) is not int or warmup_calls < 0:
            raise ValueError("warmup_calls must be a nonnegative integer")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.warmup_calls = warmup_calls
        self._clock_ns = time.perf_counter_ns if clock_ns is None else clock_ns
        self._cuda_synchronize = (
            torch.cuda.synchronize if cuda_synchronize is None else cuda_synchronize
        )
        self._started_ns: int | None = None
        self._durations_ns: list[int] = []

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self._cuda_synchronize(self.device)

    def begin_call(self) -> None:
        """Start one action-producing forward measurement."""

        if self._started_ns is not None:
            raise RuntimeError("an inference latency measurement is already active")
        self._synchronize()
        started = self._clock_ns()
        if type(started) is not int or started < 0:
            raise RuntimeError("perf_counter_ns returned an invalid timestamp")
        self._started_ns = started

    def end_call(self) -> None:
        """Finish one successfully returned action-producing forward."""

        if self._started_ns is None:
            raise RuntimeError("no inference latency measurement is active")
        self._synchronize()
        ended = self._clock_ns()
        elapsed = ended - self._started_ns
        self._started_ns = None
        if type(ended) is not int or elapsed < 0:
            raise RuntimeError("perf_counter_ns was not monotonic")
        self._durations_ns.append(elapsed)

    @staticmethod
    def _percentile(sorted_values: list[int], quantile: float) -> float:
        """Return a linearly interpolated percentile in integer-clock units."""

        if not sorted_values:
            raise ValueError("cannot calculate a percentile without samples")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        position = (len(sorted_values) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(sorted_values[lower])
        fraction = position - lower
        return float(
            sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * fraction
        )

    def report(self, *, expected_call_count: int) -> dict[str, Any]:
        """Build a finite, fixed-schema latency report in milliseconds."""

        if self._started_ns is not None:
            raise RuntimeError("an inference latency measurement is still active")
        if type(expected_call_count) is not int or expected_call_count <= 0:
            raise ValueError("expected_call_count must be a positive integer")
        call_count = len(self._durations_ns)
        if call_count != expected_call_count:
            raise RuntimeError(
                "inference latency call count mismatch: "
                f"observed {call_count}, expected {expected_call_count}"
            )
        excluded = min(self.warmup_calls, call_count)
        samples = sorted(self._durations_ns[excluded:])
        if not samples:
            raise RuntimeError("warmup exclusion left no inference latency samples")
        nanoseconds_to_milliseconds = 1.0e-6
        total_ms = float(sum(samples) * nanoseconds_to_milliseconds)
        report = {
            "schema_version": 1,
            "source": "exact_policy_act_calls_used_by_evaluation_control_path",
            "call_site": (
                "policy.act(normalized_observation, recurrent_state, "
                "deterministic=True)"
            ),
            "clock": "time.perf_counter_ns_monotonic",
            "device": str(self.device),
            "measurement_unit": "milliseconds_per_vectorized_policy_call",
            "batch_size": self.batch_size,
            "cuda_synchronized_before_and_after_call": self.device.type == "cuda",
            "activity_recorder_hooks_in_scope": True,
            "action_producing_call_count": call_count,
            "warmup_calls_excluded_from_statistics": excluded,
            "warmup_exclusion_justification": (
                "exclude prefix calls that can include one-time lazy CUDA/kernel "
                "initialization; no extra forwards were executed"
            ),
            "sample_count": len(samples),
            "total_ms": total_ms,
            "mean_ms": total_ms / len(samples),
            "p50_ms": self._percentile(samples, 0.50)
            * nanoseconds_to_milliseconds,
            "p95_ms": self._percentile(samples, 0.95)
            * nanoseconds_to_milliseconds,
            "p99_ms": self._percentile(samples, 0.99)
            * nanoseconds_to_milliseconds,
            "max_ms": float(samples[-1] * nanoseconds_to_milliseconds),
            "all_action_producing_calls_total_ms": float(
                sum(self._durations_ns) * nanoseconds_to_milliseconds
            ),
        }
        numeric_keys = (
            "total_ms",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "all_action_producing_calls_total_ms",
        )
        if not all(
            math.isfinite(report[key]) and report[key] >= 0.0
            for key in numeric_keys
        ):
            raise RuntimeError("inference latency report contains an invalid value")
        if not (
            report["p50_ms"]
            <= report["p95_ms"]
            <= report["p99_ms"]
            <= report["max_ms"]
        ):
            raise RuntimeError("inference latency percentiles are inconsistent")
        return report


def _canonical_sha256(value: Any) -> str:
    from hashlib import sha256

    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _profile_for_task(task: str) -> str:
    if task == TASK_ID:
        return CONTRACT_PROFILE_V1
    if task in {WIDE_TASK_ID, WIDE_WIND_TASK_ID}:
        return CONTRACT_PROFILE_V2
    raise ValueError(f"unsupported command task {task!r}")


def _protocol_name_for_task(task: str) -> str:
    return EVALUATION_PROTOCOL_V1 if task == TASK_ID else EVALUATION_PROTOCOL_V2


def _protocol_payload_for_task(task: str) -> dict[str, Any]:
    return protocol_v1_payload() if task == TASK_ID else protocol_v2_payload(task=task)


def _protocol_sha256_for_task(task: str) -> str:
    return protocol_v1_sha256() if task == TASK_ID else protocol_v2_sha256(task=task)


def expected_manifest_ids(
    task: str = TASK_ID,
) -> tuple[str, str, dict[str, Any]]:
    """Rebuild the task/evaluation identities stored by ``drone_train.py``."""

    if task == TASK_ID:
        from g1_fly_control.tasks.crazyflie.command_logic import (
            COMMAND_TRACKING_CONTRACT_SHA256,
            command_follow_contract_payload,
            command_training_contract_payload,
        )

        compact = command_follow_contract_payload()
        complete = command_training_contract_payload()
        contract_sha256 = COMMAND_TRACKING_CONTRACT_SHA256
    elif task in {WIDE_TASK_ID, WIDE_WIND_TASK_ID}:
        from g1_fly_control.tasks.crazyflie.command_wide_logic import (
            command_wide_contract_payload,
            command_wide_training_contract_payload,
            command_wide_training_contract_sha256,
        )

        wind_enabled = task == WIDE_WIND_TASK_ID
        compact = command_wide_contract_payload()
        complete = command_wide_training_contract_payload(
            wind_enabled=wind_enabled
        )
        contract_sha256 = command_wide_training_contract_sha256(
            wind_enabled=wind_enabled
        )
    else:
        raise ValueError(f"unsupported command task {task!r}")
    if _canonical_sha256(complete) != contract_sha256:
        raise RuntimeError("current command-training contract checksum is inconsistent")
    contract_profile = _profile_for_task(task)
    evaluation_protocol = _protocol_name_for_task(task)
    task_payload = {
        "task": task,
        "episode_steps": EPISODE_STEPS,
        "control_dt_s": 0.02,
        "observation_width": 12,
        "action_width": 4,
        "contract_profile": contract_profile,
        "command_training_contract": complete,
        "command_follow_contract": compact,
        "command_training_contract_sha256": contract_sha256,
    }
    evaluation = {
        "schema_version": 1,
        "protocol": evaluation_protocol,
        "task": task,
        "evaluation_protocol": _protocol_payload_for_task(task),
        "evaluation_protocol_sha256": _protocol_sha256_for_task(task),
        "command_follow_contract": compact,
        "command_training_contract_sha256": contract_sha256,
        "command_training_contract": complete,
    }
    evaluation["manifest_id"] = _canonical_sha256(evaluation)
    return _canonical_sha256(task_payload), evaluation["manifest_id"], evaluation


def validate_evaluation_checkpoint(
    spec: CommandCheckpointSpec,
    payload: Mapping[str, Any],
    *,
    current_source_set: str,
    requested_policy: str | None,
    requested_fingerprint: str | None,
    requested_training_seed: int | None,
    requested_task: str | None = None,
) -> dict[str, Any]:
    """Fail closed on task, protocol, completion, source, and launch identity."""

    task = spec.resolved_config.get("task")
    if task not in TASK_IDS:
        raise ValueError("checkpoint does not contain a supported command task")
    if requested_task is not None and requested_task != task:
        raise ValueError("--task differs from checkpoint task")
    task_manifest_id, evaluation_manifest_id, manifest = expected_manifest_ids(task)
    if spec.task_manifest_id != task_manifest_id:
        raise ValueError("checkpoint task manifest differs from current CommandFollow task")
    if spec.evaluation_manifest_id != evaluation_manifest_id:
        raise ValueError("checkpoint evaluation manifest differs from the selected command protocol")
    if spec.fingerprints.get("source_set") != current_source_set:
        raise ValueError("checkpoint source-set fingerprint differs from current runtime")
    resolved = spec.resolved_config
    if resolved.get("contract_profile") != _profile_for_task(task):
        raise ValueError("checkpoint contract_profile does not match its command task")
    if resolved.get("evaluation_protocol") != _protocol_name_for_task(task):
        raise ValueError("checkpoint evaluation_protocol does not match its command task")
    if requested_policy is not None and requested_policy != spec.controller:
        raise ValueError("--policy differs from checkpoint controller")
    reproduction = spec.fingerprints.get("reproduction")
    if requested_fingerprint is not None and requested_fingerprint != reproduction:
        raise ValueError("--expected_fingerprint differs from checkpoint")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("status") != "completed":
        raise ValueError("held-out evaluation requires a completed checkpoint")
    seed = metadata.get("seed", resolved.get("seed"))
    if type(seed) is not int:
        raise ValueError("checkpoint training seed is missing")
    if requested_training_seed is not None and requested_training_seed != seed:
        raise ValueError("--training_seed differs from checkpoint")
    counters = payload.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("checkpoint counters are missing")
    interactions = counters.get("total_interactions")
    declared = resolved.get("total_interactions")
    if type(interactions) is not int or type(declared) is not int or interactions != declared:
        raise ValueError("checkpoint did not complete its declared interaction budget")
    return {
        "training_seed": seed,
        "total_interactions": interactions,
        "reproduction_fingerprint": reproduction,
        "task_manifest_id": task_manifest_id,
        "evaluation_manifest_id": evaluation_manifest_id,
        "evaluation_manifest": manifest,
    }


def _atomic_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish one JSON result without ever replacing prior evidence."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".lock")
    temporary = destination.with_name(destination.name + f".tmp-{uuid.uuid4().hex}")
    with lock.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite evaluation: {destination}")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def control_quality_metrics(
    rollout: CommandRollout,
    acceleration_body: torch.Tensor,
    jerk_body: torch.Tensor,
    projected_gravity_body: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    primary_score: Mapping[str, Any],
) -> dict[str, Any]:
    """Report acceleration/response, stability, and survival independently."""

    shape3 = (*rollout.alive.shape, 3)
    for name, value in (
        ("acceleration_body", acceleration_body),
        ("jerk_body", jerk_body),
        ("projected_gravity_body", projected_gravity_body),
        ("angular_velocity_body", angular_velocity_body),
    ):
        if value.shape != shape3 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite with shape {list(shape3)}")
    alive = rollout.alive
    denominator = max(1, int(alive.sum()))

    def vector_rms(value: torch.Tensor) -> float:
        return float((value.square().sum(dim=-1)[alive].sum() / denominator).sqrt())

    acceleration_rms = vector_rms(acceleration_body)
    jerk_rms = vector_rms(jerk_body)
    tilt_rms = float(
        (projected_gravity_body[:, :, :2].square().sum(dim=-1)[alive].sum() / denominator).sqrt()
    )
    angular_rms = vector_rms(angular_velocity_body)
    commands = rollout.effective_commands[:, :, :3]
    commanded = alive & (commands.square().sum(dim=-1) > 1.0e-8)
    wrong_acceleration_fraction = (
        float(((acceleration_body * commands).sum(dim=-1)[commanded] < 0.0).float().mean())
        if bool(commanded.any())
        else 0.0
    )
    survival = float(alive.float().mean())
    invalid = int(rollout.invalid.sum())
    response_score = float(primary_score["component_scores"]["response"])
    acceleration_score = 100.0 * math.exp(-((jerk_rms / 80.0) ** 2)) * max(
        0.0, 1.0 - wrong_acceleration_fraction
    )
    stability_score = 100.0 * math.exp(
        -((tilt_rms / 0.25) ** 2) - ((angular_rms / 1.5) ** 2)
    )
    survival_score = 100.0 * survival * (1.0 if invalid == 0 else 0.0)
    return {
        "component_scores_0_100": {
            "acceleration_quality": acceleration_score,
            "command_response": response_score,
            "flight_stability": stability_score,
            "survival_not_die": survival_score,
        },
        "raw": {
            "acceleration_rms_m_s2": acceleration_rms,
            "jerk_rms_m_s3": jerk_rms,
            "wrong_direction_acceleration_fraction": wrong_acceleration_fraction,
            "projected_gravity_xy_rms": tilt_rms,
            "angular_velocity_rms_rad_s": angular_rms,
            "survival_fraction": survival,
            "invalid_state_count": invalid,
        },
        "interpretation": (
            "Independent control diagnostics; none is inferred from aggregate reward."
        ),
    }


def physical_wind_telemetry_report(
    *,
    task: str,
    force_world: torch.Tensor,
    torque_world: torch.Tensor,
    category_code: torch.Tensor,
    interval_observed: torch.Tensor,
    vehicle_weight_n: float,
) -> dict[str, Any]:
    """Audit the physical wrench applied during each evaluated interval.

    The environment snapshots ``terminal_applied_wind_*`` in ``_get_dones``
    before Isaac Lab advances the wind schedule or auto-resets an environment.
    Consequently each row supplied here is the wrench that was actually active
    during that call to ``env.step`` rather than the wrench prepared for the
    next interval.  Schedule equality is evaluated only through an episode's
    first terminal interval; post-termination auto-reset activity is retained
    for finite/bounds auditing but cannot be attributed to the original held-
    out episode.
    """

    from g1_fly_control.tasks.crazyflie.command_wide_logic import (
        HELD_OUT_PULSE_DURATION_STEPS,
        HELD_OUT_PULSE_START_STEPS,
        HELD_OUT_WIND_SEED,
        WIND_CATEGORIES,
        WIND_REFERENCE_ARM_M,
        held_out_wind_at_step,
        wind_evaluation_protocol_payload,
    )

    if task not in {WIDE_TASK_ID, WIDE_WIND_TASK_ID}:
        raise ValueError("physical wind telemetry is only defined for command-v2 tasks")
    expected_vector_shape = (EPISODE_STEPS, EPISODE_COUNT, 3)
    expected_scalar_shape = (EPISODE_STEPS, EPISODE_COUNT)
    for name, value in (("force_world", force_world), ("torque_world", torque_world)):
        if value.shape != expected_vector_shape:
            raise ValueError(f"{name} must have shape {list(expected_vector_shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains nonfinite physical telemetry")
    if category_code.shape != expected_scalar_shape:
        raise ValueError(f"category_code must have shape {list(expected_scalar_shape)}")
    if (
        interval_observed.shape != expected_scalar_shape
        or interval_observed.dtype != torch.bool
    ):
        raise ValueError(
            f"interval_observed must be bool with shape {list(expected_scalar_shape)}"
        )
    if not math.isfinite(vehicle_weight_n) or vehicle_weight_n <= 0.0:
        raise ValueError("vehicle_weight_n must be finite and positive")

    # Run all comparisons on CPU so the report builder remains deterministic
    # and can be covered without launching Isaac Sim.
    force = force_world.detach().cpu()
    torque = torque_world.detach().cpu()
    categories = category_code.detach().to(device="cpu", dtype=torch.long)
    observed = interval_observed.detach().to(device="cpu", dtype=torch.bool)
    expected_force = torch.zeros_like(force)
    expected_torque = torch.zeros_like(torque)
    expected_categories = torch.zeros_like(categories)
    if task == WIDE_WIND_TASK_ID:
        category_to_code = {name: index for index, name in enumerate(WIND_CATEGORIES)}
        for episode_index in range(EPISODE_COUNT):
            for step in range(EPISODE_STEPS):
                scheduled = held_out_wind_at_step(
                    episode_index,
                    step,
                    seed=HELD_OUT_WIND_SEED,
                )
                expected_force[step, episode_index] = force.new_tensor(
                    scheduled.force_ratio_world
                ) * vehicle_weight_n
                expected_torque[step, episode_index] = torque.new_tensor(
                    scheduled.torque_ratio_world
                ) * vehicle_weight_n * WIND_REFERENCE_ARM_M
                expected_categories[step, episode_index] = category_to_code[
                    scheduled.category
                ]

    force_norm = torch.linalg.vector_norm(force, dim=-1)
    torque_norm = torch.linalg.vector_norm(torque, dim=-1)
    expected_force_norm = torch.linalg.vector_norm(expected_force, dim=-1)
    expected_torque_norm = torch.linalg.vector_norm(expected_torque, dim=-1)
    actual_active = (force_norm > 0.0) | (torque_norm > 0.0)
    expected_active = (expected_force_norm > 0.0) | (expected_torque_norm > 0.0)
    observed_count = int(observed.sum())
    force_limit_ratio = 0.0
    torque_limit_ratio = 0.0
    if task == WIDE_WIND_TASK_ID:
        wind_protocol = wind_evaluation_protocol_payload()
        force_limit_ratio = float(wind_protocol["force_to_weight_ratio_limit"])
        torque_limit_ratio = float(
            wind_protocol["torque_to_weight_arm_ratio_limit"]
        )
    force_limit_n = vehicle_weight_n * force_limit_ratio
    torque_limit_nm = vehicle_weight_n * WIND_REFERENCE_ARM_M * torque_limit_ratio
    force_tolerance = max(1.0e-8, force_limit_n * 1.0e-6)
    torque_tolerance = max(1.0e-10, torque_limit_nm * 1.0e-6)

    expected_force_matches = bool(
        observed_count
        and torch.allclose(
            force[observed],
            expected_force[observed],
            rtol=1.0e-6,
            atol=force_tolerance,
        )
    )
    expected_torque_matches = bool(
        observed_count
        and torch.allclose(
            torque[observed],
            expected_torque[observed],
            rtol=1.0e-6,
            atol=torque_tolerance,
        )
    )
    expected_categories_match = bool(
        observed_count
        and torch.equal(categories[observed], expected_categories[observed])
    )
    pulse_window_matches = bool(
        observed_count and torch.equal(actual_active[observed], expected_active[observed])
    )
    force_within_bound = bool((force_norm <= force_limit_n + force_tolerance).all())
    torque_within_bound = bool(
        (torque_norm <= torque_limit_nm + torque_tolerance).all()
    )
    still_exact_zero = bool(
        torch.count_nonzero(force) == 0
        and torch.count_nonzero(torque) == 0
        and torch.count_nonzero(categories) == 0
    )
    complete_protocol_observed = observed_count == EPISODE_COUNT * EPISODE_STEPS
    wind_nonzero_observed = bool((actual_active & observed).any())
    category_codes_valid = bool(
        ((categories >= 0) & (categories < len(WIND_CATEGORIES))).all()
    )
    integrity_passed = bool(
        force_within_bound
        and torque_within_bound
        and category_codes_valid
        and expected_force_matches
        and expected_torque_matches
        and expected_categories_match
        and pulse_window_matches
        and (task != WIDE_TASK_ID or still_exact_zero)
    )

    def _rms(norm: torch.Tensor, mask: torch.Tensor) -> float:
        selected = norm[mask]
        return float(selected.square().mean().sqrt()) if selected.numel() else 0.0

    per_episode: list[dict[str, Any]] = []
    for episode_index in range(EPISODE_COUNT):
        episode_observed = observed[:, episode_index]
        episode_expected_active = expected_active[:, episode_index]
        episode_actual_active = actual_active[:, episode_index]
        per_episode.append(
            {
                "episode_index": episode_index,
                "observed_intervals_through_first_done": int(episode_observed.sum()),
                "planned_pulse_intervals": int(episode_expected_active.sum()),
                "observed_planned_pulse_intervals": int(
                    (episode_observed & episode_expected_active).sum()
                ),
                "observed_nonzero_wrench_intervals": int(
                    (episode_observed & episode_actual_active).sum()
                ),
                "maximum_force_norm_n_observed": float(
                    force_norm[:, episode_index][episode_observed].max()
                )
                if bool(episode_observed.any())
                else 0.0,
                "maximum_torque_norm_nm_observed": float(
                    torque_norm[:, episode_index][episode_observed].max()
                )
                if bool(episode_observed.any())
                else 0.0,
                "expected_schedule_matches_observed": bool(
                    torch.allclose(
                        force[:, episode_index][episode_observed],
                        expected_force[:, episode_index][episode_observed],
                        rtol=1.0e-6,
                        atol=force_tolerance,
                    )
                    and torch.allclose(
                        torque[:, episode_index][episode_observed],
                        expected_torque[:, episode_index][episode_observed],
                        rtol=1.0e-6,
                        atol=torque_tolerance,
                    )
                    and torch.equal(
                        categories[:, episode_index][episode_observed],
                        expected_categories[:, episode_index][episode_observed],
                    )
                ),
            }
        )

    return {
        "source": (
            "terminal_applied_wind_force_world/torque_world captured immediately "
            "after env.step for the physical interval just simulated"
        ),
        "frame": "world",
        "application_point": "body_center_of_mass",
        "condition": "wind" if task == WIDE_WIND_TASK_ID else "still_air",
        "vehicle_weight_n": vehicle_weight_n,
        "reference_arm_m": WIND_REFERENCE_ARM_M,
        "limits": {
            "force_to_weight_ratio": force_limit_ratio,
            "torque_to_weight_arm_ratio": torque_limit_ratio,
            "maximum_force_norm_n": force_limit_n,
            "maximum_torque_norm_nm": torque_limit_nm,
        },
        "protocol": {
            "seed": HELD_OUT_WIND_SEED if task == WIDE_WIND_TASK_ID else None,
            "pulse_start_steps": (
                list(HELD_OUT_PULSE_START_STEPS)
                if task == WIDE_WIND_TASK_ID
                else []
            ),
            "pulse_duration_steps": (
                HELD_OUT_PULSE_DURATION_STEPS
                if task == WIDE_WIND_TASK_ID
                else 0
            ),
        },
        "samples": {
            "planned_intervals": EPISODE_COUNT * EPISODE_STEPS,
            "observed_intervals_through_first_done": observed_count,
            "planned_pulse_intervals": int(expected_active.sum()),
            "observed_planned_pulse_intervals": int((expected_active & observed).sum()),
            "observed_nonzero_wrench_intervals": int((actual_active & observed).sum()),
        },
        "measured": {
            "maximum_force_norm_n_all_simulated_intervals": float(force_norm.max()),
            "maximum_torque_norm_nm_all_simulated_intervals": float(torque_norm.max()),
            "force_rms_n_observed": _rms(force_norm, observed),
            "torque_rms_nm_observed": _rms(torque_norm, observed),
        },
        "integrity": {
            "all_values_finite": True,
            "category_codes_valid_all_simulated_intervals": category_codes_valid,
            "force_within_declared_bound": force_within_bound,
            "torque_within_declared_bound": torque_within_bound,
            "expected_force_matches_on_observed_intervals": expected_force_matches,
            "expected_torque_matches_on_observed_intervals": expected_torque_matches,
            "expected_category_matches_on_observed_intervals": expected_categories_match,
            "pulse_window_matches_on_observed_intervals": pulse_window_matches,
            "still_air_exact_zero_all_simulated_intervals": (
                still_exact_zero if task == WIDE_TASK_ID else None
            ),
            "complete_600_step_protocol_observed_for_all_episodes": (
                complete_protocol_observed
            ),
            "wind_nonzero_wrench_observed": (
                wind_nonzero_observed if task == WIDE_WIND_TASK_ID else None
            ),
            "passed": integrity_passed,
        },
        "per_episode": per_episode,
    }


def summarize_activity(
    recorder: TrainedActivityRecorder,
    sums: torch.Tensor,
    squares: torch.Tensor,
    active_counts: torch.Tensor,
    samples: torch.Tensor,
) -> dict[str, Any]:
    """Summarize authenticated actual-forward activity by command segment."""

    units = len(recorder.roles)
    expected = (len(SEGMENT_NAMES), units)
    if any(value.shape != expected for value in (sums, squares, active_counts)):
        raise ValueError("activity accumulators have incompatible shapes")
    if samples.shape != (len(SEGMENT_NAMES),):
        raise ValueError("activity sample counts have incompatible shape")
    sums = sums.double().cpu()
    squares = squares.double().cpu()
    active_counts = active_counts.double().cpu()
    samples = samples.long().cpu()
    roles = tuple(recorder.roles)
    unit_ids = tuple(recorder.unit_ids)
    if len(unit_ids) != units:
        raise ValueError("activity stable-ID width differs from roles")

    def summarize_rows(total: torch.Tensor, square: torch.Tensor, active: torch.Tensor, count: int) -> dict[str, Any]:
        divisor = max(1, count)
        by_role = {}
        for role in dict.fromkeys(roles):
            indices = [index for index, value in enumerate(roles) if value == role]
            by_role[role] = {
                "unit_count": len(indices),
                "mean_absolute_activity_per_unit": float(total[indices].sum() / (divisor * len(indices))),
                "active_fraction_per_unit": float(active[indices].sum() / (divisor * len(indices))),
            }
        return {
            "sample_count": count,
            "mean_absolute_activity_per_unit": float(total.sum() / (divisor * units)),
            "rms_activity_per_unit": float((square.sum() / (divisor * units)).sqrt()),
            "active_fraction_per_unit": float(active.sum() / (divisor * units)),
            "roles": by_role,
        }

    overall_sum = sums.sum(dim=0)
    overall_square = squares.sum(dim=0)
    overall_active = active_counts.sum(dim=0)
    overall_samples = int(samples.sum())
    ranked = sorted(range(units), key=lambda index: (-float(overall_sum[index]), unit_ids[index]))
    per_unit = [
        {
            "index": index,
            "id": unit_ids[index],
            "role": roles[index],
            "absolute_activity_sum": float(overall_sum[index]),
            "mean_absolute_activity": float(overall_sum[index] / max(1, overall_samples)),
            "rms_activity": float((overall_square[index] / max(1, overall_samples)).sqrt()),
            "active_count": int(overall_active[index]),
            "active_fraction": float(overall_active[index] / max(1, overall_samples)),
        }
        for index in range(units)
    ]
    return {
        "source": "exact_forward_pass_that_produced_each_evaluated_action",
        "controller": recorder.spec.controller,
        "kind": "sampled_lif_spikes" if "lif" in recorder.spec.controller else "engineering_absolute_activations",
        "unit_count": units,
        "role_counts": dict(Counter(roles)),
        "role_provenance": recorder.role_provenance,
        "overall": summarize_rows(overall_sum, overall_square, overall_active, overall_samples),
        "segments": {
            name: summarize_rows(sums[index], squares[index], active_counts[index], int(samples[index]))
            for index, name in enumerate(SEGMENT_NAMES)
        },
        "per_unit": per_unit,
        "top_units": [per_unit[index] for index in ranked[:10]],
    }


def _state_finite_rows(state: Any, batch: int, device: torch.device) -> torch.Tensor:
    if state is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    if isinstance(state, torch.Tensor):
        return torch.isfinite(state).all(dim=1)
    tensors = tuple(getattr(state, name) for name in ("membrane", "spikes", "synapse", "refractory"))
    result = torch.ones(batch, dtype=torch.bool, device=device)
    for value in tensors:
        result &= torch.isfinite(value).all(dim=1)
    return result


def _reset_state_rows(policy: torch.nn.Module, state: Any, done: torch.Tensor) -> Any:
    from g1_fly_control.crazyflie.controllers import reset_controller_state

    return reset_controller_state(policy, state, done)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=TASK_IDS)
    parser.add_argument(
        "--protocol",
        choices=(EVALUATION_PROTOCOL_V1, EVALUATION_PROTOCOL_V2),
    )
    parser.add_argument("--policy", choices=SUPPORTED_CONTROLLER_KINDS)
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--training_seed", type=int)
    parser.add_argument(
        "--asset_mirror",
        type=Path,
        default=Path.home() / ".cache/flyg1/isaac-5.1-offline",
    )
    return parser


def _run(
    args: argparse.Namespace,
    spec: CommandCheckpointSpec,
    launch_identity: dict[str, Any],
    current_source_set: str,
) -> dict[str, Any]:
    global _LAST_MEMORY_GATE, _LAST_MEMORY_SAMPLES

    import gymnasium as gym
    import isaaclab_tasks.direct.quadcopter  # noqa: F401
    import g1_fly_control.tasks.crazyflie  # noqa: F401
    from drone_bootstrap import selected_env_cfg
    from g1_fly_control.crazyflie.memory import assess, reset_cuda_peak, snapshot
    from g1_fly_control.tasks.crazyflie.offline_assets import configure_offline_scene

    task = str(spec.resolved_config["task"])
    device = torch.device(args.device)
    cfg = selected_env_cfg(
        task,
        EPISODE_COUNT,
        deterministic_evaluation=True,
        contract_profile=_profile_for_task(task),
    )
    cfg.episode_length_s = EPISODE_STEPS * 0.02
    cfg.scene.num_envs = EPISODE_COUNT
    cfg.scene.clone_in_fabric = False
    cfg.sim.device = str(device)
    cfg.debug_vis = False
    asset_report = configure_offline_scene(cfg, mirror_root=args.asset_mirror)
    env = None
    recorder = None
    memory_samples: list[dict[str, Any]] = []
    memory_gate: dict[str, Any] | None = None

    def record_memory(stage: str, step: int | None = None) -> None:
        nonlocal memory_gate
        global _LAST_MEMORY_GATE, _LAST_MEMORY_SAMPLES
        memory_samples.append(snapshot(stage, device, step=step))
        memory_gate = assess(memory_samples)
        _LAST_MEMORY_SAMPLES = list(memory_samples)
        _LAST_MEMORY_GATE = dict(memory_gate)
        if not memory_gate["passed"]:
            raise RuntimeError("evaluation memory gate failed: " + "; ".join(memory_gate["failures"]))

    reset_cuda_peak(device)
    try:
        env = gym.make(task, cfg=cfg, render_mode=None).unwrapped
        record_memory("environment_loaded", 0)
        policy, normalizer, controller_report = load_command_controller(
            spec, device=device, expected_source_set=current_source_set
        )
        recorder = TrainedActivityRecorder(policy, spec)
        record_memory("policy_loaded", 0)
        if env.command_tracking_contract != launch_identity["evaluation_manifest"]["command_training_contract"]:
            raise RuntimeError("live environment command contract differs from checkpoint evaluation contract")
        if task == WIDE_WIND_TASK_ID:
            set_wind_evaluation_mode = getattr(env, "set_wind_evaluation_mode", None)
            if not callable(set_wind_evaluation_mode):
                raise RuntimeError("wide-wind task lacks held-out wind evaluation mode")
            wind_seed = int(
                _protocol_payload_for_task(task)["wind_protocol"]["seed"]
            )
            set_wind_evaluation_mode(
                list(range(EPISODE_COUNT)),
                seed=wind_seed,
            )
        observation, _ = env.reset(
            seed=_protocol_payload_for_task(task)["evaluation_seed"]
        )
        observation = env.clear_manual_command()
        env.episode_length_buf.zero_()
        state = initial_policy_state(policy, EPISODE_COUNT, device)
        commands, segment_names = (
            batched_v1_held_out_commands(device=device)
            if task == TASK_ID
            else batched_v2_held_out_commands(device=device)
        )
        active = torch.ones(EPISODE_COUNT, dtype=torch.bool, device=device)
        terminated_final = torch.zeros_like(active)
        truncated_final = torch.zeros_like(active)
        failure_cause = torch.zeros(EPISODE_COUNT, dtype=torch.long, device=device)

        effective_trace = torch.zeros_like(commands)
        linear_trace = torch.zeros((EPISODE_STEPS, EPISODE_COUNT, 3), device=device)
        yaw_trace = torch.zeros((EPISODE_STEPS, EPISODE_COUNT), device=device)
        action_trace = torch.zeros_like(commands)
        position_trace = torch.zeros((EPISODE_STEPS, EPISODE_COUNT, 3), device=device)
        alive_trace = torch.zeros((EPISODE_STEPS, EPISODE_COUNT), dtype=torch.bool, device=device)
        invalid_trace = torch.zeros_like(alive_trace)
        acceleration_trace = torch.zeros_like(linear_trace)
        jerk_trace = torch.zeros_like(linear_trace)
        gravity_trace = torch.zeros_like(linear_trace)
        angular_trace = torch.zeros_like(linear_trace)
        reward_trace = torch.zeros((EPISODE_STEPS, EPISODE_COUNT), device=device)
        wide_task = task in {WIDE_TASK_ID, WIDE_WIND_TASK_ID}
        wind_force_trace = (
            torch.zeros_like(linear_trace) if wide_task else None
        )
        wind_torque_trace = (
            torch.zeros_like(linear_trace) if wide_task else None
        )
        wind_category_trace = (
            torch.zeros(
                (EPISODE_STEPS, EPISODE_COUNT),
                dtype=torch.long,
                device=device,
            )
            if wide_task
            else None
        )
        wind_observed_trace = (
            torch.zeros(
                (EPISODE_STEPS, EPISODE_COUNT),
                dtype=torch.bool,
                device=device,
            )
            if wide_task
            else None
        )
        vehicle_weight_n = (
            float(getattr(env, "_robot_weight")) if wide_task else None
        )
        reward_names = tuple(sorted(env.reward_components))
        reward_component_sums = {
            name: torch.zeros(EPISODE_COUNT, device=device) for name in reward_names
        }
        units = len(recorder.roles)
        activity_sums = torch.zeros((len(SEGMENT_NAMES), units), device=device)
        activity_squares = torch.zeros_like(activity_sums)
        activity_active = torch.zeros_like(activity_sums)
        activity_samples = torch.zeros(len(SEGMENT_NAMES), dtype=torch.long, device=device)
        hover_action = commands.new_tensor([2.0 / 1.9 - 1.0, 0.0, 0.0, 0.0])
        inference_latency = AuthenticatedInferenceLatency(
            device=device,
            batch_size=EPISODE_COUNT,
        )

        for step in range(EPISODE_STEPS):
            if wind_observed_trace is not None:
                # ``active`` still identifies the original held-out episode at
                # this point; it is updated only after terminal telemetry from
                # the just-simulated interval has been captured.
                wind_observed_trace[step] = active
            requested = commands[step].clone()
            requested[~active] = 0.0
            observation = env.set_manual_command_body(requested)
            effective = env.effective_command_body.detach().clone()
            normalized = normalizer.normalize(observation["policy"])
            recorder.begin_step()
            with torch.no_grad():
                inference_latency.begin_call()
                output = policy.act(normalized, state, deterministic=True)
                inference_latency.end_call()
            candidate_state = output.state
            activity_values = recorder.activity_batch(candidate_state)
            action = output.action
            finite_rows = (
                torch.isfinite(action).all(dim=1)
                & (action.abs().amax(dim=1) <= 1.0 + 1.0e-6)
                & _state_finite_rows(candidate_state, EPISODE_COUNT, device)
                & torch.isfinite(activity_values).all(dim=1)
                & torch.isfinite(observation["policy"]).all(dim=1)
            )
            invalid_now = active & ~finite_rows
            simulator_action = action.clone()
            simulator_action[~finite_rows | ~active] = hover_action
            next_observation, reward, terminated, truncated, _ = env.step(simulator_action)
            if wide_task:
                terminal_wind_force = getattr(
                    env, "terminal_applied_wind_force_world", None
                )
                terminal_wind_torque = getattr(
                    env, "terminal_applied_wind_torque_world", None
                )
                terminal_wind_category = getattr(
                    env, "terminal_wind_category_code", None
                )
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (
                        terminal_wind_force,
                        terminal_wind_torque,
                        terminal_wind_category,
                    )
                ):
                    raise RuntimeError(
                        "command-v2 environment lacks terminal physical-wind telemetry"
                    )
                if (
                    terminal_wind_force.shape != (EPISODE_COUNT, 3)
                    or terminal_wind_torque.shape != (EPISODE_COUNT, 3)
                    or terminal_wind_category.shape != (EPISODE_COUNT,)
                ):
                    raise RuntimeError("terminal physical-wind telemetry shape changed")
                assert wind_force_trace is not None
                assert wind_torque_trace is not None
                assert wind_category_trace is not None
                wind_force_trace[step] = terminal_wind_force.detach()
                wind_torque_trace[step] = terminal_wind_torque.detach()
                wind_category_trace[step] = terminal_wind_category.detach()
            terminal_obs = env.drone_terminal_observation.detach()
            linear = terminal_obs[:, 0:3] + effective[:, 0:3]
            angular = terminal_obs[:, 3:6].clone()
            angular[:, 2] += effective[:, 3]
            numeric = torch.cat(
                (
                    linear,
                    angular,
                    env.terminal_position_w,
                    env.terminal_linear_acceleration_body,
                    env.terminal_linear_jerk_body,
                    terminal_obs[:, 6:9],
                ),
                dim=1,
            )
            invalid_now |= active & ~torch.isfinite(numeric).all(dim=1)
            if wide_task:
                assert wind_force_trace is not None
                assert wind_torque_trace is not None
                wind_finite = torch.isfinite(wind_force_trace[step]).all(dim=1)
                wind_finite &= torch.isfinite(wind_torque_trace[step]).all(dim=1)
                invalid_now |= active & ~wind_finite
            # Include telemetry failures discovered after stepping before
            # advancing recurrent state or the protocol's active mask.
            done = terminated | truncated | invalid_now
            safe_numeric = torch.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
            linear = safe_numeric[:, 0:3]
            angular = safe_numeric[:, 3:6]
            positions = safe_numeric[:, 6:9]
            accelerations = safe_numeric[:, 9:12]
            jerks = safe_numeric[:, 12:15]
            gravity = safe_numeric[:, 15:18]
            alive_now = active & ~terminated & ~invalid_now
            segment_index = step // SEGMENT_STEPS
            activity_mask = active & finite_rows
            selected_activity = activity_values[activity_mask]
            if selected_activity.numel():
                activity_sums[segment_index] += selected_activity.abs().sum(dim=0)
                activity_squares[segment_index] += selected_activity.square().sum(dim=0)
                threshold = 0.0 if "lif" in spec.controller else ACTIVITY_THRESHOLD
                activity_active[segment_index] += (selected_activity.abs() > threshold).sum(dim=0)
                activity_samples[segment_index] += int(activity_mask.sum())

            effective_trace[step] = effective
            linear_trace[step] = linear
            yaw_trace[step] = angular[:, 2]
            action_trace[step] = torch.nan_to_num(simulator_action).clamp(-1.0, 1.0)
            position_trace[step] = positions
            alive_trace[step] = alive_now
            invalid_trace[step] = invalid_now
            acceleration_trace[step] = accelerations
            jerk_trace[step] = jerks
            gravity_trace[step] = gravity
            angular_trace[step] = angular
            reward_trace[step] = torch.nan_to_num(reward)
            for name in reward_names:
                reward_component_sums[name] += torch.where(
                    active, torch.nan_to_num(env.reward_components[name]), 0.0
                )
            newly_terminated = active & terminated
            newly_truncated = active & truncated
            terminated_final |= newly_terminated | invalid_now
            truncated_final |= newly_truncated
            failure_cause = torch.where(
                newly_terminated,
                env.terminal_failure_cause,
                failure_cause,
            )
            failure_cause = torch.where(
                invalid_now,
                torch.full_like(failure_cause, 4),
                failure_cause,
            )
            active &= ~done
            state = _reset_state_rows(policy, candidate_state, done)
            observation = next_observation
            if step + 1 in {150, 300, 450, 600}:
                record_memory("steady_state", step + 1)

        record_memory("evaluation_end", EPISODE_STEPS)
        if memory_gate is None:
            raise RuntimeError("memory gate was not assessed")
        rollout = CommandRollout(
            effective_trace.cpu(),
            linear_trace.cpu(),
            yaw_trace.cpu(),
            action_trace.cpu(),
            position_trace.cpu(),
            alive_trace.cpu(),
            invalid_trace.cpu(),
        )
        score = (
            score_command_rollout_v1(rollout)
            if task == TASK_ID
            else score_command_rollout_v2(rollout, task=task)
        )
        quality = control_quality_metrics(
            rollout,
            acceleration_trace.cpu(),
            jerk_trace.cpu(),
            gravity_trace.cpu(),
            angular_trace.cpu(),
            score,
        )
        physical_wind = None
        if wide_task:
            assert wind_force_trace is not None
            assert wind_torque_trace is not None
            assert wind_category_trace is not None
            assert wind_observed_trace is not None
            assert vehicle_weight_n is not None
            physical_wind = physical_wind_telemetry_report(
                task=task,
                force_world=wind_force_trace,
                torque_world=wind_torque_trace,
                category_code=wind_category_trace,
                interval_observed=wind_observed_trace,
                vehicle_weight_n=vehicle_weight_n,
            )
            if not physical_wind["integrity"]["passed"]:
                failed = [
                    name
                    for name, value in physical_wind["integrity"].items()
                    if value is False
                ]
                raise RuntimeError(
                    "physical wind telemetry integrity failed: " + ", ".join(failed)
                )
        reward_component_cpu = {
            name: value.cpu() for name, value in reward_component_sums.items()
        }
        episodes = []
        for index in range(EPISODE_COUNT):
            episode_rollout = CommandRollout(
                rollout.effective_commands[:, index : index + 1],
                rollout.linear_velocity_body[:, index : index + 1],
                rollout.yaw_rate_body[:, index : index + 1],
                rollout.actions[:, index : index + 1],
                rollout.positions_world[:, index : index + 1],
                rollout.alive[:, index : index + 1],
                rollout.invalid[:, index : index + 1],
            )
            episode_score = (
                score_command_rollout_v1(episode_rollout)
                if task == TASK_ID
                else score_command_rollout_v2(episode_rollout, task=task)
            )
            episode_quality = control_quality_metrics(
                episode_rollout,
                acceleration_trace[:, index : index + 1].cpu(),
                jerk_trace[:, index : index + 1].cpu(),
                gravity_trace[:, index : index + 1].cpu(),
                angular_trace[:, index : index + 1].cpu(),
                episode_score,
            )
            episodes.append(
                {
                    "episode_index": index,
                    "score": episode_score,
                    "control_quality": episode_quality,
                    "terminated": bool(terminated_final[index]),
                    "truncated": bool(truncated_final[index]),
                    "failure_cause": int(failure_cause[index]),
                    "alive_steps": int(alive_trace[:, index].sum()),
                    "invalid_steps": int(invalid_trace[:, index].sum()),
                    "reward_total": float(reward_trace[:, index].sum()),
                    "reward_components": {
                        name: float(value[index]) for name, value in reward_component_cpu.items()
                    },
                    **(
                        {"physical_wind": physical_wind["per_episode"][index]}
                        if physical_wind is not None
                        else {}
                    ),
                }
            )
        activity_report = summarize_activity(
            recorder,
            activity_sums,
            activity_squares,
            activity_active,
            activity_samples,
        )
        inference_latency_report = inference_latency.report(
            expected_call_count=EPISODE_STEPS
        )
        termination_count = int(terminated_final.sum())
        truncation_count = int(truncated_final.sum())
        failure_counts = Counter(int(value) for value in failure_cause.cpu().tolist())
        return {
            "schema_version": 1,
            "analysis_kind": (
                "crazyflie_command_follow_heldout_v1"
                if task == TASK_ID
                else "crazyflie_command_follow_heldout_v2"
            ),
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": task,
            "controller": spec.controller,
            "checkpoint": {
                "path": str(spec.path),
                "sha256": spec.sha256,
                **launch_identity,
            },
            "protocol": _protocol_payload_for_task(task),
            "protocol_sha256": _protocol_sha256_for_task(task),
            "episodes_requested": EPISODE_COUNT,
            "episodes_evaluated": EPISODE_COUNT,
            "steps_per_episode": EPISODE_STEPS,
            "vectorized_environment_count": EPISODE_COUNT,
            "deterministic_actions": True,
            "policy_action_source": "actual_trained_controller_no_assist",
            "summary": {
                "score": score,
                "control_quality": quality,
                "termination_count": termination_count,
                "truncation_count": truncation_count,
                "survived_full_horizon_count": sum(
                    not row["terminated"] and row["truncated"] for row in episodes
                ),
                "invalid_episode_count": sum(row["invalid_steps"] > 0 for row in episodes),
                "failure_cause_counts": {str(key): value for key, value in sorted(failure_counts.items())},
                "reward_total_mean": float(reward_trace.sum(dim=0).mean()),
                "reward_component_mean_per_episode": {
                    name: float(value.mean()) for name, value in reward_component_cpu.items()
                },
                **(
                    {
                        "physical_wind": {
                            key: value
                            for key, value in physical_wind.items()
                            if key != "per_episode"
                        }
                    }
                    if physical_wind is not None
                    else {}
                ),
            },
            "episodes": episodes,
            "segments": [
                {
                    "index": index,
                    "name": name,
                    "steps": SEGMENT_STEPS,
                    "command_sample_count": EPISODE_COUNT * SEGMENT_STEPS,
                }
                for index, name in enumerate(SEGMENT_NAMES)
            ],
            "activity": activity_report,
            "inference_latency": inference_latency_report,
            "controller_report": controller_report,
            "memory_samples": memory_samples,
            "memory_gate": memory_gate,
            "asset": asset_report,
            "integrity": {
                "task_manifest_matched": True,
                "evaluation_manifest_matched": True,
                "source_set_matched": True,
                "checkpoint_completed_budget": True,
                "all_actions_finite_and_bounded": int(invalid_trace.sum()) == 0,
                "activity_from_actual_forward": True,
                "inference_latency_from_actual_forward": True,
                "segment_names": list(segment_names[::SEGMENT_STEPS]),
                **(
                    {
                        "physical_wind_telemetry_passed": physical_wind[
                            "integrity"
                        ]["passed"],
                        "physical_wind_uses_terminal_actual_interval": True,
                    }
                    if physical_wind is not None
                    else {}
                ),
            },
        }
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = _parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"refusing to overwrite evaluation: {args.output}")
    if not args.headless:
        parser.error("command held-out evaluation requires --headless")
    if torch.device(args.device) != torch.device("cuda:0"):
        parser.error("command held-out evaluation requires --device cuda:0")
    try:
        from drone_bootstrap import canonical_sha256, source_hashes
        from g1_fly_control.crazyflie.checkpoint import read_checkpoint
        from g1_fly_control.tasks.crazyflie.offline_assets import verify_asset_mirror

        spec = inspect_command_checkpoint(args.checkpoint)
        checkpoint_task = str(spec.resolved_config["task"])
        expected_protocol = _protocol_name_for_task(checkpoint_task)
        if args.protocol is not None and args.protocol != expected_protocol:
            raise ValueError(
                f"--protocol={args.protocol} does not match {checkpoint_task} ({expected_protocol})"
            )
        args.task = checkpoint_task if args.task is None else args.task
        args.protocol = expected_protocol
        payload = read_checkpoint(spec.path, map_location="cpu", resolve_external_history=False)
        current_source_set = canonical_sha256(source_hashes())
        launch_identity = validate_evaluation_checkpoint(
            spec,
            payload,
            current_source_set=current_source_set,
            requested_policy=args.policy,
            requested_fingerprint=args.expected_fingerprint,
            requested_training_seed=args.training_seed,
            requested_task=args.task,
        )
        verify_asset_mirror(args.asset_mirror)
    except Exception as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "RESOLVED",
                "task": checkpoint_task,
                "controller": spec.controller,
                "checkpoint_sha256": spec.sha256,
                "episodes": EPISODE_COUNT,
                "steps": EPISODE_STEPS,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    app = AppLauncher(args).app
    try:
        result = _run(args, spec, launch_identity, current_source_set)
        _atomic_json_no_overwrite(args.output, result)
        print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "analysis_kind": (
                "crazyflie_command_follow_heldout_v1"
                if args.task == TASK_ID
                else "crazyflie_command_follow_heldout_v2"
            ),
            "status": "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": args.task,
            "controller": spec.controller,
            "checkpoint": str(spec.path),
            "checkpoint_sha256": spec.sha256,
            "phase_error": f"{type(exc).__name__}: {exc}",
            "memory_samples": _LAST_MEMORY_SAMPLES,
            "memory_gate": _LAST_MEMORY_GATE,
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_json_no_overwrite(args.output, failure)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
