#!/usr/bin/env python3
"""Run finite-state, schedule, controller, and memory smoke gates for Crazyflie."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from drone_bootstrap import (
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_COMMAND_V1,
    CONTRACT_PROFILE_COMMAND_V2,
    CONTRACT_PROFILE_SURVIVAL_V2,
    CONTRACT_PROFILES,
    COMMAND_TASK_ID,
    COMMAND_V2_TASK_IDS,
    COMMAND_WIDE_WIND_TASK_ID,
    DEFAULT_CONTRACT_PROFILE,
    DEFAULT_CONNECTOME,
    DEFAULT_OPTIC_CONNECTOME,
    DEFAULT_WING_CONNECTOME,
    DEFAULT_REWIRE_MANIFEST,
    ROOT,
    launch_environment,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    task_contract_payload,
    validate_contract_profile,
)
from g1_fly_control.tasks.crazyflie.logic import (
    GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
)
from g1_fly_control.tasks.crazyflie.command_logic import (
    COMMAND_CURRICULUM,
    COMMAND_TRACKING_CONTRACT_SHA256,
    FAILURE_HIGH_HEIGHT as COMMAND_FAILURE_HIGH_HEIGHT,
    FAILURE_LOW_HEIGHT as COMMAND_FAILURE_LOW_HEIGHT,
    FAILURE_NONFINITE as COMMAND_FAILURE_NONFINITE,
    FAILURE_WORKSPACE_ESCAPE as COMMAND_FAILURE_WORKSPACE_ESCAPE,
    MAXIMUM_COMMAND_HOLD_STEPS,
    bound_command_body,
    classify_command_failures,
    sample_scheduled_command,
)


LEGACY_CUSTOM_TASKS = {
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
    "FlyCrazyflie-Mixed-v0",
}
COMMAND_TASKS = {COMMAND_TASK_ID, *COMMAND_V2_TASK_IDS}
CUSTOM_TASKS = LEGACY_CUSTOM_TASKS | COMMAND_TASKS
GATE_C_TASKS = LEGACY_CUSTOM_TASKS - {"FlyCrazyflie-Mixed-v0"}
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
LEG_CONNECTOME_POLICIES = frozenset(
    {
        "frozen_lif_original",
        "frozen_lif_degree_rewired",
        "leg_wing_lif",
        "leg_optic_lif",
        "leg_wing_optic_lif",
    }
)
WING_CONNECTOME_POLICIES = frozenset(
    {"wing_lif", "leg_wing_lif", "wing_optic_lif", "leg_wing_optic_lif"}
)
OPTIC_CONNECTOME_POLICIES = frozenset(
    {"optic_lif", "leg_optic_lif", "wing_optic_lif", "leg_wing_optic_lif"}
)
UNMATCHED_MULTI_CONNECTOME_POLICIES = frozenset(
    {
        "wing_lif",
        "leg_wing_lif",
        "leg_optic_lif",
        "wing_optic_lif",
        "leg_wing_optic_lif",
    }
)
PROBE_SEED = 20260916
SUBMITTED_IMPULSE_ABS_TOL_N_S = GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S
TERMINAL_SNAPSHOT_ABS_TOL = 2.0e-4
FULL_CURRICULUM_INTERACTIONS = 1_000_000
COMMAND_FULL_CURRICULUM_INTERACTIONS = 500_000
COMMAND_V2_FULL_CURRICULUM_INTERACTIONS = 1_000_000
COMMAND_MINIMUM_SMOKE_STEPS = 500
COMMAND_REWARD_COMPONENTS = (
    "linear_tracking",
    "yaw_tracking",
    "tracking_progress",
    "wrong_direction_acceleration",
    "jerk",
    "target_retention",
    "attitude_stability",
    "angular_stability",
    "control_effort",
    "action_smoothness",
    "survival",
    "failure",
    "total",
)
_COMMAND_CATEGORY_TO_CODE = {
    "hover": 0,
    "cardinal": 1,
    "diagonal": 2,
    "full_simultaneous": 3,
}


def _contract_profile_for_task(task: str, requested: str | None) -> str:
    """Resolve the CLI default without changing native or legacy defaults."""

    if requested is None:
        requested = (
            CONTRACT_PROFILE_COMMAND_V1
            if task == COMMAND_TASK_ID
            else CONTRACT_PROFILE_COMMAND_V2
            if task in COMMAND_V2_TASK_IDS
            else DEFAULT_CONTRACT_PROFILE
        )
    return validate_contract_profile(requested, task=task)


def _default_smoke_report_path(
    *, task: str, seed: int, num_envs: int, steps: int
) -> Path:
    """Return the additive report location used by the plan's short commands."""

    slug = "".join(character.lower() if character.isalnum() else "-" for character in task)
    slug = "-".join(part for part in slug.split("-") if part)
    return (
        ROOT
        / "runs"
        / "crazyflie_command_smoke"
        / f"{slug}-seed{seed}-{num_envs}env-{steps}steps.json"
    )


def _steady_state_sample_steps(total_steps: int, warmup_steps: int) -> tuple[int, ...]:
    """Return four equally spaced samples after one full episode warm-up.

    Isaac/PhysX and the task logger allocate a small amount of host memory on
    the first automatic episode reset.  Samples before that boundary are not
    like-for-like steady-state observations.  The required 1,000-step smoke
    leaves a 400-step measured window after the custom 600-step horizon (and a
    500-step window for the native task).  Short diagnostics retain quartile
    sampling but cannot claim the full steady-state memory gate.
    """

    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    start = warmup_steps if total_steps - warmup_steps >= 4 else 0
    span = total_steps - start
    samples = tuple(start + max(1, round(span * fraction / 4)) for fraction in range(1, 5))
    if len(set(samples)) != 4 or samples[-1] != total_steps:
        raise ValueError("smoke length is too short for four distinct memory samples")
    return samples


def _select_full_training_curriculum(
    env: Any,
    contract_profile: str = DEFAULT_CONTRACT_PROFILE,
) -> dict[str, Any]:
    """Select and verify the full-distribution stage before a custom smoke reset."""

    profile = validate_contract_profile(contract_profile)
    env.set_training_interactions(FULL_CURRICULUM_INTERACTIONS)
    task_contract = task_contract_payload(profile)
    expected_contract = task_contract["training_curriculum"]
    expected_stages = expected_contract["stages"]
    expected_stage_index = len(expected_stages) - 1
    active_stage = env.active_training_curriculum_stage_payload
    full_stage = env.full_training_curriculum_stage_payload
    if (
        env.training_interactions != FULL_CURRICULUM_INTERACTIONS
        or env.active_training_curriculum_stage_index != expected_stage_index
        or active_stage != expected_stages[-1]
        or full_stage != expected_stages[-1]
    ):
        raise RuntimeError("Custom smoke did not select the frozen full curriculum stage")
    return {
        "selection_timing": "before_first_explicit_smoke_reset",
        "contract_profile": profile,
        "requested_training_interactions": FULL_CURRICULUM_INTERACTIONS,
        "selected_training_interactions": env.training_interactions,
        "active_stage_index": env.active_training_curriculum_stage_index,
        "active_stage": active_stage,
        "full_stage": full_stage,
        "contract": task_contract,
    }


def _select_full_command_curriculum(
    env: Any,
    *,
    task: str = COMMAND_TASK_ID,
) -> dict[str, Any]:
    """Select and verify the selected command task's final curriculum stage."""

    profile = (
        CONTRACT_PROFILE_COMMAND_V1
        if task == COMMAND_TASK_ID
        else CONTRACT_PROFILE_COMMAND_V2
    )
    profile = validate_contract_profile(profile, task=task)
    interactions = (
        COMMAND_FULL_CURRICULUM_INTERACTIONS
        if task == COMMAND_TASK_ID
        else COMMAND_V2_FULL_CURRICULUM_INTERACTIONS
    )
    env.set_training_interactions(interactions)
    task_contract = task_contract_payload(profile, task=task)
    expected_stages = task_contract[
        "curriculum" if task == COMMAND_TASK_ID else "command_curriculum"
    ]
    expected_stage_index = len(expected_stages) - 1
    active_stage = env.active_training_curriculum_stage_payload
    full_stage = env.full_training_curriculum_stage_payload
    if (
        env.training_interactions != interactions
        or env.active_training_curriculum_stage_index != expected_stage_index
        or active_stage != expected_stages[-1]
        or full_stage != expected_stages[-1]
    ):
        raise RuntimeError("Command smoke did not select the final curriculum stage")
    return {
        "selection_timing": "before_first_explicit_smoke_reset",
        "contract_profile": profile,
        "requested_training_interactions": interactions,
        "selected_training_interactions": env.training_interactions,
        "active_stage_index": env.active_training_curriculum_stage_index,
        "active_stage": active_stage,
        "full_stage": full_stage,
        "contract": task_contract,
    }


def _assess_command_schedule_state(
    state: Mapping[str, Any],
    *,
    task: str = COMMAND_TASK_ID,
) -> dict[str, Any]:
    """Authenticate one live schedule cursor against the pure seeded sampler."""

    if task in COMMAND_V2_TASK_IDS:
        return _assess_wide_command_schedule_state(state, task=task)

    required = {
        "schema_version",
        "kind",
        "contract_sha256",
        "num_envs",
        "command_schedule_seed",
        "training_interactions",
        "next_command_segment_index",
        "command_steps_remaining",
        "requested_command_body",
        "command_category_code",
        "command_stage_index",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        return {
            "passed": False,
            "reason": "schedule state fields changed",
            "missing_fields": sorted(required - set(state)) if isinstance(state, Mapping) else sorted(required),
            "unexpected_fields": sorted(set(state) - required) if isinstance(state, Mapping) else [],
        }
    count = state["num_envs"]
    seed = state["command_schedule_seed"]
    interactions = state["training_interactions"]
    if (
        state["schema_version"] != 1
        or state["kind"] != "flyg1.crazyflie.command-schedule.v1"
        or state["contract_sha256"] != COMMAND_TRACKING_CONTRACT_SHA256
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not isinstance(interactions, int)
        or isinstance(interactions, bool)
        or interactions < 0
    ):
        return {"passed": False, "reason": "schedule state header is invalid"}

    next_indices = torch.as_tensor(state["next_command_segment_index"], dtype=torch.long)
    remaining = torch.as_tensor(state["command_steps_remaining"], dtype=torch.long)
    requested = torch.as_tensor(state["requested_command_body"], dtype=torch.float64)
    categories = torch.as_tensor(state["command_category_code"], dtype=torch.long)
    stages = torch.as_tensor(state["command_stage_index"], dtype=torch.long)
    vector_shape = (count,)
    shapes_valid = (
        requested.shape == (count, 4)
        and all(
            value.shape == vector_shape
            for value in (next_indices, remaining, categories, stages)
        )
    )
    if not shapes_valid:
        return {
            "passed": False,
            "reason": "schedule state tensor shapes changed",
            "requested_shape": list(requested.shape),
            "vector_shapes": [
                list(value.shape)
                for value in (next_indices, remaining, categories, stages)
            ],
        }

    range_valid = bool(
        torch.isfinite(requested).all()
        and torch.all(next_indices >= 1)
        and torch.all(remaining >= 1)
        and torch.all(remaining <= MAXIMUM_COMMAND_HOLD_STEPS)
        and torch.all(categories >= 0)
        and torch.all(categories < len(_COMMAND_CATEGORY_TO_CODE))
        and torch.all(stages >= 0)
        and torch.all(stages < len(COMMAND_CURRICULUM))
        and torch.allclose(
            requested,
            bound_command_body(requested),
            rtol=0.0,
            atol=1.0e-7,
        )
    )
    per_environment: list[dict[str, Any]] = []
    for env_id in range(count):
        segment_index = int(next_indices[env_id]) - 1
        stage_index = int(stages[env_id])
        stage_start = COMMAND_CURRICULUM[stage_index].start_interactions
        expected = sample_scheduled_command(
            seed=seed,
            environment_id=env_id,
            segment_index=segment_index,
            total_interactions=stage_start,
        )
        command_matches = bool(
            torch.allclose(
                requested[env_id],
                requested.new_tensor(expected.command),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        category_matches = (
            int(categories[env_id]) == _COMMAND_CATEGORY_TO_CODE[expected.category]
        )
        stage_matches = expected.stage_index == stage_index
        hold_matches = 1 <= int(remaining[env_id]) <= expected.hold_steps
        interaction_matches = stage_start <= interactions
        per_environment.append(
            {
                "environment_id": env_id,
                "segment_index": segment_index,
                "category": expected.category,
                "category_code": int(categories[env_id]),
                "stage_index": stage_index,
                "remaining_steps": int(remaining[env_id]),
                "expected_hold_steps": expected.hold_steps,
                "command_matches": command_matches,
                "category_matches": category_matches,
                "stage_matches": stage_matches,
                "interaction_clock_covers_stage": interaction_matches,
                "passed": bool(
                    command_matches
                    and category_matches
                    and stage_matches
                    and hold_matches
                    and interaction_matches
                ),
            }
        )
    return {
        "passed": bool(range_valid and all(row["passed"] for row in per_environment)),
        "range_valid": range_valid,
        "num_envs": count,
        "command_schedule_seed": seed,
        "training_interactions": interactions,
        "next_command_segment_index": next_indices.tolist(),
        "command_category_code": categories.tolist(),
        "command_stage_index": stages.tolist(),
        "per_environment": per_environment,
    }


def _assess_wide_command_schedule_state(
    state: Mapping[str, Any],
    *,
    task: str,
) -> dict[str, Any]:
    """Authenticate the common command cursor inside a command_v2 snapshot."""

    from g1_fly_control.tasks.crazyflie.command_wide_logic import (
        COMMAND_WIDE_SCHEDULE_STATE_KIND,
        command_wide_contract_payload,
        command_wide_training_contract_payload,
        command_wide_training_contract_sha256,
        sample_wide_scheduled_command,
    )

    required = {
        "schema_version",
        "kind",
        "contract_sha256",
        "num_envs",
        "command_schedule_seed",
        "training_interactions",
        "next_command_segment_index",
        "command_steps_remaining",
        "requested_command_body",
        "command_category_code",
        "command_stage_index",
    }
    if not isinstance(state, Mapping) or not required.issubset(state):
        return {
            "passed": False,
            "reason": "wide schedule state lacks required command cursor fields",
            "missing_fields": (
                sorted(required - set(state)) if isinstance(state, Mapping) else sorted(required)
            ),
        }
    count = state["num_envs"]
    seed = state["command_schedule_seed"]
    interactions = state["training_interactions"]
    wind_enabled = task == COMMAND_WIDE_WIND_TASK_ID
    expected_hash = command_wide_training_contract_sha256(
        wind_enabled=wind_enabled
    )
    if (
        state["schema_version"] != 2
        or state["kind"] != COMMAND_WIDE_SCHEDULE_STATE_KIND
        or state["contract_sha256"] != expected_hash
        or type(count) is not int
        or count < 1
        or type(seed) is not int
        or seed < 0
        or type(interactions) is not int
        or interactions < 0
    ):
        return {"passed": False, "reason": "wide schedule state header is invalid"}
    next_indices = torch.as_tensor(state["next_command_segment_index"], dtype=torch.long)
    remaining = torch.as_tensor(state["command_steps_remaining"], dtype=torch.long)
    requested = torch.as_tensor(state["requested_command_body"], dtype=torch.float64)
    categories = torch.as_tensor(state["command_category_code"], dtype=torch.long)
    stages = torch.as_tensor(state["command_stage_index"], dtype=torch.long)
    vectors = (next_indices, remaining, categories, stages)
    if requested.shape != (count, 4) or any(value.shape != (count,) for value in vectors):
        return {"passed": False, "reason": "wide schedule state tensor shapes changed"}
    compact = command_wide_contract_payload()
    complete = command_wide_training_contract_payload(wind_enabled=wind_enabled)
    curriculum = complete["command_curriculum"]
    bounded = bound_command_body(
        requested,
        maximum_horizontal_speed_m_s=float(compact["maximum_horizontal_speed_m_s"]),
        maximum_vertical_speed_m_s=float(compact["maximum_vertical_speed_m_s"]),
        maximum_yaw_rate_rad_s=float(compact["maximum_yaw_rate_rad_s"]),
    )
    range_valid = bool(
        torch.isfinite(requested).all()
        and torch.all(next_indices >= 1)
        and torch.all(remaining >= 1)
        and torch.all(remaining <= int(complete["command_hold_steps_inclusive"][1]))
        and torch.all(categories >= 0)
        and torch.all(categories < len(_COMMAND_CATEGORY_TO_CODE))
        and torch.all(stages >= 0)
        and torch.all(stages < len(curriculum))
        and torch.allclose(requested, bounded, rtol=0.0, atol=1.0e-7)
    )
    per_environment: list[dict[str, Any]] = []
    for env_id in range(count):
        segment_index = int(next_indices[env_id]) - 1
        stage_index = int(stages[env_id])
        stage_start = int(curriculum[stage_index]["start_interactions"])
        expected = sample_wide_scheduled_command(
            seed=seed,
            environment_id=env_id,
            segment_index=segment_index,
            total_interactions=stage_start,
        )
        command_matches = bool(
            torch.allclose(
                requested[env_id],
                requested.new_tensor(expected.command),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        row_passed = bool(
            command_matches
            and int(categories[env_id]) == _COMMAND_CATEGORY_TO_CODE[expected.category]
            and expected.stage_index == stage_index
            and 1 <= int(remaining[env_id]) <= expected.hold_steps
            and stage_start <= interactions
        )
        per_environment.append(
            {
                "environment_id": env_id,
                "segment_index": segment_index,
                "stage_index": stage_index,
                "category": expected.category,
                "passed": row_passed,
            }
        )
    return {
        "passed": bool(range_valid and all(row["passed"] for row in per_environment)),
        "range_valid": range_valid,
        "num_envs": count,
        "command_schedule_seed": seed,
        "training_interactions": interactions,
        "next_command_segment_index": next_indices.tolist(),
        "command_category_code": categories.tolist(),
        "command_stage_index": stages.tolist(),
        "per_environment": per_environment,
        "additional_state_fields": sorted(set(state) - required),
    }


def _assess_command_reward_components(
    components: Mapping[str, Any], *, num_envs: int
) -> dict[str, Any]:
    """Verify the complete finite/sign-correct command reward decomposition."""

    expected = set(COMMAND_REWARD_COMPONENTS)
    actual = set(components) if isinstance(components, Mapping) else set()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    shapes: dict[str, list[int]] = {}
    finite: dict[str, bool] = {}
    values: dict[str, torch.Tensor] = {}
    for name in sorted(actual & expected):
        value = torch.as_tensor(components[name])
        values[name] = value
        shapes[name] = list(value.shape)
        finite[name] = bool(torch.isfinite(value).all())
    shapes_valid = all(tuple(shape) == (num_envs,) for shape in shapes.values())
    finite_valid = all(finite.values())
    sign_checks = {
        "wrong_direction_acceleration_nonpositive": bool(
            "wrong_direction_acceleration" in values
            and torch.all(values["wrong_direction_acceleration"] <= 1.0e-8)
        ),
        "jerk_nonpositive": bool(
            "jerk" in values and torch.all(values["jerk"] <= 1.0e-8)
        ),
        "attitude_stability_nonpositive": bool(
            "attitude_stability" in values
            and torch.all(values["attitude_stability"] <= 1.0e-8)
        ),
        "angular_stability_nonpositive": bool(
            "angular_stability" in values
            and torch.all(values["angular_stability"] <= 1.0e-8)
        ),
        "survival_nonnegative": bool(
            "survival" in values and torch.all(values["survival"] >= -1.0e-8)
        ),
        "failure_nonpositive": bool(
            "failure" in values and torch.all(values["failure"] <= 1.0e-8)
        ),
    }
    return {
        "passed": bool(
            not missing
            and not unexpected
            and shapes_valid
            and finite_valid
            and all(sign_checks.values())
        ),
        "missing": missing,
        "unexpected": unexpected,
        "shapes": shapes,
        "finite": finite,
        "sign_checks": sign_checks,
    }


def _assess_command_terminal_snapshot(
    *,
    done: torch.Tensor,
    terminal_mask: torch.Tensor,
    terminal_observation: torch.Tensor,
    terminal_tracking_error_body: torch.Tensor,
    terminal_linear_acceleration_body: torch.Tensor,
    terminal_linear_jerk_body: torch.Tensor,
    terminal_failure_cause: torch.Tensor,
    terminal_position_w: torch.Tensor,
    terminal_command_target_position_w: torch.Tensor,
    failure_reward: torch.Tensor,
    expected_failure_cause: int,
) -> dict[str, Any]:
    """Check coherent, finite pre-reset safety telemetry for command control."""

    done_mask = torch.as_tensor(done, dtype=torch.bool)
    count = done_mask.numel()
    fields = {
        "terminal_mask": (torch.as_tensor(terminal_mask, dtype=torch.bool), (count,)),
        "terminal_observation": (torch.as_tensor(terminal_observation), (count, 12)),
        "terminal_tracking_error_body": (
            torch.as_tensor(terminal_tracking_error_body),
            (count, 4),
        ),
        "terminal_linear_acceleration_body": (
            torch.as_tensor(terminal_linear_acceleration_body),
            (count, 3),
        ),
        "terminal_linear_jerk_body": (
            torch.as_tensor(terminal_linear_jerk_body),
            (count, 3),
        ),
        "terminal_failure_cause": (
            torch.as_tensor(terminal_failure_cause, dtype=torch.long),
            (count,),
        ),
        "terminal_position_w": (torch.as_tensor(terminal_position_w), (count, 3)),
        "terminal_command_target_position_w": (
            torch.as_tensor(terminal_command_target_position_w),
            (count, 3),
        ),
        "failure_reward": (torch.as_tensor(failure_reward), (count,)),
    }
    shapes = {name: list(value.shape) for name, (value, _) in fields.items()}
    shapes_valid = tuple(done_mask.shape) == (count,) and all(
        tuple(value.shape) == expected for value, expected in fields.values()
    )
    selected = done_mask
    if not shapes_valid or not bool(selected.any()):
        return {
            "passed": False,
            "reason": "terminal snapshot shape mismatch or no selected terminal row",
            "shapes": shapes,
        }
    mask_exact = bool(torch.equal(fields["terminal_mask"][0], done_mask))
    finite = all(
        bool(torch.isfinite(value[selected]).all())
        for name, (value, _) in fields.items()
        if name not in {"terminal_mask", "terminal_failure_cause"}
    )
    causes = fields["terminal_failure_cause"][0][selected]
    cause_exact = bool(torch.all(causes == int(expected_failure_cause)))
    failure_penalty_applied = bool(
        torch.all(fields["failure_reward"][0][selected] < 0.0)
    )
    observation_error_matches = bool(
        torch.allclose(
            fields["terminal_observation"][0][selected, :3],
            fields["terminal_tracking_error_body"][0][selected, :3],
            rtol=0.0,
            atol=TERMINAL_SNAPSHOT_ABS_TOL,
        )
        and torch.allclose(
            fields["terminal_observation"][0][selected, 5],
            fields["terminal_tracking_error_body"][0][selected, 3],
            rtol=0.0,
            atol=TERMINAL_SNAPSHOT_ABS_TOL,
        )
    )
    return {
        "passed": bool(
            mask_exact
            and finite
            and cause_exact
            and failure_penalty_applied
            and observation_error_matches
        ),
        "shapes": shapes,
        "finite": finite,
        "terminal_mask_exact": mask_exact,
        "failure_cause_exact": cause_exact,
        "failure_penalty_applied": failure_penalty_applied,
        "observation_tracking_error_consistent": observation_error_matches,
        "expected_failure_cause": int(expected_failure_cause),
        "observed_failure_causes": causes.tolist(),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _obs_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["policy"]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Environment did not return a tensor policy observation")
    return value


def _reset_state(state: Any, done: torch.Tensor) -> Any:
    if state is None:
        return None
    if hasattr(state, "masked_reset"):
        return state.masked_reset(done)
    if isinstance(state, torch.Tensor):
        return state.masked_fill(done[:, None], 0.0)
    raise TypeError(f"Unknown recurrent state type: {type(state).__name__}")


def _counter(env: Any, *names: str) -> int:
    for name in names:
        value = getattr(env, name, None)
        if value is not None:
            return int(torch.as_tensor(value).sum().item())
    return 0


def _tensor_value(env: Any, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(env, name, None)
        if value is not None:
            return torch.as_tensor(value)
    return None


def _build_policy(
    kind: str,
    device: torch.device,
    manifest: Path,
    rewire_seed: int,
    rewire_manifest: Path,
    wing_manifest: Path,
    optic_manifest: Path,
):
    from g1_fly_control.crazyflie.controllers import build_controller
    result = build_controller(
        kind,
        observation_dim=12,
        action_dim=4,
        device=device,
        connectome_manifest=manifest,
        wing_connectome_manifest=wing_manifest,
        optic_connectome_manifest=optic_manifest,
        rewire_seed=rewire_seed,
        rewire_manifest_path=(rewire_manifest if kind == "frozen_lif_degree_rewired" else None),
        enforce_parameter_match=kind not in UNMATCHED_MULTI_CONNECTOME_POLICIES,
    )
    if isinstance(result, tuple):
        return result
    return result.policy, result.report


def _horizontal(value: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(value).clone()
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ValueError("impulse/momentum tensors must end in three world-frame coordinates")
    result[..., 2] = 0.0
    return result


def _assess_gust_impulse_evidence(
    *,
    expected_impulse_w: torch.Tensor,
    submitted_impulse_w: torch.Tensor,
    baseline_delta_momentum_w: torch.Tensor,
    gust_delta_momentum_w: torch.Tensor,
    recorded_gust_delta_momentum_w: torch.Tensor,
    response_absolute_tolerance_n_s: float,
    response_relative_tolerance: float,
) -> dict[str, Any]:
    """Assess submitted and measured gust evidence without Isaac dependencies."""

    tensors = tuple(
        torch.as_tensor(value, dtype=torch.float64)
        for value in (
            expected_impulse_w,
            submitted_impulse_w,
            baseline_delta_momentum_w,
            gust_delta_momentum_w,
            recorded_gust_delta_momentum_w,
        )
    )
    expected, submitted, baseline, gust, recorded = tensors
    if expected.ndim != 2 or expected.shape[1] != 3 or any(
        value.shape != expected.shape for value in tensors[1:]
    ):
        raise ValueError("gust evidence must use matching [num_envs, 3] tensors")
    if (
        not math.isfinite(response_absolute_tolerance_n_s)
        or response_absolute_tolerance_n_s < 0.0
        or not math.isfinite(response_relative_tolerance)
        or response_relative_tolerance < 0.0
    ):
        raise ValueError("gust-response tolerances must be finite and non-negative")

    expected_h = _horizontal(expected)
    submitted_h = _horizontal(submitted)
    baseline_h = _horizontal(baseline)
    gust_h = _horizontal(gust)
    recorded_h = _horizontal(recorded)
    paired_response_h = gust_h - baseline_h
    submitted_error = torch.linalg.vector_norm(submitted_h - expected_h, dim=-1)
    recorded_error = torch.linalg.vector_norm(recorded_h - gust_h, dim=-1)
    response_error = torch.linalg.vector_norm(paired_response_h - expected_h, dim=-1)
    expected_norm = torch.linalg.vector_norm(expected_h, dim=-1)
    response_tolerance = torch.maximum(
        torch.full_like(expected_norm, float(response_absolute_tolerance_n_s)),
        expected_norm * float(response_relative_tolerance),
    )
    finite = all(torch.isfinite(value).all().item() for value in tensors)
    per_environment_passed = (
        (expected_norm > 0.0)
        & (submitted_error <= SUBMITTED_IMPULSE_ABS_TOL_N_S)
        & (recorded_error <= SUBMITTED_IMPULSE_ABS_TOL_N_S)
        & (response_error <= response_tolerance)
    )
    return {
        "method": "paired_identical_action_rollouts_same_live_environment_v1",
        "horizontal_only": True,
        "expected_impulse_w_n_s": expected_h.tolist(),
        "submitted_force_time_integral_w_n_s": submitted_h.tolist(),
        "baseline_mass_delta_velocity_w_n_s": baseline_h.tolist(),
        "gust_mass_delta_velocity_w_n_s": gust_h.tolist(),
        "environment_recorded_gust_mass_delta_velocity_w_n_s": recorded_h.tolist(),
        "paired_measured_gust_response_w_n_s": paired_response_h.tolist(),
        "submitted_vector_error_n_s": submitted_error.tolist(),
        "recorded_measurement_consistency_error_n_s": recorded_error.tolist(),
        "paired_response_vector_error_n_s": response_error.tolist(),
        "submitted_absolute_tolerance_n_s": SUBMITTED_IMPULSE_ABS_TOL_N_S,
        "physical_response_absolute_tolerance_n_s": float(response_absolute_tolerance_n_s),
        "physical_response_relative_tolerance": float(response_relative_tolerance),
        "effective_physical_response_tolerance_n_s": response_tolerance.tolist(),
        "finite": bool(finite),
        "per_environment_passed": per_environment_passed.tolist(),
        "passed": bool(finite and torch.all(per_environment_passed).item()),
    }


def _assess_terminal_snapshot(
    *,
    done: torch.Tensor,
    terminal_mask: torch.Tensor,
    terminal_observation: torch.Tensor,
    terminal_goal_w: torch.Tensor,
    terminal_position_w: torch.Tensor,
    terminal_distance_m: torch.Tensor,
    terminal_speed_mps: torch.Tensor,
    terminal_failure_cause: torch.Tensor,
    expected_failure_cause: int,
) -> dict[str, Any]:
    """Check that terminal fields describe one coherent pre-reset state."""

    done_mask = torch.as_tensor(done, dtype=torch.bool)
    mask = torch.as_tensor(terminal_mask, dtype=torch.bool)
    observation = torch.as_tensor(terminal_observation)
    goal = torch.as_tensor(terminal_goal_w)
    position = torch.as_tensor(terminal_position_w)
    distance = torch.as_tensor(terminal_distance_m)
    speed = torch.as_tensor(terminal_speed_mps)
    cause = torch.as_tensor(terminal_failure_cause, dtype=torch.long)
    count = done_mask.numel()
    expected_shapes = {
        "terminal_mask": (count,),
        "terminal_observation": (count, 12),
        "terminal_goal_w": (count, 3),
        "terminal_position_w": (count, 3),
        "terminal_distance_m": (count,),
        "terminal_speed_mps": (count,),
        "terminal_failure_cause": (count,),
    }
    actual = {
        "terminal_mask": tuple(mask.shape),
        "terminal_observation": tuple(observation.shape),
        "terminal_goal_w": tuple(goal.shape),
        "terminal_position_w": tuple(position.shape),
        "terminal_distance_m": tuple(distance.shape),
        "terminal_speed_mps": tuple(speed.shape),
        "terminal_failure_cause": tuple(cause.shape),
    }
    if tuple(done_mask.shape) != (count,) or any(
        actual[name] != shape for name, shape in expected_shapes.items()
    ):
        raise ValueError(f"terminal snapshot shape mismatch: {actual}")
    selected = done_mask
    if not bool(selected.any()):
        return {"passed": False, "reason": "no terminal row was selected"}
    finite = bool(
        torch.isfinite(observation[selected]).all()
        and torch.isfinite(goal[selected]).all()
        and torch.isfinite(position[selected]).all()
        and torch.isfinite(distance[selected]).all()
        and torch.isfinite(speed[selected]).all()
    )
    observation_distance = torch.linalg.vector_norm(observation[:, 9:12], dim=-1)
    observation_speed = torch.linalg.vector_norm(observation[:, :3], dim=-1)
    world_distance = torch.linalg.vector_norm(goal - position, dim=-1)
    distance_consistent = bool(
        torch.allclose(
            observation_distance[selected], distance[selected],
            rtol=1.0e-4, atol=TERMINAL_SNAPSHOT_ABS_TOL,
        )
        and torch.allclose(
            world_distance[selected], distance[selected],
            rtol=1.0e-4, atol=TERMINAL_SNAPSHOT_ABS_TOL,
        )
    )
    speed_consistent = bool(
        torch.allclose(
            observation_speed[selected], speed[selected],
            rtol=1.0e-4, atol=TERMINAL_SNAPSHOT_ABS_TOL,
        )
    )
    mask_exact = bool(torch.equal(mask, done_mask))
    cause_exact = bool(torch.all(cause[selected] == int(expected_failure_cause)))
    return {
        "finite": finite,
        "terminal_mask_exact": mask_exact,
        "distance_consistent": distance_consistent,
        "speed_consistent": speed_consistent,
        "failure_cause_exact": cause_exact,
        "expected_failure_cause": int(expected_failure_cause),
        "observed_failure_causes": cause[selected].tolist(),
        "passed": finite and mask_exact and distance_consistent and speed_consistent and cause_exact,
    }


def _probe_plan(env: Any) -> dict[str, torch.Tensor]:
    """Build a deterministic, valid plan for live acceptance probes."""

    origins = env._terrain.env_origins.clone()
    count = int(env.num_envs)
    device = torch.device(env.device)
    root = torch.zeros((count, 13), dtype=torch.float32, device=device)
    root[:, :2] = origins[:, :2]
    root[:, 2] = 0.75
    root[:, 3] = 1.0  # Isaac root-state quaternion order is wxyz.

    offsets = torch.tensor(
        ((0.8, 0.0, 0.75), (-0.8, 0.0, 0.75), (0.0, 0.8, 0.75), (0.0, -0.8, 0.75)),
        dtype=torch.float32,
        device=device,
    )
    targets = offsets.unsqueeze(0).repeat(count, 1, 1)
    targets[..., :2] += origins[:, None, :2]
    if env.scenario != "waypoint_switch":
        targets = targets[:, :1, :]
    gust_directions = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(count, 1, 1)
    return {
        "initial_root_state_w": root,
        "targets_w": targets,
        "gust_directions_w": gust_directions,
    }


def _write_probe_kinematics(
    env: Any,
    position_w: torch.Tensor,
    *,
    linear_velocity_w: torch.Tensor | None = None,
) -> None:
    count = int(env.num_envs)
    device = torch.device(env.device)
    ids = torch.arange(count, dtype=torch.long, device=device)
    pose = torch.zeros((count, 7), dtype=torch.float32, device=device)
    pose[:, :3] = position_w
    pose[:, 3] = 1.0
    velocity = torch.zeros((count, 6), dtype=torch.float32, device=device)
    if linear_velocity_w is not None:
        velocity[:, :3] = linear_velocity_w
    env._robot.write_root_pose_to_sim(pose, ids)
    env._robot.write_root_velocity_to_sim(velocity, ids)


def _hover_action(env: Any, hover_action: float) -> torch.Tensor:
    action = torch.zeros((int(env.num_envs), 4), dtype=torch.float32, device=env.device)
    action[:, 0] = hover_action
    return action


def _run_command_failure_probes(env: Any, hover_action: float) -> dict[str, Any]:
    """Exercise all finite command-task safety resets plus nonfinite logic."""

    origins = env._terrain.env_origins.clone()
    probes = (
        (
            "low_height",
            COMMAND_FAILURE_LOW_HEIGHT,
            lambda: torch.cat(
                (
                    origins[:, :2],
                    torch.full_like(
                        origins[:, 2:3],
                        max(0.01, float(env.cfg.hard_minimum_height_m) - 0.05),
                    ),
                ),
                dim=-1,
            ),
        ),
        (
            "high_height",
            COMMAND_FAILURE_HIGH_HEIGHT,
            lambda: torch.cat(
                (
                    origins[:, :2],
                    torch.full_like(
                        origins[:, 2:3],
                        float(env.cfg.hard_maximum_height_m) + 0.20,
                    ),
                ),
                dim=-1,
            ),
        ),
        (
            "workspace_escape",
            COMMAND_FAILURE_WORKSPACE_ESCAPE,
            lambda: torch.stack(
                (
                    origins[:, 0]
                    + float(env.cfg.hard_workspace_xy_limit_m)
                    + 0.20,
                    origins[:, 1],
                    torch.full_like(origins[:, 2], float(env.cfg.spawn_height_m)),
                ),
                dim=-1,
            ),
        ),
    )
    results: dict[str, Any] = {}
    for name, expected_cause, position_factory in probes:
        env.reset(seed=PROBE_SEED)
        _write_probe_kinematics(env, position_factory())
        _, reward, terminated, truncated, _ = env.step(
            _hover_action(env, hover_action)
        )
        done = terminated | truncated
        snapshot = _assess_command_terminal_snapshot(
            done=done,
            terminal_mask=env.terminal_mask,
            terminal_observation=env.drone_terminal_observation,
            terminal_tracking_error_body=env.terminal_tracking_error_body,
            terminal_linear_acceleration_body=env.terminal_linear_acceleration_body,
            terminal_linear_jerk_body=env.terminal_linear_jerk_body,
            terminal_failure_cause=env.terminal_failure_cause,
            terminal_position_w=env.terminal_position_w,
            terminal_command_target_position_w=env.terminal_command_target_position_w,
            failure_reward=env.reward_components["failure"],
            expected_failure_cause=expected_cause,
        )
        reward_check = _assess_command_reward_components(
            env.reward_components, num_envs=int(env.num_envs)
        )
        results[name] = {
            "expected_failure_cause": expected_cause,
            "terminated": terminated.tolist(),
            "truncated": truncated.tolist(),
            "reward_finite": bool(torch.isfinite(reward).all()),
            "post_reset_episode_length_zero": bool(
                torch.all(env.episode_length_buf == 0)
            ),
            "terminal_snapshot": snapshot,
            "reward_components": reward_check,
            "passed": bool(
                torch.all(terminated).item()
                and not torch.any(truncated).item()
                and torch.isfinite(reward).all().item()
                and torch.all(env.episode_length_buf == 0).item()
                and snapshot["passed"]
                and reward_check["passed"]
            ),
        }

    synthetic = classify_command_failures(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([False]),
        minimum_height_m=float(env.cfg.hard_minimum_height_m),
        maximum_height_m=float(env.cfg.hard_maximum_height_m),
        workspace_xy_limit_m=float(env.cfg.hard_workspace_xy_limit_m),
    )
    nonfinite_passed = bool(
        synthetic.terminated.item()
        and synthetic.cause.item() == COMMAND_FAILURE_NONFINITE
    )
    results["nonfinite_logic_without_physics_injection"] = {
        "method": "pure_helper_shared_with_CommandFollowEnv._get_dones",
        "injected_nonfinite_into_simulator": False,
        "expected_failure_cause": COMMAND_FAILURE_NONFINITE,
        "observed_failure_cause": int(synthetic.cause.item()),
        "terminated": bool(synthetic.terminated.item()),
        "passed": nonfinite_passed,
    }
    return {
        "probes": results,
        "passed": all(value["passed"] for value in results.values()),
    }


def _terminal_snapshot_from_env(
    env: Any, done: torch.Tensor, *, expected_failure_cause: int
) -> dict[str, Any]:
    return _assess_terminal_snapshot(
        done=done,
        terminal_mask=env.terminal_mask,
        terminal_observation=env.drone_terminal_observation,
        terminal_goal_w=env.terminal_goal_w,
        terminal_position_w=env.terminal_position_w,
        terminal_distance_m=env.terminal_distance_m,
        terminal_speed_mps=env.terminal_speed_mps,
        terminal_failure_cause=env.terminal_failure_cause,
        expected_failure_cause=expected_failure_cause,
    )


def _run_horizon_and_schedule_probe(env: Any, hover_action: float) -> dict[str, Any]:
    """Run one reset-free, stabilized 600-decision protocol episode."""

    from g1_fly_control.tasks.crazyflie.logic import EPISODE_STEPS, FAILURE_NONE

    plan = _probe_plan(env)
    env.set_episode_plan(**plan)
    observation, _ = env.reset(seed=PROBE_SEED)
    observation = _obs_tensor(observation)
    started_at_zero = bool(torch.all(env.episode_length_buf == 0))
    max_episode_length = int(env.max_episode_length)
    switch_event_steps: list[int] = []
    gust_event_steps: list[int] = []
    previous_switch = env.switch_count.clone()
    previous_gust = env.gust_count.clone()
    unexpected_done_decisions: list[int] = []
    timeout_decisions: list[int] = []
    terminal_snapshot: dict[str, Any] = {"passed": False, "reason": "timeout was not observed"}
    timeout_terminated: list[bool] = []
    timeout_truncated: list[bool] = []

    with torch.no_grad():
        for decision in range(1, EPISODE_STEPS + 1):
            pre_step_index = int(env.episode_length_buf[0].item())
            # This is a schedule/ending probe, not a controller comparison.
            # Re-pinning to the active target keeps one episode alive without
            # resets while the real simulator, step counter, wrench API, done
            # logic, and automatic reset path all execute normally.
            _write_probe_kinematics(env, env._desired_pos_w.clone())
            next_observation, reward, terminated, truncated, _ = env.step(
                _hover_action(env, hover_action)
            )
            next_observation = _obs_tensor(next_observation)
            if not bool(torch.isfinite(next_observation).all() and torch.isfinite(reward).all()):
                unexpected_done_decisions.append(decision)
                break
            current_switch = env.switch_count.clone()
            current_gust = env.gust_count.clone()
            if bool(torch.any(current_switch > previous_switch)):
                # Switches are installed after DirectRLEnv increments the
                # completed-decision counter.
                switch_event_steps.append(decision)
            if bool(torch.any(current_gust > previous_gust)):
                # Gust configuration happens before physics at the zero-based
                # interval index.
                gust_event_steps.append(pre_step_index)
            previous_switch = current_switch
            previous_gust = current_gust
            done = terminated | truncated
            if bool(done.any()):
                if decision != EPISODE_STEPS:
                    unexpected_done_decisions.append(decision)
                else:
                    timeout_decisions.append(decision)
                    timeout_terminated = terminated.tolist()
                    timeout_truncated = truncated.tolist()
                    terminal_snapshot = _terminal_snapshot_from_env(
                        env, done, expected_failure_cause=FAILURE_NONE
                    )
                break
            observation = next_observation

    expected_switch_steps = list(env.cfg.switch_steps) if env.scenario == "waypoint_switch" else []
    expected_gust_steps = list(env.cfg.gust_steps) if env.scenario == "gust_recovery" else []
    terminal_switch_count = env.terminal_switch_count.tolist() if timeout_decisions else []
    terminal_gust_count = env.terminal_gust_count.tolist() if timeout_decisions else []
    expected_event_count = 3
    switch_schedule_passed = env.scenario != "waypoint_switch" or (
        switch_event_steps == expected_switch_steps
        and terminal_switch_count == [expected_event_count] * int(env.num_envs)
    )
    gust_schedule_passed = env.scenario != "gust_recovery" or (
        gust_event_steps == expected_gust_steps
        and terminal_gust_count == [expected_event_count] * int(env.num_envs)
    )
    submitted_impulse_passed = True
    submitted_impulse_error = None
    if env.scenario == "gust_recovery" and timeout_decisions:
        applied = env.terminal_gust_event_applied_impulse_w
        expected = env.terminal_gust_event_expected_impulse_w
        submitted_impulse_error = float(
            torch.linalg.vector_norm(applied - expected, dim=-1).max().item()
        )
        submitted_impulse_passed = submitted_impulse_error <= SUBMITTED_IMPULSE_ABS_TOL_N_S

    timeout_exact = (
        timeout_decisions == [EPISODE_STEPS]
        and timeout_terminated == [False] * int(env.num_envs)
        and timeout_truncated == [True] * int(env.num_envs)
        and bool(torch.all(env.episode_length_buf == 0))
    )
    result = {
        "method": "live_stabilized_uninterrupted_episode_v1",
        "stabilization": (
            "Before each live control decision, root pose is set to the active target with identity "
            "attitude and zero velocity; no reset occurs before the real 600-step timeout."
        ),
        "started_at_step_zero": started_at_zero,
        "reported_max_episode_length": max_episode_length,
        "expected_max_episode_length": EPISODE_STEPS,
        "timeout_decisions": timeout_decisions,
        "timeout_terminated": timeout_terminated,
        "timeout_truncated": timeout_truncated,
        "post_timeout_episode_length_zero": bool(torch.all(env.episode_length_buf == 0)),
        "unexpected_done_decisions": unexpected_done_decisions,
        "switch_event_step_indices": switch_event_steps,
        "expected_switch_event_step_indices": expected_switch_steps,
        "terminal_switch_count": terminal_switch_count,
        "gust_event_step_indices": gust_event_steps,
        "expected_gust_event_step_indices": expected_gust_steps,
        "terminal_gust_count": terminal_gust_count,
        "submitted_impulse_max_vector_error_n_s": submitted_impulse_error,
        "terminal_snapshot": terminal_snapshot,
        "checks": {
            "max_episode_length_is_600": max_episode_length == EPISODE_STEPS,
            "exact_timeout_is_truncation_not_termination": timeout_exact,
            "terminal_snapshot_consistent": terminal_snapshot.get("passed") is True,
            "switch_schedule_all_three_exact": switch_schedule_passed,
            "gust_schedule_all_three_exact": gust_schedule_passed,
            "all_gust_force_time_integrals_match": submitted_impulse_passed,
        },
    }
    result["passed"] = started_at_zero and not unexpected_done_decisions and all(
        result["checks"].values()
    )
    return result


def _run_paired_gust_response_probe(env: Any, hover_action: float) -> dict[str, Any]:
    """Measure external horizontal impulse with paired live rollouts."""

    from g1_fly_control.tasks.crazyflie.logic import (
        GUST_RESPONSE_ABS_TOL_N_S,
        GUST_RESPONSE_REL_TOL,
        MIXED_SCENARIO_NAMES,
    )

    if env.scenario != "gust_recovery":
        return {"applicable": False, "passed": True}
    original_scenario = env.scenario
    plan = _probe_plan(env)
    start_step = int(env.cfg.gust_steps[0])
    duration = int(env.cfg.gust_duration_steps)
    action = _hover_action(env, hover_action)
    mass = float(env.robot_mass_kg)
    baseline_done = False
    gust_done = False
    try:
        # Baseline: exact same environment, reset state, actions, and five
        # control decisions, with only the scenario's external gust disabled.
        env.scenario = "waypoint_reach"
        env.set_episode_plan(**plan)
        env.reset(seed=PROBE_SEED)
        # Fixed-scenario environments initialize this per-row code once in
        # __init__. Merely changing ``env.scenario`` for this paired probe does
        # not update it, while gust scheduling deliberately reads the per-row
        # code so mixed environments work. Set the probe rows explicitly so
        # the baseline truly differs only by the absence of the external gust.
        env.episode_scenario_code.fill_(MIXED_SCENARIO_NAMES.index("waypoint_reach"))
        env.episode_length_buf.fill_(start_step)
        baseline_start_position = env._robot.data.root_pos_w.clone()
        baseline_start_velocity = env._robot.data.root_lin_vel_w.clone()
        for _ in range(duration):
            _, _, terminated, truncated, _ = env.step(action)
            baseline_done |= bool((terminated | truncated).any())
        baseline_end_velocity = env._robot.data.root_lin_vel_w.clone()
        baseline_delta_momentum = mass * (baseline_end_velocity - baseline_start_velocity)

        # Gust: reset to the same immutable plan and execute the same actions.
        env.scenario = original_scenario
        env.set_episode_plan(**plan)
        env.reset(seed=PROBE_SEED)
        env.episode_scenario_code.fill_(MIXED_SCENARIO_NAMES.index("gust_recovery"))
        env.episode_length_buf.fill_(start_step)
        gust_start_position = env._robot.data.root_pos_w.clone()
        gust_start_velocity = env._robot.data.root_lin_vel_w.clone()
        for _ in range(duration):
            _, _, terminated, truncated, _ = env.step(action)
            gust_done |= bool((terminated | truncated).any())
        gust_end_velocity = env._robot.data.root_lin_vel_w.clone()
        gust_delta_momentum = mass * (gust_end_velocity - gust_start_velocity)
        submitted = env.gust_event_applied_impulse_w[:, 0, :].clone()
        expected = env.gust_event_expected_impulse_w[:, 0, :].clone()
        recorded = env.gust_event_measured_delta_momentum_w[:, 0, :].clone()
        assessment = _assess_gust_impulse_evidence(
            expected_impulse_w=expected,
            submitted_impulse_w=submitted,
            baseline_delta_momentum_w=baseline_delta_momentum,
            gust_delta_momentum_w=gust_delta_momentum,
            recorded_gust_delta_momentum_w=recorded,
            response_absolute_tolerance_n_s=GUST_RESPONSE_ABS_TOL_N_S,
            response_relative_tolerance=GUST_RESPONSE_REL_TOL,
        )
        start_state_error = float(
            torch.maximum(
                torch.abs(gust_start_position - baseline_start_position).max(),
                torch.abs(gust_start_velocity - baseline_start_velocity).max(),
            ).item()
        )
        assessment.update(
            {
                "applicable": True,
                "gust_event_index": 0,
                "gust_start_step": start_step,
                "control_decisions": duration,
                "physics_steps": duration * int(env.cfg.decimation),
                "duration_s": duration * float(env.step_dt),
                "robot_mass_kg": mass,
                "identical_start_state_max_abs_error": start_state_error,
                "baseline_terminated_or_truncated": baseline_done,
                "gust_terminated_or_truncated": gust_done,
            }
        )
        assessment["passed"] = bool(
            assessment["passed"]
            and start_state_error <= 1.0e-7
            and not baseline_done
            and not gust_done
        )
        return assessment
    finally:
        env.scenario = original_scenario


def _run_forced_failure_probes(env: Any, hover_action: float) -> dict[str, Any]:
    """Exercise finite live failures and the pure nonfinite decision branch."""

    from g1_fly_control.tasks.crazyflie.logic import (
        FAILURE_HIGH_HEIGHT,
        FAILURE_LOW_HEIGHT,
        FAILURE_NONFINITE,
        FAILURE_WORKSPACE_ESCAPE,
        classify_failure_causes,
    )

    plan = _probe_plan(env)
    origins = env._terrain.env_origins.clone()
    probes = (
        (
            "low_height",
            FAILURE_LOW_HEIGHT,
            lambda: torch.cat(
                (origins[:, :2], torch.full_like(origins[:, 2:3], 0.02)), dim=-1
            ),
        ),
        (
            "high_height",
            FAILURE_HIGH_HEIGHT,
            lambda: torch.cat(
                (
                    origins[:, :2],
                    torch.full_like(origins[:, 2:3], float(env.cfg.maximum_height_m) + 0.20),
                ),
                dim=-1,
            ),
        ),
        (
            "workspace_escape",
            FAILURE_WORKSPACE_ESCAPE,
            lambda: torch.stack(
                (
                    origins[:, 0] + float(env.cfg.workspace_xy_limit_m) + 0.20,
                    origins[:, 1],
                    torch.full_like(origins[:, 2], 0.75),
                ),
                dim=-1,
            ),
        ),
    )
    results: dict[str, Any] = {}
    for name, expected_cause, position_factory in probes:
        env.set_episode_plan(**plan)
        env.reset(seed=PROBE_SEED)
        _write_probe_kinematics(env, position_factory())
        _, reward, terminated, truncated, _ = env.step(_hover_action(env, hover_action))
        done = terminated | truncated
        snapshot = _terminal_snapshot_from_env(
            env, done, expected_failure_cause=expected_cause
        )
        results[name] = {
            "expected_failure_cause": expected_cause,
            "terminated": terminated.tolist(),
            "truncated": truncated.tolist(),
            "reward_finite": bool(torch.isfinite(reward).all()),
            "post_reset_episode_length_zero": bool(torch.all(env.episode_length_buf == 0)),
            "terminal_snapshot": snapshot,
            "passed": bool(
                torch.all(terminated).item()
                and not torch.any(truncated).item()
                and torch.isfinite(reward).all().item()
                and torch.all(env.episode_length_buf == 0).item()
                and snapshot["passed"]
            ),
        }

    # Never put NaN/Inf into PhysX.  This synthetic call executes the exact
    # pure helper used by CrazyflieEnv._get_dones.
    synthetic = classify_failure_causes(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([False]),
        minimum_height_m=float(env.cfg.minimum_height_m),
        maximum_height_m=float(env.cfg.maximum_height_m),
        workspace_xy_limit_m=float(env.cfg.workspace_xy_limit_m),
    )
    nonfinite_passed = bool(
        synthetic.terminated.item() and synthetic.cause.item() == FAILURE_NONFINITE
    )
    results["nonfinite_logic_without_physics_injection"] = {
        "method": "pure_helper_shared_with_CrazyflieEnv._get_dones",
        "injected_nonfinite_into_simulator": False,
        "expected_failure_cause": FAILURE_NONFINITE,
        "observed_failure_cause": int(synthetic.cause.item()),
        "terminated": bool(synthetic.terminated.item()),
        "passed": nonfinite_passed,
    }
    return {
        "probes": results,
        "passed": all(value["passed"] for value in results.values()),
    }


def _run_reset_isolation_probe(env: Any) -> dict[str, Any]:
    if int(env.num_envs) < 2:
        return {
            "tested": False,
            "passed": True,
            "gate_c_acceptance_satisfied": False,
            "note": "requires num_envs >= 2; the one-environment runtime smoke remains allowed",
        }

    plan = _probe_plan(env)
    env.set_episode_plan(**plan)
    env.reset(seed=PROBE_SEED)
    _write_probe_kinematics(env, env._desired_pos_w.clone())
    env.step(_hover_action(env, 2.0 / float(env.cfg.thrust_to_weight) - 1.0))
    fields = {
        "episode_length_buf": env.episode_length_buf[1:].clone(),
        "root_state_w": env._robot.data.root_state_w[1:].clone(),
        "desired_pos_w": env._desired_pos_w[1:].clone(),
        "target_sequence_w": env._target_sequence_w[1:].clone(),
        "switch_count": env.switch_count[1:].clone(),
        "gust_count": env.gust_count[1:].clone(),
        "success_count": env.success_count[1:].clone(),
        "gust_event_applied_impulse_w": env.gust_event_applied_impulse_w[1:].clone(),
    }
    reset_row_previous_episode_step = int(env.episode_length_buf[0].item())
    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
    reset_row_reinitialized = bool(
        reset_row_previous_episode_step > 0 and int(env.episode_length_buf[0].item()) == 0
    )
    unchanged = {
        "episode_length_buf": torch.equal(fields["episode_length_buf"], env.episode_length_buf[1:]),
        "root_state_w": torch.equal(fields["root_state_w"], env._robot.data.root_state_w[1:]),
        "desired_pos_w": torch.equal(fields["desired_pos_w"], env._desired_pos_w[1:]),
        "target_sequence_w": torch.equal(fields["target_sequence_w"], env._target_sequence_w[1:]),
        "switch_count": torch.equal(fields["switch_count"], env.switch_count[1:]),
        "gust_count": torch.equal(fields["gust_count"], env.gust_count[1:]),
        "success_count": torch.equal(fields["success_count"], env.success_count[1:]),
        "gust_event_applied_impulse_w": torch.equal(
            fields["gust_event_applied_impulse_w"], env.gust_event_applied_impulse_w[1:]
        ),
    }
    return {
        "tested": True,
        "reset_environment_zero_reinitialized": reset_row_reinitialized,
        "passed": reset_row_reinitialized and all(unchanged.values()),
        "gate_c_acceptance_satisfied": reset_row_reinitialized and all(unchanged.values()),
        "unaffected_environment_fields": unchanged,
    }


def _run_gate_c_acceptance_probes(env: Any, hover_action: float) -> dict[str, Any]:
    horizon = _run_horizon_and_schedule_probe(env, hover_action)
    gust_response = _run_paired_gust_response_probe(env, hover_action)
    failures = _run_forced_failure_probes(env, hover_action)
    reset_isolation = _run_reset_isolation_probe(env)
    result = {
        "horizon_schedule_and_terminal": horizon,
        "gust_impulse_evidence": gust_response,
        "failure_termination": failures,
        "reset_isolation": reset_isolation,
    }
    result["runtime_tier_passed"] = bool(
        horizon["passed"]
        and gust_response["passed"]
        and failures["passed"]
        and reset_isolation["passed"]
    )
    result["full_gate_c_acceptance_passed"] = bool(
        result["runtime_tier_passed"] and reset_isolation["gate_c_acceptance_satisfied"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contract_profile",
        choices=CONTRACT_PROFILES,
        default=None,
        help=(
            "Explicit task contract. Defaults to command_v1 for CommandFollow "
            "or command_v2 for Wide tasks, and balanced_v3 for native/legacy tasks."
        ),
    )
    parser.add_argument(
        "--offline_only",
        action="store_true",
        help="Require the already-pinned local Crazyflie USD and procedural ground",
    )
    parser.add_argument("--hover_centered_random_actions", action="store_true")
    parser.add_argument("--random_action_scale", type=float, default=0.02)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--ppo_update", action="store_true", help="Run one real collect/backward/update after smoke steps")
    parser.add_argument("--connectome_manifest", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--wing_connectome_manifest", type=Path, default=DEFAULT_WING_CONNECTOME)
    parser.add_argument("--optic_connectome_manifest", type=Path, default=DEFAULT_OPTIC_CONNECTOME)
    parser.add_argument("--rewire_seed", type=int, default=20260916)
    parser.add_argument("--rewire_manifest", type=Path, default=DEFAULT_REWIRE_MANIFEST)
    parser.add_argument("--output_report", type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.task is None:
        parser.error("--task is required")
    if args.task != "Isaac-Quadcopter-Direct-v0" and args.task not in CUSTOM_TASKS:
        parser.error("--task must be the native quadcopter or a project task ID")
    try:
        args.contract_profile = _contract_profile_for_task(
            args.task, args.contract_profile
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_report is None:
        args.output_report = _default_smoke_report_path(
            task=args.task,
            seed=args.seed,
            num_envs=args.num_envs,
            steps=args.steps,
        )
    if (
        args.contract_profile == CONTRACT_PROFILE_BALANCED_V4
        and args.task == "FlyCrazyflie-Mixed-v0"
    ):
        parser.error("balanced_v4 does not support Mixed training")
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive")
    if not math.isfinite(args.random_action_scale) or not 0 <= args.random_action_scale <= 1:
        parser.error("--random_action_scale must be finite and in [0, 1]")
    if args.ppo_update and not args.policy:
        parser.error("--ppo_update requires --policy")
    connectome = args.connectome_manifest.resolve()
    wing_connectome = args.wing_connectome_manifest.resolve()
    optic_connectome = args.optic_connectome_manifest.resolve()
    rewire_manifest_path = args.rewire_manifest.resolve()
    if args.policy in LEG_CONNECTOME_POLICIES and not connectome.is_file():
        parser.error(f"Connectome manifest does not exist: {connectome}")
    if args.policy in WING_CONNECTOME_POLICIES and not wing_connectome.is_file():
        parser.error(f"Wing connectome manifest does not exist: {wing_connectome}")
    if args.policy in OPTIC_CONNECTOME_POLICIES and not optic_connectome.is_file():
        parser.error(f"Optic connectome manifest does not exist: {optic_connectome}")
    if args.policy and not rewire_manifest_path.is_file():
        parser.error(f"Rewire manifest does not exist: {rewire_manifest_path}")

    resolved_config = {
        "kind": "smoke",
        "task": args.task,
        "contract_profile": args.contract_profile,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "policy": args.policy,
        "hover_centered_random_actions": args.hover_centered_random_actions,
        "random_action_scale": args.random_action_scale,
        "ppo_update": args.ppo_update,
        "offline_only": args.offline_only,
        "connectome_manifest": (
            str(connectome)
            if args.policy in LEG_CONNECTOME_POLICIES
            else str(wing_connectome)
            if args.policy in {"wing_lif", "wing_optic_lif"}
            else str(optic_connectome)
            if args.policy == "optic_lif"
            else None
        ),
        "wing_connectome_manifest": (
            str(wing_connectome) if args.policy in WING_CONNECTOME_POLICIES else None
        ),
        "optic_connectome_manifest": (
            str(optic_connectome) if args.policy in OPTIC_CONNECTOME_POLICIES else None
        ),
        "rewire_seed": args.rewire_seed,
        "rewire_manifest": str(rewire_manifest_path),
        "headless": bool(getattr(args, "headless", False)),
        "survival_first_contract": (
            task_contract_payload(args.contract_profile)
            if args.task in LEGACY_CUSTOM_TASKS
            and args.contract_profile == CONTRACT_PROFILE_SURVIVAL_V2
            else None
        ),
        "balanced_task_contract": (
            task_contract_payload(args.contract_profile)
            if args.task in LEGACY_CUSTOM_TASKS
            and args.contract_profile == CONTRACT_PROFILE_BALANCED_V3
            else None
        ),
        "balanced_v4_task_contract": (
            task_contract_payload(args.contract_profile)
            if args.task in LEGACY_CUSTOM_TASKS
            and args.contract_profile == CONTRACT_PROFILE_BALANCED_V4
            else None
        ),
        "command_follow_contract": (
            task_contract_payload(args.contract_profile, task=args.task)
            if args.task in COMMAND_TASKS
            else None
        ),
        "curriculum_probe_interactions": (
            COMMAND_FULL_CURRICULUM_INTERACTIONS
            if args.task == COMMAND_TASK_ID
            else COMMAND_V2_FULL_CURRICULUM_INTERACTIONS
            if args.task in COMMAND_V2_TASK_IDS
            else FULL_CURRICULUM_INTERACTIONS
            if args.task in LEGACY_CUSTOM_TASKS
            else None
        ),
    }
    try:
        fingerprint_rewire_manifest = load_fingerprint_rewire_manifest(
            rewire_manifest_path,
            expected_seed=args.rewire_seed,
        )
        fingerprint, payload = reproduction_fingerprint(
            resolved_config=resolved_config,
            connectome_manifest=(
                wing_connectome
                if args.policy in {"wing_lif", "wing_optic_lif"}
                else optic_connectome
                if args.policy == "optic_lif"
                else connectome
                if args.policy in LEG_CONNECTOME_POLICIES
                else None
            ),
            rewired_manifest=fingerprint_rewire_manifest,
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"cannot resolve reproduction fingerprint: {exc}")
    print(json.dumps({
        "status": "RESOLVED",
        "resolved_config": resolved_config,
        "fingerprint": fingerprint,
        "outputs": {"report": str(args.output_report.resolve())},
    }, sort_keys=True), flush=True)

    torch.manual_seed(args.seed)
    app = AppLauncher(args).app
    env = None
    report: dict[str, Any] = {}
    curriculum_report: dict[str, Any] | None = None
    try:
        from g1_fly_control.crazyflie.memory import assess, reset_cuda_peak, snapshot

        env = launch_environment(
            args.task,
            args.num_envs,
            # Mixed is deliberately training-only.  Planned deterministic
            # evaluation belongs to the three fixed held-out scenarios.
            deterministic_evaluation=args.task in GATE_C_TASKS,
            mixed_scenario_seed=(
                args.seed if args.task == "FlyCrazyflie-Mixed-v0" else None
            ),
            command_schedule_seed=(
                args.seed if args.task in COMMAND_TASKS else None
            ),
            contract_profile=args.contract_profile,
        )
        device = torch.device(env.device)
        reset_cuda_peak(device)
        memory_samples = [snapshot("environment_loaded", device, step=0)]
        offline_scene_report = getattr(env, "_flyg1_offline_scene_report", None)
        if not isinstance(offline_scene_report, Mapping):
            raise RuntimeError("Environment lacks verified offline-scene provenance")
        if args.task in COMMAND_TASKS:
            curriculum_report = _select_full_command_curriculum(
                env,
                task=args.task,
            )
        elif args.task in LEGACY_CUSTOM_TASKS:
            curriculum_report = _select_full_training_curriculum(
                env,
                args.contract_profile,
            )
        observation, _ = env.reset(seed=args.seed)
        observation = _obs_tensor(observation)
        deterministic_episode_start = int(env.episode_length_buf.abs().max().item()) == 0
        if observation.shape != (args.num_envs, 12):
            raise RuntimeError(f"Expected observation shape {(args.num_envs, 12)}, received {tuple(observation.shape)}")
        policy = None
        policy_report = None
        policy_normalizer = None
        state = None
        frozen_checksum_before = None
        if args.policy:
            from g1_fly_control.crazyflie.controllers import controller_core_checksum
            from g1_fly_control.crazyflie.normalization import RunningMeanVariance

            policy, policy_report = _build_policy(
                args.policy,
                device,
                connectome,
                args.rewire_seed,
                rewire_manifest_path,
                wing_connectome,
                optic_connectome,
            )
            policy.eval()
            policy_normalizer = RunningMeanVariance.create(12, device=device)
            state = policy.initial_state(args.num_envs, device=device) if hasattr(policy, "initial_state") else None
            frozen_checksum_before = controller_core_checksum(policy)
        hover_action = 2.0 / float(env.cfg.thrust_to_weight) - 1.0
        terminated_count = 0
        truncated_count = 0
        invalid_count = 0
        max_observation_abs = 0.0
        max_reward_abs = 0.0
        max_action_abs = 0.0
        observed_switch_events = 0
        observed_gust_events = 0
        previous_switch_counter = _tensor_value(env, "switch_count", "_switch_count", "drone_switch_count")
        previous_gust_counter = _tensor_value(env, "gust_count", "_gust_count", "drone_gust_count")
        previous_switch_counter = previous_switch_counter.clone() if previous_switch_counter is not None else None
        previous_gust_counter = previous_gust_counter.clone() if previous_gust_counter is not None else None
        maximum_completed_submitted_impulse_error = 0.0
        completed_submitted_impulse_integrals = 0
        previous_completed_impulse_mask: torch.Tensor | None = None
        command_schedule_snapshots_validated = 0
        command_schedule_first_failure: dict[str, Any] | None = None
        command_schedule_transition_valid = True
        command_changes_by_environment = [0 for _ in range(args.num_envs)]
        command_categories_seen = [set() for _ in range(args.num_envs)]
        command_reward_snapshots_validated = 0
        command_reward_first_failure: dict[str, Any] | None = None
        previous_command_segment_indices: torch.Tensor | None = None
        maximum_wind_force_norm_n = 0.0
        maximum_wind_torque_norm_nm = 0.0
        wind_telemetry_finite = True
        if args.task in COMMAND_TASKS:
            initial_schedule_state = env.command_schedule_state_dict()
            initial_schedule_check = _assess_command_schedule_state(
                initial_schedule_state,
                task=args.task,
            )
            command_schedule_snapshots_validated += 1
            if not initial_schedule_check["passed"]:
                command_schedule_first_failure = {
                    "step": 0,
                    "assessment": initial_schedule_check,
                }
            previous_command_segment_indices = torch.as_tensor(
                initial_schedule_state["next_command_segment_index"],
                dtype=torch.long,
            )
            for env_id, category in enumerate(
                initial_schedule_state["command_category_code"]
            ):
                command_categories_seen[env_id].add(int(category))
        memory_warmup_steps = int(getattr(env, "max_episode_length", 0))
        memory_sample_steps = _steady_state_sample_steps(args.steps, memory_warmup_steps)
        sample_steps = set(memory_sample_steps)
        terminal_observation_seen = False
        with torch.no_grad():
            for step in range(1, args.steps + 1):
                if policy is not None:
                    if policy_normalizer is None:
                        raise RuntimeError("Policy smoke loop lacks its fixed normalizer")
                    policy_observation = policy_normalizer.normalize(observation)
                    output = policy.act(policy_observation, state, deterministic=True)
                    action = output.action
                    state = output.state
                else:
                    action = torch.zeros((args.num_envs, 4), device=device)
                    if args.task in CUSTOM_TASKS and not args.hover_centered_random_actions:
                        # This diagnostic controller is deliberately outside
                        # the comparison set.  It only keeps the perturbed
                        # vehicle alive long enough to exercise 3/6/9 s task
                        # schedules while retaining the native action mapping.
                        position_z = env._robot.data.root_pos_w[:, 2]
                        linear_velocity_b = env._robot.data.root_lin_vel_b
                        angular_velocity_b = env._robot.data.root_ang_vel_b
                        gravity_b = env._robot.data.projected_gravity_b
                        action[:, 0] = (
                            hover_action
                            + 0.35 * (0.75 - position_z)
                            - 0.18 * env._robot.data.root_lin_vel_w[:, 2]
                            + 0.20 * (1.0 + gravity_b[:, 2])
                        )
                        action[:, 1] = (
                            0.08 * gravity_b[:, 1]
                            - 0.02 * angular_velocity_b[:, 0]
                            + 0.015 * linear_velocity_b[:, 1]
                        )
                        action[:, 2] = (
                            -0.08 * gravity_b[:, 0]
                            - 0.02 * angular_velocity_b[:, 1]
                            - 0.015 * linear_velocity_b[:, 0]
                        )
                        action[:, 3] = -0.01 * angular_velocity_b[:, 2]
                        action.clamp_(-1.0, 1.0)
                    else:
                        action[:, 0] = hover_action
                    if args.hover_centered_random_actions:
                        perturbation = (2 * torch.rand_like(action) - 1) * args.random_action_scale
                        action = (action + perturbation).clamp(-1.0, 1.0)
                if action.shape != (args.num_envs, 4) or not torch.isfinite(action).all():
                    raise RuntimeError("Controller emitted a nonfinite action or changed the 4-value contract")
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_observation = _obs_tensor(next_observation)
                if next_observation.shape != (args.num_envs, 12):
                    raise RuntimeError(
                        "Environment changed the 12-value policy observation contract"
                    )
                finite = torch.isfinite(next_observation).all() and torch.isfinite(reward).all()
                if not bool(finite):
                    invalid_count += 1
                    raise FloatingPointError(f"Nonfinite simulator state at control step {step}")
                done = terminated | truncated
                if args.task in COMMAND_TASKS:
                    if args.task in COMMAND_V2_TASK_IDS:
                        wind_force = env.applied_wind_force_world
                        wind_torque = env.applied_wind_torque_world
                        wind_telemetry_finite &= bool(
                            wind_force.shape == (args.num_envs, 3)
                            and wind_torque.shape == (args.num_envs, 3)
                            and torch.isfinite(wind_force).all()
                            and torch.isfinite(wind_torque).all()
                        )
                        maximum_wind_force_norm_n = max(
                            maximum_wind_force_norm_n,
                            float(torch.linalg.vector_norm(wind_force, dim=1).max()),
                        )
                        maximum_wind_torque_norm_nm = max(
                            maximum_wind_torque_norm_nm,
                            float(torch.linalg.vector_norm(wind_torque, dim=1).max()),
                        )
                    schedule_state = env.command_schedule_state_dict()
                    schedule_check = _assess_command_schedule_state(
                        schedule_state,
                        task=args.task,
                    )
                    command_schedule_snapshots_validated += 1
                    if (
                        not schedule_check["passed"]
                        and command_schedule_first_failure is None
                    ):
                        command_schedule_first_failure = {
                            "step": step,
                            "assessment": schedule_check,
                        }
                    current_indices = torch.as_tensor(
                        schedule_state["next_command_segment_index"],
                        dtype=torch.long,
                    )
                    if previous_command_segment_indices is None:
                        raise RuntimeError("Command smoke lost its schedule cursor")
                    segment_delta = current_indices - previous_command_segment_indices
                    if bool(
                        torch.any(segment_delta < 0)
                        | torch.any(segment_delta > 1)
                    ):
                        command_schedule_transition_valid = False
                    for env_id, delta in enumerate(segment_delta.tolist()):
                        command_changes_by_environment[env_id] += max(0, int(delta))
                    previous_command_segment_indices = current_indices
                    for env_id, category in enumerate(
                        schedule_state["command_category_code"]
                    ):
                        command_categories_seen[env_id].add(int(category))

                    reward_check = _assess_command_reward_components(
                        env.reward_components, num_envs=args.num_envs
                    )
                    command_reward_snapshots_validated += 1
                    if (
                        not reward_check["passed"]
                        and command_reward_first_failure is None
                    ):
                        command_reward_first_failure = {
                            "step": step,
                            "assessment": reward_check,
                        }
                if bool(done.any()):
                    terminal = getattr(env, "drone_terminal_observation", None)
                    if terminal is None:
                        terminal = getattr(env, "flyg1_terminal_observation", None)
                    terminal_observation_seen |= terminal is not None and bool(torch.isfinite(terminal).all())
                    state = _reset_state(state, done)
                terminated_count += int(terminated.sum())
                truncated_count += int(truncated.sum())
                max_observation_abs = max(max_observation_abs, float(next_observation.abs().max()))
                max_reward_abs = max(max_reward_abs, float(reward.abs().max()))
                max_action_abs = max(max_action_abs, float(action.abs().max()))
                observation = next_observation
                current_switch = _tensor_value(env, "switch_count", "_switch_count", "drone_switch_count")
                current_gust = _tensor_value(env, "gust_count", "_gust_count", "drone_gust_count")
                if current_switch is not None and previous_switch_counter is not None:
                    observed_switch_events += int((current_switch - previous_switch_counter).clamp_min(0).sum())
                    previous_switch_counter = current_switch.clone()
                if current_gust is not None and previous_gust_counter is not None:
                    observed_gust_events += int((current_gust - previous_gust_counter).clamp_min(0).sum())
                    previous_gust_counter = current_gust.clone()
                event_applied = _tensor_value(env, "gust_event_applied_impulse_w")
                event_expected = _tensor_value(env, "gust_event_expected_impulse_w")
                if event_applied is not None and event_expected is not None:
                    expected_norm = torch.linalg.vector_norm(event_expected, dim=-1)
                    applied_norm = torch.linalg.vector_norm(event_applied, dim=-1)
                    completed = (expected_norm > 0) & torch.isclose(
                        applied_norm, expected_norm, rtol=1.0e-4, atol=1.0e-7
                    )
                    if previous_completed_impulse_mask is None:
                        previous_completed_impulse_mask = torch.zeros_like(completed)
                    newly_completed = completed & ~previous_completed_impulse_mask
                    if bool(newly_completed.any()):
                        errors = torch.linalg.vector_norm(event_applied - event_expected, dim=-1)[newly_completed]
                        maximum_completed_submitted_impulse_error = max(
                            maximum_completed_submitted_impulse_error, float(errors.max())
                        )
                        completed_submitted_impulse_integrals += int(newly_completed.sum())
                    previous_completed_impulse_mask = completed.clone()
                if step in sample_steps:
                    memory_samples.append(snapshot("steady_state", device, step=step))
        command_failure_probes = None
        gate_c_acceptance = None
        if args.task in COMMAND_TASKS:
            try:
                command_failure_probes = _run_command_failure_probes(
                    env, hover_action
                )
                memory_samples.append(
                    snapshot("command_failure_probes", device, step=args.steps)
                )
            finally:
                # A forced safety reset must not leak into an optional PPO update.
                env.reset(seed=args.seed)
        elif args.task in GATE_C_TASKS:
            try:
                gate_c_acceptance = _run_gate_c_acceptance_probes(env, hover_action)
                memory_samples.append(snapshot("gate_c_acceptance_probes", device, step=args.steps))
            finally:
                # Probe plans must never leak into a learning-path smoke update.
                env.clear_episode_plan()
                env.reset(seed=args.seed)
        ppo_metrics = None
        if args.ppo_update:
            from g1_fly_control.crazyflie.normalization import NormalizedEnv, RunningMeanVariance
            from g1_fly_control.training import PPOConfig, RecurrentPPO

            policy.train()
            normalizer = RunningMeanVariance.create(12, device=device)
            normalized_env = NormalizedEnv(env, normalizer, training=True)
            runner = RecurrentPPO(policy, PPOConfig(horizon=4, ppo_epochs=1, target_kl=0.5))
            reset_cuda_peak(device)
            rollout, _, _ = runner.collect(normalized_env)
            ppo_metrics = runner.update(rollout)
            ppo_metrics["all_finite"] = all(math.isfinite(float(value)) for value in ppo_metrics.values())
            memory_samples.append(snapshot("ppo_forward_backward_update", device, step=args.steps))
        if policy is not None:
            from g1_fly_control.crazyflie.controllers import controller_core_checksum
            frozen_checksum_after = controller_core_checksum(policy)
        else:
            frozen_checksum_after = None
        memory_gate = assess(memory_samples)
        switch_count = max(observed_switch_events, _counter(env, "switch_count", "_switch_count", "drone_switch_count"))
        gust_count = max(observed_gust_events, _counter(env, "gust_count", "_gust_count", "drone_gust_count"))
        reset_isolation = (
            gate_c_acceptance["reset_isolation"]
            if gate_c_acceptance is not None
            else {"tested": False, "passed": True, "gate_c_acceptance_satisfied": False}
        )
        horizon_probe = (
            gate_c_acceptance["horizon_schedule_and_terminal"]
            if gate_c_acceptance is not None else None
        )
        gust_evidence = (
            gate_c_acceptance["gust_impulse_evidence"]
            if gate_c_acceptance is not None else None
        )
        failure_probe = (
            gate_c_acceptance["failure_termination"]
            if gate_c_acceptance is not None else None
        )
        submitted_impulse_report = None
        expected_impulse_report = None
        submitted_impulse_error_report = None
        if gust_evidence is not None and gust_evidence.get("applicable") is True:
            submitted_impulse_report = gust_evidence["submitted_force_time_integral_w_n_s"]
            expected_impulse_report = gust_evidence["expected_impulse_w_n_s"]
            submitted_impulse_error_report = max(gust_evidence["submitted_vector_error_n_s"])
        custom_checks = {}
        command_schedule_evidence = None
        command_reward_evidence = None
        if args.task in COMMAND_TASKS:
            expected_categories = set(_COMMAND_CATEGORY_TO_CODE.values())
            command_schedule_evidence = {
                "validated_snapshot_count": command_schedule_snapshots_validated,
                "expected_snapshot_count": args.steps + 1,
                "first_failure": command_schedule_first_failure,
                "transition_increments_are_zero_or_one": command_schedule_transition_valid,
                "command_changes_by_environment": command_changes_by_environment,
                "categories_seen_by_environment": [
                    sorted(categories) for categories in command_categories_seen
                ],
                "all_environments_changed_command": all(
                    changes > 0 for changes in command_changes_by_environment
                ),
                "all_environments_saw_all_categories": all(
                    categories == expected_categories
                    for categories in command_categories_seen
                ),
            }
            command_schedule_evidence["passed"] = bool(
                command_schedule_snapshots_validated == args.steps + 1
                and command_schedule_first_failure is None
                and command_schedule_transition_valid
                and command_schedule_evidence["all_environments_changed_command"]
                and command_schedule_evidence["all_environments_saw_all_categories"]
            )
            command_reward_evidence = {
                "required_components": list(COMMAND_REWARD_COMPONENTS),
                "validated_snapshot_count": command_reward_snapshots_validated,
                "expected_snapshot_count": args.steps,
                "first_failure": command_reward_first_failure,
                "passed": bool(
                    command_reward_snapshots_validated == args.steps
                    and command_reward_first_failure is None
                ),
            }
            custom_checks = {
                "minimum_500_control_steps": args.steps >= COMMAND_MINIMUM_SMOKE_STEPS,
                "final_command_curriculum_selected": (
                    curriculum_report is not None
                    and curriculum_report["active_stage_index"]
                    == len(
                        curriculum_report["contract"].get(
                            "curriculum",
                            curriculum_report["contract"].get("command_curriculum", []),
                        )
                    )
                    - 1
                    and curriculum_report["active_stage"]
                    == curriculum_report["full_stage"]
                ),
                "deterministic_seeded_command_schedule": command_schedule_evidence,
                "finite_named_reward_components": command_reward_evidence,
                "finite_12_value_observations": invalid_count == 0,
                "bounded_four_value_actions": max_action_abs <= 1.000001,
                "safety_reset_and_failure_telemetry": (
                    command_failure_probes
                    if command_failure_probes is not None
                    else {"passed": False, "reason": "probes did not run"}
                ),
            }
            if args.task in COMMAND_V2_TASK_IDS:
                maximum_force_limit_n = float(env._robot_weight) * 0.15
                maximum_torque_limit_nm = (
                    float(env._robot_weight) * 0.046 * 0.10
                )
                wind_enabled = args.task == COMMAND_WIDE_WIND_TASK_ID
                wind_physics_evidence = {
                    "wind_enabled": wind_enabled,
                    "telemetry_finite_and_shaped": wind_telemetry_finite,
                    "maximum_force_norm_n": maximum_wind_force_norm_n,
                    "maximum_force_limit_n": maximum_force_limit_n,
                    "maximum_torque_norm_nm": maximum_wind_torque_norm_nm,
                    "maximum_torque_limit_nm": maximum_torque_limit_nm,
                    "nonzero_wrench_seen": bool(
                        maximum_wind_force_norm_n > 0.0
                        or maximum_wind_torque_norm_nm > 0.0
                    ),
                }
                wind_physics_evidence["passed"] = bool(
                    wind_telemetry_finite
                    and maximum_wind_force_norm_n <= maximum_force_limit_n + 1.0e-7
                    and maximum_wind_torque_norm_nm
                    <= maximum_torque_limit_nm + 1.0e-9
                    and (
                        wind_physics_evidence["nonzero_wrench_seen"]
                        if wind_enabled
                        else (
                            maximum_wind_force_norm_n == 0.0
                            and maximum_wind_torque_norm_nm == 0.0
                        )
                    )
                )
                custom_checks["physical_world_frame_wind"] = wind_physics_evidence
        elif args.task in GATE_C_TASKS:
            custom_checks = {
                "full_training_curriculum_selected": (
                    curriculum_report is not None
                    and curriculum_report["active_stage_index"] == 3
                    and curriculum_report["active_stage"]["name"] == "full"
                ),
                "deterministic_evaluation_started_at_step_zero": (
                    deterministic_episode_start
                    and horizon_probe is not None
                    and horizon_probe["started_at_step_zero"]
                ),
                "exact_600_step_timeout_is_truncation": (
                    horizon_probe is not None
                    and horizon_probe["checks"]["max_episode_length_is_600"]
                    and horizon_probe["checks"]["exact_timeout_is_truncation_not_termination"]
                ),
                "terminal_snapshot_consistent": (
                    horizon_probe is not None
                    and horizon_probe["checks"]["terminal_snapshot_consistent"]
                ),
                "reset_isolation": reset_isolation,
                "forced_failure_codes_and_nonfinite_logic": (
                    failure_probe is not None and failure_probe["passed"]
                ),
                "switch_schedule_all_three_uninterrupted": (
                    horizon_probe is not None
                    and horizon_probe["checks"]["switch_schedule_all_three_exact"]
                ),
                "gust_schedule_all_three_uninterrupted": (
                    horizon_probe is not None
                    and horizon_probe["checks"]["gust_schedule_all_three_exact"]
                ),
                "gust_submitted_force_time_integrals_match": (
                    args.task != "FlyCrazyflie-GustRecovery-v0"
                    or (
                        horizon_probe is not None
                        and horizon_probe["checks"]["all_gust_force_time_integrals_match"]
                    )
                ),
                "gust_paired_horizontal_physical_response_matches": (
                    args.task != "FlyCrazyflie-GustRecovery-v0"
                    or (gust_evidence is not None and gust_evidence["passed"])
                ),
            }
        elif args.task == "FlyCrazyflie-Mixed-v0":
            scenario_starts = env.mixed_scenario_episode_starts.detach().cpu().tolist()
            custom_checks = {
                "full_training_curriculum_selected": (
                    curriculum_report is not None
                    and curriculum_report["active_stage_index"] == 3
                    and curriculum_report["active_stage"]["name"] == "full"
                ),
                "training_episode_started_at_step_zero": deterministic_episode_start,
                "mixed_scenario_starts_recorded": (
                    len(scenario_starts) == 3
                    and sum(int(count) for count in scenario_starts) >= args.num_envs
                ),
                "per_environment_round_robin_balance_preserved": (
                    len(scenario_starts) == 3
                    and max(scenario_starts) - min(scenario_starts) <= args.num_envs
                ),
            }
        passed = (
            invalid_count == 0
            and max_action_abs <= 1.000001
            and memory_gate["passed"]
            and (policy is None or frozen_checksum_before == frozen_checksum_after)
            and (ppo_metrics is None or ppo_metrics["all_finite"])
            and (args.task not in CUSTOM_TASKS or all(
                value if isinstance(value, bool) else value.get("passed", True)
                for value in custom_checks.values()
            ))
        )
        report = {
            "schema_version": 1,
            "status": "PASS" if passed else "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": args.task,
            "contract_profile": args.contract_profile,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "policy": args.policy,
            "ppo_update": args.ppo_update,
            "offline_only_requested": args.offline_only,
            "offline_scene": dict(offline_scene_report),
            "observations_finite": invalid_count == 0,
            "invalid_state_count": invalid_count,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "max_observation_abs": max_observation_abs,
            "max_reward_abs": max_reward_abs,
            "max_action_abs": max_action_abs,
            "switch_count": switch_count,
            "gust_count": gust_count,
            "mixed_scenario_episode_starts": (
                env.mixed_scenario_episode_starts.detach().cpu().tolist()
                if args.task == "FlyCrazyflie-Mixed-v0"
                else None
            ),
            "submitted_gust_force_time_integral_w_n_s": (
                submitted_impulse_report
            ),
            "expected_gust_impulse_w_n_s": expected_impulse_report,
            "submitted_gust_impulse_max_vector_error_n_s": submitted_impulse_error_report,
            "completed_submitted_gust_impulse_integrals": completed_submitted_impulse_integrals,
            "maximum_completed_submitted_gust_impulse_vector_error_n_s": (
                maximum_completed_submitted_impulse_error
                if completed_submitted_impulse_integrals else None
            ),
            "normal_smoke_terminal_observation_seen": terminal_observation_seen,
            "gate_c_acceptance": gate_c_acceptance,
            "command_schedule_evidence": command_schedule_evidence,
            "command_reward_evidence": command_reward_evidence,
            "command_failure_probes": command_failure_probes,
            "full_gate_c_acceptance_passed": (
                gate_c_acceptance["full_gate_c_acceptance_passed"]
                if gate_c_acceptance is not None else None
            ),
            "custom_checks": custom_checks,
            "policy_report": policy_report,
            "ppo_metrics": ppo_metrics,
            "frozen_core_checksum_before": frozen_checksum_before,
            "frozen_core_checksum_after": frozen_checksum_after,
            "memory_samples": memory_samples,
            "memory_gate": memory_gate,
            "memory_sampling_contract": {
                "scheme": "four_equally_spaced_samples_after_first_episode_boundary_v1",
                "first_episode_warmup_steps": memory_warmup_steps,
                "sample_steps": list(memory_sample_steps),
                "full_steady_state_window_available": args.steps - memory_warmup_steps >= 4,
            },
            "training_curriculum": (
                {
                    **curriculum_report,
                    "final_training_interactions": env.training_interactions,
                    "final_active_stage_index": env.active_training_curriculum_stage_index,
                    "final_active_stage": env.active_training_curriculum_stage_payload,
                }
                if curriculum_report is not None else None
            ),
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
        }
        _atomic_json(args.output_report.resolve(), report)
        print(json.dumps({
            "status": report["status"], "report": str(args.output_report.resolve()),
            "task": args.task, "num_envs": args.num_envs, "steps": args.steps,
            "policy": args.policy, "memory_gate": memory_gate,
            "terminated_count": terminated_count, "truncated_count": truncated_count,
            "switch_count": switch_count, "gust_count": gust_count,
            "resolved_config": resolved_config, "fingerprint": fingerprint,
        }, indent=2, sort_keys=True))
        return 0 if passed else 1
    except BaseException:
        traceback.print_exc()
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": args.task,
            "contract_profile": args.contract_profile,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "policy": args.policy,
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
            "training_curriculum": curriculum_report,
            "error": traceback.format_exc(),
        }
        _atomic_json(args.output_report.resolve(), failure)
        print(json.dumps({
            "status": "FAIL",
            "report": str(args.output_report.resolve()),
            "resolved_config": resolved_config,
            "fingerprint": fingerprint,
            "failure_reason": failure["error"],
        }, indent=2, sort_keys=True))
        return 1
    finally:
        if env is not None:
            env.close()
        # Preserve the smoke gate's process status; see the explicit
        # flush/os._exit sequence at the entrypoint below.


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
