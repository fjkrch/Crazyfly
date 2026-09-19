"""Isaac-independent bookkeeping for the Crazyflie scenarios.

The simulator environment owns tensors describing the vehicle.  This module
only defines the experiment timing and small, deterministic state machines so
that their boundary semantics can be tested without starting Isaac Sim.

Step convention
---------------
``step`` is a zero-based control-interval index.  An event with start ``s``
and duration ``d`` is active on ``s <= step < s + d``.  Consequently the
0.10 s gust starting at step 150 is applied on steps 150--154 and has ended at
the boundary at step 155.  Its recovery window contains the 100 interval
indices 155--254.  A recovery dwell completed on the last of those intervals
has latency exactly 2.0 s.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import torch


CONTROL_FREQUENCY_HZ = 50
CONTROL_DT_S = 1.0 / CONTROL_FREQUENCY_HZ

EPISODE_DURATION_S = 12.0
EPISODE_STEPS = 600

DEFAULT_SWITCH_TIMES_S = (3.0, 6.0, 9.0)
DEFAULT_SWITCH_STEPS = (150, 300, 450)
DEFAULT_GUST_TIMES_S = (3.0, 6.0, 9.0)
DEFAULT_GUST_STEPS = (150, 300, 450)
# More explicit aliases are useful at call sites that also handle event ends.
DEFAULT_GUST_START_STEPS = DEFAULT_GUST_STEPS

GUST_DURATION_S = 0.10
GUST_DURATION_STEPS = 5
GUST_DELTA_V_M_S = 0.75

SUCCESS_DISTANCE_M = 0.20
SUCCESS_SPEED_M_S = 0.25
SUCCESS_DWELL_S = 0.50
SUCCESS_DWELL_STEPS = 25

RECOVERY_WINDOW_S = 2.0
RECOVERY_WINDOW_STEPS = 100

# Gate-C compares the horizontal momentum response from two otherwise
# identical live rollouts: one without the scheduled external wrench and one
# with it.  The absolute term covers small solver/float noise, while the
# relative term scales with the mass-derived impulse.  These tolerances are
# part of the predeclared task contract and must not be fitted after a run.
GUST_RESPONSE_ABS_TOL_N_S = 5.0e-4
GUST_RESPONSE_REL_TOL = 0.10
# A submitted gust is reward-eligible only when its force-time integral agrees
# with the frozen, mass-derived world-frame impulse to this componentwise
# tolerance.  This is deliberately tighter than the measured-response gate:
# it authenticates the commanded disturbance, not noisy vehicle dynamics.
GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S = 1.0e-6
AUDITED_CRAZYFLIE_MASS_KG = 0.028200002387166023
CRAZYFLIE_MASS_ABS_TOL_KG = 1.0e-9

FAILURE_NONE = 0
FAILURE_LOW_HEIGHT = 1
FAILURE_HIGH_HEIGHT = 2
FAILURE_WORKSPACE_ESCAPE = 3
FAILURE_NONFINITE = 4
FAILURE_CAUSE_NAMES = {
    FAILURE_NONE: "none",
    FAILURE_LOW_HEIGHT: "ground_or_low_height",
    FAILURE_HIGH_HEIGHT: "high_height_escape",
    FAILURE_WORKSPACE_ESCAPE: "horizontal_workspace_escape",
    FAILURE_NONFINITE: "nonfinite_state",
}

# The fourth ID is training-only.  Each environment advances through this
# ordered tuple independently at episode resets, producing a real concurrent
# mixture without changing the three official held-out evaluation scenarios.
MIXED_SCENARIO_CONTRACT_VERSION = "crazyflie_mixed_round_robin_v1"
MIXED_SCENARIO_NAMES = (
    "waypoint_reach",
    "waypoint_switch",
    "gust_recovery",
)
SWITCH_TARGET_CURRICULUM_VERSION = "crazyflie_switch_followup_full_v1"
BALANCED_SWITCH_TARGET_CURRICULUM_VERSION = (
    "crazyflie_balanced_switch_active_stage_v1"
)
BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION = (
    "crazyflie_balanced_switch_active_stage_v2"
)


def switch_target_curriculum_payload() -> dict[str, object]:
    """Describe the feasible, task-specific Switch target sequence contract."""

    return {
        "version": SWITCH_TARGET_CURRICULUM_VERSION,
        "target_0_distribution": "active_episode_reset_curriculum_stage",
        "targets_1_through_3_distribution": "final_full_curriculum_stage",
        "applies_to_scenario": "waypoint_switch",
        "non_switch_mixed_rows_repeat_target_0": True,
        "reason": "early curriculum volumes cannot satisfy four consecutive minimum separations",
    }


def balanced_switch_target_curriculum_payload() -> dict[str, object]:
    """Describe balanced-v3 Switch targets without bypassing its curriculum."""

    return {
        "version": BALANCED_SWITCH_TARGET_CURRICULUM_VERSION,
        "target_0_distribution": "active_episode_reset_curriculum_stage",
        "targets_1_through_3_distribution": (
            "active_episode_reset_curriculum_stage"
        ),
        "applies_to_scenario": "waypoint_switch",
        "non_switch_mixed_rows_repeat_target_0": True,
        "reason": (
            "every balanced-v3 stage supports four consecutive target "
            "separations without exposing the full distribution early"
        ),
    }


def balanced_v4_switch_target_curriculum_payload() -> dict[str, object]:
    """Describe balanced-v4 Switch targets within the active stage."""

    return {
        "version": BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION,
        "target_0_distribution": "active_episode_reset_curriculum_stage",
        "targets_1_through_3_distribution": (
            "active_episode_reset_curriculum_stage"
        ),
        "applies_to_scenario": "waypoint_switch",
        "non_switch_mixed_rows_repeat_target_0": True,
        "mixed_training_supported": False,
        "reason": (
            "every balanced-v4 stage supports four consecutive target "
            "separations without exposing the full distribution early"
        ),
    }


def mixed_scenario_codes(
    environment_ids: torch.Tensor,
    episode_indices: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Assign deterministic, auditable per-environment mixed scenarios.

    Episode index zero uses ``(environment_id + seed) mod 3`` and every reset
    advances that environment by one scenario.  Thus each environment sees
    each task exactly once per three of its episodes, while multiple tasks are
    active concurrently in a vector environment.
    """

    env_ids = torch.as_tensor(environment_ids)
    episodes = torch.as_tensor(episode_indices, device=env_ids.device)
    if env_ids.shape != episodes.shape:
        raise ValueError("environment_ids and episode_indices must have matching shapes")
    if env_ids.dtype == torch.bool or env_ids.is_floating_point() or env_ids.is_complex():
        raise TypeError("environment_ids must use an integer dtype")
    if episodes.dtype == torch.bool or episodes.is_floating_point() or episodes.is_complex():
        raise TypeError("episode_indices must use an integer dtype")
    if torch.any(env_ids < 0) or torch.any(episodes < 0):
        raise ValueError("environment IDs and episode indices must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("mixed scenario seed must be a non-negative integer")
    return torch.remainder(env_ids.to(torch.long) + episodes.to(torch.long) + seed, 3)


def mixed_scenario_contract_payload(*, seed: int = 0) -> dict[str, object]:
    """Return the fingerprintable training-only mixture contract."""

    # Reuse the assignment validator without depending on a simulator.
    mixed_scenario_codes(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), seed=seed)
    return {
        "version": MIXED_SCENARIO_CONTRACT_VERSION,
        "training_only": True,
        "scenario_names_by_code": list(MIXED_SCENARIO_NAMES),
        "assignment": "(environment_id + per_environment_episode_index + seed) mod 3",
        "seed": seed,
        "exact_balance_window_per_environment_episodes": 3,
    }


REWARD_CONTRACT_VERSION = "crazyflie_survival_first_reward_v2"
DEFAULT_PROGRESS_REWARD_SCALE = 0.5
DEFAULT_PROGRESS_CLIP_M = 0.05
DEFAULT_SUCCESS_BONUS = 5.0
DEFAULT_SURVIVAL_REWARD_PER_INTERVAL = 0.02
DEFAULT_CONTROL_EFFORT_PENALTY_SCALE = 0.01
DEFAULT_ACTION_CHANGE_PENALTY_SCALE = 0.001
DEFAULT_FAILURE_PENALTY = 20.0
# The installed action mapping produces vehicle weight when
# (action[0] + 1) / 2 * 1.9 == 1.  Effort is therefore measured around this
# physically neutral collective command, not around the under-thrust command
# action[0] == 0.  Native body moments are much more sensitive near hover, so
# their normalized actions are expressed relative to the fixed 0.005
# reference before squaring.  Both squared metrics are capped to keep a bad
# command from dominating every other declared interval term.
DEFAULT_COLLECTIVE_HOVER_ACTION = 2.0 / 1.9 - 1.0
DEFAULT_MOMENT_ACTION_REFERENCE = 0.005
DEFAULT_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP = 4.0
DEFAULT_ACTION_CHANGE_NORMALIZED_SQUARED_CAP = 4.0

TRAINING_CURRICULUM_VERSION = "crazyflie_survival_first_curriculum_v2"

# The balanced task is a separate opt-in protocol.  The survival-v2 names and
# defaults above remain the legacy authority used by CrazyflieEnvCfg.
BALANCED_REWARD_CONTRACT_VERSION = "crazyflie_balanced_task_reward_v3"
BALANCED_PROGRESS_POTENTIAL_SCALE_M = 2.0
BALANCED_PROGRESS_REWARD_SCALE = 1.0
BALANCED_PROXIMITY_REWARD_SCALE = 0.015
BALANCED_PROXIMITY_DISTANCE_SCALE_M = 0.50
BALANCED_DWELL_REWARD_PER_INTERVAL = 0.020
BALANCED_BRAKING_PENALTY_SCALE = 0.010
BALANCED_BRAKING_SPEED_REFERENCE_M_S = 0.50
BALANCED_BRAKING_NORMALIZED_SQUARED_CAP = 4.0
BALANCED_SUCCESS_BONUS = 3.0
BALANCED_RETENTION_PENALTY_SCALE = 0.040
BALANCED_RETENTION_RAMP_M = 0.50
BALANCED_SURVIVAL_REWARD_PER_INTERVAL = 0.010
BALANCED_CONTROL_EFFORT_PENALTY_SCALE = 0.0025
BALANCED_COLLECTIVE_EFFORT_REFERENCE = 0.15
BALANCED_MOMENT_EFFORT_REFERENCE = 0.020
BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP = 4.0
BALANCED_ACTION_CHANGE_PENALTY_SCALE = 0.0005
BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE = 0.10
BALANCED_MOMENT_ACTION_CHANGE_REFERENCE = 0.010
BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP = 4.0
BALANCED_BOUNDARY_PENALTY_SCALE = 0.050
BALANCED_BOUNDARY_LOW_ONSET_M = 0.30
BALANCED_BOUNDARY_LOW_WIDTH_M = 0.20
BALANCED_BOUNDARY_HIGH_ONSET_M = 1.70
BALANCED_BOUNDARY_HIGH_WIDTH_M = 0.20
BALANCED_BOUNDARY_XY_ONSET_M = 2.25
BALANCED_BOUNDARY_XY_WIDTH_M = 0.25
BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP = 4.0
BALANCED_FAILURE_PENALTY = 25.0
BALANCED_TRAINING_CURRICULUM_VERSION = "crazyflie_balanced_task_curriculum_v3"

# Balanced-v4 is additive: all v2/v3 constants above remain immutable.  It
# strengthens only goal acquisition, advances full-distribution exposure for
# bounded proofs, and adds an authenticated one-shot gust-recovery event.
BALANCED_V4_REWARD_CONTRACT_VERSION = "crazyflie_balanced_task_reward_v4"
BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M = 2.0
BALANCED_V4_PROGRESS_REWARD_SCALE = 4.0
BALANCED_V4_PROXIMITY_REWARD_SCALE = 0.150
BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M = 0.80
BALANCED_V4_GUST_RECOVERY_BONUS = 3.0
BALANCED_V4_TRAINING_CURRICULUM_VERSION = "crazyflie_balanced_task_curriculum_v4"


@dataclass(frozen=True)
class TrainingCurriculumStage:
    """One reset/goal distribution selected by completed interactions.

    ``start_interactions`` is inclusive.  A stage is selected only when an
    episode resets; advancing the global interaction clock never interrupts or
    mutates an episode that is already in flight.
    """

    name: str
    start_interactions: int
    spawn_height_m: float
    spawn_position_xy_half_range_m: float
    spawn_position_z_half_range_m: float
    spawn_yaw_half_range_rad: float
    spawn_linear_velocity_half_range_mps: float
    spawn_angular_velocity_half_range_radps: float
    goal_xy_min_m: float
    goal_xy_max_m: float
    goal_z_min_m: float
    goal_z_max_m: float
    minimum_goal_separation_m: float

    def payload(self) -> dict[str, str | int | float]:
        """Return the canonical, JSON-safe representation used in fingerprints."""

        return {
            "name": self.name,
            "start_interactions": self.start_interactions,
            "spawn_height_m": self.spawn_height_m,
            "spawn_position_xy_half_range_m": self.spawn_position_xy_half_range_m,
            "spawn_position_z_half_range_m": self.spawn_position_z_half_range_m,
            "spawn_yaw_half_range_rad": self.spawn_yaw_half_range_rad,
            "spawn_linear_velocity_half_range_mps": self.spawn_linear_velocity_half_range_mps,
            "spawn_angular_velocity_half_range_radps": self.spawn_angular_velocity_half_range_radps,
            "goal_xy_min_m": self.goal_xy_min_m,
            "goal_xy_max_m": self.goal_xy_max_m,
            "goal_z_min_m": self.goal_z_min_m,
            "goal_z_max_m": self.goal_z_max_m,
            "minimum_goal_separation_m": self.minimum_goal_separation_m,
        }

    def reset_distribution_payload(self) -> dict[str, float]:
        """Return reset/goal values, excluding stage identity and timing."""

        return {
            key: float(value)
            for key, value in self.payload().items()
            if key not in {"name", "start_interactions"}
        }


DEFAULT_TRAINING_CURRICULUM = (
    TrainingCurriculumStage(
        name="vertical_lift",
        start_interactions=0,
        spawn_height_m=1.00,
        spawn_position_xy_half_range_m=0.01,
        spawn_position_z_half_range_m=0.005,
        spawn_yaw_half_range_rad=0.02,
        spawn_linear_velocity_half_range_mps=0.0,
        spawn_angular_velocity_half_range_radps=0.0,
        goal_xy_min_m=-0.02,
        goal_xy_max_m=0.02,
        goal_z_min_m=1.45,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.40,
    ),
    TrainingCurriculumStage(
        name="near",
        start_interactions=200_000,
        spawn_height_m=0.85,
        spawn_position_xy_half_range_m=0.025,
        spawn_position_z_half_range_m=0.01,
        spawn_yaw_half_range_rad=0.05,
        spawn_linear_velocity_half_range_mps=0.02,
        spawn_angular_velocity_half_range_radps=0.03,
        goal_xy_min_m=-0.50,
        goal_xy_max_m=0.50,
        goal_z_min_m=1.00,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.50,
    ),
    TrainingCurriculumStage(
        name="mid",
        start_interactions=500_000,
        spawn_height_m=0.70,
        spawn_position_xy_half_range_m=0.05,
        spawn_position_z_half_range_m=0.02,
        spawn_yaw_half_range_rad=0.10,
        spawn_linear_velocity_half_range_mps=0.04,
        spawn_angular_velocity_half_range_radps=0.07,
        goal_xy_min_m=-1.00,
        goal_xy_max_m=1.00,
        goal_z_min_m=0.75,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.65,
    ),
    TrainingCurriculumStage(
        name="full",
        start_interactions=1_000_000,
        spawn_height_m=0.50,
        spawn_position_xy_half_range_m=0.10,
        spawn_position_z_half_range_m=0.05,
        spawn_yaw_half_range_rad=0.25,
        spawn_linear_velocity_half_range_mps=0.10,
        spawn_angular_velocity_half_range_radps=0.20,
        goal_xy_min_m=-2.0,
        goal_xy_max_m=2.0,
        goal_z_min_m=0.50,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.75,
    ),
)


BALANCED_TRAINING_CURRICULUM = (
    TrainingCurriculumStage(
        name="near_3d",
        start_interactions=0,
        spawn_height_m=1.00,
        spawn_position_xy_half_range_m=0.01,
        spawn_position_z_half_range_m=0.005,
        spawn_yaw_half_range_rad=0.02,
        spawn_linear_velocity_half_range_mps=0.0,
        spawn_angular_velocity_half_range_radps=0.0,
        goal_xy_min_m=-0.35,
        goal_xy_max_m=0.35,
        goal_z_min_m=0.80,
        goal_z_max_m=1.20,
        minimum_goal_separation_m=0.40,
    ),
    TrainingCurriculumStage(
        name="local_3d",
        start_interactions=200_000,
        spawn_height_m=0.85,
        spawn_position_xy_half_range_m=0.025,
        spawn_position_z_half_range_m=0.01,
        spawn_yaw_half_range_rad=0.05,
        spawn_linear_velocity_half_range_mps=0.02,
        spawn_angular_velocity_half_range_radps=0.03,
        goal_xy_min_m=-0.75,
        goal_xy_max_m=0.75,
        goal_z_min_m=0.65,
        goal_z_max_m=1.35,
        minimum_goal_separation_m=0.50,
    ),
    TrainingCurriculumStage(
        name="mid_3d",
        start_interactions=500_000,
        spawn_height_m=0.70,
        spawn_position_xy_half_range_m=0.05,
        spawn_position_z_half_range_m=0.02,
        spawn_yaw_half_range_rad=0.10,
        spawn_linear_velocity_half_range_mps=0.04,
        spawn_angular_velocity_half_range_radps=0.07,
        goal_xy_min_m=-1.25,
        goal_xy_max_m=1.25,
        goal_z_min_m=0.60,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.65,
    ),
    TrainingCurriculumStage(
        name="full",
        start_interactions=1_000_000,
        spawn_height_m=0.50,
        spawn_position_xy_half_range_m=0.10,
        spawn_position_z_half_range_m=0.05,
        spawn_yaw_half_range_rad=0.25,
        spawn_linear_velocity_half_range_mps=0.10,
        spawn_angular_velocity_half_range_radps=0.20,
        goal_xy_min_m=-2.0,
        goal_xy_max_m=2.0,
        goal_z_min_m=0.50,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.75,
    ),
)


# The distributions are intentionally identical to balanced-v3.  Only the
# interaction boundaries change, so a 500k bounded proof completes hundreds
# of episodes on the same full distribution used by held-out evaluation.
BALANCED_V4_TRAINING_CURRICULUM = (
    TrainingCurriculumStage(
        name="near_3d",
        start_interactions=0,
        spawn_height_m=1.00,
        spawn_position_xy_half_range_m=0.01,
        spawn_position_z_half_range_m=0.005,
        spawn_yaw_half_range_rad=0.02,
        spawn_linear_velocity_half_range_mps=0.0,
        spawn_angular_velocity_half_range_radps=0.0,
        goal_xy_min_m=-0.35,
        goal_xy_max_m=0.35,
        goal_z_min_m=0.80,
        goal_z_max_m=1.20,
        minimum_goal_separation_m=0.40,
    ),
    TrainingCurriculumStage(
        name="local_3d",
        start_interactions=50_000,
        spawn_height_m=0.85,
        spawn_position_xy_half_range_m=0.025,
        spawn_position_z_half_range_m=0.01,
        spawn_yaw_half_range_rad=0.05,
        spawn_linear_velocity_half_range_mps=0.02,
        spawn_angular_velocity_half_range_radps=0.03,
        goal_xy_min_m=-0.75,
        goal_xy_max_m=0.75,
        goal_z_min_m=0.65,
        goal_z_max_m=1.35,
        minimum_goal_separation_m=0.50,
    ),
    TrainingCurriculumStage(
        name="mid_3d",
        start_interactions=125_000,
        spawn_height_m=0.70,
        spawn_position_xy_half_range_m=0.05,
        spawn_position_z_half_range_m=0.02,
        spawn_yaw_half_range_rad=0.10,
        spawn_linear_velocity_half_range_mps=0.04,
        spawn_angular_velocity_half_range_radps=0.07,
        goal_xy_min_m=-1.25,
        goal_xy_max_m=1.25,
        goal_z_min_m=0.60,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.65,
    ),
    TrainingCurriculumStage(
        name="full",
        start_interactions=250_000,
        spawn_height_m=0.50,
        spawn_position_xy_half_range_m=0.10,
        spawn_position_z_half_range_m=0.05,
        spawn_yaw_half_range_rad=0.25,
        spawn_linear_velocity_half_range_mps=0.10,
        spawn_angular_velocity_half_range_radps=0.20,
        goal_xy_min_m=-2.0,
        goal_xy_max_m=2.0,
        goal_z_min_m=0.50,
        goal_z_max_m=1.50,
        minimum_goal_separation_m=0.75,
    ),
)


def validate_training_curriculum(
    stages: Sequence[TrainingCurriculumStage],
) -> tuple[TrainingCurriculumStage, ...]:
    """Validate and freeze an interaction-gated reset curriculum."""

    normalized = tuple(stages)
    if not normalized:
        raise ValueError("training curriculum must contain at least one stage")
    if any(not isinstance(stage, TrainingCurriculumStage) for stage in normalized):
        raise TypeError("training curriculum entries must be TrainingCurriculumStage values")
    if normalized[0].start_interactions != 0:
        raise ValueError("training curriculum must begin at interaction 0")
    if len({stage.name for stage in normalized}) != len(normalized):
        raise ValueError("training curriculum stage names must be unique")

    previous_start = -1
    for stage in normalized:
        if not stage.name or stage.name.strip() != stage.name:
            raise ValueError("training curriculum stage names must be nonempty and trimmed")
        start = stage.start_interactions
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("stage start_interactions must be non-negative integers")
        if start <= previous_start:
            raise ValueError("training curriculum stage starts must be strictly increasing")
        previous_start = start

        values = stage.reset_distribution_payload()
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("training curriculum reset values must be finite")
        half_ranges = (
            stage.spawn_position_xy_half_range_m,
            stage.spawn_position_z_half_range_m,
            stage.spawn_yaw_half_range_rad,
            stage.spawn_linear_velocity_half_range_mps,
            stage.spawn_angular_velocity_half_range_radps,
        )
        if stage.spawn_height_m <= 0.0 or any(value < 0.0 for value in half_ranges):
            raise ValueError("spawn height must be positive and perturbation ranges non-negative")
        if stage.spawn_height_m - stage.spawn_position_z_half_range_m <= 0.0:
            raise ValueError("spawn distribution must remain above zero height")
        if stage.goal_xy_min_m >= stage.goal_xy_max_m:
            raise ValueError("goal XY minimum must be below its maximum")
        if stage.goal_z_min_m >= stage.goal_z_max_m or stage.goal_z_min_m <= 0.0:
            raise ValueError("goal Z bounds must be positive and strictly increasing")
        if stage.minimum_goal_separation_m <= 0.0:
            raise ValueError("minimum goal separation must be positive")

    return normalized


def validate_monotonic_training_interactions(previous: int, current: int) -> int:
    """Validate a checkpointable interaction-clock update and return it."""

    for name, value in (("previous", previous), ("current", current)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} training interactions must be a non-negative integer")
    if current < previous:
        raise ValueError("training interactions cannot decrease")
    return current


def advance_vector_training_interactions(current: int, num_envs: int) -> int:
    """Advance once after a real vector step, before terminal rows reset."""

    validate_monotonic_training_interactions(current, current)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    return current + num_envs


def training_curriculum_stage_index(
    total_interactions: int,
    stages: Sequence[TrainingCurriculumStage] = DEFAULT_TRAINING_CURRICULUM,
) -> int:
    """Select the inclusive stage index for an exact interaction count."""

    if isinstance(total_interactions, bool) or not isinstance(total_interactions, int):
        raise ValueError("total_interactions must be a non-negative integer")
    if total_interactions < 0:
        raise ValueError("total_interactions must be a non-negative integer")
    normalized = validate_training_curriculum(stages)
    selected = 0
    for index, stage in enumerate(normalized):
        if total_interactions < stage.start_interactions:
            break
        selected = index
    return selected


def select_training_curriculum_stage(
    total_interactions: int,
    stages: Sequence[TrainingCurriculumStage] = DEFAULT_TRAINING_CURRICULUM,
) -> TrainingCurriculumStage:
    """Return the stage selected by ``training_curriculum_stage_index``."""

    normalized = validate_training_curriculum(stages)
    return normalized[training_curriculum_stage_index(total_interactions, normalized)]


def reset_training_curriculum_stage_index(
    total_interactions: int,
    stages: Sequence[TrainingCurriculumStage] = DEFAULT_TRAINING_CURRICULUM,
    *,
    deterministic_eval: bool = False,
    episode_plan_installed: bool = False,
) -> int:
    """Select a new episode's stage, with evaluation forced to full scope."""

    normalized = validate_training_curriculum(stages)
    if deterministic_eval or episode_plan_installed:
        # Still validate the interaction count so a bad restored counter cannot
        # hide behind evaluation mode.
        training_curriculum_stage_index(total_interactions, normalized)
        return len(normalized) - 1
    return training_curriculum_stage_index(total_interactions, normalized)


def training_curriculum_payload(
    stages: Sequence[TrainingCurriculumStage] = DEFAULT_TRAINING_CURRICULUM,
    *,
    version: str = TRAINING_CURRICULUM_VERSION,
) -> dict[str, object]:
    """Return the versioned canonical payload required in resolved configs."""

    if not isinstance(version, str) or not version:
        raise ValueError("training curriculum version must be a nonempty string")
    normalized = validate_training_curriculum(stages)
    return {"version": version, "stages": [stage.payload() for stage in normalized]}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_curriculum_sha256(
    stages: Sequence[TrainingCurriculumStage] = DEFAULT_TRAINING_CURRICULUM,
    *,
    version: str = TRAINING_CURRICULUM_VERSION,
) -> str:
    """Hash the exact version and numeric reset curriculum."""

    return _canonical_sha256(training_curriculum_payload(stages, version=version))


def reward_contract_payload(
    *,
    progress_scale: float = DEFAULT_PROGRESS_REWARD_SCALE,
    progress_clip_m: float = DEFAULT_PROGRESS_CLIP_M,
    success_bonus: float = DEFAULT_SUCCESS_BONUS,
    survival_reward_per_interval: float = DEFAULT_SURVIVAL_REWARD_PER_INTERVAL,
    control_effort_scale: float = DEFAULT_CONTROL_EFFORT_PENALTY_SCALE,
    collective_hover_action: float = DEFAULT_COLLECTIVE_HOVER_ACTION,
    moment_action_reference: float = DEFAULT_MOMENT_ACTION_REFERENCE,
    control_effort_normalized_squared_cap: float = (
        DEFAULT_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP
    ),
    action_change_scale: float = DEFAULT_ACTION_CHANGE_PENALTY_SCALE,
    action_change_normalized_squared_cap: float = (
        DEFAULT_ACTION_CHANGE_NORMALIZED_SQUARED_CAP
    ),
    failure_penalty: float = DEFAULT_FAILURE_PENALTY,
    version: str = REWARD_CONTRACT_VERSION,
) -> dict[str, str | float]:
    """Return and validate the frozen survival-first reward contract."""

    if not isinstance(version, str) or not version:
        raise ValueError("reward contract version must be a nonempty string")
    values = {
        "progress_scale": float(progress_scale),
        "progress_clip_m": float(progress_clip_m),
        "success_bonus": float(success_bonus),
        "survival_reward_per_interval": float(survival_reward_per_interval),
        "control_effort_scale": float(control_effort_scale),
        "collective_hover_action": float(collective_hover_action),
        "moment_action_reference": float(moment_action_reference),
        "control_effort_normalized_squared_cap": float(
            control_effort_normalized_squared_cap
        ),
        "action_change_scale": float(action_change_scale),
        "action_change_normalized_squared_cap": float(
            action_change_normalized_squared_cap
        ),
        "failure_penalty": float(failure_penalty),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("reward contract values must be finite")
    nonnegative = (
        values["progress_scale"],
        values["success_bonus"],
        values["survival_reward_per_interval"],
        values["control_effort_scale"],
        values["action_change_scale"],
        values["failure_penalty"],
    )
    if any(value < 0.0 for value in nonnegative):
        raise ValueError("reward scales must be non-negative")
    positive = (
        values["progress_clip_m"],
        values["moment_action_reference"],
        values["control_effort_normalized_squared_cap"],
        values["action_change_normalized_squared_cap"],
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("reward clips, references, and caps must be positive")
    if not -1.0 < values["collective_hover_action"] < 1.0:
        raise ValueError("collective hover action must lie strictly inside [-1, 1]")
    return {"version": version, **values}


def reward_contract_sha256(**kwargs: float | str | bool) -> str:
    """Hash the exact version and numeric survival-first reward contract."""

    return _canonical_sha256(reward_contract_payload(**kwargs))  # type: ignore[arg-type]


DEFAULT_TRAINING_CURRICULUM_SHA256 = training_curriculum_sha256()
DEFAULT_REWARD_CONTRACT_SHA256 = reward_contract_sha256()


def survival_first_contract_payload() -> dict[str, object]:
    """Return the single CPU-safe authority consumed by config fingerprints."""

    return {
        "reward": reward_contract_payload(),
        "reward_sha256": DEFAULT_REWARD_CONTRACT_SHA256,
        "training_curriculum": training_curriculum_payload(),
        "training_curriculum_sha256": DEFAULT_TRAINING_CURRICULUM_SHA256,
    }


def balanced_reward_contract_payload(
    *,
    progress_potential_scale_m: float = BALANCED_PROGRESS_POTENTIAL_SCALE_M,
    progress_scale: float = BALANCED_PROGRESS_REWARD_SCALE,
    proximity_scale: float = BALANCED_PROXIMITY_REWARD_SCALE,
    proximity_distance_scale_m: float = BALANCED_PROXIMITY_DISTANCE_SCALE_M,
    dwell_reward_per_interval: float = BALANCED_DWELL_REWARD_PER_INTERVAL,
    braking_scale: float = BALANCED_BRAKING_PENALTY_SCALE,
    braking_speed_reference_m_s: float = BALANCED_BRAKING_SPEED_REFERENCE_M_S,
    braking_normalized_squared_cap: float = BALANCED_BRAKING_NORMALIZED_SQUARED_CAP,
    success_bonus: float = BALANCED_SUCCESS_BONUS,
    retention_scale: float = BALANCED_RETENTION_PENALTY_SCALE,
    retention_ramp_m: float = BALANCED_RETENTION_RAMP_M,
    survival_reward_per_interval: float = BALANCED_SURVIVAL_REWARD_PER_INTERVAL,
    control_effort_scale: float = BALANCED_CONTROL_EFFORT_PENALTY_SCALE,
    collective_hover_action: float = DEFAULT_COLLECTIVE_HOVER_ACTION,
    collective_effort_reference: float = BALANCED_COLLECTIVE_EFFORT_REFERENCE,
    moment_effort_reference: float = BALANCED_MOMENT_EFFORT_REFERENCE,
    control_effort_normalized_squared_cap: float = (
        BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP
    ),
    action_change_scale: float = BALANCED_ACTION_CHANGE_PENALTY_SCALE,
    collective_action_change_reference: float = (
        BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE
    ),
    moment_action_change_reference: float = BALANCED_MOMENT_ACTION_CHANGE_REFERENCE,
    action_change_normalized_squared_cap: float = (
        BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP
    ),
    boundary_scale: float = BALANCED_BOUNDARY_PENALTY_SCALE,
    boundary_low_onset_m: float = BALANCED_BOUNDARY_LOW_ONSET_M,
    boundary_low_width_m: float = BALANCED_BOUNDARY_LOW_WIDTH_M,
    boundary_high_onset_m: float = BALANCED_BOUNDARY_HIGH_ONSET_M,
    boundary_high_width_m: float = BALANCED_BOUNDARY_HIGH_WIDTH_M,
    boundary_xy_onset_m: float = BALANCED_BOUNDARY_XY_ONSET_M,
    boundary_xy_width_m: float = BALANCED_BOUNDARY_XY_WIDTH_M,
    boundary_normalized_squared_cap: float = (
        BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP
    ),
    failure_penalty: float = BALANCED_FAILURE_PENALTY,
    success_distance_m: float = SUCCESS_DISTANCE_M,
    success_speed_m_s: float = SUCCESS_SPEED_M_S,
    success_dwell_steps: int = SUCCESS_DWELL_STEPS,
    minimum_height_m: float = 0.10,
    maximum_height_m: float = 2.00,
    workspace_xy_limit_m: float = 2.75,
    version: str = BALANCED_REWARD_CONTRACT_VERSION,
) -> dict[str, object]:
    """Return and validate the additive balanced-task reward contract."""

    if not isinstance(version, str) or not version:
        raise ValueError("reward contract version must be a nonempty string")
    if (
        isinstance(success_dwell_steps, bool)
        or not isinstance(success_dwell_steps, int)
        or success_dwell_steps <= 0
    ):
        raise ValueError("success_dwell_steps must be a positive integer")
    values = {
        "progress_potential_scale_m": float(progress_potential_scale_m),
        "progress_scale": float(progress_scale),
        "proximity_scale": float(proximity_scale),
        "proximity_distance_scale_m": float(proximity_distance_scale_m),
        "dwell_reward_per_interval": float(dwell_reward_per_interval),
        "braking_scale": float(braking_scale),
        "braking_speed_reference_m_s": float(braking_speed_reference_m_s),
        "braking_normalized_squared_cap": float(braking_normalized_squared_cap),
        "success_bonus": float(success_bonus),
        "retention_scale": float(retention_scale),
        "retention_ramp_m": float(retention_ramp_m),
        "survival_reward_per_interval": float(survival_reward_per_interval),
        "control_effort_scale": float(control_effort_scale),
        "collective_hover_action": float(collective_hover_action),
        "collective_effort_reference": float(collective_effort_reference),
        "moment_effort_reference": float(moment_effort_reference),
        "control_effort_normalized_squared_cap": float(
            control_effort_normalized_squared_cap
        ),
        "action_change_scale": float(action_change_scale),
        "collective_action_change_reference": float(
            collective_action_change_reference
        ),
        "moment_action_change_reference": float(moment_action_change_reference),
        "action_change_normalized_squared_cap": float(
            action_change_normalized_squared_cap
        ),
        "boundary_scale": float(boundary_scale),
        "boundary_low_onset_m": float(boundary_low_onset_m),
        "boundary_low_width_m": float(boundary_low_width_m),
        "boundary_high_onset_m": float(boundary_high_onset_m),
        "boundary_high_width_m": float(boundary_high_width_m),
        "boundary_xy_onset_m": float(boundary_xy_onset_m),
        "boundary_xy_width_m": float(boundary_xy_width_m),
        "boundary_normalized_squared_cap": float(
            boundary_normalized_squared_cap
        ),
        "failure_penalty": float(failure_penalty),
        "success_distance_m": float(success_distance_m),
        "success_speed_m_s": float(success_speed_m_s),
        "minimum_height_m": float(minimum_height_m),
        "maximum_height_m": float(maximum_height_m),
        "workspace_xy_limit_m": float(workspace_xy_limit_m),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("balanced reward contract values must be finite")
    nonnegative = (
        "progress_scale",
        "proximity_scale",
        "dwell_reward_per_interval",
        "braking_scale",
        "success_bonus",
        "retention_scale",
        "survival_reward_per_interval",
        "control_effort_scale",
        "action_change_scale",
        "boundary_scale",
        "failure_penalty",
        "success_distance_m",
        "success_speed_m_s",
    )
    if any(values[name] < 0.0 for name in nonnegative):
        raise ValueError("balanced reward scales and success thresholds must be non-negative")
    positive = (
        "progress_potential_scale_m",
        "proximity_distance_scale_m",
        "braking_speed_reference_m_s",
        "braking_normalized_squared_cap",
        "retention_ramp_m",
        "collective_effort_reference",
        "moment_effort_reference",
        "control_effort_normalized_squared_cap",
        "collective_action_change_reference",
        "moment_action_change_reference",
        "action_change_normalized_squared_cap",
        "boundary_low_width_m",
        "boundary_high_width_m",
        "boundary_xy_width_m",
        "boundary_normalized_squared_cap",
        "workspace_xy_limit_m",
    )
    if any(values[name] <= 0.0 for name in positive):
        raise ValueError("balanced reward references, widths, and caps must be positive")
    if not -1.0 < values["collective_hover_action"] < 1.0:
        raise ValueError("collective hover action must lie strictly inside [-1, 1]")
    if values["minimum_height_m"] >= values["maximum_height_m"]:
        raise ValueError("height failure bounds are inconsistent")
    if not (
        values["minimum_height_m"]
        < values["boundary_low_onset_m"]
        <= values["boundary_high_onset_m"]
        < values["maximum_height_m"]
    ):
        raise ValueError("height boundary barrier must lie inside failure bounds")
    if not 0.0 < values["boundary_xy_onset_m"] < values["workspace_xy_limit_m"]:
        raise ValueError("horizontal boundary barrier must lie inside the workspace limit")
    return {
        "version": version,
        **values,
        "success_dwell_steps": success_dwell_steps,
        "progress_potential": "scale_m*tanh(distance_m/scale_m)",
        "boundary_xy_frame": "environment_local",
        "boundary_z_frame": "world",
        "positive_terms_zero_on_failure": True,
        "negative_progress_retained_on_failure": True,
        "success_terminates_episode": False,
    }


def balanced_reward_contract_sha256(**kwargs: object) -> str:
    """Hash the exact version and numeric balanced reward contract."""

    return _canonical_sha256(balanced_reward_contract_payload(**kwargs))


def balanced_training_curriculum_payload() -> dict[str, object]:
    """Return the separate balanced-v3 reset curriculum payload."""

    return training_curriculum_payload(
        BALANCED_TRAINING_CURRICULUM,
        version=BALANCED_TRAINING_CURRICULUM_VERSION,
    )


def balanced_training_curriculum_sha256() -> str:
    """Hash the exact balanced-v3 reset curriculum."""

    return training_curriculum_sha256(
        BALANCED_TRAINING_CURRICULUM,
        version=BALANCED_TRAINING_CURRICULUM_VERSION,
    )


BALANCED_REWARD_CONTRACT_SHA256 = balanced_reward_contract_sha256()
BALANCED_TRAINING_CURRICULUM_SHA256 = balanced_training_curriculum_sha256()


def balanced_task_contract_payload() -> dict[str, object]:
    """Return the additive balanced-v3 task authority for fingerprints."""

    return {
        "reward": balanced_reward_contract_payload(),
        "reward_sha256": BALANCED_REWARD_CONTRACT_SHA256,
        "training_curriculum": balanced_training_curriculum_payload(),
        "training_curriculum_sha256": BALANCED_TRAINING_CURRICULUM_SHA256,
        "switch_targets": balanced_switch_target_curriculum_payload(),
    }


def balanced_v4_reward_contract_payload(
    *,
    progress_potential_scale_m: float = BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M,
    progress_scale: float = BALANCED_V4_PROGRESS_REWARD_SCALE,
    proximity_scale: float = BALANCED_V4_PROXIMITY_REWARD_SCALE,
    proximity_distance_scale_m: float = BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M,
    gust_recovery_bonus: float = BALANCED_V4_GUST_RECOVERY_BONUS,
    gust_submitted_impulse_abs_tol_n_s: float = (
        GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S
    ),
    audited_robot_mass_kg: float = AUDITED_CRAZYFLIE_MASS_KG,
    audited_robot_mass_abs_tol_kg: float = CRAZYFLIE_MASS_ABS_TOL_KG,
    gust_delta_velocity_m_s: float = GUST_DELTA_V_M_S,
    version: str = BALANCED_V4_REWARD_CONTRACT_VERSION,
    **shared_overrides: object,
) -> dict[str, object]:
    """Return and validate the additive balanced-v4 reward contract.

    The shared terms are serialized through the v3 validator but use a new
    version and explicit v4 goal scales.  The returned payload then records
    every provenance rule required for the gust-recovery event bonus.
    """

    authentication_values = {
        "gust_recovery_bonus": float(gust_recovery_bonus),
        "gust_submitted_impulse_abs_tol_n_s": float(
            gust_submitted_impulse_abs_tol_n_s
        ),
        "audited_robot_mass_kg": float(audited_robot_mass_kg),
        "audited_robot_mass_abs_tol_kg": float(audited_robot_mass_abs_tol_kg),
        "gust_delta_velocity_m_s": float(gust_delta_velocity_m_s),
    }
    if any(not math.isfinite(value) for value in authentication_values.values()):
        raise ValueError("balanced-v4 gust authentication values must be finite")
    if authentication_values["gust_recovery_bonus"] < 0.0:
        raise ValueError("balanced-v4 gust recovery bonus must be non-negative")
    if (
        authentication_values["gust_submitted_impulse_abs_tol_n_s"] <= 0.0
        or authentication_values["audited_robot_mass_kg"] <= 0.0
        or authentication_values["audited_robot_mass_abs_tol_kg"] <= 0.0
        or authentication_values["gust_delta_velocity_m_s"] <= 0.0
    ):
        raise ValueError("balanced-v4 gust authentication tolerances and mass must be positive")
    shared = balanced_reward_contract_payload(
        progress_potential_scale_m=progress_potential_scale_m,
        progress_scale=progress_scale,
        proximity_scale=proximity_scale,
        proximity_distance_scale_m=proximity_distance_scale_m,
        version=version,
        **shared_overrides,
    )
    return {
        **shared,
        **authentication_values,
        "gust_recovery_bonus_candidate": (
            "newly_completed_post_gust_success_dwell"
        ),
        "gust_recovery_bonus_scenario": "gust_recovery",
        "gust_expected_impulse": (
            "predeclared_world_xy_unit_direction*audited_robot_mass_kg*"
            "gust_delta_velocity_m_s"
        ),
        "gust_recovery_bonus_requires_authenticated_submitted_impulse": True,
        "gust_recovery_bonus_one_shot_per_scheduled_gust": True,
        "gust_recovery_bonus_zero_outside_scenario": True,
        "gust_recovery_bonus_zero_on_failure": True,
    }


def balanced_v4_reward_contract_sha256(**kwargs: object) -> str:
    """Hash the exact version and numeric balanced-v4 reward contract."""

    return _canonical_sha256(balanced_v4_reward_contract_payload(**kwargs))


def balanced_v4_training_curriculum_payload() -> dict[str, object]:
    """Return the accelerated balanced-v4 reset curriculum payload."""

    return training_curriculum_payload(
        BALANCED_V4_TRAINING_CURRICULUM,
        version=BALANCED_V4_TRAINING_CURRICULUM_VERSION,
    )


def balanced_v4_training_curriculum_sha256() -> str:
    """Hash the exact balanced-v4 reset curriculum."""

    return training_curriculum_sha256(
        BALANCED_V4_TRAINING_CURRICULUM,
        version=BALANCED_V4_TRAINING_CURRICULUM_VERSION,
    )


BALANCED_V4_REWARD_CONTRACT_SHA256 = balanced_v4_reward_contract_sha256()
BALANCED_V4_TRAINING_CURRICULUM_SHA256 = (
    balanced_v4_training_curriculum_sha256()
)


def balanced_v4_task_contract_payload() -> dict[str, object]:
    """Return the additive balanced-v4 task authority for fingerprints."""

    return {
        "reward": balanced_v4_reward_contract_payload(),
        "reward_sha256": BALANCED_V4_REWARD_CONTRACT_SHA256,
        "training_curriculum": balanced_v4_training_curriculum_payload(),
        "training_curriculum_sha256": BALANCED_V4_TRAINING_CURRICULUM_SHA256,
        "switch_targets": balanced_v4_switch_target_curriculum_payload(),
    }


@dataclass(frozen=True)
class FailureClassification:
    """Pure, vectorized failure classification used by the live environment."""

    terminated: torch.Tensor
    cause: torch.Tensor
    low_height: torch.Tensor
    high_height: torch.Tensor
    workspace_escape: torch.Tensor
    nonfinite: torch.Tensor


def classify_failure_causes(
    root_position_w: torch.Tensor,
    environment_origins_w: torch.Tensor,
    finite_state: torch.Tensor,
    *,
    minimum_height_m: float,
    maximum_height_m: float,
    workspace_xy_limit_m: float,
) -> FailureClassification:
    """Classify terminal failures without requiring or mutating a simulator.

    Keeping this decision pure lets Gate C exercise the nonfinite branch
    without ever injecting NaN/Inf into PhysX.  The live environment calls the
    same helper, so the probe cannot drift from execution semantics.
    """

    position = torch.as_tensor(root_position_w)
    origins = torch.as_tensor(
        environment_origins_w, device=position.device, dtype=position.dtype
    )
    finite = torch.as_tensor(finite_state, device=position.device, dtype=torch.bool)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("root_position_w must have shape [num_envs, 3]")
    if origins.shape != position.shape:
        raise ValueError("environment_origins_w must match root_position_w")
    if finite.shape != (position.shape[0],):
        raise ValueError("finite_state must have one value per environment")
    bounds = (minimum_height_m, maximum_height_m, workspace_xy_limit_m)
    if any(not math.isfinite(float(value)) for value in bounds):
        raise ValueError("failure bounds must be finite")
    if minimum_height_m >= maximum_height_m or workspace_xy_limit_m <= 0.0:
        raise ValueError("failure bounds are inconsistent")

    # A nonfinite position is a nonfinite failure even if a caller supplied an
    # erroneously optimistic finite_state mask.
    finite = finite & torch.isfinite(position).all(dim=-1) & torch.isfinite(origins).all(dim=-1)
    height = position[:, 2]
    local_xy = position[:, :2] - origins[:, :2]
    low = finite & (height < float(minimum_height_m))
    high = finite & (height > float(maximum_height_m))
    workspace = finite & torch.any(torch.abs(local_xy) > float(workspace_xy_limit_m), dim=-1)
    nonfinite = ~finite
    cause = torch.full_like(finite, FAILURE_NONE, dtype=torch.long)
    cause[low] = FAILURE_LOW_HEIGHT
    cause[high] = FAILURE_HIGH_HEIGHT
    cause[workspace] = FAILURE_WORKSPACE_ESCAPE
    cause[nonfinite] = FAILURE_NONFINITE
    return FailureClassification(
        terminated=low | high | workspace | nonfinite,
        cause=cause,
        low_height=low,
        high_height=high,
        workspace_escape=workspace,
        nonfinite=nonfinite,
    )


def interval_reward_terms(
    distance_progress_m: torch.Tensor,
    newly_successful: torch.Tensor,
    actions: torch.Tensor,
    action_delta: torch.Tensor,
    failed: torch.Tensor,
    *,
    progress_scale: float,
    progress_clip_m: float,
    success_bonus: float,
    survival_reward_per_interval: float,
    control_effort_scale: float,
    collective_hover_action: float,
    moment_action_reference: float,
    control_effort_normalized_squared_cap: float,
    action_change_scale: float,
    action_change_normalized_squared_cap: float,
    failure_penalty: float,
) -> dict[str, torch.Tensor]:
    """Compute the frozen per-control-interval survival-first reward."""

    if (
        actions.ndim != 2
        or actions.shape[1] != 4
        or action_delta.shape != actions.shape
    ):
        raise ValueError(
            "actions and action_delta must have matching [batch, 4] shapes"
        )
    batch = actions.shape[0]
    for name, value in {
        "distance_progress_m": distance_progress_m,
        "newly_successful": newly_successful,
        "failed": failed,
    }.items():
        if value.shape != (batch,):
            raise ValueError(f"{name} must have one value per action row")
    safe_action = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    safe_delta = torch.nan_to_num(action_delta, nan=0.0, posinf=2.0, neginf=-2.0)
    safe_progress = torch.nan_to_num(
        distance_progress_m,
        nan=0.0,
        posinf=float(progress_clip_m),
        neginf=-float(progress_clip_m),
    ).clamp(-float(progress_clip_m), float(progress_clip_m))
    failed_bool = failed.to(dtype=torch.bool)
    valid = (~failed_bool).to(safe_progress.dtype)
    progress = safe_progress * float(progress_scale)
    success = newly_successful.to(safe_progress.dtype) * float(success_bonus)
    survival = valid * float(survival_reward_per_interval)
    effort_normalized_squared = (
        (safe_action[:, 0] - float(collective_hover_action)).square()
        + torch.sum(
            (safe_action[:, 1:] / float(moment_action_reference)).square(), dim=-1
        )
    ).clamp(max=float(control_effort_normalized_squared_cap))
    effort = -float(control_effort_scale) * effort_normalized_squared
    action_change_normalized_squared = (
        safe_delta[:, 0].square()
        + torch.sum(
            (safe_delta[:, 1:] / float(moment_action_reference)).square(),
            dim=-1,
        )
    ).clamp(max=float(action_change_normalized_squared_cap))
    change = -float(action_change_scale) * action_change_normalized_squared
    failure = -float(failure_penalty) * failed_bool.to(safe_progress.dtype)
    total = torch.nan_to_num(
        progress
        + success
        + survival
        + effort
        + change
        + failure,
        nan=-float(failure_penalty),
        posinf=float(success_bonus),
        neginf=-float(failure_penalty),
    )
    return {
        "progress": progress,
        "success_bonus": success,
        "survival": survival,
        "control_effort": effort,
        "action_change": change,
        "failure": failure,
        "total": total,
    }


def balanced_boundary_metric(
    root_position_w: torch.Tensor,
    environment_origins_w: torch.Tensor,
    *,
    low_onset_m: float = BALANCED_BOUNDARY_LOW_ONSET_M,
    low_width_m: float = BALANCED_BOUNDARY_LOW_WIDTH_M,
    high_onset_m: float = BALANCED_BOUNDARY_HIGH_ONSET_M,
    high_width_m: float = BALANCED_BOUNDARY_HIGH_WIDTH_M,
    xy_onset_m: float = BALANCED_BOUNDARY_XY_ONSET_M,
    xy_width_m: float = BALANCED_BOUNDARY_XY_WIDTH_M,
    normalized_squared_cap: float = BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP,
) -> torch.Tensor:
    """Return the capped balanced-v3 soft barrier metric.

    Horizontal coordinates are relative to each environment origin.  Height
    deliberately follows the absolute-world convention used by the installed
    task's hard termination.  Nonfinite rows receive the maximum metric.
    """

    position = torch.as_tensor(root_position_w)
    origins = torch.as_tensor(
        environment_origins_w, device=position.device, dtype=position.dtype
    )
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("root_position_w must have shape [batch, 3]")
    if origins.shape != position.shape:
        raise ValueError("environment_origins_w must match root_position_w")
    values = (
        low_onset_m,
        low_width_m,
        high_onset_m,
        high_width_m,
        xy_onset_m,
        xy_width_m,
        normalized_squared_cap,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("boundary values must be finite")
    if low_width_m <= 0.0 or high_width_m <= 0.0 or xy_width_m <= 0.0:
        raise ValueError("boundary widths must be positive")
    if normalized_squared_cap <= 0.0 or low_onset_m > high_onset_m or xy_onset_m <= 0.0:
        raise ValueError("boundary onsets and cap are inconsistent")

    finite = torch.isfinite(position).all(dim=-1) & torch.isfinite(origins).all(dim=-1)
    safe_position = torch.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    safe_origins = torch.nan_to_num(origins, nan=0.0, posinf=0.0, neginf=0.0)
    local_xy = safe_position[:, :2] - safe_origins[:, :2]
    height = safe_position[:, 2]
    metric = (
        torch.relu((float(low_onset_m) - height) / float(low_width_m)).square()
        + torch.relu((height - float(high_onset_m)) / float(high_width_m)).square()
        + torch.sum(
            torch.relu(
                (torch.abs(local_xy) - float(xy_onset_m)) / float(xy_width_m)
            ).square(),
            dim=-1,
        )
    ).clamp(max=float(normalized_squared_cap))
    return torch.where(
        finite,
        metric,
        torch.full_like(metric, float(normalized_squared_cap)),
    )


def balanced_interval_reward_terms(
    previous_distance_m: torch.Tensor,
    current_distance_m: torch.Tensor,
    speed_m_s: torch.Tensor,
    newly_successful: torch.Tensor,
    success_latched: torch.Tensor,
    actions: torch.Tensor,
    action_delta: torch.Tensor,
    failed: torch.Tensor,
    root_position_w: torch.Tensor,
    environment_origins_w: torch.Tensor,
    *,
    progress_potential_scale_m: float = BALANCED_PROGRESS_POTENTIAL_SCALE_M,
    progress_scale: float = BALANCED_PROGRESS_REWARD_SCALE,
    proximity_scale: float = BALANCED_PROXIMITY_REWARD_SCALE,
    proximity_distance_scale_m: float = BALANCED_PROXIMITY_DISTANCE_SCALE_M,
    dwell_reward_per_interval: float = BALANCED_DWELL_REWARD_PER_INTERVAL,
    braking_scale: float = BALANCED_BRAKING_PENALTY_SCALE,
    braking_speed_reference_m_s: float = BALANCED_BRAKING_SPEED_REFERENCE_M_S,
    braking_normalized_squared_cap: float = BALANCED_BRAKING_NORMALIZED_SQUARED_CAP,
    success_bonus: float = BALANCED_SUCCESS_BONUS,
    retention_scale: float = BALANCED_RETENTION_PENALTY_SCALE,
    retention_ramp_m: float = BALANCED_RETENTION_RAMP_M,
    survival_reward_per_interval: float = BALANCED_SURVIVAL_REWARD_PER_INTERVAL,
    control_effort_scale: float = BALANCED_CONTROL_EFFORT_PENALTY_SCALE,
    collective_hover_action: float = DEFAULT_COLLECTIVE_HOVER_ACTION,
    collective_effort_reference: float = BALANCED_COLLECTIVE_EFFORT_REFERENCE,
    moment_effort_reference: float = BALANCED_MOMENT_EFFORT_REFERENCE,
    control_effort_normalized_squared_cap: float = (
        BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP
    ),
    action_change_scale: float = BALANCED_ACTION_CHANGE_PENALTY_SCALE,
    collective_action_change_reference: float = (
        BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE
    ),
    moment_action_change_reference: float = BALANCED_MOMENT_ACTION_CHANGE_REFERENCE,
    action_change_normalized_squared_cap: float = (
        BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP
    ),
    boundary_scale: float = BALANCED_BOUNDARY_PENALTY_SCALE,
    boundary_low_onset_m: float = BALANCED_BOUNDARY_LOW_ONSET_M,
    boundary_low_width_m: float = BALANCED_BOUNDARY_LOW_WIDTH_M,
    boundary_high_onset_m: float = BALANCED_BOUNDARY_HIGH_ONSET_M,
    boundary_high_width_m: float = BALANCED_BOUNDARY_HIGH_WIDTH_M,
    boundary_xy_onset_m: float = BALANCED_BOUNDARY_XY_ONSET_M,
    boundary_xy_width_m: float = BALANCED_BOUNDARY_XY_WIDTH_M,
    boundary_normalized_squared_cap: float = (
        BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP
    ),
    failure_penalty: float = BALANCED_FAILURE_PENALTY,
    success_distance_m: float = SUCCESS_DISTANCE_M,
    success_speed_m_s: float = SUCCESS_SPEED_M_S,
) -> dict[str, torch.Tensor]:
    """Compute the additive balanced-v3 reward from policy-visible task state."""

    if actions.ndim != 2 or actions.shape[1] != 4 or action_delta.shape != actions.shape:
        raise ValueError("actions and action_delta must have matching [batch, 4] shapes")
    batch = actions.shape[0]
    vectors = {
        "previous_distance_m": previous_distance_m,
        "current_distance_m": current_distance_m,
        "speed_m_s": speed_m_s,
        "newly_successful": newly_successful,
        "success_latched": success_latched,
        "failed": failed,
    }
    if any(value.shape != (batch,) for value in vectors.values()):
        raise ValueError("balanced reward vectors must have one value per action row")
    if root_position_w.shape != (batch, 3) or environment_origins_w.shape != (batch, 3):
        raise ValueError("positions and environment origins must have shape [batch, 3]")

    safe_previous = torch.where(
        torch.isfinite(previous_distance_m) & (previous_distance_m >= 0.0),
        previous_distance_m,
        torch.full_like(previous_distance_m, float("inf")),
    )
    safe_current = torch.where(
        torch.isfinite(current_distance_m) & (current_distance_m >= 0.0),
        current_distance_m,
        torch.full_like(current_distance_m, float("inf")),
    )
    safe_speed = torch.where(
        torch.isfinite(speed_m_s) & (speed_m_s >= 0.0),
        speed_m_s,
        torch.full_like(speed_m_s, float("inf")),
    )
    failed_bool = failed.to(dtype=torch.bool)
    valid = (~failed_bool).to(dtype=actions.dtype)

    potential_scale = float(progress_potential_scale_m)
    previous_potential = potential_scale * torch.tanh(safe_previous / potential_scale)
    current_potential = potential_scale * torch.tanh(safe_current / potential_scale)
    progress_raw = float(progress_scale) * (previous_potential - current_potential)
    progress = torch.where(
        failed_bool,
        torch.minimum(progress_raw, torch.zeros_like(progress_raw)),
        progress_raw,
    )

    q = torch.exp(
        -0.5 * (safe_current / float(proximity_distance_scale_m)).square()
    )
    proximity = valid * float(proximity_scale) * q
    in_tube = (
        (safe_current <= float(success_distance_m))
        & (safe_speed <= float(success_speed_m_s))
    )
    dwell = valid * in_tube.to(actions.dtype) * float(dwell_reward_per_interval)
    braking_metric = (safe_speed / float(braking_speed_reference_m_s)).square().clamp(
        max=float(braking_normalized_squared_cap)
    )
    braking = -float(braking_scale) * q * braking_metric
    success = (
        valid
        * newly_successful.to(dtype=actions.dtype)
        * float(success_bonus)
    )
    retention_metric = ((safe_current - float(success_distance_m)) / float(retention_ramp_m)).clamp(
        min=0.0, max=1.0
    )
    retention = (
        -float(retention_scale)
        * success_latched.to(dtype=actions.dtype)
        * retention_metric
    )
    survival = valid * float(survival_reward_per_interval)

    safe_action = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    safe_delta = torch.nan_to_num(action_delta, nan=0.0, posinf=2.0, neginf=-2.0)
    effort_metric = (
        ((safe_action[:, 0] - float(collective_hover_action)) / float(
            collective_effort_reference
        )).square()
        + torch.sum(
            (safe_action[:, 1:] / float(moment_effort_reference)).square(), dim=-1
        )
    ).clamp(max=float(control_effort_normalized_squared_cap))
    effort = -float(control_effort_scale) * effort_metric
    change_metric = (
        (safe_delta[:, 0] / float(collective_action_change_reference)).square()
        + torch.sum(
            (safe_delta[:, 1:] / float(moment_action_change_reference)).square(),
            dim=-1,
        )
    ).clamp(max=float(action_change_normalized_squared_cap))
    change = -float(action_change_scale) * change_metric

    boundary_metric = balanced_boundary_metric(
        root_position_w,
        environment_origins_w,
        low_onset_m=boundary_low_onset_m,
        low_width_m=boundary_low_width_m,
        high_onset_m=boundary_high_onset_m,
        high_width_m=boundary_high_width_m,
        xy_onset_m=boundary_xy_onset_m,
        xy_width_m=boundary_xy_width_m,
        normalized_squared_cap=boundary_normalized_squared_cap,
    )
    boundary = -float(boundary_scale) * boundary_metric
    failure = -float(failure_penalty) * failed_bool.to(dtype=actions.dtype)
    total = torch.nan_to_num(
        progress
        + proximity
        + dwell
        + braking
        + success
        + retention
        + survival
        + effort
        + change
        + boundary
        + failure,
        nan=-float(failure_penalty),
        posinf=float(success_bonus),
        neginf=-float(failure_penalty),
    )
    return {
        "progress": progress,
        "proximity": proximity,
        "dwell": dwell,
        "braking": braking,
        "success_bonus": success,
        "retention": retention,
        "survival": survival,
        "control_effort": effort,
        "action_change": change,
        "boundary": boundary,
        "failure": failure,
        "total": total,
    }


def authenticated_gust_recovery_mask(
    newly_recovered: torch.Tensor,
    current_event_index: torch.Tensor,
    submitted_impulses_w_n_s: torch.Tensor,
    expected_impulses_w_n_s: torch.Tensor,
    *,
    max_abs_error_n_s: float = GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
) -> torch.Tensor:
    """Authenticate one-shot recovery candidates against submitted impulses.

    This helper deliberately authenticates the force-time integral sent to
    simulation.  It does not use measured momentum response, which is a noisy
    evaluation metric rather than provenance for the reward event.
    """

    candidates = torch.as_tensor(newly_recovered)
    indices = torch.as_tensor(current_event_index, device=candidates.device)
    submitted = torch.as_tensor(
        submitted_impulses_w_n_s, device=candidates.device
    )
    expected = torch.as_tensor(
        expected_impulses_w_n_s,
        device=candidates.device,
        dtype=submitted.dtype,
    )
    if candidates.ndim != 1 or candidates.dtype != torch.bool:
        raise TypeError("newly_recovered must be a one-dimensional Boolean tensor")
    if indices.shape != candidates.shape:
        raise ValueError("current_event_index must have one value per recovery candidate")
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise TypeError("current_event_index must use an integer dtype")
    if (
        submitted.ndim != 3
        or submitted.shape[0] != candidates.shape[0]
        or submitted.shape[2] != 3
        or submitted.shape[1] < 1
        or expected.shape != submitted.shape
    ):
        raise ValueError(
            "submitted and expected impulses must share shape [batch, events, 3]"
        )
    if submitted.dtype == torch.bool or not submitted.is_floating_point():
        raise TypeError("gust impulse tensors must use a floating-point dtype")
    if (
        isinstance(max_abs_error_n_s, bool)
        or not math.isfinite(float(max_abs_error_n_s))
        or float(max_abs_error_n_s) <= 0.0
    ):
        raise ValueError("max_abs_error_n_s must be finite and positive")

    event_count = submitted.shape[1]
    valid_index = (indices >= 0) & (indices < event_count)
    safe_index = indices.to(torch.long).clamp(min=0, max=event_count - 1)
    rows = torch.arange(candidates.shape[0], device=candidates.device)
    selected_submitted = submitted[rows, safe_index]
    selected_expected = expected[rows, safe_index]
    finite = (
        torch.isfinite(selected_submitted).all(dim=-1)
        & torch.isfinite(selected_expected).all(dim=-1)
    )
    nonzero = (
        torch.linalg.vector_norm(selected_submitted, dim=-1) > 0.0
    ) & (torch.linalg.vector_norm(selected_expected, dim=-1) > 0.0)
    max_error = torch.amax(
        torch.abs(selected_submitted - selected_expected), dim=-1
    )
    return (
        candidates
        & valid_index
        & finite
        & nonzero
        & (max_error <= float(max_abs_error_n_s))
    )


def balanced_v4_interval_reward_terms(
    previous_distance_m: torch.Tensor,
    current_distance_m: torch.Tensor,
    speed_m_s: torch.Tensor,
    newly_successful: torch.Tensor,
    success_latched: torch.Tensor,
    authenticated_gust_recovery: torch.Tensor,
    actions: torch.Tensor,
    action_delta: torch.Tensor,
    failed: torch.Tensor,
    root_position_w: torch.Tensor,
    environment_origins_w: torch.Tensor,
    *,
    progress_potential_scale_m: float = BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M,
    progress_scale: float = BALANCED_V4_PROGRESS_REWARD_SCALE,
    proximity_scale: float = BALANCED_V4_PROXIMITY_REWARD_SCALE,
    proximity_distance_scale_m: float = BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M,
    gust_recovery_bonus: float = BALANCED_V4_GUST_RECOVERY_BONUS,
    **shared_reward_kwargs: float,
) -> dict[str, torch.Tensor]:
    """Compute balanced-v4 terms and its authenticated Gust-only event."""

    if (
        authenticated_gust_recovery.shape != (actions.shape[0],)
        or authenticated_gust_recovery.dtype != torch.bool
    ):
        raise TypeError(
            "authenticated_gust_recovery must be a batch-sized Boolean tensor"
        )
    if (
        isinstance(gust_recovery_bonus, bool)
        or not math.isfinite(float(gust_recovery_bonus))
        or float(gust_recovery_bonus) < 0.0
    ):
        raise ValueError("gust_recovery_bonus must be finite and non-negative")
    terms = balanced_interval_reward_terms(
        previous_distance_m,
        current_distance_m,
        speed_m_s,
        newly_successful,
        success_latched,
        actions,
        action_delta,
        failed,
        root_position_w,
        environment_origins_w,
        progress_potential_scale_m=progress_potential_scale_m,
        progress_scale=progress_scale,
        proximity_scale=proximity_scale,
        proximity_distance_scale_m=proximity_distance_scale_m,
        **shared_reward_kwargs,
    )
    failed_bool = failed.to(dtype=torch.bool)
    recovery = (
        (~failed_bool).to(dtype=actions.dtype)
        * authenticated_gust_recovery.to(dtype=actions.dtype)
        * float(gust_recovery_bonus)
    )
    failure_penalty = float(
        shared_reward_kwargs.get("failure_penalty", BALANCED_FAILURE_PENALTY)
    )
    total = torch.nan_to_num(
        terms["total"] + recovery,
        nan=-failure_penalty,
        posinf=float(shared_reward_kwargs.get("success_bonus", BALANCED_SUCCESS_BONUS))
        + float(gust_recovery_bonus),
        neginf=-failure_penalty,
    )
    return {
        **terms,
        "gust_recovery_bonus": recovery,
        "total": total,
    }


def seconds_to_control_steps(seconds: float, *, control_hz: int = CONTROL_FREQUENCY_HZ) -> int:
    """Convert an exactly representable control-grid time to a step count.

    Silently rounding an off-grid schedule would change the frozen protocol,
    so a non-grid-aligned value is rejected.
    """

    if isinstance(seconds, bool) or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        raise ValueError("seconds must be a finite, non-negative number")
    if isinstance(control_hz, bool) or int(control_hz) != control_hz or control_hz <= 0:
        raise ValueError("control_hz must be a positive integer")
    exact = float(seconds) * int(control_hz)
    rounded = round(exact)
    if not math.isclose(exact, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{seconds!r} s is not aligned to the {control_hz} Hz control grid")
    return int(rounded)


def control_steps_to_seconds(steps: int, *, control_hz: int = CONTROL_FREQUENCY_HZ) -> float:
    """Convert a non-negative integral control-step count to seconds."""

    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if isinstance(control_hz, bool) or int(control_hz) != control_hz or control_hz <= 0:
        raise ValueError("control_hz must be a positive integer")
    return int(steps) / int(control_hz)


def schedule_steps(
    times_s: Sequence[float], *, control_hz: int = CONTROL_FREQUENCY_HZ
) -> tuple[int, ...]:
    """Return a strictly increasing schedule expressed in control steps."""

    result = tuple(seconds_to_control_steps(value, control_hz=control_hz) for value in times_s)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("schedule times must be strictly increasing")
    return result


def _validate_schedule(starts: Sequence[int], duration_steps: int) -> tuple[int, ...]:
    if isinstance(duration_steps, bool) or int(duration_steps) != duration_steps or duration_steps <= 0:
        raise ValueError("duration_steps must be a positive integer")
    normalized = tuple(int(start) for start in starts)
    if any(isinstance(start, bool) or int(start) != start or start < 0 for start in starts):
        raise ValueError("event starts must be non-negative integers")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("event starts must be strictly increasing")
    if any(left + int(duration_steps) > right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("event intervals must not overlap")
    return normalized


def schedule_active_mask(
    step: int | torch.Tensor,
    starts: Sequence[int],
    duration_steps: int,
) -> bool | torch.Tensor:
    """Return whether any half-open scheduled interval is active.

    Tensor input produces a Boolean tensor of the same shape and may live on
    either CPU or CUDA.  Scalar input produces a Python ``bool``.
    """

    normalized = _validate_schedule(starts, duration_steps)
    if isinstance(step, torch.Tensor):
        if step.dtype == torch.bool or step.is_floating_point() or step.is_complex():
            raise TypeError("step tensors must have an integer dtype")
        active = torch.zeros_like(step, dtype=torch.bool)
        for start in normalized:
            active |= (step >= start) & (step < start + int(duration_steps))
        return active
    if isinstance(step, bool) or int(step) != step:
        raise TypeError("step must be an integer or integer tensor")
    step_int = int(step)
    return any(start <= step_int < start + int(duration_steps) for start in normalized)


def scheduled_event_index(
    step: int | torch.Tensor,
    starts: Sequence[int],
    duration_steps: int = 1,
) -> int | torch.Tensor:
    """Return the active event index, or ``-1`` when no event is active."""

    normalized = _validate_schedule(starts, duration_steps)
    if isinstance(step, torch.Tensor):
        if step.dtype == torch.bool or step.is_floating_point() or step.is_complex():
            raise TypeError("step tensors must have an integer dtype")
        result = torch.full_like(step, -1, dtype=torch.long)
        for index, start in enumerate(normalized):
            mask = (step >= start) & (step < start + int(duration_steps))
            result = torch.where(mask, index, result)
        return result
    if isinstance(step, bool) or int(step) != step:
        raise TypeError("step must be an integer or integer tensor")
    for index, start in enumerate(normalized):
        if start <= int(step) < start + int(duration_steps):
            return index
    return -1


@dataclass(frozen=True)
class RecoveryWindow:
    """Half-open post-gust window in zero-based control-step coordinates."""

    gust_start_step: int
    gust_end_step: int
    start_step: int
    stop_step: int

    @property
    def duration_steps(self) -> int:
        return self.stop_step - self.start_step

    def contains(self, step: int) -> bool:
        return self.start_step <= int(step) < self.stop_step


def recovery_window(
    gust_start_step: int,
    *,
    gust_duration_steps: int = GUST_DURATION_STEPS,
    recovery_window_steps: int = RECOVERY_WINDOW_STEPS,
) -> RecoveryWindow:
    """Build the recovery interval that begins immediately after a gust."""

    if isinstance(gust_start_step, bool) or int(gust_start_step) != gust_start_step or gust_start_step < 0:
        raise ValueError("gust_start_step must be a non-negative integer")
    if (
        isinstance(gust_duration_steps, bool)
        or int(gust_duration_steps) != gust_duration_steps
        or gust_duration_steps <= 0
    ):
        raise ValueError("gust_duration_steps must be a positive integer")
    if (
        isinstance(recovery_window_steps, bool)
        or int(recovery_window_steps) != recovery_window_steps
        or recovery_window_steps <= 0
    ):
        raise ValueError("recovery_window_steps must be a positive integer")
    gust_end = int(gust_start_step) + int(gust_duration_steps)
    return RecoveryWindow(
        gust_start_step=int(gust_start_step),
        gust_end_step=gust_end,
        start_step=gust_end,
        stop_step=gust_end + int(recovery_window_steps),
    )


def gust_impulse_n_s(mass_kg: float, *, delta_v_m_s: float = GUST_DELTA_V_M_S) -> float:
    """Desired horizontal impulse, ``mass * delta-v``, in N s."""

    mass = float(mass_kg)
    delta_v = float(delta_v_m_s)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if not math.isfinite(delta_v) or delta_v <= 0.0:
        raise ValueError("delta_v_m_s must be finite and positive")
    return mass * delta_v


def gust_force_n(
    mass_kg: float,
    *,
    delta_v_m_s: float = GUST_DELTA_V_M_S,
    duration_s: float = GUST_DURATION_S,
) -> float:
    """Constant force magnitude implementing the requested gust impulse.

    ``force = (mass * delta-v) / duration``
    """

    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    return gust_impulse_n_s(mass_kg, delta_v_m_s=delta_v_m_s) / duration


def horizontal_gust_force_vector_n(
    mass_kg: float,
    direction: Sequence[float],
    *,
    delta_v_m_s: float = GUST_DELTA_V_M_S,
    duration_s: float = GUST_DURATION_S,
) -> tuple[float, float, float]:
    """Return a world-frame horizontal force vector applied at the body COM.

    A two-vector is interpreted as world X/Y.  A three-vector is accepted only
    when its Z component is zero (within numerical tolerance).  Direction is
    normalized so the stored direction cannot accidentally alter gust strength.
    """

    values = tuple(float(value) for value in direction)
    if len(values) == 2:
        x, y = values
    elif len(values) == 3:
        x, y, z = values
        if not math.isfinite(z) or not math.isclose(z, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("gust direction must be horizontal (world-frame z must be zero)")
    else:
        raise ValueError("gust direction must have two or three components")
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("gust direction must be finite")
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError("gust direction must be non-zero")
    magnitude = gust_force_n(mass_kg, delta_v_m_s=delta_v_m_s, duration_s=duration_s)
    return magnitude * x / norm, magnitude * y / norm, 0.0


def constant_force_impulse_n_s(
    force_n: float, applied_steps: int, *, control_dt_s: float = CONTROL_DT_S
) -> float:
    """Measured scalar impulse for a constant force over control intervals."""

    force = float(force_n)
    dt = float(control_dt_s)
    if not math.isfinite(force):
        raise ValueError("force_n must be finite")
    if isinstance(applied_steps, bool) or int(applied_steps) != applied_steps or applied_steps < 0:
        raise ValueError("applied_steps must be a non-negative integer")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    return force * int(applied_steps) * dt


def success_mask(
    distance_m: float | torch.Tensor,
    speed_m_s: float | torch.Tensor,
    *,
    distance_threshold_m: float = SUCCESS_DISTANCE_M,
    speed_threshold_m_s: float = SUCCESS_SPEED_M_S,
) -> bool | torch.Tensor:
    """Whether state is finite and inside the inclusive success tube."""

    if distance_threshold_m < 0.0 or speed_threshold_m_s < 0.0:
        raise ValueError("success thresholds must be non-negative")
    if isinstance(distance_m, torch.Tensor) or isinstance(speed_m_s, torch.Tensor):
        if isinstance(distance_m, torch.Tensor):
            distance = distance_m
            speed = torch.as_tensor(speed_m_s, device=distance.device, dtype=distance.dtype)
        else:
            speed = speed_m_s
            assert isinstance(speed, torch.Tensor)
            distance = torch.as_tensor(distance_m, device=speed.device, dtype=speed.dtype)
        distance, speed = torch.broadcast_tensors(distance, speed)
        return (
            torch.isfinite(distance)
            & torch.isfinite(speed)
            & (distance >= 0.0)
            & (speed >= 0.0)
            & (distance <= float(distance_threshold_m))
            & (speed <= float(speed_threshold_m_s))
        )
    distance_value = float(distance_m)
    speed_value = float(speed_m_s)
    return (
        math.isfinite(distance_value)
        and math.isfinite(speed_value)
        and 0.0 <= distance_value <= float(distance_threshold_m)
        and 0.0 <= speed_value <= float(speed_threshold_m_s)
    )


@dataclass(frozen=True)
class SuccessDwellUpdate:
    dwell_steps: torch.Tensor
    success_latched: torch.Tensor
    in_success_tube: torch.Tensor
    newly_successful: torch.Tensor


def advance_success_dwell(
    dwell_steps: torch.Tensor,
    success_latched: torch.Tensor,
    distance_m: torch.Tensor,
    speed_m_s: torch.Tensor,
    *,
    required_steps: int = SUCCESS_DWELL_STEPS,
) -> SuccessDwellUpdate:
    """Advance a continuous success dwell and a one-shot success latch.

    Leaving the tube resets the dwell to zero.  The latch remains set until a
    target reset, preventing repeated success bonuses for one target.
    """

    if isinstance(required_steps, bool) or int(required_steps) != required_steps or required_steps <= 0:
        raise ValueError("required_steps must be a positive integer")
    if dwell_steps.dtype == torch.bool or dwell_steps.is_floating_point() or dwell_steps.is_complex():
        raise TypeError("dwell_steps must have an integer dtype")
    if success_latched.dtype != torch.bool:
        raise TypeError("success_latched must have Boolean dtype")
    if dwell_steps.shape != success_latched.shape:
        raise ValueError("dwell_steps and success_latched must have matching shapes")
    inside = success_mask(distance_m, speed_m_s)
    assert isinstance(inside, torch.Tensor)
    if inside.shape != dwell_steps.shape:
        raise ValueError("distance, speed, and dwell state must have matching broadcast shape")
    updated_dwell = torch.where(
        inside,
        torch.clamp(dwell_steps + 1, max=int(required_steps)),
        torch.zeros_like(dwell_steps),
    )
    newly_successful = (updated_dwell >= int(required_steps)) & ~success_latched
    updated_latch = success_latched | newly_successful
    return SuccessDwellUpdate(updated_dwell, updated_latch, inside, newly_successful)


# Short alias for callers that name the operation after its state variable.
update_success_dwell = advance_success_dwell


@dataclass(frozen=True)
class TargetTrackingUpdate:
    distance_m: torch.Tensor
    progress_m: torch.Tensor
    in_success_tube: torch.Tensor
    newly_successful: torch.Tensor
    success_latched: torch.Tensor
    dwell_steps: torch.Tensor
    target_age_steps: torch.Tensor


@dataclass
class TargetTrackingState:
    """Vectorized target-specific progress, dwell, and timer state."""

    previous_distance_m: torch.Tensor
    dwell_steps: torch.Tensor
    success_latched: torch.Tensor
    target_age_steps: torch.Tensor
    first_success_after_steps: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device | str = "cpu") -> "TargetTrackingState":
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        return cls(
            previous_distance_m=torch.zeros(int(num_envs), device=device),
            dwell_steps=torch.zeros(int(num_envs), dtype=torch.long, device=device),
            success_latched=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            target_age_steps=torch.zeros(int(num_envs), dtype=torch.long, device=device),
            first_success_after_steps=torch.full((int(num_envs),), -1, dtype=torch.long, device=device),
        )

    @property
    def num_envs(self) -> int:
        return int(self.previous_distance_m.numel())

    def reset(self, env_ids: torch.Tensor, current_distance_m: torch.Tensor) -> None:
        """Reset every target-specific field for selected environments.

        Seeding ``previous_distance_m`` with distance to the *new* target is
        what makes progress on the switch interval exactly zero.
        """

        if env_ids.dtype == torch.bool:
            if env_ids.shape != self.previous_distance_m.shape:
                raise ValueError("Boolean env_ids must have one entry per environment")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten()
        else:
            ids = env_ids.to(device=self.previous_distance_m.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError("env_ids contains an out-of-range environment index")
        distance = torch.as_tensor(
            current_distance_m,
            device=self.previous_distance_m.device,
            dtype=self.previous_distance_m.dtype,
        ).flatten()
        if distance.numel() == self.num_envs:
            distance = distance[ids]
        elif distance.numel() != ids.numel():
            raise ValueError("current_distance_m must cover either all environments or selected env_ids")
        if torch.any(~torch.isfinite(distance)) or torch.any(distance < 0.0):
            raise ValueError("current_distance_m must be finite and non-negative at target reset")
        self.previous_distance_m[ids] = distance
        self.dwell_steps[ids] = 0
        self.success_latched[ids] = False
        self.target_age_steps[ids] = 0
        self.first_success_after_steps[ids] = -1

    def update(self, distance_m: torch.Tensor, speed_m_s: torch.Tensor) -> TargetTrackingUpdate:
        distance = torch.as_tensor(
            distance_m,
            device=self.previous_distance_m.device,
            dtype=self.previous_distance_m.dtype,
        )
        speed = torch.as_tensor(speed_m_s, device=distance.device, dtype=distance.dtype)
        if distance.shape != self.previous_distance_m.shape or speed.shape != distance.shape:
            raise ValueError("distance and speed must contain one scalar per environment")
        progress = self.previous_distance_m - distance
        dwell = advance_success_dwell(self.dwell_steps, self.success_latched, distance, speed)
        self.target_age_steps.add_(1)
        self.first_success_after_steps.copy_(
            torch.where(
                dwell.newly_successful,
                self.target_age_steps,
                self.first_success_after_steps,
            )
        )
        self.previous_distance_m.copy_(distance)
        self.dwell_steps.copy_(dwell.dwell_steps)
        self.success_latched.copy_(dwell.success_latched)
        return TargetTrackingUpdate(
            distance_m=distance.clone(),
            progress_m=progress,
            in_success_tube=dwell.in_success_tube,
            newly_successful=dwell.newly_successful,
            success_latched=dwell.success_latched,
            dwell_steps=dwell.dwell_steps,
            target_age_steps=self.target_age_steps.clone(),
        )


@dataclass(frozen=True)
class RecoveryTrackingUpdate:
    in_window: torch.Tensor
    newly_recovered: torch.Tensor
    newly_failed: torch.Tensor
    active: torch.Tensor
    recovery_latency_s: torch.Tensor


@dataclass
class RecoveryTrackingState:
    """One non-overlapping recovery attempt per environment at a time."""

    active: torch.Tensor
    stable_before_gust: torch.Tensor
    dwell_steps: torch.Tensor
    recovered: torch.Tensor
    start_step: torch.Tensor
    stop_step: torch.Tensor
    completion_step: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device | str = "cpu") -> "RecoveryTrackingState":
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        count = int(num_envs)
        return cls(
            active=torch.zeros(count, dtype=torch.bool, device=device),
            stable_before_gust=torch.zeros(count, dtype=torch.bool, device=device),
            dwell_steps=torch.zeros(count, dtype=torch.long, device=device),
            recovered=torch.zeros(count, dtype=torch.bool, device=device),
            start_step=torch.full((count,), -1, dtype=torch.long, device=device),
            stop_step=torch.full((count,), -1, dtype=torch.long, device=device),
            completion_step=torch.full((count,), -1, dtype=torch.long, device=device),
        )

    @property
    def num_envs(self) -> int:
        return int(self.active.numel())

    def begin_after_gust(
        self,
        env_ids: torch.Tensor,
        gust_start_step: int | torch.Tensor,
        stable_before_gust: bool | torch.Tensor,
        *,
        gust_duration_steps: int = GUST_DURATION_STEPS,
        recovery_window_steps: int = RECOVERY_WINDOW_STEPS,
    ) -> None:
        """Reset dwell and start a fresh attempt after a scheduled gust."""

        if env_ids.dtype == torch.bool:
            if env_ids.shape != self.active.shape:
                raise ValueError("Boolean env_ids must have one entry per environment")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten()
        else:
            ids = env_ids.to(device=self.active.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError("env_ids contains an out-of-range environment index")
        if torch.any(self.active[ids]):
            raise RuntimeError("cannot overlap recovery attempts for one environment")
        if gust_duration_steps <= 0 or recovery_window_steps <= 0:
            raise ValueError("gust duration and recovery window must be positive")
        gust_start = torch.as_tensor(gust_start_step, device=self.active.device, dtype=torch.long).flatten()
        if gust_start.numel() == 1:
            gust_start = gust_start.expand(ids.numel())
        elif gust_start.numel() == self.num_envs:
            gust_start = gust_start[ids]
        elif gust_start.numel() != ids.numel():
            raise ValueError("gust_start_step must be scalar, selected-env, or all-env sized")
        stable = torch.as_tensor(stable_before_gust, device=self.active.device, dtype=torch.bool).flatten()
        if stable.numel() == 1:
            stable = stable.expand(ids.numel())
        elif stable.numel() == self.num_envs:
            stable = stable[ids]
        elif stable.numel() != ids.numel():
            raise ValueError("stable_before_gust must be scalar, selected-env, or all-env sized")
        start = gust_start + int(gust_duration_steps)
        self.active[ids] = True
        self.stable_before_gust[ids] = stable
        self.dwell_steps[ids] = 0
        self.recovered[ids] = False
        self.start_step[ids] = start
        self.stop_step[ids] = start + int(recovery_window_steps)
        self.completion_step[ids] = -1

    def update(
        self,
        episode_step: torch.Tensor,
        distance_m: torch.Tensor,
        speed_m_s: torch.Tensor,
        *,
        terminated: torch.Tensor | None = None,
        required_dwell_steps: int = SUCCESS_DWELL_STEPS,
    ) -> RecoveryTrackingUpdate:
        if (
            isinstance(required_dwell_steps, bool)
            or int(required_dwell_steps) != required_dwell_steps
            or required_dwell_steps <= 0
        ):
            raise ValueError("required_dwell_steps must be a positive integer")
        step = episode_step.to(device=self.active.device, dtype=torch.long)
        distance = distance_m.to(device=self.active.device)
        speed = speed_m_s.to(device=self.active.device)
        if step.shape != self.active.shape or distance.shape != step.shape or speed.shape != step.shape:
            raise ValueError("episode_step, distance, and speed must have one value per environment")
        terminated_mask = (
            torch.zeros_like(self.active)
            if terminated is None
            else terminated.to(device=self.active.device, dtype=torch.bool)
        )
        if terminated_mask.shape != self.active.shape:
            raise ValueError("terminated must have one value per environment")
        in_window = self.active & (step >= self.start_step) & (step < self.stop_step)
        inside = success_mask(distance, speed)
        assert isinstance(inside, torch.Tensor)
        self.dwell_steps.copy_(
            torch.where(
                in_window & inside,
                torch.clamp(self.dwell_steps + 1, max=int(required_dwell_steps)),
                torch.where(in_window, torch.zeros_like(self.dwell_steps), self.dwell_steps),
            )
        )
        newly_recovered = in_window & (self.dwell_steps >= int(required_dwell_steps))
        self.recovered |= newly_recovered
        self.completion_step.copy_(torch.where(newly_recovered, step, self.completion_step))
        expired = self.active & (step >= self.stop_step)
        newly_failed = (expired | (self.active & terminated_mask)) & ~newly_recovered
        self.active &= ~(newly_recovered | newly_failed)
        latency_steps = self.completion_step - self.start_step + 1
        latency_s = torch.where(
            newly_recovered,
            latency_steps.to(dtype=distance.dtype) * CONTROL_DT_S,
            torch.full_like(distance, torch.nan),
        )
        return RecoveryTrackingUpdate(
            in_window=in_window,
            newly_recovered=newly_recovered,
            newly_failed=newly_failed,
            active=self.active.clone(),
            recovery_latency_s=latency_s,
        )


# Import-time assertions make accidental protocol edits fail loudly.
assert seconds_to_control_steps(EPISODE_DURATION_S) == EPISODE_STEPS
assert schedule_steps(DEFAULT_SWITCH_TIMES_S) == DEFAULT_SWITCH_STEPS
assert schedule_steps(DEFAULT_GUST_TIMES_S) == DEFAULT_GUST_STEPS
assert seconds_to_control_steps(GUST_DURATION_S) == GUST_DURATION_STEPS
assert seconds_to_control_steps(SUCCESS_DWELL_S) == SUCCESS_DWELL_STEPS
assert seconds_to_control_steps(RECOVERY_WINDOW_S) == RECOVERY_WINDOW_STEPS


__all__ = [
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_DT_S",
    "EPISODE_DURATION_S",
    "EPISODE_STEPS",
    "DEFAULT_SWITCH_TIMES_S",
    "DEFAULT_SWITCH_STEPS",
    "DEFAULT_GUST_TIMES_S",
    "DEFAULT_GUST_STEPS",
    "DEFAULT_GUST_START_STEPS",
    "GUST_DURATION_S",
    "GUST_DURATION_STEPS",
    "GUST_DELTA_V_M_S",
    "SUCCESS_DISTANCE_M",
    "SUCCESS_SPEED_M_S",
    "SUCCESS_DWELL_S",
    "SUCCESS_DWELL_STEPS",
    "RECOVERY_WINDOW_S",
    "RECOVERY_WINDOW_STEPS",
    "GUST_RESPONSE_ABS_TOL_N_S",
    "GUST_RESPONSE_REL_TOL",
    "GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S",
    "AUDITED_CRAZYFLIE_MASS_KG",
    "CRAZYFLIE_MASS_ABS_TOL_KG",
    "FAILURE_NONE",
    "FAILURE_LOW_HEIGHT",
    "FAILURE_HIGH_HEIGHT",
    "FAILURE_WORKSPACE_ESCAPE",
    "FAILURE_NONFINITE",
    "FAILURE_CAUSE_NAMES",
    "MIXED_SCENARIO_CONTRACT_VERSION",
    "MIXED_SCENARIO_NAMES",
    "SWITCH_TARGET_CURRICULUM_VERSION",
    "BALANCED_SWITCH_TARGET_CURRICULUM_VERSION",
    "BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION",
    "mixed_scenario_codes",
    "mixed_scenario_contract_payload",
    "switch_target_curriculum_payload",
    "balanced_switch_target_curriculum_payload",
    "balanced_v4_switch_target_curriculum_payload",
    "REWARD_CONTRACT_VERSION",
    "DEFAULT_PROGRESS_REWARD_SCALE",
    "DEFAULT_PROGRESS_CLIP_M",
    "DEFAULT_SUCCESS_BONUS",
    "DEFAULT_SURVIVAL_REWARD_PER_INTERVAL",
    "DEFAULT_CONTROL_EFFORT_PENALTY_SCALE",
    "DEFAULT_ACTION_CHANGE_PENALTY_SCALE",
    "DEFAULT_FAILURE_PENALTY",
    "DEFAULT_COLLECTIVE_HOVER_ACTION",
    "DEFAULT_MOMENT_ACTION_REFERENCE",
    "DEFAULT_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP",
    "DEFAULT_ACTION_CHANGE_NORMALIZED_SQUARED_CAP",
    "DEFAULT_REWARD_CONTRACT_SHA256",
    "BALANCED_REWARD_CONTRACT_VERSION",
    "BALANCED_PROGRESS_POTENTIAL_SCALE_M",
    "BALANCED_PROGRESS_REWARD_SCALE",
    "BALANCED_PROXIMITY_REWARD_SCALE",
    "BALANCED_PROXIMITY_DISTANCE_SCALE_M",
    "BALANCED_DWELL_REWARD_PER_INTERVAL",
    "BALANCED_BRAKING_PENALTY_SCALE",
    "BALANCED_BRAKING_SPEED_REFERENCE_M_S",
    "BALANCED_BRAKING_NORMALIZED_SQUARED_CAP",
    "BALANCED_SUCCESS_BONUS",
    "BALANCED_RETENTION_PENALTY_SCALE",
    "BALANCED_RETENTION_RAMP_M",
    "BALANCED_SURVIVAL_REWARD_PER_INTERVAL",
    "BALANCED_CONTROL_EFFORT_PENALTY_SCALE",
    "BALANCED_COLLECTIVE_EFFORT_REFERENCE",
    "BALANCED_MOMENT_EFFORT_REFERENCE",
    "BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP",
    "BALANCED_ACTION_CHANGE_PENALTY_SCALE",
    "BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE",
    "BALANCED_MOMENT_ACTION_CHANGE_REFERENCE",
    "BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP",
    "BALANCED_BOUNDARY_PENALTY_SCALE",
    "BALANCED_BOUNDARY_LOW_ONSET_M",
    "BALANCED_BOUNDARY_LOW_WIDTH_M",
    "BALANCED_BOUNDARY_HIGH_ONSET_M",
    "BALANCED_BOUNDARY_HIGH_WIDTH_M",
    "BALANCED_BOUNDARY_XY_ONSET_M",
    "BALANCED_BOUNDARY_XY_WIDTH_M",
    "BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP",
    "BALANCED_FAILURE_PENALTY",
    "BALANCED_REWARD_CONTRACT_SHA256",
    "BALANCED_V4_REWARD_CONTRACT_VERSION",
    "BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M",
    "BALANCED_V4_PROGRESS_REWARD_SCALE",
    "BALANCED_V4_PROXIMITY_REWARD_SCALE",
    "BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M",
    "BALANCED_V4_GUST_RECOVERY_BONUS",
    "BALANCED_V4_REWARD_CONTRACT_SHA256",
    "TRAINING_CURRICULUM_VERSION",
    "BALANCED_TRAINING_CURRICULUM_VERSION",
    "BALANCED_V4_TRAINING_CURRICULUM_VERSION",
    "TrainingCurriculumStage",
    "DEFAULT_TRAINING_CURRICULUM",
    "DEFAULT_TRAINING_CURRICULUM_SHA256",
    "BALANCED_TRAINING_CURRICULUM",
    "BALANCED_TRAINING_CURRICULUM_SHA256",
    "BALANCED_V4_TRAINING_CURRICULUM",
    "BALANCED_V4_TRAINING_CURRICULUM_SHA256",
    "validate_training_curriculum",
    "validate_monotonic_training_interactions",
    "advance_vector_training_interactions",
    "training_curriculum_stage_index",
    "select_training_curriculum_stage",
    "reset_training_curriculum_stage_index",
    "training_curriculum_payload",
    "training_curriculum_sha256",
    "reward_contract_payload",
    "reward_contract_sha256",
    "survival_first_contract_payload",
    "balanced_reward_contract_payload",
    "balanced_reward_contract_sha256",
    "balanced_training_curriculum_payload",
    "balanced_training_curriculum_sha256",
    "balanced_task_contract_payload",
    "balanced_v4_reward_contract_payload",
    "balanced_v4_reward_contract_sha256",
    "balanced_v4_training_curriculum_payload",
    "balanced_v4_training_curriculum_sha256",
    "balanced_v4_task_contract_payload",
    "FailureClassification",
    "classify_failure_causes",
    "interval_reward_terms",
    "balanced_boundary_metric",
    "balanced_interval_reward_terms",
    "authenticated_gust_recovery_mask",
    "balanced_v4_interval_reward_terms",
    "seconds_to_control_steps",
    "control_steps_to_seconds",
    "schedule_steps",
    "schedule_active_mask",
    "scheduled_event_index",
    "RecoveryWindow",
    "recovery_window",
    "gust_impulse_n_s",
    "gust_force_n",
    "horizontal_gust_force_vector_n",
    "constant_force_impulse_n_s",
    "success_mask",
    "SuccessDwellUpdate",
    "advance_success_dwell",
    "update_success_dwell",
    "TargetTrackingUpdate",
    "TargetTrackingState",
    "RecoveryTrackingUpdate",
    "RecoveryTrackingState",
]
