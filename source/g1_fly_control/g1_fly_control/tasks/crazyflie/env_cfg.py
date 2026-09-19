"""Configuration for the additive Crazyflie waypoint environments."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnvCfg

from .adapter import validate_upstream_contract
from .logic import (
    BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP,
    BALANCED_ACTION_CHANGE_PENALTY_SCALE,
    BALANCED_BOUNDARY_HIGH_ONSET_M,
    BALANCED_BOUNDARY_HIGH_WIDTH_M,
    BALANCED_BOUNDARY_LOW_ONSET_M,
    BALANCED_BOUNDARY_LOW_WIDTH_M,
    BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP,
    BALANCED_BOUNDARY_PENALTY_SCALE,
    BALANCED_BOUNDARY_XY_ONSET_M,
    BALANCED_BOUNDARY_XY_WIDTH_M,
    BALANCED_BRAKING_NORMALIZED_SQUARED_CAP,
    BALANCED_BRAKING_PENALTY_SCALE,
    BALANCED_BRAKING_SPEED_REFERENCE_M_S,
    BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE,
    BALANCED_COLLECTIVE_EFFORT_REFERENCE,
    BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP,
    BALANCED_CONTROL_EFFORT_PENALTY_SCALE,
    BALANCED_DWELL_REWARD_PER_INTERVAL,
    BALANCED_FAILURE_PENALTY,
    BALANCED_MOMENT_ACTION_CHANGE_REFERENCE,
    BALANCED_MOMENT_EFFORT_REFERENCE,
    BALANCED_PROGRESS_POTENTIAL_SCALE_M,
    BALANCED_PROGRESS_REWARD_SCALE,
    BALANCED_PROXIMITY_DISTANCE_SCALE_M,
    BALANCED_PROXIMITY_REWARD_SCALE,
    BALANCED_RETENTION_PENALTY_SCALE,
    BALANCED_RETENTION_RAMP_M,
    BALANCED_REWARD_CONTRACT_SHA256,
    BALANCED_REWARD_CONTRACT_VERSION,
    BALANCED_SUCCESS_BONUS,
    BALANCED_SURVIVAL_REWARD_PER_INTERVAL,
    BALANCED_SWITCH_TARGET_CURRICULUM_VERSION,
    BALANCED_TRAINING_CURRICULUM,
    BALANCED_TRAINING_CURRICULUM_SHA256,
    BALANCED_TRAINING_CURRICULUM_VERSION,
    BALANCED_V4_GUST_RECOVERY_BONUS,
    BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M,
    BALANCED_V4_PROGRESS_REWARD_SCALE,
    BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M,
    BALANCED_V4_PROXIMITY_REWARD_SCALE,
    BALANCED_V4_REWARD_CONTRACT_SHA256,
    BALANCED_V4_REWARD_CONTRACT_VERSION,
    BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION,
    BALANCED_V4_TRAINING_CURRICULUM,
    BALANCED_V4_TRAINING_CURRICULUM_SHA256,
    BALANCED_V4_TRAINING_CURRICULUM_VERSION,
    AUDITED_CRAZYFLIE_MASS_KG,
    CRAZYFLIE_MASS_ABS_TOL_KG,
    DEFAULT_ACTION_CHANGE_NORMALIZED_SQUARED_CAP,
    DEFAULT_ACTION_CHANGE_PENALTY_SCALE,
    DEFAULT_COLLECTIVE_HOVER_ACTION,
    DEFAULT_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP,
    DEFAULT_CONTROL_EFFORT_PENALTY_SCALE,
    DEFAULT_FAILURE_PENALTY,
    DEFAULT_GUST_STEPS,
    DEFAULT_MOMENT_ACTION_REFERENCE,
    DEFAULT_PROGRESS_CLIP_M,
    DEFAULT_PROGRESS_REWARD_SCALE,
    DEFAULT_REWARD_CONTRACT_SHA256,
    DEFAULT_SUCCESS_BONUS,
    DEFAULT_SURVIVAL_REWARD_PER_INTERVAL,
    DEFAULT_SWITCH_STEPS,
    DEFAULT_TRAINING_CURRICULUM,
    DEFAULT_TRAINING_CURRICULUM_SHA256,
    EPISODE_DURATION_S,
    GUST_DELTA_V_M_S,
    GUST_DURATION_STEPS,
    GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S,
    MIXED_SCENARIO_CONTRACT_VERSION,
    RECOVERY_WINDOW_STEPS,
    REWARD_CONTRACT_VERSION,
    SUCCESS_DISTANCE_M,
    SUCCESS_DWELL_STEPS,
    SUCCESS_SPEED_M_S,
    SWITCH_TARGET_CURRICULUM_VERSION,
    TRAINING_CURRICULUM_VERSION,
    TrainingCurriculumStage,
)

# Importing this module is part of constructing an Isaac-backed environment.
# Validate before any simulator resources are allocated.
UPSTREAM_CONTRACT = validate_upstream_contract()


@configclass
class CrazyflieEnvCfg(QuadcopterEnvCfg):
    """Shared, frozen environment contract for all three project scenarios."""

    # Preserve the upstream 100 Hz physics and decimation=2 control mapping.
    episode_length_s = EPISODE_DURATION_S
    observation_space = 12
    action_space = 4
    debug_vis = False

    scenario: str = "waypoint_reach"
    deterministic_eval: bool = False
    switch_target_curriculum_version: str = SWITCH_TARGET_CURRICULUM_VERSION

    # Full-distribution reset contract.  The final curriculum stage must match
    # these values exactly; deterministic evaluation always uses this stage.
    spawn_height_m: float = 0.50
    spawn_position_xy_half_range_m: float = 0.10
    spawn_position_z_half_range_m: float = 0.05
    spawn_yaw_half_range_rad: float = 0.25
    spawn_linear_velocity_half_range_mps: float = 0.10
    spawn_angular_velocity_half_range_radps: float = 0.20

    # Native command volume, expressed relative to each environment origin.
    goal_xy_min_m: float = -2.0
    goal_xy_max_m: float = 2.0
    goal_z_min_m: float = 0.5
    goal_z_max_m: float = 1.5
    minimum_goal_separation_m: float = 0.75

    # Fixed, interaction-gated training reset curriculum.  Selection happens
    # only on episode reset, so a stage boundary never interrupts a flight.
    training_curriculum_version: str = TRAINING_CURRICULUM_VERSION
    training_curriculum_sha256: str = DEFAULT_TRAINING_CURRICULUM_SHA256
    training_curriculum: tuple[TrainingCurriculumStage, ...] = DEFAULT_TRAINING_CURRICULUM

    # Preserve native vertical failure bounds and add an explicit horizontal
    # escape boundary with margin outside the target volume.
    minimum_height_m: float = 0.10
    maximum_height_m: float = 2.00
    workspace_xy_limit_m: float = 2.75

    # Success and held-out event schedules at 50 Hz.
    success_distance_m: float = SUCCESS_DISTANCE_M
    success_speed_mps: float = SUCCESS_SPEED_M_S
    success_dwell_steps: int = SUCCESS_DWELL_STEPS
    switch_steps: tuple[int, int, int] = DEFAULT_SWITCH_STEPS
    gust_steps: tuple[int, int, int] = DEFAULT_GUST_STEPS
    gust_duration_steps: int = GUST_DURATION_STEPS
    gust_velocity_delta_mps: float = GUST_DELTA_V_M_S
    recovery_window_steps: int = RECOVERY_WINDOW_STEPS

    # Frozen survival-first reward. Progress, valid survival, and both command
    # penalties are per interval; success/failure are one-time events.
    reward_contract_version: str = REWARD_CONTRACT_VERSION
    reward_contract_sha256: str = DEFAULT_REWARD_CONTRACT_SHA256
    progress_reward_scale: float = DEFAULT_PROGRESS_REWARD_SCALE
    progress_reward_clip_m: float = DEFAULT_PROGRESS_CLIP_M
    success_bonus: float = DEFAULT_SUCCESS_BONUS
    survival_reward_per_interval: float = DEFAULT_SURVIVAL_REWARD_PER_INTERVAL
    control_effort_penalty_scale: float = DEFAULT_CONTROL_EFFORT_PENALTY_SCALE
    collective_hover_action: float = DEFAULT_COLLECTIVE_HOVER_ACTION
    moment_action_reference: float = DEFAULT_MOMENT_ACTION_REFERENCE
    control_effort_normalized_squared_cap: float = (
        DEFAULT_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP
    )
    action_change_penalty_scale: float = DEFAULT_ACTION_CHANGE_PENALTY_SCALE
    action_change_normalized_squared_cap: float = (
        DEFAULT_ACTION_CHANGE_NORMALIZED_SQUARED_CAP
    )
    failure_penalty: float = DEFAULT_FAILURE_PENALTY


@configclass
class WaypointReachEnvCfg(CrazyflieEnvCfg):
    scenario: str = "waypoint_reach"


@configclass
class WaypointSwitchEnvCfg(CrazyflieEnvCfg):
    scenario: str = "waypoint_switch"


@configclass
class GustRecoveryEnvCfg(CrazyflieEnvCfg):
    scenario: str = "gust_recovery"


@configclass
class MixedTrainingEnvCfg(CrazyflieEnvCfg):
    """Training-only deterministic per-environment mixture of all scenarios."""

    scenario: str = "mixed"
    mixed_scenario_contract_version: str = MIXED_SCENARIO_CONTRACT_VERSION
    mixed_scenario_seed: int = 0


@configclass
class BalancedV3EnvCfg(CrazyflieEnvCfg):
    """Active balanced-task contract; survival-v2 remains explicit compatibility."""

    switch_target_curriculum_version: str = (
        BALANCED_SWITCH_TARGET_CURRICULUM_VERSION
    )
    training_curriculum_version: str = BALANCED_TRAINING_CURRICULUM_VERSION
    training_curriculum_sha256: str = BALANCED_TRAINING_CURRICULUM_SHA256
    training_curriculum: tuple[TrainingCurriculumStage, ...] = (
        BALANCED_TRAINING_CURRICULUM
    )

    reward_contract_version: str = BALANCED_REWARD_CONTRACT_VERSION
    reward_contract_sha256: str = BALANCED_REWARD_CONTRACT_SHA256
    balanced_progress_potential_scale_m: float = BALANCED_PROGRESS_POTENTIAL_SCALE_M
    balanced_progress_reward_scale: float = BALANCED_PROGRESS_REWARD_SCALE
    proximity_reward_scale: float = BALANCED_PROXIMITY_REWARD_SCALE
    proximity_distance_scale_m: float = BALANCED_PROXIMITY_DISTANCE_SCALE_M
    dwell_reward_per_interval: float = BALANCED_DWELL_REWARD_PER_INTERVAL
    braking_penalty_scale: float = BALANCED_BRAKING_PENALTY_SCALE
    braking_speed_reference_mps: float = BALANCED_BRAKING_SPEED_REFERENCE_M_S
    braking_normalized_squared_cap: float = BALANCED_BRAKING_NORMALIZED_SQUARED_CAP
    success_bonus: float = BALANCED_SUCCESS_BONUS
    retention_penalty_scale: float = BALANCED_RETENTION_PENALTY_SCALE
    retention_ramp_m: float = BALANCED_RETENTION_RAMP_M
    survival_reward_per_interval: float = BALANCED_SURVIVAL_REWARD_PER_INTERVAL
    control_effort_penalty_scale: float = BALANCED_CONTROL_EFFORT_PENALTY_SCALE
    collective_effort_reference: float = BALANCED_COLLECTIVE_EFFORT_REFERENCE
    moment_effort_reference: float = BALANCED_MOMENT_EFFORT_REFERENCE
    control_effort_normalized_squared_cap: float = (
        BALANCED_CONTROL_EFFORT_NORMALIZED_SQUARED_CAP
    )
    action_change_penalty_scale: float = BALANCED_ACTION_CHANGE_PENALTY_SCALE
    collective_action_change_reference: float = (
        BALANCED_COLLECTIVE_ACTION_CHANGE_REFERENCE
    )
    moment_action_change_reference: float = BALANCED_MOMENT_ACTION_CHANGE_REFERENCE
    action_change_normalized_squared_cap: float = (
        BALANCED_ACTION_CHANGE_NORMALIZED_SQUARED_CAP
    )
    boundary_penalty_scale: float = BALANCED_BOUNDARY_PENALTY_SCALE
    boundary_low_onset_m: float = BALANCED_BOUNDARY_LOW_ONSET_M
    boundary_low_width_m: float = BALANCED_BOUNDARY_LOW_WIDTH_M
    boundary_high_onset_m: float = BALANCED_BOUNDARY_HIGH_ONSET_M
    boundary_high_width_m: float = BALANCED_BOUNDARY_HIGH_WIDTH_M
    boundary_xy_onset_m: float = BALANCED_BOUNDARY_XY_ONSET_M
    boundary_xy_width_m: float = BALANCED_BOUNDARY_XY_WIDTH_M
    boundary_normalized_squared_cap: float = (
        BALANCED_BOUNDARY_NORMALIZED_SQUARED_CAP
    )
    failure_penalty: float = BALANCED_FAILURE_PENALTY


@configclass
class BalancedWaypointReachEnvCfg(BalancedV3EnvCfg):
    scenario: str = "waypoint_reach"


@configclass
class BalancedWaypointSwitchEnvCfg(BalancedV3EnvCfg):
    scenario: str = "waypoint_switch"


@configclass
class BalancedGustRecoveryEnvCfg(BalancedV3EnvCfg):
    scenario: str = "gust_recovery"


@configclass
class BalancedMixedTrainingEnvCfg(BalancedV3EnvCfg):
    scenario: str = "mixed"
    mixed_scenario_contract_version: str = MIXED_SCENARIO_CONTRACT_VERSION
    mixed_scenario_seed: int = 0


@configclass
class BalancedV4EnvCfg(BalancedV3EnvCfg):
    """Goal-strengthened task contract with authenticated gust recovery."""

    switch_target_curriculum_version: str = (
        BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION
    )
    training_curriculum_version: str = BALANCED_V4_TRAINING_CURRICULUM_VERSION
    training_curriculum_sha256: str = BALANCED_V4_TRAINING_CURRICULUM_SHA256
    training_curriculum: tuple[TrainingCurriculumStage, ...] = (
        BALANCED_V4_TRAINING_CURRICULUM
    )

    reward_contract_version: str = BALANCED_V4_REWARD_CONTRACT_VERSION
    reward_contract_sha256: str = BALANCED_V4_REWARD_CONTRACT_SHA256
    balanced_progress_potential_scale_m: float = (
        BALANCED_V4_PROGRESS_POTENTIAL_SCALE_M
    )
    balanced_progress_reward_scale: float = BALANCED_V4_PROGRESS_REWARD_SCALE
    proximity_reward_scale: float = BALANCED_V4_PROXIMITY_REWARD_SCALE
    proximity_distance_scale_m: float = BALANCED_V4_PROXIMITY_DISTANCE_SCALE_M
    gust_recovery_bonus: float = BALANCED_V4_GUST_RECOVERY_BONUS
    gust_submitted_impulse_abs_tol_n_s: float = (
        GUST_SUBMITTED_IMPULSE_ABS_TOL_N_S
    )
    audited_robot_mass_kg: float = AUDITED_CRAZYFLIE_MASS_KG
    audited_robot_mass_abs_tol_kg: float = CRAZYFLIE_MASS_ABS_TOL_KG


@configclass
class BalancedV4WaypointReachEnvCfg(BalancedV4EnvCfg):
    scenario: str = "waypoint_reach"


@configclass
class BalancedV4WaypointSwitchEnvCfg(BalancedV4EnvCfg):
    scenario: str = "waypoint_switch"


@configclass
class BalancedV4GustRecoveryEnvCfg(BalancedV4EnvCfg):
    scenario: str = "gust_recovery"


# Explicit aliases make config discovery friendly without changing the stable
# entry-point names above.
CrazyflieWaypointReachEnvCfg = WaypointReachEnvCfg
CrazyflieWaypointSwitchEnvCfg = WaypointSwitchEnvCfg
CrazyflieGustRecoveryEnvCfg = GustRecoveryEnvCfg
CrazyflieMixedTrainingEnvCfg = MixedTrainingEnvCfg
CrazyflieBalancedWaypointReachEnvCfg = BalancedWaypointReachEnvCfg
CrazyflieBalancedWaypointSwitchEnvCfg = BalancedWaypointSwitchEnvCfg
CrazyflieBalancedGustRecoveryEnvCfg = BalancedGustRecoveryEnvCfg
CrazyflieBalancedMixedTrainingEnvCfg = BalancedMixedTrainingEnvCfg
CrazyflieBalancedV4WaypointReachEnvCfg = BalancedV4WaypointReachEnvCfg
CrazyflieBalancedV4WaypointSwitchEnvCfg = BalancedV4WaypointSwitchEnvCfg
CrazyflieBalancedV4GustRecoveryEnvCfg = BalancedV4GustRecoveryEnvCfg


__all__ = [
    "BalancedGustRecoveryEnvCfg",
    "BalancedMixedTrainingEnvCfg",
    "BalancedV3EnvCfg",
    "BalancedWaypointReachEnvCfg",
    "BalancedWaypointSwitchEnvCfg",
    "BalancedV4EnvCfg",
    "BalancedV4GustRecoveryEnvCfg",
    "BalancedV4WaypointReachEnvCfg",
    "BalancedV4WaypointSwitchEnvCfg",
    "CrazyflieBalancedGustRecoveryEnvCfg",
    "CrazyflieBalancedMixedTrainingEnvCfg",
    "CrazyflieBalancedWaypointReachEnvCfg",
    "CrazyflieBalancedWaypointSwitchEnvCfg",
    "CrazyflieBalancedV4GustRecoveryEnvCfg",
    "CrazyflieBalancedV4WaypointReachEnvCfg",
    "CrazyflieBalancedV4WaypointSwitchEnvCfg",
    "CrazyflieEnvCfg",
    "CrazyflieGustRecoveryEnvCfg",
    "CrazyflieMixedTrainingEnvCfg",
    "CrazyflieWaypointReachEnvCfg",
    "CrazyflieWaypointSwitchEnvCfg",
    "GustRecoveryEnvCfg",
    "MixedTrainingEnvCfg",
    "UPSTREAM_CONTRACT",
    "WaypointReachEnvCfg",
    "WaypointSwitchEnvCfg",
]
