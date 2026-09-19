"""Crazyflie waypoint, switch, and gust-recovery direct environments."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv

from .adapter import validate_runtime_contract
from .env_cfg import CrazyflieEnvCfg
from .logic import (
    BALANCED_REWARD_CONTRACT_VERSION,
    BALANCED_SWITCH_TARGET_CURRICULUM_VERSION,
    BALANCED_V4_REWARD_CONTRACT_VERSION,
    BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION,
    FAILURE_CAUSE_NAMES,
    FAILURE_HIGH_HEIGHT,
    FAILURE_LOW_HEIGHT,
    FAILURE_NONE,
    FAILURE_NONFINITE,
    FAILURE_WORKSPACE_ESCAPE,
    MIXED_SCENARIO_CONTRACT_VERSION,
    MIXED_SCENARIO_NAMES,
    REWARD_CONTRACT_VERSION,
    SWITCH_TARGET_CURRICULUM_VERSION,
    advance_vector_training_interactions,
    authenticated_gust_recovery_mask,
    balanced_interval_reward_terms,
    balanced_reward_contract_sha256,
    balanced_switch_target_curriculum_payload,
    balanced_v4_interval_reward_terms,
    balanced_v4_reward_contract_sha256,
    balanced_v4_switch_target_curriculum_payload,
    classify_failure_causes,
    gust_force_n,
    gust_impulse_n_s,
    interval_reward_terms,
    mixed_scenario_codes,
    mixed_scenario_contract_payload,
    switch_target_curriculum_payload,
    reset_training_curriculum_stage_index,
    reward_contract_sha256,
    training_curriculum_sha256,
    training_curriculum_stage_index,
    validate_monotonic_training_interactions,
    validate_training_curriculum,
)


class CrazyflieEnv(QuadcopterEnv):
    """Thin project-specific extension of NVIDIA's direct quadcopter task.

    The parent class remains responsible for scene construction, the Crazyflie
    asset, and the exact normalized action-to-wrench mapping.  This subclass
    changes only episode/task logic and adds evaluation instrumentation.
    """

    cfg: CrazyflieEnvCfg

    def __init__(self, cfg: CrazyflieEnvCfg, render_mode: str | None = None, **kwargs: Any):
        if cfg.scenario not in {"waypoint_reach", "waypoint_switch", "gust_recovery", "mixed"}:
            raise ValueError(f"Unsupported Crazyflie scenario {cfg.scenario!r}")
        balanced_v3 = cfg.reward_contract_version == BALANCED_REWARD_CONTRACT_VERSION
        balanced_v4 = cfg.reward_contract_version == BALANCED_V4_REWARD_CONTRACT_VERSION
        balanced_task = balanced_v3 or balanced_v4
        if not balanced_task and cfg.reward_contract_version != REWARD_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported Crazyflie reward contract {cfg.reward_contract_version!r}"
            )
        expected_switch_contract = (
            BALANCED_V4_SWITCH_TARGET_CURRICULUM_VERSION
            if balanced_v4
            else BALANCED_SWITCH_TARGET_CURRICULUM_VERSION
            if balanced_v3
            else SWITCH_TARGET_CURRICULUM_VERSION
        )
        if cfg.switch_target_curriculum_version != expected_switch_contract:
            raise ValueError("switch target curriculum version does not match the implementation")
        if cfg.scenario == "mixed":
            if balanced_v4:
                raise ValueError("balanced-v4 intentionally does not support Mixed training")
            if bool(cfg.deterministic_eval):
                raise ValueError("the mixed task is training-only and cannot use deterministic evaluation")
            contract_version = getattr(cfg, "mixed_scenario_contract_version", None)
            if contract_version != MIXED_SCENARIO_CONTRACT_VERSION:
                raise ValueError("mixed scenario contract version does not match the implementation")
            mixed_scenario_contract_payload(seed=getattr(cfg, "mixed_scenario_seed", -1))
        curriculum = validate_training_curriculum(cfg.training_curriculum)
        curriculum_hash = training_curriculum_sha256(
            curriculum, version=cfg.training_curriculum_version
        )
        if curriculum_hash != cfg.training_curriculum_sha256:
            raise ValueError("training curriculum payload does not match its frozen SHA-256")
        if balanced_v4:
            reward_hash = balanced_v4_reward_contract_sha256(
                progress_potential_scale_m=cfg.balanced_progress_potential_scale_m,
                progress_scale=cfg.balanced_progress_reward_scale,
                proximity_scale=cfg.proximity_reward_scale,
                proximity_distance_scale_m=cfg.proximity_distance_scale_m,
                dwell_reward_per_interval=cfg.dwell_reward_per_interval,
                braking_scale=cfg.braking_penalty_scale,
                braking_speed_reference_m_s=cfg.braking_speed_reference_mps,
                braking_normalized_squared_cap=cfg.braking_normalized_squared_cap,
                success_bonus=cfg.success_bonus,
                retention_scale=cfg.retention_penalty_scale,
                retention_ramp_m=cfg.retention_ramp_m,
                survival_reward_per_interval=cfg.survival_reward_per_interval,
                control_effort_scale=cfg.control_effort_penalty_scale,
                collective_hover_action=cfg.collective_hover_action,
                collective_effort_reference=cfg.collective_effort_reference,
                moment_effort_reference=cfg.moment_effort_reference,
                control_effort_normalized_squared_cap=(
                    cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=cfg.action_change_penalty_scale,
                collective_action_change_reference=(
                    cfg.collective_action_change_reference
                ),
                moment_action_change_reference=cfg.moment_action_change_reference,
                action_change_normalized_squared_cap=(
                    cfg.action_change_normalized_squared_cap
                ),
                boundary_scale=cfg.boundary_penalty_scale,
                boundary_low_onset_m=cfg.boundary_low_onset_m,
                boundary_low_width_m=cfg.boundary_low_width_m,
                boundary_high_onset_m=cfg.boundary_high_onset_m,
                boundary_high_width_m=cfg.boundary_high_width_m,
                boundary_xy_onset_m=cfg.boundary_xy_onset_m,
                boundary_xy_width_m=cfg.boundary_xy_width_m,
                boundary_normalized_squared_cap=cfg.boundary_normalized_squared_cap,
                failure_penalty=cfg.failure_penalty,
                success_distance_m=cfg.success_distance_m,
                success_speed_m_s=cfg.success_speed_mps,
                success_dwell_steps=cfg.success_dwell_steps,
                minimum_height_m=cfg.minimum_height_m,
                maximum_height_m=cfg.maximum_height_m,
                workspace_xy_limit_m=cfg.workspace_xy_limit_m,
                gust_recovery_bonus=cfg.gust_recovery_bonus,
                gust_submitted_impulse_abs_tol_n_s=(
                    cfg.gust_submitted_impulse_abs_tol_n_s
                ),
                audited_robot_mass_kg=cfg.audited_robot_mass_kg,
                audited_robot_mass_abs_tol_kg=cfg.audited_robot_mass_abs_tol_kg,
                gust_delta_velocity_m_s=cfg.gust_velocity_delta_mps,
                version=cfg.reward_contract_version,
            )
        elif balanced_v3:
            reward_hash = balanced_reward_contract_sha256(
                progress_potential_scale_m=cfg.balanced_progress_potential_scale_m,
                progress_scale=cfg.balanced_progress_reward_scale,
                proximity_scale=cfg.proximity_reward_scale,
                proximity_distance_scale_m=cfg.proximity_distance_scale_m,
                dwell_reward_per_interval=cfg.dwell_reward_per_interval,
                braking_scale=cfg.braking_penalty_scale,
                braking_speed_reference_m_s=cfg.braking_speed_reference_mps,
                braking_normalized_squared_cap=cfg.braking_normalized_squared_cap,
                success_bonus=cfg.success_bonus,
                retention_scale=cfg.retention_penalty_scale,
                retention_ramp_m=cfg.retention_ramp_m,
                survival_reward_per_interval=cfg.survival_reward_per_interval,
                control_effort_scale=cfg.control_effort_penalty_scale,
                collective_hover_action=cfg.collective_hover_action,
                collective_effort_reference=cfg.collective_effort_reference,
                moment_effort_reference=cfg.moment_effort_reference,
                control_effort_normalized_squared_cap=(
                    cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=cfg.action_change_penalty_scale,
                collective_action_change_reference=(
                    cfg.collective_action_change_reference
                ),
                moment_action_change_reference=cfg.moment_action_change_reference,
                action_change_normalized_squared_cap=(
                    cfg.action_change_normalized_squared_cap
                ),
                boundary_scale=cfg.boundary_penalty_scale,
                boundary_low_onset_m=cfg.boundary_low_onset_m,
                boundary_low_width_m=cfg.boundary_low_width_m,
                boundary_high_onset_m=cfg.boundary_high_onset_m,
                boundary_high_width_m=cfg.boundary_high_width_m,
                boundary_xy_onset_m=cfg.boundary_xy_onset_m,
                boundary_xy_width_m=cfg.boundary_xy_width_m,
                boundary_normalized_squared_cap=cfg.boundary_normalized_squared_cap,
                failure_penalty=cfg.failure_penalty,
                success_distance_m=cfg.success_distance_m,
                success_speed_m_s=cfg.success_speed_mps,
                success_dwell_steps=cfg.success_dwell_steps,
                minimum_height_m=cfg.minimum_height_m,
                maximum_height_m=cfg.maximum_height_m,
                workspace_xy_limit_m=cfg.workspace_xy_limit_m,
                version=cfg.reward_contract_version,
            )
        else:
            reward_hash = reward_contract_sha256(
                progress_scale=cfg.progress_reward_scale,
                progress_clip_m=cfg.progress_reward_clip_m,
                success_bonus=cfg.success_bonus,
                survival_reward_per_interval=cfg.survival_reward_per_interval,
                control_effort_scale=cfg.control_effort_penalty_scale,
                collective_hover_action=cfg.collective_hover_action,
                moment_action_reference=cfg.moment_action_reference,
                control_effort_normalized_squared_cap=(
                    cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=cfg.action_change_penalty_scale,
                action_change_normalized_squared_cap=(
                    cfg.action_change_normalized_squared_cap
                ),
                failure_penalty=cfg.failure_penalty,
                version=cfg.reward_contract_version,
            )
        if reward_hash != cfg.reward_contract_sha256:
            raise ValueError("reward payload does not match its frozen SHA-256")
        expected_full_distribution = {
            "spawn_height_m": float(cfg.spawn_height_m),
            "spawn_position_xy_half_range_m": float(cfg.spawn_position_xy_half_range_m),
            "spawn_position_z_half_range_m": float(cfg.spawn_position_z_half_range_m),
            "spawn_yaw_half_range_rad": float(cfg.spawn_yaw_half_range_rad),
            "spawn_linear_velocity_half_range_mps": float(
                cfg.spawn_linear_velocity_half_range_mps
            ),
            "spawn_angular_velocity_half_range_radps": float(
                cfg.spawn_angular_velocity_half_range_radps
            ),
            "goal_xy_min_m": float(cfg.goal_xy_min_m),
            "goal_xy_max_m": float(cfg.goal_xy_max_m),
            "goal_z_min_m": float(cfg.goal_z_min_m),
            "goal_z_max_m": float(cfg.goal_z_max_m),
            "minimum_goal_separation_m": float(cfg.minimum_goal_separation_m),
        }
        if curriculum[-1].reset_distribution_payload() != expected_full_distribution:
            raise ValueError("final curriculum stage must equal the full task distribution")
        super().__init__(cfg, render_mode, **kwargs)
        validate_runtime_contract(self)
        if balanced_v4 and not math.isclose(
            float(self._robot_mass.item()),
            float(cfg.audited_robot_mass_kg),
            rel_tol=0.0,
            abs_tol=float(cfg.audited_robot_mass_abs_tol_kg),
        ):
            raise ValueError(
                "balanced-v4 Crazyflie mass differs from the audited reward provenance"
            )

        self.scenario = cfg.scenario
        self.deterministic_eval = bool(cfg.deterministic_eval)
        self._balanced_v3 = balanced_v3
        self._balanced_v4 = balanced_v4
        self._balanced_task = balanced_task
        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        fixed_scenario_code = {
            "waypoint_reach": 0,
            "waypoint_switch": 1,
            "gust_recovery": 2,
            "mixed": 0,
        }[self.scenario]
        self.episode_scenario_code = torch.full(
            (self.num_envs,), fixed_scenario_code, dtype=torch.long, device=self.device
        )
        self.terminal_scenario_code = self.episode_scenario_code.clone()
        self.mixed_episode_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.mixed_scenario_episode_starts = torch.zeros(
            len(MIXED_SCENARIO_NAMES), dtype=torch.long, device=self.device
        )
        # The trainer writes an initial restart checkpoint before the first
        # environment step.  Publish the immutable codebook as an instance
        # audit field immediately instead of waiting for ``_get_dones`` to
        # mirror it into ``extras``.
        self.mixed_scenario_names_by_code = MIXED_SCENARIO_NAMES
        self._training_curriculum = curriculum
        self._training_interactions = 0
        self._active_training_curriculum_stage_index = training_curriculum_stage_index(
            0, curriculum
        )
        self.episode_curriculum_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_curriculum_start_interactions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        stage_count = len(curriculum)
        self.curriculum_episode_completions = torch.zeros(
            stage_count, dtype=torch.long, device=self.device
        )
        self.curriculum_failure_terminations = torch.zeros(
            stage_count, dtype=torch.long, device=self.device
        )
        self.curriculum_successful_episodes = torch.zeros(
            stage_count, dtype=torch.long, device=self.device
        )

        # Command/target state.
        self._previous_actions = torch.zeros_like(self._actions)
        self._action_delta = torch.zeros_like(self._actions)
        self._target_sequence_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._reward_previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._distance = torch.zeros(self.num_envs, device=self.device)
        self._distance_progress = torch.zeros(self.num_envs, device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        self._target_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.switch_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_switch_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._switched_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Success is target-specific; cumulative success_count survives target
        # switches but is cleared on episode reset.
        self.success_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Gust forces are applied in the world frame at the body COM through
        # the instantaneous wrench composer, leaving the parent's local-frame
        # thrust/moment mapping untouched.
        self._gust_directions_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.gust_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.gust_direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.gust_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gust_applied_impulse_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.gust_measured_delta_momentum_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.gust_stable_before = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gust_event_applied_impulse_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.gust_event_measured_delta_momentum_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.gust_event_stable_before = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
        self.gust_event_recovered = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
        self.gust_event_recovery_latency_s = torch.full(
            (self.num_envs, 3), float("nan"), device=self.device
        )
        self.gust_recovery_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gust_recovery_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gust_recovered_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_authenticated_gust_recovery_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._current_recovery_event = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._active_gust_event = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._gust_ending_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gust_start_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._gust_recovery_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gust_recovery_deadline_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gust_force_magnitude_n = gust_force_n(
            float(self._robot_mass.item()),
            delta_v_m_s=self.cfg.gust_velocity_delta_mps,
            duration_s=self.cfg.gust_duration_steps * self.step_dt,
        )
        self.gust_expected_impulse_magnitude_n_s = gust_impulse_n_s(
            float(self._robot_mass.item()), delta_v_m_s=self.cfg.gust_velocity_delta_mps
        )

        # Failure, terminal, and reward instrumentation.  Terminal tensors are
        # snapshots from before DirectRLEnv performs its automatic reset.
        self.failure_cause = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.drone_terminal_observation = torch.zeros(self.num_envs, 12, device=self.device)
        self.crazyflie_terminal_observation = self.drone_terminal_observation
        self.flyg1_terminal_observation = self.drone_terminal_observation  # documented compatibility alias
        self.terminal_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.terminal_position_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.terminal_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_failure_cause = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_distance_m = torch.zeros(self.num_envs, device=self.device)
        self.terminal_speed_mps = torch.zeros(self.num_envs, device=self.device)
        self.terminal_switch_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_gust_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_gust_event_applied_impulse_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.terminal_gust_event_expected_impulse_w = torch.zeros(
            self.num_envs, 3, 3, device=self.device
        )
        self.terminal_gust_event_measured_delta_momentum_w = torch.zeros(
            self.num_envs, 3, 3, device=self.device
        )
        self.terminal_gust_event_stable_before = torch.zeros(
            self.num_envs, 3, dtype=torch.bool, device=self.device
        )
        self.terminal_gust_event_recovered = torch.zeros(
            self.num_envs, 3, dtype=torch.bool, device=self.device
        )
        self.terminal_gust_event_recovery_latency_s = torch.full(
            (self.num_envs, 3), float("nan"), device=self.device
        )
        self.terminal_mechanical_work_proxy = torch.zeros(self.num_envs, device=self.device)
        self.terminal_curriculum_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_curriculum_start_interactions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reward_components: dict[str, torch.Tensor] = {}
        self.mechanical_work_proxy_step = torch.zeros(self.num_envs, device=self.device)
        self.mechanical_work_proxy = torch.zeros(self.num_envs, device=self.device)
        balanced_reward_component_names = (
            "progress",
            "proximity",
            "dwell",
            "braking",
            "success_bonus",
            "retention",
            "survival",
            "control_effort",
            "action_change",
            "boundary",
            "failure",
        )
        reward_component_names = (
            (
                *balanced_reward_component_names,
                "gust_recovery_bonus",
                "total",
            )
            if self._balanced_v4
            else (*balanced_reward_component_names, "total")
            if self._balanced_task
            else (
                "progress",
                "success_bonus",
                "survival",
                "control_effort",
                "action_change",
                "failure",
                "total",
            )
        )
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (*reward_component_names, "mechanical_work_proxy")
        }

        # Optional immutable evaluation plan, used on the next and subsequent
        # explicit resets until clear_episode_plan() is called.
        self._planned_initial_root_state_w: torch.Tensor | None = None
        self._planned_targets_w: torch.Tensor | None = None
        self._planned_gust_directions_w: torch.Tensor | None = None

    @property
    def target_sequence_w(self) -> torch.Tensor:
        return self._target_sequence_w

    @property
    def target_elapsed_steps(self) -> torch.Tensor:
        return self._target_elapsed_steps

    @property
    def robot_mass_kg(self) -> float:
        return float(self._robot_mass.item())

    @property
    def training_interactions(self) -> int:
        return self._training_interactions

    @property
    def active_training_curriculum_stage_index(self) -> int:
        return self._active_training_curriculum_stage_index

    def _scenario_mask(self, scenario: str) -> torch.Tensor:
        """Return the per-environment mask for fixed or mixed training."""

        try:
            code = MIXED_SCENARIO_NAMES.index(scenario)
        except ValueError as exc:
            raise ValueError(f"Unknown scenario {scenario!r}") from exc
        return self.episode_scenario_code == code

    @property
    def active_training_curriculum_stage_payload(self) -> dict[str, str | int | float]:
        return self._training_curriculum[
            self._active_training_curriculum_stage_index
        ].payload()

    @property
    def full_training_curriculum_stage_payload(self) -> dict[str, str | int | float]:
        return self._training_curriculum[-1].payload()

    def set_training_interactions(self, total_interactions: int) -> None:
        """Advance the reset curriculum clock without mutating active episodes."""

        self._training_interactions = validate_monotonic_training_interactions(
            self._training_interactions, total_interactions
        )
        self._active_training_curriculum_stage_index = training_curriculum_stage_index(
            self._training_interactions, self._training_curriculum
        )

    def _episode_plan_is_installed(self) -> bool:
        return any(
            value is not None
            for value in (
                self._planned_initial_root_state_w,
                self._planned_targets_w,
                self._planned_gust_directions_w,
            )
        )

    def _stage_index_for_new_episode(self) -> int:
        return reset_training_curriculum_stage_index(
            self._training_interactions,
            self._training_curriculum,
            deterministic_eval=self.deterministic_eval,
            episode_plan_installed=self._episode_plan_is_installed(),
        )

    def _curriculum_stage_for_ids(self, ids: torch.Tensor):
        indices = torch.unique(self.episode_curriculum_stage_index[ids])
        if indices.numel() != 1:
            raise RuntimeError("one reset batch cannot mix curriculum stages")
        return self._training_curriculum[int(indices.item())]

    @property
    def aggregate_force_b(self) -> torch.Tensor:
        """Controller-commanded aggregate force in the body frame, in newtons."""

        return self._thrust[:, 0, :]

    @property
    def aggregate_moment_b(self) -> torch.Tensor:
        """Controller-commanded aggregate moment in the body frame, in N m."""

        return self._moment[:, 0, :]

    @property
    def gust_vector_w(self) -> torch.Tensor:
        """Current external gust-force vector in the world frame, in newtons."""

        return self.gust_force_w

    @property
    def gust_actual_impulse_w(self) -> torch.Tensor:
        """Force-time integral actually submitted for the most recent gust."""

        return self.gust_applied_impulse_w

    @property
    def gust_expected_impulse_w(self) -> torch.Tensor:
        """Desired impulse vector for the most recent gust, in N s."""

        return self.gust_direction_w * self.gust_expected_impulse_magnitude_n_s

    @property
    def gust_event_expected_impulse_w(self) -> torch.Tensor:
        """Desired impulse vectors for all three predeclared gusts, in N s."""

        return self._gust_directions_w * self.gust_expected_impulse_magnitude_n_s

    def set_episode_plan(
        self,
        *,
        initial_root_state_w: torch.Tensor | Sequence[Sequence[float]] | None = None,
        targets_w: torch.Tensor | Sequence[Any] | None = None,
        gust_directions_w: torch.Tensor | Sequence[Any] | None = None,
    ) -> None:
        """Install a deterministic per-environment plan for future resets.

        Root states use Isaac's 13-value ``(position, wxyz quaternion, linear
        velocity, angular velocity)`` world-frame order. Targets are world
        coordinates with shape ``(num_envs, targets, 3)``; switch scenarios
        require four targets. Gust directions have shape ``(num_envs, 3, 2|3)``
        and are normalized in the horizontal world plane here.
        """

        if self.scenario == "mixed":
            raise ValueError(
                "FlyCrazyflie-Mixed-v0 is training-only; use the three official task IDs for planned evaluation"
            )

        def tensor_or_none(value: Any) -> torch.Tensor | None:
            if value is None:
                return None
            return torch.as_tensor(value, dtype=torch.float32, device=self.device).clone()

        root = tensor_or_none(initial_root_state_w)
        targets = tensor_or_none(targets_w)
        gusts = tensor_or_none(gust_directions_w)
        if root is not None:
            if root.shape != (self.num_envs, 13) or not torch.isfinite(root).all():
                raise ValueError(f"initial_root_state_w must be finite with shape {(self.num_envs, 13)}")
        if targets is not None:
            if targets.ndim == 2 and targets.shape == (self.num_envs, 3):
                targets = targets[:, None, :]
            required = 4 if self.scenario in {"waypoint_switch", "mixed"} else 1
            if targets.ndim != 3 or targets.shape[0] != self.num_envs or targets.shape[2] != 3:
                raise ValueError("targets_w must have shape (num_envs, targets, 3)")
            if targets.shape[1] < required or not torch.isfinite(targets).all():
                raise ValueError(f"scenario {self.scenario!r} requires {required} finite target(s)")
            targets = targets[:, :required, :]
        if gusts is not None:
            if gusts.ndim != 3 or gusts.shape[:2] != (self.num_envs, 3) or gusts.shape[2] not in (2, 3):
                raise ValueError("gust_directions_w must have shape (num_envs, 3, 2|3)")
            horizontal = torch.zeros(self.num_envs, 3, 3, device=self.device)
            horizontal[..., :2] = gusts[..., :2]
            norm = torch.linalg.vector_norm(horizontal[..., :2], dim=-1, keepdim=True)
            if not torch.isfinite(horizontal).all() or torch.any(norm <= 1.0e-8):
                raise ValueError("gust directions must be finite and nonzero in the horizontal plane")
            gusts = horizontal / norm

        self._planned_initial_root_state_w = root
        self._planned_targets_w = targets
        self._planned_gust_directions_w = gusts

    def clear_episode_plan(self) -> None:
        self._planned_initial_root_state_w = None
        self._planned_targets_w = None
        self._planned_gust_directions_w = None

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        if options:
            episode_plan = options.get("episode_plan")
            if episode_plan is not None:
                if not isinstance(episode_plan, dict):
                    raise TypeError("options['episode_plan'] must be a mapping")
                self.set_episode_plan(**episode_plan)
        observations, extras = super().reset(seed=seed, options=options)
        # This explicit assignment is an acceptance invariant: every full
        # evaluation episode begins at decision step zero.
        self.episode_length_buf.zero_()
        return observations, extras

    def _compute_policy_observation(self) -> torch.Tensor:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        return torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": self._compute_policy_observation()}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        previous = self._previous_actions.clone()
        super()._pre_physics_step(actions)  # preserves NVIDIA's exact clamp and wrench mapping
        self._action_delta.copy_(self._actions - previous)
        self._previous_actions.copy_(self._actions)
        self._configure_gust_for_step()

    def _configure_gust_for_step(self) -> None:
        self.gust_force_w.zero_()
        self._active_gust_event.fill_(-1)
        self._gust_ending_this_step.zero_()
        gust_envs = self._scenario_mask("gust_recovery")
        if not torch.any(gust_envs):
            return

        step = self.episode_length_buf
        for event_index, start in enumerate(self.cfg.gust_steps):
            onset = (step == start) & gust_envs
            active = (
                (step >= start)
                & (step < start + self.cfg.gust_duration_steps)
                & gust_envs
            )
            ending = (step == start + self.cfg.gust_duration_steps - 1) & gust_envs
            if torch.any(onset):
                distance = torch.linalg.vector_norm(
                    self._desired_pos_w - self._robot.data.root_pos_w, dim=-1
                )
                speed = torch.linalg.vector_norm(self._robot.data.root_lin_vel_w, dim=-1)
                stable = (
                    torch.isfinite(distance)
                    & torch.isfinite(speed)
                    & (distance <= self.cfg.success_distance_m)
                    & (speed <= self.cfg.success_speed_mps)
                )
                direction = self._gust_directions_w[:, event_index, :]
                self.gust_count[onset] += 1
                self.gust_direction_w[onset] = direction[onset]
                self.gust_applied_impulse_w[onset] = 0.0
                self.gust_measured_delta_momentum_w[onset] = 0.0
                self.gust_stable_before[onset] = stable[onset]
                self.gust_event_stable_before[onset, event_index] = stable[onset]
                self._gust_start_velocity_w[onset] = self._robot.data.root_lin_vel_w[onset]
                self.gust_recovery_active[onset] = True
                self.gust_recovery_dwell[onset] = 0
                self._current_recovery_event[onset] = event_index
                self._gust_recovery_start_step[onset] = start + self.cfg.gust_duration_steps
                self._gust_recovery_deadline_step[onset] = (
                    start + self.cfg.gust_duration_steps + self.cfg.recovery_window_steps
                )
            self.gust_force_w[active] = (
                self._gust_directions_w[active, event_index, :] * self.gust_force_magnitude_n
            )
            self._active_gust_event[active] = event_index
            self._gust_ending_this_step |= ending

    def _apply_action(self) -> None:
        super()._apply_action()
        active = self._active_gust_event >= 0
        if not torch.any(active):
            return
        self._robot.instantaneous_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self.gust_force_w[:, None, :],
            is_global=True,
        )
        impulse_increment = self.gust_force_w * self.physics_dt
        self.gust_applied_impulse_w[active] += impulse_increment[active]
        for event_index in range(3):
            event_mask = self._active_gust_event == event_index
            self.gust_event_applied_impulse_w[event_mask, event_index, :] += impulse_increment[event_mask]

    def _apply_switch_schedule(self) -> None:
        self._switched_this_step.zero_()
        switch_envs = self._scenario_mask("waypoint_switch")
        if not torch.any(switch_envs):
            return
        for switch_index, switch_step in enumerate(self.cfg.switch_steps):
            mask = (self.episode_length_buf == switch_step) & switch_envs
            if not torch.any(mask):
                continue
            self._desired_pos_w[mask] = self._target_sequence_w[mask, switch_index + 1, :]
            current_distance = torch.linalg.vector_norm(
                self._desired_pos_w[mask] - self._robot.data.root_pos_w[mask], dim=-1
            )
            self._previous_distance[mask] = torch.nan_to_num(
                current_distance, nan=0.0, posinf=0.0, neginf=0.0
            )
            self.success_dwell[mask] = 0
            self.success_latched[mask] = False
            self._target_elapsed_steps[mask] = 0
            self.switch_count[mask] += 1
            self.last_switch_step[mask] = switch_step
            self._switched_this_step |= mask

    def _update_task_state(self) -> torch.Tensor:
        self._apply_switch_schedule()
        self._reward_previous_distance.copy_(self._previous_distance)
        raw_distance = torch.linalg.vector_norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=-1
        )
        raw_speed = torch.linalg.vector_norm(self._robot.data.root_lin_vel_w, dim=-1)
        finite_metrics = torch.isfinite(raw_distance) & torch.isfinite(raw_speed)
        self._distance.copy_(torch.nan_to_num(raw_distance, nan=1.0e6, posinf=1.0e6, neginf=1.0e6))
        self._speed.copy_(torch.nan_to_num(raw_speed, nan=1.0e6, posinf=1.0e6, neginf=1.0e6))
        self._distance_progress.copy_(self._previous_distance - self._distance)
        self._distance_progress[~finite_metrics | self._switched_this_step] = 0.0
        self._previous_distance.copy_(self._distance)

        inside_tube = (
            finite_metrics
            & (self._distance <= self.cfg.success_distance_m)
            & (self._speed <= self.cfg.success_speed_mps)
        )
        # A switch occurs at the boundary after the preceding interval.  The
        # new target's dwell starts with the following control interval, so it
        # remains exactly zero on the switch-returned observation.
        target_dwell_sample = inside_tube & ~self._switched_this_step
        self.success_dwell.copy_(
            torch.where(target_dwell_sample, self.success_dwell + 1, 0)
        )
        self._new_success_this_step.copy_(
            (self.success_dwell >= self.cfg.success_dwell_steps) & ~self.success_latched
        )
        self.success_latched |= self._new_success_this_step
        self.success_count += self._new_success_this_step.long()

        self._update_gust_recovery(inside_tube)
        self._target_elapsed_steps += (~self._switched_this_step).long()
        return finite_metrics

    def _update_gust_recovery(self, inside_tube: torch.Tensor) -> None:
        self._new_authenticated_gust_recovery_this_step.zero_()
        gust_envs = self._scenario_mask("gust_recovery")
        if not torch.any(gust_envs):
            return

        ending = self._gust_ending_this_step
        if torch.any(ending):
            measured = self._robot_mass * (
                self._robot.data.root_lin_vel_w - self._gust_start_velocity_w
            )
            measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
            self.gust_measured_delta_momentum_w[ending] = measured[ending]
            for event_index in range(3):
                mask = ending & (self._active_gust_event == event_index)
                self.gust_event_measured_delta_momentum_w[mask, event_index, :] = measured[mask]

        in_window = (
            self.gust_recovery_active
            & gust_envs
            & (self.episode_length_buf > self._gust_recovery_start_step)
            & (self.episode_length_buf <= self._gust_recovery_deadline_step)
        )
        self.gust_recovery_dwell.copy_(
            torch.where(in_window & inside_tube, self.gust_recovery_dwell + 1, 0)
        )
        recovered = in_window & (self.gust_recovery_dwell >= self.cfg.success_dwell_steps)
        if self._balanced_v4:
            self._new_authenticated_gust_recovery_this_step.copy_(
                authenticated_gust_recovery_mask(
                    recovered,
                    self._current_recovery_event,
                    self.gust_event_applied_impulse_w,
                    self.gust_event_expected_impulse_w,
                    max_abs_error_n_s=(
                        self.cfg.gust_submitted_impulse_abs_tol_n_s
                    ),
                )
            )
        for event_index in range(3):
            mask = recovered & (self._current_recovery_event == event_index)
            if torch.any(mask):
                latency = (
                    self.episode_length_buf[mask] - self._gust_recovery_start_step[mask]
                ).float() * self.step_dt
                self.gust_event_recovered[mask, event_index] = True
                self.gust_event_recovery_latency_s[mask, event_index] = latency
        self.gust_recovered_count += recovered.long()
        self.gust_recovery_active[recovered] = False
        expired = self.gust_recovery_active & (
            self.episode_length_buf >= self._gust_recovery_deadline_step
        )
        self.gust_recovery_active[expired] = False

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        finite_metrics = self._update_task_state()
        root_state = torch.cat(
            (
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self._robot.data.projected_gravity_b,
                self._actions,
            ),
            dim=-1,
        )
        finite_state = finite_metrics & torch.isfinite(root_state).all(dim=-1)
        failure = classify_failure_causes(
            self._robot.data.root_pos_w,
            self._terrain.env_origins,
            finite_state,
            minimum_height_m=self.cfg.minimum_height_m,
            maximum_height_m=self.cfg.maximum_height_m,
            workspace_xy_limit_m=self.cfg.workspace_xy_limit_m,
        )
        terminated = failure.terminated
        self.failure_cause.copy_(failure.cause)
        time_out = (self.episode_length_buf >= self.max_episode_length) & ~terminated

        self.drone_terminal_observation.copy_(self._compute_policy_observation())
        self.terminal_goal_w.copy_(self._desired_pos_w)
        self.terminal_position_w.copy_(self._robot.data.root_pos_w)
        self.terminal_success.copy_(self.success_latched)
        self.terminal_success_count.copy_(self.success_count)
        self.terminal_failure_cause.copy_(self.failure_cause)
        self.terminal_distance_m.copy_(self._distance)
        self.terminal_speed_mps.copy_(self._speed)
        self.terminal_switch_count.copy_(self.switch_count)
        self.terminal_gust_count.copy_(self.gust_count)
        self.terminal_gust_event_applied_impulse_w.copy_(self.gust_event_applied_impulse_w)
        self.terminal_gust_event_expected_impulse_w.copy_(self.gust_event_expected_impulse_w)
        self.terminal_gust_event_measured_delta_momentum_w.copy_(
            self.gust_event_measured_delta_momentum_w
        )
        self.terminal_gust_event_stable_before.copy_(self.gust_event_stable_before)
        self.terminal_gust_event_recovered.copy_(self.gust_event_recovered)
        self.terminal_gust_event_recovery_latency_s.copy_(self.gust_event_recovery_latency_s)
        self.terminal_curriculum_stage_index.copy_(self.episode_curriculum_stage_index)
        self.terminal_curriculum_start_interactions.copy_(
            self.episode_curriculum_start_interactions
        )
        self.terminal_scenario_code.copy_(self.episode_scenario_code)
        self.terminal_mask.copy_(terminated | time_out)
        done = terminated | time_out
        required_successes = torch.where(
            self._scenario_mask("waypoint_switch"),
            torch.full_like(self.success_count, 4),
            torch.ones_like(self.success_count),
        )
        episode_succeeded = self.success_count >= required_successes
        for stage_index in range(len(self._training_curriculum)):
            stage_done = done & (self.episode_curriculum_stage_index == stage_index)
            self.curriculum_episode_completions[stage_index] += torch.count_nonzero(stage_done)
            self.curriculum_failure_terminations[stage_index] += torch.count_nonzero(
                stage_done & terminated
            )
            self.curriculum_successful_episodes[stage_index] += torch.count_nonzero(
                stage_done & episode_succeeded
            )
        self.extras["drone_terminal_observation"] = self.drone_terminal_observation
        self.extras["crazyflie_terminal_observation"] = self.crazyflie_terminal_observation
        self.extras["flyg1_terminal_observation"] = self.flyg1_terminal_observation
        self.extras["terminal_goal_w"] = self.terminal_goal_w
        self.extras["terminal_position_w"] = self.terminal_position_w
        self.extras["terminal_success"] = self.terminal_success
        self.extras["terminal_success_count"] = self.terminal_success_count
        self.extras["terminal_failure_cause"] = self.terminal_failure_cause
        self.extras["terminal_distance_m"] = self.terminal_distance_m
        self.extras["terminal_speed_mps"] = self.terminal_speed_mps
        self.extras["terminal_switch_count"] = self.terminal_switch_count
        self.extras["terminal_gust_count"] = self.terminal_gust_count
        self.extras["terminal_gust_event_applied_impulse_w"] = (
            self.terminal_gust_event_applied_impulse_w
        )
        self.extras["terminal_gust_event_expected_impulse_w"] = (
            self.terminal_gust_event_expected_impulse_w
        )
        self.extras["terminal_gust_event_measured_delta_momentum_w"] = (
            self.terminal_gust_event_measured_delta_momentum_w
        )
        self.extras["terminal_gust_event_stable_before"] = self.terminal_gust_event_stable_before
        self.extras["terminal_gust_event_recovered"] = self.terminal_gust_event_recovered
        self.extras["terminal_gust_event_recovery_latency_s"] = (
            self.terminal_gust_event_recovery_latency_s
        )
        self.extras["terminal_mechanical_work_proxy"] = self.terminal_mechanical_work_proxy
        self.extras["terminal_curriculum_stage_index"] = self.terminal_curriculum_stage_index
        self.extras["terminal_curriculum_start_interactions"] = (
            self.terminal_curriculum_start_interactions
        )
        self.extras["episode_scenario_code"] = self.episode_scenario_code
        self.extras["terminal_scenario_code"] = self.terminal_scenario_code
        self.extras["mixed_scenario_episode_starts"] = self.mixed_scenario_episode_starts
        self.extras["mixed_scenario_names_by_code"] = MIXED_SCENARIO_NAMES
        self.extras["switch_target_curriculum"] = (
            balanced_v4_switch_target_curriculum_payload()
            if self._balanced_v4
            else balanced_switch_target_curriculum_payload()
            if self._balanced_v3
            else switch_target_curriculum_payload()
        )
        self.extras["curriculum_episode_completions"] = self.curriculum_episode_completions
        self.extras["curriculum_failure_terminations"] = self.curriculum_failure_terminations
        self.extras["curriculum_successful_episodes"] = self.curriculum_successful_episodes
        self.extras["terminal_mask"] = self.terminal_mask
        # One vector step has now produced ``num_envs`` real interactions.
        # Advance before DirectRLEnv auto-resets terminal rows so a reset at an
        # exact stage boundary immediately uses the new distribution.  The
        # trainer's pre-rollout setter subsequently supplies the same value.
        self.set_training_interactions(
            advance_vector_training_interactions(self._training_interactions, self.num_envs)
        )
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._balanced_v4:
            self.reward_components = balanced_v4_interval_reward_terms(
                self._reward_previous_distance,
                self._distance,
                self._speed,
                self._new_success_this_step,
                self.success_latched,
                self._new_authenticated_gust_recovery_this_step,
                self._actions,
                self._action_delta,
                self.reset_terminated,
                self._robot.data.root_pos_w,
                self._terrain.env_origins,
                progress_potential_scale_m=(
                    self.cfg.balanced_progress_potential_scale_m
                ),
                progress_scale=self.cfg.balanced_progress_reward_scale,
                proximity_scale=self.cfg.proximity_reward_scale,
                proximity_distance_scale_m=self.cfg.proximity_distance_scale_m,
                dwell_reward_per_interval=self.cfg.dwell_reward_per_interval,
                braking_scale=self.cfg.braking_penalty_scale,
                braking_speed_reference_m_s=self.cfg.braking_speed_reference_mps,
                braking_normalized_squared_cap=(
                    self.cfg.braking_normalized_squared_cap
                ),
                success_bonus=self.cfg.success_bonus,
                retention_scale=self.cfg.retention_penalty_scale,
                retention_ramp_m=self.cfg.retention_ramp_m,
                survival_reward_per_interval=self.cfg.survival_reward_per_interval,
                control_effort_scale=self.cfg.control_effort_penalty_scale,
                collective_hover_action=self.cfg.collective_hover_action,
                collective_effort_reference=self.cfg.collective_effort_reference,
                moment_effort_reference=self.cfg.moment_effort_reference,
                control_effort_normalized_squared_cap=(
                    self.cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=self.cfg.action_change_penalty_scale,
                collective_action_change_reference=(
                    self.cfg.collective_action_change_reference
                ),
                moment_action_change_reference=(
                    self.cfg.moment_action_change_reference
                ),
                action_change_normalized_squared_cap=(
                    self.cfg.action_change_normalized_squared_cap
                ),
                boundary_scale=self.cfg.boundary_penalty_scale,
                boundary_low_onset_m=self.cfg.boundary_low_onset_m,
                boundary_low_width_m=self.cfg.boundary_low_width_m,
                boundary_high_onset_m=self.cfg.boundary_high_onset_m,
                boundary_high_width_m=self.cfg.boundary_high_width_m,
                boundary_xy_onset_m=self.cfg.boundary_xy_onset_m,
                boundary_xy_width_m=self.cfg.boundary_xy_width_m,
                boundary_normalized_squared_cap=(
                    self.cfg.boundary_normalized_squared_cap
                ),
                failure_penalty=self.cfg.failure_penalty,
                success_distance_m=self.cfg.success_distance_m,
                success_speed_m_s=self.cfg.success_speed_mps,
                gust_recovery_bonus=self.cfg.gust_recovery_bonus,
            )
        elif self._balanced_v3:
            self.reward_components = balanced_interval_reward_terms(
                self._reward_previous_distance,
                self._distance,
                self._speed,
                self._new_success_this_step,
                self.success_latched,
                self._actions,
                self._action_delta,
                self.reset_terminated,
                self._robot.data.root_pos_w,
                self._terrain.env_origins,
                progress_potential_scale_m=(
                    self.cfg.balanced_progress_potential_scale_m
                ),
                progress_scale=self.cfg.balanced_progress_reward_scale,
                proximity_scale=self.cfg.proximity_reward_scale,
                proximity_distance_scale_m=self.cfg.proximity_distance_scale_m,
                dwell_reward_per_interval=self.cfg.dwell_reward_per_interval,
                braking_scale=self.cfg.braking_penalty_scale,
                braking_speed_reference_m_s=self.cfg.braking_speed_reference_mps,
                braking_normalized_squared_cap=(
                    self.cfg.braking_normalized_squared_cap
                ),
                success_bonus=self.cfg.success_bonus,
                retention_scale=self.cfg.retention_penalty_scale,
                retention_ramp_m=self.cfg.retention_ramp_m,
                survival_reward_per_interval=self.cfg.survival_reward_per_interval,
                control_effort_scale=self.cfg.control_effort_penalty_scale,
                collective_hover_action=self.cfg.collective_hover_action,
                collective_effort_reference=self.cfg.collective_effort_reference,
                moment_effort_reference=self.cfg.moment_effort_reference,
                control_effort_normalized_squared_cap=(
                    self.cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=self.cfg.action_change_penalty_scale,
                collective_action_change_reference=(
                    self.cfg.collective_action_change_reference
                ),
                moment_action_change_reference=(
                    self.cfg.moment_action_change_reference
                ),
                action_change_normalized_squared_cap=(
                    self.cfg.action_change_normalized_squared_cap
                ),
                boundary_scale=self.cfg.boundary_penalty_scale,
                boundary_low_onset_m=self.cfg.boundary_low_onset_m,
                boundary_low_width_m=self.cfg.boundary_low_width_m,
                boundary_high_onset_m=self.cfg.boundary_high_onset_m,
                boundary_high_width_m=self.cfg.boundary_high_width_m,
                boundary_xy_onset_m=self.cfg.boundary_xy_onset_m,
                boundary_xy_width_m=self.cfg.boundary_xy_width_m,
                boundary_normalized_squared_cap=(
                    self.cfg.boundary_normalized_squared_cap
                ),
                failure_penalty=self.cfg.failure_penalty,
                success_distance_m=self.cfg.success_distance_m,
                success_speed_m_s=self.cfg.success_speed_mps,
            )
        else:
            self.reward_components = interval_reward_terms(
                self._distance_progress,
                self._new_success_this_step,
                self._actions,
                self._action_delta,
                self.reset_terminated,
                progress_scale=self.cfg.progress_reward_scale,
                progress_clip_m=self.cfg.progress_reward_clip_m,
                success_bonus=self.cfg.success_bonus,
                survival_reward_per_interval=self.cfg.survival_reward_per_interval,
                control_effort_scale=self.cfg.control_effort_penalty_scale,
                collective_hover_action=self.cfg.collective_hover_action,
                moment_action_reference=self.cfg.moment_action_reference,
                control_effort_normalized_squared_cap=(
                    self.cfg.control_effort_normalized_squared_cap
                ),
                action_change_scale=self.cfg.action_change_penalty_scale,
                action_change_normalized_squared_cap=(
                    self.cfg.action_change_normalized_squared_cap
                ),
                failure_penalty=self.cfg.failure_penalty,
            )
        total = self.reward_components["total"]

        force_power = torch.abs(
            torch.sum(self.aggregate_force_b * self._robot.data.root_lin_vel_b, dim=-1)
        )
        moment_power = torch.abs(
            torch.sum(self.aggregate_moment_b * self._robot.data.root_ang_vel_b, dim=-1)
        )
        self.mechanical_work_proxy_step.copy_(
            torch.nan_to_num((force_power + moment_power) * self.step_dt, nan=0.0, posinf=0.0, neginf=0.0)
        )
        self.mechanical_work_proxy += self.mechanical_work_proxy_step
        self.terminal_mechanical_work_proxy.copy_(self.mechanical_work_proxy)

        for key, value in self.reward_components.items():
            self._episode_sums[key] += value
        self._episode_sums["mechanical_work_proxy"] += self.mechanical_work_proxy_step
        self.extras["reward_components"] = self.reward_components
        return total

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        if env_ids is None:
            ids = self._all_env_ids
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return

        final_distance = torch.linalg.vector_norm(
            self._desired_pos_w[ids] - self._robot.data.root_pos_w[ids], dim=-1
        )
        log = {
            f"Episode_Reward/{key}": torch.mean(values[ids]).item()
            for key, values in self._episode_sums.items()
        }
        log.update(
            {
                "Episode_Termination/failure": torch.count_nonzero(self.reset_terminated[ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[ids]).item(),
                "Metrics/final_distance_to_goal": torch.nan_to_num(final_distance, nan=1.0e6).mean().item(),
                "Metrics/successes": self.success_count[ids].sum().item(),
                "Metrics/gusts": self.gust_count[ids].sum().item(),
                "Metrics/gust_recoveries": self.gust_recovered_count[ids].sum().item(),
            }
        )
        if self.scenario == "mixed":
            for code, name in enumerate(MIXED_SCENARIO_NAMES):
                log[f"Mixed/completed_{name}"] = torch.count_nonzero(
                    (self.episode_scenario_code[ids] == code)
                    & (self.reset_terminated[ids] | self.reset_time_outs[ids])
                ).item()
        self.extras["log"] = log

        self._robot.reset(ids)
        DirectRLEnv._reset_idx(self, ids)
        # Unlike the native task, never randomize this buffer on a full reset.
        self.episode_length_buf[ids] = 0

        new_stage_index = self._stage_index_for_new_episode()
        self.episode_curriculum_stage_index[ids] = new_stage_index
        self.episode_curriculum_start_interactions[ids] = self._training_interactions
        if self.scenario == "mixed":
            assigned = mixed_scenario_codes(
                ids,
                self.mixed_episode_index[ids],
                seed=self.cfg.mixed_scenario_seed,
            )
            self.episode_scenario_code[ids] = assigned
            self.mixed_episode_index[ids] += 1
            self.mixed_scenario_episode_starts += torch.bincount(
                assigned, minlength=len(MIXED_SCENARIO_NAMES)
            )

        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self._action_delta[ids] = 0.0
        root_state = self._make_initial_root_state(ids)
        joint_pos = self._robot.data.default_joint_pos[ids]
        joint_vel = self._robot.data.default_joint_vel[ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)

        targets = self._make_target_sequence(ids, root_state[:, :3])
        self._target_sequence_w[ids] = targets
        self._desired_pos_w[ids] = targets[:, 0, :]
        self._make_gust_directions(ids)

        initial_distance = torch.linalg.vector_norm(
            self._desired_pos_w[ids] - root_state[:, :3], dim=-1
        )
        self._previous_distance[ids] = initial_distance
        self._reward_previous_distance[ids] = initial_distance
        self._distance[ids] = initial_distance
        self._distance_progress[ids] = 0.0
        self._speed[ids] = torch.linalg.vector_norm(root_state[:, 7:10], dim=-1)
        self._target_elapsed_steps[ids] = 0
        self.switch_count[ids] = 0
        self.last_switch_step[ids] = -1
        self._switched_this_step[ids] = False
        self.success_dwell[ids] = 0
        self.success_latched[ids] = False
        self.success_count[ids] = 0
        self._new_success_this_step[ids] = False
        self.failure_cause[ids] = FAILURE_NONE

        self.gust_force_w[ids] = 0.0
        self.gust_direction_w[ids] = 0.0
        self.gust_count[ids] = 0
        self.gust_applied_impulse_w[ids] = 0.0
        self.gust_measured_delta_momentum_w[ids] = 0.0
        self.gust_stable_before[ids] = False
        self.gust_event_applied_impulse_w[ids] = 0.0
        self.gust_event_measured_delta_momentum_w[ids] = 0.0
        self.gust_event_stable_before[ids] = False
        self.gust_event_recovered[ids] = False
        self.gust_event_recovery_latency_s[ids] = float("nan")
        self.gust_recovery_active[ids] = False
        self.gust_recovery_dwell[ids] = 0
        self.gust_recovered_count[ids] = 0
        self._new_authenticated_gust_recovery_this_step[ids] = False
        self._current_recovery_event[ids] = -1
        self._active_gust_event[ids] = -1
        self._gust_ending_this_step[ids] = False
        self._gust_start_velocity_w[ids] = 0.0
        self._gust_recovery_start_step[ids] = 0
        self._gust_recovery_deadline_step[ids] = 0

        self.mechanical_work_proxy_step[ids] = 0.0
        self.mechanical_work_proxy[ids] = 0.0
        for value in self._episode_sums.values():
            value[ids] = 0.0

    def _make_initial_root_state(self, ids: torch.Tensor) -> torch.Tensor:
        if self._planned_initial_root_state_w is not None:
            root_state = self._planned_initial_root_state_w[ids].clone()
        else:
            stage = self._curriculum_stage_for_ids(ids)
            root_state = self._robot.data.default_root_state[ids].clone()
            root_state[:, :3] += self._terrain.env_origins[ids]
            count = ids.numel()
            root_state[:, :2] += self._uniform(
                (count, 2),
                -stage.spawn_position_xy_half_range_m,
                stage.spawn_position_xy_half_range_m,
            )
            root_state[:, 2] = self._terrain.env_origins[ids, 2] + stage.spawn_height_m
            root_state[:, 2] += self._uniform(
                (count,),
                -stage.spawn_position_z_half_range_m,
                stage.spawn_position_z_half_range_m,
            )
            yaw = self._uniform(
                (count,), -stage.spawn_yaw_half_range_rad, stage.spawn_yaw_half_range_rad
            )
            zeros = torch.zeros_like(yaw)
            root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
            root_state[:, 7:10] = self._uniform(
                (count, 3),
                -stage.spawn_linear_velocity_half_range_mps,
                stage.spawn_linear_velocity_half_range_mps,
            )
            root_state[:, 10:13] = self._uniform(
                (count, 3),
                -stage.spawn_angular_velocity_half_range_radps,
                stage.spawn_angular_velocity_half_range_radps,
            )
        if not torch.isfinite(root_state).all():
            raise ValueError("episode plan produced a nonfinite initial root state")
        return root_state

    def _make_target_sequence(self, ids: torch.Tensor, initial_position_w: torch.Tensor) -> torch.Tensor:
        if self._planned_targets_w is not None:
            planned = self._planned_targets_w[ids]
            if planned.shape[1] == 1:
                targets = planned.expand(-1, 4, -1).clone()
            else:
                targets = planned[:, :4, :].clone()
            self._validate_target_sequence(ids, initial_position_w, targets)
            return targets

        # Target zero follows the active reset curriculum exactly.  Survival-
        # v2 retains its declared full-distribution Switch followups because
        # its vertical-lift volume is too small for four separations.  Every
        # balanced-v3 volume is feasible, so it keeps all four targets within
        # the active stage instead of exposing the full task early.
        targets = torch.zeros(ids.numel(), 4, 3, device=self.device)
        first = self._sample_separated_targets(ids, initial_position_w)
        targets[:, 0, :] = first
        targets[:, 1:, :] = first[:, None, :]
        switch_rows = self._scenario_mask("waypoint_switch")[ids]
        switch_ids = ids[switch_rows]
        reference = first[switch_rows]
        for target_index in range(1, 4):
            if switch_ids.numel() == 0:
                break
            target = self._sample_separated_targets(
                switch_ids,
                reference,
                use_full_distribution=not self._balanced_task,
            )
            targets[switch_rows, target_index, :] = target
            reference = target
        return targets

    def _sample_separated_targets(
        self,
        ids: torch.Tensor,
        reference_w: torch.Tensor,
        *,
        use_full_distribution: bool = False,
    ) -> torch.Tensor:
        count = ids.numel()
        origins = self._terrain.env_origins[ids]
        stage = (
            self._training_curriculum[-1]
            if use_full_distribution
            else self._curriculum_stage_for_ids(ids)
        )
        target = torch.empty(count, 3, device=self.device)
        valid = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(64):
            candidate = torch.empty_like(target)
            candidate[:, :2] = self._uniform(
                (count, 2), stage.goal_xy_min_m, stage.goal_xy_max_m
            )
            candidate[:, :2] += origins[:, :2]
            candidate[:, 2] = self._uniform(
                (count,), stage.goal_z_min_m, stage.goal_z_max_m
            )
            candidate_valid = (
                torch.linalg.vector_norm(candidate - reference_w, dim=-1)
                >= stage.minimum_goal_separation_m
            )
            take = ~valid & candidate_valid
            target[take] = candidate[take]
            valid |= candidate_valid
            if bool(torch.all(valid)):
                return target

        # Deterministic far-corner fallback makes the separation guarantee
        # explicit even under an adversarial/randomly mocked generator.
        missing = ~valid
        ref_local_x = reference_w[:, 0] - origins[:, 0]
        ref_local_y = reference_w[:, 1] - origins[:, 1]
        target[missing, 0] = origins[missing, 0] + torch.where(
            ref_local_x[missing] <= 0,
            torch.full_like(ref_local_x[missing], stage.goal_xy_max_m),
            torch.full_like(ref_local_x[missing], stage.goal_xy_min_m),
        )
        target[missing, 1] = origins[missing, 1] + torch.where(
            ref_local_y[missing] <= 0,
            torch.full_like(ref_local_y[missing], stage.goal_xy_max_m),
            torch.full_like(ref_local_y[missing], stage.goal_xy_min_m),
        )
        target[missing, 2] = stage.goal_z_max_m
        distance = torch.linalg.vector_norm(target - reference_w, dim=-1)
        if torch.any(distance < stage.minimum_goal_separation_m):
            raise RuntimeError("Could not sample a target satisfying minimum separation")
        return target

    def _validate_target_sequence(
        self, ids: torch.Tensor, initial_position_w: torch.Tensor, targets: torch.Tensor
    ) -> None:
        if not torch.isfinite(targets).all():
            raise ValueError("planned targets must be finite")
        local_xy = targets[..., :2] - self._terrain.env_origins[ids, None, :2]
        within_xy = (local_xy >= self.cfg.goal_xy_min_m) & (local_xy <= self.cfg.goal_xy_max_m)
        within_z = (targets[..., 2] >= self.cfg.goal_z_min_m) & (
            targets[..., 2] <= self.cfg.goal_z_max_m
        )
        if not bool(torch.all(within_xy)) or not bool(torch.all(within_z)):
            raise ValueError("planned target lies outside the validated native goal workspace")
        target_count = 4 if self.scenario in {"waypoint_switch", "mixed"} else 1
        checked_targets = targets[:, :target_count, :]
        references = torch.cat(
            (initial_position_w[:, None, :], checked_targets[:, :-1, :]), dim=1
        )
        if torch.any(
            torch.linalg.vector_norm(checked_targets - references, dim=-1)
            < self.cfg.minimum_goal_separation_m
        ):
            raise ValueError("planned targets violate minimum start/consecutive target separation")

    def _make_gust_directions(self, ids: torch.Tensor) -> None:
        if self._planned_gust_directions_w is not None:
            self._gust_directions_w[ids] = self._planned_gust_directions_w[ids]
            return
        angles = self._uniform((ids.numel(), 3), 0.0, 2.0 * torch.pi)
        self._gust_directions_w[ids, :, 0] = torch.cos(angles)
        self._gust_directions_w[ids, :, 1] = torch.sin(angles)
        self._gust_directions_w[ids, :, 2] = 0.0

    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(low, high)


__all__ = [
    "CrazyflieEnv",
    "FAILURE_CAUSE_NAMES",
    "FAILURE_HIGH_HEIGHT",
    "FAILURE_LOW_HEIGHT",
    "FAILURE_NONE",
    "FAILURE_NONFINITE",
    "FAILURE_WORKSPACE_ESCAPE",
]
