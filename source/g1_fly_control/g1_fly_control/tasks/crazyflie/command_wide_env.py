"""Command-v2 Crazyflie environment with optional deterministic physical wind."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from .command_env import CommandFollowEnv
from .command_wide_env_cfg import CommandFollowWideEnvCfg
from .command_wide_logic import (
    COMMAND_CATEGORIES,
    COMMAND_CURRICULUM,
    COMMAND_WIDE_CONTRACT_VERSION,
    COMMAND_WIDE_SCHEDULE_STATE_KIND,
    COMMAND_WIDE_STILL_CONTRACT_SHA256,
    COMMAND_WIDE_WIND_CONTRACT_SHA256,
    CONTROL_DT_S,
    EPISODE_STEPS,
    HARD_MAXIMUM_HEIGHT_M,
    HARD_MINIMUM_HEIGHT_M,
    HARD_WORKSPACE_XY_M,
    HELD_OUT_EPISODES,
    HELD_OUT_EPISODE_STEPS,
    HELD_OUT_WIND_SEED,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HEIGHT_M,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_WIND_CALM_STEPS,
    MAXIMUM_WIND_PULSE_STEPS,
    MAXIMUM_YAW_RATE_RAD_S,
    MINIMUM_COMMAND_HOLD_STEPS,
    MINIMUM_HEIGHT_M,
    SOFT_WORKSPACE_XY_M,
    TRAINING_WIND_SEED,
    WIND_CATEGORIES,
    WIND_CURRICULUM,
    WIND_EVALUATION_PROTOCOL_SHA256,
    WIND_REFERENCE_ARM_M,
    apply_wide_command_safety_envelope,
    bound_wide_command_body,
    command_curriculum_stage_index,
    command_wide_reward_terms,
    command_wide_training_contract_payload,
    held_out_wind_at_step,
    sample_training_wind,
    sample_wide_scheduled_command,
    wind_curriculum_stage_index,
)


_COMMAND_TO_CODE = {name: index for index, name in enumerate(COMMAND_CATEGORIES)}
_WIND_TO_CODE = {name: index for index, name in enumerate(WIND_CATEGORIES)}
_WIND_MODE_TRAINING = "training"
_WIND_MODE_HELD_OUT = "heldout"


class CommandFollowWideEnv(CommandFollowEnv):
    """Track wide commands, optionally under a real world-frame wind wrench.

    The policy interface is unchanged: 12 observations and four native
    aggregate-wrench actions.  Wind never enters the observation directly.
    It is submitted to the Crazyflie's permanent wrench composer at its center
    of mass with ``is_global=True`` and is composed with the native body-frame
    motor wrench immediately before Isaac Lab writes the physics buffers.
    """

    cfg: CommandFollowWideEnvCfg

    def __init__(
        self,
        cfg: CommandFollowWideEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self._wind_mode = _WIND_MODE_TRAINING
        self._wind_active_seed = int(cfg.wind_schedule_seed)
        self._next_wind_segment_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._wind_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._wind_force_ratio_world = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._wind_torque_ratio_world = torch.zeros_like(
            self._wind_force_ratio_world
        )
        self._applied_wind_force_world = torch.zeros_like(
            self._wind_force_ratio_world
        )
        self._applied_wind_torque_world = torch.zeros_like(
            self._wind_force_ratio_world
        )
        self._wind_category_code = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._wind_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._heldout_episode_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._heldout_step_cursor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_applied_wind_force_world = torch.zeros_like(
            self._applied_wind_force_world
        )
        self.terminal_applied_wind_torque_world = torch.zeros_like(
            self._applied_wind_torque_world
        )
        self.terminal_wind_category_code = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_wind_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _validate_cfg(cfg: CommandFollowWideEnvCfg) -> None:
        if not isinstance(cfg.wind_enabled, bool):
            raise ValueError("wind_enabled must be bool")
        expected_sha = (
            COMMAND_WIDE_WIND_CONTRACT_SHA256
            if cfg.wind_enabled
            else COMMAND_WIDE_STILL_CONTRACT_SHA256
        )
        if cfg.command_tracking_contract_version != COMMAND_WIDE_CONTRACT_VERSION:
            raise ValueError("command-v2 contract version changed")
        if cfg.command_tracking_contract_sha256 != expected_sha:
            raise ValueError("command-v2 contract SHA-256 changed")
        if cfg.observation_space != 12 or cfg.action_space != 4:
            raise ValueError("command-v2 requires 12 observations and four actions")
        if cfg.decimation != 2 or not math.isclose(float(cfg.sim.dt), 0.01):
            raise ValueError("command-v2 requires native 50 Hz control")
        if not math.isclose(float(cfg.episode_length_s), 12.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("command-v2 episodes must contain exactly 600 control steps")
        if cfg.command_hold_min_steps != MINIMUM_COMMAND_HOLD_STEPS:
            raise ValueError("minimum command hold steps differ from command-v2")
        if cfg.command_hold_max_steps != MAXIMUM_COMMAND_HOLD_STEPS:
            raise ValueError("maximum command hold steps differ from command-v2")
        exact_float_fields = {
            "maximum_horizontal_speed_m_s": MAXIMUM_HORIZONTAL_SPEED_M_S,
            "maximum_vertical_speed_m_s": MAXIMUM_VERTICAL_SPEED_M_S,
            "maximum_yaw_rate_rad_s": MAXIMUM_YAW_RATE_RAD_S,
            "minimum_height_m": MINIMUM_HEIGHT_M,
            "maximum_height_m": MAXIMUM_HEIGHT_M,
            "workspace_xy_limit_m": SOFT_WORKSPACE_XY_M,
            "hard_minimum_height_m": HARD_MINIMUM_HEIGHT_M,
            "hard_maximum_height_m": HARD_MAXIMUM_HEIGHT_M,
            "hard_workspace_xy_limit_m": HARD_WORKSPACE_XY_M,
            "wind_reference_arm_m": WIND_REFERENCE_ARM_M,
        }
        for name, expected in exact_float_fields.items():
            if not math.isclose(
                float(getattr(cfg, name)), expected, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"{name} differs from command-v2")
        for name in ("command_schedule_seed", "wind_schedule_seed", "held_out_wind_seed"):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if cfg.wind_schedule_seed != TRAINING_WIND_SEED:
            raise ValueError("training wind seed differs from the paired contract")
        if cfg.held_out_wind_seed != HELD_OUT_WIND_SEED:
            raise ValueError("held-out wind seed differs from the paired protocol")
        if cfg.wind_schedule_seed == cfg.held_out_wind_seed:
            raise ValueError("training and held-out wind seeds must be distinct")

    @property
    def command_tracking_contract(self) -> dict[str, Any]:
        return command_wide_training_contract_payload(
            wind_enabled=bool(self.cfg.wind_enabled)
        )

    @property
    def active_training_curriculum_stage_index(self) -> int:
        return command_curriculum_stage_index(self._training_interactions)

    @property
    def active_training_curriculum_stage_payload(self) -> dict[str, int | float]:
        return COMMAND_CURRICULUM[
            self.active_training_curriculum_stage_index
        ].payload()

    @property
    def full_training_curriculum_stage_payload(self) -> dict[str, int | float]:
        return COMMAND_CURRICULUM[-1].payload()

    @property
    def active_wind_curriculum_stage_index(self) -> int:
        return wind_curriculum_stage_index(self._training_interactions)

    @property
    def active_wind_curriculum_stage_payload(self) -> dict[str, int | float]:
        return WIND_CURRICULUM[self.active_wind_curriculum_stage_index].payload()

    @property
    def applied_wind_force_world(self) -> torch.Tensor:
        """Finite physical force submitted to Isaac, shape ``[num_envs, 3]``."""

        return self._applied_wind_force_world

    @property
    def applied_wind_torque_world(self) -> torch.Tensor:
        """Finite physical torque submitted to Isaac, shape ``[num_envs, 3]``."""

        return self._applied_wind_torque_world

    @property
    def wind_mode(self) -> str:
        return self._wind_mode

    @property
    def _schedule_independent_of_episode_resets(self) -> bool:
        """Advance command-v2 schedules on the global control clock."""

        return True

    def _refresh_effective_command(self) -> None:
        finite = torch.isfinite(
            torch.cat(
                (
                    self._requested_command_body,
                    self._robot.data.root_pos_w,
                    self._robot.data.root_quat_w,
                    self._command_target_position_w,
                    self._workspace_origin_w,
                ),
                dim=1,
            )
        ).all(dim=1)
        self._effective_command_body.zero_()
        if bool(finite.any()):
            self._effective_command_body[finite] = apply_wide_command_safety_envelope(
                self._requested_command_body[finite],
                self._robot.data.root_pos_w[finite],
                self._robot.data.root_quat_w[finite],
                self._command_target_position_w[finite],
                self._workspace_origin_w[finite],
                minimum_height_m=self.cfg.minimum_height_m,
                maximum_height_m=self.cfg.maximum_height_m,
                maximum_horizontal_offset_m=self.cfg.workspace_xy_limit_m,
            )

    def set_manual_command_body(
        self, command_body: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        value = torch.as_tensor(
            command_body,
            dtype=self._requested_command_body.dtype,
            device=self.device,
        )
        if value.shape != (self.num_envs, 4):
            raise ValueError(f"manual command must have shape [{self.num_envs}, 4]")
        bounded = bound_wide_command_body(value)
        if not torch.allclose(value, bounded, rtol=0.0, atol=1.0e-6):
            raise ValueError("manual command exceeds the command-v2 speed envelope")
        self._manual_command_mode.fill_(True)
        self._requested_command_body.copy_(value)
        self._refresh_effective_command()
        return {"policy": self._compute_policy_observation()}

    def _sample_scheduled_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        rows: list[tuple[float, float, float, float]] = []
        holds: list[int] = []
        categories: list[int] = []
        stages: list[int] = []
        for env_id, segment_index in zip(
            env_ids.detach().cpu().tolist(),
            self._next_command_segment_index[env_ids].detach().cpu().tolist(),
            strict=True,
        ):
            segment = sample_wide_scheduled_command(
                seed=int(self.cfg.command_schedule_seed),
                environment_id=int(env_id),
                segment_index=int(segment_index),
                total_interactions=self._training_interactions,
            )
            rows.append(segment.command)
            holds.append(segment.hold_steps)
            categories.append(_COMMAND_TO_CODE[segment.category])
            stages.append(segment.stage_index)
        self._requested_command_body[env_ids] = torch.tensor(
            rows, dtype=self._requested_command_body.dtype, device=self.device
        )
        self._command_steps_remaining[env_ids] = torch.tensor(
            holds, dtype=torch.long, device=self.device
        )
        self._command_category_code[env_ids] = torch.tensor(
            categories, dtype=torch.long, device=self.device
        )
        self._command_stage_index[env_ids] = torch.tensor(
            stages, dtype=torch.long, device=self.device
        )
        self._next_command_segment_index[env_ids] += 1

    def _set_wind_rows(
        self,
        env_ids: torch.Tensor,
        force_ratios: list[tuple[float, float, float]],
        torque_ratios: list[tuple[float, float, float]],
        holds: list[int],
        categories: list[int],
        stages: list[int],
    ) -> None:
        self._wind_force_ratio_world[env_ids] = torch.tensor(
            force_ratios,
            dtype=self._wind_force_ratio_world.dtype,
            device=self.device,
        )
        self._wind_torque_ratio_world[env_ids] = torch.tensor(
            torque_ratios,
            dtype=self._wind_torque_ratio_world.dtype,
            device=self.device,
        )
        self._wind_steps_remaining[env_ids] = torch.tensor(
            holds, dtype=torch.long, device=self.device
        )
        self._wind_category_code[env_ids] = torch.tensor(
            categories, dtype=torch.long, device=self.device
        )
        self._wind_stage_index[env_ids] = torch.tensor(
            stages, dtype=torch.long, device=self.device
        )
        self._refresh_applied_wind(env_ids)

    def _refresh_applied_wind(self, env_ids: torch.Tensor | None = None) -> None:
        ids = self._all_env_ids if env_ids is None else env_ids
        if not self.cfg.wind_enabled:
            self._wind_force_ratio_world[ids] = 0.0
            self._wind_torque_ratio_world[ids] = 0.0
        self._applied_wind_force_world[ids] = (
            self._wind_force_ratio_world[ids] * float(self._robot_weight)
        )
        self._applied_wind_torque_world[ids] = (
            self._wind_torque_ratio_world[ids]
            * float(self._robot_weight)
            * float(self.cfg.wind_reference_arm_m)
        )
        if not bool(torch.isfinite(self._applied_wind_force_world[ids]).all()):
            raise FloatingPointError("scheduled wind force is nonfinite")
        if not bool(torch.isfinite(self._applied_wind_torque_world[ids]).all()):
            raise FloatingPointError("scheduled wind torque is nonfinite")

    def _set_calm_wind(self, env_ids: torch.Tensor) -> None:
        self._wind_force_ratio_world[env_ids] = 0.0
        self._wind_torque_ratio_world[env_ids] = 0.0
        self._wind_steps_remaining[env_ids] = 0
        self._wind_category_code[env_ids] = _WIND_TO_CODE["calm"]
        self._wind_stage_index[env_ids] = wind_curriculum_stage_index(
            self._training_interactions
        )
        self._refresh_applied_wind(env_ids)

    def _sample_training_winds(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if not self.cfg.wind_enabled:
            self._set_calm_wind(env_ids)
            return
        force_ratios: list[tuple[float, float, float]] = []
        torque_ratios: list[tuple[float, float, float]] = []
        holds: list[int] = []
        categories: list[int] = []
        stages: list[int] = []
        for env_id, segment_index in zip(
            env_ids.detach().cpu().tolist(),
            self._next_wind_segment_index[env_ids].detach().cpu().tolist(),
            strict=True,
        ):
            segment = sample_training_wind(
                seed=int(self.cfg.wind_schedule_seed),
                environment_id=int(env_id),
                segment_index=int(segment_index),
                total_interactions=self._training_interactions,
            )
            force_ratios.append(segment.force_ratio_world)
            torque_ratios.append(segment.torque_ratio_world)
            holds.append(segment.hold_steps)
            categories.append(_WIND_TO_CODE[segment.category])
            stages.append(segment.stage_index)
        self._set_wind_rows(
            env_ids, force_ratios, torque_ratios, holds, categories, stages
        )
        self._next_wind_segment_index[env_ids] += 1

    def _load_heldout_wind(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        force_ratios: list[tuple[float, float, float]] = []
        torque_ratios: list[tuple[float, float, float]] = []
        holds: list[int] = []
        categories: list[int] = []
        stages: list[int] = []
        for episode_index, step in zip(
            self._heldout_episode_index[env_ids].detach().cpu().tolist(),
            self._heldout_step_cursor[env_ids].detach().cpu().tolist(),
            strict=True,
        ):
            segment = held_out_wind_at_step(
                int(episode_index), int(step), seed=self._wind_active_seed
            )
            force_ratios.append(segment.force_ratio_world)
            torque_ratios.append(segment.torque_ratio_world)
            holds.append(segment.hold_steps)
            categories.append(_WIND_TO_CODE[segment.category])
            stages.append(segment.stage_index)
        self._set_wind_rows(
            env_ids, force_ratios, torque_ratios, holds, categories, stages
        )

    def set_wind_evaluation_mode(
        self,
        episode_indices: Sequence[int] | torch.Tensor,
        *,
        seed: int = HELD_OUT_WIND_SEED,
    ) -> None:
        """Select the deterministic 16-episode held-out physical-wind protocol."""

        if not self.cfg.wind_enabled:
            raise RuntimeError("held-out wind mode requires the wind-enabled task")
        if seed != int(self.cfg.held_out_wind_seed):
            raise ValueError("held-out wind seed differs from the declared protocol")
        values = torch.as_tensor(
            episode_indices, dtype=torch.long, device=self.device
        )
        if values.shape != (self.num_envs,):
            raise ValueError(
                f"episode_indices must have shape [{self.num_envs}]"
            )
        if bool((values < 0).any()) or bool((values >= HELD_OUT_EPISODES).any()):
            raise ValueError("held-out episode indices must be in [0, 15]")
        self._wind_mode = _WIND_MODE_HELD_OUT
        self._wind_active_seed = int(seed)
        self._heldout_episode_index.copy_(values)
        self._heldout_step_cursor.zero_()
        self._load_heldout_wind(self._all_env_ids)

    def clear_wind_evaluation_mode(self) -> None:
        """Return to the deterministic seeded training-wind stream."""

        self._wind_mode = _WIND_MODE_TRAINING
        self._wind_active_seed = int(self.cfg.wind_schedule_seed)
        self._heldout_episode_index.zero_()
        self._heldout_step_cursor.zero_()
        self._set_calm_wind(self._all_env_ids)

    def _advance_wind_for_next_interval(self) -> None:
        continuing = ~(self.reset_terminated | self.reset_time_outs)
        if self._wind_mode == _WIND_MODE_HELD_OUT:
            self._heldout_step_cursor[continuing] += 1
            valid = continuing & (
                self._heldout_step_cursor < HELD_OUT_EPISODE_STEPS
            )
            if bool(valid.any()):
                self._load_heldout_wind(self._all_env_ids[valid])
            return
        if not self.cfg.wind_enabled:
            self._set_calm_wind(self._all_env_ids)
            return
        # Training wind is paired across controllers.  Every environment row
        # therefore advances once per global control interval, including rows
        # which terminate at that boundary.  Episode resets must not consume
        # an extra segment or pause this clock.
        self._wind_steps_remaining -= 1
        due = self._wind_steps_remaining <= 0
        if bool(due.any()):
            self._sample_training_winds(self._all_env_ids[due])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._refresh_applied_wind()

    def _apply_action(self) -> None:
        # This call converts the declared world-frame wind into the current
        # link frame.  The native body-frame thrust/moment is then composed
        # into the same buffer before DirectRLEnv calls scene.write_data_to_sim.
        # Resetting also invalidates WrenchComposer's link-pose cache, so the
        # world-to-body transform is refreshed after every physics substep.
        self._robot.permanent_wrench_composer.reset()
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._applied_wind_force_world.unsqueeze(1),
            torques=self._applied_wind_torque_world.unsqueeze(1),
            body_ids=self._body_id,
            is_global=True,
        )
        self._robot.permanent_wrench_composer.add_forces_and_torques(
            forces=self._thrust,
            torques=self._moment,
            body_ids=self._body_id,
            is_global=False,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        done = terminated | time_out
        self.terminal_applied_wind_force_world.copy_(
            self._applied_wind_force_world
        )
        self.terminal_applied_wind_torque_world.copy_(
            self._applied_wind_torque_world
        )
        self.terminal_wind_category_code.copy_(self._wind_category_code)
        self.terminal_wind_stage_index.copy_(self._wind_stage_index)
        self.extras.update(
            {
                "command_v2_contract_sha256": self.cfg.command_tracking_contract_sha256,
                "wind_evaluation_protocol_sha256": WIND_EVALUATION_PROTOCOL_SHA256,
                "wind_enabled": bool(self.cfg.wind_enabled),
                "wind_mode": self._wind_mode,
                "applied_wind_force_world": self._applied_wind_force_world,
                "applied_wind_torque_world": self._applied_wind_torque_world,
                "wind_force_ratio_world": self._wind_force_ratio_world,
                "wind_torque_ratio_world": self._wind_torque_ratio_world,
                "wind_steps_remaining": self._wind_steps_remaining,
                "wind_category_code": self._wind_category_code,
                "wind_curriculum_stage_index": self._wind_stage_index,
                "heldout_wind_episode_index": self._heldout_episode_index,
                "heldout_wind_step_cursor": self._heldout_step_cursor,
                "terminal_applied_wind_force_world": self.terminal_applied_wind_force_world,
                "terminal_applied_wind_torque_world": self.terminal_applied_wind_torque_world,
                "terminal_wind_category_code": self.terminal_wind_category_code,
                "terminal_wind_stage_index": self.terminal_wind_stage_index,
                "terminal_wind_mask": done,
            }
        )
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        target_error_b = self._target_error_body()
        self.reward_components = command_wide_reward_terms(
            self._tracking_error_body,
            self._previous_tracking_error_body,
            self.linear_acceleration_body,
            self.linear_jerk_body,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            target_error_b,
            self._actions,
            self._previous_actions,
            self.reset_terminated,
            step_dt_s=CONTROL_DT_S,
        )
        for name, value in self.reward_components.items():
            self._episode_sums[name] += value
        self._episode_linear_error_sum += torch.linalg.vector_norm(
            self._tracking_error_body[:, :3], dim=1
        )
        self._episode_yaw_error_sum += torch.abs(self._tracking_error_body[:, 3])
        self._episode_tracking_samples += 1
        self.extras["reward_components"] = self.reward_components
        self._advance_command_for_next_interval()
        self._advance_wind_for_next_interval()
        self._previous_linear_velocity_b.copy_(self._robot.data.root_lin_vel_b)
        self._previous_linear_acceleration_body.copy_(
            self.linear_acceleration_body
        )
        return self.reward_components["total"]

    def _zero_physical_wrench_on_reset(self, env_ids: torch.Tensor) -> None:
        zeros = torch.zeros(
            env_ids.numel(), 1, 3, device=self.device, dtype=self._thrust.dtype
        )
        self._robot.permanent_wrench_composer.reset(env_ids)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=self._body_id,
            env_ids=env_ids,
            is_global=True,
        )

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        if env_ids is None:
            ids = self._all_env_ids
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        super()._reset_idx(ids)
        self._zero_physical_wrench_on_reset(ids)
        if self._wind_mode == _WIND_MODE_HELD_OUT:
            self._heldout_step_cursor[ids] = 0
            self._load_heldout_wind(ids)
        elif self.cfg.wind_enabled:
            # Initialize once, then retain the shared global wind cursor across
            # episode resets.  The physical composer was zeroed above and the
            # preserved scheduled ratios are re-applied for the next interval.
            uninitialized = ids[self._next_wind_segment_index[ids] == 0]
            if uninitialized.numel():
                self._sample_training_winds(uninitialized)
            self._refresh_applied_wind(ids)
        else:
            self._set_calm_wind(ids)

    def command_schedule_state_dict(self) -> dict[str, Any]:
        """Serialize every command and wind cursor needed for clean resume."""

        return {
            "schema_version": 2,
            "kind": COMMAND_WIDE_SCHEDULE_STATE_KIND,
            "contract_sha256": self.cfg.command_tracking_contract_sha256,
            "num_envs": self.num_envs,
            "wind_enabled": bool(self.cfg.wind_enabled),
            "command_schedule_seed": int(self.cfg.command_schedule_seed),
            "wind_schedule_seed": int(self.cfg.wind_schedule_seed),
            "held_out_wind_seed": int(self.cfg.held_out_wind_seed),
            "wind_active_seed": self._wind_active_seed,
            "wind_mode": self._wind_mode,
            "training_interactions": self._training_interactions,
            "next_command_segment_index": self._next_command_segment_index.detach().cpu().tolist(),
            "command_steps_remaining": self._command_steps_remaining.detach().cpu().tolist(),
            "requested_command_body": self._requested_command_body.detach().cpu().tolist(),
            "command_category_code": self._command_category_code.detach().cpu().tolist(),
            "command_stage_index": self._command_stage_index.detach().cpu().tolist(),
            "next_wind_segment_index": self._next_wind_segment_index.detach().cpu().tolist(),
            "wind_steps_remaining": self._wind_steps_remaining.detach().cpu().tolist(),
            "wind_force_ratio_world": self._wind_force_ratio_world.detach().cpu().tolist(),
            "wind_torque_ratio_world": self._wind_torque_ratio_world.detach().cpu().tolist(),
            "applied_wind_force_world": self._applied_wind_force_world.detach().cpu().tolist(),
            "applied_wind_torque_world": self._applied_wind_torque_world.detach().cpu().tolist(),
            "wind_category_code": self._wind_category_code.detach().cpu().tolist(),
            "wind_stage_index": self._wind_stage_index.detach().cpu().tolist(),
            "heldout_episode_index": self._heldout_episode_index.detach().cpu().tolist(),
            "heldout_step_cursor": self._heldout_step_cursor.detach().cpu().tolist(),
            "wind_evaluation_protocol_sha256": WIND_EVALUATION_PROTOCOL_SHA256,
        }

    def load_command_schedule_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate and restore a command-v2 clean-boundary schedule state."""

        required = {
            "schema_version",
            "kind",
            "contract_sha256",
            "num_envs",
            "wind_enabled",
            "command_schedule_seed",
            "wind_schedule_seed",
            "held_out_wind_seed",
            "wind_active_seed",
            "wind_mode",
            "training_interactions",
            "next_command_segment_index",
            "command_steps_remaining",
            "requested_command_body",
            "command_category_code",
            "command_stage_index",
            "next_wind_segment_index",
            "wind_steps_remaining",
            "wind_force_ratio_world",
            "wind_torque_ratio_world",
            "applied_wind_force_world",
            "applied_wind_torque_world",
            "wind_category_code",
            "wind_stage_index",
            "heldout_episode_index",
            "heldout_step_cursor",
            "wind_evaluation_protocol_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("command-v2 schedule state fields changed")
        scalar_expected = {
            "schema_version": 2,
            "kind": COMMAND_WIDE_SCHEDULE_STATE_KIND,
            "contract_sha256": self.cfg.command_tracking_contract_sha256,
            "num_envs": self.num_envs,
            "wind_enabled": bool(self.cfg.wind_enabled),
            "command_schedule_seed": int(self.cfg.command_schedule_seed),
            "wind_schedule_seed": int(self.cfg.wind_schedule_seed),
            "held_out_wind_seed": int(self.cfg.held_out_wind_seed),
            "wind_evaluation_protocol_sha256": WIND_EVALUATION_PROTOCOL_SHA256,
        }
        for name, expected in scalar_expected.items():
            if state[name] != expected:
                raise ValueError(f"command-v2 schedule state {name} differs")
        mode = state["wind_mode"]
        if mode not in (_WIND_MODE_TRAINING, _WIND_MODE_HELD_OUT):
            raise ValueError("command-v2 schedule state has invalid wind mode")
        if mode == _WIND_MODE_HELD_OUT and not self.cfg.wind_enabled:
            raise ValueError("still-air state cannot use held-out wind mode")
        expected_active_seed = (
            int(self.cfg.held_out_wind_seed)
            if mode == _WIND_MODE_HELD_OUT
            else int(self.cfg.wind_schedule_seed)
        )
        if state["wind_active_seed"] != expected_active_seed:
            raise ValueError("command-v2 schedule state active wind seed differs")
        interactions = state["training_interactions"]
        if not isinstance(interactions, int) or isinstance(interactions, bool) or interactions < 0:
            raise ValueError("command-v2 state has invalid interactions")

        vector_names = (
            "next_command_segment_index",
            "command_steps_remaining",
            "command_category_code",
            "command_stage_index",
            "next_wind_segment_index",
            "wind_steps_remaining",
            "wind_category_code",
            "wind_stage_index",
            "heldout_episode_index",
            "heldout_step_cursor",
        )
        vectors = {
            name: torch.as_tensor(state[name], dtype=torch.long, device=self.device)
            for name in vector_names
        }
        if any(value.shape != (self.num_envs,) for value in vectors.values()):
            raise ValueError("command-v2 schedule vector shape differs")
        matrix_names = (
            "wind_force_ratio_world",
            "wind_torque_ratio_world",
            "applied_wind_force_world",
            "applied_wind_torque_world",
        )
        matrices = {
            name: torch.as_tensor(
                state[name], dtype=self._wind_force_ratio_world.dtype, device=self.device
            )
            for name in matrix_names
        }
        requested = torch.as_tensor(
            state["requested_command_body"],
            dtype=self._requested_command_body.dtype,
            device=self.device,
        )
        if requested.shape != (self.num_envs, 4):
            raise ValueError("command-v2 requested command shape differs")
        if any(value.shape != (self.num_envs, 3) for value in matrices.values()):
            raise ValueError("command-v2 wind matrix shape differs")
        if not bool(torch.isfinite(requested).all()) or not all(
            bool(torch.isfinite(value).all()) for value in matrices.values()
        ):
            raise ValueError("command-v2 schedule state contains nonfinite values")
        if not torch.allclose(
            requested, bound_wide_command_body(requested), rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("command-v2 state contains an out-of-range command")

        next_command = vectors["next_command_segment_index"]
        command_remaining = vectors["command_steps_remaining"]
        command_categories = vectors["command_category_code"]
        command_stages = vectors["command_stage_index"]
        command_uninitialized = next_command == 0
        command_is_uninitialized = bool(command_uninitialized.all())
        if (
            bool((next_command < 0).any())
            or bool((command_remaining < 0).any())
            or bool((command_remaining > MAXIMUM_COMMAND_HOLD_STEPS).any())
            or bool((command_categories < 0).any())
            or bool((command_categories >= len(COMMAND_CATEGORIES)).any())
            or bool((command_stages < 0).any())
            or bool((command_stages >= len(COMMAND_CURRICULUM)).any())
            or (bool(command_uninitialized.any()) and not command_is_uninitialized)
        ):
            raise ValueError("command-v2 state contains an invalid command cursor")
        if command_is_uninitialized:
            if (
                interactions != 0
                or bool(torch.count_nonzero(command_remaining))
                or bool(torch.count_nonzero(command_categories))
                or bool(torch.count_nonzero(command_stages))
                or bool(torch.count_nonzero(requested))
            ):
                raise ValueError("command-v2 uninitialized command cursor is not pristine")
        else:
            if bool((command_remaining < 1).any()):
                raise ValueError("command-v2 state contains an expired command cursor")
            for env_id in range(self.num_envs):
                stage_index = int(command_stages[env_id])
                stage_start = COMMAND_CURRICULUM[stage_index].start_interactions
                if stage_start > interactions:
                    raise ValueError("command-v2 command stage is ahead of its clock")
                expected = sample_wide_scheduled_command(
                    seed=int(self.cfg.command_schedule_seed),
                    environment_id=env_id,
                    segment_index=int(next_command[env_id]) - 1,
                    total_interactions=stage_start,
                )
                if (
                    expected.stage_index != stage_index
                    or _COMMAND_TO_CODE[expected.category] != int(command_categories[env_id])
                    or int(command_remaining[env_id]) > expected.hold_steps
                    or not torch.allclose(
                        requested[env_id], requested.new_tensor(expected.command), rtol=0.0, atol=1.0e-6
                    )
                ):
                    raise ValueError("command-v2 command cursor is not deterministic")

        next_wind = vectors["next_wind_segment_index"]
        wind_remaining = vectors["wind_steps_remaining"]
        wind_categories = vectors["wind_category_code"]
        wind_stages = vectors["wind_stage_index"]
        heldout_episodes = vectors["heldout_episode_index"]
        heldout_steps = vectors["heldout_step_cursor"]
        force_ratios = matrices["wind_force_ratio_world"]
        torque_ratios = matrices["wind_torque_ratio_world"]
        applied_force = matrices["applied_wind_force_world"]
        applied_torque = matrices["applied_wind_torque_world"]
        wind_uninitialized = next_wind == 0
        wind_is_uninitialized = bool(wind_uninitialized.all())
        if (
            bool((next_wind < 0).any())
            or bool((wind_remaining < 0).any())
            or bool((wind_categories < 0).any())
            or bool((wind_categories >= len(WIND_CATEGORIES)).any())
            or bool((wind_stages < 0).any())
            or bool((wind_stages >= len(WIND_CURRICULUM)).any())
            or bool((heldout_episodes < 0).any())
            or bool((heldout_episodes >= HELD_OUT_EPISODES).any())
            or bool((heldout_steps < 0).any())
            or bool((heldout_steps >= HELD_OUT_EPISODE_STEPS).any())
            or (
                self.cfg.wind_enabled
                and mode == _WIND_MODE_TRAINING
                and bool(wind_uninitialized.any())
                and not wind_is_uninitialized
            )
        ):
            raise ValueError("command-v2 state contains an invalid wind cursor")
        if (
            self.cfg.wind_enabled
            and mode == _WIND_MODE_TRAINING
            and command_is_uninitialized != wind_is_uninitialized
        ):
            raise ValueError("command and wind initialization boundaries differ")
        expected_force = force_ratios * float(self._robot_weight)
        expected_torque = (
            torque_ratios
            * float(self._robot_weight)
            * float(self.cfg.wind_reference_arm_m)
        )
        if not torch.allclose(applied_force, expected_force, rtol=0.0, atol=1.0e-7):
            raise ValueError("command-v2 physical force disagrees with its ratio")
        if not torch.allclose(applied_torque, expected_torque, rtol=0.0, atol=1.0e-9):
            raise ValueError("command-v2 physical torque disagrees with its ratio")

        for env_id in range(self.num_envs):
            if not self.cfg.wind_enabled:
                expected_wind = None
                if (
                    int(next_wind[env_id]) != 0
                    or int(wind_remaining[env_id]) != 0
                    or int(wind_categories[env_id]) != _WIND_TO_CODE["calm"]
                    or bool(torch.count_nonzero(force_ratios[env_id]))
                    or bool(torch.count_nonzero(torque_ratios[env_id]))
                ):
                    raise ValueError("still-air state contains a nonzero wind cursor")
                continue
            if mode == _WIND_MODE_HELD_OUT:
                expected_wind = held_out_wind_at_step(
                    int(heldout_episodes[env_id]),
                    int(heldout_steps[env_id]),
                    seed=expected_active_seed,
                )
                if int(wind_remaining[env_id]) != expected_wind.hold_steps:
                    raise ValueError("held-out wind remaining steps differ")
            else:
                if int(next_wind[env_id]) == 0:
                    if (
                        not command_is_uninitialized
                        or interactions != 0
                        or int(wind_remaining[env_id]) != 0
                        or int(wind_categories[env_id]) != _WIND_TO_CODE["calm"]
                        or int(wind_stages[env_id]) != 0
                        or bool(torch.count_nonzero(force_ratios[env_id]))
                        or bool(torch.count_nonzero(torque_ratios[env_id]))
                    ):
                        raise ValueError("training wind has an invalid uninitialized cursor")
                    continue
                if int(wind_remaining[env_id]) < 1:
                    raise ValueError("training wind state has an expired segment")
                stage_index = int(wind_stages[env_id])
                stage_start = WIND_CURRICULUM[stage_index].start_interactions
                if stage_start > interactions:
                    raise ValueError("wind stage is ahead of its interaction clock")
                expected_wind = sample_training_wind(
                    seed=int(self.cfg.wind_schedule_seed),
                    environment_id=env_id,
                    segment_index=int(next_wind[env_id]) - 1,
                    total_interactions=stage_start,
                )
                if int(wind_remaining[env_id]) > expected_wind.hold_steps:
                    raise ValueError("training wind remaining steps differ")
            if (
                expected_wind.stage_index != int(wind_stages[env_id])
                or _WIND_TO_CODE[expected_wind.category] != int(wind_categories[env_id])
                or not torch.allclose(
                    force_ratios[env_id],
                    force_ratios.new_tensor(expected_wind.force_ratio_world),
                    rtol=0.0,
                    atol=1.0e-7,
                )
                or not torch.allclose(
                    torque_ratios[env_id],
                    torque_ratios.new_tensor(expected_wind.torque_ratio_world),
                    rtol=0.0,
                    atol=1.0e-7,
                )
            ):
                raise ValueError("command-v2 wind cursor is not deterministic")

        self._training_interactions = interactions
        self._wind_mode = mode
        self._wind_active_seed = expected_active_seed
        self._next_command_segment_index.copy_(next_command)
        self._command_steps_remaining.copy_(command_remaining)
        self._requested_command_body.copy_(requested)
        self._command_category_code.copy_(command_categories)
        self._command_stage_index.copy_(command_stages)
        self._next_wind_segment_index.copy_(next_wind)
        self._wind_steps_remaining.copy_(wind_remaining)
        self._wind_force_ratio_world.copy_(force_ratios)
        self._wind_torque_ratio_world.copy_(torque_ratios)
        self._applied_wind_force_world.copy_(applied_force)
        self._applied_wind_torque_world.copy_(applied_torque)
        self._wind_category_code.copy_(wind_categories)
        self._wind_stage_index.copy_(wind_stages)
        self._heldout_episode_index.copy_(heldout_episodes)
        self._heldout_step_cursor.copy_(heldout_steps)
        self._manual_command_mode.zero_()
        self._command_target_position_w.copy_(self._robot.data.root_pos_w)
        self._refresh_effective_command()


__all__ = ["CommandFollowWideEnv"]
