"""Single-task Crazyflie body-velocity/yaw command-following environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv

from .command_env_cfg import CommandFollowEnvCfg
from .command_logic import (
    COMMAND_CURRICULUM,
    COMMAND_TRACKING_CONTRACT_SHA256,
    COMMAND_TRACKING_CONTRACT_VERSION,
    CONTROL_DT_S,
    FAILURE_CAUSE_NAMES,
    FAILURE_NONE,
    HARD_MAXIMUM_HEIGHT_M,
    HARD_MINIMUM_HEIGHT_M,
    HARD_WORKSPACE_XY_M,
    MAXIMUM_COMMAND_HOLD_STEPS,
    MAXIMUM_HEIGHT_M,
    MAXIMUM_HORIZONTAL_SPEED_M_S,
    MAXIMUM_VERTICAL_SPEED_M_S,
    MAXIMUM_YAW_RATE_RAD_S,
    MINIMUM_COMMAND_HOLD_STEPS,
    MINIMUM_HEIGHT_M,
    SOFT_WORKSPACE_XY_M,
    apply_command_safety_envelope,
    bound_command_body,
    classify_command_failures,
    command_conditioned_observation,
    command_follow_reward_terms,
    command_tracking_contract_payload,
    curriculum_stage_index,
    integrate_command_target,
    sample_scheduled_command,
)


COMMAND_SCHEDULE_STATE_KIND = "flyg1.crazyflie.command-schedule.v1"
_COMMAND_CATEGORIES = ("hover", "cardinal", "diagonal", "full_simultaneous")
_CATEGORY_TO_CODE = {name: index for index, name in enumerate(_COMMAND_CATEGORIES)}


class CommandFollowEnv(QuadcopterEnv):
    """Track held body-frame linear/yaw commands using native wrench actions.

    The installed Crazyflie articulation, action mapping, physics step, and
    decimation remain owned by :class:`QuadcopterEnv`.  This class replaces
    only command generation, observation/reward/termination logic, and reset
    instrumentation.
    """

    cfg: CommandFollowEnvCfg

    def __init__(
        self,
        cfg: CommandFollowEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._validate_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self._all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._workspace_origin_w = self._terrain.env_origins.clone()
        self._requested_command_body = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._effective_command_body = torch.zeros_like(
            self._requested_command_body
        )
        self._tracking_error_body = torch.zeros_like(
            self._requested_command_body
        )
        self._previous_tracking_error_body = torch.zeros_like(
            self._tracking_error_body
        )
        self._previous_linear_velocity_b = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.linear_acceleration_body = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._previous_linear_acceleration_body = torch.zeros_like(
            self.linear_acceleration_body
        )
        self.linear_jerk_body = torch.zeros_like(self.linear_acceleration_body)
        self._command_target_position_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._action_delta = torch.zeros_like(self._actions)
        self._manual_command_mode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._training_interactions = 0
        self._next_command_segment_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._command_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._command_category_code = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._command_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.failure_cause = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.drone_terminal_observation = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self.crazyflie_terminal_observation = self.drone_terminal_observation
        self.flyg1_terminal_observation = self.drone_terminal_observation
        self.terminal_tracking_error_body = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.terminal_linear_acceleration_body = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_linear_jerk_body = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_failure_cause = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_position_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_command_target_position_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_command_category_code = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_command_stage_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_command_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        reward_names = (
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
        self.reward_components: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_names
        }
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_names
        }
        self._episode_linear_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_yaw_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_tracking_samples = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _validate_cfg(cfg: CommandFollowEnvCfg) -> None:
        if cfg.command_tracking_contract_version != COMMAND_TRACKING_CONTRACT_VERSION:
            raise ValueError("command tracking contract version changed")
        if cfg.command_tracking_contract_sha256 != COMMAND_TRACKING_CONTRACT_SHA256:
            raise ValueError("command tracking contract SHA-256 changed")
        if cfg.observation_space != 12 or cfg.action_space != 4:
            raise ValueError("command-following environment requires 12 observations and four actions")
        if cfg.decimation != 2 or not math.isclose(float(cfg.sim.dt), 0.01):
            raise ValueError("command-following environment requires native 50 Hz control")
        if not math.isclose(float(cfg.episode_length_s), 12.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("command-following episodes must contain exactly 600 control steps")
        if cfg.command_hold_min_steps != MINIMUM_COMMAND_HOLD_STEPS:
            raise ValueError("minimum command hold steps differ from the contract")
        if cfg.command_hold_max_steps != MAXIMUM_COMMAND_HOLD_STEPS:
            raise ValueError("maximum command hold steps differ from the contract")
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
        }
        for name, expected in exact_float_fields.items():
            if not math.isclose(
                float(getattr(cfg, name)), expected, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"{name} differs from the command contract")
        if (
            not isinstance(cfg.command_schedule_seed, int)
            or isinstance(cfg.command_schedule_seed, bool)
            or cfg.command_schedule_seed < 0
        ):
            raise ValueError("command schedule seed must be a non-negative integer")

    @property
    def command_tracking_contract(self) -> dict[str, Any]:
        return command_tracking_contract_payload()

    @property
    def training_interactions(self) -> int:
        return self._training_interactions

    @property
    def active_training_curriculum_stage_index(self) -> int:
        return curriculum_stage_index(self._training_interactions)

    @property
    def active_training_curriculum_stage_payload(self) -> dict[str, int | float]:
        return COMMAND_CURRICULUM[
            self.active_training_curriculum_stage_index
        ].payload()

    @property
    def full_training_curriculum_stage_payload(self) -> dict[str, int | float]:
        return COMMAND_CURRICULUM[-1].payload()

    @property
    def requested_command_body(self) -> torch.Tensor:
        return self._requested_command_body

    @property
    def effective_command_body(self) -> torch.Tensor:
        return self._effective_command_body

    @property
    def tracking_error_body(self) -> torch.Tensor:
        return self._tracking_error_body

    @property
    def command_target_position_w(self) -> torch.Tensor:
        return self._command_target_position_w

    def set_training_interactions(self, value: int) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("training interactions must be a non-negative integer")
        self._training_interactions = value

    def _target_error_body(self) -> torch.Tensor:
        target_error_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._command_target_position_w,
        )
        return target_error_b

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
            self._effective_command_body[finite] = apply_command_safety_envelope(
                self._requested_command_body[finite],
                self._robot.data.root_pos_w[finite],
                self._robot.data.root_quat_w[finite],
                self._command_target_position_w[finite],
                self._workspace_origin_w[finite],
                minimum_height_m=self.cfg.minimum_height_m,
                maximum_height_m=self.cfg.maximum_height_m,
                maximum_horizontal_offset_m=self.cfg.workspace_xy_limit_m,
            )

    def _update_tracking_error(self) -> None:
        self._tracking_error_body[:, :3] = (
            self._robot.data.root_lin_vel_b - self._effective_command_body[:, :3]
        )
        self._tracking_error_body[:, 3] = (
            self._robot.data.root_ang_vel_b[:, 2]
            - self._effective_command_body[:, 3]
        )

    def _compute_policy_observation(self) -> torch.Tensor:
        target_error_b = self._target_error_body()
        observation = command_conditioned_observation(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._effective_command_body,
            target_error_b,
        )
        self._tracking_error_body[:, :3].copy_(observation[:, :3])
        self._tracking_error_body[:, 3].copy_(observation[:, 5])
        return observation

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_effective_command()
        return {"policy": self._compute_policy_observation()}

    def set_manual_command_body(
        self, command_body: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Set a physical body command and return its fresh policy observation."""

        value = torch.as_tensor(
            command_body,
            dtype=self._requested_command_body.dtype,
            device=self.device,
        )
        if value.shape != (self.num_envs, 4):
            raise ValueError(
                f"manual command must have shape [{self.num_envs}, 4]"
            )
        bounded = bound_command_body(
            value,
            maximum_horizontal_speed_m_s=self.cfg.maximum_horizontal_speed_m_s,
            maximum_vertical_speed_m_s=self.cfg.maximum_vertical_speed_m_s,
            maximum_yaw_rate_rad_s=self.cfg.maximum_yaw_rate_rad_s,
        )
        if not torch.allclose(value, bounded, rtol=0.0, atol=1.0e-6):
            raise ValueError("manual command exceeds the trained speed envelope")
        self._manual_command_mode.fill_(True)
        self._requested_command_body.copy_(value)
        self._refresh_effective_command()
        return {"policy": self._compute_policy_observation()}

    def clear_manual_command(self) -> dict[str, torch.Tensor]:
        """Enter manual hover while retaining the current integrated hold target."""

        return self.set_manual_command_body(
            torch.zeros_like(self._requested_command_body)
        )

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
            segment = sample_scheduled_command(
                seed=int(self.cfg.command_schedule_seed),
                environment_id=int(env_id),
                segment_index=int(segment_index),
                total_interactions=self._training_interactions,
            )
            rows.append(segment.command)
            holds.append(segment.hold_steps)
            categories.append(_CATEGORY_TO_CODE[segment.category])
            stages.append(segment.stage_index)
        self._requested_command_body[env_ids] = torch.tensor(
            rows,
            dtype=self._requested_command_body.dtype,
            device=self.device,
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

    def command_schedule_state_dict(self) -> dict[str, Any]:
        """Return the complete semantic schedule cursor for clean checkpoints."""

        return {
            "schema_version": 1,
            "kind": COMMAND_SCHEDULE_STATE_KIND,
            "contract_sha256": COMMAND_TRACKING_CONTRACT_SHA256,
            "num_envs": self.num_envs,
            "command_schedule_seed": int(self.cfg.command_schedule_seed),
            "training_interactions": self._training_interactions,
            "next_command_segment_index": self._next_command_segment_index.detach().cpu().tolist(),
            "command_steps_remaining": self._command_steps_remaining.detach().cpu().tolist(),
            "requested_command_body": self._requested_command_body.detach().cpu().tolist(),
            "command_category_code": self._command_category_code.detach().cpu().tolist(),
            "command_stage_index": self._command_stage_index.detach().cpu().tolist(),
        }

    def load_command_schedule_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a validated schedule cursor after the simulator reset boundary."""

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
            raise ValueError("command schedule state fields changed")
        if state["schema_version"] != 1 or state["kind"] != COMMAND_SCHEDULE_STATE_KIND:
            raise ValueError("unsupported command schedule state")
        if state["contract_sha256"] != COMMAND_TRACKING_CONTRACT_SHA256:
            raise ValueError("command schedule state uses a different task contract")
        if state["num_envs"] != self.num_envs:
            raise ValueError("command schedule state environment count differs")
        if state["command_schedule_seed"] != int(self.cfg.command_schedule_seed):
            raise ValueError("command schedule state seed differs")
        interactions = state["training_interactions"]
        if (
            not isinstance(interactions, int)
            or isinstance(interactions, bool)
            or interactions < 0
        ):
            raise ValueError("command schedule state has invalid interactions")

        next_indices = torch.as_tensor(
            state["next_command_segment_index"], dtype=torch.long, device=self.device
        )
        remaining = torch.as_tensor(
            state["command_steps_remaining"], dtype=torch.long, device=self.device
        )
        requested = torch.as_tensor(
            state["requested_command_body"],
            dtype=self._requested_command_body.dtype,
            device=self.device,
        )
        categories = torch.as_tensor(
            state["command_category_code"], dtype=torch.long, device=self.device
        )
        stages = torch.as_tensor(
            state["command_stage_index"], dtype=torch.long, device=self.device
        )
        vector_shape = (self.num_envs,)
        if any(value.shape != vector_shape for value in (next_indices, remaining, categories, stages)):
            raise ValueError("command schedule vector width differs")
        if requested.shape != (self.num_envs, 4):
            raise ValueError("command schedule command shape differs")
        if (
            bool((next_indices < 0).any())
            or bool((remaining < 1).any())
            or bool((remaining > self.cfg.command_hold_max_steps).any())
            or bool((categories < 0).any())
            or bool((categories >= len(_COMMAND_CATEGORIES)).any())
            or bool((stages < 0).any())
            or bool((stages >= len(COMMAND_CURRICULUM)).any())
        ):
            raise ValueError("command schedule state contains an out-of-range cursor")
        bounded = bound_command_body(requested)
        if not torch.allclose(requested, bounded, rtol=0.0, atol=1.0e-6):
            raise ValueError("command schedule state contains an out-of-range command")
        for env_id in range(self.num_envs):
            if int(next_indices[env_id]) < 1:
                raise ValueError("command schedule state has no sampled segment")
            stage_index = int(stages[env_id])
            stage_start = COMMAND_CURRICULUM[stage_index].start_interactions
            if stage_start > interactions:
                raise ValueError("command schedule state stage is ahead of its interaction clock")
            expected = sample_scheduled_command(
                seed=int(self.cfg.command_schedule_seed),
                environment_id=env_id,
                segment_index=int(next_indices[env_id]) - 1,
                total_interactions=stage_start,
            )
            expected_command = requested.new_tensor(expected.command)
            if (
                expected.stage_index != stage_index
                or _CATEGORY_TO_CODE[expected.category] != int(categories[env_id])
                or int(remaining[env_id]) > expected.hold_steps
                or not torch.allclose(
                    requested[env_id], expected_command, rtol=0.0, atol=1.0e-6
                )
            ):
                raise ValueError("command schedule state disagrees with its deterministic sequence")
        self._training_interactions = interactions
        self._next_command_segment_index.copy_(next_indices)
        self._command_steps_remaining.copy_(remaining)
        self._requested_command_body.copy_(requested)
        self._command_category_code.copy_(categories)
        self._command_stage_index.copy_(stages)
        self._manual_command_mode.zero_()
        # A checkpoint resumes at a clean environment-reset boundary.  Keep
        # the restored command cursor, but establish a new reachable hold point.
        self._command_target_position_w.copy_(self._robot.data.root_pos_w)
        self._refresh_effective_command()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._previous_tracking_error_body.copy_(self._tracking_error_body)
        super()._pre_physics_step(actions)
        self._action_delta.copy_(self._actions - self._previous_actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_effective_command()
        observation = self._compute_policy_observation()
        self.linear_acceleration_body.copy_(
            (self._robot.data.root_lin_vel_b - self._previous_linear_velocity_b)
            / CONTROL_DT_S
        )
        self.linear_jerk_body.copy_(
            (
                self.linear_acceleration_body
                - self._previous_linear_acceleration_body
            )
            / CONTROL_DT_S
        )
        root_state = torch.cat(
            (
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self._robot.data.projected_gravity_b,
                self._actions,
                self._requested_command_body,
                self._effective_command_body,
                self._command_target_position_w,
            ),
            dim=1,
        )
        finite_state = torch.isfinite(root_state).all(dim=1)
        failure = classify_command_failures(
            self._robot.data.root_pos_w,
            self._workspace_origin_w,
            finite_state,
            minimum_height_m=self.cfg.hard_minimum_height_m,
            maximum_height_m=self.cfg.hard_maximum_height_m,
            workspace_xy_limit_m=self.cfg.hard_workspace_xy_limit_m,
        )
        terminated = failure.terminated
        time_out = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        done = terminated | time_out
        self.failure_cause.copy_(failure.cause)
        self.drone_terminal_observation.copy_(observation)
        self.terminal_tracking_error_body.copy_(self._tracking_error_body)
        self.terminal_linear_acceleration_body.copy_(
            self.linear_acceleration_body
        )
        self.terminal_linear_jerk_body.copy_(self.linear_jerk_body)
        self.terminal_failure_cause.copy_(failure.cause)
        self.terminal_position_w.copy_(self._robot.data.root_pos_w)
        self.terminal_command_target_position_w.copy_(
            self._command_target_position_w
        )
        self.terminal_mask.copy_(done)
        self.terminal_command_category_code.copy_(self._command_category_code)
        self.terminal_command_stage_index.copy_(self._command_stage_index)
        self.terminal_command_steps_remaining.copy_(
            self._command_steps_remaining
        )
        self._training_interactions += self.num_envs
        self.extras.update(
            {
                "drone_terminal_observation": self.drone_terminal_observation,
                "crazyflie_terminal_observation": self.crazyflie_terminal_observation,
                "flyg1_terminal_observation": self.flyg1_terminal_observation,
                "terminal_tracking_error_body": self.terminal_tracking_error_body,
                "terminal_linear_acceleration_body": self.terminal_linear_acceleration_body,
                "terminal_linear_jerk_body": self.terminal_linear_jerk_body,
                "terminal_failure_cause": self.terminal_failure_cause,
                "terminal_position_w": self.terminal_position_w,
                "terminal_command_target_position_w": self.terminal_command_target_position_w,
                "terminal_mask": self.terminal_mask,
                "terminal_command_category_code": self.terminal_command_category_code,
                "terminal_command_stage_index": self.terminal_command_stage_index,
                "terminal_command_steps_remaining": self.terminal_command_steps_remaining,
                "requested_command_body": self._requested_command_body,
                "effective_command_body": self._effective_command_body,
                "tracking_error_body": self._tracking_error_body,
                "linear_acceleration_body": self.linear_acceleration_body,
                "linear_jerk_body": self.linear_jerk_body,
                "command_steps_remaining": self._command_steps_remaining,
                "command_category_code": self._command_category_code,
                "command_curriculum_stage_index": self._command_stage_index,
            }
        )
        return terminated, time_out

    def _advance_command_for_next_interval(self) -> None:
        continuing = ~(self.reset_terminated | self.reset_time_outs)
        finite = torch.isfinite(
            torch.cat(
                (
                    self._command_target_position_w,
                    self._effective_command_body,
                    self._robot.data.root_quat_w,
                    self._workspace_origin_w,
                ),
                dim=1,
            )
        ).all(dim=1) & continuing
        if bool(finite.any()):
            self._command_target_position_w[finite] = integrate_command_target(
                self._command_target_position_w[finite],
                self._effective_command_body[finite],
                self._robot.data.root_quat_w[finite],
                self._workspace_origin_w[finite],
                dt_s=CONTROL_DT_S,
                minimum_height_m=self.cfg.minimum_height_m,
                maximum_height_m=self.cfg.maximum_height_m,
                maximum_horizontal_offset_m=self.cfg.workspace_xy_limit_m,
            )
        self._desired_pos_w.copy_(self._command_target_position_w)
        scheduled = ~self._manual_command_mode
        if not self._schedule_independent_of_episode_resets:
            scheduled &= continuing
        self._command_steps_remaining[scheduled] -= 1
        due = scheduled & (self._command_steps_remaining <= 0)
        if bool(due.any()):
            self._sample_scheduled_commands(self._all_env_ids[due])
        self._refresh_effective_command()

    @property
    def _schedule_independent_of_episode_resets(self) -> bool:
        """Whether every scheduled row advances on the shared global clock.

        The original command-v1 task retains its episode-local schedule.  The
        command-v2 subclass overrides this flag so paired still/wind runs see
        the same command at every global control interval even when their
        episode termination patterns differ.
        """

        return False

    def _get_rewards(self) -> torch.Tensor:
        target_error_b = self._target_error_body()
        self.reward_components = command_follow_reward_terms(
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
        self._previous_linear_velocity_b.copy_(self._robot.data.root_lin_vel_b)
        self._previous_linear_acceleration_body.copy_(
            self.linear_acceleration_body
        )
        return self.reward_components["total"]

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        if env_ids is None:
            ids = self._all_env_ids
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return

        samples = self._episode_tracking_samples[ids].clamp_min(1).to(torch.float32)
        log = {
            f"Episode_Reward/{name}": float(values[ids].mean())
            for name, values in self._episode_sums.items()
        }
        log.update(
            {
                "Episode_Termination/failure": int(
                    torch.count_nonzero(self.reset_terminated[ids]).item()
                ),
                "Episode_Termination/time_out": int(
                    torch.count_nonzero(self.reset_time_outs[ids]).item()
                ),
                "Metrics/linear_tracking_error_mean_m_s": float(
                    (self._episode_linear_error_sum[ids] / samples).mean()
                ),
                "Metrics/yaw_tracking_error_mean_rad_s": float(
                    (self._episode_yaw_error_sum[ids] / samples).mean()
                ),
            }
        )
        for code, name in FAILURE_CAUSE_NAMES.items():
            if code:
                log[f"Failure/{name}"] = int(
                    torch.count_nonzero(
                        self.terminal_failure_cause[ids] == code
                    ).item()
                )
        self.extras["log"] = log

        self._robot.reset(ids)
        DirectRLEnv._reset_idx(self, ids)
        self.episode_length_buf[ids] = 0

        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self._action_delta[ids] = 0.0
        root_state = self._robot.data.default_root_state[ids].clone()
        root_state[:, :3] = self._terrain.env_origins[ids]
        root_state[:, 2] += float(self.cfg.spawn_height_m)
        root_state[:, 7:] = 0.0
        joint_pos = self._robot.data.default_joint_pos[ids]
        joint_vel = self._robot.data.default_joint_vel[ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)

        self._workspace_origin_w[ids] = self._terrain.env_origins[ids]
        self._command_target_position_w[ids] = root_state[:, :3]
        self._desired_pos_w[ids] = root_state[:, :3]
        manual = self._manual_command_mode[ids]
        if bool(manual.any()):
            manual_ids = ids[manual]
            self._requested_command_body[manual_ids] = 0.0
            self._effective_command_body[manual_ids] = 0.0
        scheduled_ids = ids[~manual]
        if self._schedule_independent_of_episode_resets:
            # A global schedule is initialized exactly once.  Later episode
            # resets retain its current command/cursor; only the integrated
            # episode target above is re-anchored to the new spawn state.
            scheduled_ids = scheduled_ids[
                self._next_command_segment_index[scheduled_ids] == 0
            ]
        if scheduled_ids.numel():
            self._sample_scheduled_commands(scheduled_ids)
        self._refresh_effective_command()
        # The simulator data view is synchronized after reset returns.  Build
        # the known zero-velocity reset errors directly instead of reading a
        # possibly stale pre-reset terminal row.
        self._tracking_error_body[ids, :3] = -self._effective_command_body[ids, :3]
        self._tracking_error_body[ids, 3] = -self._effective_command_body[ids, 3]
        self._previous_tracking_error_body[ids] = self._tracking_error_body[ids]
        self._previous_linear_velocity_b[ids] = 0.0
        self.linear_acceleration_body[ids] = 0.0
        self._previous_linear_acceleration_body[ids] = 0.0
        self.linear_jerk_body[ids] = 0.0

        self.failure_cause[ids] = FAILURE_NONE
        for value in self._episode_sums.values():
            value[ids] = 0.0
        self._episode_linear_error_sum[ids] = 0.0
        self._episode_yaw_error_sum[ids] = 0.0
        self._episode_tracking_samples[ids] = 0


__all__ = ["COMMAND_SCHEDULE_STATE_KIND", "CommandFollowEnv"]
