"""Immutable physics-scale normalization for every Crazyflie controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from g1_fly_control.crazyflie.stabilization import (
    FIXED_OBSERVATION_CLIP,
    FIXED_OBSERVATION_SCALE,
    OBSERVATION_WIDTH,
)
from g1_fly_control.tasks.crazyflie.command_logic import COMMAND_FOLLOW_TASK_ID


COMMAND_TASK_ID = COMMAND_FOLLOW_TASK_ID
COMMAND_WIDTH = 4


@dataclass
class FixedPhysicsScaleNormalizer:
    """Serializable fixed scaling; ``update`` deliberately changes nothing."""

    scale: torch.Tensor
    clip: float = FIXED_OBSERVATION_CLIP

    @classmethod
    def create(
        cls, width: int, *, device: str | torch.device = "cpu"
    ) -> "FixedPhysicsScaleNormalizer":
        if width != OBSERVATION_WIDTH:
            raise ValueError(
                f"Crazyflie fixed normalization requires width {OBSERVATION_WIDTH}"
            )
        return cls(
            scale=torch.tensor(
                FIXED_OBSERVATION_SCALE, dtype=torch.float32, device=device
            )
        )

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != OBSERVATION_WIDTH:
            raise ValueError(f"Expected [batch, {OBSERVATION_WIDTH}] values")
        # Kept as a no-op so the existing drone-only environment adapter can
        # use one interface in training and evaluation.  No rollout can alter
        # normalization or contaminate it with crash states.

    def normalize(self, values: torch.Tensor, *, clip: float | None = None) -> torch.Tensor:
        if clip is not None and float(clip) != self.clip:
            raise ValueError(
                f"Fixed normalization clip is immutable at {self.clip}, got {clip}"
            )
        if values.ndim < 1 or values.shape[-1] != OBSERVATION_WIDTH:
            raise ValueError(
                f"Observation must end in exactly {OBSERVATION_WIDTH} values"
            )
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return (values / scale).clamp(-self.clip, self.clip)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1 or values.shape[-1] != OBSERVATION_WIDTH:
            raise ValueError(
                f"Observation must end in exactly {OBSERVATION_WIDTH} values"
            )
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return values * scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "fixed_physics_scale_v1",
            "scale": self.scale.detach().cpu(),
            "clip": self.clip,
            "update_rule": "immutable_no_updates",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = {"schema_version", "kind", "scale", "clip", "update_rule"}
        if set(state) != required:
            raise ValueError(
                "Fixed normalizer state fields changed: "
                f"expected {sorted(required)}, got {sorted(state)}"
            )
        if state["schema_version"] != 1 or state["kind"] != "fixed_physics_scale_v1":
            raise ValueError("Checkpoint uses an incompatible observation normalizer")
        if state["update_rule"] != "immutable_no_updates":
            raise ValueError("Checkpoint normalizer unexpectedly permits updates")
        if float(state["clip"]) != FIXED_OBSERVATION_CLIP:
            raise ValueError("Checkpoint fixed observation clip changed")
        value = torch.as_tensor(
            state["scale"], dtype=self.scale.dtype, device=self.scale.device
        )
        expected = torch.tensor(
            FIXED_OBSERVATION_SCALE,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )
        if value.shape != expected.shape or not torch.equal(value, expected):
            raise ValueError("Checkpoint fixed observation scales changed")
        self.scale.copy_(expected)


# Retain the established drone-script import name while making the changed
# fixed contract explicit in reports and type names.  No frozen G1 module uses
# this drone-only class.
RunningMeanVariance = FixedPhysicsScaleNormalizer


class NormalizedEnv:
    """Narrow adapter that normalizes policy observations for the shared PPO path."""

    def __init__(
        self,
        env: Any,
        normalizer: RunningMeanVariance,
        *,
        training: bool,
        task: str | None = None,
    ) -> None:
        self.env = env
        self.unwrapped = self
        self.normalizer = normalizer
        self.training = bool(training)
        if task is not None and (not isinstance(task, str) or not task):
            raise ValueError("task must be a non-empty string when provided")
        self.task = task
        self._command_tracking = (
            task == COMMAND_TASK_ID
            or getattr(env, "command_tracking_contract", None) is not None
        )
        num_envs = int(getattr(env, "num_envs", 0))
        device = getattr(env, "device", "cpu")
        self._episode_return = torch.zeros(num_envs, device=device)
        self._episode_reward_components: dict[str, torch.Tensor] = {}
        self._interval_completed_episodes = 0
        self._interval_episode_return_sum = 0.0
        self._interval_successful_episodes = 0
        self._interval_success_count = 0
        self._interval_failure_terminations = 0
        self._interval_time_limit_truncations = 0
        self._interval_failure_causes = {str(code): 0 for code in range(1, 5)}
        self._interval_reward_component_sums: dict[str, float] = {}
        self._interval_reward_component_step_sums: dict[str, float] = {}
        self._interval_reward_component_sample_counts: dict[str, int] = {}
        self._interval_final_distance_sum = 0.0
        self._interval_final_speed_sum = 0.0
        self._interval_command_sample_count = 0
        self._interval_command_linear_squared_error_sum = 0.0
        self._interval_command_yaw_squared_error_sum = 0.0
        self._interval_requested_command_sum = torch.zeros(COMMAND_WIDTH, device=device)
        self._interval_effective_command_sum = torch.zeros(COMMAND_WIDTH, device=device)
        self._interval_effective_command_squared_sum = torch.zeros(
            COMMAND_WIDTH, device=device
        )
        self._interval_effective_command_max_abs = torch.zeros(
            COMMAND_WIDTH, device=device
        )
        self._interval_command_projection_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @staticmethod
    def _tensor(observation: Any) -> torch.Tensor:
        if isinstance(observation, dict):
            observation = observation["policy"]
        if not isinstance(observation, torch.Tensor):
            raise TypeError("Crazyflie policy observation must be a tensor or policy dictionary")
        return observation

    def _convert(self, observation: Any) -> dict[str, torch.Tensor]:
        raw = self._tensor(observation)
        if self.training:
            self.normalizer.update(raw)
        return {"policy": self.normalizer.normalize(raw)}

    def _command_vector(self, name: str, *, device: torch.device) -> torch.Tensor:
        value = getattr(self.env, name, None)
        if value is None:
            raise RuntimeError(
                f"{COMMAND_TASK_ID} lacks required command telemetry {name}"
            )
        tensor = torch.as_tensor(value, device=device).detach()
        expected = (self._episode_return.numel(), COMMAND_WIDTH)
        if tensor.shape != expected:
            raise RuntimeError(
                f"{COMMAND_TASK_ID} telemetry {name} must have shape {expected}"
            )
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(
                f"{COMMAND_TASK_ID} telemetry {name} must be finite floating point"
            )
        return tensor

    def _accumulate_command_tracking(self, done: torch.Tensor) -> None:
        contract = getattr(self.env, "command_tracking_contract", None)
        if not isinstance(contract, Mapping) or not contract:
            raise RuntimeError(
                f"{COMMAND_TASK_ID} lacks a non-empty command_tracking_contract"
            )
        device = self._episode_return.device
        requested = self._command_vector("requested_command_body", device=device)
        effective = self._command_vector("effective_command_body", device=device)
        tracking_error = self._command_vector("tracking_error_body", device=device)
        if bool(done.any()):
            terminal_error = self._command_vector(
                "terminal_tracking_error_body", device=device
            )
            tracking_error = tracking_error.clone()
            tracking_error[done] = terminal_error[done]

        self._interval_command_sample_count += int(tracking_error.shape[0])
        self._interval_command_linear_squared_error_sum += float(
            tracking_error[:, :3].square().sum().item()
        )
        self._interval_command_yaw_squared_error_sum += float(
            tracking_error[:, 3].square().sum().item()
        )
        self._interval_requested_command_sum += requested.sum(dim=0)
        self._interval_effective_command_sum += effective.sum(dim=0)
        self._interval_effective_command_squared_sum += effective.square().sum(dim=0)
        self._interval_effective_command_max_abs = torch.maximum(
            self._interval_effective_command_max_abs,
            effective.abs().amax(dim=0),
        )
        projected = (requested - effective).abs().amax(dim=1) > 1.0e-7
        self._interval_command_projection_count += int(projected.sum().item())

    def reset(self, *args: Any, **kwargs: Any):
        observation, extras = self.env.reset(*args, **kwargs)
        return self._convert(observation), extras

    def step(self, action: torch.Tensor):
        observation, reward, terminated, truncated, extras = self.env.step(action)
        if self.training:
            self._accumulate_training_metrics(reward, terminated, truncated)
        terminal = getattr(self.env, "drone_terminal_observation", None)
        if terminal is None:
            terminal = getattr(self.env, "flyg1_terminal_observation", None)
        if terminal is not None:
            normalized_terminal = self.normalizer.normalize(terminal)
            self.drone_terminal_observation = normalized_terminal
            self.flyg1_terminal_observation = normalized_terminal
        return self._convert(observation), reward, terminated, truncated, extras

    @torch.no_grad()
    def _accumulate_training_metrics(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Retain completed-episode measurements before the next update.

        DirectRLEnv auto-resets rows inside ``step``.  The environment's
        terminal tensors are snapshots taken before that reset, so consuming
        them here prevents failures and successes from being lost or counted
        repeatedly as a current-state gauge.
        """

        self._episode_return += reward.detach()
        components = getattr(self.env, "reward_components", {})
        if isinstance(components, dict):
            for name, value in components.items():
                tensor = torch.as_tensor(value, device=self._episode_return.device).detach()
                if tensor.shape != self._episode_return.shape:
                    raise RuntimeError(
                        f"reward component {name!r} must have shape "
                        f"{tuple(self._episode_return.shape)}"
                    )
                if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(
                        f"reward component {name!r} must be finite floating point"
                    )
                accumulator = self._episode_reward_components.setdefault(
                    str(name), torch.zeros_like(self._episode_return)
                )
                accumulator += tensor
                component_name = str(name)
                self._interval_reward_component_step_sums[component_name] = (
                    self._interval_reward_component_step_sums.get(component_name, 0.0)
                    + float(tensor.sum().item())
                )
                self._interval_reward_component_sample_counts[component_name] = (
                    self._interval_reward_component_sample_counts.get(component_name, 0)
                    + int(tensor.numel())
                )
        done = (terminated | truncated).bool()
        if self._command_tracking:
            self._accumulate_command_tracking(done)
        if not bool(done.any()):
            return
        count = int(done.sum().item())
        self._interval_completed_episodes += count
        self._interval_episode_return_sum += float(self._episode_return[done].sum().item())
        if not self._command_tracking:
            terminal_success_count = torch.as_tensor(
                getattr(self.env, "terminal_success_count"), device=done.device
            )[done]
            terminal_scenario_code = torch.as_tensor(
                getattr(self.env, "terminal_scenario_code"), device=done.device
            )[done].to(torch.long)
            if terminal_scenario_code.shape != terminal_success_count.shape:
                raise RuntimeError(
                    "terminal scenario codes must align with terminal success counts"
                )
            if bool(((terminal_scenario_code < 0) | (terminal_scenario_code > 2)).any()):
                raise RuntimeError("terminal scenario codes must be Reach=0, Switch=1, or Gust=2")
            # A Switch episode succeeds only after target 0 and all three
            # switched targets have completed their dwell. Reach and Gust
            # retain their single-target episode criterion.
            required_successes = torch.where(
                terminal_scenario_code == 1,
                torch.full_like(terminal_success_count, 4),
                torch.ones_like(terminal_success_count),
            )
            self._interval_success_count += int(terminal_success_count.sum().item())
            self._interval_successful_episodes += int(
                (terminal_success_count >= required_successes).sum().item()
            )
        self._interval_failure_terminations += int(terminated.sum().item())
        self._interval_time_limit_truncations += int(truncated.sum().item())
        causes = torch.as_tensor(getattr(self.env, "terminal_failure_cause"), device=done.device)[done]
        for code in range(1, 5):
            self._interval_failure_causes[str(code)] += int((causes == code).sum().item())
        if not self._command_tracking:
            terminal_distance = torch.as_tensor(
                getattr(self.env, "terminal_distance_m"), device=done.device
            )[done]
            terminal_speed = torch.as_tensor(
                getattr(self.env, "terminal_speed_mps"), device=done.device
            )[done]
            self._interval_final_distance_sum += float(terminal_distance.sum().item())
            self._interval_final_speed_sum += float(terminal_speed.sum().item())
        for name, accumulator in self._episode_reward_components.items():
            self._interval_reward_component_sums[name] = (
                self._interval_reward_component_sums.get(name, 0.0)
                + float(accumulator[done].sum().item())
            )
            accumulator[done] = 0.0
        self._episode_return[done] = 0.0

    def consume_training_metrics(self) -> dict[str, Any]:
        """Return and clear update-local completed-episode aggregates."""

        count = self._interval_completed_episodes
        result: dict[str, Any] = {
            "completed_episode_count": count,
            "episodic_return_mean": (
                self._interval_episode_return_sum / count if count else None
            ),
            "episodic_return_sum": self._interval_episode_return_sum,
            "failure_termination_count": self._interval_failure_terminations,
            "time_limit_truncation_count": self._interval_time_limit_truncations,
            "failure_cause_counts": dict(self._interval_failure_causes),
            "episodic_reward_component_means": {
                name: value / count for name, value in self._interval_reward_component_sums.items()
            } if count else {},
            "rollout_reward_component_means": {
                name: value / self._interval_reward_component_sample_counts[name]
                for name, value in self._interval_reward_component_step_sums.items()
            },
        }
        if self._command_tracking:
            samples = self._interval_command_sample_count
            result.update({
                "command_tracking_sample_count": samples,
                "command_tracking_rmse_linear_mps": (
                    math.sqrt(
                        self._interval_command_linear_squared_error_sum / samples
                    )
                    if samples else None
                ),
                "command_tracking_rmse_yaw_radps": (
                    math.sqrt(
                        self._interval_command_yaw_squared_error_sum / samples
                    )
                    if samples else None
                ),
                "requested_command_mean_body": (
                    (self._interval_requested_command_sum / samples).tolist()
                    if samples else None
                ),
                "effective_command_mean_body": (
                    (self._interval_effective_command_sum / samples).tolist()
                    if samples else None
                ),
                "effective_command_rms_body": (
                    torch.sqrt(
                        self._interval_effective_command_squared_sum / samples
                    ).tolist()
                    if samples else None
                ),
                "effective_command_max_abs_body": (
                    self._interval_effective_command_max_abs.tolist()
                    if samples else None
                ),
                "command_projection_fraction": (
                    self._interval_command_projection_count / samples
                    if samples else None
                ),
            })
        else:
            result.update({
                "successful_episode_count": self._interval_successful_episodes,
                "target_success_count": self._interval_success_count,
                "final_goal_distance_mean_m": (
                    self._interval_final_distance_sum / count if count else None
                ),
                "final_speed_mean_m_s": (
                    self._interval_final_speed_sum / count if count else None
                ),
            })
        self._interval_completed_episodes = 0
        self._interval_episode_return_sum = 0.0
        self._interval_successful_episodes = 0
        self._interval_success_count = 0
        self._interval_failure_terminations = 0
        self._interval_time_limit_truncations = 0
        self._interval_failure_causes = {str(code): 0 for code in range(1, 5)}
        self._interval_reward_component_sums = {}
        self._interval_reward_component_step_sums = {}
        self._interval_reward_component_sample_counts = {}
        self._interval_final_distance_sum = 0.0
        self._interval_final_speed_sum = 0.0
        self._interval_command_sample_count = 0
        self._interval_command_linear_squared_error_sum = 0.0
        self._interval_command_yaw_squared_error_sum = 0.0
        self._interval_requested_command_sum.zero_()
        self._interval_effective_command_sum.zero_()
        self._interval_effective_command_squared_sum.zero_()
        self._interval_effective_command_max_abs.zero_()
        self._interval_command_projection_count = 0
        return result

    def close(self) -> None:
        self.env.close()
