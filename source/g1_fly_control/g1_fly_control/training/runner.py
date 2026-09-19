"""Small recurrent PPO reference runner for Isaac Lab vector environments.

It is intentionally explicit about stored policy-space actions, reset masks, and
sequence replay. It is suitable for smoke/pilot runs; large studies should record
the resolved config and budget rather than silently changing these defaults.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from g1_fly_control.policies.actor_critic import FrozenLIFActorCritic, TanhDiagonalGaussian
from g1_fly_control.policies.lif_core import LIFState
from g1_fly_control.policies.gru import GRUActorCritic

from .recurrent_storage import RecurrentRollout, RecurrentRolloutBuilder


@dataclass(frozen=True)
class PPOConfig:
    horizon: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.002
    learning_rate: float = 3.0e-4
    ppo_epochs: int = 2
    max_grad_norm: float = 1.0
    burn_in: int = 0
    target_kl: float | None = None


def policy_observation(observation: Any) -> torch.Tensor:
    """Extract the documented policy group from Isaac Lab/Gym observations."""
    if isinstance(observation, dict):
        observation = observation.get("policy", observation.get("actor", observation))
    if not isinstance(observation, torch.Tensor):
        raise TypeError(f"Expected Tensor or observation dict, got {type(observation).__name__}")
    return observation


class RecurrentPPO:
    def __init__(self, policy: nn.Module, config: PPOConfig = PPOConfig()) -> None:
        if config.target_kl is not None and (not math.isfinite(config.target_kl) or config.target_kl <= 0):
            raise ValueError("PPO target_kl must be positive and finite when set.")
        if isinstance(policy, FrozenLIFActorCritic) and not policy.core.dense_recurrent_weights.numel():
            raise ValueError(
                "Frozen-LIF PPO replay requires deterministic recurrent accumulation; "
                "graphs above the current dense limit need a validated sparse backend."
            )
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self._observation: torch.Tensor | None = None
        self._state: LIFState | torch.Tensor | None = None
        self._reset_before: torch.Tensor | None = None
        self._replay_debug = os.environ.get("FLYG1_REPLAY_DIAGNOSTICS") == "1"
        self._debug_collected_means: list[torch.Tensor] = []
        self._debug_collected_actions: list[torch.Tensor] = []
        self._debug_collected_states: list[LIFState] = []
        self._debug_replay_means: list[torch.Tensor] = []
        self._debug_replay_states: list[LIFState] = []
        self._debug_immediate_log_probs: list[torch.Tensor] = []
        self._debug_log_stds: list[torch.Tensor] = []
        self._debug_observation_mutation_max = 0.0
        self._debug_action_mutation_max = 0.0

    def _initial_state(self, batch_size: int, device: torch.device) -> LIFState | None:
        if hasattr(self.policy, "initial_state"):
            return self.policy.initial_state(batch_size, device=device)
        return None

    @staticmethod
    def _state_reset(state: LIFState | torch.Tensor | None, done: torch.Tensor) -> LIFState | torch.Tensor | None:
        if isinstance(state, LIFState):
            return state.masked_reset(done)
        if isinstance(state, torch.Tensor):
            return state.masked_fill(done[:, None], 0.0)
        return None

    @torch.no_grad()
    def collect(self, env, *, progress_callback: Callable[[int, str, dict[str, float | int]], None] | None = None
                ) -> tuple[RecurrentRollout, LIFState | torch.Tensor | None, torch.Tensor]:
        if self._observation is None:
            observation, _ = env.reset()
            observation = policy_observation(observation)
            state = self._initial_state(observation.shape[0], observation.device)
            reset_before = torch.ones(observation.shape[0], dtype=torch.bool, device=observation.device)
        else:
            observation, state = self._observation, self._state
            reset_before = self._reset_before
            if reset_before is None:
                raise RuntimeError("Missing reset mask for continued rollout.")
        builder = RecurrentRolloutBuilder(reset_before, state)
        if self._replay_debug:
            self._debug_collected_means = []
            self._debug_collected_actions = []
            self._debug_collected_states = []
            self._debug_immediate_log_probs = []
            self._debug_log_stds = []
            self._debug_observation_mutation_max = 0.0
            self._debug_action_mutation_max = 0.0
        for step_index in range(self.config.horizon):
            if progress_callback is not None:
                progress_callback(step_index, "before_policy", {
                    "reset_before_count": int(reset_before.sum()),
                    "max_episode_length_buf": int(env.episode_length_buf.max()),
                })
            observation_before_step = observation.clone() if self._replay_debug else None
            output = self.policy.act(observation, state)
            action_before_step = output.action.clone() if self._replay_debug else None
            if self._replay_debug:
                self._debug_collected_means.append(output.mean.detach().clone())
                self._debug_collected_actions.append(action_before_step.detach().clone())
                if isinstance(output.state, LIFState):
                    self._debug_collected_states.append(output.state.clone())
                self._debug_log_stds.append(self.policy.log_std.detach().clone())
                self._debug_immediate_log_probs.append(
                    TanhDiagonalGaussian(output.mean, self.policy.log_std).log_prob(output.action).detach().clone()
                )
            if progress_callback is not None:
                progress_callback(step_index, "before_env_step", {
                    "action_max_abs": float(output.action.abs().max()),
                    "action_mean_abs": float(output.action.abs().mean()),
                })
            next_observation, reward, terminated, truncated, _ = env.step(output.action)
            if observation_before_step is not None and action_before_step is not None:
                self._debug_observation_mutation_max = max(
                    self._debug_observation_mutation_max,
                    float((observation - observation_before_step).abs().max()),
                )
                self._debug_action_mutation_max = max(
                    self._debug_action_mutation_max,
                    float((output.action - action_before_step).abs().max()),
                )
            if progress_callback is not None:
                progress_callback(step_index, "after_env_step", {
                    "terminated_count": int(terminated.sum()),
                    "truncated_count": int(truncated.sum()),
                    "max_episode_length_buf": int(env.episode_length_buf.max()),
                })
            next_observation = policy_observation(next_observation)
            if bool(truncated.any()):
                terminal_obs = getattr(env.unwrapped, "flyg1_terminal_observation", None)
                if terminal_obs is None:
                    raise RuntimeError("Timeout bootstrapping requires a captured terminal observation.")
                # Capture uses the same observation manager before Isaac Lab's
                # auto-reset. The separate value network does not need LIF state.
                terminal_value = self.policy.critic(terminal_obs).squeeze(-1)
                reward = reward + self.config.gamma * terminal_value * truncated.to(reward.dtype)
            builder.append(
                observation, output.action, reward, terminated, truncated, output.log_prob, output.value
            )
            state = self._state_reset(output.state, terminated | truncated)
            observation = next_observation
            reset_before = terminated | truncated
        rollout = builder.build()
        # This value corresponds to the nonterminal post-rollout observation/state.
        bootstrap = self.policy.act(observation, state, deterministic=True).value
        rollout.compute_returns(bootstrap, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        self._observation, self._state, self._reset_before = observation, state, reset_before
        return rollout, state, observation

    def _evaluate(self, rollout: RecurrentRollout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.policy, (FrozenLIFActorCritic, GRUActorCritic)):
            state = rollout.initial_state
            if state is None:
                raise RuntimeError("Recurrent rollout is missing its sequence-start state.")
            replay_means: list[torch.Tensor] = []
            replay_states: list[LIFState] = []
            hook = None
            state_hook = None
            if self._replay_debug and isinstance(self.policy, FrozenLIFActorCritic):
                hook = self.policy.decoder.register_forward_hook(
                    lambda _module, _inputs, output: replay_means.append(output.detach().clone())
                )
                state_hook = self.policy.core.register_forward_hook(
                    lambda _module, _inputs, output: replay_states.append(LIFState(*(
                        item.detach().clone() for item in
                        (output.membrane, output.spikes, output.synapse, output.refractory)
                    )))
                )
            try:
                log_prob, entropy, values, _ = self.policy.evaluate_sequence(
                    rollout.observations, rollout.actions, state, rollout.reset_before, burn_in=self.config.burn_in
                )
            finally:
                if hook is not None:
                    hook.remove()
                if state_hook is not None:
                    state_hook.remove()
            if hook is not None:
                self._debug_replay_means = replay_means
                self._debug_replay_states = replay_states
            return log_prob, entropy, values
        flat_obs = rollout.observations.flatten(0, 1)
        flat_actions = rollout.actions.flatten(0, 1)
        log_prob, entropy, values, _ = self.policy.evaluate_actions(flat_obs, flat_actions)
        shape = rollout.rewards.shape
        return log_prob.view(shape), entropy.view(shape), values.view(shape)

    def update(self, rollout: RecurrentRollout) -> dict[str, float]:
        if rollout.advantages is None or rollout.returns is None:
            raise ValueError("Call compute_returns before PPO update.")
        if not 0 <= self.config.burn_in < rollout.rewards.shape[0]:
            raise ValueError("PPO burn_in must be shorter than rollout horizon.")
        loss_slice = slice(self.config.burn_in, None)
        old_log_probs = rollout.old_log_probs[loss_slice]
        target_returns = rollout.returns[loss_slice]
        with torch.no_grad():
            before_log_prob, _, _ = self._evaluate(rollout)
            if not isinstance(self.policy, (FrozenLIFActorCritic, GRUActorCritic)):
                before_log_prob = before_log_prob[loss_slice]
            mismatch = (before_log_prob - old_log_probs).abs().max()
            if mismatch > 5e-4:
                if self._replay_debug:
                    differences = (before_log_prob - old_log_probs).abs()
                    maximum_flat_index = int(differences.flatten().argmax())
                    mismatch_step, mismatch_env = divmod(maximum_flat_index, differences.shape[1])
                    diagnostic: dict[str, Any] = {
                        "kind": "ppo_replay_mismatch", "max_abs_log_prob_diff": float(mismatch),
                        "step": mismatch_step, "environment": mismatch_env,
                        "action_max_abs_at_step": float(rollout.actions[mismatch_step, mismatch_env].abs().max()),
                        "reset_before_at_step": bool(rollout.reset_before[mismatch_step, mismatch_env]),
                        "observation_mutation_max_after_env_step": self._debug_observation_mutation_max,
                        "action_mutation_max_after_env_step": self._debug_action_mutation_max,
                        "stored_log_prob": float(old_log_probs[mismatch_step, mismatch_env]),
                        "replayed_log_prob": float(before_log_prob[mismatch_step, mismatch_env]),
                    }
                    if len(self._debug_immediate_log_probs) == rollout.rewards.shape[0]:
                        diagnostic["immediate_recomputed_log_prob"] = float(
                            self._debug_immediate_log_probs[mismatch_step][mismatch_env]
                        )
                        diagnostic["collection_log_std_max_abs_diff_from_replay"] = float(
                            (self._debug_log_stds[mismatch_step] - self.policy.log_std).abs().max()
                        )
                        diagnostic["collection_log_std_mean"] = float(self._debug_log_stds[mismatch_step].mean())
                        diagnostic["replay_log_std_mean"] = float(self.policy.log_std.detach().mean())
                        collected_actions = torch.stack(self._debug_collected_actions)
                        diagnostic["collection_vs_rollout_action_max_abs_diff"] = float(
                            (collected_actions - rollout.actions).abs().max()
                        )
                        diagnostic["collection_vs_rollout_action_diff_at_mismatch"] = float(
                            (collected_actions[mismatch_step, mismatch_env]
                             - rollout.actions[mismatch_step, mismatch_env]).abs().max()
                        )
                        diagnostic["manual_log_prob_collected_mean_stored_action"] = float(
                            TanhDiagonalGaussian(
                                self._debug_collected_means[mismatch_step],
                                self._debug_log_stds[mismatch_step],
                            ).log_prob(rollout.actions[mismatch_step])[mismatch_env]
                        )
                    if (isinstance(self.policy, FrozenLIFActorCritic)
                            and len(self._debug_replay_means) == rollout.rewards.shape[0]):
                        first_replay_means = self._debug_replay_means
                        first_replay_states = self._debug_replay_states
                        mean_differences = [
                            float((replay_mean - collected_mean).abs().max())
                            for replay_mean, collected_mean in zip(
                                first_replay_means, self._debug_collected_means, strict=True
                            )
                        ]
                        diagnostic["first_replay_mean_diff_by_step"] = mean_differences
                        diagnostic["first_mean_divergence_step"] = next(
                            (step for step, difference in enumerate(mean_differences) if difference > 1e-5), None
                        )
                        if len(first_replay_states) == len(self._debug_collected_states):
                            state_differences = [{
                                "membrane": float((first.membrane - collected.membrane).abs().max()),
                                "synapse": float((first.synapse - collected.synapse).abs().max()),
                                "spike_count": int((first.spikes != collected.spikes).sum()),
                                "refractory_count": int((first.refractory != collected.refractory).sum()),
                            } for first, collected in zip(
                                first_replay_states, self._debug_collected_states, strict=True
                            )]
                            diagnostic["first_state_divergence_step"] = next((
                                step for step, row in enumerate(state_differences)
                                if any(value != 0 for value in row.values())
                            ), None)
                            diagnostic["first_spike_divergence_step"] = next((
                                step for step, row in enumerate(state_differences) if row["spike_count"]
                            ), None)
                            if diagnostic["first_state_divergence_step"] is not None:
                                diagnostic["first_state_divergence"] = state_differences[
                                    diagnostic["first_state_divergence_step"]
                                ]
                            if diagnostic["first_spike_divergence_step"] is not None:
                                diagnostic["first_spike_divergence"] = state_differences[
                                    diagnostic["first_spike_divergence_step"]
                                ]
                        second_log_prob, _, _ = self._evaluate(rollout)
                        diagnostic["second_replay_log_prob_at_mismatch"] = float(
                            second_log_prob[mismatch_step, mismatch_env]
                        )
                        diagnostic["first_vs_second_replay_log_prob_max_abs_diff"] = float(
                            (before_log_prob - second_log_prob).abs().max()
                        )
                        diagnostic["first_vs_second_replay_mean_max_abs_diff"] = max(
                            float((first - second).abs().max())
                            for first, second in zip(first_replay_means, self._debug_replay_means, strict=True)
                        )
                        bundle_path = os.environ.get("FLYG1_REPLAY_BUNDLE_PATH")
                        if bundle_path:
                            destination = Path(bundle_path)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            torch.save({
                                "policy": self.policy.state_dict(),
                                "observations": rollout.observations,
                                "actions": rollout.actions,
                                "reset_before": rollout.reset_before,
                                "old_log_probs": rollout.old_log_probs,
                                "initial_state": rollout.initial_state,
                                "collection_means": torch.stack(self._debug_collected_means),
                                "first_replay_means": torch.stack(first_replay_means),
                                "first_replay_log_probs": before_log_prob,
                                "collection_states": self._debug_collected_states,
                                "first_replay_states": first_replay_states,
                            }, destination)
                            diagnostic["bundle_path"] = str(destination)
                    print(json.dumps(diagnostic, sort_keys=True), flush=True)
                raise RuntimeError(f"Stored/replayed policy log probabilities differ before PPO update: {float(mismatch):.6g}")
        # Normalize across independent vector environments/time samples, not episode rows.
        selected_advantages = rollout.advantages[loss_slice]
        advantages = (selected_advantages - selected_advantages.mean()) / (selected_advantages.std(unbiased=False) + 1e-8)
        metrics: dict[str, float] = {}
        accepted_epochs = 0
        for _ in range(self.config.ppo_epochs):
            # A guarded update can be rolled back without disturbing earlier
            # accepted epochs, including Adam's moment estimates and step count.
            policy_before = deepcopy(self.policy.state_dict()) if self.config.target_kl is not None else None
            optimizer_before = deepcopy(self.optimizer.state_dict()) if self.config.target_kl is not None else None
            log_prob, entropy, values = self._evaluate(rollout)
            if not isinstance(self.policy, (FrozenLIFActorCritic, GRUActorCritic)):
                log_prob, entropy, values = log_prob[loss_slice], entropy[loss_slice], values[loss_slice]
            ratio = (log_prob - old_log_probs).exp()
            surrogate_a = ratio * advantages
            surrogate_b = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
            policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
            value_loss = 0.5 * (values - target_returns).square().mean()
            entropy_bonus = entropy.mean()
            loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy_bonus
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss; checkpoint and simulator state were not advanced.")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            finite_parameters = torch.stack([
                torch.isfinite(parameter).all() for parameter in self.policy.parameters()
            ]).all()
            if not bool(finite_parameters):
                if policy_before is not None and optimizer_before is not None:
                    self.policy.load_state_dict(policy_before)
                    self.optimizer.load_state_dict(optimizer_before)
                raise FloatingPointError("Non-finite policy parameters after PPO optimizer step.")
            with torch.no_grad():
                post_log_prob, _, _ = self._evaluate(rollout)
                if not isinstance(self.policy, (FrozenLIFActorCritic, GRUActorCritic)):
                    post_log_prob = post_log_prob[loss_slice]
                # The nonnegative ratio-based estimator is more reliable than
                # mean(old_log_prob - new_log_prob) on a short sampled rollout.
                log_ratio = (post_log_prob - old_log_probs).double()
                post_step_kl = (torch.expm1(log_ratio) - log_ratio).mean().clamp_min(0.0)
                if not bool(torch.isfinite(post_step_kl)):
                    if policy_before is not None and optimizer_before is not None:
                        self.policy.load_state_dict(policy_before)
                        self.optimizer.load_state_dict(optimizer_before)
                    raise FloatingPointError("Non-finite post-step PPO KL.")
                rejected = self.config.target_kl is not None and (
                    float(post_step_kl) > self.config.target_kl
                )
                if rejected:
                    pre_log_ratio = (log_prob.detach() - old_log_probs).double()
                    accepted_kl = (torch.expm1(pre_log_ratio) - pre_log_ratio).mean().clamp_min(0.0)
                else:
                    accepted_kl = post_step_kl
            if rejected:
                if policy_before is None or optimizer_before is None:
                    raise RuntimeError("PPO KL rollback state was not captured.")
                self.policy.load_state_dict(policy_before)
                self.optimizer.load_state_dict(optimizer_before)
            else:
                accepted_epochs += 1
            metrics = {
                "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()), "entropy": float(entropy_bonus.detach()),
                "grad_norm": float(grad_norm.detach()), "approx_kl": float(accepted_kl),
                "attempted_kl": float(post_step_kl), "rejected_step": float(rejected),
                "accepted_epochs": float(accepted_epochs),
            }
            if rejected:
                break
        return metrics

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
