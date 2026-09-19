"""Ordered recurrent rollout storage with explicit reset masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from g1_fly_control.policies.lif_core import LIFState


@dataclass
class RecurrentRollout:
    observations: torch.Tensor  # [T, B, O]
    actions: torch.Tensor  # [T, B, A], post-tanh policy actions
    rewards: torch.Tensor  # [T, B]
    terminated: torch.Tensor  # [T, B]
    truncated: torch.Tensor  # [T, B]
    reset_before: torch.Tensor  # [T, B]
    old_log_probs: torch.Tensor  # [T, B]
    values: torch.Tensor  # [T, B]
    initial_state: LIFState | torch.Tensor | None = None  # state at the rollout boundary
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def validate(self) -> None:
        time, batch = self.rewards.shape
        tensors = {
            "observations": self.observations.shape[:2], "actions": self.actions.shape[:2],
            "terminated": self.terminated.shape, "truncated": self.truncated.shape,
            "reset_before": self.reset_before.shape, "old_log_probs": self.old_log_probs.shape,
            "values": self.values.shape,
        }
        invalid = {name: shape for name, shape in tensors.items() if shape != (time, batch)}
        if invalid:
            raise ValueError(f"Rollout dimensions disagree with rewards [T,B]: {invalid}")
        if not torch.isfinite(self.observations).all() or not torch.isfinite(self.actions).all():
            raise ValueError("Rollout contains non-finite observations/actions.")
        if not torch.isfinite(self.rewards).all() or not torch.isfinite(self.old_log_probs).all():
            raise ValueError("Rollout contains non-finite rewards/log probabilities.")
        if isinstance(self.initial_state, LIFState) and self.initial_state.membrane.shape[0] != batch:
            raise ValueError("Initial LIF state batch differs from rollout batch.")
        if isinstance(self.initial_state, torch.Tensor) and self.initial_state.shape[0] != batch:
            raise ValueError("Initial recurrent state batch differs from rollout batch.")

    def compute_returns(self, bootstrap_value: torch.Tensor, *, gamma: float, gae_lambda: float) -> None:
        self.validate()
        if bootstrap_value.shape != self.values.shape[1:]:
            raise ValueError("Bootstrap values must have shape [batch].")
        advantages = torch.zeros_like(self.rewards)
        advantage = torch.zeros_like(bootstrap_value)
        next_value = bootstrap_value
        for step in range(self.rewards.shape[0] - 1, -1, -1):
            # Both true terminations and auto-reset time limits end this trajectory.
            # The caller folds gamma * V(terminal_observation) into the reward for
            # time limits before this backward pass, so no reset observation leaks in.
            nonterminal = (~(self.terminated[step] | self.truncated[step])).to(self.rewards.dtype)
            delta = self.rewards[step] + gamma * next_value * nonterminal - self.values[step]
            advantage = delta + gamma * gae_lambda * nonterminal * advantage
            advantages[step] = advantage
            next_value = self.values[step]
        self.advantages = advantages
        self.returns = advantages + self.values


class RecurrentRolloutBuilder:
    """Append on-policy transitions before converting them to contiguous tensors."""

    def __init__(self, initial_reset: torch.Tensor, initial_state: LIFState | torch.Tensor | None = None) -> None:
        self._reset_before = initial_reset.bool()
        self._initial_state = initial_state.clone() if initial_state is not None else None
        self._items: list[tuple[torch.Tensor, ...]] = []

    def append(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._items.append(tuple(
            tensor.detach().clone() for tensor in
            (observation, action, reward, terminated.bool(), truncated.bool(), self._reset_before, log_prob, value)
        ))
        self._reset_before = (terminated | truncated).bool()

    def build(self) -> RecurrentRollout:
        if not self._items:
            raise ValueError("Cannot build an empty rollout.")
        fields = tuple(torch.stack([item[index] for item in self._items]) for index in range(8))
        result = RecurrentRollout(*fields, initial_state=self._initial_state)
        result.validate()
        return result
