"""Conventional trainable recurrent comparison with the same bounded action head."""

from __future__ import annotations

import torch
from torch import nn

from .actor_critic import PolicyOutput, TanhDiagonalGaussian, _mlp


class GRUActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_dim, self.action_dim, self.hidden_dim = observation_dim, action_dim, hidden_dim
        self.gru = nn.GRUCell(observation_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = _mlp([observation_dim, hidden_dim, hidden_dim, 1])
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def act(self, observations: torch.Tensor, state: torch.Tensor, *, deterministic: bool = False, **_: object) -> PolicyOutput:
        next_state = self.gru(observations, state)
        mean = self.actor(next_state)
        action, log_prob = TanhDiagonalGaussian(mean, self.log_std).sample(deterministic)
        return PolicyOutput(action, log_prob, self.critic(observations).squeeze(-1), next_state, mean)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_state = self.gru(observations, state)
        distribution = TanhDiagonalGaussian(self.actor(next_state), self.log_std)
        return distribution.log_prob(actions), distribution.entropy(), self.critic(observations).squeeze(-1), next_state

    def evaluate_sequence(
        self, observations: torch.Tensor, actions: torch.Tensor, initial_state: torch.Tensor, reset_before: torch.Tensor,
        *, burn_in: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or actions.shape[:2] != observations.shape[:2] or reset_before.shape != observations.shape[:2]:
            raise ValueError("GRU sequences require aligned [time, batch, feature] tensors and reset masks.")
        state = initial_state
        log_probs, entropies, values = [], [], []
        for step in range(observations.shape[0]):
            state = state.masked_fill(reset_before[step, :, None], 0.0)
            log_prob, entropy, value, state = self.evaluate_actions(observations[step], actions[step], state)
            if step >= burn_in:
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values), state
