"""Policies that keep the biological core on the actor path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from .lif_core import LIFCore, LIFState


def _mlp(widths: Iterable[int], *, activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    dimensions = list(widths)
    if len(dimensions) < 2:
        raise ValueError("MLP needs input and output dimensions.")
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        layers.append(nn.Linear(source, target))
        if index != len(dimensions) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    state: LIFState | None
    mean: torch.Tensor


class TanhDiagonalGaussian:
    """A bounded distribution with log probabilities in the transformed action space."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.log_std = log_std.expand_as(mean)
        self.normal = Normal(mean, self.log_std.exp())

    def sample(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.mean if deterministic else self.normal.rsample()
        action = torch.tanh(latent)
        return action, self.log_prob(action, latent=latent)

    def log_prob(self, action: torch.Tensor, *, latent: torch.Tensor | None = None) -> torch.Tensor:
        bounded = action.clamp(-1 + 1e-6, 1 - 1e-6)
        latent = torch.atanh(bounded) if latent is None else latent
        correction = torch.log(1.0 - bounded.square() + 1e-6)
        return (self.normal.log_prob(latent) - correction).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # Base-distribution entropy is a documented PPO regularizer approximation.
        return self.normal.entropy().sum(dim=-1)


class FrozenLIFActorCritic(nn.Module):
    """Small trainable adapters around a frozen, differentiable-through LIF core."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        core: LIFCore,
        *,
        input_indices: list[int] | torch.Tensor,
        output_indices: list[int] | torch.Tensor,
        adapter_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.core = core
        self.register_buffer("input_indices", torch.as_tensor(input_indices, dtype=torch.long))
        self.register_buffer("output_indices", torch.as_tensor(output_indices, dtype=torch.long))
        if self.input_indices.numel() == 0 or self.output_indices.numel() == 0:
            raise ValueError("At least one circuit input and output neuron is required.")
        if self.input_indices.min() < 0 or self.input_indices.max() >= core.num_neurons:
            raise ValueError("Input subset is outside core neuron range.")
        if self.output_indices.min() < 0 or self.output_indices.max() >= core.num_neurons:
            raise ValueError("Output subset is outside core neuron range.")
        self.encoder = _mlp([observation_dim, adapter_hidden_dim, self.input_indices.numel()])
        self.decoder = _mlp([self.output_indices.numel(), adapter_hidden_dim, action_dim])
        self.critic = _mlp([observation_dim, critic_hidden_dim, critic_hidden_dim, 1])
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) + sum(buffer.numel() for buffer in self.buffers())

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> LIFState:
        return self.core.initial_state(batch_size, device=device)

    def _mean_and_state(
        self, observations: torch.Tensor, state: LIFState, *, ablate_outgoing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, LIFState]:
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"Expected observations [batch, {self.observation_dim}].")
        encoded = self.encoder(observations)
        current = torch.zeros(
            (observations.shape[0], self.core.num_neurons), dtype=observations.dtype, device=observations.device
        )
        current[:, self.input_indices] = encoded
        state = self.core(current, state, ablate_outgoing=ablate_outgoing)
        return self.decoder(state.synapse[:, self.output_indices]), state

    def act(
        self,
        observations: torch.Tensor,
        state: LIFState,
        *,
        deterministic: bool = False,
        ablate_outgoing: torch.Tensor | None = None,
    ) -> PolicyOutput:
        mean, next_state = self._mean_and_state(observations, state, ablate_outgoing=ablate_outgoing)
        distribution = TanhDiagonalGaussian(mean, self.log_std)
        action, log_prob = distribution.sample(deterministic=deterministic)
        return PolicyOutput(action, log_prob, self.critic(observations).squeeze(-1), next_state, mean)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state: LIFState,
        *,
        ablate_outgoing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LIFState]:
        mean, next_state = self._mean_and_state(observations, state, ablate_outgoing=ablate_outgoing)
        distribution = TanhDiagonalGaussian(mean, self.log_std)
        return distribution.log_prob(actions), distribution.entropy(), self.critic(observations).squeeze(-1), next_state

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_state: LIFState,
        reset_before: torch.Tensor,
        *,
        burn_in: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LIFState]:
        """Replay ordered rollout data with resets and differentiable truncated BPTT.

        ``reset_before[t]`` masks environments whose state must be zeroed before
        observation ``t``. Burn-in runs dynamics but omits its losses.
        """
        if observations.ndim != 3 or actions.shape[:2] != observations.shape[:2]:
            raise ValueError("Sequences must be [time, batch, features].")
        if reset_before.shape != observations.shape[:2]:
            raise ValueError("reset_before must be [time, batch].")
        if not 0 <= burn_in < observations.shape[0]:
            raise ValueError("burn_in must be in [0, sequence length).")
        state = initial_state
        log_probs, entropies, values = [], [], []
        for step in range(observations.shape[0]):
            state = state.masked_reset(reset_before[step])
            log_prob, entropy, value, state = self.evaluate_actions(observations[step], actions[step], state)
            if step >= burn_in:
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values), state


class MLPActorCritic(nn.Module):
    """Ordinary-policy engineering baseline using the identical action distribution."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor = _mlp([observation_dim, hidden_dim, hidden_dim, action_dim])
        self.critic = _mlp([observation_dim, hidden_dim, hidden_dim, 1])
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def act(self, observations: torch.Tensor, state: None = None, *, deterministic: bool = False, **_: object) -> PolicyOutput:
        mean = self.actor(observations)
        distribution = TanhDiagonalGaussian(mean, self.log_std)
        action, log_prob = distribution.sample(deterministic=deterministic)
        return PolicyOutput(action, log_prob, self.critic(observations).squeeze(-1), None, mean)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor, state: None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        distribution = TanhDiagonalGaussian(self.actor(observations), self.log_std)
        return distribution.log_prob(actions), distribution.entropy(), self.critic(observations).squeeze(-1), None

