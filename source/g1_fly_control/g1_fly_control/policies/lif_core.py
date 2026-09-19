"""Batched frozen LIF dynamics with a straight-through surrogate derivative."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import nn


@dataclass
class LIFState:
    membrane: torch.Tensor
    spikes: torch.Tensor
    synapse: torch.Tensor
    refractory: torch.Tensor

    def clone(self) -> "LIFState":
        return LIFState(*(item.clone() for item in (self.membrane, self.spikes, self.synapse, self.refractory)))

    def masked_reset(self, mask: torch.Tensor) -> "LIFState":
        """Functionally reset selected batch rows while preserving autograd for others."""
        mask = mask.bool().reshape(-1, 1)
        return LIFState(
            self.membrane.masked_fill(mask, 0.0),
            self.spikes.masked_fill(mask, 0.0),
            self.synapse.masked_fill(mask, 0.0),
            self.refractory.masked_fill(mask, 0),
        )


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_minus_threshold: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(voltage_minus_threshold)
        ctx.beta = beta
        return (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        # Fast-sigmoid surrogate; the forward spike remains exactly hard/binary.
        return grad_output * (beta / (1.0 + beta * x.abs()).square()), None


class LIFCore(nn.Module):
    """Frozen recurrent LIF core with independent state for each environment.

    ``edge_index[0]`` is presynaptic and ``edge_index[1]`` postsynaptic.  All
    connectivity/model tensors are buffers, so optimizers cannot update them; no
    ``no_grad`` or detach is used in the forward path, preserving gradients to the encoder.
    """

    # The current MaleCNS subset has 256 neurons. A fixed dense multiply avoids
    # CUDA atomic reduction-order changes in index_add that can cross a hard
    # spike threshold between rollout collection and PPO replay. Full graphs
    # need a separately validated deterministic sparse backend.
    DENSE_REPLAY_LIMIT = 1024

    def __init__(
        self,
        num_neurons: int,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
        *,
        dt: float = 0.002,
        tau_membrane: float = 0.020,
        tau_synapse: float = 0.010,
        threshold: float = 1.0,
        reset_value: float = 0.0,
        refractory_steps: int = 1,
        surrogate_beta: float = 10.0,
        neural_substeps: int = 5,
    ) -> None:
        super().__init__()
        if num_neurons <= 0 or edge_index.shape[0] != 2 or edge_index.shape[1] != weights.numel():
            raise ValueError("Invalid LIF graph dimensions.")
        if dt <= 0 or tau_membrane <= 0 or tau_synapse <= 0 or neural_substeps < 1:
            raise ValueError("LIF time constants and neural_substeps must be positive.")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be non-negative.")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_neurons):
            raise ValueError("LIF edge index is outside neuron range.")
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.tau_membrane = float(tau_membrane)
        self.tau_synapse = float(tau_synapse)
        self.threshold = float(threshold)
        self.reset_value = float(reset_value)
        self.refractory_steps = int(refractory_steps)
        self.surrogate_beta = float(surrogate_beta)
        self.neural_substeps = int(neural_substeps)
        self.register_buffer("edge_index", edge_index.to(dtype=torch.long).contiguous())
        self.register_buffer("weights", weights.to(dtype=torch.float32).contiguous())
        self.register_buffer("dense_recurrent_weights", self._build_dense_recurrent_weights(), persistent=False)
        self.register_buffer("membrane_decay", torch.tensor(float(torch.exp(torch.tensor(-dt / tau_membrane)))))
        self.register_buffer("synapse_decay", torch.tensor(float(torch.exp(torch.tensor(-dt / tau_synapse)))))

    def _build_dense_recurrent_weights(self) -> torch.Tensor:
        if self.num_neurons > self.DENSE_REPLAY_LIMIT:
            return torch.empty(0, dtype=torch.float32, device=self.weights.device)
        dense_cpu = torch.zeros((self.num_neurons, self.num_neurons), dtype=torch.float32)
        if self.edge_index.numel():
            edge_cpu = self.edge_index.detach().cpu()
            dense_cpu.index_put_(
                (edge_cpu[1], edge_cpu[0]), self.weights.detach().cpu(), accumulate=True
            )
        return dense_cpu.to(self.weights.device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # This derived, nonpersistent buffer must follow checkpoint graph values.
        self.dense_recurrent_weights = self._build_dense_recurrent_weights()

    @property
    def frozen_checksum(self) -> str:
        hash_ = sha256()
        for tensor in (self.edge_index, self.weights, self.membrane_decay, self.synapse_decay):
            hash_.update(tensor.detach().cpu().numpy().tobytes())
        return hash_.hexdigest()

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> LIFState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        target_device = device if device is not None else self.weights.device
        shape = (batch_size, self.num_neurons)
        return LIFState(
            membrane=torch.zeros(shape, device=target_device, dtype=self.weights.dtype),
            spikes=torch.zeros(shape, device=target_device, dtype=self.weights.dtype),
            synapse=torch.zeros(shape, device=target_device, dtype=self.weights.dtype),
            refractory=torch.zeros(shape, device=target_device, dtype=torch.long),
        )

    def reset(self, state: LIFState, env_ids: torch.Tensor | list[int] | None = None) -> LIFState:
        if env_ids is None:
            return self.initial_state(state.membrane.shape[0], device=state.membrane.device)
        mask = torch.zeros(state.membrane.shape[0], dtype=torch.bool, device=state.membrane.device)
        mask[torch.as_tensor(env_ids, device=state.membrane.device, dtype=torch.long)] = True
        return state.masked_reset(mask)

    def _recurrent_current(self, filtered_spikes: torch.Tensor) -> torch.Tensor:
        if self.dense_recurrent_weights.numel():
            return filtered_spikes @ self.dense_recurrent_weights.T
        current = torch.zeros_like(filtered_spikes)
        if self.edge_index.numel() == 0:
            return current
        messages = filtered_spikes[:, self.edge_index[0]] * self.weights
        return current.index_add(1, self.edge_index[1], messages)

    def step(self, injected_current: torch.Tensor, state: LIFState, *, ablate_outgoing: torch.Tensor | None = None) -> LIFState:
        if injected_current.shape != state.membrane.shape:
            raise ValueError("Injected current must have shape [batch, num_neurons].")
        active = state.refractory <= 0
        voltage = self.membrane_decay * state.membrane + (1.0 - self.membrane_decay) * (
            injected_current + self._recurrent_current(state.synapse)
        )
        voltage = torch.where(active, voltage, torch.full_like(voltage, self.reset_value))
        hard_spikes = _SurrogateSpike.apply(voltage - self.threshold, self.surrogate_beta) * active.to(voltage.dtype)
        recurrent_spikes = hard_spikes
        if ablate_outgoing is not None:
            mask = ablate_outgoing.to(device=voltage.device, dtype=torch.bool).reshape(1, -1)
            if mask.shape[1] != self.num_neurons:
                raise ValueError("ablate_outgoing must have one element per neuron.")
            recurrent_spikes = hard_spikes.masked_fill(mask, 0.0)
        membrane = torch.where(hard_spikes.bool(), torch.full_like(voltage, self.reset_value), voltage)
        refractory = torch.clamp(state.refractory - 1, min=0)
        refractory = torch.where(hard_spikes.bool(), torch.full_like(refractory, self.refractory_steps), refractory)
        synapse = self.synapse_decay * state.synapse + recurrent_spikes
        return LIFState(membrane, hard_spikes, synapse, refractory)

    def forward(self, injected_current: torch.Tensor, state: LIFState, *, ablate_outgoing: torch.Tensor | None = None) -> LIFState:
        for _ in range(self.neural_substeps):
            state = self.step(injected_current, state, ablate_outgoing=ablate_outgoing)
        return state

    def dense_recurrent_current(self, filtered_spikes: torch.Tensor) -> torch.Tensor:
        """Reference calculation used by tests; W is indexed [post, pre]."""
        if self.dense_recurrent_weights.numel():
            matrix = self.dense_recurrent_weights
        else:
            matrix = torch.zeros((self.num_neurons, self.num_neurons), dtype=self.weights.dtype, device=self.weights.device)
            matrix.index_put_((self.edge_index[1], self.edge_index[0]), self.weights, accumulate=True)
        return filtered_spikes @ matrix.T
