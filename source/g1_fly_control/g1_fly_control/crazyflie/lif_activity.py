"""Bounded, rollout-wide activity summaries for Crazyflie LIF controllers.

The training path deliberately keeps only one per-neuron spike-count vector
plus a fixed-size scalar accumulator on the policy device.  It never retains
state traces and performs its single device-to-host synchronization only when
the completed rollout is summarized.
"""

from __future__ import annotations

import math
from types import TracebackType
from typing import Any

import torch
from torch import nn


def scheduled_measurement_due(
    update_index: int,
    *,
    total_updates: int,
    checkpoint_every_updates: int,
) -> bool:
    """Return the shared first/checkpoint/final measurement cadence."""

    values = (update_index, total_updates, checkpoint_every_updates)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("measurement update indices and cadence must be positive integers")
    if update_index > total_updates:
        raise ValueError("measurement update index exceeds total updates")
    return (
        update_index == 1
        or update_index == total_updates
        or update_index % checkpoint_every_updates == 0
    )


class RolloutLIFActivityAccumulator:
    """Accumulate one post-final-substep LIF state per control decision.

    ``RecurrentPPO.collect`` evaluates the policy once per rollout step and
    once more for the bootstrap value.  The first ``horizon`` core outputs are
    counted, while the one bootstrap output is observed but intentionally
    excluded.  Consequently, every reported activity fraction has exactly
    ``horizon * num_envs`` neuron samples in its denominator.
    """

    SCHEMA_VERSION = 1
    MAX_NEURONS = 256
    _EXPECTED_BOOTSTRAP_CALLS = 1
    _MEMBRANE_L2_SUM = 0
    _MEMBRANE_L2_SQUARE_SUM = 1
    _MEMBRANE_L2_MAX = 2
    _SYNAPSE_L2_SUM = 3
    _SYNAPSE_L2_SQUARE_SUM = 4
    _SYNAPSE_L2_MAX = 5
    _NONFINITE_VALUE_COUNT = 6
    _NONBINARY_SPIKE_COUNT = 7
    _STAT_COUNT = 8

    def __init__(
        self,
        core: nn.Module,
        *,
        horizon: int,
        num_envs: int,
        control_dt_s: float,
    ) -> None:
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if (
            not isinstance(control_dt_s, (int, float))
            or not math.isfinite(float(control_dt_s))
            or not 0.0 < float(control_dt_s)
        ):
            raise ValueError("control_dt_s must be positive")
        num_neurons = getattr(core, "num_neurons", None)
        if type(num_neurons) is not int or not 0 < num_neurons <= self.MAX_NEURONS:
            raise ValueError(
                f"Crazyflie LIF activity requires 1..{self.MAX_NEURONS} neurons; "
                f"received {num_neurons!r}"
            )
        weights = getattr(core, "weights", None)
        if not isinstance(weights, torch.Tensor):
            raise TypeError("LIF core must expose its device through a weights tensor")
        neural_substeps = getattr(core, "neural_substeps", None)
        neural_dt_s = getattr(core, "dt", None)
        if type(neural_substeps) is not int or neural_substeps <= 0:
            raise ValueError("LIF core neural_substeps must be a positive integer")
        if (
            not isinstance(neural_dt_s, (int, float))
            or not math.isfinite(float(neural_dt_s))
            or not 0.0 < float(neural_dt_s)
        ):
            raise ValueError("LIF core dt must be positive")

        self.core = core
        self.horizon = horizon
        self.num_envs = num_envs
        self.num_neurons = num_neurons
        self.control_dt_s = float(control_dt_s)
        self.neural_substeps = neural_substeps
        self.neural_dt_s = float(neural_dt_s)
        # Float32 counts are exact for all configured rollout sample counts and
        # avoid slow FP64 reductions on consumer GPUs.
        self._spike_counts = torch.zeros(num_neurons, dtype=torch.float32, device=weights.device)
        self._stats = torch.zeros(self._STAT_COUNT, dtype=torch.float32, device=weights.device)
        self._forward_calls = 0
        self._counted_calls = 0
        self._ignored_calls = 0
        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def retained_tensor_numel(self) -> int:
        """Number of persistent device values; independent of rollout length."""

        return self._spike_counts.numel() + self._stats.numel()

    def _observe(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self._forward_calls += 1
        if self._counted_calls >= self.horizon:
            self._ignored_calls += 1
            return
        required = ("membrane", "spikes", "synapse")
        if any(not hasattr(output, name) for name in required):
            raise TypeError("LIF core hook output lacks membrane, spikes, or synapse")
        membrane = output.membrane.detach()
        spikes = output.spikes.detach()
        synapse = output.synapse.detach()
        expected_shape = (self.num_envs, self.num_neurons)
        if membrane.shape != expected_shape or spikes.shape != expected_shape or synapse.shape != expected_shape:
            raise RuntimeError(
                "LIF activity state shape mismatch: expected "
                f"{expected_shape}, got membrane={tuple(membrane.shape)}, "
                f"spikes={tuple(spikes.shape)}, synapse={tuple(synapse.shape)}"
            )
        if membrane.device != self._stats.device or spikes.device != self._stats.device:
            raise RuntimeError("LIF activity tensors moved away from the controller device")
        if synapse.device != self._stats.device:
            raise RuntimeError("LIF synapse tensor moved away from the controller device")

        finite_spikes = torch.isfinite(spikes)
        self._spike_counts.add_(torch.where(finite_spikes, spikes, 0.0).float().sum(dim=0))
        membrane_l2 = torch.linalg.vector_norm(membrane.float(), dim=-1)
        synapse_l2 = torch.linalg.vector_norm(synapse.float(), dim=-1)
        self._stats[self._MEMBRANE_L2_SUM].add_(membrane_l2.sum())
        self._stats[self._MEMBRANE_L2_SQUARE_SUM].add_(membrane_l2.square().sum())
        self._stats[self._MEMBRANE_L2_MAX].copy_(
            torch.maximum(self._stats[self._MEMBRANE_L2_MAX], membrane_l2.max())
        )
        self._stats[self._SYNAPSE_L2_SUM].add_(synapse_l2.sum())
        self._stats[self._SYNAPSE_L2_SQUARE_SUM].add_(synapse_l2.square().sum())
        self._stats[self._SYNAPSE_L2_MAX].copy_(
            torch.maximum(self._stats[self._SYNAPSE_L2_MAX], synapse_l2.max())
        )
        nonfinite = (
            (~torch.isfinite(membrane)).sum()
            + (~finite_spikes).sum()
            + (~torch.isfinite(synapse)).sum()
        )
        nonbinary = (finite_spikes & (spikes != 0.0) & (spikes != 1.0)).sum()
        self._stats[self._NONFINITE_VALUE_COUNT].add_(nonfinite.float())
        self._stats[self._NONBINARY_SPIKE_COUNT].add_(nonbinary.float())
        self._counted_calls += 1

    def __enter__(self) -> "RolloutLIFActivityAccumulator":
        if self._hook_handle is not None:
            raise RuntimeError("LIF activity accumulator cannot be entered twice")
        if self._forward_calls or self._counted_calls or self._ignored_calls:
            raise RuntimeError("LIF activity accumulator instances are single-use")
        self._hook_handle = self.core.register_forward_hook(self._observe)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def summary(self) -> dict[str, Any]:
        """Return the validated rollout summary after one batched host transfer."""

        if self._hook_handle is not None:
            raise RuntimeError("Cannot summarize LIF activity while its hook is active")
        expected_calls = self.horizon + self._EXPECTED_BOOTSTRAP_CALLS
        if self._forward_calls != expected_calls:
            raise RuntimeError(
                "Unexpected LIF core call count during rollout: "
                f"expected {expected_calls}, observed {self._forward_calls}"
            )
        if self._counted_calls != self.horizon or self._ignored_calls != self._EXPECTED_BOOTSTRAP_CALLS:
            raise RuntimeError("LIF activity did not count the exact rollout and bootstrap boundaries")

        # This is the only device-to-host synchronization in the accumulator.
        packed = torch.cat((self._spike_counts, self._stats)).detach().cpu()
        counts = packed[: self.num_neurons]
        stats = packed[self.num_neurons :]
        nonfinite_count = int(stats[self._NONFINITE_VALUE_COUNT].item())
        nonbinary_count = int(stats[self._NONBINARY_SPIKE_COUNT].item())
        if nonfinite_count:
            raise RuntimeError(f"LIF rollout activity contains {nonfinite_count} nonfinite state values")
        if nonbinary_count:
            raise RuntimeError(f"LIF rollout activity contains {nonbinary_count} nonbinary spike values")
        if not torch.isfinite(packed).all().item():
            raise RuntimeError("LIF rollout activity accumulator produced a nonfinite summary")

        sample_count = self.horizon * self.num_envs
        total_spikes = float(counts.sum().item())
        spike_fraction = total_spikes / (sample_count * self.num_neurons)
        saturated_count = int((counts == sample_count).sum().item())
        dead_count = int((counts == 0).sum().item())
        membrane_mean = float(stats[self._MEMBRANE_L2_SUM].item()) / sample_count
        synapse_mean = float(stats[self._SYNAPSE_L2_SUM].item()) / sample_count
        membrane_rms = (
            float(stats[self._MEMBRANE_L2_SQUARE_SUM].item()) / sample_count
        ) ** 0.5
        synapse_rms = (
            float(stats[self._SYNAPSE_L2_SQUARE_SUM].item()) / sample_count
        ) ** 0.5
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sampling_semantics": "post_final_neural_substep_once_per_control_decision",
            "bootstrap_excluded": True,
            "saturation_definition": "spiked_in_every_env_at_every_counted_control_decision",
            "norm_definition": "l2_across_neurons_then_summarized_across_env_control_samples",
            "sampled_spike_rate_definition": "sampled_spike_fraction_divided_by_control_dt_s",
            "control_dt_s": self.control_dt_s,
            "neural_dt_s": self.neural_dt_s,
            "neural_substeps_per_control_step": self.neural_substeps,
            "rollout_control_steps": self.horizon,
            "environment_count": self.num_envs,
            "neuron_count": self.num_neurons,
            "neuron_sample_count": sample_count,
            "core_forward_call_count": self._forward_calls,
            "ignored_bootstrap_forward_call_count": self._ignored_calls,
            "rollout_total_sampled_spikes": int(total_spikes),
            "rollout_sampled_spike_fraction": spike_fraction,
            "rollout_sampled_spike_rate_hz_per_neuron": spike_fraction / self.control_dt_s,
            "rollout_dead_neuron_fraction": dead_count / self.num_neurons,
            "rollout_saturated_neuron_fraction": saturated_count / self.num_neurons,
            "rollout_per_neuron_spike_count_min": int(counts.min().item()),
            "rollout_per_neuron_spike_count_mean": float(counts.mean().item()),
            "rollout_per_neuron_spike_count_max": int(counts.max().item()),
            "rollout_membrane_l2_mean": membrane_mean,
            "rollout_membrane_l2_rms": membrane_rms,
            "rollout_membrane_l2_max": float(stats[self._MEMBRANE_L2_MAX].item()),
            "rollout_synapse_l2_mean": synapse_mean,
            "rollout_synapse_l2_rms": synapse_rms,
            "rollout_synapse_l2_max": float(stats[self._SYNAPSE_L2_MAX].item()),
            "nonfinite_state_value_count": nonfinite_count,
            "nonbinary_spike_value_count": nonbinary_count,
            "retained_device_value_count": self.retained_tensor_numel,
        }


__all__ = ["RolloutLIFActivityAccumulator", "scheduled_measurement_due"]
