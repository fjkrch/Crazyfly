from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from g1_fly_control.crazyflie.lif_activity import (
    RolloutLIFActivityAccumulator,
    scheduled_measurement_due,
)
from g1_fly_control.policies import LIFState


class _Core(nn.Module):
    def __init__(self, num_neurons: int = 3) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.neural_substeps = 5
        self.dt = 0.002
        self.register_buffer("weights", torch.zeros(1))

    def forward(self, state: LIFState) -> LIFState:
        return state


def _state(
    spikes: list[list[float]],
    *,
    membrane: list[list[float]] | None = None,
    synapse: list[list[float]] | None = None,
) -> LIFState:
    spike_tensor = torch.tensor(spikes, dtype=torch.float32)
    membrane_tensor = torch.tensor(membrane, dtype=torch.float32) if membrane is not None else spike_tensor
    synapse_tensor = torch.tensor(synapse, dtype=torch.float32) if synapse is not None else spike_tensor * 2.0
    return LIFState(
        membrane=membrane_tensor,
        spikes=spike_tensor,
        synapse=synapse_tensor,
        refractory=torch.zeros_like(spike_tensor, dtype=torch.long),
    )


def test_rollout_summary_counts_across_steps_and_reset_rows_and_excludes_bootstrap():
    core = _Core()
    accumulator = RolloutLIFActivityAccumulator(core, horizon=2, num_envs=2, control_dt_s=0.02)
    with accumulator:
        core(_state([[1, 0, 1], [0, 0, 1]]))
        # Reset the first environment through the real recurrent-state API.
        # Historical activity remains in the rollout counts instead of being
        # erased when that row is zeroed.
        second = _state([[1, 1, 1], [1, 0, 1]]).masked_reset(torch.tensor([True, False]))
        core(second)
        core(_state([[1, 1, 1], [1, 1, 1]]))  # bootstrap: excluded

    summary = accumulator.summary()
    assert summary["schema_version"] == 1
    assert summary["neuron_sample_count"] == 4
    assert summary["rollout_total_sampled_spikes"] == 5
    assert summary["rollout_sampled_spike_fraction"] == pytest.approx(5 / 12)
    assert summary["rollout_sampled_spike_rate_hz_per_neuron"] == pytest.approx((5 / 12) / 0.02)
    assert summary["rollout_dead_neuron_fraction"] == pytest.approx(1 / 3)
    assert summary["rollout_saturated_neuron_fraction"] == 0.0
    assert summary["rollout_per_neuron_spike_count_min"] == 0
    assert summary["rollout_per_neuron_spike_count_max"] == 3
    assert summary["ignored_bootstrap_forward_call_count"] == 1
    assert summary["retained_device_value_count"] == core.num_neurons + 8


def test_saturation_and_norm_summaries_use_all_environment_control_samples():
    core = _Core(num_neurons=2)
    accumulator = RolloutLIFActivityAccumulator(core, horizon=2, num_envs=1, control_dt_s=0.02)
    with accumulator:
        core(_state([[1, 0]], membrane=[[3, 4]], synapse=[[0, 2]]))
        core(_state([[1, 1]], membrane=[[0, 0]], synapse=[[3, 4]]))
        core(_state([[0, 0]]))

    summary = accumulator.summary()
    assert summary["rollout_saturated_neuron_fraction"] == 0.5
    assert summary["rollout_dead_neuron_fraction"] == 0.0
    assert summary["rollout_membrane_l2_mean"] == pytest.approx(2.5)
    assert summary["rollout_membrane_l2_rms"] == pytest.approx(math.sqrt(12.5))
    assert summary["rollout_membrane_l2_max"] == pytest.approx(5.0)
    assert summary["rollout_synapse_l2_mean"] == pytest.approx(3.5)
    assert summary["rollout_synapse_l2_rms"] == pytest.approx(math.sqrt(14.5))
    assert summary["rollout_synapse_l2_max"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("spikes", "message"),
    [
        ([[float("nan"), 0, 0]], "nonfinite"),
        ([[0.5, 0, 0]], "nonbinary"),
    ],
)
def test_summary_fails_closed_on_invalid_activity(spikes, message):
    core = _Core()
    accumulator = RolloutLIFActivityAccumulator(core, horizon=1, num_envs=1, control_dt_s=0.02)
    with accumulator:
        core(_state(spikes))
        core(_state([[0, 0, 0]]))
    with pytest.raises(RuntimeError, match=message):
        accumulator.summary()


def test_call_count_shape_and_neuron_boundaries_fail_closed():
    core = _Core()
    too_short = RolloutLIFActivityAccumulator(core, horizon=2, num_envs=1, control_dt_s=0.02)
    with too_short:
        core(_state([[0, 0, 0]]))
    with pytest.raises(RuntimeError, match="call count"):
        too_short.summary()

    wrong_shape = RolloutLIFActivityAccumulator(core, horizon=1, num_envs=2, control_dt_s=0.02)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        with wrong_shape:
            core(_state([[0, 0, 0]]))

    with pytest.raises(ValueError, match="1..256"):
        RolloutLIFActivityAccumulator(_Core(num_neurons=257), horizon=1, num_envs=1, control_dt_s=0.02)


def test_hook_is_removed_after_context_exit_and_storage_is_rollout_length_constant():
    core = _Core(num_neurons=256)
    short = RolloutLIFActivityAccumulator(core, horizon=1, num_envs=1, control_dt_s=0.02)
    long = RolloutLIFActivityAccumulator(core, horizon=10_000, num_envs=4, control_dt_s=0.02)
    assert short.retained_tensor_numel == long.retained_tensor_numel == 264

    with short:
        core(_state([[0.0] * 256]))
        core(_state([[0.0] * 256]))
    calls = short._forward_calls
    core(_state([[1.0] * 256]))
    assert short._forward_calls == calls


def test_activity_uses_first_checkpoint_and_final_measurement_cadence():
    sampled = [
        update
        for update in range(1, 251)
        if scheduled_measurement_due(
            update,
            total_updates=250,
            checkpoint_every_updates=100,
        )
    ]
    assert sampled == [1, 100, 200, 250]
    assert all(
        scheduled_measurement_due(update, total_updates=2, checkpoint_every_updates=1)
        for update in (1, 2)
    )
    with pytest.raises(ValueError, match="positive integers"):
        scheduled_measurement_due(0, total_updates=2, checkpoint_every_updates=1)
