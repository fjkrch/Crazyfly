from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from g1_fly_control.connectome import load_connectome
from g1_fly_control.policies import FrozenLIFActorCritic, GRUActorCritic, LIFCore, MLPActorCritic
from g1_fly_control.training import PPOConfig, RecurrentPPO


class _ToyVectorEnv:
    """CPU-only rollout harness; it does not claim Isaac physics behavior."""

    def __init__(self):
        self.unwrapped = self
        self.step_count = 0
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return {"policy": torch.ones(2, 3)}, {}

    def step(self, action):
        self.step_count += 1
        observation = torch.ones(2, 3) * (1 + self.step_count * 0.01)
        return {"policy": observation}, -action.square().sum(dim=-1), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}


def test_frozen_lif_recurrent_ppo_replays_actions_and_preserves_core():
    circuit = load_connectome("tests/fixtures/synthetic_circuit/manifest.json", allow_synthetic=True)
    core = LIFCore(circuit.num_neurons, circuit.edge_index, circuit.weights, neural_substeps=2)
    policy = FrozenLIFActorCritic(3, 2, core, input_indices=[0], output_indices=[3])
    runner = RecurrentPPO(policy, PPOConfig(horizon=4, ppo_epochs=1, burn_in=1))
    before = core.frozen_checksum
    env = _ToyVectorEnv()
    rollout, _, _ = runner.collect(env)
    metrics = runner.update(rollout)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert core.frozen_checksum == before
    second_rollout, _, _ = runner.collect(env)
    assert env.reset_count == 1
    assert env.step_count == 8
    assert not second_rollout.reset_before[0].any()
    assert second_rollout.initial_state is not None
    assert second_rollout.initial_state.membrane.abs().sum() > 0
    second_metrics = runner.update(second_rollout)
    assert torch.isfinite(torch.tensor(second_metrics["loss"]))


def test_large_lif_graph_requires_validated_deterministic_replay_backend():
    core = LIFCore(1025, torch.empty((2, 0), dtype=torch.long), torch.empty(0))
    policy = FrozenLIFActorCritic(3, 2, core, input_indices=[0], output_indices=[1])
    with pytest.raises(ValueError, match="deterministic recurrent accumulation"):
        RecurrentPPO(policy)


def test_gru_rollout_preserves_state_across_update_boundary():
    env = _ToyVectorEnv()
    policy = GRUActorCritic(3, 2, hidden_dim=8)
    runner = RecurrentPPO(policy, PPOConfig(horizon=3, ppo_epochs=1))
    first, _, _ = runner.collect(env)
    runner.update(first)
    second, _, _ = runner.collect(env)
    assert env.reset_count == 1
    assert env.step_count == 6
    assert not second.reset_before[0].any()
    assert second.initial_state is not None
    assert second.initial_state.abs().sum() > 0
    metrics = runner.update(second)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["rejected_step"] == 0.0
    assert metrics["accepted_epochs"] == 1.0
    assert metrics["approx_kl"] == metrics["attempted_kl"]
    assert metrics["approx_kl"] > 0.0


def _assert_same_state(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_state(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for item, saved in zip(actual, expected, strict=True):
            _assert_same_state(item, saved)
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


def test_kl_gate_restores_policy_and_populated_adam_state():
    torch.manual_seed(7)
    env = _ToyVectorEnv()
    runner = RecurrentPPO(GRUActorCritic(3, 2, hidden_dim=8), PPOConfig(horizon=4, ppo_epochs=1))
    warmup, _, _ = runner.collect(env)
    runner.update(warmup)
    assert runner.optimizer.state_dict()["state"]  # Adam moments exist before guarded step.

    runner.config = replace(runner.config, target_kl=1e-10)
    runner.optimizer.param_groups[0]["lr"] = 0.1
    before_policy = deepcopy(runner.policy.state_dict())
    before_optimizer = deepcopy(runner.optimizer.state_dict())
    rollout, _, _ = runner.collect(env)
    metrics = runner.update(rollout)

    assert metrics["rejected_step"] == 1.0
    assert metrics["accepted_epochs"] == 0.0
    assert metrics["attempted_kl"] > runner.config.target_kl
    assert metrics["approx_kl"] <= runner.config.target_kl
    _assert_same_state(runner.policy.state_dict(), before_policy)
    _assert_same_state(runner.optimizer.state_dict(), before_optimizer)


def test_nonfinite_gradient_stops_before_optimizer_step():
    torch.manual_seed(9)
    env = _ToyVectorEnv()
    runner = RecurrentPPO(MLPActorCritic(3, 2, hidden_dim=8), PPOConfig(horizon=4, ppo_epochs=1))
    before_policy = deepcopy(runner.policy.state_dict())
    rollout, _, _ = runner.collect(env)
    hook = runner.policy.log_std.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            runner.update(rollout)
    finally:
        hook.remove()
    _assert_same_state(runner.policy.state_dict(), before_policy)
    assert not runner.optimizer.state_dict()["state"]
