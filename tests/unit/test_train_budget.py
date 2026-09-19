"""Budget and model-size accounting without launching Isaac Sim."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from train import policy_parameter_counts, resolve_training_budget  # noqa: E402


def test_interaction_budget_rounds_up_to_complete_rollouts():
    assert resolve_training_budget(num_envs=16, horizon=32, max_iterations=10, interaction_budget=1024) == (2, 1024)
    assert resolve_training_budget(num_envs=16, horizon=32, max_iterations=10, interaction_budget=1025) == (3, 1536)
    assert resolve_training_budget(num_envs=16, horizon=32, max_iterations=10, interaction_budget=1) == (1, 512)
    assert resolve_training_budget(num_envs=16, horizon=32, max_iterations=10, interaction_budget=None) == (10, 5120)


@pytest.mark.parametrize("values", [
    {"num_envs": 0, "horizon": 32, "max_iterations": 10, "interaction_budget": None},
    {"num_envs": 16, "horizon": 0, "max_iterations": 10, "interaction_budget": None},
    {"num_envs": 16, "horizon": 32, "max_iterations": 0, "interaction_budget": None},
    {"num_envs": 16, "horizon": 32, "max_iterations": 10, "interaction_budget": 0},
])
def test_interaction_budget_rejects_nonpositive_inputs(values):
    with pytest.raises(ValueError):
        resolve_training_budget(**values)


def test_parameter_counts_include_fixed_synapses_but_not_graph_indices():
    from g1_fly_control.policies import FrozenLIFActorCritic, LIFCore

    core = LIFCore(
        3,
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([0.3, 0.4]),
    )
    policy = FrozenLIFActorCritic(4, 2, core, input_indices=[0], output_indices=[2])
    counts = policy_parameter_counts(policy)
    assert counts["model_total_parameters"] == counts["model_trainable_parameters"] + 2
    assert counts["model_frozen_parameters"] == 2
