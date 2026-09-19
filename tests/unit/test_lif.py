import torch

from g1_fly_control.policies import FrozenLIFActorCritic, LIFCore
from g1_fly_control.policies.gru import GRUActorCritic


def core():
    return LIFCore(4, torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([0.5, -0.25, 0.75]), neural_substeps=2)


def test_recurrent_current_matches_reference_and_core_is_frozen():
    model = core()
    spikes = torch.randn(3, 4)
    assert torch.allclose(model._recurrent_current(spikes), model.dense_recurrent_current(spikes))
    before = model.frozen_checksum
    policy = FrozenLIFActorCritic(5, 2, model, input_indices=[0, 1], output_indices=[2, 3])
    result = policy.act(torch.randn(3, 5), policy.initial_state(3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    optimizer.zero_grad()
    (result.mean.square().mean() + result.value.mean()).backward()
    assert policy.encoder[0].weight.grad is not None
    optimizer.step()
    assert model.frozen_checksum == before


def test_small_core_dense_current_matches_ordered_edges_and_input_gradient():
    # Duplicate edges exercise the precomputed aggregation as well as signs.
    edge_index = torch.tensor([[0, 0, 1, 2, 2], [1, 1, 2, 0, 1]])
    weights = torch.tensor([0.5, -0.2, 0.75, -1.0, 0.1])
    model = LIFCore(3, edge_index, weights)
    assert model.dense_recurrent_weights.shape == (3, 3)
    assert "dense_recurrent_weights" not in model.state_dict()
    spikes = torch.tensor([[0.2, 0.4, -0.1], [0.8, -0.5, 0.3]], requires_grad=True)
    messages = spikes[:, edge_index[0]] * weights
    edge_result = torch.zeros_like(spikes).index_add(1, edge_index[1], messages)
    dense_result = model._recurrent_current(spikes)
    assert torch.allclose(dense_result, edge_result, atol=1e-7)
    dense_result.sum().backward(retain_graph=True)
    dense_gradient = spikes.grad.clone()
    spikes.grad.zero_()
    edge_result.sum().backward()
    assert torch.allclose(dense_gradient, spikes.grad, atol=1e-7)


def test_derived_dense_current_rebuilds_after_checkpoint_load():
    first = LIFCore(3, torch.tensor([[0, 1], [1, 2]]), torch.tensor([0.5, -0.3]))
    second = LIFCore(3, torch.tensor([[2, 0], [1, 2]]), torch.tensor([0.9, 0.1]))
    spikes = torch.tensor([[1.0, 2.0, 3.0]])
    expected = second._recurrent_current(spikes).clone()
    assert not torch.equal(first._recurrent_current(spikes), expected)
    first.load_state_dict(second.state_dict())
    assert torch.equal(first._recurrent_current(spikes), expected)
    assert first.frozen_checksum == second.frozen_checksum


def test_reset_isolation_and_checkpoint_deterministic_action():
    torch.manual_seed(7)
    model = core()
    state = model.initial_state(2)
    advanced = model(torch.ones(2, 4), state)
    reset = model.reset(advanced, [0])
    assert torch.all(reset.membrane[0] == 0)
    assert torch.allclose(reset.membrane[1], advanced.membrane[1])
    policy = FrozenLIFActorCritic(5, 2, model, input_indices=[0], output_indices=[3])
    observation = torch.ones(1, 5)
    first = policy.act(observation, policy.initial_state(1), deterministic=True).action
    second = policy.act(observation, policy.initial_state(1), deterministic=True).action
    assert torch.equal(first, second)


def test_gru_sequence_replays_stored_log_probabilities():
    torch.manual_seed(12)
    policy = GRUActorCritic(5, 2, hidden_dim=8)
    state = policy.initial_state(3)
    observations = torch.randn(4, 3, 5)
    actions, stored_log_probs = [], []
    reset_mask = torch.zeros(4, 3, dtype=torch.bool)
    reset_mask[0] = True
    reset_mask[2, 1] = True
    with torch.no_grad():
        for step in range(4):
            state = state.masked_fill(reset_mask[step, :, None], 0.0)
            output = policy.act(observations[step], state)
            actions.append(output.action)
            stored_log_probs.append(output.log_prob)
            state = output.state
    replay, _, _, _ = policy.evaluate_sequence(observations, torch.stack(actions), policy.initial_state(3), reset_mask)
    assert torch.allclose(replay, torch.stack(stored_log_probs), atol=1e-5)


def test_frozen_core_passes_nonzero_gradient_to_encoder_and_decoder():
    model = LIFCore(2, torch.tensor([[0], [1]]), torch.tensor([1.0]), threshold=0.01, neural_substeps=4)
    policy = FrozenLIFActorCritic(3, 1, model, input_indices=[0], output_indices=[1])
    for module in list(policy.encoder.modules()) + list(policy.decoder.modules()):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.constant_(module.weight, 0.2)
            torch.nn.init.constant_(module.bias, 0.2)
    output = policy.act(torch.ones(2, 3), policy.initial_state(2), deterministic=True)
    output.mean.sum().backward()
    assert policy.encoder[0].weight.grad.abs().sum() > 0
    assert policy.decoder[-1].weight.grad.abs().sum() > 0
