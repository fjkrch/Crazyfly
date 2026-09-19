import torch

from g1_fly_control.tasks.g1.logic import bounded_position_targets, default_centered_targets, initial_goal_state, invalid_simulation, update_goal_reward, workspace_escape


def test_action_targets_are_hard_bounded():
    result = bounded_position_targets(torch.tensor([[-2.0, 0.0, 3.0]]), torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(result, torch.tensor([[-1.0, 0.0, 3.0]]))
    centered = default_centered_targets(
        torch.tensor([[-2.0, 0.0, 3.0]]),
        torch.tensor([[-1.0, -2.0, -3.0]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[0.5, 1.5, -1.0]]),
    )
    assert torch.equal(centered, torch.tensor([[-1.0, 1.5, 3.0]]))


def test_non_upright_or_torso_contact_is_not_a_termination_signal():
    # This task's pure termination logic only has invalid numerical state/workspace terms.
    root_xy = torch.tensor([[0.0, 0.0]])
    assert not workspace_escape(root_xy, torch.zeros_like(root_xy), radius=8.0).item()
    prone_orientation_and_torso_contact = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert not invalid_simulation(prone_orientation_and_torso_contact).item()


def test_goal_success_requires_sustained_hold_and_reset_prevents_progress_spike():
    state = initial_goal_state(1, "cpu")
    root = torch.zeros(1, 2)
    state.reset(torch.tensor([0]), root, torch.tensor([[1.0, 0.0]]))
    state.reset(torch.tensor([0]), root, torch.tensor([[0.1, 0.0]]))
    first_progress, success, _ = update_goal_reward(state, root, hold_steps=3)
    assert first_progress.item() == 0.0
    assert not success.item()
    update_goal_reward(state, root, hold_steps=3)
    _, success, _ = update_goal_reward(state, root, hold_steps=3)
    assert success.item() == 1.0
    _, success, _ = update_goal_reward(state, root, hold_steps=3)
    assert not success.item()
