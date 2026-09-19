import torch

from g1_fly_control.training.recurrent_storage import RecurrentRolloutBuilder


def test_rollout_masks_and_returns_are_finite():
    builder = RecurrentRolloutBuilder(torch.ones(2, dtype=torch.bool))
    for step in range(3):
        builder.append(torch.ones(2, 4), torch.zeros(2, 1), torch.ones(2), torch.tensor([False, step == 1]), torch.zeros(2, dtype=torch.bool), torch.zeros(2), torch.zeros(2))
    rollout = builder.build()
    assert rollout.reset_before[0].all()
    assert rollout.reset_before[2, 1]
    rollout.compute_returns(torch.zeros(2), gamma=0.99, gae_lambda=0.95)
    assert torch.isfinite(rollout.advantages).all()

