import pytest
import torch

from g1_fly_control.crazyflie.controllers import (
    CRAZYFLIE_HOVER_ACTION,
    CRAZYFLIE_RESIDUAL_LATENT_SCALE,
    CrazyflieFrozenLIFActorCritic,
    MatchedGRUActorCritic,
    MatchedMLPActorCritic,
    build_controller,
    controller_core_checksum,
)
from g1_fly_control.crazyflie.stabilization import (
    FIXED_OBSERVATION_CLIP,
    FIXED_OBSERVATION_SCALE,
    PRIOR_ACTION_ATANH_EPS,
    compose_crazyflie_residual_mean,
    crazyflie_stabilization_prior_action,
    crazyflie_stabilization_prior_latent,
    normalize_physical_observation,
    reconstruct_physical_observation,
    stabilization_contract_payload,
)
from g1_fly_control.policies.actor_critic import FrozenLIFActorCritic


def _upright_raw(batch: int = 1) -> torch.Tensor:
    observation = torch.zeros(batch, 12)
    observation[:, 8] = -1.0
    return observation


def _actor_head(policy: torch.nn.Module) -> torch.nn.Linear:
    if isinstance(policy, CrazyflieFrozenLIFActorCritic):
        return policy.decoder[-1]
    if isinstance(policy, MatchedGRUActorCritic):
        return policy.actor
    assert isinstance(policy, MatchedMLPActorCritic)
    return policy.actor[-1]


def test_fixed_physics_scaling_round_trip_and_clip() -> None:
    scale = torch.tensor(FIXED_OBSERVATION_SCALE)
    raw = scale * torch.linspace(-4.5, 4.5, 12)
    normalized = normalize_physical_observation(raw)
    torch.testing.assert_close(normalized, raw / scale)
    torch.testing.assert_close(reconstruct_physical_observation(normalized), raw)

    extreme = torch.stack((-10.0 * scale, 10.0 * scale))
    clipped = normalize_physical_observation(extreme)
    assert clipped.min().item() == -FIXED_OBSERVATION_CLIP
    assert clipped.max().item() == FIXED_OBSERVATION_CLIP


def test_upright_stationary_prior_is_exact_hover() -> None:
    normalized = normalize_physical_observation(_upright_raw(3))
    prior = crazyflie_stabilization_prior_action(normalized)
    expected = torch.tensor(CRAZYFLIE_HOVER_ACTION).expand_as(prior)
    torch.testing.assert_close(prior, expected, rtol=0.0, atol=1.0e-7)


def test_prior_exact_mapping_signs_and_goal_independence() -> None:
    raw = _upright_raw(2)
    raw[:, 0:3] = torch.tensor([1.5, -2.0, -0.5])
    raw[:, 3:6] = torch.tensor([0.25, -0.4, 0.7])
    raw[:, 6:9] = torch.tensor([0.2, -0.3, -0.9])
    raw[0, 9:12] = torch.tensor([-1.0, 0.5, 2.0])
    raw[1, 9:12] = torch.tensor([3.0, -4.0, -2.0])
    normalized = normalize_physical_observation(raw)
    prior = crazyflie_stabilization_prior_action(normalized)

    expected = torch.tensor(
        [
            CRAZYFLIE_HOVER_ACTION[0] - 0.18 * -0.5 + 0.20 * (1.0 - 0.9),
            0.08 * -0.3 - 0.02 * 0.25 + 0.015 * -2.0,
            -0.08 * 0.2 - 0.02 * -0.4 - 0.015 * 1.5,
            -0.01 * 0.7,
        ]
    )
    torch.testing.assert_close(prior[0], expected, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(prior[1], expected, rtol=0.0, atol=1.0e-7)


def test_prior_is_finite_and_strictly_inside_tanh_inverse_domain() -> None:
    normalized = torch.full((4, 12), FIXED_OBSERVATION_CLIP)
    normalized[1::2].neg_()
    prior = crazyflie_stabilization_prior_action(normalized)
    latent = crazyflie_stabilization_prior_latent(normalized)
    limit = 1.0 - PRIOR_ACTION_ATANH_EPS
    assert torch.isfinite(prior).all() and torch.isfinite(latent).all()
    assert float(prior.abs().max()) <= limit
    torch.testing.assert_close(torch.tanh(latent), prior)


def test_bounded_residual_latent_composition_is_exact() -> None:
    observation = normalize_physical_observation(_upright_raw(2))
    logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [100.0, -100.0, 100.0, -100.0]]
    )
    composed = compose_crazyflie_residual_mean(observation, logits)
    prior = crazyflie_stabilization_prior_latent(observation)
    torch.testing.assert_close(composed[0], prior[0])
    expected_delta = torch.tensor(CRAZYFLIE_RESIDUAL_LATENT_SCALE) * torch.tensor(
        [1.0, -1.0, 1.0, -1.0]
    )
    torch.testing.assert_close(composed[1] - prior[1], expected_delta)


def test_plan_v2_contract_records_prior_and_residual_exactly() -> None:
    contract = stabilization_contract_payload()
    assert contract["version"] == (
        "crazyflie_shared_stabilization_bounded_residual_v2"
    )
    assert contract["prior"]["goal_independent"] is True
    assert contract["prior"]["ignored_goal_indices"] == [9, 10, 11]
    assert "vertical_waypoint_servo" not in contract
    assert contract["residual"]["latent_scale"] == list(
        CRAZYFLIE_RESIDUAL_LATENT_SCALE
    )


@pytest.mark.parametrize(
    "kind", ("frozen_lif", "frozen_lif_rewired", "gru", "mlp")
)
def test_all_controller_conditions_use_identical_fixed_stack_and_counts(kind: str) -> None:
    policy, report = build_controller(kind)
    trainable_before = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    core_before = controller_core_checksum(policy)
    head = _actor_head(policy)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

    observation = normalize_physical_observation(_upright_raw(2))
    observation[:, 0:9] += torch.tensor(
        [0.2, -0.1, 0.05, 0.02, -0.03, 0.04, 0.1, -0.2, 0.05]
    )
    state = policy.initial_state(2) if getattr(policy, "is_recurrent", False) else None
    output = policy.act(observation, state, deterministic=True)
    expected = torch.tanh(
        compose_crazyflie_residual_mean(observation, torch.zeros(2, 4))
    )
    torch.testing.assert_close(output.action, expected, rtol=0.0, atol=1.0e-7)

    assert not policy.crazyflie_residual_latent_scale.requires_grad
    assert tuple(policy.crazyflie_residual_latent_scale.tolist()) == pytest.approx(
        CRAZYFLIE_RESIDUAL_LATENT_SCALE
    )
    assert sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    ) == trainable_before == report["total_trainable_parameters"]
    assert controller_core_checksum(policy) == core_before
    expected_counts = {
        "frozen_lif": (4_776, 18_305, 23_081),
        "frozen_lif_rewired": (4_776, 18_305, 23_081),
        "gru": (4_793, 18_305, 23_098),
        "mlp": (4_827, 18_305, 23_132),
    }
    assert (
        report["actor_trainable_parameters"],
        report["critic_trainable_parameters"],
        report["total_trainable_parameters"],
    ) == expected_counts[kind]


@pytest.mark.parametrize(
    "kind", ("frozen_lif", "frozen_lif_rewired", "gru", "mlp")
)
def test_trainable_residual_path_itself_retains_all_goal_gradients(kind: str) -> None:
    """Do not let fixed-servo gradients mask a disconnected learned actor."""

    torch.manual_seed(91)
    policy, _ = build_controller(kind)
    observation = torch.tensor(
        [[0.1, -0.05, 0.02, 0.01, -0.02, 0.03, 0.0, 0.0, -1.0, 0.2, -0.15, 0.3]],
        requires_grad=True,
    )
    if isinstance(policy, CrazyflieFrozenLIFActorCritic):
        state = policy.initial_state(1)
        for _ in range(12):
            residual_logits, state = FrozenLIFActorCritic._mean_and_state(
                policy, observation, state
            )
    elif isinstance(policy, MatchedGRUActorCritic):
        state = policy.initial_state(1)
        for _ in range(12):
            state = policy.gru(observation, state)
        residual_logits = policy.actor(state)
    else:
        assert isinstance(policy, MatchedMLPActorCritic)
        residual_logits = policy.actor(observation)

    gradient = torch.autograd.grad(residual_logits.sum(), observation)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, 9:12]) == 3


@pytest.mark.parametrize("kind", ("frozen_lif", "gru", "mlp"))
def test_composed_distribution_log_probability_replays_exactly(kind: str) -> None:
    torch.manual_seed(20260916)
    policy, _ = build_controller(kind)
    observation = normalize_physical_observation(_upright_raw(3))
    observation[:, 9:12] = torch.randn(3, 3)
    state = policy.initial_state(3) if getattr(policy, "is_recurrent", False) else None
    output = policy.act(observation, state)
    replay_state = policy.initial_state(3) if getattr(policy, "is_recurrent", False) else None
    log_prob, _entropy, _value, _ = policy.evaluate_actions(
        observation, output.action, replay_state
    )
    # Collection has the sampled pre-tanh latent while replay reconstructs it
    # with atanh(action), so the existing distribution permits float32
    # round-off at a few ulps while preserving the exact density equation.
    torch.testing.assert_close(log_prob, output.log_prob, rtol=0.0, atol=5.0e-6)
