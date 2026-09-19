"""CPU-only contracts for the additive all-connectome LIF combinations."""

from pathlib import Path

import pytest
import torch

from g1_fly_control.crazyflie.controllers import (
    CrazyflieCombinedConnectomeActorCritic,
    build_controller,
    controller_core_checksum,
    controller_core_checksums,
    named_lif_core_indices,
    named_lif_cores,
    named_lif_encoders,
    reset_controller_state,
    verify_frozen_core,
)


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (
        "leg_optic_lif",
        ("leg", "optic"),
        9_224,
        6_731,
        2_048,
        "independent_leg_and_optic_cores_concat_motor_readouts_v1",
    ),
    (
        "wing_optic_lif",
        ("wing", "optic"),
        9_224,
        10_492,
        2_048,
        "independent_wing_and_optic_cores_concat_motor_readouts_v1",
    ),
    (
        "leg_wing_optic_lif",
        ("leg", "wing", "optic"),
        13_672,
        15_595,
        3_072,
        "independent_leg_and_wing_and_optic_cores_concat_motor_readouts_v1",
    ),
)


@pytest.mark.parametrize(
    ("kind", "labels", "actor", "frozen", "dynamic_state", "fusion"), CASES
)
def test_combination_counts_provenance_and_ordered_core_api(
    kind, labels, actor, frozen, dynamic_state, fusion
):
    policy, report = build_controller(kind)

    assert isinstance(policy, CrazyflieCombinedConnectomeActorCritic)
    assert tuple(report["core_labels"]) == labels
    assert tuple(label for label, _core in named_lif_cores(policy)) == labels
    assert tuple(label for label, _inputs, _outputs in named_lif_core_indices(policy)) == labels
    assert tuple(label for label, _encoder in named_lif_encoders(policy)) == labels
    assert tuple(report["connectome_manifests"]) == labels
    assert tuple(report["connectome_checksums"]) == labels
    assert tuple(report["connectome_manifest_fingerprints"]) == labels
    assert tuple(report["per_core_checksums"]) == labels
    assert tuple(report["per_core_population_indices"]) == labels
    assert all(
        Path(path)
        == ROOT
        / "data"
        / ("connectome" if label == "leg" else f"connectome_{label}")
        / "manifest.json"
        for label, path in report["connectome_manifests"].items()
    )

    assert report["fusion_contract"] == fusion
    assert report["actor_trainable_parameters"] == actor
    assert report["critic_trainable_parameters"] == 18_305
    assert report["total_trainable_parameters"] == actor + 18_305
    assert report["frozen_synaptic_weights"] == frozen
    assert report["frozen_parameters"] == frozen
    assert report["total_dynamic_state_per_environment"] == dynamic_state
    assert report["continuous_dynamic_state_per_environment"] == 3 * dynamic_state // 4
    assert report["discrete_dynamic_state_per_environment"] == dynamic_state // 4
    assert report["input_population_size"] == 32 * len(labels)
    assert report["output_population_size"] == 24 * len(labels)
    assert policy.decoder[0].in_features == 24 * len(labels)

    checksum = controller_core_checksum(policy)
    assert checksum == report["core_checksum"]
    assert controller_core_checksums(policy) == report["per_core_checksums"]
    verify_frozen_core(policy, checksum)
    for (label, core), (_index_label, inputs, outputs) in zip(
        named_lif_cores(policy), named_lif_core_indices(policy), strict=True
    ):
        assert not any(True for _ in core.parameters())
        assert core.weights.requires_grad is False
        assert len(report["per_core_checksums"][label]) == 64
        assert report["per_core_population_indices"][label][
            "input_indices"
        ] == inputs.tolist()
        assert report["per_core_population_indices"][label][
            "output_indices"
        ] == outputs.tolist()


@pytest.mark.parametrize(
    ("kind", "labels", "actor", "frozen", "dynamic_state", "fusion"), CASES
)
def test_combination_forward_reset_gradients_and_frozen_checksums(
    kind, labels, actor, frozen, dynamic_state, fusion
):
    del actor, frozen, dynamic_state, fusion
    torch.manual_seed(8128)
    policy, report = build_controller(kind)
    before = report["core_checksum"]
    core_buffer_snapshots = {
        label: {name: value.detach().clone() for name, value in core.named_buffers()}
        for label, core in named_lif_cores(policy)
    }
    observation = torch.randn(3, 12)
    state = policy.initial_state(3)
    assert state.membrane.shape == (3, 256 * len(labels))

    loss = torch.zeros(())
    output = None
    for _ in range(50):
        output = policy.act(observation, state, deterministic=True)
        state = output.state
        loss = loss + output.mean.square().sum()
    assert output is not None
    assert output.action.shape == (3, 4)
    assert torch.isfinite(output.action).all()
    assert output.action.abs().max() <= 1.0
    loss.backward()

    for label, encoder in named_lif_encoders(policy):
        gradients = [
            parameter.grad for parameter in encoder.parameters() if parameter.grad is not None
        ]
        assert gradients, label
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(int(torch.count_nonzero(gradient)) for gradient in gradients) > 0
    assert controller_core_checksum(policy) == before
    for label, core in named_lif_cores(policy):
        assert all(
            torch.equal(core_buffer_snapshots[label][name], value)
            for name, value in core.named_buffers()
        )

    reset = reset_controller_state(policy, output.state, torch.tensor([False, True, False]))
    assert reset is not None
    for field in (reset.membrane, reset.spikes, reset.synapse, reset.refractory):
        assert torch.count_nonzero(field[1]) == 0
    torch.testing.assert_close(reset.membrane[0], output.state.membrane[0])


@pytest.mark.parametrize(
    ("kind", "labels", "actor", "frozen", "dynamic_state", "fusion"), CASES
)
def test_combination_state_dict_reconstructs_exact_policy(
    kind, labels, actor, frozen, dynamic_state, fusion
):
    del actor, frozen, dynamic_state, fusion
    torch.manual_seed(410)
    first, first_report = build_controller(kind)
    state_dict = {name: value.detach().clone() for name, value in first.state_dict().items()}
    torch.manual_seed(999)
    restored, restored_report = build_controller(kind)
    restored.load_state_dict(state_dict, strict=True)

    observation = torch.randn(2, 12)
    first_output = first.act(observation, first.initial_state(2), deterministic=True)
    restored_output = restored.act(observation, restored.initial_state(2), deterministic=True)
    torch.testing.assert_close(first_output.action, restored_output.action)
    torch.testing.assert_close(first_output.mean, restored_output.mean)
    assert first_report["core_checksum"] == restored_report["core_checksum"]
    assert tuple(label for label, _core in named_lif_cores(restored)) == labels


def test_triple_aggregate_checksum_covers_every_independent_core():
    policy, report = build_controller("leg_wing_optic_lif")
    cores = named_lif_cores(policy)
    assert len({id(core) for _label, core in cores}) == 3
    assert len({core.weights.data_ptr() for _label, core in cores}) == 3

    baseline = report["core_checksum"]
    per_core_baseline = controller_core_checksums(policy)
    for label, core in cores:
        original = core.weights[0].detach().clone()
        core.weights[0] += 0.125
        assert controller_core_checksum(policy) != baseline, label
        changed = controller_core_checksums(policy)
        assert changed[label] != per_core_baseline[label]
        assert all(
            changed[other] == per_core_baseline[other]
            for other, _other_core in cores
            if other != label
        )
        core.weights[0].copy_(original)
        assert controller_core_checksum(policy) == baseline
        assert controller_core_checksums(policy) == per_core_baseline


@pytest.mark.parametrize(
    ("alias", "canonical"),
    (
        ("combined_leg_optic_lif", "leg_optic_lif"),
        ("frozen_lif_leg_optic", "leg_optic_lif"),
        ("combined_wing_optic_lif", "wing_optic_lif"),
        ("frozen_lif_wing_optic", "wing_optic_lif"),
        ("combined_leg_wing_optic_lif", "leg_wing_optic_lif"),
        ("frozen_lif_leg_wing_optic", "leg_wing_optic_lif"),
        ("all_connectome_lif", "leg_wing_optic_lif"),
    ),
)
def test_combination_aliases_resolve_to_canonical_kind(alias, canonical):
    _policy, report = build_controller(alias)
    assert report["controller_kind"] == canonical
