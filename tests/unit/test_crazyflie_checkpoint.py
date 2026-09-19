"""CPU-only acceptance tests for Crazyflie controllers and checkpoints."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import g1_fly_control.crazyflie.checkpoint as checkpoint_module
from g1_fly_control.connectome import load_connectome
from g1_fly_control.crazyflie.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointValidationError,
    HistoryReferenceAccumulator,
    ResumeCounters,
    append_history_record,
    append_validated_history_record_inplace,
    build_history_reference,
    build_history_reference_from_segments,
    capture_rng_states,
    checkpoint_sha256,
    link_checkpoint_snapshot,
    load_checkpoint,
    load_history_reference,
    read_checkpoint,
    restore_rng_states,
    save_checkpoint,
    save_checkpoint_boundary,
    validate_history,
    validate_history_alignment,
    validate_interaction_alignment,
    warm_start_actor_from_checkpoint,
    write_history_segment,
)
from g1_fly_control.crazyflie.controllers import (
    CRAZYFLIE_HOVER_ACTION,
    CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE,
    CRAZYFLIE_INITIAL_LATENT_STD,
    CRAZYFLIE_THRUST_TO_WEIGHT,
    DEFAULT_REWIRE_MANIFEST_FILE_SHA256,
    DEFAULT_REWIRE_SEED,
    CrazyflieCombinedLegWingActorCritic,
    CrazyflieFrozenLIFActorCritic,
    MatchedGRUActorCritic,
    MatchedMLPActorCritic,
    build_controller,
    controller_core_checksum,
    default_rewire_manifest_path,
    initialize_crazyflie_actor,
    load_rewire_manifest,
    reset_controller_state,
    rewire_manifest_checksum,
    validate_rewire_manifest,
)
from g1_fly_control.policies import LIFCore


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_circuit" / "manifest.json"
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["frozen_lif", "frozen_lif_rewired", "gru", "mlp"])
def test_all_controllers_are_bounded_matched_and_use_common_critic(kind):
    policy, report = build_controller(
        kind,
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_swaps=1,
    )
    observation = torch.randn(3, 12)
    state = policy.initial_state(3) if getattr(policy, "is_recurrent", False) else None
    output = policy.act(observation, state)
    assert output.action.shape == (3, 4)
    assert torch.isfinite(output.action).all()
    assert (output.action >= -1.0).all() and (output.action <= 1.0).all()
    assert report["actor_parameter_match_passed"]
    assert report["actor_parameter_deviation_fraction"] <= 0.10
    assert report["critic_trainable_parameters"] == 18_305
    assert report["total_trainable_parameters"] == (
        report["actor_trainable_parameters"] + report["critic_trainable_parameters"]
    )


@pytest.mark.parametrize("kind", ["frozen_lif", "frozen_lif_rewired", "gru", "mlp"])
def test_all_controllers_share_hover_safe_actor_initialization(kind):
    torch.manual_seed(1234)
    policy, report = build_controller(
        kind,
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_swaps=1,
    )
    observation = torch.zeros(3, 12)
    observation[:, 8] = -1.0  # fixed-normalized upright projected gravity
    state = policy.initial_state(3) if getattr(policy, "is_recurrent", False) else None
    output = policy.act(observation, state, deterministic=True)
    target = torch.tensor(CRAZYFLIE_HOVER_ACTION).expand_as(output.action)

    assert (CRAZYFLIE_HOVER_ACTION[0] + 1.0) * CRAZYFLIE_THRUST_TO_WEIGHT / 2.0 == pytest.approx(1.0)
    torch.testing.assert_close(output.action, target, rtol=0.0, atol=1.0e-3)
    torch.testing.assert_close(
        policy.log_std.detach(),
        torch.tensor(CRAZYFLIE_INITIAL_LATENT_STD).log(),
        rtol=0.0,
        atol=1.0e-7,
    )

    if isinstance(policy, CrazyflieFrozenLIFActorCritic):
        head = policy.decoder[-1]
    elif isinstance(policy, MatchedGRUActorCritic):
        head = policy.actor
    else:
        assert isinstance(policy, MatchedMLPActorCritic)
        head = policy.actor[-1]
    expected_magnitude = CRAZYFLIE_INITIAL_HEAD_WEIGHT_SCALE / np.sqrt(head.in_features)
    torch.testing.assert_close(
        head.weight.detach().abs(),
        torch.full_like(head.weight, expected_magnitude),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert torch.count_nonzero(head.weight) == head.weight.numel()
    assert torch.linalg.matrix_rank(head.weight.detach()).item() == 4
    assert report["actor_initialization"]["scheme"] == (
        "shared_stabilization_bounded_residual_v2_full_rank_walsh"
    )
    assert report["actor_initialization"]["final_head_sign_pattern"] == (
        "walsh_h8_rows_1_2_3_4_repeated"
    )
    assert report["actor_initialization"]["final_head_weight_rank"] == 4
    assert report["actor_initialization"]["target_deterministic_action"] == list(
        CRAZYFLIE_HOVER_ACTION
    )
    assert report["actor_initialization"]["initial_latent_std"] == list(
        CRAZYFLIE_INITIAL_LATENT_STD
    )
    assert report["actor_initialization"]["final_head_all_weights_nonzero"] is True


@pytest.mark.parametrize("kind", ["frozen_lif", "frozen_lif_rewired"])
def test_hover_initialization_does_not_change_frozen_lif_core(kind):
    policy, report = build_controller(kind)
    before = controller_core_checksum(policy)
    core_buffers = {
        name: value.detach().clone() for name, value in policy.core.named_buffers()
    }
    trainable_before = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    initialization = initialize_crazyflie_actor(policy)
    assert initialization["scheme"] == (
        "shared_stabilization_bounded_residual_v2_full_rank_walsh"
    )
    assert controller_core_checksum(policy) == before == report["core_checksum"]
    assert all(
        torch.equal(core_buffers[name], value)
        for name, value in policy.core.named_buffers()
    )
    assert sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    ) == trainable_before


@pytest.mark.parametrize("kind", ["frozen_lif", "frozen_lif_rewired"])
def test_hover_initialized_lif_keeps_nonzero_encoder_and_decoder_gradients(kind):
    torch.manual_seed(9)
    policy, report = build_controller(kind)
    before = report["core_checksum"]
    observation = torch.ones(2, 12)
    state = policy.initial_state(2)
    loss = torch.zeros(())
    for _ in range(12):
        output = policy.act(observation, state, deterministic=True)
        state = output.state
        loss = loss + output.mean[:, 0].sum()
    loss.backward()

    for parameter in (
        policy.encoder[0].weight,
        policy.decoder[0].weight,
        policy.decoder[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert controller_core_checksum(policy) == before


def test_real_graph_default_widths_and_exact_parameter_counts():
    expected = {
        "frozen_lif": (4_776, 23_081, 5_103, 1_024),
        "gru": (4_793, 23_098, 0, 33),
        "mlp": (4_827, 23_132, 0, 0),
    }
    for kind, counts in expected.items():
        _policy, report = build_controller(kind)
        assert (
            report["actor_trainable_parameters"],
            report["total_trainable_parameters"],
            report["frozen_parameters"],
            report["total_dynamic_state_per_environment"],
        ) == counts


@pytest.mark.parametrize(
    ("kind", "actor", "frozen", "state"),
    [
        ("wing_lif", 4_776, 8_864, 1_024),
        ("leg_wing_lif", 9_224, 13_967, 2_048),
    ],
)
def test_wing_extension_counts_bounds_and_frozen_provenance(kind, actor, frozen, state):
    policy, report = build_controller(kind)
    before = controller_core_checksum(policy)
    recurrent_state = policy.initial_state(3)
    output = policy.act(torch.randn(3, 12), recurrent_state, deterministic=True)

    assert output.action.shape == (3, 4)
    assert torch.isfinite(output.action).all()
    assert output.action.abs().max() <= 1.0
    assert report["parameter_matching_required"] is False
    assert report["actor_trainable_parameters"] == actor
    assert report["frozen_parameters"] == frozen
    assert report["total_dynamic_state_per_environment"] == state
    assert report["widths"]["adapter_hidden_dim"] == 64
    assert controller_core_checksum(policy) == before == report["core_checksum"]

    done = torch.tensor([False, True, False])
    reset = reset_controller_state(policy, output.state, done)
    assert reset is not None
    assert torch.count_nonzero(reset.membrane[1]) == 0
    assert torch.count_nonzero(reset.spikes[1]) == 0
    assert torch.count_nonzero(reset.synapse[1]) == 0
    assert torch.count_nonzero(reset.refractory[1]) == 0
    torch.testing.assert_close(reset.membrane[0], output.state.membrane[0])

    if kind == "wing_lif":
        assert report["connectome_manifests"] == {
            "primary": str(ROOT / "data" / "connectome_wing" / "manifest.json")
        }
        assert set(report["per_core_checksums"]) == {"primary"}
    else:
        assert isinstance(policy, CrazyflieCombinedLegWingActorCritic)
        assert set(report["per_core_checksums"]) == {"leg", "wing"}
        assert report["fusion_contract"] == (
            "independent_leg_and_wing_cores_concat_motor_readouts_v1"
        )
        assert not any(True for _ in policy.core.parameters())
        assert not any(True for _ in policy.wing_core.parameters())


@pytest.mark.parametrize("kind", ["wing_lif", "leg_wing_lif"])
def test_wing_extension_gradients_reach_each_encoder_without_changing_cores(kind):
    torch.manual_seed(9001)
    policy, report = build_controller(kind)
    before = report["core_checksum"]
    observation = torch.randn(2, 12)
    state = policy.initial_state(2)
    loss = torch.zeros(())
    for _ in range(50):
        output = policy.act(observation, state, deterministic=True)
        state = output.state
        loss = loss + output.mean.square().sum()
    loss.backward()

    encoders = [policy.encoder]
    if isinstance(policy, CrazyflieCombinedLegWingActorCritic):
        encoders.append(policy.wing_encoder)
    for encoder in encoders:
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(int(torch.count_nonzero(gradient)) for gradient in gradients) > 0
    assert controller_core_checksum(policy) == before


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("frozen_lif_original", "frozen_lif"),
        ("frozen_lif_degree_rewired", "frozen_lif_rewired"),
        ("gru_matched", "gru"),
        ("mlp_normal", "mlp"),
    ],
)
def test_matrix_condition_aliases_resolve_exactly(alias, canonical):
    _policy, report = build_controller(
        alias,
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_swaps=1,
    )
    assert report["controller_kind"] == canonical


def test_rewire_is_deterministic_and_preserves_declared_constraints():
    original, original_report = build_controller(
        "frozen_lif", connectome_manifest=FIXTURE, allow_synthetic=True
    )
    first, first_report = build_controller(
        "frozen_lif_rewired",
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_seed=117,
        rewire_swaps=1,
    )
    second, second_report = build_controller(
        "frozen_lif_rewired",
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_seed=117,
        rewire_swaps=1,
    )
    assert torch.equal(first.core.edge_index, second.core.edge_index)
    assert torch.equal(first.core.weights, original.core.weights)
    assert first_report["rewire_manifest_checksum"] == second_report["rewire_manifest_checksum"]
    assert first_report["core_checksum"] == second_report["core_checksum"]
    assert first_report["core_checksum"] != original_report["core_checksum"]
    assert all(first_report["rewire_manifest"]["invariants"].values())
    assert first_report["input_population_size"] == original_report["input_population_size"]
    assert first_report["output_population_size"] == original_report["output_population_size"]


def test_checked_in_primary_rewire_matches_fresh_generation_and_configs():
    artifact = default_rewire_manifest_path()
    assert artifact == ROOT / "configs" / "experiments" / "crazyflie_rewire_seed_20260916.json"
    assert sha256(artifact.read_bytes()).hexdigest() == DEFAULT_REWIRE_MANIFEST_FILE_SHA256
    circuit = load_connectome(ROOT / "data" / "connectome" / "manifest.json")
    edge_index, weights, manifest = load_rewire_manifest(
        artifact,
        circuit,
        expected_seed=DEFAULT_REWIRE_SEED,
        expected_file_sha256=DEFAULT_REWIRE_MANIFEST_FILE_SHA256,
        verify_fresh_generation=True,
    )
    assert manifest["checksum"] == "1cdf4e2ac63424ad6af84cd8fc865eceed34e434f02ff918ec31edb523362fa2"
    assert manifest["requested_swaps"] == manifest["completed_swaps"] == 51_030
    assert manifest["attempts"] == 70_456
    assert edge_index.shape == circuit.edge_index.shape == (2, 5_103)
    assert weights.shape == circuit.weights.shape == (5_103,)
    assert all(manifest["invariants"].values())
    assert rewire_manifest_checksum(manifest) == manifest["checksum"]

    policy, report = build_controller("frozen_lif_degree_rewired")
    assert torch.equal(policy.core.edge_index.cpu(), edge_index)
    assert torch.equal(policy.core.weights.cpu(), weights)
    assert report["rewire_manifest_source"] == "frozen_artifact"
    assert report["rewire_manifest_path"] == str(artifact)
    assert report["rewire_manifest_file_sha256"] == DEFAULT_REWIRE_MANIFEST_FILE_SHA256
    assert report["rewire_manifest_checksum"] == manifest["checksum"]

    for name in ("crazyflie_integration.json", "crazyflie_main.json"):
        config = json.loads((ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8"))
        assert config["rewire_seed"] == DEFAULT_REWIRE_SEED
        assert config["rewire_manifest"] == "configs/experiments/crazyflie_rewire_seed_20260916.json"
        assert config["rewire_manifest_sha256"] == DEFAULT_REWIRE_MANIFEST_FILE_SHA256


def test_frozen_rewire_manifest_rejects_tampered_content():
    circuit = load_connectome(ROOT / "data" / "connectome" / "manifest.json")
    manifest = json.loads(default_rewire_manifest_path().read_text(encoding="utf-8"))
    checksum_tamper = deepcopy(manifest)
    checksum_tamper["weights"][0] += 0.125
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_rewire_manifest(checksum_tamper, circuit, expected_seed=DEFAULT_REWIRE_SEED)

    invariant_tamper = deepcopy(manifest)
    invariant_tamper["edge_index"][1][0] = invariant_tamper["edge_index"][0][0]
    invariant_tamper["checksum"] = rewire_manifest_checksum(invariant_tamper)
    with pytest.raises(ValueError, match="failed validation"):
        validate_rewire_manifest(invariant_tamper, circuit, expected_seed=DEFAULT_REWIRE_SEED)


def test_lif_gradient_crosses_frozen_core_and_checksum_stays_fixed():
    core = LIFCore(
        2,
        torch.tensor([[0], [1]]),
        torch.tensor([1.0]),
        threshold=0.01,
        neural_substeps=4,
    )
    policy = CrazyflieFrozenLIFActorCritic(
        12,
        4,
        core,
        input_indices=[0],
        output_indices=[1],
        adapter_hidden_dim=8,
        critic_hidden_dim=16,
    )
    for module in list(policy.encoder.modules()) + list(policy.decoder.modules()):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.constant_(module.weight, 0.2)
            torch.nn.init.constant_(module.bias, 0.2)
    before = controller_core_checksum(policy)
    output = policy.act(torch.ones(2, 12), policy.initial_state(2), deterministic=True)
    output.mean.sum().backward()
    assert policy.encoder[0].weight.grad is not None
    assert torch.isfinite(policy.encoder[0].weight.grad).all()
    assert policy.encoder[0].weight.grad.abs().sum() > 0
    assert policy.decoder[-1].weight.grad is not None
    assert torch.isfinite(policy.decoder[-1].weight.grad).all()
    assert policy.decoder[-1].weight.grad.abs().sum() > 0
    torch.optim.Adam(policy.parameters(), lr=1e-3).step()
    assert controller_core_checksum(policy) == before


def test_recurrent_reset_is_per_environment_and_mlp_has_no_state():
    lif, _ = build_controller("frozen_lif", connectome_manifest=FIXTURE, allow_synthetic=True)
    lif_state = lif.initial_state(2)
    lif_state.membrane[:] = torch.tensor([[1.0], [2.0]])
    lif_state.spikes[:] = 1
    lif_state.synapse[:] = 3
    lif_state.refractory[:] = 1
    lif_reset = reset_controller_state(lif, lif_state, torch.tensor([True, False]))
    assert torch.count_nonzero(lif_reset.membrane[0]) == 0
    assert torch.equal(lif_reset.membrane[1], lif_state.membrane[1])

    gru = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    gru_state = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    gru_reset = reset_controller_state(gru, gru_state, torch.tensor([False, True]))
    assert torch.equal(gru_reset[0], gru_state[0])
    assert torch.count_nonzero(gru_reset[1]) == 0

    mlp = MatchedMLPActorCritic(12, 4, hidden_dims=(8, 8), critic_hidden_dim=16)
    assert reset_controller_state(mlp, None, torch.tensor([True, False])) is None
    assert not any(
        isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase))
        for module in mlp.actor.modules()
    )


def _checkpoint_inputs(policy, optimizer):
    counters = ResumeCounters(completed_updates=2, total_interactions=64, completed_episodes=3)
    history = [
        {"completed_updates": 1, "total_interactions": 32, "loss": 1.0},
        {"completed_updates": 2, "total_interactions": 64, "loss": 0.5},
    ]
    return {
        "policy": policy,
        "optimizer": optimizer,
        "normalizers": {"observation": {"count": 64}, "reward": {"count": 64}},
        "counters": counters,
        "recurrent_state": torch.arange(14, dtype=torch.float32).reshape(2, 7),
        "resolved_config": {"task": "FlyCrazyflie-WaypointReach-v0", "num_envs": 2},
        "command": ["drone_train.py", "--seed", "0"],
        "task_manifest_id": "task-manifest-test",
        "evaluation_manifest_id": "evaluation-manifest-test",
        "fingerprints": {"code": "abc", "controller": "def", "environment": "ghi"},
        "history": history,
        "interactions_per_update": 32,
    }


def _warm_start_checkpoint_inputs(policy, optimizer, *, controller, report):
    inputs = _checkpoint_inputs(policy, optimizer)
    inputs["resolved_config"] = {
        **inputs["resolved_config"],
        "controller": controller,
        "survival_first_contract": {"version": "source-contract"},
    }
    inputs["fingerprints"] = {
        "reproduction": "source-reproduction",
        "source_set": "source-code",
        "connectome": report["connectome_checksum"],
        "frozen_core": report["core_checksum"],
        "rewire_manifest": report.get("rewire_manifest_checksum"),
    }
    inputs["metadata"] = {"controller_report": report}
    return inputs


@pytest.mark.parametrize(
    "controller",
    [
        "frozen_lif_original",
        "frozen_lif_degree_rewired",
        "gru_matched",
        "mlp_normal",
    ],
)
def test_actor_warm_start_loads_only_actor_and_resets_all_other_state(
    tmp_path, controller
):
    torch.manual_seed(11)
    source, report = build_controller(
        controller,
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_swaps=1,
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.add_(5.0 if name.startswith("critic.") else 0.25)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=3e-4)
    path = tmp_path / f"{controller}.pt"
    save_checkpoint(
        path,
        **_warm_start_checkpoint_inputs(
            source, source_optimizer, controller=controller, report=report
        ),
    )

    torch.manual_seed(99)
    target, target_report = build_controller(
        controller,
        connectome_manifest=FIXTURE,
        allow_synthetic=True,
        rewire_swaps=1,
    )
    target_before = {name: value.clone() for name, value in target.state_dict().items()}
    result = warm_start_actor_from_checkpoint(
        path,
        policy=target,
        expected_controller=controller,
        expected_connectome_checksum=target_report["connectome_checksum"],
        expected_frozen_core_checksum=target_report["core_checksum"],
    )
    source_state = source.state_dict()
    target_state = target.state_dict()
    assert result["source_checkpoint"]["absolute_path"] == str(path.resolve())
    assert result["source_checkpoint"]["sha256"] == checkpoint_sha256(path)
    assert result["loaded_policy_state_keys"]
    assert result["reset_policy_state_keys"]
    assert set(result["loaded_policy_state_keys"]).isdisjoint(
        result["reset_policy_state_keys"]
    )
    for name in result["loaded_policy_state_keys"]:
        assert torch.equal(target_state[name], source_state[name])
    for name in result["reset_policy_state_keys"]:
        assert torch.equal(target_state[name], target_before[name])
        assert not torch.equal(target_state[name], source_state[name])
    for name in result["verified_fixed_policy_state_keys"]:
        assert torch.equal(target_state[name], target_before[name])
    assert result["compatibility"] == {
        "policy_class_verified": True,
        "controller_verified": True,
        "connectome_checksum_verified": True,
        "frozen_core_checksum_verified": True,
        "fixed_policy_state_verified": True,
        "cross_task_and_reward_fingerprint_change_allowed": True,
    }


def test_actor_warm_start_fails_before_mutation_on_fixed_buffer_or_connectome_change(
    tmp_path,
):
    source, report = build_controller(
        "gru_matched", connectome_manifest=FIXTURE, allow_synthetic=True
    )
    path = tmp_path / "source.pt"
    save_checkpoint(
        path,
        **_warm_start_checkpoint_inputs(
            source,
            torch.optim.Adam(source.parameters(), lr=3e-4),
            controller="gru_matched",
            report=report,
        ),
    )
    target, target_report = build_controller(
        "gru_matched", connectome_manifest=FIXTURE, allow_synthetic=True
    )
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="connectome checksum"):
        warm_start_actor_from_checkpoint(
            path,
            policy=target,
            expected_controller="gru_matched",
            expected_connectome_checksum="wrong-connectome",
            expected_frozen_core_checksum=None,
        )
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())

    damaged = torch.load(path, map_location="cpu", weights_only=False)
    damaged["policy_state"]["crazyflie_residual_latent_scale"] += 1.0
    damaged_path = tmp_path / "damaged-fixed-buffer.pt"
    torch.save(damaged, damaged_path)
    with pytest.raises(CheckpointCompatibilityError, match="fixed policy state"):
        warm_start_actor_from_checkpoint(
            damaged_path,
            policy=target,
            expected_controller="gru_matched",
            expected_connectome_checksum=target_report["connectome_checksum"],
            expected_frozen_core_checksum=None,
        )
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())


def test_atomic_checkpoint_round_trip_and_resume_has_no_duplicate_count(tmp_path):
    torch.manual_seed(5)
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    path = tmp_path / "checkpoints" / "update-000002.pt"
    arguments = _checkpoint_inputs(policy, optimizer)
    save_checkpoint(path, scheduler=scheduler, **arguments)
    assert path.is_file()
    assert len(checkpoint_sha256(path)) == 64
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, scheduler=scheduler, **arguments)

    reloaded = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    reloaded_optimizer = torch.optim.Adam(reloaded.parameters(), lr=9e-4)
    reloaded_scheduler = torch.optim.lr_scheduler.StepLR(reloaded_optimizer, step_size=2)
    payload = load_checkpoint(
        path,
        policy=reloaded,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
        expected_fingerprints=arguments["fingerprints"],
        expected_config=arguments["resolved_config"],
        expected_task_manifest_id="task-manifest-test",
        expected_evaluation_manifest_id="evaluation-manifest-test",
        for_resume=True,
    )
    assert payload["next_update_index"] == 2
    assert payload["resume_counters"]["completed_updates"] == 2
    assert payload["resume_counters"]["total_interactions"] == 64
    assert payload["resume_counters"]["resume_count"] == 1
    assert payload["resume_counters"]["resume_reset"] is True
    assert torch.equal(payload["recurrent_state"], arguments["recurrent_state"])
    advanced = ResumeCounters.from_value(payload["resume_counters"]).advanced(rollout_interactions=32)
    assert advanced.completed_updates == 3
    assert advanced.total_interactions == 96
    appended = append_history_record(
        payload["history"],
        {"completed_updates": 3, "total_interactions": 96, "loss": 0.25},
    )
    assert len(appended) == 3


def test_checkpoint_boundary_serializes_once_and_links_identical_archive(tmp_path, monkeypatch):
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    latest = tmp_path / "checkpoints" / "latest.pt"
    numbered = tmp_path / "checkpoints" / "update-00000002.pt"
    calls = 0
    original_save = torch.save

    def counted_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", counted_save)
    save_checkpoint_boundary(latest, numbered, **_checkpoint_inputs(policy, optimizer))

    assert calls == 1
    assert latest.stat().st_ino == numbered.stat().st_ino
    assert checkpoint_sha256(latest) == checkpoint_sha256(numbered)
    assert torch.load(latest, map_location="cpu", weights_only=False)["counters"] == torch.load(
        numbered, map_location="cpu", weights_only=False
    )["counters"]


def test_replacing_latest_preserves_prior_numbered_checkpoint(tmp_path):
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    latest = tmp_path / "checkpoints" / "latest.pt"
    first = tmp_path / "checkpoints" / "update-00000002.pt"
    second = tmp_path / "checkpoints" / "update-00000003.pt"
    first_arguments = _checkpoint_inputs(policy, optimizer)
    save_checkpoint_boundary(latest, first, **first_arguments)
    first_inode = first.stat().st_ino
    first_sha = checkpoint_sha256(first)

    second_arguments = _checkpoint_inputs(policy, optimizer)
    second_arguments["counters"] = ResumeCounters(3, 96, completed_episodes=4)
    second_arguments["history"] = [
        *second_arguments["history"],
        {"completed_updates": 3, "total_interactions": 96, "loss": 0.25},
    ]
    save_checkpoint_boundary(latest, second, **second_arguments)

    assert first.stat().st_ino == first_inode
    assert checkpoint_sha256(first) == first_sha
    assert latest.stat().st_ino == second.stat().st_ino
    assert latest.stat().st_ino != first_inode
    assert checkpoint_sha256(latest) == checkpoint_sha256(second)
    assert torch.load(first, map_location="cpu", weights_only=False)["counters"][
        "completed_updates"
    ] == 2


def test_checkpoint_link_failure_leaves_restartable_latest(tmp_path, monkeypatch):
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    latest = tmp_path / "checkpoints" / "latest.pt"
    numbered = tmp_path / "checkpoints" / "update-00000002.pt"
    arguments = _checkpoint_inputs(policy, optimizer)
    original_link = os.link

    def fail_link(*_args, **_kwargs):
        raise OSError("injected link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected link failure"):
        save_checkpoint_boundary(latest, numbered, **arguments)
    assert latest.is_file()
    assert not numbered.exists()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["counters"]["completed_updates"] == 2
    assert payload["counters"]["total_interactions"] == 64

    monkeypatch.setattr(os, "link", original_link)
    link_checkpoint_snapshot(latest, numbered)
    assert latest.stat().st_ino == numbered.stat().st_ino


def test_checkpoint_archive_rejects_cross_parent_and_existing_destination(tmp_path):
    source = tmp_path / "one" / "latest.pt"
    source.parent.mkdir()
    source.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="share the source checkpoint directory"):
        link_checkpoint_snapshot(source, tmp_path / "two" / "update.pt")
    destination = source.parent / "update.pt"
    destination.write_bytes(b"prior")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        link_checkpoint_snapshot(source, destination)
    assert destination.read_bytes() == b"prior"


def test_constant_time_live_history_append_checks_only_the_validated_boundary():
    history = [
        {"completed_updates": 1, "total_interactions": 100, "loss": 1.0},
        {"completed_updates": 2, "total_interactions": 200, "loss": 0.5},
    ]
    append_validated_history_record_inplace(
        history,
        {"completed_updates": 3, "total_interactions": 300, "loss": 0.25},
    )
    assert [row["completed_updates"] for row in history] == [1, 2, 3]
    with pytest.raises(CheckpointValidationError, match="strictly increasing"):
        append_validated_history_record_inplace(
            history,
            {"completed_updates": 3, "total_interactions": 400},
        )
    with pytest.raises(TypeError, match="mutable list"):
        append_validated_history_record_inplace(  # type: ignore[arg-type]
            tuple(history),
            {"completed_updates": 4, "total_interactions": 400},
        )


def test_external_segmented_history_round_trip_is_compact_and_tamper_evident(tmp_path):
    torch.manual_seed(7)
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = [
        {
            "completed_updates": update,
            "total_interactions": update * 32,
            "loss": 1.0 / update,
            "payload": "x" * 2048,
        }
        for update in range(1, 101)
    ]
    checkpoint = tmp_path / "checkpoints" / "update-00000100.pt"
    first = write_history_segment(
        tmp_path / "history" / "rows-00000001-00000050-resume-0000.jsonl",
        history[:50],
        reference_directory=checkpoint.parent,
    )
    second = write_history_segment(
        tmp_path / "history" / "rows-00000051-00000100-resume-0000.jsonl",
        history[50:],
        reference_directory=checkpoint.parent,
    )
    reference = build_history_reference(history, [first, second])
    streamed_reference = build_history_reference_from_segments(
        [first, second],
        checkpoint_path=checkpoint,
        interactions_per_update=32,
    )
    assert streamed_reference == reference
    arguments = _checkpoint_inputs(policy, optimizer)
    arguments.update({
        "counters": ResumeCounters(100, 3200, completed_episodes=3),
        "history": history,
        "history_reference": reference,
    })
    save_checkpoint(checkpoint, **arguments)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert raw["history"] == []
    assert raw["history_reference"]["row_count"] == 100
    assert checkpoint.stat().st_size < sum(
        (tmp_path / "history" / name).stat().st_size
        for name in (
            "rows-00000001-00000050-resume-0000.jsonl",
            "rows-00000051-00000100-resume-0000.jsonl",
        )
    )
    assert load_history_reference(reference, checkpoint_path=checkpoint) == history

    reloaded = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    loaded = load_checkpoint(
        checkpoint,
        policy=reloaded,
        optimizer=torch.optim.Adam(reloaded.parameters(), lr=3e-4),
        expected_fingerprints=arguments["fingerprints"],
        expected_config=arguments["resolved_config"],
        expected_task_manifest_id="task-manifest-test",
        expected_evaluation_manifest_id="evaluation-manifest-test",
        for_resume=True,
    )
    assert loaded["history"] == history
    assert loaded["next_update_index"] == 100

    compact_checkpoint = checkpoint.with_name("update-00000100-compact.pt")
    compact_arguments = dict(arguments)
    compact_arguments["history"] = []
    save_checkpoint(compact_checkpoint, **compact_arguments)
    compact_policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    compact_loaded = load_checkpoint(
        compact_checkpoint,
        policy=compact_policy,
        optimizer=torch.optim.Adam(compact_policy.parameters(), lr=3e-4),
        expected_fingerprints=arguments["fingerprints"],
        expected_config=arguments["resolved_config"],
        expected_task_manifest_id="task-manifest-test",
        expected_evaluation_manifest_id="evaluation-manifest-test",
        for_resume=True,
        materialize_external_history=False,
    )
    assert compact_loaded["history"] == []
    assert compact_loaded["history_reference"] == reference
    assert compact_loaded["next_update_index"] == 100

    segment_path = tmp_path / "history" / "rows-00000051-00000100-resume-0000.jsonl"
    segment_path.write_bytes(segment_path.read_bytes() + b"{}\n")
    with pytest.raises(CheckpointValidationError, match="size or checksum"):
        load_history_reference(reference, checkpoint_path=checkpoint)
    with pytest.raises(
        CheckpointValidationError,
        match="size or checksum|missing explicit counter",
    ):
        build_history_reference_from_segments(
            [first, second],
            checkpoint_path=checkpoint,
            interactions_per_update=32,
        )


def test_streamed_history_reference_supports_empty_initial_boundary_and_partial_resume(
    tmp_path,
):
    torch.manual_seed(11)
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    initialized = tmp_path / "checkpoints" / "update-00000000.pt"
    empty_reference = build_history_reference_from_segments(
        [], checkpoint_path=initialized, interactions_per_update=32
    )
    assert empty_reference == build_history_reference([], [])
    initial_arguments = _checkpoint_inputs(policy, optimizer)
    initial_arguments.update({
        "counters": ResumeCounters(0, 0),
        "history": [],
        "history_reference": empty_reference,
    })
    save_checkpoint(initialized, **initial_arguments)
    assert read_checkpoint(initialized)["history_reference"] == empty_reference

    all_rows = [
        {
            "completed_updates": update,
            "total_interactions": update * 32,
            "loss": 1.0 / update,
        }
        for update in range(1, 224)
    ]
    segment_specs = ((1, 100), (101, 200), (201, 223))
    segments = []
    max_pending_rows = 0
    for first_update, last_update in segment_specs:
        pending = all_rows[first_update - 1:last_update]
        max_pending_rows = max(max_pending_rows, len(pending))
        segments.append(write_history_segment(
            tmp_path / "history" / (
                f"rows-{first_update:08d}-{last_update:08d}-resume-0000.jsonl"
            ),
            pending,
            reference_directory=initialized.parent,
        ))
        pending.clear()
        assert pending == []
    assert max_pending_rows == 100
    resumed_reference = build_history_reference_from_segments(
        segments,
        checkpoint_path=initialized,
        interactions_per_update=32,
    )
    assert resumed_reference == build_history_reference(all_rows, segments)

    resumed_checkpoint = initialized.with_name("update-00000223.pt")
    resumed_arguments = _checkpoint_inputs(policy, optimizer)
    resumed_arguments.update({
        "counters": ResumeCounters(223, 223 * 32),
        "history": [],
        "history_reference": resumed_reference,
    })
    save_checkpoint(resumed_checkpoint, **resumed_arguments)
    target = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    loaded = load_checkpoint(
        resumed_checkpoint,
        policy=target,
        optimizer=torch.optim.Adam(target.parameters(), lr=3e-4),
        expected_fingerprints=resumed_arguments["fingerprints"],
        expected_config=resumed_arguments["resolved_config"],
        expected_task_manifest_id="task-manifest-test",
        expected_evaluation_manifest_id="evaluation-manifest-test",
        for_resume=True,
        materialize_external_history=False,
    )
    assert loaded["history"] == []
    assert loaded["history_reference"]["row_count"] == 223
    assert loaded["resume_counters"]["completed_updates"] == 223
    assert loaded["resume_counters"]["total_interactions"] == 223 * 32


def _history_rows(first: int, last: int, *, interactions_per_update: int = 32):
    return [
        {
            "completed_updates": update,
            "total_interactions": update * interactions_per_update,
            "label": f"บิน-{update}",
            "nested": {"values": [update, -0.0, True]},
        }
        for update in range(first, last + 1)
    ]


def _write_test_history_segment(tmp_path, rows, name):
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    return checkpoint, write_history_segment(
        tmp_path / "history" / name,
        rows,
        reference_directory=checkpoint.parent,
    )


def _clear_history_accumulator_cache():
    with checkpoint_module._HISTORY_ACCUMULATOR_CACHE_LOCK:
        checkpoint_module._HISTORY_ACCUMULATOR_CACHE.clear()


def test_incremental_history_digest_is_exact_closed_v1_and_streamed_row_by_row(tmp_path):
    rows = _history_rows(1, 5)
    checkpoint, first = _write_test_history_segment(
        tmp_path, rows[:2], "rows-00000001-00000002.jsonl"
    )
    _, second = _write_test_history_segment(
        tmp_path, rows[2:], "rows-00000003-00000005.jsonl"
    )

    accumulator = HistoryReferenceAccumulator(
        checkpoint_path=checkpoint,
        interactions_per_update=32,
    )
    first_reference = accumulator.synchronize([first])
    assert first_reference == build_history_reference(rows[:2], [first])
    final_reference = accumulator.synchronize([first, second])
    assert final_reference == build_history_reference(rows, [first, second])
    assert accumulator.validated_segment_count == 2
    assert accumulator.row_count == 5
    assert set(final_reference) == {
        "schema_version",
        "storage",
        "row_count",
        "history_sha256",
        "last_completed_updates",
        "last_total_interactions",
        "segments",
    }
    assert all(
        set(segment) == {
            "path",
            "sha256",
            "byte_count",
            "row_count",
            "first_completed_updates",
            "first_total_interactions",
            "last_completed_updates",
            "last_total_interactions",
        }
        for segment in final_reference["segments"]
    )
    expected_first_bytes = b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        for row in rows[:2]
    )
    first_path = (checkpoint.parent / first["path"]).resolve()
    assert first_path.read_bytes() == expected_first_bytes
    assert first["sha256"] == sha256(expected_first_bytes).hexdigest()
    assert first["byte_count"] == len(expected_first_bytes)


def test_training_reference_cache_reconstructs_once_then_reads_only_new_segments(
    tmp_path, monkeypatch
):
    rows = _history_rows(1, 6)
    checkpoint, first = _write_test_history_segment(
        tmp_path, rows[:3], "rows-00000001-00000003.jsonl"
    )
    _, second = _write_test_history_segment(
        tmp_path, rows[3:], "rows-00000004-00000006.jsonl"
    )
    first_path = (checkpoint.parent / first["path"]).resolve()
    second_path = (checkpoint.parent / second["path"]).resolve()
    opened: list[Path] = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode and "b" in mode and path.suffix == ".jsonl":
            opened.append(path.resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    _clear_history_accumulator_cache()

    first_reference = build_history_reference_from_segments(
        [first], checkpoint_path=checkpoint, interactions_per_update=32
    )
    assert opened == [first_path]
    assert first_reference == build_history_reference(rows[:3], [first])

    opened.clear()
    assert build_history_reference_from_segments(
        [first], checkpoint_path=checkpoint, interactions_per_update=32
    ) == first_reference
    assert opened == []

    final_reference = build_history_reference_from_segments(
        [first, second], checkpoint_path=checkpoint, interactions_per_update=32
    )
    assert opened == [second_path]
    assert final_reference == build_history_reference(rows, [first, second])

    # A new process/resume has no in-memory state: old segments are scanned
    # exactly once, then the reconstructed prefix is reused.
    _clear_history_accumulator_cache()
    opened.clear()
    assert build_history_reference_from_segments(
        [first, second], checkpoint_path=checkpoint, interactions_per_update=32
    ) == final_reference
    assert opened == [first_path, second_path]
    opened.clear()
    build_history_reference_from_segments(
        [first, second], checkpoint_path=checkpoint, interactions_per_update=32
    )
    assert opened == []


def test_cached_history_tamper_forces_cryptographic_revalidation(tmp_path):
    rows = _history_rows(1, 3)
    checkpoint, segment = _write_test_history_segment(
        tmp_path, rows, "rows-00000001-00000003.jsonl"
    )
    _clear_history_accumulator_cache()
    expected = build_history_reference_from_segments(
        [segment], checkpoint_path=checkpoint, interactions_per_update=32
    )
    segment_path = (checkpoint.parent / segment["path"]).resolve()
    original = segment_path.read_bytes()
    tampered = original.replace(b'"values":[2,-0.0,true]', b'"values":[9,-0.0,true]')
    assert tampered != original and len(tampered) == len(original)
    segment_path.write_bytes(tampered)

    with pytest.raises(CheckpointValidationError, match="size or checksum"):
        build_history_reference_from_segments(
            [segment], checkpoint_path=checkpoint, interactions_per_update=32
        )

    # A failed rebuild never poisons the cache; restoring the authenticated
    # bytes permits a clean reconstruction on the next call.
    segment_path.write_bytes(original)
    assert build_history_reference_from_segments(
        [segment], checkpoint_path=checkpoint, interactions_per_update=32
    ) == expected

    forged = dict(segment)
    forged["sha256"] = "0" * 64
    with pytest.raises(CheckpointValidationError, match="size or checksum"):
        build_history_reference_from_segments(
            [forged], checkpoint_path=checkpoint, interactions_per_update=32
        )


def test_incremental_history_continuity_failure_is_transactional(tmp_path):
    first_rows = _history_rows(1, 2)
    checkpoint, first = _write_test_history_segment(
        tmp_path, first_rows, "rows-00000001-00000002.jsonl"
    )
    accumulator = HistoryReferenceAccumulator.reconstruct(
        [first], checkpoint_path=checkpoint, interactions_per_update=32
    )
    prior_reference = accumulator.reference()
    _, skipped = _write_test_history_segment(
        tmp_path, _history_rows(4, 4), "rows-00000004-00000004.jsonl"
    )

    with pytest.raises(CheckpointValidationError, match="skips or invents"):
        accumulator.synchronize([first, skipped])
    assert accumulator.row_count == 2
    assert accumulator.validated_segment_count == 1
    assert accumulator.reference() == prior_reference

    _, contiguous = _write_test_history_segment(
        tmp_path, _history_rows(3, 3), "rows-00000003-00000003.jsonl"
    )
    assert accumulator.synchronize([first, contiguous]) == build_history_reference(
        _history_rows(1, 3), [first, contiguous]
    )


def test_history_segment_write_failures_never_publish_partial_files(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    destination = tmp_path / "history" / "rows.jsonl"
    invalid = _history_rows(1, 2)
    invalid[1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        write_history_segment(
            destination, invalid, reference_directory=checkpoint.parent
        )
    assert not destination.exists()
    assert list(destination.parent.glob(".rows.jsonl.*.tmp")) == []

    def fail_link(*_args, **_kwargs):
        raise OSError("injected publish failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected publish failure"):
        write_history_segment(
            destination, _history_rows(1, 2), reference_directory=checkpoint.parent
        )
    assert not destination.exists()
    assert list(destination.parent.glob(".rows.jsonl.*.tmp")) == []


def test_history_segment_atomic_publish_never_overwrites_existing_destination(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    destination = tmp_path / "history" / "rows.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"preserve-me")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_history_segment(
            destination, _history_rows(1, 2), reference_directory=checkpoint.parent
        )
    assert destination.read_bytes() == b"preserve-me"
    assert list(destination.parent.glob(".rows.jsonl.*.tmp")) == []


def test_history_validation_detects_path_replacement_during_scan(tmp_path, monkeypatch):
    rows = _history_rows(1, 2)
    checkpoint, segment = _write_test_history_segment(
        tmp_path, rows, "rows-00000001-00000002.jsonl"
    )
    original_signature = checkpoint_module._history_file_signature(
        (checkpoint.parent / segment["path"]).resolve()
    )
    replacement_signature = (
        original_signature[0],
        original_signature[1] + 1,
        *original_signature[2:],
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_history_file_signature",
        lambda _path: replacement_signature,
    )
    accumulator = HistoryReferenceAccumulator(
        checkpoint_path=checkpoint, interactions_per_update=32
    )
    with pytest.raises(CheckpointValidationError, match="changed while"):
        accumulator.synchronize([segment])
    assert accumulator.row_count == 0
    assert accumulator.validated_segment_count == 0


def test_history_accumulator_cache_is_bounded(tmp_path):
    _clear_history_accumulator_cache()
    for index in range(checkpoint_module._HISTORY_ACCUMULATOR_CACHE_LIMIT + 3):
        build_history_reference_from_segments(
            [],
            checkpoint_path=tmp_path / str(index) / "checkpoints" / "latest.pt",
            interactions_per_update=32,
        )
    assert len(checkpoint_module._HISTORY_ACCUMULATOR_CACHE) == (
        checkpoint_module._HISTORY_ACCUMULATOR_CACHE_LIMIT
    )


def test_incompatible_fingerprint_rejected_before_policy_mutation(tmp_path):
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, **_checkpoint_inputs(policy, optimizer))

    target = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        load_checkpoint(
            path,
            policy=target,
            optimizer=torch.optim.Adam(target.parameters(), lr=3e-4),
            expected_fingerprints={"code": "wrong"},
            for_resume=True,
        )
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())

    tainted = load_checkpoint(
        path,
        policy=target,
        expected_fingerprints={"code": "wrong"},
        allow_incompatible=True,
    )
    assert tainted["tainted"] is True
    assert tainted["taint_reasons"]


def test_survival_v2_residual_scale_is_rejected_before_policy_mutation(tmp_path):
    policy = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, **_checkpoint_inputs(policy, optimizer))

    incompatible = torch.load(path, map_location="cpu", weights_only=False)
    incompatible["policy_state"]["crazyflie_residual_latent_scale"] = torch.tensor(
        [0.04, 0.08, 0.08, 0.05]
    )
    incompatible_path = tmp_path / "survival-v3-checkpoint.pt"
    torch.save(incompatible, incompatible_path)

    target = MatchedGRUActorCritic(12, 4, hidden_dim=7, critic_hidden_dim=16)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(
        CheckpointCompatibilityError,
        match="crazyflie_residual_latent_scale",
    ):
        load_checkpoint(incompatible_path, policy=target)
    assert all(
        torch.equal(value, before[name])
        for name, value in target.state_dict().items()
    )


def test_history_and_counter_validation_prevent_double_counting():
    history = [{"completed_updates": 1, "total_interactions": 32}]
    with pytest.raises(CheckpointValidationError, match="duplicate"):
        validate_history([*history, {"completed_updates": 1, "total_interactions": 32}])
    with pytest.raises(CheckpointValidationError, match="Counter mismatch"):
        validate_interaction_alignment(ResumeCounters(2, 63), 32)
    validate_interaction_alignment(ResumeCounters(2, 64), 32)
    resumed = ResumeCounters(2, 64).resumed(reset_environments=True)
    assert (resumed.completed_updates, resumed.total_interactions) == (2, 64)


def test_history_alignment_rejects_skipped_or_invented_update_rows():
    skipped = [
        {"completed_updates": 0, "total_interactions": 0},
        {"completed_updates": 2, "total_interactions": 64},
        {"completed_updates": 3, "total_interactions": 96},
    ]
    with pytest.raises(CheckpointValidationError, match="skips or invents"):
        validate_history_alignment(skipped, ResumeCounters(3, 96), 32)

    missing = [
        {"completed_updates": 1, "total_interactions": 32},
        {"completed_updates": 3, "total_interactions": 96},
    ]
    with pytest.raises(CheckpointValidationError, match="rows for 3 completed updates"):
        validate_history_alignment(missing, ResumeCounters(3, 96), 32)


def test_python_numpy_and_torch_rng_round_trip():
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    states = capture_rng_states(environment={"counter": 7}, task_schedule=(1, 2), include_cuda=False)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restored = restore_rng_states(states)
    actual = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert restored == {"environment": {"counter": 7}, "task_schedule": (1, 2)}
