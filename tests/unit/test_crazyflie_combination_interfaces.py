"""CPU-only contracts for the additive multi-connectome interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_evaluate as evaluator  # noqa: E402
import drone_smoke_env as smoke_env  # noqa: E402
import drone_train as trainer  # noqa: E402
from g1_fly_control.crazyflie import checkpoint as checkpoint_module  # noqa: E402
from g1_fly_control.crazyflie.controllers import build_controller  # noqa: E402
from g1_fly_control.crazyflie.trained_keyboard import (  # noqa: E402
    COMBINATION_CONTROLLER_KINDS,
    SUPPORTED_CONTROLLER_KINDS,
    CommandCheckpointSpec,
    TrainedActivityRecorder,
    _controller_build_arguments,
    inspect_command_checkpoint,
)
from g1_fly_control.tasks.crazyflie.command_logic import (  # noqa: E402
    command_follow_contract_payload,
)


EXPECTED_LABELS = {
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}


def _training_args(policy: str) -> argparse.Namespace:
    return argparse.Namespace(
        task=trainer.COMMAND_WIDE_TASK,
        contract_profile=trainer.COMMAND_V2_CONTRACT_PROFILE,
        evaluation_protocol=trainer.COMMAND_V2_EVALUATION_PROTOCOL,
        policy=policy,
        seed=0,
        num_envs=40,
        total_interactions=1_000_000,
        horizon=100,
        microbatch_size=40,
        ppo_epochs=2,
        learning_rate=3.0e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        max_grad_norm=1.0,
        target_kl=0.05,
        checkpoint_every_updates=25,
        connectome_manifest=trainer.DEFAULT_CONNECTOME.resolve(),
        wing_connectome_manifest=trainer.DEFAULT_WING_CONNECTOME.resolve(),
        optic_connectome_manifest=trainer.DEFAULT_OPTIC_CONNECTOME.resolve(),
        rewire_seed=20260916,
        rewire_manifest=trainer.DEFAULT_REWIRE_MANIFEST.resolve(),
        warm_start_checkpoint=None,
    )


def _spec(policy: str) -> CommandCheckpointSpec:
    labels = EXPECTED_LABELS[policy]
    paths = {
        "leg": trainer.DEFAULT_CONNECTOME,
        "wing": trainer.DEFAULT_WING_CONNECTOME,
        "optic": trainer.DEFAULT_OPTIC_CONNECTOME,
    }
    return CommandCheckpointSpec(
        path=Path("/tmp/combo.pt"),
        sha256="x",
        controller=policy,
        canonical_controller=policy,
        resolved_config={},
        controller_report={
            "controller_kind": policy,
            "core_labels": list(labels),
            "widths": {"adapter_hidden_dim": 64, "critic_hidden_dim": 128},
            "connectome_manifests": {
                label: str(paths[label].resolve()) for label in labels
            },
        },
        command_follow_contract=command_follow_contract_payload(),
        fingerprints={},
        task_manifest_id="task",
        evaluation_manifest_id="evaluation",
        observation_normalizer_state={},
    )


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_combination_standalone_identity_hashes_every_core(policy: str) -> None:
    args = _training_args(policy)
    resolved, evaluation = trainer.standalone_resolved_config(args)
    composition = resolved["lif_connectome_composition"]

    assert policy in trainer.POLICIES
    assert tuple(composition["core_labels"]) == EXPECTED_LABELS[policy]
    assert tuple(composition["connectomes"]) == EXPECTED_LABELS[policy]
    assert composition["fusion_contract"] == trainer.COMBINATION_LIF_FUSION_CONTRACTS[
        policy
    ]
    assert composition["recurrent_cross_core_edges"] is False
    assert composition["parameter_matching_required"] is False
    for identity in composition["connectomes"].values():
        assert len(identity["manifest_sha256"]) == 64
        assert len(identity["neurons_path"]["sha256"]) == 64
        assert len(identity["edges_path"]["sha256"]) == 64
    assert evaluation["task"] == trainer.COMMAND_WIDE_TASK


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_checkpoint_reconstruction_routes_every_declared_manifest(policy: str) -> None:
    spec = _spec(policy)
    arguments = _controller_build_arguments(spec)
    labels = EXPECTED_LABELS[policy]

    assert arguments["widths"] == {
        "adapter_hidden_dim": 64,
        "critic_hidden_dim": 128,
    }
    expected_argument = {
        "leg": "connectome_manifest",
        "wing": "wing_connectome_manifest",
        "optic": "optic_connectome_manifest",
    }
    for label in labels:
        assert arguments[expected_argument[label]] == spec.controller_report[
            "connectome_manifests"
        ][label]
    rebuilt, report = build_controller(
        policy,
        observation_dim=12,
        action_dim=4,
        device="cpu",
        **arguments,
    )
    assert tuple(report["core_labels"]) == labels
    assert tuple(report["connectome_manifests"]) == labels
    assert rebuilt.initial_state(2).spikes.shape == (2, 256 * len(labels))


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_activity_roles_cover_every_core_in_recurrent_state_order(policy: str) -> None:
    spec = _spec(policy)
    recorder = TrainedActivityRecorder(torch.nn.Identity(), spec)
    labels = EXPECTED_LABELS[policy]
    unit_count = 256 * len(labels)
    spikes = torch.zeros((2, unit_count))
    for index in range(len(labels)):
        spikes[:, index * 256] = 1.0
    state = SimpleNamespace(
        membrane=torch.zeros_like(spikes),
        spikes=spikes,
        synapse=torch.zeros_like(spikes),
        refractory=torch.zeros_like(spikes),
    )

    assert recorder.activity_batch(state).shape == (2, unit_count)
    assert tuple(recorder.role_provenance) == labels
    assert len(recorder.roles) == len(recorder.unit_ids) == unit_count
    assert all(
        any(role.startswith(f"{label}:") for role in recorder.roles)
        for label in labels
    )
    assert len(set(recorder.unit_ids)) == unit_count
    recorder.close()


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_checkpoint_inspector_accepts_exact_combination_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str
) -> None:
    checkpoint = tmp_path / f"{policy}.pt"
    checkpoint.write_bytes(policy.encode())
    payload = {
        "resolved_config": {
            "task": "FlyCrazyflie-CommandFollow-v0",
            "controller": policy,
            "command_follow_contract": command_follow_contract_payload(),
        },
        "metadata": {
            "controller_report": {
                "controller_kind": policy,
                "observation_dim": 12,
                "action_dim": 4,
            }
        },
        "normalizers": {"observation": {"kind": "unit-test"}},
        "fingerprints": {"source_set": "source"},
        "task_manifest_id": "task-manifest",
        "evaluation_manifest_id": "evaluation-manifest",
    }
    monkeypatch.setattr(
        checkpoint_module, "read_checkpoint", lambda *_args, **_kwargs: payload
    )

    spec = inspect_command_checkpoint(checkpoint)
    assert spec.controller == spec.canonical_controller == policy
    assert policy in SUPPORTED_CONTROLLER_KINDS


def test_generic_training_activity_preserves_all_ordered_core_summaries() -> None:
    class Accumulator:
        def __init__(self, value: int) -> None:
            self.value = value

        def summary(self) -> dict[str, int]:
            return {"sampled_spike_count": self.value}

    policy = SimpleNamespace(
        fusion_contract=trainer.COMBINATION_LIF_FUSION_CONTRACTS[
            "leg_wing_optic_lif"
        ]
    )
    result = trainer._summarize_lif_activity(
        policy,
        {
            "leg": Accumulator(1),
            "wing": Accumulator(2),
            "optic": Accumulator(3),
        },
    )

    assert tuple(result) == ("composition", "leg", "wing", "optic")
    assert result["leg"]["sampled_spike_count"] == 1
    assert result["wing"]["sampled_spike_count"] == 2
    assert result["optic"]["sampled_spike_count"] == 3


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_evaluator_parser_accepts_combination_policy(policy: str) -> None:
    args = evaluator._parser().parse_args(
        ["--checkpoint", "checkpoint.pt", "--output", "evaluation.json", "--policy", policy]
    )
    assert args.policy == policy


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_smoke_parser_accepts_combination_policy(policy: str) -> None:
    assert policy in smoke_env.POLICIES


@pytest.mark.parametrize("policy", COMBINATION_CONTROLLER_KINDS)
def test_smoke_builder_routes_all_combination_manifests(policy: str) -> None:
    built, report = smoke_env._build_policy(
        policy,
        torch.device("cpu"),
        trainer.DEFAULT_CONNECTOME.resolve(),
        20260916,
        trainer.DEFAULT_REWIRE_MANIFEST.resolve(),
        trainer.DEFAULT_WING_CONNECTOME.resolve(),
        trainer.DEFAULT_OPTIC_CONNECTOME.resolve(),
    )
    assert tuple(report["core_labels"]) == EXPECTED_LABELS[policy]
    assert report["parameter_matching_required"] is False
    assert built.initial_state(1).spikes.shape[1] == 256 * len(EXPECTED_LABELS[policy])


def test_training_report_gate_requires_exact_ordered_core_provenance() -> None:
    args = _training_args("leg_wing_optic_lif")
    labels = EXPECTED_LABELS[args.policy]
    paths = {
        "leg": args.connectome_manifest,
        "wing": args.wing_connectome_manifest,
        "optic": args.optic_connectome_manifest,
    }
    report = {
        "controller_kind": args.policy,
        "core_labels": list(labels),
        "fusion_contract": trainer.COMBINATION_LIF_FUSION_CONTRACTS[args.policy],
        "connectome_manifests": {
            label: str(paths[label]) for label in labels
        },
        "connectome_checksums": {label: "b" * 64 for label in labels},
        "per_core_checksums": {label: "a" * 64 for label in labels},
    }
    trainer._validate_combination_controller_report(args, report)

    report["core_labels"] = ["optic", "wing", "leg"]
    with pytest.raises(RuntimeError, match="core ordering"):
        trainer._validate_combination_controller_report(args, report)
