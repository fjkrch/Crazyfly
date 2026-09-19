from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from g1_fly_control.crazyflie import checkpoint as checkpoint_module
from g1_fly_control.crazyflie.trained_keyboard import (
    COMMAND_FOLLOW_CONTRACT_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    PUBLIC_CONTROLLER_KINDS,
    ActionDecision,
    CommandCheckpointSpec,
    RuntimeSpeeds,
    SafetyEnvelope,
    TASK_ID,
    TrainedActivityRecorder,
    _controller_build_arguments,
    arbitrate_action,
    command_conditioned_observation,
    inspect_command_checkpoint,
    policy_state_is_finite,
    reconstruct_physical_observation,
    reset_policy_state,
    safety_reason,
    scaled_command_body,
)
from g1_fly_control.tasks.crazyflie.command_logic import (
    command_follow_contract_payload,
)


def test_runtime_script_selects_the_command_v1_environment_contract() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/crazyflie_trained_keyboard.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "selected_env_cfg"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    profile = keywords["contract_profile"]
    assert isinstance(profile, ast.Constant)
    assert profile.value == "command_v1"

    # The task enforces its authenticated 600-step horizon.  The viewer may
    # reset on truncation, but must not mutate the task contract to extend it.
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "episode_length_s"
            for target in node.targets
        )
        for node in ast.walk(tree)
    )


def test_runtime_viewer_scene_guides_are_visual_only_and_bounded() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/crazyflie_trained_keyboard.py"
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "FlightSceneVisuals" in classes
    scene_source = ast.get_source_segment(source, classes["FlightSceneVisuals"])
    assert scene_source is not None
    assert "VisualizationMarkers" in scene_source
    assert "collision_props" not in scene_source
    assert "rigid_props" not in scene_source
    assert "mass_props" not in scene_source
    assert "deque(maxlen=int(trail_points))" in scene_source
    assert '"physics_effect": "none_visual_markers_only"' in scene_source

    option_strings = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    assert {"--no_scene_guides", "--no_flight_trail", "--trail_points"} <= option_strings


def test_continuous_viewer_suppresses_only_timeout_without_mutating_cfg() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/crazyflie_trained_keyboard.py"
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "--continuous" in source
    assert "original_get_dones = env._get_dones" in source
    assert "return terminated, torch.zeros_like(time_out)" in source
    assert "env._get_dones = continuous_viewer_dones" in source
    # Neither the authenticated config nor its calculated timeout is changed.
    assert "cfg.episode_length_s" not in source
    assert "cfg.max_episode_length" not in source
    assert "env.cfg.episode_length_s" not in source
    assert "env.max_episode_length =" not in source


def test_runtime_speed_defaults_and_exact_training_limits_pass() -> None:
    contract = command_follow_contract_payload()
    RuntimeSpeeds().validate(contract)
    RuntimeSpeeds(0.8, 0.4, 1.2).validate(contract)


@pytest.mark.parametrize(
    "speeds",
    [
        RuntimeSpeeds(-0.1, 0.2, 0.3),
        RuntimeSpeeds(0.81, 0.2, 0.3),
        RuntimeSpeeds(0.2, 0.41, 0.3),
        RuntimeSpeeds(0.2, 0.3, 1.21),
        RuntimeSpeeds(float("nan"), 0.3, 0.8),
    ],
)
def test_runtime_speeds_reject_negative_nonfinite_or_out_of_training_range(
    speeds: RuntimeSpeeds,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        speeds.validate(command_follow_contract_payload())


def test_simultaneous_four_axis_command_scales_without_axis_loss() -> None:
    inverse_sqrt_two = 2.0**-0.5
    result = scaled_command_body(
        (inverse_sqrt_two, inverse_sqrt_two, 1.0, -1.0),
        RuntimeSpeeds(0.6, 0.3, 1.0),
    )
    assert result == pytest.approx(
        (0.6 * inverse_sqrt_two, 0.6 * inverse_sqrt_two, 0.3, -1.0)
    )
    assert (result[0] ** 2 + result[1] ** 2) ** 0.5 == pytest.approx(0.6)


def test_command_observation_has_exact_error_signs_and_round_trips_physics() -> None:
    physical = torch.tensor(
        [[0.7, -0.2, 0.4, 0.1, 0.2, 0.9, 0.0, 0.0, -1.0, 99.0, 99.0, 99.0]]
    )
    command = torch.tensor([[0.5, -0.4, 0.1, 0.3]])
    target_error = torch.tensor([[1.0, 2.0, 3.0]])
    result = command_conditioned_observation(physical, command, target_error)
    assert result[0, 0:3].tolist() == pytest.approx([0.2, 0.2, 0.3])
    assert float(result[0, 5]) == pytest.approx(0.6)
    assert result[0, 9:12].tolist() == pytest.approx([1.0, 2.0, 3.0])
    reconstructed = reconstruct_physical_observation(result, command)
    assert torch.equal(reconstructed[:, 0:9], physical[:, 0:9])
    assert torch.equal(reconstructed[:, 9:12], target_error)


def _safe_physical_observation() -> torch.Tensor:
    observation = torch.zeros((1, 12))
    observation[:, 8] = -1.0
    return observation


def test_safety_envelope_accepts_nominal_hover_and_labels_each_major_breach() -> None:
    envelope = SafetyEnvelope()
    physical = _safe_physical_observation()
    origin = torch.tensor([[0.0, 0.0, 0.5]])
    assert safety_reason(physical, origin.clone(), origin, envelope) is None

    low = origin.clone()
    low[:, 2] = envelope.minimum_height_m
    assert safety_reason(physical, low, origin, envelope) == "below_minimum_height"
    far = origin.clone()
    far[:, 0] = envelope.maximum_horizontal_offset_m
    assert safety_reason(physical, far, origin, envelope) == "horizontal_workspace_limit"
    tilted = physical.clone()
    tilted[:, 8] = 0.0
    assert safety_reason(tilted, origin, origin, envelope) == "excessive_tilt"


def test_valid_policy_action_is_returned_as_the_exact_primary_tensor() -> None:
    policy_action = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    state = torch.zeros((1, 3))
    fallback = torch.zeros((1, 4))
    decision = arbitrate_action(policy_action, state, fallback)
    assert isinstance(decision, ActionDecision)
    assert decision.source == "trained_policy"
    assert decision.fallback_reason is None
    assert decision.action is policy_action


@pytest.mark.parametrize(
    ("action", "state", "runtime_reason", "expected"),
    [
        (torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), None, None, "nonfinite_policy_action"),
        (torch.tensor([[1.1, 0.0, 0.0, 0.0]]), None, None, "out_of_bounds_policy_action"),
        (torch.zeros((1, 4)), torch.tensor([[float("inf")]]), None, "nonfinite_policy_state"),
        (torch.zeros((1, 4)), None, "excessive_tilt", "excessive_tilt"),
    ],
)
def test_invalid_or_unsafe_policy_uses_exact_deterministic_fallback(
    action: torch.Tensor,
    state: object,
    runtime_reason: str | None,
    expected: str,
) -> None:
    fallback = torch.tensor([[0.05, 0.01, -0.01, 0.0]])
    decision = arbitrate_action(
        action,
        state,
        fallback,
        runtime_safety_reason=runtime_reason,
    )
    assert decision.source == "deterministic_flight_assist"
    assert decision.fallback_reason == expected
    assert decision.action is fallback


def test_policy_state_finite_supports_mlp_gru_and_lif_state_shapes() -> None:
    assert policy_state_is_finite(None)
    assert policy_state_is_finite(torch.zeros((1, 4)))
    lif = SimpleNamespace(
        membrane=torch.zeros((1, 2)),
        spikes=torch.zeros((1, 2)),
        synapse=torch.zeros((1, 2)),
        refractory=torch.zeros((1, 2)),
    )
    assert policy_state_is_finite(lif)
    lif.membrane[0, 0] = float("nan")
    assert not policy_state_is_finite(lif)


def test_recurrent_reset_discards_fallback_state_and_calls_exact_initializer() -> None:
    class Policy:
        def __init__(self) -> None:
            self.calls: list[tuple[int, torch.device | str]] = []

        def initial_state(self, batch_size: int, *, device: torch.device | str):
            self.calls.append((batch_size, device))
            return torch.full((batch_size, 2), 7.0)

    policy = Policy()
    stale = torch.full((1, 2), 99.0)
    reset = reset_policy_state(policy, stale, device="cpu")
    assert policy.calls == [(1, "cpu")]
    assert torch.equal(reset, torch.full((1, 2), 7.0))
    assert not torch.equal(reset, stale)


def _checkpoint_payload(controller: str) -> dict[str, object]:
    canonical = {
        "frozen_lif_original": "frozen_lif",
        "frozen_lif_degree_rewired": "frozen_lif_rewired",
        "wing_lif": "wing_lif",
        "leg_wing_lif": "leg_wing_lif",
        "optic_lif": "optic_lif",
        "gru_matched": "gru",
        "mlp_normal": "mlp",
    }[controller]
    return {
        "resolved_config": {
            "task": TASK_ID,
            "controller": controller,
            "command_follow_contract": command_follow_contract_payload(),
        },
        "metadata": {
            "controller_report": {
                "controller_kind": canonical,
                "observation_dim": 12,
                "action_dim": 4,
            }
        },
        "normalizers": {"observation": {"kind": "unit-test"}},
        "fingerprints": {"source_set": "source"},
        "task_manifest_id": "task-manifest",
        "evaluation_manifest_id": "evaluation-manifest",
    }


@pytest.mark.parametrize("controller", PUBLIC_CONTROLLER_KINDS)
def test_checkpoint_inspector_accepts_each_declared_controller_without_isaac(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, controller: str
) -> None:
    checkpoint = tmp_path / f"{controller}.pt"
    checkpoint.write_bytes(controller.encode())
    monkeypatch.setattr(
        checkpoint_module,
        "read_checkpoint",
        lambda *_args, **_kwargs: _checkpoint_payload(controller),
    )
    spec = inspect_command_checkpoint(checkpoint)
    assert spec.controller == controller
    assert spec.command_follow_contract["version"] == COMMAND_FOLLOW_CONTRACT_VERSION
    assert (
        spec.command_follow_contract["observation_contract_version"]
        == OBSERVATION_CONTRACT_VERSION
    )


def test_checkpoint_inspector_rejects_old_waypoint_checkpoint_before_isaac(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "old-waypoint.pt"
    checkpoint.write_bytes(b"old")
    payload = _checkpoint_payload("mlp_normal")
    payload["resolved_config"]["task"] = "FlyCrazyflie-WaypointReach-v0"
    monkeypatch.setattr(
        checkpoint_module, "read_checkpoint", lambda *_args, **_kwargs: payload
    )
    with pytest.raises(ValueError, match="checkpoint task must be"):
        inspect_command_checkpoint(checkpoint)


def test_checkpoint_inspector_rejects_changed_command_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "wrong-contract.pt"
    checkpoint.write_bytes(b"wrong")
    payload = _checkpoint_payload("gru_matched")
    payload["resolved_config"]["command_follow_contract"][
        "maximum_horizontal_speed_m_s"
    ] = 99.0
    monkeypatch.setattr(
        checkpoint_module, "read_checkpoint", lambda *_args, **_kwargs: payload
    )
    with pytest.raises(ValueError, match="differs from current task contract"):
        inspect_command_checkpoint(checkpoint)


def _spec(controller: str, report: dict[str, object]) -> CommandCheckpointSpec:
    return CommandCheckpointSpec(
        path=Path("/tmp/checkpoint.pt"),
        sha256="x",
        controller=controller,
        canonical_controller="x",
        resolved_config={},
        controller_report=report,
        command_follow_contract=command_follow_contract_payload(),
        fingerprints={},
        task_manifest_id="task",
        evaluation_manifest_id="evaluation",
        observation_normalizer_state={},
    )


def test_controller_build_arguments_filter_none_widths_and_route_combined_manifests() -> None:
    spec = _spec(
        "leg_wing_lif",
        {
            "widths": {
                "adapter_hidden_dim": 64,
                "critic_hidden_dim": 128,
                "gru_hidden_dim": None,
            },
            "connectome_manifests": {"leg": "/leg.json", "wing": "/wing.json"},
        },
    )
    result = _controller_build_arguments(spec)
    assert result["widths"] == {
        "adapter_hidden_dim": 64,
        "critic_hidden_dim": 128,
    }
    assert result["connectome_manifest"] == "/leg.json"
    assert result["wing_connectome_manifest"] == "/wing.json"


def test_gru_activity_is_taken_from_actual_returned_policy_state() -> None:
    policy = torch.nn.Identity()
    spec = _spec(
        "gru_matched",
        {
            "widths": {"gru_hidden_dim": 3},
            "connectome_manifests": {"primary": "/unused.json"},
        },
    )
    recorder = TrainedActivityRecorder(policy, spec)
    state = torch.tensor([[0.0, -0.2, 0.4]])
    summary = recorder.update(state)
    assert summary["active_neurons"] == 2
    assert recorder.last_activity.tolist() == pytest.approx([0.0, 0.2, 0.4])
    recorder.close()


def test_gru_activity_batch_preserves_every_evaluator_row() -> None:
    policy = torch.nn.Identity()
    spec = _spec(
        "gru_matched",
        {
            "widths": {"gru_hidden_dim": 3},
            "connectome_manifests": {"primary": "/unused.json"},
        },
    )
    recorder = TrainedActivityRecorder(policy, spec)
    state = torch.tensor([[0.0, -0.2, 0.4], [-0.5, 0.6, -0.7]])
    activity = recorder.activity_batch(state)
    assert activity.shape == (2, 3)
    assert torch.allclose(
        activity,
        torch.tensor([[0.0, 0.2, 0.4], [0.5, 0.6, 0.7]]),
    )
    recorder.close()


def test_lif_activity_batch_uses_actual_spike_rows_without_reduction() -> None:
    manifest = (
        Path(__file__).resolve().parents[2] / "data" / "connectome" / "manifest.json"
    )
    spec = _spec(
        "frozen_lif_original",
        {
            "widths": {"adapter_hidden_dim": 64},
            "connectome_manifests": {"primary": str(manifest)},
        },
    )
    recorder = TrainedActivityRecorder(torch.nn.Identity(), spec)
    spikes = torch.zeros((2, len(recorder.roles)))
    spikes[0, 3] = 1.0
    spikes[1, 9] = 1.0
    state = SimpleNamespace(
        spikes=spikes,
        membrane=torch.zeros_like(spikes),
        synapse=torch.zeros_like(spikes),
        refractory=torch.zeros_like(spikes),
    )
    activity = recorder.activity_batch(state)
    assert torch.equal(activity, spikes)
    recorder.close()


def test_mlp_activity_hooks_observe_the_same_actor_forward_pass() -> None:
    policy = SimpleNamespace(
        actor=torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.ELU(),
            torch.nn.Linear(3, 2),
            torch.nn.ELU(),
            torch.nn.Linear(2, 1),
        )
    )
    spec = _spec(
        "mlp_normal",
        {
            "widths": {"mlp_hidden_dims": [3, 2]},
            "connectome_manifests": {"primary": "/unused.json"},
        },
    )
    recorder = TrainedActivityRecorder(policy, spec)
    recorder.begin_step()
    policy.actor(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
    activity_batch = recorder.activity_batch(None)
    assert activity_batch.shape == (2, 5)
    assert bool((activity_batch >= 0.0).all())
    summary = recorder.update(None)
    assert recorder.last_activity.shape == (5,)
    assert 0 <= summary["active_neurons"] <= 5
    recorder.close()


def test_runtime_script_declares_checkpoint_primary_action_and_manual_api() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "crazyflie_trained_keyboard.py"
    ).read_text(encoding="utf-8")
    assert 'required=True' in source
    assert "set_manual_command_body" in source
    assert "TRAINED POLICY CONTROLS ACTION" in source
    assert '"trained_policy"' in source
    assert "fallback_action_steps" in source
