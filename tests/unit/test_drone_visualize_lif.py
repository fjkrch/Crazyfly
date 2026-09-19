from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from drone_visualize_lif import (  # noqa: E402
    BrainWindow,
    CONTROL_DT_S,
    CLOSE_FOLLOW_CAMERA_REFRESH_STEPS,
    DEFAULT_CLOSE_FOLLOW_CAMERA_OFFSET_M,
    LIFActivityTracker,
    MAX_VISUALIZATION_STEPS,
    _deterministic_lif_action,
    _close_follow_camera_pose,
    _brain_window_packet,
    _format_sidecar,
    _pace_deadline,
    _render_activity_bar,
    _resolved_config,
    _steady_state_sample_steps,
    _set_close_follow_camera,
    _validate_camera_offset,
    _validate_lif_controller,
)


def _state(
    spikes: list[float],
    *,
    membrane: list[float] | None = None,
    synapse: list[float] | None = None,
) -> SimpleNamespace:
    spike_tensor = torch.tensor([spikes], dtype=torch.float32)
    membrane_tensor = torch.tensor(
        [membrane if membrane is not None else spikes], dtype=torch.float32
    )
    synapse_tensor = torch.tensor(
        [synapse if synapse is not None else spikes], dtype=torch.float32
    )
    return SimpleNamespace(
        membrane=membrane_tensor,
        spikes=spike_tensor,
        synapse=synapse_tensor,
        refractory=torch.zeros_like(spike_tensor, dtype=torch.long),
    )


@pytest.mark.parametrize(
    "controller", ("frozen_lif_original", "frozen_lif_degree_rewired")
)
def test_only_the_two_declared_lif_checkpoints_are_accepted(controller: str) -> None:
    assert _validate_lif_controller(controller) == controller


@pytest.mark.parametrize("controller", ("gru_matched", "mlp_normal", None))
def test_non_lif_checkpoint_is_rejected_before_isaac(controller) -> None:
    with pytest.raises(ValueError, match="LIF visualization requires"):
        _validate_lif_controller(controller)


def test_memory_schedule_is_four_bounded_increasing_quarters() -> None:
    assert _steady_state_sample_steps(600) == (150, 300, 450, 600)
    assert _steady_state_sample_steps(5) == (2, 3, 4, 5)
    with pytest.raises(ValueError, match=">= 4"):
        _steady_state_sample_steps(3)


def test_realtime_deadline_uses_the_50_hz_control_contract() -> None:
    assert _pace_deadline(10.0, 25, 1.0) == pytest.approx(10.5)
    assert _pace_deadline(10.0, 25, 2.0) == pytest.approx(10.25)
    with pytest.raises(ValueError, match="positive"):
        _pace_deadline(10.0, 0, 1.0)


def test_close_follow_camera_offset_and_pose_are_finite_and_close() -> None:
    offset = _validate_camera_offset(DEFAULT_CLOSE_FOLLOW_CAMERA_OFFSET_M)
    eye, target = _close_follow_camera_pose((2.0, -1.0, 1.5), offset)
    assert target == [2.0, -1.0, 1.5]
    assert eye == pytest.approx([0.85, -2.15, 2.05])
    assert CLOSE_FOLLOW_CAMERA_REFRESH_STEPS == 5

    with pytest.raises(ValueError, match="three finite"):
        _validate_camera_offset((1.0, float("nan"), 1.0))
    with pytest.raises(ValueError, match="distance"):
        _validate_camera_offset((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="root position"):
        _close_follow_camera_pose((1.0, 2.0), offset)


def test_close_follow_camera_uses_env0_and_sim_camera_api() -> None:
    class Sim:
        def __init__(self) -> None:
            self.call = None

        def set_camera_view(self, *, eye, target) -> None:
            self.call = {"eye": eye, "target": target}

    sim = Sim()
    env = SimpleNamespace(
        sim=sim,
        _robot=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[3.0, 4.0, 2.0], [99.0, 99.0, 99.0]])
            )
        ),
    )
    pose = _set_close_follow_camera(env, (-1.0, -1.0, 0.5))
    assert sim.call == {"eye": [2.0, 3.0, 2.5], "target": [3.0, 4.0, 2.0]}
    assert pose == {"eye_w_m": [2.0, 3.0, 2.5], "target_w_m": [3.0, 4.0, 2.0]}


def test_brain_window_packet_is_small_finite_and_cpu_only() -> None:
    frame = {
        "step": 10,
        "requested_steps": 100,
        "neuron_count": 4,
        "lif_activity": {
            "final_substep_spike_count": 2,
            "window_sampled_spike_rate_hz_per_neuron": 12.5,
            "membrane_l2": 3.0,
            "synapse_l2": 4.0,
        },
    }
    packet = _brain_window_packet(frame, torch.tensor([0.0, 0.5, 1.0, 0.25]))
    assert packet["step"] == 10
    assert packet["spike_count"] == 2
    assert packet["window_activity"] == [0.0, 0.5, 1.0, 0.25]
    with pytest.raises(ValueError, match="finite"):
        _brain_window_packet(frame, torch.tensor([0.0, float("inf")]))


def test_brain_window_cleanup_terminates_an_unresponsive_child() -> None:
    class Channel:
        def __init__(self) -> None:
            self.values = []
            self.closed = False

        def put_nowait(self, value) -> None:
            self.values.append(value)

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            pass

    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False

        def join(self, timeout) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    window = BrainWindow.__new__(BrainWindow)
    window._frames = Channel()
    window._status = Channel()
    window._process = Process()
    window.backend = "test"
    window.update_count = 3
    window.cleanup = "running"
    audit = window.close()
    assert window._process.terminated is True
    assert window._frames.values == [None]
    assert window._frames.closed is True
    assert window._status.closed is True
    assert audit["cleanup"] == "terminated"


def test_policy_helper_cannot_request_stochastic_inference() -> None:
    class Policy:
        def __init__(self) -> None:
            self.deterministic = None

        def act(self, observation, state, *, deterministic):
            self.deterministic = deterministic
            return observation, state

    policy = Policy()
    observation = torch.zeros(1, 12)
    state = object()
    assert _deterministic_lif_action(policy, observation, state) == (
        observation,
        state,
    )
    assert policy.deterministic is True


def test_activity_tracker_reports_bounded_final_substep_samples() -> None:
    tracker = LIFActivityTracker(window_steps=2, bar_width=4)
    first = tracker.update(
        _state([1, 0, 0, 1], membrane=[3, 4, 0, 0], synapse=[0, 0, 0, 2])
    )
    second = tracker.update(
        _state([0, 1, 0, 1], membrane=[0, 0, 0, 0], synapse=[0, 3, 4, 0])
    )
    assert first["final_substep_spike_count"] == 2
    assert second["window_control_steps"] == 2
    assert second["window_sampled_spike_fraction"] == pytest.approx(0.5)
    assert second["window_sampled_spike_rate_hz_per_neuron"] == pytest.approx(
        0.5 / CONTROL_DT_S
    )
    assert len(second["relative_activity_bar"]) == 4

    summary = tracker.summary()
    assert summary["sampled_control_steps"] == 2
    assert summary["neuron_count"] == 4
    assert summary["total_sampled_spikes"] == 4
    assert summary["sampled_spike_fraction"] == pytest.approx(0.5)
    assert summary["sampled_spike_rate_hz_per_neuron"] == pytest.approx(25.0)
    assert summary["dead_neuron_fraction"] == pytest.approx(0.25)
    assert summary["saturated_neuron_fraction"] == pytest.approx(0.25)
    assert summary["membrane_l2_mean"] == pytest.approx(2.5)
    assert summary["synapse_l2_mean"] == pytest.approx(3.5)
    assert summary["unobserved_internal_neural_substeps_counted"] is False


def test_activity_window_is_bounded_and_rejects_invalid_lif_state() -> None:
    tracker = LIFActivityTracker(window_steps=2, bar_width=4)
    tracker.update(_state([1, 0, 0, 0]))
    tracker.update(_state([0, 1, 0, 0]))
    current = tracker.update(_state([0, 0, 1, 0]))
    assert current["window_control_steps"] == 2
    assert tracker.summary()["sampled_control_steps"] == 3

    with pytest.raises(FloatingPointError, match="not binary"):
        LIFActivityTracker(window_steps=1, bar_width=4).update(
            _state([0.5, 0, 0, 0])
        )
    bad_shape = _state([0, 0, 0, 0])
    bad_shape.spikes = bad_shape.spikes.repeat(2, 1)
    with pytest.raises(ValueError, match=r"\[1, neurons\]"):
        LIFActivityTracker(window_steps=1, bar_width=4).update(bad_shape)


def test_relative_activity_bar_has_fixed_width_and_zero_state_is_blank() -> None:
    assert _render_activity_bar(torch.zeros(8), 8) == " " * 8
    bar = _render_activity_bar(torch.arange(1, 9), 4)
    assert len(bar) == 4
    assert bar[-1] == "@"


def test_resolved_contract_hardcodes_one_env_deterministic_and_no_mutation() -> None:
    args = argparse.Namespace(
        task="FlyCrazyflie-WaypointReach-v0",
        steps=100,
        seed=7,
        headless=False,
        realtime_rate=1.0,
        display_interval=10,
        activity_window_steps=25,
        activity_bar_width=48,
        close_follow_camera=False,
        camera_offset=DEFAULT_CLOSE_FOLLOW_CAMERA_OFFSET_M,
        brain_window=False,
    )
    resolved = _resolved_config(args, {"controller": "frozen_lif_original"})
    assert resolved["num_envs"] == 1
    assert resolved["deterministic_actions"] is True
    assert resolved["viewer_enabled"] is True
    assert resolved["checkpoint_mutation_allowed"] is False
    assert resolved["maximum_allowed_steps"] == MAX_VISUALIZATION_STEPS == 600
    assert resolved["camera"]["mode"] == "default_viewer"
    assert resolved["camera"]["active"] is False
    assert resolved["brain_window"]["active"] is False


def test_resolved_contract_records_active_close_follow_camera() -> None:
    args = argparse.Namespace(
        task="FlyCrazyflie-GustRecovery-v0",
        steps=100,
        seed=7,
        headless=False,
        realtime_rate=1.0,
        display_interval=10,
        activity_window_steps=25,
        activity_bar_width=48,
        close_follow_camera=True,
        camera_offset=(-1.0, -1.0, 0.5),
        brain_window=True,
    )
    resolved = _resolved_config(args, {"controller": "frozen_lif_original"})
    assert resolved["camera"] == {
        "mode": "close_follow_env0",
        "requested": True,
        "active": True,
        "world_frame_eye_offset_m": [-1.0, -1.0, 0.5],
        "refresh_interval_control_steps": 5,
        "update_backend": "reused_usd_matrix_op_no_kit_commands",
    }
    assert resolved["brain_window"] == {
        "requested": True,
        "active": True,
        "separate_process": True,
        "backend": "tkinter",
        "update_interval_control_steps": 10,
        "terminal_sidecar_retained": True,
    }


def test_sidecar_labels_sample_semantics_and_flight_outcome() -> None:
    frame = {
        "step": 10,
        "requested_steps": 100,
        "simulation_time_s": 0.2,
        "wall_time_s": 0.2,
        "realtime_factor": 1.0,
        "goal_distance_m": 0.5,
        "speed_m_s": 0.1,
        "reward": 0.02,
        "success_count": 0,
        "terminated": False,
        "truncated": False,
        "failure_cause": 0,
        "action": [0.05, 0.0, 0.0, 0.0],
        "neuron_count": 4,
        "lif_activity": {
            "final_substep_spike_count": 1,
            "window_sampled_spike_rate_hz_per_neuron": 2.5,
            "relative_activity_bar": " @  ",
            "membrane_l2": 1.0,
            "synapse_l2": 2.0,
        },
    }
    text = _format_sidecar(frame, None)
    assert "final-substep spikes=1/4" in text
    assert "RUNNING" in text
    assert "memory=pending" in text


def test_documentation_exposes_viewer_and_headless_commands() -> None:
    spec = (ROOT / "docs/crazyflie_task_spec.md").read_text(encoding="utf-8")
    assert "drone_visualize_lif.py" in spec
    assert "terminal LIF sidecar" in spec
    assert "--headless" in spec
