"""CPU-only checks for the held-out replay's reference gate and NPZ contract."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluation_protocol import HeldOutEvents  # noqa: E402
from record_heldout_regimes import _paired_plan_sha256, _reference, _reference_checks, _safe_output_path, _write_npz  # noqa: E402
from regime_metrics import summarize_recording  # noqa: E402


TASK = "FlyG1-PushRecovery-FreePosture-v0"


def _evaluation(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = {"task": "FlyG1-GoalReach-FreePosture-v0", "policy": "frozen_lif", "seed": 7}
    schedule = HeldOutEvents(101, 16, TASK)
    result = {
        "status": "executed", "checkpoint": str(checkpoint), "task": TASK,
        "training_seed": 7, "evaluation_seed": 101, "n_episodes": 16,
        "ablation": None,
        "execution": {"control_steps": 1000},
        "scenario": {
            "task": TASK, "evaluation_protocol": "heldout_v1",
            "policy_action_mode": "deterministic", "schedule": schedule.manifest,
        },
        "episodes": [
            {"episode_id": index, "seed": 7,
             "success": False, "time_to_target_s": None, "schedule_events": [],
             "initial_state_sha256": "a" * 64,
             "paired_plan_sha256": _paired_plan_sha256(schedule, index)}
            for index in range(16)
        ],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path, checkpoint, metadata, result


def test_reference_accepts_json_roundtrip_and_rejects_wrong_replay_identity(tmp_path):
    path, checkpoint, metadata, result = _evaluation(tmp_path)
    reference, schedule = _reference(path, checkpoint, metadata)
    assert len(reference["episodes"]) == 16
    assert schedule.manifest["sha256"] == result["scenario"]["schedule"]["sha256"]

    result["episodes"][5]["paired_plan_sha256"] = "0" * 64
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="paired plan"):
        _reference(path, checkpoint, metadata)

    result["episodes"][5]["paired_plan_sha256"] = _paired_plan_sha256(schedule, 5)
    result["scenario"]["schedule"]["sha256"] = "0" * 64
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule"):
        _reference(path, checkpoint, metadata)

    result["scenario"]["schedule"] = schedule.manifest
    result["evaluation_seed"] = 102
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="seed must be 101"):
        _reference(path, checkpoint, metadata)


def test_written_recording_round_trips_through_regime_analyzer(tmp_path):
    root = np.full((4, 16, 13), np.nan, dtype=np.float32)
    forces = np.full((4, 16, 2, 3), np.nan, dtype=np.float32)
    valid = np.ones((4, 16), dtype=np.bool_)
    valid[2:, 0] = False  # Auto-reset rows for episode 0 remain invalid NaNs.
    for index in range(16):
        count = 2 if index == 0 else 4
        root[:count, index] = 0.0
        root[:count, index, 2] = 0.8
        root[:count, index, 3] = 1.0  # identity quaternion in Isaac wxyz order
        forces[:count, index] = 0.0
        forces[:count, index, 0, 2] = 25.0
    path = tmp_path / "recording.npz"
    _write_npz(
        path, root_state=root, forces=forces, valid=valid,
        body_names=["left_ankle_roll_link", "torso_link"],
        metadata={
            "condition": "frozen_lif_original", "scenario": TASK,
            "training_seed": 7, "evaluation_seed": 101,
            "control_dt_s": 0.02,
            "sample_phase": "pre_action_before_first_episode_reset",
        },
    )
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["root_state"].shape == (4, 16, 13)
        assert arrays["contact_net_forces_w"].shape == (4, 16, 2, 3)
    summary = summarize_recording(path, force_threshold_n=20.0)
    assert summary["per_training_seed"]["episode_count"] == 16
    assert [item["valid_control_steps"] for item in summary["episodes"][:2]] == [2, 4]
    assert summary["episodes"][0]["net_body_force"]["left_ankle_roll_link"]["fraction_at_or_above_threshold"] == 1.0


def test_output_path_cannot_replace_a_source_file(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    evaluation = tmp_path / "evaluation.json"
    connectome = tmp_path / "connectome.json"
    for source in (checkpoint, evaluation, connectome):
        with pytest.raises(ValueError, match="must differ"):
            _safe_output_path(source, checkpoint, evaluation, connectome)
    assert _safe_output_path(tmp_path / "trace.npz", checkpoint, evaluation, connectome) == tmp_path / "trace.npz"


def test_companion_replay_checks_outcomes_events_and_steps(tmp_path):
    _, _, _, reference = _evaluation(tmp_path)
    event = {"kind": "target", "ordinal": 0, "time_s": 0.0,
             "relative_offset_xy_m": [2.0, 0.0], "goal_xy_world_m": [2.0, 0.0]}
    for episode in reference["episodes"]:
        episode["schedule_events"] = [event]
    replay_events = [[dict(event)] for _ in range(16)]
    success = [False] * 16
    times = [None] * 16
    matched = _reference_checks(reference, replay_events, success, times, 1000)
    assert matched["required_match"] is True
    assert sum(matched["full_event_match_by_episode"]) == 16

    changed_world = [[{**event, "goal_xy_world_m": [2.1, 0.0]}] for _ in range(16)]
    companion = _reference_checks(reference, changed_world, success, times, 1000)
    assert companion["required_match"] is True
    assert not any(companion["full_event_match_by_episode"])

    wrong_success = success.copy()
    wrong_success[3] = True
    assert _reference_checks(reference, replay_events, wrong_success, times, 1000)["required_match"] is False
    assert _reference_checks(reference, replay_events, success, times, 999)["required_match"] is False
    wrong_event = [[{**event, "ordinal": 1}] for _ in range(16)]
    assert _reference_checks(reference, wrong_event, success, times, 1000)["required_match"] is False
