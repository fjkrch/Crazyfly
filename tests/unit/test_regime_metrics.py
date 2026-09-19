"""Post-hoc regime metrics use only valid, named, pre-action CPU samples."""

import json
import math
import sys

import numpy as np
import pytest

from scripts.regime_metrics import SAMPLE_PHASE, main, summarize_recording, summarize_recordings


def _recording(tmp_path, name="trace.npz", *, training_seed=0, evaluation_seed=101,
               condition="frozen_lif_original", scenario="GoalReach", **overrides):
    root = np.zeros((4, 2, 13), dtype=np.float64)
    root[:, :, 3] = 1.0  # identity wxyz
    root[:, 0, 2] = [1.0, 1.2, 0.8, 1.0]
    root[:2, 1, 2] = [0.4, 0.6]
    # 90 degrees around x, with a deliberately non-unit but valid quaternion.
    root[1, 0, 3:7] = [math.sqrt(2), math.sqrt(2), 0.0, 0.0]
    root[2, 0, 3:7] = [0.0, 1.0, 0.0, 0.0]  # inverted root axis
    root[:2, 1, 3:7] = [0.0, 1.0, 0.0, 0.0]
    force = np.zeros((4, 2, 2, 3), dtype=np.float64)
    force[:, 0, 0, 0] = [0.0, 20.0, 30.0, 10.0]
    force[:2, 1, 0, 0] = [40.0, 0.0]
    force[:2, 1, 1, 1] = [20.0, 40.0]
    valid = np.array([[True, True], [True, True], [True, False], [True, False]])
    # Poison post-episode auto-reset rows. The analyzer must never read them.
    root[2:, 1] = np.nan
    force[2:, 1] = np.nan
    metadata = {
        "condition": condition, "scenario": scenario,
        "training_seed": training_seed, "evaluation_seed": evaluation_seed,
        "control_dt_s": 0.02, "sample_phase": SAMPLE_PHASE,
    }
    arrays = {
        "root_state": root, "contact_net_forces_w": force, "valid_step": valid,
        "episode_id": np.array([0, 1], dtype=np.int64),
        "body_names": np.array(["left_palm_link", "torso_link"]),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    arrays.update(overrides)
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return path


def test_posthoc_summary_masks_finished_episodes_normalizes_quaternion_and_uses_named_forces(tmp_path):
    path = _recording(tmp_path)
    result = summarize_recording(path, force_threshold_n=20.0)
    first, second = result["episodes"]
    assert [row["valid_control_steps"] for row in result["episodes"]] == [4, 2]
    assert first["root_height_m"]["mean"] == pytest.approx(1.0)
    assert second["root_height_m"]["mean"] == pytest.approx(0.5)
    upright = first["root_up_axis_alignment_cosine"]
    assert upright["mean"] == pytest.approx(0.25)
    assert upright["fraction_at_or_above_positive_threshold"] == pytest.approx(0.5)
    assert upright["fraction_at_or_below_negative_threshold"] == pytest.approx(0.25)
    left = first["net_body_force"]["left_palm_link"]
    assert left["mean_magnitude_n"] == pytest.approx(15.0)
    assert left["fraction_at_or_above_threshold"] == pytest.approx(0.5)  # equality counts
    assert second["net_body_force"]["torso_link"]["fraction_at_or_above_threshold"] == 1.0
    seed = result["per_training_seed"]
    # Equal episode weighting differs from pooled timestep weighting: (1 + .5)/2.
    assert seed["root_height_m"]["mean_of_episode_means"] == pytest.approx(0.75)
    assert seed["valid_control_steps"] == 6
    assert seed["net_body_force"]["left_palm_link"]["max_observed_magnitude_n"] == 40.0
    assert result["definitions"]["force_threshold_n"] == 20.0
    assert "ground support unknown" in result["definitions"]["contact_measure"]
    assert len(result["recording_sha256"]) == 64


def test_mask_must_be_true_prefix_and_valid_samples_must_be_finite(tmp_path):
    path = _recording(tmp_path, "gap.npz", valid_step=np.array([
        [True, True], [False, True], [True, False], [False, False]
    ]))
    with pytest.raises(ValueError, match="true prefix"):
        summarize_recording(path, force_threshold_n=20.0)

    force = np.zeros((4, 2, 2, 3))
    force[0, 0, 0, 0] = np.nan
    path = _recording(tmp_path, "bad_force.npz", contact_net_forces_w=force)
    with pytest.raises(ValueError, match="non-finite"):
        summarize_recording(path, force_threshold_n=20.0)


def test_zero_quaternion_and_missing_body_names_are_rejected(tmp_path):
    root = np.zeros((4, 2, 13))
    root[:, :, 3] = 1.0
    root[0, 0, 3:7] = 0.0
    path = _recording(tmp_path, "bad_quat.npz", root_state=root)
    with pytest.raises(ValueError, match="invalid root quaternion"):
        summarize_recording(path, force_threshold_n=20.0)

    path = _recording(tmp_path, "names.npz", body_names=np.array(["torso_link", "torso_link"]))
    with pytest.raises(ValueError, match="duplicates"):
        summarize_recording(path, force_threshold_n=20.0)


def test_thresholds_and_pre_action_metadata_are_explicit(tmp_path):
    path = _recording(tmp_path)
    for threshold in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="force_threshold_n"):
            summarize_recording(path, force_threshold_n=threshold)
    with pytest.raises(ValueError, match="upright_cos_threshold"):
        summarize_recording(path, force_threshold_n=20.0, upright_cos_threshold=1.1)
    bad_metadata = {
        "condition": "frozen_lif_original", "scenario": "GoalReach", "training_seed": 0,
        "evaluation_seed": 101, "control_dt_s": 0.02, "sample_phase": "post_action_auto_reset",
    }
    path = _recording(tmp_path, "phase.npz", metadata_json=np.asarray(json.dumps(bad_metadata)))
    with pytest.raises(ValueError, match="sample_phase"):
        summarize_recording(path, force_threshold_n=20.0)


def test_multiple_recordings_group_by_training_seed_and_reject_duplicate_episode_ids(tmp_path):
    first = _recording(tmp_path, "first.npz", training_seed=0, evaluation_seed=101)
    second = _recording(tmp_path, "second.npz", training_seed=0, evaluation_seed=102)
    third = _recording(tmp_path, "third.npz", training_seed=1, evaluation_seed=101)
    result = summarize_recordings([first, second, third], force_threshold_n=20.0)
    assert [(row["training_seed"], row["episode_count"]) for row in result["per_training_seed"]] == [
        (0, 4), (1, 2)
    ]
    assert result["per_training_seed"][0]["valid_control_steps"] == 12
    assert len(result["recordings"]) == 3
    with pytest.raises(ValueError, match="Duplicate recorded evaluation episode"):
        summarize_recordings([first, first], force_threshold_n=20.0)


def test_cli_writes_inspectable_seed_summary(tmp_path, monkeypatch):
    recording = _recording(tmp_path)
    output = tmp_path / "regimes.json"
    monkeypatch.setattr(sys, "argv", [
        "regime_metrics.py", str(recording), "--force-threshold-n", "20", "--output", str(output)
    ])
    assert main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "descriptive_posthoc_summary"
    assert len(result["per_training_seed"]) == 1
    assert result["per_training_seed"][0]["episode_count"] == 2
