"""The regime comparison validates the full 60-cell replay lattice on CPU."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation_protocol import schedule_manifest
from scripts.summarize_regime_matrix import (
    CONDITIONS, EPISODES, SCENARIOS, SEEDS, render_markdown, summarize_regime_matrix,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _matrix(tmp_path: Path) -> tuple[Path, dict]:
    root = np.zeros((2, EPISODES, 13), dtype=np.float32)
    forces = np.zeros((2, EPISODES, 2, 3), dtype=np.float32)
    valid = np.ones((2, EPISODES), dtype=np.bool_)
    names = np.asarray(["left_foot_link", "torso_link"])
    code_sha = {"scripts/record_heldout_regimes.py": "a" * 64}
    source_fingerprint = "b" * 64
    jobs = []
    batch_rows = []
    for ci, condition in enumerate(CONDITIONS):
        for seed in SEEDS:
            checkpoint = tmp_path / f"{condition}-{seed}.pt"
            checkpoint.write_bytes(f"checkpoint-{condition}-{seed}".encode())
            main_job = {
                "id": f"{condition}__seed-{seed}", "condition": condition,
                "seed": seed, "status": "passed", "checkpoint": str(checkpoint),
                "evaluations": {},
            }
            for scenario in SCENARIOS:
                tag = f"{condition}-{seed}-{scenario}"
                evaluation_path = tmp_path / f"{tag}.json"
                recording_path = tmp_path / f"{tag}.npz"
                schedule = schedule_manifest(101, EPISODES, scenario)
                schedule_sha = schedule["sha256"]
                initial = [sha256(f"initial-{seed}-{i}".encode()).hexdigest() for i in range(EPISODES)]
                paired = [sha256(json.dumps({
                    "scenario_schedule_sha256": schedule_sha,
                    "environment_plan": schedule["plan"][i],
                }, sort_keys=True).encode()).hexdigest() for i in range(EPISODES)]
                evaluation = {
                    "status": "executed", "task": scenario, "training_seed": seed,
                    "evaluation_seed": 101, "n_episodes": EPISODES, "ablation": None,
                    "checkpoint": str(checkpoint),
                    "execution": {"control_steps": 2},
                    "scenario": {"task": scenario, "evaluation_protocol": "heldout_v1",
                                 "policy_action_mode": "deterministic", "control_dt_s": 0.02,
                                 "schedule": schedule},
                    "episodes": [{"episode_id": i, "seed": seed,
                                  "success": False, "time_to_target_s": None,
                                  "schedule_events": [],
                                  "initial_state_sha256": initial[i],
                                  "paired_plan_sha256": paired[i]} for i in range(EPISODES)],
                }
                _write_json(evaluation_path, evaluation)
                root.fill(0)
                root[:, :, 2] = 0.5 + 0.1 * seed + 0.2 * ci
                root[:, :, 3] = 1.0
                if ci == 1:  # inverted at the second of two steps
                    root[1, :, 3] = 0.0
                    root[1, :, 4] = 1.0
                forces.fill(0)
                forces[0, :, 0, 2] = 25.0
                if ci > 0:
                    forces[1, :, 0, 2] = 25.0
                metadata = {
                    "schema_version": "heldout_regimes_v1",
                    "condition": condition, "scenario": scenario, "task": scenario,
                    "training_seed": seed, "evaluation_seed": 101,
                    "episode_count": EPISODES, "control_dt_s": 0.02,
                    "sample_phase": "pre_action_before_first_episode_reset",
                    "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
                    "evaluation_json": str(evaluation_path),
                    "evaluation_json_sha256": _sha(evaluation_path),
                    "scenario_schedule_sha256": schedule_sha,
                    "initial_state_sha256": initial,
                    "paired_plan_sha256": paired,
                    "execution_source_fingerprint": source_fingerprint,
                    "code_sha256": code_sha,
                    "connectome_manifest": None,
                    "connectome_manifest_sha256": None,
                    "reference_checks": {
                        "required_match": True,
                        "success_match_by_episode": [True] * EPISODES,
                        "target_time_match_by_episode": [True] * EPISODES,
                        "scheduled_event_match_by_episode": [True] * EPISODES,
                        "full_event_match_by_episode": [True] * EPISODES,
                        "control_steps_match": True,
                        "replay_success_by_episode": [False] * EPISODES,
                        "replay_target_time_s_by_episode": [None] * EPISODES,
                        "replay_control_steps": 2,
                    },
                }
                np.savez_compressed(
                    recording_path, root_state=root, contact_net_forces_w=forces,
                    valid_step=valid, episode_id=np.arange(EPISODES), body_names=names,
                    metadata_json=np.asarray(json.dumps(metadata)),
                )
                main_job["evaluations"][scenario] = {
                    "status": "passed", "result_file": str(evaluation_path),
                }
                batch_rows.append({
                    "id": f"{condition}__seed-{seed}__{scenario}",
                    "condition": condition, "training_seed": seed,
                    "evaluation_seed": 101, "scenario": scenario,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
                    "evaluation_json": str(evaluation_path),
                    "evaluation_json_sha256": _sha(evaluation_path),
                    "connectome_manifest": None, "connectome_manifest_sha256": None,
                    "output_npz": str(recording_path), "output_npz_sha256": _sha(recording_path),
                    "status": "passed", "log": str(tmp_path / f"{tag}.log"),
                })
            jobs.append(main_job)
    main = {
        "status": "complete", "execution_source_fingerprint": source_fingerprint,
        "config": {"conditions": list(CONDITIONS), "seeds": list(SEEDS),
                   "evaluation": {"scenario_tasks": list(SCENARIOS), "episodes": EPISODES}},
        "jobs": jobs,
    }
    main_path = tmp_path / "main.json"
    _write_json(main_path, main)
    manifest = {
        "schema_version": "regime_matrix_v1", "status": "complete", "counts": {"passed": 60},
        "main_matrix": str(main_path), "main_matrix_sha256": _sha(main_path),
        "execution_source_fingerprint": source_fingerprint,
        "recorder_code_sha256": code_sha, "jobs": batch_rows,
    }
    manifest_path = tmp_path / "regime_batch.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def test_full_report_uses_five_independent_seeds_and_exact_named_force_fractions(tmp_path):
    manifest_path, _ = _matrix(tmp_path)
    report = summarize_regime_matrix(manifest_path=manifest_path, force_threshold_n=20.0)
    assert report["status"] == "complete"
    assert (report["validated_recordings"], report["validated_episodes"]) == (60, 960)
    assert all(len(value) == 64 for value in report["analysis_code_sha256"].values())
    assert report["body_names"] == ["left_foot_link", "torso_link"]
    scenario = SCENARIOS[0]
    original = report["by_scenario"][scenario][CONDITIONS[0]]
    height = original["metrics"]["root_height_m_mean_of_episode_means"]
    assert height["n_training_seeds"] == 5
    assert height["mean"] == pytest.approx(0.7)
    assert height["sample_sd"] == pytest.approx(math.sqrt(0.025))
    foot = original["metrics"]["net_body_force_threshold_fraction::left_foot_link"]
    assert foot == {"n_training_seeds": 5, "mean": 0.5, "sample_sd": 0.0}
    assert original["metrics"]["net_body_force_threshold_fraction::torso_link"]["mean"] == 0.0
    delta = report["paired_difference_vs_original"][scenario][CONDITIONS[1]]
    assert delta["training_seeds"] == list(SEEDS)
    assert delta["metrics"]["root_height_m_mean_of_episode_means"]["mean"] == pytest.approx(0.2)
    assert delta["metrics"]["net_body_force_threshold_fraction::left_foot_link"]["mean"] == 0.5
    assert delta["metrics"]["root_up_axis_positive_threshold_fraction"]["mean"] == -0.5
    markdown = render_markdown(report)
    assert "left_foot_link" in markdown and "ground-support" in markdown
    assert "Paired root-pose differences" in markdown


def test_missing_or_tampered_records_cannot_be_reported_complete(tmp_path):
    manifest_path, manifest = _matrix(tmp_path)
    manifest["jobs"][0]["status"] = "ready"
    _write_json(manifest_path, manifest)
    partial = summarize_regime_matrix(manifest_path=manifest_path)
    assert partial["status"] == "incomplete"
    assert partial["validated_recordings"] == 59
    assert len(partial["missing_recordings"]) == 1

    manifest["jobs"][0]["status"] = "passed"
    path = Path(manifest["jobs"][0]["evaluation_json"])
    evaluation = json.loads(path.read_text())
    evaluation["episodes"][0]["paired_plan_sha256"] = "0" * 64
    _write_json(path, evaluation)
    _write_json(manifest_path, manifest)
    bad = summarize_regime_matrix(manifest_path=manifest_path)
    assert bad["status"] == "incomplete"
    assert bad["validated_recordings"] == 59
    assert any("evaluation_json_sha256" in error for error in bad["validation_errors"])


def test_inconsistent_top_level_batch_status_cannot_claim_complete(tmp_path):
    manifest_path, manifest = _matrix(tmp_path)
    manifest["status"] = "partial_failure"
    manifest["counts"] = {"passed": 59, "failed": 1}
    _write_json(manifest_path, manifest)
    report = summarize_regime_matrix(manifest_path=manifest_path)
    assert report["status"] == "incomplete"
    assert any("status/counts" in error for error in report["validation_errors"])


def test_companion_replay_mismatch_cannot_claim_complete(tmp_path):
    manifest_path, manifest = _matrix(tmp_path)
    row = manifest["jobs"][0]
    path = Path(row["output_npz"])
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["reference_checks"]["required_match"] = False
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    row["output_npz_sha256"] = _sha(path)
    _write_json(manifest_path, manifest)
    report = summarize_regime_matrix(manifest_path=manifest_path)
    assert report["status"] == "incomplete"
    assert report["validated_recordings"] == 59
    assert any("companion replay" in error for error in report["validation_errors"])


def test_recording_hash_and_body_name_mismatch_are_rejected(tmp_path):
    manifest_path, manifest = _matrix(tmp_path)
    row = manifest["jobs"][1]
    path = Path(row["output_npz"])
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["body_names"] = np.asarray(["torso_link", "left_foot_link"])
    np.savez_compressed(path, **arrays)
    row["output_npz_sha256"] = _sha(path)
    _write_json(manifest_path, manifest)
    report = summarize_regime_matrix(manifest_path=manifest_path)
    assert report["status"] == "incomplete"
    assert report["validated_recordings"] == 59
    assert any("body list/order differs" in error for error in report["validation_errors"])


def test_direct_npz_inputs_require_full_lattice_and_matching_episode_pairing(tmp_path):
    manifest_path, manifest = _matrix(tmp_path)
    recordings = [row["output_npz"] for row in manifest["jobs"]]
    direct = summarize_regime_matrix(recordings)
    assert direct["status"] == "descriptive_only"
    assert direct["batch_manifest"] is None

    # A plausible NPZ with re-labelled paired plans is not the held-out replay.
    path = Path(recordings[1])
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["paired_plan_sha256"][0] = "0" * 64
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    bad = summarize_regime_matrix(recordings)
    assert bad["status"] == "incomplete"
    assert bad["validated_recordings"] == 59
    assert any("paired_plan_sha256" in error for error in bad["validation_errors"])


def test_coherently_relabelled_schedule_and_mixed_source_are_rejected(tmp_path):
    _, manifest = _matrix(tmp_path)
    recordings = [row["output_npz"] for row in manifest["jobs"]]
    path = Path(recordings[1])
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    evaluation_path = Path(metadata["evaluation_json"])
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["scenario"]["schedule"]["plan"][0]["target_offsets_m"][0][0] += 0.1
    _write_json(evaluation_path, evaluation)
    metadata["evaluation_json_sha256"] = _sha(evaluation_path)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    bad = summarize_regime_matrix(recordings)
    assert bad["validated_recordings"] == 59
    assert any("deterministic heldout_v1" in error for error in bad["validation_errors"])

    # Keep the rest valid but mix a second execution source in one direct NPZ.
    path = Path(recordings[2])
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["execution_source_fingerprint"] = "f" * 64
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    bad = summarize_regime_matrix(recordings)
    assert bad["validated_recordings"] == 58
    assert any("different execution source fingerprints" in error for error in bad["validation_errors"])
