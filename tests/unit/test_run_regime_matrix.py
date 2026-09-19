"""CPU-only queue gates, identity checks, and sequential launch planning."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_regime_matrix as queue  # noqa: E402


def _main_matrix(tmp_path):
    jobs = []
    for condition in queue.CONDITIONS:
        for seed in range(5):
            checkpoint = tmp_path / f"{condition}__seed-{seed}.pt"
            checkpoint.write_bytes(f"checkpoint {condition} {seed}".encode())
            evaluations = {}
            for scenario in queue.SCENARIOS:
                path = tmp_path / f"{condition}__seed-{seed}__{scenario}.json"
                path.write_text(json.dumps({
                    "scenario": {"schedule": {"sha256": "a" * 64}},
                    "execution": {"control_steps": 1},
                    "episodes": [{"initial_state_sha256": "b" * 64,
                                  "paired_plan_sha256": "c" * 64,
                                  "success": False, "time_to_target_s": None,
                                  "schedule_events": []} for _ in range(16)],
                }), encoding="utf-8")
                evaluations[scenario] = {"status": "passed", "result_file": str(path)}
            jobs.append({"id": f"{condition}__seed-{seed}", "condition": condition,
                         "seed": seed, "status": "passed", "checkpoint": str(checkpoint),
                         "evaluations": evaluations})
    connectome = tmp_path / "connectome.json"
    connectome.write_text("{}", encoding="utf-8")
    main = {
        "schema_version": 2, "status": "complete", "counts": {"passed": 20},
        "execution_source_fingerprint": queue._execution_source_fingerprint(),
        "config": {"conditions": list(queue.CONDITIONS), "seeds": list(range(5)),
                   "evaluation": {"protocol": "heldout_v1", "episodes": 16,
                                  "seed": 101, "scenario_tasks": list(queue.SCENARIOS)}},
        "prerequisites": {"real_connectome": {"status": "PASS", "path": str(connectome)}},
        "jobs": jobs,
    }
    path = tmp_path / "main.json"
    path.write_text(json.dumps(main), encoding="utf-8")
    return path, main


def _report(monkeypatch):
    monkeypatch.setattr(queue, "summarize", lambda path: {
        "status": "complete", "validated_jobs": 20, "expected_jobs": 20,
        "validation_errors": [], "source_execution_fingerprint": queue._execution_source_fingerprint(),
        "manifest_sha256": queue._sha256(path),
    })


def _args(main_path, *, execute=False, resume=False, max_jobs=None):
    return argparse.Namespace(main_matrix=main_path, execute=execute, resume=resume,
                              max_jobs=max_jobs, dry_run=not execute)


def _recording(path, row, manifest, *, checkpoint_sha=None):
    metadata = {
        "schema_version": "heldout_regimes_v1", "sample_phase": "pre_action_before_first_episode_reset",
        "condition": row["condition"], "scenario": row["scenario"], "task": row["scenario"],
        "training_seed": row["training_seed"], "evaluation_seed": 101, "episode_count": 16,
        "control_dt_s": 0.02, "checkpoint": row["checkpoint"],
        "checkpoint_sha256": checkpoint_sha or row["checkpoint_sha256"],
        "evaluation_json": row["evaluation_json"],
        "evaluation_json_sha256": row["evaluation_json_sha256"],
        "connectome_manifest": row["connectome_manifest"],
        "connectome_manifest_sha256": row["connectome_manifest_sha256"],
        "execution_source_fingerprint": manifest["execution_source_fingerprint"],
        "code_sha256": manifest["recorder_code_sha256"],
        "scenario_schedule_sha256": "a" * 64,
        "initial_state_sha256": ["b" * 64] * 16,
        "paired_plan_sha256": ["c" * 64] * 16,
        "reference_checks": {
            "required_match": True,
            "success_match_by_episode": [True] * 16,
            "target_time_match_by_episode": [True] * 16,
            "scheduled_event_match_by_episode": [True] * 16,
            "full_event_match_by_episode": [True] * 16,
            "control_steps_match": True,
            "replay_success_by_episode": [False] * 16,
            "replay_target_time_s_by_episode": [None] * 16,
            "replay_control_steps": 1,
        },
    }
    root = np.zeros((1, 16, 13), dtype=np.float32)
    root[:, :, 3] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, root_state=root,
                        contact_net_forces_w=np.zeros((1, 16, 1, 3), dtype=np.float32),
                        valid_step=np.ones((1, 16), dtype=np.bool_),
                        episode_id=np.arange(16, dtype=np.int32), body_names=np.asarray(["torso_link"]),
                        metadata_json=np.asarray(json.dumps(metadata)))


def test_incomplete_main_matrix_is_rejected_before_planning(tmp_path, monkeypatch):
    path, main = _main_matrix(tmp_path)
    main["status"] = "running"
    main["counts"] = {"running": 1, "ready": 19}
    path.write_text(json.dumps(main), encoding="utf-8")
    monkeypatch.setattr(queue, "summarize", lambda path: pytest.fail("validator must not run"))
    output = tmp_path / "regimes.json"
    with pytest.raises(ValueError, match="20 passed jobs"):
        queue._main_locked(_args(path), output)
    assert not output.exists()


def test_plan_has_60_paired_jobs_and_dry_run_never_launches(tmp_path, monkeypatch):
    path, _ = _main_matrix(tmp_path)
    _report(monkeypatch)
    monkeypatch.setattr(queue, "_run", lambda *args: pytest.fail("dry-run started Isaac"))
    output = tmp_path / "regimes.json"
    assert queue._main_locked(_args(path), output) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert manifest["counts"] == {"ready": 60}
    assert len(manifest["jobs"]) == 60
    assert manifest["main_matrix"] == str(path.resolve())
    assert manifest["main_matrix_sha256"] == queue._sha256(path)
    assert [row["scenario"] for row in manifest["jobs"][:3]] == list(queue.SCENARIOS)
    assert "--connectome_manifest" in queue._command(manifest["jobs"][0])
    assert "--connectome_manifest" not in queue._command(manifest["jobs"][-1])
    assert queue._command(manifest["jobs"][0])[-3:] == [
        "--headless", "--connectome_manifest", manifest["jobs"][0]["connectome_manifest"]]


def test_recording_resume_requires_content_and_metadata_identity(tmp_path, monkeypatch):
    path, main = _main_matrix(tmp_path)
    _report(monkeypatch)
    output = tmp_path / "regimes.json"
    planned = queue._new_manifest(path, main, queue._sha256(path), output)
    row = planned["jobs"][0]
    recording = Path(row["output_npz"])
    _recording(recording, row, planned)
    row["status"] = "passed"
    row["output_npz_sha256"] = queue._sha256(recording)
    assert queue._recording_valid(row, planned) == (True, None)
    queue._save(output, planned)
    resumed = queue._resume_manifest(output, queue._new_manifest(path, main, queue._sha256(path), output))
    assert resumed["jobs"][0]["status"] == "passed"
    _recording(recording, row, planned, checkpoint_sha="0" * 64)
    row["output_npz_sha256"] = queue._sha256(recording)
    queue._save(output, planned)
    resumed = queue._resume_manifest(output, queue._new_manifest(path, main, queue._sha256(path), output))
    assert resumed["jobs"][0]["status"] == "ready"
    assert "not reusable" in resumed["jobs"][0]["reason"]

    # A PASS assertion without a stored artifact hash also cannot be reused.
    row["output_npz_sha256"] = None
    queue._save(output, planned)
    resumed = queue._resume_manifest(output, queue._new_manifest(path, main, queue._sha256(path), output))
    assert resumed["jobs"][0]["status"] == "ready"


def test_execute_is_sequential_and_resumable_without_simulator_in_test(tmp_path, monkeypatch):
    path, _ = _main_matrix(tmp_path)
    _report(monkeypatch)
    launched = []

    def fake_run(command, log):
        assert "--headless" in command
        launched.append(command)
        artifact = Path(command[command.index("--output") + 1])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"fake recording")
        return {"exit_code": 0, "log": str(log), "wall_time_s": 0.01}

    monkeypatch.setattr(queue, "_run", fake_run)
    monkeypatch.setattr(queue, "_recording_valid", lambda row, manifest: (True, None))
    output = tmp_path / "regimes.json"
    assert queue._main_locked(_args(path, execute=True, max_jobs=2), output) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"passed": 2, "ready": 58}
    assert len(launched) == 2
    assert launched[0][launched[0].index("--output") + 1] != launched[1][launched[1].index("--output") + 1]


def test_successful_process_without_valid_npz_is_failed_and_stops_queue(tmp_path, monkeypatch):
    path, _ = _main_matrix(tmp_path)
    _report(monkeypatch)
    launched = []

    def fake_run(command, log):
        launched.append(command)
        artifact = Path(command[command.index("--output") + 1])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"corrupt NPZ")
        return {"exit_code": 0, "log": str(log), "wall_time_s": 0.01}

    monkeypatch.setattr(queue, "_run", fake_run)
    output = tmp_path / "regimes.json"
    assert queue._main_locked(_args(path, execute=True), output) == 1
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"failed": 1, "ready": 59}
    assert len(launched) == 1
    assert "validation failed" in manifest["jobs"][0]["reason"]
