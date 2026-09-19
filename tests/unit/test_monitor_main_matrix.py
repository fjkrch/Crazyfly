"""CPU-only monitor tests; no matrix command is launched."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import monitor_main_matrix as monitor  # noqa: E402


def _fixture(tmp_path: Path):
    queue = tmp_path / "matrix.json"
    command = ["/python", str(tmp_path / "train.py"), "--seed", "0"]
    manifest = {
        "status": "running", "counts": {"running": 1},
        "config": {"task": "Reach", "training": {"num_envs": 16, "horizon": 32},
                   "evaluation": {"scenario_tasks": ["Reach"]}},
        "jobs": [{"id": "lif__seed-0", "status": "running", "task": "Reach",
                  "policy": "frozen_lif", "seed": 0, "interaction_budget": 1024,
                  "training_command": command, "evaluations": {}}],
    }
    queue.write_text(json.dumps(manifest))
    log = tmp_path / "matrix" / "logs" / "lif__seed-0__train.log"
    log.parent.mkdir(parents=True)
    proc = tmp_path / "proc"
    child = proc / "123"
    child.mkdir(parents=True)
    (child / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in command) + b"\0")
    (child / "cwd").symlink_to(tmp_path, target_is_directory=True)
    return queue, log, proc


def test_tail_ignores_rollout_progress_and_partial_json(tmp_path):
    queue, log, proc = _fixture(tmp_path)
    saved_dir = tmp_path / "matrix" / "training" / "20260913T000000Z" / "Reach" / "frozen_lif" / "seed-0"
    saved_dir.mkdir(parents=True)
    (saved_dir / "checkpoint.pt").write_bytes(b"unverified")
    log.write_text('{"iteration": 0, "loss": 0.1}\n'
                   '{"rollout_progress": "collect", "iteration": 1}\n'
                   '{"iteration": 1, "loss":')
    report = monitor.snapshot(queue, proc_root=proc, now=log.stat().st_mtime + 1)
    job = report["active_jobs"][0]
    assert job["completed_updates"] == 1
    assert job["expected_updates"] == 2
    assert job["completed_interactions_observed"] == 512
    assert job["training_pids"] == [123]
    assert job["health"] == "active_process"
    assert job["checkpoint"]["observed_paths"] == [str(saved_dir / "checkpoint.pt")]
    assert job["checkpoint"]["runner_verified"] is False
    assert job["evaluations"]["Reach"]["file_exists"] is False


def test_stale_or_missing_process_is_attention_not_terminal(tmp_path):
    queue, log, proc = _fixture(tmp_path)
    log.write_text('{"iteration": 1, "loss": NaN}\n')
    (proc / "123" / "cmdline").write_bytes(b"other\0")
    report = monitor.snapshot(queue, proc_root=proc, now=log.stat().st_mtime + 301)
    job = report["active_jobs"][0]
    assert job["health"] == "needs_inspection"
    assert job["health_is_terminal"] is False
    assert job["active_stage_log_stale"] is True
    assert job["nonfinite_metric_keys_last_update"] == ["loss"]
    assert report["queue_status"] == "running"


def test_evaluation_uses_its_own_log_after_training_finishes(tmp_path):
    queue, train_log, proc = _fixture(tmp_path)
    train_log.write_text('{"iteration": 1, "loss": 0.1}\n')
    manifest = json.loads(queue.read_text())
    job = manifest["jobs"][0]
    job["status"] = "training_complete"
    eval_command = ["/python", str(tmp_path / "evaluate.py"), "--task", "Reach"]
    result = tmp_path / "matrix" / "evaluations" / job["id"] / "Reach.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"status": "executed"}')
    job["evaluations"] = {"Reach": {"status": "running", "command": eval_command,
                                    "result_file": str(result)}}
    queue.write_text(json.dumps(manifest))
    (proc / "123" / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in eval_command) + b"\0")
    eval_log = train_log.with_name(f"{job['id']}__Reach__evaluate.log")
    eval_log.write_text("evaluating\n")
    now = train_log.stat().st_mtime + 400
    os.utime(eval_log, (now - 1, now - 1))
    observed = monitor.snapshot(queue, proc_root=proc, now=now)["active_jobs"][0]
    assert observed["health"] == "active_process"
    assert observed["evaluation_pids"] == {"Reach": [123]}
    assert observed["evaluations"]["Reach"]["observed_file_status"] == "executed"
    assert observed["evaluations"]["Reach"]["runner_verified"] is False


def test_checkpoint_file_is_observed_before_runner_verifies_it(tmp_path):
    queue, log, proc = _fixture(tmp_path)
    log.write_text('{"iteration": 1, "loss": 0.1}\n')
    checkpoint = tmp_path / "matrix" / "training" / "20260913T000000Z" / "Reach" / "frozen_lif" / "seed-0" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unfinished checkpoint")
    report = monitor.snapshot(queue, proc_root=proc, now=log.stat().st_mtime + 1)
    observed = report["active_jobs"][0]["checkpoint"]
    assert observed["observed_paths"] == [str(checkpoint)]
    assert observed["queue_path_exists"] is False
    assert observed["runner_verified"] is False
