"""CPU-only queue tests: no Isaac Sim process is launched."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_matrix  # noqa: E402
from evaluation_protocol import schedule_manifest  # noqa: E402


def _write_config(tmp_path: Path, *, seeds=(0, 1)) -> Path:
    config = {
        "task": "FlyG1-GoalReach-FreePosture-v0",
        "interaction_budget": 1024,
        "conditions": list(run_matrix.POLICIES),
        "seeds": list(seeds),
        "training": {"num_envs": 16, "horizon": 32},
        "evaluation": {
            "scenario_tasks": sorted(run_matrix.TASKS), "seed": 101, "episodes": 2,
            "held_out_targets": True, "protocol": "heldout_v1", "reset_regimes": ["standing"],
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _main(monkeypatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", *map(str, arguments)])
    return run_matrix.main()


def test_dry_run_emits_all_jobs_without_starting_processes(tmp_path, monkeypatch):
    config = _write_config(tmp_path, seeds=range(5))
    output = tmp_path / "matrix.json"
    monkeypatch.setattr(run_matrix, "_run", lambda *_: pytest.fail("dry run launched a process"))
    assert _main(monkeypatch, "--config", config, "--dry_run", "--output", output) == 0
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "dry_run"
    assert len(manifest["jobs"]) == 20
    assert len({job["id"] for job in manifest["jobs"]}) == 20
    assert all("--interaction_budget" in job["training_command"] for job in manifest["jobs"])
    for job in manifest["jobs"]:
        command = run_matrix._eval_command(job | {"checkpoint": "dummy.pt"}, manifest["config"],
                                           "FlyG1-GoalReach-FreePosture-v0", tmp_path / "out.json", None)
        assert command[command.index("--protocol") + 1] == "heldout_v1"
    assert (tmp_path / "matrix_summary.json").is_file()
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--config", config, "--dry_run", "--output", output)


def test_matrix_records_condition_learning_rate_and_kl_guard(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, seeds=(0,))
    config = json.loads(config_path.read_text())
    config["training"].update({"target_kl": 0.05, "learning_rate_by_condition": {
        "frozen_lif_original": 1e-6, "gru_trainable": 3e-4,
    }})
    config_path.write_text(json.dumps(config))
    output = tmp_path / "guarded.json"
    assert _main(monkeypatch, "--config", config_path, "--dry_run", "--output", output) == 0
    jobs = {job["condition"]: job for job in json.loads(output.read_text())["jobs"]}
    command = jobs["frozen_lif_original"]["training_command"]
    assert command[command.index("--learning_rate") + 1] == "1e-06"
    assert command[command.index("--target_kl") + 1] == "0.05"
    command = jobs["mlp_engineering_baseline"]["training_command"]
    assert "--learning_rate" not in command
    assert command[command.index("--target_kl") + 1] == "0.05"


def test_missing_gate_and_synthetic_connectome_start_no_jobs(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    fixture = Path(__file__).resolve().parents[1] / "fixtures/synthetic_circuit/manifest.json"
    monkeypatch.setattr(run_matrix, "_run", lambda *_: pytest.fail("blocked queue launched a process"))
    assert _main(monkeypatch, "--config", config, "--execute", "--output", output,
                 "--connectome_manifest", fixture) == 2
    manifest = json.loads(output.read_text())
    assert manifest["counts"] == {"blocked_missing_connectome": 4, "waiting_smoke_gate": 4}
    assert manifest["prerequisites"]["real_connectome"]["status"] == "BLOCKED"
    assert "Synthetic" in manifest["prerequisites"]["real_connectome"]["reason"]


def test_stale_smoke_report_cannot_start_training(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    smoke = tmp_path / "stale_smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "task": "FlyG1-GoalReach-FreePosture-v0",
                                 "num_envs": 16, "steps": 1000, "random_actions": True, "random_action_scale": 0.1,
                                 "simulator_source_fingerprint": "stale"}))
    monkeypatch.setattr(run_matrix, "_run", lambda *_: pytest.fail("stale gate launched training"))
    assert _main(monkeypatch, "--config", config, "--execute", "--output", output,
                 "--smoke_report", smoke) == 2
    manifest = json.loads(output.read_text())
    assert manifest["counts"] == {"blocked_missing_connectome": 4, "waiting_smoke_gate": 4}


def test_queue_only_marks_ready_jobs_and_execute_picks_them_up(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "task": "FlyG1-GoalReach-FreePosture-v0",
                                 "num_envs": 16, "steps": 1000, "random_actions": True, "random_action_scale": 0.1,
                                 "simulator_source_fingerprint": run_matrix.simulator_source_fingerprint()}))
    assert _main(monkeypatch, "--config", config, "--dry_run", "--output", output) == 0
    monkeypatch.setattr(run_matrix, "_run", lambda *_: pytest.fail("queue-only launched a process"))
    assert _main(monkeypatch, "--config", config, "--queue_only", "--resume", "--output", output,
                 "--smoke_report", smoke) == 0
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "queued_with_blockers"
    assert manifest["counts"] == {"blocked_missing_connectome": 4, "ready": 4}
    assert manifest["prerequisites"]["useful_learning_pilot"]["status"] == "UNVERIFIED"
    picked = []

    def fake_job(job, manifest, output, connectome):
        picked.append(job["id"])
        job["status"] = "passed"

    monkeypatch.setattr(run_matrix, "_run_job", fake_job)
    assert _main(monkeypatch, "--config", config, "--execute", "--resume", "--output", output,
                 "--max_jobs", "1") == 0
    assert picked == ["gru_trainable__seed-0"]


def test_execute_resumes_and_evaluates_each_scenario(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "task": "FlyG1-GoalReach-FreePosture-v0",
                                 "num_envs": 16, "steps": 1000, "random_actions": True, "random_action_scale": 0.1,
                                 "simulator_source_fingerprint": run_matrix.simulator_source_fingerprint()}),
                     encoding="utf-8")
    commands = []

    def fake_run(command, log):
        commands.append(command)
        if "train.py" in command[1]:
            policy = command[command.index("--policy") + 1]
            seed = int(command[command.index("--seed") + 1])
            run_dir = tmp_path / f"{policy}-{seed}"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint.pt"
            train_manifest = run_dir / "manifest.json"
            metadata = {"status": "executed_matrix_training", "task": "FlyG1-GoalReach-FreePosture-v0",
                        "policy": policy, "seed": seed, "checkpoint_path": str(checkpoint),
                        "requested_interaction_budget": 1024, "actual_environment_interactions": 1024,
                        "completed_iterations": 2, "num_envs": 16, "connectome_checksum": None,
                        "rewire_seed": None}
            train_manifest.write_text(json.dumps(metadata))
            torch.save({"format": 1, "policy": {"dummy": torch.tensor(1.)},
                        "optimizer": {}, "metadata": metadata}, checkpoint)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Isaac log\n" + json.dumps({"status": "PASS", "checkpoint": str(checkpoint),
                                                         "run_manifest": str(train_manifest)}) + "\n")
        else:
            result_file = Path(command[command.index("--output") + 1])
            result_file.parent.mkdir(parents=True, exist_ok=True)
            task = command[command.index("--task") + 1]
            seed = int(Path(command[command.index("--checkpoint") + 1]).parent.name.split("-")[-1])
            result_file.write_text(json.dumps({"status": "executed",
                                               "checkpoint": command[command.index("--checkpoint") + 1],
                                               "task": task,
                                               "training_seed": seed,
                                               "evaluation_seed": int(command[command.index("--seed") + 1]),
                                               "n_episodes": int(command[command.index("--episodes") + 1]),
                                               "n_independent_training_seeds": 1,
                                               "per_seed": [{"seed": seed, "episodes": 2, "success_rate": 0.0}],
                                               "episodes": [{"seed": seed, "episode_id": i, "success": False,
                                                             "schedule_events": [], "initial_state_sha256": "a" * 64,
                                                             "paired_plan_sha256": "b" * 64} for i in range(2)],
                                               "scenario": {"evaluation_protocol": "heldout_v1", "held_out_targets_verified": True,
                                                            "paired_disturbance_schedule": True,
                                                            "schedule": schedule_manifest(101, 2, task)},
                                               "summary": {"success_rate": 0.0}}))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("evaluation log\n")
        return {"exit_code": 0, "wall_time_s": 0.1, "log": str(log)}

    monkeypatch.setattr(run_matrix, "_run", fake_run)
    assert _main(monkeypatch, "--config", config, "--execute", "--output", output,
                 "--smoke_report", smoke, "--max_jobs", "1") == 0
    manifest = json.loads(output.read_text())
    assert manifest["counts"] == {"blocked_missing_connectome": 4, "passed": 1, "pending": 3}
    passed = next(job for job in manifest["jobs"] if job["status"] == "passed")
    assert set(passed["evaluations"]) == run_matrix.TASKS
    assert all(entry["status"] == "passed" for entry in passed["evaluations"].values())
    assert len(commands) == 4  # one training job plus three scenarios
    assert _main(monkeypatch, "--config", config, "--execute", "--resume", "--output", output,
                 "--max_jobs", "1") == 0
    manifest = json.loads(output.read_text())
    assert manifest["counts"] == {"blocked_missing_connectome": 4, "passed": 2, "pending": 2}
    assert len(commands) == 8


def test_resume_rejects_config_change(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    assert _main(monkeypatch, "--config", config, "--dry_run", "--output", output) == 0
    changed = json.loads(config.read_text())
    changed["interaction_budget"] = 2048
    config.write_text(json.dumps(changed))
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--config", config, "--resume", "--execute", "--output", output)


def test_resume_rejects_training_or_evaluation_source_change(tmp_path, monkeypatch, capsys):
    config = _write_config(tmp_path)
    output = tmp_path / "matrix.json"
    assert _main(monkeypatch, "--config", config, "--dry_run", "--output", output) == 0
    monkeypatch.setattr(run_matrix, "_execution_source_fingerprint", lambda: "changed")
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--config", config, "--resume", "--execute", "--output", output)
    assert "source changed" in capsys.readouterr().err


@pytest.mark.parametrize("mismatch", ["under_budget", "checkpoint_path"])
def test_training_manifest_mismatch_cannot_be_marked_passed(tmp_path, monkeypatch, mismatch):
    config = _write_config(tmp_path, seeds=(0,))
    output = tmp_path / "matrix.json"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "task": "FlyG1-GoalReach-FreePosture-v0",
                                 "num_envs": 16, "steps": 1000, "random_actions": True, "random_action_scale": 0.1,
                                 "simulator_source_fingerprint": run_matrix.simulator_source_fingerprint()}))

    def fake_run(command, log):
        assert "train.py" in command[1]  # evaluation must never start
        run_dir = tmp_path / "underbudget"
        run_dir.mkdir()
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        train_manifest = run_dir / "manifest.json"
        train_manifest.write_text(json.dumps({"task": "FlyG1-GoalReach-FreePosture-v0",
                                              "policy": "gru", "seed": 0,
                                              "checkpoint_path": str(checkpoint if mismatch == "under_budget" else run_dir / "different.pt"),
                                              "requested_interaction_budget": 1024,
                                              "actual_environment_interactions": 512 if mismatch == "under_budget" else 1024}))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"status": "PASS", "checkpoint": str(checkpoint),
                                   "run_manifest": str(train_manifest)}))
        return {"exit_code": 0, "wall_time_s": 0.1, "log": str(log)}

    monkeypatch.setattr(run_matrix, "_run", fake_run)
    assert _main(monkeypatch, "--config", config, "--execute", "--output", output,
                 "--smoke_report", smoke, "--max_jobs", "1") == 1
    manifest = json.loads(output.read_text())
    gru = next(job for job in manifest["jobs"] if job["condition"] == "gru_trainable")
    assert gru["status"] == "failed"
    assert gru["evaluations"] == {}


def test_queue_only_preserves_partial_and_failed_jobs(tmp_path, monkeypatch):
    config = _write_config(tmp_path, seeds=(0,))
    output = tmp_path / "matrix.json"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "task": "FlyG1-GoalReach-FreePosture-v0",
                                 "num_envs": 16, "steps": 1000, "random_actions": True,
                                 "random_action_scale": 0.1,
                                 "simulator_source_fingerprint": run_matrix.simulator_source_fingerprint()}))
    assert _main(monkeypatch, "--config", config, "--dry_run", "--output", output) == 0
    manifest = json.loads(output.read_text())
    statuses = ["training_complete", "failed", "running", "ready"]
    for job, status in zip(manifest["jobs"], statuses):
        job["status"] = status
        job["evaluations"] = {"sentinel": {"status": "passed", "result_file": "sentinel.json"}}
        if status != "ready":
            run_dir = tmp_path / job["id"]
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint.pt"
            metadata = {"status": "executed_matrix_training", "task": job["task"],
                        "policy": job["policy"], "seed": job["seed"],
                        "checkpoint_path": str(checkpoint), "requested_interaction_budget": 1024,
                        "actual_environment_interactions": 1024, "completed_iterations": 2,
                        "num_envs": 16, "connectome_checksum": None,
                        "rewire_seed": job["seed"] if job["policy"] == "frozen_lif_rewired" else None}
            (run_dir / "manifest.json").write_text(json.dumps(metadata))
            torch.save({"format": 1, "policy": {"dummy": torch.tensor(1.)},
                        "optimizer": {}, "metadata": metadata}, checkpoint)
            job["checkpoint"] = str(checkpoint)
            job["run_manifest"] = str(run_dir / "manifest.json")
    output.write_text(json.dumps(manifest))
    monkeypatch.setattr(run_matrix, "_run", lambda *_: pytest.fail("queue-only launched a process"))
    assert _main(monkeypatch, "--config", config, "--queue_only", "--resume", "--output", output,
                 "--smoke_report", smoke) == 0
    updated = json.loads(output.read_text())
    assert [job["status"] for job in updated["jobs"]] == statuses
    assert all("sentinel" in job["evaluations"] for job in updated["jobs"])


def test_evaluation_record_count_required():
    result = {"n_independent_training_seeds": 1, "per_seed": [{"seed": 0, "episodes": 2}],
              "episodes": [{"seed": 0, "episode_id": i, "success": False,
                            "schedule_events": [], "initial_state_sha256": "a" * 64,
                            "paired_plan_sha256": "b" * 64} for i in range(2)]}
    assert run_matrix._valid_episode_records(result, seed=0, episodes=2, heldout=True)
    assert not run_matrix._valid_episode_records(result | {"episodes": []}, seed=0, episodes=2, heldout=True)
    assert not run_matrix._valid_episode_records(result | {"per_seed": []}, seed=0, episodes=2, heldout=True)
    assert not run_matrix._valid_episode_records(result | {"episodes": [{"seed": 0, "episode_id": 0}]},
                                                 seed=0, episodes=2, heldout=True)


def test_child_watchdog_stops_idle_process_but_allows_long_active_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "IDLE_TIMEOUT_S", 0.25)
    monkeypatch.setattr(run_matrix, "WATCHDOG_POLL_S", 0.05)
    idle = run_matrix._run([sys.executable, "-c", "import time; print('started'); time.sleep(2)"],
                           tmp_path / "idle.log")
    assert idle["stalled"] is True and idle["exit_code"] == 124
    assert "Matrix watchdog" in (tmp_path / "idle.log").read_text()
    active = run_matrix._run([sys.executable, "-c",
                              "import time; [(print(i, flush=True), time.sleep(0.1)) for i in range(8)]"],
                             tmp_path / "active.log")
    assert active["stalled"] is False and active["exit_code"] == 0
    assert active["wall_time_s"] > 0.7


def test_resume_revalidates_passed_checkpoint_and_each_evaluation(tmp_path):
    config_path = _write_config(tmp_path, seeds=(0,))
    config = json.loads(config_path.read_text())
    config["conditions"] = ["gru_trainable"]
    config_path.write_text(json.dumps(config))
    manifest = run_matrix._new_manifest(config, config_path, tmp_path / "matrix.json", None)
    job = manifest["jobs"][0]
    checkpoint = tmp_path / "checkpoint.pt"
    run_manifest = tmp_path / "manifest.json"
    metadata = {"status": "executed_matrix_training", "task": job["task"],
                "policy": job["policy"], "seed": job["seed"], "checkpoint_path": str(checkpoint),
                "requested_interaction_budget": 1024, "actual_environment_interactions": 1024,
                "completed_iterations": 2, "num_envs": 16, "connectome_checksum": None,
                "rewire_seed": None}
    run_manifest.write_text(json.dumps(metadata))
    torch.save({"format": 1, "policy": {"dummy": torch.tensor(1.)},
                "optimizer": {}, "metadata": metadata}, checkpoint)
    job.update({"checkpoint": str(checkpoint), "run_manifest": str(run_manifest), "status": "passed"})
    for scenario in config["evaluation"]["scenario_tasks"]:
        result_file = tmp_path / f"{scenario}.json"
        result = {"status": "executed", "checkpoint": str(checkpoint), "task": scenario,
                  "training_seed": 0, "evaluation_seed": 101, "n_episodes": 2,
                  "n_independent_training_seeds": 1,
                  "per_seed": [{"seed": 0, "episodes": 2, "success_rate": 0.0}],
                  "episodes": [{"seed": 0, "episode_id": i, "success": False,
                                "schedule_events": [], "initial_state_sha256": "a" * 64,
                                "paired_plan_sha256": "b" * 64} for i in range(2)],
                  "scenario": {"evaluation_protocol": "heldout_v1", "held_out_targets_verified": True,
                               "paired_disturbance_schedule": True,
                               "schedule": schedule_manifest(101, 2, scenario)}}
        result_file.write_text(json.dumps(result))
        job["evaluations"][scenario] = {"status": "passed", "result_file": str(result_file)}
    run_matrix._reconcile_resume_artifacts(manifest)
    assert job["status"] == "passed"
    damaged = config["evaluation"]["scenario_tasks"][0]
    Path(job["evaluations"][damaged]["result_file"]).write_text("{}")
    run_matrix._reconcile_resume_artifacts(manifest)
    assert job["status"] == "training_complete"
    assert job["evaluations"][damaged]["status"] == "needs_rerun"
    assert job["checkpoint"] == str(checkpoint)
    checkpoint.write_bytes(b"not a checkpoint")
    run_matrix._reconcile_resume_artifacts(manifest)
    assert job["status"] == "ready"
    assert "checkpoint" not in job and job["evaluations"] == {}
