"""CPU-only gates for the dedicated CommandFollow comparison queue."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_parallel_gate as parallel_gate  # noqa: E402
import crazyflie_command_queue as queue  # noqa: E402


CONFIG = ROOT / "configs" / "experiments" / "crazyflie_command_seed0_500k.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue.validate_config(CONFIG)
    output = tmp_path_factory.mktemp("command-queue") / "output"
    return config, queue.build_queue(config, output)


def test_exact_six_job_lif_first_shape_and_commands(built):
    config, value = built
    assert value["kind"] == "crazyflie_command_comparison_queue_v1"
    assert value["status"] == "dry_run"
    assert value["dry_run"] is True
    assert value["job_count"] == 6
    assert value["predicted_training_interactions"] == 3_000_000
    assert value["predicted_evaluation_episodes"] == 96
    assert [job["controller"] for job in value["jobs"]] == list(queue.CONTROLLERS)
    assert [job["architecture_class"] for job in value["jobs"]] == [
        "lif", "lif", "lif", "lif", "baseline", "baseline"
    ]
    assert all(job["task"] == queue.TASK for job in value["jobs"])
    assert all(job["seed"] == 0 for job in value["jobs"])
    assert all(job["total_interactions"] == 500_000 for job in value["jobs"])
    assert all(job["expected_updates"] == 125 for job in value["jobs"])
    assert len({job["run_dir"] for job in value["jobs"]}) == 6
    assert len({job["evaluation_output"] for job in value["jobs"]}) == 6
    assert value["config_file_sha256"] == config["_config_sha256"]

    forbidden = ("WaypointReach", "WaypointSwitch", "GustRecovery", "Mixed")
    for job in value["jobs"]:
        training = job["training_command"]
        assert training[training.index("--task") + 1] == queue.TASK
        assert training[training.index("--contract_profile") + 1] == "command_v1"
        assert training[training.index("--evaluation_protocol") + 1] == "command_v1"
        assert training[training.index("--num_envs") + 1] == "40"
        assert training[training.index("--total_interactions") + 1] == "500000"
        assert training[training.index("--horizon") + 1] == "100"
        assert training[training.index("--pause_file") + 1] == job["pause_file"]
        assert "--headless" in training
        evaluation = job["evaluation_command"]
        assert evaluation[evaluation.index("--protocol") + 1] == "command_v1"
        assert evaluation[evaluation.index("--policy") + 1] == job["controller"]
        assert "--episodes" not in evaluation and "--steps" not in evaluation
        assert not any(term in " ".join(training + evaluation) for term in forbidden)


def test_parameter_count_disclosure_is_complete(built):
    _, value = built
    reports = value["controller_reports"]
    assert {name: report["actor_trainable_parameters"] for name, report in reports.items()} == {
        "frozen_lif_original": 4_776,
        "frozen_lif_degree_rewired": 4_776,
        "wing_lif": 4_776,
        "leg_wing_lif": 9_224,
        "gru_matched": 4_793,
        "mlp_normal": 4_827,
    }
    assert value["parameter_matching"]["leg_wing_is_not_parameter_matched"] is True


def test_closed_config_rejects_any_budget_change(tmp_path):
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["total_interactions_per_job"] = 499_999
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="total_interactions_per_job"):
        queue.validate_config(path)


def test_dry_run_payload_lists_every_command_without_launching(built):
    _, value = built
    payload = queue._dry_run_payload(value)
    assert payload["status"] == "verified_dry_run_no_processes_launched"
    assert payload["job_count"] == 6
    assert payload["training_interactions_per_job"] == 500_000
    assert payload["predicted_training_interactions"] == 3_000_000
    assert payload["predicted_evaluation_episodes"] == 96
    assert len(payload["cells"]) == 6
    assert [cell["controller"] for cell in payload["cells"]] == list(queue.CONTROLLERS)


def test_dry_run_cli_writes_atomic_queue_and_never_popen(
    built, tmp_path: Path, monkeypatch, capsys
):
    config, _ = built
    output = tmp_path / "new-output"
    resolved = dict(config)
    resolved["_output_root"] = str(output)
    monkeypatch.setattr(queue, "validate_config", lambda _path: dict(resolved))
    monkeypatch.setattr(
        queue,
        "_start_child",
        lambda *_args, **_kwargs: pytest.fail("dry-run launched a trainer/evaluator"),
    )
    assert queue.main(["--config", str(CONFIG), "--dry_run"]) == 0
    persisted = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "dry_run"
    assert persisted["revision"] == 1
    assert (output / "queue_summary.json").is_file()
    assert json.loads(capsys.readouterr().out)["job_count"] == 6
    with pytest.raises(SystemExit):
        queue.main(["--config", str(CONFIG), "--dry_run"])


def test_resource_limits_are_strict_and_missing_telemetry_fails(monkeypatch):
    def nvidia(value: str, code: int = 0):
        monkeypatch.setattr(
            queue.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=code, stdout=value, stderr="failed" if code else ""
            ),
        )

    monkeypatch.setattr(queue, "_swap_out_pages", lambda: 10)
    monkeypatch.setattr(queue, "_ram_used_percent", lambda: 10.0)
    nvidia("6963.2\n")
    assert queue.resource_snapshot()["passed"] is False
    nvidia("6963.19\n")
    monkeypatch.setattr(queue, "_ram_used_percent", lambda: 90.0)
    assert queue.resource_snapshot()["passed"] is False
    monkeypatch.setattr(queue, "_ram_used_percent", lambda: 89.99)
    assert queue.resource_snapshot()["passed"] is True
    nvidia("", code=1)
    with pytest.raises(RuntimeError, match="telemetry failed"):
        queue.resource_snapshot()


def _receipt(config: dict, path: Path) -> dict:
    artifacts = []
    for index in range(10):
        artifact = path.parent / f"artifact-{index}.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact-{index}".encode())
        artifacts.append({
            "path": str(artifact.resolve()),
            "size_bytes": artifact.stat().st_size,
            "sha256": queue.sha256_file(artifact),
        })
    launches = [
        {
            "controller": controller,
            "task": queue.TASK,
            "num_envs": 40,
            "pause_exit_code": 3,
            "resume_exit_code": 0,
            "paused_updates": 1,
            "paused_interactions": 4_000,
            "completed_updates": 2,
            "completed_interactions": 8_000,
        }
        for controller in ("frozen_lif_original", "frozen_lif_degree_rewired")
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "contract": queue.parallel_gate_contract(config),
        "checks": {
            "exactly_two_processes": True,
            "isolated_run_and_checkpoint_directories": True,
            "clean_checkpoints_passed": True,
            "simultaneous_gpu_compute_overlap": True,
            "no_missing_or_nonfinite_telemetry": True,
            "no_sustained_paging": True,
            "sequential_resume_completed": True,
        },
        "telemetry_summary": {
            "max_device_gpu_used_mib": 6_000.0,
            "max_system_ram_percent": 75.0,
            "simultaneous_gpu_compute_overlap_samples": 2,
            "swap_out_growth_pages": 0,
        },
        "evidence": {
            "controllers": ["frozen_lif_original", "frozen_lif_degree_rewired"],
            "parallel_launches": launches,
            "sequential_resume_order": [
                "frozen_lif_original", "frozen_lif_degree_rewired"
            ],
            "telemetry_sample_count": 3,
            "artifacts": artifacts,
        },
        "completed_utc": "2026-09-18T00:00:00+00:00",
    }
    return {"payload": payload, "payload_sha256": queue.canonical_sha256(payload)}


def test_parallel_two_requires_current_authenticated_command_receipt(
    built, tmp_path: Path
):
    config, value = built
    missing = tmp_path / "missing" / "parallel_gate_receipt.json"
    with pytest.raises(ValueError, match="authenticated command-specific"):
        queue.execute_queue(
            json.loads(json.dumps(value)), tmp_path / "queue.json",
            max_parallel=2, resume=False, config=config,
            parallel_gate_path=missing,
        )

    receipt_path = tmp_path / "gate" / "parallel_gate_receipt.json"
    receipt = _receipt(config, receipt_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert queue.validate_parallel_gate_receipt(receipt_path, config)["status"] == "PASS"
    receipt["payload"]["telemetry_summary"]["max_device_gpu_used_mib"] = 6963.2
    receipt["payload_sha256"] = queue.canonical_sha256(receipt["payload"])
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="strict limits"):
        queue.validate_parallel_gate_receipt(receipt_path, config)


def test_loaded_queue_rejects_stale_fingerprint(built, tmp_path: Path, monkeypatch):
    config, value = built
    copied = json.loads(json.dumps(value))
    queue_path = tmp_path / "queue.json"
    copied["queue_file"] = str(queue_path)
    gate_path = Path(copied["parallel_gate"]["path"])
    monkeypatch.setattr(queue, "_job_fingerprint", lambda *_args: ("f" * 64, {}))
    with pytest.raises(ValueError, match="fingerprint is stale"):
        queue._validate_loaded_queue(copied, config, queue_path, gate_path)


def test_existing_checkpoint_is_resumed_and_attempt_log_is_exclusive(
    built, tmp_path: Path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][0]))
    job["run_dir"] = str(tmp_path / "run")
    job["checkpoint"] = str(tmp_path / "run" / "checkpoints" / "latest.pt")
    Path(job["checkpoint"]).parent.mkdir(parents=True)
    Path(job["checkpoint"]).write_bytes(b"checkpoint")
    job["attempts"] = []

    class Process:
        pid = 12345

    captured = {}

    def popen(command, **kwargs):
        captured["command"] = command
        return Process()

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    handle = queue._start_child(job, "training", tmp_path)
    assert captured["command"][-1] == "--resume"
    assert Path(handle["record"]["log"]).is_file()
    handle["stream"].close()
    with pytest.raises(FileExistsError):
        # Reset attempt numbering deliberately to prove logs cannot overwrite.
        job["attempts"] = []
        queue._start_child(job, "training", tmp_path)


def test_parallel_gate_specs_are_bounded_exact_and_sequentially_resumable(
    built, tmp_path: Path
):
    config, _ = built
    specs = parallel_gate.gate_specs(tmp_path / "gate", config)
    assert [spec["controller"] for spec in specs] == [
        "frozen_lif_original", "frozen_lif_degree_rewired"
    ]
    assert len({spec["run_dir"] for spec in specs}) == 2
    for spec in specs:
        pause = spec["pause_command"]
        resume = spec["resume_command"]
        assert pause[pause.index("--task") + 1] == queue.TASK
        assert pause[pause.index("--num_envs") + 1] == "40"
        assert pause[pause.index("--total_interactions") + 1] == "8000"
        assert pause[pause.index("--pause_after_updates") + 1] == "1"
        assert "--resume" not in pause
        assert "--resume" in resume
        assert "--pause_after_updates" not in resume


def test_queue_source_has_no_legacy_run_or_queue_mutation_paths():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "crazyflie_neural_comparison_seed0_500k" not in source
    assert "drone_run_matrix.py" not in source
    assert "execute_drone_matrix.sh" not in source
    assert 'log.open("xb")' in source
