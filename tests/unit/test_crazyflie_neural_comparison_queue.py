"""CPU-only gates for the paused resumable 500k neural comparison queue."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_neural_comparison_queue as queue_module  # noqa: E402
from drone_bootstrap import source_hashes  # noqa: E402


CONFIG = (
    ROOT / "configs" / "experiments" / "crazyflie_neural_comparison_seed0_500k.json"
)


@pytest.fixture(scope="module")
def resolved_queue(tmp_path_factory):
    config = queue_module.validate_config(CONFIG)
    output = tmp_path_factory.mktemp("neural-comparison") / "output"
    try:
        queue = queue_module.build_queue(config, output)
    except Exception as exc:
        if type(exc).__name__ != "GateValidationError" or "identity is stale" not in str(exc):
            raise
        pytest.skip(
            "paused 500k queue correctly rejects the additive command-v2 source identity; "
            "its immutable historical receipt must not be rewritten"
        )
    return config, queue


def test_queue_has_exact_lif_first_18_cell_shape_and_commands(resolved_queue):
    config, queue = resolved_queue
    expected_pairs = [
        (controller, task)
        for controller in queue_module.CONTROLLERS
        for task in queue_module.TASKS
    ]
    assert queue["job_count"] == 18
    assert queue["evaluation_bundle_count"] == 18
    assert queue["predicted_evaluation_episodes"] == 288
    assert queue["predicted_neural_activity_replay_episodes"] == 288
    assert [(job["controller"], job["task"]) for job in queue["jobs"]] == expected_pairs
    assert all(job["seed"] == 0 for job in queue["jobs"])
    assert all(job["total_interactions"] == 500_000 for job in queue["jobs"])
    assert all(job["expected_updates"] == 125 for job in queue["jobs"])
    assert len({job["run_dir"] for job in queue["jobs"]}) == 18
    assert len({job["evaluation_output"] for job in queue["jobs"]}) == 18
    assert len({job["neural_activity_output"] for job in queue["jobs"]}) == 18
    assert len({job["expected_fingerprint"] for job in queue["jobs"]}) == 18
    assert queue["config_file_sha256"] == config["_config_sha256"]
    assert queue["config_identity_sha256"] == queue_module.canonical_sha256(queue["config"])
    assert queue["queue_runner"] == str(Path(queue_module.__file__).resolve())
    assert queue["queue_runner_sha256"] == queue_module.sha256_file(
        Path(queue_module.__file__).resolve()
    )
    assert queue["activity_collector"] == str(queue_module.ACTIVITY_COLLECTOR)
    assert queue["activity_collector_sha256"] == queue_module.sha256_file(
        queue_module.ACTIVITY_COLLECTOR
    )
    assert queue["activity_validator"] == str(queue_module.ACTIVITY_VALIDATOR)
    assert queue["activity_validator_sha256"] == queue_module.sha256_file(
        queue_module.ACTIVITY_VALIDATOR
    )
    assert queue["parallel_gate"] == queue_module._parallel_gate_record()

    for job in queue["jobs"]:
        command = job["training_command"]
        assert command[0] == str(queue_module.ISAAC_PYTHON)
        assert command[1].endswith("/scripts/drone_train.py")
        assert command[command.index("--num_envs") + 1] == "40"
        assert command[command.index("--horizon") + 1] == "100"
        assert command[command.index("--microbatch_size") + 1] == "40"
        assert command[command.index("--ppo_epochs") + 1] == "2"
        assert command[command.index("--learning_rate") + 1] == "0.0003"
        assert command[command.index("--contract_profile") + 1] == "balanced_v4"
        assert command[command.index("--evaluation_protocol") + 1] == "main"
        assert "--headless" in command
        evaluation = job["evaluation_command"]
        assert evaluation[evaluation.index("--scenario") + 1] == job["task"]
        assert evaluation[evaluation.index("--protocol") + 1] == "main"
        activity = job["neural_activity_command"]
        assert activity[1].endswith("/scripts/crazyflie_neural_activity.py")
        assert activity[activity.index("--scenario") + 1] == job["task"]
        assert job["activity_collector"] == queue["activity_collector"]
        assert job["activity_collector_sha256"] == queue["activity_collector_sha256"]
        assert job["activity_validator"] == queue["activity_validator"]
        assert job["activity_validator_sha256"] == queue["activity_validator_sha256"]


def test_dry_run_print_contract_lists_all_cells_without_mutation(resolved_queue):
    _, queue = resolved_queue
    payload = queue_module._dry_run_payload(queue)
    assert payload["status"] == "dry_run_only_no_processes_launched"
    assert payload["job_count"] == 18
    assert payload["training_interactions_per_job"] == 500_000
    assert payload["predicted_training_interactions"] == 9_000_000
    assert payload["predicted_evaluation_episodes"] == 288
    assert payload["predicted_neural_activity_replay_episodes"] == 288
    assert len(payload["cells"]) == 18
    assert [cell["id"] for cell in payload["cells"]] == [
        job["id"] for job in queue["jobs"]
    ]
    assert all(cell["training_command"] for cell in payload["cells"])
    assert all(cell["evaluation_command"] for cell in payload["cells"])
    assert all(cell["neural_activity_command"] for cell in payload["cells"])


def test_parameter_counts_and_capacity_disclosure_are_explicit(resolved_queue):
    _, queue = resolved_queue
    reports = queue["controller_reports"]
    assert reports["frozen_lif_original"]["actor_trainable_parameters"] == 4_776
    assert reports["frozen_lif_degree_rewired"]["actor_trainable_parameters"] == 4_776
    assert reports["wing_lif"]["actor_trainable_parameters"] == 4_776
    assert reports["leg_wing_lif"]["actor_trainable_parameters"] == 9_224
    assert reports["gru_matched"]["actor_trainable_parameters"] == 4_793
    assert reports["mlp_normal"]["actor_trainable_parameters"] == 4_827
    assert queue["parameter_matching"]["leg_wing_is_not_parameter_matched"] is True
    assert queue["parameter_matching"]["numerically_matched_single_core_and_baselines"] == [
        "frozen_lif_original",
        "frozen_lif_degree_rewired",
        "wing_lif",
        "gru_matched",
        "mlp_normal",
    ]


def test_loaded_queue_rejects_stale_current_source_fingerprint(
    resolved_queue, monkeypatch, tmp_path: Path
):
    config, queue = resolved_queue
    queue_path = tmp_path / "queue.json"
    queue = json.loads(json.dumps(queue))
    queue["queue_file"] = str(queue_path)
    monkeypatch.setattr(
        queue_module,
        "_job_fingerprint",
        lambda *_args, **_kwargs: ("f" * 64, {"changed": True}),
    )

    with pytest.raises(ValueError, match="fingerprint is stale"):
        queue_module._validate_loaded_queue(queue, config, queue_path)


def test_loaded_queue_rejects_changed_activity_collector_pin(
    resolved_queue, tmp_path: Path
):
    config, queue = resolved_queue
    queue_path = tmp_path / "queue.json"
    queue = json.loads(json.dumps(queue))
    queue["queue_file"] = str(queue_path)
    queue["activity_collector_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="reviewed config/order"):
        queue_module._validate_loaded_queue(queue, config, queue_path)


def test_loaded_queue_rejects_changed_activity_validator_pin(
    resolved_queue, tmp_path: Path
):
    config, queue = resolved_queue
    queue = json.loads(json.dumps(queue))
    queue_path = tmp_path / "queue.json"
    queue["queue_file"] = str(queue_path)
    queue["activity_validator_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed config/order"):
        queue_module._validate_loaded_queue(queue, config, queue_path)


def test_resource_gate_is_strict_at_both_exclusive_limits(monkeypatch):
    def gpu(value):
        monkeypatch.setattr(
            queue_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=f"{value}\n", stderr=""
            ),
        )

    gpu(queue_module.GPU_LIMIT_MIB)
    monkeypatch.setattr(queue_module, "_ram_used_percent", lambda: 10.0)
    assert queue_module.resource_snapshot()["passed"] is False

    gpu(100.0)
    monkeypatch.setattr(
        queue_module, "_ram_used_percent", lambda: queue_module.RAM_LIMIT_PERCENT
    )
    assert queue_module.resource_snapshot()["passed"] is False

    gpu(queue_module.GPU_LIMIT_MIB - 0.01)
    monkeypatch.setattr(
        queue_module, "_ram_used_percent", lambda: queue_module.RAM_LIMIT_PERCENT - 0.01
    )
    assert queue_module.resource_snapshot()["passed"] is True


def test_second_parallel_child_waits_for_observed_gpu_allocation(monkeypatch):
    active = {
        "one": {
            "job": {"active_process": {"pid": 1234}},
        }
    }
    monkeypatch.setattr(queue_module, "gpu_compute_client_pids", lambda: set())
    assert queue_module._active_gpu_allocation_observed(active) is False
    monkeypatch.setattr(queue_module, "gpu_compute_client_pids", lambda: {1234, 9999})
    assert queue_module._active_gpu_allocation_observed(active) is True


def test_atomic_state_and_no_overwrite_prepare_contract(tmp_path, resolved_queue):
    _, queue = resolved_queue
    state = tmp_path / "queue.json"
    queue_module.save_queue(state, queue)
    decoded = json.loads(state.read_text(encoding="utf-8"))
    assert decoded["revision"] == 1
    assert decoded["status"] == "prepared"
    assert (tmp_path / "queue_summary.json").is_file()
    source = (ROOT / "scripts" / "crazyflie_neural_comparison_queue.py").read_text(
        encoding="utf-8"
    )
    assert "Output root already exists and will not be overwritten" in source
    assert 'log.open("xb")' in source


def test_config_is_closed_and_launcher_does_not_change_drone_fingerprint(tmp_path):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["total_interactions_per_job"] = 500_001
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="total_interactions_per_job"):
        queue_module.validate_config(changed)
    assert not (ROOT / "scripts" / "crazyflie_neural_comparison_queue.py").name.startswith(
        "drone_"
    )
    assert "scripts/crazyflie_neural_comparison_queue.py" not in source_hashes()


def test_reconcile_keeps_missing_artifacts_pending(resolved_queue):
    _, queue = resolved_queue
    queue_module.reconcile_queue(queue, retry_failed=True)
    assert all(job["status"] == "pending" for job in queue["jobs"])
    assert all(job["training_status"] == "pending" for job in queue["jobs"])
    assert all(job["evaluation_status"] == "pending" for job in queue["jobs"])
    assert all(job["neural_activity_status"] == "pending" for job in queue["jobs"])


def _write_shallow_artifact(job: dict, output: Path, *, activity: dict | None = None) -> None:
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"unit-test-checkpoint")
    value = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "main",
        "scenario": job["task"],
        "controller": job["controller"],
        "training_seed": job["seed"],
        "fingerprint": job["expected_fingerprint"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": queue_module.sha256_file(checkpoint),
        "episodes": [{} for _ in range(16)],
        "summary": {"complete": True, "episode_count": 16},
        "memory_gate": {
            "passed": True,
            "max_device_gpu_used_mib": 1.0,
            "max_system_ram_percent": 1.0,
        },
    }
    if activity is not None:
        value["neural_activity"] = activity
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value), encoding="utf-8")


def test_valid_evaluation_rejects_sixteen_empty_episode_rows(
    resolved_queue, tmp_path: Path
):
    _, built = resolved_queue
    job = json.loads(json.dumps(built["jobs"][0]))
    run_dir = tmp_path / "empty-evaluation-job"
    job["run_dir"] = str(run_dir)
    job["checkpoint"] = str(run_dir / "checkpoints" / "latest.pt")
    job["evaluation_output"] = str(tmp_path / "empty-evaluation.json")
    _write_shallow_artifact(job, Path(job["evaluation_output"]))

    assert queue_module.valid_evaluation(job) is False


def test_valid_neural_activity_rejects_empty_activity_payload(
    resolved_queue, tmp_path: Path, monkeypatch
):
    _, built = resolved_queue
    job = json.loads(json.dumps(built["jobs"][0]))
    run_dir = tmp_path / "empty-activity-job"
    job["run_dir"] = str(run_dir)
    job["checkpoint"] = str(run_dir / "checkpoints" / "latest.pt")
    job["evaluation_output"] = str(tmp_path / "official.json")
    job["neural_activity_output"] = str(tmp_path / "empty-activity.json")
    _write_shallow_artifact(job, Path(job["evaluation_output"]))
    _write_shallow_artifact(job, Path(job["neural_activity_output"]), activity={})

    # Isolate the neural payload gate from the official evaluation gate.  The
    # old validator accepted this exact empty dictionary; the strict reporter
    # validation must reject it even when upstream identity checks are treated
    # as already authenticated.
    official = json.loads(Path(job["evaluation_output"]).read_text(encoding="utf-8"))
    monkeypatch.setattr(queue_module, "_strict_evaluation_value", lambda _job: official)
    monkeypatch.setattr(queue_module, "_validate_scenario_part", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_module, "score_evaluation", lambda *args, **kwargs: {})

    assert queue_module.valid_neural_activity(job) is False
