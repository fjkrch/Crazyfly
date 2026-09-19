"""CPU-only tests for the Crazyflie queue and immutable evaluation plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_run_matrix as matrix  # noqa: E402
import drone_train as training_script  # noqa: E402
from drone_bootstrap import (  # noqa: E402
    UPSTREAM_DIRECT_RL_ENV,
    UPSTREAM_ISAACLAB_MATH,
    execution_source_paths,
    rollout_rng_contract,
)
from drone_evaluation_protocol import SCENARIOS, generate_manifest, load_protocol, validate_manifest  # noqa: E402
from g1_fly_control.crazyflie.controllers import build_controller  # noqa: E402
from g1_fly_control.crazyflie.stabilization import stabilization_contract_payload  # noqa: E402
from g1_fly_control.tasks.crazyflie.logic import AUDITED_CRAZYFLIE_MASS_KG  # noqa: E402
from g1_fly_control.tasks.crazyflie.metrics import (  # noqa: E402
    EpisodeSummary,
    GustRecoveryOutcome,
    summarize_episodes,
)


def _cheap_fingerprint(_config, controller, seed):
    payload = {"controller": controller, "seed": seed}
    return matrix.canonical_sha256(payload), payload


@pytest.mark.parametrize(
    ("config_name", "label", "expected_seeds"),
    (
        ("crazyflie_balanced_v3_main.json", "main", [0, 1, 2, 3, 4]),
        ("crazyflie_balanced_v3_integration.json", "integration", [0]),
    ),
)
def test_balanced_v3_configs_validate_as_the_mixed_task(
    config_name, label, expected_seeds
):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / config_name)

    assert config["label"] == label
    assert config["task"] == matrix.BALANCED_TASK == "FlyCrazyflie-Mixed-v0"
    assert config["_contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
    assert config["contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
    assert config["seeds"] == expected_seeds
    assert config["balanced_task_contract"] == matrix.balanced_task_contract_payload()
    assert config["memory_acceptance"] == matrix.memory_acceptance_contract_payload()
    assert config["memory_acceptance"] == training_script.memory_acceptance_contract_payload()
    assert config["memory_acceptance"]["policy_version"] == (
        "crazyflie_memory_acceptance_v2"
    )
    assert config["memory_acceptance"]["system_ram_percent_exclusive"] == 90.0
    assert config["memory_acceptance"]["rss_growth_disposition"] == "warning_only"
    assert "survival_first_contract" not in config


@pytest.mark.parametrize(
    ("config_name", "label", "expected_seeds"),
    (
        ("crazyflie_task_separated_v1_balanced_v3_schema_dryrun_integration.json", "integration", [0]),
        ("crazyflie_task_separated_v1_balanced_v3_schema_dryrun_main.json", "main", [0, 1, 2, 3, 4]),
    ),
)
def test_task_separated_v1_umbrella_and_scalar_cells_validate(
    config_name, label, expected_seeds
):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / config_name)

    assert config["label"] == label
    assert config["matrix_layout"] == matrix.TASK_SEPARATED_LAYOUT
    assert config["execution_readiness"] == matrix.TASK_SEPARATED_SCHEMA_DRY_RUN
    assert config["tasks"] == list(SCENARIOS)
    assert config["controllers"] == list(matrix.CONTROLLERS)
    assert config["seeds"] == expected_seeds
    assert "task" not in config
    assert config["evaluation"]["matched_task_only"] is True
    assert set(config["_task_configs"]) == set(SCENARIOS)
    for task, cell in config["_task_configs"].items():
        assert cell["task"] == task
        assert cell["matrix_layout"] == matrix.TASK_SEPARATED_CELL_LAYOUT
        assert cell["_contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
        assert cell["balanced_task_contract"] == matrix.balanced_task_contract_payload()
        assert cell["evaluation"].get("matched_task_only") is None


@pytest.mark.parametrize(
    ("config_name", "expected_jobs", "expected_episodes"),
    (
        ("crazyflie_task_separated_v1_balanced_v3_schema_dryrun_integration.json", 12, 24),
        ("crazyflie_task_separated_v1_balanced_v3_schema_dryrun_main.json", 60, 960),
    ),
)
def test_task_separated_queue_is_task_first_and_matched_task_only(
    tmp_path, monkeypatch, config_name, expected_jobs, expected_episodes
):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / config_name)
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / f"{config['label']}.json")

    expected_cells = [
        (task, controller, seed)
        for task in SCENARIOS
        for controller in matrix.CONTROLLERS
        for seed in config["seeds"]
    ]
    assert [
        (job["task"], job["controller"], job["seed"]) for job in queue["jobs"]
    ] == expected_cells
    assert queue["job_count"] == expected_jobs
    assert queue["evaluation_bundle_count"] == expected_jobs
    assert queue["predicted_evaluation_episodes"] == expected_episodes
    assert queue["resource_limits"]["gpu_compute_utilization_target_percent"] == {
        "minimum_inclusive": 50.0,
        "maximum_inclusive": 90.0,
        "above_maximum_disposition": "scale_down_and_retest_before_long_run",
    }
    assert len({job["id"] for job in queue["jobs"]}) == expected_jobs
    assert len({job["run_dir"] for job in queue["jobs"]}) == expected_jobs
    for job in queue["jobs"]:
        assert job["task"] in job["id"]
        assert len(job["evaluations"]) == 1
        assert job["evaluations"][0]["scenario"] == job["task"]
        train_command = job["training_command"]
        assert train_command[train_command.index("--task") + 1] == job["task"]
        cell_config_path = Path(
            train_command[train_command.index("--matrix_config") + 1]
        )
        cell_config = matrix.validate_config(cell_config_path)
        assert cell_config["task"] == job["task"]
        assert cell_config["_matrix_layout"] == matrix.TASK_SEPARATED_CELL_LAYOUT


def test_task_separated_switch_fingerprint_uses_balanced_curriculum(monkeypatch):
    config = matrix.validate_config(
        ROOT / "configs" / "experiments" / "crazyflie_task_separated_v1_balanced_v3_schema_dryrun_integration.json"
    )
    switch = config["_task_configs"]["FlyCrazyflie-WaypointSwitch-v0"]
    captured = {}

    def fake_fingerprint(**kwargs):
        captured.update(kwargs)
        return "b" * 64, kwargs

    monkeypatch.setattr(matrix, "reproduction_fingerprint", fake_fingerprint)
    fingerprint, _ = matrix.job_fingerprint(switch, matrix.CONTROLLERS[0], 0)

    assert fingerprint == "b" * 64
    resolved = captured["resolved_config"]
    assert resolved["task"] == "FlyCrazyflie-WaypointSwitch-v0"
    assert resolved["contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
    assert resolved["switch_target_curriculum"] == (
        matrix.balanced_switch_target_curriculum_payload()
    )
    assert "mixed_scenario_contract" not in resolved


def test_task_separated_balanced_v3_schema_dry_run_rejects_execute(
    tmp_path, monkeypatch
):
    output = tmp_path / "must-not-be-created.json"
    monkeypatch.setattr(sys, "argv", [
        "drone_run_matrix.py",
        "--config",
        str(
            ROOT
            / "configs"
            / "experiments"
            / "crazyflie_task_separated_v1_balanced_v3_schema_dryrun_integration.json"
        ),
        "--execute",
        "--output",
        str(output),
    ])

    with pytest.raises(SystemExit):
        matrix.main()
    assert not output.exists()


def test_task_separated_scalar_cell_matches_trainer_fingerprint():
    cell_path = (
        ROOT
        / "configs"
        / "experiments"
        / "crazyflie_task_separated_v1_balanced_v3_schema_dryrun_integration_reach.json"
    )
    cell = matrix.validate_config(cell_path)
    fingerprint, _ = matrix.job_fingerprint(cell, "mlp_normal", 0)
    args = SimpleNamespace(
        task=cell["task"],
        contract_profile=cell["_contract_profile"],
        policy="mlp_normal",
        seed=0,
        num_envs=cell["training"]["num_envs"],
        total_interactions=cell["total_interactions"],
        horizon=cell["training"]["horizon"],
        microbatch_size=cell["training"]["microbatch_size"],
        ppo_epochs=cell["training"]["ppo_epochs"],
        learning_rate=cell["training"]["learning_rate"],
        gamma=cell["training"]["gamma"],
        gae_lambda=cell["training"]["gae_lambda"],
        clip_ratio=cell["training"]["clip_ratio"],
        value_coefficient=cell["training"]["value_coefficient"],
        entropy_coefficient=cell["training"]["entropy_coefficient"],
        max_grad_norm=cell["training"]["max_grad_norm"],
        target_kl=cell["training"]["target_kl"],
        checkpoint_every_updates=cell["training"]["checkpoint_every_updates"],
        connectome_manifest=Path(cell["_connectome_path"]),
        rewire_seed=cell["rewire_seed"],
        rewire_manifest=Path(cell["_rewire_manifest_path"]),
        matrix_config=cell_path,
        expected_fingerprint=fingerprint,
        warm_start_checkpoint=None,
    )

    resolved, trainer_fingerprint, _, protocol = (
        training_script._resolved_training_config(args)
    )
    assert trainer_fingerprint == fingerprint
    assert resolved == matrix.resolved_job_config(cell, "mlp_normal", 0)
    assert protocol == cell["_evaluation_manifest"]


def test_balanced_v4_seed0_comparison_is_exact_controller_first_12_job_queue(
    tmp_path, monkeypatch
):
    config_path = (
        ROOT
        / "configs"
        / "experiments"
        / "crazyflie_task_separated_v1_balanced_v4_seed0_comparison.json"
    )
    config = matrix.validate_config(config_path)
    assert config["label"] == matrix.COMPARISON_LABEL == "comparison"
    assert config["_matrix_layout"] == matrix.TASK_SEPARATED_COMPARISON_LAYOUT
    assert config["execution_readiness"] == (
        matrix.TASK_SEPARATED_COMPARISON_READINESS
    )
    assert config["contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V4
    assert config["controllers"] == list(matrix.CONTROLLERS)
    assert config["tasks"] == list(SCENARIOS)
    assert config["seeds"] == [0]
    assert config["total_interactions"] == 1_000_000
    assert config["training"] == matrix._comparison_training_contract()
    assert config["evaluation"]["protocol"] == "main"
    assert config["evaluation"]["episodes_per_scenario"] == 16
    assert config["evaluation"]["matched_task_only"] is True
    assert config["learning_rate_selection"] == {
        "receipt": "runs/crazyflie-balanced-v4-lif-reach-lr-screen-selection.json",
        "receipt_sha256": matrix.COMPARISON_SELECTION_RECEIPT_SHA256,
        "selection_id": matrix.COMPARISON_SELECTION_ID,
        "selected_candidate_id": "lr-3e-4",
        "selected_learning_rate": 3.0e-4,
        "application": (
            "common_learning_rate_all_four_controllers_all_three_tasks_"
            "seed0_comparison_only"
        ),
    }
    for task, cell in config["_task_configs"].items():
        assert cell["task"] == task
        assert cell["_contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V4
        assert cell["balanced_v4_task_contract"] == (
            matrix.balanced_v4_task_contract_payload()
        )
        assert "balanced_task_contract" not in cell

    # Keep this queue-shape test independent of the workstation's mutable
    # paired-smoke receipt.  Authenticated concurrency-2 behavior has its own
    # receipt tests below; this case deliberately exercises the recorded
    # fail-closed fallback path.
    fallback_decision = {
        "status": "fallback_sequential",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 1,
        "fallback_applied": True,
        "fallback_reason": "test fixture: paired-smoke receipt unavailable",
        "receipt": str(matrix.COMPARISON_CONCURRENCY_RECEIPT.resolve()),
        "receipt_sha256": None,
        "receipt_id": None,
        "source_set_sha256": None,
        "reports": [],
        "aggregate_overlap_memory_gate": None,
    }
    monkeypatch.setattr(
        matrix,
        "comparison_concurrency_decision",
        lambda _config: dict(fallback_decision),
    )
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / "comparison.json")
    expected_cells = [
        (controller, task, 0)
        for controller in matrix.CONTROLLERS
        for task in SCENARIOS
    ]
    assert [
        (job["controller"], job["task"], job["seed"]) for job in queue["jobs"]
    ] == expected_cells
    assert queue["job_count"] == 12
    assert queue["evaluation_bundle_count"] == 12
    assert queue["predicted_evaluation_episodes"] == 192
    assert "sequential_process_limit" not in queue
    assert queue["requested_max_concurrent_isaac_processes"] == 2
    assert queue["max_concurrent_isaac_processes"] == 1
    assert queue["concurrency_decision"]["status"] == "fallback_sequential"
    assert queue["concurrency_decision"]["fallback_applied"] is True
    assert queue["resource_limits"]["max_concurrent_isaac_processes"] == 1
    assert queue["resource_limits"]["device_gpu_used_mib_exclusive"] == 6963.2
    assert queue["resource_limits"]["system_ram_percent_exclusive"] == 90.0
    assert queue["resource_limits"]["gpu_compute_utilization_policy"] == {
        "maximum_inclusive": 100.0,
        "disposition": "telemetry_only_not_an_acceptance_gate",
    }
    for job in queue["jobs"]:
        assert job["total_interactions"] == 1_000_000
        assert job["training_resources"] == {
            "num_envs": 4,
            "horizon": 100,
            "ppo_epochs": 2,
            "microbatch_size": 4,
            "num_workers": 0,
            "precision": "float32",
        }
        assert len(job["evaluations"]) == 1
        assert job["evaluations"][0]["scenario"] == job["task"]
        assert job["evaluations"][0]["protocol"] == "main"
        command = job["training_command"]
        assert command[command.index("--contract_profile") + 1] == "balanced_v4"
        assert command[command.index("--num_envs") + 1] == "4"
        assert command[command.index("--microbatch_size") + 1] == "4"
        assert command[command.index("--learning_rate") + 1] == "0.0003"
        assert command[command.index("--total_interactions") + 1] == "1000000"
        assert job["total_interactions"] // (
            job["training_resources"]["num_envs"]
            * job["training_resources"]["horizon"]
        ) == 2500
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--warm_start_checkpoint" not in command


@pytest.mark.parametrize(
    "cell_name",
    (
        "crazyflie_task_separated_v1_balanced_v4_seed0_comparison_reach.json",
        "crazyflie_task_separated_v1_balanced_v4_seed0_comparison_switch.json",
        "crazyflie_task_separated_v1_balanced_v4_seed0_comparison_gust.json",
    ),
)
def test_balanced_v4_comparison_cells_match_trainer_fingerprint(cell_name):
    cell_path = ROOT / "configs" / "experiments" / cell_name
    cell = matrix.validate_config(cell_path)
    controller = "frozen_lif_original"
    fingerprint, _ = matrix.job_fingerprint(cell, controller, 0)
    args = SimpleNamespace(
        task=cell["task"],
        contract_profile=cell["_contract_profile"],
        policy=controller,
        seed=0,
        num_envs=4,
        total_interactions=1_000_000,
        horizon=100,
        microbatch_size=4,
        ppo_epochs=2,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        max_grad_norm=1.0,
        target_kl=0.05,
        checkpoint_every_updates=100,
        connectome_manifest=Path(cell["_connectome_path"]),
        rewire_seed=cell["rewire_seed"],
        rewire_manifest=Path(cell["_rewire_manifest_path"]),
        matrix_config=cell_path,
        expected_fingerprint=fingerprint,
        warm_start_checkpoint=None,
    )
    resolved, trainer_fingerprint, _, protocol = (
        training_script._resolved_training_config(args)
    )
    assert trainer_fingerprint == fingerprint
    assert resolved == matrix.resolved_job_config(cell, controller, 0)
    assert resolved["contract_profile"] == "balanced_v4"
    assert resolved["matrix"]["learning_rate_selection"][
        "selection_id"
    ] == matrix.COMPARISON_SELECTION_ID
    if cell["task"] == "FlyCrazyflie-WaypointSwitch-v0":
        assert resolved["switch_target_curriculum"] == (
            matrix.balanced_v4_switch_target_curriculum_payload()
        )
    assert protocol == load_protocol("main")


def test_balanced_v4_comparison_rejects_changed_lr_selection(tmp_path):
    source = (
        ROOT
        / "configs"
        / "experiments"
        / "crazyflie_task_separated_v1_balanced_v4_seed0_comparison_base.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["learning_rate_selection"]["receipt_sha256"] = "0" * 64
    changed = tmp_path / "changed-selection.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate_selection declaration changed"):
        matrix.validate_config(changed)


def test_balanced_v4_comparison_execute_requires_reviewed_dry_run(
    tmp_path, monkeypatch
):
    output = tmp_path / "missing-reviewed-queue.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drone_run_matrix.py",
            "--config",
            str(
                ROOT
                / "configs"
                / "experiments"
                / "crazyflie_task_separated_v1_balanced_v4_seed0_comparison.json"
            ),
            "--execute",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        matrix.main()
    assert not output.exists()


def _passing_comparison_concurrency_decision():
    return {
        "status": "paired_smoke_pass",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 2,
        "fallback_applied": False,
        "fallback_reason": None,
        "receipt": str(matrix.COMPARISON_CONCURRENCY_RECEIPT.resolve()),
        "receipt_sha256": "a" * 64,
        "receipt_id": "b" * 64,
        "source_set_sha256": "c" * 64,
        "reports": [],
        "aggregate_overlap_memory_gate": {"passed": True},
    }


def test_paired_smoke_source_auth_requires_the_complete_current_set(
    tmp_path, monkeypatch
):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    current = {
        str(first): matrix.sha256_file(first),
        str(second): matrix.sha256_file(second),
    }
    monkeypatch.setattr(matrix, "source_hashes", lambda: dict(current))

    assert matrix._validate_current_source_hashes(
        {"source_sha256": dict(current)}, label="unit paired smoke"
    ) == matrix.canonical_sha256(current)

    incomplete = {str(first): current[str(first)]}
    with pytest.raises(ValueError, match="complete current executable source set"):
        matrix._validate_current_source_hashes(
            {"source_sha256": incomplete}, label="unit paired smoke"
        )

    monkeypatch.setattr(
        matrix, "source_hashes", lambda: {str(first): current[str(first)]}
    )
    with pytest.raises(ValueError, match="complete current executable source set"):
        matrix._validate_current_source_hashes(
            {"source_sha256": dict(current)}, label="unit paired smoke"
        )


def _comparison_runtime_queue(tmp_path, monkeypatch, decision):
    config = matrix.validate_config(
        ROOT
        / "configs"
        / "experiments"
        / "crazyflie_task_separated_v1_balanced_v4_seed0_comparison.json"
    )
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    monkeypatch.setattr(
        matrix, "comparison_concurrency_decision", lambda _config: deepcopy(decision)
    )
    queue_path = tmp_path / "comparison.json"
    queue = matrix.build_queue(config, queue_path)
    return queue_path, queue


def _install_fake_comparison_runtime(monkeypatch, *, fail_ids=(), pause=False):
    trained = set()
    evaluated = set()
    active = 0
    maximum_active = 0
    started_training = []
    runtime_lock = threading.Lock()

    def fake_valid_training(job):
        return job["id"] in trained

    def fake_valid_evaluation(bundle, _job, _expected):
        return bundle["output"] in evaluated

    def fake_run(command, _log):
        nonlocal active, maximum_active
        with runtime_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if any(str(item).endswith("drone_train.py") for item in command):
                run_dir = Path(command[command.index("--run_dir") + 1])
                job_id = run_dir.name
                with runtime_lock:
                    started_training.append(job_id)
                time.sleep(0.02)
                if pause:
                    Path(command[command.index("--pause_file") + 1]).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    Path(command[command.index("--pause_file") + 1]).write_text(
                        "{}", encoding="utf-8"
                    )
                    return {"exit_code": 3, "wall_time_s": 0.02, "log": "unit"}
                if job_id in fail_ids:
                    return {"exit_code": 1, "wall_time_s": 0.02, "log": "unit"}
                trained.add(job_id)
            else:
                output = command[command.index("--output") + 1]
                time.sleep(0.01)
                evaluated.add(output)
            return {"exit_code": 0, "wall_time_s": 0.02, "log": "unit"}
        finally:
            with runtime_lock:
                active -= 1

    monkeypatch.setattr(matrix, "_valid_training", fake_valid_training)
    monkeypatch.setattr(matrix, "_valid_evaluation", fake_valid_evaluation)
    monkeypatch.setattr(matrix, "_run", fake_run)
    return {
        "trained": trained,
        "evaluated": evaluated,
        "started_training": started_training,
        "maximum_active": lambda: maximum_active,
    }


def test_comparison_two_worker_scheduler_bounds_concurrency_and_keeps_lif_first(
    tmp_path, monkeypatch
):
    decision = _passing_comparison_concurrency_decision()
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, decision)
    observed = _install_fake_comparison_runtime(monkeypatch)

    assert matrix.execute(queue, queue_path, max_jobs=4) == 0
    assert observed["maximum_active"]() == 2
    expected_first_four = [job["id"] for job in queue["jobs"][:4]]
    assert set(observed["started_training"]) == set(expected_first_four)
    assert len(observed["started_training"]) == len(set(observed["started_training"])) == 4
    assert all(
        queue["jobs"][index]["status"] == "completed" for index in range(4)
    )
    assert all(job["status"] == "pending" for job in queue["jobs"][4:])
    assert not any(
        job_id.startswith("FlyCrazyflie") and "gru_matched" in job_id
        for job_id in observed["started_training"]
    )


def test_comparison_two_worker_scheduler_preserves_failure_and_other_job(
    tmp_path, monkeypatch
):
    decision = _passing_comparison_concurrency_decision()
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, decision)
    failed_id = queue["jobs"][0]["id"]
    observed = _install_fake_comparison_runtime(
        monkeypatch, fail_ids={failed_id}
    )

    assert matrix.execute(queue, queue_path, max_jobs=2) == 1
    assert observed["maximum_active"]() == 2
    assert queue["jobs"][0]["status"] == "failed"
    assert "Training command failed" in queue["jobs"][0]["failure"]
    assert queue["jobs"][1]["status"] == "completed"
    assert all(job["status"] == "pending" for job in queue["jobs"][2:])


def test_comparison_two_worker_scheduler_stops_submission_on_pause(
    tmp_path, monkeypatch
):
    decision = _passing_comparison_concurrency_decision()
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, decision)
    observed = _install_fake_comparison_runtime(monkeypatch, pause=True)

    assert matrix.execute(queue, queue_path) == 3
    assert observed["maximum_active"]() == 2
    assert len(observed["started_training"]) == 2
    assert all(job["status"] == "paused" for job in queue["jobs"][:2])
    assert all(job["status"] == "pending" for job in queue["jobs"][2:])


def test_comparison_interrupt_kills_children_before_executor_shutdown(
    tmp_path, monkeypatch
):
    decision = _passing_comparison_concurrency_decision()
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, decision)
    events = []
    submitted = []

    class InterruptFuture:
        def result(self):
            raise KeyboardInterrupt

        def cancel(self):
            events.append("cancel")
            return True

    class AuditExecutor:
        def __init__(self, *, max_workers, thread_name_prefix):
            assert max_workers == 2
            assert thread_name_prefix == "crazyflie-comparison"
            self.stop_events = []

        def submit(self, _function, *_args, **kwargs):
            self.stop_events.append(kwargs["stop_submitting"])
            future = InterruptFuture()
            submitted.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures=False):
            assert wait is True
            assert cancel_futures is True
            assert self.stop_events
            assert all(event.is_set() for event in self.stop_events)
            events.append("shutdown")

    monkeypatch.setattr(matrix, "ThreadPoolExecutor", AuditExecutor)
    monkeypatch.setattr(
        matrix,
        "wait",
        lambda futures, return_when: (set(futures), set()),
    )
    monkeypatch.setattr(
        matrix, "_terminate_active_children", lambda: events.append("terminate")
    )

    with pytest.raises(KeyboardInterrupt):
        matrix.execute(queue, queue_path, max_jobs=2)

    assert len(submitted) == 2
    assert events[0] == "terminate"
    assert events[1:-1] == ["cancel", "cancel"]
    assert events[-1] == "shutdown"


def test_comparison_missing_receipt_falls_back_to_real_sequential_execution(
    tmp_path, monkeypatch
):
    decision = {
        "status": "fallback_sequential",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 1,
        "fallback_applied": True,
        "fallback_reason": "FileNotFoundError: unit missing receipt",
        "receipt": str(matrix.COMPARISON_CONCURRENCY_RECEIPT.resolve()),
        "receipt_sha256": None,
        "receipt_id": None,
        "source_set_sha256": None,
        "reports": [],
        "aggregate_overlap_memory_gate": None,
    }
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, decision)
    observed = _install_fake_comparison_runtime(monkeypatch)
    monkeypatch.setattr(
        matrix,
        "_execute_comparison_concurrent",
        lambda *_args, **_kwargs: pytest.fail("fallback must not use two-worker path"),
    )

    assert matrix.execute(queue, queue_path, max_jobs=2) == 0
    assert observed["maximum_active"]() == 1
    assert observed["started_training"] == [job["id"] for job in queue["jobs"][:2]]
    assert queue["max_concurrent_isaac_processes"] == 1


def test_comparison_execute_rejects_forged_concurrency_upgrade(
    tmp_path, monkeypatch
):
    fallback = {
        "status": "fallback_sequential",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 1,
        "fallback_applied": True,
        "fallback_reason": "FileNotFoundError: unit missing receipt",
        "receipt": str(matrix.COMPARISON_CONCURRENCY_RECEIPT.resolve()),
        "receipt_sha256": None,
        "receipt_id": None,
        "source_set_sha256": None,
        "reports": [],
        "aggregate_overlap_memory_gate": None,
    }
    queue_path, queue = _comparison_runtime_queue(tmp_path, monkeypatch, fallback)
    queue["concurrency_decision"] = _passing_comparison_concurrency_decision()
    queue["max_concurrent_isaac_processes"] = 2
    queue["resource_limits"]["max_concurrent_isaac_processes"] = 2

    with pytest.raises(ValueError, match="decision.*changed"):
        matrix.execute(queue, queue_path)


@pytest.mark.parametrize("include_both", [True, False], ids=["both", "neither"])
def test_matrix_config_requires_exactly_one_task_contract(tmp_path, include_both):
    source = ROOT / "configs" / "experiments" / "crazyflie_balanced_v3_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    if include_both:
        config["survival_first_contract"] = matrix.survival_first_contract_payload()
    else:
        config.pop("balanced_task_contract")
    changed = tmp_path / f"contracts-{'both' if include_both else 'neither'}.json"
    changed.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        matrix.validate_config(changed)


@pytest.mark.parametrize(
    "config_name", ("crazyflie_main.json", "crazyflie_integration.json")
)
def test_legacy_survival_matrix_configs_remain_accepted(config_name):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / config_name)

    assert config["task"] == matrix.SURVIVAL_TASK
    assert config["_contract_profile"] == matrix.CONTRACT_PROFILE_SURVIVAL_V2
    assert config["survival_first_contract"] == matrix.survival_first_contract_payload()
    assert "balanced_task_contract" not in config


def test_balanced_main_queue_has_exact_order_budget_and_training_cli(tmp_path, monkeypatch):
    config = matrix.validate_config(
        ROOT / "configs" / "experiments" / "crazyflie_balanced_v3_main.json"
    )
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / "balanced-main.json")

    expected_controllers = (
        "frozen_lif_original",
        "frozen_lif_degree_rewired",
        "gru_matched",
        "mlp_normal",
    )
    assert matrix.CONTROLLERS == expected_controllers
    expected_cells = [
        (controller, seed)
        for controller in expected_controllers
        for seed in (0, 1, 2, 3, 4)
    ]
    assert [(job["controller"], job["seed"]) for job in queue["jobs"]] == expected_cells
    assert queue["job_count"] == len(queue["jobs"]) == 20
    assert queue["evaluation_bundle_count"] == 60
    assert queue["predicted_evaluation_episodes"] == 960
    assert all(len(job["evaluations"]) == 3 for job in queue["jobs"])
    assert all(
        tuple(bundle["scenario"] for bundle in job["evaluations"]) == SCENARIOS
        for job in queue["jobs"]
    )
    for job in queue["jobs"]:
        command = job["training_command"]
        assert job["task"] == matrix.BALANCED_TASK
        assert job["contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
        assert command[command.index("--task") + 1] == "FlyCrazyflie-Mixed-v0"
        assert command[command.index("--contract_profile") + 1] == "balanced_v3"


def test_balanced_per_seed_mixed_contract_is_in_resolved_fingerprint(monkeypatch):
    config = matrix.validate_config(
        ROOT / "configs" / "experiments" / "crazyflie_balanced_v3_main.json"
    )
    captured = []

    def fake_fingerprint(*, resolved_config, **_kwargs):
        captured.append(deepcopy(resolved_config))
        mixed_contract = resolved_config["mixed_scenario_contract"]
        return matrix.canonical_sha256(mixed_contract), {"resolved_config": resolved_config}

    monkeypatch.setattr(matrix, "reproduction_fingerprint", fake_fingerprint)
    fingerprints = {}
    for seed in (0, 4):
        fingerprints[seed], _ = matrix.job_fingerprint(config, matrix.CONTROLLERS[0], seed)

    for resolved, seed in zip(captured, (0, 4), strict=True):
        expected_mixed = matrix.mixed_scenario_contract_payload(seed=seed)
        assert resolved == matrix.resolved_job_config(config, matrix.CONTROLLERS[0], seed)
        assert resolved["task"] == matrix.BALANCED_TASK
        assert resolved["contract_profile"] == matrix.CONTRACT_PROFILE_BALANCED_V3
        assert resolved["mixed_scenario_contract"] == expected_mixed
        assert resolved["switch_target_curriculum"] == (
            matrix.balanced_switch_target_curriculum_payload()
        )
        assert fingerprints[seed] == matrix.canonical_sha256(expected_mixed)
    assert fingerprints[0] != fingerprints[4]


def test_main_config_enumerates_exact_complete_matrix(tmp_path, monkeypatch):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_main.json")
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / "main.json")
    assert queue["job_count"] == 20
    assert queue["evaluation_bundle_count"] == 60
    assert queue["predicted_evaluation_episodes"] == 960
    assert queue["resource_limits"]["device_gpu_used_mib_exclusive"] == 6963.2
    assert queue["resource_limits"]["system_ram_percent_exclusive"] == 90.0
    assert queue["resource_limits"]["policy_version"] == "crazyflie_memory_acceptance_v2"
    assert queue["resource_limits"]["rss_growth_disposition"] == "warning_only"
    assert queue["resource_limits"]["max_concurrent_isaac_processes"] == 1
    assert len({(job["controller"], job["seed"]) for job in queue["jobs"]}) == 20
    assert len({job["run_dir"] for job in queue["jobs"]}) == 20
    assert all(job["total_interactions"] == 5_000_000 for job in queue["jobs"])
    assert queue["config"]["training"]["horizon"] == 100
    assert queue["config"]["training"]["learning_rate"] == 3.0e-5
    assert {job["seed"] for job in queue["jobs"]} == {0, 1, 2, 3, 4}
    assert tuple(dict.fromkeys(job["controller"] for job in queue["jobs"])) == matrix.CONTROLLERS
    assert all(len(job["evaluations"]) == 3 for job in queue["jobs"])
    assert all(tuple(item["scenario"] for item in job["evaluations"]) == SCENARIOS for job in queue["jobs"])
    evaluation = queue["config"]["evaluation"]
    assert evaluation["policy_inference_device"] == matrix.POLICY_INFERENCE_DEVICE
    assert evaluation["policy_inference_backend"] == matrix.POLICY_INFERENCE_BACKEND
    assert evaluation["policy_inference_precision"] == "float32"
    assert evaluation["policy_inference_graph"] == matrix._policy_graph_config(4)
    assert evaluation["policy_bridge"] == matrix.POLICY_BRIDGE_CONTRACT
    assert all("--total_interactions" in job["training_command"] for job in queue["jobs"])
    assert all("--headless" in job["training_command"] for job in queue["jobs"])
    assert all(
        bundle["command"][bundle["command"].index("--device") + 1] == "cuda:0"
        for job in queue["jobs"]
        for bundle in job["evaluations"]
    )
    for flag in (
        "--microbatch_size",
        "--gamma",
        "--gae_lambda",
        "--clip_ratio",
        "--value_coefficient",
        "--entropy_coefficient",
        "--max_grad_norm",
        "--rewire_manifest",
    ):
        assert all(flag in job["training_command"] for job in queue["jobs"])


def test_matrix_config_rejects_conflicting_memory_acceptance(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["memory_acceptance"] = {
        **matrix.memory_acceptance_contract_payload(),
        "rss_growth_disposition": "hard_failure",
    }
    changed = tmp_path / "wrong-memory-policy.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="crazyflie_memory_acceptance_v2"):
        matrix.validate_config(changed)


def test_lif_proof_protocol_is_available_and_enters_standalone_fingerprint(monkeypatch):
    args = SimpleNamespace(
        task="FlyCrazyflie-Mixed-v0",
        contract_profile="balanced_v3",
        policy="frozen_lif_original",
        seed=0,
        num_envs=1,
        total_interactions=500_000,
        horizon=100,
        microbatch_size=1,
        ppo_epochs=2,
        learning_rate=3.0e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        max_grad_norm=1.0,
        target_kl=0.05,
        checkpoint_every_updates=100,
        connectome_manifest=training_script.DEFAULT_CONNECTOME,
        wing_connectome_manifest=training_script.DEFAULT_WING_CONNECTOME,
        rewire_seed=20260916,
        rewire_manifest=training_script.DEFAULT_REWIRE_MANIFEST,
        evaluation_protocol="lif_proof",
        warm_start_checkpoint=None,
        matrix_config=None,
        expected_fingerprint=None,
    )
    captured = {}

    def fake_fingerprint(**kwargs):
        captured.update(kwargs)
        return "f" * 64, {"source_sha256": {}}

    monkeypatch.setattr(training_script, "reproduction_fingerprint", fake_fingerprint)
    monkeypatch.setattr(
        training_script,
        "load_fingerprint_rewire_manifest",
        lambda *_args, **_kwargs: {"unit": True},
    )
    resolved, fingerprint, _payload, protocol = training_script._resolved_training_config(args)
    assert "lif_proof" in training_script.EVALUATION_PROTOCOLS
    assert resolved["evaluation_protocol"] == "lif_proof"
    assert resolved["memory_acceptance"] == training_script.memory_acceptance_contract_payload()
    assert protocol == load_protocol("lif_proof")
    assert captured["evaluation_manifest"] == protocol
    assert fingerprint == "f" * 64


def test_integration_config_is_four_controller_seed_zero(tmp_path, monkeypatch):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_integration.json")
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / "integration.json")
    assert queue["label"] == "integration"
    assert queue["job_count"] == 4
    assert queue["evaluation_bundle_count"] == 12
    assert queue["predicted_evaluation_episodes"] == 24
    assert queue["config"]["training"]["horizon"] == 25
    assert queue["config"]["training"]["learning_rate"] == 1.0e-4
    assert {job["seed"] for job in queue["jobs"]} == {0}
    assert queue["config"]["evaluation"]["policy_inference_graph"] == matrix._policy_graph_config(2)


def test_matrix_config_requires_exact_rollout_rng_contract(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    assert config["rollout_rng_contract"] == rollout_rng_contract()
    config["rollout_rng_contract"]["resume_behavior"] = "silently_reseed"
    changed = tmp_path / "wrong-rollout-rng.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout_rng_contract"):
        matrix.validate_config(changed)


def test_matrix_config_freezes_survival_reward_and_reset_curriculum(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_main.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    contract = config["survival_first_contract"]
    assert contract == matrix.survival_first_contract_payload()
    assert contract["reward"] == {
        "action_change_normalized_squared_cap": 4.0,
        "action_change_scale": 0.001,
        "collective_hover_action": 2.0 / 1.9 - 1.0,
        "control_effort_normalized_squared_cap": 4.0,
        "control_effort_scale": 0.01,
        "failure_penalty": 20.0,
        "moment_action_reference": 0.005,
        "progress_clip_m": 0.05,
        "progress_scale": 0.5,
        "success_bonus": 5.0,
        "survival_reward_per_interval": 0.02,
        "version": "crazyflie_survival_first_reward_v2",
    }
    assert [
        stage["start_interactions"]
        for stage in contract["training_curriculum"]["stages"]
    ] == [0, 200_000, 500_000, 1_000_000]

    config["survival_first_contract"]["reward"]["progress_scale"] = 10.0
    changed = tmp_path / "wrong-survival-contract.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="survival_first_contract"):
        matrix.validate_config(changed)


def test_matrix_config_freezes_shared_stabilization_and_residual_contract(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_main.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    assert config["stabilization_and_residual_contract"] == stabilization_contract_payload()
    assert config["stabilization_and_residual_contract"]["version"] == (
        "crazyflie_shared_stabilization_bounded_residual_v2"
    )
    assert config["stabilization_and_residual_contract"]["residual"]["latent_scale"] == [
        0.35, 0.08, 0.08, 0.05
    ]
    config["stabilization_and_residual_contract"]["residual"]["latent_scale"][0] = 0.04
    changed = tmp_path / "wrong-stabilization-contract.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="stabilization_and_residual_contract"):
        matrix.validate_config(changed)


@pytest.mark.parametrize(
    ("config_name", "field", "wrong_value"),
    (
        ("crazyflie_main.json", "horizon", 25),
        ("crazyflie_main.json", "learning_rate", 1.0e-4),
        ("crazyflie_integration.json", "learning_rate", 3.0e-5),
    ),
)
def test_matrix_config_rejects_unreviewed_rollout_or_learning_rate(
    tmp_path, config_name, field, wrong_value
):
    source = ROOT / "configs" / "experiments" / config_name
    config = json.loads(source.read_text(encoding="utf-8"))
    config["training"][field] = wrong_value
    changed = tmp_path / f"wrong-{field}.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        matrix.validate_config(changed)


def test_matrix_config_requires_exact_cuda_graph_policy_inference(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["evaluation"]["policy_inference_device"] = "cpu"
    changed = tmp_path / "wrong-policy-device.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="policy inference device must be cuda:0"):
        matrix.validate_config(changed)


def test_rewire_artifact_is_validated_and_included_in_fingerprint(monkeypatch):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_integration.json")
    captured = {}

    def fake_fingerprint(**kwargs):
        captured.update(kwargs)
        return "a" * 64, kwargs

    monkeypatch.setattr(matrix, "reproduction_fingerprint", fake_fingerprint)
    fingerprint, _ = matrix.job_fingerprint(config, matrix.CONTROLLERS[0], 0)
    assert fingerprint == "a" * 64
    assert captured["rewired_manifest"] == config["_rewire_manifest"]
    assert captured["rewired_manifest"]["seed"] == config["rewire_seed"]
    assert matrix.sha256_file(config["_rewire_manifest_path"]) == config["rewire_manifest_sha256"]


def test_rewire_artifact_rejects_file_or_content_mismatch(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["rewire_manifest_sha256"] = "0" * 64
    wrong_file_hash = tmp_path / "wrong-file-hash.json"
    wrong_file_hash.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        matrix.validate_config(wrong_file_hash)

    artifact = json.loads(
        (ROOT / config["rewire_manifest"]).read_text(encoding="utf-8")
    )
    artifact["seed"] += 1
    mutated_artifact = tmp_path / "mutated-rewire.json"
    mutated_artifact.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    config["rewire_manifest"] = str(mutated_artifact)
    config["rewire_manifest_sha256"] = matrix.sha256_file(mutated_artifact)
    wrong_content = tmp_path / "wrong-content.json"
    wrong_content.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="content validation"):
        matrix.validate_config(wrong_content)


def test_main_manifest_is_deterministic_valid_and_held_out():
    checked_in = load_protocol("main")
    regenerated = generate_manifest(seed=101, episodes_per_scenario=16, label="main")
    assert checked_in == regenerated
    assert checked_in["manifest_id"] == validate_manifest(regenerated)["manifest_id"]
    assert all(len(checked_in["scenarios"][scenario]) == 16 for scenario in SCENARIOS)
    hashes = [
        plan["plan_sha256"]
        for scenario in SCENARIOS
        for plan in checked_in["scenarios"][scenario]
    ]
    assert len(hashes) == len(set(hashes)) == 48


def test_manifest_rejects_mutation():
    manifest = generate_manifest(seed=101, episodes_per_scenario=2, label="integration")
    manifest["scenarios"][SCENARIOS[0]][0]["initial_state"]["yaw_rad"] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(manifest)


def test_manifest_rejects_rechecksummed_semantic_mutation():
    manifest = generate_manifest(seed=101, episodes_per_scenario=2, label="integration")
    plan = manifest["scenarios"][SCENARIOS[0]][0]
    plan["targets_relative_to_env_origin_m"][0][0] += 0.01
    plan_without_hash = dict(plan)
    plan_without_hash.pop("plan_sha256")
    plan["plan_sha256"] = matrix.canonical_sha256(plan_without_hash)
    manifest_without_id = dict(manifest)
    manifest_without_id.pop("manifest_id")
    manifest["manifest_id"] = matrix.canonical_sha256(manifest_without_id)
    with pytest.raises(ValueError, match="frozen seed-derived"):
        validate_manifest(manifest, expected_episodes=2)


def test_manifest_rejects_rechecksummed_unknown_schema_field():
    manifest = generate_manifest(seed=101, episodes_per_scenario=2, label="integration")
    manifest["unreviewed_extension"] = True
    manifest_without_id = dict(manifest)
    manifest_without_id.pop("manifest_id")
    manifest["manifest_id"] = matrix.canonical_sha256(manifest_without_id)
    with pytest.raises(ValueError, match="closed version-1 schema"):
        validate_manifest(manifest, expected_episodes=2)


def test_protocol_writer_preserves_differing_existing_file_and_accepts_identical(
    tmp_path, monkeypatch
):
    import drone_evaluation_protocol as protocol

    output = tmp_path / "protocol.json"
    different = b'{"preserve": true}\n'
    output.write_bytes(different)
    argv = [
        "drone_evaluation_protocol.py", "--seed", "101", "--episodes", "2",
        "--label", "integration", "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        protocol.main()
    assert output.read_bytes() == different

    output.unlink()
    expected = generate_manifest(seed=101, episodes_per_scenario=2, label="integration")
    output.write_text(json.dumps(expected), encoding="utf-8")
    before = output.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    assert protocol.main() == 0
    assert output.read_bytes() == before


def test_invalid_budget_that_would_overshoot_is_rejected(tmp_path):
    source = ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["total_interactions"] = 51
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 50"):
        matrix.validate_config(path)


def test_main_execute_requires_separate_authorization(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "drone_run_matrix.py",
        "--config", str(ROOT / "configs" / "experiments" / "crazyflie_main.json"),
        "--execute",
    ])
    with pytest.raises(SystemExit):
        matrix.main()


def _integration_job(tmp_path, monkeypatch):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_integration.json")
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    queue = matrix.build_queue(config, tmp_path / "integration.json")
    return config, queue, queue["jobs"][0]


def _memory_evidence(stages):
    samples = []
    for index, stage in enumerate(stages):
        samples.append({
            "timestamp_utc": f"2026-09-16T00:00:{index:02d}+00:00",
            "stage": stage,
            "step": index,
            "process_rss_mib": 100.0,
            "system_ram_total_gib": 32.0,
            "system_ram_used_gib": 4.0,
            "system_ram_available_gib": 28.0,
            "system_ram_percent": 12.5,
            "system_swap_total_gib": 8.0,
            "system_swap_used_gib": 0.0,
            "system_swap_percent": 0.0,
            "system_swap_in_mib": 0.0,
            "system_swap_out_mib": 0.0,
            "compute_device_type": "cuda",
            "gpu_devices": [{
                "index": 0,
                "name": "unit-gpu",
                "driver_version": "unit",
                "total_mib": 8192.0,
                "used_mib": 512.0,
            }],
            "torch_allocated_mib": 32.0,
            "torch_reserved_mib": 64.0,
            "torch_peak_allocated_mib": 32.0,
            "torch_peak_reserved_mib": 64.0,
        })
    return samples, matrix.assess_memory(samples)


def test_v2_memory_validator_accepts_rss_warning_but_requires_policy_metadata():
    stages = (
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "optimizer_update",
        "optimizer_update",
        "optimizer_update",
        "training_complete",
    )
    samples, _ = _memory_evidence(stages)
    optimizer_samples = [
        sample for sample in samples if sample["stage"] == "optimizer_update"
    ]
    for index, sample in enumerate(optimizer_samples):
        sample["process_rss_mib"] = 100.0 + index
    gate = matrix.assess_memory(samples)
    assert gate["passed"] is True
    assert gate["monotonic_process_growth_detected"] is True
    assert gate["warnings"]
    assert matrix._valid_training_memory(samples, gate)

    legacy_gate = deepcopy(gate)
    legacy_gate.pop("policy_version")
    assert not matrix._valid_training_memory(samples, legacy_gate)


@pytest.mark.parametrize(
    "failure_kind", ("ram_cap", "gpu_cap", "missing_gpu", "paging", "nonfinite")
)
def test_v2_memory_validator_rejects_every_hard_gate(failure_kind):
    stages = (
        "environment_loaded",
        "graph_captured",
        "steady_state",
        "steady_state",
        "steady_state",
        "steady_state",
        "evaluation_end",
    )
    samples, _ = _memory_evidence(stages)
    if failure_kind == "ram_cap":
        samples[-1]["system_ram_percent"] = 90.0
    elif failure_kind == "gpu_cap":
        samples[-1]["gpu_devices"][0]["used_mib"] = 6963.2
    elif failure_kind == "missing_gpu":
        samples[-1]["gpu_devices"] = []
    elif failure_kind == "paging":
        steady = [sample for sample in samples if sample["stage"] == "steady_state"]
        for index, sample in enumerate(steady):
            sample["system_swap_out_mib"] = float(index)
    else:
        samples[-1]["process_rss_mib"] = float("nan")
    gate = matrix.assess_memory(samples)
    assert gate["passed"] is False
    assert not matrix._valid_evaluation_memory(samples, gate)


def test_trainer_helpers_fail_on_nonfinite_code_4_but_allow_rss_warning():
    assert training_script._nonfinite_failure_count(
        {"failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0}}
    ) == 0
    with pytest.raises(RuntimeError, match="cause code 4"):
        training_script._raise_for_nonfinite_failure_count(1, completed_updates=7)

    samples, _ = _memory_evidence(
        ("optimizer_update",) * 4
    )
    for index, sample in enumerate(samples):
        sample["process_rss_mib"] = 100.0 + index
    warning_gate = matrix.assess_memory(samples)
    assert warning_gate["warnings"]
    training_script._memory_gate_hard_failure(warning_gate, stage="unit")

    failed_gate = deepcopy(warning_gate)
    failed_gate["passed"] = False
    failed_gate["failures"] = ["unit hard failure"]
    with pytest.raises(RuntimeError, match="unit hard failure"):
        training_script._memory_gate_hard_failure(failed_gate, stage="unit")


def test_failure_runtime_evidence_preserves_counters_history_samples_and_gate():
    class UnitCounters:
        @staticmethod
        def as_dict():
            return {"completed_updates": 3, "total_interactions": 75}

    samples, expected_gate = _memory_evidence(("optimizer_update",))
    history_reference = {"row_count": 3, "segments": []}
    evidence = training_script._failure_runtime_evidence(
        counters=UnitCounters(),
        history_reference=history_reference,
        pending_history_rows=1,
        memory_samples=samples,
    )
    assert evidence["counters"] == {
        "completed_updates": 3,
        "total_interactions": 75,
    }
    assert evidence["history_reference"] == history_reference
    assert evidence["pending_uncommitted_history_rows"] == 1
    assert evidence["memory_samples"] == samples
    assert evidence["memory_gate"] == expected_gate
    assert evidence["recomputed_memory_gate"] == expected_gate
    assert evidence["memory_gate_recomputation_error"] is None


def test_history_validator_rejects_nonfinite_failure_code_4():
    assert matrix._history_row_has_no_nonfinite_failure(
        {"failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0}}
    )
    assert not matrix._history_row_has_no_nonfinite_failure(
        {"failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 1}}
    )
    assert not matrix._history_row_has_no_nonfinite_failure({})
    assert not matrix._history_row_has_no_nonfinite_failure(
        {"failure_cause_counts": {"4": 0}}
    )


def test_training_artifact_requires_completed_exact_checkpoint_and_manifest(tmp_path, monkeypatch):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    monkeypatch.setattr(
        matrix, "_valid_checkpoint_history", lambda *_args, **_kwargs: True
    )
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-one")
    expected_report = {
        "controller_kind": "frozen_lif",
        "core_checksum": "unit-frozen-core",
        "actor_trainable_parameters": 123,
        "total_trainable_parameters": 456,
        "actor_parameter_match_passed": True,
    }
    checkpoint_payload = {
        "metadata": {
            "controller_report": expected_report,
            "core_checksum_before": expected_report["core_checksum"],
            "core_checksum_after": expected_report["core_checksum"],
        },
        "fingerprints": {"frozen_core": expected_report["core_checksum"]},
        "core_checksum": expected_report["core_checksum"],
    }
    monkeypatch.setattr(matrix, "_expected_controller_report", lambda _job: expected_report)
    monkeypatch.setattr(matrix, "_read_training_checkpoint", lambda _path: checkpoint_payload)
    memory_samples, memory_gate = _memory_evidence((
        "environment_loaded", "controller_loaded", "rollout",
        "optimizer_update", "training_complete",
    ))
    manifest_path = Path(job["run_dir"]) / "training_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": job["task"],
        "contract_profile": job["contract_profile"],
        "controller": job["controller"],
        "seed": job["seed"],
        "requested_interactions": job["total_interactions"],
        "environment_interactions": job["total_interactions"],
        "completed_updates": job["total_interactions"] // (
            job["training_resources"]["num_envs"] * job["training_resources"]["horizon"]
        ),
        "fingerprint": job["expected_fingerprint"],
        "fingerprint_payload": job["fingerprint_payload"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "controller_report": expected_report,
        "core_checksum_before": expected_report["core_checksum"],
        "core_checksum_after": expected_report["core_checksum"],
        "memory_samples": memory_samples,
        "memory_gate": memory_gate,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": matrix.sha256_file(checkpoint),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert matrix._valid_training(job)

    for field, invalid in (
        ("status", "paused"),
        ("environment_interactions", job["total_interactions"] - 1),
        ("completed_updates", 0),
        ("fingerprint", "f" * 64),
        ("evaluation_manifest_id", "wrong-manifest"),
        ("memory_gate", {**memory_gate, "passed": False}),
        ("controller_report", {**expected_report, "actor_trainable_parameters": 124}),
        ("core_checksum_after", "mutated-core"),
        ("checkpoint_sha256", "0" * 64),
    ):
        changed = dict(manifest)
        changed[field] = invalid
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not matrix._valid_training(job), field

    forged_memory = deepcopy(manifest)
    forged_memory["memory_samples"][0]["gpu_devices"][0]["used_mib"] = 7000.0
    forged_memory["memory_gate"] = memory_gate
    manifest_path.write_text(json.dumps(forged_memory), encoding="utf-8")
    assert not matrix._valid_training(job)

    checkpoint_payload["core_checksum"] = "mutated-checkpoint-core"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not matrix._valid_training(job)
    checkpoint_payload["core_checksum"] = expected_report["core_checksum"]

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-two")
    assert not matrix._valid_training(job)


def test_truncated_checkpoint_history_validation_fails_closed(tmp_path, monkeypatch):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    metadata = {
        "completed_updates": 2,
        "history_reference": {},
    }
    assert matrix._valid_checkpoint_history(checkpoint, metadata, job) is False


def test_matrix_checkpoint_read_does_not_materialize_external_history(
    tmp_path, monkeypatch
):
    from g1_fly_control.crazyflie import checkpoint as checkpoint_module

    observed = {}

    def fake_read(path, *, map_location, resolve_external_history):
        observed.update({
            "path": path,
            "map_location": map_location,
            "resolve_external_history": resolve_external_history,
        })
        return {"validated": True}

    monkeypatch.setattr(checkpoint_module, "read_checkpoint", fake_read)
    source = tmp_path / "latest.pt"
    assert matrix._read_training_checkpoint(source) == {"validated": True}
    assert observed == {
        "path": source,
        "map_location": "cpu",
        "resolve_external_history": False,
    }


def test_external_checkpoint_history_is_stream_validated_without_materializing(
    tmp_path, monkeypatch
):
    from g1_fly_control.crazyflie.checkpoint import (
        build_history_reference_from_segments,
        write_history_segment,
    )

    _, _, job = _integration_job(tmp_path, monkeypatch)
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "completed_updates": update,
            "total_interactions": update * 25,
            "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
        }
        for update in (1, 2)
    ]
    segment = write_history_segment(
        Path(job["run_dir"]) / "history" / "rows-00000001-00000002.jsonl",
        rows,
        reference_directory=checkpoint.parent,
    )
    reference = build_history_reference_from_segments(
        [segment], checkpoint_path=checkpoint, interactions_per_update=25
    )
    metadata = {"completed_updates": 2, "history_reference": reference}
    payload = {
        "counters": {"completed_updates": 2, "total_interactions": 50},
        "history": [],
        "history_reference": reference,
        "fingerprints": {"reproduction": job["expected_fingerprint"]},
        "interactions_per_update": 25,
    }
    assert matrix._valid_checkpoint_history(
        checkpoint, metadata, job, payload=payload
    )

    segment_path = Path(job["run_dir"]) / "history" / "rows-00000001-00000002.jsonl"
    segment_path.write_bytes(segment_path.read_bytes() + b"{}\n")
    assert not matrix._valid_checkpoint_history(
        checkpoint, metadata, job, payload=payload
    )


def test_inline_legacy_checkpoint_history_remains_supported(tmp_path, monkeypatch):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    checkpoint = Path(job["checkpoint"])
    metadata = {"completed_updates": 2, "history_reference": None}
    payload = {
        "counters": {"completed_updates": 2, "total_interactions": 50},
        "history": [
            {
                "completed_updates": 1,
                "total_interactions": 25,
                "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            },
            {
                "completed_updates": 2,
                "total_interactions": 50,
                "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            },
        ],
        "history_reference": None,
        "fingerprints": {"reproduction": job["expected_fingerprint"]},
        "interactions_per_update": 25,
    }
    assert matrix._valid_checkpoint_history(
        checkpoint, metadata, job, payload=payload
    )


def test_training_validator_fails_closed_on_corrupt_checkpoint(tmp_path, monkeypatch):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-a-torch-checkpoint")
    (Path(job["run_dir"]) / "training_manifest.json").write_text(
        json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8"
    )
    assert matrix._valid_training(job) is False


def test_precheckpoint_attempt_is_recoverably_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(training_script, "ROOT", tmp_path)
    run_dir = tmp_path / "runs" / "job"
    run_dir.mkdir(parents=True)
    marker = run_dir / "training_manifest.json"
    marker.write_text('{"status":"failed"}\n', encoding="utf-8")
    archive = training_script._prepare_run_directory(run_dir, resume=False)
    assert archive is not None
    assert (archive / marker.name).read_text(encoding="utf-8") == '{"status":"failed"}\n'
    assert (run_dir / "checkpoints").is_dir()
    assert not marker.exists()

    latest = run_dir / "checkpoints" / "latest.pt"
    latest.write_bytes(b"checkpoint")
    assert training_script._prepare_run_directory(run_dir, resume=True) is None
    with pytest.raises(ValueError, match="use --resume"):
        training_script._prepare_run_directory(run_dir, resume=False)


def _training_rng_probe() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(8)


def test_controller_reseed_is_independent_of_startup_rng_consumption():
    states = []
    for startup_draws in (1, 37):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        for _ in range(startup_draws):
            _training_rng_probe()
        report = training_script._initialize_controller_rng(17)
        policy, _ = build_controller("gru")
        states.append({name: value.detach().clone() for name, value in policy.state_dict().items()})
        assert report["controller_construction_reseed_applied"] is True

    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])


def test_fresh_rollout_reseed_is_architecture_independent():
    probes = []
    for kind in ("frozen_lif", "frozen_lif_rewired", "gru", "mlp"):
        random.seed(700 + len(kind))
        np.random.seed(800 + len(kind))
        torch.manual_seed(900 + len(kind))
        # These constructors deliberately consume different quantities of the
        # global torch stream.  The post-construction boundary must erase that
        # difference before task resets and policy exploration begin.
        build_controller(kind)
        report = training_script._initialize_rollout_rng(23, resume_restored=False)
        probes.append(_training_rng_probe())
        assert report["scheme"] == "post_construction_reseed_v1"
        assert report["fresh_reseed_applied"] is True
        assert report["checkpoint_rng_preserved"] is False

    reference = probes[0]
    for probe in probes[1:]:
        assert probe[0] == reference[0]
        assert probe[1] == reference[1]
        assert torch.equal(probe[2], reference[2])


def test_resume_rollout_rng_helper_preserves_restored_streams():
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    expected = _training_rng_probe()

    # A resumed process may preseed its throw-away controller construction.
    # Loading the checkpoint then restores these saved states, and the
    # post-construction helper must leave them untouched.
    training_script._initialize_controller_rng(999)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    report = training_script._initialize_rollout_rng(999, resume_restored=True)
    actual = _training_rng_probe()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert report["fresh_reseed_applied"] is False
    assert report["checkpoint_rng_preserved"] is True


def _matrix_memory_evidence():
    return _memory_evidence(
        ("environment_loaded", "graph_captured", "steady_state", "evaluation_end")
    )


def _evaluation_artifact(job, bundle):
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"evaluation-checkpoint")
    plans = load_protocol("integration")["scenarios"][bundle["scenario"]]
    records = [
        EpisodeSummary(
            scenario=bundle["scenario"],
            episode_id=episode_id,
            success=False,
            terminated=False,
            truncated=True,
            failure_reason=None,
            time_to_first_success_s=None,
            final_goal_error_m=1.0 + episode_id,
            integrated_goal_error_m_s=5.0 + episode_id,
            mean_speed_inside_target_region_m_s=None,
            crash=False,
            out_of_bounds=False,
            invalid_state=False,
            command_effort=2.0,
            command_smoothness=0.5,
            aggregate_wrench_mechanical_work_proxy_j=0.25,
            gust_outcomes=(
                tuple(
                    GustRecoveryOutcome(
                        gust_index=gust_index,
                        gust_start_step=gust["start_step"],
                        applied=True,
                        stable_before_gust=True,
                        recovered=False,
                        recovery_latency_s=None,
                        max_displacement_m=0.2,
                        post_gust_error_integral_m_s=0.3,
                    )
                    for gust_index, gust in enumerate(plans[episode_id]["gusts"])
                )
                if bundle["scenario"] == SCENARIOS[2]
                else ()
            ),
        )
        for episode_id in range(2)
    ]
    summary = summarize_episodes(records, expected_episode_count=2)
    for row, plan in zip(summary["episodes"], plans, strict=True):
        row.update({"plan": plan, "plan_sha256": plan["plan_sha256"]})
        if bundle["scenario"] == SCENARIOS[2]:
            expected_impulses = [
                [
                    AUDITED_CRAZYFLIE_MASS_KG
                    * gust["desired_mass_normalized_delta_velocity_m_s"]
                    * gust["direction_world_xy"][0],
                    AUDITED_CRAZYFLIE_MASS_KG
                    * gust["desired_mass_normalized_delta_velocity_m_s"]
                    * gust["direction_world_xy"][1],
                    0.0,
                ]
                for gust in plan["gusts"]
            ]
            row.update({
                "robot_mass_kg": AUDITED_CRAZYFLIE_MASS_KG,
                "gust_applied_impulse_w_n_s": deepcopy(expected_impulses),
                "gust_expected_impulse_w_n_s": deepcopy(expected_impulses),
                "gust_impulse_max_abs_error_n_s": 0.0,
            })
    memory_samples, memory_gate = _matrix_memory_evidence()
    return {
        "schema_version": 1,
        "status": "completed",
        "label": bundle["protocol_label"],
        "protocol": bundle["protocol"],
        "evaluation_seed": bundle["evaluation_seed"],
        "evaluation_manifest_id": bundle["evaluation_manifest_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": matrix.sha256_file(checkpoint),
        "training_seed": job["seed"],
        "controller": job["controller"],
        "fingerprint": job["expected_fingerprint"],
        "fingerprint_payload": job["fingerprint_payload"],
        "deterministic_actions": True,
        "simulation_device": matrix.POLICY_INFERENCE_DEVICE,
        "simulation_device_type": "cuda",
        "policy_inference_device": matrix.POLICY_INFERENCE_DEVICE,
        "policy_inference_device_type": "cuda",
        "policy_inference_backend": matrix.POLICY_INFERENCE_BACKEND,
        "policy_inference_precision": matrix.POLICY_INFERENCE_PRECISION,
        "policy_inference_graph": matrix._policy_graph_artifact(2),
        "policy_bridge": matrix.POLICY_BRIDGE_CONTRACT,
        "memory_samples": memory_samples,
        "memory_gate": memory_gate,
        "scenario": bundle["scenario"],
        "episodes": summary["episodes"],
        "summary": summary,
    }


def test_evaluation_artifact_requires_checkpoint_protocol_plans_and_exact_summary(tmp_path, monkeypatch):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    bundle = job["evaluations"][0]
    output = Path(bundle["output"])
    output.parent.mkdir(parents=True)
    artifact = _evaluation_artifact(job, bundle)
    output.write_text(json.dumps(artifact), encoding="utf-8")
    assert matrix._valid_evaluation(bundle, job, expected=2)

    mutations = []
    wrong_protocol = deepcopy(artifact)
    wrong_protocol["protocol"] = "main"
    mutations.append(wrong_protocol)
    wrong_manifest = deepcopy(artifact)
    wrong_manifest["evaluation_manifest_id"] = "wrong"
    mutations.append(wrong_manifest)
    duplicate_episode = deepcopy(artifact)
    duplicate_episode["episodes"][1]["episode_id"] = 0
    mutations.append(duplicate_episode)
    wrong_plan = deepcopy(artifact)
    wrong_plan["episodes"][0]["plan"]["initial_state"]["yaw_rad"] += 0.01
    mutations.append(wrong_plan)
    wrong_summary = deepcopy(artifact)
    wrong_summary["summary"]["success_count"] = 2
    mutations.append(wrong_summary)
    wrong_checkpoint_hash = deepcopy(artifact)
    wrong_checkpoint_hash["checkpoint_sha256"] = "0" * 64
    mutations.append(wrong_checkpoint_hash)
    wrong_policy_device = deepcopy(artifact)
    wrong_policy_device["policy_inference_device"] = "cpu"
    mutations.append(wrong_policy_device)
    wrong_bridge = deepcopy(artifact)
    wrong_bridge["policy_bridge"]["gpu_to_cpu_policy_tensor_transfers_per_decision"] = 1
    mutations.append(wrong_bridge)
    wrong_graph = deepcopy(artifact)
    wrong_graph["policy_inference_graph"]["bitwise_parity_verified"] = False
    mutations.append(wrong_graph)
    wrong_fingerprint_payload = deepcopy(artifact)
    wrong_fingerprint_payload["fingerprint_payload"]["tampered"] = True
    mutations.append(wrong_fingerprint_payload)
    wrong_memory = deepcopy(artifact)
    wrong_memory["memory_gate"]["device_gpu_telemetry_complete"] = False
    mutations.append(wrong_memory)
    for changed in mutations:
        output.write_text(json.dumps(changed), encoding="utf-8")
        assert not matrix._valid_evaluation(bundle, job, expected=2)

    output.write_text(json.dumps(artifact), encoding="utf-8")
    Path(job["checkpoint"]).write_bytes(b"changed-after-evaluation")
    assert not matrix._valid_evaluation(bundle, job, expected=2)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: row.__setitem__("robot_mass_kg", 1.0),
        lambda row: row["gust_applied_impulse_w_n_s"][0].__setitem__(0, 0.5),
        lambda row: row["gust_expected_impulse_w_n_s"][0].__setitem__(1, -0.5),
        lambda row: row.__setitem__("gust_impulse_max_abs_error_n_s", 0.5),
    ),
    ids=("mass", "applied-direction", "expected-direction", "max-error"),
)
def test_gust_evaluation_resume_rejects_tampered_impulse_evidence(
    tmp_path, monkeypatch, mutate
):
    _, _, job = _integration_job(tmp_path, monkeypatch)
    bundle = job["evaluations"][2]
    assert bundle["scenario"] == SCENARIOS[2]
    output = Path(bundle["output"])
    output.parent.mkdir(parents=True)
    artifact = _evaluation_artifact(job, bundle)
    output.write_text(json.dumps(artifact), encoding="utf-8")
    assert matrix._valid_evaluation(bundle, job, expected=2)

    changed = deepcopy(artifact)
    mutate(changed["episodes"][0])
    output.write_text(json.dumps(changed), encoding="utf-8")
    assert not matrix._valid_evaluation(bundle, job, expected=2)


def test_resume_consumes_archives_and_records_persistent_pause_request(tmp_path, monkeypatch):
    _, queue, job = _integration_job(tmp_path, monkeypatch)
    queue_path = tmp_path / "integration.json"
    matrix.save_queue(queue_path, queue)
    pause_file, request, created = matrix.write_pause_request(queue, queue_path)
    assert created is True
    assert pause_file.is_file()
    assert request["status"] == "requested"
    same_path, same_request, created_again = matrix.write_pause_request(queue, queue_path)
    assert same_path == pause_file
    assert same_request == request
    assert created_again is False

    job["status"] = "paused"
    job["pause_reason"] = "test request"
    event = matrix.consume_pause_request(queue, queue_path)
    matrix.save_queue(queue_path, queue)
    assert event["pause_request_consumed"] is True
    assert event["resumed_jobs"] == [job["id"]]
    assert not pause_file.exists()
    assert Path(event["request_archive"]).is_file()
    assert matrix.sha256_file(event["request_archive"]) == event["request_sha256"]
    assert job["status"] == "pending"
    persisted = json.loads(queue_path.read_text(encoding="utf-8"))
    assert persisted["resume_history"][-1] == event


def test_resume_queue_recomputes_and_rejects_stale_commands_or_fingerprints(tmp_path, monkeypatch):
    config, queue, _ = _integration_job(tmp_path, monkeypatch)
    queue_path = tmp_path / "integration.json"
    matrix.validate_resume_queue(queue, config, queue_path)
    queue["jobs"][0]["training_command"][0] = "/tmp/untrusted-python"
    with pytest.raises(ValueError, match="command or reproduction fingerprint is stale"):
        matrix.validate_resume_queue(queue, config, queue_path)


def test_completed_job_is_revalidated_before_it_can_be_skipped(tmp_path, monkeypatch):
    _, queue, job = _integration_job(tmp_path, monkeypatch)
    queue_path = tmp_path / "integration.json"
    job["status"] = "completed"
    for bundle in job["evaluations"]:
        bundle["status"] = "completed"
    calls = []
    monkeypatch.setattr(matrix, "_valid_training", lambda _job: False)
    monkeypatch.setattr(matrix, "_valid_evaluation", lambda _bundle, _job, _expected: False)

    def failed_run(command, log):
        calls.append((command, log))
        return {"exit_code": 1, "wall_time_s": 0.0, "log": str(log)}

    monkeypatch.setattr(matrix, "_run", failed_run)
    assert matrix.execute(queue, queue_path, max_jobs=1) == 1
    assert len(calls) == 1
    assert job["artifact_revalidation"]["result"] == "rejected_stale_or_incomplete_artifact"
    assert job["status"] == "failed"


def test_training_failure_is_recorded_and_later_job_still_runs(tmp_path, monkeypatch):
    _, queue, first = _integration_job(tmp_path, monkeypatch)
    second = queue["jobs"][1]
    queue_path = tmp_path / "integration.json"
    trained: set[str] = set()
    calls: list[str] = []
    monkeypatch.setattr(matrix, "_valid_training", lambda job: job["id"] in trained)
    monkeypatch.setattr(matrix, "_valid_evaluation", lambda *_args: True)

    def run_training(command, log):
        controller = command[command.index("--policy") + 1]
        calls.append(controller)
        if controller == first["controller"]:
            return {"exit_code": 1, "wall_time_s": 0.0, "log": str(log)}
        trained.add(second["id"])
        return {"exit_code": 0, "wall_time_s": 0.0, "log": str(log)}

    monkeypatch.setattr(matrix, "_run", run_training)
    assert matrix.execute(queue, queue_path, max_jobs=2) == 1
    assert calls == [first["controller"], second["controller"]]
    assert first["status"] == "failed"
    assert second["status"] == "completed"
    assert queue["jobs"][2]["status"] == "pending"


def test_evaluation_failure_does_not_strand_scenarios_or_later_job(tmp_path, monkeypatch):
    _, queue, first = _integration_job(tmp_path, monkeypatch)
    second = queue["jobs"][1]
    queue_path = tmp_path / "integration.json"
    completed_outputs: set[str] = set()
    calls: list[str] = []
    first_failure = first["evaluations"][0]["output"]
    monkeypatch.setattr(matrix, "_valid_training", lambda _job: True)
    monkeypatch.setattr(
        matrix,
        "_valid_evaluation",
        lambda bundle, _job, _expected: bundle["output"] in completed_outputs,
    )

    def run_evaluation(command, log):
        output = command[command.index("--output") + 1]
        calls.append(output)
        if output == first_failure:
            return {"exit_code": 1, "wall_time_s": 0.0, "log": str(log)}
        completed_outputs.add(output)
        return {"exit_code": 0, "wall_time_s": 0.0, "log": str(log)}

    monkeypatch.setattr(matrix, "_run", run_evaluation)
    assert matrix.execute(queue, queue_path, max_jobs=2) == 1
    assert len(calls) == 6
    assert first["status"] == "failed"
    assert [bundle["status"] for bundle in first["evaluations"]] == [
        "failed", "completed", "completed"
    ]
    assert second["status"] == "completed"


@pytest.mark.parametrize("child_exit_code", [0, 1])
def test_resume_is_not_masked_by_stale_paused_manifest(
    tmp_path, monkeypatch, child_exit_code
):
    _, queue, job = _integration_job(tmp_path, monkeypatch)
    queue_path = tmp_path / "integration.json"
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True)
    Path(job["checkpoint"]).parent.mkdir(parents=True)
    Path(job["checkpoint"]).write_bytes(b"stale-paused-checkpoint")
    (run_dir / "training_manifest.json").write_text(
        json.dumps({"status": "paused"}), encoding="utf-8"
    )
    monkeypatch.setattr(matrix, "_valid_training", lambda _job: False)
    monkeypatch.setattr(
        matrix, "_run",
        lambda _command, log: {
            "exit_code": child_exit_code, "wall_time_s": 0.0, "log": str(log)
        },
    )
    assert matrix.execute(queue, queue_path, max_jobs=1) == 1
    assert job["status"] == "failed"


def test_pause_is_honored_immediately_after_evaluation_bundle(tmp_path, monkeypatch):
    _, queue, job = _integration_job(tmp_path, monkeypatch)
    queue_path = tmp_path / "integration.json"
    pause_file = Path(queue["pause_file"])
    completed: set[str] = set()
    calls: list[str] = []
    monkeypatch.setattr(matrix, "_valid_training", lambda _job: True)
    monkeypatch.setattr(
        matrix, "_valid_evaluation",
        lambda bundle, _job, _expected: bundle["scenario"] in completed,
    )

    def finish_one_evaluation(_command, log):
        scenario = job["evaluations"][len(calls)]["scenario"]
        calls.append(scenario)
        completed.add(scenario)
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        pause_file.write_text('{"status":"requested"}\n', encoding="utf-8")
        return {"exit_code": 0, "wall_time_s": 0.0, "log": str(log)}

    monkeypatch.setattr(matrix, "_run", finish_one_evaluation)
    assert matrix.execute(queue, queue_path, max_jobs=1) == 3
    assert calls == [SCENARIOS[0]]
    assert job["evaluations"][0]["status"] == "completed"
    assert job["status"] == "paused"


def test_fingerprint_source_set_includes_executed_parent_initializers():
    paths = {path.resolve() for path in execution_source_paths()}
    package = ROOT / "source" / "g1_fly_control" / "g1_fly_control"
    assert (package / "__init__.py").resolve() in paths
    assert (package / "tasks" / "__init__.py").resolve() in paths
    assert UPSTREAM_DIRECT_RL_ENV.resolve() in paths
    assert UPSTREAM_ISAACLAB_MATH.resolve() in paths


def test_standalone_training_defaults_use_tuned_shared_lif_settings():
    assert training_script.STANDALONE_HORIZON == 100
    assert training_script.STANDALONE_LEARNING_RATE == 3.0e-5


def test_training_curriculum_counter_is_fail_closed_and_exact():
    class Environment:
        def __init__(self):
            self.calls = []
            self.training_interactions = 0
            self.active_training_curriculum_stage_index = 0
            self.active_training_curriculum_stage_payload = {"name": "stabilize"}

        def set_training_interactions(self, value):
            self.calls.append(value)
            self.training_interactions = value

    env = Environment()
    training_script._set_environment_training_interactions(env, 150_000)
    assert env.calls == [150_000]
    with pytest.raises(ValueError, match="non-negative integer"):
        training_script._set_environment_training_interactions(env, -1)
    with pytest.raises(RuntimeError, match="lacks set_training_interactions"):
        training_script._set_environment_training_interactions(object(), 0)

    snapshot = training_script._environment_curriculum_snapshot(
        env,
        expected_interactions=150_000,
    )
    assert snapshot == {
        "training_interactions": 150_000,
        "active_stage_index": 0,
        "active_stage_name": "stabilize",
        "active_stage": {"name": "stabilize"},
    }
    with pytest.raises(RuntimeError, match="curriculum clock disagrees"):
        training_script._environment_curriculum_snapshot(
            env,
            expected_interactions=150_400,
        )


def test_main_execute_cannot_create_an_unreviewed_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    monkeypatch.setattr(sys, "argv", [
        "drone_run_matrix.py",
        "--config", str(ROOT / "configs" / "experiments" / "crazyflie_main.json"),
        "--output", str(tmp_path / "main.json"),
        "--execute",
        "--authorize_main",
    ])
    with pytest.raises(SystemExit):
        matrix.main()


def test_dry_run_refuses_overwrite_and_resume_preview_preserves_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    output = tmp_path / "integration.json"
    base_argv = [
        "drone_run_matrix.py",
        "--config", str(ROOT / "configs" / "experiments" / "crazyflie_integration.json"),
        "--output", str(output),
        "--dry_run",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    assert matrix.main() == 0
    queue = json.loads(output.read_text(encoding="utf-8"))
    queue["preservation_marker"] = "must-survive"
    output.write_text(json.dumps(queue), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", base_argv)
    with pytest.raises(SystemExit):
        matrix.main()

    monkeypatch.setattr(sys, "argv", [*base_argv, "--resume"])
    assert matrix.main() == 0
    preserved = json.loads(output.read_text(encoding="utf-8"))
    assert preserved["preservation_marker"] == "must-survive"


def test_pause_request_rejects_corrupted_queue_path(tmp_path, monkeypatch):
    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_integration.json")
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    output = tmp_path / "integration.json"
    queue = matrix.build_queue(config, output)
    queue["pause_file"] = str(tmp_path / "g1-victim" / "pause.request")
    matrix.save_queue(output, queue)
    monkeypatch.setattr(sys, "argv", [
        "drone_run_matrix.py",
        "--config", str(ROOT / "configs" / "experiments" / "crazyflie_integration.json"),
        "--output", str(output),
        "--request_pause",
    ])
    with pytest.raises(SystemExit):
        matrix.main()
    assert not (tmp_path / "g1-victim" / "pause.request").exists()


def test_global_lock_collision_preserves_pause_request_and_paused_queue(tmp_path, monkeypatch):
    from contextlib import contextmanager

    config = matrix.validate_config(ROOT / "configs" / "experiments" / "crazyflie_integration.json")
    monkeypatch.setattr(matrix, "job_fingerprint", _cheap_fingerprint)
    output = tmp_path / "integration.json"
    queue = matrix.build_queue(config, output)
    queue["jobs"][0]["status"] = "paused"
    queue["jobs"][0]["pause_reason"] = "preserve me"
    matrix.save_queue(output, queue)
    pause_file, _, _ = matrix.write_pause_request(queue, output)
    original_lock = matrix.queue_lock

    @contextmanager
    def collide_on_global(path):
        if Path(path).name == ".crazyflie_isaac":
            raise RuntimeError("synthetic global collision")
        with original_lock(path):
            yield

    monkeypatch.setattr(matrix, "queue_lock", collide_on_global)
    monkeypatch.setattr(sys, "argv", [
        "drone_run_matrix.py",
        "--config", str(ROOT / "configs" / "experiments" / "crazyflie_integration.json"),
        "--output", str(output),
        "--execute", "--resume",
    ])
    with pytest.raises(SystemExit):
        matrix.main()
    assert pause_file.is_file()
    preserved = json.loads(output.read_text(encoding="utf-8"))
    assert preserved["jobs"][0]["status"] == "paused"
    assert preserved["jobs"][0]["pause_reason"] == "preserve me"


def test_matrix_logs_are_attempt_numbered_instead_of_overwritten(tmp_path):
    log = tmp_path / "job.log"
    command = [sys.executable, "-c", "print('attempt')"]
    first = matrix._run(command, log)
    second = matrix._run(command, log)
    assert first["exit_code"] == second["exit_code"] == 0
    assert first["log"] == str(log)
    assert second["log"] != first["log"]
    assert Path(first["log"]).read_text(encoding="utf-8").count("attempt") >= 1
    assert Path(second["log"]).read_text(encoding="utf-8").count("attempt") >= 1
