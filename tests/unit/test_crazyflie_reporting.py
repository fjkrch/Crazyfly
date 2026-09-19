"""Strict provenance checks used by the Crazyflie report generator."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_summarize_matrix as reporting  # noqa: E402


def test_completed_training_uses_runner_strict_validation(tmp_path, monkeypatch):
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    (run_dir / "training_manifest.json").write_text(
        json.dumps({
            "status": "completed",
            "controller": "frozen_lif_original",
            "seed": 0,
            "fingerprint": "f" * 64,
            "environment_interactions": 50,
            "controller_report": {},
            "memory_gate": {"passed": True},
        }),
        encoding="utf-8",
    )
    job = {
        "run_dir": str(run_dir),
        "controller": "frozen_lif_original",
        "seed": 0,
        "expected_fingerprint": "f" * 64,
        "total_interactions": 50,
        "checkpoint": str(run_dir / "missing.pt"),
    }
    monkeypatch.setattr(reporting.matrix_runner, "_valid_training", lambda _job: False)
    _details, errors = reporting._training_details(job, require_complete=True)
    assert any("queue runner's strict" in error for error in errors)


def test_completed_evaluation_uses_runner_strict_validation(tmp_path, monkeypatch):
    output = tmp_path / "evaluation.json"
    output.write_text(
        json.dumps({
            "status": "completed",
            "scenario": "FlyCrazyflie-WaypointReach-v0",
            "training_seed": 0,
            "fingerprint": "f" * 64,
            "episodes": [],
            "summary": {},
        }),
        encoding="utf-8",
    )
    bundle = {"output": str(output), "scenario": "FlyCrazyflie-WaypointReach-v0"}
    job = {
        "seed": 0,
        "controller": "frozen_lif_original",
        "expected_fingerprint": "f" * 64,
        "checkpoint": str(tmp_path / "missing.pt"),
    }
    monkeypatch.setattr(
        reporting.matrix_runner,
        "_valid_evaluation",
        lambda _bundle, _job, _expected: False,
    )
    _parsed, errors = reporting._evaluation_result(bundle, job, expected_episodes=2)
    assert any("queue runner's strict" in error for error in errors)


def test_design_validation_enforces_label_specific_training_resources():
    config = reporting.matrix_runner.validate_config(
        ROOT / "configs" / "experiments" / "crazyflie_integration.json"
    )
    acceptance = reporting.matrix_runner.memory_acceptance_contract_payload()
    queue = {
        "schema_version": 1,
        "label": "integration",
        "config": config,
        "config_sha256": reporting.matrix_runner.canonical_sha256(config),
        "job_count": 4,
        "evaluation_bundle_count": 12,
        "predicted_evaluation_episodes": 24,
        "sequential_process_limit": 1,
        "resource_limits": {
            "policy_version": acceptance["policy_version"],
            "device_gpu_used_mib_exclusive": acceptance[
                "device_gpu_used_mib_exclusive"
            ],
            "system_ram_percent_exclusive": acceptance[
                "system_ram_percent_exclusive"
            ],
            "rss_growth_tolerance_mib": acceptance["rss_growth_tolerance_mib"],
            "rss_growth_disposition": "warning_only",
            "swap_out_growth_tolerance_mib": acceptance[
                "swap_out_growth_tolerance_mib"
            ],
            "sustained_paging_disposition": "hard_failure",
            "gpu_telemetry_required_for_cuda": True,
            "finite_numeric_telemetry_required": True,
        },
    }
    *_, errors = reporting._validate_design(queue)
    assert errors == []

    config["training"]["num_envs"] = 2
    queue["config_sha256"] = reporting.matrix_runner.canonical_sha256(config)
    *_, errors = reporting._validate_design(queue)
    assert any("training.num_envs must be 1" in error for error in errors)


def test_markdown_visibly_surfaces_memory_warning_count_and_messages():
    report = {
        "label": "integration",
        "status": "complete",
        "queue_status": "completed",
        "completion": {
            "validated_completed_jobs": 1,
            "planned_jobs": 1,
            "planned_evaluation_episodes_fixed_denominator": 6,
            "validated_evaluation_episodes": 6,
        },
        "memory_warnings": {
            "policy_version": "crazyflie_memory_acceptance_v2",
            "count": 1,
            "messages": ["process RSS rose monotonically by 3.0 MiB"],
            "records": [{
                "job_id": "frozen_lif_original__seed-0",
                "scope": "training",
                "scenario": None,
                "message": "process RSS rose monotonically by 3.0 MiB",
            }],
        },
        "execution_status_table": [],
        "seed_table": [],
        "checkpoint_scenario_table": [],
        "episode_table": [],
        "controller_comparison": {
            "by_scenario": {},
            "paired_differences_vs_original": {},
        },
        "validation_errors": [],
        "interpretation": "unit",
    }
    rendered = reporting.markdown(report)
    assert "Memory warnings (warning-only)" in rendered
    assert "Count: **1**" in rendered
    assert "process RSS rose monotonically by 3.0 MiB" in rendered


def test_summary_rejects_queue_with_stale_current_fingerprint(tmp_path, monkeypatch):
    config = json.loads(
        (ROOT / "configs" / "experiments" / "crazyflie_integration.json").read_text(
            encoding="utf-8"
        )
    )
    queue_path = tmp_path / "queue.json"
    queue = {
        "schema_version": 1,
        "label": "integration",
        "config": config,
        "config_path": str(ROOT / "configs" / "experiments" / "crazyflie_integration.json"),
        "config_sha256": reporting.matrix_runner.canonical_sha256(config),
        "output": str(queue_path),
        "job_count": 4,
        "evaluation_bundle_count": 12,
        "predicted_evaluation_episodes": 24,
        "sequential_process_limit": 1,
        "jobs": [],
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    monkeypatch.setattr(reporting.matrix_runner, "validate_config", lambda _path: config)

    def reject_stale(_queue, _config, _path):
        raise ValueError("stale reproduction fingerprint")

    monkeypatch.setattr(reporting.matrix_runner, "validate_resume_queue", reject_stale)
    report = reporting.summarize(queue_path, bootstrap_samples=100)
    assert report["status"] == "incomplete"
    assert report["fingerprint_scope"] == "per_job"
    assert report["resolved_config"] == config
    assert report["config_sha256"] == queue["config_sha256"]
    assert any(
        "incompatible with current source/config/runtime fingerprint" in error
        for error in report["validation_errors"]
    )
