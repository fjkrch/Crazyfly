"""CPU-only gates for the review-only leg/wing extension queue."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_wing_run_matrix as wing_matrix  # noqa: E402


def test_wing_main_dry_run_is_complete_and_non_executable(tmp_path):
    config, base = wing_matrix.validate_wing_config(
        ROOT / "configs" / "experiments" / "crazyflie_wing_main.json"
    )
    queue = wing_matrix.build_queue(config, base, tmp_path / "wing-main.json")
    assert queue["status"] == "dry_run"
    assert queue["execution_authorized"] is False
    assert queue["sequential_process_limit"] == 1
    assert queue["job_count"] == 15
    assert queue["evaluation_bundle_count"] == 45
    assert queue["predicted_evaluation_episodes"] == 720
    assert queue["resource_limits"]["device_gpu_used_mib_exclusive"] == 6963.2
    assert queue["resource_limits"]["system_ram_percent_exclusive"] == 90.0
    assert queue["resource_limits"]["policy_version"] == "crazyflie_memory_acceptance_v2"
    assert queue["resource_limits"]["rss_growth_disposition"] == "warning_only"
    assert queue["memory_acceptance"]["policy_version"] == "crazyflie_memory_acceptance_v2"
    assert {
        (job["controller"], job["seed"]) for job in queue["jobs"]
    } == {(controller, seed) for controller in wing_matrix.CONTROLLERS for seed in wing_matrix.SEEDS}
    assert all(job["total_interactions"] == 5_000_000 for job in queue["jobs"])
    assert all(len(job["evaluations"]) == 3 for job in queue["jobs"])
    assert all(
        job["training_command"][job["training_command"].index("--evaluation_protocol") + 1]
        == "main"
        for job in queue["jobs"]
    )
    assert all(
        "--wing_connectome_manifest" in job["training_command"]
        for job in queue["jobs"]
    )
    assert queue["controller_reports"]["frozen_lif_original"]["actor_trainable_parameters"] == 4776
    assert queue["controller_reports"]["wing_lif"]["actor_trainable_parameters"] == 4776
    assert queue["controller_reports"]["leg_wing_lif"]["actor_trainable_parameters"] == 9224


def test_wing_matrix_cli_exposes_no_execute_mode():
    source = (ROOT / "scripts" / "drone_wing_run_matrix.py").read_text(encoding="utf-8")
    assert 'add_argument("--execute"' not in source
    assert "execution_authorized\": False" in source
