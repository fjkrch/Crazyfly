"""Focused CPU-only tests for neural activity and six-controller reporting."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_neural_activity as activity  # noqa: E402
import summarize_neural_comparison as reporting  # noqa: E402


def _design_queue(tmp_path: Path) -> dict:
    jobs = []
    current_sources = reporting.source_hashes()
    for controller in reporting.CONTROLLERS:
        for task in reporting.TASKS:
            identifier = (
                f"{controller}__"
                f"{task.removeprefix('FlyCrazyflie-').removesuffix('-v0').lower()}__seed-0"
            )
            run_dir = tmp_path / "jobs" / identifier
            payload = {
                "schema_version": 1,
                "cell": identifier,
                "source_sha256": current_sources,
                "resolved_config": {
                    "task": task,
                    "controller": controller,
                    "seed": 0,
                    "total_interactions": 500_000,
                    "num_envs": 40,
                    "horizon": 100,
                    "microbatch_size": 40,
                    "ppo_epochs": 2,
                    "learning_rate": 0.0003,
                    "evaluation_protocol": "main",
                    "contract_profile": "balanced_v4",
                },
            }
            jobs.append(
                {
                    "id": identifier,
                    "controller": controller,
                    "task": task,
                    "seed": 0,
                    "status": "pending",
                    "total_interactions": 500_000,
                    "run_dir": str(run_dir),
                    "checkpoint": str(run_dir / "checkpoints" / "latest.pt"),
                    "training_manifest": str(run_dir / "training_manifest.json"),
                    "evaluation_output": str(tmp_path / "evaluation" / f"{identifier}.json"),
                    "neural_activity_output": str(
                        tmp_path / "activity" / f"{identifier}.json"
                    ),
                    "expected_fingerprint": reporting.canonical_sha256(payload),
                    "fingerprint_payload": payload,
                }
            )
    return {
        "schema_version": 1,
        "status": "pending",
        "config_path": str(reporting.CONFIG_PATH),
        "job_count": 18,
        "predicted_training_interactions": 9_000_000,
        "predicted_evaluation_episodes": 288,
        "jobs": jobs,
    }


def test_six_controller_contract_and_budget_are_exact() -> None:
    assert reporting.CONTROLLERS == (
        "frozen_lif_original",
        "frozen_lif_degree_rewired",
        "wing_lif",
        "leg_wing_lif",
        "gru_matched",
        "mlp_normal",
    )
    assert reporting.EXPECTED_JOB_COUNT == 18
    assert reporting.EXPECTED_TOTAL_INTERACTIONS == 9_000_000
    assert reporting.EXPECTED_EVALUATION_EPISODES == 288


def test_design_validator_accepts_exact_shape_and_rejects_missing_cell(tmp_path: Path) -> None:
    queue = _design_queue(tmp_path)
    result = reporting.validate_design(queue, queue_path=tmp_path / "queue.json")
    assert result["config"]["total_interactions_per_job"] == 500_000

    queue["jobs"].pop()
    with pytest.raises(ValueError, match="exactly 18 jobs"):
        reporting.validate_design(queue, queue_path=tmp_path / "queue.json")


def test_incomplete_mode_has_no_winner_or_partial_rank(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(_design_queue(tmp_path)), encoding="utf-8")
    report = reporting.build_report(queue_path, allow_incomplete=True)
    assert report["status"] == "incomplete_or_invalid_no_winner"
    assert report["winner"] is None
    assert report["completion"]["valid_cells"] == 0
    assert all(row["rank"] is None for row in report["overall_controller_table"])

    with pytest.raises(ValueError, match="Comparison is incomplete/invalid"):
        reporting.build_report(queue_path, allow_incomplete=False)


def test_lif_role_rates_dead_saturated_and_top_ids_recompute() -> None:
    manifest = ROOT / "data" / "connectome_wing" / "manifest.json"
    neurons = json.loads((manifest.parent / "neurons.json").read_text(encoding="utf-8"))
    roles = [row["annotations"]["model_role"] for row in neurons]
    counts = [0] * 256
    sensory = roles.index("wing_sensory_input")
    motor = roles.index("wing_motor_output")
    intrinsic = roles.index("vnc_interneuron")
    counts[sensory] = 100
    counts[motor] = 50
    counts[intrinsic] = 25
    result = activity.summarize_lif_counts(
        counts,
        active_control_decisions=100,
        manifest_path=manifest,
    )
    assert result["sampled_spike_count"] == 175
    assert result["saturated_neuron_count"] == 1
    assert result["dead_neuron_count"] == 253
    assert result["roles"]["wing_sensory_input"]["saturated_neuron_count"] == 1
    assert result["roles"]["wing_motor_output"]["sampled_spike_count"] == 50
    assert result["roles"]["vnc_interneuron"]["sampled_spike_count"] == 25
    assert result["top_neurons"][0]["index"] == sensory
    assert result["top_neurons"][0]["sampled_spike_rate_hz"] == pytest.approx(50.0)
    assert len(result["per_neuron"]) == 256


def test_lif_summary_rejects_impossible_spike_count() -> None:
    with pytest.raises(ValueError, match="outside their denominator"):
        activity.summarize_lif_counts(
            [101] + [0] * 255,
            active_control_decisions=100,
            manifest_path=ROOT / "data" / "connectome" / "manifest.json",
        )


def test_engineering_units_keep_nonbiological_semantics_and_raw_accumulators() -> None:
    result = activity.summarize_engineering_units(
        [2.0, 4.0],
        [1.0, 9.0],
        [5, 10],
        observation_count=10,
        layer="gru_hidden",
    )
    assert result["kind"] == "engineering_units_not_biological_neurons"
    assert "not spikes or hertz" in result["activation_definition"]
    assert result["top_units"][0]["index"] == 1
    assert result["per_unit"][1]["absolute_activation_sum"] == 4.0
    assert result["per_unit"][1]["active_sample_count"] == 10


def test_activity_payload_recomputes_from_full_counts() -> None:
    core = activity.summarize_lif_counts(
        [0] * 256,
        active_control_decisions=10,
        manifest_path=ROOT / "data" / "connectome" / "manifest.json",
    )
    payload = {
        "active_control_decisions": 10,
        "cores": {"leg": core},
        "engineering_layers": {},
    }
    assert reporting.canonical_sha256(reporting._recompute_activity_payload(payload)) == (
        reporting.canonical_sha256(payload)
    )

    payload["cores"]["leg"]["dead_neuron_count"] = 0
    assert reporting.canonical_sha256(reporting._recompute_activity_payload(payload)) != (
        reporting.canonical_sha256(payload)
    )


def test_curve_plot_marks_incomplete_and_is_png(tmp_path: Path) -> None:
    report = {
        "status": "incomplete_or_invalid_no_winner",
        "controller_task_table": [
            {
                "controller": controller,
                "task": task,
                "training_evidence": {"training_curve": None},
            }
            for controller in reporting.CONTROLLERS
            for task in reporting.TASKS
        ],
    }
    output = tmp_path / "curves.png"
    reporting.plot_training_curves(report, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
