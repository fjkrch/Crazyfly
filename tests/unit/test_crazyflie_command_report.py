"""Focused gates for the command-only comparison report and plots."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_queue as queue_module  # noqa: E402
import crazyflie_command_report as report_module  # noqa: E402


CONFIG_SOURCE = ROOT / "configs/experiments/crazyflie_command_seed0_500k.json"


def _canonical(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def report_queue(tmp_path_factory):
    sandbox = tmp_path_factory.mktemp("command-report")
    config_path = sandbox / "configs/experiments/crazyflie_command_seed0_500k.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG_SOURCE.read_bytes())
    config = queue_module.validate_config(config_path)
    output_root = sandbox / "runs/crazyflie_command_seed0_500k"
    queue = queue_module.build_queue(config, output_root)
    queue_path = output_root / "queue.json"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sandbox, config_path, output_root, queue_path, queue


def _memory_gate():
    return {
        "passed": True,
        "max_device_gpu_used_mib": 1000.0,
        "max_system_ram_percent": 40.0,
    }


def _history_rows():
    return [
        {
            "completed_updates": index,
            "total_interactions": index * 4000,
            "mean_rollout_reward": index / 1000.0,
            "loss": 1.0 / index,
            "policy_loss": -1.0 / (index + 1),
            "value_loss": 0.5 / index,
        }
        for index in range(1, 126)
    ]


def _write_completed_cell(job: dict, controller_report: dict) -> None:
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"authenticated-unit-test-checkpoint")
    checkpoint_sha = report_module._sha256_file(checkpoint)
    history_dir = checkpoint.parent.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    rows = _history_rows()
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    history_path = history_dir / "rows-00000001-00000125-resume-0000-test.jsonl"
    history_path.write_bytes(payload)
    reference = {
        "schema_version": 1,
        "storage": "immutable_jsonl_segments",
        "row_count": 125,
        "history_sha256": _canonical(rows),
        "last_completed_updates": 125,
        "last_total_interactions": 500000,
        "segments": [
            {
                "path": "../history/" + history_path.name,
                "sha256": sha256(payload).hexdigest(),
                "byte_count": len(payload),
                "row_count": 125,
                "first_completed_updates": 1,
                "first_total_interactions": 4000,
                "last_completed_updates": 125,
                "last_total_interactions": 500000,
            }
        ],
    }
    core = controller_report["core_checksum"]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": report_module.TASK,
        "contract_profile": "command_v1",
        "controller": job["controller"],
        "seed": 0,
        "num_envs": 40,
        "horizon": 100,
        "requested_interactions": 500000,
        "environment_interactions": 500000,
        "completed_updates": 125,
        "fingerprint": job["expected_fingerprint"],
        "fingerprint_payload": job["fingerprint_payload"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config": {
            "task": report_module.TASK,
            "contract_profile": "command_v1",
            "controller": job["controller"],
            "seed": 0,
            "total_interactions": 500000,
            "evaluation_protocol": "command_v1",
            "command_training_contract_sha256": job["command_training_contract_sha256"],
        },
        "controller_report": controller_report,
        "memory_gate": _memory_gate(),
        "command_schedule": {
            "command_training_contract_sha256": job["command_training_contract_sha256"],
            "state": {"training_interactions": 500000},
        },
        "history_reference": reference,
        "core_checksum_before": core,
        "core_checksum_after": core,
    }
    training_path = Path(job["training_manifest"])
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    raw = {
        "steps": 600,
        "episodes": 16,
        "linear_tracking_rmse_m_s": 0.1,
        "yaw_tracking_rmse_rad_s": 0.2,
        "wrong_direction_fraction": 0.01,
        "mean_command_projection_ratio": 0.9,
        "response_latency_mean_s": 0.2,
        "overshoot_ratio_mean": 0.1,
        "brake_settling_mean_s": 0.3,
        "hover_speed_rms_m_s": 0.04,
        "hover_drift_mean_m": 0.05,
        "survival_fraction": 1.0,
        "invalid_state_count": 0,
        "action_effort_rms": 0.1,
        "action_delta_rms": 0.02,
    }
    import math

    score_components = {
        "linear_tracking": 100.0 * math.exp(-((raw["linear_tracking_rmse_m_s"] / 0.35) ** 2)),
        "yaw_tracking": 100.0 * math.exp(-((raw["yaw_tracking_rmse_rad_s"] / 0.50) ** 2)),
        "direction": 100.0 * (1.0 - raw["wrong_direction_fraction"]),
        "response": 100.0 * math.exp(-(raw["response_latency_mean_s"] / 0.80)),
        "braking": 100.0 * math.exp(-(raw["brake_settling_mean_s"] / 1.00)),
        "hover": 100.0
        * math.exp(
            -((raw["hover_speed_rms_m_s"] / 0.15) ** 2)
            - ((raw["hover_drift_mean_m"] / 0.15) ** 2)
        ),
        "safety": 100.0,
        "effort": 100.0 * math.exp(-((raw["action_effort_rms"] / 0.25) ** 2)),
        "smoothness": 100.0 * math.exp(-((raw["action_delta_rms"] / 0.08) ** 2)),
    }
    score_weights = {
        "linear_tracking": 0.30,
        "yaw_tracking": 0.10,
        "direction": 0.10,
        "response": 0.10,
        "braking": 0.10,
        "hover": 0.10,
        "safety": 0.15,
        "effort": 0.025,
        "smoothness": 0.025,
    }
    total_score = sum(score_components[name] * weight for name, weight in score_weights.items())
    quality_raw = {
        "acceleration_rms_m_s2": 0.4,
        "jerk_rms_m_s3": 1.0,
        "wrong_direction_acceleration_fraction": 0.01,
        "projected_gravity_xy_rms": 0.03,
        "angular_velocity_rms_rad_s": 0.2,
        "survival_fraction": 1.0,
        "invalid_state_count": 0,
    }
    quality = {
        "component_scores_0_100": {
            "acceleration_quality": 100.0
            * math.exp(-((quality_raw["jerk_rms_m_s3"] / 80.0) ** 2))
            * (1.0 - quality_raw["wrong_direction_acceleration_fraction"]),
            "command_response": score_components["response"],
            "flight_stability": 100.0
            * math.exp(
                -((quality_raw["projected_gravity_xy_rms"] / 0.25) ** 2)
                - ((quality_raw["angular_velocity_rms_rad_s"] / 1.5) ** 2)
            ),
            "survival_not_die": 100.0,
        },
        "raw": quality_raw,
        "interpretation": "unit test",
    }
    reward_components = {name: 0.1 for name in report_module.REWARD_COMPONENTS}
    activity = {
        "source": "exact_forward_pass_that_produced_each_evaluated_action",
        "controller": job["controller"],
        "kind": "sampled_lif_spikes",
        "unit_count": 2,
        "role_counts": {"leg": 1, "motor": 1},
        "role_provenance": "unit test",
        "overall": {
            "sample_count": 9600,
            "mean_absolute_activity_per_unit": 0.2,
            "rms_activity_per_unit": 0.3,
            "active_fraction_per_unit": 0.4,
            "roles": {
                "leg": {
                    "unit_count": 1,
                    "mean_absolute_activity_per_unit": 0.2,
                    "active_fraction_per_unit": 0.4,
                },
                "motor": {
                    "unit_count": 1,
                    "mean_absolute_activity_per_unit": 0.3,
                    "active_fraction_per_unit": 0.5,
                },
            },
        },
        "segments": {},
        "per_unit": [],
        "top_units": [],
    }
    evaluation = {
        "schema_version": 1,
        "analysis_kind": report_module.ANALYSIS_KIND,
        "status": "PASS",
        "task": report_module.TASK,
        "controller": job["controller"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "training_seed": 0,
            "total_interactions": 500000,
            "reproduction_fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
            "evaluation_manifest": job["evaluation_manifest"],
        },
        "protocol": job["evaluation_manifest"]["evaluation_protocol"],
        "protocol_sha256": job["evaluation_manifest"]["evaluation_protocol_sha256"],
        "episodes_requested": 16,
        "episodes_evaluated": 16,
        "steps_per_episode": 600,
        "vectorized_environment_count": 16,
        "deterministic_actions": True,
        "policy_action_source": "actual_trained_controller_no_assist",
        "summary": {
            "score": {
                "score": total_score,
                "component_scores": score_components,
                "component_weights": score_weights,
                "raw": raw,
            },
            "control_quality": quality,
            "reward_component_mean_per_episode": reward_components,
        },
        "episodes": [
            {"episode_index": index, "reward_components": reward_components}
            for index in range(16)
        ],
        "activity": activity,
        "controller_report": controller_report,
        "memory_gate": _memory_gate(),
        "integrity": {
            "task_manifest_matched": True,
            "evaluation_manifest_matched": True,
            "source_set_matched": True,
            "checkpoint_completed_budget": True,
            "all_actions_finite_and_bounded": True,
            "activity_from_actual_forward": True,
        },
    }
    evaluation_path = Path(job["evaluation_output"])
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")


def test_pending_jobs_are_explicit_na_and_never_complete(report_queue, monkeypatch):
    sandbox, config_path, _output_root, queue_path, _queue = report_queue
    monkeypatch.setattr(report_module, "ROOT", sandbox)
    data = report_module.collect_report_data(config_path, queue_path)
    assert not data.complete
    assert all(not cell.complete for cell in data.cells)
    markdown = report_module.render_markdown(
        data, {name: Path(filename) for name, filename in report_module.PLOT_NAMES.items()}
    )
    assert "INCOMPLETE — 0/6 verified" in markdown
    assert markdown.count("N/A (") >= 6
    assert "all six jobs independently verified" not in markdown


def test_one_authenticated_complete_cell_and_five_na(report_queue, monkeypatch):
    sandbox, config_path, _output_root, queue_path, original_queue = report_queue
    monkeypatch.setattr(report_module, "ROOT", sandbox)
    queue = json.loads(json.dumps(original_queue))
    first = queue["jobs"][0]
    first.update(status="completed", training_status="completed", evaluation_status="completed")
    _write_completed_cell(first, queue["controller_reports"][first["controller"]])
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = report_module.collect_report_data(config_path, queue_path)
    assert [cell.complete for cell in data.cells] == [True, False, False, False, False, False]
    assert 0.0 <= data.cells[0].score["score"] <= 100.0
    assert len(data.cells[0].history) == 125
    markdown = report_module.render_markdown(
        data, {name: Path(filename) for name, filename in report_module.PLOT_NAMES.items()}
    )
    assert "INCOMPLETE — 1/6 verified" in markdown
    assert f"{data.cells[0].score['score']:.3f}" in markdown
    assert "Acceleration /100" in markdown
    assert "Frozen-core integrity" in markdown


def test_checkpoint_tamper_turns_completed_cell_into_na(report_queue, monkeypatch):
    sandbox, config_path, _output_root, queue_path, _queue = report_queue
    monkeypatch.setattr(report_module, "ROOT", sandbox)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    checkpoint = Path(queue["jobs"][0]["checkpoint"])
    checkpoint.write_bytes(b"tampered")
    data = report_module.collect_report_data(config_path, queue_path)
    assert data.cells[0].complete is False
    assert "checkpoint_sha256" in data.cells[0].reason or "checkpoint" in data.cells[0].reason


def test_plots_render_for_incomplete_data_and_are_png(report_queue, monkeypatch, tmp_path):
    sandbox, config_path, _output_root, queue_path, _queue = report_queue
    monkeypatch.setattr(report_module, "ROOT", sandbox)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    _write_completed_cell(
        queue["jobs"][0], queue["controller_reports"][queue["jobs"][0]["controller"]]
    )
    data = report_module.collect_report_data(config_path, queue_path)
    assert data.cells[0].complete is True
    destinations = {name: tmp_path / filename for name, filename in report_module.PLOT_NAMES.items()}
    report_module.render_plots(data, destinations)
    for path in destinations.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_no_overwrite_publication(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report_module._publish_no_overwrite(source, destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_float32_episode_mean_tolerance_is_narrow():
    # Real evaluator summaries use float32 reductions, while the reporter
    # recomputes from JSON episode rows with Python float64.
    assert report_module._float32_episode_mean_matches(20.0, 20.0 + 1.7e-6)
    assert not report_module._float32_episode_mean_matches(20.0, 20.0 + 1.0e-5)
