"""Focused gates for the additive 14-cell command report v2."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_report as report_module  # noqa: E402


CONFIG_SOURCE = (
    ROOT / "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json"
)


def _canonical(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _controller_report(controller: str) -> dict:
    lif = controller in report_module.V2_LIF_CONTROLLERS
    return {
        "controller": report_module.V2_POLICY_BY_CONTROLLER[controller],
        "actor_trainable_parameters": 4776 if controller != "leg_wing_lif" else 9224,
        "total_trainable_parameters": 23081 if controller != "leg_wing_lif" else 27529,
        "frozen_parameters": 2048 if lif else 0,
        "parameter_matching_required": controller != "leg_wing_lif",
        "actor_parameter_match_passed": controller != "leg_wing_lif",
        **({"core_checksum": sha256(controller.encode()).hexdigest()} if lif else {}),
    }


@pytest.fixture()
def v2_queue(tmp_path, monkeypatch):
    sandbox = tmp_path
    monkeypatch.setattr(report_module, "ROOT", sandbox)
    config_path = sandbox / "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG_SOURCE.read_bytes())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = sandbox / config["output_root"]
    queue_path = output_root / "queue.json"
    jobs = []
    for controller in report_module.V2_CONTROLLERS:
        for task in report_module.V2_TASKS:
            index = len(jobs) + 1
            condition = "still" if task == report_module.V2_TASKS[0] else "wind"
            identifier = f"{index:02d}__{controller}__{condition}__seed-0"
            run_dir = output_root / "jobs" / identifier
            eval_protocol = {
                "version": "unit-test-v2",
                "task": task,
                "wind_enabled": condition == "wind",
            }
            evaluation_manifest = {
                "manifest_id": sha256(f"manifest-{identifier}".encode()).hexdigest(),
                "evaluation_protocol": eval_protocol,
                "evaluation_protocol_sha256": _canonical(eval_protocol),
            }
            jobs.append(
                {
                    "id": identifier,
                    "controller": controller,
                    "policy": report_module.V2_POLICY_BY_CONTROLLER[controller],
                    "architecture_class": (
                        "lif" if controller in report_module.V2_LIF_CONTROLLERS else "baseline"
                    ),
                    "task": task,
                    "evaluation_protocol": "command_v2",
                    "seed": 0,
                    "command_schedule_seed": 0,
                    "paired_task_seed_key": f"{controller}__seed-0",
                    "contract_profile": "command_v2",
                    "status": "pending",
                    "training_status": "pending",
                    "evaluation_status": "pending",
                    "total_interactions": 1_000_000,
                    "expected_updates": 250,
                    "run_dir": str(run_dir),
                    "checkpoint": str(run_dir / "checkpoints/latest.pt"),
                    "training_manifest": str(run_dir / "training_manifest.json"),
                    "evaluation_output": str(
                        output_root / "evaluations" / identifier / "heldout.json"
                    ),
                    "expected_fingerprint": sha256(f"fingerprint-{identifier}".encode()).hexdigest(),
                    "fingerprint_payload": {"job": identifier},
                    "evaluation_manifest": evaluation_manifest,
                    "evaluation_manifest_id": evaluation_manifest["manifest_id"],
                    "command_training_contract_sha256": sha256(
                        f"contract-{task}".encode()
                    ).hexdigest(),
                    "attempts": [],
                }
            )
    queue_runner = ROOT / "scripts/crazyflie_command_queue_v2.py"
    trainer = ROOT / "scripts/drone_train.py"
    evaluator = ROOT / "scripts/crazyflie_command_evaluate.py"
    queue = {
        "schema_version": 2,
        "kind": report_module.V2_QUEUE_KIND,
        "status": "dry_run",
        "dry_run": True,
        "config_path": str(config_path),
        "config_file_sha256": report_module._sha256_file(config_path),
        "config_identity_sha256": _canonical(config),
        "config": config,
        "queue_runner": str(queue_runner),
        "queue_runner_sha256": report_module._sha256_file(queue_runner),
        "trainer": str(trainer),
        "trainer_sha256": report_module._sha256_file(trainer),
        "evaluator": str(evaluator),
        "evaluator_sha256": report_module._sha256_file(evaluator),
        "output_root": str(output_root),
        "queue_file": str(queue_path),
        "job_count": 14,
        "predicted_training_interactions": 14_000_000,
        "predicted_evaluation_episodes": 224,
        "controller_order": list(report_module.V2_CONTROLLERS),
        "task_order": list(report_module.V2_TASKS),
        "lif_first": True,
        "maximum_parallel": 1,
        "controller_reports": {
            controller: _controller_report(controller)
            for controller in report_module.V2_CONTROLLERS
        },
        "counts": {"pending": 14},
        "jobs": jobs,
    }
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path, output_root, queue_path, queue


def _memory_gate():
    return {
        "passed": True,
        "max_device_gpu_used_mib": 1000.0,
        "max_system_ram_percent": 40.0,
    }


def _write_complete_cell(job: dict, controller_report: dict) -> None:
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"authenticated-v2-unit-test-checkpoint")
    checkpoint_sha = report_module._sha256_file(checkpoint)
    rows = [
        {
            "completed_updates": index,
            "total_interactions": index * 4000,
            "mean_rollout_reward": index / 1000.0,
            "loss": 1.0 / index,
            "policy_loss": -1.0 / (index + 1),
            "value_loss": 0.5 / index,
        }
        for index in range(1, 251)
    ]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    history_dir = checkpoint.parent.parent / "history"
    history_dir.mkdir(parents=True)
    history_path = history_dir / "rows-00000001-00000250-resume-0000-test.jsonl"
    history_path.write_bytes(payload)
    history_reference = {
        "schema_version": 1,
        "storage": "immutable_jsonl_segments",
        "row_count": 250,
        "history_sha256": _canonical(rows),
        "last_completed_updates": 250,
        "last_total_interactions": 1_000_000,
        "segments": [
            {
                "path": "../history/" + history_path.name,
                "sha256": sha256(payload).hexdigest(),
                "byte_count": len(payload),
                "row_count": 250,
            }
        ],
    }
    core = controller_report.get("core_checksum")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": job["task"],
        "contract_profile": "command_v2",
        "controller": job["policy"],
        "seed": 0,
        "num_envs": 40,
        "horizon": 100,
        "requested_interactions": 1_000_000,
        "environment_interactions": 1_000_000,
        "completed_updates": 250,
        "fingerprint": job["expected_fingerprint"],
        "fingerprint_payload": job["fingerprint_payload"],
        "evaluation_manifest_id": job["evaluation_manifest_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config": {
            "task": job["task"],
            "contract_profile": "command_v2",
            "controller": job["policy"],
            "seed": 0,
            "total_interactions": 1_000_000,
            "evaluation_protocol": "command_v2",
            "command_training_contract_sha256": job["command_training_contract_sha256"],
        },
        "controller_report": controller_report,
        "memory_gate": _memory_gate(),
        "command_schedule": {
            "command_training_contract_sha256": job["command_training_contract_sha256"],
            "state": {"training_interactions": 1_000_000},
        },
        "history_reference": history_reference,
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
    scores = {
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
    weights = {
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
            "command_response": scores["response"],
            "flight_stability": 100.0
            * math.exp(
                -((quality_raw["projected_gravity_xy_rms"] / 0.25) ** 2)
                - ((quality_raw["angular_velocity_rms_rad_s"] / 1.5) ** 2)
            ),
            "survival_not_die": 100.0,
        },
        "raw": quality_raw,
    }
    reward_components = {name: 0.1 for name in report_module.REWARD_COMPONENTS}
    per_unit = [
        {
            "index": 0,
            "id": "L-optic-001",
            "role": "optic",
            "absolute_activity_sum": 2880.0,
            "mean_absolute_activity": 0.3,
            "rms_activity": 0.4,
            "active_count": 2880,
            "active_fraction": 0.3,
        },
        {
            "index": 1,
            "id": "motor-001",
            "role": "motor",
            "absolute_activity_sum": 1920.0,
            "mean_absolute_activity": 0.2,
            "rms_activity": 0.3,
            "active_count": 1920,
            "active_fraction": 0.2,
        },
    ]
    activity = {
        "source": "exact_forward_pass_that_produced_each_evaluated_action",
        "controller": job["policy"],
        "kind": "sampled_lif_spikes",
        "unit_count": 2,
        "overall": {
            "sample_count": 9600,
            "mean_absolute_activity_per_unit": 0.25,
            "rms_activity_per_unit": 0.35,
            "active_fraction_per_unit": 0.25,
            "roles": {
                "optic": {
                    "unit_count": 1,
                    "mean_absolute_activity_per_unit": 0.3,
                    "active_fraction_per_unit": 0.3,
                },
                "motor": {
                    "unit_count": 1,
                    "mean_absolute_activity_per_unit": 0.2,
                    "active_fraction_per_unit": 0.2,
                },
            },
        },
        "per_unit": per_unit,
        "top_units": list(per_unit),
    }
    wind_condition = job["task"] == report_module.V2_TASKS[1]
    physical_wind = {
        "source": "terminal actual interval unit test",
        "frame": "world",
        "application_point": "body_center_of_mass",
        "condition": "wind" if wind_condition else "still_air",
        "samples": {
            "planned_intervals": 9600,
            "observed_intervals_through_first_done": 9600,
            "planned_pulse_intervals": 2000 if wind_condition else 0,
            "observed_planned_pulse_intervals": 2000 if wind_condition else 0,
            "observed_nonzero_wrench_intervals": 2000 if wind_condition else 0,
        },
        "measured": {
            "maximum_force_norm_n_all_simulated_intervals": 0.1 if wind_condition else 0.0,
            "maximum_torque_norm_nm_all_simulated_intervals": 0.001 if wind_condition else 0.0,
            "force_rms_n_observed": 0.02 if wind_condition else 0.0,
            "torque_rms_nm_observed": 0.0002 if wind_condition else 0.0,
        },
        "integrity": {
            "all_values_finite": True,
            "category_codes_valid_all_simulated_intervals": True,
            "force_within_declared_bound": True,
            "torque_within_declared_bound": True,
            "expected_force_matches_on_observed_intervals": True,
            "expected_torque_matches_on_observed_intervals": True,
            "expected_category_matches_on_observed_intervals": True,
            "pulse_window_matches_on_observed_intervals": True,
            "still_air_exact_zero_all_simulated_intervals": (
                None if wind_condition else True
            ),
            "complete_600_step_protocol_observed_for_all_episodes": True,
            "wind_nonzero_wrench_observed": True if wind_condition else None,
            "passed": True,
        },
    }
    evaluation = {
        "schema_version": 1,
        "analysis_kind": report_module.V2_ANALYSIS_KIND,
        "status": "PASS",
        "task": job["task"],
        "controller": job["policy"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "training_seed": 0,
            "total_interactions": 1_000_000,
            "reproduction_fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
            "evaluation_manifest": job["evaluation_manifest"],
        },
        "protocol": job["evaluation_manifest"]["evaluation_protocol"],
        "protocol_sha256": job["evaluation_manifest"]["evaluation_protocol_sha256"],
        "episodes_requested": 16,
        "episodes_evaluated": 16,
        "steps_per_episode": 600,
        "deterministic_actions": True,
        "policy_action_source": "actual_trained_controller_no_assist",
        "summary": {
            "score": {
                "score": sum(scores[name] * weight for name, weight in weights.items()),
                "component_scores": scores,
                "component_weights": weights,
                "raw": raw,
            },
            "control_quality": quality,
            "reward_component_mean_per_episode": reward_components,
            "physical_wind": physical_wind,
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
            "physical_wind_telemetry_passed": True,
            "physical_wind_uses_terminal_actual_interval": True,
        },
    }
    evaluation_path = Path(job["evaluation_output"])
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")


def test_pending_v2_matrix_is_explicit_na(v2_queue):
    config_path, _root, queue_path, _queue = v2_queue
    data = report_module.collect_v2_report_data(config_path, queue_path)
    assert not data.complete
    assert len(data.cells) == 14
    assert all(not cell.complete for cell in data.cells)
    markdown = report_module.render_v2_markdown(
        data,
        {name: Path(filename) for name, filename in report_module.V2_PLOT_NAMES.items()},
    )
    assert "INCOMPLETE — 0/14 verified" in markdown
    assert "Wind effect (wind minus still)" in markdown
    assert "Most active LIF neurons" in markdown
    assert "N/A (incomplete pair)" in markdown


def test_one_authenticated_v2_lif_cell_reports_roles_and_top_neurons(v2_queue):
    config_path, _root, queue_path, queue = v2_queue
    first = queue["jobs"][0]
    first.update(status="completed", training_status="completed", evaluation_status="completed")
    _write_complete_cell(first, queue["controller_reports"][first["controller"]])
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = report_module.collect_v2_report_data(config_path, queue_path)
    assert [cell.complete for cell in data.cells].count(True) == 1
    assert len(data.cells[0].history) == 250
    markdown = report_module.render_v2_markdown(
        data,
        {name: Path(filename) for name, filename in report_module.V2_PLOT_NAMES.items()},
    )
    assert "INCOMPLETE — 1/14 verified" in markdown
    assert "L-optic-001" in markdown
    assert "optic" in markdown
    assert "Physical wind evidence" in markdown


def test_v2_top_neuron_tamper_rejects_numeric_cell(v2_queue):
    config_path, _root, queue_path, queue = v2_queue
    first = queue["jobs"][0]
    first.update(status="completed", training_status="completed", evaluation_status="completed")
    _write_complete_cell(first, queue["controller_reports"][first["controller"]])
    evaluation_path = Path(first["evaluation_output"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["activity"]["top_units"].reverse()
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = report_module.collect_v2_report_data(config_path, queue_path)
    assert data.cells[0].complete is False
    assert "top-unit rank" in data.cells[0].reason


def test_v2_complete_still_wind_pair_produces_authenticated_delta(v2_queue):
    config_path, _root, queue_path, queue = v2_queue
    for job in queue["jobs"][:2]:
        job.update(status="completed", training_status="completed", evaluation_status="completed")
        _write_complete_cell(job, queue["controller_reports"][job["controller"]])
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = report_module.collect_v2_report_data(config_path, queue_path)
    assert data.cells[0].complete and data.cells[1].complete
    assert report_module._v2_delta(
        data.cells[0], data.cells[1], lambda cell: report_module._score_value(cell, "total")
    ) == pytest.approx(0.0)
    markdown = report_module.render_v2_markdown(
        data,
        {name: Path(filename) for name, filename in report_module.V2_PLOT_NAMES.items()},
    )
    assert "| original_lif | paired | 0.000" in markdown
    assert '"condition":"wind"' in markdown


def test_v2_wind_without_observed_physical_push_is_rejected(v2_queue):
    config_path, _root, queue_path, queue = v2_queue
    wind = queue["jobs"][1]
    wind.update(status="completed", training_status="completed", evaluation_status="completed")
    _write_complete_cell(wind, queue["controller_reports"][wind["controller"]])
    evaluation_path = Path(wind["evaluation_output"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["summary"]["physical_wind"]["integrity"][
        "wind_nonzero_wrench_observed"
    ] = False
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = report_module.collect_v2_report_data(config_path, queue_path)
    assert data.cells[1].complete is False
    assert "did not observe a nonzero physical wrench" in data.cells[1].reason


def test_v2_incomplete_plots_are_png(v2_queue, tmp_path):
    config_path, _root, queue_path, _queue = v2_queue
    data = report_module.collect_v2_report_data(config_path, queue_path)
    destinations = {
        name: tmp_path / filename for name, filename in report_module.V2_PLOT_NAMES.items()
    }
    report_module.render_v2_plots(data, destinations)
    for path in destinations.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
