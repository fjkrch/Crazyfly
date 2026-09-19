"""CPU gates for the fail-closed six-cell combination reporter."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_combinations_queue_v3 as queue_module  # noqa: E402
import crazyflie_command_combinations_report as report  # noqa: E402


CONFIG = ROOT / "configs/experiments/crazyflie_command_combinations_seed0_1m.json"


def _canonical(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue_module.validate_config(CONFIG)
    output = tmp_path_factory.mktemp("v3-report") / "matrix"
    queue = queue_module.build_queue(config, output)
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return public, queue, output


def _completed_queue(value):
    # Production queue/evaluation writers canonicalize mapping keys.  Exercise
    # the persisted identity rather than relying on construction-time order.
    completed = json.loads(json.dumps(value, sort_keys=True))
    completed.update(status="completed", dry_run=False, counts={"completed": 6})
    for job in completed["jobs"]:
        job.update(status="completed", training_status="completed", evaluation_status="completed")
    return completed


def _memory_gate():
    return {
        "passed": True,
        "device_gpu_telemetry_complete": True,
        "sustained_paging_detected": False,
        "failures": [],
        "limits": {
            "gpu_used_mib_exclusive": 6963.2,
            "system_ram_percent_exclusive": 90.0,
        },
        "max_device_gpu_used_mib": 3200.0,
        "max_process_rss_mib": 4100.0,
        "max_system_ram_percent": 42.0,
        "max_torch_allocated_mib": 100.0,
        "max_torch_reserved_mib": 120.0,
    }


def _latency():
    return {
        "schema_version": 1,
        "source": "exact_policy_act_calls_used_by_evaluation_control_path",
        "call_site": "policy.act(normalized_observation, recurrent_state, deterministic=True)",
        "clock": "time.perf_counter_ns_monotonic",
        "device": "cuda:0",
        "measurement_unit": "milliseconds_per_vectorized_policy_call",
        "batch_size": 16,
        "cuda_synchronized_before_and_after_call": True,
        "activity_recorder_hooks_in_scope": True,
        "action_producing_call_count": 600,
        "warmup_calls_excluded_from_statistics": 10,
        "warmup_exclusion_justification": "unit-test real-call prefix",
        "sample_count": 590,
        "total_ms": 59.0,
        "mean_ms": 0.1,
        "p50_ms": 0.09,
        "p95_ms": 0.15,
        "p99_ms": 0.18,
        "max_ms": 0.2,
        "all_action_producing_calls_total_ms": 60.0,
    }


def _score(evaluation_manifest):
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
    components = {
        "linear_tracking": 100.0 * math.exp(-((raw["linear_tracking_rmse_m_s"] / 0.35) ** 2)),
        "yaw_tracking": 100.0 * math.exp(-((raw["yaw_tracking_rmse_rad_s"] / 0.50) ** 2)),
        "direction": 100.0 * (1.0 - raw["wrong_direction_fraction"]),
        "response": 100.0 * math.exp(-(raw["response_latency_mean_s"] / 0.80)),
        "braking": 100.0 * math.exp(-(raw["brake_settling_mean_s"] / 1.00)),
        "hover": 100.0 * math.exp(
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
    return {
        "score": sum(components[name] * weights[name] for name in weights),
        "component_scores": components,
        "component_weights": weights,
        "raw": raw,
        "protocol": evaluation_manifest["evaluation_protocol"],
        "protocol_sha256": evaluation_manifest["evaluation_protocol_sha256"],
    }


def _quality(score):
    raw = {
        "acceleration_rms_m_s2": 0.4,
        "jerk_rms_m_s3": 1.0,
        "wrong_direction_acceleration_fraction": 0.01,
        "projected_gravity_xy_rms": 0.03,
        "angular_velocity_rms_rad_s": 0.2,
        "survival_fraction": 1.0,
        "invalid_state_count": 0,
    }
    return {
        "component_scores_0_100": {
            "acceleration_quality": 100.0
            * math.exp(-((raw["jerk_rms_m_s3"] / 80.0) ** 2))
            * (1.0 - raw["wrong_direction_acceleration_fraction"]),
            "command_response": score["component_scores"]["response"],
            "flight_stability": 100.0
            * math.exp(
                -((raw["projected_gravity_xy_rms"] / 0.25) ** 2)
                - ((raw["angular_velocity_rms_rad_s"] / 1.5) ** 2)
            ),
            "survival_not_die": 100.0,
        },
        "raw": raw,
    }


def _activity(job):
    components = tuple(job["combination_components"])
    identities = job["fingerprint_payload"]["resolved_config"][
        "lif_connectome_composition"
    ]["connectomes"]
    provenance = {
        label: {
            "manifest": identities[label]["manifest"],
            "manifest_sha256": identities[label]["manifest_sha256"],
            "neurons": identities[label]["neurons_path"]["path"],
            "neurons_sha256": identities[label]["neurons_path"]["sha256"],
            "neuron_count": len(report.EXPECTED_ROLES[label]),
        }
        for label in components
    }
    per_unit = []
    for role in sorted(set().union(*(report.EXPECTED_ROLES[label] for label in components))):
        index = len(per_unit)
        core = role.split(":", 1)[0]
        mean = 0.05 + 0.01 * index
        per_unit.append({
            "index": index,
            "id": f"{core}:unit-{index:03d}",
            "role": role,
            "absolute_activity_sum": mean * 9600,
            "mean_absolute_activity": mean,
            "rms_activity": math.sqrt(mean),
            "active_count": int(mean * 9600),
            "active_fraction": mean,
        })
    roles = {
        unit["role"]: {
            "unit_count": 1,
            "mean_absolute_activity_per_unit": unit["mean_absolute_activity"],
            "active_fraction_per_unit": unit["active_fraction"],
        }
        for unit in per_unit
    }
    top = sorted(
        per_unit,
        key=lambda unit: (-unit["mean_absolute_activity"], unit["id"]),
    )[: min(10, len(per_unit))]
    return {
        "source": "exact_forward_pass_that_produced_each_evaluated_action",
        "controller": job["policy"],
        "kind": "sampled_lif_spikes",
        "unit_count": len(per_unit),
        "role_provenance": provenance,
        "overall": {
            "sample_count": 9600,
            "mean_absolute_activity_per_unit": sum(
                unit["mean_absolute_activity"] for unit in per_unit
            ) / len(per_unit),
            "rms_activity_per_unit": math.sqrt(sum(
                unit["rms_activity"] ** 2 for unit in per_unit
            ) / len(per_unit)),
            "active_fraction_per_unit": sum(
                unit["active_fraction"] for unit in per_unit
            ) / len(per_unit),
            "roles": roles,
        },
        "per_unit": per_unit,
        "top_units": [
            {
                "id": unit["id"],
                "role": unit["role"],
                "mean_absolute_activity": unit["mean_absolute_activity"],
                "rms_activity": unit["rms_activity"],
                "active_fraction": unit["active_fraction"],
            }
            for unit in top
        ],
    }


def _physical_wind(wind):
    return {
        "frame": "world",
        "condition": "wind" if wind else "still_air",
        "samples": {
            "planned_intervals": 9600,
            "observed_intervals_through_first_done": 9600,
        },
        "measured": {
            "maximum_force_norm_n_all_simulated_intervals": 0.1 if wind else 0.0,
            "maximum_torque_norm_nm_all_simulated_intervals": 0.001 if wind else 0.0,
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
            "still_air_exact_zero_all_simulated_intervals": True if not wind else None,
            "wind_nonzero_wrench_observed": True if wind else None,
            "passed": True,
        },
    }


def _write_complete_artifacts(job, controller_report):
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{job['id']}".encode())
    checkpoint_sha = report._sha256_file(checkpoint)
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
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "rows-00000001-00000250-resume-0000-test.jsonl"
    history_path.write_bytes(payload)
    history_reference = {
        "schema_version": 1,
        "storage": "immutable_jsonl_segments",
        "row_count": 250,
        "history_sha256": _canonical(rows),
        "last_completed_updates": 250,
        "last_total_interactions": 1_000_000,
        "segments": [{
            "path": "../history/" + history_path.name,
            "sha256": sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "row_count": 250,
        }],
    }
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
        "resolved_config": job["fingerprint_payload"]["resolved_config"],
        "controller_report": controller_report,
        "memory_gate": _memory_gate(),
        "training_wall_time_s": 60.0,
        "command_schedule": {
            "command_training_contract_sha256": job["command_training_contract_sha256"],
            "state": {"training_interactions": 1_000_000},
        },
        "history_reference": history_reference,
        "core_checksum_before": controller_report["core_checksum"],
        "core_checksum_after": controller_report["core_checksum"],
        "per_core_checksums_before": controller_report["per_core_checksums"],
        "per_core_checksums_after": controller_report["per_core_checksums"],
    }
    training_path = Path(job["training_manifest"])
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    score = _score(job["evaluation_manifest"])
    quality = _quality(score)
    rewards = {name: 0.1 for name in report.common.REWARD_COMPONENTS}
    episodes = [
        {
            "episode_index": index,
            "alive_steps": 600,
            "invalid_steps": 0,
            "terminated": False,
            "truncated": True,
            "failure_cause": 0,
            "reward_total": 0.1,
            "reward_components": rewards,
        }
        for index in range(16)
    ]
    evaluation = {
        "schema_version": 1,
        "analysis_kind": report.ANALYSIS_KIND,
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
            "score": score,
            "control_quality": quality,
            "reward_component_mean_per_episode": rewards,
            "reward_total_mean": 0.1,
            "survived_full_horizon_count": 16,
            "termination_count": 0,
            "truncation_count": 16,
            "invalid_episode_count": 0,
            "failure_cause_counts": {"0": 16},
            "physical_wind": _physical_wind(job["task"] == report.TASKS[1]),
        },
        "episodes": episodes,
        "activity": _activity(job),
        "inference_latency": _latency(),
        "controller_report": controller_report,
        "memory_gate": _memory_gate(),
        "integrity": {
            "task_manifest_matched": True,
            "evaluation_manifest_matched": True,
            "source_set_matched": True,
            "checkpoint_completed_budget": True,
            "all_actions_finite_and_bounded": True,
            "activity_from_actual_forward": True,
            "inference_latency_from_actual_forward": True,
            "physical_wind_telemetry_passed": True,
            "physical_wind_uses_terminal_actual_interval": True,
        },
    }
    evaluation_path = Path(job["evaluation_output"])
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")


def test_real_config_is_exactly_the_three_requested_combinations():
    config = report._read_json(CONFIG)
    output = report._validate_config(config, CONFIG)
    assert output.name == "crazyflie_command_combinations_seed0_1m"
    assert config["controllers"] == list(report.CONTROLLERS)
    assert all("gru" not in name and "mlp" not in name for name in config["controllers"])
    assert config["total_interactions_per_job"] == 1_000_000


def test_queue_requires_complete_six_cell_identity_and_equal_two_core_capacity(built):
    config, base, output = built
    value = _completed_queue(base)
    snapshots = {}
    jobs = report._validate_queue(
        value,
        config,
        config_path=CONFIG.resolve(),
        config_sha256=report._sha256_file(CONFIG),
        queue_path=Path(value["queue_file"]),
        output_root=output.resolve(),
        snapshots=snapshots,
    )
    assert len(jobs) == 6
    assert len(snapshots) >= 6
    assert value["controller_reports"]["leg_optic_lif"]["actor_trainable_parameters"] == 9224
    assert value["controller_reports"]["wing_optic_lif"]["actor_trainable_parameters"] == 9224
    assert value["controller_reports"]["leg_wing_optic_lif"]["actor_trainable_parameters"] == 13672
    wing_report = value["controller_reports"]["wing_optic_lif"]
    assert wing_report["core_labels"] == ["wing", "optic"]
    assert list(wing_report["per_core_checksums"]) == ["optic", "wing"]

    pending = json.loads(json.dumps(value))
    pending.update(status="pending", dry_run=False, counts={"pending": 6})
    with pytest.raises(ValueError, match="not globally completed"):
        report._validate_queue(
            pending,
            config,
            config_path=CONFIG.resolve(),
            config_sha256=report._sha256_file(CONFIG),
            queue_path=Path(pending["queue_file"]),
            output_root=output.resolve(),
            snapshots={},
        )


def test_queue_rejects_source_and_two_core_capacity_drift(built):
    config, base, output = built
    changed = _completed_queue(base)
    changed["jobs"][0]["fingerprint_payload"]["source_sha256"][
        "scripts/drone_train.py"
    ] = "0" * 64
    changed["jobs"][0]["expected_fingerprint"] = _canonical(
        changed["jobs"][0]["fingerprint_payload"]
    )
    with pytest.raises(ValueError, match="SHA-256 changed"):
        report._validate_queue(
            changed,
            config,
            config_path=CONFIG.resolve(),
            config_sha256=report._sha256_file(CONFIG),
            queue_path=Path(changed["queue_file"]),
            output_root=output.resolve(),
            snapshots={},
        )

    changed = _completed_queue(base)
    changed["controller_reports"]["wing_optic_lif"]["actor_trainable_parameters"] += 1
    with pytest.raises(ValueError, match="actor parameter count drifted"):
        report._validate_queue(
            changed,
            config,
            config_path=CONFIG.resolve(),
            config_sha256=report._sha256_file(CONFIG),
            queue_path=Path(changed["queue_file"]),
            output_root=output.resolve(),
            snapshots={},
        )


def test_complete_cell_validates_history_metrics_memory_and_all_core_activity(built):
    _config, base, output = built
    value = _completed_queue(base)
    job = value["jobs"][0]
    controller_report = value["controller_reports"][job["controller"]]
    _write_complete_artifacts(job, controller_report)
    cell = report._validate_cell(
        job, controller_report=controller_report, output_root=output.resolve()
    )
    assert cell.complete
    assert len(cell.history) == 250
    assert cell.components == ("leg", "optic")
    assert set(cell.activity["role_provenance"]) == {"leg", "optic"}
    assert cell.train_memory["max_device_gpu_used_mib"] == 3200.0


def test_cell_rejects_missing_activity_core_and_paging(built):
    _config, base, output = built
    value = _completed_queue(base)
    job = value["jobs"][1]
    controller_report = value["controller_reports"][job["controller"]]
    _write_complete_artifacts(job, controller_report)
    evaluation_path = Path(job["evaluation_output"])
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["activity"]["role_provenance"].pop("optic")
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="every core"):
        report._validate_cell(
            job, controller_report=controller_report, output_root=output.resolve()
        )

    _write_complete_artifacts(job, controller_report)
    manifest_path = Path(job["training_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["memory_gate"]["sustained_paging_detected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sustained paging"):
        report._validate_cell(
            job, controller_report=controller_report, output_root=output.resolve()
        )


def test_memory_gate_rejects_exact_gpu_ceiling():
    gate = _memory_gate()
    gate["max_device_gpu_used_mib"] = 6963.2
    with pytest.raises(ValueError, match="strict GPU/RAM cap"):
        report._validate_memory_gate(gate, "unit memory")


def test_inference_latency_requires_actual_synchronized_600_call_evidence():
    assert report._validate_inference_latency(_latency())["p95_ms"] == 0.15
    changed = _latency()
    changed["cuda_synchronized_before_and_after_call"] = False
    with pytest.raises(ValueError, match="cuda_synchronized"):
        report._validate_inference_latency(changed)
    changed = _latency()
    changed["sample_count"] = 589
    with pytest.raises(ValueError, match="sample_count"):
        report._validate_inference_latency(changed)
    with pytest.raises(ValueError, match="must be an object"):
        report._validate_inference_latency(None)


def test_terminal_v2_dependencies_revalidate_as_14_cells_and_224_episodes():
    snapshots = {}
    cells = report._collect_historical_v2_cells(
        report._read_json(CONFIG), snapshots
    )
    assert len(cells) == 14
    assert sum(16 for _cell in cells) == 224
    assert all(cell.complete for cell in cells)
    expected = {
        (ROOT / "runs/crazyflie_command_optic_wind_seed0_1m/queue.json").resolve(),
        (ROOT / "docs/crazyflie_command_optic_wind_report.md").resolve(),
    }
    assert expected.issubset(snapshots)


def test_report_source_contains_tier_and_no_baseline_claims():
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert "Exact fair comparison: revision-3 two-core tier" in source
    assert "older reset-dependent command schedule" in source
    assert "Isolated three-core tier" in source
    assert "exact action-producing `policy.act` calls" in source
    assert "descriptive only" in source
    assert "gru_matched_2core" not in source
    assert "mlp_matched_2core" not in source


def test_complete_markdown_and_all_six_plots_render_from_verified_cells(
    built, tmp_path
):
    _config, base, output = built
    value = _completed_queue(base)
    cells = []
    for job in value["jobs"]:
        controller_report = value["controller_reports"][job["controller"]]
        _write_complete_artifacts(job, controller_report)
        cells.append(report._validate_cell(
            job,
            controller_report=controller_report,
            output_root=output.resolve(),
        ))
    data = report.CombinationReportData(
        config_path=CONFIG,
        queue_path=Path(value["queue_file"]),
        output_root=output.resolve(),
        config_sha256="a" * 64,
        queue_sha256="b" * 64,
        cells=cells,
        historical_cells=report._collect_historical_v2_cells(
            report._read_json(CONFIG), {}
        ),
        input_hashes={},
        generated_at_utc="2026-09-18T00:00:00+00:00",
    )
    assert data.complete
    assert len(data.historical_cells) + len(data.cells) == 20
    assert (len(data.historical_cells) + len(data.cells)) * 16 == 320
    destinations = {
        name: tmp_path / filename for name, filename in report.PLOT_NAMES.items()
    }
    markdown = report.render_markdown(data, destinations)
    assert "COMPLETE — all 6 revision-3 jobs independently verified" in markdown
    assert "Authenticated combined 20-cell study" in markdown
    assert "320 held-out episodes" in markdown
    assert "Exact fair comparison: revision-3 two-core tier" in markdown
    assert "older reset-dependent command schedule" in markdown
    assert "Isolated three-core tier" in markdown
    assert "0.1000" in markdown
    assert "descriptive only" in markdown
    report.render_plots(data, destinations)
    assert set(destinations) == {
        "score", "wind_delta", "reward", "loss", "activity", "efficiency"
    }
    for path in destinations.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    data.output_root = tmp_path / "published-plots"
    report_path = tmp_path / "docs" / "combined-report.md"
    result = report.write_report_bundle(data, report_path)
    assert result["status"] == "COMPLETE"
    assert result["new_verified_jobs"] == 6
    assert result["historical_verified_jobs"] == 14
    assert result["combined_verified_cells"] == 20
    assert result["combined_evaluation_episodes"] == 320
    assert len(result["report_sha256"]) == 64
    assert report_path.is_file()
    assert all(len(item["sha256"]) == 64 for item in result["plots"].values())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report.write_report_bundle(data, report_path)
