"""CPU gates for the fail-closed fresh 60-cell revision-4 reporter."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_all_fair_queue_v4 as queue  # noqa: E402
import crazyflie_command_all_fair_report_v4 as report  # noqa: E402


CONFIG = (
    ROOT
    / "configs/experiments/crazyflie_command_all_fair_seeds0_1_2_1m.json"
)


def _canonical(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


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
        "call_site": (
            "policy.act(normalized_observation, recurrent_state, deterministic=True)"
        ),
        "clock": "time.perf_counter_ns_monotonic",
        "device": "cuda:0",
        "measurement_unit": "milliseconds_per_vectorized_policy_call",
        "batch_size": 16,
        "cuda_synchronized_before_and_after_call": True,
        "activity_recorder_hooks_in_scope": True,
        "action_producing_call_count": 600,
        "warmup_calls_excluded_from_statistics": 10,
        "warmup_exclusion_justification": (
            "exclude prefix calls that can include one-time lazy CUDA/kernel "
            "initialization; no extra forwards were executed"
        ),
        "sample_count": 590,
        "total_ms": 59.0,
        "mean_ms": 0.1,
        "p50_ms": 0.09,
        "p95_ms": 0.15,
        "p99_ms": 0.18,
        "max_ms": 0.2,
        "all_action_producing_calls_total_ms": 60.0,
    }


def _score(evaluation_manifest, *, seed=0):
    # Tiny seed variation exercises raw-seed aggregation without changing the
    # exact scoring definitions.
    raw = {
        "steps": 600,
        "episodes": 16,
        "linear_tracking_rmse_m_s": 0.10 + 0.005 * seed,
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
        "linear_tracking": 100.0
        * math.exp(-((raw["linear_tracking_rmse_m_s"] / 0.35) ** 2)),
        "yaw_tracking": 100.0
        * math.exp(-((raw["yaw_tracking_rmse_rad_s"] / 0.50) ** 2)),
        "direction": 100.0 * (1.0 - raw["wrong_direction_fraction"]),
        "response": 100.0
        * math.exp(-(raw["response_latency_mean_s"] / 0.80)),
        "braking": 100.0
        * math.exp(-(raw["brake_settling_mean_s"] / 1.00)),
        "hover": 100.0
        * math.exp(
            -((raw["hover_speed_rms_m_s"] / 0.15) ** 2)
            - ((raw["hover_drift_mean_m"] / 0.15) ** 2)
        ),
        "safety": 100.0,
        "effort": 100.0 * math.exp(-((raw["action_effort_rms"] / 0.25) ** 2)),
        "smoothness": 100.0
        * math.exp(-((raw["action_delta_rms"] / 0.08) ** 2)),
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
            "still_air_exact_zero_all_simulated_intervals": (
                True if not wind else None
            ),
            "complete_600_step_protocol_observed_for_all_episodes": True,
            "wind_nonzero_wrench_observed": True if wind else None,
            "passed": True,
        },
    }


def _activity_summary(per_unit, sample_count):
    roles = {}
    for role in dict.fromkeys(unit["role"] for unit in per_unit):
        selected = [unit for unit in per_unit if unit["role"] == role]
        roles[role] = {
            "unit_count": len(selected),
            "mean_absolute_activity_per_unit": sum(
                unit["mean_absolute_activity"] for unit in selected
            )
            / len(selected),
            "active_fraction_per_unit": sum(
                unit["active_fraction"] for unit in selected
            )
            / len(selected),
        }
    return {
        "sample_count": sample_count,
        "mean_absolute_activity_per_unit": sum(
            unit["mean_absolute_activity"] for unit in per_unit
        )
        / len(per_unit),
        "rms_activity_per_unit": math.sqrt(
            sum(unit["rms_activity"] ** 2 for unit in per_unit) / len(per_unit)
        ),
        "active_fraction_per_unit": sum(
            unit["active_fraction"] for unit in per_unit
        )
        / len(per_unit),
        "roles": roles,
    }


def _activity(job):
    controller = job["controller"]
    if controller in queue.LIF_CONTROLLERS:
        components = queue.COMPONENTS_BY_CONTROLLER[controller]
        manifests = job["expected_connectome_manifests"]
        provenance = {}
        identities = []
        for component in components:
            key = component if len(components) > 1 else "primary"
            manifest = Path(manifests[key])
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            neurons_path = (manifest.parent / manifest_value["neurons_path"]).resolve()
            neurons = json.loads(neurons_path.read_text(encoding="utf-8"))
            provenance[component] = queue._manifest_activity_identity(str(manifest))
            identities.extend(
                (
                    f"{component}:{row['id']}",
                    f"{component}:{row['annotations']['model_role']}",
                )
                for row in neurons
            )
        kind = "sampled_lif_spikes"
    elif controller == "gru_matched":
        provenance = {
            "engineering": {
                "kind": "matched_gru_hidden_state",
                "biological_roles": False,
                "unit_count": 33,
            }
        }
        identities = [(f"gru:hidden:{index}", "gru:hidden") for index in range(33)]
        kind = "engineering_absolute_activations"
    else:
        provenance = {
            "engineering": {
                "kind": "matched_mlp_post_activation_hidden_units",
                "activation_values": "absolute_actual_post_activation_output",
                "biological_roles": False,
                "layer_widths": [61, 61],
                "unit_count": 122,
            }
        }
        identities = [
            (f"mlp:hidden_{layer}:{index}", f"mlp:hidden_{layer}")
            for layer in range(2)
            for index in range(61)
        ]
        kind = "engineering_absolute_activations"
    per_unit = []
    for index, (stable_id, role) in enumerate(identities):
        mean = 0.02 + index * 0.00001
        rms = mean * 1.1
        active = 0.1 + (index % 7) * 0.001
        per_unit.append(
            {
                "index": index,
                "id": stable_id,
                "role": role,
                "absolute_activity_sum": mean * 9600,
                "mean_absolute_activity": mean,
                "rms_activity": rms,
                "active_count": int(active * 9600),
                "active_fraction": active,
            }
        )
    role_counts = {}
    for unit in per_unit:
        role_counts[unit["role"]] = role_counts.get(unit["role"], 0) + 1
    ranked = sorted(
        per_unit, key=lambda unit: (-unit["mean_absolute_activity"], unit["id"])
    )[:10]
    return {
        "source": "exact_forward_pass_that_produced_each_evaluated_action",
        "controller": job["policy"],
        "kind": kind,
        "unit_count": len(per_unit),
        "role_counts": role_counts,
        "role_provenance": provenance,
        "overall": _activity_summary(per_unit, 9600),
        "segments": {
            name: _activity_summary(per_unit, 800) for name in report.SEGMENT_NAMES
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
            for unit in ranked
        ],
    }


def _schedule(job):
    seed = job["seed"]
    wind = job["task"] == queue.TASKS[1]
    requested = [[0.1 * (seed + 1), -0.1, 0.05, 0.2] for _ in range(40)]
    zeros3 = [[0.0, 0.0, 0.0] for _ in range(40)]
    force = (
        [[0.01 * (seed + 1), 0.0, 0.0] for _ in range(40)]
        if wind
        else zeros3
    )
    torque = (
        [[0.0, 0.001 * (seed + 1), 0.0] for _ in range(40)]
        if wind
        else zeros3
    )
    state = {
        "schema_version": 2,
        "kind": "flyg1.crazyflie.command-wide-schedule.v2",
        "contract_sha256": job["command_training_contract_sha256"],
        "num_envs": 40,
        "wind_enabled": wind,
        "command_schedule_seed": seed,
        "wind_schedule_seed": report.TRAINING_WIND_SEED,
        "held_out_wind_seed": report.HELD_OUT_WIND_SEED,
        "wind_active_seed": report.TRAINING_WIND_SEED,
        "wind_mode": "training",
        "training_interactions": 1_000_000,
        "next_command_segment_index": [250 + seed] * 40,
        "command_steps_remaining": [25] * 40,
        "requested_command_body": requested,
        "command_category_code": [1] * 40,
        "command_stage_index": [3] * 40,
        "next_wind_segment_index": ([200 + seed] * 40 if wind else [0] * 40),
        "wind_steps_remaining": ([30] * 40 if wind else [0] * 40),
        "wind_force_ratio_world": force,
        "wind_torque_ratio_world": torque,
        "applied_wind_force_world": force,
        "applied_wind_torque_world": torque,
        "wind_category_code": ([1] * 40 if wind else [0] * 40),
        "wind_stage_index": [3] * 40,
        "heldout_episode_index": [0] * 40,
        "heldout_step_cursor": [0] * 40,
        "wind_evaluation_protocol_sha256": "d" * 64,
    }
    return {
        "command_follow_contract": {"test": True},
        "command_training_contract_sha256": job[
            "command_training_contract_sha256"
        ],
        "state_sha256": _canonical(state),
        "state": state,
    }


def _write_attempts(job, output):
    attempts = []
    for index, phase in enumerate(("training", "evaluation"), start=1):
        log = output / "logs" / f"{job['id']}__{phase}__attempt-{index:04d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"{job['id']} {phase} complete\n", encoding="utf-8")
        attempts.append(
            {
                "attempt": index,
                "phase": phase,
                "started_utc": f"2026-09-18T00:00:0{index}+00:00",
                "finished_utc": f"2026-09-18T00:01:0{index}+00:00",
                "command": list(job[f"{phase}_command"]),
                "log": str(log),
                "archived_invalid_output": None,
                "pid": 1000 + index,
                "exit_code": 0,
                "resource_precheck": {
                    "passed": True,
                    "gpu_used_mib": 500.0,
                    "system_ram_percent": 20.0,
                },
            }
        )
    job["attempts"] = attempts
    job["failure_history"] = []


def _write_artifacts(job, controller_report):
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{job['id']}".encode())
    checkpoint_sha = report._sha256_file(checkpoint)
    rows = [
        {
            "completed_updates": index,
            "total_interactions": index * 4000,
            "mean_rollout_reward": index / 1000.0 + job["seed"] * 0.01,
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
        "segments": [
            {
                "path": "../history/" + history_path.name,
                "sha256": sha256(payload).hexdigest(),
                "byte_count": len(payload),
                "row_count": 250,
            }
        ],
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": job["task"],
        "contract_profile": "command_v2",
        "controller": job["policy"],
        "seed": job["seed"],
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
        "training_wall_time_s": 60.0 + job["seed"],
        "command_schedule": _schedule(job),
        "history_reference": history_reference,
        "core_checksum_before": controller_report.get("core_checksum"),
        "core_checksum_after": controller_report.get("core_checksum"),
        "per_core_checksums_before": controller_report.get(
            "per_core_checksums", {}
        ),
        "per_core_checksums_after": controller_report.get(
            "per_core_checksums", {}
        ),
    }
    manifest_path = Path(job["training_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    score = _score(job["evaluation_manifest"], seed=job["seed"])
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
            "training_seed": job["seed"],
            "total_interactions": 1_000_000,
            "reproduction_fingerprint": job["expected_fingerprint"],
            "evaluation_manifest_id": job["evaluation_manifest_id"],
            "evaluation_manifest": job["evaluation_manifest"],
        },
        "protocol": job["evaluation_manifest"]["evaluation_protocol"],
        "protocol_sha256": job["evaluation_manifest"][
            "evaluation_protocol_sha256"
        ],
        "episodes_requested": 16,
        "episodes_evaluated": 16,
        "steps_per_episode": 600,
        "vectorized_environment_count": 16,
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
            "physical_wind": _physical_wind(job["task"] == queue.TASKS[1]),
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
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue.validate_config(CONFIG)
    output = tmp_path_factory.mktemp("all-fair-v4-report") / "matrix"
    config = dict(config)
    config["_output_root"] = str(output.resolve())
    value = queue.build_queue(config, output)
    value = json.loads(json.dumps(value, sort_keys=True))
    value.update(status="completed", dry_run=False, counts={"completed": 60})
    for job in value["jobs"]:
        job.update(
            status="completed",
            training_status="completed",
            evaluation_status="completed",
        )
        _write_attempts(job, output)
        _write_artifacts(job, value["controller_reports"][job["controller"]])
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / "queue.json"
    queue_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return config, value, output, queue_path


@pytest.fixture(scope="module")
def data(built):
    config, _value, output, queue_path = built
    validated = dict(config)
    validated["_output_root"] = str(output.resolve())
    original = report._validate_config
    report._validate_config = lambda _path: (dict(validated), output.resolve())
    try:
        return report.collect_report_data(CONFIG, queue_path)
    finally:
        report._validate_config = original


def test_contract_is_exactly_10_by_2_by_3_and_never_merges_history():
    assert len(report.CONTROLLERS) == 10
    assert report.SEEDS == (0, 1, 2)
    assert report.JOB_COUNT == 60
    assert report.JOB_COUNT * report.EPISODES == 960
    assert report.JOB_COUNT * report.INTERACTIONS == 60_000_000
    assert not hasattr(report.AllFairReportData, "historical_cells")
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert "historical_cells_merged\": 0" in source
    assert "Cross-tier comparisons are descriptive only" in source
    assert "do not establish causal" in source


def test_controller_reports_validate_all_capacity_tiers_and_baselines(built):
    _config, value, _output, _queue_path = built
    for controller in report.CONTROLLERS:
        controller_report = value["controller_reports"][controller]
        report._validate_controller_report(controller, controller_report)
        assert (
            controller_report["actor_trainable_parameters"]
            == queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
        )
    assert queue.ACTOR_PARAMETERS_BY_CONTROLLER["original_lif"] == 4776
    assert queue.ACTOR_PARAMETERS_BY_CONTROLLER["leg_wing_lif"] == 9224
    assert queue.ACTOR_PARAMETERS_BY_CONTROLLER["leg_wing_optic_lif"] == 13672
    assert queue.ACTOR_PARAMETERS_BY_CONTROLLER["gru_matched"] == 4793
    assert queue.ACTOR_PARAMETERS_BY_CONTROLLER["mlp_normal"] == 4827

    for controller in ("original_lif", "leg_wing_lif", "leg_wing_optic_lif"):
        changed = json.loads(
            json.dumps(value["controller_reports"][controller])
        )
        changed["core_checksum"] = "0" * 64
        with pytest.raises(ValueError, match="does not bind its per-core checksums"):
            report._validate_controller_report(controller, changed)


def test_queue_refuses_any_partial_matrix_and_validates_immutable_attempts(built):
    config, value, output, queue_path = built
    snapshots = {}
    jobs = report._validate_queue(
        value,
        config,
        config_path=CONFIG.resolve(),
        queue_path=queue_path.resolve(),
        output_root=output.resolve(),
        snapshots=snapshots,
    )
    assert len(jobs) == 60
    assert sum(len(job["attempts"]) for job in jobs) == 120
    assert len([path for path in snapshots if path.parent == output / "logs"]) == 120

    partial = json.loads(json.dumps(value))
    partial.update(status="partial_failed", counts={"completed": 59, "failed": 1})
    partial["jobs"][17].update(
        status="failed", evaluation_status="failed"
    )
    with pytest.raises(ValueError, match="status differs|counts differs"):
        report._validate_queue(
            partial,
            config,
            config_path=CONFIG.resolve(),
            queue_path=queue_path.resolve(),
            output_root=output.resolve(),
            snapshots={},
        )

    malformed_launch = json.loads(json.dumps(value["jobs"][0]))
    malformed_launch["attempts"][0]["exit_code"] = None
    malformed_launch["attempts"][0].pop("launch_error", None)
    with pytest.raises(ValueError, match="launch failure lacks"):
        report._validate_attempts(malformed_launch, output.resolve(), {})


def test_failure_isolation_receipts_must_reconcile_exactly(built):
    config, value, output, queue_path = built
    changed = json.loads(json.dumps(value))
    failure = {
        "index": 1,
        "utc": "2026-09-18T00:00:00+00:00",
        "phase": "evaluation",
        "exit_code": 1,
        "reason": "evaluation exited 1; artifact validation failed",
    }
    changed["jobs"][3]["failure_history"] = [failure]
    changed["jobs"][3]["failure"] = failure["reason"]
    changed["events"].append(
        {
            "event": "job_failure_isolated",
            "job": changed["jobs"][3]["id"],
            **failure,
        }
    )
    report._validate_queue(
        changed,
        config,
        config_path=CONFIG.resolve(),
        queue_path=queue_path.resolve(),
        output_root=output.resolve(),
        snapshots={},
    )
    changed["events"][-1]["reason"] = "tampered"
    with pytest.raises(ValueError, match="lacks one queue event"):
        report._validate_queue(
            changed,
            config,
            config_path=CONFIG.resolve(),
            queue_path=queue_path.resolve(),
            output_root=output.resolve(),
            snapshots={},
        )


def test_all_60_cells_validate_budget_history_memory_activity_and_latency(data):
    assert data.complete
    assert len(data.cells) == 60
    assert sum(16 for _cell in data.cells) == 960
    assert {(cell.controller, cell.seed, cell.condition) for cell in data.cells} == {
        (controller, seed, condition)
        for controller in report.CONTROLLERS
        for seed in report.SEEDS
        for condition in ("still", "wind")
    }
    assert all(len(cell.history) == 250 for cell in data.cells)
    assert all(cell.inference_latency["sample_count"] == 590 for cell in data.cells)
    assert all(cell.train_memory["max_device_gpu_used_mib"] < 6963.2 for cell in data.cells)
    assert all(cell.eval_memory["max_device_gpu_used_mib"] < 6963.2 for cell in data.cells)
    assert {cell.activity["kind"] for cell in data.cells} == {
        "sampled_lif_spikes",
        "engineering_absolute_activations",
    }
    assert Path(report.__file__).resolve() in data.input_hashes
    assert Path(report.v3_helpers.__file__).resolve() in data.input_hashes
    assert Path(report.common.__file__).resolve() in data.input_hashes


def test_cell_rejects_missing_latency_and_per_core_drift(built):
    _config, value, output, _queue_path = built
    job = value["jobs"][0]
    controller_report = value["controller_reports"][job["controller"]]
    evaluation_path = Path(job["evaluation_output"])
    original_evaluation = evaluation_path.read_bytes()
    evaluation = json.loads(original_evaluation)
    evaluation.pop("inference_latency")
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation validator"):
        report._validate_cell(
            job, controller_report=controller_report, output_root=output.resolve()
        )
    evaluation_path.write_bytes(original_evaluation)

    manifest_path = Path(job["training_manifest"])
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["per_core_checksums_after"]["primary"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training validator"):
        report._validate_cell(
            job, controller_report=controller_report, output_root=output.resolve()
        )
    manifest_path.write_bytes(original_manifest)


def test_paired_schedule_gate_rejects_one_controller_drift(data):
    report._validate_paired_schedules(data.cells)
    changed = list(data.cells)
    target = next(
        cell
        for cell in changed
        if cell.controller == "mlp_normal" and cell.seed == 2 and cell.condition == "wind"
    )
    original = target.command_schedule_identity_sha256
    target.command_schedule_identity_sha256 = "0" * 64
    try:
        with pytest.raises(ValueError, match="reset-invariant command schedule"):
            report._validate_paired_schedules(changed)
    finally:
        target.command_schedule_identity_sha256 = original


def test_schedule_uses_contract_wind_seeds_independent_of_training_seed(built):
    _config, value, _output, _queue_path = built
    job = next(job for job in value["jobs"] if job["seed"] == 2)
    schedule = _schedule(job)
    report._validate_schedule(schedule, job=job)
    assert schedule["state"]["command_schedule_seed"] == 2
    assert schedule["state"]["wind_schedule_seed"] == report.TRAINING_WIND_SEED
    assert schedule["state"]["wind_active_seed"] == report.TRAINING_WIND_SEED
    assert schedule["state"]["held_out_wind_seed"] == report.HELD_OUT_WIND_SEED

    for field in (
        "wind_schedule_seed",
        "held_out_wind_seed",
        "wind_active_seed",
    ):
        changed = json.loads(json.dumps(schedule))
        changed["state"][field] += 1
        changed["state_sha256"] = _canonical(changed["state"])
        with pytest.raises(ValueError, match=rf"{field} differs"):
            report._validate_schedule(changed, job=job)


def test_activity_gate_covers_each_lif_core_and_engineering_layer(data):
    for cell in data.cells:
        groups = report._activity_groups(cell)
        if cell.controller in report.LIF_CONTROLLERS:
            assert set(groups) == set(report.COMPONENTS[cell.controller])
        elif cell.controller == "gru_matched":
            assert set(groups) == {"gru:hidden"}
        else:
            assert set(groups) == {"mlp:hidden_0", "mlp:hidden_1"}
        assert all(count > 0 for _mean, _active, count in groups.values())


def test_activity_gate_requires_exact_overall_and_segment_sample_counts(built):
    _config, value, _output, _queue_path = built
    job = value["jobs"][0]
    evaluation = json.loads(Path(job["evaluation_output"]).read_text(encoding="utf-8"))

    changed = json.loads(json.dumps(evaluation["activity"]))
    changed["overall"]["sample_count"] = 0
    with pytest.raises(ValueError, match="sample count differs from the held-out protocol"):
        report._validate_activity(changed, job=job, episodes=evaluation["episodes"])

    changed = json.loads(json.dumps(evaluation["activity"]))
    changed["segments"][report.SEGMENT_NAMES[0]]["sample_count"] = 799
    changed["segments"][report.SEGMENT_NAMES[1]]["sample_count"] = 801
    with pytest.raises(ValueError, match="sample count differs from the held-out protocol"):
        report._validate_activity(changed, job=job, episodes=evaluation["episodes"])


def test_activity_gate_derives_early_termination_counts_from_episodes(built):
    _config, value, _output, _queue_path = built
    job = value["jobs"][0]
    evaluation = json.loads(Path(job["evaluation_output"]).read_text(encoding="utf-8"))
    episodes = json.loads(json.dumps(evaluation["episodes"]))
    episodes[0].update(alive_steps=124, terminated=True, truncated=False)
    overall_count, segment_counts = report._expected_activity_sample_counts(
        episodes, job_id=job["id"]
    )
    assert overall_count == 9_125
    assert list(segment_counts.values())[:4] == [800, 800, 775, 750]

    activity = json.loads(json.dumps(evaluation["activity"]))
    activity["overall"]["sample_count"] = overall_count
    for name, count in segment_counts.items():
        activity["segments"][name]["sample_count"] = count
    report._validate_activity(activity, job=job, episodes=episodes)

    activity["segments"][report.SEGMENT_NAMES[2]]["sample_count"] += 1
    with pytest.raises(ValueError, match="sample count differs from the held-out protocol"):
        report._validate_activity(activity, job=job, episodes=episodes)


def test_markdown_plots_and_no_overwrite_bundle_are_all_v4_only(data, tmp_path):
    destinations = {
        name: tmp_path / filename for name, filename in report.PLOT_NAMES.items()
    }
    markdown = report.render_markdown(data, destinations)
    assert "all 60 fresh revision-4 cells independently verified" in markdown
    assert "60,000,000 training interactions" in markdown
    assert "960 evaluation episodes" in markdown
    assert "Per-seed 20-cell results" in markdown
    assert "Seed 0" in markdown and "Seed 1" in markdown and "Seed 2" in markdown
    assert "raw seeds" in markdown
    assert "Activity differences are descriptive correlations only" in markdown
    assert "none of their scores" in markdown
    report.render_plots(data, destinations)
    assert set(destinations) == set(report.PLOT_NAMES)
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in destinations.values())

    original_output = data.output_root
    data.output_root = tmp_path / "published-plots"
    try:
        report_path = tmp_path / "docs" / "all-fair-v4.md"
        result = report.write_report_bundle(data, report_path)
        assert result["status"] == "COMPLETE"
        assert result["verified_jobs"] == 60
        assert result["training_interactions"] == 60_000_000
        assert result["evaluation_episodes"] == 960
        assert result["historical_cells_merged"] == 0
        assert len(result["report_sha256"]) == 64
        assert set(result["plots"]) == set(report.PLOT_NAMES)
        assert all(len(item["sha256"]) == 64 for item in result["plots"].values())
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            report.write_report_bundle(data, report_path)
    finally:
        data.output_root = original_output


def test_memory_and_latency_helpers_reject_boundary_or_malformed_values():
    gate = _memory_gate()
    gate["max_device_gpu_used_mib"] = 6963.2
    with pytest.raises(ValueError, match="strict GPU/RAM cap"):
        report.v3_helpers._validate_memory_gate(gate, "test memory")
    latency = _latency()
    latency["sample_count"] = 589
    with pytest.raises(ValueError, match="sample_count"):
        report.v3_helpers._validate_inference_latency(latency)
    with pytest.raises(ValueError, match="must be an object"):
        report.v3_helpers._validate_inference_latency(None)
