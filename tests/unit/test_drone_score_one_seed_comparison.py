from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_score_one_seed_comparison as scorer  # noqa: E402
from drone_bootstrap import canonical_sha256, sha256_file  # noqa: E402
from drone_evaluation_protocol import load_protocol  # noqa: E402
from g1_fly_control.tasks.crazyflie.logic import AUDITED_CRAZYFLIE_MASS_KG  # noqa: E402
from g1_fly_control.tasks.crazyflie.metrics import (  # noqa: E402
    EpisodeSummary,
    GustRecoveryOutcome,
    SwitchOutcome,
    summarize_episodes,
)


def _result(scenario: str, *, perfect: bool = True) -> dict[str, object]:
    records = []
    for episode_id in range(scorer.EPISODES_PER_CELL):
        switch_outcomes = ()
        gust_outcomes = ()
        if scenario == scorer.TASKS[1]:
            switch_outcomes = tuple(
                SwitchOutcome(
                    switch_index=index,
                    switch_step=(150, 300, 450)[index],
                    success=perfect,
                    latency_s=0.5 if perfect else None,
                    failure_reason=None if perfect else "target_segment_ended_without_success",
                )
                for index in range(3)
            )
        if scenario == scorer.TASKS[2]:
            gust_outcomes = tuple(
                GustRecoveryOutcome(
                    gust_index=index,
                    gust_start_step=(150, 300, 450)[index],
                    applied=True,
                    stable_before_gust=True,
                    recovered=perfect,
                    recovery_latency_s=0.5 if perfect else None,
                    max_displacement_m=0.0 if perfect else 1.0,
                    post_gust_error_integral_m_s=0.0 if perfect else 2.0,
                    failure_reason=None if perfect else "recovery_window_expired_or_not_reached",
                )
                for index in range(3)
            )
        records.append(
            EpisodeSummary(
                scenario=scenario,
                episode_id=episode_id,
                success=perfect,
                terminated=False,
                truncated=True,
                failure_reason=None,
                time_to_first_success_s=0.5 if perfect else None,
                final_goal_error_m=0.0 if perfect else 1.0,
                integrated_goal_error_m_s=0.0 if perfect else 12.0,
                mean_speed_inside_target_region_m_s=0.0 if perfect else None,
                crash=False,
                out_of_bounds=False,
                invalid_state=False,
                command_effort=0.0,
                command_smoothness=0.0,
                aggregate_wrench_mechanical_work_proxy_j=0.0,
                switch_outcomes=switch_outcomes,
                gust_outcomes=gust_outcomes,
            )
        )
    summary = summarize_episodes(records, expected_episode_count=scorer.EPISODES_PER_CELL)
    event_count = (
        4 if scenario == scorer.TASKS[1] and perfect
        else 1 if perfect
        else 0
    )
    for row in summary["episodes"]:
        row["target_success_event_count"] = event_count
    protocol = load_protocol("main")
    for episode_id, row in enumerate(summary["episodes"]):
        plan = deepcopy(protocol["scenarios"][scenario][episode_id])
        row["plan"] = plan
        row["plan_sha256"] = plan["plan_sha256"]
        if scenario == scorer.TASKS[2]:
            expected = [
                [
                    AUDITED_CRAZYFLIE_MASS_KG
                    * float(gust["desired_mass_normalized_delta_velocity_m_s"])
                    * float(gust["direction_world_xy"][0]),
                    AUDITED_CRAZYFLIE_MASS_KG
                    * float(gust["desired_mass_normalized_delta_velocity_m_s"])
                    * float(gust["direction_world_xy"][1]),
                    0.0,
                ]
                for gust in plan["gusts"]
            ]
            row.update(
                {
                    "robot_mass_kg": AUDITED_CRAZYFLIE_MASS_KG,
                    "gust_applied_impulse_w_n_s": deepcopy(expected),
                    "gust_expected_impulse_w_n_s": deepcopy(expected),
                    "gust_impulse_max_abs_error_n_s": 0.0,
                }
            )
        else:
            row.update(
                {
                    "robot_mass_kg": None,
                    "gust_applied_impulse_w_n_s": [],
                    "gust_expected_impulse_w_n_s": [],
                    "gust_impulse_max_abs_error_n_s": None,
                }
            )
    # The production scorer consumes JSON, where dataclass tuples are arrays.
    return json.loads(json.dumps({"episodes": summary["episodes"], "summary": summary}))


def test_perfect_fixed_score_is_exactly_100_for_every_task() -> None:
    for scenario in scorer.TASKS:
        scored = scorer.score_evaluation(_result(scenario), scenario)
        assert scored["score"] == pytest.approx(100.0)
        assert set(scored["score_components_points"]) == set(
            scorer.SCORE_CONTRACT["weights_points"]
        )
        assert scored["raw_metrics"]["task_event_completion_rate"] == 1.0
        assert scored["raw_metrics"]["all_required_event_episode_rate"] == 1.0


def test_zero_event_policy_keeps_nonbinary_tracking_safety_and_control_score() -> None:
    scored = scorer.score_evaluation(_result(scorer.TASKS[0], perfect=False), scorer.TASKS[0])
    assert scored["raw_metrics"]["task_event_success_count"] == 0
    assert scored["score_components_points"]["task_event_completion"] == 0.0
    assert scored["score_components_points"]["all_required_events"] == 0.0
    assert scored["score_components_points"]["fixed_horizon_censored_latency"] == 0.0
    # The score is deliberately not binary: safe finite flight, tracking, and
    # smooth bounded commands remain visible when target success is zero.
    assert 25.0 < scored["score"] < 55.0


def test_event_detail_tampering_is_rejected_even_when_aggregate_summary_is_unchanged() -> None:
    result = _result(scorer.TASKS[1])
    result["episodes"][0]["target_success_event_count"] = 0
    result["summary"]["episodes"][0]["target_success_event_count"] = 0
    with pytest.raises(ValueError, match="success disagrees"):
        scorer.score_evaluation(result, scorer.TASKS[1])


@pytest.mark.parametrize(
    "field",
    ("success", "terminated", "truncated", "crash", "out_of_bounds", "invalid_state"),
)
def test_episode_boolean_score_and_failure_fields_require_exact_bool(field: str) -> None:
    result = _result(scorer.TASKS[0])
    replacement = int(result["episodes"][0][field])
    result["episodes"][0][field] = replacement
    result["summary"]["episodes"][0][field] = replacement
    with pytest.raises(ValueError, match=rf"Episode {field} field must be Boolean"):
        scorer.score_evaluation(result, scorer.TASKS[0])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complete_queue(tmp_path: Path) -> Path:
    protocol = load_protocol("main")
    protocol_sha256 = canonical_sha256(protocol)
    queue_path = tmp_path / "queue.json"
    config_path = tmp_path / "config.json"
    jobs = []
    for controller in scorer.CONTROLLERS:
        for task in scorer.TASKS:
            plans = protocol["scenarios"][task]
            identifier = f"{task}__{controller}__seed-0"
            run_dir = tmp_path / "artifacts" / "jobs" / identifier
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(identifier.encode("utf-8"))
            fingerprint_payload = {
                "schema_version": 1,
                "resolved_config": {"task": task, "controller": controller, "seed": 0},
            }
            fingerprint = canonical_sha256(fingerprint_payload)
            controller_report = {
                "actor_trainable_parameters": 4776,
                "critic_trainable_parameters": 18305,
                "total_trainable_parameters": 23081,
                "frozen_parameters": 5103 if "lif" in controller else 0,
                "total_dynamic_state_per_environment": 1024,
                "actor_parameter_match_passed": True,
            }
            training_manifest = {
                "status": "completed",
                "task": task,
                "contract_profile": "balanced_v4",
                "controller": controller,
                "seed": 0,
                "requested_interactions": scorer.INTERACTIONS_PER_JOB,
                "environment_interactions": scorer.INTERACTIONS_PER_JOB,
                "completed_updates": 2500,
                "completed_episodes": 100,
                "training_wall_time_s": 100.0,
                "resume_count": 0,
                "warm_start": None,
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "evaluation_manifest_id": protocol["manifest_id"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "memory_gate": {"passed": True, "warnings": []},
                "controller_report": controller_report,
                "core_checksum_before": "core",
                "core_checksum_after": "core",
                "history_reference": {"segments": []},
                "ppo": {"loss": 0.1},
            }
            _write_json(run_dir / "training_manifest.json", training_manifest)

            evaluation = _result(task)
            output = tmp_path / "artifacts" / "evaluations" / identifier / f"{task}.json"
            result = {
                "schema_version": 1,
                "status": "completed",
                "label": "main",
                "protocol": "main",
                "scenario": task,
                "evaluation_seed": 101,
                "evaluation_manifest_id": protocol["manifest_id"],
                "training_seed": 0,
                "controller": controller,
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "memory_gate": {"passed": True, "warnings": []},
                "episodes": evaluation["episodes"],
                "summary": evaluation["summary"],
            }
            _write_json(output, result)
            bundle = {
                "scenario": task,
                "status": "completed",
                "output": str(output),
                "protocol": "main",
                "protocol_label": "main",
                "evaluation_manifest_id": protocol["manifest_id"],
                "evaluation_manifest_sha256": protocol_sha256,
                "evaluation_seed": 101,
                "expected_episode_ids": list(range(16)),
                "expected_plan_sha256": [plan["plan_sha256"] for plan in plans],
                "expected_plan_object_sha256": [canonical_sha256(plan) for plan in plans],
                "command": ["python", "scripts/drone_evaluate.py"],
            }
            jobs.append(
                {
                    "id": identifier,
                    "controller": controller,
                    "seed": 0,
                    "task": task,
                    "contract_profile": "balanced_v4",
                    "status": "completed",
                    "total_interactions": scorer.INTERACTIONS_PER_JOB,
                    "training_resources": {
                        "num_envs": 4,
                        "horizon": 100,
                        "ppo_epochs": 2,
                        "microbatch_size": 4,
                    },
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "expected_fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                    "evaluation_manifest_id": protocol["manifest_id"],
                    "controller_report": controller_report,
                    "evaluations": [bundle],
                }
            )
    config = {
        "schema_version": 1,
        "label": "comparison",
        "matrix_layout": scorer.matrix_runner.TASK_SEPARATED_COMPARISON_LAYOUT,
        "execution_readiness": scorer.matrix_runner.TASK_SEPARATED_COMPARISON_READINESS,
        "controllers": list(scorer.CONTROLLERS),
        "tasks": list(scorer.TASKS),
        "seeds": [0],
        "total_interactions": scorer.INTERACTIONS_PER_JOB,
        "training": {
            "num_envs": 4,
            "horizon": 100,
            "ppo_epochs": 2,
            "microbatch_size": 4,
            "num_workers": 0,
            "precision": "float32",
            "learning_rate": 0.0003,
            "checkpoint_every_updates": 100,
        },
        "evaluation": {
            "protocol": "main",
            "seed": 101,
            "episodes_per_scenario": 16,
            "scenarios": list(scorer.TASKS),
            "matched_task_only": True,
        },
    }
    _write_json(config_path, config)
    concurrency_decision = {
        "status": "fallback_sequential",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 1,
        "fallback_applied": True,
        "fallback_reason": "unit-test missing receipt",
        "receipt": str(tmp_path / "paired-smoke.json"),
        "receipt_sha256": None,
        "receipt_id": None,
        "source_set_sha256": None,
        "reports": [],
        "aggregate_overlap_memory_gate": None,
    }
    queue = {
        "schema_version": 1,
        "label": "comparison",
        "status": "completed",
        "dry_run": False,
        "counts": {"completed": 12},
        "config_path": str(config_path),
        "output": str(queue_path),
        "requested_max_concurrent_isaac_processes": 2,
        "max_concurrent_isaac_processes": 1,
        "concurrency_decision": concurrency_decision,
        "resource_limits": {"max_concurrent_isaac_processes": 1},
        "job_count": 12,
        "evaluation_bundle_count": 12,
        "predicted_evaluation_episodes": 192,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "jobs": jobs,
    }
    _write_json(queue_path, queue)
    return queue_path


def _trust_current_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scorer.matrix_runner,
        "validate_config",
        lambda path: {
            "_matrix_layout": scorer.matrix_runner.TASK_SEPARATED_COMPARISON_LAYOUT
        },
    )
    monkeypatch.setattr(
        scorer.matrix_runner,
        "validate_resume_queue",
        lambda queue, config, output: None,
    )


def _history_summary_fixture() -> dict[str, object]:
    return {
        "row_count": 2500,
        "first_completed_updates": 1,
        "last_completed_updates": 2500,
        "first_total_interactions": 400,
        "last_total_interactions": scorer.INTERACTIONS_PER_JOB,
        "scalar_metrics": {
            "loss": {
                "observation_count": 2500,
                "first": 1.0,
                "last": 0.1,
                "mean": 0.5,
                "minimum": 0.1,
                "maximum": 1.0,
            }
        },
        "count_totals": {"target_success_count": 4},
        "failure_cause_count_totals": {"1": 0, "2": 0, "3": 0, "4": 0},
        "history_reference": {"history_sha256": "a" * 64, "segments": []},
    }


def test_complete_report_validates_12_cells_192_episodes_and_self_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    # Binary checkpoint/history authentication is extensively exercised by
    # the runner's own tests.  Here the two calls are isolated so this test can
    # exercise the scorer's independent design, detail, and score validation
    # without manufacturing Torch checkpoint internals.
    monkeypatch.setattr(scorer.matrix_runner, "_valid_training", lambda job: True)
    monkeypatch.setattr(
        scorer.matrix_runner,
        "_valid_evaluation",
        lambda bundle, job, expected: expected == 16,
    )
    monkeypatch.setattr(
        scorer, "_training_history_summary", lambda checkpoint, manifest: _history_summary_fixture()
    )
    _trust_current_queue(monkeypatch)
    report = scorer.build_report(queue_path)
    assert report["status"] == "complete"
    assert report["completion"] == {
        "valid_cells": 12,
        "planned_cells": 12,
        "validated_evaluation_episodes": 192,
        "planned_evaluation_episodes": 192,
    }
    assert len(report["controller_task_table"]) == 12
    assert all(row["macro_mean_score"] == pytest.approx(100.0) for row in report["overall_controller_table"])
    assert all(row["rank"] == 1 for row in report["overall_controller_table"])
    copy = dict(report)
    claimed = copy.pop("report_sha256")
    assert canonical_sha256(copy) == claimed
    round_trip = json.loads(json.dumps(report, allow_nan=False))
    round_trip_claimed = round_trip.pop("report_sha256")
    assert canonical_sha256(round_trip) == round_trip_claimed
    rendered = scorer.markdown(report)
    for required in (
        "Execution concurrency:",
        "Components S/E/A/L/T/C",
        "16/16 (1.000)",
        "0/16; 0/16; 0/16",
        "Train GPU/RAM",
        "Actor/trainable/frozen/state",
        "Training events; last loss",
        "Checkpoint SHA-256",
    ):
        assert required in rendered
    assert report["confidence_intervals"] is None


def test_invalid_completed_cell_is_visible_as_na_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    first_output = Path(queue["jobs"][0]["evaluations"][0]["output"])
    result = json.loads(first_output.read_text(encoding="utf-8"))
    result["episodes"][0]["target_success_event_count"] = 0
    _write_json(first_output, result)
    monkeypatch.setattr(scorer.matrix_runner, "_valid_training", lambda job: True)
    monkeypatch.setattr(scorer.matrix_runner, "_valid_evaluation", lambda bundle, job, expected: True)
    monkeypatch.setattr(
        scorer, "_training_history_summary", lambda checkpoint, manifest: _history_summary_fixture()
    )
    _trust_current_queue(monkeypatch)
    report = scorer.build_report(queue_path)
    assert report["status"] == "incomplete_or_invalid"
    assert report["completion"]["valid_cells"] == 11
    invalid = report["controller_task_table"][0]
    assert invalid["score"] is None
    assert invalid["status"] == "invalid_or_incomplete"
    assert any("score recomputation rejected" in issue for issue in invalid["issues"])


def test_design_rejects_any_shape_other_than_12_jobs_and_192_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["predicted_evaluation_episodes"] = 191
    _trust_current_queue(monkeypatch)
    with pytest.raises(ValueError, match="192 evaluation episodes"):
        scorer.validate_design(queue, queue_path=queue_path)


def test_score_contract_is_frozen_and_raw_numerators_are_explicit() -> None:
    assert (
        scorer.SCORE_CONTRACT_SHA256
        == "8c64242e63c3fe78b7a2e6668756ebb518c6dd6d4f992d1651acd9022381fa60"
    )
    assert sum(scorer.SCORE_CONTRACT["weights_points"].values()) == 100.0
    scored = scorer.score_evaluation(_result(scorer.TASKS[0]), scorer.TASKS[0])
    raw = scored["raw_metrics"]
    assert raw["nonterminated_episode_count"] == 16
    assert raw["survival_denominator"] == 16
    assert raw["strict_task_complete_episode_count"] == 16
    assert raw["score_evidence"]["task_event_completion"] == {
        "numerator": 16,
        "denominator": 16,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("robot_mass_kg", 1.0, "audited cf2x mass"),
        (
            "gust_applied_impulse_w_n_s",
            [[0.0, 0.0, 0.0]] * 3,
            "applied flag disagrees",
        ),
        (
            "gust_expected_impulse_w_n_s",
            [[1.0, 0.0, 0.0]] * 3,
            "expected gust impulse disagrees",
        ),
        ("gust_impulse_max_abs_error_n_s", 1.0, "max error does not recompute"),
    ],
)
def test_gust_score_rejects_unauthenticated_impulse_evidence(
    field: str, replacement: object, message: str
) -> None:
    result = _result(scorer.TASKS[2])
    result["episodes"][0][field] = deepcopy(replacement)
    result["summary"]["episodes"][0][field] = deepcopy(replacement)
    with pytest.raises(ValueError, match=message):
        scorer.score_evaluation(result, scorer.TASKS[2])


def test_nonidentical_scores_never_collapse_into_an_approximate_tie() -> None:
    rows = [
        {
            "controller": "higher",
            "macro_mean_score": 50.0000000000004,
            "worst_task_score": 40.0,
            "macro_task_event_completion_rate": 0.5,
            "rank": None,
        },
        {
            "controller": "lower",
            "macro_mean_score": 50.0,
            "worst_task_score": 40.0,
            "macro_task_event_completion_rate": 0.5,
            "rank": None,
        },
        {
            "controller": "exact-tie",
            "macro_mean_score": 50.0,
            "worst_task_score": 40.0,
            "macro_task_event_completion_rate": 0.5,
            "rank": None,
        },
    ]
    scorer._rank_overall(rows)
    assert rows[0]["rank"] == 1
    assert rows[1]["rank"] == rows[2]["rank"] == 2


def test_fallback_and_authenticated_parallel_concurrency_schemas_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    _trust_current_queue(monkeypatch)
    scorer.validate_design(queue, queue_path=queue_path)

    queue["max_concurrent_isaac_processes"] = 2
    queue["resource_limits"]["max_concurrent_isaac_processes"] = 2
    queue["concurrency_decision"] = {
        "status": "paired_smoke_pass",
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 2,
        "fallback_applied": False,
        "fallback_reason": None,
        "receipt": str(tmp_path / "paired-smoke.json"),
        "receipt_sha256": "a" * 64,
        "receipt_id": "b" * 64,
        "source_set_sha256": "c" * 64,
        "reports": [{"slot": 0}, {"slot": 1}],
        "aggregate_overlap_memory_gate": {"passed": True},
    }
    scorer.validate_design(queue, queue_path=queue_path)

    queue["concurrency_decision"]["aggregate_overlap_memory_gate"]["passed"] = False
    with pytest.raises(ValueError, match="paired-smoke"):
        scorer.validate_design(queue, queue_path=queue_path)


def test_current_matrix_dry_run_schema_validates_without_legacy_sequential_field(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs/experiments/crazyflie_task_separated_v1_balanced_v4_seed0_comparison.json"
    config = scorer.matrix_runner.validate_config(config_path)
    queue_path = tmp_path / "current-schema-queue.json"
    queue = scorer.matrix_runner.build_queue(config, queue_path)
    queue["dry_run"] = True
    queue["counts"] = {"pending": 12}
    assert "sequential_process_limit" not in queue
    assert queue["max_concurrent_isaac_processes"] in {1, 2}
    scorer.validate_design(queue, queue_path=queue_path)


def test_design_rejects_status_order_and_stale_current_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    _trust_current_queue(monkeypatch)

    wrong_status = deepcopy(queue)
    wrong_status["status"] = "failed"
    with pytest.raises(ValueError, match="queue status is inconsistent"):
        scorer.validate_design(wrong_status, queue_path=queue_path)

    wrong_order = deepcopy(queue)
    wrong_order["jobs"][0], wrong_order["jobs"][1] = (
        wrong_order["jobs"][1],
        wrong_order["jobs"][0],
    )
    with pytest.raises(ValueError, match="controller-first"):
        scorer.validate_design(wrong_order, queue_path=queue_path)

    monkeypatch.setattr(
        scorer.matrix_runner,
        "validate_resume_queue",
        lambda queue, config, output: (_ for _ in ()).throw(
            ValueError("stale source fingerprint")
        ),
    )
    with pytest.raises(ValueError, match="stale source fingerprint"):
        scorer.validate_design(queue, queue_path=queue_path)


def test_training_history_summary_reports_observed_metrics_and_counts() -> None:
    rows = []
    for update in (1, 2):
        row = {
            field: float(update) for field in scorer.TRAINING_SCALAR_METRICS
        }
        row.update(
            {
                field: update for field in scorer.TRAINING_COUNT_METRICS
            }
        )
        row.update(
            {
                "completed_updates": update,
                "total_interactions": update * 400,
                "failure_cause_counts": {
                    "1": update,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                },
            }
        )
        rows.append(row)
    summary = scorer._summarize_training_history(
        rows,
        {
            "history_sha256": "d" * 64,
            "row_count": 2,
            "segments": [{"path": "rows.jsonl", "sha256": "e" * 64, "row_count": 2}],
        },
    )
    assert summary["scalar_metrics"]["loss"] == {
        "observation_count": 2,
        "first": 1.0,
        "last": 2.0,
        "mean": 1.5,
        "minimum": 1.0,
        "maximum": 2.0,
    }
    assert summary["count_totals"]["target_success_count"] == 3
    assert summary["failure_cause_count_totals"]["1"] == 3


def test_evaluation_toctou_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    job = queue["jobs"][0]
    bundle = job["evaluations"][0]
    output = Path(bundle["output"])

    def mutate_after_initial_read(bundle: object, job: object, expected: int) -> bool:
        output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return True

    monkeypatch.setattr(
        scorer.matrix_runner, "_valid_evaluation", mutate_after_initial_read
    )
    _, _, issues = scorer._evaluation_evidence(job, bundle)
    assert any("changed while the report" in issue for issue in issues)


def _trusted_report_for_snapshot_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    queue_path = _complete_queue(tmp_path)
    monkeypatch.setattr(scorer.matrix_runner, "_valid_training", lambda job: True)
    monkeypatch.setattr(
        scorer.matrix_runner, "_valid_evaluation", lambda bundle, job, expected: True
    )
    monkeypatch.setattr(
        scorer,
        "_training_history_summary",
        lambda checkpoint, manifest: _history_summary_fixture(),
    )
    _trust_current_queue(monkeypatch)
    return queue_path, scorer.build_report(queue_path)


def test_final_snapshot_rejects_queue_and_early_cell_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path, report = _trusted_report_for_snapshot_test(tmp_path, monkeypatch)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    first_evaluation = Path(queue["jobs"][0]["evaluations"][0]["output"])
    original_evaluation = first_evaluation.read_bytes()
    first_evaluation.write_bytes(original_evaluation + b"\n")
    with pytest.raises(ValueError, match="artifact changed before publication"):
        scorer._revalidate_report_artifacts(report)

    # Restore the exact evaluation bytes, then prove the queue itself is also
    # held immutable across the complete 12-cell report construction window.
    first_evaluation.write_bytes(original_evaluation)
    queue_path.write_text(queue_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="queue changed before report publication"):
        scorer._revalidate_report_artifacts(report)


def test_final_snapshot_rejects_history_mutation_and_source_addition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = _complete_queue(tmp_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    first = queue["jobs"][0]
    checkpoint = Path(first["checkpoint"])
    manifest_path = Path(first["run_dir"]) / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment = checkpoint.parent.parent / "history" / "segment.jsonl"
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_text('{"completed_updates":1}\n', encoding="utf-8")
    manifest["history_reference"] = {
        "segments": [{"path": "../history/segment.jsonl"}]
    }
    _write_json(manifest_path, manifest)

    monkeypatch.setattr(scorer.matrix_runner, "_valid_training", lambda job: True)
    monkeypatch.setattr(
        scorer.matrix_runner, "_valid_evaluation", lambda bundle, job, expected: True
    )
    monkeypatch.setattr(
        scorer,
        "_training_history_summary",
        lambda checkpoint, manifest: _history_summary_fixture(),
    )
    _trust_current_queue(monkeypatch)
    source_state = {"extra": False}
    original_sources = scorer.source_hashes()

    def current_sources() -> dict[str, str]:
        result = dict(original_sources)
        if source_state["extra"]:
            result["scripts/new_drone_source.py"] = "f" * 64
        return result

    monkeypatch.setattr(scorer, "source_hashes", current_sources)
    report = scorer.build_report(queue_path)
    segment.write_text('{"completed_updates":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed before publication"):
        scorer._revalidate_report_artifacts(report)

    segment.write_text('{"completed_updates":1}\n', encoding="utf-8")
    source_state["extra"] = True
    with pytest.raises(ValueError, match="source set changed"):
        scorer._revalidate_report_artifacts(report)


def test_report_pair_is_no_clobber_and_rolls_back_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    scorer._publish_report_pair(json_path, "json", markdown_path, "markdown")
    assert json_path.read_text(encoding="utf-8") == "json"
    assert markdown_path.read_text(encoding="utf-8") == "markdown"
    with pytest.raises(FileExistsError):
        scorer._publish_report_pair(json_path, "new", tmp_path / "other.md", "new")
    assert json_path.read_text(encoding="utf-8") == "json"
    assert not (tmp_path / "other.md").exists()

    rollback_json = tmp_path / "rollback.json"
    rollback_markdown = tmp_path / "rollback.md"
    original_link = os.link
    calls = 0

    def fail_second_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(destination).write_text("foreign", encoding="utf-8")
            raise FileExistsError(str(destination))
        original_link(source, destination)

    monkeypatch.setattr(scorer.os, "link", fail_second_link)
    with pytest.raises(FileExistsError):
        scorer._publish_report_pair(
            rollback_json, "ours-json", rollback_markdown, "ours-markdown"
        )
    assert not rollback_json.exists()
    assert rollback_markdown.read_text(encoding="utf-8") == "foreign"


def test_concurrent_report_writers_never_clobber_or_leave_a_half_pair(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "race.json"
    markdown_path = tmp_path / "race.md"
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(label: str) -> None:
        barrier.wait()
        try:
            scorer._publish_report_pair(
                json_path, f"json-{label}", markdown_path, f"markdown-{label}"
            )
            outcomes.append(f"pass-{label}")
        except FileExistsError:
            outcomes.append(f"blocked-{label}")

    threads = [threading.Thread(target=writer, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(item.startswith("pass-") for item in outcomes) == 1
    assert sum(item.startswith("blocked-") for item in outcomes) == 1
    winner = next(item.removeprefix("pass-") for item in outcomes if item.startswith("pass-"))
    assert json_path.read_text(encoding="utf-8") == f"json-{winner}"
    assert markdown_path.read_text(encoding="utf-8") == f"markdown-{winner}"
