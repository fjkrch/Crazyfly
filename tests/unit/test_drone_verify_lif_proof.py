from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_verify_lif_proof as proof  # noqa: E402
import drone_run_matrix as matrix  # noqa: E402
from drone_evaluate import AUDITED_CRAZYFLIE_MASS_KG  # noqa: E402
from drone_evaluation_protocol import SCENARIOS, load_protocol  # noqa: E402
from g1_fly_control.tasks.crazyflie.logic import balanced_task_contract_payload  # noqa: E402


def _memory_samples() -> list[dict[str, object]]:
    stages = (
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    )
    return [
        {
            "timestamp_utc": f"2026-09-16T00:00:0{index}+00:00",
            "stage": stage,
            "step": index,
            "process_rss_mib": 1000.0,
            "system_ram_total_gib": 32.0,
            "system_ram_used_gib": 12.0,
            "system_ram_available_gib": 20.0,
            "system_ram_percent": 40.0,
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
                "used_mib": 1024.0,
            }],
            "torch_allocated_mib": 64.0,
            "torch_reserved_mib": 128.0,
            "torch_peak_allocated_mib": 128.0,
            "torch_peak_reserved_mib": 256.0,
        }
        for index, stage in enumerate(stages)
    ]


def _training_fixture(tmp_path: Path):
    checkpoint = (tmp_path / "run" / "checkpoints" / "latest.pt").resolve()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unit-checkpoint")
    checkpoint_sha = proof.sha256_file(checkpoint)
    expected_config = {
        "kind": "standalone_training",
        "task": proof.EXPECTED_TASK,
        "contract_profile": proof.EXPECTED_PROFILE,
        "controller": proof.EXPECTED_CONTROLLER,
        "seed": proof.EXPECTED_SEED,
        "num_envs": proof.EXPECTED_NUM_ENVS,
        "total_interactions": proof.EXPECTED_INTERACTIONS,
        "horizon": proof.EXPECTED_HORIZON,
        "evaluation_protocol": proof.EXPECTED_PROTOCOL,
        "balanced_task_contract": balanced_task_contract_payload(),
        "mixed_scenario_contract": {"seed": 0, "schedule": "unit"},
    }
    report = {
        "controller_kind": proof.EXPECTED_CONTROLLER_KIND,
        "core_checksum": "core-checksum",
        "connectome_checksum": "connectome-checksum",
    }
    current_payload = {
        "source_sha256": {"scripts/unit.py": "source-checksum"},
        "rewired_manifest_sha256": "rewire-checksum",
        "resolved_config": expected_config,
    }
    current_fingerprint = "f" * 64
    protocol = {"manifest_id": "lif-proof-manifest", "evaluation_seed": 101}
    samples = _memory_samples()
    gate = proof.assess_memory(samples)
    curriculum = proof._expected_curriculum_snapshot(expected_config)
    mixed = {"contract": expected_config["mixed_scenario_contract"]}
    history = [
        {
            "completed_updates": index,
            "total_interactions": index * proof.EXPECTED_INTERACTIONS_PER_UPDATE,
            "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            "metrics": {"loss": 0.25, "note": None, "finite": True},
        }
        for index in range(1, proof.EXPECTED_UPDATES + 1)
    ]
    history_reference = {
        "row_count": proof.EXPECTED_UPDATES,
        "last_completed_updates": proof.EXPECTED_UPDATES,
        "last_total_interactions": proof.EXPECTED_INTERACTIONS,
    }
    payload = {
        "policy_class": "unit.FrozenLIF",
        "resolved_config": expected_config,
        "interactions_per_update": proof.EXPECTED_INTERACTIONS_PER_UPDATE,
        "counters": {
            "completed_updates": proof.EXPECTED_UPDATES,
            "total_interactions": proof.EXPECTED_INTERACTIONS,
        },
        "history": history,
        "history_reference": history_reference,
        "metadata": {
            "status": "completed",
            "contract_profile": proof.EXPECTED_PROFILE,
            "controller": proof.EXPECTED_CONTROLLER,
            "seed": proof.EXPECTED_SEED,
            "requested_interactions": proof.EXPECTED_INTERACTIONS,
            "interactions_per_update": proof.EXPECTED_INTERACTIONS_PER_UPDATE,
            "controller_report": report,
            "core_checksum_before": report["core_checksum"],
            "core_checksum_after": report["core_checksum"],
            "training_curriculum": curriculum,
            "mixed_scenario": mixed,
            "memory_samples": samples,
        },
        "core_checksum": report["core_checksum"],
        "fingerprints": {
            "reproduction": current_fingerprint,
            "source_set": proof.canonical_sha256(current_payload["source_sha256"]),
            "connectome": report["connectome_checksum"],
            "frozen_core": report["core_checksum"],
            "rewire_manifest": current_payload["rewired_manifest_sha256"],
        },
        "task_manifest_id": "task-manifest",
        "evaluation_manifest_id": protocol["manifest_id"],
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": proof.EXPECTED_TASK,
        "contract_profile": proof.EXPECTED_PROFILE,
        "controller": proof.EXPECTED_CONTROLLER,
        "seed": proof.EXPECTED_SEED,
        "num_envs": proof.EXPECTED_NUM_ENVS,
        "horizon": proof.EXPECTED_HORIZON,
        "requested_interactions": proof.EXPECTED_INTERACTIONS,
        "environment_interactions": proof.EXPECTED_INTERACTIONS,
        "completed_updates": proof.EXPECTED_UPDATES,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "fingerprint": current_fingerprint,
        "fingerprint_payload": current_payload,
        "resolved_config": expected_config,
        "task_manifest_id": payload["task_manifest_id"],
        "evaluation_manifest_id": protocol["manifest_id"],
        "controller_report": report,
        "core_checksum_before": report["core_checksum"],
        "core_checksum_after": report["core_checksum"],
        "training_curriculum": curriculum,
        "mixed_scenario": mixed,
        "history_reference": history_reference,
        "memory_samples": samples,
        "memory_gate": gate,
    }
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "expected_config": expected_config,
        "report": report,
        "current_payload": current_payload,
        "current_fingerprint": current_fingerprint,
        "protocol": protocol,
        "payload": payload,
        "manifest": manifest,
    }


def _validate_training(fixture):
    return proof._validate_checkpoint_and_manifest(
        checkpoint=fixture["checkpoint"],
        checkpoint_sha256=fixture["checkpoint_sha"],
        payload=fixture["payload"],
        manifest=fixture["manifest"],
        expected_config=fixture["expected_config"],
        expected_policy_class="unit.FrozenLIF",
        expected_controller_report=fixture["report"],
        current_fingerprint=fixture["current_fingerprint"],
        current_fingerprint_payload=fixture["current_payload"],
        protocol=fixture["protocol"],
    )


def test_exact_500k_original_lif_checkpoint_and_training_manifest_pass(tmp_path):
    fixture = _training_fixture(tmp_path)
    gate = _validate_training(fixture)
    assert gate["passed"] is True
    assert fixture["payload"]["counters"] == {
        "completed_updates": 1250,
        "total_interactions": 500_000,
    }


@pytest.mark.parametrize(
    ("corruption", "match"),
    (
        ("missing_counts", "lacks exact failure-cause counts"),
        ("nonfinite_state", "records a nonfinite-state termination"),
        ("nan_metric", "contains a nonfinite float"),
        ("counter_gap", "breaks exact update/interaction continuity"),
    ),
)
def test_training_gate_rejects_bad_or_nonfinite_authenticated_history(
    tmp_path, corruption, match
):
    fixture = _training_fixture(tmp_path)
    row = fixture["payload"]["history"][17]
    if corruption == "missing_counts":
        row.pop("failure_cause_counts")
    elif corruption == "nonfinite_state":
        row["failure_cause_counts"]["4"] = 1
    elif corruption == "nan_metric":
        row["metrics"]["loss"] = float("nan")
    elif corruption == "counter_gap":
        row["total_interactions"] += proof.EXPECTED_INTERACTIONS_PER_UPDATE
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)
    with pytest.raises(proof.ProofGateError, match=match):
        _validate_training(fixture)


def test_proof_curriculum_snapshot_uses_real_inclusive_start_interactions_contract():
    contract = balanced_task_contract_payload()
    stages = contract["training_curriculum"]["stages"]
    assert [stage["start_interactions"] for stage in stages] == [
        0,
        200_000,
        500_000,
        1_000_000,
    ]
    assert all("min_total_interactions" not in stage for stage in stages)
    snapshot = proof._expected_curriculum_snapshot(
        {"balanced_task_contract": contract}
    )
    assert snapshot == {
        "training_interactions": 500_000,
        "active_stage_index": 2,
        "active_stage_name": stages[2]["name"],
        "active_stage": stages[2],
    }


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    (
        ("counters", "total_interactions", 499_600, "500,000 interactions"),
        ("counters", "completed_updates", 1249, "1,250 updates"),
        ("metadata", "controller", "frozen_lif_degree_rewired", "controller"),
        ("metadata", "contract_profile", "survival_v2", "contract_profile"),
        ("manifest", "status", "paused", "status"),
        ("manifest", "checkpoint_sha256", "0" * 64, "checkpoint_sha256"),
        ("manifest", "environment_interactions", 499_600, "environment_interactions"),
    ),
)
def test_training_gate_rejects_wrong_identity_or_boundary(
    tmp_path, target, field, value, match
):
    fixture = _training_fixture(tmp_path)
    if target in {"counters", "metadata"}:
        fixture["payload"][target][field] = value
    else:
        fixture[target][field] = value
    with pytest.raises(proof.ProofGateError, match=match):
        _validate_training(fixture)


def test_training_gate_rejects_stale_source_fingerprint(tmp_path):
    fixture = _training_fixture(tmp_path)
    fixture["current_payload"]["source_sha256"]["scripts/unit.py"] = "changed"
    with pytest.raises(proof.ProofGateError, match="fingerprints are stale"):
        _validate_training(fixture)


def test_training_gate_rejects_changed_frozen_core(tmp_path):
    fixture = _training_fixture(tmp_path)
    fixture["payload"]["metadata"]["core_checksum_after"] = "mutated"
    with pytest.raises(proof.ProofGateError, match="core changed"):
        _validate_training(fixture)


def test_training_gate_rejects_forged_or_over_limit_memory(tmp_path):
    fixture = _training_fixture(tmp_path)
    fixture["manifest"]["memory_samples"][0]["gpu_devices"][0]["used_mib"] = (
        proof.GPU_LIMIT_MIB
    )
    with pytest.raises(proof.ProofGateError, match="memory gate does not recompute"):
        _validate_training(fixture)

    fixture = _training_fixture(tmp_path / "actual-limit")
    fixture["manifest"]["memory_samples"][0]["gpu_devices"][0]["used_mib"] = (
        proof.GPU_LIMIT_MIB
    )
    fixture["payload"]["metadata"]["memory_samples"] = deepcopy(
        fixture["manifest"]["memory_samples"]
    )
    fixture["manifest"]["memory_gate"] = proof.assess_memory(
        fixture["manifest"]["memory_samples"]
    )
    with pytest.raises(proof.ProofGateError, match="memory gate failed"):
        _validate_training(fixture)


def test_training_gate_accepts_policy_v2_monotonic_rss_warning(tmp_path):
    fixture = _training_fixture(tmp_path)
    samples = _memory_samples()
    samples.extend(
        {
            **samples[-1],
            "stage": "optimizer_update",
            "process_rss_mib": rss,
        }
        for rss in (1000.0, 1001.0, 1002.0, 1003.0)
    )
    gate = proof.assess_memory(samples)
    assert gate["policy_version"] == proof.EXPECTED_MEMORY_POLICY_VERSION
    assert gate["passed"] is True
    assert gate["monotonic_process_growth_detected"] is True
    assert gate["warnings"]
    fixture["payload"]["metadata"]["memory_samples"] = deepcopy(samples)
    fixture["manifest"]["memory_samples"] = deepcopy(samples)
    fixture["manifest"]["memory_gate"] = gate
    assert _validate_training(fixture)["passed"] is True


def _evaluation_fixture(tmp_path: Path):
    checkpoint = (tmp_path / "run" / "checkpoints" / "latest.pt").resolve()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    current_payload = {"resolved_config": {"proof": True}}
    report = {"controller_kind": proof.EXPECTED_CONTROLLER_KIND}
    protocol = {"manifest_id": "lif-proof-manifest", "evaluation_seed": 101}
    proof_plans = load_protocol("lif_proof")
    results = {}
    flattened = []
    scenario_memory_gates = {}
    for scenario in SCENARIOS:
        episodes = []
        for episode_id in range(proof.EXPECTED_EPISODES_PER_SCENARIO):
            plan = proof_plans["scenarios"][scenario][episode_id]
            row = {
                "scenario": scenario,
                "episode_id": episode_id,
                "success": scenario == SCENARIOS[0] and episode_id == 0,
                "invalid_state": False,
                "target_success_event_count": (
                    1 if episode_id == 0 and scenario in SCENARIOS[:2] else 0
                ),
                "plan": plan,
                "plan_sha256": plan["plan_sha256"],
                "robot_mass_kg": None,
                "gust_applied_impulse_w_n_s": [[0.0, 0.0, 0.0] for _ in range(3)],
                "gust_expected_impulse_w_n_s": [[0.0, 0.0, 0.0] for _ in range(3)],
                "gust_impulse_max_abs_error_n_s": None,
                "switch_outcomes": [],
                "gust_outcomes": [],
            }
            if scenario == SCENARIOS[1]:
                row["switch_outcomes"] = [
                    {"success": False} for _ in range(3)
                ]
            if scenario == SCENARIOS[2]:
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
                row["robot_mass_kg"] = AUDITED_CRAZYFLIE_MASS_KG
                row["gust_applied_impulse_w_n_s"] = [
                    list(vector) for vector in expected_impulses
                ]
                row["gust_expected_impulse_w_n_s"] = [
                    list(vector) for vector in expected_impulses
                ]
                row["gust_impulse_max_abs_error_n_s"] = 0.0
                row["gust_outcomes"] = [
                    {
                        "gust_index": gust_index,
                        "applied": True,
                        "recovered": episode_id == 0 and gust_index == 0,
                    }
                    for gust_index in range(3)
                ]
            episodes.append(row)
        flattened.extend(episodes)
        scenario_gate = proof.assess_memory(_memory_samples())
        scenario_memory_gates[scenario] = scenario_gate
        results[scenario] = {
            "episodes": episodes,
            "summary": {
                "success_count": sum(row["success"] for row in episodes),
                "invalid_state_count": 0,
            },
            "memory_samples": [],
            "memory_gate": scenario_gate,
        }
    evaluation = {
        "schema_version": 1,
        "status": "completed",
        "label": "lif_proof",
        "protocol": proof.EXPECTED_PROTOCOL,
        "scenario": "all",
        "episodes": flattened,
        "scenario_results": results,
        "total_episodes": len(SCENARIOS) * proof.EXPECTED_EPISODES_PER_SCENARIO,
        "evaluation_seed": 101,
        "evaluation_manifest_id": protocol["manifest_id"],
        "training_seed": proof.EXPECTED_SEED,
        "controller": proof.EXPECTED_CONTROLLER,
        "deterministic_actions": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": proof.sha256_file(checkpoint),
        "fingerprint": "f" * 64,
        "fingerprint_payload": current_payload,
        "controller_report": report,
        "generated_at_utc": "2026-09-16T00:00:00+00:00",
        "memory_gate": {
            "policy_version": proof.EXPECTED_MEMORY_POLICY_VERSION,
            "passed": True,
            "failures": [],
            "warnings": [],
            "device_gpu_telemetry_complete": True,
            "max_device_gpu_used_mib": 1024.0,
            "max_system_ram_percent": 40.0,
            "max_process_rss_mib": 1000.0,
            "monotonic_process_growth_detected": False,
            "sustained_paging_detected": False,
            "limits": next(iter(scenario_memory_gates.values()))["limits"],
            "scenario_gates": scenario_memory_gates,
        },
    }
    return checkpoint, protocol, report, current_payload, evaluation


def _validate_evaluation(fixture):
    checkpoint, protocol, report, current_payload, evaluation = fixture
    rebuilt = deepcopy(evaluation)
    rebuilt["generated_at_utc"] = "rebuilt-at-a-different-time"
    return proof._validate_evaluation(
        evaluation_path=checkpoint.parents[1] / "evaluation.json",
        evaluation=evaluation,
        rebuilt=rebuilt,
        checkpoint=checkpoint,
        checkpoint_sha256=proof.sha256_file(checkpoint),
        current_fingerprint="f" * 64,
        current_fingerprint_payload=current_payload,
        expected_controller_report=report,
        protocol=protocol,
    )


def test_evaluation_requires_one_boolean_success_in_every_scenario(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    # _atomic_json(sort_keys=True) alphabetizes nested mappings when the real
    # merged artifact is reloaded; scenario identity must not depend on that
    # serialization order.
    fixture[-1]["scenario_results"] = {
        scenario: fixture[-1]["scenario_results"][scenario]
        for scenario in sorted(SCENARIOS)
    }
    scores, memory = _validate_evaluation(fixture)
    assert {
        scenario: score["event_episode_count"] for scenario, score in scores.items()
    } == {scenario: 1 for scenario in SCENARIOS}
    assert scores[SCENARIOS[0]]["strict_full_episode_success_count"] == 1
    assert scores[SCENARIOS[1]]["strict_full_episode_success_count"] == 0
    assert scores[SCENARIOS[2]]["strict_full_episode_success_count"] == 0
    assert all(score["event_score"] == "1/5" for score in scores.values())
    assert all(score["event_episode_rate"] == pytest.approx(0.2) for score in scores.values())
    assert memory["passed"] is True


def test_evaluation_accepts_policy_v2_monotonic_rss_warning(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    scenario = SCENARIOS[0]
    warning = "process RSS rose monotonically in the steady-state window"
    gate = fixture[-1]["memory_gate"]["scenario_gates"][scenario]
    gate["monotonic_process_growth_detected"] = True
    gate["warnings"] = [warning]
    fixture[-1]["memory_gate"]["monotonic_process_growth_detected"] = True
    fixture[-1]["memory_gate"]["warnings"] = [f"{scenario}: {warning}"]

    scores, memory = _validate_evaluation(fixture)

    assert scores[scenario]["gate_passed"] is True
    assert memory["warnings"] == [f"{scenario}: {warning}"]


def test_evaluation_rejects_non_v2_memory_policy(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    scenario = SCENARIOS[0]
    fixture[-1]["memory_gate"]["scenario_gates"][scenario]["policy_version"] = "legacy"
    with pytest.raises(proof.ProofGateError, match="RAM/VRAM/paging gate failed"):
        _validate_evaluation(fixture)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_evaluation_rejects_zero_success_in_each_scenario(tmp_path, scenario):
    fixture = _evaluation_fixture(tmp_path)
    evaluation = fixture[-1]
    for row in evaluation["scenario_results"][scenario]["episodes"]:
        row["success"] = False
        row["target_success_event_count"] = 0
        for outcome in row["switch_outcomes"]:
            outcome["success"] = False
        for outcome in row["gust_outcomes"]:
            outcome["recovered"] = False
    evaluation["scenario_results"][scenario]["summary"]["success_count"] = 0
    with pytest.raises(
        proof.ProofGateError,
        match=f"zero held-out event-success episodes for {scenario}",
    ):
        _validate_evaluation(fixture)


@pytest.mark.parametrize(
    ("corruption", "match"),
    (
        ("missing_impulse", "exactly three gust vectors"),
        ("wrong_expected", "Recorded expected gust impulse"),
        ("wrong_applied", "Submitted gust impulse differs"),
        ("wrong_max_error", "max error does not recompute"),
    ),
)
def test_evaluation_proof_scorer_rejects_tampered_gust_evidence(
    tmp_path, corruption, match
):
    fixture = _evaluation_fixture(tmp_path)
    row = fixture[-1]["scenario_results"][SCENARIOS[2]]["episodes"][0]
    if corruption == "missing_impulse":
        row.pop("gust_applied_impulse_w_n_s")
    elif corruption == "wrong_expected":
        row["gust_expected_impulse_w_n_s"][0][0] += 0.01
    elif corruption == "wrong_applied":
        row["gust_applied_impulse_w_n_s"][0][0] += 0.01
    elif corruption == "wrong_max_error":
        row["gust_impulse_max_abs_error_n_s"] = 0.01
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)
    with pytest.raises(proof.ProofGateError, match=match):
        _validate_evaluation(fixture)


def test_evaluation_rejects_truthy_non_boolean_success_and_invalid_state(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    scenario = SCENARIOS[0]
    fixture[-1]["scenario_results"][scenario]["episodes"][0]["success"] = 1
    with pytest.raises(proof.ProofGateError, match="not Boolean"):
        _validate_evaluation(fixture)

    fixture = _evaluation_fixture(tmp_path / "invalid")
    fixture[-1]["scenario_results"][scenario]["episodes"][0]["invalid_state"] = True
    fixture[-1]["scenario_results"][scenario]["summary"]["invalid_state_count"] = 1
    with pytest.raises(proof.ProofGateError, match="nonfinite/invalid state"):
        _validate_evaluation(fixture)


def test_evaluation_rejects_missing_or_inconsistent_episode_event_count(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    scenario = SCENARIOS[0]
    fixture[-1]["scenario_results"][scenario]["episodes"][0].pop(
        "target_success_event_count"
    )
    with pytest.raises(proof.ProofGateError, match="target-success event count is invalid"):
        _validate_evaluation(fixture)

    fixture = _evaluation_fixture(tmp_path / "inconsistent")
    fixture[-1]["scenario_results"][scenario]["episodes"][0][
        "target_success_event_count"
    ] = 0
    with pytest.raises(proof.ProofGateError, match="Strict episode success disagrees"):
        _validate_evaluation(fixture)


def test_evaluation_rejects_published_bundle_that_differs_from_rebuilt_parts(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    fixture[-1]["deterministic_actions"] = False
    checkpoint, protocol, report, current_payload, evaluation = fixture
    rebuilt = deepcopy(evaluation)
    rebuilt["deterministic_actions"] = True
    with pytest.raises(proof.ProofGateError, match="differs from its strictly rebuilt"):
        proof._validate_evaluation(
            evaluation_path=tmp_path / "evaluation.json",
            evaluation=evaluation,
            rebuilt=rebuilt,
            checkpoint=checkpoint,
            checkpoint_sha256=proof.sha256_file(checkpoint),
            current_fingerprint="f" * 64,
            current_fingerprint_payload=current_payload,
            expected_controller_report=report,
            protocol=protocol,
        )


def test_evaluation_rejects_failed_scenario_memory_gate(tmp_path):
    fixture = _evaluation_fixture(tmp_path)
    scenario = SCENARIOS[2]
    gate = fixture[-1]["memory_gate"]["scenario_gates"][scenario]
    gate["sustained_paging_detected"] = True
    with pytest.raises(proof.ProofGateError, match="RAM/VRAM/paging gate failed"):
        _validate_evaluation(fixture)


def test_part_paths_and_sha_must_match_the_attempt_directory(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    attempt = "attempt-one"
    paths = proof._scenario_part_paths(evaluation_path, attempt)
    for scenario, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"scenario": scenario}), encoding="utf-8")
    merged = {
        "scenario_attempt_id": attempt,
        "scenario_part_artifacts": {
            scenario: {
                "path": str(paths[scenario].resolve()),
                "sha256": proof.sha256_file(paths[scenario]),
            }
            for scenario in sorted(SCENARIOS)
        },
    }
    assert proof._validated_part_paths(evaluation_path, merged) == paths

    merged["scenario_part_artifacts"][SCENARIOS[0]]["sha256"] = "0" * 64
    with pytest.raises(proof.ProofGateError, match="SHA-256 changed"):
        proof._validated_part_paths(evaluation_path, merged)


def test_verifier_defaults_are_fixed_and_missing_proof_is_read_only(tmp_path):
    before = list(tmp_path.rglob("*"))
    with pytest.raises(proof.ProofGateError, match="Missing proof checkpoint"):
        proof.verify_lif_proof(
            checkpoint=tmp_path / "missing.pt",
            training_manifest=tmp_path / "missing-training.json",
            evaluation=tmp_path / "missing-evaluation.json",
            main_config=tmp_path / "missing-main.json",
        )
    assert list(tmp_path.rglob("*")) == before
    assert proof.PROOF_CHECKPOINT == proof.ROOT / "runs/crazyflie-balanced-v3-lif-proof/checkpoints/latest.pt"
    assert proof.PROOF_EVALUATION == proof.ROOT / "runs/crazyflie-balanced-v3-lif-proof/evaluation.json"


def test_main_launcher_stops_on_proof_failure_before_queue_code_or_mutation(tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs" / "experiments"
    runs = project / "runs"
    scripts.mkdir(parents=True)
    configs.mkdir(parents=True)
    runs.mkdir(parents=True)
    launcher_source = ROOT / "scripts" / "execute_drone_matrix.sh"
    launcher = scripts / launcher_source.name
    shutil.copyfile(launcher_source, launcher)
    (configs / "crazyflie_balanced_v3_main.json").write_text("{}\n", encoding="utf-8")
    queue = runs / "crazyflie_balanced_v3_main_v1.json"
    original_queue = b'{"reviewed": true}\n'
    queue.write_bytes(original_queue)

    invocation_log = tmp_path / "invocations.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$PROOF_TEST_LOG\"\n"
        "if [[ \"$1\" == *drone_verify_lif_proof.py ]]; then exit 7; fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(launcher), "--authorize_main"],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "ISAAC_PYTHON": str(fake_python),
            "PROOF_TEST_LOG": str(invocation_log),
        },
    )
    assert completed.returncode == 7
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert invocations == [str(scripts / "drone_verify_lif_proof.py")]
    assert queue.read_bytes() == original_queue


def test_main_launcher_orders_gate_before_authorization_and_runner():
    source = (ROOT / "scripts" / "execute_drone_matrix.sh").read_text(encoding="utf-8")
    gate = source.index("drone_verify_lif_proof.py")
    queue_rebuild = source.index("import drone_run_matrix as matrix")
    authorization = source.index("MAIN MATRIX AUTHORIZATION ACCEPTED")
    runner = source.index("--execute")
    assert gate < queue_rebuild < authorization < runner


def test_direct_execute_cannot_mutate_main_queue_without_verified_proof(tmp_path):
    queue = {"label": "main", "dry_run": True, "status": "pending"}
    before = deepcopy(queue)
    with pytest.raises(ValueError, match="requires the exact passing"):
        matrix.execute(queue, tmp_path / "main.json")
    assert queue == before


def test_direct_main_cli_checks_proof_before_queue_lock_or_mutation(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "main.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "reviewed.json"
    original = b'{"reviewed": true}\n'
    output.write_bytes(original)
    config = {"label": "main", "_config_path": str(config_path.resolve())}
    monkeypatch.setattr(matrix, "validate_config", lambda _path: config)
    calls: list[str] = []

    def reject_proof(_config, _path):
        calls.append("proof")
        raise proof.ProofGateError("unit proof absent")

    def forbidden_lock(_path):
        calls.append("lock")
        raise AssertionError("queue lock must not be reached")

    monkeypatch.setattr(matrix, "_verify_main_lif_proof", reject_proof)
    monkeypatch.setattr(matrix, "queue_lock", forbidden_lock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drone_run_matrix.py",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--execute",
            "--resume",
            "--authorize_main",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        matrix.main()
    assert raised.value.code == 2
    assert calls == ["proof"]
    assert output.read_bytes() == original
    assert "Main empirical proof gate failed" in capsys.readouterr().err
