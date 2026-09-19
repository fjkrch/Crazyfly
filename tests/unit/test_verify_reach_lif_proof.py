from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_reach_lif_proof as proof  # noqa: E402


def _history_rows() -> list[dict[str, object]]:
    config, _ = proof._expected_resolved_config(1.0e-4)
    rows: list[dict[str, object]] = []
    completed = 0
    for update in range(1, proof.EXPECTED_UPDATES + 1):
        interactions = update * proof.EXPECTED_INTERACTIONS_PER_UPDATE
        snapshot = proof._expected_curriculum_snapshot(config, interactions)
        finished = 4 if update % 6 == 0 else 0
        completed += finished
        rows.append({
            "completed_updates": update,
            "total_interactions": interactions,
            "rollout_start_interactions": (
                interactions - proof.EXPECTED_INTERACTIONS_PER_UPDATE
            ),
            "training_curriculum_interactions": interactions,
            "active_training_curriculum_stage_index": snapshot["active_stage_index"],
            "active_training_curriculum_stage_name": snapshot["active_stage_name"],
            "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            "completed_episode_count": finished,
            "target_success_count": 0,
            "successful_episode_count": 0,
            "failure_termination_count": 0,
            "time_limit_truncation_count": finished,
            "completed_episodes": completed,
            "rejected_step": False,
            "loss": 0.1,
        })
    return rows


def test_exact_balanced_v4_reach_history_passes_and_recomputes():
    config, protocol = proof._expected_resolved_config(1.0e-4)
    summary = proof._validate_training_history(_history_rows(), config)

    assert config["task"] == proof.EXPECTED_TASK
    assert config["contract_profile"] == proof.EXPECTED_PROFILE
    assert config["seed"] == 5
    assert config["total_interactions"] == 500_000
    assert config["checkpoint_every_updates"] == 100
    assert protocol["evaluation_seed"] == 101
    assert summary == {
        "completed_episodes": (proof.EXPECTED_UPDATES // 6) * 4,
        "target_success_events": 0,
        "strict_successes": 0,
        "time_limit_truncations": (proof.EXPECTED_UPDATES // 6) * 4,
        "ppo_rejected_updates": 0,
        "physical_failures": 0,
        "nonfinite_failures": 0,
    }


def test_training_history_reports_safely_rolled_back_ppo_updates():
    config, _ = proof._expected_resolved_config(1.0e-4)
    rows = _history_rows()
    rows[17]["rejected_step"] = True
    rows[918]["rejected_step"] = True

    summary = proof._validate_training_history(rows, config)

    assert summary["ppo_rejected_updates"] == 2


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("gap", "continuity"),
        ("crash", "crash/OOB/nonfinite"),
        ("nonfinite", "Nonfinite float"),
        ("rejected_type", "invalid rejected_step evidence"),
        ("stage", "curriculum stage"),
    ),
)
def test_training_history_fails_closed(mutation: str, match: str):
    config, _ = proof._expected_resolved_config(1.0e-4)
    rows = _history_rows()
    row = rows[17]
    if mutation == "gap":
        row["total_interactions"] = int(row["total_interactions"]) + 400
    elif mutation == "crash":
        row["failure_cause_counts"]["1"] = 1  # type: ignore[index]
    elif mutation == "nonfinite":
        row["loss"] = math.nan
    elif mutation == "rejected_type":
        row["rejected_step"] = 1
    elif mutation == "stage":
        row["active_training_curriculum_stage_name"] = "forged"
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)
    with pytest.raises(proof.ReachProofGateError, match=match):
        proof._validate_training_history(rows, config)


def _history_reference(
    *, second_resume: int = 1, gap: int = 0, split_update: int = 100
) -> dict[str, object]:
    return {
        "segments": [
            {
                "path": "../history/rows-00000001-00000100-resume-0000-1.jsonl",
                "last_completed_updates": split_update,
                "last_total_interactions": split_update * 400,
            },
            {
                "path": f"../history/rows-00000101-00001250-resume-{second_resume:04d}-2.jsonl",
                "first_completed_updates": split_update + 1 + gap,
                "first_total_interactions": (split_update + 1 + gap) * 400,
            },
        ]
    }


def test_resume_generation_evidence_requires_one_contiguous_resume():
    assert proof._resume_split_from_history_reference(_history_reference()) == 100

    with pytest.raises(proof.ReachProofGateError, match="generations 0 and 1"):
        proof._resume_split_from_history_reference(_history_reference(second_resume=2))
    with pytest.raises(proof.ReachProofGateError, match="update-100 resume boundary"):
        proof._resume_split_from_history_reference(_history_reference(gap=1))
    with pytest.raises(proof.ReachProofGateError, match="update-100 resume boundary"):
        proof._resume_split_from_history_reference(_history_reference(split_update=99))


def _reach_evaluation(*, success: bool = True, crash: bool = False) -> dict[str, object]:
    episodes = []
    for episode in range(proof.EXPECTED_EPISODES):
        event = 1 if success and episode == 0 else 0
        episodes.append({
            "target_success_event_count": event,
            "success": bool(event),
            "crash": crash and episode == 1,
            "out_of_bounds": False,
            "invalid_state": False,
            "terminated": crash and episode == 1,
            "truncated": not (crash and episode == 1),
            "completed_steps": 12 if crash and episode == 1 else 600,
            "failure_reason": "ground_or_low_height" if crash and episode == 1 else None,
            "time_to_first_success_s": 2.0 if event else None,
        })
    return {
        "episodes": episodes,
        "summary": {
            "episode_count": 5,
            "expected_episode_count": 5,
            "success_count": int(success),
            "crash_count": int(crash),
            "out_of_bounds_count": 0,
            "invalid_state_count": 0,
            "termination_count": int(crash),
            "truncation_count": 5 - int(crash),
        },
    }


def test_reach_outcome_gate_accepts_one_of_five_genuine_events():
    assert proof._validate_reach_outcomes(_reach_evaluation()) == {
        "event_episode_count": 1,
        "strict_success_count": 1,
        "crash_count": 0,
        "out_of_bounds_count": 0,
        "invalid_state_count": 0,
    }


def test_reach_outcome_gate_rejects_zero_events_and_any_crash():
    with pytest.raises(proof.ReachProofGateError, match="zero genuine"):
        proof._validate_reach_outcomes(_reach_evaluation(success=False))
    with pytest.raises(proof.ReachProofGateError, match="crashed, escaped"):
        proof._validate_reach_outcomes(_reach_evaluation(crash=True))


def _memory_samples() -> list[dict[str, object]]:
    stages = (
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    )
    return [
        {
            "timestamp_utc": f"2026-09-17T00:00:{index:02d}+00:00",
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


def test_training_memory_gate_recomputes_and_requires_two_processes():
    samples = _memory_samples()
    gate = proof.assess_memory(samples)
    assert proof._validate_memory(samples, gate, label="Training")["passed"] is True

    one_process = samples[4:]
    one_process_gate = proof.assess_memory(one_process)
    with pytest.raises(proof.ReachProofGateError, match="two trainer processes"):
        proof._validate_memory(one_process, one_process_gate, label="Training")


def test_profile_hashes_recompute_separately():
    config, _ = proof._expected_resolved_config(3.0e-4)
    hashes = proof._profile_hashes(config)
    assert set(hashes) == {
        "task_contract_sha256",
        "reward_sha256",
        "training_curriculum_sha256",
        "switch_targets_sha256",
        "reward_version_sha256",
        "training_curriculum_version_sha256",
        "switch_targets_version_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_failure_output_reports_verifier_sha256(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_reach_lif_proof.py",
            "--selection",
            str(missing),
            "--checkpoint",
            str(missing),
            "--training_manifest",
            str(missing),
            "--evaluation",
            str(missing),
        ],
    )
    assert proof.main() == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "FAIL"
    assert failure["verifier"] == str(proof.SCRIPT_PATH)
    assert failure["verifier_sha256"] == proof.sha256_file(proof.SCRIPT_PATH)
