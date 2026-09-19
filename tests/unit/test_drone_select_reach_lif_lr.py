from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_select_reach_lif_lr as selector  # noqa: E402
from g1_fly_control.crazyflie.checkpoint import (  # noqa: E402
    build_history_reference,
    write_history_segment,
)


def _memory_samples(*, gpu_used_mib: float = 1024.0) -> list[dict[str, object]]:
    stages = (
        "environment_loaded",
        "controller_loaded",
        "rollout",
        "optimizer_update",
        "training_complete",
    )
    return [
        {
            "timestamp_utc": f"2026-09-17T00:00:0{index}+00:00",
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
            "gpu_devices": [
                {
                    "index": 0,
                    "name": "unit-gpu",
                    "driver_version": "unit",
                    "total_mib": 8192.0,
                    "used_mib": gpu_used_mib,
                }
            ],
            "torch_allocated_mib": 64.0,
            "torch_reserved_mib": 128.0,
            "torch_peak_allocated_mib": 128.0,
            "torch_peak_reserved_mib": 256.0,
        }
        for index, stage in enumerate(stages)
    ]


def _history_rows(
    *, success_events: int, final_distance: float, progress: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for update in range(1, selector.EXPECTED_UPDATES + 1):
        complete = update == selector.EXPECTED_UPDATES
        rows.append(
            {
                "completed_updates": update,
                "total_interactions": update * selector.EXPECTED_INTERACTIONS_PER_UPDATE,
                "completed_episode_count": 1 if complete else 0,
                "target_success_count": success_events if complete else 0,
                "successful_episode_count": 1 if complete and success_events else 0,
                "failure_termination_count": 0,
                "time_limit_truncation_count": 1 if complete else 0,
                "failure_cause_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
                "final_goal_distance_mean_m": final_distance if complete else None,
                "episodic_reward_component_means": {"progress": progress} if complete else {},
                "loss": 0.1,
            }
        )
    return rows


def _v4_contract() -> dict[str, object]:
    reward = {"version": "unit-balanced-v4", "progress_scale": 4.0}
    curriculum = {"version": "unit-balanced-v4-curriculum", "stages": []}
    return {
        "reward": reward,
        "reward_sha256": selector._canonical_sha256(reward),
        "training_curriculum": curriculum,
        "training_curriculum_sha256": selector._canonical_sha256(curriculum),
    }


def _resolved_config(learning_rate: float, *, rewire_manifest: Path) -> dict[str, object]:
    return {
        **selector._expected_resolved_fields(learning_rate),
        "rewire_seed": 20260916,
        "rewire_manifest": str(rewire_manifest.resolve()),
        "rewire_manifest_file_sha256": selector._sha256_file(rewire_manifest),
        "rollout_rng_contract": {"scheme": "post_construction_reseed_v1"},
        "memory_acceptance": {"policy_version": selector.EXPECTED_MEMORY_POLICY},
        "balanced_v4_task_contract": _v4_contract(),
    }


def _install_candidate(
    sandbox: Path,
    declaration: dict[str, object],
    *,
    source: Path,
    success_events: int,
    final_distance: float,
    progress: float,
    gpu_used_mib: float = 1024.0,
) -> dict[str, object]:
    run_dir = sandbox / str(declaration["run_dir"])
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unit checkpoint " + str(declaration["candidate_id"]).encode())
    rows = _history_rows(
        success_events=success_events,
        final_distance=final_distance,
        progress=progress,
    )
    history_path = run_dir / "history" / "rows.jsonl"
    segment = write_history_segment(
        history_path,
        rows,
        reference_directory=checkpoint.parent,
    )
    history_reference = build_history_reference(rows, [segment])
    learning_rate = float(declaration["learning_rate"])
    rewire_manifest = sandbox / "rewire-manifest.json"
    if not rewire_manifest.exists():
        rewire_manifest.write_text('{"unit": true}\n')
    resolved = _resolved_config(learning_rate, rewire_manifest=rewire_manifest)
    source_hashes = {str(source.resolve()): selector._sha256_file(source)}
    fingerprint_payload = {
        "schema_version": 1,
        "source_sha256": source_hashes,
        "resolved_config": resolved,
        "runtime": {"name": "unit"},
    }
    fingerprint = selector._canonical_sha256(fingerprint_payload)
    core = "c" * 64
    controller_report = {
        "controller_kind": selector.EXPECTED_CONTROLLER_KIND,
        "core_checksum": core,
    }
    samples = _memory_samples(gpu_used_mib=gpu_used_mib)
    memory_gate = selector.assess_memory(samples)
    curriculum = {"active_stage_name": "near_3d", "training_interactions": 40_000}
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "task": selector.EXPECTED_TASK,
        "contract_profile": selector.EXPECTED_PROFILE,
        "controller": selector.EXPECTED_CONTROLLER,
        "seed": selector.EXPECTED_SEED,
        "num_envs": selector.EXPECTED_NUM_ENVS,
        "horizon": selector.EXPECTED_HORIZON,
        "requested_interactions": selector.EXPECTED_INTERACTIONS,
        "environment_interactions": selector.EXPECTED_INTERACTIONS,
        "completed_updates": selector.EXPECTED_UPDATES,
        "completed_episodes": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": selector._sha256_file(checkpoint),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "resolved_config": resolved,
        "task_manifest_id": "task-manifest",
        "evaluation_manifest_id": "evaluation-manifest",
        "controller_report": controller_report,
        "core_checksum_before": core,
        "core_checksum_after": core,
        "training_curriculum": curriculum,
        "history_reference": history_reference,
        "memory_samples": samples,
        "memory_gate": memory_gate,
    }
    manifest_path = run_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checkpoint_payload = {
        "resolved_config": resolved,
        "interactions_per_update": selector.EXPECTED_INTERACTIONS_PER_UPDATE,
        "counters": {
            "completed_updates": selector.EXPECTED_UPDATES,
            "total_interactions": selector.EXPECTED_INTERACTIONS,
            "completed_episodes": 1,
        },
        "metadata": {
            "status": "completed",
            "contract_profile": selector.EXPECTED_PROFILE,
            "controller": selector.EXPECTED_CONTROLLER,
            "seed": selector.EXPECTED_SEED,
            "requested_interactions": selector.EXPECTED_INTERACTIONS,
            "interactions_per_update": selector.EXPECTED_INTERACTIONS_PER_UPDATE,
            "controller_report": controller_report,
            "core_checksum_before": core,
            "core_checksum_after": core,
            "memory_samples": samples,
            "training_curriculum": curriculum,
        },
        "core_checksum": core,
        "fingerprints": {
            "reproduction": fingerprint,
            "source_set": selector._canonical_sha256(source_hashes),
            "frozen_core": core,
        },
        "task_manifest_id": "task-manifest",
        "evaluation_manifest_id": "evaluation-manifest",
        "history": [],
        "history_reference": history_reference,
    }
    return {
        "checkpoint": checkpoint.resolve(),
        "checkpoint_payload": checkpoint_payload,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


@pytest.fixture
def screen_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "configs" / "experiments" / selector.DEFAULT_CONFIG.name
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(selector.DEFAULT_CONFIG, config_path)
    monkeypatch.setattr(selector, "ROOT", tmp_path)
    monkeypatch.setattr(selector, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(selector, "EXPECTED_CONFIG_SHA256", selector._sha256_file(config_path))
    config = json.loads(config_path.read_text())
    source = tmp_path / "source.py"
    source.write_text("FROZEN = True\n")
    artifacts = [
        _install_candidate(
            tmp_path,
            config["candidates"][0],
            source=source,
            success_events=1,
            final_distance=0.25,
            progress=2.0,
        ),
        _install_candidate(
            tmp_path,
            config["candidates"][1],
            source=source,
            success_events=2,
            final_distance=0.90,
            progress=-1.0,
        ),
    ]
    payloads = {item["checkpoint"]: item["checkpoint_payload"] for item in artifacts}

    def loader(path: Path) -> dict[str, object]:
        return deepcopy(payloads[path.resolve()])

    return {
        "config_path": config_path,
        "config": config,
        "artifacts": artifacts,
        "source": source,
        "loader": loader,
    }


def test_predeclared_config_is_exact_and_missing_runs_fail_closed(tmp_path, monkeypatch):
    config_path = tmp_path / selector.DEFAULT_CONFIG.name
    shutil.copyfile(selector.DEFAULT_CONFIG, config_path)
    monkeypatch.setattr(selector, "ROOT", tmp_path)
    monkeypatch.setattr(selector, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(selector, "EXPECTED_CONFIG_SHA256", selector._sha256_file(config_path))
    config = selector._validate_declaration(config_path)
    assert [row["learning_rate"] for row in config["candidates"]] == [1.0e-4, 3.0e-4]
    with pytest.raises(selector.SelectionGateError, match="run directory is missing"):
        selector.select_learning_rate(config_path, checkpoint_loader=lambda _: {})


def test_selector_authenticates_both_candidates_and_events_win(screen_fixture):
    result = selector.select_learning_rate(
        screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
    )
    assert result["selected_candidate_id"] == "lr-3e-4"
    assert result["selected_learning_rate"] == 3.0e-4
    assert result["held_out_evaluation_used"] is False
    assert result["all_hard_gates_passed"] is True
    output = Path(result["selection_output"])
    assert output.is_file()
    assert result["selection_output_sha256"] == selector._sha256_file(output)
    with pytest.raises(selector.SelectionGateError, match="already exists"):
        selector.select_learning_rate(
            screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
        )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        ((2, 1.0, 0.0, 3.0e-4), (1, 0.1, 100.0, 1.0e-4), "left"),
        ((2, 0.5, 0.0, 3.0e-4), (2, 0.7, 100.0, 1.0e-4), "left"),
        ((2, 0.5, 3.0, 3.0e-4), (2, 0.5, 2.0, 1.0e-4), "left"),
        ((2, 0.5, 3.0, 3.0e-4), (2, 0.5, 3.0, 1.0e-4), "right"),
    ),
)
def test_selection_order_is_lexicographic(left, right, expected):
    def evidence(name, values):
        successes, distance, progress, learning_rate = values
        return {
            "candidate_id": name,
            "learning_rate": learning_rate,
            "metrics": {
                "training_target_success_event_count": successes,
                "final_goal_distance_over_completed_pilot_episodes_m": distance,
                "cumulative_progress_reward": progress,
            },
        }

    winner, ranking = selector._select_winner(
        [evidence("left", left), evidence("right", right)]
    )
    assert winner["candidate_id"] == expected
    assert ranking[0]["candidate_id"] == expected


def test_selector_rejects_source_fingerprint_drift(screen_fixture):
    screen_fixture["source"].write_text("FROZEN = False\n")
    with pytest.raises(selector.SelectionGateError, match="fingerprinted source changed"):
        selector.select_learning_rate(
            screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
        )


def test_selector_rejects_checkpoint_hash_drift(screen_fixture):
    screen_fixture["artifacts"][0]["checkpoint"].write_bytes(b"changed")
    with pytest.raises(selector.SelectionGateError, match="checkpoint SHA-256 differs"):
        selector.select_learning_rate(
            screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
        )


def test_selector_rejects_memory_failure(screen_fixture):
    artifact = screen_fixture["artifacts"][0]
    manifest = json.loads(artifact["manifest_path"].read_text())
    samples = _memory_samples(gpu_used_mib=7000.0)
    manifest["memory_samples"] = samples
    manifest["memory_gate"] = selector.assess_memory(samples)
    artifact["manifest_path"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact["checkpoint_payload"]["metadata"]["memory_samples"] = samples
    with pytest.raises(selector.SelectionGateError, match="memory hard gate failed"):
        selector.select_learning_rate(
            screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
        )


@pytest.mark.parametrize(("code", "match"), (("1", "physical/crash"), ("4", "nonfinite")))
def test_history_metrics_rejects_physical_and_nonfinite_failures(code, match):
    rows = _history_rows(success_events=0, final_distance=0.5, progress=1.0)
    row = rows[-1]
    row["failure_cause_counts"][code] = 1
    row["failure_termination_count"] = 1
    row["time_limit_truncation_count"] = 0
    with pytest.raises(selector.SelectionGateError, match=match):
        selector._history_metrics(rows, candidate_id="unit")


def test_config_byte_change_is_rejected_before_results(screen_fixture):
    with screen_fixture["config_path"].open("a", encoding="utf-8") as stream:
        stream.write(" \n")
    with pytest.raises(selector.SelectionGateError, match="SHA-256 changed"):
        selector.select_learning_rate(
            screen_fixture["config_path"], checkpoint_loader=screen_fixture["loader"]
        )
