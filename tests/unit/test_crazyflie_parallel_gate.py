"""CPU-only tests for the authenticated two-process Crazyflie gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_parallel_gate as gate  # noqa: E402


def _sample(index: int, *, gpu: float = 5000.0, ram: float = 50.0, swap: float = 0.0):
    return {
        "timestamp_utc": f"2026-09-18T00:00:0{index}.000+00:00",
        "monotonic_seconds": float(index),
        "system_ram_percent": ram,
        "system_ram_used_mib": 1000.0,
        "system_swap_out_mib": swap,
        "gpu_devices": [
            {
                "index": 0,
                "uuid": "GPU-unit",
                "name": "unit",
                "total_mib": 8192.0,
                "used_mib": gpu,
            }
        ],
        "compute_processes": [
            {"pid": 101, "gpu_uuid": "GPU-unit", "used_mib": 2000.0},
            {"pid": 202, "gpu_uuid": "GPU-unit", "used_mib": 2000.0},
        ],
        "tracked_processes": {
            "reach": {
                "root_pid": 101,
                "root_alive": True,
                "process_tree_pids": [101],
                "matched_compute_pids": [101],
            },
            "switch": {
                "root_pid": 202,
                "root_alive": True,
                "process_tree_pids": [202],
                "matched_compute_pids": [202],
            },
        },
    }


def test_commands_are_exact_current_shape_and_isolated(tmp_path: Path):
    initial = gate.trainer_commands(tmp_path)
    resume = gate.trainer_commands(tmp_path, resume=True)
    assert [row["task"] for row in initial] == [task for _, task in gate.TASK_SPECS]
    assert len({row["run_dir"] for row in initial}) == 2
    assert len({row["checkpoint_path"] for row in initial}) == 2
    for row in initial:
        command = row["command"]
        assert command[0] == str(gate.ISAAC_PYTHON)
        assert command[1] == str(gate.TRAIN_SCRIPT)
        assert command[command.index("--num_envs") + 1] == "40"
        assert command[command.index("--total_interactions") + 1] == "8000"
        assert command[command.index("--horizon") + 1] == "100"
        assert command[command.index("--microbatch_size") + 1] == "40"
        assert command[command.index("--ppo_epochs") + 1] == "2"
        assert command[command.index("--learning_rate") + 1] == "0.0003"
        assert command[command.index("--checkpoint_every_updates") + 1] == "100"
        assert command[command.index("--pause_after_updates") + 1] == "1"
        assert "--resume" not in command
    assert all("--resume" in row["command"] for row in resume)
    assert all("--pause_after_updates" not in row["command"] for row in resume)


def test_external_telemetry_requires_real_simultaneous_compute_overlap():
    samples = [_sample(index) for index in range(4)]
    result = gate.analyze_telemetry(
        samples, expected_root_pids={"reach": 101, "switch": 202}
    )
    assert result["simultaneous_gpu_compute_overlap_sample_count"] == 4
    assert result["gpu_limit_passed"] is True
    assert result["ram_limit_passed"] is True
    assert result["no_new_sustained_swap_out_passed"] is True

    samples[0]["tracked_processes"]["reach"]["matched_compute_pids"] = []
    with pytest.raises(gate.GateValidationError, match="intersection"):
        gate.analyze_telemetry(samples)


def test_external_limits_are_exclusive_and_sustained_swap_fails():
    at_gpu_limit = [_sample(index) for index in range(4)]
    at_gpu_limit[2]["gpu_devices"][0]["used_mib"] = gate.GPU_LIMIT_MIB
    assert gate.analyze_telemetry(at_gpu_limit)["gpu_limit_passed"] is False

    at_ram_limit = [_sample(index) for index in range(4)]
    at_ram_limit[1]["system_ram_percent"] = gate.RAM_LIMIT_PERCENT
    assert gate.analyze_telemetry(at_ram_limit)["ram_limit_passed"] is False

    paging = [_sample(index, swap=float(index)) for index in range(4)]
    result = gate.analyze_telemetry(paging)
    assert result["sustained_swap_out_detected"] is True
    assert result["no_new_sustained_swap_out_passed"] is False


def test_paused_manifest_requires_clean_checkpoint_and_current_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "reach"
    checkpoint = run_dir / "checkpoints" / "update-00000001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unit-checkpoint")
    sources = {"scripts/drone_train.py": "a" * 64}
    resolved = {
        "kind": "standalone_training",
        "task": "FlyCrazyflie-WaypointReach-v0",
        "contract_profile": "balanced_v4",
        "controller": "frozen_lif_original",
        "seed": 0,
        "num_envs": 40,
        "total_interactions": 8000,
        "horizon": 100,
        "microbatch_size": 40,
        "ppo_epochs": 2,
        "learning_rate": 0.0003,
        "checkpoint_every_updates": 100,
        "evaluation_protocol": "main",
    }
    fingerprint_payload = {
        "source_sha256": sources,
        "resolved_config": resolved,
    }
    fingerprint = gate.canonical_sha256(fingerprint_payload)
    manifest = {
        "schema_version": 1,
        "status": "paused",
        "contract_profile": "balanced_v4",
        "controller": "frozen_lif_original",
        "seed": 0,
        "environment_interactions": 4000,
        "requested_interactions": 8000,
        "completed_updates": 1,
        "checkpoint": str(run_dir / "checkpoints" / "latest.pt"),
        "checkpoint_sha256": gate.sha256_file(checkpoint),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "memory_gate": {
            "passed": True,
            "failures": [],
            "device_gpu_telemetry_complete": True,
            "max_device_gpu_used_mib": 5000.0,
            "max_system_ram_percent": 50.0,
            "sustained_paging_detected": False,
        },
    }
    monkeypatch.setattr(
        gate,
        "_read_clean_checkpoint",
        lambda _path: {
            "resolved_config": resolved,
            "counters": {
                "completed_updates": 1,
                "total_interactions": 4000,
                "resume_count": 0,
            },
            "metadata": {
                "status": "paused",
                "core_checksum_before": "core",
                "core_checksum_after": "core",
            },
            "core_checksum": "core",
            "fingerprints": {"reproduction": fingerprint},
        },
    )
    result = gate.validate_paused_training(
        manifest,
        task="FlyCrazyflie-WaypointReach-v0",
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        expected_source_hashes=sources,
    )
    assert result["checkpoint_internal_validation"] == "PASS"
    stale = json.loads(json.dumps(manifest))
    stale["fingerprint_payload"]["source_sha256"] = {"changed": "b" * 64}
    stale["fingerprint"] = gate.canonical_sha256(stale["fingerprint_payload"])
    with pytest.raises(gate.GateValidationError, match="source hashes are stale"):
        gate.validate_paused_training(
            stale,
            task="FlyCrazyflie-WaypointReach-v0",
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            expected_source_hashes=sources,
        )


def test_atomic_receipt_creation_never_overwrites(tmp_path: Path):
    target = tmp_path / "receipt.json"
    gate._atomic_create_json(target, {"first": True})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        gate._atomic_create_json(target, {"second": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"first": True}
