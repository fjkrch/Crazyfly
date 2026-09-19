"""Pure-CPU safety gates for the stock G1/Go1 memory-smoke supervisor."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import default_locomotion_memory_smoke_v1 as smoke  # noqa: E402
import default_locomotion_queue_v1 as queue  # noqa: E402


CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1.json"
)


@pytest.fixture(scope="module")
def config():
    return queue.validate_config(CONFIG)


def _safe_sample():
    return {
        "timestamp_utc": "2026-09-19T00:00:00.000+00:00",
        "gpu_index": 0,
        "gpu_used_mib": 1000.0,
        "gpu_total_mib": 8151.0,
        "gpu_utilization_percent": 25.0,
        "system_ram_percent": 30.0,
        "system_ram_used_mib": 7000.0,
        "system_ram_total_mib": 23728.0,
        "swap_used_mib": 500.0,
        "swap_out_bytes": 100,
    }


def _passed_result(spec):
    return {
        "status": "passed",
        "passed": True,
        "task": spec["task_id"],
        "num_envs": 1024,
        "max_iterations": 1,
        "memory": {
            "sample_count": 5,
            "owned_compute_sample_count": 4,
            "max_device_gpu_used_mib": 6000.0,
            "max_system_ram_percent": 70.0,
            "sustained_paging_detected": False,
        },
        "checkpoint": {"sha256": "a" * 64},
        "native_artifacts": {"agent_yaml": {"sha256": "b" * 64}},
        "lingering_compute_processes": [],
        "hard_resource_failure": False,
    }


def test_safety_first_order_and_exact_one_update_commands():
    assert [
        (spec["robot"], spec["terrain"], spec["task_id"])
        for spec in smoke.SMOKE_SPECS
    ] == [
        ("g1", "rough", "Isaac-Velocity-Rough-G1-v0"),
        ("go1", "rough", "Isaac-Velocity-Rough-Unitree-Go1-v0"),
        ("g1", "flat", "Isaac-Velocity-Flat-G1-v0"),
        ("go1", "flat", "Isaac-Velocity-Flat-Unitree-Go1-v0"),
    ]
    assert len({spec["task_id"] for spec in smoke.SMOKE_SPECS}) == 4
    for spec in smoke.SMOKE_SPECS:
        command = smoke.smoke_command(spec)
        assert command[:2] == [str(queue.ISAAC_PYTHON), str(queue.TRAINER)]
        for option, expected in (
            ("--task", spec["task_id"]),
            ("--agent", "rsl_rl_cfg_entry_point"),
            ("--seed", "0"),
            ("--num_envs", "1024"),
            ("--max_iterations", "1"),
            ("--device", "cuda:0"),
            ("--logger", "tensorboard"),
            ("--run_name", "seed_0"),
        ):
            assert command[command.index(option) + 1] == expected
        assert "--headless" in command
        assert "--resume" not in command
        assert "--load_run" not in command
        assert "--checkpoint" not in command
        assert "-Play-" not in spec["task_id"]


@pytest.mark.parametrize(
    ("gpu", "ram", "paging", "expected"),
    [
        (6963.199, 89.999, 2, None),
        (6963.2, 20.0, 0, "gpu_memory_limit"),
        (7000.0, 20.0, 0, "gpu_memory_limit"),
        (1000.0, 90.0, 0, "system_ram_limit"),
        (1000.0, 95.0, 0, "system_ram_limit"),
        (1000.0, 20.0, 3, "sustained_swap_out"),
        (math.nan, 20.0, 0, "missing_or_invalid_resource_telemetry"),
    ],
)
def test_resource_limits_are_strict_and_nonfinite_fails_closed(
    gpu, ram, paging, expected
):
    sample = _safe_sample()
    sample["gpu_used_mib"] = gpu
    sample["system_ram_percent"] = ram
    result = smoke.resource_violation(sample, paging)
    if expected is None:
        assert result is None
    else:
        assert result.startswith(expected)


def test_checkpoint_validation_counts_distribution_parameter_and_rejects_nonfinite(
    tmp_path
):
    spec = smoke.SMOKE_SPECS[0]
    checkpoint = tmp_path / "model_0.pt"
    actor_count = spec["actor_trainable_parameter_count"]
    critic_count = spec["critic_parameter_count"]
    payload = {
        "actor_state_dict": {"all": torch.zeros(actor_count)},
        "critic_state_dict": {"all": torch.zeros(critic_count)},
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": None,
    }
    torch.save(payload, checkpoint)
    result = smoke.validate_checkpoint(checkpoint, spec)
    assert result["actor_trainable_parameter_count"] == actor_count
    assert result["critic_trainable_parameter_count"] == critic_count
    assert result["all_values_finite"] is True
    payload["actor_state_dict"]["all"][0] = float("nan")
    torch.save(payload, checkpoint)
    with pytest.raises(smoke.SmokeGateError, match="non-finite"):
        smoke.validate_checkpoint(checkpoint, spec)


def test_task_result_requires_three_live_owned_samples_and_strict_caps():
    spec = smoke.SMOKE_SPECS[0]
    value = _passed_result(spec)
    smoke._validate_task_result(value, spec)
    for mutation in (
        lambda row: row["memory"].update(sample_count=2),
        lambda row: row["memory"].update(owned_compute_sample_count=2),
        lambda row: row["memory"].update(max_device_gpu_used_mib=6963.2),
        lambda row: row["memory"].update(max_system_ram_percent=90.0),
        lambda row: row["memory"].update(sustained_paging_detected=True),
        lambda row: row.update(checkpoint=None),
        lambda row: row.update(native_artifacts=None),
        lambda row: row.update(lingering_compute_processes=[{"pid": 1}]),
    ):
        changed = json.loads(json.dumps(value))
        mutation(changed)
        with pytest.raises(smoke.SmokeGateError):
            smoke._validate_task_result(changed, spec)


def test_unsafe_global_preflight_launches_no_child(
    config, tmp_path, monkeypatch
):
    isolated = dict(config)
    isolated["_output_root"] = str(tmp_path / "matrix")
    monkeypatch.setattr(smoke, "sample_resources", _safe_sample)
    monkeypatch.setattr(
        smoke,
        "_compute_processes",
        lambda: [{"pid": 123, "process_name": "foreign", "used_memory_mib": 10.0}],
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsafe preflight launched a child")

    monkeypatch.setattr(smoke.subprocess, "Popen", forbidden)
    result = smoke.run_gate(isolated)
    assert result["passed"] is False
    assert result["blocked_reason"] == "preexisting_nvidia_compute_processes"
    assert result["task_results"] == []
    assert not (tmp_path / "matrix" / "gates" / smoke.RECEIPT_NAME).exists()


def test_normal_task_failure_isolated_but_hard_resource_failure_stops(
    config, tmp_path, monkeypatch
):
    monkeypatch.setattr(smoke, "sample_resources", _safe_sample)
    monkeypatch.setattr(smoke, "_compute_processes", lambda: [])
    calls = []

    def local_failure(spec, _attempt):
        calls.append(spec["task_id"])
        return {
            "passed": False,
            "status": "failed",
            "failure": "trainer_exit_code:1",
            "hard_resource_failure": False,
        }

    monkeypatch.setattr(smoke, "run_task_smoke", local_failure)
    isolated = dict(config)
    isolated["_output_root"] = str(tmp_path / "local")
    result = smoke.run_gate(isolated)
    assert result["passed"] is False
    assert len(calls) == 4

    calls.clear()

    def hard_failure(spec, _attempt):
        calls.append(spec["task_id"])
        return {
            "passed": False,
            "status": "failed",
            "failure": "gpu_memory_limit:7000>=6963.2MiB",
            "hard_resource_failure": True,
        }

    monkeypatch.setattr(smoke, "run_task_smoke", hard_failure)
    isolated["_output_root"] = str(tmp_path / "hard")
    result = smoke.run_gate(isolated)
    assert result["passed"] is False
    assert len(calls) == 1
    assert result["blocked_reason"].startswith("hard_resource_failure:")


def test_canonical_receipt_binds_all_four_tasks_and_current_source(
    config, tmp_path
):
    receipt = {
        "schema_version": 1,
        "kind": smoke.GATE_KIND,
        "status": "passed",
        "passed": True,
        "config_sha256": queue.sha256_file(Path(config["_config_path"])),
        "queue_builder_sha256": queue.sha256_file(Path(queue.__file__).resolve()),
        "isaaclab_commit": queue.ISAACLAB_COMMIT,
        "num_envs": 1024,
        "maximum_parallel": 1,
        "task_results": [_passed_result(spec) for spec in smoke.SMOKE_SPECS],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert smoke.validate_canonical_receipt(config, path)["passed"] is True
    receipt["task_results"][1]["task"] = receipt["task_results"][0]["task"]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(smoke.SmokeGateError):
        smoke.validate_canonical_receipt(config, path)


def test_owned_compute_process_filter_uses_process_group(monkeypatch):
    monkeypatch.setattr(smoke.os, "getpgid", lambda pid: {10: 100, 11: 200}[pid])
    rows = [
        {"pid": 10, "process_name": "owned", "used_memory_mib": 1.0},
        {"pid": 11, "process_name": "foreign", "used_memory_mib": 1.0},
    ]
    assert smoke._owned_compute_processes(rows, 100) == [rows[0]]
