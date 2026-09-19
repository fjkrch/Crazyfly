from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from g1_fly_control.crazyflie.command_evaluation import (
    CommandRollout,
    EPISODE_COUNT,
    EPISODE_STEPS,
    batched_held_out_commands,
    score_command_rollout,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "crazyflie_command_evaluate.py"
spec = importlib.util.spec_from_file_location("crazyflie_command_evaluate_test", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _nominal_rollout() -> CommandRollout:
    commands, _ = batched_held_out_commands()
    return CommandRollout(
        effective_commands=commands,
        linear_velocity_body=commands[:, :, :3].clone(),
        yaw_rate_body=commands[:, :, 3].clone(),
        actions=torch.zeros_like(commands),
        positions_world=torch.zeros(EPISODE_STEPS, EPISODE_COUNT, 3),
        alive=torch.ones(EPISODE_STEPS, EPISODE_COUNT, dtype=torch.bool),
        invalid=torch.zeros(EPISODE_STEPS, EPISODE_COUNT, dtype=torch.bool),
    )


def test_evaluator_identity_is_exact_16_by_600_command_v1() -> None:
    task_id, evaluation_id, manifest = module.expected_manifest_ids()
    assert len(task_id) == 64
    assert len(evaluation_id) == 64
    assert manifest["manifest_id"] == evaluation_id
    assert manifest["protocol"] == "command_v1"
    assert manifest["evaluation_protocol"]["episodes"] == 16
    assert manifest["evaluation_protocol"]["steps_per_episode"] == 600


def test_quality_reports_acceleration_stability_and_survival_separately() -> None:
    rollout = _nominal_rollout()
    vectors = torch.zeros(EPISODE_STEPS, EPISODE_COUNT, 3)
    score = score_command_rollout(rollout)
    result = module.control_quality_metrics(
        rollout, vectors, vectors, vectors, vectors, score
    )
    assert set(result["component_scores_0_100"]) == {
        "acceleration_quality",
        "command_response",
        "flight_stability",
        "survival_not_die",
    }
    assert result["component_scores_0_100"]["acceleration_quality"] == 100.0
    assert result["component_scores_0_100"]["flight_stability"] == 100.0
    assert result["component_scores_0_100"]["survival_not_die"] == 100.0


def test_atomic_result_writer_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation.json"
    module._atomic_json_no_overwrite(destination, {"status": "PASS"})
    assert destination.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        module._atomic_json_no_overwrite(destination, {"status": "FAIL"})
    assert '"PASS"' in destination.read_text(encoding="utf-8")


class _FakeClock:
    def __init__(self, timestamps: list[int]) -> None:
        self._timestamps = iter(timestamps)

    def __call__(self) -> int:
        return next(self._timestamps)


def test_inference_latency_reports_exact_calls_and_excludes_prefix_warmup() -> None:
    # Durations are 1, 2, 3, 4, and 5 ms.  The first two real calls remain in
    # the all-call total but are excluded from steady-state statistics.
    clock = _FakeClock(
        [
            0,
            1_000_000,
            2_000_000,
            4_000_000,
            5_000_000,
            8_000_000,
            9_000_000,
            13_000_000,
            14_000_000,
            19_000_000,
        ]
    )
    latency = module.AuthenticatedInferenceLatency(
        device=torch.device("cpu"),
        batch_size=16,
        warmup_calls=2,
        clock_ns=clock,
    )
    for _ in range(5):
        latency.begin_call()
        latency.end_call()
    report = latency.report(expected_call_count=5)
    assert report["source"] == "exact_policy_act_calls_used_by_evaluation_control_path"
    assert report["action_producing_call_count"] == 5
    assert report["warmup_calls_excluded_from_statistics"] == 2
    assert report["sample_count"] == 3
    assert report["batch_size"] == 16
    assert report["total_ms"] == pytest.approx(12.0)
    assert report["mean_ms"] == pytest.approx(4.0)
    assert report["p50_ms"] == pytest.approx(4.0)
    assert report["p95_ms"] == pytest.approx(4.9)
    assert report["p99_ms"] == pytest.approx(4.98)
    assert report["max_ms"] == pytest.approx(5.0)
    assert report["all_action_producing_calls_total_ms"] == pytest.approx(15.0)
    assert report["cuda_synchronized_before_and_after_call"] is False


def test_inference_latency_cuda_syncs_exactly_before_and_after_each_call() -> None:
    synchronized: list[str] = []
    latency = module.AuthenticatedInferenceLatency(
        device=torch.device("cuda:0"),
        batch_size=16,
        warmup_calls=0,
        clock_ns=_FakeClock([10, 20, 30, 50]),
        cuda_synchronize=lambda device: synchronized.append(str(device)),
    )
    for _ in range(2):
        latency.begin_call()
        latency.end_call()
    report = latency.report(expected_call_count=2)
    assert synchronized == ["cuda:0"] * 4
    assert report["cuda_synchronized_before_and_after_call"] is True
    assert report["sample_count"] == 2
    assert report["total_ms"] == pytest.approx(30.0e-6)


def test_inference_latency_fails_closed_on_call_mismatch_or_invalid_clock() -> None:
    latency = module.AuthenticatedInferenceLatency(
        device=torch.device("cpu"),
        batch_size=1,
        warmup_calls=0,
        clock_ns=_FakeClock([20, 10]),
    )
    latency.begin_call()
    with pytest.raises(RuntimeError, match="not monotonic"):
        latency.end_call()

    missing = module.AuthenticatedInferenceLatency(
        device=torch.device("cpu"),
        batch_size=1,
        warmup_calls=0,
        clock_ns=_FakeClock([0, 1]),
    )
    missing.begin_call()
    missing.end_call()
    with pytest.raises(RuntimeError, match="call count mismatch"):
        missing.report(expected_call_count=2)


def test_inference_latency_rejects_excluding_every_sample() -> None:
    latency = module.AuthenticatedInferenceLatency(
        device=torch.device("cpu"),
        batch_size=1,
        warmup_calls=1,
        clock_ns=_FakeClock([0, 1]),
    )
    latency.begin_call()
    latency.end_call()
    with pytest.raises(RuntimeError, match="left no"):
        latency.report(expected_call_count=1)
