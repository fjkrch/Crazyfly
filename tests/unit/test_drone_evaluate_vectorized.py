from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import textwrap

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_evaluate as evaluator  # noqa: E402
from drone_bootstrap import sha256_file  # noqa: E402
from drone_evaluation_protocol import SCENARIOS, generate_manifest, load_protocol  # noqa: E402
from drone_evaluate import (  # noqa: E402
    _CudaGraphPolicyRunner,
    _deterministic_actor_step,
    _evaluation_batch_size,
    _evaluate_batch,
    _merge_scenario_parts,
    _scenario_child_command,
    _scenario_part_paths,
    _update_success_timing,
    _validated_policy_device,
)
from g1_fly_control.crazyflie.controllers import build_controller, reset_controller_state  # noqa: E402
from g1_fly_control.policies.lif_core import LIFState  # noqa: E402
from g1_fly_control.crazyflie.memory import assess as assess_memory  # noqa: E402
from g1_fly_control.tasks.crazyflie.metrics import (  # noqa: E402
    EpisodeSummary,
    GustRecoveryOutcome,
    SwitchOutcome,
    summarize_episodes,
)


def _empty_timing(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch,), -1, dtype=torch.long),
        torch.full((batch, 3), -1, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
    )


def test_vectorized_success_timing_matches_scalar_first_event_semantics() -> None:
    first, switches, previous = _empty_timing()
    event_ids = torch.arange(3, dtype=torch.long)

    first, switches, previous = _update_success_timing(
        current_success_count=torch.tensor([1, 1, 1, 1]),
        current_switch_count=torch.tensor([1, 1, 2, 4]),
        previous_success_count=previous,
        active=torch.tensor([True, False, True, True]),
        decision=25,
        first_success_step=first,
        switch_success_step=switches,
        switch_event_ids=event_ids,
    )
    assert first.tolist() == [25, -1, 25, 25]
    assert switches.tolist() == [
        [25, -1, -1],
        [-1, -1, -1],
        [-1, 25, -1],
        [-1, -1, -1],
    ]
    # Inactive rows retain their previous count, exactly as completed rows did
    # in the former scalar evaluator.
    assert previous.tolist() == [1, 0, 1, 1]

    first, switches, previous = _update_success_timing(
        current_success_count=torch.tensor([2, 1, 1, 2]),
        current_switch_count=torch.tensor([2, 1, 2, 2]),
        previous_success_count=previous,
        active=torch.ones(4, dtype=torch.bool),
        decision=100,
        first_success_step=first,
        switch_success_step=switches,
        switch_event_ids=event_ids,
    )
    assert first.tolist() == [25, 100, 25, 25]
    assert switches.tolist() == [
        [25, 100, -1],
        [100, -1, -1],
        [-1, 25, -1],
        [-1, 100, -1],
    ]
    assert previous.tolist() == [2, 1, 1, 2]


def test_vectorized_success_timing_rejects_mismatched_event_ids() -> None:
    first, switches, previous = _empty_timing(batch=1)
    with pytest.raises(ValueError, match="one entry per tracked switch"):
        _update_success_timing(
            current_success_count=torch.ones(1, dtype=torch.long),
            current_switch_count=torch.ones(1, dtype=torch.long),
            previous_success_count=previous,
            active=torch.ones(1, dtype=torch.bool),
            decision=1,
            first_success_step=first,
            switch_success_step=switches,
            switch_event_ids=torch.arange(2, dtype=torch.long),
        )


def test_evaluation_decision_loop_has_cuda_graph_and_bounded_completion_poll() -> None:
    source = textwrap.dedent(inspect.getsource(_evaluate_batch))
    tree = ast.parse(source)
    decision_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "decision"
    )
    forbidden = []
    item_calls = []
    for node in ast.walk(decision_loop):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"cpu", "numpy", "tolist"}:
            forbidden.append(node.func.attr)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "item":
            item_calls.append(node)
        if isinstance(node.func, ast.Name) and node.func.id in {"int", "float"}:
            forbidden.append(node.func.id)
    assert forbidden == []
    # The sole scalar device-to-host read is the explicitly rate-limited
    # all-complete poll that prevents DirectRLEnv from auto-resetting and
    # simulating unrelated episodes after every plan has terminated. Policy
    # traffic is tensor-only and is audited separately below.
    assert len(item_calls) == 1
    poll_if = next(
        node for node in ast.walk(decision_loop)
        if isinstance(node, ast.If) and any(call is item_calls[0] for call in ast.walk(node.test))
    )
    assert isinstance(poll_if.test, ast.BoolOp)
    assert isinstance(poll_if.test.op, ast.And)
    assert any(isinstance(node, ast.Mod) for node in ast.walk(poll_if.test))
    assert len(poll_if.body) == 1 and isinstance(poll_if.body[0], ast.Break)
    interval_assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "completion_poll_interval"
                for target in node.targets)
    )
    assert isinstance(interval_assignment.value, ast.Constant)
    assert interval_assignment.value.value == 25
    cpu_transfers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cpu"
    ]
    assert len(cpu_transfers) == 1

    step_source = textwrap.dedent(inspect.getsource(_CudaGraphPolicyRunner.step))
    step_tree = ast.parse(step_source)
    forbidden_step_calls = []
    copy_calls = []
    replay_calls = []
    for node in ast.walk(step_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"cpu", "cuda", "to", "numpy", "tolist", "item"}:
            forbidden_step_calls.append(node.func.attr)
        if node.func.attr == "copy_":
            copy_calls.append(node)
        if node.func.attr == "replay":
            replay_calls.append(node)
    assert forbidden_step_calls == []
    assert len(copy_calls) == 2
    assert len(replay_calls) == 1


def _assert_state_equal(left, right) -> None:
    if isinstance(left, LIFState):
        assert isinstance(right, LIFState)
        for left_value, right_value in zip(
            (left.membrane, left.spikes, left.synapse, left.refractory),
            (right.membrane, right.spikes, right.synapse, right.refractory),
            strict=True,
        ):
            assert torch.equal(left_value, right_value)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    else:
        assert left is None and right is None


@pytest.mark.parametrize(
    "kind",
    ("frozen_lif_original", "frozen_lif_degree_rewired", "gru_matched", "mlp_normal"),
)
def test_action_only_path_matches_full_deterministic_actor_on_cpu(kind: str) -> None:
    torch.manual_seed(20260916)
    policy, _ = build_controller(kind, device="cpu")
    policy.eval()
    batch = 4
    direct_state = policy.initial_state(batch, device="cpu") if hasattr(policy, "initial_state") else None
    action_only_state = (
        policy.initial_state(batch, device="cpu") if hasattr(policy, "initial_state") else None
    )
    reset_masks = (
        torch.zeros(batch, dtype=torch.bool),
        torch.tensor([True, False, True, False]),
    )
    generator = torch.Generator().manual_seed(101)

    with torch.inference_mode():
        for reset_mask in reset_masks:
            observation = torch.randn(batch, 12, generator=generator)
            direct_state = reset_controller_state(policy, direct_state, reset_mask)
            direct = policy.act(observation, direct_state, deterministic=True)
            action_only, action_only_state = _deterministic_actor_step(
                policy, observation, action_only_state, reset_mask
            )
            assert torch.equal(action_only, direct.action)
            _assert_state_equal(action_only_state, direct.state)
            direct_state = direct.state


def test_primary_policy_device_is_exact_cuda_zero() -> None:
    assert _validated_policy_device("cuda:0") == torch.device("cuda:0")
    for invalid in ("cpu", "cuda", "cuda:1"):
        with pytest.raises(ValueError, match="requires exactly cuda:0"):
            _validated_policy_device(invalid)


def test_lif_proof_protocol_is_distinct_and_uses_five_deterministic_plans() -> None:
    proof_protocol = load_protocol("lif_proof")
    integration = load_protocol("integration")
    main = load_protocol("main")

    assert proof_protocol["label"] == "lif_proof"
    assert proof_protocol["evaluation_seed"] == 101
    assert proof_protocol["episodes_per_scenario"] == 5
    assert all(len(proof_protocol["scenarios"][scenario]) == 5 for scenario in SCENARIOS)
    assert proof_protocol["manifest_id"] not in {
        integration["manifest_id"],
        main["manifest_id"],
    }
    assert integration["episodes_per_scenario"] == 2
    assert main["episodes_per_scenario"] == 16
    assert _evaluation_batch_size(2) == 2
    assert _evaluation_batch_size(5) == 1
    assert _evaluation_batch_size(16) == 4


def test_cuda_graph_runner_fails_closed_for_cpu_policy() -> None:
    policy, _ = build_controller("mlp_normal", device="cpu")
    policy.eval()
    with pytest.raises(RuntimeError, match="parameter/buffer"):
        _CudaGraphPolicyRunner(policy, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
@pytest.mark.parametrize(
    "kind",
    ("frozen_lif_original", "frozen_lif_degree_rewired", "gru_matched", "mlp_normal"),
)
def test_cuda_graph_runner_passes_runtime_bitwise_gate(kind: str) -> None:
    torch.manual_seed(20260916)
    policy, _ = build_controller(kind, device="cuda:0")
    policy.eval()
    runner = _CudaGraphPolicyRunner(policy, 2)
    assert runner.report == evaluator._expected_policy_graph_report(2)
    observation = torch.linspace(-1.0, 1.0, 24, device="cuda:0").reshape(2, 12)
    reset = torch.tensor([False, True], device="cuda:0")
    action, finite = runner.step(observation, reset)
    torch.cuda.synchronize("cuda:0")
    assert action.device == torch.device("cuda:0")
    assert finite.tolist() == [True, True]


def _unit_episode(scenario: str) -> EpisodeSummary:
    switches = ()
    gusts = ()
    if scenario == SCENARIOS[1]:
        switches = tuple(
            SwitchOutcome(
                switch_index=index,
                switch_step=step,
                success=False,
                latency_s=None,
                failure_reason="target_segment_ended_without_success",
            )
            for index, step in enumerate((150, 300, 450))
        )
    elif scenario == SCENARIOS[2]:
        gusts = tuple(
            GustRecoveryOutcome(
                gust_index=index,
                gust_start_step=step,
                applied=True,
                stable_before_gust=False,
                recovered=False,
                recovery_latency_s=None,
                max_displacement_m=0.5,
                post_gust_error_integral_m_s=1.0,
                failure_reason="recovery_window_expired_or_not_reached",
            )
            for index, step in enumerate((150, 300, 450))
        )
    return EpisodeSummary(
        scenario=scenario,
        episode_id=0,
        success=False,
        terminated=False,
        truncated=True,
        failure_reason="time_limit",
        time_to_first_success_s=None,
        final_goal_error_m=1.0,
        integrated_goal_error_m_s=12.0,
        mean_speed_inside_target_region_m_s=None,
        crash=False,
        out_of_bounds=False,
        invalid_state=False,
        command_effort=0.0,
        command_smoothness=0.0,
        aggregate_wrench_mechanical_work_proxy_j=0.0,
        completed_steps=600,
        evaluation_seed=101,
        switch_outcomes=switches,
        gust_outcomes=gusts,
    )


def _unit_fingerprint_payload() -> dict:
    return {
        "source_sha256": {"unit": "b" * 64},
        "resolved_config": {"kind": "unit_evaluation"},
    }


def _unit_fingerprint() -> str:
    return evaluator.canonical_sha256(_unit_fingerprint_payload())


def _unit_memory_evidence(*, monotonic_rss: bool = False) -> tuple[list[dict], dict]:
    samples = []
    stages = ["environment_loaded", "graph_captured"]
    stages.extend(["steady_state"] * (4 if monotonic_rss else 1))
    stages.append("evaluation_end")
    steady_index = 0
    for index, stage in enumerate(stages):
        if stage == "steady_state":
            rss = 100.0 + (steady_index if monotonic_rss else 0.0)
            steady_index += 1
        else:
            rss = 100.0
        samples.append({
            "timestamp_utc": f"2026-09-16T00:00:0{index}+00:00",
            "stage": stage,
            "step": index,
            "process_rss_mib": rss,
            "system_ram_total_gib": 32.0,
            "system_ram_used_gib": 4.0,
            "system_ram_available_gib": 28.0,
            "system_ram_percent": 12.5,
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
                "used_mib": 512.0,
            }],
            "torch_allocated_mib": 32.0,
            "torch_reserved_mib": 64.0,
            "torch_peak_allocated_mib": 32.0,
            "torch_peak_reserved_mib": 64.0,
        })
    return samples, assess_memory(samples)


def _unit_part(
    scenario: str,
    protocol: dict,
    checkpoint: Path,
    *,
    monotonic_rss: bool = False,
) -> dict:
    memory_samples, memory_gate = _unit_memory_evidence(monotonic_rss=monotonic_rss)
    summary = summarize_episodes([_unit_episode(scenario)], expected_episode_count=1)
    plan = protocol["scenarios"][scenario][0]
    if scenario == SCENARIOS[2]:
        expected_gust_impulses = [
            [
                evaluator.AUDITED_CRAZYFLIE_MASS_KG
                * gust["desired_mass_normalized_delta_velocity_m_s"]
                * gust["direction_world_xy"][0],
                evaluator.AUDITED_CRAZYFLIE_MASS_KG
                * gust["desired_mass_normalized_delta_velocity_m_s"]
                * gust["direction_world_xy"][1],
                0.0,
            ]
            for gust in plan["gusts"]
        ]
    else:
        expected_gust_impulses = [[0.0, 0.0, 0.0] for _ in range(3)]
    summary["episodes"][0].update({
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "target_success_event_count": 0,
        "robot_mass_kg": (
            evaluator.AUDITED_CRAZYFLIE_MASS_KG if scenario == SCENARIOS[2] else None
        ),
        "gust_applied_impulse_w_n_s": [list(vector) for vector in expected_gust_impulses],
        "gust_expected_impulse_w_n_s": [list(vector) for vector in expected_gust_impulses],
        "gust_impulse_max_abs_error_n_s": 0.0 if scenario == SCENARIOS[2] else None,
    })
    return {
        "schema_version": 1,
        "status": "completed",
        "label": protocol["label"],
        "protocol": "unit",
        "evaluation_seed": protocol["evaluation_seed"],
        "evaluation_manifest_id": protocol["manifest_id"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_seed": 7,
        "controller": "frozen_lif_original",
        "fingerprint": _unit_fingerprint(),
        "fingerprint_payload": _unit_fingerprint_payload(),
        "controller_report": {"controller_kind": "frozen_lif"},
        "generated_at_utc": "2026-09-16T00:00:00+00:00",
        "deterministic_actions": True,
        "simulation_device": "cuda:0",
        "simulation_device_type": "cuda",
        "policy_inference_device": "cuda:0",
        "policy_inference_device_type": "cuda",
        "policy_inference_backend": evaluator.POLICY_INFERENCE_BACKEND,
        "policy_inference_precision": evaluator.POLICY_INFERENCE_PRECISION,
        "policy_inference_graph": evaluator._expected_policy_graph_report(1),
        "policy_bridge": dict(evaluator.POLICY_BRIDGE_CONTRACT),
        "memory_samples": memory_samples,
        "memory_gate": memory_gate,
        "failure_denominator_rule": "Every planned episode remains in the denominator",
        "censoring_rule": "Unsuccessful episode/attempt assigned its fixed observation horizon",
        "integrated_error_censoring_rule": (
            "For early termination, carry the final observed goal error through the fixed episode "
            "or post-gust recovery window"
        ),
        "scenario": scenario,
        "episodes": summary["episodes"],
        "summary": summary,
    }


def _unit_args(checkpoint: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint.resolve(),
        output=output.resolve(),
        protocol="unit",
        expected_fingerprint=_unit_fingerprint(),
        training_seed=7,
        policy="frozen_lif_original",
        headless=True,
        device="cuda:0",
    )


def test_all_scenarios_child_command_forwards_only_declared_launch_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = _unit_args(checkpoint, tmp_path / "all.json")
    part_paths = _scenario_part_paths(args.output, "attempt-one")
    command = _scenario_child_command(args, SCENARIOS[0], part_paths[SCENARIOS[0]])

    assert command[0] == sys.executable
    assert command[1] == str((ROOT / "scripts" / "drone_evaluate.py").resolve())
    assert "--all_scenarios" not in command
    assert command[command.index("--scenario") + 1] == SCENARIOS[0]
    assert command[command.index("--checkpoint") + 1] == str(checkpoint.resolve())
    assert command[command.index("--expected_fingerprint") + 1] == _unit_fingerprint()
    assert command[command.index("--training_seed") + 1] == "7"
    assert command[command.index("--policy") + 1] == "frozen_lif_original"
    assert "--headless" in command
    assert command[command.index("--device") + 1] == "cuda:0"
    assert tuple(part_paths) == SCENARIOS
    assert len(set(part_paths.values())) == 3
    assert all(path.parent.name == "attempt-one" for path in part_paths.values())
    assert all(path.parent.parent.name == "all.json.parts" for path in part_paths.values())


def test_all_scenarios_merge_recomputes_parts_and_preserves_explicit_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol = generate_manifest(seed=101, episodes_per_scenario=1, label="unit")
    paths = _scenario_part_paths(tmp_path / "all.json", "merge-attempt")
    for index, (scenario, path) in enumerate(paths.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _unit_part(
                    scenario,
                    protocol,
                    checkpoint,
                    monotonic_rss=index == 0,
                )
            ),
            encoding="utf-8",
        )

    merged = _merge_scenario_parts(
        paths,
        protocol_name="unit",
        protocol=protocol,
        checkpoint=checkpoint,
        expected_fingerprint=_unit_fingerprint(),
        expected_training_seed=7,
        expected_policy="frozen_lif_original",
    )
    assert merged["scenario"] == "all"
    assert merged["scenario_attempt_id"] == "merge-attempt"
    assert merged["total_episodes"] == 3
    assert [row["scenario"] for row in merged["episodes"]] == list(SCENARIOS)
    assert tuple(merged["scenario_results"]) == SCENARIOS
    assert tuple(merged["scenario_part_artifacts"]) == SCENARIOS
    assert merged["memory_gate"]["policy_version"] == evaluator.MEMORY_POLICY_VERSION
    assert merged["memory_gate"]["passed"] is True
    assert merged["memory_gate"]["monotonic_process_growth_detected"] is True
    assert merged["memory_gate"]["sustained_paging_detected"] is False
    assert merged["memory_gate"]["limits"]["rss_growth_disposition"] == "warning_only"
    assert len(merged["memory_gate"]["warnings"]) == 1
    assert merged["memory_gate"]["warnings"][0].startswith(f"{SCENARIOS[0]}: ")
    for scenario, artifact in merged["scenario_part_artifacts"].items():
        assert artifact["path"] == str(paths[scenario].resolve())
        assert artifact["sha256"] == sha256_file(paths[scenario])

    tampered = json.loads(paths[SCENARIOS[1]].read_text(encoding="utf-8"))
    tampered["summary"]["success_rate"] = 1.0
    paths[SCENARIOS[1]].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not recompute exactly"):
        _merge_scenario_parts(
            paths,
            protocol_name="unit",
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=_unit_fingerprint(),
            expected_training_seed=7,
            expected_policy="frozen_lif_original",
        )

    paths[SCENARIOS[1]].write_text(
        json.dumps(_unit_part(SCENARIOS[1], protocol, checkpoint)), encoding="utf-8"
    )
    payload_tamper = _unit_part(SCENARIOS[1], protocol, checkpoint)
    payload_tamper["fingerprint_payload"]["tampered"] = True
    paths[SCENARIOS[1]].write_text(json.dumps(payload_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint payload mismatch"):
        _merge_scenario_parts(
            paths,
            protocol_name="unit",
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=_unit_fingerprint(),
            expected_training_seed=7,
            expected_policy="frozen_lif_original",
        )


@pytest.mark.parametrize(
    ("corruption", "match"),
    (
        ("expected_vector", "Recorded expected gust impulse"),
        ("applied_flag", "applied flag disagrees"),
        ("reported_error", "max error does not recompute"),
        ("nonfinite", "finite numeric evidence"),
        ("mass", "audited cf2x mass"),
    ),
)
def test_gust_scenario_part_rejects_tampered_impulse_evidence(
    tmp_path: Path, corruption: str, match: str
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol = generate_manifest(seed=101, episodes_per_scenario=1, label="unit")
    scenario = SCENARIOS[2]
    # Scenario parts are validated after being read from JSON; round-trip the
    # dataclass-derived tuple fields so this tamper test exercises that exact
    # artifact representation.
    part = json.loads(json.dumps(_unit_part(scenario, protocol, checkpoint)))
    row = part["episodes"][0]
    if corruption == "expected_vector":
        row["gust_expected_impulse_w_n_s"][0][0] += 0.01
    elif corruption == "applied_flag":
        row["gust_applied_impulse_w_n_s"][0] = [0.0, 0.0, 0.0]
    elif corruption == "reported_error":
        row["gust_impulse_max_abs_error_n_s"] = 0.5
    elif corruption == "nonfinite":
        row["gust_applied_impulse_w_n_s"][0][0] = float("nan")
    elif corruption == "mass":
        row["robot_mass_kg"] = 1.0
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)

    with pytest.raises(ValueError, match=match):
        evaluator._validate_scenario_part(
            part,
            scenario=scenario,
            protocol_name="unit",
            protocol=protocol,
            checkpoint=checkpoint,
            expected_fingerprint=_unit_fingerprint(),
            expected_training_seed=7,
            expected_policy="frozen_lif_original",
        )


def test_all_scenarios_successful_reruns_preserve_prior_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "resolved_config": {"kind": "unit_evaluation"},
        "fingerprints": {"reproduction": _unit_fingerprint()},
    }, checkpoint)
    output = tmp_path / "all.json"
    args = _unit_args(checkpoint, output)
    args.expected_fingerprint = None  # Match the standalone section-14 command.
    protocol = generate_manifest(seed=101, episodes_per_scenario=1, label="unit")
    attempt_ids = iter(("attempt-first", "attempt-second"))
    monkeypatch.setattr(evaluator, "load_protocol", lambda _name: protocol)
    monkeypatch.setattr(evaluator, "_new_scenario_attempt_id", lambda: next(attempt_ids))

    def run_child(command, *, check):
        assert check is False
        assert command[command.index("--expected_fingerprint") + 1] == _unit_fingerprint()
        scenario = command[command.index("--scenario") + 1]
        part_path = Path(command[command.index("--output") + 1])
        assert not part_path.exists()
        part_path.write_text(
            json.dumps(_unit_part(scenario, protocol, checkpoint)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluator.subprocess, "run", run_child)
    assert evaluator._run_all_scenarios(args) == 0
    first_paths = _scenario_part_paths(output, "attempt-first")
    first_bytes = {scenario: path.read_bytes() for scenario, path in first_paths.items()}
    assert evaluator._run_all_scenarios(args) == 0
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["scenario_attempt_id"] == "attempt-second"
    assert all(path.read_bytes() == first_bytes[scenario] for scenario, path in first_paths.items())
    assert list((tmp_path / "all.json.history").glob("*.json"))


def test_all_scenarios_parent_runs_children_sequentially_and_stops_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "resolved_config": {"kind": "unit_evaluation"},
        "fingerprints": {"reproduction": _unit_fingerprint()},
    }, checkpoint)
    output = tmp_path / "all.json"
    args = _unit_args(checkpoint, output)
    protocol = generate_manifest(seed=101, episodes_per_scenario=1, label="unit")
    calls: list[str] = []

    monkeypatch.setattr(evaluator, "load_protocol", lambda _name: protocol)
    monkeypatch.setattr(evaluator, "_new_scenario_attempt_id", lambda: "failed-attempt")

    def run_child(command, *, check):
        assert check is False
        scenario = command[command.index("--scenario") + 1]
        calls.append(scenario)
        if scenario == SCENARIOS[1]:
            return SimpleNamespace(returncode=9)
        part_path = Path(command[command.index("--output") + 1])
        part_path.write_text(
            json.dumps(_unit_part(scenario, protocol, checkpoint)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluator.subprocess, "run", run_child)
    assert evaluator._run_all_scenarios(args) == 1
    assert calls == list(SCENARIOS[:2])
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failed_scenario"] == SCENARIOS[1]
    assert failure["completed_scenarios"] == [SCENARIOS[0]]
    assert failure["failed_phase"] == "run_scenario_child"
    assert failure["resolved_config"] == {"kind": "unit_evaluation"}
    assert failure["fingerprint"] == _unit_fingerprint()
    assert failure["scenario_attempt_id"] == "failed-attempt"
    assert not _scenario_part_paths(output, "failed-attempt")[SCENARIOS[2]].exists()
