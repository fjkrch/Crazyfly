from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from drone_bootstrap import (  # noqa: E402
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_COMMAND_V1,
    COMMAND_TASK_ID,
)
from drone_smoke_env import (  # noqa: E402
    COMMAND_FULL_CURRICULUM_INTERACTIONS,
    COMMAND_REWARD_COMPONENTS,
    CUSTOM_TASKS,
    GATE_C_TASKS,
    LEGACY_CUSTOM_TASKS,
    _assess_command_reward_components,
    _assess_command_schedule_state,
    _assess_command_terminal_snapshot,
    _contract_profile_for_task,
    _default_smoke_report_path,
    _select_full_command_curriculum,
)
from g1_fly_control.tasks.crazyflie.command_logic import (  # noqa: E402
    COMMAND_CURRICULUM,
    COMMAND_TRACKING_CONTRACT_SHA256,
    FAILURE_LOW_HEIGHT,
    sample_scheduled_command,
)


def _schedule_state(*, seed: int = 7, count: int = 4) -> dict[str, object]:
    stage_index = len(COMMAND_CURRICULUM) - 1
    rows = []
    holds = []
    categories = []
    next_indices = []
    category_to_code = {
        "hover": 0,
        "cardinal": 1,
        "diagonal": 2,
        "full_simultaneous": 3,
    }
    for env_id in range(count):
        segment_index = env_id + 4
        segment = sample_scheduled_command(
            seed=seed,
            environment_id=env_id,
            segment_index=segment_index,
            total_interactions=COMMAND_FULL_CURRICULUM_INTERACTIONS,
        )
        rows.append(list(segment.command))
        holds.append(segment.hold_steps)
        categories.append(category_to_code[segment.category])
        next_indices.append(segment_index + 1)
    return {
        "schema_version": 1,
        "kind": "flyg1.crazyflie.command-schedule.v1",
        "contract_sha256": COMMAND_TRACKING_CONTRACT_SHA256,
        "num_envs": count,
        "command_schedule_seed": seed,
        "training_interactions": COMMAND_FULL_CURRICULUM_INTERACTIONS,
        "next_command_segment_index": next_indices,
        "command_steps_remaining": holds,
        "requested_command_body": rows,
        "command_category_code": categories,
        "command_stage_index": [stage_index] * count,
    }


def test_command_task_is_additive_and_gets_its_own_implicit_contract():
    assert COMMAND_TASK_ID in CUSTOM_TASKS
    assert COMMAND_TASK_ID not in LEGACY_CUSTOM_TASKS
    assert COMMAND_TASK_ID not in GATE_C_TASKS
    assert _contract_profile_for_task(COMMAND_TASK_ID, None) == CONTRACT_PROFILE_COMMAND_V1
    assert (
        _contract_profile_for_task("Isaac-Quadcopter-Direct-v0", None)
        == CONTRACT_PROFILE_BALANCED_V3
    )
    with pytest.raises(ValueError, match="requires contract_profile=command_v1"):
        _contract_profile_for_task(COMMAND_TASK_ID, CONTRACT_PROFILE_BALANCED_V3)


def test_plan_command_can_use_an_additive_default_report_path():
    result = _default_smoke_report_path(
        task=COMMAND_TASK_ID, seed=3, num_envs=4, steps=500
    )
    assert result == (
        ROOT
        / "runs/crazyflie_command_smoke/"
        "flycrazyflie-commandfollow-v0-seed3-4env-500steps.json"
    )


def test_command_smoke_selects_and_reports_the_final_stage_before_reset():
    stages = [stage.payload() for stage in COMMAND_CURRICULUM]

    class Environment:
        training_interactions = 0
        active_training_curriculum_stage_index = 0

        def set_training_interactions(self, value):
            self.training_interactions = value
            self.active_training_curriculum_stage_index = 2

        @property
        def active_training_curriculum_stage_payload(self):
            return stages[self.active_training_curriculum_stage_index]

        @property
        def full_training_curriculum_stage_payload(self):
            return stages[-1]

    report = _select_full_command_curriculum(Environment())
    assert report["contract_profile"] == CONTRACT_PROFILE_COMMAND_V1
    assert report["requested_training_interactions"] == 500_000
    assert report["selected_training_interactions"] == 500_000
    assert report["active_stage_index"] == 2
    assert report["active_stage"] == stages[-1]
    assert report["full_stage"] == stages[-1]


def test_command_schedule_assessment_authenticates_seed_segment_and_bounds():
    state = _schedule_state()
    report = _assess_command_schedule_state(state)
    assert report["passed"] is True
    assert report["range_valid"] is True
    assert len(report["per_environment"]) == 4

    changed = dict(state)
    changed["requested_command_body"] = [row[:] for row in state["requested_command_body"]]
    changed["requested_command_body"][0][0] += 0.01
    failure = _assess_command_schedule_state(changed)
    assert failure["passed"] is False
    assert failure["per_environment"][0]["command_matches"] is False

    out_of_range = dict(state)
    out_of_range["command_steps_remaining"] = list(
        state["command_steps_remaining"]
    )
    out_of_range["command_steps_remaining"][0] = 101
    assert _assess_command_schedule_state(out_of_range)["passed"] is False


def test_command_reward_assessment_requires_acceleration_stability_and_survival_terms():
    components = {
        name: torch.zeros(4, dtype=torch.float32)
        for name in COMMAND_REWARD_COMPONENTS
    }
    components["survival"].fill_(0.001)
    report = _assess_command_reward_components(components, num_envs=4)
    assert report["passed"] is True
    assert report["missing"] == []
    assert all(report["sign_checks"].values())

    missing = dict(components)
    del missing["tracking_progress"]
    missing_report = _assess_command_reward_components(missing, num_envs=4)
    assert missing_report["passed"] is False
    assert missing_report["missing"] == ["tracking_progress"]

    wrong_sign = dict(components)
    wrong_sign["jerk"] = torch.ones(4)
    assert _assess_command_reward_components(wrong_sign, num_envs=4)["passed"] is False


def test_command_terminal_assessment_checks_failure_reward_and_all_telemetry():
    done = torch.tensor([True, True])
    observation = torch.zeros(2, 12)
    tracking = torch.zeros(2, 4)
    observation[:, :3] = tracking[:, :3]
    observation[:, 5] = tracking[:, 3]
    kwargs = {
        "done": done,
        "terminal_mask": done.clone(),
        "terminal_observation": observation,
        "terminal_tracking_error_body": tracking,
        "terminal_linear_acceleration_body": torch.zeros(2, 3),
        "terminal_linear_jerk_body": torch.zeros(2, 3),
        "terminal_failure_cause": torch.full((2,), FAILURE_LOW_HEIGHT),
        "terminal_position_w": torch.tensor([[0.0, 0.0, 0.1]]).repeat(2, 1),
        "terminal_command_target_position_w": torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        "failure_reward": torch.full((2,), -5.0),
        "expected_failure_cause": FAILURE_LOW_HEIGHT,
    }
    assert _assess_command_terminal_snapshot(**kwargs)["passed"] is True

    no_penalty = dict(kwargs)
    no_penalty["failure_reward"] = torch.zeros(2)
    failed = _assess_command_terminal_snapshot(**no_penalty)
    assert failed["passed"] is False
    assert failed["failure_penalty_applied"] is False


def test_exact_command_smoke_contract_is_wired_into_the_runner_source():
    source = (ROOT / "scripts/drone_smoke_env.py").read_text(encoding="utf-8")
    assert "COMMAND_MINIMUM_SMOKE_STEPS = 500" in source
    assert '"tracking_progress"' in source
    assert '"wrong_direction_acceleration"' in source
    assert '"attitude_stability"' in source
    assert '"angular_stability"' in source
    assert '"survival"' in source
    assert '"failure"' in source
    assert "_run_command_failure_probes" in source
    assert "command_schedule_seed" in source
