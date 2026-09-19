from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from drone_smoke_env import (  # noqa: E402
    CUSTOM_TASKS,
    GATE_C_TASKS,
    _assess_gust_impulse_evidence,
    _assess_terminal_snapshot,
    _select_full_training_curriculum,
    _steady_state_sample_steps,
)
from drone_bootstrap import (  # noqa: E402
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_SURVIVAL_V2,
    DEFAULT_CONTRACT_PROFILE,
    launch_environment,
    selected_env_cfg,
    task_contract_payload,
    validate_contract_profile,
)
from g1_fly_control.tasks.crazyflie.logic import (  # noqa: E402
    balanced_task_contract_payload,
    balanced_v4_task_contract_payload,
    survival_first_contract_payload,
)
from g1_fly_control.tasks.crazyflie.registration import (  # noqa: E402
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    MIXED_TRAINING_TASK_ID,
    TASK_IDS,
    TASK_TO_CFG,
)


def test_memory_samples_exclude_the_first_episode_lazy_allocation_boundary():
    assert _steady_state_sample_steps(1000, 600) == (700, 800, 900, 1000)
    assert _steady_state_sample_steps(1000, 500) == (625, 750, 875, 1000)
    assert _steady_state_sample_steps(100, 600) == (25, 50, 75, 100)
    with pytest.raises(ValueError, match="positive integer"):
        _steady_state_sample_steps(0, 0)


def test_mixed_training_smoke_is_not_misrepresented_as_a_gate_c_evaluation():
    assert "FlyCrazyflie-Mixed-v0" in CUSTOM_TASKS
    assert "FlyCrazyflie-Mixed-v0" not in GATE_C_TASKS
    assert GATE_C_TASKS == {
        "FlyCrazyflie-WaypointReach-v0",
        "FlyCrazyflie-WaypointSwitch-v0",
        "FlyCrazyflie-GustRecovery-v0",
    }


def test_direct_project_registrations_default_to_active_balanced_configs():
    expected = {
        TASK_IDS[0]: "BalancedWaypointReachEnvCfg",
        TASK_IDS[1]: "BalancedWaypointSwitchEnvCfg",
        TASK_IDS[2]: "BalancedGustRecoveryEnvCfg",
        MIXED_TRAINING_TASK_ID: "BalancedMixedTrainingEnvCfg",
        COMMAND_FOLLOW_TASK_ID: "CommandFollowEnvCfg",
        COMMAND_FOLLOW_WIDE_TASK_ID: "CommandFollowWideEnvCfg",
        COMMAND_FOLLOW_WIDE_WIND_TASK_ID: "CommandFollowWideWindEnvCfg",
    }
    assert set(TASK_TO_CFG) == set(expected)
    for task_id, cfg_name in expected.items():
        assert TASK_TO_CFG[task_id].endswith(f":{cfg_name}")


@pytest.mark.parametrize(
    ("contract_profile", "contract"),
    (
        (CONTRACT_PROFILE_SURVIVAL_V2, survival_first_contract_payload()),
        (CONTRACT_PROFILE_BALANCED_V3, balanced_task_contract_payload()),
        (CONTRACT_PROFILE_BALANCED_V4, balanced_v4_task_contract_payload()),
    ),
)
def test_custom_smoke_selects_and_reports_the_full_curriculum_before_reset(
    contract_profile, contract
):
    stages = contract["training_curriculum"]["stages"]

    class Environment:
        training_interactions = 0
        active_training_curriculum_stage_index = 0

        def set_training_interactions(self, value):
            self.training_interactions = value
            self.active_training_curriculum_stage_index = 3

        @property
        def active_training_curriculum_stage_payload(self):
            return stages[self.active_training_curriculum_stage_index]

        @property
        def full_training_curriculum_stage_payload(self):
            return stages[-1]

    report = _select_full_training_curriculum(Environment(), contract_profile)
    assert report["requested_training_interactions"] == 1_000_000
    assert report["selected_training_interactions"] == 1_000_000
    assert report["active_stage_index"] == 3
    assert report["active_stage"]["name"] == "full"
    assert report["contract_profile"] == contract_profile
    assert report["contract"] == contract


def test_contract_profile_validation_and_payloads_fail_closed():
    assert validate_contract_profile(CONTRACT_PROFILE_SURVIVAL_V2) == "survival_v2"
    assert validate_contract_profile(CONTRACT_PROFILE_BALANCED_V3) == "balanced_v3"
    assert validate_contract_profile(CONTRACT_PROFILE_BALANCED_V4) == "balanced_v4"
    assert DEFAULT_CONTRACT_PROFILE == CONTRACT_PROFILE_BALANCED_V3
    assert task_contract_payload("survival_v2") == survival_first_contract_payload()
    assert task_contract_payload("balanced_v3") == balanced_task_contract_payload()
    assert task_contract_payload("balanced_v4") == balanced_v4_task_contract_payload()
    with pytest.raises(ValueError, match="contract_profile must be one of"):
        validate_contract_profile("balanced")
    # The project profile selector is inert for the installed native task, so
    # shared CLIs accept either recognized value without changing its config.
    assert validate_contract_profile(
        "balanced_v3", task="Isaac-Quadcopter-Direct-v0"
    ) == "balanced_v3"
    assert validate_contract_profile(
        "survival_v2", task="Isaac-Quadcopter-Direct-v0"
    ) == "survival_v2"
    assert validate_contract_profile(
        "balanced_v4", task="Isaac-Quadcopter-Direct-v0"
    ) == "balanced_v4"


def test_selected_env_cfg_uses_balanced_default_and_preserves_explicit_survival(monkeypatch):
    import sys
    import types

    class Scene:
        num_envs = 0

    def cfg_class(name):
        class Config:
            def __init__(self):
                self.scene = Scene()
                self.debug_vis = True
                self.deterministic_eval = False

        Config.__name__ = name
        return Config

    adapter = types.ModuleType("g1_fly_control.tasks.crazyflie.adapter")
    adapter.validate_upstream_contract = lambda: None
    env_cfg = types.ModuleType("g1_fly_control.tasks.crazyflie.env_cfg")
    names = (
        "GustRecoveryEnvCfg",
        "MixedTrainingEnvCfg",
        "WaypointReachEnvCfg",
        "WaypointSwitchEnvCfg",
        "BalancedGustRecoveryEnvCfg",
        "BalancedMixedTrainingEnvCfg",
        "BalancedWaypointReachEnvCfg",
        "BalancedWaypointSwitchEnvCfg",
        "BalancedV4GustRecoveryEnvCfg",
        "BalancedV4WaypointReachEnvCfg",
        "BalancedV4WaypointSwitchEnvCfg",
    )
    for name in names:
        setattr(env_cfg, name, cfg_class(name))
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setitem(sys.modules, env_cfg.__name__, env_cfg)

    balanced = selected_env_cfg("FlyCrazyflie-WaypointReach-v0", 3)
    legacy = selected_env_cfg(
        "FlyCrazyflie-WaypointReach-v0",
        4,
        deterministic_evaluation=True,
        contract_profile="survival_v2",
    )
    v4 = selected_env_cfg(
        "FlyCrazyflie-WaypointReach-v0",
        5,
        contract_profile="balanced_v4",
    )
    assert type(balanced).__name__ == "BalancedWaypointReachEnvCfg"
    assert balanced.scene.num_envs == 3
    assert type(legacy).__name__ == "WaypointReachEnvCfg"
    assert legacy.scene.num_envs == 4
    assert legacy.deterministic_eval is True
    assert type(v4).__name__ == "BalancedV4WaypointReachEnvCfg"
    assert v4.scene.num_envs == 5
    with pytest.raises(ValueError, match="Unknown Crazyflie task"):
        selected_env_cfg(
            "FlyCrazyflie-Mixed-v0", 2, contract_profile="balanced_v4"
        )


def test_native_cfg_is_identical_for_both_project_profile_selectors(monkeypatch):
    import types

    class Scene:
        num_envs = 0

    class NativeConfig:
        def __init__(self):
            self.scene = Scene()
            self.debug_vis = True

    module_name = "isaaclab_tasks.direct.quadcopter.quadcopter_env"
    native_module = types.ModuleType(module_name)
    native_module.QuadcopterEnvCfg = NativeConfig
    monkeypatch.setitem(sys.modules, module_name, native_module)

    balanced = selected_env_cfg(
        "Isaac-Quadcopter-Direct-v0", 2, contract_profile="balanced_v3"
    )
    survival = selected_env_cfg(
        "Isaac-Quadcopter-Direct-v0", 2, contract_profile="survival_v2"
    )
    v4 = selected_env_cfg(
        "Isaac-Quadcopter-Direct-v0", 2, contract_profile="balanced_v4"
    )
    assert type(balanced) is type(survival) is type(v4) is NativeConfig
    assert balanced.scene.num_envs == survival.scene.num_envs == v4.scene.num_envs == 2
    assert balanced.debug_vis is survival.debug_vis is v4.debug_vis is False


def test_paired_gust_evidence_keeps_expected_submitted_and_measured_values_distinct():
    report = _assess_gust_impulse_evidence(
        expected_impulse_w=torch.tensor([[0.02, 0.0, 0.0]]),
        submitted_impulse_w=torch.tensor([[0.02, 0.0, 0.0]]),
        baseline_delta_momentum_w=torch.tensor([[0.001, 0.0, 0.01]]),
        gust_delta_momentum_w=torch.tensor([[0.021, 0.0, 0.03]]),
        recorded_gust_delta_momentum_w=torch.tensor([[0.021, 0.0, 0.03]]),
        response_absolute_tolerance_n_s=5.0e-4,
        response_relative_tolerance=0.10,
    )
    assert report["passed"] is True
    assert report["expected_impulse_w_n_s"][0] == pytest.approx([0.02, 0.0, 0.0])
    assert report["submitted_force_time_integral_w_n_s"][0] == pytest.approx(
        [0.02, 0.0, 0.0]
    )
    assert report["paired_measured_gust_response_w_n_s"][0] == pytest.approx(
        [0.02, 0.0, 0.0]
    )
    assert report["horizontal_only"] is True


def test_paired_gust_evidence_fails_closed_outside_predeclared_tolerance():
    report = _assess_gust_impulse_evidence(
        expected_impulse_w=torch.tensor([[0.02, 0.0, 0.0]]),
        submitted_impulse_w=torch.tensor([[0.02, 0.0, 0.0]]),
        baseline_delta_momentum_w=torch.zeros(1, 3),
        gust_delta_momentum_w=torch.tensor([[0.03, 0.0, 0.0]]),
        recorded_gust_delta_momentum_w=torch.tensor([[0.03, 0.0, 0.0]]),
        response_absolute_tolerance_n_s=5.0e-4,
        response_relative_tolerance=0.10,
    )
    assert report["passed"] is False
    assert report["paired_response_vector_error_n_s"][0] > report[
        "effective_physical_response_tolerance_n_s"
    ][0]


def test_terminal_snapshot_assessment_checks_pre_reset_geometry_speed_mask_and_cause():
    observation = torch.zeros(2, 12)
    observation[0, :3] = torch.tensor([3.0, 4.0, 0.0])
    observation[0, 9:12] = torch.tensor([0.6, 0.8, 0.0])
    report = _assess_terminal_snapshot(
        done=torch.tensor([True, False]),
        terminal_mask=torch.tensor([True, False]),
        terminal_observation=observation,
        terminal_goal_w=torch.tensor([[1.0, 0.0, 0.75], [0.0, 0.0, 0.0]]),
        terminal_position_w=torch.tensor([[0.0, 0.0, 0.75], [0.0, 0.0, 0.0]]),
        terminal_distance_m=torch.tensor([1.0, 0.0]),
        terminal_speed_mps=torch.tensor([5.0, 0.0]),
        terminal_failure_cause=torch.tensor([3, 0]),
        expected_failure_cause=3,
    )
    assert report["passed"] is True

    wrong_mask = _assess_terminal_snapshot(
        done=torch.tensor([True, False]),
        terminal_mask=torch.tensor([False, False]),
        terminal_observation=observation,
        terminal_goal_w=torch.tensor([[1.0, 0.0, 0.75], [0.0, 0.0, 0.0]]),
        terminal_position_w=torch.tensor([[0.0, 0.0, 0.75], [0.0, 0.0, 0.0]]),
        terminal_distance_m=torch.tensor([1.0, 0.0]),
        terminal_speed_mps=torch.tensor([5.0, 0.0]),
        terminal_failure_cause=torch.tensor([3, 0]),
        expected_failure_cause=3,
    )
    assert wrong_mask["passed"] is False
    assert wrong_mask["terminal_mask_exact"] is False


def test_live_environment_and_audit_source_pin_the_instantaneous_world_frame_gust_contract():
    env_source = (
        ROOT
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/env.py"
    ).read_text(encoding="utf-8")
    audit_source = (ROOT / "scripts/drone_inspect_asset.py").read_text(encoding="utf-8")
    smoke_source = (ROOT / "scripts/drone_smoke_env.py").read_text(encoding="utf-8")
    spec = (ROOT / "docs/crazyflie_task_spec.md").read_text(encoding="utf-8")

    assert "instantaneous_wrench_composer.set_forces_and_torques" in env_source
    assert "is_global=True" in env_source
    assert "failure = classify_failure_causes(" in env_source
    assert '"api": "Articulation.instantaneous_wrench_composer.set_forces_and_torques"' in audit_source
    assert '"force_frame": "world"' in audit_source
    assert '"application_point": "body_center_of_mass_no_offset_argument"' in audit_source
    assert "paired_identical_action_rollouts_same_live_environment_v1" in smoke_source
    assert "without_physics_injection" in smoke_source
    assert "max(0.0005 N s, 10% * expected_norm)" in spec
