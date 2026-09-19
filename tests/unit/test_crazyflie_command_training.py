from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_train as trainer  # noqa: E402
from drone_bootstrap import (  # noqa: E402
    observation_contract_for_task,
    selected_env_cfg,
    task_contract_payload,
    validate_contract_profile,
)
from g1_fly_control.crazyflie.command_evaluation import (  # noqa: E402
    EPISODE_COUNT,
    EPISODE_STEPS,
    EVALUATION_SEED,
    protocol_payload,
    protocol_sha256,
)
from g1_fly_control.tasks.crazyflie.command_logic import (  # noqa: E402
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_TRACKING_CONTRACT_SHA256,
    command_follow_contract_payload,
    command_training_contract_payload,
)


def _args(policy: str) -> argparse.Namespace:
    return argparse.Namespace(
        task=COMMAND_FOLLOW_TASK_ID,
        contract_profile=trainer.COMMAND_CONTRACT_PROFILE,
        evaluation_protocol=trainer.COMMAND_EVALUATION_PROTOCOL,
        policy=policy,
        seed=0,
        num_envs=40,
        total_interactions=500_000,
        horizon=100,
        microbatch_size=40,
        ppo_epochs=2,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        max_grad_norm=1.0,
        target_kl=0.05,
        checkpoint_every_updates=25,
        connectome_manifest=trainer.DEFAULT_CONNECTOME,
        wing_connectome_manifest=trainer.DEFAULT_WING_CONNECTOME,
        rewire_seed=20260916,
        rewire_manifest=trainer.DEFAULT_REWIRE_MANIFEST,
        warm_start_checkpoint=None,
    )


@pytest.mark.parametrize("policy", trainer.POLICIES)
def test_command_standalone_config_fingerprints_contract_for_all_six_controllers(
    policy: str,
) -> None:
    resolved, evaluation = trainer.standalone_resolved_config(_args(policy))

    assert resolved["task"] == COMMAND_FOLLOW_TASK_ID
    assert resolved["contract_profile"] == "command_v1"
    assert resolved["controller"] == policy
    assert resolved["command_follow_contract"] == command_follow_contract_payload()
    assert resolved["command_training_contract"] == command_training_contract_payload()
    assert (
        resolved["command_training_contract_sha256"]
        == COMMAND_TRACKING_CONTRACT_SHA256
    )
    assert resolved["evaluation_protocol"] == "command_v1"
    assert "mixed_scenario_contract" not in resolved
    assert "switch_target_curriculum" not in resolved
    assert ("wing_extension" in resolved) is (
        policy in {"wing_lif", "leg_wing_lif"}
    )
    assert evaluation["task"] == COMMAND_FOLLOW_TASK_ID
    assert evaluation["evaluation_protocol"] == protocol_payload()
    assert evaluation["evaluation_protocol_sha256"] == protocol_sha256()
    assert evaluation["evaluation_protocol"]["evaluation_seed"] == EVALUATION_SEED
    assert evaluation["evaluation_protocol"]["episodes"] == EPISODE_COUNT == 16
    assert evaluation["evaluation_protocol"]["steps_per_episode"] == EPISODE_STEPS == 600
    assert len(evaluation["manifest_id"]) == 64


@pytest.mark.parametrize(
    ("task", "profile", "protocol", "message"),
    (
        (
            COMMAND_FOLLOW_TASK_ID,
            "balanced_v4",
            "command_v1",
            "requires contract_profile=command_v1",
        ),
        (
            COMMAND_FOLLOW_TASK_ID,
            "command_v1",
            "integration",
            "requires evaluation_protocol=command_v1",
        ),
        (
            "FlyCrazyflie-WaypointReach-v0",
            "command_v1",
            "integration",
            "valid only",
        ),
    ),
)
def test_command_contract_and_protocol_cannot_be_mixed_with_waypoint_profiles(
    task: str,
    profile: str,
    protocol: str,
    message: str,
) -> None:
    args = _args("mlp_normal")
    args.task = task
    args.contract_profile = profile
    args.evaluation_protocol = protocol
    with pytest.raises(ValueError, match=message):
        trainer.standalone_resolved_config(args)


class _ScheduleEnv:
    def __init__(self, state: dict | None = None):
        self.num_envs = 2
        self.training_interactions = 12_000
        self.command_tracking_contract = command_training_contract_payload()
        self._state = state or {
            "schema_version": 1,
            "kind": "crazyflie_command_schedule_state_v1",
            "contract_sha256": COMMAND_TRACKING_CONTRACT_SHA256,
            "num_envs": 2,
            "command_schedule_seed": 0,
            "training_interactions": self.training_interactions,
            "next_command_segment_index": [7, 8],
            "command_steps_remaining": [12, 34],
            "requested_command_body": [
                [0.25, 0.0, 0.0, 0.0],
                [0.0, -0.25, 0.15, -0.4],
            ],
            "command_category_code": [1, 3],
            "command_stage_index": [0, 0],
        }

    def command_schedule_state_dict(self):
        return deepcopy(self._state)

    def load_command_schedule_state_dict(self, state):
        if state["contract_sha256"] != COMMAND_TRACKING_CONTRACT_SHA256:
            raise ValueError("different task contract")
        if state["num_envs"] != self.num_envs:
            raise ValueError("environment count differs")
        self._state = deepcopy(state)
        self.training_interactions = state["training_interactions"]


def test_command_schedule_state_round_trips_for_checkpoint_resume() -> None:
    source = _ScheduleEnv()
    state = trainer._capture_command_task_schedule_state(
        source, task=COMMAND_FOLLOW_TASK_ID
    )
    assert state is not None

    destination = _ScheduleEnv()
    provenance = trainer._restore_command_task_schedule_state(
        destination,
        task=COMMAND_FOLLOW_TASK_ID,
        state=state,
    )

    assert destination.command_schedule_state_dict() == state
    assert provenance == {
        "schema_version": 1,
        "restored_from_checkpoint": True,
        "restored_before_resume_environment_reset": True,
        "state_sha256": trainer.canonical_sha256(state),
        "checkpoint_state": state,
    }
    snapshot = trainer._command_schedule_snapshot(
        destination, task=COMMAND_FOLLOW_TASK_ID
    )
    assert snapshot is not None
    assert snapshot["state"] == state
    assert snapshot["state_sha256"] == trainer.canonical_sha256(state)


def test_command_schedule_capture_rejects_wrong_contract() -> None:
    env = _ScheduleEnv()
    env._state["contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="different task contract"):
        trainer._capture_command_task_schedule_state(
            env, task=COMMAND_FOLLOW_TASK_ID
        )


def test_command_curriculum_and_live_telemetry_do_not_require_waypoint_fields() -> None:
    env = SimpleNamespace(
        command_tracking_contract=command_training_contract_payload(),
        training_interactions=250_000,
        active_training_curriculum_stage_index=2,
        active_training_curriculum_stage_payload={
            "start_interactions": 250_000,
            "maximum_horizontal_speed_m_s": 0.8,
            "maximum_vertical_speed_m_s": 0.4,
            "maximum_yaw_rate_rad_s": 1.2,
        },
        _robot=SimpleNamespace(
            data=SimpleNamespace(root_lin_vel_w=torch.tensor([[3.0, 4.0, 0.0]]))
        ),
    )

    curriculum = trainer._environment_curriculum_snapshot(
        env, expected_interactions=250_000
    )
    telemetry = trainer._live_task_telemetry(
        env, task=COMMAND_FOLLOW_TASK_ID
    )

    assert curriculum["active_stage_name"] == "command_stage_2"
    assert telemetry == {"speed_mean_m_s": 5.0}
    assert "goal_distance_mean_m" not in telemetry


def test_bootstrap_selects_only_the_command_v1_config(monkeypatch) -> None:
    command_cfg = type(
        "CommandFollowEnvCfg",
        (),
        {
            "__init__": lambda self: (
                setattr(self, "scene", SimpleNamespace(num_envs=0)),
                setattr(self, "debug_vis", True),
            )[-1],
        },
    )
    adapter = SimpleNamespace(validate_upstream_contract=lambda: None)
    config_module = SimpleNamespace(CommandFollowEnvCfg=command_cfg)
    monkeypatch.setitem(
        sys.modules,
        "g1_fly_control.tasks.crazyflie.adapter",
        adapter,
    )
    monkeypatch.setitem(
        sys.modules,
        "g1_fly_control.tasks.crazyflie.command_env_cfg",
        config_module,
    )

    cfg = selected_env_cfg(
        COMMAND_FOLLOW_TASK_ID,
        7,
        contract_profile="command_v1",
    )

    assert cfg.scene.num_envs == 7
    assert cfg.debug_vis is False
    assert validate_contract_profile(
        "command_v1", task=COMMAND_FOLLOW_TASK_ID
    ) == "command_v1"
    assert task_contract_payload("command_v1") == command_training_contract_payload()
    observation = observation_contract_for_task(COMMAND_FOLLOW_TASK_ID)
    assert observation["width"] == 12
    assert observation["names"] == command_training_contract_payload()[
        "observation_order"
    ]
    with pytest.raises(ValueError, match="requires contract_profile=command_v1"):
        selected_env_cfg(
            COMMAND_FOLLOW_TASK_ID,
            1,
            contract_profile="balanced_v4",
        )
