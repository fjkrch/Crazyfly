"""CPU-safe contract tests for the training-only mixed Crazyflie task."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_train as training_script  # noqa: E402

from g1_fly_control.tasks.crazyflie.registration import (
    COMMAND_FOLLOW_TASK_ID,
    COMMAND_FOLLOW_WIDE_TASK_ID,
    COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    MIXED_TRAINING_TASK_ID,
    REGISTERED_TASK_IDS,
    TASK_IDS,
    TASK_TO_CFG,
    register_tasks,
)


def test_mixed_task_is_registered_but_not_an_official_evaluation_id():
    assert TASK_IDS == (
        "FlyCrazyflie-WaypointReach-v0",
        "FlyCrazyflie-WaypointSwitch-v0",
        "FlyCrazyflie-GustRecovery-v0",
    )
    assert MIXED_TRAINING_TASK_ID == "FlyCrazyflie-Mixed-v0"
    assert REGISTERED_TASK_IDS == TASK_IDS + (
        MIXED_TRAINING_TASK_ID,
        COMMAND_FOLLOW_TASK_ID,
        COMMAND_FOLLOW_WIDE_TASK_ID,
        COMMAND_FOLLOW_WIDE_WIND_TASK_ID,
    )
    assert TASK_TO_CFG[MIXED_TRAINING_TASK_ID].endswith(
        ":BalancedMixedTrainingEnvCfg"
    )


def test_mixed_registration_is_idempotent():
    assert register_tasks() == REGISTERED_TASK_IDS
    assert register_tasks() == REGISTERED_TASK_IDS
    assert gym.spec(MIXED_TRAINING_TASK_ID).kwargs["env_cfg_entry_point"].endswith(
        ":BalancedMixedTrainingEnvCfg"
    )


def _initial_mixed_environment_audit_fields() -> dict[str, object]:
    """Fields present before the first simulator step/checkpoint."""

    return {
        "num_envs": 4,
        "mixed_scenario_names_by_code": (
            "waypoint_reach",
            "waypoint_switch",
            "gust_recovery",
        ),
        "mixed_episode_index": torch.zeros(4, dtype=torch.long),
        "mixed_scenario_episode_starts": torch.zeros(3, dtype=torch.long),
        "episode_scenario_code": torch.zeros(4, dtype=torch.long),
        "terminal_scenario_code": torch.zeros(4, dtype=torch.long),
        "terminal_mask": torch.zeros(4, dtype=torch.bool),
    }


def test_initial_mixed_checkpoint_snapshot_has_every_required_audit_field():
    env = SimpleNamespace(extras={}, **_initial_mixed_environment_audit_fields())

    snapshot = training_script._mixed_scenario_snapshot(
        env,
        task=MIXED_TRAINING_TASK_ID,
        seed=0,
    )

    assert snapshot is not None
    assert snapshot["scenario_names_by_code"] == [
        "waypoint_reach",
        "waypoint_switch",
        "gust_recovery",
    ]
    assert snapshot["episode_starts_by_scenario"] == {
        "waypoint_reach": 0,
        "waypoint_switch": 0,
        "gust_recovery": 0,
    }
    assert snapshot["total_episode_starts"] == 0
    assert snapshot["episode_start_proportions"] == {
        "waypoint_reach": 0.0,
        "waypoint_switch": 0.0,
        "gust_recovery": 0.0,
    }
    assert snapshot["current_environment_counts_by_scenario"] == {
        "waypoint_reach": 4,
        "waypoint_switch": 0,
        "gust_recovery": 0,
    }
    assert snapshot["last_step_terminal_counts_by_scenario"] == {
        "waypoint_reach": 0,
        "waypoint_switch": 0,
        "gust_recovery": 0,
    }
    assert snapshot["episode_scenario_codes"] == [0, 0, 0, 0]
    assert snapshot["terminal_scenario_codes"] == [0, 0, 0, 0]
    assert snapshot["terminal_mask"] == [False, False, False, False]
    assert snapshot["task_schedule_state"] == {
        "schema_version": 1,
        "kind": training_script.MIXED_TASK_SCHEDULE_STATE_KIND,
        "contract": training_script.mixed_scenario_contract_payload(seed=0),
        "num_envs": 4,
        "per_environment_episode_index": [0, 0, 0, 0],
        "scenario_episode_starts": [0, 0, 0],
        "next_scenario_codes": [0, 1, 2, 0],
        "total_episode_starts": 0,
    }


@pytest.mark.parametrize("missing_name", tuple(_initial_mixed_environment_audit_fields()))
def test_mixed_checkpoint_snapshot_fails_closed_for_each_missing_audit_field(missing_name):
    fields = _initial_mixed_environment_audit_fields()
    fields.pop(missing_name)
    env = SimpleNamespace(extras={}, **fields)

    with pytest.raises(RuntimeError, match=missing_name):
        training_script._mixed_scenario_snapshot(
            env,
            task=MIXED_TRAINING_TASK_ID,
            seed=0,
        )


def test_environment_publishes_mixed_name_codebook_before_first_step():
    source = (
        ROOT
        / "source/g1_fly_control/g1_fly_control/tasks/crazyflie/env.py"
    ).read_text(encoding="utf-8")
    constructor_assignment = "self.mixed_scenario_names_by_code = MIXED_SCENARIO_NAMES"
    extras_assignment = 'self.extras["mixed_scenario_names_by_code"] = MIXED_SCENARIO_NAMES'

    assert source.index(constructor_assignment) < source.index(extras_assignment)


def _schedule_env(num_envs: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        mixed_episode_index=torch.zeros(num_envs, dtype=torch.long),
        mixed_scenario_episode_starts=torch.zeros(3, dtype=torch.long),
    )


def _start_mixed_episodes(
    env: SimpleNamespace,
    env_ids: list[int],
    *,
    seed: int,
) -> list[int]:
    ids = torch.tensor(env_ids, dtype=torch.long)
    assigned = training_script.mixed_scenario_codes(
        ids, env.mixed_episode_index[ids], seed=seed
    )
    env.mixed_episode_index[ids] += 1
    env.mixed_scenario_episode_starts += torch.bincount(assigned, minlength=3)
    return assigned.tolist()


def test_mixed_pause_resume_preserves_sequence_and_cumulative_provenance():
    seed = 7
    uninterrupted = _schedule_env()
    for env_ids in ([0, 1, 2, 3], [0, 2], [1], [0, 1, 3], [2, 3]):
        _start_mixed_episodes(uninterrupted, list(env_ids), seed=seed)

    checkpoint_state = training_script._capture_mixed_task_schedule_state(
        uninterrupted, task=MIXED_TRAINING_TASK_ID, seed=seed
    )
    assert checkpoint_state is not None
    assert checkpoint_state["total_episode_starts"] == 12
    assert sum(checkpoint_state["scenario_episode_starts"]) == 12

    resumed = _schedule_env()
    provenance = training_script._restore_mixed_task_schedule_state(
        resumed,
        task=MIXED_TRAINING_TASK_ID,
        seed=seed,
        state=deepcopy(checkpoint_state),
    )
    assert provenance == {
        "schema_version": 1,
        "restored_from_checkpoint": True,
        "restored_before_resume_environment_reset": True,
        "checkpoint_state": checkpoint_state,
    }
    assert resumed.mixed_episode_index.tolist() == uninterrupted.mixed_episode_index.tolist()
    assert (
        resumed.mixed_scenario_episode_starts.tolist()
        == uninterrupted.mixed_scenario_episode_starts.tolist()
    )

    # RecurrentPPO deliberately performs a full environment reset after a
    # restart. Both timelines must therefore assign the same next scenario to
    # every environment and retain exactly the same cumulative evidence.
    uninterrupted_next = _start_mixed_episodes(
        uninterrupted, [0, 1, 2, 3], seed=seed
    )
    resumed_next = _start_mixed_episodes(resumed, [0, 1, 2, 3], seed=seed)
    assert resumed_next == uninterrupted_next
    assert resumed.mixed_episode_index.tolist() == uninterrupted.mixed_episode_index.tolist()
    assert (
        resumed.mixed_scenario_episode_starts.tolist()
        == uninterrupted.mixed_scenario_episode_starts.tolist()
    )
    assert training_script._capture_mixed_task_schedule_state(
        resumed, task=MIXED_TRAINING_TASK_ID, seed=seed
    ) == training_script._capture_mixed_task_schedule_state(
        uninterrupted, task=MIXED_TRAINING_TASK_ID, seed=seed
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda state: state.pop("next_scenario_codes"), "closed schema"),
        (
            lambda state: state.__setitem__("scenario_episode_starts", [99, 0, 0]),
            "cumulative starts",
        ),
        (
            lambda state: state.__setitem__("next_scenario_codes", [2, 2, 2, 2]),
            "next scenario codes",
        ),
        (
            lambda state: state.__setitem__(
                "per_environment_episode_index", [1.0, 1.0, 1.0, 1.0]
            ),
            "integer dtype",
        ),
        (
            lambda state: state.__setitem__("total_episode_starts", 999),
            "total episode starts",
        ),
    ),
)
def test_mixed_resume_fails_closed_on_malformed_schedule_state(mutation, message):
    seed = 3
    source = _schedule_env()
    _start_mixed_episodes(source, [0, 1, 2, 3], seed=seed)
    state = training_script._capture_mixed_task_schedule_state(
        source, task=MIXED_TRAINING_TASK_ID, seed=seed
    )
    assert state is not None
    mutation(state)
    destination = _schedule_env()

    with pytest.raises(RuntimeError, match=message):
        training_script._restore_mixed_task_schedule_state(
            destination,
            task=MIXED_TRAINING_TASK_ID,
            seed=seed,
            state=state,
        )
    assert destination.mixed_episode_index.tolist() == [0, 0, 0, 0]
    assert destination.mixed_scenario_episode_starts.tolist() == [0, 0, 0]


def test_fixed_tasks_reject_injected_mixed_schedule_state_without_mutation():
    env = _schedule_env()
    with pytest.raises(RuntimeError, match="fixed task checkpoint"):
        training_script._restore_mixed_task_schedule_state(
            env,
            task="FlyCrazyflie-WaypointReach-v0",
            seed=0,
            state={"unexpected": True},
        )
    assert env.mixed_episode_index.tolist() == [0, 0, 0, 0]
