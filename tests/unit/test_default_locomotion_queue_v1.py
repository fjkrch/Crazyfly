"""Pure-CPU gates for the additive stock G1/Go1 locomotion dry run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import default_locomotion_queue_v1 as queue  # noqa: E402


CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1.json"
)


@pytest.fixture(scope="module")
def built():
    config = queue.validate_config(CONFIG)
    return config, *queue.build_queue_bundle(config)


def _all_jobs(shards):
    return [
        job
        for robot in ("g1", "go1")
        for job in shards[robot]["jobs"]
    ]


def test_exact_train_tasks_three_seeds_and_no_play_variants(built):
    _, index, shards = built
    expected_tasks = [
        "Isaac-Velocity-Flat-G1-v0",
        "Isaac-Velocity-Rough-G1-v0",
        "Isaac-Velocity-Flat-Unitree-Go1-v0",
        "Isaac-Velocity-Rough-Unitree-Go1-v0",
    ]
    assert index["task_ids"] == expected_tasks
    assert index["seeds"] == [0, 1, 2]
    jobs = _all_jobs(shards)
    assert len(jobs) == 12
    assert all("-Play-" not in job["task"] for job in jobs)
    assert [job["task"] for job in jobs] == [
        task for task in expected_tasks for _seed in range(3)
    ]
    assert [job["seed"] for job in jobs] == [0, 1, 2] * 4


def test_g1_and_go1_have_physically_separate_six_job_queues(built):
    _, index, shards = built
    root = Path(index["output_root"])
    assert set(shards) == {"g1", "go1"}
    assert shards["g1"]["queue_file"] == str(root / "g1" / "queue.json")
    assert shards["go1"]["queue_file"] == str(root / "go1" / "queue.json")
    for robot in ("g1", "go1"):
        assert shards[robot]["job_count"] == 6
        assert shards[robot]["counts"] == {"planned": 6}
        robot_root = root / robot
        for job in shards[robot]["jobs"]:
            assert job["robot"] == robot
            assert Path(job["working_directory"]).is_relative_to(robot_root)
            assert not Path(job["working_directory"]).is_relative_to(ROOT / "runs")
    assert {
        job["working_directory"] for job in _all_jobs(shards)
    }.__len__() == 12


def test_only_planar_velocity_and_yaw_commands_are_declared(built):
    config, index, shards = built
    expected_axes = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
    assert config["command_contract"]["axes"] == expected_axes
    assert index["command_contract"]["vertical_command_present"] is False
    assert index["command_contract"]["base_height_command_present"] is False
    assert index["command_contract"]["up_down_command_present"] is False
    for job in _all_jobs(shards):
        contract = job["command_contract"]
        assert contract["axes"] == expected_axes
        assert contract["vertical_command_present"] is False
        assert contract["up_down_command_present"] is False
        assert set(contract["ranges"]) == {
            "lin_vel_x_mps",
            "lin_vel_y_mps",
            "ang_vel_z_radps",
        }
    g1_rough = next(job for job in _all_jobs(shards) if job["id"] == "g1-rough-seed-0")
    assert g1_rough["command_contract"]["ranges"]["lin_vel_y_mps"] == [0.0, 0.0]


def test_commands_use_canonical_stock_trainer_and_only_naming_safety_overrides(built):
    _, _, shards = built
    for job in _all_jobs(shards):
        command = job["training_command"]
        assert command[:2] == [str(queue.ISAAC_PYTHON), str(queue.TRAINER)]
        for option, expected in (
            ("--task", job["task"]),
            ("--agent", "rsl_rl_cfg_entry_point"),
            ("--seed", str(job["seed"])),
            ("--num_envs", "1024"),
            ("--max_iterations", str(job["max_iterations"])),
            ("--device", "cuda:0"),
            ("--logger", "tensorboard"),
            ("--run_name", f"seed_{job['seed']}"),
        ):
            assert command[command.index(option) + 1] == expected
        assert "--headless" in command
        assert not any("hydra" in token.lower() for token in command)
        assert job["stock_defaults"] == {
            "rewards": True,
            "observations": True,
            "actions": True,
            "ppo": True,
            "default_experiment_name": job["native_log_root"].split("/")[-1],
            "save_interval": 50,
            "max_iterations": job["max_iterations"],
        }


def test_default_iteration_interaction_and_parameter_counts(built):
    _, index, shards = built
    expected = {
        ("g1", "flat"): (1500, 36_864_000, 123, 37, 85_925, 85_962, 81_281),
        ("g1", "rough"): (3000, 73_728_000, 310, 37, 328_229, 328_266, 323_585),
        ("go1", "flat"): (300, 7_372_800, 48, 12, 40_844, 40_856, 39_425),
        ("go1", "rough"): (1500, 36_864_000, 235, 12, 286_604, 286_616, 285_185),
    }
    for job in _all_jobs(shards):
        iterations, interactions, obs, action, actor_mlp, actor_total, critic = expected[
            (job["robot"], job["terrain"])
        ]
        assert job["max_iterations"] == iterations
        assert job["num_envs"] == 1024
        assert job["num_steps_per_env"] == 24
        assert job["expected_interactions"] == interactions
        assert job["expected_interactions"] == 1024 * 24 * iterations
        assert job["expected_final_checkpoint_basename"] == f"model_{iterations - 1}.pt"
        assert (job["observation_width"], job["action_width"]) == (obs, action)
        assert job["policy"]["actor_mlp_parameter_count"] == actor_mlp
        assert job["policy"]["actor_distribution_parameter_count"] == action
        assert job["policy"]["actor_trainable_parameter_count"] == actor_total
        assert actor_total == actor_mlp + action
        assert job["policy"]["critic_parameter_count"] == critic
    assert shards["g1"]["predicted_training_interactions"] == 331_776_000
    assert shards["go1"]["predicted_training_interactions"] == 132_710_400
    assert index["predicted_training_interactions"] == 464_486_400


def test_strict_resource_and_resume_contract_is_fail_closed(built):
    config, index, shards = built
    limits = config["queue"]
    assert limits["maximum_parallel"] == 1
    assert limits["gpu_used_mib_exclusive"] == 6963.2
    assert limits["system_ram_percent_exclusive"] == 90.0
    assert limits["resource_poll_interval_seconds"] == 5.0
    assert limits["sustained_paging_sample_count"] == 3
    assert limits["dry_run_only_until_reviewed"] is True
    assert index["execution_authorized"] is False
    assert all(shard["execution_authorized"] is False for shard in shards.values())
    for job in _all_jobs(shards):
        assert "restarts" in job["resume_semantics"]
        assert "not claimed exact" in job["resume_semantics"]


def test_installed_sources_and_frozen_artifacts_match_before_receipts(built):
    config, index, _ = built
    assert config["_source_verification"] == {
        "commit": queue.ISAACLAB_COMMIT,
        "file_count": 10,
        "file_set_sha256": "36fca3501177012b0e1abc498f1e56ee1d6b4e40b6425451c70227ab5dcf4d15",
        "all_match": True,
    }
    assert config["_frozen_g1_verification"] == {
        "file_count": 37,
        "mismatched": 0,
        "all_match": True,
    }
    assert config["_frozen_runs_verification"] == {
        "file_count": 1429,
        "byte_count": 336_929_906,
        "tree_sha256": "c609886c330fdb9462a4bf92b5c2fcffc8a49cbc212d03110ab5914e13df00f3",
        "all_match": True,
    }
    assert index["preservation_verification"]["custom_g1"]["all_match"] is True
    assert index["preservation_verification"]["frozen_runs"]["all_match"] is True


def test_dry_run_writes_two_queue_files_and_never_launches_training(
    built, tmp_path, monkeypatch
):
    config, _, _ = built
    isolated = dict(config)
    isolated["_output_root"] = str(tmp_path / "matrix")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run attempted to launch a process")

    monkeypatch.setattr(queue.subprocess, "Popen", forbidden)
    index = queue.write_dry_run(isolated)
    assert index["status"] == "verified_dry_run"
    assert (tmp_path / "matrix" / "matrix_index.json").is_file()
    assert (tmp_path / "matrix" / "g1" / "queue.json").is_file()
    assert (tmp_path / "matrix" / "go1" / "queue.json").is_file()
    assert not list((tmp_path / "matrix").rglob("model_*.pt"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        queue.write_dry_run(isolated)


def test_execute_is_explicitly_blocked_until_review_and_four_memory_smokes(built):
    with pytest.raises(RuntimeError, match="fail-closed"):
        queue.main(["--config", str(CONFIG), "--execute"])


def test_config_rejects_play_task_vertical_axis_and_frozen_runs_output(tmp_path):
    original = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutations = []
    play = json.loads(json.dumps(original))
    play["robots"][0]["tasks"][0]["task_id"] = "Isaac-Velocity-Flat-G1-Play-v0"
    mutations.append(play)
    vertical = json.loads(json.dumps(original))
    vertical["command_contract"]["axes"].append("lin_vel_z")
    vertical["command_contract"]["vertical_command_present"] = True
    mutations.append(vertical)
    frozen = json.loads(json.dumps(original))
    frozen["output_root"] = "runs/default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1"
    mutations.append(frozen)
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(mutation), encoding="utf-8")
        with pytest.raises(ValueError):
            queue.validate_config(path, verify_preservation=False)
