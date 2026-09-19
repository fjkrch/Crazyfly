"""Pure gates for the additive 14-cell wide-command matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_queue_v2 as queue  # noqa: E402


CONFIG = (
    ROOT / "configs" / "experiments"
    / "crazyflie_command_optic_wind_seed0_1m.json"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue.validate_config(CONFIG)
    root = tmp_path_factory.mktemp("command-v2") / "matrix"
    return config, queue.build_queue(config, root)


def test_exact_14_cell_shape_is_lif_first_and_task_paired(built):
    _, value = built
    expected = [
        (controller, task)
        for controller in queue.CONTROLLERS
        for task in queue.TASKS
    ]
    assert value["kind"] == "crazyflie_command_big_matrix_queue_v2"
    assert value["job_count"] == 14
    assert value["predicted_training_interactions"] == 14_000_000
    assert value["predicted_evaluation_episodes"] == 224
    assert [(job["controller"], job["task"]) for job in value["jobs"]] == expected
    assert all(job["architecture_class"] == "lif" for job in value["jobs"][:10])
    assert all(job["architecture_class"] == "baseline" for job in value["jobs"][10:])
    assert all(job["seed"] == 0 for job in value["jobs"])
    assert all(job["total_interactions"] == 1_000_000 for job in value["jobs"])
    assert all(job["expected_updates"] == 250 for job in value["jobs"])
    assert len({job["id"] for job in value["jobs"]}) == 14
    assert len({job["run_dir"] for job in value["jobs"]}) == 14
    assert len({job["evaluation_output"] for job in value["jobs"]}) == 14
    assert value["maximum_parallel"] == 1
    assert value["resource_limits"]["maximum_parallel"] == 1

    for index in range(0, 14, 2):
        still, wind = value["jobs"][index : index + 2]
        assert still["controller"] == wind["controller"]
        assert still["seed"] == wind["seed"] == 0
        assert still["command_schedule_seed"] == wind["command_schedule_seed"] == 0
        assert still["paired_task_seed_key"] == wind["paired_task_seed_key"]
        assert still["task"] == queue.TASKS[0]
        assert wind["task"] == queue.TASKS[1]
        assert still["expected_fingerprint"] != wind["expected_fingerprint"]


def test_every_command_uses_wide_v2_explicit_task_and_optic_manifest(built):
    config, value = built
    for job in value["jobs"]:
        train = job["training_command"]
        evaluate = job["evaluation_command"]
        assert train[train.index("--task") + 1] == job["task"]
        assert train[train.index("--contract_profile") + 1] == "command_v2"
        assert train[train.index("--evaluation_protocol") + 1] == "command_v2"
        assert train[train.index("--policy") + 1] == job["policy"]
        assert train[train.index("--total_interactions") + 1] == "1000000"
        assert train[train.index("--num_envs") + 1] == "40"
        assert train[train.index("--optic_connectome_manifest") + 1] == config["_optic_manifest"]
        assert "--headless" in train
        assert evaluate[evaluate.index("--task") + 1] == job["task"]
        assert evaluate[evaluate.index("--protocol") + 1] == "command_v2"
        assert evaluate[evaluate.index("--policy") + 1] == job["policy"]
        assert "--headless" in evaluate
        text = " ".join(train + evaluate)
        assert "FlyCrazyflie-CommandFollow-v0" not in text
        assert "FlyCrazyflie-CommandFollowWind-v0" not in text


def test_aliases_are_only_labels_and_canonical_policy_ids_reach_children(built):
    _, value = built
    original = value["jobs"][0]
    rewired = value["jobs"][2]
    assert original["controller"] == "original_lif"
    assert original["policy"] == "frozen_lif_original"
    assert rewired["controller"] == "rewired_lif"
    assert rewired["policy"] == "frozen_lif_degree_rewired"
    assert value["controller_reports"]["optic_lif"]["actor_trainable_parameters"] == 4_776
    assert value["controller_reports"]["optic_lif"]["actor_parameter_match_passed"] is True


def test_expanded_envelope_and_per_task_contract_hashes_are_pinned(built):
    config, value = built
    assert config["command_envelope"] == {
        "maximum_horizontal_speed_mps": 1.0,
        "maximum_vertical_speed_mps": 0.5,
        "maximum_yaw_rate_radps": 1.5,
        "minimum_hold_steps": 25,
        "maximum_hold_steps": 100,
        "control_dt_s": 0.02,
        "simultaneous_axes": True,
    }
    still_hashes = {
        job["command_training_contract_sha256"]
        for job in value["jobs"] if job["task"] == queue.TASKS[0]
    }
    wind_hashes = {
        job["command_training_contract_sha256"]
        for job in value["jobs"] if job["task"] == queue.TASKS[1]
    }
    assert still_hashes == {queue.COMMAND_WIDE_STILL_CONTRACT_SHA256}
    assert wind_hashes == {queue.COMMAND_WIDE_WIND_CONTRACT_SHA256}
    assert still_hashes != wind_hashes


def test_prior_paging_evidence_forces_permanent_sequential_policy(built):
    config, value = built
    event = config["_prior_paging_event"]
    assert event["event"] == "hard_resource_gate"
    assert event["sample"]["sustained_paging"] is True
    assert config["queue"]["default_max_parallel"] == 1
    assert config["queue"]["maximum_parallel"] == 1
    assert value["parallelism_disposition"]["disposition"] == "v2_max_parallel_fixed_to_one"
    parser_source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "--max_parallel" not in parser_source
    assert "maximum_parallel_after_gate" not in parser_source


def test_execution_rejects_any_in_memory_parallelism_expansion(built, tmp_path):
    _, value = built
    changed = json.loads(json.dumps(value))
    changed["maximum_parallel"] = 2
    with pytest.raises(ValueError, match="permanently sequential"):
        queue.execute_queue(changed, tmp_path / "queue.json", resume=False)


def test_memory_limits_are_strict_exclusive_and_paging_is_a_hard_gate(built):
    config, value = built
    assert config["queue"]["gpu_used_mib_exclusive"] == 6963.2
    assert config["queue"]["system_ram_percent_exclusive"] == 90.0
    assert config["queue"]["sustained_paging_sample_count"] == 3
    assert value["resource_limits"] == {
        "gpu_used_mib_exclusive": 6963.2,
        "system_ram_percent_exclusive": 90.0,
        "sustained_paging_sample_count": 3,
        "default_max_parallel": 1,
        "maximum_parallel": 1,
    }


def test_closed_config_rejects_budget_task_and_parallel_changes(tmp_path: Path):
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    for field, value, message in (
        ("total_interactions_per_job", 500_000, "total_interactions_per_job"),
        ("tasks", ["FlyCrazyflie-CommandFollow-v0"], "tasks"),
    ):
        changed = json.loads(json.dumps(base))
        changed[field] = value
        path = tmp_path / f"changed-{field}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            queue.validate_config(path)
    changed = json.loads(json.dumps(base))
    changed["queue"]["maximum_parallel"] = 2
    path = tmp_path / "changed-parallel.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="queue differs"):
        queue.validate_config(path)


def test_dry_run_cli_persists_queue_without_starting_child(
    built, tmp_path: Path, monkeypatch, capsys
):
    config, _ = built
    output = tmp_path / "output"
    resolved = dict(config)
    resolved["_output_root"] = str(output)
    monkeypatch.setattr(queue, "validate_config", lambda _path: dict(resolved))
    monkeypatch.setattr(
        queue, "_start_child",
        lambda *_args, **_kwargs: pytest.fail("dry run launched a child"),
    )
    assert queue.main(["--config", str(CONFIG), "--dry_run"]) == 0
    persisted = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "dry_run"
    assert persisted["revision"] == 1
    assert persisted["counts"] == {"pending": 14}
    assert json.loads(capsys.readouterr().out)["job_count"] == 14
    with pytest.raises(SystemExit):
        queue.main(["--config", str(CONFIG), "--dry_run"])


def test_loaded_queue_rejects_changed_source_or_fingerprint(built, tmp_path, monkeypatch):
    config, value = built
    copied = json.loads(json.dumps(value))
    path = tmp_path / "queue.json"
    copied["queue_file"] = str(path)
    monkeypatch.setattr(
        queue, "_job_fingerprint",
        lambda *_args: ("f" * 64, {"stale": True}, {"manifest_id": "stale"}),
    )
    with pytest.raises(ValueError, match="fingerprint is stale"):
        queue._validate_loaded_queue(copied, config, path)


def test_existing_checkpoint_adds_resume_and_logs_never_overwrite(
    built, tmp_path: Path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][0]))
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    job.update(run_dir=str(run_dir), checkpoint=str(checkpoint), attempts=[])

    class Process:
        pid = 12345

    captured = {}

    def popen(command, **_kwargs):
        captured["command"] = command
        return Process()

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    handle = queue._start_child(job, "training", tmp_path)
    assert captured["command"][-1] == "--resume"
    handle["stream"].close()
    job["attempts"] = []
    with pytest.raises(FileExistsError):
        queue._start_child(job, "training", tmp_path)


def test_v1_queue_config_and_tests_remain_byte_exact():
    expected = {
        "scripts/crazyflie_command_queue.py": "fa24d00158ad0051a18321cead9a78119de65ee91e5848038f140d83643a5ebc",
        "scripts/crazyflie_command_parallel_gate.py": "9a9b5fa57e4e61a2143a9f8850c4001f1e98f447cbccfd3442378e47d8b5bac0",
        "configs/experiments/crazyflie_command_seed0_500k.json": "9120a52eb24ceb456ede809f12cdfff89497429680ce9a0932d734285492b830",
        "tests/unit/test_crazyflie_command_queue.py": "dccc5239bcaf004617f4bbc79d6247009c7860cf231a077ba6a1f91979cc3333",
    }
    assert {relative: queue.sha256_file(ROOT / relative) for relative in expected} == expected


def test_v2_source_never_invokes_old_matrix_shell_or_network():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "execute_drone_matrix.sh" not in source
    assert "drone_run_matrix.py" not in source
    assert "requests." not in source
    assert "urllib" not in source
    assert 'log.open("xb")' in source
