"""Pure-CPU gates for the reset-invariant 60-cell all-fair queue."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_all_fair_queue_v4 as queue  # noqa: E402


CONFIG = (
    ROOT / "configs" / "experiments"
    / "crazyflie_command_all_fair_seeds0_1_2_1m.json"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue.validate_config(CONFIG)
    root = tmp_path_factory.mktemp("command-all-fair-v4") / "matrix"
    return config, queue.build_queue(config, root)


def test_exact_60_cell_three_seed_shape_and_lif_first_order(built):
    _, value = built
    expected = [
        (controller, seed, task)
        for controller in queue.CONTROLLERS
        for seed in queue.SEEDS
        for task in queue.TASKS
    ]
    assert value["schema_version"] == 4
    assert value["kind"] == "crazyflie_command_all_fair_queue_v4"
    assert value["job_count"] == 60
    assert value["predicted_training_interactions"] == 60_000_000
    assert value["predicted_evaluation_episodes"] == 960
    assert [
        (job["controller"], job["seed"], job["task"])
        for job in value["jobs"]
    ] == expected
    assert all(job["architecture_class"] == "lif" for job in value["jobs"][:48])
    assert all(
        job["architecture_class"] == "baseline" for job in value["jobs"][48:]
    )
    assert len({job["id"] for job in value["jobs"]}) == 60
    assert len({job["run_dir"] for job in value["jobs"]}) == 60
    assert len({job["evaluation_output"] for job in value["jobs"]}) == 60
    assert value["lif_job_count"] == 48
    assert value["baseline_job_count"] == 12
    assert value["maximum_parallel"] == 1
    assert value["failure_isolation"] is True


def test_still_wind_are_adjacent_and_paired_within_controller_seed(built):
    _, value = built
    for index in range(0, 60, 2):
        still, wind = value["jobs"][index : index + 2]
        assert still["controller"] == wind["controller"]
        assert still["seed"] == wind["seed"]
        assert still["command_schedule_seed"] == wind["command_schedule_seed"]
        assert still["paired_task_seed_key"] == wind["paired_task_seed_key"]
        assert still["task"] == queue.TASKS[0]
        assert wind["task"] == queue.TASKS[1]
        assert still["expected_fingerprint"] != wind["expected_fingerprint"]


def test_every_command_has_exact_seed_task_budget_and_v2_hyperparameters(built):
    config, value = built
    for job in value["jobs"]:
        train = job["training_command"]
        evaluate = job["evaluation_command"]
        for option, expected in (
            ("--task", job["task"]),
            ("--policy", job["policy"]),
            ("--seed", str(job["seed"])),
            ("--contract_profile", "command_v2"),
            ("--evaluation_protocol", "command_v2"),
            ("--total_interactions", "1000000"),
            ("--num_envs", "40"),
            ("--horizon", "100"),
            ("--microbatch_size", "40"),
            ("--ppo_epochs", "2"),
            ("--learning_rate", "0.0003"),
        ):
            assert train[train.index(option) + 1] == expected
        assert train[train.index("--connectome_manifest") + 1] == config["_leg_manifest"]
        assert train[train.index("--wing_connectome_manifest") + 1] == config["_wing_manifest"]
        assert train[train.index("--optic_connectome_manifest") + 1] == config["_optic_manifest"]
        assert evaluate[evaluate.index("--task") + 1] == job["task"]
        assert evaluate[evaluate.index("--policy") + 1] == job["policy"]
        assert evaluate[evaluate.index("--training_seed") + 1] == str(job["seed"])
        assert evaluate[evaluate.index("--protocol") + 1] == "command_v2"
        assert "--headless" in train and "--headless" in evaluate


def test_controller_reports_have_exact_actor_critic_state_and_core_contract(built):
    _, value = built
    assert set(value["controller_reports"]) == set(queue.CONTROLLERS)
    for controller, report in value["controller_reports"].items():
        assert report["actor_trainable_parameters"] == queue.ACTOR_PARAMETERS_BY_CONTROLLER[controller]
        assert report["critic_trainable_parameters"] == queue.CRITIC_PARAMETERS_BY_CONTROLLER[controller]
        assert report["total_dynamic_state_per_environment"] == queue.DYNAMIC_STATE_BY_CONTROLLER[controller]
        assert report["frozen_parameters"] == queue.FROZEN_PARAMETERS_BY_CONTROLLER[controller]
        assert report["fusion_contract"] == queue.FUSION_BY_CONTROLLER[controller]
        matching_job = next(
            job for job in value["jobs"] if job["controller"] == controller
        )
        assert queue._valid_controller_report(matching_job, report)


def test_controller_report_gate_rejects_each_exact_capacity_drift(built):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][0]))
    report = json.loads(json.dumps(value["controller_reports"][job["controller"]]))
    report["actor_trainable_parameters"] += 1
    assert not queue._valid_controller_report(job, report)
    report = json.loads(json.dumps(value["controller_reports"][job["controller"]]))
    report["per_core_checksums"]["primary"] = "f" * 64
    assert not queue._valid_controller_report(job, report)


def test_activity_gate_is_controller_specific_for_all_ten_controllers(built):
    _, value = built
    for controller in queue.CONTROLLERS:
        job = next(row for row in value["jobs"] if row["controller"] == controller)
        if controller in queue.LIF_CONTROLLERS:
            labels = queue.COMPONENTS_BY_CONTROLLER[controller]
            manifests = job["expected_connectome_manifests"]
            provenance = {}
            per_unit = []
            for label in labels:
                key = label if len(labels) > 1 else "primary"
                identity = queue._manifest_activity_identity(manifests[key])
                assert identity is not None
                provenance[label] = identity
                per_unit.extend(
                    {"id": f"{label}:unit-{index}"}
                    for index in range(identity["neuron_count"])
                )
        elif controller == "gru_matched":
            provenance = {
                "engineering": {
                    "kind": "matched_gru_hidden_state",
                    "biological_roles": False,
                    "unit_count": 33,
                }
            }
            per_unit = [{"id": f"gru:hidden:{index}"} for index in range(33)]
        else:
            provenance = {
                "engineering": {
                    "kind": "matched_mlp_post_activation_hidden_units",
                    "activation_values": "absolute_actual_post_activation_output",
                    "biological_roles": False,
                    "layer_widths": [61, 61],
                    "unit_count": 122,
                }
            }
            per_unit = [
                {"id": f"mlp:hidden_{index // 61}:{index % 61}"}
                for index in range(122)
            ]
        activity = {
            "controller": job["policy"],
            "unit_count": len(per_unit),
            "role_provenance": provenance,
            "per_unit": per_unit,
        }
        assert queue._valid_activity(job, activity)
        activity["per_unit"] = activity["per_unit"][:-1]
        activity["unit_count"] -= 1
        assert not queue._valid_activity(job, activity)


def test_live_contract_is_reset_invariant_and_hashes_are_imported(built):
    config, value = built
    assert config["_live_contract_hashes"] == queue.CONTRACT_SHA_BY_TASK
    assert value["live_command_training_contract_sha256"] == queue.CONTRACT_SHA_BY_TASK
    for task in queue.TASKS:
        wind = task == queue.TASKS[1]
        payload = queue.command_wide_training_contract_payload(wind_enabled=wind)
        clock = payload["training_schedule_clock"]
        assert clock["schedule_independent_of_episode_termination"] is True
        assert clock["episode_reset_advances_command_cursor"] is False
        assert clock["episode_reset_advances_training_wind_cursor"] is False


def test_v2_is_pinned_as_historical_only_and_v3_is_untouched(built):
    config, value = built
    historical = config["historical_v2_identity"]
    assert historical["fair_evidence_eligible"] is False
    assert value["historical_v2_fair_evidence_eligible"] is False
    for field, hash_field in (
        ("config", "config_sha256"),
        ("runner", "runner_sha256"),
        ("completed_queue", "completed_queue_sha256"),
        ("completed_report", "completed_report_sha256"),
    ):
        assert queue.sha256_file(ROOT / historical[field]) == historical[hash_field]
    preserved = config["preserved_v3_identity"]
    assert queue.sha256_file(ROOT / preserved["config"]) == preserved["config_sha256"]
    assert queue.sha256_file(ROOT / preserved["runner"]) == preserved["runner_sha256"]


def test_final_default_does_not_use_preserved_seed0_draft():
    draft = (
        ROOT / "configs" / "experiments"
        / "crazyflie_command_all_fair_seed0_1m.json"
    )
    assert draft.is_file()
    assert queue.DEFAULT_CONFIG == CONFIG
    assert "seeds0_1_2" in queue.DEFAULT_CONFIG.name


def test_all_60_fingerprints_bind_seed_source_and_live_contract(built):
    _, value = built
    assert len({job["expected_fingerprint"] for job in value["jobs"]}) == 60
    for job in value["jobs"]:
        resolved = job["fingerprint_payload"]["resolved_config"]
        assert resolved["seed"] == job["seed"]
        assert resolved["command_training_contract_sha256"] == job[
            "command_training_contract_sha256"
        ]
        assert isinstance(job["fingerprint_payload"]["source_sha256"], dict)
        assert job["evaluation_manifest_id"] == job["evaluation_manifest"]["manifest_id"]


def test_loaded_queue_reauthenticates_all_60_fingerprints_and_source(built):
    config, value = built
    config = dict(config)
    config["_output_root"] = value["output_root"]
    copied = json.loads(json.dumps(value))
    path = Path(value["queue_file"])
    queue._validate_loaded_queue(copied, config, path)
    copied["jobs"][17]["seed"] = 99
    with pytest.raises((KeyError, ValueError), match="stale|differs"):
        queue._validate_loaded_queue(copied, config, path)


def test_loaded_queue_rejects_every_launch_identity_mutation(
    built, monkeypatch
):
    config, canonical = built
    config = dict(config)
    config["_output_root"] = canonical["output_root"]
    path = Path(canonical["queue_file"])
    monkeypatch.setattr(queue, "build_queue", lambda *_args, **_kwargs: canonical)

    def replace_command_option(value, command_field, option, replacement):
        command = value["jobs"][0][command_field]
        command[command.index(option) + 1] = replacement

    mutations = {
        "output root": lambda value: value.__setitem__("output_root", "/tmp/wrong-root"),
        "queue path": lambda value: value.__setitem__("queue_file", "/tmp/wrong-queue.json"),
        "global pause path": lambda value: value.__setitem__("global_pause_file", "/tmp/wrong-pause"),
        "embedded config": lambda value: value["config"]["queue"].__setitem__(
            "resource_poll_interval_seconds", 0.01
        ),
        "run path": lambda value: value["jobs"][0].__setitem__("run_dir", "/tmp/wrong-run"),
        "checkpoint path": lambda value: value["jobs"][0].__setitem__("checkpoint", "/tmp/wrong.pt"),
        "manifest path": lambda value: value["jobs"][0].__setitem__("training_manifest", "/tmp/wrong.json"),
        "job pause path": lambda value: value["jobs"][0].__setitem__("pause_file", "/tmp/wrong-pause"),
        "evaluation path": lambda value: value["jobs"][0].__setitem__("evaluation_output", "/tmp/wrong-heldout.json"),
        "training command": lambda value: replace_command_option(
            value, "training_command", "--total_interactions", "1"
        ),
        "evaluation command": lambda value: replace_command_option(
            value, "evaluation_command", "--task", queue.TASKS[1]
        ),
        "interaction budget": lambda value: value["jobs"][0].__setitem__("total_interactions", 1),
        "update budget": lambda value: value["jobs"][0].__setitem__("expected_updates", 1),
        "controller priority": lambda value: value["jobs"][0].__setitem__("controller_priority", 9),
        "seed priority": lambda value: value["jobs"][0].__setitem__("seed_priority", 9),
        "task priority": lambda value: value["jobs"][0].__setitem__("task_priority", 9),
        "capacity": lambda value: value["jobs"][0].__setitem__(
            "expected_actor_trainable_parameters", 1
        ),
        "controller report hash": lambda value: value["jobs"][0].__setitem__(
            "controller_report_sha256", "f" * 64
        ),
        "fingerprint": lambda value: value["jobs"][0].__setitem__(
            "expected_fingerprint", "f" * 64
        ),
    }
    for label, mutate in mutations.items():
        candidate = json.loads(json.dumps(canonical))
        mutate(candidate)
        with pytest.raises(ValueError, match="immutable|canonical|differs") as error:
            queue._validate_loaded_queue(candidate, config, path)
        assert label, str(error.value)


def test_closed_config_rejects_seed_budget_parallel_and_history_drift(tmp_path):
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    changes = [
        ("seeds", [0], "seeds"),
        ("total_interactions_per_job", 500_000, "total_interactions_per_job"),
        ("controllers", ["original_lif"], "controllers"),
    ]
    for field, replacement, message in changes:
        changed = json.loads(json.dumps(base))
        changed[field] = replacement
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
    changed = json.loads(json.dumps(base))
    changed["historical_v2_identity"]["fair_evidence_eligible"] = True
    path = tmp_path / "changed-history.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="historical_v2_identity"):
        queue.validate_config(path)


def test_latency_gate_requires_exact_authenticated_live_forward_schema():
    valid = {
        "schema_version": 1,
        "source": "exact_policy_act_calls_used_by_evaluation_control_path",
        "call_site": (
            "policy.act(normalized_observation, recurrent_state, deterministic=True)"
        ),
        "clock": "time.perf_counter_ns_monotonic",
        "device": "cuda:0",
        "measurement_unit": "milliseconds_per_vectorized_policy_call",
        "batch_size": 16,
        "cuda_synchronized_before_and_after_call": True,
        "activity_recorder_hooks_in_scope": True,
        "action_producing_call_count": 600,
        "warmup_calls_excluded_from_statistics": 10,
        "warmup_exclusion_justification": (
            "exclude prefix calls that can include one-time lazy CUDA/kernel "
            "initialization; no extra forwards were executed"
        ),
        "sample_count": 590,
        "total_ms": 59.0,
        "mean_ms": 0.1,
        "p50_ms": 0.08,
        "p95_ms": 0.12,
        "p99_ms": 0.15,
        "max_ms": 0.2,
        "all_action_producing_calls_total_ms": 60.0,
    }
    assert queue.v3_queue._valid_inference_latency(valid)
    valid["cuda_synchronized_before_and_after_call"] = False
    assert not queue.v3_queue._valid_inference_latency(valid)


def test_dry_run_cli_persists_60_cells_without_starting_child(
    built, tmp_path, monkeypatch, capsys
):
    config, _ = built
    output = tmp_path / "output"
    resolved = dict(config)
    resolved["_output_root"] = str(output)
    monkeypatch.setattr(queue, "validate_config", lambda _path: dict(resolved))
    monkeypatch.setattr(
        queue,
        "_start_child",
        lambda *_args, **_kwargs: pytest.fail("dry run launched a child"),
    )
    assert queue.main(["--config", str(CONFIG), "--dry_run"]) == 0
    persisted = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "queue_summary.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "dry_run"
    assert persisted["counts"] == {"pending": 60}
    assert summary["job_count"] == 60
    assert summary["predicted_training_interactions"] == 60_000_000
    assert summary["predicted_evaluation_episodes"] == 960
    assert json.loads(capsys.readouterr().out)["job_count"] == 60
    with pytest.raises(SystemExit):
        queue.main(["--config", str(CONFIG), "--dry_run"])


class _FakeStream:
    def close(self):
        return None


class _FakeProcess:
    def __init__(self, job, exit_code):
        self.job = job
        self.exit_code = exit_code

    def poll(self):
        if self.exit_code == 0:
            self.job["eval_ok"] = True
        return self.exit_code


class _SequencedProcess:
    def __init__(self, job, exit_codes, *, succeed_on_zero=False):
        self.job = job
        self.exit_codes = iter(exit_codes)
        self.succeed_on_zero = succeed_on_zero

    def poll(self):
        value = next(self.exit_codes)
        if value == 0 and self.succeed_on_zero:
            self.job["eval_ok"] = True
        return value


def _mini_execution_queue(tmp_path: Path):
    jobs = []
    for index, name in enumerate(("first", "middle", "last"), start=1):
        jobs.append(
            {
                "id": name,
                "status": "pending",
                "training_status": "completed",
                "evaluation_status": "pending",
                "train_ok": True,
                "eval_ok": False,
                "checkpoint": str(tmp_path / name / "latest.pt"),
                "pause_file": str(tmp_path / name / "pause.request"),
                "attempts": [],
                "failure_history": [],
                "evaluation_command": ["evaluate", name],
                "training_command": ["train", name],
            }
        )
    return {
        "job_count": 3,
        "jobs": jobs,
        "maximum_parallel": 1,
        "global_pause_file": str(tmp_path / "pause.request"),
        "output_root": str(tmp_path),
        "config": {"queue": {"resource_poll_interval_seconds": 0.0}},
        "events": [],
        "revision": 0,
        "dry_run": True,
    }


def _patch_execution(monkeypatch, *, middle_fails):
    launched = []
    monkeypatch.setattr(queue, "valid_training", lambda job: job["train_ok"])
    monkeypatch.setattr(queue, "valid_evaluation", lambda job: job["eval_ok"])
    monkeypatch.setattr(
        queue,
        "_resource_sample",
        lambda _queue, stage: {"passed": True, "stage": stage, "swap_out_pages": 1},
    )
    monkeypatch.setattr(queue.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(queue.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        queue,
        "save_queue",
        lambda _path, value: queue._refresh_status(value),
    )

    def start(job, phase, _root):
        launched.append(job["id"])
        attempt = len(job["attempts"]) + 1
        record = {
            "attempt": attempt,
            "phase": phase,
            "started_utc": "start",
            "command": list(job[f"{phase}_command"]),
            "log": f"attempt-{attempt}.log",
            "archived_invalid_output": None,
            "pid": 1000 + attempt,
        }
        job["attempts"].append(record)
        job["status"] = "running"
        job[f"{phase}_status"] = "running"
        exit_code = 1 if middle_fails[0] and job["id"] == "middle" else 0
        return {
            "process": _FakeProcess(job, exit_code),
            "stream": _FakeStream(),
            "phase": phase,
            "job": job,
            "record": record,
        }

    monkeypatch.setattr(queue, "_start_child", start)
    return launched


def test_middle_failure_is_isolated_later_job_completes_and_resume_retries(
    tmp_path, monkeypatch
):
    value = _mini_execution_queue(tmp_path)
    middle_fails = [True]
    launched = _patch_execution(monkeypatch, middle_fails=middle_fails)
    result = queue.execute_queue(value, tmp_path / "queue.json", resume=False)
    assert result == 1
    assert launched == ["first", "middle", "last"]
    assert [job["status"] for job in value["jobs"]] == [
        "completed", "failed", "completed"
    ]
    assert value["status"] == "partial_failed"
    middle = value["jobs"][1]
    assert len(middle["failure_history"]) == 1
    assert middle["failure_history"][0]["phase"] == "evaluation"
    assert middle["attempts"][0]["exit_code"] == 1

    launched.clear()
    middle_fails[0] = False
    result = queue.execute_queue(value, tmp_path / "queue.json", resume=True)
    assert result == 0
    assert launched == ["middle"]
    assert all(job["status"] == "completed" for job in value["jobs"])
    assert value["status"] == "completed"
    assert len(middle["failure_history"]) == 1
    assert len(middle["attempts"]) == 2


def test_unsafe_precheck_defers_without_launching(tmp_path, monkeypatch):
    value = _mini_execution_queue(tmp_path)
    value["jobs"] = value["jobs"][:1]
    value["job_count"] = 1
    monkeypatch.setattr(queue, "valid_training", lambda job: job["train_ok"])
    monkeypatch.setattr(queue, "valid_evaluation", lambda job: job["eval_ok"])
    monkeypatch.setattr(
        queue,
        "_resource_sample",
        lambda _queue, stage: {
            "passed": False,
            "stage": stage,
            "telemetry_error": "unit-test",
        },
    )
    monkeypatch.setattr(queue.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        queue,
        "save_queue",
        lambda _path, payload: queue._refresh_status(payload),
    )
    monkeypatch.setattr(
        queue,
        "_start_child",
        lambda *_args, **_kwargs: pytest.fail("unsafe precheck launched child"),
    )
    assert queue.execute_queue(value, tmp_path / "queue.json", resume=False) == 4
    assert value["jobs"][0]["status"] == "pending"
    assert value["status"] == "incomplete"
    assert value["resource_block"]["telemetry_error"] == "unit-test"
    assert value["events"][-1]["event"] == "resource_precheck_deferred_without_launch"


def _patch_resource_execution_basics(monkeypatch):
    monkeypatch.setattr(queue, "valid_training", lambda job: job["train_ok"])
    monkeypatch.setattr(queue, "valid_evaluation", lambda job: job["eval_ok"])
    monkeypatch.setattr(queue.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(queue.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        queue,
        "save_queue",
        lambda _path, payload: queue._refresh_status(payload),
    )


def test_live_training_resource_breach_pauses_only_cell_then_later_cell_runs(
    tmp_path, monkeypatch
):
    value = _mini_execution_queue(tmp_path)
    value["jobs"] = [value["jobs"][0], value["jobs"][2]]
    value["job_count"] = 2
    first, later = value["jobs"]
    first["train_ok"] = False
    stages = []
    samples = iter(
        [
            {"passed": True, "swap_out_pages": 1},
            {
                "passed": False,
                "sustained_paging": False,
                "gpu_used_mib": 7000.0,
                "system_ram_percent": 50.0,
            },
            {"passed": True, "swap_out_pages": 1},
        ]
    )
    monkeypatch.setattr(
        queue,
        "_resource_sample",
        lambda _queue, stage: stages.append(stage) or {**next(samples), "stage": stage},
    )
    _patch_resource_execution_basics(monkeypatch)
    launched = []

    def start(job, phase, _root):
        launched.append((job["id"], phase))
        record = {
            "attempt": 1,
            "phase": phase,
            "started_utc": "start",
            "command": list(job[f"{phase}_command"]),
            "log": "attempt.log",
            "archived_invalid_output": None,
            "pid": 4321,
        }
        job["attempts"].append(record)
        job["active_process"] = {"pid": 4321, "phase": phase}
        job["status"] = job[f"{phase}_status"] = "running"
        process = (
            _SequencedProcess(job, [None, 3])
            if job is first
            else _SequencedProcess(job, [0], succeed_on_zero=True)
        )
        return {
            "process": process,
            "stream": _FakeStream(),
            "phase": phase,
            "job": job,
            "record": record,
        }

    monkeypatch.setattr(queue, "_start_child", start)
    assert queue.execute_queue(value, tmp_path / "queue.json", resume=False) == 1
    assert launched == [("first", "training"), ("last", "evaluation")]
    assert first["status"] == "paused"
    assert later["status"] == "completed"
    assert Path(first["pause_file"]).is_file()
    assert first["resource_gate_history"][0]["disposition"] == "paused_at_clean_checkpoint"
    assert stages == ["before_next_job", "active_child", "before_next_job"]


def test_live_evaluation_resource_breach_sigterms_only_cell_then_later_runs(
    tmp_path, monkeypatch
):
    value = _mini_execution_queue(tmp_path)
    value["jobs"] = [value["jobs"][0], value["jobs"][2]]
    value["job_count"] = 2
    first, later = value["jobs"]
    samples = iter(
        [
            {"passed": True, "swap_out_pages": 1},
            {
                "passed": False,
                "sustained_paging": False,
                "gpu_used_mib": 7000.0,
                "system_ram_percent": 50.0,
            },
            {"passed": True, "swap_out_pages": 1},
        ]
    )
    monkeypatch.setattr(
        queue,
        "_resource_sample",
        lambda _queue, stage: {**next(samples), "stage": stage},
    )
    _patch_resource_execution_basics(monkeypatch)
    killed = []
    monkeypatch.setattr(queue.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    launched = []

    def start(job, phase, _root):
        launched.append(job["id"])
        pid = 5000 + len(launched)
        record = {
            "attempt": 1,
            "phase": phase,
            "started_utc": "start",
            "command": list(job[f"{phase}_command"]),
            "log": "attempt.log",
            "archived_invalid_output": None,
            "pid": pid,
        }
        job["attempts"].append(record)
        job["active_process"] = {"pid": pid, "phase": phase}
        job["status"] = job[f"{phase}_status"] = "running"
        process = (
            _SequencedProcess(job, [None, -15])
            if job is first
            else _SequencedProcess(job, [0], succeed_on_zero=True)
        )
        return {
            "process": process,
            "stream": _FakeStream(),
            "phase": phase,
            "job": job,
            "record": record,
        }

    monkeypatch.setattr(queue, "_start_child", start)
    assert queue.execute_queue(value, tmp_path / "queue.json", resume=False) == 1
    assert launched == ["first", "last"]
    assert killed == [(5001, queue.signal.SIGTERM)]
    assert first["status"] == "failed"
    assert first["failure_history"][0]["phase"] == "evaluation"
    assert later["status"] == "completed"


def test_sustained_paging_remains_global_pause_and_stops_later_launch(
    tmp_path, monkeypatch
):
    value = _mini_execution_queue(tmp_path)
    value["jobs"] = value["jobs"][:2]
    value["job_count"] = 2
    first = value["jobs"][0]
    first["train_ok"] = False
    samples = iter(
        [
            {"passed": True, "swap_out_pages": 1},
            {"passed": False, "sustained_paging": True, "swap_out_pages": 3},
        ]
    )
    monkeypatch.setattr(
        queue,
        "_resource_sample",
        lambda _queue, stage: {**next(samples), "stage": stage},
    )
    _patch_resource_execution_basics(monkeypatch)
    launched = []

    def start(job, phase, _root):
        launched.append(job["id"])
        record = {
            "attempt": 1,
            "phase": phase,
            "started_utc": "start",
            "command": list(job[f"{phase}_command"]),
            "log": "attempt.log",
            "archived_invalid_output": None,
            "pid": 6001,
        }
        job["attempts"].append(record)
        job["active_process"] = {"pid": 6001, "phase": phase}
        job["status"] = job[f"{phase}_status"] = "running"
        return {
            "process": _SequencedProcess(job, [None, 3]),
            "stream": _FakeStream(),
            "phase": phase,
            "job": job,
            "record": record,
        }

    monkeypatch.setattr(queue, "_start_child", start)
    assert queue.execute_queue(value, tmp_path / "queue.json", resume=False) == 4
    assert launched == ["first"]
    assert Path(value["global_pause_file"]).is_file()
    assert value["jobs"][0]["status"] == "paused"
    assert value["jobs"][1]["status"] == "paused"
    assert any(event["event"] == "hard_global_resource_gate" for event in value["events"])


def test_status_never_reports_completion_until_every_cell_passes():
    value = {
        "dry_run": False,
        "job_count": 3,
        "jobs": [{"status": "completed"}, {"status": "failed"}, {"status": "completed"}],
    }
    queue._refresh_status(value)
    assert value["status"] == "partial_failed"
    value["jobs"][1]["status"] = "pending"
    queue._refresh_status(value)
    assert value["status"] == "incomplete"
    value["jobs"][1]["status"] = "completed"
    queue._refresh_status(value)
    assert value["status"] == "completed"


def test_v4_source_never_invokes_old_matrix_shell_or_network():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "execute_drone_matrix.sh" not in source
    assert "drone_run_matrix.py" not in source
    assert "requests." not in source
    assert "urllib" not in source
