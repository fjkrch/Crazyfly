"""Pure CPU gates for the additive six-cell LIF-combination matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import crazyflie_command_combinations_queue_v3 as queue  # noqa: E402


CONFIG = (
    ROOT / "configs" / "experiments"
    / "crazyflie_command_combinations_seed0_1m.json"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config = queue.validate_config(CONFIG)
    root = tmp_path_factory.mktemp("command-combinations-v3") / "matrix"
    return config, queue.build_queue(config, root)


def test_exact_six_cell_shape_is_lif_only_and_task_paired(built):
    _, value = built
    expected = [
        (controller, task)
        for controller in queue.CONTROLLERS
        for task in queue.TASKS
    ]
    assert value["schema_version"] == 3
    assert value["kind"] == "crazyflie_command_combinations_queue_v3"
    assert value["job_count"] == 6
    assert value["predicted_training_interactions"] == 6_000_000
    assert value["predicted_evaluation_episodes"] == 96
    assert [(job["controller"], job["task"]) for job in value["jobs"]] == expected
    assert all(job["architecture_class"] == "lif" for job in value["jobs"])
    assert all(job["seed"] == 0 for job in value["jobs"])
    assert all(job["total_interactions"] == 1_000_000 for job in value["jobs"])
    assert all(job["expected_updates"] == 250 for job in value["jobs"])
    assert len({job["id"] for job in value["jobs"]}) == 6
    assert len({job["run_dir"] for job in value["jobs"]}) == 6
    assert len({job["evaluation_output"] for job in value["jobs"]}) == 6
    assert value["maximum_parallel"] == 1

    for index in range(0, 6, 2):
        still, wind = value["jobs"][index : index + 2]
        assert still["controller"] == wind["controller"]
        assert still["seed"] == wind["seed"] == 0
        assert still["command_schedule_seed"] == wind["command_schedule_seed"] == 0
        assert still["paired_task_seed_key"] == wind["paired_task_seed_key"]
        assert still["task"] == queue.TASKS[0]
        assert wind["task"] == queue.TASKS[1]
        assert still["expected_fingerprint"] != wind["expected_fingerprint"]


def test_exact_controller_components_and_no_baselines(built):
    config, value = built
    expected = {
        "leg_optic_lif": ["leg", "optic"],
        "wing_optic_lif": ["wing", "optic"],
        "leg_wing_optic_lif": ["leg", "wing", "optic"],
    }
    assert config["comparison_contract"]["combinations"] == expected
    assert config["comparison_contract"]["matched_gru_or_mlp_jobs_in_this_follow_on"] is False
    assert value["controller_order"] == list(expected)
    for job in value["jobs"]:
        assert job["combination_components"] == expected[job["controller"]]
        assert job["capacity_tier_core_count"] == len(expected[job["controller"]])
        assert job["policy"].endswith("_lif")
        assert "gru" not in job["policy"] and "mlp" not in job["policy"]


def test_every_command_uses_explicit_v2_task_and_all_manifests(built):
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
        assert train[train.index("--connectome_manifest") + 1] == config["_leg_manifest"]
        assert train[train.index("--wing_connectome_manifest") + 1] == config["_wing_manifest"]
        assert train[train.index("--optic_connectome_manifest") + 1] == config["_optic_manifest"]
        assert "--headless" in train
        assert evaluate[evaluate.index("--task") + 1] == job["task"]
        assert evaluate[evaluate.index("--protocol") + 1] == "command_v2"
        assert evaluate[evaluate.index("--policy") + 1] == job["policy"]
        assert "--headless" in evaluate


def test_controller_reports_expose_capacity_and_frozen_core_evidence(built):
    _, value = built
    reports = value["controller_reports"]
    assert set(reports) == set(queue.CONTROLLERS)
    expected_actor = {
        "leg_optic_lif": 9_224,
        "wing_optic_lif": 9_224,
        "leg_wing_optic_lif": 13_672,
    }
    expected_state = {
        "leg_optic_lif": 2_048,
        "wing_optic_lif": 2_048,
        "leg_wing_optic_lif": 3_072,
    }
    for label, report in reports.items():
        assert report["core_labels"] == list(queue.COMPONENTS_BY_CONTROLLER[label])
        assert report["actor_trainable_parameters"] == expected_actor[label]
        assert report["critic_trainable_parameters"] == 18_305
        assert report["total_dynamic_state_per_environment"] == expected_state[label]
        assert set(report["per_core_checksums"]) == set(queue.COMPONENTS_BY_CONTROLLER[label])
        assert all(len(value) == 64 for value in report["per_core_checksums"].values())


def test_config_pins_exact_capacity_and_completed_v2_provenance(built):
    config, _ = built
    assert config["comparison_contract"]["capacity_contract"] == {
        "actor_trainable_parameters": {
            "leg_optic_lif": 9_224,
            "wing_optic_lif": 9_224,
            "leg_wing_optic_lif": 13_672,
        },
        "critic_trainable_parameters": {
            "leg_optic_lif": 18_305,
            "wing_optic_lif": 18_305,
            "leg_wing_optic_lif": 18_305,
        },
        "total_dynamic_state_per_environment": {
            "leg_optic_lif": 2_048,
            "wing_optic_lif": 2_048,
            "leg_wing_optic_lif": 3_072,
        },
    }
    base = config["base_matrix_identity"]
    assert queue.sha256_file(ROOT / base["completed_queue"]) == base[
        "completed_queue_sha256"
    ]
    assert queue.sha256_file(ROOT / base["completed_report"]) == base[
        "completed_report_sha256"
    ]
    completed = json.loads((ROOT / base["completed_queue"]).read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["counts"] == {"completed": 14}
    assert len(completed["jobs"]) == 14
    assert all(
        job["status"] == job["training_status"] == job["evaluation_status"] == "completed"
        for job in completed["jobs"]
    )


def test_expanded_envelope_and_task_contract_hashes_are_pinned(built):
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


def test_fingerprints_bind_every_component_manifest(built):
    _, value = built
    for job in value["jobs"]:
        resolved = job["fingerprint_payload"]["resolved_config"]
        identity = json.dumps(resolved, sort_keys=True)
        for component in job["combination_components"]:
            assert component in identity
        assert job["evaluation_manifest_id"] == job["evaluation_manifest"]["manifest_id"]


def test_strict_resource_gates_and_historical_paging_evidence(built):
    config, value = built
    assert config["_prior_paging_event"]["event"] == "hard_resource_gate"
    assert config["_prior_paging_event"]["sample"]["sustained_paging"] is True
    assert value["resource_limits"] == {
        "gpu_used_mib_exclusive": 6963.2,
        "system_ram_percent_exclusive": 90.0,
        "sustained_paging_sample_count": 3,
        "default_max_parallel": 1,
        "maximum_parallel": 1,
    }
    assert value["parallelism_disposition"]["disposition"] == "v3_max_parallel_fixed_to_one"


def test_execution_rejects_in_memory_parallelism_expansion(built, tmp_path):
    _, value = built
    changed = json.loads(json.dumps(value))
    changed["maximum_parallel"] = 2
    with pytest.raises(ValueError, match="permanently sequential"):
        queue.execute_queue(changed, tmp_path / "queue.json", resume=False)


def test_closed_config_rejects_budget_controller_and_parallel_changes(tmp_path):
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    for field, replacement, message in (
        ("total_interactions_per_job", 500_000, "total_interactions_per_job"),
        ("controllers", ["leg_optic_lif"], "controllers"),
    ):
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
    changed["base_matrix_identity"]["completed_queue_sha256"] = "0" * 64
    path = tmp_path / "changed-v2-provenance.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="base_matrix_identity"):
        queue.validate_config(path)
    changed = json.loads(json.dumps(base))
    changed["comparison_contract"]["capacity_contract"][
        "actor_trainable_parameters"
    ]["wing_optic_lif"] = 9_225
    path = tmp_path / "changed-capacity.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison_contract"):
        queue.validate_config(path)


def test_dry_run_cli_persists_v3_queue_without_starting_child(
    built, tmp_path: Path, monkeypatch, capsys
):
    config, _ = built
    output = tmp_path / "output"
    resolved = dict(config)
    resolved["_output_root"] = str(output)
    monkeypatch.setattr(queue, "validate_config", lambda _path: dict(resolved))
    monkeypatch.setattr(
        queue.v2_queue, "_start_child",
        lambda *_args, **_kwargs: pytest.fail("dry run launched a child"),
    )
    assert queue.main(["--config", str(CONFIG), "--dry_run"]) == 0
    persisted = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "queue_summary.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "dry_run"
    assert persisted["revision"] == 1
    assert persisted["counts"] == {"pending": 6}
    assert summary["schema_version"] == 3
    assert summary["job_count"] == 6
    assert json.loads(capsys.readouterr().out)["job_count"] == 6
    with pytest.raises(SystemExit):
        queue.main(["--config", str(CONFIG), "--dry_run"])


def test_sorted_json_roundtrip_preserves_wing_optic_order_authority(
    built, tmp_path
):
    config, value = built
    persisted = json.loads(json.dumps(value))
    path = tmp_path / "queue.json"
    persisted["queue_file"] = str(path)
    queue.save_queue(path, persisted)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    queue._validate_loaded_queue(loaded, config, path)

    wing_job = next(
        job for job in loaded["jobs"]
        if job["controller"] == "wing_optic_lif" and job["task"] == queue.TASKS[0]
    )
    wing_report = loaded["controller_reports"]["wing_optic_lif"]
    assert wing_report["core_labels"] == ["wing", "optic"]
    assert list(wing_report["per_core_checksums"]) == ["optic", "wing"]
    assert queue._valid_combination_report(wing_job, wing_report) is True


def test_loaded_queue_rejects_changed_source_or_fingerprint(built, tmp_path, monkeypatch):
    config, value = built
    copied = json.loads(json.dumps(value))
    path = tmp_path / "queue.json"
    copied["queue_file"] = str(path)
    monkeypatch.setattr(
        queue,
        "_job_fingerprint",
        lambda *_args: ("f" * 64, {"stale": True}, {"manifest_id": "stale"}),
    )
    with pytest.raises(ValueError, match="fingerprint is stale"):
        queue._validate_loaded_queue(copied, config, path)


def test_existing_checkpoint_adds_resume_and_attempt_logs_never_overwrite(
    built, tmp_path: Path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][2]))
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

    monkeypatch.setattr(queue.v2_queue.subprocess, "Popen", popen)
    handle = queue._start_child(job, "training", tmp_path)
    assert captured["command"][-1] == "--resume"
    handle["stream"].close()
    job["attempts"] = []
    with pytest.raises(FileExistsError):
        queue._start_child(job, "training", tmp_path)


def test_pause_request_uses_v3_schema(built, tmp_path):
    _, value = built
    changed = json.loads(json.dumps(value))
    changed["global_pause_file"] = str(tmp_path / "pause.request")
    queue._request_pause(changed, tmp_path / "queue.json", "unit_test")
    payload = json.loads((tmp_path / "pause.request").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["reason"] == "unit_test"


def test_v3_training_validator_rejects_any_per_core_checksum_drift(
    built, tmp_path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][0]))
    original_report = json.loads(
        json.dumps(value["controller_reports"][job["controller"]])
    )
    manifest_path = tmp_path / "training_manifest.json"
    job["training_manifest"] = str(manifest_path)
    manifest = {
        "controller_report": json.loads(json.dumps(original_report)),
        "core_checksum_before": job["expected_core_checksum"],
        "core_checksum_after": job["expected_core_checksum"],
        "per_core_checksums_before": job["expected_per_core_checksums"],
        "per_core_checksums_after": job["expected_per_core_checksums"],
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(queue, "_v2_valid_training", lambda _job: True)
    assert queue.valid_training(job) is True

    first = job["combination_components"][0]
    manifest["controller_report"]["per_core_checksums"][first] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert queue.valid_training(job) is False

    manifest["controller_report"] = original_report
    manifest["core_checksum_after"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert queue.valid_training(job) is False

    manifest["core_checksum_after"] = job["expected_core_checksum"]
    manifest["per_core_checksums_after"] = {
        **job["expected_per_core_checksums"],
        first: "d" * 64,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert queue.valid_training(job) is False


def test_v3_evaluation_validator_requires_activity_from_every_core(
    built, tmp_path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][4]))
    report = json.loads(json.dumps(value["controller_reports"][job["controller"]]))
    identities = job["fingerprint_payload"]["resolved_config"][
        "lif_connectome_composition"
    ]["connectomes"]
    provenance = {
        label: {
            "manifest": identities[label]["manifest"],
            "manifest_sha256": identities[label]["manifest_sha256"],
            "neurons": identities[label]["neurons_path"]["path"],
            "neurons_sha256": identities[label]["neurons_path"]["sha256"],
            "neuron_count": 1,
        }
        for label in job["combination_components"]
    }
    activity = {
        "controller": job["policy"],
        "unit_count": len(job["combination_components"]),
        "role_provenance": provenance,
        "per_unit": [
            {"id": f"{label}:unit-0"}
            for label in job["combination_components"]
        ],
    }
    output = tmp_path / "heldout.json"
    job["evaluation_output"] = str(output)
    latency = {
        "schema_version": 1,
        "source": "exact_policy_act_calls_used_by_evaluation_control_path",
        "call_site": (
            "policy.act(normalized_observation, recurrent_state, "
            "deterministic=True)"
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
    value_out = {
        "controller_report": report,
        "activity": activity,
        "inference_latency": latency,
        "integrity": {"inference_latency_from_actual_forward": True},
    }
    output.write_text(json.dumps(value_out, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(queue, "_v2_valid_evaluation", lambda _job: True)
    assert queue.valid_evaluation(job) is True

    missing = job["combination_components"][-1]
    removed_provenance = value_out["activity"]["role_provenance"].pop(missing)
    output.write_text(json.dumps(value_out, sort_keys=True), encoding="utf-8")
    assert queue.valid_evaluation(job) is False

    value_out["activity"]["role_provenance"][missing] = removed_provenance
    value_out["inference_latency"]["source"] = "synthetic_benchmark"
    output.write_text(json.dumps(value_out, sort_keys=True), encoding="utf-8")
    assert queue.valid_evaluation(job) is False


def test_v3_evaluation_validator_rejects_unauthenticated_latency(
    built, tmp_path, monkeypatch
):
    _, value = built
    job = json.loads(json.dumps(value["jobs"][0]))
    output = tmp_path / "heldout.json"
    job["evaluation_output"] = str(output)
    monkeypatch.setattr(queue, "_v2_valid_evaluation", lambda _job: True)
    output.write_text(
        json.dumps({
            "controller_report": value["controller_reports"][job["controller"]],
            "activity": {},
            "inference_latency": {
                "source": "synthetic_benchmark",
                "action_producing_call_count": 600,
            },
            "integrity": {"inference_latency_from_actual_forward": False},
        }),
        encoding="utf-8",
    )
    assert queue.valid_evaluation(job) is False


def test_v2_config_and_runner_remain_byte_exact():
    expected = {
        "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json":
            "bf7369d74d119784eb6a932cbeb57734fa93b7fdda5457e3e46decfb6f140c44",
        "scripts/crazyflie_command_queue_v2.py":
            "16d7035665045ebd2128a3b369e98f49828d9f3b5c3da0685a377c9cb8447dd4",
    }
    assert {relative: queue.sha256_file(ROOT / relative) for relative in expected} == expected


def test_v3_source_never_invokes_old_matrix_shell_or_network():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "execute_drone_matrix.sh" not in source
    assert "drone_run_matrix.py" not in source
    assert "requests." not in source
    assert "urllib" not in source
