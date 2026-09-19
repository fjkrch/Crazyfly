"""Matrix conclusions must use validated training-seed results."""

import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import pytest

from scripts.summarize_matrix import markdown, summarize


def _matrix(tmp_path: Path, *, second_complete: bool = True, mismatch: bool = False) -> Path:
    scenario = "FlyG1-GoalReach-FreePosture-v0"
    jobs = []
    for condition in ("frozen_lif_original", "gru_trainable"):
        for seed in (0, 1):
            job_id = f"{condition}__seed-{seed}"
            checkpoint = str(tmp_path / job_id / "checkpoint.pt")
            result_path = tmp_path / job_id / "evaluation.json"
            result_path.parent.mkdir()
            Path(checkpoint).write_bytes(b"fixture checkpoint")
            policy = "frozen_lif" if condition == "frozen_lif_original" else "gru"
            total = 100 if policy == "frozen_lif" else 120
            trainable = 80 if policy == "frozen_lif" else 120
            training_manifest = {
                "status": "executed_matrix_training", "task": scenario, "policy": policy,
                "seed": seed, "checkpoint_path": checkpoint, "num_envs": 16,
                "ppo_config": {"horizon": 32}, "requested_interaction_budget": 512,
                "completed_iterations": 1, "actual_environment_interactions": 512,
                "history": [{
                    "iteration": 0, "loss": 0.1, "policy_loss": -0.01,
                    "value_loss": 0.2, "entropy": 1.0, "grad_norm": 0.5,
                    "approx_kl": 0.01, "attempted_kl": 0.01,
                    "rejected_step": 0.0, "accepted_epochs": 2.0,
                }],
                "model_total_parameters": total, "model_trainable_parameters": trainable,
                "model_frozen_parameters": total - trainable,
                "training_wall_time_s": float(10 + seed),
                "memory_samples": [
                    {
                        "stage": "rollout", "ram_used_gib": 7.0 + seed,
                        "ram_total_gib": 24.0, "ram_percent": 35.0 + seed,
                        "process_rss_mib": 1000.0 + seed,
                        "torch_cuda_peak_allocated_mib": 20.0,
                        "gpu_telemetry": "ok", "gpu_devices": [{
                            "gpu_index": "0", "gpu_total_mib": 8192.0,
                            "gpu_used_mib": 2000.0 + 100 * seed,
                        }],
                    },
                    {
                        "stage": "optimizer_update", "ram_used_gib": 8.0 + seed,
                        "ram_total_gib": 24.0, "ram_percent": 40.0 + seed,
                        "process_rss_mib": 1200.0 + seed,
                        "torch_cuda_peak_allocated_mib": 30.0,
                        "gpu_telemetry": "ok", "gpu_devices": [{
                            "gpu_index": "0", "gpu_total_mib": 8192.0,
                            "gpu_used_mib": 2200.0 + 100 * seed,
                        }],
                    },
                ],
            }
            training_path = result_path.parent / "manifest.json"
            training_path.write_text(json.dumps(training_manifest))
            training_metadata = {field: training_manifest[field] for field in (
                "requested_interaction_budget", "actual_environment_interactions",
                "completed_iterations", "model_total_parameters", "model_trainable_parameters",
                "model_frozen_parameters", "training_wall_time_s",
            )}
            success = condition == "gru_trainable"
            progress = float(seed + success)
            episode = {
                "episode_id": 0, "seed": seed, "success": success,
                "xy_progress_m": progress,
                "accumulated_goal_relative_progress_m": progress,
                "time_to_target_s": 2.0 if success else None,
                "mechanical_work_proxy": 10.0 + seed,
                "excessive_impact": 0.0,
                "joint_limit_frequency": 0.0,
                "saturation_frequency": 0.0,
                "goal_switch_count": 0, "push_count": 0,
                "push_impulse_vector_n_s": [0.0, 0.0, 0.0],
                "push_impulse_magnitude_n_s": 0.0,
                "recovery_success": None, "recovery_time_s": None,
                "recovery_attempt_count": 0, "recovery_success_count": 0,
                "paired_plan_sha256": "same-plan", "initial_state_sha256": (
                    "different" if mismatch and success else "same-initial"
                ),
            }
            per_seed = {
                "seed": seed, "episodes": 1,
                "success_rate": float(success),
                "mean_xy_progress_m": progress,
                "mean_accumulated_goal_relative_progress_m": progress,
                "mean_time_to_target_s": 2.0 if success else None,
                "mean_work_proxy": 10.0 + seed,
                "mean_excessive_impact": 0.0,
                "mean_joint_limit_frequency": 0.0,
                "mean_saturation_frequency": 0.0,
                "mean_goal_switch_count": 0.0,
                "mean_push_count": 0.0,
                "mean_push_impulse_magnitude_n_s": 0.0,
                "recovery_success_rate": None,
                "mean_recovery_time_s": None,
                "recovery_attempt_count": 0,
                "recovery_success_count": 0,
            }
            result = {
                "status": "executed", "task": scenario, "checkpoint": checkpoint,
                "training_seed": seed, "evaluation_seed": 101, "n_episodes": 1,
                "n_independent_training_seeds": 1,
                "per_seed": [per_seed],
                "episodes": [episode],
                "scenario": {"evaluation_protocol": "heldout_v1", "schedule": {"sha256": "a" * 64}},
            }
            result_path.write_text(json.dumps(result))
            passed = second_complete or seed == 0
            jobs.append({"id": job_id, "condition": condition, "seed": seed,
                         "task": scenario, "policy": policy, "interaction_budget": 512,
                         "status": "passed" if passed else "ready", "checkpoint": checkpoint,
                         "run_manifest": str(training_path), "training_metadata": training_metadata,
                         "evaluations": {scenario: {"status": "passed", "result_file": str(result_path)}} if passed else {}})
    manifest = {"config": {"task": scenario, "conditions": ["frozen_lif_original", "gru_trainable"],
                           "seeds": [0, 1], "interaction_budget": 512,
                           "training": {"num_envs": 16, "horizon": 32},
                           "evaluation": {"scenario_tasks": [scenario], "seed": 101,
                                                        "episodes": 1, "protocol": "heldout_v1"}},
                "jobs": jobs, "execution_source_fingerprint": "version"}
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(manifest))
    return path


def test_complete_matrix_aggregates_independent_seeds_and_paired_differences(tmp_path):
    report = summarize(_matrix(tmp_path))
    assert report["status"] == "complete"
    assert report["validated_jobs"] == 4
    scenario = "FlyG1-GoalReach-FreePosture-v0"
    gru = report["by_scenario"][scenario]["gru_trainable"]
    assert gru["metrics"]["success_rate"]["n_training_seeds"] == 2
    assert gru["metrics"]["mean_xy_progress_m"]["mean"] == 1.5
    difference = report["paired_differences_vs_original"][scenario]["gru_trainable"]
    assert difference["paired_training_seeds"] == [0, 1]
    assert difference["metrics"]["mean_xy_progress_m"]["mean"] == 1.0
    assert report["pairing"][scenario]["initial_state_shared"] is True
    resources = report["training_resources"]["by_condition"]["gru_trainable"]
    assert resources["metrics"]["model_total_parameters"]["mean"] == 120
    assert resources["metrics"]["model_trainable_parameters"]["mean"] == 120
    assert resources["metrics"]["actual_environment_interactions"]["mean"] == 512
    assert resources["metrics"]["training_wall_time_s"]["mean"] == 10.5
    assert resources["metrics"]["max_sampled_gpu_used_mib"]["mean"] == 2250
    assert resources["metrics"]["max_sampled_ram_percent"]["mean"] == 40.5
    assert len(report["training_manifest_sha256"]) == 4
    rendered = markdown(report)
    assert "gru_trainable" in rendered
    assert "Training cost and memory" in rendered
    assert "Time to target" in rendered
    assert "Excessive impact" in rendered
    assert "Max sampled GPU used" in rendered


def test_unfinished_jobs_remain_incomplete_without_inventing_results(tmp_path):
    report = summarize(_matrix(tmp_path, second_complete=False))
    assert report["status"] == "incomplete"
    assert report["validated_jobs"] == 2
    assert len(report["missing_or_unfinished_jobs"]) == 2


def test_mismatched_paired_initial_states_invalidate_complete_claim(tmp_path):
    report = summarize(_matrix(tmp_path, mismatch=True))
    assert report["status"] == "incomplete"
    assert report["pairing"]["FlyG1-GoalReach-FreePosture-v0"]["initial_state_shared"] is False
    assert report["validation_errors"]


@pytest.mark.parametrize("row,field,value", [
    ("per_seed", "mean_work_proxy", float("nan")),
    ("per_seed", "mean_work_proxy", "MISSING"),
    ("per_seed", "mean_work_proxy", 999.0),
    ("episodes", "mechanical_work_proxy", float("nan")),
    ("episodes", "joint_limit_frequency", "MISSING"),
    ("episodes", "time_to_target_s", "MISSING"),
])
def test_missing_nonfinite_or_inconsistent_metrics_invalidate_complete_claim(tmp_path, row, field, value):
    manifest = _matrix(tmp_path)
    result_path = tmp_path / "frozen_lif_original__seed-0" / "evaluation.json"
    result = json.loads(result_path.read_text())
    target = result[row][0]
    if value == "MISSING":
        target.pop(field)
    else:
        target[field] = value
    result_path.write_text(json.dumps(result))
    report = summarize(manifest)
    assert report["status"] == "incomplete"
    assert report["validation_errors"]
    assert report["validated_jobs"] == 3


@pytest.mark.parametrize("field,value", [
    ("model_total_parameters", 999),
    ("training_wall_time_s", float("nan")),
    ("actual_environment_interactions", 1024),
    ("memory_samples.1.gpu_devices.0.gpu_used_mib", float("nan")),
    ("memory_samples.1.ram_percent", "MISSING"),
])
def test_invalid_training_metadata_makes_report_incomplete(tmp_path, field, value):
    manifest = _matrix(tmp_path)
    training_path = tmp_path / "frozen_lif_original__seed-0" / "manifest.json"
    training = json.loads(training_path.read_text())
    parts = field.split(".")
    target = training
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    if value == "MISSING":
        target.pop(parts[-1])
    else:
        target[parts[-1]] = value
    training_path.write_text(json.dumps(training))
    report = summarize(manifest)
    assert report["status"] == "incomplete"
    assert report["validated_jobs"] == 3
    assert any("frozen_lif_original__seed-0" in error for error in report["validation_errors"])


@pytest.mark.parametrize("corruption", [
    "missing_history", "truncated_history", "wrong_iteration", "duplicate_iteration",
    "missing_loss", "nan_loss", "infinite_kl", "text_entropy", "boolean_grad_norm",
])
def test_corrupt_training_history_makes_report_incomplete(tmp_path, corruption):
    manifest = _matrix(tmp_path)
    training_path = tmp_path / "frozen_lif_original__seed-0" / "manifest.json"
    training = json.loads(training_path.read_text())
    history = training["history"]
    if corruption == "missing_history":
        training.pop("history")
    elif corruption == "truncated_history":
        history.clear()
    elif corruption == "wrong_iteration":
        history[0]["iteration"] = 1
    elif corruption == "duplicate_iteration":
        history.append(deepcopy(history[0]))
    elif corruption == "missing_loss":
        history[0].pop("loss")
    elif corruption == "nan_loss":
        history[0]["loss"] = float("nan")
    elif corruption == "infinite_kl":
        history[0]["attempted_kl"] = float("inf")
    elif corruption == "text_entropy":
        history[0]["entropy"] = "1.0"
    elif corruption == "boolean_grad_norm":
        history[0]["grad_norm"] = True
    training_path.write_text(json.dumps(training))
    report = summarize(manifest)
    assert report["status"] == "incomplete"
    assert report["validated_jobs"] == 3
    assert any("frozen_lif_original__seed-0: training history" in error
               for error in report["validation_errors"])


def test_full_shape_heldout_matrix_uses_training_seeds_and_rejects_one_corrupt_episode(tmp_path):
    """Sixteen paired episodes per scenario must still yield only five replicates."""
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template = json.loads(_matrix(template_dir).read_text())
    source_job = template["jobs"][0]
    source_training = json.loads(Path(source_job["run_manifest"]).read_text())
    source_result_path = next(iter(source_job["evaluations"].values()))["result_file"]
    source_result = json.loads(Path(source_result_path).read_text())
    conditions = {
        "frozen_lif_original": "frozen_lif",
        "frozen_lif_degree_rewired": "frozen_lif_rewired",
        "gru_trainable": "gru",
        "mlp_engineering_baseline": "mlp",
    }
    scenarios = (
        "FlyG1-GoalReach-FreePosture-v0",
        "FlyG1-GoalSwitch-FreePosture-v0",
        "FlyG1-PushRecovery-FreePosture-v0",
    )
    root = tmp_path / "full_matrix"
    jobs = []
    for condition_index, (condition, policy) in enumerate(conditions.items()):
        for seed in range(5):
            job_id = f"{condition}__seed-{seed}"
            job_dir = root / job_id
            job_dir.mkdir(parents=True)
            checkpoint = job_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"fixture checkpoint")
            training = deepcopy(source_training)
            training.update(policy=policy, seed=seed, checkpoint_path=str(checkpoint))
            training["model_total_parameters"] = 100 + 20 * condition_index
            training["model_trainable_parameters"] = 80 if condition_index < 2 else 100 + 20 * condition_index
            training["model_frozen_parameters"] = (
                training["model_total_parameters"] - training["model_trainable_parameters"]
            )
            training["training_wall_time_s"] = float(10 + seed + condition_index)
            training_path = job_dir / "manifest.json"
            training_path.write_text(json.dumps(training))
            training_metadata = {field: training[field] for field in (
                "requested_interaction_budget", "actual_environment_interactions",
                "completed_iterations", "model_total_parameters", "model_trainable_parameters",
                "model_frozen_parameters", "training_wall_time_s",
            )}
            evaluations = {}
            for scenario in scenarios:
                switch = "GoalSwitch" in scenario
                recovery = "PushRecovery" in scenario
                episodes = []
                for episode_id in range(16):
                    success = episode_id < 2 * seed + condition_index
                    recovered = recovery and episode_id < seed
                    episode = deepcopy(source_result["episodes"][0])
                    episode.update(
                        episode_id=episode_id, seed=seed, success=success,
                        xy_progress_m=None if switch else seed + condition_index / 2 + episode_id / 10,
                        accumulated_goal_relative_progress_m=seed + condition_index / 2 + episode_id / 10,
                        time_to_target_s=1 + episode_id / 10 if success else None,
                        mechanical_work_proxy=10 + seed + episode_id / 10,
                        goal_switch_count=2 if switch else 0,
                        push_count=2 if recovery else 0,
                        push_impulse_vector_n_s=[20.0, 0.0, 0.0] if recovery else [0.0, 0.0, 0.0],
                        push_impulse_magnitude_n_s=20.0 if recovery else 0.0,
                        recovery_attempt_count=1 if recovery else 0,
                        recovery_success_count=int(recovered),
                        recovery_success=recovered if recovery else None,
                        recovery_time_s=1 + episode_id / 10 if recovered else None,
                        paired_plan_sha256=f"{scenario}:plan:{episode_id}",
                        initial_state_sha256=f"{scenario}:initial:{episode_id}",
                    )
                    episodes.append(episode)
                successful = [row for row in episodes if row["success"]]
                recovered = [row for row in episodes if row["recovery_success_count"]]
                attempts = sum(row["recovery_attempt_count"] for row in episodes)
                per_seed = {
                    "seed": seed, "episodes": 16,
                    "success_rate": len(successful) / 16,
                    "mean_xy_progress_m": None if switch else mean(row["xy_progress_m"] for row in episodes),
                    "mean_accumulated_goal_relative_progress_m": mean(
                        row["accumulated_goal_relative_progress_m"] for row in episodes
                    ),
                    "mean_time_to_target_s": mean(row["time_to_target_s"] for row in successful) if successful else None,
                    "mean_work_proxy": mean(row["mechanical_work_proxy"] for row in episodes),
                    "mean_excessive_impact": 0.0,
                    "mean_joint_limit_frequency": 0.0,
                    "mean_saturation_frequency": 0.0,
                    "mean_goal_switch_count": 2.0 if switch else 0.0,
                    "mean_push_count": 2.0 if recovery else 0.0,
                    "mean_push_impulse_magnitude_n_s": 20.0 if recovery else 0.0,
                    "recovery_success_rate": len(recovered) / attempts if attempts else None,
                    "mean_recovery_time_s": mean(row["recovery_time_s"] for row in recovered) if recovered else None,
                    "recovery_attempt_count": attempts,
                    "recovery_success_count": len(recovered),
                }
                result = deepcopy(source_result)
                result.update(
                    task=scenario, checkpoint=str(checkpoint), training_seed=seed,
                    n_episodes=16, per_seed=[per_seed], episodes=episodes,
                )
                result["scenario"]["schedule"]["sha256"] = "a" * 64
                result_path = job_dir / f"{scenario}.json"
                result_path.write_text(json.dumps(result))
                evaluations[scenario] = {"status": "passed", "result_file": str(result_path)}
            jobs.append({
                "id": job_id, "condition": condition, "seed": seed,
                "task": scenarios[0], "policy": policy, "interaction_budget": 512,
                "status": "passed", "checkpoint": str(checkpoint),
                "run_manifest": str(training_path), "training_metadata": training_metadata,
                "evaluations": evaluations,
            })
    manifest = {
        "config": {
            "task": scenarios[0], "conditions": list(conditions), "seeds": list(range(5)),
            "interaction_budget": 512, "training": {"num_envs": 16, "horizon": 32},
            "evaluation": {"scenario_tasks": list(scenarios), "seed": 101,
                           "episodes": 16, "protocol": "heldout_v1"},
        },
        "jobs": jobs, "execution_source_fingerprint": "version",
    }
    manifest_path = root / "matrix.json"
    manifest_path.write_text(json.dumps(manifest))

    report = summarize(manifest_path)
    assert report["status"] == "complete", report["validation_errors"]
    assert (report["expected_jobs"], report["validated_jobs"]) == (20, 20)
    assert len(report["evaluation_result_sha256"]) == 60
    assert len(report["training_manifest_sha256"]) == 20
    reach = report["by_scenario"][scenarios[0]]["frozen_lif_original"]["metrics"]
    assert reach["success_rate"]["n_training_seeds"] == 5
    assert reach["success_rate"]["mean"] == pytest.approx(0.25)
    assert reach["success_rate"]["sample_sd"] == pytest.approx(stdev(seed / 8 for seed in range(5)))
    assert reach["mean_xy_progress_m"]["mean"] == pytest.approx(2.75)
    assert reach["mean_xy_progress_m"]["sample_sd"] == pytest.approx(stdev(range(5)))
    assert report["paired_differences_vs_original"][scenarios[0]]["gru_trainable"]["paired_training_seeds"] == list(range(5))
    assert report["by_scenario"][scenarios[1]]["frozen_lif_original"]["metrics"]["mean_xy_progress_m"] is None
    assert report["by_scenario"][scenarios[2]]["frozen_lif_original"]["metrics"]["recovery_success_rate"]["n_training_seeds"] == 5
    assert all(all(value for key, value in entry.items() if key.endswith("_shared"))
               for entry in report["pairing"].values())

    result_path = root / "gru_trainable__seed-3" / f"{scenarios[2]}.json"
    original = result_path.read_text()
    result = json.loads(original)
    result["episodes"][7]["mechanical_work_proxy"] += 1.0
    result_path.write_text(json.dumps(result))
    corrupt_episode = summarize(manifest_path)
    assert corrupt_episode["status"] == "incomplete"
    assert corrupt_episode["validated_jobs"] == 19
    assert any("gru_trainable__seed-3" in error and "mean_work_proxy" in error
               for error in corrupt_episode["validation_errors"])

    result = json.loads(original)
    result["per_seed"][0]["mean_work_proxy"] += 0.01
    result_path.write_text(json.dumps(result))
    corrupt_mean = summarize(manifest_path)
    assert corrupt_mean["status"] == "incomplete"
    assert corrupt_mean["validated_jobs"] == 19
    assert any("gru_trainable__seed-3" in error and "mean_work_proxy" in error
               for error in corrupt_mean["validation_errors"])
