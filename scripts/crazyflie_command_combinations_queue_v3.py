#!/usr/bin/env python3
"""Fail-closed sequential queue for the six-cell LIF-combination follow-on.

Revision 3 is additive: it never adopts or mutates the completed revision-2
queue.  The three frozen multi-connectome LIF policies are each trained once
in still air and once with the reviewed physical-wind task.  ``--dry_run``
materializes an immutable six-cell plan; ``--execute`` is the only mode that
can launch Isaac children.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source" / "g1_fly_control"
for import_path in (SCRIPT_DIR, SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import crazyflie_command_queue as v1_queue  # noqa: E402
import crazyflie_command_queue_v2 as v2_queue  # noqa: E402
import drone_train  # noqa: E402
from drone_bootstrap import (  # noqa: E402
    canonical_sha256,
    load_fingerprint_rewire_manifest,
    reproduction_fingerprint,
    sha256_file,
)
from g1_fly_control.crazyflie.controllers import build_controller  # noqa: E402
from g1_fly_control.tasks.crazyflie.command_wide_logic import (  # noqa: E402
    COMMAND_WIDE_STILL_CONTRACT_SHA256,
    COMMAND_WIDE_WIND_CONTRACT_SHA256,
    command_wide_contract_payload,
    command_wide_training_contract_payload,
    command_wide_training_contract_sha256,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "experiments"
    / "crazyflie_command_combinations_seed0_1m.json"
)
ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
TASKS = (
    "FlyCrazyflie-CommandFollowWide-v0",
    "FlyCrazyflie-CommandFollowWideWind-v0",
)
PROTOCOL_BY_TASK = {task: "command_v2" for task in TASKS}
CONTROLLERS = (
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
)
POLICY_BY_CONTROLLER = {controller: controller for controller in CONTROLLERS}
COMPONENTS_BY_CONTROLLER = {
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
}
FUSION_BY_CONTROLLER = {
    "leg_optic_lif": "independent_leg_and_optic_cores_concat_motor_readouts_v1",
    "wing_optic_lif": "independent_wing_and_optic_cores_concat_motor_readouts_v1",
    "leg_wing_optic_lif": (
        "independent_leg_and_wing_and_optic_cores_concat_motor_readouts_v1"
    ),
}
ACTOR_PARAMETERS_BY_CONTROLLER = {
    "leg_optic_lif": 9_224,
    "wing_optic_lif": 9_224,
    "leg_wing_optic_lif": 13_672,
}
CRITIC_PARAMETERS_BY_CONTROLLER = {
    controller: 18_305 for controller in CONTROLLERS
}
DYNAMIC_STATE_BY_CONTROLLER = {
    "leg_optic_lif": 2_048,
    "wing_optic_lif": 2_048,
    "leg_wing_optic_lif": 3_072,
}
CONTRACT_SHA_BY_TASK = {
    TASKS[0]: COMMAND_WIDE_STILL_CONTRACT_SHA256,
    TASKS[1]: COMMAND_WIDE_WIND_CONTRACT_SHA256,
}
TOTAL_INTERACTIONS = 1_000_000
EPISODES_PER_JOB = 16
STEPS_PER_EPISODE = 600
JOB_COUNT = 6
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
QUEUE_KIND = "crazyflie_command_combinations_queue_v3"
QUEUE_SCHEMA_VERSION = 3
QUEUE_FILE_NAME = "queue.json"
TRAINER = (SCRIPT_DIR / "drone_train.py").resolve()
EVALUATOR = (SCRIPT_DIR / "crazyflie_command_evaluate.py").resolve()
V1_RUNNER = (SCRIPT_DIR / "crazyflie_command_queue.py").resolve()
V2_RUNNER = (SCRIPT_DIR / "crazyflie_command_queue_v2.py").resolve()
PRIOR_QUEUE = (
    ROOT / "runs" / "crazyflie_command_seed0_500k" / "queue.json"
).resolve()
BASE_CONFIG = (
    ROOT / "configs" / "experiments"
    / "crazyflie_command_optic_wind_seed0_1m.json"
).resolve()
BASE_COMPLETED_QUEUE = (
    ROOT / "runs" / "crazyflie_command_optic_wind_seed0_1m" / "queue.json"
).resolve()
BASE_COMPLETED_REPORT = (
    ROOT / "docs" / "crazyflie_command_optic_wind_report.md"
).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return v1_queue._read_json(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    v1_queue._atomic_json(path, value)


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _expected_config_keys() -> set[str]:
    return {
        "schema_version", "kind", "label", "output_root", "isaac_python",
        "tasks", "task_protocols", "contract_profile", "controllers",
        "trainer_policies", "seed", "total_interactions_per_job", "training",
        "command_envelope", "evaluation", "connectomes", "rewire", "queue",
        "comparison_contract", "base_matrix_identity",
        "prior_parallel_paging_evidence",
    }


def _parser_choices(parser: argparse.ArgumentParser, option: str) -> set[str]:
    for action in parser._actions:
        if option in action.option_strings:
            return set(action.choices or ())
    return set()


def _validate_prior_paging_evidence(declaration: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "queue": "runs/crazyflie_command_seed0_500k/queue.json",
        "queue_sha256": "648ad8b8476a1b4873e23eedacefeb803e3d7887d73d171e3e1512245ef5b99c",
        "required_event": "hard_resource_gate",
        "required_sustained_paging": True,
        "disposition": "v3_max_parallel_fixed_to_one",
    }
    if declaration != expected:
        raise ValueError("prior_parallel_paging_evidence differs from reviewed evidence")
    if not PRIOR_QUEUE.is_file() or sha256_file(PRIOR_QUEUE) != expected["queue_sha256"]:
        raise ValueError("completed v1 queue paging evidence is missing or changed")
    prior = _read_json(PRIOR_QUEUE)
    matches = [
        event for event in prior.get("events", [])
        if isinstance(event, dict)
        and event.get("event") == "hard_resource_gate"
        and isinstance(event.get("sample"), dict)
        and event["sample"].get("sustained_paging") is True
    ]
    if len(matches) != 1:
        raise ValueError("v1 queue does not contain the unique required paging event")
    return matches[0]


def validate_runtime_interfaces() -> dict[str, Any]:
    """Prove the three policies reach both trainer and evaluator before launch."""

    missing: list[str] = []
    required_policies = set(POLICY_BY_CONTROLLER.values())
    trainer_policies = set(getattr(drone_train, "POLICIES", ()))
    trainer_tasks = set(getattr(drone_train, "TRAINING_TASKS", ()))
    trainer_protocols = set(getattr(drone_train, "EVALUATION_PROTOCOLS", ()))
    if not required_policies.issubset(trainer_policies):
        missing.append(f"trainer policies {sorted(required_policies - trainer_policies)}")
    if not set(TASKS).issubset(trainer_tasks):
        missing.append(f"trainer tasks {sorted(set(TASKS) - trainer_tasks)}")
    if not set(PROTOCOL_BY_TASK.values()).issubset(trainer_protocols):
        missing.append(
            f"trainer protocols {sorted(set(PROTOCOL_BY_TASK.values()) - trainer_protocols)}"
        )
    signature = inspect.signature(build_controller)
    for argument in (
        "connectome_manifest", "wing_connectome_manifest",
        "optic_connectome_manifest",
    ):
        if argument not in signature.parameters:
            missing.append(f"build_controller {argument}")
    try:
        import crazyflie_command_evaluate as evaluator

        parser = evaluator._parser()
        eval_tasks = _parser_choices(parser, "--task")
        eval_protocols = _parser_choices(parser, "--protocol")
        eval_policies = _parser_choices(parser, "--policy")
    except Exception as exc:
        missing.append(f"evaluator parser ({type(exc).__name__}: {exc})")
        eval_tasks, eval_protocols, eval_policies = set(), set(), set()
    if not set(TASKS).issubset(eval_tasks):
        missing.append(f"evaluator tasks {sorted(set(TASKS) - eval_tasks)}")
    if not set(PROTOCOL_BY_TASK.values()).issubset(eval_protocols):
        missing.append(
            f"evaluator protocols {sorted(set(PROTOCOL_BY_TASK.values()) - eval_protocols)}"
        )
    if not required_policies.issubset(eval_policies):
        missing.append(f"evaluator policies {sorted(required_policies - eval_policies)}")
    if missing:
        raise ValueError("v3 runtime interfaces are incomplete: " + "; ".join(missing))
    return {
        "trainer_policies": sorted(required_policies),
        "trainer_tasks": list(TASKS),
        "evaluation_protocols": dict(PROTOCOL_BY_TASK),
        "evaluator_explicit_task_argument": True,
        "multi_connectome_arguments": [
            "--connectome_manifest", "--wing_connectome_manifest",
            "--optic_connectome_manifest",
        ],
    }


def validate_config(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = _read_json(path)
    if set(config) != _expected_config_keys():
        raise ValueError("v3 config top-level fields differ from the closed schema")
    exact = {
        "schema_version": 3,
        "kind": "crazyflie_command_combinations_matrix_v3",
        "label": "crazyflie_command_combinations_seed0_1m",
        "output_root": "runs/crazyflie_command_combinations_seed0_1m",
        "isaac_python": str(ISAAC_PYTHON),
        "tasks": list(TASKS),
        "task_protocols": PROTOCOL_BY_TASK,
        "contract_profile": "command_v2",
        "controllers": list(CONTROLLERS),
        "trainer_policies": POLICY_BY_CONTROLLER,
        "seed": 0,
        "total_interactions_per_job": TOTAL_INTERACTIONS,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be exactly {expected!r}")
    expected_training = {
        "num_envs": 40, "horizon": 100, "microbatch_size": 40,
        "ppo_epochs": 2, "learning_rate": 0.0003, "gamma": 0.99,
        "gae_lambda": 0.95, "clip_ratio": 0.2, "value_coefficient": 0.5,
        "entropy_coefficient": 0.002, "max_grad_norm": 1.0,
        "target_kl": 0.05, "checkpoint_every_updates": 25,
        "precision": "float32", "device": "cuda:0",
    }
    if config.get("training") != expected_training:
        raise ValueError("training differs from the reviewed v2 PPO contract")
    if TOTAL_INTERACTIONS % (40 * 100):
        raise ValueError("v3 interaction budget does not divide into whole PPO updates")
    expected_envelope = {
        "maximum_horizontal_speed_mps": 1.0,
        "maximum_vertical_speed_mps": 0.5,
        "maximum_yaw_rate_radps": 1.5,
        "minimum_hold_steps": 25,
        "maximum_hold_steps": 100,
        "control_dt_s": 0.02,
        "simultaneous_axes": True,
    }
    if config.get("command_envelope") != expected_envelope:
        raise ValueError("command_envelope differs from the reviewed expanded envelope")
    live = command_wide_contract_payload()
    if (
        live.get("maximum_horizontal_speed_m_s") != 1.0
        or live.get("maximum_vertical_speed_m_s") != 0.5
        or live.get("maximum_yaw_rate_rad_s") != 1.5
    ):
        raise ValueError("live command contract differs from the v3 envelope")
    for task in TASKS:
        wind_enabled = task == TASKS[1]
        payload = command_wide_training_contract_payload(wind_enabled=wind_enabled)
        if (
            command_wide_training_contract_sha256(wind_enabled=wind_enabled)
            != CONTRACT_SHA_BY_TASK[task]
            or canonical_sha256(payload) != CONTRACT_SHA_BY_TASK[task]
        ):
            raise ValueError(f"live wide-command contract hash is inconsistent for {task}")
    expected_evaluation = {
        "episodes_per_job": 16,
        "steps_per_episode": 600,
        "deterministic_actions": True,
        "activity_from_actual_controller": True,
        "device": "cuda:0",
        "script": "scripts/crazyflie_command_evaluate.py",
        "explicit_task_argument": True,
        "analysis_kind": "crazyflie_command_follow_heldout_v2",
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("evaluation differs from the reviewed held-out contract")
    expected_queue = {
        "default_max_parallel": 1,
        "maximum_parallel": 1,
        "lif_first": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
        "sustained_paging_sample_count": 3,
        "network_independent": True,
        "immutable_attempt_logs": True,
    }
    if config.get("queue") != expected_queue:
        raise ValueError("queue differs from the strict sequential v3 policy")
    expected_comparison = {
        "same_task_stream_budget_seed_and_ppo_hyperparameters": True,
        "paired_command_seed_across_still_and_wind": True,
        "lif_cells_before_any_baseline": True,
        "controllers_are_all_frozen_multi_connectome_lif": True,
        "combinations": {key: list(value) for key, value in COMPONENTS_BY_CONTROLLER.items()},
        "fusion_by_controller": dict(FUSION_BY_CONTROLLER),
        "capacity_contract": {
            "actor_trainable_parameters": dict(ACTOR_PARAMETERS_BY_CONTROLLER),
            "critic_trainable_parameters": dict(CRITIC_PARAMETERS_BY_CONTROLLER),
            "total_dynamic_state_per_environment": dict(
                DYNAMIC_STATE_BY_CONTROLLER
            ),
        },
        "parameter_count_reporting_required": True,
        "cross_capacity_claims_are_descriptive_only": True,
        "matched_gru_or_mlp_jobs_in_this_follow_on": False,
    }
    if config.get("comparison_contract") != expected_comparison:
        raise ValueError("comparison_contract differs from reviewed v3 declaration")
    expected_base = {
        "config": "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json",
        "config_sha256": "bf7369d74d119784eb6a932cbeb57734fa93b7fdda5457e3e46decfb6f140c44",
        "completed_queue": "runs/crazyflie_command_optic_wind_seed0_1m/queue.json",
        "completed_queue_sha256": "68f4fee83d659edb439691aa79c5f4b6decc610c1e35e869b1caddd04cf5bddc",
        "completed_report": "docs/crazyflie_command_optic_wind_report.md",
        "completed_report_sha256": "7988eabbb82c710dda724f36bfb19852439cafd984562b997ef238d41b7a6fed",
        "required_queue_status": "completed",
        "required_job_count": 14,
        "required_completed_jobs": 14,
        "relation": "additive_follow_on_no_v2_mutation",
    }
    if config.get("base_matrix_identity") != expected_base:
        raise ValueError("base_matrix_identity differs from the frozen v2 declaration")
    if not BASE_CONFIG.is_file() or sha256_file(BASE_CONFIG) != expected_base["config_sha256"]:
        raise ValueError("frozen v2 base config is missing or changed")
    if (
        not BASE_COMPLETED_QUEUE.is_file()
        or sha256_file(BASE_COMPLETED_QUEUE)
        != expected_base["completed_queue_sha256"]
    ):
        raise ValueError("immutable completed v2 queue is missing or changed")
    if (
        not BASE_COMPLETED_REPORT.is_file()
        or sha256_file(BASE_COMPLETED_REPORT)
        != expected_base["completed_report_sha256"]
    ):
        raise ValueError("immutable completed v2 report is missing or changed")
    completed_v2 = _read_json(BASE_COMPLETED_QUEUE)
    completed_jobs = completed_v2.get("jobs")
    if (
        completed_v2.get("schema_version") != 2
        or completed_v2.get("kind") != "crazyflie_command_big_matrix_queue_v2"
        or completed_v2.get("status") != "completed"
        or completed_v2.get("dry_run") is not False
        or completed_v2.get("job_count") != 14
        or completed_v2.get("counts") != {"completed": 14}
        or not isinstance(completed_jobs, list)
        or len(completed_jobs) != 14
        or len({job.get("id") for job in completed_jobs if isinstance(job, dict)})
        != 14
        or any(
            not isinstance(job, dict)
            or job.get("status") != "completed"
            or job.get("training_status") != "completed"
            or job.get("evaluation_status") != "completed"
            or "active_process" in job
            for job in completed_jobs
        )
        or not isinstance(completed_v2.get("finished_utc"), str)
    ):
        raise ValueError("immutable v2 queue is not the terminal 14-job result")
    prior_event = _validate_prior_paging_evidence(
        config.get("prior_parallel_paging_evidence", {})
    )
    connectomes = config.get("connectomes")
    if not isinstance(connectomes, dict) or set(connectomes) != {
        "leg_manifest", "leg_manifest_sha256", "wing_manifest",
        "wing_manifest_sha256", "optic_manifest", "optic_manifest_sha256",
    }:
        raise ValueError("connectomes differs from the v3 schema")
    for label in ("leg", "wing", "optic"):
        relative = connectomes[f"{label}_manifest"]
        expected_hash = connectomes[f"{label}_manifest_sha256"]
        artifact = (ROOT / relative).resolve()
        if (
            not isinstance(relative, str) or not isinstance(expected_hash, str)
            or len(expected_hash) != 64 or not artifact.is_file()
            or sha256_file(artifact) != expected_hash
        ):
            raise ValueError(f"Pinned {label} connectome is missing or changed")
    rewire = config.get("rewire")
    expected_rewire = {
        "seed": 20260916,
        "manifest": "configs/experiments/crazyflie_rewire_seed_20260916.json",
        "manifest_sha256": "6c2a10b879d22741b0d17110d052953adc4f081e9ecac636fabdf0ba68d57b14",
    }
    if rewire != expected_rewire:
        raise ValueError("rewire identity differs from the frozen declaration")
    rewire_path = (ROOT / rewire["manifest"]).resolve()
    if not rewire_path.is_file() or sha256_file(rewire_path) != rewire["manifest_sha256"]:
        raise ValueError("Pinned rewire manifest is missing or changed")
    if not ISAAC_PYTHON.is_file() or not os.access(ISAAC_PYTHON, os.X_OK):
        raise ValueError(f"Isaac Python is not executable: {ISAAC_PYTHON}")
    for script in (TRAINER, EVALUATOR, V1_RUNNER, V2_RUNNER):
        if not script.is_file():
            raise ValueError(f"Required local script is missing: {script}")
    interface = validate_runtime_interfaces()

    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    config["_output_root"] = str((ROOT / config["output_root"]).resolve())
    config["_leg_manifest"] = str((ROOT / connectomes["leg_manifest"]).resolve())
    config["_wing_manifest"] = str((ROOT / connectomes["wing_manifest"]).resolve())
    config["_optic_manifest"] = str((ROOT / connectomes["optic_manifest"]).resolve())
    config["_rewire_manifest"] = str(rewire_path)
    config["_interface"] = interface
    config["_prior_paging_event"] = prior_event
    return config


def _training_args(config: Mapping[str, Any], task: str, policy: str) -> SimpleNamespace:
    training = config["training"]
    return SimpleNamespace(
        task=task, contract_profile="command_v2", policy=policy, seed=0,
        num_envs=40, total_interactions=TOTAL_INTERACTIONS, horizon=100,
        microbatch_size=40, ppo_epochs=training["ppo_epochs"],
        learning_rate=training["learning_rate"], gamma=training["gamma"],
        gae_lambda=training["gae_lambda"], clip_ratio=training["clip_ratio"],
        value_coefficient=training["value_coefficient"],
        entropy_coefficient=training["entropy_coefficient"],
        max_grad_norm=training["max_grad_norm"], target_kl=training["target_kl"],
        checkpoint_every_updates=training["checkpoint_every_updates"],
        connectome_manifest=Path(config["_leg_manifest"]),
        wing_connectome_manifest=Path(config["_wing_manifest"]),
        optic_connectome_manifest=Path(config["_optic_manifest"]),
        rewire_seed=config["rewire"]["seed"],
        rewire_manifest=Path(config["_rewire_manifest"]),
        evaluation_protocol=PROTOCOL_BY_TASK[task], warm_start_checkpoint=None,
    )


def _primary_manifest(args: SimpleNamespace) -> Path:
    return (
        args.wing_connectome_manifest
        if args.policy == "wing_optic_lif"
        else args.connectome_manifest
    )


def _job_fingerprint(
    config: Mapping[str, Any], task: str, policy: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    args = _training_args(config, task, policy)
    resolved, evaluation = drone_train.standalone_resolved_config(args)
    rewired = load_fingerprint_rewire_manifest(
        args.rewire_manifest,
        expected_file_sha256=config["rewire"]["manifest_sha256"],
        expected_seed=args.rewire_seed,
    )
    fingerprint, payload = reproduction_fingerprint(
        resolved_config=resolved,
        evaluation_manifest=evaluation,
        connectome_manifest=_primary_manifest(args),
        rewired_manifest=rewired,
    )
    return fingerprint, payload, evaluation


def _training_command(
    config: Mapping[str, Any], task: str, policy: str, run_dir: Path,
    fingerprint: str, pause_file: Path,
) -> list[str]:
    training = config["training"]
    return [
        config["isaac_python"], str(TRAINER),
        "--task", task, "--contract_profile", "command_v2",
        "--policy", policy, "--seed", "0", "--num_envs", "40",
        "--total_interactions", str(TOTAL_INTERACTIONS),
        "--horizon", "100", "--microbatch_size", "40",
        "--ppo_epochs", str(training["ppo_epochs"]),
        "--learning_rate", str(training["learning_rate"]),
        "--gamma", str(training["gamma"]),
        "--gae_lambda", str(training["gae_lambda"]),
        "--clip_ratio", str(training["clip_ratio"]),
        "--value_coefficient", str(training["value_coefficient"]),
        "--entropy_coefficient", str(training["entropy_coefficient"]),
        "--max_grad_norm", str(training["max_grad_norm"]),
        "--target_kl", str(training["target_kl"]),
        "--checkpoint_every_updates", str(training["checkpoint_every_updates"]),
        "--connectome_manifest", config["_leg_manifest"],
        "--wing_connectome_manifest", config["_wing_manifest"],
        "--optic_connectome_manifest", config["_optic_manifest"],
        "--rewire_seed", str(config["rewire"]["seed"]),
        "--rewire_manifest", config["_rewire_manifest"],
        "--evaluation_protocol", PROTOCOL_BY_TASK[task],
        "--run_dir", str(run_dir), "--expected_fingerprint", fingerprint,
        "--pause_file", str(pause_file), "--device", training["device"],
        "--headless",
    ]


def _evaluation_command(
    config: Mapping[str, Any], task: str, policy: str, checkpoint: Path,
    output: Path, fingerprint: str,
) -> list[str]:
    return [
        config["isaac_python"], str(EVALUATOR),
        "--task", task, "--checkpoint", str(checkpoint),
        "--output", str(output), "--protocol", PROTOCOL_BY_TASK[task],
        "--device", config["evaluation"]["device"],
        "--expected_fingerprint", fingerprint, "--training_seed", "0",
        "--policy", policy, "--headless",
    ]


def _controller_reports(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for label in CONTROLLERS:
        policy, report = build_controller(
            POLICY_BY_CONTROLLER[label], observation_dim=12, action_dim=4,
            device="cpu", connectome_manifest=config["_leg_manifest"],
            wing_connectome_manifest=config["_wing_manifest"],
            optic_connectome_manifest=config["_optic_manifest"],
            rewire_seed=config["rewire"]["seed"],
            enforce_parameter_match=False,
        )
        del policy
        report = dict(report)
        if report.get("fusion_contract") != FUSION_BY_CONTROLLER[label]:
            raise RuntimeError(f"{label} fusion contract differs from v3 config")
        if (
            report.get("actor_trainable_parameters")
            != ACTOR_PARAMETERS_BY_CONTROLLER[label]
            or report.get("critic_trainable_parameters")
            != CRITIC_PARAMETERS_BY_CONTROLLER[label]
            or report.get("total_dynamic_state_per_environment")
            != DYNAMIC_STATE_BY_CONTROLLER[label]
        ):
            raise RuntimeError(f"{label} capacity differs from the closed v3 contract")
        reports[label] = report
    return reports


def build_queue(config: Mapping[str, Any], output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root or config["_output_root"]).resolve()
    controller_reports = _controller_reports(config)
    jobs: list[dict[str, Any]] = []
    for controller_priority, label in enumerate(CONTROLLERS):
        policy = POLICY_BY_CONTROLLER[label]
        for task_priority, task in enumerate(TASKS):
            task_slug = "still" if task == TASKS[0] else "wind"
            identifier = f"{len(jobs) + 1:02d}__{label}__{task_slug}__seed-0"
            run_dir = root / "jobs" / identifier
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            pause_file = run_dir / "pause.request"
            evaluation_output = root / "evaluations" / identifier / "heldout.json"
            fingerprint, payload, evaluation_manifest = _job_fingerprint(
                config, task, policy
            )
            controller_report = controller_reports[label]
            jobs.append({
                "id": identifier,
                "controller": label,
                "policy": policy,
                "controller_priority": controller_priority,
                "task_priority": task_priority,
                "architecture_class": "lif",
                "combination_components": list(COMPONENTS_BY_CONTROLLER[label]),
                "capacity_tier_core_count": len(COMPONENTS_BY_CONTROLLER[label]),
                "task": task,
                "evaluation_protocol": PROTOCOL_BY_TASK[task],
                "seed": 0,
                "command_schedule_seed": 0,
                "paired_task_seed_key": f"{label}__seed-0",
                "contract_profile": "command_v2",
                "status": "pending",
                "training_status": "pending",
                "evaluation_status": "pending",
                "total_interactions": TOTAL_INTERACTIONS,
                "expected_updates": 250,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "training_manifest": str(run_dir / "training_manifest.json"),
                "pause_file": str(pause_file),
                "evaluation_output": str(evaluation_output),
                "expected_fingerprint": fingerprint,
                "fingerprint_payload": payload,
                "evaluation_manifest": evaluation_manifest,
                "evaluation_manifest_id": evaluation_manifest["manifest_id"],
                "command_training_contract_sha256": CONTRACT_SHA_BY_TASK[task],
                "controller_report_sha256": canonical_sha256(controller_report),
                "expected_core_checksum": controller_report["core_checksum"],
                "expected_per_core_checksums": controller_report["per_core_checksums"],
                "expected_connectome_manifests": controller_report[
                    "connectome_manifests"
                ],
                "expected_connectome_checksums": controller_report[
                    "connectome_checksums"
                ],
                "training_command": _training_command(
                    config, task, policy, run_dir, fingerprint, pause_file
                ),
                "evaluation_command": _evaluation_command(
                    config, task, policy, checkpoint, evaluation_output, fingerprint
                ),
                "attempts": [],
            })
    expected_pairs = [
        (controller, task) for controller in CONTROLLERS for task in TASKS
    ]
    if (
        len(jobs) != JOB_COUNT
        or [(job["controller"], job["task"]) for job in jobs] != expected_pairs
        or any(job["architecture_class"] != "lif" for job in jobs)
    ):
        raise RuntimeError("v3 queue did not resolve the exact six-cell LIF order")
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": QUEUE_KIND,
        "label": config["label"],
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "revision": 0,
        "status": "dry_run",
        "dry_run": True,
        "config_path": config["_config_path"],
        "config_file_sha256": config["_config_sha256"],
        "config_identity_sha256": canonical_sha256(_public_config(config)),
        "queue_runner": str(Path(__file__).resolve()),
        "queue_runner_sha256": sha256_file(Path(__file__).resolve()),
        "v1_helper": str(V1_RUNNER),
        "v1_helper_sha256": sha256_file(V1_RUNNER),
        "v2_helper": str(V2_RUNNER),
        "v2_helper_sha256": sha256_file(V2_RUNNER),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "base_matrix_config": str(BASE_CONFIG),
        "base_matrix_config_sha256": sha256_file(BASE_CONFIG),
        "base_completed_queue": str(BASE_COMPLETED_QUEUE),
        "base_completed_queue_sha256": sha256_file(BASE_COMPLETED_QUEUE),
        "base_completed_report": str(BASE_COMPLETED_REPORT),
        "base_completed_report_sha256": sha256_file(BASE_COMPLETED_REPORT),
        "output_root": str(root),
        "queue_file": str(root / QUEUE_FILE_NAME),
        "global_pause_file": str(root / "pause.request"),
        "job_count": JOB_COUNT,
        "predicted_training_interactions": JOB_COUNT * TOTAL_INTERACTIONS,
        "predicted_evaluation_episodes": JOB_COUNT * EPISODES_PER_JOB,
        "controller_order": list(CONTROLLERS),
        "task_order": list(TASKS),
        "task_protocols": dict(PROTOCOL_BY_TASK),
        "lif_cell_count": JOB_COUNT,
        "lif_first": True,
        "maximum_parallel": 1,
        "parallelism_disposition": config["prior_parallel_paging_evidence"],
        "resource_limits": {
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "sustained_paging_sample_count": 3,
            "default_max_parallel": 1,
            "maximum_parallel": 1,
        },
        "controller_reports": controller_reports,
        "counts": {"pending": JOB_COUNT},
        "config": _public_config(config),
        "interface_contract": config["_interface"],
        "jobs": jobs,
        "events": [{"utc": _utc_now(), "event": "verified_v3_dry_run_created"}],
    }


def _refresh_status(queue: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in queue["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    queue["counts"] = dict(sorted(counts.items()))
    if queue.get("dry_run"):
        queue["status"] = "dry_run"
    elif counts.get("running"):
        queue["status"] = "running"
    elif counts.get("paused"):
        queue["status"] = "paused"
    elif counts.get("failed"):
        queue["status"] = "failed"
    elif counts.get("completed") == JOB_COUNT:
        queue["status"] = "completed"
    elif queue.get("resource_block"):
        queue["status"] = "blocked_resource"
    else:
        queue["status"] = "pending"


def save_queue(path: Path, queue: dict[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_utc"] = _utc_now()
    _refresh_status(queue)
    _atomic_json(path, queue)
    _atomic_json(path.with_name("queue_summary.json"), {
        "schema_version": 3,
        "kind": QUEUE_KIND,
        "status": queue["status"],
        "dry_run": queue["dry_run"],
        "revision": queue["revision"],
        "updated_utc": queue["updated_utc"],
        "counts": queue["counts"],
        "job_count": JOB_COUNT,
        "predicted_training_interactions": 6_000_000,
        "predicted_evaluation_episodes": 96,
        "maximum_parallel": 1,
        "resource_block": queue.get("resource_block"),
        "jobs": [
            {key: job[key] for key in (
                "id", "controller", "policy", "task", "status",
                "training_status", "evaluation_status", "checkpoint",
            )}
            for job in queue["jobs"]
        ],
    })


# The v2 execution engine is deliberately generic over job commands.  Keep its
# mature process supervision, but tighten artifact validation for multi-core
# provenance before delegating any state transition to it.
_v2_valid_training = v2_queue.valid_training
_v2_valid_evaluation = v2_queue.valid_evaluation
reconcile_queue = v2_queue.reconcile_queue
_start_child = v2_queue._start_child


def _valid_combination_report(
    job: Mapping[str, Any], report: Any
) -> bool:
    """Validate ordered per-core identities, not only the aggregate checksum."""

    if not isinstance(report, dict):
        return False
    labels = list(COMPONENTS_BY_CONTROLLER.get(str(job.get("controller")), ()))
    if not labels:
        return False
    per_core = report.get("per_core_checksums")
    manifests = report.get("connectome_manifests")
    connectomes = report.get("connectome_checksums")
    populations = report.get("per_core_population_indices")
    fingerprints = report.get("connectome_manifest_fingerprints")
    ordered_maps = (per_core, manifests, connectomes, populations, fingerprints)
    if any(
        not isinstance(value, dict) or set(value) != set(labels)
        for value in ordered_maps
    ):
        return False
    return (
        report.get("controller_kind") == job.get("policy")
        and report.get("core_labels") == labels
        and report.get("fusion_contract")
        == FUSION_BY_CONTROLLER.get(str(job.get("controller")))
        and report.get("core_checksum") == job.get("expected_core_checksum")
        and per_core == job.get("expected_per_core_checksums")
        and manifests == job.get("expected_connectome_manifests")
        and connectomes == job.get("expected_connectome_checksums")
        and all(isinstance(value, str) and len(value) == 64 for value in per_core.values())
        and isinstance(report.get("actor_trainable_parameters"), int)
        and report["actor_trainable_parameters"]
        == ACTOR_PARAMETERS_BY_CONTROLLER[job["controller"]]
        and isinstance(report.get("critic_trainable_parameters"), int)
        and report["critic_trainable_parameters"]
        == CRITIC_PARAMETERS_BY_CONTROLLER[job["controller"]]
        and report.get("total_dynamic_state_per_environment")
        == DYNAMIC_STATE_BY_CONTROLLER[job["controller"]]
        and isinstance(report.get("frozen_parameters"), int)
        and report["frozen_parameters"] > 0
        and report.get("parameter_matching_required") is False
        and canonical_sha256(report) == job.get("controller_report_sha256")
    )


def valid_training(job: Mapping[str, Any]) -> bool:
    """Require exact multi-core provenance plus frozen aggregate before/after."""

    if not _v2_valid_training(job):
        return False
    try:
        manifest = _read_json(Path(job["training_manifest"]))
        report = manifest["controller_report"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    aggregate = job.get("expected_core_checksum")
    expected_per_core = job.get("expected_per_core_checksums")
    return (
        _valid_combination_report(job, report)
        and manifest.get("core_checksum_before") == aggregate
        and manifest.get("core_checksum_after") == aggregate
        and manifest.get("per_core_checksums_before") == expected_per_core
        and manifest.get("per_core_checksums_after") == expected_per_core
    )


def _valid_activity_provenance(job: Mapping[str, Any], activity: Any) -> bool:
    if not isinstance(activity, dict) or activity.get("controller") != job.get("policy"):
        return False
    labels = list(COMPONENTS_BY_CONTROLLER.get(str(job.get("controller")), ()))
    provenance = activity.get("role_provenance")
    per_unit = activity.get("per_unit")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != set(labels)
        or not isinstance(per_unit, list)
        or activity.get("unit_count") != len(per_unit)
    ):
        return False
    try:
        identities = job["fingerprint_payload"]["resolved_config"][
            "lif_connectome_composition"
        ]["connectomes"]
    except (KeyError, TypeError):
        return False
    for label in labels:
        observed = provenance.get(label)
        identity = identities.get(label) if isinstance(identities, dict) else None
        if not isinstance(observed, dict) or not isinstance(identity, dict):
            return False
        neurons = identity.get("neurons_path")
        if (
            observed.get("manifest") != identity.get("manifest")
            or observed.get("manifest_sha256") != identity.get("manifest_sha256")
            or not isinstance(neurons, dict)
            or observed.get("neurons") != neurons.get("path")
            or observed.get("neurons_sha256") != neurons.get("sha256")
            or not isinstance(observed.get("neuron_count"), int)
            or observed["neuron_count"] < 1
        ):
            return False
    prefixes = {
        str(row.get("id", "")).split(":", 1)[0]
        for row in per_unit if isinstance(row, dict)
    }
    return prefixes == set(labels)


def _valid_inference_latency(value: Any) -> bool:
    """Require timings from the exact 600 action-producing evaluator calls."""

    if not isinstance(value, dict):
        return False
    numeric_keys = (
        "total_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
        "all_action_producing_calls_total_ms",
    )
    numbers = {key: value.get(key) for key in numeric_keys}
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
        or float(number) < 0.0
        for number in numbers.values()
    ):
        return False
    return (
        value.get("schema_version") == 1
        and value.get("source")
        == "exact_policy_act_calls_used_by_evaluation_control_path"
        and value.get("call_site")
        == "policy.act(normalized_observation, recurrent_state, deterministic=True)"
        and value.get("clock") == "time.perf_counter_ns_monotonic"
        and value.get("device") == "cuda:0"
        and value.get("measurement_unit")
        == "milliseconds_per_vectorized_policy_call"
        and value.get("batch_size") == 16
        and value.get("cuda_synchronized_before_and_after_call") is True
        and value.get("activity_recorder_hooks_in_scope") is True
        and value.get("action_producing_call_count") == 600
        and value.get("warmup_calls_excluded_from_statistics") == 10
        and value.get("warmup_exclusion_justification")
        == (
            "exclude prefix calls that can include one-time lazy CUDA/kernel "
            "initialization; no extra forwards were executed"
        )
        and value.get("sample_count") == 590
        and math.isclose(
            float(numbers["mean_ms"]),
            float(numbers["total_ms"]) / 590.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and float(numbers["all_action_producing_calls_total_ms"])
        >= float(numbers["total_ms"])
        and float(numbers["p50_ms"])
        <= float(numbers["p95_ms"])
        <= float(numbers["p99_ms"])
        <= float(numbers["max_ms"])
    )


def valid_evaluation(job: Mapping[str, Any]) -> bool:
    """Require evaluation reconstruction/activity from every declared core."""

    if not _v2_valid_evaluation(job):
        return False
    try:
        value = _read_json(Path(job["evaluation_output"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        _valid_combination_report(job, value.get("controller_report"))
        and _valid_activity_provenance(job, value.get("activity"))
        and _valid_inference_latency(value.get("inference_latency"))
        and value.get("integrity", {}).get(
            "inference_latency_from_actual_forward"
        ) is True
    )


def _pause_path(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        _atomic_json(path, payload)


def _request_pause(queue: Mapping[str, Any], queue_path: Path, reason: str) -> None:
    payload = {
        "schema_version": 3,
        "status": "requested",
        "utc": _utc_now(),
        "reason": reason,
        "queue": str(queue_path),
        "pid": os.getpid(),
    }
    _pause_path(Path(queue["global_pause_file"]), payload)
    for job in queue["jobs"]:
        active = job.get("active_process")
        if isinstance(active, dict) and active.get("phase") == "training":
            _pause_path(Path(job["pause_file"]), {**payload, "job": job["id"]})


def execute_queue(queue: dict[str, Any], queue_path: Path, *, resume: bool) -> int:
    """Run v2's reviewed supervisor with v3 persistence/pause identities."""

    if queue.get("maximum_parallel") != 1:
        raise ValueError("v3 execution is permanently sequential after prior paging")
    original_save = v2_queue.save_queue
    original_pause = v2_queue._request_pause
    original_valid_training = v2_queue.valid_training
    original_valid_evaluation = v2_queue.valid_evaluation
    try:
        v2_queue.save_queue = save_queue
        v2_queue._request_pause = _request_pause
        v2_queue.valid_training = valid_training
        v2_queue.valid_evaluation = valid_evaluation
        return v2_queue.execute_queue(queue, queue_path, resume=resume)
    finally:
        v2_queue.save_queue = original_save
        v2_queue._request_pause = original_pause
        v2_queue.valid_training = original_valid_training
        v2_queue.valid_evaluation = original_valid_evaluation


def _validate_loaded_queue(
    queue: dict[str, Any], config: Mapping[str, Any], path: Path
) -> None:
    expected_pairs = [
        (controller, task) for controller in CONTROLLERS for task in TASKS
    ]
    if (
        queue.get("schema_version") != 3
        or queue.get("kind") != QUEUE_KIND
        or queue.get("config_file_sha256") != config["_config_sha256"]
        or queue.get("config_identity_sha256") != canonical_sha256(_public_config(config))
        or queue.get("queue_runner_sha256") != sha256_file(Path(__file__).resolve())
        or queue.get("v1_helper_sha256") != sha256_file(V1_RUNNER)
        or queue.get("v2_helper_sha256") != sha256_file(V2_RUNNER)
        or queue.get("trainer_sha256") != sha256_file(TRAINER)
        or queue.get("evaluator_sha256") != sha256_file(EVALUATOR)
        or Path(queue.get("base_matrix_config", "")).resolve() != BASE_CONFIG
        or queue.get("base_matrix_config_sha256") != sha256_file(BASE_CONFIG)
        or Path(queue.get("base_completed_queue", "")).resolve()
        != BASE_COMPLETED_QUEUE
        or queue.get("base_completed_queue_sha256")
        != sha256_file(BASE_COMPLETED_QUEUE)
        or Path(queue.get("base_completed_report", "")).resolve()
        != BASE_COMPLETED_REPORT
        or queue.get("base_completed_report_sha256")
        != sha256_file(BASE_COMPLETED_REPORT)
        or Path(queue.get("queue_file", "")).resolve() != path.resolve()
        or queue.get("job_count") != JOB_COUNT
        or queue.get("predicted_training_interactions") != 6_000_000
        or queue.get("predicted_evaluation_episodes") != 96
        or queue.get("maximum_parallel") != 1
        or [(job.get("controller"), job.get("task")) for job in queue.get("jobs", [])]
        != expected_pairs
        or any(job.get("architecture_class") != "lif" for job in queue.get("jobs", []))
    ):
        raise ValueError("Existing v3 queue differs from reviewed config/source/order")
    for job in queue["jobs"]:
        fingerprint, payload, evaluation = _job_fingerprint(
            config, job["task"], job["policy"]
        )
        report = queue.get("controller_reports", {}).get(job["controller"])
        if (
            job.get("expected_fingerprint") != fingerprint
            or job.get("fingerprint_payload") != payload
            or job.get("evaluation_manifest") != evaluation
            or job.get("evaluation_manifest_id") != evaluation.get("manifest_id")
            or job.get("command_training_contract_sha256")
            != CONTRACT_SHA_BY_TASK[job["task"]]
            or job.get("combination_components")
            != list(COMPONENTS_BY_CONTROLLER[job["controller"]])
            or not _valid_combination_report(job, report)
        ):
            raise ValueError(f"Existing v3 queue fingerprint is stale: {job.get('id')}")


def _dry_run_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "verified_v3_dry_run_no_processes_launched",
        "queue": queue["queue_file"],
        "job_count": JOB_COUNT,
        "training_interactions_per_job": TOTAL_INTERACTIONS,
        "predicted_training_interactions": 6_000_000,
        "predicted_evaluation_episodes": 96,
        "maximum_parallel": 1,
        "controller_order": list(CONTROLLERS),
        "task_order": list(TASKS),
        "task_protocols": dict(PROTOCOL_BY_TASK),
        "cells": [
            {
                "id": job["id"],
                "controller": job["controller"],
                "policy": job["policy"],
                "task": job["task"],
                "protocol": job["evaluation_protocol"],
                "seed": 0,
                "total_interactions": TOTAL_INTERACTIONS,
                "expected_updates": 250,
                "run_dir": job["run_dir"],
                "checkpoint": job["checkpoint"],
                "evaluation_output": job["evaluation_output"],
                "training_command": job["training_command"],
                "evaluation_command": job["evaluation_command"],
            }
            for job in queue["jobs"]
        ],
    }


def _status_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": queue["status"],
        "queue": queue["queue_file"],
        "revision": queue["revision"],
        "counts": queue["counts"],
        "job_count": JOB_COUNT,
        "training_interactions": 6_000_000,
        "evaluation_episodes": 96,
        "maximum_parallel": 1,
        "last_resource_precheck": queue.get("last_resource_precheck"),
        "last_resource_sample": queue.get("last_resource_sample"),
    }


def _assert_fresh_dry_run_destination(output_root: Path) -> None:
    protected = (
        output_root / QUEUE_FILE_NAME,
        output_root / "queue_summary.json",
        output_root / "jobs",
        output_root / "evaluations",
        output_root / "logs",
        output_root / "pause.request",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise ValueError(
            "v3 dry-run destination contains queue outputs and will not be overwritten: "
            + ", ".join(existing)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry_run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--pause", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    try:
        config = validate_config(args.config)
        output_root = Path(config["_output_root"])
        queue_path = output_root / QUEUE_FILE_NAME
        if args.dry_run:
            _assert_fresh_dry_run_destination(output_root)
            queue = build_queue(config, output_root)
            save_queue(queue_path, queue)
            print(json.dumps(_dry_run_payload(queue), indent=2, sort_keys=True))
            return 0
        if not queue_path.is_file():
            raise ValueError(f"Verified v3 dry-run queue does not exist: {queue_path}")
        queue = _read_json(queue_path)
        _validate_loaded_queue(queue, config, queue_path)
        if args.status:
            print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
            return 0
        if args.pause:
            _request_pause(queue, queue_path, "explicit_cli_request")
            print(json.dumps({"status": "pause_requested", "queue": str(queue_path)}, indent=2))
            return 0
        with v1_queue.queue_lock(queue_path):
            result = execute_queue(queue, queue_path, resume=args.resume)
        print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
        return result
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
