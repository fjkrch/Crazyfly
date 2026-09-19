#!/usr/bin/env python3
"""Failure-isolated sequential queue for the reset-invariant all-fair matrix.

Revision 4 is a new 10-controller by two-task by three-seed experiment.  It
does not adopt historical v2/v3 results as fair evidence and never modifies
their files.  ``--dry_run`` creates the authenticated 60-cell queue; only
``--execute`` launches children.  A failed cell is preserved and skipped so
later cells can run, while ``--execute --resume`` explicitly retries failed or
paused cells from their existing clean checkpoint boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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
import crazyflie_command_combinations_queue_v3 as v3_queue  # noqa: E402
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
    / "crazyflie_command_all_fair_seeds0_1_2_1m.json"
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
    "original_lif",
    "rewired_lif",
    "wing_lif",
    "leg_wing_lif",
    "optic_lif",
    "leg_optic_lif",
    "wing_optic_lif",
    "leg_wing_optic_lif",
    "gru_matched",
    "mlp_normal",
)
POLICY_BY_CONTROLLER = {
    "original_lif": "frozen_lif_original",
    "rewired_lif": "frozen_lif_degree_rewired",
    "wing_lif": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "leg_optic_lif": "leg_optic_lif",
    "wing_optic_lif": "wing_optic_lif",
    "leg_wing_optic_lif": "leg_wing_optic_lif",
    "gru_matched": "gru_matched",
    "mlp_normal": "mlp_normal",
}
SEEDS = (0, 1, 2)
LIF_CONTROLLERS = CONTROLLERS[:8]
BASELINE_CONTROLLERS = CONTROLLERS[8:]
COMPONENTS_BY_CONTROLLER = {
    "original_lif": ("leg",),
    "rewired_lif": ("leg",),
    "wing_lif": ("wing",),
    "leg_wing_lif": ("leg", "wing"),
    "optic_lif": ("optic",),
    "leg_optic_lif": ("leg", "optic"),
    "wing_optic_lif": ("wing", "optic"),
    "leg_wing_optic_lif": ("leg", "wing", "optic"),
    "gru_matched": (),
    "mlp_normal": (),
}
FUSION_BY_CONTROLLER = {
    "original_lif": None,
    "rewired_lif": None,
    "wing_lif": None,
    "leg_wing_lif": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
    "optic_lif": None,
    "leg_optic_lif": "independent_leg_and_optic_cores_concat_motor_readouts_v1",
    "wing_optic_lif": "independent_wing_and_optic_cores_concat_motor_readouts_v1",
    "leg_wing_optic_lif": (
        "independent_leg_and_wing_and_optic_cores_concat_motor_readouts_v1"
    ),
    "gru_matched": None,
    "mlp_normal": None,
}
REPORT_KIND_BY_CONTROLLER = {
    "original_lif": "frozen_lif",
    "rewired_lif": "frozen_lif_rewired",
    "wing_lif": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "leg_optic_lif": "leg_optic_lif",
    "wing_optic_lif": "wing_optic_lif",
    "leg_wing_optic_lif": "leg_wing_optic_lif",
    "gru_matched": "gru",
    "mlp_normal": "mlp",
}
ACTOR_PARAMETERS_BY_CONTROLLER = {
    "original_lif": 4_776,
    "rewired_lif": 4_776,
    "wing_lif": 4_776,
    "leg_wing_lif": 9_224,
    "optic_lif": 4_776,
    "leg_optic_lif": 9_224,
    "wing_optic_lif": 9_224,
    "leg_wing_optic_lif": 13_672,
    "gru_matched": 4_793,
    "mlp_normal": 4_827,
}
CRITIC_PARAMETERS_BY_CONTROLLER = {controller: 18_305 for controller in CONTROLLERS}
DYNAMIC_STATE_BY_CONTROLLER = {
    "original_lif": 1_024,
    "rewired_lif": 1_024,
    "wing_lif": 1_024,
    "leg_wing_lif": 2_048,
    "optic_lif": 1_024,
    "leg_optic_lif": 2_048,
    "wing_optic_lif": 2_048,
    "leg_wing_optic_lif": 3_072,
    "gru_matched": 33,
    "mlp_normal": 0,
}
FROZEN_PARAMETERS_BY_CONTROLLER = {
    "original_lif": 5_103,
    "rewired_lif": 5_103,
    "wing_lif": 8_864,
    "leg_wing_lif": 13_967,
    "optic_lif": 1_628,
    "leg_optic_lif": 6_731,
    "wing_optic_lif": 10_492,
    "leg_wing_optic_lif": 15_595,
    "gru_matched": 0,
    "mlp_normal": 0,
}
PARAMETER_MATCH_REQUIRED_BY_CONTROLLER = {
    "original_lif": True,
    "rewired_lif": True,
    "wing_lif": False,
    "leg_wing_lif": False,
    "optic_lif": True,
    "leg_optic_lif": False,
    "wing_optic_lif": False,
    "leg_wing_optic_lif": False,
    "gru_matched": True,
    "mlp_normal": True,
}
CONTRACT_SHA_BY_TASK = {
    TASKS[0]: COMMAND_WIDE_STILL_CONTRACT_SHA256,
    TASKS[1]: COMMAND_WIDE_WIND_CONTRACT_SHA256,
}
TOTAL_INTERACTIONS = 1_000_000
EPISODES_PER_JOB = 16
STEPS_PER_EPISODE = 600
JOB_COUNT = len(CONTROLLERS) * len(SEEDS) * len(TASKS)
TOTAL_MATRIX_INTERACTIONS = JOB_COUNT * TOTAL_INTERACTIONS
TOTAL_EVALUATION_EPISODES = JOB_COUNT * EPISODES_PER_JOB
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
QUEUE_KIND = "crazyflie_command_all_fair_queue_v4"
QUEUE_SCHEMA_VERSION = 4
QUEUE_FILE_NAME = "queue.json"
TRAINER = (SCRIPT_DIR / "drone_train.py").resolve()
EVALUATOR = (SCRIPT_DIR / "crazyflie_command_evaluate.py").resolve()
V1_RUNNER = (SCRIPT_DIR / "crazyflie_command_queue.py").resolve()
V2_RUNNER = (SCRIPT_DIR / "crazyflie_command_queue_v2.py").resolve()
V3_RUNNER = (SCRIPT_DIR / "crazyflie_command_combinations_queue_v3.py").resolve()
PRIOR_QUEUE = (ROOT / "runs" / "crazyflie_command_seed0_500k" / "queue.json").resolve()


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
        "trainer_policies", "seeds", "total_interactions_per_job", "training",
        "command_envelope", "evaluation", "connectomes", "rewire", "queue",
        "comparison_contract", "historical_v2_identity", "preserved_v3_identity",
        "prior_parallel_paging_evidence",
    }


def _parser_choices(parser: argparse.ArgumentParser, option: str) -> set[str]:
    for action in parser._actions:
        if option in action.option_strings:
            return set(action.choices or ())
    return set()


def _expected_pairs() -> list[tuple[str, int, str]]:
    return [
        (controller, seed, task)
        for controller in CONTROLLERS
        for seed in SEEDS
        for task in TASKS
    ]


def _validate_prior_paging_evidence(declaration: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "queue": "runs/crazyflie_command_seed0_500k/queue.json",
        "queue_sha256": "648ad8b8476a1b4873e23eedacefeb803e3d7887d73d171e3e1512245ef5b99c",
        "required_event": "hard_resource_gate",
        "required_sustained_paging": True,
        "disposition": "v4_max_parallel_fixed_to_one",
    }
    if declaration != expected:
        raise ValueError("prior_parallel_paging_evidence differs from reviewed evidence")
    if not PRIOR_QUEUE.is_file() or sha256_file(PRIOR_QUEUE) != expected["queue_sha256"]:
        raise ValueError("completed v1 paging evidence is missing or changed")
    prior = _read_json(PRIOR_QUEUE)
    matches = [
        event for event in prior.get("events", [])
        if isinstance(event, dict)
        and event.get("event") == "hard_resource_gate"
        and isinstance(event.get("sample"), dict)
        and event["sample"].get("sustained_paging") is True
    ]
    if len(matches) != 1:
        raise ValueError("v1 queue lacks the unique sustained-paging event")
    return matches[0]


def _validate_live_contracts() -> dict[str, str]:
    compact = command_wide_contract_payload()
    if (
        compact.get("maximum_horizontal_speed_m_s") != 1.0
        or compact.get("maximum_vertical_speed_m_s") != 0.5
        or compact.get("maximum_yaw_rate_rad_s") != 1.5
    ):
        raise ValueError("live command envelope differs from the fair v4 envelope")
    expected_clock = {
        "basis": "global_control_intervals_per_environment",
        "interaction_clock": "completed_control_intervals_times_num_envs",
        "same_interval_for_every_environment": True,
        "schedule_independent_of_episode_termination": True,
        "episode_termination_gates_command_cursor": False,
        "episode_reset_advances_command_cursor": False,
        "episode_termination_gates_training_wind_cursor": False,
        "episode_reset_advances_training_wind_cursor": False,
    }
    result: dict[str, str] = {}
    for task in TASKS:
        wind_enabled = task == TASKS[1]
        payload = command_wide_training_contract_payload(wind_enabled=wind_enabled)
        digest = command_wide_training_contract_sha256(wind_enabled=wind_enabled)
        if digest != CONTRACT_SHA_BY_TASK[task] or canonical_sha256(payload) != digest:
            raise ValueError(f"live command-v2 contract hash is inconsistent for {task}")
        if payload.get("training_interaction_budget") != TOTAL_INTERACTIONS:
            raise ValueError(f"live command-v2 budget differs for {task}")
        if payload.get("training_schedule_clock") != expected_clock:
            raise ValueError(f"live command-v2 schedule is not reset-invariant for {task}")
        result[task] = digest
    return result


def validate_runtime_interfaces() -> dict[str, Any]:
    missing: list[str] = []
    required_policies = set(POLICY_BY_CONTROLLER.values())
    if not required_policies.issubset(set(getattr(drone_train, "POLICIES", ()))):
        missing.append("trainer policies")
    if not set(TASKS).issubset(set(getattr(drone_train, "TRAINING_TASKS", ()))):
        missing.append("trainer tasks")
    signature = inspect.signature(build_controller)
    for argument in (
        "connectome_manifest", "wing_connectome_manifest",
        "optic_connectome_manifest", "rewire_manifest_path",
    ):
        if argument not in signature.parameters:
            missing.append(f"build_controller {argument}")
    try:
        import crazyflie_command_evaluate as evaluator

        parser = evaluator._parser()
        if not set(TASKS).issubset(_parser_choices(parser, "--task")):
            missing.append("evaluator tasks")
        if not required_policies.issubset(_parser_choices(parser, "--policy")):
            missing.append("evaluator policies")
        if "command_v2" not in _parser_choices(parser, "--protocol"):
            missing.append("evaluator command_v2 protocol")
        if getattr(evaluator, "INFERENCE_LATENCY_WARMUP_CALLS", None) != 10:
            missing.append("evaluator authenticated inference latency")
    except Exception as exc:
        missing.append(f"evaluator parser ({type(exc).__name__}: {exc})")
    if missing:
        raise ValueError("v4 runtime interfaces are incomplete: " + "; ".join(missing))
    return {
        "trainer_policies": sorted(required_policies),
        "trainer_tasks": list(TASKS),
        "evaluation_protocols": dict(PROTOCOL_BY_TASK),
        "authenticated_inference_latency": True,
        "multi_connectome_arguments": [
            "--connectome_manifest", "--wing_connectome_manifest",
            "--optic_connectome_manifest",
        ],
    }


def validate_config(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = _read_json(path)
    if set(config) != _expected_config_keys():
        raise ValueError("v4 config top-level fields differ from the closed schema")
    exact = {
        "schema_version": 4,
        "kind": "crazyflie_command_all_fair_matrix_v4",
        "label": "crazyflie_command_all_fair_seeds0_1_2_1m",
        "output_root": "runs/crazyflie_command_all_fair_seeds0_1_2_1m",
        "isaac_python": str(ISAAC_PYTHON),
        "tasks": list(TASKS),
        "task_protocols": PROTOCOL_BY_TASK,
        "contract_profile": "command_v2",
        "controllers": list(CONTROLLERS),
        "trainer_policies": POLICY_BY_CONTROLLER,
        "seeds": list(SEEDS),
        "total_interactions_per_job": TOTAL_INTERACTIONS,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be exactly {expected!r}")
    expected_training = {
        "num_envs": 40,
        "horizon": 100,
        "microbatch_size": 40,
        "ppo_epochs": 2,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "checkpoint_every_updates": 25,
        "precision": "float32",
        "device": "cuda:0",
    }
    if config.get("training") != expected_training:
        raise ValueError("training differs from the reviewed v2 PPO contract")
    if TOTAL_INTERACTIONS % (
        expected_training["num_envs"] * expected_training["horizon"]
    ):
        raise ValueError("v4 interaction budget does not divide into PPO updates")
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
        raise ValueError("command_envelope differs from the fair v4 envelope")
    live_contract_hashes = _validate_live_contracts()
    expected_evaluation = {
        "episodes_per_job": EPISODES_PER_JOB,
        "steps_per_episode": STEPS_PER_EPISODE,
        "deterministic_actions": True,
        "activity_from_actual_controller": True,
        "authenticated_inference_latency_required": True,
        "latency_warmup_calls": 10,
        "device": "cuda:0",
        "script": "scripts/crazyflie_command_evaluate.py",
        "explicit_task_argument": True,
        "analysis_kind": "crazyflie_command_follow_heldout_v2",
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("evaluation differs from the authenticated v4 contract")
    expected_queue = {
        "default_max_parallel": 1,
        "maximum_parallel": 1,
        "lif_first": True,
        "failure_isolation": True,
        "retry_failed_or_paused_only_with_resume": True,
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
        "sustained_paging_sample_count": 3,
        "network_independent": True,
        "immutable_attempt_logs": True,
    }
    if config.get("queue") != expected_queue:
        raise ValueError("queue differs from the failure-isolated sequential policy")
    expected_comparison = {
        "same_task_stream_budget_seed_and_ppo_hyperparameters": True,
        "still_and_wind_paired_within_each_controller_and_seed": True,
        "all_lif_jobs_before_all_gru_and_mlp_jobs": True,
        "reset_invariant_global_training_schedule_required": True,
        "seeds": list(SEEDS),
        "lif_controllers": list(LIF_CONTROLLERS),
        "baseline_controllers": list(BASELINE_CONTROLLERS),
        "components": {
            key: list(value) for key, value in COMPONENTS_BY_CONTROLLER.items()
        },
        "capacity_contract": {
            "actor_trainable_parameters": dict(ACTOR_PARAMETERS_BY_CONTROLLER),
            "critic_trainable_parameters": dict(CRITIC_PARAMETERS_BY_CONTROLLER),
            "total_dynamic_state_per_environment": dict(
                DYNAMIC_STATE_BY_CONTROLLER
            ),
        },
        "parameter_count_reporting_required": True,
        "cross_capacity_claims_are_descriptive_only": True,
    }
    if config.get("comparison_contract") != expected_comparison:
        raise ValueError("comparison_contract differs from the all-fair declaration")
    expected_v2 = {
        "config": "configs/experiments/crazyflie_command_optic_wind_seed0_1m.json",
        "config_sha256": "bf7369d74d119784eb6a932cbeb57734fa93b7fdda5457e3e46decfb6f140c44",
        "runner": "scripts/crazyflie_command_queue_v2.py",
        "runner_sha256": "16d7035665045ebd2128a3b369e98f49828d9f3b5c3da0685a377c9cb8447dd4",
        "completed_queue": "runs/crazyflie_command_optic_wind_seed0_1m/queue.json",
        "completed_queue_sha256": "68f4fee83d659edb439691aa79c5f4b6decc610c1e35e869b1caddd04cf5bddc",
        "completed_report": "docs/crazyflie_command_optic_wind_report.md",
        "completed_report_sha256": "7988eabbb82c710dda724f36bfb19852439cafd984562b997ef238d41b7a6fed",
        "required_queue_status": "completed",
        "required_job_count": 14,
        "fair_evidence_eligible": False,
        "disposition": "preserved_historical_only_pre_reset_invariant_schedule",
    }
    historical = config.get("historical_v2_identity")
    if historical != expected_v2:
        raise ValueError("historical_v2_identity differs from frozen evidence")
    for field, hash_field in (
        ("config", "config_sha256"),
        ("runner", "runner_sha256"),
        ("completed_queue", "completed_queue_sha256"),
        ("completed_report", "completed_report_sha256"),
    ):
        artifact = (ROOT / historical[field]).resolve()
        if not artifact.is_file() or sha256_file(artifact) != historical[hash_field]:
            raise ValueError(f"historical v2 {field} is missing or changed")
    completed_v2 = _read_json((ROOT / historical["completed_queue"]).resolve())
    completed_jobs = completed_v2.get("jobs")
    if (
        completed_v2.get("status") != "completed"
        or completed_v2.get("job_count") != 14
        or completed_v2.get("counts") != {"completed": 14}
        or not isinstance(completed_jobs, list)
        or len(completed_jobs) != 14
        or any(
            not isinstance(job, dict)
            or job.get("status") != "completed"
            or job.get("training_status") != "completed"
            or job.get("evaluation_status") != "completed"
            for job in completed_jobs
        )
    ):
        raise ValueError("historical v2 queue is not its terminal 14-job artifact")
    expected_v3 = {
        "config": "configs/experiments/crazyflie_command_combinations_seed0_1m.json",
        "config_sha256": "34df6d0480634f48bc9c86acb43e169406ee54e8b1017459ac9668d3f19d1223",
        "runner": "scripts/crazyflie_command_combinations_queue_v3.py",
        "runner_sha256": "3a4b5c320a244c2961e29e1ad57b8529ff309e8b3aab38eccb1c250c634b9d93",
        "relation": "preserved_not_adopted_or_mutated",
    }
    preserved_v3 = config.get("preserved_v3_identity")
    if preserved_v3 != expected_v3:
        raise ValueError("preserved_v3_identity differs from frozen files")
    for field, hash_field in (("config", "config_sha256"), ("runner", "runner_sha256")):
        artifact = (ROOT / preserved_v3[field]).resolve()
        if not artifact.is_file() or sha256_file(artifact) != preserved_v3[hash_field]:
            raise ValueError(f"preserved v3 {field} is missing or changed")
    connectomes = config.get("connectomes")
    if not isinstance(connectomes, dict) or set(connectomes) != {
        "leg_manifest", "leg_manifest_sha256", "wing_manifest",
        "wing_manifest_sha256", "optic_manifest", "optic_manifest_sha256",
    }:
        raise ValueError("connectomes differs from the v4 schema")
    for label in ("leg", "wing", "optic"):
        relative = connectomes[f"{label}_manifest"]
        expected_hash = connectomes[f"{label}_manifest_sha256"]
        artifact = (ROOT / relative).resolve()
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not artifact.is_file()
            or sha256_file(artifact) != expected_hash
        ):
            raise ValueError(f"pinned {label} connectome is missing or changed")
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
        raise ValueError("pinned rewire manifest is missing or changed")
    prior_event = _validate_prior_paging_evidence(
        config.get("prior_parallel_paging_evidence", {})
    )
    if not ISAAC_PYTHON.is_file() or not os.access(ISAAC_PYTHON, os.X_OK):
        raise ValueError(f"Isaac Python is not executable: {ISAAC_PYTHON}")
    for script in (TRAINER, EVALUATOR, V1_RUNNER, V2_RUNNER, V3_RUNNER):
        if not script.is_file():
            raise ValueError(f"required local script is missing: {script}")
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
    config["_live_contract_hashes"] = live_contract_hashes
    return config


def _training_args(
    config: Mapping[str, Any], task: str, policy: str, seed: int
) -> SimpleNamespace:
    training = config["training"]
    return SimpleNamespace(
        task=task,
        contract_profile="command_v2",
        policy=policy,
        seed=seed,
        num_envs=40,
        total_interactions=TOTAL_INTERACTIONS,
        horizon=100,
        microbatch_size=40,
        ppo_epochs=training["ppo_epochs"],
        learning_rate=training["learning_rate"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        clip_ratio=training["clip_ratio"],
        value_coefficient=training["value_coefficient"],
        entropy_coefficient=training["entropy_coefficient"],
        max_grad_norm=training["max_grad_norm"],
        target_kl=training["target_kl"],
        checkpoint_every_updates=training["checkpoint_every_updates"],
        connectome_manifest=Path(config["_leg_manifest"]),
        wing_connectome_manifest=Path(config["_wing_manifest"]),
        optic_connectome_manifest=Path(config["_optic_manifest"]),
        rewire_seed=config["rewire"]["seed"],
        rewire_manifest=Path(config["_rewire_manifest"]),
        evaluation_protocol=PROTOCOL_BY_TASK[task],
        warm_start_checkpoint=None,
    )


def _primary_manifest(args: SimpleNamespace) -> Path:
    if args.policy in {"wing_lif", "wing_optic_lif"}:
        return args.wing_connectome_manifest
    if args.policy == "optic_lif":
        return args.optic_connectome_manifest
    return args.connectome_manifest


def _job_fingerprint(
    config: Mapping[str, Any], task: str, policy: str, seed: int
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    args = _training_args(config, task, policy, seed)
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
    config: Mapping[str, Any], task: str, policy: str, seed: int,
    run_dir: Path, fingerprint: str, pause_file: Path,
) -> list[str]:
    training = config["training"]
    return [
        config["isaac_python"],
        str(TRAINER),
        "--task", task,
        "--contract_profile", "command_v2",
        "--policy", policy,
        "--seed", str(seed),
        "--num_envs", "40",
        "--total_interactions", str(TOTAL_INTERACTIONS),
        "--horizon", "100",
        "--microbatch_size", "40",
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
        "--run_dir", str(run_dir),
        "--expected_fingerprint", fingerprint,
        "--pause_file", str(pause_file),
        "--device", training["device"],
        "--headless",
    ]


def _evaluation_command(
    config: Mapping[str, Any], task: str, policy: str, seed: int,
    checkpoint: Path, output: Path, fingerprint: str,
) -> list[str]:
    return [
        config["isaac_python"],
        str(EVALUATOR),
        "--task", task,
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--protocol", PROTOCOL_BY_TASK[task],
        "--device", config["evaluation"]["device"],
        "--expected_fingerprint", fingerprint,
        "--training_seed", str(seed),
        "--policy", policy,
        "--headless",
    ]


def _controller_reports(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for label in CONTROLLERS:
        policy_kind = POLICY_BY_CONTROLLER[label]
        kwargs: dict[str, Any] = {}
        if policy_kind == "frozen_lif_degree_rewired":
            kwargs["rewire_manifest_path"] = config["_rewire_manifest"]
        if label in {"leg_optic_lif", "wing_optic_lif", "leg_wing_optic_lif"}:
            kwargs["enforce_parameter_match"] = False
        policy, report = build_controller(
            policy_kind,
            observation_dim=12,
            action_dim=4,
            device="cpu",
            connectome_manifest=config["_leg_manifest"],
            wing_connectome_manifest=config["_wing_manifest"],
            optic_connectome_manifest=config["_optic_manifest"],
            rewire_seed=config["rewire"]["seed"],
            **kwargs,
        )
        del policy
        report = dict(report)
        if (
            report.get("controller_kind") != REPORT_KIND_BY_CONTROLLER[label]
            or report.get("actor_trainable_parameters")
            != ACTOR_PARAMETERS_BY_CONTROLLER[label]
            or report.get("critic_trainable_parameters")
            != CRITIC_PARAMETERS_BY_CONTROLLER[label]
            or report.get("total_dynamic_state_per_environment")
            != DYNAMIC_STATE_BY_CONTROLLER[label]
            or report.get("frozen_parameters")
            != FROZEN_PARAMETERS_BY_CONTROLLER[label]
            or report.get("parameter_matching_required")
            is not PARAMETER_MATCH_REQUIRED_BY_CONTROLLER[label]
            or report.get("fusion_contract") != FUSION_BY_CONTROLLER[label]
        ):
            raise RuntimeError(f"{label} differs from the exact v4 capacity contract")
        reports[label] = report
    return reports


def build_queue(
    config: Mapping[str, Any], output_root: Path | None = None
) -> dict[str, Any]:
    root = Path(output_root or config["_output_root"]).resolve()
    reports = _controller_reports(config)
    jobs: list[dict[str, Any]] = []
    for controller_priority, controller in enumerate(CONTROLLERS):
        policy = POLICY_BY_CONTROLLER[controller]
        for seed_priority, seed in enumerate(SEEDS):
            for task_priority, task in enumerate(TASKS):
                task_slug = "still" if task == TASKS[0] else "wind"
                identifier = (
                    f"{len(jobs) + 1:03d}__{controller}__{task_slug}__seed-{seed}"
                )
                run_dir = root / "jobs" / identifier
                checkpoint = run_dir / "checkpoints" / "latest.pt"
                pause_file = run_dir / "pause.request"
                evaluation_output = (
                    root / "evaluations" / identifier / "heldout.json"
                )
                fingerprint, payload, evaluation_manifest = _job_fingerprint(
                    config, task, policy, seed
                )
                report = reports[controller]
                jobs.append(
                    {
                        "id": identifier,
                        "controller": controller,
                        "policy": policy,
                        "controller_priority": controller_priority,
                        "seed_priority": seed_priority,
                        "task_priority": task_priority,
                        "architecture_class": (
                            "lif" if controller in LIF_CONTROLLERS else "baseline"
                        ),
                        "controller_components": list(
                            COMPONENTS_BY_CONTROLLER[controller]
                        ),
                        "task": task,
                        "evaluation_protocol": PROTOCOL_BY_TASK[task],
                        "seed": seed,
                        "command_schedule_seed": seed,
                        "paired_task_seed_key": f"{controller}__seed-{seed}",
                        "contract_profile": "command_v2",
                        "status": "pending",
                        "training_status": "pending",
                        "evaluation_status": "pending",
                        "total_interactions": TOTAL_INTERACTIONS,
                        "expected_updates": 250,
                        "run_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "training_manifest": str(
                            run_dir / "training_manifest.json"
                        ),
                        "pause_file": str(pause_file),
                        "evaluation_output": str(evaluation_output),
                        "expected_fingerprint": fingerprint,
                        "fingerprint_payload": payload,
                        "evaluation_manifest": evaluation_manifest,
                        "evaluation_manifest_id": evaluation_manifest["manifest_id"],
                        "command_training_contract_sha256": CONTRACT_SHA_BY_TASK[task],
                        "controller_report_sha256": canonical_sha256(report),
                        "expected_actor_trainable_parameters": (
                            ACTOR_PARAMETERS_BY_CONTROLLER[controller]
                        ),
                        "expected_critic_trainable_parameters": (
                            CRITIC_PARAMETERS_BY_CONTROLLER[controller]
                        ),
                        "expected_dynamic_state_per_environment": (
                            DYNAMIC_STATE_BY_CONTROLLER[controller]
                        ),
                        "expected_core_checksum": report.get("core_checksum"),
                        "expected_per_core_checksums": report.get(
                            "per_core_checksums", {}
                        ),
                        "expected_connectome_manifests": report.get(
                            "connectome_manifests", {}
                        ),
                        "expected_connectome_checksums": report.get(
                            "connectome_checksums", {}
                        ),
                        "training_command": _training_command(
                            config, task, policy, seed, run_dir, fingerprint, pause_file
                        ),
                        "evaluation_command": _evaluation_command(
                            config, task, policy, seed, checkpoint,
                            evaluation_output, fingerprint,
                        ),
                        "attempts": [],
                        "failure_history": [],
                    }
                )
    if (
        len(jobs) != JOB_COUNT
        or [
            (job["controller"], job["seed"], job["task"]) for job in jobs
        ]
        != _expected_pairs()
        or any(job["architecture_class"] != "lif" for job in jobs[:48])
        or any(job["architecture_class"] != "baseline" for job in jobs[48:])
    ):
        raise RuntimeError("v4 queue did not resolve the exact LIF-first 60-cell order")
    historical = config["historical_v2_identity"]
    preserved_v3 = config["preserved_v3_identity"]
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
        "v2_supervisor_helper": str(V2_RUNNER),
        "v2_supervisor_helper_sha256": sha256_file(V2_RUNNER),
        "v3_validator_helper": str(V3_RUNNER),
        "v3_validator_helper_sha256": sha256_file(V3_RUNNER),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "historical_v2": dict(historical),
        "preserved_v3": dict(preserved_v3),
        "historical_v2_fair_evidence_eligible": False,
        "output_root": str(root),
        "queue_file": str(root / QUEUE_FILE_NAME),
        "global_pause_file": str(root / "pause.request"),
        "job_count": JOB_COUNT,
        "predicted_training_interactions": TOTAL_MATRIX_INTERACTIONS,
        "predicted_evaluation_episodes": TOTAL_EVALUATION_EPISODES,
        "controller_order": list(CONTROLLERS),
        "seed_order": list(SEEDS),
        "task_order": list(TASKS),
        "task_protocols": dict(PROTOCOL_BY_TASK),
        "live_command_training_contract_sha256": dict(CONTRACT_SHA_BY_TASK),
        "lif_job_count": len(LIF_CONTROLLERS) * len(SEEDS) * len(TASKS),
        "baseline_job_count": len(BASELINE_CONTROLLERS) * len(SEEDS) * len(TASKS),
        "lif_first": True,
        "maximum_parallel": 1,
        "failure_isolation": True,
        "retry_failed_or_paused_only_with_resume": True,
        "parallelism_disposition": config["prior_parallel_paging_evidence"],
        "resource_limits": {
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "sustained_paging_sample_count": 3,
            "default_max_parallel": 1,
            "maximum_parallel": 1,
        },
        "controller_reports": reports,
        "counts": {"pending": JOB_COUNT},
        "config": _public_config(config),
        "interface_contract": config["_interface"],
        "jobs": jobs,
        "events": [
            {"utc": _utc_now(), "event": "verified_v4_dry_run_created"}
        ],
    }


def _refresh_status(queue: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for job in queue["jobs"]:
        status = str(job["status"])
        counts[status] = counts.get(status, 0) + 1
    queue["counts"] = dict(sorted(counts.items()))
    total = int(queue.get("job_count", len(queue["jobs"])))
    if queue.get("dry_run"):
        queue["status"] = "dry_run"
    elif counts.get("running"):
        queue["status"] = "running"
    elif counts.get("completed") == total:
        queue["status"] = "completed"
    elif counts.get("paused"):
        queue["status"] = "paused"
    elif counts.get("failed") and counts.get("completed"):
        queue["status"] = "partial_failed"
    else:
        queue["status"] = "incomplete"


def save_queue(path: Path, queue: dict[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    queue["updated_utc"] = _utc_now()
    _refresh_status(queue)
    _atomic_json(path, queue)
    _atomic_json(
        path.with_name("queue_summary.json"),
        {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "kind": QUEUE_KIND,
            "status": queue["status"],
            "dry_run": queue["dry_run"],
            "revision": queue["revision"],
            "updated_utc": queue["updated_utc"],
            "counts": queue["counts"],
            "job_count": JOB_COUNT,
            "predicted_training_interactions": TOTAL_MATRIX_INTERACTIONS,
            "predicted_evaluation_episodes": TOTAL_EVALUATION_EPISODES,
            "maximum_parallel": 1,
            "failure_isolation": True,
            "resource_block": queue.get("resource_block"),
            "jobs": [
                {
                    key: job[key]
                    for key in (
                        "id", "controller", "policy", "seed", "task", "status",
                        "training_status", "evaluation_status", "checkpoint",
                    )
                }
                for job in queue["jobs"]
            ],
        },
    )


def _memory_gate_passed(value: Any) -> bool:
    return v2_queue._memory_gate_passed(value)


def _valid_controller_report(job: Mapping[str, Any], report: Any) -> bool:
    if not isinstance(report, dict):
        return False
    controller = str(job.get("controller"))
    if controller not in CONTROLLERS:
        return False
    if controller in {
        "leg_optic_lif", "wing_optic_lif", "leg_wing_optic_lif"
    } and not v3_queue._valid_combination_report(job, report):
        return False
    expected_core = job.get("expected_core_checksum")
    expected_per_core = job.get("expected_per_core_checksums")
    if controller in LIF_CONTROLLERS:
        if (
            not isinstance(expected_core, str)
            or len(expected_core) != 64
            or not isinstance(expected_per_core, dict)
            or not expected_per_core
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in expected_per_core.values()
            )
        ):
            return False
    elif expected_core is not None or expected_per_core != {}:
        return False
    return (
        report.get("controller_kind") == REPORT_KIND_BY_CONTROLLER[controller]
        and report.get("actor_trainable_parameters")
        == ACTOR_PARAMETERS_BY_CONTROLLER[controller]
        and report.get("critic_trainable_parameters")
        == CRITIC_PARAMETERS_BY_CONTROLLER[controller]
        and report.get("total_dynamic_state_per_environment")
        == DYNAMIC_STATE_BY_CONTROLLER[controller]
        and report.get("frozen_parameters")
        == FROZEN_PARAMETERS_BY_CONTROLLER[controller]
        and report.get("parameter_matching_required")
        is PARAMETER_MATCH_REQUIRED_BY_CONTROLLER[controller]
        and report.get("fusion_contract") == FUSION_BY_CONTROLLER[controller]
        and report.get("core_checksum") == expected_core
        and report.get("per_core_checksums") == expected_per_core
        and report.get("connectome_manifests")
        == job.get("expected_connectome_manifests")
        and report.get("connectome_checksums")
        == job.get("expected_connectome_checksums")
        and canonical_sha256(report) == job.get("controller_report_sha256")
    )


def valid_training(job: Mapping[str, Any]) -> bool:
    try:
        manifest = _read_json(Path(job["training_manifest"]))
        checkpoint = Path(job["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
        schedule = manifest["command_schedule"]
        report = manifest["controller_report"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "completed"
        and manifest.get("task") == job.get("task")
        and manifest.get("contract_profile") == "command_v2"
        and manifest.get("controller") == job.get("policy")
        and manifest.get("seed") == job.get("seed")
        and manifest.get("num_envs") == 40
        and manifest.get("horizon") == 100
        and manifest.get("requested_interactions") == TOTAL_INTERACTIONS
        and manifest.get("environment_interactions") == TOTAL_INTERACTIONS
        and manifest.get("completed_updates") == job.get("expected_updates") == 250
        and manifest.get("fingerprint") == job.get("expected_fingerprint")
        and manifest.get("fingerprint_payload") == job.get("fingerprint_payload")
        and manifest.get("evaluation_manifest_id")
        == job.get("evaluation_manifest_id")
        and Path(manifest.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and manifest.get("checkpoint_sha256") == checkpoint_hash
        and _valid_controller_report(job, report)
        and manifest.get("core_checksum_before")
        == job.get("expected_core_checksum")
        and manifest.get("core_checksum_after")
        == job.get("expected_core_checksum")
        and manifest.get("per_core_checksums_before")
        == job.get("expected_per_core_checksums")
        and manifest.get("per_core_checksums_after")
        == job.get("expected_per_core_checksums")
        and isinstance(schedule, dict)
        and schedule.get("command_training_contract_sha256")
        == job.get("command_training_contract_sha256")
        and isinstance(schedule.get("state"), dict)
        and schedule["state"].get("training_interactions") == TOTAL_INTERACTIONS
        and _memory_gate_passed(manifest.get("memory_gate"))
    )


def _manifest_activity_identity(manifest_value: str) -> dict[str, Any] | None:
    try:
        manifest = Path(manifest_value).expanduser().resolve()
        raw = _read_json(manifest)
        neurons = (manifest.parent / raw["neurons_path"]).resolve()
        rows = json.loads(neurons.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "neurons": str(neurons),
        "neurons_sha256": sha256_file(neurons),
        "neuron_count": len(rows),
    }


def _valid_activity(job: Mapping[str, Any], activity: Any) -> bool:
    if not isinstance(activity, dict) or activity.get("controller") != job.get("policy"):
        return False
    per_unit = activity.get("per_unit")
    if (
        not isinstance(per_unit, list)
        or not per_unit
        or activity.get("unit_count") != len(per_unit)
        or any(not isinstance(row, dict) for row in per_unit)
    ):
        return False
    controller = str(job.get("controller"))
    if controller in {"leg_optic_lif", "wing_optic_lif", "leg_wing_optic_lif"}:
        if not v3_queue._valid_activity_provenance(job, activity):
            return False
    provenance = activity.get("role_provenance")
    if controller in LIF_CONTROLLERS:
        labels = COMPONENTS_BY_CONTROLLER[controller]
        manifests = job.get("expected_connectome_manifests")
        if not isinstance(provenance, dict) or set(provenance) != set(labels):
            return False
        if not isinstance(manifests, dict):
            return False
        expected_counts: dict[str, int] = {}
        for label in labels:
            manifest_key = label if len(labels) > 1 else "primary"
            manifest_value = manifests.get(manifest_key)
            if not isinstance(manifest_value, str):
                return False
            identity = _manifest_activity_identity(manifest_value)
            if provenance.get(label) != identity or identity is None:
                return False
            expected_counts[label] = int(identity["neuron_count"])
        observed_counts = {label: 0 for label in labels}
        for row in per_unit:
            prefix = str(row.get("id", "")).split(":", 1)[0]
            if prefix not in observed_counts:
                return False
            observed_counts[prefix] += 1
        return (
            observed_counts == expected_counts
            and activity.get("unit_count") == sum(expected_counts.values())
        )
    if not isinstance(provenance, dict) or set(provenance) != {"engineering"}:
        return False
    engineering = provenance["engineering"]
    if controller == "gru_matched":
        return (
            engineering
            == {
                "kind": "matched_gru_hidden_state",
                "biological_roles": False,
                "unit_count": 33,
            }
            and len(per_unit) == 33
            and all(str(row.get("id", "")).startswith("gru:hidden:") for row in per_unit)
        )
    return (
        engineering
        == {
            "kind": "matched_mlp_post_activation_hidden_units",
            "activation_values": "absolute_actual_post_activation_output",
            "biological_roles": False,
            "layer_widths": [61, 61],
            "unit_count": 122,
        }
        and len(per_unit) == 122
        and all(str(row.get("id", "")).startswith("mlp:hidden_") for row in per_unit)
    )


def _valid_physical_wind(job: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
    try:
        physical = value["summary"]["physical_wind"]
        integrity = physical["integrity"]
    except (KeyError, TypeError):
        return False
    common = (
        isinstance(physical, dict)
        and isinstance(integrity, dict)
        and integrity.get("passed") is True
        and integrity.get("force_within_declared_bound") is True
        and integrity.get("torque_within_declared_bound") is True
        and value.get("integrity", {}).get("physical_wind_telemetry_passed") is True
        and value.get("integrity", {}).get(
            "physical_wind_uses_terminal_actual_interval"
        ) is True
    )
    if job.get("task") == TASKS[0]:
        return bool(
            common
            and physical.get("condition") == "still_air"
            and integrity.get("still_air_exact_zero_all_simulated_intervals") is True
        )
    return bool(
        common
        and physical.get("condition") == "wind"
        and integrity.get("wind_nonzero_wrench_observed") is True
        and integrity.get("expected_force_matches_on_observed_intervals") is True
        and integrity.get("expected_torque_matches_on_observed_intervals") is True
    )


def valid_evaluation(job: Mapping[str, Any]) -> bool:
    try:
        value = _read_json(Path(job["evaluation_output"]))
        checkpoint = Path(job["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_record = value["checkpoint"]
        summary = value["summary"]
        integrity = value["integrity"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    quality_keys = {
        "acceleration_quality", "command_response", "flight_stability",
        "survival_not_die",
    }
    return (
        value.get("schema_version") == 1
        and value.get("analysis_kind") == "crazyflie_command_follow_heldout_v2"
        and value.get("status") == "PASS"
        and value.get("task") == job.get("task")
        and value.get("controller") == job.get("policy")
        and isinstance(checkpoint_record, dict)
        and Path(checkpoint_record.get("path", "")).resolve() == checkpoint.resolve()
        and checkpoint_record.get("sha256") == checkpoint_hash
        and checkpoint_record.get("training_seed") == job.get("seed")
        and checkpoint_record.get("total_interactions") == TOTAL_INTERACTIONS
        and checkpoint_record.get("reproduction_fingerprint")
        == job.get("expected_fingerprint")
        and checkpoint_record.get("evaluation_manifest_id")
        == job.get("evaluation_manifest_id")
        and checkpoint_record.get("evaluation_manifest")
        == job.get("evaluation_manifest")
        and value.get("protocol")
        == job.get("evaluation_manifest", {}).get("evaluation_protocol")
        and value.get("protocol_sha256")
        == job.get("evaluation_manifest", {}).get("evaluation_protocol_sha256")
        and value.get("episodes_requested") == EPISODES_PER_JOB
        and value.get("episodes_evaluated") == EPISODES_PER_JOB
        and value.get("steps_per_episode") == STEPS_PER_EPISODE
        and value.get("vectorized_environment_count") == EPISODES_PER_JOB
        and value.get("deterministic_actions") is True
        and value.get("policy_action_source")
        == "actual_trained_controller_no_assist"
        and isinstance(value.get("episodes"), list)
        and len(value["episodes"]) == EPISODES_PER_JOB
        and all(isinstance(row, dict) and row for row in value["episodes"])
        and isinstance(summary, dict)
        and isinstance(summary.get("score"), dict)
        and set(summary.get("control_quality", {}).get(
            "component_scores_0_100", {}
        ))
        == quality_keys
        and _valid_controller_report(job, value.get("controller_report"))
        and _valid_activity(job, value.get("activity"))
        and v3_queue._valid_inference_latency(value.get("inference_latency"))
        and _valid_physical_wind(job, value)
        and _memory_gate_passed(value.get("memory_gate"))
        and all(
            integrity.get(key) is True
            for key in (
                "task_manifest_matched", "evaluation_manifest_matched",
                "source_set_matched", "checkpoint_completed_budget",
                "all_actions_finite_and_bounded", "activity_from_actual_forward",
                "inference_latency_from_actual_forward",
            )
        )
        and v1_queue._finite_tree(value)
    )


def _process_alive(job: Mapping[str, Any]) -> bool:
    return v2_queue._process_alive(job)


def reconcile_queue(queue: dict[str, Any], *, retry_failed: bool) -> None:
    """Reconcile artifacts without silently retrying an isolated failure."""

    for job in queue["jobs"]:
        if job.get("status") == "running" and _process_alive(job):
            job["recovered_detached_process"] = True
            continue
        job.pop("active_process", None)
        try:
            training_ok = valid_training(job)
            evaluation_ok = training_ok and valid_evaluation(job)
        except Exception as exc:
            _record_job_failure(
                queue,
                job,
                phase="artifact_reconciliation",
                reason=f"artifact reconciliation failed: {type(exc).__name__}: {exc}",
                exit_code=None,
            )
            continue
        if evaluation_ok:
            job["training_status"] = "completed"
            job["evaluation_status"] = "completed"
            job["status"] = "completed"
            continue
        old_status = job.get("status")
        if old_status in {"failed", "paused"} and not retry_failed:
            if training_ok:
                job["training_status"] = "completed"
            continue
        job["training_status"] = "completed" if training_ok else "pending"
        job["evaluation_status"] = "pending"
        job["status"] = "pending"
        if retry_failed and old_status in {"failed", "paused"}:
            job["last_retry_requested_utc"] = _utc_now()


def _next_phase(job: dict[str, Any]) -> str | None:
    if not valid_training(job):
        return "training"
    job["training_status"] = "completed"
    if not valid_evaluation(job):
        return "evaluation"
    job["evaluation_status"] = "completed"
    job["status"] = "completed"
    return None


def _archive_invalid_evaluation(job: Mapping[str, Any]) -> str | None:
    output = Path(job["evaluation_output"])
    if not output.exists() or valid_evaluation(job):
        return None
    archive_dir = output.parent / "invalid_attempts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{output.stem}-{time.time_ns()}{output.suffix}"
    os.replace(output, archive)
    return str(archive)


def _start_child(
    job: dict[str, Any], phase: str, output_root: Path
) -> dict[str, Any]:
    command = list(job[f"{phase}_command"])
    if phase == "training" and Path(job["checkpoint"]).is_file():
        command.append("--resume")
    archived = _archive_invalid_evaluation(job) if phase == "evaluation" else None
    attempt = len(job["attempts"]) + 1
    log = output_root / "logs" / f"{job['id']}__{phase}__attempt-{attempt:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("xb")
    record = {
        "attempt": attempt,
        "phase": phase,
        "started_utc": _utc_now(),
        "command": command,
        "log": str(log),
        "archived_invalid_output": archived,
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException as exc:
        stream.close()
        record.update(
            finished_utc=_utc_now(),
            exit_code=None,
            launch_error=f"{type(exc).__name__}: {exc}",
        )
        job["attempts"].append(record)
        raise
    record["pid"] = process.pid
    job["attempts"].append(record)
    job["active_process"] = {
        "pid": process.pid,
        "phase": phase,
        "started_utc": record["started_utc"],
        "log": str(log),
    }
    job["status"] = "running"
    job[f"{phase}_status"] = "running"
    return {
        "process": process,
        "stream": stream,
        "phase": phase,
        "job": job,
        "record": record,
    }


def _record_job_failure(
    queue: dict[str, Any], job: dict[str, Any], *, phase: str,
    reason: str, exit_code: int | None,
) -> None:
    failure = {
        "index": len(job.setdefault("failure_history", [])) + 1,
        "utc": _utc_now(),
        "phase": phase,
        "exit_code": exit_code,
        "reason": reason,
    }
    job["failure_history"].append(failure)
    job["failure"] = reason
    if phase in {"training", "evaluation"}:
        job[f"{phase}_status"] = "failed"
    job["status"] = "failed"
    queue["events"].append(
        {"utc": failure["utc"], "event": "job_failure_isolated", "job": job["id"], **failure}
    )


def _finish_child(
    queue: dict[str, Any], handle: dict[str, Any], exit_code: int | None
) -> None:
    if handle.get("stream") is not None:
        handle["stream"].close()
    job = handle["job"]
    phase = handle["phase"]
    record = handle["record"]
    record.update(finished_utc=_utc_now(), exit_code=exit_code)
    job.pop("active_process", None)
    resource_gate = handle.get("job_local_resource_gate")
    if isinstance(resource_gate, dict):
        record["job_local_resource_gate"] = resource_gate
        if phase == "training" and exit_code == 3:
            job.setdefault("resource_gate_history", []).append(
                {
                    "utc": _utc_now(),
                    "phase": phase,
                    "disposition": "paused_at_clean_checkpoint",
                    "sample": resource_gate,
                }
            )
            job["training_status"] = "paused"
            job["status"] = "paused"
            queue["events"].append(
                {
                    "utc": _utc_now(),
                    "event": "job_resource_gate_isolated",
                    "job": job["id"],
                    "phase": phase,
                    "disposition": "paused_at_clean_checkpoint",
                }
            )
            return
        _record_job_failure(
            queue,
            job,
            phase=phase,
            reason=(
                f"{phase} exceeded the live per-job resource gate; "
                f"child exited {exit_code!r}"
            ),
            exit_code=exit_code,
        )
        return
    valid = valid_training(job) if phase == "training" else valid_evaluation(job)
    if valid:
        job[f"{phase}_status"] = "completed"
        job["status"] = "completed" if phase == "evaluation" else "pending"
        return
    if phase == "training" and exit_code == 3:
        job["training_status"] = "paused"
        job["status"] = "paused"
        return
    reason = f"{phase} exited {exit_code!r}; artifact validation failed"
    _record_job_failure(
        queue, job, phase=phase, reason=reason, exit_code=exit_code
    )


def _pause_path(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        _atomic_json(path, payload)


def _request_pause(
    queue: Mapping[str, Any], queue_path: Path, reason: str
) -> None:
    payload = {
        "schema_version": QUEUE_SCHEMA_VERSION,
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


def _request_job_resource_stop(
    queue: dict[str, Any], queue_path: Path, active: dict[str, Any],
    sample: Mapping[str, Any],
) -> None:
    """Stop only the resource-heavy cell; do not block later safe cells."""

    if "job_local_resource_gate" in active:
        return
    job = active["job"]
    phase = active["phase"]
    active["job_local_resource_gate"] = dict(sample)
    payload = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "status": "requested",
        "utc": _utc_now(),
        "reason": "job_local_live_resource_gate",
        "queue": str(queue_path),
        "job": job["id"],
        "pid": os.getpid(),
        "sample": dict(sample),
    }
    if phase == "training":
        _pause_path(Path(job["pause_file"]), payload)
        disposition = "training_pause_requested_for_clean_checkpoint"
    else:
        active_process = job.get("active_process", {})
        child_pid = active_process.get("pid")
        if not isinstance(child_pid, int) or child_pid < 1:
            raise RuntimeError("active evaluation lacks a valid child process group")
        try:
            os.killpg(child_pid, signal.SIGTERM)
            disposition = "evaluation_process_group_sigterm"
        except ProcessLookupError:
            # The process can exit between poll() and the stop request.  The
            # resource-gate marker still makes this attempt invalid below.
            disposition = "evaluation_process_group_already_exited"
    active["record"]["job_local_resource_stop"] = {
        "utc": payload["utc"],
        "disposition": disposition,
        "sample": dict(sample),
    }
    queue["events"].append(
        {
            "utc": payload["utc"],
            "event": "job_resource_gate_stop_requested",
            "job": job["id"],
            "phase": phase,
            "disposition": disposition,
            "sample": dict(sample),
        }
    )


def _consume_pauses(queue: dict[str, Any]) -> None:
    paths = [Path(queue["global_pause_file"])] + [
        Path(job["pause_file"]) for job in queue["jobs"]
    ]
    for path in paths:
        if not path.exists():
            continue
        archive_dir = path.parent / "pause_requests"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{path.stem}-consumed-{time.time_ns()}{path.suffix}"
        os.replace(path, archive)
        queue["events"].append(
            {"utc": _utc_now(), "event": "pause_consumed", "archive": str(archive)}
        )


def _next_eligible(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the next pending cell, skipping isolated failures and pauses."""

    for job in queue["jobs"]:
        if job.get("status") == "pending":
            return job
    return None


def _resource_sample(queue: dict[str, Any], *, stage: str) -> dict[str, Any]:
    try:
        sample = v1_queue.resource_snapshot()
        swap = int(sample["swap_out_pages"])
        samples = list(queue.get("resource_gate_swap_samples", []))
        samples.append(swap)
        samples = samples[-3:]
        queue["resource_gate_swap_samples"] = samples
        sustained = len(samples) == 3 and samples[0] < samples[1] < samples[2]
        sample["sustained_paging"] = sustained
        sample["passed"] = bool(sample.get("passed") and not sustained)
    except Exception as exc:
        sample = {
            "utc": _utc_now(),
            "passed": False,
            "sustained_paging": None,
            "telemetry_error": f"{type(exc).__name__}: {exc}",
        }
    sample["stage"] = stage
    return sample


def execute_queue(
    queue: dict[str, Any], queue_path: Path, *, resume: bool
) -> int:
    """Execute sequentially while isolating cell failures and global hazards."""

    if queue.get("maximum_parallel") != 1:
        raise ValueError("v4 execution is permanently sequential after prior paging")
    if resume:
        _consume_pauses(queue)
    elif Path(queue["global_pause_file"]).exists() or any(
        Path(job["pause_file"]).exists() for job in queue["jobs"]
    ):
        raise RuntimeError("pause request exists; use --execute --resume")
    reconcile_queue(queue, retry_failed=resume)
    queue["dry_run"] = False
    queue.pop("resource_block", None)
    queue.setdefault("started_utc", _utc_now())
    queue["events"].append(
        {
            "utc": _utc_now(),
            "event": "failure_isolated_sequential_runner_started",
            "pid": os.getpid(),
            "resume": resume,
            "max_parallel": 1,
        }
    )
    save_queue(queue_path, queue)
    stop = False
    resource_stop = False
    active: dict[str, Any] | None = None

    def signal_handler(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        _request_pause(queue, queue_path, f"signal_{signum}")

    handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.signal(signum, signal_handler)
    if hasattr(signal, "SIGHUP"):
        handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for job in queue["jobs"]:
        if job.get("status") == "running" and _process_alive(job):
            attempts = job.get("attempts", [])
            record = attempts[-1] if attempts else dict(job["active_process"])
            active = {
                "process": None,
                "stream": None,
                "phase": job["active_process"]["phase"],
                "job": job,
                "record": record,
            }
            break
    try:
        while True:
            if active is not None:
                process = active["process"]
                exit_code = None if process is None else process.poll()
                alive = (
                    _process_alive(active["job"])
                    if process is None
                    else exit_code is None
                )
                if not alive:
                    try:
                        _finish_child(queue, active, exit_code)
                    except Exception as exc:
                        phase = str(active["phase"])
                        job = active["job"]
                        job.pop("active_process", None)
                        _record_job_failure(
                            queue,
                            job,
                            phase=phase,
                            reason=(
                                "post-child artifact validation failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            exit_code=exit_code,
                        )
                    active = None
                    save_queue(queue_path, queue)
                    continue
                if "job_local_resource_gate" not in active:
                    sample = _resource_sample(queue, stage="active_child")
                    queue["last_resource_sample"] = sample
                    queue.setdefault("resource_samples", []).append(sample)
                    queue["resource_samples"] = queue["resource_samples"][-200:]
                    if not sample.get("passed"):
                        global_hazard = (
                            sample.get("sustained_paging") is True
                            or "telemetry_error" in sample
                        )
                        if global_hazard:
                            stop = True
                            resource_stop = True
                            queue["resource_block"] = sample
                            queue["events"].append(
                                {
                                    "utc": _utc_now(),
                                    "event": "hard_global_resource_gate",
                                    "sample": sample,
                                }
                            )
                            _request_pause(queue, queue_path, "hard_global_resource_gate")
                            if active["phase"] == "evaluation":
                                try:
                                    _request_job_resource_stop(
                                        queue, queue_path, active, sample
                                    )
                                except Exception as exc:
                                    queue["events"].append(
                                        {
                                            "utc": _utc_now(),
                                            "event": "global_eval_stop_request_failed",
                                            "job": active["job"]["id"],
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                    )
                        else:
                            try:
                                _request_job_resource_stop(
                                    queue, queue_path, active, sample
                                )
                            except Exception as exc:
                                stop = True
                                resource_stop = True
                                queue["resource_block"] = {
                                    **sample,
                                    "resource_stop_error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                }
                                queue["events"].append(
                                    {
                                        "utc": _utc_now(),
                                        "event": "job_resource_stop_failed_global_defer",
                                        "job": active["job"]["id"],
                                        "sample": queue["resource_block"],
                                    }
                                )
                                _request_pause(
                                    queue, queue_path,
                                    "job_resource_stop_failed_global_defer",
                                )
                        save_queue(queue_path, queue)
            if stop or Path(queue["global_pause_file"]).exists():
                if active is None:
                    for job in queue["jobs"]:
                        if job.get("status") == "pending":
                            job["status"] = "paused"
                    queue["events"].append(
                        {"utc": _utc_now(), "event": "runner_paused"}
                    )
                    save_queue(queue_path, queue)
                    return 4 if resource_stop else 3
            elif active is None:
                job = _next_eligible(queue)
                if job is None:
                    queue["finished_utc"] = _utc_now()
                    if all(item.get("status") == "completed" for item in queue["jobs"]):
                        queue["events"].append(
                            {"utc": _utc_now(), "event": "runner_completed_all_cells"}
                        )
                        save_queue(queue_path, queue)
                        return 0
                    queue["events"].append(
                        {
                            "utc": _utc_now(),
                            "event": "runner_finished_with_isolated_failures",
                            "failed_jobs": [
                                item["id"] for item in queue["jobs"]
                                if item.get("status") == "failed"
                            ],
                        }
                    )
                    save_queue(queue_path, queue)
                    return 1
                snapshot = _resource_sample(queue, stage="before_next_job")
                queue["last_resource_precheck"] = snapshot
                queue.setdefault("resource_prechecks", []).append(snapshot)
                queue["resource_prechecks"] = queue["resource_prechecks"][-100:]
                if not snapshot.get("passed"):
                    queue["resource_block"] = snapshot
                    queue["events"].append(
                        {
                            "utc": _utc_now(),
                            "event": "resource_precheck_deferred_without_launch",
                            "job": job["id"],
                            "sample": snapshot,
                        }
                    )
                    save_queue(queue_path, queue)
                    return 4
                try:
                    phase = _next_phase(job)
                except Exception as exc:
                    _record_job_failure(
                        queue,
                        job,
                        phase="artifact_validation",
                        reason=(
                            "pre-launch artifact validation failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        exit_code=None,
                    )
                    save_queue(queue_path, queue)
                    continue
                if phase is None:
                    save_queue(queue_path, queue)
                    continue
                try:
                    active = _start_child(job, phase, Path(queue["output_root"]))
                    active["record"]["resource_precheck"] = snapshot
                except Exception as exc:
                    reason = f"{phase} launch failed: {type(exc).__name__}: {exc}"
                    _record_job_failure(
                        queue, job, phase=phase, reason=reason, exit_code=None
                    )
                    save_queue(queue_path, queue)
                    continue
                save_queue(queue_path, queue)
            time.sleep(
                float(queue["config"]["queue"]["resource_poll_interval_seconds"])
            )
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _validate_loaded_queue(
    queue: dict[str, Any], config: Mapping[str, Any], path: Path
) -> None:
    # Rebuild the complete dry-run identity from the authenticated config and
    # current source.  Runtime state (status/attempts/events/resource samples)
    # is intentionally mutable, but every launch-defining field must remain
    # byte-for-byte equivalent to this canonical queue.  In particular, a
    # queue file is not allowed to smuggle in a different command, budget, or
    # output path after the reviewed dry run was created.
    canonical = build_queue(config, Path(config["_output_root"]))
    mutable_queue_fields = {
        "created_utc", "updated_utc", "revision", "status", "dry_run",
        "counts", "jobs", "events",
    }
    for field, expected in canonical.items():
        if field in mutable_queue_fields:
            continue
        if queue.get(field) != expected:
            raise ValueError(
                f"existing v4 queue immutable field differs from canonical "
                f"dry run: {field}"
            )

    actual_jobs = queue.get("jobs")
    if not isinstance(actual_jobs, list) or len(actual_jobs) != len(canonical["jobs"]):
        raise ValueError("existing v4 queue job list differs from canonical dry run")
    mutable_job_fields = {
        "status", "training_status", "evaluation_status", "attempts",
        "failure_history",
    }
    for index, (job, expected_job) in enumerate(
        zip(actual_jobs, canonical["jobs"], strict=True)
    ):
        if not isinstance(job, dict):
            raise ValueError(f"existing v4 queue job {index} is not a mapping")
        for field, expected in expected_job.items():
            if field in mutable_job_fields:
                continue
            if job.get(field) != expected:
                identifier = expected_job["id"]
                raise ValueError(
                    f"existing v4 queue immutable job field differs from "
                    f"canonical dry run: {identifier}.{field}"
                )

    observed_pairs = [
        (job.get("controller"), job.get("seed"), job.get("task"))
        for job in actual_jobs
    ]
    if (
        queue.get("schema_version") != QUEUE_SCHEMA_VERSION
        or queue.get("kind") != QUEUE_KIND
        or queue.get("config_file_sha256") != config["_config_sha256"]
        or queue.get("config_identity_sha256")
        != canonical_sha256(_public_config(config))
        or queue.get("queue_runner_sha256")
        != sha256_file(Path(__file__).resolve())
        or queue.get("v1_helper_sha256") != sha256_file(V1_RUNNER)
        or queue.get("v2_supervisor_helper_sha256") != sha256_file(V2_RUNNER)
        or queue.get("v3_validator_helper_sha256") != sha256_file(V3_RUNNER)
        or queue.get("trainer_sha256") != sha256_file(TRAINER)
        or queue.get("evaluator_sha256") != sha256_file(EVALUATOR)
        or queue.get("historical_v2") != config["historical_v2_identity"]
        or queue.get("preserved_v3") != config["preserved_v3_identity"]
        or queue.get("historical_v2_fair_evidence_eligible") is not False
        or Path(queue.get("queue_file", "")).resolve() != path.resolve()
        or queue.get("job_count") != JOB_COUNT
        or queue.get("predicted_training_interactions")
        != TOTAL_MATRIX_INTERACTIONS
        or queue.get("predicted_evaluation_episodes")
        != TOTAL_EVALUATION_EPISODES
        or queue.get("controller_order") != list(CONTROLLERS)
        or queue.get("seed_order") != list(SEEDS)
        or queue.get("task_order") != list(TASKS)
        or queue.get("live_command_training_contract_sha256")
        != CONTRACT_SHA_BY_TASK
        or queue.get("maximum_parallel") != 1
        or queue.get("failure_isolation") is not True
        or queue.get("retry_failed_or_paused_only_with_resume") is not True
        or observed_pairs != _expected_pairs()
        or any(
            job.get("architecture_class") != "lif"
            for job in queue.get("jobs", [])[:48]
        )
        or any(
            job.get("architecture_class") != "baseline"
            for job in queue.get("jobs", [])[48:]
        )
    ):
        raise ValueError("existing v4 queue differs from reviewed config/source/order")
    reports = queue.get("controller_reports")
    if not isinstance(reports, dict) or set(reports) != set(CONTROLLERS):
        raise ValueError("existing v4 queue controller reports are incomplete")
    for job in queue["jobs"]:
        fingerprint, payload, evaluation = _job_fingerprint(
            config, job["task"], job["policy"], job["seed"]
        )
        report = reports.get(job["controller"])
        if (
            job.get("policy") != POLICY_BY_CONTROLLER[job["controller"]]
            or job.get("command_schedule_seed") != job.get("seed")
            or job.get("paired_task_seed_key")
            != f"{job['controller']}__seed-{job['seed']}"
            or job.get("expected_fingerprint") != fingerprint
            or job.get("fingerprint_payload") != payload
            or job.get("evaluation_manifest") != evaluation
            or job.get("evaluation_manifest_id") != evaluation.get("manifest_id")
            or job.get("command_training_contract_sha256")
            != CONTRACT_SHA_BY_TASK[job["task"]]
            or job.get("controller_components")
            != list(COMPONENTS_BY_CONTROLLER[job["controller"]])
            or not _valid_controller_report(job, report)
        ):
            raise ValueError(f"existing v4 queue fingerprint is stale: {job.get('id')}")


def _dry_run_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "verified_v4_dry_run_no_processes_launched",
        "queue": queue["queue_file"],
        "job_count": JOB_COUNT,
        "training_interactions_per_job": TOTAL_INTERACTIONS,
        "predicted_training_interactions": TOTAL_MATRIX_INTERACTIONS,
        "predicted_evaluation_episodes": TOTAL_EVALUATION_EPISODES,
        "maximum_parallel": 1,
        "failure_isolation": True,
        "controller_order": list(CONTROLLERS),
        "seed_order": list(SEEDS),
        "task_order": list(TASKS),
        "task_protocols": dict(PROTOCOL_BY_TASK),
        "cells": [
            {
                "id": job["id"],
                "controller": job["controller"],
                "policy": job["policy"],
                "seed": job["seed"],
                "task": job["task"],
                "protocol": job["evaluation_protocol"],
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
        "training_interactions": TOTAL_MATRIX_INTERACTIONS,
        "evaluation_episodes": TOTAL_EVALUATION_EPISODES,
        "maximum_parallel": 1,
        "failure_isolation": True,
        "last_resource_precheck": queue.get("last_resource_precheck"),
        "last_resource_sample": queue.get("last_resource_sample"),
        "resource_block": queue.get("resource_block"),
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
            "v4 dry-run destination contains queue outputs and will not be "
            "overwritten: " + ", ".join(existing)
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
            raise ValueError(f"verified v4 dry-run queue does not exist: {queue_path}")
        queue = _read_json(queue_path)
        _validate_loaded_queue(queue, config, queue_path)
        if args.status:
            print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
            return 0
        if args.pause:
            _request_pause(queue, queue_path, "explicit_cli_request")
            print(
                json.dumps(
                    {"status": "pause_requested", "queue": str(queue_path)},
                    indent=2,
                )
            )
            return 0
        with v1_queue.queue_lock(queue_path):
            result = execute_queue(queue, queue_path, resume=args.resume)
        print(json.dumps(_status_payload(queue), indent=2, sort_keys=True))
        return result
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
