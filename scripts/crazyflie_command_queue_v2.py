#!/usr/bin/env python3
"""Resumable sequential queue for the additive 14-cell command matrix v2.

The v2 queue is intentionally separate from the completed six-cell v1 queue.
It compares still-air and wind CommandFollow tasks at one million interactions
for seven controllers.  ``--dry_run`` atomically materializes the reviewed
queue without launching Isaac; only ``--execute`` starts a child.  Concurrency
is fixed at one because the prior v1 two-process run observed sustained paging.
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
    / "crazyflie_command_optic_wind_seed0_1m.json"
)
ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
TASKS = (
    "FlyCrazyflie-CommandFollowWide-v0",
    "FlyCrazyflie-CommandFollowWideWind-v0",
)
PROTOCOL_BY_TASK = {
    TASKS[0]: "command_v2",
    TASKS[1]: "command_v2",
}
CONTROLLERS = (
    "original_lif",
    "rewired_lif",
    "wing_lif",
    "leg_wing_lif",
    "optic_lif",
    "gru_matched",
    "mlp_normal",
)
POLICY_BY_CONTROLLER = {
    "original_lif": "frozen_lif_original",
    "rewired_lif": "frozen_lif_degree_rewired",
    "wing_lif": "wing_lif",
    "leg_wing_lif": "leg_wing_lif",
    "optic_lif": "optic_lif",
    "gru_matched": "gru_matched",
    "mlp_normal": "mlp_normal",
}
LIF_CONTROLLERS = CONTROLLERS[:5]
CONTRACT_SHA_BY_TASK = {
    TASKS[0]: COMMAND_WIDE_STILL_CONTRACT_SHA256,
    TASKS[1]: COMMAND_WIDE_WIND_CONTRACT_SHA256,
}
TOTAL_INTERACTIONS = 1_000_000
EPISODES_PER_JOB = 16
STEPS_PER_EPISODE = 600
JOB_COUNT = 14
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
QUEUE_KIND = "crazyflie_command_big_matrix_queue_v2"
QUEUE_SCHEMA_VERSION = 2
QUEUE_FILE_NAME = "queue.json"
EVALUATOR = (SCRIPT_DIR / "crazyflie_command_evaluate.py").resolve()
TRAINER = (SCRIPT_DIR / "drone_train.py").resolve()
V1_RUNNER = (SCRIPT_DIR / "crazyflie_command_queue.py").resolve()
PRIOR_QUEUE = (
    ROOT / "runs" / "crazyflie_command_seed0_500k" / "queue.json"
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
        "comparison_contract", "prior_parallel_paging_evidence",
    }


def _validate_prior_paging_evidence(declaration: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "queue": "runs/crazyflie_command_seed0_500k/queue.json",
        "queue_sha256": "648ad8b8476a1b4873e23eedacefeb803e3d7887d73d171e3e1512245ef5b99c",
        "required_event": "hard_resource_gate",
        "required_sustained_paging": True,
        "disposition": "v2_max_parallel_fixed_to_one",
    }
    if declaration != expected:
        raise ValueError("prior_parallel_paging_evidence differs from reviewed v1 evidence")
    if not PRIOR_QUEUE.is_file() or sha256_file(PRIOR_QUEUE) != expected["queue_sha256"]:
        raise ValueError("completed v1 queue evidence is missing or changed")
    prior = _read_json(PRIOR_QUEUE)
    matches = [
        event for event in prior.get("events", [])
        if isinstance(event, dict)
        and event.get("event") == "hard_resource_gate"
        and isinstance(event.get("sample"), dict)
        and event["sample"].get("sustained_paging") is True
    ]
    if len(matches) != 1:
        raise ValueError("v1 queue does not contain the unique paging event required by v2")
    return matches[0]


def _parser_choices(parser: argparse.ArgumentParser, option: str) -> set[str]:
    for action in parser._actions:
        if option in action.option_strings:
            return set(action.choices or ())
    return set()


def validate_runtime_interfaces() -> dict[str, Any]:
    """Fail before dry-run if trainer/evaluator/controller v2 APIs are absent."""

    missing: list[str] = []
    trainer_policies = set(getattr(drone_train, "POLICIES", ()))
    trainer_tasks = set(getattr(drone_train, "TRAINING_TASKS", ()))
    trainer_protocols = set(getattr(drone_train, "EVALUATION_PROTOCOLS", ()))
    required_policies = set(POLICY_BY_CONTROLLER.values())
    if not required_policies.issubset(trainer_policies):
        missing.append(f"trainer policies {sorted(required_policies - trainer_policies)}")
    if not set(TASKS).issubset(trainer_tasks):
        missing.append(f"trainer tasks {sorted(set(TASKS) - trainer_tasks)}")
    if not set(PROTOCOL_BY_TASK.values()).issubset(trainer_protocols):
        missing.append(
            f"trainer protocols {sorted(set(PROTOCOL_BY_TASK.values()) - trainer_protocols)}"
        )
    if "optic_connectome_manifest" not in inspect.signature(
        drone_train.standalone_resolved_config
    ).parameters and "optic_lif" in trainer_policies:
        # The resolved-config function receives a Namespace, so the actual
        # proof is the trainer parser/source plus the controller builder below.
        if "--optic_connectome_manifest" not in TRAINER.read_text(encoding="utf-8"):
            missing.append("trainer --optic_connectome_manifest")
    if "optic_connectome_manifest" not in inspect.signature(build_controller).parameters:
        missing.append("build_controller optic_connectome_manifest")

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
        raise ValueError("v2 runtime interfaces are incomplete: " + "; ".join(missing))
    return {
        "trainer_policies": sorted(required_policies),
        "trainer_tasks": list(TASKS),
        "evaluation_protocols": dict(PROTOCOL_BY_TASK),
        "evaluator_explicit_task_argument": True,
        "optic_manifest_argument": "--optic_connectome_manifest",
    }


def validate_config(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = _read_json(path)
    if set(config) != _expected_config_keys():
        raise ValueError("v2 config top-level fields differ from the closed schema")
    exact = {
        "schema_version": 2,
        "kind": "crazyflie_command_optic_wind_matrix_v2",
        "label": "crazyflie_command_optic_wind_seed0_1m",
        "output_root": "runs/crazyflie_command_optic_wind_seed0_1m",
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
    if TOTAL_INTERACTIONS % (expected_training["num_envs"] * expected_training["horizon"]):
        raise ValueError("v2 interaction budget does not divide into whole PPO updates")
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
        raise ValueError("command_envelope differs from expanded v2 envelope")
    command_contract = command_wide_contract_payload()
    if (
        command_contract.get("maximum_horizontal_speed_m_s") != 1.0
        or command_contract.get("maximum_vertical_speed_m_s") != 0.5
        or command_contract.get("maximum_yaw_rate_rad_s") != 1.5
    ):
        raise ValueError("live command contract has not adopted the expanded v2 envelope")
    for task in TASKS:
        wind_enabled = task == TASKS[1]
        complete = command_wide_training_contract_payload(wind_enabled=wind_enabled)
        if (
            command_wide_training_contract_sha256(wind_enabled=wind_enabled)
            != CONTRACT_SHA_BY_TASK[task]
            or canonical_sha256(complete) != CONTRACT_SHA_BY_TASK[task]
        ):
            raise ValueError(f"live wide command contract hash is inconsistent for {task}")
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
        raise ValueError("evaluation differs from the reviewed v2 held-out contract")
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
        raise ValueError("queue differs from strict sequential v2 policy")
    expected_comparison = {
        "same_task_stream_budget_seed_and_ppo_hyperparameters": True,
        "paired_command_seed_across_still_and_wind": True,
        "lif_cells_before_gru_and_mlp": True,
        "parameter_matched_controllers": [
            "original_lif", "rewired_lif", "wing_lif", "optic_lif",
            "gru_matched", "mlp_normal",
        ],
        "leg_wing_parameter_matching_required": False,
        "leg_wing_fusion": "independent_leg_and_wing_cores_concat_motor_readouts_v1",
        "optic_architecture": "optic_lobe_lif",
        "optic_parameter_matching_required": True,
    }
    if config.get("comparison_contract") != expected_comparison:
        raise ValueError("comparison contract differs from reviewed v2 declaration")
    _validate_prior_paging_evidence(config.get("prior_parallel_paging_evidence", {}))

    connectomes = config.get("connectomes")
    if not isinstance(connectomes, dict) or set(connectomes) != {
        "leg_manifest", "leg_manifest_sha256", "wing_manifest",
        "wing_manifest_sha256", "optic_manifest", "optic_manifest_sha256",
    }:
        raise ValueError("connectomes differs from v2 schema")
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
        raise ValueError("rewire identity differs from reviewed v2 declaration")
    rewire_path = (ROOT / rewire["manifest"]).resolve()
    if not rewire_path.is_file() or sha256_file(rewire_path) != rewire["manifest_sha256"]:
        raise ValueError("Pinned rewire manifest is missing or changed")
    if not ISAAC_PYTHON.is_file() or not os.access(ISAAC_PYTHON, os.X_OK):
        raise ValueError(f"Isaac Python is not executable: {ISAAC_PYTHON}")
    for script in (TRAINER, EVALUATOR, V1_RUNNER):
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
    config["_prior_paging_event"] = _validate_prior_paging_evidence(
        config["prior_parallel_paging_evidence"]
    )
    return config


def _training_args(
    config: Mapping[str, Any], task: str, policy: str
) -> SimpleNamespace:
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
    selected_manifest = (
        args.optic_connectome_manifest if policy == "optic_lif"
        else args.wing_connectome_manifest if policy == "wing_lif"
        else args.connectome_manifest
    )
    fingerprint, payload = reproduction_fingerprint(
        resolved_config=resolved, evaluation_manifest=evaluation,
        connectome_manifest=selected_manifest, rewired_manifest=rewired,
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
        policy = POLICY_BY_CONTROLLER[label]
        _, report = build_controller(
            policy, observation_dim=12, action_dim=4, device="cpu",
            connectome_manifest=config["_leg_manifest"],
            wing_connectome_manifest=config["_wing_manifest"],
            optic_connectome_manifest=config["_optic_manifest"],
            rewire_seed=config["rewire"]["seed"],
            rewire_manifest_path=(
                config["_rewire_manifest"]
                if policy == "frozen_lif_degree_rewired" else None
            ),
        )
        reports[label] = report
    return reports


def build_queue(config: Mapping[str, Any], output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root or config["_output_root"]).resolve()
    evaluator_hash = sha256_file(EVALUATOR)
    trainer_hash = sha256_file(TRAINER)
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
            jobs.append({
                "id": identifier, "controller": label, "policy": policy,
                "controller_priority": controller_priority,
                "task_priority": task_priority,
                "architecture_class": (
                    "lif" if label in LIF_CONTROLLERS else "baseline"
                ),
                "task": task, "evaluation_protocol": PROTOCOL_BY_TASK[task],
                "seed": 0, "command_schedule_seed": 0,
                "paired_task_seed_key": f"{label}__seed-0",
                "contract_profile": "command_v2",
                "status": "pending", "training_status": "pending",
                "evaluation_status": "pending",
                "total_interactions": TOTAL_INTERACTIONS,
                "expected_updates": 250,
                "run_dir": str(run_dir), "checkpoint": str(checkpoint),
                "training_manifest": str(run_dir / "training_manifest.json"),
                "pause_file": str(pause_file),
                "evaluation_output": str(evaluation_output),
                "expected_fingerprint": fingerprint,
                "fingerprint_payload": payload,
                "evaluation_manifest": evaluation_manifest,
                "evaluation_manifest_id": evaluation_manifest["manifest_id"],
                "command_training_contract_sha256": CONTRACT_SHA_BY_TASK[task],
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
        or any(job["architecture_class"] != "lif" for job in jobs[:10])
        or any(job["architecture_class"] != "baseline" for job in jobs[10:])
    ):
        raise RuntimeError("v2 queue did not resolve exact LIF-first 14-cell order")
    return {
        "schema_version": QUEUE_SCHEMA_VERSION, "kind": QUEUE_KIND,
        "label": config["label"], "created_utc": _utc_now(),
        "updated_utc": _utc_now(), "revision": 0,
        "status": "dry_run", "dry_run": True,
        "config_path": config["_config_path"],
        "config_file_sha256": config["_config_sha256"],
        "config_identity_sha256": canonical_sha256(_public_config(config)),
        "queue_runner": str(Path(__file__).resolve()),
        "queue_runner_sha256": sha256_file(Path(__file__).resolve()),
        "v1_helper": str(V1_RUNNER), "v1_helper_sha256": sha256_file(V1_RUNNER),
        "trainer": str(TRAINER), "trainer_sha256": trainer_hash,
        "evaluator": str(EVALUATOR), "evaluator_sha256": evaluator_hash,
        "output_root": str(root), "queue_file": str(root / QUEUE_FILE_NAME),
        "global_pause_file": str(root / "pause.request"),
        "job_count": JOB_COUNT,
        "predicted_training_interactions": JOB_COUNT * TOTAL_INTERACTIONS,
        "predicted_evaluation_episodes": JOB_COUNT * EPISODES_PER_JOB,
        "controller_order": list(CONTROLLERS), "task_order": list(TASKS),
        "task_protocols": dict(PROTOCOL_BY_TASK), "lif_cell_count": 10,
        "lif_first": True, "maximum_parallel": 1,
        "parallelism_disposition": config["prior_parallel_paging_evidence"],
        "resource_limits": {
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "sustained_paging_sample_count": 3,
            "default_max_parallel": 1, "maximum_parallel": 1,
        },
        "controller_reports": _controller_reports(config),
        "counts": {"pending": JOB_COUNT}, "config": _public_config(config),
        "interface_contract": config["_interface"], "jobs": jobs,
        "events": [{"utc": _utc_now(), "event": "verified_v2_dry_run_created"}],
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
        "schema_version": 2, "kind": QUEUE_KIND, "status": queue["status"],
        "dry_run": queue["dry_run"], "revision": queue["revision"],
        "updated_utc": queue["updated_utc"], "counts": queue["counts"],
        "job_count": JOB_COUNT,
        "predicted_training_interactions": 14_000_000,
        "predicted_evaluation_episodes": 224,
        "maximum_parallel": 1, "resource_block": queue.get("resource_block"),
        "jobs": [
            {key: job[key] for key in (
                "id", "controller", "policy", "task", "status",
                "training_status", "evaluation_status", "checkpoint",
            )}
            for job in queue["jobs"]
        ],
    })


def _memory_gate_passed(value: Any) -> bool:
    return v1_queue._memory_gate_passed(value)


def valid_training(job: Mapping[str, Any]) -> bool:
    try:
        manifest = _read_json(Path(job["training_manifest"]))
        checkpoint = Path(job["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
        schedule = manifest["command_schedule"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "completed"
        and manifest.get("task") == job["task"]
        and manifest.get("contract_profile") == "command_v2"
        and manifest.get("controller") == job["policy"]
        and manifest.get("seed") == 0 and manifest.get("num_envs") == 40
        and manifest.get("horizon") == 100
        and manifest.get("requested_interactions") == TOTAL_INTERACTIONS
        and manifest.get("environment_interactions") == TOTAL_INTERACTIONS
        and manifest.get("completed_updates") == job["expected_updates"] == 250
        and manifest.get("fingerprint") == job["expected_fingerprint"]
        and manifest.get("fingerprint_payload") == job["fingerprint_payload"]
        and manifest.get("evaluation_manifest_id") == job["evaluation_manifest_id"]
        and Path(manifest.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and manifest.get("checkpoint_sha256") == checkpoint_hash
        and manifest.get("core_checksum_before") == manifest.get("core_checksum_after")
        and isinstance(schedule, dict)
        and schedule.get("command_training_contract_sha256")
        == job["command_training_contract_sha256"]
        and isinstance(schedule.get("state"), dict)
        and schedule["state"].get("training_interactions") == TOTAL_INTERACTIONS
        and _memory_gate_passed(manifest.get("memory_gate"))
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
        and value.get("status") == "PASS" and value.get("task") == job["task"]
        and value.get("controller") == job["policy"]
        and isinstance(checkpoint_record, dict)
        and Path(checkpoint_record.get("path", "")).resolve() == checkpoint.resolve()
        and checkpoint_record.get("sha256") == checkpoint_hash
        and checkpoint_record.get("training_seed") == 0
        and checkpoint_record.get("total_interactions") == TOTAL_INTERACTIONS
        and checkpoint_record.get("reproduction_fingerprint") == job["expected_fingerprint"]
        and checkpoint_record.get("evaluation_manifest_id") == job["evaluation_manifest_id"]
        and checkpoint_record.get("evaluation_manifest") == job["evaluation_manifest"]
        and value.get("protocol") == job["evaluation_manifest"].get("evaluation_protocol")
        and value.get("protocol_sha256")
        == job["evaluation_manifest"].get("evaluation_protocol_sha256")
        and value.get("episodes_requested") == EPISODES_PER_JOB
        and value.get("episodes_evaluated") == EPISODES_PER_JOB
        and value.get("steps_per_episode") == STEPS_PER_EPISODE
        and value.get("vectorized_environment_count") == EPISODES_PER_JOB
        and value.get("deterministic_actions") is True
        and value.get("policy_action_source") == "actual_trained_controller_no_assist"
        and isinstance(value.get("episodes"), list)
        and len(value["episodes"]) == EPISODES_PER_JOB
        and all(isinstance(row, dict) and row for row in value["episodes"])
        and isinstance(summary, dict) and isinstance(summary.get("score"), dict)
        and set(summary.get("control_quality", {}).get("component_scores_0_100", {}))
        == quality_keys
        and isinstance(value.get("activity"), dict)
        and _memory_gate_passed(value.get("memory_gate"))
        and all(integrity.get(key) is True for key in (
            "task_manifest_matched", "evaluation_manifest_matched",
            "source_set_matched", "checkpoint_completed_budget",
            "all_actions_finite_and_bounded", "activity_from_actual_forward",
        ))
        and v1_queue._finite_tree(value)
    )


def _process_alive(job: Mapping[str, Any]) -> bool:
    active = job.get("active_process")
    if not isinstance(active, dict) or not v1_queue._pid_alive(active.get("pid")):
        return False
    try:
        command = Path(f"/proc/{active['pid']}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode(errors="replace")
    except OSError:
        return False
    expected = "drone_train.py" if active.get("phase") == "training" else "crazyflie_command_evaluate.py"
    target = job["run_dir"] if active.get("phase") == "training" else job["evaluation_output"]
    return expected in command and target in command


def reconcile_queue(queue: dict[str, Any], *, retry_failed: bool) -> None:
    for job in queue["jobs"]:
        if valid_training(job):
            job["training_status"] = "completed"
            if valid_evaluation(job):
                job["evaluation_status"] = "completed"
                job["status"] = "completed"
                job.pop("active_process", None)
                continue
            job["evaluation_status"] = "pending"
        else:
            job["training_status"] = "pending"
            job["evaluation_status"] = "pending"
        if _process_alive(job):
            job["status"] = "running"
            job["recovered_detached_process"] = True
        else:
            job.pop("active_process", None)
            if retry_failed or job.get("status") not in {"failed", "paused"}:
                job["status"] = "pending"
                job.pop("failure", None)


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


def _start_child(job: dict[str, Any], phase: str, output_root: Path) -> dict[str, Any]:
    command = list(job[f"{phase}_command"])
    if phase == "training" and Path(job["checkpoint"]).is_file():
        command.append("--resume")
    archived = _archive_invalid_evaluation(job) if phase == "evaluation" else None
    attempt = len(job["attempts"]) + 1
    log = output_root / "logs" / f"{job['id']}__{phase}__attempt-{attempt:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("xb")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    record = {
        "attempt": attempt, "phase": phase, "pid": process.pid,
        "started_utc": _utc_now(), "command": command, "log": str(log),
        "archived_invalid_output": archived,
    }
    job["attempts"].append(record)
    job["active_process"] = {
        "pid": process.pid, "phase": phase,
        "started_utc": record["started_utc"], "log": str(log),
    }
    job["status"] = "running"
    job[f"{phase}_status"] = "running"
    return {"process": process, "stream": stream, "phase": phase, "job": job, "record": record}


def _finish_child(handle: dict[str, Any], exit_code: int | None) -> None:
    handle["stream"].close() if handle.get("stream") is not None else None
    job, phase, record = handle["job"], handle["phase"], handle["record"]
    record.update(finished_utc=_utc_now(), exit_code=exit_code)
    job.pop("active_process", None)
    valid = valid_training(job) if phase == "training" else valid_evaluation(job)
    if valid:
        job[f"{phase}_status"] = "completed"
        job["status"] = "completed" if phase == "evaluation" else "pending"
    elif phase == "training" and exit_code == 3:
        job["training_status"] = "paused"
        job["status"] = "paused"
    else:
        job[f"{phase}_status"] = "failed"
        job["status"] = "failed"
        job["failure"] = f"{phase} exited {exit_code!r}; artifact validation failed"


def _pause_path(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        _atomic_json(path, payload)


def _request_pause(queue: Mapping[str, Any], queue_path: Path, reason: str) -> None:
    payload = {
        "schema_version": 2, "status": "requested", "utc": _utc_now(),
        "reason": reason, "queue": str(queue_path), "pid": os.getpid(),
    }
    _pause_path(Path(queue["global_pause_file"]), payload)
    for job in queue["jobs"]:
        active = job.get("active_process")
        if isinstance(active, dict) and active.get("phase") == "training":
            _pause_path(Path(job["pause_file"]), {**payload, "job": job["id"]})


def _consume_pauses(queue: dict[str, Any]) -> None:
    paths = [Path(queue["global_pause_file"])] + [Path(job["pause_file"]) for job in queue["jobs"]]
    for path in paths:
        if not path.exists():
            continue
        archive_dir = path.parent / "pause_requests"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{path.stem}-consumed-{time.time_ns()}{path.suffix}"
        os.replace(path, archive)
        queue["events"].append({"utc": _utc_now(), "event": "pause_consumed", "archive": str(archive)})


def _next_eligible(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    """Honor exact order; a failed/paused earlier cell blocks later cells."""

    for job in queue["jobs"]:
        if job["status"] == "completed":
            continue
        return job if job["status"] == "pending" else None
    return None


def execute_queue(
    queue: dict[str, Any], queue_path: Path, *, resume: bool,
) -> int:
    if queue.get("maximum_parallel") != 1:
        raise ValueError("v2 execution is permanently sequential after prior paging")
    if resume:
        _consume_pauses(queue)
    elif Path(queue["global_pause_file"]).exists() or any(
        Path(job["pause_file"]).exists() for job in queue["jobs"]
    ):
        raise RuntimeError("Pause request exists; use --execute --resume")
    reconcile_queue(queue, retry_failed=resume)
    queue["dry_run"] = False
    queue.pop("resource_block", None)
    queue.setdefault("started_utc", _utc_now())
    queue["events"].append({
        "utc": _utc_now(), "event": "sequential_runner_started",
        "pid": os.getpid(), "resume": resume, "max_parallel": 1,
    })
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
            active = {
                "process": None, "stream": None,
                "phase": job["active_process"]["phase"], "job": job,
                "record": {"recovered_by_pid_monitor": True, **job["active_process"]},
            }
            break
    swap_samples: list[int] = []
    try:
        while True:
            if active is not None:
                process = active["process"]
                exit_code = None if process is None else process.poll()
                alive = _process_alive(active["job"]) if process is None else exit_code is None
                if not alive:
                    _finish_child(active, exit_code)
                    active = None
                    save_queue(queue_path, queue)
                else:
                    try:
                        sample = v1_queue.resource_snapshot()
                        swap_samples.append(int(sample["swap_out_pages"]))
                        swap_samples = swap_samples[-3:]
                        sustained = len(swap_samples) == 3 and swap_samples[0] < swap_samples[1] < swap_samples[2]
                        sample["sustained_paging"] = sustained
                        sample["passed"] = bool(sample["passed"] and not sustained)
                    except Exception as exc:
                        sample = {"utc": _utc_now(), "passed": False, "telemetry_error": f"{type(exc).__name__}: {exc}"}
                    queue["last_resource_sample"] = sample
                    queue.setdefault("resource_samples", []).append(sample)
                    queue["resource_samples"] = queue["resource_samples"][-200:]
                    if not sample["passed"]:
                        stop = True
                        resource_stop = True
                        queue["resource_block"] = sample
                        queue["events"].append({"utc": _utc_now(), "event": "hard_resource_gate", "sample": sample})
                        _request_pause(queue, queue_path, "hard_resource_gate")
                        save_queue(queue_path, queue)
            if stop or Path(queue["global_pause_file"]).exists():
                if active is None:
                    for job in queue["jobs"]:
                        if job["status"] == "pending":
                            job["status"] = "paused"
                    queue["events"].append({"utc": _utc_now(), "event": "runner_paused"})
                    save_queue(queue_path, queue)
                    return 4 if resource_stop else 3
            elif active is None:
                job = _next_eligible(queue)
                if job is None:
                    if all(item["status"] == "completed" for item in queue["jobs"]):
                        queue["finished_utc"] = _utc_now()
                        queue["events"].append({"utc": _utc_now(), "event": "runner_completed"})
                        save_queue(queue_path, queue)
                        return 0
                    queue["finished_utc"] = _utc_now()
                    save_queue(queue_path, queue)
                    return 1
                try:
                    snapshot = v1_queue.resource_snapshot()
                except Exception as exc:
                    snapshot = {"utc": _utc_now(), "passed": False, "telemetry_error": f"{type(exc).__name__}: {exc}"}
                queue["last_resource_precheck"] = snapshot
                queue.setdefault("resource_prechecks", []).append(snapshot)
                queue["resource_prechecks"] = queue["resource_prechecks"][-100:]
                if not snapshot.get("passed"):
                    queue["resource_block"] = snapshot
                    save_queue(queue_path, queue)
                    return 4
                phase = _next_phase(job)
                if phase is None:
                    save_queue(queue_path, queue)
                    continue
                active = _start_child(job, phase, Path(queue["output_root"]))
                active["record"]["resource_precheck"] = snapshot
                save_queue(queue_path, queue)
            time.sleep(float(queue["config"]["queue"]["resource_poll_interval_seconds"]))
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _validate_loaded_queue(
    queue: dict[str, Any], config: Mapping[str, Any], path: Path
) -> None:
    expected_pairs = [
        (controller, task) for controller in CONTROLLERS for task in TASKS
    ]
    if (
        queue.get("schema_version") != 2 or queue.get("kind") != QUEUE_KIND
        or queue.get("config_file_sha256") != config["_config_sha256"]
        or queue.get("config_identity_sha256") != canonical_sha256(_public_config(config))
        or queue.get("queue_runner_sha256") != sha256_file(Path(__file__).resolve())
        or queue.get("v1_helper_sha256") != sha256_file(V1_RUNNER)
        or queue.get("trainer_sha256") != sha256_file(TRAINER)
        or queue.get("evaluator_sha256") != sha256_file(EVALUATOR)
        or Path(queue.get("queue_file", "")).resolve() != path.resolve()
        or queue.get("job_count") != 14
        or queue.get("predicted_training_interactions") != 14_000_000
        or queue.get("predicted_evaluation_episodes") != 224
        or queue.get("maximum_parallel") != 1
        or [(job.get("controller"), job.get("task")) for job in queue.get("jobs", [])]
        != expected_pairs
    ):
        raise ValueError("Existing v2 queue differs from reviewed config/source/order")
    for job in queue["jobs"]:
        fingerprint, payload, evaluation = _job_fingerprint(
            config, job["task"], job["policy"]
        )
        if (
            job.get("expected_fingerprint") != fingerprint
            or job.get("fingerprint_payload") != payload
            or job.get("evaluation_manifest") != evaluation
            or job.get("evaluation_manifest_id") != evaluation.get("manifest_id")
            or job.get("command_training_contract_sha256")
            != CONTRACT_SHA_BY_TASK[job["task"]]
        ):
            raise ValueError(f"Existing v2 queue fingerprint is stale: {job.get('id')}")


def _dry_run_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "verified_v2_dry_run_no_processes_launched",
        "queue": queue["queue_file"], "job_count": 14,
        "training_interactions_per_job": TOTAL_INTERACTIONS,
        "predicted_training_interactions": 14_000_000,
        "predicted_evaluation_episodes": 224,
        "maximum_parallel": 1, "controller_order": list(CONTROLLERS),
        "task_order": list(TASKS), "task_protocols": dict(PROTOCOL_BY_TASK),
        "cells": [
            {
                "id": job["id"], "controller": job["controller"],
                "policy": job["policy"], "task": job["task"],
                "protocol": job["evaluation_protocol"], "seed": 0,
                "total_interactions": TOTAL_INTERACTIONS,
                "expected_updates": 250, "run_dir": job["run_dir"],
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
        "status": queue["status"], "queue": queue["queue_file"],
        "revision": queue["revision"], "counts": queue["counts"],
        "job_count": 14, "training_interactions": 14_000_000,
        "evaluation_episodes": 224, "maximum_parallel": 1,
        "last_resource_precheck": queue.get("last_resource_precheck"),
        "last_resource_sample": queue.get("last_resource_sample"),
    }


def _assert_fresh_dry_run_destination(output_root: Path) -> None:
    """Permit unrelated prelaunch evidence, but never adopt/replace queue outputs."""

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
            "v2 dry-run destination contains queue outputs and will not be overwritten: "
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
            raise ValueError(f"Verified v2 dry-run queue does not exist: {queue_path}")
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
