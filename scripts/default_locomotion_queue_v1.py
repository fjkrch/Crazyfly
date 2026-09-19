#!/usr/bin/env python3
"""Build the reviewed stock Isaac Lab G1/Go1 locomotion queues.

This queue is intentionally dry-run only.  It records the exact native RSL-RL
commands, but it cannot launch them until the four one-update GPU memory smokes
have been measured and the user has reviewed the result.  Existing custom G1
and Crazyflie sources and every file below ``runs/`` remain out of scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1.json"
)
ISAAC_PYTHON = Path(
    "/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
)
ISAACLAB_ROOT = Path("/home/chayanin/Downloads/IsaacLab")
TRAINER = (
    ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
)
ISAACLAB_COMMIT = "b4c321024792976150ca55fddb26fa34480d974e"
OUTPUT_ROOT_RELATIVE = Path(
    "stock_isaaclab_runs/default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1"
)
SEEDS = (0, 1, 2)
NUM_ENVS = 1024
STEPS_PER_ENV = 24
GPU_LIMIT_MIB = 6963.2
RAM_LIMIT_PERCENT = 90.0
QUEUE_SCHEMA_VERSION = 1
CONTROLLER = "stock_rsl_rl_default_mlp"


TASK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "robot": "g1",
        "output_subdirectory": "g1",
        "terrain": "flat",
        "task_id": "Isaac-Velocity-Flat-G1-v0",
        "default_experiment_name": "g1_flat",
        "default_max_iterations": 1500,
        "default_num_steps_per_env": 24,
        "default_save_interval": 50,
        "observation_width": 123,
        "action_width": 37,
        "actor_hidden_dims": [256, 128, 128],
        "critic_hidden_dims": [256, 128, 128],
        "actor_mlp_parameter_count": 85_925,
        "actor_distribution_parameter_count": 37,
        "actor_trainable_parameter_count": 85_962,
        "critic_parameter_count": 81_281,
        "command_ranges": {
            "lin_vel_x_mps": [0.0, 1.0],
            "lin_vel_y_mps": [-0.5, 0.5],
            "ang_vel_z_radps": [-1.0, 1.0],
        },
    },
    {
        "robot": "g1",
        "output_subdirectory": "g1",
        "terrain": "rough",
        "task_id": "Isaac-Velocity-Rough-G1-v0",
        "default_experiment_name": "g1_rough",
        "default_max_iterations": 3000,
        "default_num_steps_per_env": 24,
        "default_save_interval": 50,
        "observation_width": 310,
        "action_width": 37,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "actor_mlp_parameter_count": 328_229,
        "actor_distribution_parameter_count": 37,
        "actor_trainable_parameter_count": 328_266,
        "critic_parameter_count": 323_585,
        "command_ranges": {
            "lin_vel_x_mps": [0.0, 1.0],
            "lin_vel_y_mps": [0.0, 0.0],
            "ang_vel_z_radps": [-1.0, 1.0],
        },
    },
    {
        "robot": "go1",
        "output_subdirectory": "go1",
        "terrain": "flat",
        "task_id": "Isaac-Velocity-Flat-Unitree-Go1-v0",
        "default_experiment_name": "unitree_go1_flat",
        "default_max_iterations": 300,
        "default_num_steps_per_env": 24,
        "default_save_interval": 50,
        "observation_width": 48,
        "action_width": 12,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "actor_mlp_parameter_count": 40_844,
        "actor_distribution_parameter_count": 12,
        "actor_trainable_parameter_count": 40_856,
        "critic_parameter_count": 39_425,
        "command_ranges": {
            "lin_vel_x_mps": [-1.0, 1.0],
            "lin_vel_y_mps": [-1.0, 1.0],
            "ang_vel_z_radps": [-1.0, 1.0],
        },
    },
    {
        "robot": "go1",
        "output_subdirectory": "go1",
        "terrain": "rough",
        "task_id": "Isaac-Velocity-Rough-Unitree-Go1-v0",
        "default_experiment_name": "unitree_go1_rough",
        "default_max_iterations": 1500,
        "default_num_steps_per_env": 24,
        "default_save_interval": 50,
        "observation_width": 235,
        "action_width": 12,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "actor_mlp_parameter_count": 286_604,
        "actor_distribution_parameter_count": 12,
        "actor_trainable_parameter_count": 286_616,
        "critic_parameter_count": 285_185,
        "command_ranges": {
            "lin_vel_x_mps": [-1.0, 1.0],
            "lin_vel_y_mps": [-1.0, 1.0],
            "ang_vel_z_radps": [-1.0, 1.0],
        },
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TASK_SPECS:
        row = {
            key: value
            for key, value in spec.items()
            if key not in {"robot", "output_subdirectory"}
        }
        rows.append(row)
    return rows


def _expected_robot_config() -> list[dict[str, Any]]:
    rows = _config_task_rows()
    return [
        {
            "id": "g1",
            "output_subdirectory": "g1",
            "tasks": rows[:2],
        },
        {
            "id": "go1",
            "output_subdirectory": "go1",
            "tasks": rows[2:],
        },
    ]


def _verify_isaac_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source_identity"]
    files = source["files"]
    if canonical_sha256(files) != source["file_set_sha256"]:
        raise ValueError("source_identity file-set digest is inconsistent")
    mismatches: list[str] = []
    for raw_path, expected in files.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(raw_path)
    if mismatches:
        raise ValueError("installed Isaac source changed: " + ", ".join(mismatches))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ISAACLAB_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify Isaac Lab commit: {exc}") from exc
    if commit != ISAACLAB_COMMIT or commit != config["isaaclab_commit"]:
        raise ValueError(f"Isaac Lab commit changed: {commit}")
    return {
        "commit": commit,
        "file_count": len(files),
        "file_set_sha256": canonical_sha256(files),
        "all_match": True,
    }


def _verify_frozen_g1(config: Mapping[str, Any]) -> dict[str, Any]:
    declaration = config["preservation"]
    receipt_path = (ROOT / declaration["frozen_g1_receipt"]).resolve()
    if sha256_file(receipt_path) != declaration["frozen_g1_receipt_sha256"]:
        raise ValueError("frozen G1 receipt changed")
    receipt = _read_json(receipt_path)
    rows = receipt.get("files")
    if not isinstance(rows, list) or len(rows) != 37:
        raise ValueError("frozen G1 receipt does not contain exactly 37 files")
    mismatches: list[str] = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["before_sha256"]:
            mismatches.append(row["path"])
    if mismatches:
        raise ValueError("frozen custom G1 files changed: " + ", ".join(mismatches))
    return {"file_count": 37, "mismatched": 0, "all_match": True}


def frozen_runs_digest(root: Path) -> dict[str, Any]:
    """Reproduce the pre-existing run-tree receipt's canonical digest."""
    rows: list[tuple[str, str]] = []
    byte_count = 0
    if not root.is_dir():
        raise ValueError(f"frozen runs directory is missing: {root}")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.startswith("crazyflie_command_") for part in relative.parts):
            continue
        # The frozen receipt records paths relative to the workspace, so the
        # canonical path starts with ``runs/`` rather than with the run-tree
        # child name alone.
        canonical_relative = (Path(root.name) / relative).as_posix()
        if path.is_symlink():
            rows.append((canonical_relative, os.readlink(path)))
        elif path.is_file():
            rows.append((canonical_relative, sha256_file(path)))
            byte_count += path.stat().st_size
    digest = hashlib.sha256()
    for relative, identity in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return {
        "file_count": len(rows),
        "byte_count": byte_count,
        "tree_sha256": digest.hexdigest(),
    }


def _verify_frozen_runs(config: Mapping[str, Any]) -> dict[str, Any]:
    declaration = config["preservation"]
    receipt_path = (ROOT / declaration["frozen_runs_receipt"]).resolve()
    if sha256_file(receipt_path) != declaration["frozen_runs_receipt_sha256"]:
        raise ValueError("frozen runs receipt changed")
    observed = frozen_runs_digest(ROOT / "runs")
    expected = {
        "file_count": declaration["frozen_runs_file_count"],
        "byte_count": declaration["frozen_runs_total_bytes"],
        "tree_sha256": declaration["frozen_runs_tree_sha256"],
    }
    if observed != expected:
        raise ValueError(f"frozen runs tree changed: expected {expected}, got {observed}")
    return {**observed, "all_match": True}


def validate_config(path: Path | str, *, verify_preservation: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _read_json(config_path)
    expected_keys = {
        "schema_version",
        "kind",
        "label",
        "output_root",
        "isaac_python",
        "isaaclab_root",
        "isaaclab_commit",
        "trainer",
        "agent_entry_point",
        "controller",
        "seeds",
        "robots",
        "training",
        "command_contract",
        "queue",
        "source_identity",
        "preservation",
    }
    if set(config) != expected_keys:
        raise ValueError("config top-level fields differ from the closed v1 schema")
    exact = {
        "schema_version": 1,
        "kind": "isaaclab_stock_default_locomotion_matrix_v1",
        "label": "default_locomotion_g1_go1_flat_rough_seeds0_1_2_v1",
        "output_root": OUTPUT_ROOT_RELATIVE.as_posix(),
        "isaac_python": str(ISAAC_PYTHON),
        "isaaclab_root": str(ISAACLAB_ROOT),
        "isaaclab_commit": ISAACLAB_COMMIT,
        "trainer": str(TRAINER),
        "agent_entry_point": "rsl_rl_cfg_entry_point",
        "controller": CONTROLLER,
        "seeds": list(SEEDS),
        "robots": _expected_robot_config(),
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be exactly {expected!r}")
    expected_training = {
        "num_envs": NUM_ENVS,
        "device": "cuda:0",
        "headless": True,
        "logger": "tensorboard",
        "task_default_rewards": True,
        "task_default_observations": True,
        "task_default_actions": True,
        "task_default_ppo": True,
        "only_overrides": [
            "num_envs_for_memory_safety",
            "seed",
            "max_iterations_repeated_from_task_default",
            "device",
            "logger",
            "run_name",
        ],
        "default_task_num_envs": 4096,
        "num_envs_override_reason": (
            "common conservative pre-smoke value; lock one common value only "
            "after all four one-update memory smokes pass"
        ),
    }
    if config["training"] != expected_training:
        raise ValueError("training settings differ from the reviewed stock/default contract")
    expected_command = {
        "kind": "planar_base_velocity_only",
        "axes": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
        "vertical_command_present": False,
        "base_height_command_present": False,
        "up_down_command_present": False,
        "command_resampling_seconds": [10.0, 10.0],
        "use_train_tasks_not_play_tasks": True,
    }
    if config["command_contract"] != expected_command:
        raise ValueError("command contract must remain planar vx/vy/yaw with no z command")
    expected_queue = {
        "default_max_parallel": 1,
        "maximum_parallel": 1,
        "failure_isolation": True,
        "continue_after_job_failure": True,
        "resume_policy": "restart_incomplete_job_from_scratch_in_new_immutable_attempt",
        "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
        "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
        "resource_poll_interval_seconds": 5.0,
        "sustained_paging_sample_count": 3,
        "network_independent_after_local_assets_verified": True,
        "immutable_attempt_logs": True,
        "execution_requires_four_task_memory_smoke_receipt": True,
        "dry_run_only_until_reviewed": True,
    }
    if config["queue"] != expected_queue:
        raise ValueError("queue settings differ from the reviewed safety contract")
    output_root = (ROOT / config["output_root"]).resolve()
    safe_root = (ROOT / "stock_isaaclab_runs").resolve()
    frozen_root = (ROOT / "runs").resolve()
    if not output_root.is_relative_to(safe_root) or output_root.is_relative_to(frozen_root):
        raise ValueError("new locomotion output must be below stock_isaaclab_runs/, never runs/")
    if not ISAAC_PYTHON.is_file() or not TRAINER.is_file():
        raise ValueError("canonical Isaac interpreter or stock RSL-RL trainer is missing")
    config["_config_path"] = str(config_path)
    config["_output_root"] = str(output_root)
    config["_source_verification"] = _verify_isaac_source(config)
    if verify_preservation:
        config["_frozen_g1_verification"] = _verify_frozen_g1(config)
        config["_frozen_runs_verification"] = _verify_frozen_runs(config)
    return config


def _training_command(spec: Mapping[str, Any], seed: int) -> list[str]:
    return [
        str(ISAAC_PYTHON),
        str(TRAINER),
        "--task",
        str(spec["task_id"]),
        "--agent",
        "rsl_rl_cfg_entry_point",
        "--seed",
        str(seed),
        "--num_envs",
        str(NUM_ENVS),
        "--max_iterations",
        str(spec["default_max_iterations"]),
        "--device",
        "cuda:0",
        "--logger",
        "tensorboard",
        "--run_name",
        f"seed_{seed}",
        "--headless",
    ]


def _build_job(
    spec: Mapping[str, Any], seed: int, output_root: Path
) -> dict[str, Any]:
    robot = str(spec["robot"])
    terrain = str(spec["terrain"])
    work_dir = output_root / robot / terrain / f"seed_{seed}" / "attempt_001"
    iterations = int(spec["default_max_iterations"])
    interactions = NUM_ENVS * STEPS_PER_ENV * iterations
    return {
        "id": f"{robot}-{terrain}-seed-{seed}",
        "robot": robot,
        "terrain": terrain,
        "task": spec["task_id"],
        "seed": seed,
        "controller": CONTROLLER,
        "status": "planned",
        "attempt_count": 0,
        "attempts": [],
        "working_directory": str(work_dir),
        "native_log_root": str(
            work_dir / "logs" / "rsl_rl" / str(spec["default_experiment_name"])
        ),
        "training_command": _training_command(spec, seed),
        "num_envs": NUM_ENVS,
        "num_steps_per_env": STEPS_PER_ENV,
        "max_iterations": iterations,
        "expected_interactions": interactions,
        "expected_final_checkpoint_basename": f"model_{iterations - 1}.pt",
        "observation_width": spec["observation_width"],
        "action_width": spec["action_width"],
        "policy": {
            "class": "native_default_rsl_rl_mlp",
            "actor_hidden_dims": spec["actor_hidden_dims"],
            "critic_hidden_dims": spec["critic_hidden_dims"],
            "activation": "elu",
            "actor_mlp_parameter_count": spec["actor_mlp_parameter_count"],
            "actor_distribution_parameter_count": spec[
                "actor_distribution_parameter_count"
            ],
            "actor_trainable_parameter_count": spec[
                "actor_trainable_parameter_count"
            ],
            "critic_parameter_count": spec["critic_parameter_count"],
        },
        "command_contract": {
            "axes": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
            "ranges": spec["command_ranges"],
            "vertical_command_present": False,
            "up_down_command_present": False,
        },
        "stock_defaults": {
            "rewards": True,
            "observations": True,
            "actions": True,
            "ppo": True,
            "default_experiment_name": spec["default_experiment_name"],
            "save_interval": spec["default_save_interval"],
            "max_iterations": iterations,
        },
        "declared_overrides": [
            "num_envs:4096->1024_memory_safety",
            f"seed:{seed}",
            "device:cuda:0",
            "logger:tensorboard",
            f"run_name:seed_{seed}",
        ],
        "resume_semantics": (
            "incomplete attempts are retained and the full job restarts in a new "
            "attempt directory; native checkpoint continuation is not claimed exact"
        ),
    }


def build_queue_bundle(
    config: Mapping[str, Any], output_root: Path | str | None = None
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(output_root or config["_output_root"]).resolve()
    generated_at = _utc_now()
    jobs = [
        _build_job(spec, seed, root)
        for spec in TASK_SPECS
        for seed in SEEDS
    ]
    shards: dict[str, dict[str, Any]] = {}
    for robot in ("g1", "go1"):
        robot_jobs = [job for job in jobs if job["robot"] == robot]
        shards[robot] = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "kind": "isaaclab_stock_default_locomotion_robot_queue_v1",
            "label": config["label"],
            "robot": robot,
            "generated_at_utc": generated_at,
            "status": "verified_dry_run",
            "execution_authorized": False,
            "output_root": str(root / robot),
            "queue_file": str(root / robot / "queue.json"),
            "controller": CONTROLLER,
            "job_count": len(robot_jobs),
            "counts": {"planned": len(robot_jobs)},
            "predicted_training_interactions": sum(
                job["expected_interactions"] for job in robot_jobs
            ),
            "maximum_parallel": 1,
            "resource_limits": dict(config["queue"]),
            "jobs": robot_jobs,
        }
    index = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": "isaaclab_stock_default_locomotion_matrix_index_v1",
        "label": config["label"],
        "generated_at_utc": generated_at,
        "status": "verified_dry_run",
        "execution_authorized": False,
        "execution_blocker": (
            "review plus one-update GPU/RAM smoke receipt for every task is required"
        ),
        "config_path": config["_config_path"],
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "queue_builder": str(Path(__file__).resolve()),
        "queue_builder_sha256": sha256_file(Path(__file__).resolve()),
        "output_root": str(root),
        "controller": CONTROLLER,
        "robots": ["g1", "go1"],
        "robot_queue_files": {
            robot: shard["queue_file"] for robot, shard in shards.items()
        },
        "task_ids": [spec["task_id"] for spec in TASK_SPECS],
        "seeds": list(SEEDS),
        "job_count": len(jobs),
        "counts": {"planned": len(jobs)},
        "predicted_training_interactions": sum(
            job["expected_interactions"] for job in jobs
        ),
        "maximum_parallel": 1,
        "command_contract": dict(config["command_contract"]),
        "resource_limits": dict(config["queue"]),
        "isaac_source_verification": dict(config["_source_verification"]),
        "preservation_verification": {
            "custom_g1": dict(config.get("_frozen_g1_verification", {})),
            "frozen_runs": dict(config.get("_frozen_runs_verification", {})),
        },
        "notes": [
            "Flat is Isaac Lab's registered name for the requested plain terrain.",
            "All tasks command only planar vx, vy, and yaw; no z/height command exists.",
            "G1 rough default vy range is exactly zero, so that task does not strafe.",
            "Raw rewards are not directly comparable across robot/task defaults.",
            "No training, simulator, or GPU process is launched by this dry run.",
        ],
    }
    _validate_queue_bundle(index, shards, root)
    return index, shards


def _validate_queue_bundle(
    index: Mapping[str, Any],
    shards: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> None:
    if set(shards) != {"g1", "go1"}:
        raise ValueError("queue must have separate g1 and go1 shards")
    if index.get("job_count") != 12 or index.get("counts") != {"planned": 12}:
        raise ValueError("matrix must contain exactly 12 planned jobs")
    if index.get("predicted_training_interactions") != 464_486_400:
        raise ValueError("matrix interaction total differs from the reviewed budget")
    seen_ids: set[str] = set()
    seen_cwds: set[str] = set()
    observed_tasks: list[str] = []
    observed_jobs = 0
    for robot, shard in shards.items():
        if shard.get("job_count") != 6 or shard.get("counts") != {"planned": 6}:
            raise ValueError(f"{robot} queue must contain exactly six jobs")
        robot_root = (output_root / robot).resolve()
        if Path(str(shard["queue_file"])).resolve() != robot_root / "queue.json":
            raise ValueError(f"{robot} queue file is not isolated in its robot folder")
        for job in shard["jobs"]:
            observed_jobs += 1
            if job["robot"] != robot or "-Play-" in job["task"]:
                raise ValueError("job uses the wrong robot shard or a Play task")
            cwd = Path(job["working_directory"]).resolve()
            if not cwd.is_relative_to(robot_root):
                raise ValueError("job working directory crosses robot folders")
            if cwd.is_relative_to((ROOT / "runs").resolve()):
                raise ValueError("stock locomotion job would write into frozen runs/")
            if job["id"] in seen_ids or str(cwd) in seen_cwds:
                raise ValueError("job id or working directory is not unique")
            seen_ids.add(job["id"])
            seen_cwds.add(str(cwd))
            command = job["training_command"]
            if command[:2] != [str(ISAAC_PYTHON), str(TRAINER)]:
                raise ValueError("job does not use the canonical interpreter/trainer")
            for option, expected in (
                ("--task", job["task"]),
                ("--agent", "rsl_rl_cfg_entry_point"),
                ("--seed", str(job["seed"])),
                ("--num_envs", str(NUM_ENVS)),
                ("--max_iterations", str(job["max_iterations"])),
                ("--device", "cuda:0"),
                ("--logger", "tensorboard"),
                ("--run_name", f"seed_{job['seed']}"),
            ):
                if option not in command or command[command.index(option) + 1] != expected:
                    raise ValueError(f"job command has a wrong {option} value")
            if "--headless" not in command:
                raise ValueError("job command is not headless")
            if job["command_contract"]["axes"] != [
                "lin_vel_x",
                "lin_vel_y",
                "ang_vel_z",
            ] or job["command_contract"]["vertical_command_present"]:
                raise ValueError("job command contract contains a vertical command")
            observed_tasks.append(job["task"])
    if observed_jobs != 12:
        raise ValueError("queue bundle does not contain 12 jobs")
    expected_tasks = [
        spec["task_id"] for spec in TASK_SPECS for _seed in SEEDS
    ]
    if observed_tasks != expected_tasks:
        raise ValueError("queue task/seed order differs from the reviewed matrix")


def write_dry_run(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dry-run root: {output_root}"
        )
    index, shards = build_queue_bundle(config, output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for robot in ("g1", "go1"):
            _atomic_json(output_root / robot / "queue.json", shards[robot])
        _atomic_json(output_root / "matrix_index.json", index)
    except Exception:
        # Preserve any partial files for diagnosis; never pretend the bundle passed.
        raise
    return index


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    index_path = output_root / "matrix_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"dry-run index does not exist: {index_path}")
    index = _read_json(index_path)
    shards = {
        robot: _read_json(output_root / robot / "queue.json")
        for robot in ("g1", "go1")
    }
    _validate_queue_bundle(index, shards, output_root)
    return {
        "status": index["status"],
        "execution_authorized": index["execution_authorized"],
        "job_count": index["job_count"],
        "counts": index["counts"],
        "predicted_training_interactions": index[
            "predicted_training_interactions"
        ],
        "robot_queues": {
            robot: {
                "queue_file": shard["queue_file"],
                "job_count": shard["job_count"],
                "counts": shard["counts"],
                "predicted_training_interactions": shard[
                    "predicted_training_interactions"
                ],
            }
            for robot, shard in shards.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock Isaac Lab G1/Go1 Flat/Rough locomotion queue v1"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reserved for a later reviewed executor; never resumes native RSL exactly.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = validate_config(args.config)
    if args.resume and not args.execute:
        raise ValueError("--resume is valid only with --execute")
    if args.execute:
        raise RuntimeError(
            "execution is fail-closed: review this dry run and pass all four "
            "one-update memory smokes before adding an executor"
        )
    if args.dry_run:
        result = write_dry_run(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(json.dumps(status(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
