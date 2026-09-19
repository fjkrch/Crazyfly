#!/usr/bin/env python3
"""Shared, additive bootstrap and reproduction fingerprints for Crazyflie tools."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
ISAACLAB_ROOT = Path("/home/chayanin/Downloads/IsaacLab")
UPSTREAM_QUADCOPTER_ENV = (
    ISAACLAB_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "quadcopter" / "quadcopter_env.py"
)
UPSTREAM_QUADCOPTER_REGISTRATION = UPSTREAM_QUADCOPTER_ENV.with_name("__init__.py")
UPSTREAM_CRAZYFLIE_ASSET = (
    ISAACLAB_ROOT / "source" / "isaaclab_assets" / "isaaclab_assets" / "robots" / "quadcopter.py"
)
UPSTREAM_DIRECT_RL_ENV = (
    ISAACLAB_ROOT / "source" / "isaaclab" / "isaaclab" / "envs" / "direct_rl_env.py"
)
UPSTREAM_ISAACLAB_MATH = (
    ISAACLAB_ROOT / "source" / "isaaclab" / "isaaclab" / "utils" / "math.py"
)
UPSTREAM_OFFLINE_TERRAIN_SOURCES = tuple(
    ISAACLAB_ROOT / "source" / "isaaclab" / "isaaclab" / "terrains" / relative
    for relative in (
        "terrain_generator.py",
        "terrain_generator_cfg.py",
        "terrain_importer.py",
        "terrain_importer_cfg.py",
        "sub_terrain_cfg.py",
        "trimesh/mesh_terrains.py",
        "trimesh/mesh_terrains_cfg.py",
        "trimesh/utils.py",
        "utils.py",
    )
)
DEFAULT_CONNECTOME = ROOT / "data" / "connectome" / "manifest.json"
DEFAULT_WING_CONNECTOME = ROOT / "data" / "connectome_wing" / "manifest.json"
DEFAULT_OPTIC_CONNECTOME = ROOT / "data" / "connectome_optic" / "manifest.json"
DEFAULT_REWIRE_MANIFEST = ROOT / "configs" / "experiments" / "crazyflie_rewire_seed_20260916.json"
CRAZYFLIE_RESOLVED_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
)
CONTRACT_PROFILE_SURVIVAL_V2 = "survival_v2"
CONTRACT_PROFILE_BALANCED_V3 = "balanced_v3"
CONTRACT_PROFILE_BALANCED_V4 = "balanced_v4"
CONTRACT_PROFILE_COMMAND_V1 = "command_v1"
CONTRACT_PROFILE_COMMAND_V2 = "command_v2"
COMMAND_TASK_ID = "FlyCrazyflie-CommandFollow-v0"
COMMAND_WIDE_TASK_ID = "FlyCrazyflie-CommandFollowWide-v0"
COMMAND_WIDE_WIND_TASK_ID = "FlyCrazyflie-CommandFollowWideWind-v0"
COMMAND_V2_TASK_IDS = (COMMAND_WIDE_TASK_ID, COMMAND_WIDE_WIND_TASK_ID)
COMMAND_TASK_IDS = (COMMAND_TASK_ID, *COMMAND_V2_TASK_IDS)
DEFAULT_CONTRACT_PROFILE = CONTRACT_PROFILE_BALANCED_V3
CONTRACT_PROFILES = (
    CONTRACT_PROFILE_SURVIVAL_V2,
    CONTRACT_PROFILE_BALANCED_V3,
    CONTRACT_PROFILE_BALANCED_V4,
    CONTRACT_PROFILE_COMMAND_V1,
    CONTRACT_PROFILE_COMMAND_V2,
)


def rollout_rng_contract() -> dict[str, Any]:
    """Return the immutable two-phase training RNG contract."""

    return {
        "scheme": "post_construction_reseed_v1",
        "seed_source": "job_seed",
        "controller_construction_timing": (
            "after_app_and_environment_construction_before_policy_optimizer_"
            "scheduler_and_normalizer_construction"
        ),
        "controller_construction_streams": [
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda_all_if_available",
        ],
        "fresh_start_timing": (
            "after_environment_policy_optimizer_scheduler_and_normalizer_construction_"
            "before_initial_checkpoint_and_first_rollout"
        ),
        "fresh_start_streams": [
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda_all_if_available",
        ],
        "resume_behavior": (
            "preseed_controller_construction_then_preserve_checkpoint_restored_rng_without_post_reseed"
        ),
    }

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab-assets",
    "isaaclab-tasks",
    "isaacsim",
    "torch",
    "gymnasium",
    "numpy",
    "trimesh",
    "scipy",
)

OBSERVATION_CONTRACT = {
    "width": 12,
    "names": [
        "body_linear_velocity_x_m_s", "body_linear_velocity_y_m_s", "body_linear_velocity_z_m_s",
        "body_angular_velocity_x_rad_s", "body_angular_velocity_y_rad_s", "body_angular_velocity_z_rad_s",
        "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
        "body_goal_displacement_x_m", "body_goal_displacement_y_m", "body_goal_displacement_z_m",
    ],
    "ordering": "upstream Isaac-Quadcopter-Direct-v0",
    "clipping": [-5.0, 5.0],
    "normalization": "immutable per-dimension physics scaling; no training updates",
    "normalization_kind": "fixed_physics_scale_v1",
    "normalization_scale": [
        2.0, 2.0, 2.0,
        5.0, 5.0, 5.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
    ],
    "normalization_update_rule": "immutable_no_updates",
}

ACTION_CONTRACT = {
    "width": 4,
    "names": ["collective_thrust", "body_moment_x", "body_moment_y", "body_moment_z"],
    "policy_range": [-1.0, 1.0],
    "mapping": {
        "collective_thrust": "(a0 + 1) / 2 * 1.9 * vehicle_weight along body +Z",
        "body_moments": "a[1:4] * 0.01 N m",
    },
    "control_dt_s": 0.02,
    "control_frequency_hz": 50.0,
}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_fingerprint_rewire_manifest(
    path: str | Path = DEFAULT_REWIRE_MANIFEST,
    *,
    expected_file_sha256: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Load the frozen rewire identity before Isaac starts.

    Full graph invariants are checked by the controller builder.  This small
    CPU-safe check makes source/config fingerprints depend on the exact
    checked-in artifact and rejects a stale claimed file hash early.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Frozen rewire manifest does not exist: {source}")
    actual_file_sha256 = sha256_file(source)
    if expected_file_sha256 is not None and actual_file_sha256 != expected_file_sha256:
        raise ValueError(
            "Frozen rewire file SHA-256 mismatch: "
            f"expected={expected_file_sha256}, actual={actual_file_sha256}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode frozen rewire manifest {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Frozen rewire manifest must be a JSON object")
    from g1_fly_control.crazyflie.controllers import rewire_manifest_checksum

    declared = value.get("checksum")
    actual_content_sha256 = rewire_manifest_checksum(value)
    if declared != actual_content_sha256:
        raise ValueError(
            "Frozen rewire content checksum mismatch: "
            f"declared={declared!r}, actual={actual_content_sha256}"
        )
    if expected_seed is not None and value.get("seed") != expected_seed:
        raise ValueError(
            f"Frozen rewire seed {value.get('seed')!r} does not match expected seed {expected_seed}"
        )
    return value


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _existing(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()}, key=lambda item: str(item))


def execution_source_paths() -> list[Path]:
    """Every new or reused source file that can affect a drone artifact."""
    package = SOURCE / "g1_fly_control"
    paths: list[Path] = list((ROOT / "scripts").glob("drone_*.py"))
    paths.extend((ROOT / "scripts").glob("execute_drone_*.sh"))
    # Python executes both parent initializers before importing any nested
    # Crazyflie module, so their bytes are part of the executable identity.
    paths.extend((package / "__init__.py", package / "tasks" / "__init__.py"))
    paths.extend((package / "tasks" / "crazyflie").glob("*.py"))
    paths.extend((package / "crazyflie").glob("*.py"))
    # These frozen modules are reused by import and therefore belong in the
    # reproduction identity even though this implementation never edits them.
    for component in ("connectome", "policies", "training"):
        paths.extend((package / component).glob("*.py"))
    # The custom environment imports DirectRLEnv and two quaternion helpers
    # directly.  Their exact installed bytes affect reset/step ordering and
    # observation construction, so an Isaac commit string alone is not a
    # fail-closed reproduction identity for a locally modified checkout.
    paths.extend((
        UPSTREAM_QUADCOPTER_ENV,
        UPSTREAM_QUADCOPTER_REGISTRATION,
        UPSTREAM_CRAZYFLIE_ASSET,
        UPSTREAM_DIRECT_RL_ENV,
        UPSTREAM_ISAACLAB_MATH,
    ))
    # Every project launcher, including the native-task route, replaces the
    # cloud-backed plane with Isaac Lab's procedural mesh terrain path.  Those
    # installed sources affect mesh construction and vector-environment origins
    # and therefore belong to the fail-closed reproduction identity too.
    paths.extend(UPSTREAM_OFFLINE_TERRAIN_SOURCES)
    return _existing(paths)


def source_hashes(paths: Iterable[Path] | None = None) -> dict[str, str]:
    selected = execution_source_paths() if paths is None else _existing(paths)
    return {_relative_or_absolute(path): sha256_file(path) for path in selected}


def installed_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    result["python"] = platform.python_version()
    return result


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def runtime_identity() -> dict[str, Any]:
    commit = _command_output(["git", "-C", str(ISAACLAB_ROOT), "rev-parse", "HEAD"])
    gpu_line = _command_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits",
    ])
    try:
        import torch
        torch_identity = {
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - only for damaged installations
        torch_identity = {"import_error": f"{type(exc).__name__}: {exc}"}
    return {
        "packages": installed_versions(),
        "platform": platform.platform(),
        "isaaclab_commit": commit,
        "nvidia_smi": gpu_line,
        "torch": torch_identity,
    }


def offline_scene_fingerprint_contract(task: str) -> dict[str, Any]:
    """Return the scene provenance that actually applies to one task ID."""

    if not isinstance(task, str) or not task:
        raise ValueError("Resolved configuration lacks a task identity")
    from g1_fly_control.tasks.crazyflie.offline_assets import offline_scene_contract

    return {"applicable": True, **offline_scene_contract()}


def observation_contract_for_task(task: str) -> dict[str, Any]:
    """Return the task-specific semantics for the shared 12-value tensor."""

    if task == COMMAND_TASK_ID:
        from g1_fly_control.tasks.crazyflie.command_logic import (
            OBSERVATION_CONTRACT_VERSION,
            command_training_contract_payload,
        )

        contract = command_training_contract_payload()
        return {
            **OBSERVATION_CONTRACT,
            "names": list(contract["observation_order"]),
            "ordering": OBSERVATION_CONTRACT_VERSION,
        }
    if task in COMMAND_V2_TASK_IDS:
        from g1_fly_control.tasks.crazyflie.command_wide_logic import (
            OBSERVATION_CONTRACT_VERSION,
            command_wide_training_contract_payload,
        )

        contract = command_wide_training_contract_payload(
            wind_enabled=task == COMMAND_WIDE_WIND_TASK_ID
        )
        return {
            **OBSERVATION_CONTRACT,
            "names": list(contract["observation_order"]),
            "ordering": OBSERVATION_CONTRACT_VERSION,
        }
    return OBSERVATION_CONTRACT


def fingerprint_payload(
    *,
    resolved_config: dict[str, Any],
    evaluation_manifest: dict[str, Any] | None = None,
    connectome_manifest: str | Path | None = None,
    rewired_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = resolved_config.get("task")
    offline_contract = offline_scene_fingerprint_contract(task)

    connectome_path = Path(connectome_manifest).resolve() if connectome_manifest else None
    connectome = None
    if connectome_path is not None:
        connectome = {"path": str(connectome_path), "sha256": sha256_file(connectome_path)}
        try:
            raw = json.loads(connectome_path.read_text(encoding="utf-8"))
            for field in ("neurons_path", "edges_path"):
                relative = raw.get(field)
                if relative:
                    child = (connectome_path.parent / relative).resolve()
                    connectome[field] = {"path": str(child), "sha256": sha256_file(child)}
        except (OSError, ValueError, TypeError):
            connectome["parse_status"] = "unavailable"
    runtime = runtime_identity()
    return {
        "schema_version": 1,
        "source_sha256": source_hashes(),
        "resolved_config": resolved_config,
        "evaluation_manifest_sha256": canonical_sha256(evaluation_manifest) if evaluation_manifest else None,
        "connectome": connectome,
        "rewired_manifest_sha256": canonical_sha256(rewired_manifest) if rewired_manifest else None,
        "runtime": runtime,
        "observation_contract": observation_contract_for_task(task),
        "action_contract": ACTION_CONTRACT,
        "offline_scene_contract": offline_contract,
        "upstream": {
            "task_id": "Isaac-Quadcopter-Direct-v0",
            "environment_source": str(UPSTREAM_QUADCOPTER_ENV),
            "direct_rl_env_source": str(UPSTREAM_DIRECT_RL_ENV),
            "direct_rl_env_source_sha256": sha256_file(UPSTREAM_DIRECT_RL_ENV),
            "math_source": str(UPSTREAM_ISAACLAB_MATH),
            "math_source_sha256": sha256_file(UPSTREAM_ISAACLAB_MATH),
            "asset_config_source": str(UPSTREAM_CRAZYFLIE_ASSET),
            "asset_config_source_sha256": sha256_file(UPSTREAM_CRAZYFLIE_ASSET),
            "asset_package": "isaaclab-assets",
            "asset_package_version": runtime["packages"]["isaaclab-assets"],
            # This exact URL was resolved by the live installed-asset audit.
            # The live audit also rejects any future asset redirect/config drift.
            "resolved_usd_url": CRAZYFLIE_RESOLVED_USD_URL,
        },
    }


def reproduction_fingerprint(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    payload = fingerprint_payload(**kwargs)
    return canonical_sha256(payload), payload


def validate_contract_profile(
    contract_profile: str,
    *,
    task: str | None = None,
) -> str:
    """Validate an explicit Crazyflie task-contract selector.

    The selector applies only to project task configuration.  The upstream
    native task accepts either recognized selector for shared-CLI compatibility
    but :func:`selected_env_cfg` always returns its untouched installed config.
    Thus changing the active project default cannot change native physics or
    make the canonical native smoke command invalid.
    """

    if not isinstance(contract_profile, str) or contract_profile not in CONTRACT_PROFILES:
        raise ValueError(
            "contract_profile must be one of " + ", ".join(CONTRACT_PROFILES)
        )
    if task == COMMAND_TASK_ID and contract_profile != CONTRACT_PROFILE_COMMAND_V1:
        raise ValueError(
            f"{COMMAND_TASK_ID} requires contract_profile={CONTRACT_PROFILE_COMMAND_V1}"
        )
    if task in COMMAND_V2_TASK_IDS and contract_profile != CONTRACT_PROFILE_COMMAND_V2:
        raise ValueError(
            f"{task} requires contract_profile={CONTRACT_PROFILE_COMMAND_V2}"
        )
    if (
        contract_profile == CONTRACT_PROFILE_COMMAND_V1
        and task is not None
        and task != COMMAND_TASK_ID
    ):
        raise ValueError(
            f"contract_profile={CONTRACT_PROFILE_COMMAND_V1} is valid only for {COMMAND_TASK_ID}"
        )
    if (
        contract_profile == CONTRACT_PROFILE_COMMAND_V2
        and task is not None
        and task not in COMMAND_V2_TASK_IDS
    ):
        raise ValueError(
            "contract_profile=command_v2 is valid only for "
            + ", ".join(COMMAND_V2_TASK_IDS)
        )
    return contract_profile


def task_contract_payload(
    contract_profile: str,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """Return the exact immutable project task contract for a profile."""

    profile = validate_contract_profile(contract_profile)
    from g1_fly_control.tasks.crazyflie.logic import (
        balanced_task_contract_payload,
        balanced_v4_task_contract_payload,
        survival_first_contract_payload,
    )

    if profile == CONTRACT_PROFILE_BALANCED_V3:
        return balanced_task_contract_payload()
    if profile == CONTRACT_PROFILE_BALANCED_V4:
        return balanced_v4_task_contract_payload()
    if profile == CONTRACT_PROFILE_COMMAND_V1:
        from g1_fly_control.tasks.crazyflie.command_logic import (
            command_training_contract_payload,
        )

        return command_training_contract_payload()
    if profile == CONTRACT_PROFILE_COMMAND_V2:
        from g1_fly_control.tasks.crazyflie.command_wide_logic import (
            command_wide_training_contract_payload,
        )

        resolved_task = COMMAND_WIDE_TASK_ID if task is None else task
        validate_contract_profile(profile, task=resolved_task)
        return command_wide_training_contract_payload(
            wind_enabled=resolved_task == COMMAND_WIDE_WIND_TASK_ID
        )
    return survival_first_contract_payload()


def selected_env_cfg(
    task: str,
    num_envs: int,
    *,
    deterministic_evaluation: bool = False,
    contract_profile: str = DEFAULT_CONTRACT_PROFILE,
):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    profile = validate_contract_profile(contract_profile, task=task)
    if task == "Isaac-Quadcopter-Direct-v0":
        from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnvCfg
        cfg = QuadcopterEnvCfg()
        cfg.debug_vis = False
    elif task == COMMAND_TASK_ID:
        from g1_fly_control.tasks.crazyflie.adapter import validate_upstream_contract
        from g1_fly_control.tasks.crazyflie.command_env_cfg import CommandFollowEnvCfg

        validate_upstream_contract()
        cfg = CommandFollowEnvCfg()
        cfg.debug_vis = False
    elif task in COMMAND_V2_TASK_IDS:
        from g1_fly_control.tasks.crazyflie.adapter import validate_upstream_contract
        from g1_fly_control.tasks.crazyflie.command_wide_env_cfg import (
            CommandFollowWideEnvCfg,
            CommandFollowWideWindEnvCfg,
        )

        validate_upstream_contract()
        cfg_type = (
            CommandFollowWideWindEnvCfg
            if task == COMMAND_WIDE_WIND_TASK_ID
            else CommandFollowWideEnvCfg
        )
        cfg = cfg_type()
        cfg.debug_vis = False
    else:
        from g1_fly_control.tasks.crazyflie.adapter import validate_upstream_contract
        from g1_fly_control.tasks.crazyflie.env_cfg import (
            BalancedGustRecoveryEnvCfg,
            BalancedMixedTrainingEnvCfg,
            BalancedV4GustRecoveryEnvCfg,
            BalancedV4WaypointReachEnvCfg,
            BalancedV4WaypointSwitchEnvCfg,
            BalancedWaypointReachEnvCfg,
            BalancedWaypointSwitchEnvCfg,
            GustRecoveryEnvCfg,
            MixedTrainingEnvCfg,
            WaypointReachEnvCfg,
            WaypointSwitchEnvCfg,
        )
        validate_upstream_contract()
        configs = {
            "FlyCrazyflie-WaypointReach-v0": WaypointReachEnvCfg,
            "FlyCrazyflie-WaypointSwitch-v0": WaypointSwitchEnvCfg,
            "FlyCrazyflie-GustRecovery-v0": GustRecoveryEnvCfg,
            "FlyCrazyflie-Mixed-v0": MixedTrainingEnvCfg,
        }
        if profile == CONTRACT_PROFILE_BALANCED_V3:
            configs = {
                "FlyCrazyflie-WaypointReach-v0": BalancedWaypointReachEnvCfg,
                "FlyCrazyflie-WaypointSwitch-v0": BalancedWaypointSwitchEnvCfg,
                "FlyCrazyflie-GustRecovery-v0": BalancedGustRecoveryEnvCfg,
                "FlyCrazyflie-Mixed-v0": BalancedMixedTrainingEnvCfg,
            }
        elif profile == CONTRACT_PROFILE_BALANCED_V4:
            configs = {
                "FlyCrazyflie-WaypointReach-v0": BalancedV4WaypointReachEnvCfg,
                "FlyCrazyflie-WaypointSwitch-v0": BalancedV4WaypointSwitchEnvCfg,
                "FlyCrazyflie-GustRecovery-v0": BalancedV4GustRecoveryEnvCfg,
            }
        if task not in configs:
            raise ValueError(f"Unknown Crazyflie task {task!r}; choose one of {', '.join(configs)}")
        cfg = configs[task]()
        cfg.debug_vis = False
        if hasattr(cfg, "deterministic_eval"):
            cfg.deterministic_eval = bool(deterministic_evaluation)
    cfg.scene.num_envs = int(num_envs)
    return cfg


def launch_environment(
    task: str,
    num_envs: int,
    *,
    render_mode: str | None = None,
    deterministic_evaluation: bool = False,
    device: str | object | None = None,
    mixed_scenario_seed: int | None = None,
    command_schedule_seed: int | None = None,
    contract_profile: str = DEFAULT_CONTRACT_PROFILE,
):
    """Create a native or project DirectRLEnv after SimulationApp is live."""
    # Reject a mismatched native/custom contract claim before importing Gym or
    # registering any task entry points.
    validate_contract_profile(contract_profile, task=task)
    import gymnasium as gym
    import isaaclab_tasks.direct.quadcopter  # noqa: F401

    custom_task = task != "Isaac-Quadcopter-Direct-v0"
    if custom_task:
        from g1_fly_control.tasks.crazyflie.registration import register_tasks
        register_tasks()
    cfg = selected_env_cfg(
        task,
        num_envs,
        deterministic_evaluation=deterministic_evaluation,
        contract_profile=contract_profile,
    )
    if mixed_scenario_seed is not None:
        if not hasattr(cfg, "mixed_scenario_seed"):
            raise ValueError("mixed_scenario_seed is valid only for FlyCrazyflie-Mixed-v0")
        if (
            isinstance(mixed_scenario_seed, bool)
            or not isinstance(mixed_scenario_seed, int)
            or mixed_scenario_seed < 0
        ):
            raise ValueError("mixed_scenario_seed must be a non-negative integer")
        cfg.mixed_scenario_seed = int(mixed_scenario_seed)
    if command_schedule_seed is not None:
        if task not in COMMAND_TASK_IDS or not hasattr(cfg, "command_schedule_seed"):
            raise ValueError(
                "command_schedule_seed is valid only for "
                + ", ".join(COMMAND_TASK_IDS)
            )
        if (
            isinstance(command_schedule_seed, bool)
            or not isinstance(command_schedule_seed, int)
            or command_schedule_seed < 0
        ):
            raise ValueError("command_schedule_seed must be a non-negative integer")
        cfg.command_schedule_seed = int(command_schedule_seed)
    if device is not None:
        # AppLauncher does not rewrite an environment cfg created after Kit
        # startup. Callers selecting a CUDA device must therefore propagate it
        # explicitly to simulation as well as to their policy tensors.
        cfg.sim.device = str(device)
    # The installed native task also uses Isaac's cloud-backed GroundPlaneCfg.
    # Apply the same project-owned, verified offline substitutions used by the
    # additive tasks without modifying either installed Isaac Lab or G1 files.
    from g1_fly_control.tasks.crazyflie.offline_assets import configure_offline_scene

    offline_scene_report = configure_offline_scene(cfg)
    wrapped = gym.make(task, cfg=cfg, render_mode=render_mode)
    env = wrapped.unwrapped
    # Runtime-only audit metadata; environment behavior never depends on it.
    env._flyg1_offline_scene_report = offline_scene_report
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shared Crazyflie bootstrap helpers; importing this module does not launch Isaac."
    )
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
