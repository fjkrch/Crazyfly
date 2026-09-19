"""Pinned adapter for Isaac Lab's installed Crazyflie direct task.

This module intentionally has no Isaac Sim imports at module import time.  Pure
Python tooling can import the task package, while :func:`validate_upstream_contract`
performs the authoritative (and potentially Isaac-dependent) checks immediately
before a custom environment is loaded.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

NATIVE_TASK_ID = "Isaac-Quadcopter-Direct-v0"
NATIVE_ENV_ENTRY_POINT = "isaaclab_tasks.direct.quadcopter.quadcopter_env:QuadcopterEnv"
NATIVE_CFG_ENTRY_POINT = "isaaclab_tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvCfg"

OBSERVATION_NAMES = (
    "body_linear_velocity_x",
    "body_linear_velocity_y",
    "body_linear_velocity_z",
    "body_angular_velocity_x",
    "body_angular_velocity_y",
    "body_angular_velocity_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "body_goal_displacement_x",
    "body_goal_displacement_y",
    "body_goal_displacement_z",
)
ACTION_NAMES = (
    "normalized_collective_thrust",
    "normalized_body_moment_x",
    "normalized_body_moment_y",
    "normalized_body_moment_z",
)

EXPECTED_OBSERVATION_WIDTH = 12
EXPECTED_ACTION_WIDTH = 4
EXPECTED_PHYSICS_DT = 0.01
EXPECTED_DECIMATION = 2
EXPECTED_CONTROL_DT = 0.02
EXPECTED_NATIVE_EPISODE_LENGTH_S = 10.0
EXPECTED_THRUST_TO_WEIGHT = 1.9
EXPECTED_MOMENT_SCALE = 0.01
EXPECTED_ASSET_SUFFIX = "/Robots/Bitcraze/Crazyflie/cf2x.usd"

# These hashes pin the three authoritative files named in plan.md.  A source
# update must be deliberately audited instead of being accepted silently.
EXPECTED_SOURCE_SHA256 = {
    "quadcopter_env": "b78a48cdb04f4a215dded24c195f14818fd5fb43f6e682eeaa1a057fc6c77f23",
    "quadcopter_registration": "1cd4d4087adfd9ba2cf8b7968fa492f9c35235463d469b5e703f393c7809f8c2",
    "crazyflie_asset": "b36ebdc4a75c670cf503496734c022ade078c77f9d4c68dcd2e1fe6d8a620c72",
}


class UpstreamContractError(RuntimeError):
    """Raised when the installed NVIDIA task no longer matches the pinned contract."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UpstreamContractError(
            "Installed Isaac-Quadcopter-Direct-v0 contract mismatch: " + message
        )


def validate_upstream_contract(*, strict_source_hash: bool = True) -> dict[str, Any]:
    """Validate and describe the installed upstream Crazyflie task.

    The imports are kept inside this function because Isaac Lab task modules
    must only be imported after ``AppLauncher`` has created the simulation app.
    """

    import gymnasium as gym

    from isaaclab_assets import CRAZYFLIE_CFG
    from isaaclab_assets.robots import quadcopter as asset_module
    from isaaclab_tasks.direct.quadcopter import quadcopter_env as upstream_module
    from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg

    import isaaclab_tasks.direct.quadcopter as registration_module

    source_paths = {
        "quadcopter_env": Path(inspect.getsourcefile(upstream_module) or ""),
        "quadcopter_registration": Path(inspect.getsourcefile(registration_module) or ""),
        "crazyflie_asset": Path(inspect.getsourcefile(asset_module) or ""),
    }
    for label, path in source_paths.items():
        _require(path.is_file(), f"cannot resolve authoritative source file for {label!r}: {path}")

    source_hashes = {label: _sha256(path) for label, path in source_paths.items()}
    if strict_source_hash:
        for label, expected in EXPECTED_SOURCE_SHA256.items():
            _require(
                source_hashes[label] == expected,
                f"{label} SHA-256 is {source_hashes[label]}, expected {expected}; audit the upstream change",
            )

    cfg = QuadcopterEnvCfg()
    _require(
        cfg.observation_space == EXPECTED_OBSERVATION_WIDTH,
        f"observation_space={cfg.observation_space!r}",
    )
    _require(cfg.action_space == EXPECTED_ACTION_WIDTH, f"action_space={cfg.action_space!r}")
    _require(cfg.decimation == EXPECTED_DECIMATION, f"decimation={cfg.decimation!r}")
    _require(math.isclose(cfg.sim.dt, EXPECTED_PHYSICS_DT), f"sim.dt={cfg.sim.dt!r}")
    _require(
        math.isclose(cfg.sim.dt * cfg.decimation, EXPECTED_CONTROL_DT),
        f"control_dt={cfg.sim.dt * cfg.decimation!r}",
    )
    _require(
        math.isclose(cfg.episode_length_s, EXPECTED_NATIVE_EPISODE_LENGTH_S),
        f"native episode_length_s={cfg.episode_length_s!r}",
    )
    _require(math.isclose(cfg.thrust_to_weight, EXPECTED_THRUST_TO_WEIGHT), "thrust_to_weight changed")
    _require(math.isclose(cfg.moment_scale, EXPECTED_MOMENT_SCALE), "moment_scale changed")
    _require(cfg.robot.spawn.usd_path.endswith(EXPECTED_ASSET_SUFFIX), f"asset={cfg.robot.spawn.usd_path!r}")
    _require(
        CRAZYFLIE_CFG.spawn.usd_path.endswith(EXPECTED_ASSET_SUFFIX),
        f"CRAZYFLIE_CFG asset={CRAZYFLIE_CFG.spawn.usd_path!r}",
    )
    _require(tuple(CRAZYFLIE_CFG.init_state.pos) == (0.0, 0.0, 0.5), "default root position changed")

    _require(NATIVE_TASK_ID in gym.registry, f"native Gym task {NATIVE_TASK_ID!r} is not registered")
    spec = gym.spec(NATIVE_TASK_ID)
    _require(spec.entry_point == NATIVE_ENV_ENTRY_POINT, f"native entry point={spec.entry_point!r}")
    _require(
        spec.kwargs.get("env_cfg_entry_point") == NATIVE_CFG_ENTRY_POINT,
        f"native config entry point={spec.kwargs.get('env_cfg_entry_point')!r}",
    )
    _require(QuadcopterEnv.__module__ == upstream_module.__name__, "upstream environment class moved")

    return {
        "native_task_id": NATIVE_TASK_ID,
        "entry_point": spec.entry_point,
        "env_cfg_entry_point": spec.kwargs.get("env_cfg_entry_point"),
        "observation_width": cfg.observation_space,
        "observation_names": list(OBSERVATION_NAMES),
        "action_width": cfg.action_space,
        "action_names": list(ACTION_NAMES),
        "physics_dt_s": cfg.sim.dt,
        "decimation": cfg.decimation,
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "native_episode_length_s": cfg.episode_length_s,
        "thrust_to_weight": cfg.thrust_to_weight,
        "moment_scale_nm": cfg.moment_scale,
        "normalized_action_bounds": [-1.0, 1.0],
        "collective_thrust_mapping": "1.9 * vehicle_weight * (action[0] + 1) / 2 along body +Z",
        "body_moment_mapping": "0.01 N m * action[1:4] in the body frame",
        "native_goal_bounds_relative_to_env_origin_m": {
            "x": [-2.0, 2.0],
            "y": [-2.0, 2.0],
            "z": [0.5, 1.5],
        },
        "native_height_termination_m": [0.1, 2.0],
        "asset_usd": cfg.robot.spawn.usd_path,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "source_sha256": source_hashes,
        "versions": {
            "python": platform.python_version(),
            "isaaclab": _package_version("isaaclab"),
            "isaaclab_assets": _package_version("isaaclab-assets"),
            "isaaclab_tasks": _package_version("isaaclab-tasks"),
            "isaacsim": _package_version("isaacsim"),
            "torch": _package_version("torch"),
            "gymnasium": _package_version("gymnasium"),
        },
    }


def validate_runtime_contract(env: Any) -> None:
    """Check properties only available after a simulator-backed env exists."""

    import gymnasium as gym

    _require(
        gym.spaces.flatdim(env.single_observation_space["policy"]) == 12,
        "runtime observation width changed",
    )
    _require(gym.spaces.flatdim(env.single_action_space) == 4, "runtime action width changed")
    _require(math.isclose(float(env.physics_dt), EXPECTED_PHYSICS_DT), "runtime physics dt changed")
    _require(math.isclose(float(env.step_dt), EXPECTED_CONTROL_DT), "runtime control dt changed")
    _require(
        math.isclose(float(env.cfg.thrust_to_weight), EXPECTED_THRUST_TO_WEIGHT),
        "runtime thrust mapping changed",
    )
    _require(
        math.isclose(float(env.cfg.moment_scale), EXPECTED_MOMENT_SCALE),
        "runtime moment mapping changed",
    )
    _require(float(env._robot_mass) > 0.0, f"invalid Crazyflie mass {env._robot_mass!r}")
    _require(len(env._body_id) == 1, f"expected one body named 'body', found indices {env._body_id!r}")


__all__ = [
    "ACTION_NAMES",
    "EXPECTED_ACTION_WIDTH",
    "EXPECTED_CONTROL_DT",
    "EXPECTED_OBSERVATION_WIDTH",
    "EXPECTED_SOURCE_SHA256",
    "NATIVE_TASK_ID",
    "OBSERVATION_NAMES",
    "UpstreamContractError",
    "validate_runtime_contract",
    "validate_upstream_contract",
]
