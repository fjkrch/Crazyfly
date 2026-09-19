"""Fail-closed local assets for the Crazyflie simulator scene.

The installed Isaac Lab Crazyflie configuration normally points at an
Omniverse URL, and its stock ground-plane spawner also opens a USD asset.  The
training and evaluation launchers need neither network dependency: this module
verifies the already prepared local Crazyflie mirror byte-for-byte and replaces
the stock ground with a procedurally generated flat collision mesh.

Isaac modules are imported only inside :func:`configure_offline_scene`, after
``AppLauncher`` is live.  Hash verification and fingerprint construction stay
safe for ordinary Python tooling and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Callable


OFFLINE_SCENE_CONTRACT_VERSION = "crazyflie_local_assets_v1"
DEFAULT_ASSET_MIRROR = Path.home() / ".cache" / "flyg1" / "isaac-5.1-offline"
MINIMUM_GROUND_EXTENT_M = 20.0
GROUND_EDGE_MARGIN_M = 1.0


@dataclass(frozen=True)
class PinnedAsset:
    relative_path: str
    sha256: str


PINNED_ASSETS = (
    PinnedAsset(
        relative_path="Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        sha256="7372ac0786312c47a92603da3fcd412d560b21c3757a8f0d5e7c2bfb2233d2f4",
    ),
    PinnedAsset(
        relative_path=(
            "Isaac/Robots/Bitcraze/Crazyflie/configuration/"
            "cf2x_robot_schema.usd"
        ),
        sha256="c7a63f78ce3937c25cd05936ee73348bfdbbd0a10e82c0b8a37250730a3cbb9c",
    ),
)


class OfflineAssetError(RuntimeError):
    """Raised when a pinned local simulation asset is missing or changed."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def offline_scene_contract() -> dict[str, Any]:
    """Return the path-independent scene contract used in fingerprints."""

    return {
        "version": OFFLINE_SCENE_CONTRACT_VERSION,
        "robot_assets": [
            {"relative_path": item.relative_path, "sha256": item.sha256}
            for item in PINNED_ASSETS
        ],
        "ground": {
            "kind": "local_procedural_mesh_plane",
            "minimum_extent_m": MINIMUM_GROUND_EXTENT_M,
            "environment_origins": "deterministic_grid",
            "external_usd": False,
        },
    }


def verify_asset_mirror(
    mirror_root: str | Path = DEFAULT_ASSET_MIRROR,
) -> tuple[Path, dict[str, Any]]:
    """Verify every mirrored dependency and return the top-level robot USD.

    This function deliberately never downloads or repairs files.  A missing or
    modified mirror fails before Gym allocates a simulator environment.
    """

    root = Path(mirror_root).expanduser().resolve()
    files: list[dict[str, str]] = []
    for item in PINNED_ASSETS:
        path = root / item.relative_path
        if not path.is_file():
            raise OfflineAssetError(
                "Pinned offline Crazyflie asset is missing: "
                f"{path}. Prepare the verified local mirror before launching Isaac."
            )
        actual = _sha256_file(path)
        if actual != item.sha256:
            raise OfflineAssetError(
                "Pinned offline Crazyflie asset SHA-256 mismatch: "
                f"path={path}, expected={item.sha256}, actual={actual}"
            )
        files.append({"path": str(path), "sha256": actual})
    top_level = root / PINNED_ASSETS[0].relative_path
    return top_level, {
        "version": OFFLINE_SCENE_CONTRACT_VERSION,
        "mirror_root": str(root),
        "files": files,
    }


def procedural_ground_extent_m(
    *,
    num_envs: int,
    env_spacing_m: float,
    workspace_half_width_m: float,
) -> float:
    """Return a square plane size covering the complete environment grid."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    for label, value in (
        ("env_spacing_m", env_spacing_m),
        ("workspace_half_width_m", workspace_half_width_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive and finite")

    short_side = max(1, int(math.sqrt(num_envs)))
    long_side = int(math.ceil(num_envs / short_side))
    grid_span = (max(short_side, long_side) - 1) * env_spacing_m
    required = grid_span + 2.0 * (workspace_half_width_m + GROUND_EDGE_MARGIN_M)
    return float(max(MINIMUM_GROUND_EXTENT_M, required))


def configure_offline_scene(
    cfg: Any,
    *,
    mirror_root: str | Path = DEFAULT_ASSET_MIRROR,
    terrain_generator_factory: Callable[..., Any] | None = None,
    mesh_plane_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Mutate a Crazyflie environment cfg to use only verified local assets."""

    robot_usd, asset_report = verify_asset_mirror(mirror_root)
    if terrain_generator_factory is None or mesh_plane_factory is None:
        from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

        terrain_generator_factory = TerrainGeneratorCfg
        mesh_plane_factory = MeshPlaneTerrainCfg

    num_envs = int(cfg.scene.num_envs)
    env_spacing = float(cfg.scene.env_spacing)
    workspace_half_width = float(getattr(cfg, "workspace_xy_limit_m", 2.75))
    extent = procedural_ground_extent_m(
        num_envs=num_envs,
        env_spacing_m=env_spacing,
        workspace_half_width_m=workspace_half_width,
    )

    cfg.robot.spawn.usd_path = str(robot_usd)
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.usd_path = None
    cfg.terrain.terrain_generator = terrain_generator_factory(
        seed=0,
        size=(extent, extent),
        num_rows=1,
        num_cols=1,
        border_width=0.0,
        curriculum=False,
        use_cache=False,
        sub_terrains={"flat": mesh_plane_factory(proportion=1.0)},
    )
    # One generated mesh covers the whole scene.  Grid origins keep vectorized
    # environments separated; using the single sub-terrain origin would stack
    # every robot at (0, 0, 0).
    cfg.terrain.use_terrain_origins = False

    return {
        **asset_report,
        "robot_usd": str(robot_usd),
        "ground": {
            "kind": "local_procedural_mesh_plane",
            "extent_m": [extent, extent],
            "environment_origins": "deterministic_grid",
            "use_terrain_origins": False,
            "num_envs": num_envs,
            "env_spacing_m": env_spacing,
            "external_usd": False,
        },
    }


__all__ = [
    "DEFAULT_ASSET_MIRROR",
    "OFFLINE_SCENE_CONTRACT_VERSION",
    "OfflineAssetError",
    "PINNED_ASSETS",
    "configure_offline_scene",
    "offline_scene_contract",
    "procedural_ground_extent_m",
    "verify_asset_mirror",
]
