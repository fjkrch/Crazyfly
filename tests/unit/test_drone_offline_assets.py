from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_bootstrap
from drone_bootstrap import offline_scene_fingerprint_contract

from g1_fly_control.tasks.crazyflie.offline_assets import (
    DEFAULT_ASSET_MIRROR,
    OFFLINE_SCENE_CONTRACT_VERSION,
    OfflineAssetError,
    PINNED_ASSETS,
    configure_offline_scene,
    offline_scene_contract,
    procedural_ground_extent_m,
    verify_asset_mirror,
)


def _write_pinned_mirror(root: Path) -> None:
    """Copy the already verified bytes into an isolated test mirror."""

    for item in PINNED_ASSETS:
        source = DEFAULT_ASSET_MIRROR / item.relative_path
        destination = root / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def test_checked_in_offline_mirror_matches_pinned_hashes():
    top_level, report = verify_asset_mirror()

    assert top_level == (DEFAULT_ASSET_MIRROR / PINNED_ASSETS[0].relative_path).resolve()
    assert report["version"] == OFFLINE_SCENE_CONTRACT_VERSION
    assert [item["sha256"] for item in report["files"]] == [
        item.sha256 for item in PINNED_ASSETS
    ]


def test_mirror_verification_fails_closed_on_changed_bytes(tmp_path: Path):
    _write_pinned_mirror(tmp_path)
    damaged = tmp_path / PINNED_ASSETS[1].relative_path
    damaged.write_bytes(damaged.read_bytes() + b"changed")

    with pytest.raises(OfflineAssetError, match="SHA-256 mismatch"):
        verify_asset_mirror(tmp_path)


def test_ground_extent_covers_vectorized_grid_and_workspace():
    assert procedural_ground_extent_m(
        num_envs=4, env_spacing_m=2.5, workspace_half_width_m=2.75
    ) == 20.0
    assert procedural_ground_extent_m(
        num_envs=40, env_spacing_m=2.5, workspace_half_width_m=2.75
    ) == 22.5
    assert procedural_ground_extent_m(
        num_envs=4096, env_spacing_m=2.5, workspace_half_width_m=2.75
    ) >= 165.0


def test_configure_offline_scene_uses_local_robot_and_nonstacked_grid(tmp_path: Path):
    _write_pinned_mirror(tmp_path)
    calls: dict[str, dict] = {}

    def mesh_plane_factory(**kwargs):
        calls["mesh"] = kwargs
        return SimpleNamespace(**kwargs)

    def terrain_generator_factory(**kwargs):
        calls["generator"] = kwargs
        return SimpleNamespace(**kwargs)

    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=4, env_spacing=2.5),
        robot=SimpleNamespace(spawn=SimpleNamespace(usd_path="https://cloud/cf2x.usd")),
        terrain=SimpleNamespace(
            terrain_type="plane",
            usd_path="https://cloud/default_environment.usd",
            terrain_generator=None,
            use_terrain_origins=True,
        ),
        workspace_xy_limit_m=2.75,
    )

    report = configure_offline_scene(
        cfg,
        mirror_root=tmp_path,
        terrain_generator_factory=terrain_generator_factory,
        mesh_plane_factory=mesh_plane_factory,
    )

    expected_robot = (tmp_path / PINNED_ASSETS[0].relative_path).resolve()
    assert cfg.robot.spawn.usd_path == str(expected_robot)
    assert cfg.terrain.terrain_type == "generator"
    assert cfg.terrain.usd_path is None
    assert cfg.terrain.use_terrain_origins is False
    assert calls["generator"]["size"] == (20.0, 20.0)
    assert calls["generator"]["use_cache"] is False
    assert calls["generator"]["sub_terrains"]["flat"].proportion == 1.0
    assert report["ground"]["environment_origins"] == "deterministic_grid"
    assert report["ground"]["external_usd"] is False


def test_offline_contract_is_path_independent_and_complete():
    contract = offline_scene_contract()

    assert contract["version"] == OFFLINE_SCENE_CONTRACT_VERSION
    assert contract["ground"]["external_usd"] is False
    assert contract["robot_assets"] == [
        {"relative_path": item.relative_path, "sha256": item.sha256}
        for item in PINNED_ASSETS
    ]
    assert all(
        len(item["sha256"]) == 64
        and set(item["sha256"]) <= set("0123456789abcdef")
        for item in contract["robot_assets"]
    )


def test_fingerprint_contract_includes_offline_scene_for_native_and_custom_tasks():
    native = offline_scene_fingerprint_contract("Isaac-Quadcopter-Direct-v0")
    custom = offline_scene_fingerprint_contract("FlyCrazyflie-WaypointReach-v0")

    assert native == custom
    assert native["applicable"] is True
    assert native["version"] == OFFLINE_SCENE_CONTRACT_VERSION
    assert native["ground"]["external_usd"] is False
    assert custom["applicable"] is True
    assert custom["version"] == OFFLINE_SCENE_CONTRACT_VERSION
    assert custom["ground"]["external_usd"] is False


def test_native_launch_configures_verified_offline_scene_before_gym_make(monkeypatch):
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1, env_spacing=2.5),
        robot=SimpleNamespace(spawn=SimpleNamespace(usd_path="https://cloud/cf2x.usd")),
        terrain=SimpleNamespace(
            terrain_type="plane",
            usd_path="https://cloud/default_environment.usd",
        ),
        sim=SimpleNamespace(device="cuda:0"),
    )
    report = {
        "version": OFFLINE_SCENE_CONTRACT_VERSION,
        "robot_usd": "/verified/cf2x.usd",
        "ground": {"kind": "local_procedural_mesh_plane", "external_usd": False},
    }
    calls: list[str] = []

    def configure(candidate):
        assert candidate is cfg
        calls.append("configure")
        candidate.robot.spawn.usd_path = report["robot_usd"]
        candidate.terrain.terrain_type = "generator"
        candidate.terrain.usd_path = None
        return report

    native_env = SimpleNamespace()

    def make(task, *, cfg: object, render_mode):
        calls.append("make")
        assert task == "Isaac-Quadcopter-Direct-v0"
        assert cfg is not None
        assert cfg.terrain.terrain_type == "generator"
        assert cfg.terrain.usd_path is None
        assert cfg.robot.spawn.usd_path == report["robot_usd"]
        assert render_mode is None
        return SimpleNamespace(unwrapped=native_env)

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make
    isaaclab_tasks = ModuleType("isaaclab_tasks")
    isaaclab_tasks.__path__ = []
    direct = ModuleType("isaaclab_tasks.direct")
    direct.__path__ = []
    quadcopter = ModuleType("isaaclab_tasks.direct.quadcopter")
    isaaclab_tasks.direct = direct
    direct.quadcopter = quadcopter
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", isaaclab_tasks)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.direct", direct)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.direct.quadcopter", quadcopter)
    monkeypatch.setattr(drone_bootstrap, "selected_env_cfg", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "g1_fly_control.tasks.crazyflie.offline_assets.configure_offline_scene",
        configure,
    )

    launched = drone_bootstrap.launch_environment("Isaac-Quadcopter-Direct-v0", 1)

    assert launched is native_env
    assert calls == ["configure", "make"]
    assert launched._flyg1_offline_scene_report == report
