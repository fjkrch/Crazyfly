#!/usr/bin/env python3
"""Deterministic held-out Crazyflie episode plans shared by every controller."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from drone_bootstrap import ROOT, canonical_sha256


SCENARIOS = (
    "FlyCrazyflie-WaypointReach-v0",
    "FlyCrazyflie-WaypointSwitch-v0",
    "FlyCrazyflie-GustRecovery-v0",
)
MAIN_MANIFEST = ROOT / "configs" / "experiments" / "crazyflie_eval_manifest_v1.json"
CONTROL_DT_S = 0.02
EPISODE_STEPS = 600
EVENT_STEPS = (150, 300, 450)
MIN_TARGET_DISTANCE_M = 0.75


def _rounded(values: list[float]) -> list[float]:
    return [round(float(value), 8) for value in values]


def _point(rng: random.Random) -> list[float]:
    return _rounded([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(0.5, 1.5)])


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _separated_point(rng: random.Random, reference: list[float]) -> list[float]:
    for _ in range(10_000):
        candidate = _point(rng)
        if _distance(candidate, reference) >= MIN_TARGET_DISTANCE_M:
            return candidate
    raise RuntimeError("Could not sample a target satisfying the declared minimum separation")


def _initial_state(rng: random.Random) -> dict[str, Any]:
    return {
        "position_relative_to_env_origin_m": _rounded([
            rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0.45, 0.65),
        ]),
        "yaw_rad": round(rng.uniform(-math.pi / 12.0, math.pi / 12.0), 8),
        "linear_velocity_world_m_s": _rounded([rng.uniform(-0.08, 0.08) for _ in range(3)]),
        "angular_velocity_body_rad_s": _rounded([rng.uniform(-0.15, 0.15) for _ in range(3)]),
    }


def _plan(seed: int, episode_id: int, scenario: str) -> dict[str, Any]:
    scenario_index = SCENARIOS.index(scenario)
    scenario_seed = seed * 1_000_003 + scenario_index * 100_003 + episode_id
    rng = random.Random(scenario_seed)
    initial = _initial_state(rng)
    initial_position = initial["position_relative_to_env_origin_m"]
    first_target = _separated_point(rng, initial_position)
    targets = [first_target]
    switches: list[dict[str, Any]] = []
    if scenario == "FlyCrazyflie-WaypointSwitch-v0":
        for event_index, step in enumerate(EVENT_STEPS, start=1):
            targets.append(_separated_point(rng, targets[-1]))
            switches.append({
                "event_index": event_index,
                "step": step,
                "time_s": round(step * CONTROL_DT_S, 8),
                "target_index": event_index,
            })
    gusts: list[dict[str, Any]] = []
    if scenario == "FlyCrazyflie-GustRecovery-v0":
        for event_index, step in enumerate(EVENT_STEPS, start=1):
            angle = rng.uniform(-math.pi, math.pi)
            gusts.append({
                "event_index": event_index,
                "start_step": step,
                "start_time_s": round(step * CONTROL_DT_S, 8),
                "duration_steps": 5,
                "duration_s": 0.10,
                "direction_world_xy": _rounded([math.cos(angle), math.sin(angle)]),
                "desired_mass_normalized_delta_velocity_m_s": 0.75,
                "application_frame": "world",
                "application_point": "Crazyflie body center of mass",
            })
    result = {
        "episode_id": episode_id,
        "scenario_seed": scenario_seed,
        "initial_state": initial,
        "targets_relative_to_env_origin_m": targets,
        "switches": switches,
        "gusts": gusts,
    }
    result["plan_sha256"] = canonical_sha256(result)
    return result


def generate_manifest(*, seed: int = 101, episodes_per_scenario: int = 16, label: str = "main") -> dict[str, Any]:
    if seed < 0 or episodes_per_scenario < 1:
        raise ValueError("seed must be nonnegative and episodes_per_scenario must be positive")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "evaluation_seed": seed,
        "episodes_per_scenario": episodes_per_scenario,
        "scenario_count": len(SCENARIOS),
        "episode_horizon_steps": EPISODE_STEPS,
        "episode_horizon_s": EPISODE_STEPS * CONTROL_DT_S,
        "control_dt_s": CONTROL_DT_S,
        "deterministic_action": True,
        "success": {"distance_m": 0.20, "speed_m_s": 0.25, "dwell_steps": 25, "dwell_s": 0.50},
        "recovery": {"dwell_steps": 25, "window_steps_after_gust": 100, "window_s_after_gust": 2.0},
        "minimum_target_separation_m": MIN_TARGET_DISTANCE_M,
        "scenarios": {
            scenario: [_plan(seed, episode_id, scenario) for episode_id in range(episodes_per_scenario)]
            for scenario in SCENARIOS
        },
    }
    manifest["manifest_id"] = canonical_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any], *, expected_episodes: int | None = None) -> dict[str, Any]:
    expected_manifest_fields = {
        "schema_version", "label", "evaluation_seed", "episodes_per_scenario",
        "scenario_count", "episode_horizon_steps", "episode_horizon_s", "control_dt_s",
        "deterministic_action", "success", "recovery", "minimum_target_separation_m",
        "scenarios", "manifest_id",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("Evaluation manifest fields differ from the closed version-1 schema")
    value = deepcopy(manifest)
    claimed = value.pop("manifest_id", None)
    actual = canonical_sha256(value)
    if claimed != actual:
        raise ValueError(f"Evaluation manifest checksum mismatch: claimed={claimed!r}, actual={actual}")
    if manifest.get("schema_version") != 1:
        raise ValueError("Evaluation manifest schema_version must be 1")
    if not isinstance(manifest.get("label"), str) or not manifest["label"]:
        raise ValueError("Evaluation manifest label must be non-empty")
    if manifest.get("evaluation_seed") != 101:
        raise ValueError("Version 1 evaluation seed must be 101")
    if manifest.get("scenario_count") != len(SCENARIOS):
        raise ValueError("Evaluation manifest must declare exactly three scenarios")
    fixed_contract = {
        "episode_horizon_steps": EPISODE_STEPS,
        "episode_horizon_s": EPISODE_STEPS * CONTROL_DT_S,
        "control_dt_s": CONTROL_DT_S,
        "deterministic_action": True,
        "success": {"distance_m": 0.20, "speed_m_s": 0.25, "dwell_steps": 25, "dwell_s": 0.50},
        "recovery": {"dwell_steps": 25, "window_steps_after_gust": 100, "window_s_after_gust": 2.0},
        "minimum_target_separation_m": MIN_TARGET_DISTANCE_M,
    }
    for field, expected in fixed_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Evaluation manifest {field} differs from the frozen version-1 contract")
    episodes = manifest.get("episodes_per_scenario")
    if type(episodes) is not int or episodes < 1 or expected_episodes is not None and episodes != expected_episodes:
        raise ValueError("Evaluation manifest episode count does not match the requested protocol")
    if tuple(manifest.get("scenarios", {}).keys()) != SCENARIOS:
        raise ValueError("Evaluation manifest must contain the three scenarios in the declared order")
    if expected_episodes == 16 and manifest["label"] != "main":
        raise ValueError("The 16-episode protocol must be labeled main")
    if expected_episodes == 5 and manifest["label"] != "lif_proof":
        raise ValueError("The five-episode protocol must be labeled lif_proof")
    if expected_episodes == 2 and manifest["label"] != "integration":
        raise ValueError("The two-episode protocol must be labeled integration")
    seen: set[str] = set()
    for scenario_index, scenario in enumerate(SCENARIOS):
        plans = manifest["scenarios"][scenario]
        if len(plans) != episodes:
            raise ValueError(f"{scenario} does not contain exactly {episodes} plans")
        for episode_id, plan in enumerate(plans):
            if not isinstance(plan, dict) or set(plan) != {
                "episode_id", "scenario_seed", "initial_state",
                "targets_relative_to_env_origin_m", "switches", "gusts", "plan_sha256",
            }:
                raise ValueError("Episode plan fields differ from the closed version-1 schema")
            if plan.get("episode_id") != episode_id:
                raise ValueError("Episode IDs must be contiguous and scenario-local")
            row = dict(plan)
            row_hash = row.pop("plan_sha256", None)
            if row_hash != canonical_sha256(row) or row_hash in seen:
                raise ValueError("Episode plan checksum is invalid or duplicated")
            seen.add(row_hash)
            # A checksum proves internal consistency, not that the held-out
            # plan is the frozen seed-derived plan.  Recreate that plan and
            # require byte-independent canonical equality so a caller cannot
            # mutate a target and simply recompute both checksum layers.
            if canonical_sha256(plan) != canonical_sha256(_plan(101, episode_id, scenario)):
                raise ValueError("Episode plan differs from the frozen seed-derived protocol")
            if plan.get("scenario_seed") != 101 * 1_000_003 + scenario_index * 100_003 + episode_id:
                raise ValueError("Episode scenario seed differs from the frozen derivation")
            initial_state = plan.get("initial_state")
            if not isinstance(initial_state, dict) or set(initial_state) != {
                "position_relative_to_env_origin_m", "yaw_rad",
                "linear_velocity_world_m_s", "angular_velocity_body_rad_s",
            }:
                raise ValueError("Episode initial_state must be an object")
            initial = initial_state.get("position_relative_to_env_origin_m")
            yaw = initial_state.get("yaw_rad")
            linear = initial_state.get("linear_velocity_world_m_s")
            angular = initial_state.get("angular_velocity_body_rad_s")
            if (
                not isinstance(initial, list) or len(initial) != 3
                or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in initial)
                or not (-0.15 <= initial[0] <= 0.15 and -0.15 <= initial[1] <= 0.15 and 0.45 <= initial[2] <= 0.65)
                or not isinstance(yaw, (int, float)) or not math.isfinite(yaw) or abs(yaw) > math.pi / 12 + 1.0e-8
                or not isinstance(linear, list) or len(linear) != 3
                or not all(isinstance(item, (int, float)) and math.isfinite(item) and abs(item) <= 0.08 for item in linear)
                or not isinstance(angular, list) or len(angular) != 3
                or not all(isinstance(item, (int, float)) and math.isfinite(item) and abs(item) <= 0.15 for item in angular)
            ):
                raise ValueError("Episode initial state violates the frozen bounded distribution")
            targets = plan["targets_relative_to_env_origin_m"]
            expected_targets = 4 if scenario == SCENARIOS[1] else 1
            if (
                not isinstance(targets, list) or len(targets) != expected_targets
                or any(
                    not isinstance(target, list) or len(target) != 3
                    or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in target)
                    or not (-2.0 <= target[0] <= 2.0 and -2.0 <= target[1] <= 2.0 and 0.5 <= target[2] <= 1.5)
                    for target in targets
                )
            ):
                raise ValueError("Episode targets violate the frozen workspace or count")
            if _distance(initial, targets[0]) < manifest["minimum_target_separation_m"]:
                raise ValueError("Initial target violates minimum separation")
            if any(_distance(a, b) < manifest["minimum_target_separation_m"] for a, b in zip(targets, targets[1:])):
                raise ValueError("Consecutive targets violate minimum separation")
            switches = plan.get("switches")
            gusts = plan.get("gusts")
            if scenario == SCENARIOS[1]:
                expected_switches = [
                    {"event_index": index + 1, "step": step, "time_s": step * CONTROL_DT_S, "target_index": index + 1}
                    for index, step in enumerate(EVENT_STEPS)
                ]
                if switches != expected_switches or gusts != []:
                    raise ValueError("Switch episode schedule differs from the frozen 3/6/9 s contract")
            elif scenario == SCENARIOS[2]:
                if switches != [] or not isinstance(gusts, list) or len(gusts) != 3:
                    raise ValueError("Gust episode must contain exactly three gusts and no switches")
                for index, (gust, step) in enumerate(zip(gusts, EVENT_STEPS, strict=True), start=1):
                    direction = gust.get("direction_world_xy") if isinstance(gust, dict) else None
                    if (
                        not isinstance(gust, dict)
                        or gust.get("event_index") != index
                        or gust.get("start_step") != step
                        or gust.get("start_time_s") != step * CONTROL_DT_S
                        or gust.get("duration_steps") != 5
                        or gust.get("duration_s") != 0.10
                        or gust.get("desired_mass_normalized_delta_velocity_m_s") != 0.75
                        or gust.get("application_frame") != "world"
                        or gust.get("application_point") != "Crazyflie body center of mass"
                        or not isinstance(direction, list) or len(direction) != 2
                        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in direction)
                        or not math.isclose(math.hypot(*direction), 1.0, rel_tol=0.0, abs_tol=2.0e-8)
                    ):
                        raise ValueError("Gust event differs from the frozen magnitude, timing, frame, or direction")
            elif switches != [] or gusts != []:
                raise ValueError("WaypointReach episodes cannot contain switches or gusts")
    return manifest


def load_protocol(name: str) -> dict[str, Any]:
    if name == "integration":
        return validate_manifest(generate_manifest(seed=101, episodes_per_scenario=2, label="integration"), expected_episodes=2)
    if name == "lif_proof":
        return validate_manifest(
            generate_manifest(seed=101, episodes_per_scenario=5, label="lif_proof"),
            expected_episodes=5,
        )
    if name == "main":
        try:
            manifest = json.loads(MAIN_MANIFEST.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read main evaluation manifest {MAIN_MANIFEST}: {exc}") from exc
        return validate_manifest(manifest, expected_episodes=16)
    path = Path(name).expanduser()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read evaluation manifest {path}: {exc}") from exc
    return validate_manifest(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--label", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = validate_manifest(generate_manifest(
            seed=args.seed, episodes_per_scenario=args.episodes, label=args.label
        ), expected_episodes=args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Refusing to replace unreadable existing manifest {args.output}: {exc}")
        if canonical_sha256(existing) != canonical_sha256(manifest):
            parser.error(
                f"Refusing to overwrite immutable evaluation manifest {args.output}; use a new versioned path"
            )
    else:
        temporary = args.output.with_name(args.output.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output.resolve()),
        "manifest_id": manifest["manifest_id"],
        "episodes_per_scenario": args.episodes,
        "total_episode_plans": args.episodes * len(SCENARIOS),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
