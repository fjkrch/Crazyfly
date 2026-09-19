#!/usr/bin/env python3
"""Fly one Crazyflie from held keys using one trained command-follow checkpoint.

The trained policy is the primary and unmodified action source.  A bounded
deterministic flight assist is used only for a recorded safety fallback.  This
entry point accepts only ``FlyCrazyflie-CommandFollow-v0`` checkpoints; old
WaypointReach, WaypointSwitch, and GustRecovery checkpoints fail before Isaac
Sim is launched.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from crazyflie_keyboard import (  # noqa: E402
    CONTROL_DT_S,
    CloseFollowCamera,
    Command,
    FlightAssist,
    FlightAssistConfig,
    HeldKeyState,
    IsaacKeyboard,
    NeuralBrainWindow,
    _quat_apply_inverse,
    _runtime_state_is_finite,
    command_from_pressed,
    prepare_asset_mirror,
)
from g1_fly_control.crazyflie.trained_keyboard import (  # noqa: E402
    CommandCheckpointSpec,
    RuntimeSpeeds,
    SafetyEnvelope,
    TASK_ID,
    TrainedActivityRecorder,
    arbitrate_action,
    initial_policy_state,
    inspect_command_checkpoint,
    load_command_controller,
    reconstruct_physical_observation,
    reset_policy_state,
    safety_reason,
    scaled_command_body,
)


BRAIN_UPDATE_STEPS = 5


class FlightSceneVisuals:
    """Non-physical scene references and a bounded live flight trail.

    These objects are USD point-instancer markers with no collision, rigid-body,
    or mass properties. They make translation and altitude obvious without
    changing environment dynamics or policy observations.
    """

    def __init__(
        self,
        ground_origin_w: torch.Tensor,
        *,
        trail_points: int,
        trail_enabled: bool,
    ) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        origin = ground_origin_w.detach().to(device="cpu", dtype=torch.float32).reshape(3)
        if not bool(torch.isfinite(origin).all()):
            raise FloatingPointError("scene-guide origin must be finite")
        self._origin = origin
        self._trail = deque(maxlen=int(trail_points))
        self._trail_enabled = bool(trail_enabled)
        self._updates = 0

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Flyg1SceneReferences",
            markers={
                "grid": sim_utils.CuboidCfg(
                    size=(0.035, 0.035, 0.008),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.70, 0.70, 0.70), roughness=0.9
                    ),
                ),
                "forward_plus_x_red": sim_utils.CylinderCfg(
                    radius=0.065,
                    height=0.80,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.08, 0.04), roughness=0.65
                    ),
                ),
                "back_minus_x_orange": sim_utils.CylinderCfg(
                    radius=0.065,
                    height=0.80,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.35, 0.02), roughness=0.65
                    ),
                ),
                "left_plus_y_green": sim_utils.CylinderCfg(
                    radius=0.065,
                    height=0.80,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.95, 0.15), roughness=0.65
                    ),
                ),
                "right_minus_y_blue": sim_utils.CylinderCfg(
                    radius=0.065,
                    height=0.80,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.25, 1.0), roughness=0.65
                    ),
                ),
                "corner": sim_utils.CuboidCfg(
                    size=(0.18, 0.18, 0.18),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.70, 0.12, 0.95), roughness=0.7
                    ),
                ),
            },
        )
        self._references = VisualizationMarkers(marker_cfg)
        reference_positions: list[list[float]] = []
        reference_indices: list[int] = []
        # A 0.5 m floor grid across the full visible command workspace.
        for x_index in range(-6, 7):
            for y_index in range(-6, 7):
                reference_positions.append(
                    [
                        float(origin[0]) + 0.5 * x_index,
                        float(origin[1]) + 0.5 * y_index,
                        float(origin[2]) + 0.012,
                    ]
                )
                reference_indices.append(0)
        # Direction-colored towers: +X red, -X orange, +Y green, -Y blue.
        for offset, marker_index in (
            ((2.5, 0.0, 0.4), 1),
            ((-2.5, 0.0, 0.4), 2),
            ((0.0, 2.5, 0.4), 3),
            ((0.0, -2.5, 0.4), 4),
        ):
            reference_positions.append(
                [float(origin[i]) + float(offset[i]) for i in range(3)]
            )
            reference_indices.append(marker_index)
        for x in (-2.5, 2.5):
            for y in (-2.5, 2.5):
                reference_positions.append(
                    [float(origin[0]) + x, float(origin[1]) + y, float(origin[2]) + 0.09]
                )
                reference_indices.append(5)
        self._reference_count = len(reference_positions)
        self._references.visualize(
            translations=torch.tensor(reference_positions, dtype=torch.float32),
            marker_indices=reference_indices,
        )

        target_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Flyg1CommandTarget",
            markers={
                "integrated_command_target": sim_utils.SphereCfg(
                    radius=0.075,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.75), roughness=0.5
                    ),
                )
            },
        )
        self._target = VisualizationMarkers(target_cfg)

        self._trail_markers = None
        if self._trail_enabled:
            trail_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/Flyg1FlightTrail",
                markers={
                    "trail": sim_utils.SphereCfg(
                        radius=0.022,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.05), roughness=0.55
                        ),
                    ),
                    "current": sim_utils.SphereCfg(
                        radius=0.045,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 1.0), roughness=0.35
                        ),
                    ),
                },
            )
            self._trail_markers = VisualizationMarkers(trail_cfg)

    @staticmethod
    def _position(value: torch.Tensor, name: str) -> torch.Tensor:
        position = value.detach().to(device="cpu", dtype=torch.float32).reshape(3)
        if not bool(torch.isfinite(position).all()):
            raise FloatingPointError(f"{name} must be a finite three-value position")
        return position

    def reset(self, position_w: torch.Tensor, target_w: torch.Tensor) -> None:
        self._trail.clear()
        self.update(position_w, target_w)

    def update(self, position_w: torch.Tensor, target_w: torch.Tensor) -> None:
        position = self._position(position_w, "flight-trail position")
        target = self._position(target_w, "command target")
        self._target.visualize(translations=target.unsqueeze(0))
        if self._trail_markers is not None:
            self._trail.append(position.clone())
            points = torch.stack(tuple(self._trail), dim=0)
            indices = [0] * max(0, points.shape[0] - 1) + [1]
            self._trail_markers.visualize(
                translations=points,
                marker_indices=indices,
            )
        self._updates += 1

    def audit(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "physics_effect": "none_visual_markers_only",
            "reference_marker_count": self._reference_count,
            "grid_spacing_m": 0.5,
            "direction_towers": {
                "+x_forward": "red",
                "-x_back": "orange",
                "+y_left": "green",
                "-y_right": "blue",
            },
            "command_target_marker": "green_cyan_sphere",
            "flight_trail_enabled": self._trail_enabled,
            "flight_trail_capacity": self._trail.maxlen,
            "flight_trail_points": len(self._trail),
            "update_count": self._updates,
        }


def scripted_input(step: int) -> tuple[frozenset[str], bool, str]:
    """Finite command sequence that includes a four-axis held-key command."""

    if step < 40:
        return frozenset(), False, "initial_hold"
    if step == 40:
        return frozenset(), True, "manual_reset"
    if step < 80:
        return frozenset(), False, "post_reset_hold"
    if step < 180:
        return frozenset({"W"}), False, "forward"
    if step < 280:
        return frozenset({"W", "A", "I", "J"}), False, "four_axis"
    if step < 360:
        return frozenset(), False, "release_hold"
    if step < 440:
        return frozenset({"S", "D", "Q", "L"}), False, "reverse_four_axis"
    return frozenset(), False, "final_hold"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Completed FlyCrazyflie-CommandFollow-v0 checkpoint",
    )
    parser.add_argument("--horizontal_speed", type=float, default=0.5)
    parser.add_argument("--vertical_speed", type=float, default=0.25)
    parser.add_argument("--yaw_rate", type=float, default=0.8)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Maximum 50 Hz control steps; zero is unlimited interactive flight",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Viewer-only: disable the 600-step time-limit reset and fly until Esc; "
            "hard safety/nonfinite resets remain enabled"
        ),
    )
    parser.add_argument(
        "--scripted_smoke",
        action="store_true",
        help="Use a finite held-key sequence instead of the viewer keyboard",
    )
    parser.add_argument(
        "--asset_mirror",
        type=Path,
        default=Path.home() / ".cache/flyg1/isaac-5.1-offline",
    )
    parser.add_argument("--offline_only", action="store_true")
    parser.add_argument("--no_brain_window", action="store_true")
    parser.add_argument("--no_follow_camera", action="store_true")
    parser.add_argument(
        "--no_scene_guides",
        action="store_true",
        help="Hide the floor grid, direction towers, command-target marker, and trail",
    )
    parser.add_argument(
        "--no_flight_trail",
        action="store_true",
        help="Keep the scene landmarks but hide the live yellow flight trail",
    )
    parser.add_argument(
        "--trail_points",
        type=int,
        default=160,
        help="Number of recent 10 Hz flight positions retained in the visible trail (20..500)",
    )
    parser.add_argument("--status_hz", type=float, default=4.0)
    return parser


def _validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, spec: CommandCheckpointSpec
) -> RuntimeSpeeds:
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.scripted_smoke and args.steps == 0:
        args.steps = 520
    if args.scripted_smoke and args.steps < 520:
        parser.error("--scripted_smoke requires at least 520 steps")
    if not args.scripted_smoke and args.headless:
        parser.error("interactive keyboard flight requires a visible viewer")
    if not math.isfinite(float(args.status_hz)) or args.status_hz <= 0.0:
        parser.error("--status_hz must be finite and positive")
    if not 20 <= args.trail_points <= 500:
        parser.error("--trail_points must be in [20, 500]")
    speeds = RuntimeSpeeds(
        horizontal_m_s=float(args.horizontal_speed),
        vertical_m_s=float(args.vertical_speed),
        yaw_rate_rad_s=float(args.yaw_rate),
    )
    try:
        speeds.validate(spec.command_follow_contract)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return speeds


def _current_source_set() -> str:
    from drone_bootstrap import canonical_sha256, source_hashes

    return canonical_sha256(source_hashes())


def _requested_command_tensor(
    held: HeldKeyState, speeds: RuntimeSpeeds, device: torch.device
) -> torch.Tensor:
    unit = command_from_pressed(held.pressed)
    values = scaled_command_body(unit.as_tuple(), speeds)
    return torch.tensor([values], dtype=torch.float32, device=device)


def _fallback_action(
    assist: FlightAssist,
    physical_observation: torch.Tensor,
    command_observation: torch.Tensor,
    effective_command_body: torch.Tensor,
) -> torch.Tensor:
    """Track the environment-owned command/hold target without integrating it."""

    desired = effective_command_body[:, 0:3].clone()
    error = command_observation[:, 9:12]
    desired[:, 0:2].add_(assist.config.horizontal_position_gain * error[:, 0:2])
    desired[:, 2].add_(assist.config.vertical_position_gain * error[:, 2])
    horizontal_norm = torch.linalg.vector_norm(desired[:, 0:2], dim=1, keepdim=True)
    desired[:, 0:2].mul_(
        torch.clamp(
            assist.config.maximum_horizontal_velocity_m_s
            / horizontal_norm.clamp_min(1.0e-9),
            max=1.0,
        )
    )
    desired[:, 2].clamp_(
        -assist.config.maximum_vertical_velocity_m_s,
        assist.config.maximum_vertical_velocity_m_s,
    )
    return assist.action(
        physical_observation,
        desired,
        float(effective_command_body[0, 3]),
    )


def _environment_contract(env: Any) -> None:
    required_methods = ("set_manual_command_body", "clear_manual_command")
    required_properties = (
        "requested_command_body",
        "effective_command_body",
        "command_target_position_w",
        "tracking_error_body",
        "command_tracking_contract",
    )
    missing = [
        name
        for name in (*required_methods, *required_properties)
        if not hasattr(env, name)
    ]
    if missing:
        raise RuntimeError(
            "command-follow environment lacks its manual API: " + ", ".join(missing)
        )


def _reset_runtime(
    env: Any,
    held: HeldKeyState,
    assist: FlightAssist,
    policy: torch.nn.Module,
    state: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Any]:
    del state
    observation, _ = env.reset()
    observation = env.clear_manual_command()
    env.episode_length_buf.zero_()
    assist.reset(env._robot.data.root_pos_w.detach().clone())
    held.clear_movement()
    return observation, reset_policy_state(policy, None, device=device)


def _run(
    args: argparse.Namespace,
    simulation_app: Any,
    local_usd: Path,
    asset_report: dict[str, Any],
    spec: CommandCheckpointSpec,
    speeds: RuntimeSpeeds,
    current_source_set: str,
) -> int:
    import gymnasium as gym
    import isaaclab_tasks.direct.quadcopter  # noqa: F401
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    import g1_fly_control.tasks.crazyflie  # noqa: F401
    from drone_bootstrap import selected_env_cfg

    cfg = selected_env_cfg(TASK_ID, 1, contract_profile="command_v1")
    cfg.debug_vis = False
    cfg.scene.num_envs = 1
    cfg.scene.clone_in_fabric = False
    cfg.robot.spawn.usd_path = str(local_usd)
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=0,
        size=(20.0, 20.0),
        num_rows=1,
        num_cols=1,
        border_width=0.0,
        curriculum=False,
        use_cache=False,
        sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
    )
    cfg.terrain.use_terrain_origins = True
    cfg.sim.device = str(args.device)

    env = None
    keyboard = None
    camera = None
    scene_visuals = None
    brain_window = None
    activity = None
    policy = None
    final_observation: dict[str, torch.Tensor] | None = None
    brain_window_audit: dict[str, Any] = {
        "enabled": False,
        "separate_process": False,
    }
    held = HeldKeyState()
    executed_steps = 0
    policy_action_steps = 0
    fallback_action_steps = 0
    fallback_reasons: Counter[str] = Counter()
    manual_reset_count = 0
    termination_reset_count = 0
    truncation_reset_count = 0
    invalid_state_reset_count = 0
    maximum_action_abs = 0.0
    phase_counts: Counter[str] = Counter()
    try:
        env = gym.make(
            TASK_ID,
            cfg=cfg,
            render_mode=None if args.headless else "human",
        ).unwrapped
        _environment_contract(env)
        if args.continuous:
            # Keep the authenticated 600-step task configuration untouched and
            # suppress only the viewer instance's time-limit signal. DirectRLEnv
            # still receives every real hard-failure termination, so corrupt or
            # unsafe state cannot be hidden by this display-only mode.
            original_get_dones = env._get_dones

            def continuous_viewer_dones() -> tuple[torch.Tensor, torch.Tensor]:
                terminated, time_out = original_get_dones()
                return terminated, torch.zeros_like(time_out)

            env._get_dones = continuous_viewer_dones
        observation, _ = env.reset(seed=0)
        observation = env.clear_manual_command()
        env.episode_length_buf.zero_()
        final_observation = observation
        device = torch.device(env.device)
        if not args.headless and not args.no_scene_guides:
            scene_visuals = FlightSceneVisuals(
                env._terrain.env_origins[0],
                trail_points=args.trail_points,
                trail_enabled=not args.no_flight_trail,
            )
            scene_visuals.reset(
                env._robot.data.root_pos_w[0], env.command_target_position_w[0]
            )
        policy, normalizer, controller_report = load_command_controller(
            spec,
            device=device,
            expected_source_set=current_source_set,
        )
        state = initial_policy_state(policy, 1, device)
        activity = TrainedActivityRecorder(policy, spec)
        assist = FlightAssist(
            FlightAssistConfig(
                horizontal_speed_m_s=max(speeds.horizontal_m_s, 1.0e-9),
                vertical_speed_m_s=max(speeds.vertical_m_s, 1.0e-9),
                yaw_rate_rad_s=max(speeds.yaw_rate_rad_s, 1.0e-9),
            ),
            env._robot.data.root_pos_w.detach().clone(),
        )
        envelope = SafetyEnvelope()

        if not args.headless and not args.no_brain_window:
            brain_window = NeuralBrainWindow(
                spec.controller,
                activity.roles,
                trained_controls_action=True,
                controller_label=spec.controller,
            )
            brain_window_audit = {"enabled": True, **brain_window.audit()}
        if not args.scripted_smoke:
            keyboard = IsaacKeyboard(held)
            print(
                "\nCrazyflie TRAINED command-follow control (focus the Isaac viewer)\n"
                "  W/S forward/back | A/D left/right | I or E up | Q down\n"
                "  J/L yaw left/right | H/Space hold | R reset | Esc exit\n"
                "  Simultaneous commands such as W+A+I+J are supported.\n"
                "  Scene: +X red | -X orange | +Y green | -Y blue; "
                "green sphere=command target; yellow dots=flight trail.\n"
                f"  Controller: {spec.controller}; checkpoint: {spec.path}\n"
                "  TRAINED POLICY CONTROLS ACTION; deterministic assist is fallback only.\n"
                f"  Speeds: horizontal={speeds.horizontal_m_s:.3f} m/s, "
                f"vertical={speeds.vertical_m_s:.3f} m/s, "
                f"yaw={speeds.yaw_rate_rad_s:.3f} rad/s.\n"
            )
        if not args.headless:
            root = env._robot.data.root_pos_w[0].detach().cpu().tolist()
            env.sim.set_camera_view(
                eye=[root[0] - 1.35, root[1] - 1.35, root[2] + 0.65],
                target=root,
            )
            if not args.no_follow_camera:
                camera = CloseFollowCamera(env, (-1.35, -1.35, 0.65))

        next_status = time.monotonic()
        next_deadline = time.monotonic()
        current_phase = "interactive"
        while not held.quit_requested:
            if args.steps and executed_steps >= args.steps:
                break
            if not args.scripted_smoke and not simulation_app.is_running():
                break
            if args.scripted_smoke:
                keys, request_reset, current_phase = scripted_input(executed_steps)
                held.clear_movement()
                for key in keys:
                    held.press(key)
                if request_reset:
                    held.reset_requested = True
            if held.consume_reset():
                observation, state = _reset_runtime(
                    env, held, assist, policy, state, device
                )
                manual_reset_count += 1
                if scene_visuals is not None:
                    scene_visuals.reset(
                        env._robot.data.root_pos_w[0], env.command_target_position_w[0]
                    )

            requested_command = _requested_command_tensor(held, speeds, device)
            observation = env.set_manual_command_body(requested_command)
            command_observation = observation["policy"]
            root_position = env._robot.data.root_pos_w.detach().clone()
            root_quaternion = env._robot.data.root_quat_w.detach().clone()
            effective_command = env.effective_command_body.detach().clone()
            physical_observation = reconstruct_physical_observation(
                command_observation, effective_command
            )
            if not _runtime_state_is_finite(
                command_observation,
                physical_observation,
                root_position,
                root_quaternion,
                effective_command,
                env._robot.data.root_lin_vel_b,
                env._robot.data.root_ang_vel_b,
            ):
                invalid_state_reset_count += 1
                observation, state = _reset_runtime(
                    env, held, assist, policy, state, device
                )
                final_observation = observation
                continue

            unsafe = safety_reason(
                physical_observation,
                root_position,
                assist.origin_w,
                envelope,
            )
            if unsafe is not None:
                observation = env.clear_manual_command()
                command_observation = observation["policy"]
                effective_command = env.effective_command_body.detach().clone()
                physical_observation = reconstruct_physical_observation(
                    command_observation, effective_command
                )
                held.clear_movement()

            fallback = _fallback_action(
                assist,
                physical_observation,
                command_observation,
                effective_command,
            )
            candidate_action = None
            candidate_state = state
            policy_error = None
            activity.begin_step()
            if unsafe is None:
                try:
                    normalized = normalizer.normalize(command_observation)
                    with torch.no_grad():
                        output = policy.act(normalized, state, deterministic=True)
                    candidate_action = output.action
                    candidate_state = output.state
                except Exception as exc:
                    policy_error = f"{type(exc).__name__}:{exc}"
            decision = arbitrate_action(
                candidate_action,
                candidate_state,
                fallback,
                runtime_safety_reason=unsafe,
                policy_error=policy_error,
            )
            if decision.source == "trained_policy":
                state = candidate_state
                activity.update(state)
                policy_action_steps += 1
            else:
                state = reset_policy_state(policy, state, device=device)
                fallback_action_steps += 1
                fallback_reasons[str(decision.fallback_reason)] += 1
            action = decision.action
            maximum_action_abs = max(maximum_action_abs, float(action.abs().max()))
            next_observation, _reward_ignored, terminated, truncated, _info = env.step(action)
            executed_steps += 1
            phase_counts[current_phase] += 1
            observation = next_observation
            final_observation = observation

            if scene_visuals is not None and executed_steps % BRAIN_UPDATE_STEPS == 0:
                scene_visuals.update(
                    env._robot.data.root_pos_w[0], env.command_target_position_w[0]
                )

            if (
                brain_window is not None
                and decision.source == "trained_policy"
                and executed_steps % BRAIN_UPDATE_STEPS == 0
            ):
                brain_window.show(activity.brain_packet(executed_steps, tuple(held.pressed)))

            if not _runtime_state_is_finite(
                observation["policy"],
                env._robot.data.root_pos_w,
                env._robot.data.root_quat_w,
                env._robot.data.root_lin_vel_b,
                env._robot.data.root_ang_vel_b,
            ):
                invalid_state_reset_count += 1
                observation, state = _reset_runtime(
                    env, held, assist, policy, state, device
                )
                final_observation = observation
                continue
            if bool((terminated | truncated).any()):
                termination_reset_count += int(terminated.sum())
                truncation_reset_count += int(truncated.sum())
                observation = env.clear_manual_command()
                env.episode_length_buf.zero_()
                assist.reset(env._robot.data.root_pos_w.detach().clone())
                state = reset_policy_state(policy, state, device=device)
                held.clear_movement()
                final_observation = observation
                if scene_visuals is not None:
                    scene_visuals.reset(
                        env._robot.data.root_pos_w[0], env.command_target_position_w[0]
                    )

            if camera is not None and executed_steps % 5 == 0:
                camera.update()
            now = time.monotonic()
            if not args.headless and now >= next_status:
                position = env._robot.data.root_pos_w[0].detach().cpu().tolist()
                requested = env.requested_command_body[0].detach().cpu().tolist()
                effective = env.effective_command_body[0].detach().cpu().tolist()
                tracking = env.tracking_error_body[0].detach().cpu().tolist()
                print(
                    "\r"
                    f"keys={'+'.join(sorted(held.pressed)) or '-':<11} "
                    f"src={decision.source:<27} "
                    f"req=({requested[0]:+.2f},{requested[1]:+.2f},{requested[2]:+.2f},"
                    f"{requested[3]:+.2f}) eff=({effective[0]:+.2f},{effective[1]:+.2f},"
                    f"{effective[2]:+.2f},{effective[3]:+.2f}) "
                    f"err=({tracking[0]:+.2f},{tracking[1]:+.2f},{tracking[2]:+.2f},"
                    f"{tracking[3]:+.2f}) pos=({position[0]:+.2f},{position[1]:+.2f},"
                    f"{position[2]:+.2f}) fallback={fallback_action_steps}   ",
                    end="",
                    flush=True,
                )
                next_status = now + 1.0 / float(args.status_hz)
            if not args.headless:
                next_deadline += CONTROL_DT_S
                delay = next_deadline - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -0.5:
                    next_deadline = time.monotonic()

        if not args.headless:
            print()
        if brain_window is not None:
            brain_window_audit = {"enabled": True, **brain_window.close()}
            brain_window = None
        final_finite = bool(
            final_observation is not None
            and _runtime_state_is_finite(
                final_observation["policy"],
                env._robot.data.root_pos_w,
                env._robot.data.root_quat_w,
                env._robot.data.root_lin_vel_b,
                env._robot.data.root_ang_vel_b,
            )
        )
        failures: list[str] = []
        if not final_finite or invalid_state_reset_count:
            failures.append("runtime state was nonfinite")
        if maximum_action_abs > 1.0 + 1.0e-6:
            failures.append("an action exceeded native bounds")
        if args.scripted_smoke and policy_action_steps == 0:
            failures.append("trained policy never controlled an action")
        if args.scripted_smoke and fallback_action_steps:
            failures.append("scripted trained-policy smoke required a safety fallback")
        report = {
            "schema_version": 1,
            "status": "PASS" if not failures else "FAIL",
            "mode": "scripted_smoke" if args.scripted_smoke else "interactive_keyboard",
            "task": TASK_ID,
            "training": False,
            "checkpoint_loaded": True,
            "checkpoint": str(spec.path),
            "checkpoint_sha256": spec.sha256,
            "checkpoint_source_set": spec.fingerprints.get("source_set"),
            "current_source_set": current_source_set,
            "controller": spec.controller,
            "controller_report": controller_report,
            "command_follow_contract": spec.command_follow_contract,
            "runtime_speeds": {
                "horizontal_m_s": speeds.horizontal_m_s,
                "vertical_m_s": speeds.vertical_m_s,
                "yaw_rate_rad_s": speeds.yaw_rate_rad_s,
            },
            "continuous_viewer": {
                "enabled": bool(args.continuous),
                "time_limit_reset_disabled": bool(args.continuous),
                "authenticated_task_timeout_steps": int(env.max_episode_length),
                "viewer_timeout_signal": "suppressed" if args.continuous else "task_default",
                "hard_safety_and_nonfinite_resets_enabled": True,
                "manual_reset_key_enabled": True,
            },
            "action_source": "trained_policy_primary_deterministic_assist_fallback",
            "reward_used_for_control": False,
            "neural_monitor_controls_action": True,
            "steps": executed_steps,
            "policy_action_steps": policy_action_steps,
            "fallback_action_steps": fallback_action_steps,
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "step_accounting_passed": (
                policy_action_steps + fallback_action_steps == executed_steps
            ),
            "manual_reset_count": manual_reset_count,
            "termination_reset_count": termination_reset_count,
            "truncation_reset_count": truncation_reset_count,
            "invalid_state_reset_count": invalid_state_reset_count,
            "maximum_action_abs": maximum_action_abs,
            "final_state_finite": final_finite,
            "phase_steps": dict(phase_counts),
            "trained_activity_final": activity.last_summary,
            "trained_activity_roles": dict(Counter(activity.roles)),
            "brain_window": brain_window_audit,
            "scene_visuals": (
                scene_visuals.audit()
                if scene_visuals is not None
                else {"enabled": False, "physics_effect": "none"}
            ),
            "asset": asset_report,
            "ground": "local_procedural_flat_mesh",
            "follow_camera": (
                camera.audit()
                if camera is not None
                else {"enabled": False, "update_count": 0, "last_pose": None}
            ),
            "failures": failures,
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0 if not failures else 1
    finally:
        if activity is not None:
            activity.close()
        if brain_window is not None:
            brain_window.close()
        if keyboard is not None:
            keyboard.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = _parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        spec = inspect_command_checkpoint(args.checkpoint)
    except Exception as exc:
        parser.error(str(exc))
    speeds = _validate_args(parser, args, spec)
    try:
        current_source_set = _current_source_set()
        if spec.fingerprints.get("source_set") != current_source_set:
            raise ValueError(
                "checkpoint source-set fingerprint differs from current runtime; "
                "use the exact source that trained this checkpoint"
            )
        local_usd, asset_report = prepare_asset_mirror(
            args.asset_mirror, allow_download=not args.offline_only
        )
    except Exception as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "RESOLVED",
                "task": TASK_ID,
                "controller": spec.controller,
                "checkpoint": str(spec.path),
                "checkpoint_sha256": spec.sha256,
                "runtime_speeds": vars(speeds),
                "trained_policy_controls_action": True,
                "deterministic_assist": "fallback_only",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    simulation_app = AppLauncher(args).app
    try:
        return _run(
            args,
            simulation_app,
            local_usd,
            asset_report,
            spec,
            speeds,
            current_source_set,
        )
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
