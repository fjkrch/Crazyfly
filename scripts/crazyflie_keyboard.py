#!/usr/bin/env python3
"""Fly one Crazyflie directly from held keyboard commands.

This tool is deliberately outside the ``drone_*.py`` experiment namespace. It
does not load a checkpoint, train a policy, use reward for control, or mutate a
matrix artifact.  A deterministic flight-assist controller translates desired
body velocity into the native Crazyflie aggregate-wrench action.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import traceback
from typing import Any, Iterable
from urllib.request import urlopen

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_fly_control"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

TASK = "Isaac-Quadcopter-Direct-v0"
CONTROL_DT_S = 0.02
HOVER_ACTION = 2.0 / 1.9 - 1.0
MOVEMENT_KEYS = frozenset({"W", "S", "A", "D", "I", "E", "Q", "J", "L"})
UP_KEYS = frozenset({"I", "E"})
HOVER_KEYS = frozenset({"H", "SPACE", "SPACE_BAR"})
QUIT_KEYS = frozenset({"ESC", "ESCAPE"})
BRAIN_WINDOW_WIDTH = 920
BRAIN_WINDOW_HEIGHT = 500
BRAIN_UPDATE_STEPS = 5
BRAIN_ACTIVITY_WINDOW_STEPS = 25

ASSET_BASE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie"
)


@dataclass(frozen=True)
class AssetFile:
    relative_path: str
    sha256: str
    omniverse_cache_path: Path
    url: str


ASSET_FILES = (
    AssetFile(
        relative_path="Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        sha256="7372ac0786312c47a92603da3fcd412d560b21c3757a8f0d5e7c2bfb2233d2f4",
        omniverse_cache_path=Path(
            "/home/chayanin/.cache/ov/client/https/"
            "2b571443ab2d99c63cee2eaf5e286f5bc965493d891b4a37a71f2e172d219181.usd"
        ),
        url=f"{ASSET_BASE_URL}/cf2x.usd",
    ),
    AssetFile(
        relative_path=(
            "Isaac/Robots/Bitcraze/Crazyflie/configuration/"
            "cf2x_robot_schema.usd"
        ),
        sha256="c7a63f78ce3937c25cd05936ee73348bfdbbd0a10e82c0b8a37250730a3cbb9c",
        omniverse_cache_path=Path(
            "/home/chayanin/.cache/ov/client/https/"
            "ffd8ae223607939dc51cf85caa8d47516cd88090bb7841fc9c95e67defbb6c12.usd"
        ),
        url=f"{ASSET_BASE_URL}/configuration/cf2x_robot_schema.usd",
    ),
)


@dataclass(frozen=True)
class Command:
    forward: float
    left: float
    up: float
    yaw_left: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.forward, self.left, self.up, self.yaw_left


@dataclass(frozen=True)
class FlightAssistConfig:
    horizontal_speed_m_s: float = 0.55
    vertical_speed_m_s: float = 0.35
    yaw_rate_rad_s: float = 0.8
    horizontal_position_gain: float = 0.65
    vertical_position_gain: float = 0.8
    horizontal_velocity_gain: float = 0.04
    vertical_velocity_gain: float = 0.18
    tilt_gain: float = 0.08
    angular_rate_gain: float = 0.02
    yaw_rate_gain: float = 0.018
    maximum_horizontal_velocity_m_s: float = 0.9
    maximum_vertical_velocity_m_s: float = 0.55
    maximum_moment_action: float = 0.14
    minimum_height_m: float = 0.30
    maximum_height_m: float = 1.70
    maximum_horizontal_offset_m: float = 4.0

    def validate(self) -> None:
        values = tuple(vars(self).values())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("flight-assist configuration must be finite")
        if any(value <= 0.0 for value in values):
            raise ValueError("flight-assist configuration values must be positive")
        if self.minimum_height_m >= self.maximum_height_m:
            raise ValueError("minimum height must be below maximum height")
        if self.maximum_moment_action > 1.0:
            raise ValueError("maximum moment action cannot exceed one")


class HeldKeyState:
    """Event-safe held-key state; repeated presses never accumulate."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self.reset_requested = False
        self.quit_requested = False

    @property
    def pressed(self) -> frozenset[str]:
        return frozenset(self._pressed)

    def press(self, key: str) -> None:
        name = str(key).upper()
        if name in MOVEMENT_KEYS:
            self._pressed.add(name)
        elif name in HOVER_KEYS:
            self.clear_movement()
        elif name == "R":
            self.clear_movement()
            self.reset_requested = True
        elif name in QUIT_KEYS:
            self.clear_movement()
            self.quit_requested = True

    def release(self, key: str) -> None:
        self._pressed.discard(str(key).upper())

    def clear_movement(self) -> None:
        self._pressed.clear()

    def consume_reset(self) -> bool:
        requested = self.reset_requested
        self.reset_requested = False
        return requested


def command_from_pressed(keys: Iterable[str]) -> Command:
    """Convert held keys into one normalized multi-axis command."""

    held = {str(key).upper() for key in keys}
    forward = float("W" in held) - float("S" in held)
    left = float("A" in held) - float("D" in held)
    horizontal_norm = math.hypot(forward, left)
    if horizontal_norm > 1.0:
        forward /= horizontal_norm
        left /= horizontal_norm
    up = float(bool(held & UP_KEYS)) - float("Q" in held)
    yaw_left = float("J" in held) - float("L" in held)
    return Command(forward, left, up, yaw_left)


def _quat_apply(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of vectors by normalized WXYZ quaternions."""

    if quaternion_wxyz.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector widths must be four/three")
    q_vec = quaternion_wxyz[..., 1:4]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (
        quaternion_wxyz[..., 0:1] * uv + uuv
    )


def _quat_apply_inverse(
    quaternion_wxyz: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    conjugate = torch.cat(
        (quaternion_wxyz[..., 0:1], -quaternion_wxyz[..., 1:4]), dim=-1
    )
    return _quat_apply(conjugate, vector)


class FlightAssist:
    """Position-holding outer loop plus bounded native wrench controller."""

    def __init__(self, config: FlightAssistConfig, root_position_w: torch.Tensor):
        config.validate()
        self.config = config
        self.reset(root_position_w)

    def reset(self, root_position_w: torch.Tensor) -> None:
        if root_position_w.shape != (1, 3):
            raise ValueError("manual flight supports exactly one environment")
        self.origin_w = root_position_w.detach().clone()
        self.target_w = root_position_w.detach().clone()
        self.target_w[:, 2].clamp_(
            self.config.minimum_height_m, self.config.maximum_height_m
        )

    def desired_velocity_body(
        self,
        root_position_w: torch.Tensor,
        root_quaternion_w: torch.Tensor,
        command: Command,
        dt_s: float,
    ) -> torch.Tensor:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("control dt must be finite and positive")
        body_command = root_position_w.new_tensor(
            [[
                command.forward * self.config.horizontal_speed_m_s,
                command.left * self.config.horizontal_speed_m_s,
                0.0,
            ]]
        )
        world_command = _quat_apply(root_quaternion_w, body_command)
        world_command[:, 2] = command.up * self.config.vertical_speed_m_s
        previous_target_w = self.target_w.clone()
        self.target_w.add_(world_command * dt_s)
        xy_low = self.origin_w[:, :2] - self.config.maximum_horizontal_offset_m
        xy_high = self.origin_w[:, :2] + self.config.maximum_horizontal_offset_m
        self.target_w[:, :2] = torch.maximum(
            torch.minimum(self.target_w[:, :2], xy_high), xy_low
        )
        self.target_w[:, 2].clamp_(
            self.config.minimum_height_m, self.config.maximum_height_m
        )

        error_body = _quat_apply_inverse(
            root_quaternion_w, self.target_w - root_position_w
        )
        # Derive feed-forward velocity from the *clamped* setpoint movement.
        # At any safety boundary it therefore becomes zero instead of pushing
        # the vehicle through the bound while a key remains held.
        target_velocity_body = _quat_apply_inverse(
            root_quaternion_w, (self.target_w - previous_target_w) / dt_s
        )
        desired = target_velocity_body
        desired[:, :2].add_(
            self.config.horizontal_position_gain * error_body[:, :2]
        )
        desired[:, 2].add_(
            self.config.vertical_position_gain * error_body[:, 2]
        )
        horizontal_norm = torch.linalg.vector_norm(desired[:, :2], dim=1, keepdim=True)
        horizontal_scale = torch.clamp(
            self.config.maximum_horizontal_velocity_m_s
            / horizontal_norm.clamp_min(1.0e-9),
            max=1.0,
        )
        desired[:, :2].mul_(horizontal_scale)
        desired[:, 2].clamp_(
            -self.config.maximum_vertical_velocity_m_s,
            self.config.maximum_vertical_velocity_m_s,
        )
        return desired

    def action(
        self,
        physical_observation: torch.Tensor,
        desired_velocity_body: torch.Tensor,
        desired_yaw_rate_rad_s: float,
    ) -> torch.Tensor:
        if physical_observation.shape != (1, 12):
            raise ValueError("native observation must have shape [1, 12]")
        if desired_velocity_body.shape != (1, 3):
            raise ValueError("desired body velocity must have shape [1, 3]")
        if not torch.isfinite(physical_observation).all():
            raise FloatingPointError("nonfinite observation")
        if not torch.isfinite(desired_velocity_body).all():
            raise FloatingPointError("nonfinite desired velocity")
        velocity = physical_observation[:, 0:3]
        angular_velocity = physical_observation[:, 3:6]
        gravity = physical_observation[:, 6:9]
        velocity_error = desired_velocity_body - velocity

        collective = (
            HOVER_ACTION
            + self.config.vertical_velocity_gain * velocity_error[:, 2]
            + 0.20 * (1.0 + gravity[:, 2])
        )
        roll = (
            self.config.tilt_gain * gravity[:, 1]
            - self.config.angular_rate_gain * angular_velocity[:, 0]
            - self.config.horizontal_velocity_gain * velocity_error[:, 1]
        )
        pitch = (
            -self.config.tilt_gain * gravity[:, 0]
            - self.config.angular_rate_gain * angular_velocity[:, 1]
            + self.config.horizontal_velocity_gain * velocity_error[:, 0]
        )
        yaw = -self.config.yaw_rate_gain * (
            angular_velocity[:, 2] - float(desired_yaw_rate_rad_s)
        )
        action = torch.stack((collective, roll, pitch, yaw), dim=1)
        action[:, 1:4].clamp_(
            -self.config.maximum_moment_action,
            self.config.maximum_moment_action,
        )
        action.clamp_(-1.0, 1.0)
        if not torch.isfinite(action).all():
            raise FloatingPointError("flight assist produced nonfinite action")
        return action


def scripted_input(step: int) -> tuple[frozenset[str], bool, str]:
    """Deterministic smoke sequence using the interactive held-key semantics."""

    if step < 80:
        return frozenset(), False, "idle_before_reset"
    if step == 80:
        return frozenset(), True, "manual_reset"
    if step < 100:
        return frozenset(), False, "idle_after_reset"
    if step < 220:
        return frozenset({"W"}), False, "forward"
    if step < 300:
        return frozenset(), False, "hover_after_forward"
    if step < 420:
        return frozenset({"W", "A"}), False, "forward_left"
    if step < 500:
        return frozenset(), False, "hover_after_diagonal"
    if step < 600:
        return frozenset({"I"}), False, "up"
    if step < 700:
        return frozenset({"Q"}), False, "down"
    return frozenset(), False, "final_hover"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    temporary = destination.with_name(destination.name + ".download")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
            stream.flush()
            os.fsync(stream.fileno())
        actual = _sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"downloaded asset hash mismatch: expected={expected_sha256}, actual={actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_asset_mirror(
    mirror_root: Path, *, allow_download: bool
) -> tuple[Path, dict[str, Any]]:
    """Materialize the pinned top asset and its relative schema dependency."""

    root = mirror_root.expanduser().resolve()
    files: list[dict[str, Any]] = []
    for spec in ASSET_FILES:
        destination = root / spec.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = "existing_mirror"
        if not destination.is_file() or _sha256_file(destination) != spec.sha256:
            destination.unlink(missing_ok=True)
            if (
                spec.omniverse_cache_path.is_file()
                and _sha256_file(spec.omniverse_cache_path) == spec.sha256
            ):
                temporary = destination.with_name(destination.name + ".copy")
                shutil.copy2(spec.omniverse_cache_path, temporary)
                os.replace(temporary, destination)
                source = "verified_omniverse_cache"
            elif allow_download:
                _download_verified(spec.url, destination, spec.sha256)
                source = "verified_online_fetch"
            else:
                raise FileNotFoundError(
                    "Pinned Crazyflie asset is absent from both the offline mirror "
                    f"and verified Omniverse cache: {spec.relative_path}"
                )
        actual = _sha256_file(destination)
        if actual != spec.sha256:
            raise RuntimeError(
                f"offline asset hash mismatch for {destination}: {actual}"
            )
        files.append(
            {
                "path": str(destination),
                "sha256": actual,
                "source": source,
            }
        )
    top = root / ASSET_FILES[0].relative_path
    return top, {"mirror_root": str(root), "files": files}


class IsaacKeyboard:
    """Thin low-level Kit subscriber preserving independent held keys."""

    def __init__(self, state: HeldKeyState) -> None:
        import carb
        import omni.appwindow

        self._carb = carb
        self._state = state
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac viewer window is unavailable")
        self._keyboard = app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )

    def _on_event(self, event: Any, *_: Any) -> bool:
        name = str(event.input.name).upper()
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self._state.press(name)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._state.release(name)
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._subscription
            )
            self._subscription = None


class CloseFollowCamera:
    """Keep the stock viewport camera close to and aimed at the drone."""

    def __init__(self, env: Any, offset: tuple[float, float, float]) -> None:
        self._env = env
        self._offset = offset
        self.update_count = 0
        self.last_pose: dict[str, list[float]] | None = None
        self.update()

    def update(self) -> None:
        root = self._env._robot.data.root_pos_w[0].detach().cpu().tolist()
        eye = [root[index] + self._offset[index] for index in range(3)]
        self._env.sim.set_camera_view(eye=eye, target=root)
        self.update_count += 1
        self.last_pose = {"eye_w_m": eye, "target_w_m": root}

    def audit(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "update_count": self.update_count,
            "last_pose": self.last_pose,
        }


_ROLE_LABELS = {
    "descending_input": "Descending input",
    "wing_sensory_input": "Wing sensory",
    "vnc_interneuron": "Thoracic intrinsic",
    "wing_motor_output": "Wing motor",
    "leg:descending_input": "Leg descending",
    "leg:sensory_input": "Leg sensory",
    "leg:vnc_interneuron": "Leg VNC intrinsic",
    "leg:motor_output": "Leg motor",
    "wing:descending_input": "Wing descending",
    "wing:wing_sensory_input": "Wing sensory",
    "wing:vnc_interneuron": "Thoracic intrinsic",
    "wing:wing_motor_output": "Wing motor",
}
_ROLE_COLORS = {
    "descending_input": "#38a9ff",
    "wing_sensory_input": "#f5c84c",
    "vnc_interneuron": "#50e678",
    "wing_motor_output": "#ff66cc",
    "leg:descending_input": "#9c7cff",
    "leg:sensory_input": "#ff9e4a",
    "leg:vnc_interneuron": "#e8734a",
    "leg:motor_output": "#ff4d6d",
    "wing:descending_input": "#38a9ff",
    "wing:wing_sensory_input": "#f5c84c",
    "wing:vnc_interneuron": "#50e678",
    "wing:wing_motor_output": "#ff66cc",
}


def _brain_window_process_main(
    frame_queue: Any,
    status_queue: Any,
    mode: str,
    roles: tuple[str, ...],
    trained_controls_action: bool = False,
    controller_label: str | None = None,
) -> None:
    """Own the neural activity window in a process isolated from Isaac UI."""

    root = None
    try:
        import tkinter as tk

        if trained_controls_action:
            title = f"Crazyflie Trained {controller_label or mode} Activity"
        else:
            title = (
                "Crazyflie Wing + Thoracic Neural Activity"
                if mode == "wing_thoracic"
                else "Crazyflie Leg + Wing + Thoracic Neural Activity"
            )
        root = tk.Tk()
        root.title(title)
        root.geometry(f"{BRAIN_WINDOW_WIDTH}x{BRAIN_WINDOW_HEIGHT}")
        root.minsize(BRAIN_WINDOW_WIDTH, BRAIN_WINDOW_HEIGHT)
        canvas = tk.Canvas(
            root,
            width=BRAIN_WINDOW_WIDTH,
            height=BRAIN_WINDOW_HEIGHT,
            background="#080c12",
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        close_requested = False

        def request_close() -> None:
            nonlocal close_requested
            close_requested = True

        root.protocol("WM_DELETE_WINDOW", request_close)
        root.update_idletasks()
        root.update()
        status_queue.put(
            {
                "status": "ready",
                "backend": f"tk-{root.tk.call('info', 'patchlevel')}",
            }
        )
        no_frame = object()
        role_counts = Counter(roles)
        while not close_requested:
            packet = no_frame
            try:
                packet = frame_queue.get(timeout=0.02)
            except queue.Empty:
                pass
            if packet is None:
                close_requested = True
            elif packet is not no_frame:
                canvas.delete("all")
                canvas.create_text(
                    22,
                    16,
                    anchor="nw",
                    fill="#f4f7fb",
                    font=("DejaVu Sans", 18, "bold"),
                    text=title,
                )
                canvas.create_text(
                    22,
                    52,
                    anchor="nw",
                    fill="#ffb347",
                    font=("DejaVu Sans", 11, "bold"),
                    text=(
                        "TRAINED POLICY CONTROLS ACTION — deterministic assist is fallback only"
                        if trained_controls_action
                        else "UNTRAINED DIAGNOSTIC ONLY — deterministic flight assist controls the drone"
                    ),
                )
                canvas.create_text(
                    22,
                    83,
                    anchor="nw",
                    fill="#dce4ef",
                    font=("DejaVu Sans Mono", 11),
                    text=(
                        f"step {packet['step']}  keys {packet['keys']}  "
                        f"spikes {packet['active_neurons']}/{len(roles)}  "
                        f"membrane L2 {packet['membrane_l2']:.3f}"
                    ),
                )

                group_keys = tuple(role_counts)
                columns = min(4, len(group_keys))
                box_width = 876 / max(1, columns)
                for index, role in enumerate(group_keys):
                    row, column = divmod(index, columns)
                    x = 22 + column * box_width
                    y = 122 + row * 49
                    color = _ROLE_COLORS.get(role, "#9aa4b2")
                    active = int(packet["role_spikes"].get(role, 0))
                    canvas.create_rectangle(
                        x,
                        y,
                        x + box_width - 8,
                        y + 39,
                        outline=color,
                    )
                    canvas.create_text(
                        x + 8,
                        y + 6,
                        anchor="nw",
                        fill=color,
                        font=("DejaVu Sans", 9, "bold"),
                        text=_ROLE_LABELS.get(role, role),
                    )
                    canvas.create_text(
                        x + 8,
                        y + 22,
                        anchor="nw",
                        fill="#dce4ef",
                        font=("DejaVu Sans Mono", 9),
                        text=f"active {active}/{role_counts[role]}",
                    )

                chart_left = 22
                chart_top = 235 if len(group_keys) <= 4 else 284
                chart_width = 876
                chart_height = 155
                canvas.create_text(
                    chart_left,
                    chart_top - 11,
                    anchor="sw",
                    fill="#dce4ef",
                    font=("DejaVu Sans", 10),
                    text=(
                        f"rolling {BRAIN_ACTIVITY_WINDOW_STEPS}-step per-neuron "
                        "spike activity (role-colored)"
                    ),
                )
                canvas.create_rectangle(
                    chart_left,
                    chart_top,
                    chart_left + chart_width,
                    chart_top + chart_height,
                    outline="#4a5665",
                )
                activity = [float(value) for value in packet["activity"]]
                maximum = max(max(activity), 1.0e-9)
                bar_width = chart_width / len(activity)
                for index, raw_value in enumerate(activity):
                    normalized = min(max(raw_value / maximum, 0.0), 1.0)
                    x0 = chart_left + index * bar_width
                    x1 = max(x0 + 1.0, chart_left + (index + 1) * bar_width - 0.4)
                    y0 = chart_top + chart_height - normalized * (chart_height - 2)
                    canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        chart_top + chart_height - 1,
                        fill=_ROLE_COLORS.get(roles[index], "#9aa4b2"),
                        outline="",
                    )
                canvas.create_text(
                    chart_left,
                    chart_top + chart_height + 17,
                    anchor="nw",
                    fill="#8d9bad",
                    font=("DejaVu Sans", 9),
                    text=(
                        "Activity is from the exact trained forward pass that produced the flight action."
                        if trained_controls_action
                        else "Activity is the frozen biological graph's response to vehicle state; "
                        "its decoder output is discarded."
                    ),
                )
            root.update_idletasks()
            root.update()
    except BaseException as exc:
        try:
            status_queue.put({"status": "error", "error": repr(exc)})
        except BaseException:
            pass
    finally:
        if root is not None:
            try:
                root.destroy()
            except BaseException:
                pass


class NeuralBrainWindow:
    """A bounded, separately spawned real-time neural activity window."""

    def __init__(
        self,
        mode: str,
        roles: tuple[str, ...],
        *,
        trained_controls_action: bool = False,
        controller_label: str | None = None,
    ) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self._frames = context.Queue(maxsize=2)
        self._status = context.Queue(maxsize=2)
        self._process = context.Process(
            target=_brain_window_process_main,
            args=(
                self._frames,
                self._status,
                mode,
                roles,
                trained_controls_action,
                controller_label,
            ),
            name="crazyflie-manual-neural-window",
            daemon=True,
        )
        self._process.start()
        try:
            startup = self._status.get(timeout=5.0)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("neural window did not become ready within 5 seconds") from exc
        if startup.get("status") != "ready":
            self.close()
            raise RuntimeError(f"neural window startup failed: {startup}")
        self.backend = str(startup["backend"])
        self.update_count = 0
        self.cleanup = "running"

    def show(self, packet: dict[str, Any]) -> None:
        if not self._process.is_alive():
            return
        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(packet)
        self.update_count += 1

    def audit(self) -> dict[str, Any]:
        return {
            "backend": getattr(self, "backend", None),
            "separate_process": True,
            "update_count": getattr(self, "update_count", 0),
            "cleanup": getattr(self, "cleanup", "not_started"),
        }

    def close(self) -> dict[str, Any]:
        process = getattr(self, "_process", None)
        if process is None or getattr(self, "cleanup", None) in {"joined", "terminated"}:
            return self.audit()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                pass
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            self.cleanup = "terminated"
        else:
            self.cleanup = "joined"
        for channel in (self._frames, self._status):
            try:
                channel.close()
                channel.join_thread()
            except BaseException:
                pass
        return self.audit()


class NeuralActivityMonitor:
    """Untrained frozen-circuit activity monitor whose action is never used."""

    def __init__(self, mode: str, device: torch.device) -> None:
        from g1_fly_control.crazyflie.controllers import build_controller

        kind = {
            "wing_thoracic": "wing_lif",
            "leg_wing_thoracic": "leg_wing_lif",
        }[mode]
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        try:
            self.policy, self.report = build_controller(
                kind,
                observation_dim=12,
                action_dim=4,
                device=device,
                connectome_manifest=ROOT / "data/connectome/manifest.json",
                wing_connectome_manifest=ROOT / "data/connectome_wing/manifest.json",
                enforce_parameter_match=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.policy.eval()
        self.mode = mode
        self.state = self.policy.initial_state(1, device=device)
        self.roles = self._load_roles(mode)
        if len(self.roles) != int(self.state.spikes.shape[1]):
            raise RuntimeError(
                "neural role count does not match recurrent state width: "
                f"roles={len(self.roles)}, state={self.state.spikes.shape[1]}"
            )
        self._role_indices = {
            role: tuple(index for index, value in enumerate(self.roles) if value == role)
            for role in dict.fromkeys(self.roles)
        }
        self._activity_window: deque[torch.Tensor] = deque(
            maxlen=BRAIN_ACTIVITY_WINDOW_STEPS
        )
        self.last_summary: dict[str, Any] = {}
        self.last_activity = torch.zeros(len(self.roles), dtype=torch.float32)

    @staticmethod
    def _manifest_roles(directory: str, prefix: str = "") -> tuple[str, ...]:
        path = ROOT / "data" / directory / "neurons.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"neural role manifest is invalid: {path}")
        roles: list[str] = []
        for row in payload:
            try:
                role = str(row["annotations"]["model_role"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"neuron lacks a model role in {path}") from exc
            roles.append(f"{prefix}{role}")
        return tuple(roles)

    @classmethod
    def _load_roles(cls, mode: str) -> tuple[str, ...]:
        wing = cls._manifest_roles(
            "connectome_wing", "wing:" if mode == "leg_wing_thoracic" else ""
        )
        if mode == "wing_thoracic":
            return wing
        leg = cls._manifest_roles("connectome", "leg:")
        return leg + wing

    def reset(self, device: torch.device) -> None:
        self.state = self.policy.initial_state(1, device=device)
        self.last_summary = {}
        self._activity_window.clear()
        self.last_activity.zero_()

    def step(self, physical_observation: torch.Tensor) -> dict[str, float]:
        from g1_fly_control.crazyflie.stabilization import (
            normalize_physical_observation,
        )

        normalized = normalize_physical_observation(physical_observation)
        with torch.no_grad():
            output = self.policy.act(normalized, self.state, deterministic=True)
        self.state = output.state
        spikes = self.state.spikes
        membrane = self.state.membrane
        spike_row = spikes[0].detach().to(device="cpu", dtype=torch.float32)
        self._activity_window.append(spike_row)
        self.last_activity = torch.stack(tuple(self._activity_window)).mean(dim=0)
        role_spikes = {
            role: int(spike_row[list(indices)].sum())
            for role, indices in self._role_indices.items()
        }
        summary = {
            "spike_fraction": float(spikes.float().mean()),
            "active_neurons": int((spikes != 0).sum()),
            "membrane_l2": float(torch.linalg.vector_norm(membrane)),
            "role_spikes": role_spikes,
        }
        if self.mode == "leg_wing_thoracic":
            boundary = int(self.policy.core.num_neurons)
            summary["leg_spike_fraction"] = float(
                spikes[:, :boundary].float().mean()
            )
            summary["wing_thoracic_spike_fraction"] = float(
                spikes[:, boundary:].float().mean()
            )
        self.last_summary = summary
        return summary

    def brain_packet(self, step: int, pressed_keys: Iterable[str]) -> dict[str, Any]:
        if not self.last_summary:
            raise RuntimeError("neural monitor has no activity sample")
        return {
            "step": int(step),
            "keys": "+".join(sorted(str(key) for key in pressed_keys)) or "-",
            "active_neurons": int(self.last_summary["active_neurons"]),
            "membrane_l2": float(self.last_summary["membrane_l2"]),
            "role_spikes": dict(self.last_summary["role_spikes"]),
            "activity": [float(value) for value in self.last_activity],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripted_smoke",
        action="store_true",
        help="Run a finite deterministic command sequence instead of reading keys",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum control steps; zero means unlimited interactive flight",
    )
    parser.add_argument("--horizontal_speed", type=float, default=0.55)
    parser.add_argument("--vertical_speed", type=float, default=0.35)
    parser.add_argument("--yaw_rate", type=float, default=0.8)
    parser.add_argument(
        "--neural_monitor",
        choices=("none", "wing_thoracic", "leg_wing_thoracic"),
        default="wing_thoracic",
        help=(
            "Diagnostic frozen-circuit activity only; never controls the drone "
            "(default: wing_thoracic)"
        ),
    )
    parser.add_argument(
        "--no_brain_window",
        action="store_true",
        help="Disable the separate neural activity window in visible mode",
    )
    parser.add_argument(
        "--asset_mirror",
        type=Path,
        default=Path.home() / ".cache/flyg1/isaac-5.1-offline",
    )
    parser.add_argument(
        "--offline_only",
        action="store_true",
        help="Fail instead of downloading if the verified local caches are absent",
    )
    parser.add_argument("--status_hz", type=float, default=4.0)
    parser.add_argument("--no_follow_camera", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.scripted_smoke and args.steps == 0:
        args.steps = 800
    if args.scripted_smoke and args.steps < 800:
        parser.error("scripted smoke requires at least 800 steps")
    if not args.scripted_smoke and args.headless:
        parser.error("interactive keyboard flight requires a visible viewer")
    for name in ("horizontal_speed", "vertical_speed", "yaw_rate", "status_hz"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name} must be finite and positive")


def _phase_summary(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in phases.items():
        samples = max(1, int(values["samples"]))
        result[name] = {
            "samples": int(values["samples"]),
            "mean_forward_velocity_m_s": values["forward_velocity_sum"] / samples,
            "mean_left_velocity_m_s": values["left_velocity_sum"] / samples,
            "mean_vertical_velocity_m_s": values["vertical_velocity_sum"] / samples,
            "mean_command_projection_m_s": values["projection_sum"] / samples,
        }
    return result


def _runtime_state_is_finite(*tensors: torch.Tensor) -> bool:
    """Return whether every simulator/control tensor is finite and non-empty."""

    return bool(tensors) and all(
        isinstance(value, torch.Tensor)
        and value.numel() > 0
        and bool(torch.isfinite(value).all())
        for value in tensors
    )


def _smoke_failures(
    phase_report: dict[str, Any],
    safety_termination_count: int,
    invalid_state_reset_count: int,
    maximum_action_abs: float,
) -> list[str]:
    failures: list[str] = []
    if safety_termination_count:
        failures.append(f"{safety_termination_count} safety termination(s)")
    if invalid_state_reset_count:
        failures.append(f"{invalid_state_reset_count} nonfinite-state safety reset(s)")
    if not math.isfinite(maximum_action_abs) or maximum_action_abs > 1.000001:
        failures.append("action bound/finite invariant failed")
    thresholds = {
        "forward": ("mean_forward_velocity_m_s", 0.025),
        "forward_left": ("mean_command_projection_m_s", 0.025),
        "up": ("mean_vertical_velocity_m_s", 0.015),
        "down": ("mean_vertical_velocity_m_s", -0.015),
    }
    for phase, (metric, threshold) in thresholds.items():
        value = phase_report.get(phase, {}).get(metric)
        if value is None or not math.isfinite(value):
            failures.append(f"{phase} response is missing/nonfinite")
        elif phase == "down" and value >= threshold:
            failures.append(f"down response {value:.4f} m/s is not negative enough")
        elif phase != "down" and value <= threshold:
            failures.append(f"{phase} response {value:.4f} m/s is too small")
    return failures


def _run(args: argparse.Namespace, simulation_app: Any, local_usd: Path, asset_report: dict[str, Any]) -> int:
    import gymnasium as gym
    import isaaclab_tasks.direct.quadcopter  # noqa: F401
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    from drone_bootstrap import selected_env_cfg

    cfg = selected_env_cfg(TASK, 1)
    cfg.debug_vis = False
    cfg.episode_length_s = 24.0 * 60.0 * 60.0
    cfg.scene.num_envs = 1
    cfg.scene.clone_in_fabric = False
    cfg.robot.spawn.usd_path = str(local_usd)
    # The stock plane points at an Isaac cloud USD. Generate an equivalent
    # flat collision surface locally so offline mode covers the whole scene.
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
    monitor = None
    brain_window = None
    brain_window_audit: dict[str, Any] = {
        "enabled": False,
        "separate_process": False,
    }
    held = HeldKeyState()
    try:
        env = gym.make(
            TASK,
            cfg=cfg,
            render_mode=None if args.headless else "human",
        ).unwrapped
        observation, _ = env.reset(seed=0)
        env.episode_length_buf.zero_()
        device = torch.device(env.device)
        config = FlightAssistConfig(
            horizontal_speed_m_s=float(args.horizontal_speed),
            vertical_speed_m_s=float(args.vertical_speed),
            yaw_rate_rad_s=float(args.yaw_rate),
        )
        assist = FlightAssist(config, env._robot.data.root_pos_w.detach().clone())
        monitor = (
            None
            if args.neural_monitor == "none"
            else NeuralActivityMonitor(args.neural_monitor, device)
        )
        if monitor is not None and not args.headless and not args.no_brain_window:
            brain_window = NeuralBrainWindow(args.neural_monitor, monitor.roles)
            brain_window_audit = {
                "enabled": True,
                **brain_window.audit(),
            }
        if not args.scripted_smoke:
            keyboard = IsaacKeyboard(held)
            print(
                "\nCrazyflie manual control (viewer must have keyboard focus)\n"
                "  W/S forward/back | A/D left/right | I or E up | Q down\n"
                "  J/L yaw left/right | H/Space hover | R reset | Esc exit\n"
                "  Multi-key commands such as W+A and W+A+I are supported.\n"
                "  Flight action: deterministic assist; neural monitor: "
                f"{args.neural_monitor} (diagnostic only, action ignored).\n"
                f"  Separate neural window: {brain_window_audit['enabled']}.\n"
            )
        if not args.headless:
            root = env._robot.data.root_pos_w[0].detach().cpu().tolist()
            env.sim.set_camera_view(
                eye=[root[0] - 1.35, root[1] - 1.35, root[2] + 0.65],
                target=root,
            )
            if not args.no_follow_camera:
                try:
                    camera = CloseFollowCamera(env, (-1.35, -1.35, 0.65))
                except Exception as exc:
                    print(f"WARNING: close-follow camera disabled: {exc}")

        start_position = env._robot.data.root_pos_w.detach().clone()
        maximum_action_abs = 0.0
        safety_termination_count = 0
        termination_reset_count = 0
        truncation_reset_count = 0
        invalid_state_reset_count = 0
        finite_state_check_count = 0
        manual_reset_count = 0
        executed_steps = 0
        phases: dict[str, dict[str, Any]] = {}
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
                observation, _ = env.reset()
                env.episode_length_buf.zero_()
                assist.reset(env._robot.data.root_pos_w.detach().clone())
                held.clear_movement()
                if monitor is not None:
                    monitor.reset(device)
                manual_reset_count += 1

            command = command_from_pressed(held.pressed)
            root_position = env._robot.data.root_pos_w.detach().clone()
            root_quaternion = env._robot.data.root_quat_w.detach().clone()
            if not _runtime_state_is_finite(
                observation["policy"],
                root_position,
                root_quaternion,
                env._robot.data.root_lin_vel_b,
                env._robot.data.root_ang_vel_b,
            ):
                invalid_state_reset_count += 1
                observation, _ = env.reset()
                env.episode_length_buf.zero_()
                assist.reset(env._robot.data.root_pos_w.detach().clone())
                held.clear_movement()
                if monitor is not None:
                    monitor.reset(device)
                if not args.headless:
                    print("\nSafety reset: nonfinite simulator state detected.")
                if not _runtime_state_is_finite(
                    observation["policy"],
                    env._robot.data.root_pos_w,
                    env._robot.data.root_quat_w,
                    env._robot.data.root_lin_vel_b,
                    env._robot.data.root_ang_vel_b,
                ):
                    raise FloatingPointError(
                        "simulator remained nonfinite after safety reset"
                    )
                root_position = env._robot.data.root_pos_w.detach().clone()
                root_quaternion = env._robot.data.root_quat_w.detach().clone()
            finite_state_check_count += 1
            desired_velocity = assist.desired_velocity_body(
                root_position, root_quaternion, command, CONTROL_DT_S
            )
            action = assist.action(
                observation["policy"],
                desired_velocity,
                command.yaw_left * config.yaw_rate_rad_s,
            )
            maximum_action_abs = max(maximum_action_abs, float(action.abs().max()))
            if monitor is not None:
                neural_observation = observation["policy"].clone()
                neural_observation[:, 9:12] = _quat_apply_inverse(
                    root_quaternion, assist.target_w - root_position
                )
                monitor.step(neural_observation)
            next_observation, _reward_ignored, terminated, truncated, _info = env.step(action)
            executed_steps += 1
            observation = next_observation
            if (
                brain_window is not None
                and executed_steps % BRAIN_UPDATE_STEPS == 0
                and monitor is not None
            ):
                brain_window.show(
                    monitor.brain_packet(executed_steps, held.pressed)
                )
            if not _runtime_state_is_finite(
                observation["policy"],
                env._robot.data.root_pos_w,
                env._robot.data.root_quat_w,
                env._robot.data.root_lin_vel_b,
                env._robot.data.root_ang_vel_b,
            ):
                invalid_state_reset_count += 1
                observation, _ = env.reset()
                env.episode_length_buf.zero_()
                assist.reset(env._robot.data.root_pos_w.detach().clone())
                held.clear_movement()
                if monitor is not None:
                    monitor.reset(device)
                if not args.headless:
                    print("\nSafety reset: nonfinite simulator state detected.")
                if not _runtime_state_is_finite(
                    observation["policy"],
                    env._robot.data.root_pos_w,
                    env._robot.data.root_quat_w,
                    env._robot.data.root_lin_vel_b,
                    env._robot.data.root_ang_vel_b,
                ):
                    raise FloatingPointError(
                        "simulator remained nonfinite after safety reset"
                    )
                finite_state_check_count += 1
                continue
            finite_state_check_count += 1

            velocity = env._robot.data.root_lin_vel_b[0].detach()
            phase = phases.setdefault(
                current_phase,
                {
                    "samples": 0,
                    "forward_velocity_sum": 0.0,
                    "left_velocity_sum": 0.0,
                    "vertical_velocity_sum": 0.0,
                    "projection_sum": 0.0,
                },
            )
            phase["samples"] += 1
            phase["forward_velocity_sum"] += float(velocity[0])
            phase["left_velocity_sum"] += float(velocity[1])
            phase["vertical_velocity_sum"] += float(velocity[2])
            phase["projection_sum"] += float(
                velocity[0] * command.forward + velocity[1] * command.left
            )

            if bool((terminated | truncated).any()):
                termination_reset_count += int(terminated.sum())
                truncation_reset_count += int(truncated.sum())
                safety_termination_count += int((terminated | truncated).sum())
                env.episode_length_buf.zero_()
                assist.reset(env._robot.data.root_pos_w.detach().clone())
                held.clear_movement()
                if monitor is not None:
                    monitor.reset(device)

            if camera is not None and executed_steps % 5 == 0:
                camera.update()
            now = time.monotonic()
            if not args.headless and now >= next_status:
                position = env._robot.data.root_pos_w[0].detach().cpu().tolist()
                velocity_values = velocity.cpu().tolist()
                neural = ""
                if monitor is not None and monitor.last_summary:
                    neural = (
                        f" | neural spikes={monitor.last_summary['spike_fraction']:.3f}"
                        f" membrane={monitor.last_summary['membrane_l2']:.2f}"
                    )
                print(
                    "\r"
                    f"keys={'+'.join(sorted(held.pressed)) or '-':<9} "
                    f"cmd=({command.forward:+.2f},{command.left:+.2f},{command.up:+.1f}) "
                    f"pos=({position[0]:+.2f},{position[1]:+.2f},{position[2]:+.2f}) "
                    f"vel_b=({velocity_values[0]:+.2f},{velocity_values[1]:+.2f},"
                    f"{velocity_values[2]:+.2f}){neural}   ",
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
            brain_window_audit = {
                "enabled": True,
                **brain_window.close(),
            }
            brain_window = None
        end_position = env._robot.data.root_pos_w.detach().clone()
        displacement = end_position - start_position
        final_state_finite = _runtime_state_is_finite(
            observation["policy"],
            end_position,
            env._robot.data.root_quat_w,
            env._robot.data.root_lin_vel_b,
            env._robot.data.root_ang_vel_b,
        )
        phase_report = _phase_summary(phases)
        failures = (
            _smoke_failures(
                phase_report,
                safety_termination_count,
                invalid_state_reset_count,
                maximum_action_abs,
            )
            if args.scripted_smoke
            else []
        )
        report = {
            "schema_version": 1,
            "status": "PASS" if not failures else "FAIL",
            "mode": "scripted_smoke" if args.scripted_smoke else "interactive_keyboard",
            "task": TASK,
            "checkpoint_loaded": False,
            "training": False,
            "reward_used_for_control": False,
            "action_source": "deterministic_manual_flight_assist",
            "neural_monitor": args.neural_monitor,
            "neural_monitor_controls_action": False,
            "neural_activity_final": (
                monitor.last_summary if monitor is not None else None
            ),
            "neural_input_target": (
                "manual_flight_assist_setpoint_body_frame"
                if monitor is not None
                else None
            ),
            "neural_population": (
                {
                    "neurons": len(monitor.roles),
                    "role_counts": dict(Counter(monitor.roles)),
                    **{
                        key: monitor.report[key]
                        for key in (
                            "controller_kind",
                            "actor_trainable_parameters",
                            "total_trainable_parameters",
                            "frozen_synaptic_weights",
                            "model_total_parameters",
                            "total_dynamic_state_per_environment",
                            "core_checksum",
                        )
                    },
                }
                if monitor is not None
                else None
            ),
            "brain_window": brain_window_audit,
            "steps": executed_steps,
            "manual_reset_count": manual_reset_count,
            "safety_termination_count": safety_termination_count,
            "termination_reset_count": termination_reset_count,
            "truncation_reset_count": truncation_reset_count,
            "invalid_state_reset_count": invalid_state_reset_count,
            "finite_state_checks": {
                "passed": bool(final_state_finite and invalid_state_reset_count == 0),
                "check_count": finite_state_check_count,
                "final_state_finite": final_state_finite,
            },
            "maximum_action_abs": maximum_action_abs,
            "start_position_w_m": start_position[0].detach().cpu().tolist(),
            "end_position_w_m": end_position[0].detach().cpu().tolist(),
            "displacement_w_m": displacement[0].detach().cpu().tolist(),
            "displacement_norm_m": float(torch.linalg.vector_norm(displacement[0])),
            "phase_metrics": phase_report,
            "failures": failures,
            "asset": asset_report,
            "ground": "local_procedural_flat_mesh",
            "follow_camera": (
                camera.audit()
                if camera is not None
                else {"enabled": False, "update_count": 0, "last_pose": None}
            ),
        }
        print(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0 if not failures else 1
    finally:
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
    _validate_args(parser, args)
    try:
        local_usd, asset_report = prepare_asset_mirror(
            args.asset_mirror, allow_download=not args.offline_only
        )
    except Exception as exc:
        parser.error(str(exc))
    simulation_app = AppLauncher(args).app
    try:
        return _run(args, simulation_app, local_usd, asset_report)
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    # Kit shutdown may otherwise replace a requested non-zero SystemExit on
    # some Isaac Sim builds. Flush the report, then preserve our test status.
    _exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_code)
