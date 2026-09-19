#!/usr/bin/env python3
"""Visualize one frozen-LIF Crazyflie checkpoint with a live terminal sidecar.

The Isaac viewer is enabled by default; pass ``--headless`` to retain only the
terminal sidecar.  Inference is always deterministic, exactly one environment
is constructed, and execution ends at the first episode ending or the finite
``--steps`` limit.  This is a diagnostic playback, not held-out evaluation.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any

import torch

from drone_bootstrap import launch_environment, sha256_file
from drone_play import (
    TASKS,
    _atomic_json,
    inspect_checkpoint,
    load_policy_and_normalizer,
)


LIF_CONTROLLERS = (
    "frozen_lif_original",
    "frozen_lif_degree_rewired",
)
CONTROL_DT_S = 0.02
MAX_VISUALIZATION_STEPS = 600
MIN_VISUALIZATION_STEPS = 4
DEFAULT_DISPLAY_INTERVAL = 10
DEFAULT_ACTIVITY_WINDOW_STEPS = 25
DEFAULT_ACTIVITY_BAR_WIDTH = 48
DEFAULT_CLOSE_FOLLOW_CAMERA_OFFSET_M = (-1.15, -1.15, 0.55)
CLOSE_FOLLOW_CAMERA_REFRESH_STEPS = 5
MIN_CLOSE_FOLLOW_CAMERA_DISTANCE_M = 0.25
MAX_CLOSE_FOLLOW_CAMERA_DISTANCE_M = 10.0
BRAIN_WINDOW_TITLE = "Crazyflie LIF Brain (realtime)"
BRAIN_WINDOW_WIDTH = 760
BRAIN_WINDOW_HEIGHT = 360
VISUALIZATION_ONLY_SOURCE_FILES = (
    "scripts/drone_play.py",
    "scripts/drone_visualize_lif.py",
)


def _validate_lif_controller(controller: Any) -> str:
    """Return a declared LIF controller name or fail before Isaac starts."""

    if controller not in LIF_CONTROLLERS:
        raise ValueError(
            "LIF visualization requires frozen_lif_original or "
            f"frozen_lif_degree_rewired; checkpoint contains {controller!r}"
        )
    return str(controller)


def _steady_state_sample_steps(total_steps: int) -> tuple[int, int, int, int]:
    """Choose four bounded, increasing memory samples across the rollout."""

    if type(total_steps) is not int or total_steps < MIN_VISUALIZATION_STEPS:
        raise ValueError(
            f"total_steps must be an integer >= {MIN_VISUALIZATION_STEPS}"
        )
    points = tuple(math.ceil(total_steps * fraction / 4) for fraction in range(1, 5))
    if len(set(points)) != 4 or points[-1] != total_steps:
        raise RuntimeError("memory sample schedule did not produce four unique steps")
    return points


def _pace_deadline(
    start_monotonic_s: float,
    completed_steps: int,
    realtime_rate: float,
) -> float:
    """Return the wall-clock deadline for one completed 50 Hz control step."""

    values = (start_monotonic_s, realtime_rate)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("realtime pacing values must be finite")
    if completed_steps < 1 or realtime_rate <= 0.0:
        raise ValueError("completed_steps and realtime_rate must be positive")
    return start_monotonic_s + completed_steps * CONTROL_DT_S / realtime_rate


def _validate_camera_offset(values: Any) -> tuple[float, float, float]:
    """Validate a finite, non-degenerate world-frame eye offset in metres."""

    try:
        offset = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera offset must contain three finite numbers") from exc
    if len(offset) != 3 or any(not math.isfinite(value) for value in offset):
        raise ValueError("camera offset must contain three finite numbers")
    distance = math.sqrt(sum(value * value for value in offset))
    if not MIN_CLOSE_FOLLOW_CAMERA_DISTANCE_M <= distance <= MAX_CLOSE_FOLLOW_CAMERA_DISTANCE_M:
        raise ValueError(
            "camera offset distance must be in "
            f"[{MIN_CLOSE_FOLLOW_CAMERA_DISTANCE_M}, "
            f"{MAX_CLOSE_FOLLOW_CAMERA_DISTANCE_M}] metres"
        )
    return offset


def _close_follow_camera_pose(
    root_position_w: Any,
    offset_m: Any,
) -> tuple[list[float], list[float]]:
    """Return an eye/target pair that follows env 0 in the world frame."""

    root = tuple(float(value) for value in root_position_w)
    if len(root) != 3 or any(not math.isfinite(value) for value in root):
        raise ValueError("Crazyflie root position must contain three finite numbers")
    offset = _validate_camera_offset(offset_m)
    target = [root[0], root[1], root[2]]
    eye = [target[index] + offset[index] for index in range(3)]
    return eye, target


def _set_close_follow_camera(env: Any, offset_m: Any) -> dict[str, list[float]]:
    """Point the Isaac viewer camera at env 0's Crazyflie."""

    root = env._robot.data.root_pos_w[0].detach().to(device="cpu").tolist()
    eye, target = _close_follow_camera_pose(root, offset_m)
    env.sim.set_camera_view(eye=eye, target=target)
    return {"eye_w_m": eye, "target_w_m": target}


class CloseFollowCamera:
    """Follow env 0 without accumulating one Kit command per refresh.

    Isaac's convenience ``set_camera_view`` constructs several Kit command
    objects on every call.  That is appropriate for occasional interactive
    camera edits but caused resident memory to rise throughout a realtime
    playback.  We use it once to establish the view, then reuse one authored
    USD matrix op.  The eye offset is fixed in the world frame, so only the
    matrix translation changes while its orientation stays constant.
    """

    CAMERA_PRIM_PATH = "/OmniverseKit_Persp"

    def __init__(self, env: Any, offset_m: Any) -> None:
        self._env = env
        self._offset_m = _validate_camera_offset(offset_m)
        self.last_pose = _set_close_follow_camera(env, self._offset_m)

        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Usd, UsdGeom

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("close-follow camera requires an active Isaac viewport")
        camera_prim = viewport.stage.GetPrimAtPath(self.CAMERA_PRIM_PATH)
        camera = UsdGeom.Camera(camera_prim)
        if not camera:
            raise RuntimeError(
                f"close-follow camera prim is unavailable: {self.CAMERA_PRIM_PATH}"
            )

        self._gf = Gf
        self._time = Usd.TimeCode.Default()
        self._camera = camera
        parent_world = camera.ComputeParentToWorldTransform(self._time)
        camera_world = camera.ComputeLocalToWorldTransform(self._time)
        initial_local = camera_world * parent_world.GetInverse()
        self._base_local_transform = Gf.Matrix4d(initial_local)
        self._matrix_op = UsdGeom.Xformable(camera_prim).MakeMatrixXform()
        if not self._matrix_op.Set(initial_local, self._time):
            raise RuntimeError("failed to initialize close-follow camera matrix")

    def update(self) -> dict[str, list[float]]:
        root = self._env._robot.data.root_pos_w[0].detach().to(device="cpu").tolist()
        eye, target = _close_follow_camera_pose(root, self._offset_m)
        parent_world = self._camera.ComputeParentToWorldTransform(self._time)
        eye_in_parent = parent_world.GetInverse().Transform(self._gf.Vec3d(*eye))
        local_transform = self._gf.Matrix4d(self._base_local_transform)
        local_transform.SetTranslateOnly(eye_in_parent)
        if not self._matrix_op.Set(local_transform, self._time):
            raise RuntimeError("failed to update close-follow camera matrix")
        self.last_pose = {"eye_w_m": eye, "target_w_m": target}
        return self.last_pose


def _enable_bounded_goal_marker(env: Any) -> bool:
    """Show the native goal marker without its per-frame allocation callback."""

    enabled = bool(env.set_debug_vis(True))
    if not enabled:
        return False
    handle = getattr(env, "_debug_vis_handle", None)
    if handle is None or not hasattr(env, "goal_pos_visualizer"):
        raise RuntimeError("Crazyflie goal marker did not expose its native visualizer")
    handle.unsubscribe()
    env._debug_vis_handle = None
    env.goal_pos_visualizer.visualize(env._desired_pos_w)
    return True


def _update_bounded_goal_marker(env: Any) -> None:
    """Move the marker after a scheduled WaypointSwitch target change."""

    env.goal_pos_visualizer.visualize(env._desired_pos_w)


def _brain_window_packet(
    frame: dict[str, Any], activity: torch.Tensor
) -> dict[str, Any]:
    """Build a small, finite CPU-only payload for the isolated GUI process."""

    if not isinstance(activity, torch.Tensor) or activity.ndim != 1:
        raise TypeError("brain-window activity must be a one-dimensional tensor")
    values = activity.detach().to(device="cpu", dtype=torch.float32)
    if values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("brain-window activity must be finite and non-empty")
    lif = frame["lif_activity"]
    packet = {
        "step": int(frame["step"]),
        "requested_steps": int(frame["requested_steps"]),
        "spike_count": int(lif["final_substep_spike_count"]),
        "neuron_count": int(frame["neuron_count"]),
        "spike_rate_hz_per_neuron": float(
            lif["window_sampled_spike_rate_hz_per_neuron"]
        ),
        "membrane_l2": float(lif["membrane_l2"]),
        "synapse_l2": float(lif["synapse_l2"]),
        "window_activity": [float(value) for value in values],
    }
    numeric = (
        packet["spike_rate_hz_per_neuron"],
        packet["membrane_l2"],
        packet["synapse_l2"],
        *packet["window_activity"],
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("brain-window payload contains a nonfinite value")
    return packet


def _brain_window_process_main(
    frame_queue: Any,
    status_queue: Any,
    title: str,
) -> None:
    """Own a Tk window in a spawned process, isolated from Isaac UI.

    The Isaac environment ships ``opencv-python-headless`` on some installs, so
    OpenCV's image routines can be present while ``imshow``/``namedWindow`` are
    deliberately unavailable.  Tk is part of this Isaac Python installation
    and talks directly to the active X11 display.
    """

    root = None
    try:
        import tkinter as tk

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
        patchlevel = str(root.tk.call("info", "patchlevel"))
        status_queue.put({"status": "ready", "backend": f"tk-{patchlevel}"})
        no_frame = object()
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
                    18,
                    anchor="nw",
                    fill="#ebebeb",
                    font=("DejaVu Sans", 17, "bold"),
                    text=f"LIF step {packet['step']}/{packet['requested_steps']}",
                )
                lines = (
                    f"spikes: {packet['spike_count']}/{packet['neuron_count']}",
                    f"window spike rate: {packet['spike_rate_hz_per_neuron']:.2f} Hz/neuron",
                    f"membrane L2: {packet['membrane_l2']:.3f}",
                    f"synapse L2: {packet['synapse_l2']:.3f}",
                )
                for index, line in enumerate(lines):
                    canvas.create_text(
                        22,
                        62 + index * 28,
                        anchor="nw",
                        fill="#28d2ff" if index < 2 else "#ebebeb",
                        font=("DejaVu Sans Mono", 12),
                        text=line,
                    )

                activity = [float(value) for value in packet["window_activity"]]
                maximum = max(max(activity), 1.0e-9)
                left, top, width, height = 22, 215, 716, 115
                canvas.create_rectangle(
                    left, top, left + width, top + height, outline="#4a4a4a"
                )
                bar_width = max(width / len(activity), 1.0)
                for index, raw_value in enumerate(activity):
                    value = min(max(raw_value / maximum, 0.0), 1.0)
                    x0 = left + index * bar_width
                    x1 = max(x0 + 1.0, left + (index + 1) * bar_width - 1.0)
                    y0 = top + height - value * (height - 2)
                    canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        top + height - 1,
                        fill="#50e678",
                        outline="",
                    )
                canvas.create_text(
                    left,
                    top - 10,
                    anchor="sw",
                    fill="#ebebeb",
                    font=("DejaVu Sans", 10),
                    text="relative per-neuron activity (rolling window)",
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


class BrainWindow:
    """A bounded, separately spawned realtime LIF activity window."""

    def __init__(self) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self._frames = context.Queue(maxsize=2)
        self._status = context.Queue(maxsize=2)
        self._process = context.Process(
            target=_brain_window_process_main,
            args=(self._frames, self._status, BRAIN_WINDOW_TITLE),
            name="crazyflie-lif-brain-window",
            daemon=True,
        )
        self._process.start()
        try:
            startup = self._status.get(timeout=5.0)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("brain window did not become ready within 5 seconds") from exc
        if startup.get("status") != "ready":
            self.close()
            raise RuntimeError(f"brain window startup failed: {startup}")
        self.backend = str(startup["backend"])
        self.update_count = 0
        self.cleanup = "running"

    def show(self, frame: dict[str, Any], activity: torch.Tensor) -> None:
        packet = _brain_window_packet(frame, activity)
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


def _deterministic_lif_action(policy: Any, observation: torch.Tensor, state: Any) -> Any:
    """Call the only permitted visualization inference rule."""

    return policy.act(observation, state, deterministic=True)


def _lif_state_tensors(state: Any) -> tuple[torch.Tensor, ...]:
    """Validate and return the four one-environment LIF state tensors."""

    names = ("membrane", "spikes", "synapse", "refractory")
    values = tuple(getattr(state, name, None) for name in names)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Controller did not return a complete LIF recurrent state")
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1 or values[0].ndim != 2 or values[0].shape[0] != 1:
        raise ValueError("LIF visualization requires four matching [1, neurons] tensors")
    if values[0].shape[1] < 1:
        raise ValueError("LIF state must contain at least one neuron")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("LIF recurrent state contains a nonfinite value")
    spikes = values[1]
    if not bool(((spikes == 0) | (spikes == 1)).all()):
        raise FloatingPointError("LIF spike state is not binary")
    return values


def _render_activity_bar(activity: torch.Tensor, width: int) -> str:
    """Render a fixed-width relative per-neuron activity sidecar bar."""

    if not isinstance(activity, torch.Tensor) or activity.ndim != 1:
        raise TypeError("activity must be a one-dimensional tensor")
    if activity.numel() < 1 or not bool(torch.isfinite(activity).all()):
        raise ValueError("activity must be finite and non-empty")
    if bool((activity < 0).any()) or width < 1:
        raise ValueError("activity must be non-negative and width positive")
    width = min(int(width), int(activity.numel()))
    bins = torch.tensor_split(activity.to(dtype=torch.float32, device="cpu"), width)
    values = torch.stack([chunk.mean() for chunk in bins])
    maximum = float(values.max())
    if maximum <= 0.0:
        return " " * width
    relative = (values / maximum).clamp(0.0, 1.0)
    palette = " .:-=+*#%@"
    indices = torch.round(relative * (len(palette) - 1)).to(dtype=torch.long)
    return "".join(palette[int(index)] for index in indices)


class LIFActivityTracker:
    """Bounded CPU summary used by the terminal sidecar and diagnostic JSON."""

    def __init__(self, *, window_steps: int, bar_width: int) -> None:
        if window_steps < 1 or bar_width < 1:
            raise ValueError("activity window and bar width must be positive")
        self.window_steps = int(window_steps)
        self.bar_width = int(bar_width)
        self._window: deque[torch.Tensor] = deque(maxlen=self.window_steps)
        self._spike_counts: torch.Tensor | None = None
        self._sample_count = 0
        self._membrane_l2_sum = 0.0
        self._membrane_l2_max = 0.0
        self._synapse_l2_sum = 0.0
        self._synapse_l2_max = 0.0

    @property
    def neuron_count(self) -> int:
        return int(self._spike_counts.numel()) if self._spike_counts is not None else 0

    def update(self, state: Any) -> dict[str, Any]:
        membrane, spikes, synapse, _ = _lif_state_tensors(state)
        spike_row = spikes[0].detach().to(device="cpu", dtype=torch.float32)
        membrane_l2 = float(torch.linalg.vector_norm(membrane[0].detach().float()))
        synapse_l2 = float(torch.linalg.vector_norm(synapse[0].detach().float()))
        if self._spike_counts is None:
            self._spike_counts = torch.zeros_like(spike_row, dtype=torch.long)
        if spike_row.shape != self._spike_counts.shape:
            raise RuntimeError("LIF neuron count changed during visualization")
        self._spike_counts += spike_row.to(dtype=torch.long)
        self._window.append(spike_row)
        self._sample_count += 1
        self._membrane_l2_sum += membrane_l2
        self._membrane_l2_max = max(self._membrane_l2_max, membrane_l2)
        self._synapse_l2_sum += synapse_l2
        self._synapse_l2_max = max(self._synapse_l2_max, synapse_l2)

        window_activity = torch.stack(tuple(self._window)).mean(dim=0)
        spike_count = int(spike_row.sum())
        top_count = min(5, self.neuron_count)
        top_values, top_indices = torch.topk(window_activity, k=top_count)
        return {
            "sample_semantics": "post_final_neural_substep_once_per_control_decision",
            "final_substep_spike_count": spike_count,
            "final_substep_spike_fraction": spike_count / self.neuron_count,
            "window_control_steps": len(self._window),
            "window_sampled_spike_fraction": float(window_activity.mean()),
            "window_sampled_spike_rate_hz_per_neuron": float(
                window_activity.mean() / CONTROL_DT_S
            ),
            "membrane_l2": membrane_l2,
            "synapse_l2": synapse_l2,
            "dead_neuron_fraction_to_date": float((self._spike_counts == 0).float().mean()),
            "saturated_neuron_fraction_to_date": float(
                (self._spike_counts == self._sample_count).float().mean()
            ),
            "top_window_neurons": [
                {"index": int(index), "sampled_spike_fraction": float(value)}
                for value, index in zip(top_values, top_indices, strict=True)
            ],
            "relative_activity_bar": _render_activity_bar(
                window_activity, self.bar_width
            ),
        }

    def summary(self) -> dict[str, Any]:
        if self._spike_counts is None or self._sample_count < 1:
            raise RuntimeError("No LIF activity was sampled")
        total = int(self._spike_counts.sum())
        denominator = self._sample_count * self.neuron_count
        fraction = total / denominator
        return {
            "schema_version": 1,
            "sample_semantics": "post_final_neural_substep_once_per_control_decision",
            "unobserved_internal_neural_substeps_counted": False,
            "control_dt_s": CONTROL_DT_S,
            "sampled_control_steps": self._sample_count,
            "neuron_count": self.neuron_count,
            "total_sampled_spikes": total,
            "sampled_spike_fraction": fraction,
            "sampled_spike_rate_hz_per_neuron": fraction / CONTROL_DT_S,
            "dead_neuron_fraction": float((self._spike_counts == 0).float().mean()),
            "saturated_neuron_fraction": float(
                (self._spike_counts == self._sample_count).float().mean()
            ),
            "per_neuron_sampled_spike_count_min": int(self._spike_counts.min()),
            "per_neuron_sampled_spike_count_max": int(self._spike_counts.max()),
            "membrane_l2_mean": self._membrane_l2_sum / self._sample_count,
            "membrane_l2_max": self._membrane_l2_max,
            "synapse_l2_mean": self._synapse_l2_sum / self._sample_count,
            "synapse_l2_max": self._synapse_l2_max,
            "window_steps": self.window_steps,
            "activity_bar_width": min(self.bar_width, self.neuron_count),
        }

    def window_activity(self) -> torch.Tensor:
        if not self._window:
            raise RuntimeError("No LIF activity was sampled")
        return torch.stack(tuple(self._window)).mean(dim=0)


def _memory_brief(sample: dict[str, Any] | None) -> str:
    if sample is None:
        return "memory=pending"
    gpu = sample.get("gpu_devices") or []
    gpu_used = max((float(row["used_mib"]) for row in gpu), default=float("nan"))
    gpu_text = f"{gpu_used:.0f}MiB" if math.isfinite(gpu_used) else "unavailable"
    return (
        f"GPU={gpu_text} RSS={float(sample['process_rss_mib']):.0f}MiB "
        f"RAM={float(sample['system_ram_percent']):.1f}%"
    )


def _format_sidecar(frame: dict[str, Any], memory: dict[str, Any] | None) -> str:
    activity = frame["lif_activity"]
    action = ", ".join(f"{value:+.3f}" for value in frame["action"])
    outcome = "RUNNING"
    if frame["terminated"]:
        outcome = f"TERMINATED cause={frame['failure_cause']}"
    elif frame["truncated"]:
        outcome = "TIME_LIMIT"
    return "\n".join(
        (
            (
                f"[Crazyflie LIF] step {frame['step']:03d}/{frame['requested_steps']:03d} "
                f"sim={frame['simulation_time_s']:.2f}s wall={frame['wall_time_s']:.2f}s "
                f"RT={frame['realtime_factor']:.2f}x {outcome}"
            ),
            (
                f"flight: distance={frame['goal_distance_m']:.3f}m "
                f"speed={frame['speed_m_s']:.3f}m/s reward={frame['reward']:+.4f} "
                f"success_events={frame['success_count']}"
            ),
            (
                f"action: [{action}]  LIF final-substep spikes="
                f"{activity['final_substep_spike_count']}/{frame['neuron_count']} "
                f"window_rate={activity['window_sampled_spike_rate_hz_per_neuron']:.2f}Hz/neuron"
            ),
            (
                f"activity(relative): |{activity['relative_activity_bar']}| "
                f"membrane_L2={activity['membrane_l2']:.3f} "
                f"synapse_L2={activity['synapse_l2']:.3f}"
            ),
            _memory_brief(memory),
        )
    )


class TerminalSidecar:
    """Small ANSI-updating sidecar with a log-safe non-TTY fallback."""

    def __init__(self, *, plain: bool) -> None:
        self.ansi = bool(sys.stdout.isatty() and not plain)
        self._line_count = 0

    def show(self, text: str) -> None:
        lines = text.splitlines()
        if self.ansi and self._line_count:
            sys.stdout.write(f"\x1b[{self._line_count}A")
        for line in lines:
            if self.ansi:
                sys.stdout.write("\x1b[2K")
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self._line_count = len(lines)


def _resolved_config(args: argparse.Namespace, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "bounded_realtime_lif_visualization",
        "task": args.task,
        "num_envs": 1,
        "requested_steps": args.steps,
        "maximum_allowed_steps": MAX_VISUALIZATION_STEPS,
        "seed": args.seed,
        "deterministic_actions": True,
        "viewer_enabled": not bool(args.headless),
        "active_goal_marker": {
            "requested": not bool(args.headless),
            "meaning": "active waypoint; moves when WaypointSwitch advances",
            "update_mode": "initial_and_waypoint_switch_only",
        },
        "camera": {
            "mode": "close_follow_env0" if args.close_follow_camera else "default_viewer",
            "requested": bool(args.close_follow_camera),
            "active": bool(args.close_follow_camera and not args.headless),
            "world_frame_eye_offset_m": list(args.camera_offset),
            "refresh_interval_control_steps": CLOSE_FOLLOW_CAMERA_REFRESH_STEPS,
            "update_backend": "reused_usd_matrix_op_no_kit_commands",
        },
        "brain_window": {
            "requested": bool(args.brain_window),
            "active": bool(args.brain_window),
            "separate_process": True,
            "backend": "tkinter",
            "update_interval_control_steps": args.display_interval,
            "terminal_sidecar_retained": True,
        },
        "diagnostic_source_drift_allowed": bool(
            getattr(args, "allow_visualizer_source_drift", False)
        ),
        "terminal_lif_sidecar": True,
        "realtime_rate": args.realtime_rate,
        "display_interval_steps": args.display_interval,
        "activity_window_steps": args.activity_window_steps,
        "activity_bar_width": args.activity_bar_width,
        "stop_at_first_episode_end": True,
        "checkpoint_mutation_allowed": False,
        **identity,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Required values are checked after AppLauncher's pre-parse so --help works
    # without a checkpoint and without constructing SimulationApp.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--steps", type=int, default=MAX_VISUALIZATION_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime_rate", type=float, default=1.0)
    parser.add_argument("--display_interval", type=int, default=DEFAULT_DISPLAY_INTERVAL)
    parser.add_argument(
        "--activity_window_steps", type=int, default=DEFAULT_ACTIVITY_WINDOW_STEPS
    )
    parser.add_argument("--activity_bar_width", type=int, default=DEFAULT_ACTIVITY_BAR_WIDTH)
    parser.add_argument("--plain_console", action="store_true")
    parser.add_argument(
        "--close_follow_camera",
        action="store_true",
        help="track env 0 with a close world-frame viewer camera",
    )
    parser.add_argument(
        "--camera_offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_CLOSE_FOLLOW_CAMERA_OFFSET_M,
        help="close-follow eye offset from the Crazyflie in world metres",
    )
    parser.add_argument(
        "--brain_window",
        action="store_true",
        help="open a separate realtime Tk window for LIF activity",
    )
    parser.add_argument(
        "--allow_visualizer_source_drift",
        action="store_true",
        help=(
            "allow only audited drift in the playback/visualizer files while "
            "requiring every policy, task, contract, runtime, and config hash to match"
        ),
    )
    parser.add_argument("--expected_fingerprint")
    parser.add_argument("--output", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required")
    if not MIN_VISUALIZATION_STEPS <= args.steps <= MAX_VISUALIZATION_STEPS:
        parser.error(
            f"--steps must be between {MIN_VISUALIZATION_STEPS} and "
            f"{MAX_VISUALIZATION_STEPS}"
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not math.isfinite(args.realtime_rate) or not 0.1 <= args.realtime_rate <= 4.0:
        parser.error("--realtime_rate must be finite and in [0.1, 4.0]")
    if not 1 <= args.display_interval <= args.steps:
        parser.error("--display_interval must be in [1, --steps]")
    if not 1 <= args.activity_window_steps <= args.steps:
        parser.error("--activity_window_steps must be in [1, --steps]")
    if not 8 <= args.activity_bar_width <= 128:
        parser.error("--activity_bar_width must be between 8 and 128")
    try:
        args.camera_offset = _validate_camera_offset(args.camera_offset)
    except ValueError as exc:
        parser.error(str(exc))
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output == args.checkpoint:
        parser.error("--output must not be the checkpoint path")
    if args.output.exists():
        parser.error(f"Refusing to overwrite visualization report: {args.output}")
    return args


def main() -> int:
    args = _parse_args()
    try:
        payload, identity = inspect_checkpoint(
            args.checkpoint,
            args.expected_fingerprint,
            allowed_diagnostic_source_drift=(
                VISUALIZATION_ONLY_SOURCE_FILES
                if args.allow_visualizer_source_drift
                else ()
            ),
        )
        _validate_lif_controller(identity["controller"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Checkpoint validation failed: {exc}") from exc

    resolved = _resolved_config(args, identity)
    checkpoint_sha256_before = sha256_file(args.checkpoint)
    if checkpoint_sha256_before != identity["checkpoint_sha256"]:
        raise SystemExit("Checkpoint bytes changed during pre-launch validation")
    print(
        json.dumps(
            {"resolved_config": resolved, "fingerprint": identity["fingerprint"]},
            indent=2,
            sort_keys=True,
        )
    )

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    env = None
    wrapped = None
    device: torch.device | None = None
    memory_samples: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    controller_report: dict[str, Any] | None = None
    close_follow_camera: CloseFollowCamera | None = None
    camera_last_pose: dict[str, list[float]] | None = None
    camera_update_count = 0
    active_goal_marker_enabled = False
    active_goal_marker_update_count = 0
    brain_window: BrainWindow | None = None
    brain_window_audit: dict[str, Any] = {
        **resolved["brain_window"],
        "update_count": 0,
        "cleanup": "not_started",
    }
    core_checksum_before: str | None = None
    executed_steps = 0
    stopped_reason = "not_started"
    reward_sum = 0.0
    tracker = LIFActivityTracker(
        window_steps=args.activity_window_steps,
        bar_width=args.activity_bar_width,
    )
    sidecar = TerminalSidecar(plain=args.plain_console)
    sample_steps = set(_steady_state_sample_steps(args.steps))
    start_wall = time.perf_counter()
    try:
        from g1_fly_control.crazyflie.controllers import (
            controller_core_checksum,
            reset_controller_state,
            verify_frozen_core,
        )
        from g1_fly_control.crazyflie.memory import assess, reset_cuda_peak, snapshot
        from g1_fly_control.crazyflie.normalization import NormalizedEnv

        env = launch_environment(
            args.task,
            1,
            deterministic_evaluation=True,
            device=getattr(args, "device", None),
            contract_profile=identity["contract_profile"],
        )
        device = torch.device(env.device)
        policy, normalizer, controller_report = load_policy_and_normalizer(
            args.checkpoint, payload, identity, device
        )
        core_checksum_before = controller_core_checksum(policy)
        if core_checksum_before is None:
            raise RuntimeError("Loaded checkpoint did not construct a frozen LIF core")
        verify_frozen_core(policy, core_checksum_before)
        wrapped = NormalizedEnv(env, normalizer, training=False)
        observation, _ = wrapped.reset(seed=args.seed)
        policy_observation = observation["policy"]
        state = policy.initial_state(1, device=device)
        if not args.headless:
            active_goal_marker_enabled = _enable_bounded_goal_marker(env)
            if not active_goal_marker_enabled:
                raise RuntimeError("Crazyflie task did not enable its active-goal marker")
            active_goal_marker_update_count += 1
        if args.close_follow_camera and not args.headless:
            close_follow_camera = CloseFollowCamera(env, args.camera_offset)
            camera_last_pose = close_follow_camera.last_pose
            camera_update_count += 1
        if args.brain_window:
            brain_window = BrainWindow()
            brain_window_audit = {
                **resolved["brain_window"],
                **brain_window.audit(),
            }
        reset_cuda_peak(device)
        memory_samples.append(snapshot("environment_loaded", device, step=0))
        start_wall = time.perf_counter()
        stopped_reason = "step_limit"

        with torch.inference_mode():
            for step_index in range(1, args.steps + 1):
                if hasattr(app, "is_running") and not app.is_running():
                    stopped_reason = "viewer_closed"
                    break
                output_policy = _deterministic_lif_action(
                    policy, policy_observation, state
                )
                if output_policy.action.shape != (1, 4):
                    raise RuntimeError(
                        "LIF controller action shape changed: "
                        f"expected (1, 4), got {tuple(output_policy.action.shape)}"
                    )
                if not bool(torch.isfinite(output_policy.action).all()):
                    raise FloatingPointError("LIF controller emitted a nonfinite action")
                if float(output_policy.action.abs().max()) > 1.000001:
                    raise FloatingPointError("LIF controller emitted an out-of-range action")

                activity = tracker.update(output_policy.state)
                next_observation, reward, terminated, truncated, _ = wrapped.step(
                    output_policy.action
                )
                if (
                    active_goal_marker_enabled
                    and bool(env._switched_this_step[0])
                ):
                    _update_bounded_goal_marker(env)
                    active_goal_marker_update_count += 1
                if (
                    close_follow_camera is not None
                    and step_index % CLOSE_FOLLOW_CAMERA_REFRESH_STEPS == 0
                ):
                    camera_last_pose = close_follow_camera.update()
                    camera_update_count += 1
                done = terminated | truncated
                if not bool(torch.isfinite(next_observation["policy"]).all()):
                    raise FloatingPointError("Visualization encountered a nonfinite observation")
                if not bool(torch.isfinite(reward).all()):
                    raise FloatingPointError("Visualization encountered a nonfinite reward")

                executed_steps = step_index
                reward_value = float(reward[0])
                reward_sum += reward_value
                terminated_value = bool(terminated[0])
                truncated_value = bool(truncated[0])
                if bool(done[0]):
                    distance_m = float(env.terminal_distance_m[0])
                    speed_m_s = float(env.terminal_speed_mps[0])
                    success_count = int(env.terminal_success_count[0])
                    failure_cause = int(env.terminal_failure_cause[0])
                else:
                    physical = normalizer.denormalize(next_observation["policy"])
                    distance_m = float(torch.linalg.vector_norm(physical[0, 9:12]))
                    speed_m_s = float(torch.linalg.vector_norm(physical[0, 0:3]))
                    success_count = int(env.success_count[0])
                    failure_cause = 0

                deadline = _pace_deadline(start_wall, executed_steps, args.realtime_rate)
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                wall_time = max(time.perf_counter() - start_wall, 1.0e-9)

                if executed_steps in sample_steps:
                    memory_samples.append(
                        snapshot("steady_state", device, step=executed_steps)
                    )
                display_due = (
                    executed_steps == 1
                    or executed_steps % args.display_interval == 0
                    or bool(done[0])
                    or executed_steps == args.steps
                )
                if display_due:
                    frame = {
                        "step": executed_steps,
                        "requested_steps": args.steps,
                        "simulation_time_s": executed_steps * CONTROL_DT_S,
                        "wall_time_s": wall_time,
                        "realtime_factor": executed_steps * CONTROL_DT_S / wall_time,
                        "reward": reward_value,
                        "cumulative_reward": reward_sum,
                        "goal_distance_m": distance_m,
                        "speed_m_s": speed_m_s,
                        "success_count": success_count,
                        "terminated": terminated_value,
                        "truncated": truncated_value,
                        "failure_cause": failure_cause,
                        "action": [float(value) for value in output_policy.action[0]],
                        "neuron_count": tracker.neuron_count,
                        "lif_activity": activity,
                    }
                    telemetry.append(frame)
                    if brain_window is not None:
                        brain_window.show(frame, tracker.window_activity())
                    sidecar.show(
                        _format_sidecar(
                            frame,
                            memory_samples[-1] if memory_samples else None,
                        )
                    )

                state = reset_controller_state(policy, output_policy.state, done)
                policy_observation = next_observation["policy"]
                if terminated_value:
                    stopped_reason = "failure_termination"
                    break
                if truncated_value:
                    stopped_reason = "time_limit_truncation"
                    break

        if executed_steps and not any(
            sample.get("stage") == "steady_state"
            and sample.get("step") == executed_steps
            for sample in memory_samples
        ):
            memory_samples.append(snapshot("steady_state", device, step=executed_steps))
        memory_samples.append(snapshot("visualization_end", device, step=executed_steps))
        if brain_window is not None:
            brain_window_audit = {
                **resolved["brain_window"],
                **brain_window.close(),
            }
        memory_gate = assess(memory_samples)
        core_checksum_after = controller_core_checksum(policy)
        verify_frozen_core(policy, core_checksum_before)
        checkpoint_sha256_after = sha256_file(args.checkpoint)
        checkpoint_unchanged = checkpoint_sha256_after == checkpoint_sha256_before
        core_unchanged = core_checksum_after == core_checksum_before
        technical_failures = list(memory_gate["failures"])
        if not checkpoint_unchanged:
            technical_failures.append("checkpoint bytes changed during visualization")
        if not core_unchanged:
            technical_failures.append("frozen LIF core checksum changed during visualization")
        if executed_steps < 1:
            technical_failures.append("visualization executed zero control steps")
        activity_summary = tracker.summary() if executed_steps else None
        report = {
            "schema_version": 1,
            "status": "PASS" if not technical_failures else "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scientific_use": "diagnostic_only_not_heldout_evaluation",
            "resolved_config": resolved,
            "controller_report": controller_report,
            "executed_control_steps": executed_steps,
            "simulation_time_s": executed_steps * CONTROL_DT_S,
            "wall_time_s": max(time.perf_counter() - start_wall, 0.0),
            "stopped_reason": stopped_reason,
            "reward_sum": reward_sum,
            "flight_outcome_is_not_execution_status": True,
            "checkpoint_sha256_before": checkpoint_sha256_before,
            "checkpoint_sha256_after": checkpoint_sha256_after,
            "checkpoint_unchanged": checkpoint_unchanged,
            "frozen_core_checksum_before": core_checksum_before,
            "frozen_core_checksum_after": core_checksum_after,
            "frozen_core_unchanged": core_unchanged,
            "lif_activity_summary": activity_summary,
            "camera": {
                **resolved["camera"],
                "update_count": camera_update_count,
                "last_pose": camera_last_pose,
            },
            "active_goal_marker": {
                **resolved["active_goal_marker"],
                "enabled": active_goal_marker_enabled,
                "update_count": active_goal_marker_update_count,
                "update_mode": "initial_and_waypoint_switch_only",
            },
            "brain_window": brain_window_audit,
            "telemetry_sampling": {
                "stored_on_first_interval": True,
                "stored_every_n_intervals": args.display_interval,
                "stored_on_final_or_done_interval": True,
                "stored_frame_count": len(telemetry),
            },
            "telemetry": telemetry,
            "memory_sampling_contract": {
                "requested_quarter_steps": sorted(sample_steps),
                "steady_state_sample_count": sum(
                    sample.get("stage") == "steady_state" for sample in memory_samples
                ),
                "full_four_sample_growth_window_available": sum(
                    sample.get("stage") == "steady_state" for sample in memory_samples
                )
                >= 4,
                "early_episode_end_may_limit_growth_window": True,
            },
            "memory_samples": memory_samples,
            "memory_gate": memory_gate,
            "technical_failures": technical_failures,
        }
        _atomic_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.output),
                    "controller": identity["controller"],
                    "task": args.task,
                    "executed_control_steps": executed_steps,
                    "stopped_reason": stopped_reason,
                    "checkpoint_unchanged": checkpoint_unchanged,
                    "frozen_core_unchanged": core_unchanged,
                    "memory_gate": memory_gate,
                    "fingerprint": identity["fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "PASS" else 1
    except BaseException:
        error = traceback.format_exc()
        traceback.print_exc()
        if brain_window is not None:
            brain_window_audit = {
                **resolved["brain_window"],
                **brain_window.close(),
            }
        checkpoint_sha256_after = (
            sha256_file(args.checkpoint) if args.checkpoint.is_file() else None
        )
        memory_gate = None
        if device is not None:
            try:
                from g1_fly_control.crazyflie.memory import assess, snapshot

                memory_samples.append(
                    snapshot("visualization_failure", device, step=executed_steps)
                )
                memory_gate = assess(memory_samples)
            except BaseException:
                error += "\nMemory failure capture also failed:\n" + traceback.format_exc()
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scientific_use": "diagnostic_only_not_heldout_evaluation",
            "resolved_config": resolved,
            "controller_report": controller_report,
            "executed_control_steps": executed_steps,
            "stopped_reason": "technical_failure",
            "checkpoint_sha256_before": checkpoint_sha256_before,
            "checkpoint_sha256_after": checkpoint_sha256_after,
            "checkpoint_unchanged": checkpoint_sha256_after == checkpoint_sha256_before,
            "frozen_core_checksum_before": core_checksum_before,
            "camera": {
                **resolved["camera"],
                "update_count": camera_update_count,
                "last_pose": camera_last_pose,
            },
            "active_goal_marker": {
                **resolved["active_goal_marker"],
                "enabled": active_goal_marker_enabled,
                "update_count": active_goal_marker_update_count,
                "update_mode": "initial_and_waypoint_switch_only",
            },
            "brain_window": brain_window_audit,
            "telemetry": telemetry,
            "memory_samples": memory_samples,
            "memory_gate": memory_gate,
            "error": error,
        }
        _atomic_json(args.output, failure)
        return 1
    finally:
        if brain_window is not None:
            brain_window.close()
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
