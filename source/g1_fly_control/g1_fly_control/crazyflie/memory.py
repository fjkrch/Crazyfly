"""Process, system, PyTorch, and device-wide memory gates for Crazyflie runs."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import csv
import gc
import io
import math
from numbers import Integral, Real
import os
import subprocess
import sys
from typing import Any, Callable, Mapping, TypeVar

import psutil
import torch


MIB = 1024**2
GIB = 1024**3
MEMORY_POLICY_VERSION = "crazyflie_memory_acceptance_v2"
# User-selected machine guard, evaluated against device-wide nvidia-smi usage
# rather than only PyTorch allocations.  The comparison is exclusive: a
# sample exactly at 6.8 GiB also fails.
GPU_LIMIT_GIB = 6.8
GPU_LIMIT_MIB = GPU_LIMIT_GIB * 1024.0
RAM_LIMIT_PERCENT = 90.0
# RSS is sampled to 0.001 MiB and can drift by pages even at a true plateau.
# Treat a strictly increasing four-sample window as growth once its net rise
# exceeds this explicit measurement-noise allowance.
RSS_GROWTH_TOLERANCE_MIB = 1.0
SWAP_OUT_GROWTH_TOLERANCE_MIB = 1.0
T = TypeVar("T")


def release_checkpoint_serialization_heap() -> dict[str, Any]:
    """Return dead checkpoint-serialization allocations to the OS when possible.

    ``torch.save`` and streamed JSON validation create short-lived host objects.
    CPython and glibc may retain their freed arenas, which makes otherwise
    bounded checkpointing look like live RSS growth over a long run.  A full
    collection followed by glibc's ``malloc_trim`` releases only unreachable
    storage; live tensors and Python objects remain resident and therefore
    remain visible to the unchanged memory gate.

    The trim is a Linux/glibc optimization.  Other platforms still perform
    garbage collection and report that allocator trimming was unavailable.
    """

    collected = int(gc.collect())
    trim_supported = False
    trim_released = False
    if sys.platform.startswith("linux"):
        try:
            trim = getattr(ctypes.CDLL(None), "malloc_trim")
        except (AttributeError, OSError):
            pass
        else:
            trim_supported = True
            trim_released = bool(trim(0))
    return {
        "python_objects_collected": collected,
        "allocator_trim_supported": trim_supported,
        "allocator_trim_released": trim_released,
    }


def commit_checkpoint_then_release_heap(commit: Callable[[], T]) -> T:
    """Commit a checkpoint, then trim only after the commit frame unwinds.

    The callback owns transient metadata, RNG snapshots, and serialization
    arguments.  Its frame is gone before the cleanup runs, so only unreachable
    allocator storage can be released.  The original exception is preserved
    when a commit fails.
    """

    try:
        return commit()
    finally:
        release_checkpoint_serialization_heap()


def _nvidia_memory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    rows = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != 5:
            continue
        index, name, driver, total, used = (item.strip() for item in row)
        rows.append({
            "index": int(index),
            "name": name,
            "driver_version": driver,
            "total_mib": float(total),
            "used_mib": float(used),
        })
    return rows


def reset_cuda_peak(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def snapshot(stage: str, device: str | torch.device, *, step: int | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process(os.getpid())
    device_type = torch.device(device).type
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "step": step,
        "process_rss_mib": round(process.memory_info().rss / MIB, 3),
        "system_ram_total_gib": round(memory.total / GIB, 3),
        "system_ram_used_gib": round(memory.used / GIB, 3),
        "system_ram_available_gib": round(memory.available / GIB, 3),
        "system_ram_percent": round(memory.percent, 3),
        "system_swap_total_gib": round(swap.total / GIB, 3),
        "system_swap_used_gib": round(swap.used / GIB, 3),
        "system_swap_percent": round(swap.percent, 3),
        "system_swap_in_mib": round(swap.sin / MIB, 3),
        "system_swap_out_mib": round(swap.sout / MIB, 3),
        "compute_device_type": device_type,
        "gpu_devices": _nvidia_memory(),
    }
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        result.update({
            "torch_allocated_mib": round(torch.cuda.memory_allocated(device) / MIB, 3),
            "torch_reserved_mib": round(torch.cuda.memory_reserved(device) / MIB, 3),
            "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / MIB, 3),
            "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / MIB, 3),
        })
    else:
        result.update({
            "torch_allocated_mib": 0.0,
            "torch_reserved_mib": 0.0,
            "torch_peak_allocated_mib": 0.0,
            "torch_peak_reserved_mib": 0.0,
        })
    return result


def snapshot_after_heap_release(
    stage: str,
    device: str | torch.device,
    *,
    step: int | None = None,
) -> dict[str, Any]:
    """Sample live memory after releasing only unreachable allocator storage."""

    release_checkpoint_serialization_heap()
    return snapshot(stage, device, step=step)


def assess(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one memory sample is required")

    telemetry_failures: list[str] = []
    missing = object()

    def reject(location: str, requirement: str) -> None:
        telemetry_failures.append(
            f"invalid memory telemetry at {location}: {requirement}"
        )

    def finite_value(
        value: Any,
        location: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
    ) -> float | None:
        if value is missing:
            reject(location, "a finite non-Boolean number is required")
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            reject(location, "must be a finite non-Boolean number")
            return None
        number = float(value)
        if not math.isfinite(number):
            reject(location, "must be a finite non-Boolean number")
            return None
        if minimum is not None and (
            number <= minimum if exclusive_minimum else number < minimum
        ):
            comparison = ">" if exclusive_minimum else ">="
            reject(location, f"must be {comparison} {minimum}")
            return None
        if maximum is not None and number > maximum:
            reject(location, f"must be <= {maximum}")
            return None
        return number

    def required_text(sample: Mapping[str, Any], field: str, location: str) -> str | None:
        value = sample.get(field, missing)
        if not isinstance(value, str) or not value.strip():
            reject(location, "a non-empty string is required")
            return None
        return value

    def required_counter_or_none(
        sample: Mapping[str, Any], field: str, location: str
    ) -> int | None:
        value = sample.get(field, missing)
        if value is None:
            return None
        if (
            value is missing
            or isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            reject(location, "must be null or a nonnegative integer")
            return None
        return int(value)

    ram_values: list[float] = []
    rss_values: list[float | None] = []
    torch_peak_allocated_values: list[float | None] = []
    torch_peak_reserved_values: list[float | None] = []
    swap_out_values: list[float | None] = []
    gpu_values: list[float] = []
    stage_values: list[str | None] = []
    device_type_values: list[str | None] = []
    gpu_telemetry_complete_by_sample: list[bool] = []
    for sample_index, raw_sample in enumerate(samples):
        sample_location = f"samples[{sample_index}]"
        if not isinstance(raw_sample, Mapping):
            reject(sample_location, "must be a mapping")
            sample: Mapping[str, Any] = {}
        else:
            sample = raw_sample

        required_text(sample, "timestamp_utc", f"{sample_location}.timestamp_utc")
        stage_values.append(required_text(
            sample, "stage", f"{sample_location}.stage"
        ))
        required_counter_or_none(sample, "step", f"{sample_location}.step")
        device_type = required_text(
            sample, "compute_device_type", f"{sample_location}.compute_device_type"
        )
        if device_type is not None and device_type not in {"cpu", "cuda"}:
            reject(
                f"{sample_location}.compute_device_type",
                "must be 'cpu' or 'cuda'",
            )
            device_type = None
        device_type_values.append(device_type)

        ram_total = finite_value(
            sample.get("system_ram_total_gib", missing),
            f"{sample_location}.system_ram_total_gib",
            minimum=0.0,
            exclusive_minimum=True,
        )
        ram_used = finite_value(
            sample.get("system_ram_used_gib", missing),
            f"{sample_location}.system_ram_used_gib",
            minimum=0.0,
        )
        ram_available = finite_value(
            sample.get("system_ram_available_gib", missing),
            f"{sample_location}.system_ram_available_gib",
            minimum=0.0,
        )
        ram_value = finite_value(
            sample.get("system_ram_percent", missing),
            f"{sample_location}.system_ram_percent",
            minimum=0.0,
            maximum=100.0,
        )
        if ram_total is not None and ram_used is not None and ram_used > ram_total:
            reject(
                f"{sample_location}.system_ram_used_gib",
                "must be <= system_ram_total_gib",
            )
        if (
            ram_total is not None
            and ram_available is not None
            and ram_available > ram_total
        ):
            reject(
                f"{sample_location}.system_ram_available_gib",
                "must be <= system_ram_total_gib",
            )
        if ram_value is not None:
            ram_values.append(ram_value)
        rss_values.append(finite_value(
            sample.get("process_rss_mib", missing),
            f"{sample_location}.process_rss_mib",
            minimum=0.0,
        ))

        swap_total = finite_value(
            sample.get("system_swap_total_gib", missing),
            f"{sample_location}.system_swap_total_gib",
            minimum=0.0,
        )
        swap_used = finite_value(
            sample.get("system_swap_used_gib", missing),
            f"{sample_location}.system_swap_used_gib",
            minimum=0.0,
        )
        finite_value(
            sample.get("system_swap_percent", missing),
            f"{sample_location}.system_swap_percent",
            minimum=0.0,
            maximum=100.0,
        )
        finite_value(
            sample.get("system_swap_in_mib", missing),
            f"{sample_location}.system_swap_in_mib",
            minimum=0.0,
        )
        swap_out_values.append(finite_value(
            sample.get("system_swap_out_mib", missing),
            f"{sample_location}.system_swap_out_mib",
            minimum=0.0,
        ))
        if swap_total is not None and swap_used is not None and swap_used > swap_total:
            reject(
                f"{sample_location}.system_swap_used_gib",
                "must be <= system_swap_total_gib",
            )

        finite_value(
            sample.get("torch_allocated_mib", missing),
            f"{sample_location}.torch_allocated_mib",
            minimum=0.0,
        )
        finite_value(
            sample.get("torch_reserved_mib", missing),
            f"{sample_location}.torch_reserved_mib",
            minimum=0.0,
        )
        torch_peak_allocated_values.append(finite_value(
            sample.get("torch_peak_allocated_mib", missing),
            f"{sample_location}.torch_peak_allocated_mib",
            minimum=0.0,
        ))
        torch_peak_reserved_values.append(finite_value(
            sample.get("torch_peak_reserved_mib", missing),
            f"{sample_location}.torch_peak_reserved_mib",
            minimum=0.0,
        ))

        devices = sample.get("gpu_devices", missing)
        sample_gpu_complete = isinstance(devices, list) and bool(devices)
        if not isinstance(devices, list):
            reject(f"{sample_location}.gpu_devices", "a list is required")
            devices = []
        seen_device_indices: set[int] = set()
        for device_index, raw_device in enumerate(devices):
            device_location = f"{sample_location}.gpu_devices[{device_index}]"
            failure_count_before = len(telemetry_failures)
            if not isinstance(raw_device, Mapping):
                reject(device_location, "must be a mapping")
                sample_gpu_complete = False
                continue
            device = raw_device
            index = device.get("index", missing)
            if (
                index is missing
                or isinstance(index, bool)
                or not isinstance(index, Integral)
                or int(index) < 0
            ):
                reject(f"{device_location}.index", "a nonnegative integer is required")
                parsed_index = None
            else:
                parsed_index = int(index)
                if parsed_index in seen_device_indices:
                    reject(f"{device_location}.index", "GPU indices must be unique")
                seen_device_indices.add(parsed_index)
            required_text(device, "name", f"{device_location}.name")
            required_text(
                device, "driver_version", f"{device_location}.driver_version"
            )
            gpu_total = finite_value(
                device.get("total_mib", missing),
                f"{device_location}.total_mib",
                minimum=0.0,
                exclusive_minimum=True,
            )
            gpu_used = finite_value(
                device.get("used_mib", missing),
                f"{device_location}.used_mib",
                minimum=0.0,
            )
            if gpu_total is not None and gpu_used is not None:
                if gpu_used > gpu_total:
                    reject(
                        f"{device_location}.used_mib",
                        "must be <= total_mib",
                    )
                else:
                    gpu_values.append(gpu_used)
            if len(telemetry_failures) != failure_count_before:
                sample_gpu_complete = False
        gpu_telemetry_complete_by_sample.append(sample_gpu_complete)

    max_ram = max(ram_values, default=0.0)
    max_gpu = max(gpu_values, default=0.0)
    cuda_sample_indices = [
        index
        for index, (device_type, torch_allocated, torch_reserved) in enumerate(zip(
            device_type_values,
            torch_peak_allocated_values,
            torch_peak_reserved_values,
        ))
        if device_type == "cuda"
        or (torch_allocated is not None and torch_allocated > 0.0)
        or (torch_reserved is not None and torch_reserved > 0.0)
    ]
    missing_gpu_telemetry = bool(cuda_sample_indices) and any(
        not gpu_telemetry_complete_by_sample[index]
        for index in cuda_sample_indices
    )
    rss = [value for value in rss_values if value is not None]
    # A short smoke cannot prove absence of a leak. This gate flags a sustained,
    # material rise over the last four equally spaced steady-state observations.
    # Compare only like-for-like samples.  Environment construction and the
    # first optimizer allocation are intentional one-time jumps, so including
    # either in a steady-state leak window creates a false positive.
    steady_rss = [
        rss_value
        for stage, rss_value in zip(stage_values, rss_values)
        if stage in {"steady_state", "optimizer_update"}
        and rss_value is not None
    ]
    tail = steady_rss[-4:]
    monotonic_growth = (
        len(tail) == 4
        and all(b > a for a, b in zip(tail, tail[1:]))
        and tail[-1] - tail[0] > RSS_GROWTH_TOLERANCE_MIB
    )
    steady_swap_out = [
        swap_value
        for stage, swap_value in zip(stage_values, swap_out_values)
        if stage in {"steady_state", "optimizer_update"}
        and swap_value is not None
    ]
    swap_tail = steady_swap_out[-4:]
    sustained_paging = (
        len(swap_tail) == 4
        and all(b > a for a, b in zip(swap_tail, swap_tail[1:]))
        and swap_tail[-1] - swap_tail[0] > SWAP_OUT_GROWTH_TOLERANCE_MIB
    )
    failures = list(telemetry_failures)
    warnings = []
    if max_ram >= RAM_LIMIT_PERCENT:
        failures.append(f"system RAM reached {max_ram:.2f}% (limit < {RAM_LIMIT_PERCENT:.2f}%)")
    if max_gpu >= GPU_LIMIT_MIB:
        failures.append(f"device-wide GPU memory reached {max_gpu:.1f} MiB (limit < {GPU_LIMIT_MIB:.1f} MiB)")
    if missing_gpu_telemetry:
        failures.append("device-wide GPU telemetry was unavailable for at least one CUDA sample")
    if monotonic_growth:
        warnings.append(f"process RSS rose monotonically by {tail[-1] - tail[0]:.1f} MiB in the steady-state window")
    if sustained_paging:
        failures.append(
            f"system swap-out rose continuously by {swap_tail[-1] - swap_tail[0]:.1f} MiB "
            "in the steady-state window"
        )
    return {
        "policy_version": MEMORY_POLICY_VERSION,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "max_system_ram_percent": max_ram,
        "max_device_gpu_used_mib": max_gpu,
        "device_gpu_telemetry_complete": not missing_gpu_telemetry,
        "max_process_rss_mib": max(rss, default=0.0),
        "max_torch_allocated_mib": max(
            (value for value in torch_peak_allocated_values if value is not None),
            default=0.0,
        ),
        "max_torch_reserved_mib": max(
            (value for value in torch_peak_reserved_values if value is not None),
            default=0.0,
        ),
        "monotonic_process_growth_detected": monotonic_growth,
        "steady_state_growth_window_mib": tail,
        "sustained_paging_detected": sustained_paging,
        "steady_state_swap_out_window_mib": swap_tail,
        "limits": {
            "system_ram_percent_exclusive": RAM_LIMIT_PERCENT,
            "gpu_used_mib_exclusive": GPU_LIMIT_MIB,
            "rss_growth_tolerance_mib": RSS_GROWTH_TOLERANCE_MIB,
            "rss_growth_disposition": "warning_only",
            "swap_out_growth_tolerance_mib": SWAP_OUT_GROWTH_TOLERANCE_MIB,
            "gpu_telemetry_required_for_cuda": True,
        },
    }
