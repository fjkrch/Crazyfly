#!/usr/bin/env python3
"""Record host RAM and NVIDIA GPU memory without changing or stopping a workload.

Run alongside training, for example::

    python scripts/memory_watch.py --output runs/go2-memory.csv

GPU figures are device-wide and include simulator/driver allocations. PyTorch's
allocator figures in a training manifest are narrower and should not be added
to the device-wide figure.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

try:
    import psutil
except ImportError:  # Training can still run when the optional monitor extra is absent.
    psutil = None


GIB = 1024**3
CSV_FIELDS = (
    "timestamp_utc", "ram_used_gib", "ram_available_gib", "ram_total_gib", "ram_percent",
    "gpu_index", "gpu_name", "gpu_used_mib", "gpu_total_mib", "gpu_percent",
    "gpu_telemetry", "warning",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def read_gpu_memory() -> tuple[list[dict[str, Any]], str]:
    """Return per-device MiB use; unavailable telemetry never prevents RAM logging."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return [], "unavailable"
    if result.returncode != 0:
        return [], "query_failed"
    devices = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 4:
            continue
        index, name, total_text, used_text = (field.strip() for field in fields)
        total, used = _number(total_text), _number(used_text)
        devices.append({
            "gpu_index": index, "gpu_name": name,
            "gpu_total_mib": total, "gpu_used_mib": used,
            "gpu_percent": round(100 * used / total, 2) if used is not None and total and total > 0 else None,
        })
    return devices, "ok" if devices else "no_devices"


def read_ram_memory() -> dict[str, float]:
    if psutil is None:
        raise RuntimeError("psutil is required for RAM monitoring; install it with `python -m pip install psutil`.")
    memory = psutil.virtual_memory()
    return {
        "ram_used_gib": round(memory.used / GIB, 3),
        "ram_available_gib": round(memory.available / GIB, 3),
        "ram_total_gib": round(memory.total / GIB, 3),
        "ram_percent": round(memory.percent, 2),
    }


def training_memory_snapshot(device: Any, stage: str, *, iteration: int | None = None) -> dict[str, Any]:
    """Capture process/system RAM and synchronized PyTorch CUDA current/peak MiB.

    GPU peaks reflect the interval since ``reset_training_cuda_peak``. They omit
    non-PyTorch allocations, so use the standalone watcher for device-wide use.
    """
    import torch

    snapshot: dict[str, Any] = {"stage": stage, "timestamp_utc": _utc_now(), "iteration": iteration}
    if psutil is not None:
        process = psutil.Process()
        snapshot["process_rss_mib"] = round(process.memory_info().rss / (1024**2), 2)
        snapshot.update(read_ram_memory())
    else:
        snapshot["process_rss_mib"] = None
        snapshot["ram_telemetry"] = "psutil_unavailable"
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        snapshot.update({
            "torch_cuda_allocated_mib": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
            "torch_cuda_reserved_mib": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
            "torch_cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
            "torch_cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2),
        })
    devices, telemetry = read_gpu_memory()
    snapshot["gpu_devices"] = devices
    snapshot["gpu_telemetry"] = telemetry
    return snapshot


def reset_training_cuda_peak(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def run_monitor(output: Path, *, interval: float = 2.0, count: int | None = None, threshold: float = 85.0) -> int:
    if (
        not math.isfinite(interval) or interval <= 0
        or count is not None and count < 1
        or not math.isfinite(threshold) or not 0 < threshold <= 100
    ):
        raise ValueError("interval must be positive, count must be positive, and threshold must be in (0, 100].")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        sample = 0
        while count is None or sample < count:
            ram = read_ram_memory()
            devices, telemetry = read_gpu_memory()
            for device in devices or [{}]:
                row = {"timestamp_utc": _utc_now(), **ram, **device, "gpu_telemetry": telemetry}
                warnings = []
                if ram["ram_percent"] >= threshold:
                    warnings.append("RAM threshold")
                if device.get("gpu_percent") is not None and device["gpu_percent"] >= threshold:
                    warnings.append("GPU threshold")
                row["warning"] = "; ".join(warnings)
                writer.writerow(row)
                if warnings:
                    print(f"{row['timestamp_utc']} warning: {row['warning']}", file=sys.stderr)
            stream.flush()
            sample += 1
            if count is None or sample < count:
                time.sleep(interval)
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between polls (default: 2).")
    parser.add_argument("--count", type=int, help="Stop after this many polls; omit to run until interrupted.")
    parser.add_argument("--threshold", type=float, default=85.0, help="Warn at this RAM or GPU utilization percentage.")
    args = parser.parse_args()
    try:
        run_monitor(args.output, interval=args.interval, count=args.count, threshold=args.threshold)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
