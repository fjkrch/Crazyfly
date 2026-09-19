"""The monitor must keep logging host RAM when GPU telemetry is unavailable."""

import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import memory_watch  # noqa: E402


def _fake_psutil(percent: float = 90.0):
    return SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(
            used=22 * memory_watch.GIB,
            available=2 * memory_watch.GIB,
            total=24 * memory_watch.GIB,
            percent=percent,
        ),
        Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=512 * 1024**2)),
    )


def test_missing_nvidia_smi_preserves_ram_logging(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(memory_watch, "psutil", _fake_psutil())

    def missing_smi(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(memory_watch.subprocess, "run", missing_smi)
    output = tmp_path / "nested" / "memory.csv"
    assert memory_watch.run_monitor(output, interval=0.1, count=1) == 1
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["ram_available_gib"] == "2.0"
    assert rows[0]["gpu_telemetry"] == "unavailable"
    assert rows[0]["gpu_used_mib"] == ""
    assert rows[0]["warning"] == "RAM threshold"
    assert "RAM threshold" in capsys.readouterr().err


def test_gpu_rows_warn_on_device_usage_without_stopping(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_watch, "psutil", _fake_psutil(percent=50.0))
    monkeypatch.setattr(memory_watch.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0,
        stdout='0, "RTX 4060 Laptop GPU", 8192, 7000\n1, "N/A device", N/A, N/A\n',
    ))
    output = tmp_path / "memory.csv"
    memory_watch.run_monitor(output, interval=0.1, count=1)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["gpu_percent"] == "85.45"
    assert rows[0]["warning"] == "GPU threshold"
    assert rows[1]["gpu_percent"] == ""
    assert rows[1]["warning"] == ""


def test_cpu_training_snapshot_contains_process_and_system_memory(monkeypatch):
    monkeypatch.setattr(memory_watch, "psutil", _fake_psutil(percent=75.0))
    monkeypatch.setattr(memory_watch, "read_gpu_memory", lambda: ([], "unavailable"))
    snapshot = memory_watch.training_memory_snapshot(torch.device("cpu"), "rollout", iteration=7)
    assert snapshot["stage"] == "rollout"
    assert snapshot["iteration"] == 7
    assert snapshot["process_rss_mib"] == 512.0
    assert snapshot["ram_percent"] == 75.0
    assert snapshot["gpu_telemetry"] == "unavailable"
    assert "torch_cuda_peak_allocated_mib" not in snapshot


def test_invalid_monitor_limits_are_rejected(tmp_path):
    for kwargs in ({"interval": 0}, {"interval": float("nan")}, {"count": 0},
                   {"threshold": 0}, {"threshold": float("nan")}):
        try:
            memory_watch.run_monitor(tmp_path / "memory.csv", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
