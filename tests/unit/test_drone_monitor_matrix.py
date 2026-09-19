"""CPU-only checks for the read-only Crazyflie queue monitor."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import drone_monitor_matrix as monitor  # noqa: E402


def _fake_process(proc: Path, pid: int, command: list[str]) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
    (directory / "status").write_text("VmRSS:\t1024 kB\nVmHWM:\t2048 kB\n", encoding="utf-8")


def test_process_match_accepts_only_fresh_or_exact_resume_command(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    command = ["python", "/repo/scripts/drone_train.py", "--policy", "mlp_normal"]
    _fake_process(proc, 101, command)
    _fake_process(proc, 102, [*command, "--resume"])
    _fake_process(proc, 103, [*command, "--unrelated"])

    matches = monitor._matching_processes([command], proc)

    assert matches is not None
    assert [row["pid"] for row in matches] == [101, 102]


def test_snapshot_print_identity_is_explicitly_queue_declared(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    config = {"controllers": ["frozen_lif_original"], "seeds": [0]}
    queue_path.write_text(json.dumps({
        "schema_version": 1,
        "label": "integration",
        "status": "dry_run",
        "config": config,
        "config_sha256": "c" * 64,
        "jobs": [{
            "id": "frozen_lif_original__seed-0",
            "controller": "frozen_lif_original",
            "seed": 0,
            "status": "pending",
            "run_dir": str(tmp_path / "job"),
            "expected_fingerprint": "f" * 64,
            "training_command": [],
            "evaluations": [],
        }],
    }), encoding="utf-8")

    report = monitor.snapshot(queue_path)

    assert report["fingerprint_scope"] == "per_job"
    assert report["fingerprints_source"] == "queue_declared_expected_fingerprint"
    assert report["resolved_config"] == config
    assert report["config_sha256"] == "c" * 64
    assert report["fingerprints"] == {"frozen_lif_original__seed-0": "f" * 64}
