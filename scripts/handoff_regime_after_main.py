#!/usr/bin/env python3
"""Wait for one live main wrapper, then launch regime replay only after validation."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import ROOT
from run_matrix import _execution_source_fingerprint, _queue_lock, QueueLockedError
from summarize_matrix import summarize


MAIN = ROOT / "runs/main_matrix_malecns_v1_heldout_v3_20260913.json"
REPORT = ROOT / "runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.json"
WRAPPER = ROOT / "scripts/execute_main_matrix_v3.sh"
REGIME_WRAPPER = ROOT / "scripts/execute_regime_matrix.sh"
SIMULATOR_SCRIPTS = frozenset({
    "execute_main_matrix_v3.sh", "execute_regime_matrix.sh",
    "run_matrix.py", "run_regime_matrix.py", "ablate.py", "doctor.py",
    "train.py", "evaluate.py", "record_heldout_regimes.py",
    "smoke_env.py", "inspect_asset.py", "play.py", "record.py",
})


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _wrapper_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, or None after the specific PID exits."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])  # stat field 22: starttime
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        # /proc can remove a process between path lookup and the stat read.
        if exc.errno in {errno.ENOENT, errno.ESRCH, errno.ENOTDIR}:
            return None
        raise


def _require_current_wrapper(pid: int, expected_ticks: int | None) -> int:
    ticks = _wrapper_start_ticks(pid)
    if ticks is None or (expected_ticks is not None and ticks != expected_ticks):
        raise ValueError("The specified main wrapper is not the live process expected at handoff startup.")
    proc = Path("/proc") / str(pid)
    argv = [item.decode("utf-8", errors="replace") for item in
            (proc / "cmdline").read_bytes().split(b"\0") if item]
    cwd = (proc / "cwd").resolve(strict=True)
    if cwd != ROOT or not any(Path(arg).name == WRAPPER.name for arg in argv[1:3]):
        raise ValueError(f"PID {pid} is not the main matrix wrapper in {ROOT}.")
    return ticks


def _active_simulator_pids() -> list[int]:
    """Catch an orphaned main child or another project simulator worker."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [item.decode("utf-8", errors="replace") for item in
                    (proc / "cmdline").read_bytes().split(b"\0") if item]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg).name in SIMULATOR_SCRIPTS for arg in argv[1:4]):
            found.append(int(proc.name))
    return sorted(found)


def _validate_report() -> None:
    main = _json_object(MAIN)
    saved = _json_object(REPORT)
    if main.get("status") != "complete" or main.get("counts") != {"passed": 20}:
        raise ValueError("Main matrix has not finished with 20 passed jobs.")
    if (saved.get("status") != "complete" or type(saved.get("expected_jobs")) is not int
            or saved["expected_jobs"] != 20 or type(saved.get("validated_jobs")) is not int
            or saved["validated_jobs"] != 20 or saved.get("validation_errors") != []
            or saved.get("missing_or_unfinished_jobs") != []):
        raise ValueError("Saved comparison report is absent, incomplete, or has validation errors.")
    if (saved.get("manifest") != str(MAIN.resolve())
            or saved.get("source_execution_fingerprint") != main.get("execution_source_fingerprint")
            or main.get("execution_source_fingerprint") != _execution_source_fingerprint()):
        raise ValueError("Saved comparison report or execution source does not match this main matrix.")
    # Rebuild from on-disk checkpoint manifests and held-out episode files. Exact
    # equality also checks the report's manifest/source hashes and aggregations.
    fresh = summarize(MAIN)
    if fresh != saved:
        raise ValueError("Saved comparison report is stale or differs from a fresh source-backed validation.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-pid", type=int, help="PID of the currently running main wrapper")
    parser.add_argument("--main-start-ticks", type=int,
                        help="Optional /proc/<pid>/stat starttime to pin an exact wrapper instance")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate the current gate once; do not wait or start Isaac")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error("--poll-seconds must be in (0, 60].")
    if not args.check_only and (args.main_pid is None or args.main_pid <= 0):
        parser.error("Normal handoff requires --main-pid for the live main wrapper.")
    try:
        if args.check_only:
            _validate_report()
            print("Comparison report gate: complete and source-backed. Check-only; regime replay not started.")
            return 0
        ticks = _require_current_wrapper(args.main_pid, args.main_start_ticks)
        print(f"Waiting for main wrapper PID {args.main_pid} (start ticks {ticks}).", flush=True)
        while _wrapper_start_ticks(args.main_pid) == ticks:
            time.sleep(args.poll_seconds)
        # Keep the main queue locked through regime execution, so a resumed main
        # runner cannot begin while this separate Isaac workload is active.
        with _queue_lock(MAIN):
            active = _active_simulator_pids()
            if active:
                raise ValueError(f"Simulator workers are still active: PIDs {active}.")
            _validate_report()
            print("Main report validated; starting sequential regime replay.", flush=True)
            return subprocess.run(["bash", str(REGIME_WRAPPER)], cwd=ROOT, check=False).returncode
    except (OSError, ValueError, json.JSONDecodeError, QueueLockedError) as exc:
        print(f"Handoff refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
