#!/usr/bin/env python3
"""Report usable/blocked capabilities without launching Isaac Sim."""

from __future__ import annotations

import importlib.util
from importlib import metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


def module_version(name: str) -> dict[str, str | None]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"status": "BLOCKED", "version": None, "location": None}
    try:
        module = __import__(name)
        distribution = {"isaacsim": "isaacsim", "rsl_rl": "rsl-rl-lib"}.get(name, name)
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        return {"status": "PASS", "version": version, "location": getattr(module, "__file__", None)}
    except Exception as exc:  # imports may need SimulationApp; that is a capability, not a false failure
        return {"status": "BLOCKED", "version": type(exc).__name__, "location": spec.origin}


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main() -> int:
    report: dict[str, object] = {
        "status": "PASS",
        "project_root": str(ROOT),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "modules": {name: module_version(name) for name in ("torch", "gymnasium", "isaaclab", "isaacsim", "rsl_rl")},
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "isaac_lab_checkout": "/home/chayanin/Downloads/IsaacLab",
    }
    checkout = Path(str(report["isaac_lab_checkout"]))
    report["isaac_lab_checkout_status"] = "PASS" if checkout.is_dir() else "BLOCKED"
    if not checkout.is_dir() or any(value["status"] != "PASS" for value in report["modules"].values()):
        report["status"] = "BLOCKED"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
