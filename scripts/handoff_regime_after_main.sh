#!/usr/bin/env bash
# Wait for the current main wrapper, validate its completed report, then replay regimes.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${FLYG1_PYTHON:-/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python}"
exec "$python_bin" scripts/handoff_regime_after_main.py "$@"
