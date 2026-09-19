#!/usr/bin/env bash
# Execute or resume only the predeclared four-cell Crazyflie integration queue.

set -euo pipefail

if (($# == 1)) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "usage: bash scripts/execute_drone_integration.sh"
  echo "Execute or resume the fixed four-controller, seed-0 integration queue sequentially."
  exit 0
fi
if (($# != 0)); then
  echo "usage: bash scripts/execute_drone_integration.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python}"
CONFIG="$PROJECT_ROOT/configs/experiments/crazyflie_balanced_v3_integration.json"
QUEUE="$PROJECT_ROOT/runs/crazyflie_balanced_v3_integration_v1.json"
ARTIFACT_ROOT="$PROJECT_ROOT/runs/crazyflie_balanced_v3_integration_v1"
PAUSE_FILE="$ARTIFACT_ROOT/pause.request"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Python is not executable: $ISAAC_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Integration config is missing: $CONFIG" >&2
  exit 2
fi
if [[ -e "$QUEUE.lock" ]]; then
  echo "Queue lock exists; another runner may be active: $QUEUE.lock" >&2
  exit 2
fi

# On resume, reject source/config/fingerprint/command drift before any job is
# launched. Only runner-owned status, timestamp, resume, and run-result fields
# are mutable. A persistent pause request is archived by the runner as part of
# its recorded resume transaction; this launcher never unlinks it.
if [[ -f "$QUEUE" ]]; then
  "$ISAAC_PYTHON" - "$PROJECT_ROOT" "$CONFIG" "$QUEUE" <<'PY'
import json
from pathlib import Path
import sys

root, config_path, queue_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "scripts"))
import drone_run_matrix as matrix

config = matrix.validate_config(config_path)
actual = json.loads(queue_path.read_text(encoding="utf-8"))
expected = matrix.build_queue(config, queue_path)

def immutable_view(queue):
    value = json.loads(json.dumps(queue))
    for key in (
        "created_utc", "status", "dry_run", "counts", "updated_utc",
        "resume_history", "last_resume",
    ):
        value.pop(key, None)
    for job in value.get("jobs", []):
        for key in (
            "status", "started_utc", "finished_utc", "training_run", "failure",
            "pause_reason", "previous_pause_reason", "artifact_revalidation",
        ):
            job.pop(key, None)
        for evaluation in job.get("evaluations", []):
            evaluation.pop("status", None)
            evaluation.pop("run", None)
    return value

if immutable_view(actual) != immutable_view(expected):
    raise SystemExit(
        "integration queue immutable content differs from the fixed config/current fingerprint"
    )
PY
fi

runner=(
  "$ISAAC_PYTHON" "$PROJECT_ROOT/scripts/drone_run_matrix.py"
  --config "$CONFIG"
  --output "$QUEUE"
  --execute
)
if [[ -f "$QUEUE" ]]; then
  runner+=(--resume)
fi

echo "Executing the bounded integration queue with one Isaac process at a time."
echo "Interpreter: $ISAAC_PYTHON"
echo "Config: $CONFIG"
echo "Queue: $QUEUE"
if [[ -f "$QUEUE" && -e "$PAUSE_FILE" ]]; then
  echo "Resume will archive the persistent pause request and record the resume event."
fi

set +e
"${runner[@]}"
runner_status=$?
set -e

summary_status=0
if [[ -f "$QUEUE" ]]; then
  "$ISAAC_PYTHON" "$PROJECT_ROOT/scripts/drone_summarize_matrix.py" \
    --queue "$QUEUE" || summary_status=$?
fi
if ((runner_status != 0)); then
  exit "$runner_status"
fi
exit "$summary_status"
