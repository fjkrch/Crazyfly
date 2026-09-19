#!/usr/bin/env bash
# Explicitly authorized launcher for the reviewed 20-cell Crazyflie main queue.

set -euo pipefail

if (($# == 1)) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "usage: bash scripts/execute_drone_matrix.sh --authorize_main"
  echo "Execute or resume only an existing reviewed 20-job main dry-run queue."
  exit 0
fi
if (($# != 1)) || [[ "$1" != "--authorize_main" ]]; then
  echo "Main execution is safety-locked." >&2
  echo "After explicit user authorization, run:" >&2
  echo "  bash scripts/execute_drone_matrix.sh --authorize_main" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python}"
CONFIG="$PROJECT_ROOT/configs/experiments/crazyflie_balanced_v3_main.json"
QUEUE="$PROJECT_ROOT/runs/crazyflie_balanced_v3_main_v1.json"
ARTIFACT_ROOT="$PROJECT_ROOT/runs/crazyflie_balanced_v3_main_v1"
PAUSE_FILE="$ARTIFACT_ROOT/pause.request"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Python is not executable: $ISAAC_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Main config is missing: $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$QUEUE" ]]; then
  echo "Reviewed main dry-run queue is missing: $QUEUE" >&2
  echo "Create and review it with drone_run_matrix.py --dry_run; this launcher will not create it." >&2
  exit 2
fi
if [[ -e "$QUEUE.lock" ]]; then
  echo "Queue lock exists; another runner may be active: $QUEUE.lock" >&2
  exit 2
fi

# Revalidate the fixed, exact 500k balanced-v3 original-LIF proof before any
# main-queue reconstruction, authorization message, status mutation, or
# runner process.  This verifier is CPU/read-only and exits nonzero if any
# source, checkpoint, memory, evaluation, or per-scenario success evidence is
# absent or stale.
"$ISAAC_PYTHON" "$PROJECT_ROOT/scripts/drone_verify_lif_proof.py"

# Rebuild every immutable command/fingerprint/path from current source and
# config before forwarding the separate authorization flag to the runner.
# Runner-owned status and result fields are the only allowed differences.
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
        "reviewed main queue immutable content differs from the fixed config/current fingerprint"
    )
PY

echo "MAIN MATRIX AUTHORIZATION ACCEPTED: executing the reviewed queue sequentially."
echo "Interpreter: $ISAAC_PYTHON"
echo "Config: $CONFIG"
echo "Queue: $QUEUE"
if [[ -e "$PAUSE_FILE" ]]; then
  echo "Authorized resume will archive the persistent pause request and record the resume event."
fi

set +e
"$ISAAC_PYTHON" "$PROJECT_ROOT/scripts/drone_run_matrix.py" \
  --config "$CONFIG" \
  --output "$QUEUE" \
  --execute \
  --resume \
  --authorize_main
runner_status=$?
set -e

summary_status=0
"$ISAAC_PYTHON" "$PROJECT_ROOT/scripts/drone_summarize_matrix.py" \
  --queue "$QUEUE" || summary_status=$?
if ((runner_status != 0)); then
  exit "$runner_status"
fi
exit "$summary_status"
