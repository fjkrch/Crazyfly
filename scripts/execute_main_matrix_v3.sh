#!/usr/bin/env bash
# Run the pinned main comparison, then write its validated comparison report.
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${FLYG1_PYTHON:-/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python}"
matrix="runs/main_matrix_malecns_v1_heldout_v3_20260913.json"

"$python_bin" -u scripts/run_matrix.py \
  --config configs/experiments/main.json \
  --execute --resume --output "$matrix" \
  --smoke_report runs/smoke-16x1000-low-amplitude.json \
  --connectome_manifest data/connectome/manifest.json
run_status=$?

"$python_bin" scripts/summarize_matrix.py \
  --manifest "$matrix" \
  --output runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.json \
  --markdown runs/main_matrix_malecns_v1_heldout_v3_20260913_comparison.md
report_status=$?

if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi
exit "$report_status"
