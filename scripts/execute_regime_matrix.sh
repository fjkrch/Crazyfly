#!/usr/bin/env bash
# After the main matrix completes, record paired pose/contact traces and report them.
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${FLYG1_PYTHON:-/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python}"
main="runs/main_matrix_malecns_v1_heldout_v3_20260913.json"
regime="runs/main_matrix_malecns_v1_heldout_v3_20260913_regimes.json"
prefix="runs/main_matrix_malecns_v1_heldout_v3_20260913_regimes_comparison"

resume_args=()
if [ -f "$regime" ]; then
  resume_args=(--resume)
fi

"$python_bin" scripts/run_regime_matrix.py \
  --main_matrix "$main" --output "$regime" --execute "${resume_args[@]}"
run_status=$?

if [ ! -f "$regime" ]; then
  exit "$run_status"
fi

"$python_bin" scripts/summarize_regime_matrix.py \
  --manifest "$regime" --force-threshold-n 20 --upright-cos-threshold 0.5 \
  --output-prefix "$prefix"
report_status=$?

if [ "$run_status" -eq 0 ] && [ "$report_status" -eq 0 ]; then
  for threshold in 10 50; do
    "$python_bin" scripts/summarize_regime_matrix.py \
      --manifest "$regime" --force-threshold-n "$threshold" --upright-cos-threshold 0.5 \
      --output-prefix "${prefix}_${threshold}n"
    sensitivity_status=$?
    if [ "$sensitivity_status" -ne 0 ]; then
      exit "$sensitivity_status"
    fi
  done
fi

if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi
exit "$report_status"
