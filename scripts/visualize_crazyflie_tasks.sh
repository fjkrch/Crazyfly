#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python"
CHECKPOINT="${CRAZYFLIE_CHECKPOINT:-${ROOT_DIR}/runs/crazyflie-balanced-v3-lif-proof/checkpoints/latest.pt}"
OUTPUT_DIR="${CRAZYFLIE_VIS_DIR:-${ROOT_DIR}/runs/crazyflie-balanced-v3-lif-proof/visualizations}"
EXPECTED_FINGERPRINT="${CRAZYFLIE_EXPECTED_FINGERPRINT:-}"
EXPECTED_FINGERPRINT_ARGS=()
if [[ -n "${EXPECTED_FINGERPRINT}" ]]; then
    EXPECTED_FINGERPRINT_ARGS=(--expected_fingerprint "${EXPECTED_FINGERPRINT}")
fi
STEPS="${CRAZYFLIE_VIS_STEPS:-600}"
MODE="${1:-all}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
    echo "Usage: $0 [reach|switch|gust|all]"
    echo "Plays the exact balanced-v3 frozen-LIF proof checkpoint with a close camera and separate realtime brain window."
    echo "Overrides: CRAZYFLIE_CHECKPOINT, CRAZYFLIE_VIS_DIR, CRAZYFLIE_VIS_STEPS, CRAZYFLIE_EXPECTED_FINGERPRINT"
    exit 0
fi

if [[ ! -x "${ISAAC_PY}" ]]; then
    echo "Isaac Python not executable: ${ISAAC_PY}" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi
if pgrep -f 'scripts/drone_(train|evaluate|visualize_lif)\.py' >/dev/null; then
    echo "Another Crazyflie train/evaluate/visualize process is already running." >&2
    echo "Run this script again after that process exits; visualizations are intentionally sequential." >&2
    exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/source/g1_fly_control:${ROOT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

run_task() {
    local short_name="$1"
    local task_id="$2"
    local report="${OUTPUT_DIR}/${short_name}-close-brain-${STAMP}.json"

    echo "Starting ${task_id}"
    echo "Report: ${report}"
    "${ISAAC_PY}" "${ROOT_DIR}/scripts/drone_visualize_lif.py" \
        --checkpoint "${CHECKPOINT}" \
        --task "${task_id}" \
        --steps "${STEPS}" \
        --seed 0 \
        --realtime_rate 1.0 \
        --display_interval 5 \
        --activity_window_steps 25 \
        --activity_bar_width 48 \
        --close_follow_camera \
        --camera_offset -1.15 -1.15 0.55 \
        --brain_window \
        --device cuda:0 \
        "${EXPECTED_FINGERPRINT_ARGS[@]}" \
        --output "${report}"
}

case "${MODE}" in
    reach)
        run_task reach FlyCrazyflie-WaypointReach-v0
        ;;
    switch)
        run_task switch FlyCrazyflie-WaypointSwitch-v0
        ;;
    gust)
        run_task gust FlyCrazyflie-GustRecovery-v0
        ;;
    all)
        run_task reach FlyCrazyflie-WaypointReach-v0
        run_task switch FlyCrazyflie-WaypointSwitch-v0
        run_task gust FlyCrazyflie-GustRecovery-v0
        ;;
    *)
        echo "Usage: $0 [reach|switch|gust|all]" >&2
        exit 2
        ;;
esac
