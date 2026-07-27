#!/usr/bin/env bash
# Run the full iMED pipeline for all 20 sequences split across GPU 0 and
# GPU 1, the two halves processed concurrently (sequences within each half
# still run one at a time, sequentially, on their assigned GPU).
#
# Usage: bash scripts/imed_run_parallel.sh
#
# For an overnight/unattended run, background and detach this script itself:
#   nohup bash scripts/imed_run_parallel.sh > imed_run_parallel_launch.log 2>&1 &
#   disown
#
# Per-GPU logs: imed_run_gpu0_<timestamp>.log, imed_run_gpu1_<timestamp>.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${REPO}/scripts"
TS="$(date +%Y%m%d_%H%M%S)"

# Keep in sync with the ALL_SEQUENCES list in imed_run_all_20.sh.
SEQUENCES=(
    session_004_scene_2_tool_1
    session_004_scene_2_tool_2
    session_004_scene_2_tool_3
    session_004_scene_2_tool_4
    session_004_scene_6_tool_1
    session_004_scene_6_tool_2
    session_004_scene_6_tool_3
    session_005_scene_7_tool_1
    session_005_scene_7_tool_2
    session_005_scene_7_tool_3
    session_006_scene_7_tool_1
    session_006_scene_7_tool_2
    session_006_scene_7_tool_3
    session_007_scene_10_tool_1
    session_007_scene_10_tool_2
    session_007_scene_11_tool_1
    session_007_scene_11_tool_2
    session_007_scene_11_tool_3
    session_007_scene_5_tool_1
    session_007_scene_5_tool_2
)

# Interleave (round-robin) rather than split in half, so each GPU gets a mix
# of sessions/scenes instead of one GPU getting all the harder/longer ones.
GPU0_SEQS=()
GPU1_SEQS=()
for i in "${!SEQUENCES[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_SEQS+=("${SEQUENCES[$i]}")
    else
        GPU1_SEQS+=("${SEQUENCES[$i]}")
    fi
done

echo "GPU0 (${#GPU0_SEQS[@]} sequences): ${GPU0_SEQS[*]}"
echo "GPU1 (${#GPU1_SEQS[@]} sequences): ${GPU1_SEQS[*]}"

LOG0="${REPO}/imed_run_gpu0_${TS}.log"
LOG1="${REPO}/imed_run_gpu1_${TS}.log"

nohup bash "${SCRIPTS}/imed_run_all_20.sh" 0 "${GPU0_SEQS[@]}" > "${LOG0}" 2>&1 &
PID0=$!
nohup bash "${SCRIPTS}/imed_run_all_20.sh" 1 "${GPU1_SEQS[@]}" > "${LOG1}" 2>&1 &
PID1=$!

echo "Started GPU0 run (PID ${PID0}), log: ${LOG0}"
echo "Started GPU1 run (PID ${PID1}), log: ${LOG1}"
echo "Waiting for both to finish..."

STATUS=0
wait "${PID0}" || { echo "GPU0 run exited with a failure (see ${LOG0})"; STATUS=1; }
wait "${PID1}" || { echo "GPU1 run exited with a failure (see ${LOG1})"; STATUS=1; }

echo ""
echo "========================================"
echo "Both GPU runs finished."
echo "  GPU0 log: ${LOG0}"
echo "  GPU1 log: ${LOG1}"
echo "========================================"
exit "${STATUS}"
