#!/usr/bin/env bash
# Run the full iMED pipeline (steps 1-4) on all sequences.
# Usage: bash scripts/imed_run_all.sh [gpu_id]
# Runs sequentially; each sequence takes ~30-60 min.

set -euo pipefail

GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${REPO}/scripts"

SEQUENCES=(
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

TOTAL=${#SEQUENCES[@]}
FAILED=()

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    echo ""
    echo "########################################"
    echo "  [$(( i + 1 ))/${TOTAL}] ${SEQ}"
    echo "########################################"

    bash "${SCRIPTS}/imed_step1_prepare.sh"  "${SEQ}"          || { echo "FAILED step1: ${SEQ}"; FAILED+=("${SEQ}:step1"); continue; }
    bash "${SCRIPTS}/imed_step2_precompute.sh" "${SEQ}" "${GPU}" || { echo "FAILED step2: ${SEQ}"; FAILED+=("${SEQ}:step2"); continue; }
    bash "${SCRIPTS}/imed_step3_reconstruct.sh" "${SEQ}" "${GPU}" || { echo "FAILED step3: ${SEQ}"; FAILED+=("${SEQ}:step3"); continue; }
    bash "${SCRIPTS}/imed_step4_evaluate.sh"  "${SEQ}" "${GPU}" || { echo "FAILED step4: ${SEQ}"; FAILED+=("${SEQ}:step4"); continue; }

    echo "Done: ${SEQ}"
done

echo ""
echo "========================================"
echo "  All sequences finished."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED:"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
else
    echo "  No failures."
fi
echo "========================================"
