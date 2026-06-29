#!/usr/bin/env bash
# Re-run steps 1, 3, 4 for already-precomputed sequences (skip step 2).
# Usage: bash scripts/imed_rerun_done.sh [gpu_id]

set -euo pipefail

GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${REPO}/scripts"

SEQUENCES=(
    session_004_scene_2_tool_1
    session_004_scene_2_tool_2
    session_004_scene_2_tool_3
    session_004_scene_6_tool_1
    session_004_scene_6_tool_2
    session_004_scene_6_tool_3
    session_005_scene_7_tool_1
    session_005_scene_7_tool_2
    session_005_scene_7_tool_3
)

TOTAL=${#SEQUENCES[@]}
FAILED=()

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    echo ""
    echo "########################################"
    echo "  [$(( i + 1 ))/${TOTAL}] ${SEQ}"
    echo "########################################"

    bash "${SCRIPTS}/imed_step1_prepare.sh"   "${SEQ}"           || { FAILED+=("${SEQ}:step1"); continue; }
    bash "${SCRIPTS}/imed_step3_reconstruct.sh" "${SEQ}" "${GPU}" || { FAILED+=("${SEQ}:step3"); continue; }
    bash "${SCRIPTS}/imed_step4_evaluate.sh"  "${SEQ}" "${GPU}"  || { FAILED+=("${SEQ}:step4"); continue; }

    echo "Done: ${SEQ}"
done

echo ""
echo "========================================"
echo "  Finished. ${#FAILED[@]} failures."
for f in "${FAILED[@]:-}"; do echo "    - ${f}"; done
echo "========================================"
