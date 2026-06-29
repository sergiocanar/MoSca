#!/usr/bin/env bash
# Run full iMED pipeline on all 20 sequences.
# Step 2 (TAP tracking) is skipped when tracks already exist.
# Usage: bash scripts/imed_run_all_20.sh [gpu_id]
# Example: bash scripts/imed_run_all_20.sh 0

set -euo pipefail

GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${REPO}/scripts"

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

TOTAL=${#SEQUENCES[@]}
FAILED=()

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    WS="${REPO}/workspaces/${SEQ}"

    echo ""
    echo "########################################"
    echo "  [$(( i + 1 ))/${TOTAL}] ${SEQ}"
    echo "########################################"

    # Step 1: always re-prepare (idempotent — skips existing files)
    bash "${SCRIPTS}/imed_step1_prepare.sh" "${SEQ}" \
        || { FAILED+=("${SEQ}:step1"); continue; }

    # Step 2: TAP tracking — skip if both npz files already exist
    UNIFORM_NPZ="${WS}/uniform_dep=sensor_bootstapir_tap.npz"
    DYNAMIC_NPZ="${WS}/dynamic_dep=sensor_bootstapir_tap.npz"
    if [ -f "${UNIFORM_NPZ}" ] && [ -f "${DYNAMIC_NPZ}" ]; then
        echo "  [step2] TAP tracks already exist, skipping."
    else
        bash "${SCRIPTS}/imed_step2_precompute.sh" "${SEQ}" "${GPU}" \
            || { FAILED+=("${SEQ}:step2"); continue; }
    fi

    # Step 3: reconstruct
    bash "${SCRIPTS}/imed_step3_reconstruct.sh" "${SEQ}" "${GPU}" \
        || { FAILED+=("${SEQ}:step3"); continue; }

    # Step 4: evaluate (uses the latest logdir automatically)
    bash "${SCRIPTS}/imed_step4_evaluate.sh" "${SEQ}" "${GPU}" \
        || { FAILED+=("${SEQ}:step4"); continue; }

    echo "  Done: ${SEQ}"
done

echo ""
echo "========================================"
echo "  Finished all ${TOTAL} sequences."
echo "  Failures: ${#FAILED[@]}"
for f in "${FAILED[@]:-}"; do echo "    - ${f}"; done
echo "========================================"
