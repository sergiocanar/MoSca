#!/usr/bin/env bash
# Step 4: Render from Endoscope 1-L viewpoint and compute PSNR/SSIM.
# Usage: bash scripts/imed_step4_evaluate.sh <sequence_name> [gpu_id] [logdir] [--skip_render]
# Example: bash scripts/imed_step4_evaluate.sh session_004_scene_2_tool_1 0
#          bash scripts/imed_step4_evaluate.sh session_004_scene_2_tool_1 0 workspaces/.../logdir --skip_render

set -euo pipefail

SEQ="${1:?Usage: $0 <sequence_name> [gpu_id] [logdir] [--skip_render]}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="${REPO}/workspaces/${SEQ}"
PYTHON="/media/SSD0/scanar/anaconda3/envs/mosca/bin/python"

# Use provided logdir or auto-detect the latest one
if [ -n "${3:-}" ]; then
    LOGDIR="${3}"
else
    LOGDIR=$(ls -td "${WS}/logs"/imed_fit_*/ 2>/dev/null | head -1)
    if [ -z "${LOGDIR}" ]; then
        echo "ERROR: No logdir found under ${WS}/logs/. Run step 3 first or provide logdir as argument."
        exit 1
    fi
fi

echo "========================================"
echo "  iMED Step 4 — Evaluate NVS"
echo "  Sequence : ${SEQ}"
echo "  Workspace: ${WS}"
echo "  Logdir   : ${LOGDIR}"
echo "  GPU      : ${GPU}"
echo "========================================"

SKIP_RENDER=""
for arg in "$@"; do [ "$arg" = "--skip_render" ] && SKIP_RENDER="--skip_render"; done

cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" imed_evaluate.py \
    --ws "${WS}" \
    --logdir "${LOGDIR}" \
    ${SKIP_RENDER}

echo ""
echo "Done. Results: ${LOGDIR}/results.json"
