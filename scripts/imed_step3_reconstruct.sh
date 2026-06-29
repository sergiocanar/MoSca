#!/usr/bin/env bash
# Step 3: Run MoSca reconstruction (frozen cameras, scaffold + photometric GS).
# Usage: bash scripts/imed_step3_reconstruct.sh <sequence_name> [gpu_id]
# Example: bash scripts/imed_step3_reconstruct.sh session_004_scene_2_tool_1 0

set -euo pipefail

SEQ="${1:?Usage: $0 <sequence_name> [gpu_id]}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="${REPO}/workspaces/${SEQ}"
PYTHON="/media/SSD0/scanar/anaconda3/envs/mosca/bin/python"

echo "========================================"
echo "  iMED Step 3 — Reconstruct"
echo "  Sequence : ${SEQ}"
echo "  Workspace: ${WS}"
echo "  GPU      : ${GPU}"
echo "========================================"

cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" mosca_reconstruct.py \
    --cfg profile/imed/imed_fit.yaml \
    --ws "${WS}"

# Print the latest logdir for step 4
LOGDIR=$(ls -td "${WS}/logs"/imed_fit_*/ 2>/dev/null | head -1)
echo ""
echo "Done. Latest logdir: ${LOGDIR}"
echo "Run step 4 with:"
echo "  bash scripts/imed_step4_evaluate.sh ${SEQ} ${GPU} ${LOGDIR}"
