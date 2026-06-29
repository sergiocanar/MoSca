#!/usr/bin/env bash
# Step 2: Run TAP tracking (skip depth/flow — both are pre-provided).
# Usage: bash scripts/imed_step2_precompute.sh <sequence_name> [gpu_id]
# Example: bash scripts/imed_step2_precompute.sh session_004_scene_2_tool_1 0

set -euo pipefail

SEQ="${1:?Usage: $0 <sequence_name> [gpu_id]}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="${REPO}/workspaces/${SEQ}"
PYTHON="/media/SSD0/scanar/anaconda3/envs/mosca/bin/python"

echo "========================================"
echo "  iMED Step 2 — Precompute (TAP tracking)"
echo "  Sequence : ${SEQ}"
echo "  Workspace: ${WS}"
echo "  GPU      : ${GPU}"
echo "========================================"

cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" mosca_precompute.py \
    --cfg profile/imed/imed_prep.yaml \
    --ws "${WS}"

echo "Done. TAP tracks saved in: ${WS}"
