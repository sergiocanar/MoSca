#!/usr/bin/env bash
# Step 1: Convert an iMED sequence directory into a MoSca workspace.
# Usage: bash scripts/imed_step1_prepare.sh <sequence_name>
# Example: bash scripts/imed_step1_prepare.sh session_004_scene_2_tool_1

set -euo pipefail

SEQ="${1:?Usage: $0 <sequence_name>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMED_SEQ="${REPO}/data/iMED_NVS/${SEQ}"
WS="${REPO}/workspaces/${SEQ}"
PYTHON="/media/SSD0/scanar/anaconda3/envs/mosca/bin/python"

echo "========================================"
echo "  iMED Step 1 — Prepare Workspace"
echo "  Sequence : ${SEQ}"
echo "  Source   : ${IMED_SEQ}"
echo "  Workspace: ${WS}"
echo "========================================"

cd "${REPO}"
"${PYTHON}" imed_prepare_workspace.py \
    --imed_seq "${IMED_SEQ}" \
    --ws "${WS}"

echo "Done. Workspace ready at: ${WS}"
