#!/usr/bin/env bash
# Compute metrics for Endo-4DGS baseline renders.
# Usage: bash scripts/imed_baseline_metrics.sh <sequence_name>

set -euo pipefail

SEQ="${1:?Usage: $0 <sequence_name>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/media/SSD0/scanar/anaconda3/envs/mosca/bin/python"
BASELINE_DIR="${REPO}/baseline/imed/${SEQ}"

# Ensure data symlink exists for baseline cfg_args source_path resolution
mkdir -p "${REPO}/data/imed"
if [ ! -e "${REPO}/data/imed/${SEQ}" ]; then
    ln -sf "${REPO}/data/iMED_NVS/${SEQ}" "${REPO}/data/imed/${SEQ}"
fi

echo "========================================"
echo "  Baseline Metrics — Endo-4DGS"
echo "  Sequence : ${SEQ}"
echo "  Logdir   : ${BASELINE_DIR}"
echo "========================================"

cd "${REPO}"
"${PYTHON}" metrics.py -m "${BASELINE_DIR}"

echo ""
echo "--- ours_15000 results ---"
"${PYTHON}" -c "
import json
d = json.load(open('${BASELINE_DIR}/results.json'))
r = d.get('ours_15000', {})
print(f'  PSNR : {r[\"PSNR\"]:.4f}')
print(f'  SSIM : {r[\"SSIM\"]:.4f}')
print(f'  LPIPS: {r[\"LPIPS\"]:.4f}')
"

echo ""
echo "Done. Results: ${BASELINE_DIR}/results.json"
