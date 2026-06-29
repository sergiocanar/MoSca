#!/usr/bin/env bash
# Create a 3-way comparison video: GT | Endo-4DGS | MoSca
# Usage: bash scripts/imed_compare_video.sh <sequence_name> [mosca_logdir]

set -euo pipefail

FFMPEG="/media/SSD0/scanar/anaconda3/envs/mosca/bin/ffmpeg"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="${1:?Usage: $0 <sequence_name> [mosca_logdir]}"
WS="${REPO}/workspaces/${SEQ}"

# MoSca logdir
if [ -n "${2:-}" ]; then
    MOSCA_LOGDIR="${2}"
else
    MOSCA_LOGDIR=$(ls -td "${WS}/logs"/imed_fit_*/ 2>/dev/null | head -1)
fi

BASELINE_DIR="${REPO}/baseline/imed/${SEQ}/test/ours_15000"
MOSCA_RENDERS="${MOSCA_LOGDIR}/test/mosca/renders"
GT_DIR="${BASELINE_DIR}/gt"
BASELINE_RENDERS="${BASELINE_DIR}/renders"
OUT="${MOSCA_LOGDIR}/test/mosca/compare_3way.webm"

echo "========================================"
echo "  3-Way Comparison Video"
echo "  Sequence : ${SEQ}"
echo "  GT       : ${GT_DIR}"
echo "  Endo-4DGS: ${BASELINE_RENDERS}"
echo "  MoSca    : ${MOSCA_RENDERS}"
echo "  Output   : ${OUT}"
echo "========================================"

"${FFMPEG}" -y \
    -framerate 10 -pattern_type glob -i "${GT_DIR}/*.png" \
    -framerate 10 -pattern_type glob -i "${BASELINE_RENDERS}/*.png" \
    -framerate 10 -pattern_type glob -i "${MOSCA_RENDERS}/*.png" \
    -filter_complex "
        [0:v]scale=640:-1,drawtext=text='GT':fontcolor=white:fontsize=24:x=10:y=10[gt];
        [1:v]scale=640:-1,drawtext=text='Endo-4DGS':fontcolor=white:fontsize=24:x=10:y=10[base];
        [2:v]scale=640:-1,drawtext=text='MoSca':fontcolor=white:fontsize=24:x=10:y=10[mosca];
        [gt][base][mosca]hstack=inputs=3[out]
    " -map "[out]" \
    -c:v libvpx-vp9 -pix_fmt yuv420p "${OUT}" -loglevel error

echo "Done. Saved to: ${OUT}"
