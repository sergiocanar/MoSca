#!/usr/bin/env bash
# Step 5: Create side-by-side comparison video (render vs GT).
# Usage:
#   Single sequence: bash scripts/imed_step5_video.sh <sequence_name> [logdir]
#   All sequences:   bash scripts/imed_step5_video.sh --all

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FFMPEG="/media/SSD0/scanar/anaconda3/envs/mosca/bin/ffmpeg"

make_video() {
    local SEQ="$1"
    local WS="${REPO}/workspaces/${SEQ}"
    local LOGDIR="${2:-$(ls -td "${WS}/logs"/imed_fit_*/ 2>/dev/null | head -1)}"

    if [ -z "${LOGDIR}" ]; then
        echo "SKIP ${SEQ}: no logdir found."
        return
    fi

    local RENDERS="${LOGDIR}/test/mosca/renders"
    local GT="${LOGDIR}/test/mosca/gt"
    local OUT="${LOGDIR}/test/mosca/compare.webm"

    if [ ! -d "${RENDERS}" ] || [ -z "$(ls "${RENDERS}"/*.png 2>/dev/null)" ]; then
        echo "SKIP ${SEQ}: no renders found."
        return
    fi

    echo "Making video: ${SEQ}"
    "${FFMPEG}" -y -framerate 10 \
        -pattern_type glob -i "${RENDERS}/*.png" \
        -framerate 10 \
        -pattern_type glob -i "${GT}/*.png" \
        -filter_complex hstack \
        -c:v libvpx-vp9 -pix_fmt yuv420p "${OUT}" -loglevel error
    echo "  Saved: ${OUT}"
}

if [ "${1:-}" = "--all" ]; then
    for ws in $(find "${REPO}/workspaces" -maxdepth 1 -mindepth 1 -type d | sort); do
        SEQ=$(basename "${ws}")
        make_video "${SEQ}"
    done
else
    SEQ="${1:?Usage: $0 <sequence_name> [logdir] | --all}"
    make_video "${SEQ}" "${2:-}"
fi
