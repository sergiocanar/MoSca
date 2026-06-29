#!/usr/bin/env bash
# Collect and summarize metrics from all finished sequences.
# Usage: bash scripts/imed_collect_metrics.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS_ROOT="${REPO}/workspaces"

printf "%-35s %8s %8s %8s\n" "Sequence" "PSNR" "SSIM" "LPIPS"
printf "%-35s %8s %8s %8s\n" "-----------------------------------" "--------" "--------" "--------"

total_psnr=0
total_ssim=0
total_lpips=0
count=0

for results in $(find "${WS_ROOT}" -name "results.json" | sort); do
    seq=$(echo "${results}" | sed 's|.*/workspaces/||; s|/logs/.*||')
    psnr=$(python3 -c "import json; d=json.load(open('${results}')); print(list(d.values())[0]['PSNR'])" 2>/dev/null)
    ssim=$(python3 -c "import json; d=json.load(open('${results}')); print(list(d.values())[0]['SSIM'])" 2>/dev/null)
    lpips=$(python3 -c "import json; d=json.load(open('${results}')); print(list(d.values())[0]['LPIPS'])" 2>/dev/null)

    if [ -n "${psnr}" ]; then
        printf "%-35s %8.4f %8.4f %8.4f\n" "${seq}" "${psnr}" "${ssim}" "${lpips}"
        total_psnr=$(python3 -c "print(${total_psnr} + ${psnr})")
        total_ssim=$(python3 -c "print(${total_ssim} + ${ssim})")
        total_lpips=$(python3 -c "print(${total_lpips} + ${lpips})")
        count=$(( count + 1 ))
    fi
done

if [ "${count}" -gt 0 ]; then
    printf "%-35s %8s %8s %8s\n" "-----------------------------------" "--------" "--------" "--------"
    printf "%-35s %8.4f %8.4f %8.4f\n" "MEAN (${count}/${#} sequences)" \
        "$(python3 -c "print(${total_psnr}/${count})")" \
        "$(python3 -c "print(${total_ssim}/${count})")" \
        "$(python3 -c "print(${total_lpips}/${count})")"
fi
