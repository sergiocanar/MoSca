#!/usr/bin/env python3
"""iMED-NVS MoSca submission entrypoint.

Mirrors the shape of Endo-4DGS/imed_nvs_baseline.py: discover iMED sequences
under a read-only input root, run the MoSca pipeline per sequence, and write
rendered Endo1L views under the challenge's output contract.

Unlike the Endo-4DGS baseline, MoSca's own workspace-prep/precompute/
reconstruct scripts never write into the input sequence directory, so no
writable-symlink-view trick is needed -- /input can stay strictly read-only.

Hidden test sequences do not ship endoscope1/ at all (see
iMED_NVS_Submission_Guidelines.pdf, pitfall #5), so sequence discovery and the
whole pipeline below never require or read anything under endoscope1/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def sequence_name(sequence: Path) -> str:
    return sequence.resolve().name


def is_sequence_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "pose.txt").is_file()
        and (path / "K.txt").is_file()
        and (path / "endoscope2").is_dir()
    )


def discover_sequences(data_root: Path) -> list[Path]:
    if is_sequence_dir(data_root):
        return [data_root]
    return sorted(path for path in data_root.rglob("*") if is_sequence_dir(path))


def find_latest_logdir(ws: Path) -> Path:
    candidates = sorted((ws / "logs").glob("imed_fit_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No imed_fit_* logdir found under {ws / 'logs'}")
    return candidates[-1]


def run_sequence(sequence: Path, output: Path, repo: Path) -> Path:
    """Train + reconstruct + render one iMED sequence.

    `output` is the sequence's own output directory; renders land under
    `output/renders/`.
    """
    sequence = sequence.resolve()
    output = output.resolve()
    repo = repo.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ws = output / "_workspace"

    run(
        [
            sys.executable, "imed_prepare_workspace.py",
            "--imed_seq", str(sequence),
            "--ws", str(ws),
            "--inference",
        ],
        cwd=repo,
    )
    run(
        [
            sys.executable, "mosca_precompute.py",
            "--cfg", "profile/imed/imed_prep.yaml",
            "--ws", str(ws),
        ],
        cwd=repo,
    )
    run(
        [
            sys.executable, "mosca_reconstruct.py",
            "--cfg", "profile/imed/imed_fit.yaml",
            "--ws", str(ws),
        ],
        cwd=repo,
    )
    logdir = find_latest_logdir(ws)
    run(
        [
            sys.executable, "imed_submission_render.py",
            "--ws", str(ws),
            "--logdir", str(logdir),
            "--output", str(output),
        ],
        cwd=repo,
    )

    return output / "renders"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iMED-NVS MoSca submission")
    parser.add_argument("--repo", default=os.environ.get("MOSCA_REPO", "/workspace/MoSca"))
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("run-sequence", help="Run one iMED-NVS sequence")
    one.add_argument("--sequence", required=True)
    one.add_argument("--output", required=True)

    many = sub.add_parser("run-dataset", help="Run all detected iMED-NVS sequences under a data root")
    many.add_argument("--data-root", required=True)
    many.add_argument("--output-root", required=True)
    many.add_argument("--max-sequences", type=int, default=None)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo)

    if args.command == "run-sequence":
        run_sequence(Path(args.sequence), Path(args.output), repo=repo)
        return 0

    sequences = discover_sequences(Path(args.data_root))
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequence folders found under {args.data_root}")

    for sequence in sequences:
        output = Path(args.output_root) / sequence_name(sequence)
        run_sequence(sequence, output, repo=repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
