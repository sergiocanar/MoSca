"""Inference-only iMED NVS renderer for the Docker submission entrypoint.

Unlike imed_evaluate.py (local dev: copies GT, computes PSNR/SSIM/LPIPS), this
script has no ground truth available -- hidden test sequences do not ship
endoscope1/ at all. It only renders the Endo1L view for every training time
step and writes it under the sequential filenames the challenge evaluator
expects (see Endo-4DGS/imednvs_submission/scripts/check_outputs.py).

Usage:
    python imed_submission_render.py \\
        --ws <workspace>                       (built with imed_prepare_workspace.py --inference) \\
        --logdir <workspace>/logs/imed_fit_.../ \\
        --output /output/<sequence_name>

Output:
    <output>/renders/00000.png, 00001.png, ...  (one per training/source frame)
"""

import argparse
import os
import os.path as osp

import imageio
import numpy as np
import torch
from tqdm import tqdm

from lib_mosca.imed_render_utils import build_K_tensor, load_trained_models, render_frame


@torch.no_grad()
def render_for_submission(ws, logdir, output, device=torch.device("cuda")):
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    T = int(meta["T"])
    K1L = meta["K1L"].astype(np.float32)             # Endo1L intrinsics
    c2w_test = meta["c2w_test"].astype(np.float32)   # Endo1L in Endo2L world (mm)
    H1 = int(meta["H1"])
    W1 = int(meta["W1"])

    # World is in mm (dep_median=-1), c2w_test already in mm -> no scale needed.
    T_cw_test = np.linalg.inv(c2w_test)
    T_cw_test_t = torch.from_numpy(T_cw_test).float().to(device)
    K1L_t = build_K_tensor(K1L, device)

    s_model, d_model = load_trained_models(logdir, device)

    renders_dir = osp.join(output, "renders")
    os.makedirs(renders_dir, exist_ok=True)

    s_gs5 = s_model()
    for t in tqdm(range(T), desc="Rendering Endo1L NVS (submission)"):
        rgb = render_frame(s_gs5, d_model, t, K1L_t, T_cw_test_t, H1, W1)
        imageio.imwrite(osp.join(renders_dir, f"{t:05d}.png"), rgb)

    print(f"Renders saved to: {renders_dir}")
    return renders_dir


def main():
    parser = argparse.ArgumentParser("iMED NVS submission renderer")
    parser.add_argument("--ws", required=True, help="MoSca workspace dir (built with --inference)")
    parser.add_argument("--logdir", required=True, help="Trained model logdir")
    parser.add_argument(
        "--output", required=True,
        help="Per-sequence output dir; renders are written to <output>/renders/",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    render_for_submission(
        ws=osp.abspath(args.ws),
        logdir=osp.abspath(args.logdir),
        output=osp.abspath(args.output),
        device=device,
    )


if __name__ == "__main__":
    main()
