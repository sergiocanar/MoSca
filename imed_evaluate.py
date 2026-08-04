"""Evaluate MoSca on iMED Task 2 (Novel View Synthesis from Endoscope 1).

This script bypasses the standard mosca_evaluate.test_main() because:
  - The training camera is static (all-identity) → Sim(3) alignment in render_test()
    is degenerate.
  - The test camera pose is exactly known from pose.txt (no alignment needed).

Usage:
    python imed_evaluate.py \\
        --ws workspaces/session_004_scene_2_tool_1 \\
        --logdir workspaces/session_004_scene_2_tool_1/logs/<timestamp>

Output structure (consumed by metrics.py):
    <logdir>/test/mosca/
        renders/frame_XXXX.png    ← rendered Endo1L views
        gt/frame_XXXX.png         ← Endo1L ground-truth images
        masks/frame_XXXX.png      ← Endo1L valid-region masks, official Endo-4DGS
                                     convention: mask = 1 - raw_tool/255, i.e.
                                     nonzero = background/valid, zero = tool.
                                     (test_masks/ in the workspace stores the raw
                                     tool mask instead, nonzero = tool; it is
                                     inverted here when copied.)
        overlap_mask.png          ← auto-computed by metrics.py from pose/depth
"""

import argparse
import json
import os
import os.path as osp
import shutil

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image
from tqdm import tqdm

from lib_moca.camera import MonocularCameras
from lib_mosca.imed_render_utils import build_K_tensor, load_trained_models, render_frame
from metrics import evaluate as metrics_evaluate
from metrics import _build_global_imed_overlap_mask


def write_test_render_video(logdir, fps=12):
    """Write a clean, single-panel Endo1L novel-view video: 100% model renders,
    one frame per test image, no GT / debug panels. Output: <logdir>/test_render.mp4.
    Encoded from the already-saved renders/, so it works with or without --skip_render."""
    renders_dir = osp.join(logdir, "test", "mosca", "renders")
    if not osp.isdir(renders_dir):
        print(f"[test_render] no renders dir at {renders_dir}, skip")
        return
    frames = sorted(f for f in os.listdir(renders_dir) if f.lower().endswith(".png"))
    if not frames:
        print("[test_render] no render frames found, skip")
        return
    out = osp.join(logdir, "test_render.mp4")
    with imageio.get_writer(out, fps=fps, macro_block_size=None, quality=8) as w:
        for f in frames:
            w.append_data(imageio.imread(osp.join(renders_dir, f))[..., :3])
    print(f"[test_render] wrote clean render video: {out} ({len(frames)} frames @ {fps} fps)")


@torch.no_grad()
def render_imed_nvs(
    logdir,
    ws,
    device=torch.device("cuda"),
):
    """Render Endoscope 1-L views for all training time steps.

    The world frame is Endoscope 2-L (training camera at origin).
    The test camera (Endo1L) has a constant known pose c2w_test.
    We render scene state d_model(t) + s_model() for each t.
    """
    # --- Load metadata --------------------------------------------------------
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    T = int(meta["T"])
    K1L = meta["K1L"].astype(np.float32)      # Endo1L intrinsics
    c2w_test = meta["c2w_test"].astype(np.float32)   # Endo1L in Endo2L world (mm)
    frame_names_test = list(meta["frame_names_test"])
    H1 = int(meta["H1"])
    W1 = int(meta["W1"])

    # World is in mm (dep_median=-1), c2w_test already in mm → no scale needed.
    T_cw_test = np.linalg.inv(c2w_test)
    T_cw_test_t = torch.from_numpy(T_cw_test).float().to(device)
    K1L_t = build_K_tensor(K1L, device)

    # --- Load trained models --------------------------------------------------
    s_model, d_model = load_trained_models(logdir, device)

    # --- Output dirs ----------------------------------------------------------
    method_dir = osp.join(logdir, "test", "mosca")
    renders_dir = osp.join(method_dir, "renders")
    gt_dir      = osp.join(method_dir, "gt")
    masks_dir   = osp.join(method_dir, "masks")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Copy GT images + masks from workspace
    test_img_dir  = osp.join(ws, "test_images")
    test_mask_dir = osp.join(ws, "test_masks")
    for name in frame_names_test:
        src_img  = osp.join(test_img_dir,  f"{name}.png")
        src_mask = osp.join(test_mask_dir, f"{name}.png")
        dst_img  = osp.join(gt_dir,        f"{name}.png")
        dst_mask = osp.join(masks_dir,     f"{name}.png")
        if osp.exists(src_img) and not osp.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        if osp.exists(src_mask):
            # test_masks/ holds the raw tool mask (nonzero = tool). The
            # official Endo-4DGS metrics.py expects the opposite convention
            # (mask = 1 - raw/255, nonzero = valid/background) and uses the
            # file's value directly as the eval region -- invert here so
            # eval scores background, not the tool. Always rewritten (cheap)
            # so stale un-inverted masks from before this fix get corrected.
            raw = np.array(Image.open(src_mask))
            if raw.ndim == 3:
                raw = raw[..., 0]
            valid = (255 - raw).astype(np.uint8)
            Image.fromarray(valid).save(dst_mask)

    # --- Render each frame ----------------------------------------------------
    s_gs5 = s_model()
    for t in tqdm(range(T), desc="Rendering Endo1L NVS"):
        rgb = render_frame(s_gs5, d_model, t, K1L_t, T_cw_test_t, H1, W1)
        name = frame_names_test[t]
        imageio.imwrite(osp.join(renders_dir, f"{name}.png"), rgb)

    print(f"Renders saved to: {renders_dir}")
    return method_dir


def _psnr(pred, gt, mask):
    """pred/gt: [3,H,W] float32 [0,1], mask: [1,H,W] float32."""
    mask3 = mask.expand(3, -1, -1)
    mse = ((pred - gt) ** 2 * mask3).sum() / mask3.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse + 1e-8)


def _ssim(pred, gt, mask, window_size=11):
    """Standard windowed SSIM averaged over valid (mask=1) pixels."""
    p = pred.unsqueeze(0)   # [1,3,H,W]
    g = gt.unsqueeze(0)
    ch = 3
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    gauss /= gauss.sum()
    kern = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0).expand(ch, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(p, kern, padding=pad, groups=ch)
    mu2 = F.conv2d(g, kern, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(p * p, kern, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(g * g, kern, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(p * g, kern, padding=pad, groups=ch) - mu1mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2) + 1e-8)
    m3 = mask.unsqueeze(0).expand(1, ch, -1, -1)
    return (ssim_map * m3).sum() / m3.sum().clamp_min(1.0)


_lpips_model = None

def _lpips(pred, gt, mask):
    global _lpips_model
    import lpips as lpips_lib
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net="vgg").to(pred.device)
    m = mask.expand(3, -1, -1)
    p = (pred * m).unsqueeze(0) * 2 - 1
    g = (gt  * m).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return _lpips_model(p, g).item()


@torch.no_grad()
def compute_corrected_metrics(logdir, ws, imed_seq):
    """Compute PSNR/SSIM/LPIPS with correct mask: overlap AND NOT tool."""
    method_dir = osp.join(logdir, "test", "mosca")
    renders_dir = osp.join(method_dir, "renders")
    gt_dir      = osp.join(method_dir, "gt")
    masks_dir   = osp.join(method_dir, "masks")

    fnames = sorted(os.listdir(renders_dir))
    if not fnames:
        print("No renders found, skipping corrected metrics.")
        return

    # Build overlap mask from first render's resolution
    sample = np.array(Image.open(osp.join(renders_dir, fnames[0])))
    out_h, out_w = sample.shape[:2]
    global_mask_np = _build_global_imed_overlap_mask(imed_seq, out_h, out_w)
    global_mask = torch.from_numpy(global_mask_np).float().cuda()  # [H,W]

    psnrs, ssims, lpipss = [], [], []
    for name in tqdm(fnames, desc="Corrected metrics"):
        pred = tvf.to_tensor(np.array(Image.open(osp.join(renders_dir, name)))).cuda()  # [3,H,W]
        gt   = tvf.to_tensor(np.array(Image.open(osp.join(gt_dir,      name)))).cuda()
        # masks_dir already stores the valid-region mask (nonzero = background/
        # non-tool), inverted from raw tool masks in render_imed_nvs().
        valid_raw = np.array(Image.open(osp.join(masks_dir, name)))
        valid = tvf.to_tensor(valid_raw).cuda()
        if valid.shape[0] > 1:
            valid = valid[:1]
        valid = (valid > 0.5).float().squeeze(0)  # [H,W], 1=background/non-tool

        mask = global_mask * valid  # [H,W]: overlap AND NOT tool
        mask = mask.unsqueeze(0)    # [1,H,W]

        psnrs.append(_psnr(pred, gt, mask).item())
        ssims.append(_ssim(pred, gt, mask).item())
        lpipss.append(_lpips(pred, gt, mask))

    results = {
        "PSNR":  float(np.mean(psnrs)),
        "SSIM":  float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)),
        "note":  "overlap_mask AND NOT tool_mask; standalone recomputation, "
                 "independent of metrics.py, kept as a cross-check"
    }
    out_path = osp.join(logdir, "results_corrected.json")
    with open(out_path, "w") as f:
        json.dump({"mosca": results}, f, indent=2)
    print(f"\n--- Standalone cross-check metrics (overlap AND NOT tool) ---")
    print(f"  PSNR : {results['PSNR']:.4f}")
    print(f"  SSIM : {results['SSIM']:.4f}")
    print(f"  LPIPS: {results['LPIPS']:.4f}")
    print(f"  Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser("iMED NVS Evaluation")
    parser.add_argument("--ws", required=True, help="MoSca workspace dir")
    parser.add_argument("--logdir", required=True, help="Trained model logdir")
    parser.add_argument(
        "--skip_render", action="store_true",
        help="Skip rendering if renders already exist"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ws = osp.abspath(args.ws)
    logdir = osp.abspath(args.logdir)

    # Write cfg_args so metrics.py can detect this is an iMED scene and
    # compute the overlap mask automatically.
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    imed_seq = str(meta["imed_seq"])
    cfg_args_path = osp.join(logdir, "cfg_args")
    if not osp.exists(cfg_args_path):
        with open(cfg_args_path, "w") as f:
            f.write(f"source_path='{imed_seq}'")

    if not args.skip_render:
        render_imed_nvs(logdir=logdir, ws=ws, device=device)

    # Clean single-panel Endo1L novel-view video (100% model renders, no debug panels).
    write_test_render_video(logdir)

    # Organizers'-equivalent metrics (overlap AND NOT tool; matches the official
    # Endo-4DGS/metrics.py convention now that masks/ is written pre-inverted).
    print("\nComputing metrics.py results (overlap mask AND NOT tool) ...")
    metrics_evaluate([logdir])

    # Independent recomputation as a cross-check -- should match results.json.
    print("\nComputing standalone cross-check metrics ...")
    compute_corrected_metrics(logdir=logdir, ws=ws, imed_seq=imed_seq)


if __name__ == "__main__":
    main()
