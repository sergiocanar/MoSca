"""Convert one iMED sequence directory into a MoSca workspace.

Usage:
    python imed_prepare_workspace.py \
        --imed_seq data/iMED_NVS/session_004_scene_2_tool_1 \
        --ws workspaces/session_004_scene_2_tool_1

What it does:
  1. endoscope2/L/*.png   → ws/images/frame_XXXX.png   (sequential rename)
  2. endoscope2/depthL/*.npy → ws/sensor_depth/frame_XXXX.npz  (key "dep")
  3. endoscope2/toolL/*.png  → ws/epi/error/frame_XXXX.png.npy (pseudo-epi:
                               tool pixels = 1.0, background = 0.0)
  4. endoscope1/L/*.png   → ws/test_images/frame_XXXX.png
  5. endoscope1/toolL/*.png  → ws/test_masks/frame_XXXX.png
  6. K.txt + pose.txt     → ws/imed_meta.npz
"""

import argparse
import os
import os.path as osp
import shutil
from glob import glob
from pathlib import Path

import cv2
import imageio
import numpy as np


def sorted_frames(directory, ext):
    fns = sorted(f for f in os.listdir(directory) if f.endswith(ext))
    return [osp.join(directory, f) for f in fns]


def make_dir(*parts):
    d = osp.join(*parts)
    os.makedirs(d, exist_ok=True)
    return d


def prepare_workspace(imed_seq, ws):
    imed_seq = osp.abspath(imed_seq)
    ws = osp.abspath(ws)

    print(f"iMED sequence : {imed_seq}")
    print(f"Workspace     : {ws}")

    # --- 1. Training images: endoscope2/L/*.png → ws/images/frame_XXXX.png ------
    src_imgs = sorted_frames(osp.join(imed_seq, "endoscope2", "L"), ".png")
    assert len(src_imgs) > 0, "No PNG files in endoscope2/L"
    img_dir = make_dir(ws, "images")
    frame_names_train = []
    for i, src in enumerate(src_imgs):
        name = f"frame_{i:04d}"
        dst = osp.join(img_dir, f"{name}.png")
        if not osp.exists(dst):
            shutil.copy2(src, dst)
        frame_names_train.append(name)
    T = len(frame_names_train)
    print(f"  Training frames : {T}")

    # Read H, W from a sample image
    sample_img = imageio.imread(src_imgs[0])
    H, W = sample_img.shape[:2]
    print(f"  Training image size : H={H}, W={W}")

    # --- 2. Depth: endoscope2/depthL/*.npy → ws/sensor_depth/frame_XXXX.npz -----
    src_deps = sorted_frames(osp.join(imed_seq, "endoscope2", "depthL"), ".npy")
    assert len(src_deps) == T, (
        f"Depth count ({len(src_deps)}) != image count ({T})"
    )
    dep_dir = make_dir(ws, "sensor_depth")
    all_depths_sample = []
    for i, (src, name) in enumerate(zip(src_deps, frame_names_train)):
        dst = osp.join(dep_dir, f"{name}.npz")
        dep = np.load(src).astype(np.float32)
        # Upsample depth to match image resolution if they differ (e.g. 512×640 → 1024×1280)
        if dep.shape[0] != H or dep.shape[1] != W:
            dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_LINEAR)
        if not osp.exists(dst):
            np.savez_compressed(dst, dep=dep)
        if i % 20 == 0:  # sample every 20 frames for scale estimation
            valid = dep[dep > 1e-3]
            if len(valid) > 0:
                all_depths_sample.append(valid)
    print(f"  Depth files created : {T}")

    # --- 3. Pseudo-epi from tool masks: endoscope2/toolL/*.png → ws/epi/error/ ---
    src_tools = sorted_frames(osp.join(imed_seq, "endoscope2", "toolL"), ".png")
    assert len(src_tools) == T, (
        f"Tool mask count ({len(src_tools)}) != image count ({T})"
    )
    epi_err_dir = make_dir(ws, "epi", "error")
    train_mask_dir = make_dir(ws, "train_masks")
    for src, name in zip(src_tools, frame_names_train):
        dst = osp.join(epi_err_dir, f"{name}.png.npy")
        if not osp.exists(dst):
            mask = imageio.imread(src)
            if mask.ndim == 3:
                mask = mask[..., 0]
            epi = (mask > 127).astype(np.float32)
            np.save(dst, epi)
        dst_mask = osp.join(train_mask_dir, f"{name}.png")
        if not osp.exists(dst_mask):
            mask = imageio.imread(src)
            if mask.ndim == 3:
                mask = mask[..., 0]
            imageio.imwrite(dst_mask, mask)
    print(f"  Pseudo-epi files created : {T}")
    print(f"  Train mask files created : {T}")

    # --- 4. Test images: endoscope1/L/*.png → ws/test_images/frame_XXXX.png -----
    src_test_imgs = sorted_frames(osp.join(imed_seq, "endoscope1", "L"), ".png")
    assert len(src_test_imgs) == T, (
        f"Test image count ({len(src_test_imgs)}) != training count ({T})"
    )
    test_img_dir = make_dir(ws, "test_images")
    frame_names_test = []
    for i, src in enumerate(src_test_imgs):
        name = f"frame_{i:04d}"
        dst = osp.join(test_img_dir, f"{name}.png")
        if not osp.exists(dst):
            shutil.copy2(src, dst)
        frame_names_test.append(name)
    print(f"  Test frames : {T}")

    sample_test = imageio.imread(src_test_imgs[0])
    H1, W1 = sample_test.shape[:2]
    print(f"  Test image size : H1={H1}, W1={W1}")

    # --- 5. Test masks: endoscope1/toolL/*.png → ws/test_masks/ -----------------
    src_test_masks = sorted_frames(osp.join(imed_seq, "endoscope1", "toolL"), ".png")
    assert len(src_test_masks) == T
    test_mask_dir = make_dir(ws, "test_masks")
    for i, src in enumerate(src_test_masks):
        dst = osp.join(test_mask_dir, f"frame_{i:04d}.png")
        if not osp.exists(dst):
            shutil.copy2(src, dst)

    # --- 6. Metadata: parse K.txt + pose.txt, compute depth scale ----------------
    from data_utils.imed_helpers import parse_imed_intrinsics, parse_imed_poses

    k_matrices = parse_imed_intrinsics(osp.join(imed_seq, "K.txt"))
    K2L = k_matrices["K2_L"]
    K1L = k_matrices["K1_L"]

    c2w_by_cam = parse_imed_poses(osp.join(imed_seq, "pose.txt"))
    # cam_id=0 → Endo2L (training, identity), cam_id=1 → Endo1L (test) in mm
    c2w_train = c2w_by_cam[0]  # should be identity
    c2w_test  = c2w_by_cam[1]  # Endo1L in Endo2L world (mm)

    # Depth scale: 1 / median_mm_depth  (maps mm world → normalized ≈1 world)
    if all_depths_sample:
        median_depth_mm = float(np.median(np.concatenate(all_depths_sample)))
    else:
        # Fallback: load all depths
        depths_all = []
        for src in src_deps:
            d = np.load(src).astype(np.float32)
            depths_all.append(d[d > 1e-3].flatten())
        median_depth_mm = float(np.median(np.concatenate(depths_all)))
    depth_scale_mm_to_norm = 1.0 / median_depth_mm
    print(f"  Median depth : {median_depth_mm:.2f} mm  (scale={depth_scale_mm_to_norm:.5f})")

    meta_path = osp.join(ws, "imed_meta.npz")
    np.savez(
        meta_path,
        T=T,
        H=H,
        W=W,
        H1=H1,
        W1=W1,
        K2L=K2L,
        K1L=K1L,
        c2w_train=c2w_train,
        c2w_test=c2w_test,
        depth_scale_mm_to_norm=depth_scale_mm_to_norm,
        frame_names_train=frame_names_train,
        frame_names_test=frame_names_test,
        imed_seq=imed_seq,
    )
    print(f"  Metadata saved : {meta_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iMED workspace preparation")
    parser.add_argument("--imed_seq", required=True, help="Path to iMED sequence dir")
    parser.add_argument("--ws", required=True, help="Target MoSca workspace dir")
    args = parser.parse_args()
    prepare_workspace(args.imed_seq, args.ws)
