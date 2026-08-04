#!/usr/bin/env python
"""
Stage 2 v2 -- SAM2 MULTI-REGION propagation with RE-ANCHORING and island gating.

Reads Stage-1 v2 regions.json (per-region continuous lifetimes + seeds). For EACH
region: build its span clip, prompt SAM2 at the region seeds RE-ANCHORED every K
frames (fixes single-anchor drift/tearing), propagate fwd+bwd within the clip, and
GATE the output to connected-components that contain a region seed (drops phantom
islands; keeps small legit regions -- NOT a min-area filter). Union all regions
per frame. No deformation guard (whole-organ motion is a correct dynamic region).
Writes viz + per-frame tissue; optional UNION(tool∪tissue) -> epi/error (--write_epi).
"""
import os, os.path as osp, glob, json, argparse, shutil
import numpy as np
import imageio.v2 as iio
import cv2
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--sam2_ckpt", default="/media/SSD0/gperezsantamaria/Challenge/uniandes_NVS/MoSca/weights/sam2/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--reanchor", type=int, default=6)       # re-prompt every K frames within a region span
    ap.add_argument("--write_epi", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz")
    TIS = osp.join(OUT, "sam2_tissue"); os.makedirs(TIS, exist_ok=True)
    tmp = osp.join(OUT, "_clip")

    d = json.load(open(osp.join(OUT, "regions.json")))
    regions = d["regions"]; negs = d["negatives"]
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"]); names = [str(x) for x in meta["frame_names_train"]]; T = len(names)
    print(f"{len(regions)} regions, T={T}")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=torch.device("cuda"))

    tissue = np.zeros((T, H, W), bool)
    reg_label = np.zeros((T, H, W), np.int16)     # for colored viz

    for R in regions:
        rid = R["id"]; s, e = R["span"]
        seeds_by_t = {int(t): v for t, v in R["seeds"].items()}
        present = sorted(seeds_by_t.keys())
        if not present:
            continue
        # re-anchor frames: every K within span that have seeds, plus the richest one
        anchors = [t for t in present if (t - s) % args.reanchor == 0]
        richest = max(present, key=lambda t: len(seeds_by_t[t]))
        if richest not in anchors:
            anchors.append(richest)
        anchors = sorted(set(anchors))
        all_seed_pts = [p for t in present for p in seeds_by_t[t]]   # for island gating

        # materialize clip
        if osp.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for t in range(s, e + 1):
            img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3]
            cv2.imwrite(osp.join(tmp, f"{t - s}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.reset_state(state)
            for t in anchors:
                P = np.array(seeds_by_t[t], np.float32)
                Ng = np.array(negs.get(str(t), []), np.float32)
                pts = P if len(Ng) == 0 else np.concatenate([P, Ng], 0)
                lbl = np.concatenate([np.ones(len(P), np.int32), np.zeros(len(Ng), np.int32)]) if len(Ng) else np.ones(len(P), np.int32)
                predictor.add_new_points_or_box(state, frame_idx=t - s, obj_id=1, points=pts, labels=lbl)
            local = {}
            for rev in (False, True):
                for fidx, obj_ids, logits in predictor.propagate_in_video(state, reverse=rev):
                    m = (logits[0, 0] > 0.0).cpu().numpy()
                    local[fidx] = local.get(fidx, np.zeros(m.shape, bool)) | m
        for li, m in local.items():
            t = s + li
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            m = _island_gate(m, all_seed_pts, W, H)          # drop phantom islands (no seed inside)
            tissue[t] |= m
            reg_label[t][m & (reg_label[t] == 0)] = rid + 1
        px = int(np.mean([tissue[t].sum() for t in range(s, e + 1)]))
        print(f"  region {rid:2d} {s:3d}-{e:3d}  anchors={anchors}  mask px/frame≈{px}")

    if osp.isdir(tmp):
        shutil.rmtree(tmp)

    # save + viz
    tool = np.stack([_toolread(osp.join(ws, "train_masks", f"{n}.png")) for n in names])
    palette = _palette()
    wr = iio.get_writer(osp.join(VIZ, "sam2_masks.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    for t in range(T):
        np.save(osp.join(TIS, f"{names[t]}.npy"), tissue[t])
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3].astype(np.float32)
        lab = reg_label[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            col = np.array(palette[(rid - 1) % len(palette)], np.float32)
            sel = lab == rid
            img[sel] = img[sel] * 0.5 + col * 0.5
        te = tool[t] ^ cv2.erode(tool[t].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[te] = [255, 60, 60]
        act = sorted(int(r - 1) for r in np.unique(lab) if r > 0)
        wr.append_data(_bar(np.clip(img, 0, 255).astype(np.uint8),
                            f"{names[t]}  regions={act}  dyn px={int(tissue[t].sum())}  (colored=SAM2 tissue, red=tool)"))
    wr.close()
    cov = tissue.reshape(T, -1).sum(1)
    print(f"\ntissue: frames_with_mask={int((cov>0).sum())}/{T}  mean px(active)={int(cov[cov>0].mean()) if (cov>0).any() else 0}")
    print(f"viz -> {VIZ}/sam2_masks.mp4")

    if args.write_epi:
        epidir = osp.join(ws, "epi", "error"); bak = osp.join(ws, "epi", "error_tool_backup")
        if not osp.isdir(bak):
            shutil.copytree(epidir, bak); print(f"backed up tool epi -> {bak}")
        for t in range(T):
            np.save(osp.join(epidir, f"{names[t]}.png.npy"), (tool[t] | tissue[t]).astype(np.float32))
        print(f"WROTE union(tool∪tissue) -> {epidir}")
    else:
        print("epi/error NOT modified (pass --write_epi 1 to commit).")


def _island_gate(m, seed_pts, W, H):
    if not m.any():
        return m
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    keep_labels = set()
    for x, y in seed_pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and lab[yi, xi] > 0:
            keep_labels.add(int(lab[yi, xi]))
    if not keep_labels:                                   # SAM2 drifted off all seeds -> keep largest CC
        counts = np.bincount(lab.ravel()); counts[0] = 0
        keep_labels = {int(counts.argmax())}
    out = np.zeros_like(m)
    for l in keep_labels:
        out |= (lab == l)
    return out


def _toolread(f):
    m = iio.imread(f); return (m[..., 0] if m.ndim == 3 else m) > 127


def _palette():
    return [(66, 135, 245), (245, 130, 49), (60, 200, 120), (200, 60, 200),
            (240, 220, 40), (40, 220, 220), (200, 90, 90), (120, 120, 250),
            (250, 160, 200), (150, 250, 120)]


def _bar(canvas, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        fnt = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 36], fill=(0, 0, 0)); d.text((8, 5), text, fill=(255, 255, 255), font=fnt)
    return np.array(im)


if __name__ == "__main__":
    main()
