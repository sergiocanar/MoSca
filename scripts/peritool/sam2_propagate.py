#!/usr/bin/env python
"""
Stage 2 v3 -- SINGLE BINARY object, tool-centric, continuous.

Reads Stage-1 v3 prompts.json (per-frame near-tool moving positives + tool/static
negatives + active spans). Tiles each interaction span into overlapping K-frame
windows; for each window it prompts ONE SAM2 object at the window anchor with the
near-tool positives (+ tool/static negatives) and propagates within the window
(continuity, fresh anchoring per window -> tracks the CURRENT near-tool tissue, no
long-span accumulation flood). Gates each frame by: keep the connected component
containing a positive (drops phantom islands) AND cap to within Dcap of the tool
(no far flooding). Unions windows -> ONE binary tissue mask.

Tool is a NEGATIVE here (SAM2 never masks it); it is added back at write time:
  epi/error = tool_mask (from dataset train_masks/) UNION tissue_mask.
Writes viz + per-frame tissue; --write_epi 1 commits the union (backs up first).
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
    ap.add_argument("--win_len", type=int, default=12)      # window length (frames)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--gap_bridge", type=int, default=3)
    ap.add_argument("--cap_px", type=int, default=220)      # keep mask within this dist of tool (0=off)
    ap.add_argument("--write_epi", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz")
    TIS = osp.join(OUT, "sam2_tissue"); os.makedirs(TIS, exist_ok=True)
    tmp = osp.join(OUT, "_clip")
    d = json.load(open(osp.join(OUT, "prompts.json")))
    P = d["frames"]; active = d["active"]
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"]); names = [str(x) for x in meta["frame_names_train"]]; T = len(names)
    tool = np.stack([_toolread(osp.join(ws, "train_masks", f"{n}.png")) for n in names])
    spans = _spans(active, T, args.gap_bridge)
    print(f"active {len(active)}/{T}  spans={spans}")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=torch.device("cuda"))
    tissue = np.zeros((T, H, W), bool)

    step = max(1, args.win_len - args.overlap)
    windows = []
    for s, e in spans:
        w0 = s
        while w0 <= e:
            windows.append((w0, min(w0 + args.win_len - 1, e)))
            if w0 + args.win_len - 1 >= e:
                break
            w0 += step
    print(f"{len(windows)} windows")

    for (w0, w1) in windows:
        # anchor = frame with most positives in window
        cand = [(t, len(P[str(t)]["pos"])) for t in range(w0, w1 + 1) if P[str(t)]["pos"]]
        if not cand:
            continue
        anchor = max(cand, key=lambda z: z[1])[0]
        win_pos = [p for t in range(w0, w1 + 1) for p in P[str(t)]["pos"]]   # for island gate
        if osp.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for t in range(w0, w1 + 1):
            img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3]
            cv2.imwrite(osp.join(tmp, f"{t - w0}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        pos = np.array(P[str(anchor)]["pos"], np.float32)
        neg = np.array(P[str(anchor)]["neg"], np.float32)
        pts = pos if len(neg) == 0 else np.concatenate([pos, neg], 0)
        lbl = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)]) if len(neg) else np.ones(len(pos), np.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.reset_state(state)
            predictor.add_new_points_or_box(state, frame_idx=anchor - w0, obj_id=1, points=pts, labels=lbl)
            local = {}
            for rev in (False, True):
                for fidx, oid, logits in predictor.propagate_in_video(state, reverse=rev):
                    m = (logits[0, 0] > 0.0).cpu().numpy()
                    local[fidx] = local.get(fidx, np.zeros(m.shape, bool)) | m
        for li, m in local.items():
            t = w0 + li
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            m = _island_gate(m, win_pos, W, H)
            if args.cap_px > 0:
                cap = cv2.dilate(tool[t].astype(np.uint8), np.ones((args.cap_px, args.cap_px), np.uint8)) > 0
                m = m & cap
            tissue[t] |= m
    if osp.isdir(tmp):
        shutil.rmtree(tmp)

    # viz + save
    wr = iio.get_writer(osp.join(VIZ, "sam2_masks.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    for t in range(T):
        np.save(osp.join(TIS, f"{names[t]}.npy"), tissue[t])
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3].astype(np.float32)
        img[tissue[t]] = img[tissue[t]] * 0.5 + np.array([40, 240, 40]) * 0.5
        te = tool[t] ^ cv2.erode(tool[t].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[te] = [255, 60, 60]
        tag = "dyn" if tissue[t].any() else "-"
        wr.append_data(_bar(np.clip(img, 0, 255).astype(np.uint8),
                            f"{names[t]}  [{tag}]  green=SAM2 tissue  red=tool  px={int(tissue[t].sum())}"))
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


def _spans(active, T, gap):
    if not active:
        return []
    active = sorted(active); out = []; s = active[0]; p = active[0]
    for t in active[1:]:
        if t - p <= gap + 1:
            p = t
        else:
            out.append((s, p)); s = t; p = t
    out.append((s, p)); return out


def _island_gate(m, seed_pts, W, H):
    if not m.any():
        return m
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    keep = set()
    for x, y in seed_pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and lab[yi, xi] > 0:
            keep.add(int(lab[yi, xi]))
    if not keep:
        c = np.bincount(lab.ravel()); c[0] = 0; keep = {int(c.argmax())}
    out = np.zeros_like(m)
    for l in keep:
        out |= (lab == l)
    return out


def _toolread(f):
    m = iio.imread(f); return (m[..., 0] if m.ndim == 3 else m) > 127


def _bar(canvas, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    except Exception:
        fnt = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0)); d.text((7, 5), text, fill=(255, 255, 255), font=fnt)
    return np.array(im)


if __name__ == "__main__":
    main()
