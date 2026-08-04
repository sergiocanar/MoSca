#!/usr/bin/env python
"""
Stage 2 -- SAM2 windowed propagation (SEPARATE `sam2` env, torch>=2.3).

Reads the Stage-1 contact seeds/events, and for each CONTACT EVENT window prompts
the SAM2 *video* predictor with the rated positive contact seeds (+ high-certainty
negatives), propagates the tissue region forward+backward WITHIN THE CLIP ONLY
(dynamics are transient -- do not broadcast the whole sequence). Produces:
  - peritool/viz/sam2_masks.mp4   (train view + SAM2 tissue mask overlay)  <-- review this
  - peritool/sam2_tissue/*.npy    (per-frame tissue mask, bool [H,W])
Optionally (--write_epi 1) writes UNION(tool ∪ tissue) into epi/error/*.png.npy
(backing up the originals first). Default OFF so the mask viz is reviewed first.
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
    ap.add_argument("--max_prompt_frames", type=int, default=3)   # prompt frames per event
    ap.add_argument("--guard", type=int, default=0)               # intersect with per-frame DEFORMATION mask
    ap.add_argument("--guard_dilate", type=int, default=35)
    ap.add_argument("--guard_win", type=int, default=2)           # velocity window (frames)
    ap.add_argument("--guard_resid", type=float, default=2.5)     # non-rigid residual thresh (px/f)
    ap.add_argument("--min_cc_area", type=int, default=1500)      # drop connected components smaller than this
    ap.add_argument("--write_epi", type=int, default=0)           # write union into epi/error (gated)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz")
    TIS = osp.join(OUT, "sam2_tissue"); os.makedirs(TIS, exist_ok=True)
    tmp = osp.join(OUT, "_clip");

    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"])
    names = [str(x) for x in meta["frame_names_train"]]
    T = len(names)

    sd = np.load(osp.join(OUT, "seeds.npz"), allow_pickle=True)
    pos = json.loads(str(sd["pos"])); neg = json.loads(str(sd["neg"]))
    events = json.loads(str(sd["events"]))
    print(f"{len(events)} events, T={T}")

    # tracks for the optional over-growth guard
    tap = np.load(sorted(glob.glob(osp.join(ws, "uniform_*bootstapir_tap.npz")))[0])
    tr = tap["tracks"].astype(np.float64); visb = tap["visibility"].astype(bool)
    c0, c1 = np.nanmax(tr[..., 0]), np.nanmax(tr[..., 1]); xi, yi = (0, 1) if c0 >= c1 else (1, 0)
    TX, TY = tr[..., xi], tr[..., yi]

    from sam2.build_sam import build_sam2_video_predictor
    device = torch.device("cuda")
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)

    tissue = np.zeros((T, H, W), bool)

    for ev in events:
        s, e, anchor = ev["start"], ev["end"], ev["anchor"]
        L = e - s + 1
        # choose prompt frames within the window that actually have positive seeds
        cand = [(t, len(pos.get(names[t], []))) for t in range(s, e + 1) if len(pos.get(names[t], [])) > 0]
        if not cand:
            print(f"  event {s}-{e}: no positive seeds -> skip"); continue
        cand.sort(key=lambda z: -z[1])
        prompt_ts = sorted({anchor if len(pos.get(names[anchor], [])) else cand[0][0]} |
                           {t for t, _ in cand[:args.max_prompt_frames]})
        # materialize clip as 0.jpg,1.jpg,... (SAM2 parses int frame idx)
        if osp.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for t in range(s, e + 1):
            img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3]
            cv2.imwrite(osp.join(tmp, f"{t - s}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.reset_state(state)
            for t in prompt_ts:
                P = np.array([p for p, _ in pos.get(names[t], [])], np.float32)
                Ng = np.array(neg.get(names[t], []), np.float32)
                if len(P) == 0:
                    continue
                pts = P if len(Ng) == 0 else np.concatenate([P, Ng], 0)
                lbl = np.concatenate([np.ones(len(P), np.int32), np.zeros(len(Ng), np.int32)]) if len(Ng) else np.ones(len(P), np.int32)
                predictor.add_new_points_or_box(state, frame_idx=t - s, obj_id=1,
                                                points=pts, labels=lbl)
            local = {}
            for rev in (False, True):
                for fidx, obj_ids, logits in predictor.propagate_in_video(state, reverse=rev):
                    m = (logits[0, 0] > 0.0).cpu().numpy()
                    local[fidx] = local.get(fidx, np.zeros((m.shape[-2], m.shape[-1]), bool)) | m
        # map back to global, resize to (H,W)
        mv = None
        if args.guard:
            mv = _moved_mask(TX, TY, visb, s, e, H, W, args.guard_dilate)
        for li, m in local.items():
            t = s + li
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            if mv is not None:
                m = m & mv
            tissue[t] |= m
        print(f"  event {s:3d}-{e:3d}: prompts@{prompt_ts}  mask px/frame≈{int(tissue[s:e+1].reshape(L,-1).sum(1).mean())}")

    if osp.isdir(tmp):
        shutil.rmtree(tmp)

    # save per-frame tissue + viz overlay
    tool = np.stack([_toolread(osp.join(ws, "train_masks", f"{n}.png")) for n in names])
    wr = iio.get_writer(osp.join(VIZ, "sam2_masks.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    ev_frames = set()
    for ev in events:
        ev_frames |= set(range(ev["start"], ev["end"] + 1))
    for t in range(T):
        np.save(osp.join(TIS, f"{names[t]}.npy"), tissue[t])
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3].astype(np.float32)
        img[tissue[t]] = img[tissue[t]] * 0.45 + np.array([40, 240, 40]) * 0.55   # tissue = green
        te = tool[t] ^ cv2.erode(tool[t].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[te] = [255, 60, 60]                                                    # tool outline = red
        tag = "EVENT" if t in ev_frames else "quiet"
        wr.append_data(_bar(np.clip(img, 0, 255).astype(np.uint8),
                            f"{names[t]}  [{tag}]  SAM2 dynamic-tissue (green)  tool (red)   px={int(tissue[t].sum())}"))
    wr.close()
    cov = tissue.reshape(T, -1).sum(1)
    print(f"\ntissue mask: frames_with_mask={int((cov>0).sum())}/{T}  mean px (in-event)={int(cov[cov>0].mean()) if (cov>0).any() else 0}")
    print(f"viz -> {VIZ}/sam2_masks.mp4 ; masks -> {TIS}/")

    if args.write_epi:
        epidir = osp.join(ws, "epi", "error")
        bak = osp.join(ws, "epi", "error_tool_backup")
        if not osp.isdir(bak):
            shutil.copytree(epidir, bak); print(f"backed up tool epi -> {bak}")
        for t in range(T):
            union = (tool[t] | tissue[t]).astype(np.float32)
            np.save(osp.join(epidir, f"{names[t]}.png.npy"), union)
        print(f"WROTE union(tool∪tissue) -> {epidir} (originals in {bak})")
    else:
        print("epi/error NOT modified (pass --write_epi 1 to commit the union).")


def _toolread(f):
    m = iio.imread(f); return (m[..., 0] if m.ndim == 3 else m) > 127


def _moved_mask(TX, TY, vis, s, e, H, W, dilate):
    mv = np.zeros((H, W), np.uint8)
    a, b = s, e
    v = vis[a] & vis[b]
    sp = np.hypot(TX[b] - TX[a], TY[b] - TY[a])
    move = v & (sp > 3.0)
    xr = np.clip(np.round(TX[e]).astype(int), 0, W - 1)
    yr = np.clip(np.round(TY[e]).astype(int), 0, H - 1)
    for n in np.where(move)[0]:
        cv2.circle(mv, (xr[n], yr[n]), 10, 1, -1)
    return cv2.dilate(mv, np.ones((dilate, dilate), np.uint8)) > 0


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
