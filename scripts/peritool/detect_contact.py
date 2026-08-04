#!/usr/bin/env python
"""
Stage 1 v3 -- SINGLE-object, TOOL-CENTRIC, motion-gated prompt generator.

MoSca consumes ONE binary dynamic mask; the TOOL is the source of every interaction
and is added back later from the dataset masks (epi = tool ∪ tissue). So SAM2 only
needs TISSUE prompts, all placed NEAR THE TOOL:
  * POSITIVES = tissue that is MOVING and within a band (~band_px) of the tool
    contour  (this catches early/gradual contact e.g. the first ~40 frames, which
    v1's contour-strain missed).
  * NEGATIVES = points ON the tool (so SAM2 never masks it) + a few static-background
    points just OUTSIDE the band (bounds SAM2 growth so it can't flood far tissue).
No regions/IDs/colors. Emits per-frame pos/neg points + an `active` flag.
Outputs (workspaces/<seq>/peritool/):  prompts.json , viz/prompts.mp4
Works half-res internally; points are in full-res coords. No MoSca core edits.
"""
import os, os.path as osp, glob, json, argparse
import numpy as np
import imageio.v2 as iio
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--move_win", type=int, default=3)
    ap.add_argument("--move_thr", type=float, default=2.0)     # full-res px/f
    ap.add_argument("--splat_r", type=int, default=9)          # half-res
    ap.add_argument("--close_k", type=int, default=17)         # half-res
    ap.add_argument("--band_dil", type=int, default=30)        # near-tool band (half-res ~= 60 full)
    ap.add_argument("--neg_ring", type=int, default=16)        # static-bg negatives just outside band (half-res)
    ap.add_argument("--min_pos_area", type=int, default=150)   # half-res px to consider a frame active
    ap.add_argument("--n_pos", type=int, default=5)
    ap.add_argument("--n_tool_neg", type=int, default=4)
    ap.add_argument("--n_bg_neg", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz"); os.makedirs(VIZ, exist_ok=True)
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"]); names = [str(x) for x in meta["frame_names_train"]]; T = len(names)
    sc = args.scale; Hs, Ws = H // sc, W // sc

    tap = np.load(sorted(glob.glob(osp.join(ws, "uniform_*bootstapir_tap.npz")))[0])
    tr = tap["tracks"].astype(np.float64); vis = tap["visibility"].astype(bool)
    c0, c1 = np.nanmax(tr[..., 0]), np.nanmax(tr[..., 1]); xi, yi = (0, 1) if c0 >= c1 else (1, 0)
    X, Y = tr[..., xi], tr[..., yi]; N = X.shape[1]
    tool = np.stack([_tool(osp.join(ws, "train_masks", f"{n}.png"), Hs, Ws) for n in names])
    xr = np.clip(np.round(X / sc).astype(int), 0, Ws - 1); yr = np.clip(np.round(Y / sc).astype(int), 0, Hs - 1)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & vis
    on_tool = np.zeros((T, N), bool)
    for t in range(T):
        on_tool[t] = inb[t] & tool[t][yr[t], xr[t]]
    print(f"T={T} N={N} half={Hs}x{Ws}")

    hw = args.move_win
    prompts = {}; active = []
    dbg = {}
    for t in range(T):
        a, b = max(0, t - hw), min(T - 1, t + hw)
        v = vis[a] & vis[b] & inb[t] & (~on_tool[t])
        sp = np.hypot(X[b] - X[a], Y[b] - Y[a]) / max(b - a, 1)
        mv = v & (sp > args.move_thr)
        mm = np.zeros((Hs, Ws), np.uint8)
        for n in np.where(mv)[0]:
            cv2.circle(mm, (xr[t, n], yr[t, n]), args.splat_r, 1, -1)
        mm = _fill(cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((args.close_k, args.close_k), np.uint8)))
        band = cv2.dilate(tool[t].astype(np.uint8), np.ones((args.band_dil, args.band_dil), np.uint8)) > 0
        pos_mask = (mm > 0) & band & (~tool[t])                 # MOVING & near-tool tissue
        pos = _sample_interior(pos_mask, args.n_pos)
        # negatives: on tool + static bg just outside band
        neg = []
        er = cv2.erode(tool[t].astype(np.uint8), np.ones((9, 9), np.uint8)); ys, xs = np.where(er)
        if len(xs):
            for k in np.linspace(0, len(xs) - 1, min(args.n_tool_neg, len(xs))).astype(int):
                neg.append([xs[k], ys[k]])
        outer = cv2.dilate(band.astype(np.uint8), np.ones((args.neg_ring * 2, args.neg_ring * 2), np.uint8)) > 0
        ring = outer & (~band) & (mm == 0)
        rys, rxs = np.where(ring)
        if len(rxs):
            for k in np.linspace(0, len(rxs) - 1, min(args.n_bg_neg, len(rxs))).astype(int):
                neg.append([rxs[k], rys[k]])
        act = len(pos) > 0 and pos_mask.sum() >= args.min_pos_area
        prompts[str(t)] = dict(pos=[[float(p[0] * sc), float(p[1] * sc)] for p in pos] if act else [],
                               neg=[[float(p[0] * sc), float(p[1] * sc)] for p in neg])
        if act:
            active.append(t)
        dbg[t] = (mm > 0, band)
        if t % 40 == 0:
            print(f"  {t}/{T}  pos={len(pos) if act else 0} neg={len(neg)} active={act}")

    with open(osp.join(OUT, "prompts.json"), "w") as fp:
        json.dump(dict(frames=prompts, active=active, H=H, W=W, T=T, params=vars(args)), fp)
    # span summary
    spans = _spans(active, T)
    print(f"\nactive frames: {len(active)}/{T}  |  interaction spans: {[(s,e) for s,e in spans]}")

    # ---- viz ---------------------------------------------------------------
    wr = iio.get_writer(osp.join(VIZ, "prompts.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    for t in range(T):
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3].astype(np.float32)
        mm, band = dbg[t]
        band_f = cv2.resize(band.astype(np.uint8), (W, H), cv2.INTER_NEAREST) > 0
        be = band_f ^ cv2.erode(band_f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        mv_f = cv2.resize(mm.astype(np.uint8), (W, H), cv2.INTER_NEAREST) > 0
        me = mv_f ^ cv2.erode(mv_f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        tl_f = cv2.resize(tool[t].astype(np.uint8), (W, H), cv2.INTER_NEAREST) > 0
        te = tl_f ^ cv2.erode(tl_f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[me] = [235, 235, 60]      # moving = yellow outline
        img[be] = [80, 160, 255]      # near-tool band = blue outline
        img[te] = [255, 60, 60]       # tool = red outline
        P = prompts[str(t)]
        for p in P["pos"]:
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (30, 255, 30), -1)
        for p in P["neg"]:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 12, 2)
        tag = "ACTIVE" if t in active else "quiet"
        wr.append_data(_bar(np.clip(img, 0, 255).astype(np.uint8),
                            f"{names[t]} [{tag}]  green=+ (moving near-tool)  x=- (tool/static)  blue=band  yellow=moving"))
    wr.close()
    print(f"viz -> {VIZ}/prompts.mp4 ; prompts -> {OUT}/prompts.json")


def _spans(active, T, gap=3):
    if not active:
        return []
    active = sorted(active); spans = []; s = active[0]; p = active[0]
    for t in active[1:]:
        if t - p <= gap + 1:
            p = t
        else:
            spans.append((s, p)); s = t; p = t
    spans.append((s, p)); return spans


def _tool(f, Hs, Ws):
    m = iio.imread(f); m = (m[..., 0] if m.ndim == 3 else m) > 127
    return cv2.resize(m.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0


def _fill(m):
    ff = m.copy(); h, w = m.shape; mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    return (m | (1 - ff)).astype(np.uint8)


def _sample_interior(mask, n):
    if mask.sum() == 0:
        return []
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(dt >= 0.35 * dt.max()) if dt.max() > 0 else np.where(mask)
    if len(xs) == 0:
        return []
    cand = np.stack([xs, ys], 1).astype(float); dv = dt[ys, xs]
    sel = [int(np.argmax(dv))]
    for _ in range(min(n, len(cand)) - 1):
        d = np.min(np.linalg.norm(cand[:, None, :] - cand[sel][None, :, :], axis=-1), axis=1)
        sel.append(int(np.argmax(d)))
    return [cand[i] for i in sel]


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
