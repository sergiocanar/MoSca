#!/usr/bin/env python
"""
Stage 1 v2 -- MOTION-FIRST, per-REGION, multi-object contact detector.

Reality (from frame-by-frame review of v1): the tool moves MULTIPLE tissue regions,
each moving-because-touched, over overlapping CONTINUOUS spans; gross bulk motion of
a whole soft organ is a CORRECT dynamic region (not to be shrunk). v1's contour-strain
seeds were single-mode / threshold-gated / time-fragmented and missed gradual+concurrent
contacts.

v2 approach (this file):
  * MOTION-FIRST: per frame, build a dense moving-tissue mask from uniform-track SPEED
    (gross motion -- bulk organ motion is wanted), excluding on-tool tracks.
  * TOOL-GATED: keep moving connected-components that TOUCH the tool (adjacent to the
    dilated tool silhouette). The tool is the cause.
  * PER-REGION TRACKING: match components across time into continuous REGION lifetimes
    (multiple concurrent regions; short gaps bridged). Each region -> seed points.
Outputs (workspaces/<seq>/peritool/):
  regions.json  -- regions[{id, span, frames, seeds{t:[[x,y]..]}}], negatives{t:[..]}, params
  viz/regions.mp4 -- per-region colored overlay + seeds on the train view (review this)
No MoSca core is modified. Works at half resolution internally; seeds in full-res coords.
"""
import os, os.path as osp, glob, json, argparse
import numpy as np
import imageio.v2 as iio
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--scale", type=int, default=2)            # internal downsample factor
    ap.add_argument("--move_win", type=int, default=3)         # velocity window (frames)
    ap.add_argument("--move_thr", type=float, default=2.0)     # moving speed (full-res px/f)
    ap.add_argument("--splat_r", type=int, default=9)          # moving-point splat radius (half-res)
    ap.add_argument("--close_k", type=int, default=15)         # morphological close (half-res)
    ap.add_argument("--contact_dil", type=int, default=16)     # tool dilation for adjacency (half-res)
    ap.add_argument("--min_area", type=int, default=700)       # min region area (half-res px)
    ap.add_argument("--match_iou", type=float, default=0.15)
    ap.add_argument("--match_dist", type=float, default=70)    # centroid match (half-res px)
    ap.add_argument("--gap_bridge", type=int, default=3)
    ap.add_argument("--min_region_len", type=int, default=3)
    ap.add_argument("--n_seed", type=int, default=5)
    ap.add_argument("--n_bg_neg", type=int, default=6)
    ap.add_argument("--n_tool_neg", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz"); os.makedirs(VIZ, exist_ok=True)
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"])
    names = [str(x) for x in meta["frame_names_train"]]
    T = len(names)
    sc = args.scale; Hs, Ws = H // sc, W // sc

    tap = np.load(sorted(glob.glob(osp.join(ws, "uniform_*bootstapir_tap.npz")))[0])
    tr = tap["tracks"].astype(np.float64); vis = tap["visibility"].astype(bool)
    c0, c1 = np.nanmax(tr[..., 0]), np.nanmax(tr[..., 1]); xi, yi = (0, 1) if c0 >= c1 else (1, 0)
    X, Y = tr[..., xi], tr[..., yi]; N = X.shape[1]

    tool = np.stack([_tool(osp.join(ws, "train_masks", f"{n}.png"), Hs, Ws) for n in names])  # [T,Hs,Ws]
    print(f"T={T} N={N} half-res={Hs}x{Ws}")

    on_tool = np.zeros((T, N), bool)
    xr = np.clip(np.round(X / sc).astype(int), 0, Ws - 1); yr = np.clip(np.round(Y / sc).astype(int), 0, Hs - 1)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & vis
    for t in range(T):
        on_tool[t] = inb[t] & tool[t][yr[t], xr[t]]

    hw = args.move_win
    # ---- per-frame moving+tool-touching components -------------------------
    per_comps = []       # list over t: list of dict(mask[Hs,Ws]bool, cent(x,y), area)
    move_dbg = np.zeros((T, Hs, Ws), bool)
    for t in range(T):
        a, b = max(0, t - hw), min(T - 1, t + hw)
        v = vis[a] & vis[b] & inb[t] & (~on_tool[t])
        sp = np.hypot(X[b] - X[a], Y[b] - Y[a]) / max(b - a, 1)
        mv = v & (sp > args.move_thr)
        mm = np.zeros((Hs, Ws), np.uint8)
        for n in np.where(mv)[0]:
            cv2.circle(mm, (xr[t, n], yr[t, n]), args.splat_r, 1, -1)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((args.close_k, args.close_k), np.uint8))
        mm = _fill(mm)
        move_dbg[t] = mm > 0
        tprox = cv2.dilate(tool[t].astype(np.uint8), np.ones((args.contact_dil, args.contact_dil), np.uint8)) > 0
        ncc, lab, stats, cent = cv2.connectedComponentsWithStats(mm.astype(np.uint8), 8)
        comps = []
        for i in range(1, ncc):
            if stats[i, cv2.CC_STAT_AREA] < args.min_area:
                continue
            cm = lab == i
            if not (cm & tprox).any():                     # must touch the tool
                continue
            comps.append(dict(mask=cm, cent=np.array([cent[i, 0], cent[i, 1]]), area=int(stats[i, cv2.CC_STAT_AREA])))
        per_comps.append(comps)
        if t % 40 == 0:
            print(f"  motion {t}/{T}: {len(comps)} region(s)")

    # ---- temporal region tracking (greedy IoU/centroid) --------------------
    regions = {}; nxt = 0
    for t in range(T):
        comps = per_comps[t]
        act = [rid for rid, r in regions.items() if r["last_t"] >= t - 1 - args.gap_bridge]
        cand = []
        for ci, c in enumerate(comps):
            for rid in act:
                iou = _iou(c["mask"], regions[rid]["last_mask"])
                dist = np.linalg.norm(c["cent"] - regions[rid]["last_cent"])
                if iou >= args.match_iou or dist <= args.match_dist:
                    cand.append((iou, -dist, ci, rid))
        cand.sort(reverse=True)
        uc, ur, asg = set(), set(), {}
        for iou, nd, ci, rid in cand:
            if ci in uc or rid in ur:
                continue
            uc.add(ci); ur.add(rid); asg[ci] = rid
        for ci, c in enumerate(comps):
            rid = asg.get(ci)
            if rid is None:
                rid = nxt; nxt += 1
                regions[rid] = dict(frames={}, seeds={}, last_t=-1, last_mask=None, last_cent=None)
            r = regions[rid]
            r["frames"][t] = True
            r["seeds"][t] = _sample_interior(c["mask"], args.n_seed)
            r["last_t"] = t; r["last_mask"] = c["mask"]; r["last_cent"] = c["cent"]
            r["_lastmask_for_viz"] = None

    # keep regions with real lifetime; relabel 0..K-1
    kept = []
    for rid, r in regions.items():
        fs = sorted(r["frames"].keys())
        if fs and (fs[-1] - fs[0] + 1) >= args.min_region_len and len(fs) >= 2:
            kept.append((fs[0], rid, r, fs))
    kept.sort()
    reg_out = []
    for newid, (_, rid, r, fs) in enumerate(kept):
        seeds = {int(t): [[float(p[0] * sc), float(p[1] * sc)] for p in r["seeds"][t]] for t in fs}
        reg_out.append(dict(id=newid, span=[int(fs[0]), int(fs[-1])], frames=[int(t) for t in fs], seeds=seeds))
    print(f"\n=== regions: {len(reg_out)} ===")
    for R in reg_out:
        s, e = R["span"]; print(f"  region {R['id']:2d}: frames {s:3d}-{e:3d} ({len(R['frames'])} present)")

    # ---- negatives (static bg + tool interior), per frame ------------------
    negs = {}
    for t in range(T):
        mm = move_dbg[t]; tl = tool[t]
        occ = mm | tl
        bg = []
        gy, gx = np.mgrid[20:Hs - 20:40, 20:Ws - 20:40]
        for yy, xx in zip(gy.ravel(), gx.ravel()):
            if not occ[yy, xx]:
                bg.append([xx, yy])
        bg = bg[:: max(1, len(bg) // max(args.n_bg_neg, 1))][:args.n_bg_neg]
        er = cv2.erode(tl.astype(np.uint8), np.ones((11, 11), np.uint8)); ys, xs = np.where(er)
        tn = []
        if len(xs):
            for k in np.linspace(0, len(xs) - 1, min(args.n_tool_neg, len(xs))).astype(int):
                tn.append([int(xs[k]), int(ys[k])])
        negs[int(t)] = [[float(p[0] * sc), float(p[1] * sc)] for p in bg + tn]

    with open(osp.join(OUT, "regions.json"), "w") as fp:
        json.dump(dict(regions=reg_out, negatives=negs, params=vars(args),
                       H=H, W=W, T=T, frames_with_region=int(sum(bool(per_comps[t]) for t in range(T)))), fp)

    # ---- viz ---------------------------------------------------------------
    palette = _palette()
    # map (t) -> list of (region_id, seeds_fullres)
    byframe = {t: [] for t in range(T)}
    for R in reg_out:
        for t in R["frames"]:
            byframe[t].append(R)
    wr = iio.get_writer(osp.join(VIZ, "regions.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    for t in range(T):
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3].astype(np.float32)
        # recompute per-region comp masks at this frame for coloring (via move+tool cc)
        active_ids = [R["id"] for R in byframe[t]]
        # color the moving-region pixels by nearest active region seed cluster (cheap: recolor comps)
        for R in byframe[t]:
            col = np.array(palette[R["id"] % len(palette)], np.float32)
            for p in R["seeds"].get(t, []):
                cv2.circle(img, (int(p[0]), int(p[1])), 10, tuple(int(v) for v in col), 2)
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (30, 255, 30), -1)
        # tool outline red, moving-mask faint outline
        tl_full = cv2.resize(tool[t].astype(np.uint8), (W, H), cv2.INTER_NEAREST) > 0
        te = tl_full ^ cv2.erode(tl_full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        mv_full = cv2.resize(move_dbg[t].astype(np.uint8), (W, H), cv2.INTER_NEAREST) > 0
        me = mv_full ^ cv2.erode(mv_full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[me] = [255, 235, 60]
        img[te] = [255, 60, 60]
        for p in negs[t]:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 9, 2)
        wr.append_data(_bar(np.clip(img, 0, 255).astype(np.uint8),
                            f"{names[t]}  regions={active_ids}  (green=seeds  yellow=moving  red=tool/neg)"))
    wr.close()
    print(f"viz -> {VIZ}/regions.mp4 ; regions -> {OUT}/regions.json")


def _tool(f, Hs, Ws):
    m = iio.imread(f); m = (m[..., 0] if m.ndim == 3 else m) > 127
    return cv2.resize(m.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0


def _fill(m):
    ff = m.copy(); h, w = m.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    return (m | (1 - ff)).astype(np.uint8)


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    i = (a & b).sum(); u = (a | b).sum()
    return i / u if u else 0.0


def _sample_interior(mask, n):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dt.max() <= 0:
        ys, xs = np.where(mask)
    else:
        ys, xs = np.where(dt >= 0.35 * dt.max())
    if len(xs) == 0:
        return []
    cand = np.stack([xs, ys], 1).astype(float)
    dvals = dt[ys, xs]
    sel = [int(np.argmax(dvals))]
    for _ in range(min(n, len(cand)) - 1):
        d = np.min(np.linalg.norm(cand[:, None, :] - cand[sel][None, :, :], axis=-1), axis=1)
        sel.append(int(np.argmax(d)))
    return [cand[i] for i in sel]


def _palette():
    return [(66, 135, 245), (245, 130, 49), (60, 200, 120), (200, 60, 200),
            (240, 220, 40), (40, 220, 220), (200, 90, 90), (120, 120, 250)]


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
