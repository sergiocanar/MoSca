#!/usr/bin/env python
"""
Stage 1 -- Peri-tool CONTACT-BORDER detector (mosca env, CPU, standalone).

Reads a MoSca iMED workspace and decides, along the surgical tool's contour,
WHERE the tool is actually TOUCHING / ALTERING tissue (a contact seed) vs where
its border merely OCCLUDES static background. Output = RATED positive contact
seed points (+ negative points + temporal event windows) that Stage 2 (SAM2)
uses as prompts to extract the full dynamic tissue masks.

Signals (per tool-contour point, aggregated over a temporal window; contact is an
EVENT not a frame), using only 'clean' uniform tracks that never touch the tool:
  (1) STRAIN   : non-rigid stretch of the peri-tool tissue track field
                 (pairwise inter-track distance change; translation/rotation inv.)
  (2) DEPTH-GAP: |depth_inside_tool - depth_outside| across the edge
                 (small gap = coplanar/contact ; large jump = occlusion)
  (3) CORREL   : does adjacent tissue velocity move WITH the tool (cos-sim)
  (4) PERSIST  : does the tissue patch stay altered after the tool leaves
                 (depth before-vs-after; reverts => occlusion)  [optional]

NOTHING in MoSca core is modified. Outputs go to <ws>/peritool/.
"""
import os, os.path as osp, glob, json, argparse
import numpy as np
import imageio.v2 as iio
import cv2
from scipy.spatial import cKDTree


def load_stack(files, reader):
    return [reader(f) for f in files]


def tool_mask_read(f):
    m = iio.imread(f)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def main():
    ap = argparse.ArgumentParser("peri-tool contact detector")
    ap.add_argument("--ws", required=True)
    ap.add_argument("--window", type=int, default=7)          # temporal window (frames)
    ap.add_argument("--n_contour", type=int, default=220)      # resampled contour pts
    ap.add_argument("--annulus_r_in", type=float, default=6)
    ap.add_argument("--annulus_r_out", type=float, default=42)
    ap.add_argument("--min_neighbors", type=int, default=4)
    ap.add_argument("--gap_thresh", type=float, default=8.0)   # mm; below => contact
    ap.add_argument("--gap_soft", type=float, default=4.0)
    ap.add_argument("--persist_delta", type=int, default=8)
    ap.add_argument("--use_persist", type=int, default=1)
    ap.add_argument("--w_strain", type=float, default=0.35)
    ap.add_argument("--w_gap", type=float, default=0.25)
    ap.add_argument("--w_corr", type=float, default=0.20)
    ap.add_argument("--w_persist", type=float, default=0.20)
    ap.add_argument("--seed_th", type=float, default=0.60)
    ap.add_argument("--neg_th", type=float, default=0.25)
    ap.add_argument("--event_th", type=float, default=0.55)
    ap.add_argument("--min_event_len", type=int, default=3)
    ap.add_argument("--max_event_len", type=int, default=12)   # split long events into bursts
    ap.add_argument("--seed_offset", type=float, default=15)
    ap.add_argument("--seed_min_sep", type=float, default=25)
    ap.add_argument("--n_bg_neg", type=int, default=6)        # highest-certainty background negatives / frame
    ap.add_argument("--n_tool_neg", type=int, default=3)      # tool-interior negatives / frame
    ap.add_argument("--neg_pos_clear", type=float, default=60)  # drop negatives within this px of any positive
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ws = args.ws
    OUT = osp.join(ws, "peritool"); VIZ = osp.join(OUT, "viz")
    os.makedirs(VIZ, exist_ok=True)

    # ---- load workspace ----------------------------------------------------
    meta = np.load(osp.join(ws, "imed_meta.npz"), allow_pickle=True)
    H, W = int(meta["H"]), int(meta["W"])
    names = [str(x) for x in meta["frame_names_train"]]
    T = len(names)
    overlap = meta["overlap_mask_train"].astype(bool) if "overlap_mask_train" in meta.files else np.ones((H, W), bool)

    tap = np.load(sorted(glob.glob(osp.join(ws, "uniform_*bootstapir_tap.npz")))[0])
    tr = tap["tracks"].astype(np.float64)            # [T,N,2]
    vis = tap["visibility"].astype(bool)             # [T,N]
    c0max, c1max = np.nanmax(tr[..., 0]), np.nanmax(tr[..., 1])
    xi, yi = (0, 1) if c0max >= c1max else (1, 0)
    X = tr[..., xi]; Y = tr[..., yi]
    N = X.shape[1]

    tool = np.stack(load_stack([osp.join(ws, "train_masks", f"{n}.png") for n in names], tool_mask_read))  # [T,H,W]
    dep = np.stack(load_stack(sorted(glob.glob(osp.join(ws, "sensor_depth", "*.npz"))),
                              lambda f: np.load(f)["dep"].astype(np.float32)))  # [T,H,W]
    print(f"T={T} N={N} x-ch={xi}  tool frac(mean)={tool.mean():.3f}")

    # ---- per-track helpers -------------------------------------------------
    xr = np.clip(np.round(X).astype(int), 0, W - 1)
    yr = np.clip(np.round(Y).astype(int), 0, H - 1)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & vis
    on_tool = np.zeros((T, N), bool)
    for t in range(T):
        on_tool[t] = inb[t] & tool[t][yr[t], xr[t]]
    # tool centroid velocity per frame (for correlation)
    tool_cent = np.full((T, 2), np.nan)
    for t in range(T):
        ys, xs = np.where(tool[t])
        if len(xs) > 0:
            tool_cent[t] = [xs.mean(), ys.mean()]

    hw = args.window // 2

    # ---------------- PASS 1: raw signals per contour point -----------------
    frames = []  # each: dict(pts[K,2], nrm[K,2], strain, gap, corr, persist, ncnt)
    for t in range(T):
        m = tool[t].astype(np.uint8)
        if m.sum() < 30:
            frames.append(None); continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).squeeze(1).astype(np.float64)  # [P,2] (x,y)
        if cnt.ndim != 2 or len(cnt) < 8:
            frames.append(None); continue
        idx = np.linspace(0, len(cnt) - 1, args.n_contour).astype(int)
        C = cnt[idx]                                   # [K,2]
        # outward normals via distance transform of background
        dt = cv2.distanceTransform((~tool[t]).astype(np.uint8), cv2.DIST_L2, 5)
        gy, gx = np.gradient(dt)
        cx = np.clip(C[:, 0].astype(int), 0, W - 1); cy = np.clip(C[:, 1].astype(int), 0, H - 1)
        nrm = np.stack([gx[cy, cx], gy[cy, cx]], 1)
        nn = np.linalg.norm(nrm, axis=1, keepdims=True); nrm = nrm / np.clip(nn, 1e-6, None)
        K = len(C)

        a, b = max(0, t - hw), min(T - 1, t + hw)
        clean = vis[a] & vis[b] & (~on_tool[a:b + 1].any(0)) & \
                (X[a] >= 0) & (X[a] < W) & (Y[a] >= 0) & (Y[a] < H)
        cl = np.where(clean)[0]
        Pt = np.stack([X[t, cl], Y[t, cl]], 1)         # positions at t
        Pa = np.stack([X[a, cl], Y[a, cl]], 1)
        Pb = np.stack([X[b, cl], Y[b, cl]], 1)
        velw = (Pb - Pa) / max(b - a, 1)               # mean velocity over window
        tv = tool_cent[b] - tool_cent[a]
        tvn = tv / max(np.linalg.norm(tv), 1e-6)
        tree = cKDTree(Pt) if len(Pt) else None

        strain = np.full(K, np.nan); gap = np.full(K, np.nan)
        corr = np.full(K, np.nan); persist = np.full(K, np.nan); ncnt = np.zeros(K, int)
        for j in range(K):
            c = C[j]; n = nrm[j]
            # depth-gap across the edge
            pin = c - 8 * n; pout = c + 8 * n
            zi = _patch(dep[t], pin); zo = _patch(dep[t], pout)
            if zi > 1e-3 and zo > 1e-3:
                gap[j] = abs(zo - zi)
            if tree is not None:
                cand = tree.query_ball_point(c, args.annulus_r_out)
                if cand:
                    rel = Pt[cand] - c
                    d = np.linalg.norm(rel, axis=1)
                    outside = (rel @ n) > 0
                    sel = np.array(cand)[(d >= args.annulus_r_in) & outside]
                    ncnt[j] = len(sel)
                    if len(sel) >= args.min_neighbors:
                        # (1) strain: pairwise inter-track stretch a->b
                        pa = Pa[np.searchsorted(cl, cl[sel]) if False else sel]
                        pa = Pa[sel]; pb = Pb[sel]
                        mrows = min(len(sel), 24)
                        ii = np.random.default_rng(j).choice(len(sel), mrows, replace=False) if len(sel) > mrows else np.arange(len(sel))
                        da = _pdist(pa[ii]); db = _pdist(pb[ii])
                        strain[j] = np.mean(np.abs(db - da)) / (np.mean(da) + 1e-3)
                        # (3) correlation with tool motion
                        mv = velw[sel].mean(0)
                        mvn = mv / max(np.linalg.norm(mv), 1e-6)
                        corr[j] = float(mvn @ tvn) * min(np.linalg.norm(mv) / (np.linalg.norm(tv) / max(b - a, 1) + 1e-3), 1.0)
            # (4) persistence (depth before vs after tool passes)
            if args.use_persist:
                tpre, tpost = a - args.persist_delta, b + args.persist_delta
                if 0 <= tpre and tpost < T:
                    if not (_ontool_patch(tool[tpre], c) or _ontool_patch(tool[tpost], c)):
                        zpre = _patch(dep[tpre], c, r=6); zpost = _patch(dep[tpost], c, r=6)
                        if zpre > 1e-3 and zpost > 1e-3:
                            persist[j] = abs(zpost - zpre)
        frames.append(dict(C=C, nrm=nrm, strain=strain, gap=gap, corr=corr, persist=persist, ncnt=ncnt))
        if t % 40 == 0:
            print(f"  pass1 {t}/{T}")

    # ---------------- normalization (robust, global) ------------------------
    def robust(vals, invert=False):
        v = np.concatenate([f[vals] for f in frames if f is not None])
        v = v[np.isfinite(v)]
        lo, hi = (np.percentile(v, [5, 95]) if len(v) else (0, 1))
        return lo, hi
    s_lo, s_hi = robust("strain")
    g_lo, g_hi = robust("gap")
    c_lo, c_hi = robust("corr")
    p_lo, p_hi = robust("persist")

    def nrm01(x, lo, hi):
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    # ---------------- PASS 2: score, seeds, events, viz ---------------------
    max_score = np.zeros(T)
    seeds_pos, seeds_neg = {}, {}
    wr = iio.get_writer(osp.join(VIZ, "contact.mp4"), fps=args.fps, macro_block_size=None, quality=8)
    for t in range(T):
        f = frames[t]
        img = iio.imread(osp.join(ws, "images", f"{names[t]}.png"))[..., :3]
        left = img.copy(); left[tool[t]] = left[tool[t]] * 0.4 + np.array([40, 220, 40]) * 0.6
        right = img.copy()
        if f is not None:
            S = nrm01(f["strain"], s_lo, s_hi)
            G = 1.0 - nrm01(f["gap"], g_lo, g_hi)                  # small gap -> high
            Cc = np.clip(f["corr"], 0, None); Cc = nrm01(Cc, max(c_lo, 0), c_hi)
            P = nrm01(f["persist"], p_lo, p_hi)
            for arr in (S, G, Cc, P):
                arr[~np.isfinite(arr)] = 0.0
            score = args.w_strain * S + args.w_gap * G + args.w_corr * Cc + args.w_persist * P
            # require some tissue evidence: contour pt must have neighbors
            score[f["ncnt"] < args.min_neighbors] *= 0.3
            max_score[t] = float(score.max()) if len(score) else 0.0
            # draw contour colored by score
            col = (cv2.applyColorMap((np.clip(score, 0, 1) * 255).astype(np.uint8).reshape(-1, 1),
                                     cv2.COLORMAP_JET)[:, 0, ::-1])  # RGB
            for j in range(len(f["C"])):
                cx, cy = int(f["C"][j, 0]), int(f["C"][j, 1])
                cv2.circle(right, (cx, cy), 3, tuple(int(v) for v in col[j]), -1)
            # positive seeds (pushed outward), rated by score
            pos = []
            order = np.argsort(-score)
            for j in order:
                if score[j] < args.seed_th:
                    break
                p = f["C"][j] + args.seed_offset * f["nrm"][j]
                if p[0] < 0 or p[0] >= W or p[1] < 0 or p[1] >= H:
                    continue
                if any(np.hypot(p[0] - q[0], p[1] - q[1]) < args.seed_min_sep for q, _ in pos):
                    continue
                pos.append((p.tolist(), float(score[j])))
            # negatives: keep only the HIGHEST-CERTAINTY ones, and keep them clear
            # of positives (else SAM2 gets contradictory prompts).
            pos_pts = np.array([p for p, _ in pos], float) if pos else np.zeros((0, 2))

            def far_from_pos(p):
                return (len(pos_pts) == 0 or
                        np.all(np.hypot(pos_pts[:, 0] - p[0], pos_pts[:, 1] - p[1]) > args.neg_pos_clear))
            neg = []
            # background negatives = the LOWEST-score contour points (most certainly non-contact)
            low = np.where(score < args.neg_th)[0]
            low = low[np.argsort(score[low])]                       # ascending score => most certain first
            for j in low:
                if len(neg) >= args.n_bg_neg:
                    break
                p = f["C"][j] + args.seed_offset * f["nrm"][j]
                if not (0 <= p[0] < W and 0 <= p[1] < H) or not far_from_pos(p):
                    continue
                if any(np.hypot(p[0] - q[0], p[1] - q[1]) < args.seed_min_sep for q in neg):
                    continue
                neg.append(p.tolist())
            # a few tool-interior negatives, also cleared from positives
            er = cv2.erode(tool[t].astype(np.uint8), np.ones((21, 21), np.uint8))
            ys, xs = np.where(er)
            if len(xs) and args.n_tool_neg > 0:
                kept = 0
                for k in np.linspace(0, len(xs) - 1, min(args.n_tool_neg * 4, len(xs))).astype(int):
                    if kept >= args.n_tool_neg:
                        break
                    p = [float(xs[k]), float(ys[k])]
                    if far_from_pos(p):
                        neg.append(p); kept += 1
            seeds_pos[names[t]] = pos; seeds_neg[names[t]] = neg
            # draw seeds
            for (p, sc) in pos:
                cv2.circle(right, (int(p[0]), int(p[1])), int(5 + 10 * sc), (30, 255, 30), 2)
            for p in neg:
                cv2.drawMarker(right, (int(p[0]), int(p[1])), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 10, 2)
        te = tool[t] ^ cv2.erode(tool[t].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        right[te] = [255, 60, 60]
        canvas = _bar(np.concatenate([left, np.full((H, 12, 3), 255, np.uint8), right], 1),
                      f"{names[t]}  max_contact={max_score[t]:.2f}   LEFT tool-epi(now)   RIGHT contour=score  green=RATED seeds  x=neg")
        wr.append_data(canvas)
    wr.close()

    # ---------------- events from max_score --------------------------------
    hot = max_score > args.event_th
    events = []
    t = 0
    while t < T:
        if hot[t]:
            s = t
            while t + 1 < T and (hot[t + 1] or (t + 2 < T and hot[t + 2])):  # bridge <=1 gap
                t += 1
            if t - s + 1 >= args.min_event_len:
                # split long runs into <=max_event_len bursts, each with its own anchor
                L = t - s + 1
                nchunk = int(np.ceil(L / args.max_event_len))
                bounds = np.linspace(s, t + 1, nchunk + 1).astype(int)
                for ci in range(nchunk):
                    cs, ce = int(bounds[ci]), int(bounds[ci + 1]) - 1
                    if ce - cs + 1 < 2:
                        continue
                    anchor = cs + int(np.argmax(max_score[cs:ce + 1]))
                    events.append(dict(start=cs, end=ce, anchor=int(anchor),
                                       peak=float(max_score[cs:ce + 1].max())))
        t += 1

    # ---------------- save + timeline viz ----------------------------------
    np.savez_compressed(osp.join(OUT, "seeds.npz"),
                        max_score=max_score,
                        pos=np.array(json.dumps(seeds_pos)), neg=np.array(json.dumps(seeds_neg)),
                        events=np.array(json.dumps(events)))
    with open(osp.join(OUT, "events.json"), "w") as fp:
        json.dump({"events": events, "n_events": len(events),
                   "frames_in_events": int(sum(e["end"] - e["start"] + 1 for e in events)),
                   "params": vars(args)}, fp, indent=2)
    _timeline(max_score, events, args.event_th, osp.join(VIZ, "events_timeline.png"))

    print(f"\n=== DONE ===")
    print(f"events: {len(events)}  covering {sum(e['end']-e['start']+1 for e in events)}/{T} frames")
    for e in events:
        print(f"  frames {e['start']:3d}-{e['end']:3d} (anchor {e['anchor']}, peak {e['peak']:.2f})")
    print(f"viz -> {VIZ}/contact.mp4 , events_timeline.png")
    print(f"seeds -> {OUT}/seeds.npz , events.json")


def _patch(img, p, r=3):
    x, y = int(round(p[0])), int(round(p[1]))
    H, W = img.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1); y0, y1 = max(0, y - r), min(H, y + r + 1)
    v = img[y0:y1, x0:x1]; v = v[v > 1e-3]
    return float(np.median(v)) if v.size else 0.0


def _ontool_patch(tmask, p, r=3):
    x, y = int(round(p[0])), int(round(p[1]))
    H, W = tmask.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1); y0, y1 = max(0, y - r), min(H, y + r + 1)
    return bool(tmask[y0:y1, x0:x1].any())


def _pdist(P):
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return d[np.triu_indices(len(P), 1)]


def _bar(canvas, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        fnt = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 36], fill=(0, 0, 0)); d.text((8, 5), text, fill=(255, 255, 255), font=fnt)
    return np.array(im)


def _timeline(score, events, th, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(score, color="k", lw=1); ax.axhline(th, color="r", ls="--", lw=1, label=f"event_th={th}")
    for e in events:
        ax.axvspan(e["start"], e["end"], color="tab:green", alpha=0.3)
        ax.plot(e["anchor"], e["peak"], "g^")
    ax.set_xlabel("frame"); ax.set_ylabel("max contact score"); ax.legend(loc="upper right")
    ax.set_title(f"{len(events)} contact events"); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
