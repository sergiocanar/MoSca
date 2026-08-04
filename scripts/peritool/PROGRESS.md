# Peri-tool dynamic-mask — iteration log (story + backups)

Goal: redirect MoSca's dynamic capacity onto the deforming peri-tool TISSUE (the
tool grabs/twists/pulls) instead of the metric-excluded TOOL. Inject via
`ws/epi/error/*.png.npy` (union tool∪tissue). Working seq: session_004_scene_2_tool_1.

Backups:
- scripts: `scripts/peritool/archive/<version>/`
- outputs: `workspaces/<seq>/peritool_iters/<version>/`
- live working dir: `workspaces/<seq>/peritool/`

---

## v1 — contour-strain seeds  (ARCHIVED: archive/v1_contour_seed, peritool_iters/v1_contour_seed)
**Stage 1 (`detect_contact.py`):** per tool-contour point, score contact from
strain + depth-gap + tool-motion correlation + persistence, over a temporal
window; emit rated positive seeds + negatives + GLOBAL event windows.
**Stage 2 (`sam2_propagate.py`):** per global event, prompt SAM2 video predictor
at rated seeds, propagate within the clip; optional (mis)guided deformation guard.

**Result:** seeds correctly land on the tool tip/jaws; SAM2 makes decent masks on
short events. But user's frame-by-frame review found systematic failures:
1. **0–50:** right gland is pushed the whole time but not masked until ~51
   (missed gradual/early contact; contour-strain never crossed threshold there).
2. **76 good → 78 torn → 82 gone** though contact runs to 86 (event fragmentation
   gap at 81–82 + SAM2 single-anchor drift on deforming/specular tissue).
3. **104, 106:** lonesome phantom island artifacts (SAM2 memory false-positives).
4. **160:** tool touches BOTH right gland and middle tissue; only the right gland
   is masked, middle not until 181 (single-mode seeding misses concurrent contact).
5. **93–118 whole-organ flood is CORRECT** — soft organ genuinely all moves.

**Diagnosis:** algorithm models ONE sparsely-seeded, threshold-gated,
time-fragmented contact; reality is MULTIPLE tissue regions, each moving-because-
touched, over overlapping continuous spans. The whole-organ flood was never the
bug → the **deformation guard is wrong and is dropped**. Real problems:
(a) seeding is single-mode/sparse/strain-thresholded → misses concurrent+gradual
contacts; (b) global fragmented events → gaps drop continuous contacts;
(c) SAM2 single-anchor propagation drifts → tearing/dropout/islands.

---

## v2 — motion-first, per-region, multi-object  (IN PROGRESS)
**Objective:** mask every tissue region that is moving BECAUSE the tool touches it,
for its full extent and full duration (big or small).
**Stage 1 (`detect_contact.py` v2):** MOTION-FIRST. Per frame, build a dense
moving-tissue mask from uniform-track speed (gross motion, NOT strain — bulk organ
motion is wanted); keep connected components that TOUCH the tool (adjacent to the
dilated tool); track components across time into continuous per-REGION lifetimes
(multiple concurrent regions supported, gaps bridged). Emit per-region seed points
+ negatives (static bg + tool interior + other regions).
**Stage 2 (`sam2_propagate.py` v2):** MULTI-OBJECT (one SAM2 obj per region),
RE-ANCHOR every K frames from fresh region seeds, propagate, GATE islands.
**Result:** fixed coverage (early gland 0-50, concurrent regions) BUT overshot:
20 regions (over-fragmented), SAM2 ballooned each seed to whole organs, union
flooded ~50-60% of frame (frame 120 = 790k px), some masks swallowed the tool,
far near-static organs masked. Multi-object/colors are pointless — MoSca only
needs ONE BINARY mask. ARCHIVED: archive/v2_motion_multiregion,
peritool_iters/v2_motion_multiregion.

---

## v3 — single binary, TOOL-CENTRIC, motion-gated  (IN PROGRESS)
Insight (user): MoSca consumes ONE binary dynamic mask; the TOOL is the source of
every interaction; final `epi = tool ∪ tissue` where the tool mask comes from the
dataset `train_masks/` (SAM2 does tissue only, tool is a NEGATIVE).
**Stage 1 (`detect_contact.py` v3):** per frame emit SINGLE-object prompts, all
NEAR THE TOOL: positives = tissue that is MOVING and within a band (~70px) of the
tool contour (catches the early gland: near-tool + moving); negatives = points ON
the tool + a few static-bg just OUTSIDE the band (bounds SAM2 growth). No regions/
IDs/colors. `active[t]` = frames with positives.
**Stage 2 (`sam2_propagate.py` v3):** SINGLE binary object. Tile the active spans
into overlapping K-frame windows, seed each window at its anchor with the near-tool
positives + tool/bg negatives, propagate within window (continuity), union.
GATE: keep CC containing a positive (drop islands) + cap growth to within Dcap of
the tool (no far flooding). Output ONE binary tissue mask; `epi = tool ∪ tissue`
(gated by --write_epi). Closest to v1 (tight, tool-centric) + v2's motion-based
early detection + real continuity.
