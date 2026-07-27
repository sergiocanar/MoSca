# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoSca is a 4D scene reconstruction system for monocular videos using dynamic Gaussian splatting. It ships two submodules:
- **MoCa** (Moving Monocular Camera): standalone camera pose estimation via tracklet-based bundle adjustment
- **MoSca**: full pipeline that adds a 4D motion scaffold and dynamic/static 3D Gaussian splatting on top of MoCa

## Environment Setup

```bash
bash install.sh   # creates conda env 'mosca', PyTorch 2.1.0, CUDA 11.8
conda activate mosca
```

Weights (RAFT, SpaTracker, TAPNet/BootsTAPIR) must be placed manually:
```
weights/
├── raft_models/raft-things.pth
├── spaT_final.pth
└── tapnet/bootstapir_checkpoint_v2.pt
```

## Running the Pipeline

Each scene is a workspace directory (`--ws`) containing an `images/` subfolder.

**Full MoSca pipeline:**
```bash
CUDA_VISIBLE_DEVICES=0 python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck
CUDA_VISIBLE_DEVICES=0 python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck
```

**MoCa only (camera pose estimation):**
```bash
python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck --skip_dynamic_resample
python lite_moca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck
```

**Multi-GPU benchmark reproduction:**
```bash
bash reproduce.sh [GPU_ID] [TOTAL_NUM_GPUS]
```

**CLI config overrides** — unknown args are merged into the YAML config as dotlist:
```bash
python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck --dep_mode=uni --tap_mode=bootstapir
```

## Pipeline Stages (mosca_reconstruct.py)

1. **`static_reconstruct`** (from `lite_moca_reconstruct.py`): runs bundle adjustment (MoCa) to solve camera poses and depth scale — outputs `logs/.../bundle/bundle.pth` and `bundle_cams.pth`
2. **`photometric_warmup`** (optional, triggered by `photo_static_warm_steps > 0`): pre-optimizes static background GS before joint optimization
3. **`scaffold_reconstruct`**: builds the 4D motion scaffold (MoSca) from dynamic tracklet curves via ARAP optimization — outputs `logs/.../mosca/mosca.pth`
4. **`photometric_reconstruct`**: jointly optimizes static + dynamic GS against RGB, depth, and optional track/flow losses — outputs `photometric_*_model_*.pth` and `photometric_cam.pth`

## Code Architecture

```
lib_prior/          # 2D foundational model wrappers (run during precompute)
  moca_processor.py     # orchestrates depth, TAP, flow, seg inference
  prior_loading.py      # Saved2D: fluent loader for precomputed priors
  depth_models/         # DepthCrafter, Metric3D, UniDepth, ZoeDepth
  optical_flow/         # RAFT wrapper
  tracking/             # BootsTAPIR, CoTracker, SpaTracker wrappers
  seg/                  # SegFormer wrapper

lib_moca/           # Camera BA and track analysis
  moca.py               # moca_solve(): tracklet-based bundle adjustment entry point
  bundle.py             # static BA solver
  camera.py             # MonocularCameras: SE3 camera model with learnable focal
  epi_helpers.py        # epipolar error analysis for static/dynamic track ID

lib_mosca/          # 4D Gaussian splatting
  mosca.py              # MoSca: 4D motion scaffold (nn.Module), ARAP, multi-level
  dynamic_gs.py         # DynSCFGaussian: dynamic Gaussians skinned to scaffold nodes
  static_gs.py          # StaticGaussian: background 3DGS
  dynamic_solver.py     # get_dynamic_curves(), geometry_scf_init()
  photo_recon.py        # DynReconstructionSolver: joint photometric optimization
  scaffold_utils/       # dual-quaternion helpers, viz

lib_render/         # Gaussian rasterization backends
  render_helper.py      # backend dispatch via GS_BACKEND env var
  diff-gaussian-rasterization-alphadep-add3/  # default backend (native_add3)
  diff-gaussian-rasterization-alphadep/       # native backend
  gof-diff-gaussian-rasterization/            # GOF backend

profile/            # YAML configs per dataset
  demo/  iphone/  nvidia/  sintel/  tum/

data_utils/         # Dataset-specific GT pose loaders (iPhone, Nvidia)
eval_utils/         # Benchmark evaluation (DyCheck, Nvidia, TUM, Sintel)
```

## Key Design Patterns

**Saved2D fluent API** (`lib_prior/prior_loading.py`): precomputed priors are loaded via chained calls:
```python
s2d = Saved2D(ws).load_epi().load_dep(depth_dir, th).normalize_depth().load_track(tap_pattern).load_vos().to(device)
```

**GS backend selection**: set `GS_BACKEND` env variable to `native_add3` (default), `native`, or `gof` before running.

**Config merging**: both `mosca_precompute.py` and `mosca_reconstruct.py` load a base YAML (`--cfg`) then merge any extra CLI flags as OmegaConf dotlist. Use this to override individual parameters without editing YAMLs.

**Output structure**: each run creates a timestamped log dir under `<ws>/logs/<exp_name>_<backend>_<timestamp>/`. Source code is backed up there automatically.

**Static vs dynamic track identification**: epipolar error (`epi_th`) on TAP tracks separates static (used for BA) from dynamic (used for scaffold). The threshold and `dyn_id_cnt` count are key parameters to tune per scene.

## Dataset Modes

Configured via `mode` in the fit YAML: `iphone` (DyCheck), `nvidia`, `sintel`, `tum`, `wild`. Mode controls which GT pose loader, evaluation metric, and test-time-optimization settings are used.

## Evaluation

```bash
# Run separately on an existing log dir
python mosca_evaluate.py --ws <scene_dir> --cfg <fit.yaml> --logdir <logdir>
# Collect metrics across all scenes
jupyter notebook collect_metrics.ipynb
```

## iMED Challenge Adaptation

A separate pipeline adapts MoSca for the iMED MICCAI 2026 Challenge (EndoVis 2026, Task 2 — Deformable NVS): dual static-endoscope surgical video, training on Endoscope 2-L and testing novel-view synthesis on the held-out Endoscope 1-L viewpoint. This mode reuses the core MoSca solver but swaps in known camera poses, sensor depth, and tool masks instead of estimating them.

**Per-sequence workflow** (`scripts/imed_step{1..5}_*.sh <sequence_name> [gpu_id]`, or `scripts/imed_run_all.sh [gpu_id]` for all sequences):
```bash
python imed_prepare_workspace.py --imed_seq data/iMED_NVS/<seq> --ws workspaces/<seq>   # step 1: build workspace
python mosca_precompute.py --cfg profile/imed/imed_prep.yaml --ws workspaces/<seq>       # step 2: TAP tracking only (depth/flow pre-provided)
python mosca_reconstruct.py --cfg profile/imed/imed_fit.yaml --ws workspaces/<seq>       # step 3: frozen-camera scaffold + photometric GS
python imed_evaluate.py --ws workspaces/<seq> --logdir workspaces/<seq>/logs/<ts>        # step 4: render Endo1L, compute PSNR/SSIM
bash scripts/imed_step5_video.sh <seq>                                                    # step 5: side-by-side compare.webm
```

- `imed_prepare_workspace.py` converts a raw `data/iMED_NVS/<seq>/` directory (per-cam `L/`, `depthL/`, `toolL/`, plus `K.txt`/`pose.txt`) into a standard MoSca workspace: `images/`, `sensor_depth/`, pseudo-epi from tool masks (`epi/error/`), `test_images/`, `test_masks/`, and `imed_meta.npz` (intrinsics, poses, depth scale). Its `--inference` flag skips reading `endoscope1/` entirely (reusing the training resolution/frame count instead) — used by the Docker submission path below, since hidden test sequences don't ship `endoscope1/` at all.
- `data_utils/imed_helpers.py` parses `K.txt`/`pose.txt`. Convention: `cam_id=0` = Endoscope 2-L (training, identity pose = world origin), `cam_id=1` = Endoscope 1-L (test) in Endo2L world, units mm.
- `profile/imed/imed_prep.yaml` and `profile/imed/imed_fit.yaml` configure `mode=imed` with `dep_mode=sensor`, `flow_mode=none`, and frozen cameras.
- `lite_moca_reconstruct.py`'s `load_gt_cam()` has an `imed` branch to load the known Endo2L/Endo1L poses instead of running BA.
- `lib_mosca/imed_render_utils.py` holds the shared Endo1L render loop (`load_trained_models()`, `render_frame()`) used by both `imed_evaluate.py` (dev, with GT/metrics) and `imed_submission_render.py` (Docker submission, no GT) — keeps the two from silently diverging.
- Baselines for comparison live under `baseline/imed/<seq>/`; `scripts/imed_baseline_metrics.sh` and `scripts/imed_collect_metrics.sh` aggregate results.

### Docker Submission

Root-level `Dockerfile` + `imed_nvs_submission.py` package the same pipeline (prepare → precompute → reconstruct → render) for the challenge evaluator, structured after `Endo-4DGS/Dockerfile` + `Endo-4DGS/imed_nvs_baseline.py`. Key differences from the dev pipeline: `imed_prepare_workspace.py` runs with `--inference` (no `endoscope1/` read), and `imed_submission_render.py` writes bare sequential `renders/00000.png...` with no GT copy or metrics (there is no GT at submission time). The `native_add3` GS backend (the one actually dispatched at render time) and the plain `native` backend (pulled in transitively at import time by viz-only helpers like `lib_moca/viz_helper.py`, even though never rendered with) are both built into the image; only the `gof` backend is skipped (no code path imports it, dispatched or not). `pytorch3d`/`torch_geometric` (required by `lib_mosca/dynamic_gs.py`) and the `bootstapir` TAP checkpoint are also built in — `xformers`, `cupy`, `mmcv`, RAFT, and SpaTracker weights are all unused on this code path (`dep_mode=sensor`, `tap_mode=bootstapir`, `flow_mode=none`) and excluded.

```bash
docker build -t mosca-nvs-submission:dev .
./scripts/local_test.sh mosca-nvs-submission:dev /path/to/iMED_NVS/train ./my_test_output
```

`scripts/local_test.sh` and `scripts/check_outputs.py` are copied unchanged from `Endo-4DGS/imednvs_submission/scripts/` (generic image/input/output driver + output-shape checker). Tag/push to Synapse the same way as documented in `Endo-4DGS/README.md`'s Docker section.
