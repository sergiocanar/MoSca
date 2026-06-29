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
