FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOSCA_REPO=/workspace/MoSca \
    TORCH_HOME=/opt/torch_cache \
    XDG_CACHE_HOME=/opt/torch_cache \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential ninja-build git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 libx11-6 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade \
        pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Pinned to the exact combo MoSca/install.sh already uses (python 3.10 / torch
# 2.1.0 / cuda 11.8) -- this is also the exact combo pytorch3d ships a
# prebuilt wheel for below, so it's not just a version-parity choice.
RUN python -m pip install \
        torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# pytorch3d: prebuilt wheel for py310/cu118/torch2.1.0, avoiding a from-source
# build (~15-30 min). If this combo is ever pulled from the index, fall back
# to a source build (the devel base image ships nvcc for exactly this case):
#   python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN python -m pip install fvcore iopath \
    && python -m pip install --no-index --no-cache-dir pytorch3d \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# torch-geometric family: prebuilt wheels matched to the pinned torch/cuda
# build, no compilation needed. Required by lib_mosca/dynamic_gs.py.
RUN python -m pip install \
        pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

WORKDIR /workspace/MoSca

# Copied separately so this layer only invalidates when requirements.txt
# changes, not on every source edit.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    # requirements.txt pins GUI opencv builds; force the headless variant for
    # a container with no display server (same reasoning as the Endo-4DGS
    # baseline Dockerfile). --no-deps is required here: without it,
    # --force-reinstall also reinstalls opencv's dependencies, and
    # opencv-python-headless==4.6.0.66's numpy constraint predates numpy 2.0,
    # so pip pulls in today's numpy 2.x and clobbers the numpy==1.26.4 pin
    # from requirements.txt above -- breaking the ABI opencv's own compiled
    # .so was built against.
    && python -m pip install --force-reinstall --no-deps \
        opencv-python-headless==4.6.0.66 opencv-contrib-python-headless==4.6.0.66

# lpips/pandas/evo: not in requirements.txt (they normally come from
# jax_requirements.txt, which also pulls in jax/flax/dm-haiku -- a heavy
# stack that's genuinely unused on the iMED path). But eval_utils/*.py
# (see .dockerignore note) import lpips/pandas/evo unconditionally at module
# load time, transitively via mosca_reconstruct.py -> mosca_evaluate.py, so
# these three are needed even though mode=imed never exercises that code.
RUN python -m pip install lpips==0.1.4 pandas==2.3.3 evo==1.36.5

COPY . /workspace/MoSca

# The default GS_BACKEND (native_add3, see lib_render/render_helper.py) is the
# only one dispatched at render time -- but several viz-only helpers
# (lib_moca/viz_helper.py, lib_mosca/photo_recon_viz_utils.py,
# lib_mosca/scaffold_utils/viz_helper.py, viz_utils.py) unconditionally
# `from lib_render.gauspl_renderer_native import render_cam_pcl` at import
# time regardless of GS_BACKEND, which needs the plain "native" extension
# (package name diff_gaussian_rasterization) built too, even though it's
# never actually invoked for rendering in the iMED pipeline. The "gof"
# variant has no such transitive import anywhere, so it's still skipped.
RUN python -m pip install lib_render/simple-knn \
    && python -m pip install lib_render/diff-gaussian-rasterization-alphadep \
    && python -m pip install lib_render/diff-gaussian-rasterization-alphadep-add3

RUN chmod +x /workspace/MoSca/imed_nvs_submission.py

ENTRYPOINT ["python", "/workspace/MoSca/imed_nvs_submission.py"]
CMD ["run-dataset", "--data-root", "/input", "--output-root", "/output"]
