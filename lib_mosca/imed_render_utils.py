"""Shared iMED NVS rendering helpers, used by both the local-dev evaluator
(imed_evaluate.py, which also copies GT and computes metrics) and the Docker
submission renderer (imed_submission_render.py, which has no GT available).
Keeping the render math in one place avoids the two callers silently
diverging.
"""

import os.path as osp

import numpy as np
import torch

from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import GS_BACKEND, render


def build_K_tensor(K_np, device="cuda"):
    return torch.from_numpy(K_np).float().to(device)


def load_trained_models(logdir, device=torch.device("cuda")):
    """Load the static+dynamic Gaussian models saved by mosca_reconstruct.py."""
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
    ).to(device)
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
    ).to(device)
    s_model.eval()
    d_model.eval()
    return s_model, d_model


@torch.no_grad()
def render_frame(s_gs5, d_model, t, K, T_cw, H, W):
    """Render one frame: precomputed static Gaussians + dynamic Gaussians at time t.

    s_gs5 should be s_model() computed once by the caller (static geometry does
    not depend on t, so it's wasteful to recompute per-frame).
    Returns an [H, W, 3] float32 RGB array clipped to [0, 1].
    """
    d_gs5 = d_model(t)
    render_dict = render([s_gs5, d_gs5], H, W, K, T_cw=T_cw)
    rgb = render_dict["rgb"].permute(1, 2, 0).cpu().numpy()
    return np.clip(rgb, 0.0, 1.0)
