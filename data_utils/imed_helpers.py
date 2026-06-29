import numpy as np
import torch
import os, os.path as osp
from scipy.spatial.transform import Rotation as R


def parse_imed_intrinsics(k_path):
    """Parse K.txt → returns dict of K matrices keyed by name (K1_L, K1_R, K2_L, K2_R)."""
    with open(k_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    matrices = {}
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith("#"):
            header = line[1:].strip()
            if not header.startswith("K"):
                i += 1
                continue
            key = header.split()[0]
            rows = []
            for j in range(1, 4):
                vals = [float(v) for v in raw_lines[i + j].split()]
                rows.append(vals)
            matrices[key] = np.array(rows, dtype=np.float32)
            i += 4
            continue
        i += 1
    return matrices


def parse_imed_poses(pose_path):
    """Parse pose.txt.

    Returns dict: {cam_id: c2w (4x4 float32)}.

    Convention (from metrics.py / _build_global_imed_overlap_mask):
      cam_id=0  →  Endoscope 2-L (training camera), pose = identity,
                   i.e. the training world origin is at Endoscope 2-L.
      cam_id=1  →  Endoscope 1-L (test camera), pose in Endoscope 2-L world (mm).
    """
    with open(pose_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2, f"Expected 2 rows, got {len(lines)}"
    c2w_by_cam = {}
    for line in lines:
        parts = line.split()
        assert len(parts) == 8
        cam_id = int(parts[0])
        t = np.array([float(v) for v in parts[1:4]], dtype=np.float32)
        q = np.array([float(v) for v in parts[4:8]], dtype=np.float32)  # xyzw
        rot = R.from_quat(q).as_matrix().astype(np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = t
        c2w_by_cam[cam_id] = c2w
    assert 0 in c2w_by_cam and 1 in c2w_by_cam
    return c2w_by_cam


def k_to_fov_and_cxcy(K, H, W):
    """Convert 3x3 K matrix to (fov_x_deg, fov_y_deg, cx_ratio, cy_ratio).

    MonocularCameras uses: rel_focal = 1/tan(fov/2), fx = rel_focal * L / 2.
    Here we derive the fov that recovers K[0,0] and K[1,1] exactly.
    """
    L = min(H, W)
    fov_x = 2.0 * np.degrees(np.arctan(L / (2.0 * float(K[0, 0]))))
    fov_y = 2.0 * np.degrees(np.arctan(L / (2.0 * float(K[1, 1]))))
    cx_ratio = float(K[0, 2]) / W
    cy_ratio = float(K[1, 2]) / H
    return fov_x, fov_y, cx_ratio, cy_ratio


def load_imed_gt_poses(ws_dir):
    """Return the 8-tuple expected by load_gt_cam() in lite_moca_reconstruct.py.

    Training camera: all-identity T×4×4 (Endoscope 2-L is fixed at origin).
    Test camera: constant Endoscope 1-L pose in Endoscope 2-L world (depth-normalized).
    """
    meta = np.load(osp.join(ws_dir, "imed_meta.npz"), allow_pickle=True)

    T = int(meta["T"])
    K2L = meta["K2L"]
    K1L = meta["K1L"]
    c2w_test = meta["c2w_test"]          # Endo1L c2w in Endo2L world, mm
    H = int(meta["H"])
    W = int(meta["W"])
    H1 = int(meta["H1"])
    W1 = int(meta["W1"])
    frame_names_test = list(meta["frame_names_test"])

    # Training camera: T copies of identity
    gt_training_cam_T_wi = torch.eye(4).unsqueeze(0).expand(T, -1, -1).clone().float()

    # Training FOV + cx/cy from K2L
    fov_x2, fov_y2, cx2, cy2 = k_to_fov_and_cxcy(K2L, H, W)
    gt_training_fov = float(fov_x2)
    gt_training_cxcy_ratio = [[float(cx2), float(cy2)]]  # [0] is taken by caller

    # Test camera: T copies of the Endo1L pose
    # World is in mm (dep_median=-1 keeps depth unnormalized), so c2w_test in mm applies directly.
    c2w_test_t = torch.from_numpy(c2w_test).float().unsqueeze(0).expand(T, -1, -1).clone()
    gt_testing_cam_T_wi_list = [c2w_test_t]
    gt_testing_tids_list = [list(range(T))]
    gt_testing_fns_list = [frame_names_test]

    fov_x1, fov_y1, cx1, cy1 = k_to_fov_and_cxcy(K1L, H1, W1)
    gt_testing_fov_list = [float(fov_x1)]
    gt_testing_cxcy_ratio_list = [[float(cx1), float(cy1)]]

    return (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    )
