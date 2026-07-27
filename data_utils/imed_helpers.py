import numpy as np
import torch
import torch.nn.functional as F
import os, os.path as osp
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy import ndimage


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


def build_train_overlap_mask(imed_seq, out_h, out_w):
    """Reproject Endo2L (training) pixels into the Endo1L (test) camera bounds.

    This is the mirror of metrics._build_global_imed_overlap_mask: that function
    marks, in the *test* camera's grid, which pixels are covered by training-view
    geometry (used for eval). This function instead marks, in the *training*
    camera's grid, which training pixels have geometry landing inside the test
    frame -- i.e. the region of the training image that actually determines
    eval-region quality. Used to reweight the photometric/depth loss toward the
    pixels that matter for scoring.

    Returns a boolean [out_h, out_w] array in training (Endo2L) resolution.
    """
    k_matrices = parse_imed_intrinsics(osp.join(imed_seq, "K.txt"))
    K2 = k_matrices["K2_L"]
    K1 = k_matrices["K1_L"]
    c2w_by_cam = parse_imed_poses(osp.join(imed_seq, "pose.txt"))
    c2w_cam2 = c2w_by_cam[0]
    c2w_cam1 = c2w_by_cam[1]
    w2c_cam1 = np.linalg.inv(c2w_cam1)

    depth_files = sorted(glob(osp.join(imed_seq, "endoscope2", "depthL", "*.npy")))
    assert len(depth_files) > 0, "No depth files in endoscope2/depthL"
    depth = np.load(depth_files[0]).astype(np.float32)
    h2, w2 = depth.shape

    u, v = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
    z = depth
    valid = z > 0
    x = (u - float(K2[0, 2])) * z / float(K2[0, 0])
    y = (v - float(K2[1, 2])) * z / float(K2[1, 1])

    pts_cam2 = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    uv_flat = np.stack([u, v], axis=-1).reshape(-1, 2)  # source (u,v) for every pixel
    valid_flat = valid.reshape(-1)
    pts_cam2 = pts_cam2[valid_flat]
    uv_valid = uv_flat[valid_flat]

    pts_cam2_h = np.concatenate([pts_cam2, np.ones((pts_cam2.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (c2w_cam2 @ pts_cam2_h.T).T
    pts_cam1 = (w2c_cam1 @ pts_world.T).T[:, :3]

    z1 = pts_cam1[:, 2]
    front = z1 > 1e-6
    pts_cam1 = pts_cam1[front]
    z1 = z1[front]
    uv_front = uv_valid[front]

    u1 = float(K1[0, 0]) * (pts_cam1[:, 0] / z1) + float(K1[0, 2])
    v1 = float(K1[1, 1]) * (pts_cam1[:, 1] / z1) + float(K1[1, 2])

    h1 = int(round(float(K1[1, 2]) * 2))
    w1 = int(round(float(K1[0, 2]) * 2))
    if h1 <= 0 or w1 <= 0:
        h1, w1 = h2, w2
    in_bounds = (u1 >= 0) & (u1 < w1) & (v1 >= 0) & (v1 < h1)

    uv_hit = uv_front[in_bounds]  # training-frame source pixels whose geometry lands in Endo1L

    mask = np.zeros((h2, w2), dtype=np.uint8)
    mask[uv_hit[:, 1].astype(np.int32), uv_hit[:, 0].astype(np.int32)] = 1
    # Convert sparse reprojections into a contiguous valid support region.
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_closing(mask, structure=np.ones((11, 11), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = mask.astype(np.float32)
    if (h2, w2) != (out_h, out_w):
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(out_h, out_w), mode="nearest")
        mask = mask_t.squeeze(0).squeeze(0).numpy()
    return mask.astype(bool)


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
