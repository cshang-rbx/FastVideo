# SPDX-License-Identifier: Apache-2.0
"""
Camera utility functions for LingbotWorld Plücker-ray conditioning.

This module provides standalone helpers to compute Plücker-ray camera embeddings
from camera pose files (``poses.npy``) and intrinsic files (``intrinsics.npy``).

The main entry point is :func:`compute_cam_plucker_emb`.
"""

from __future__ import annotations

import numpy as np
import torch


# =========================================================================
# Low-level helpers
# =========================================================================


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    """Interpolate camera poses (rotation via SLERP, translation via linear).

    Args:
        src_indices: Source frame indices, shape ``(N,)``.
        src_rot_mat: Rotation matrices at source indices, ``(N, 3, 3)``.
        src_trans_vec: Translation vectors at source indices, ``(N, 3)``.
        tgt_indices: Target frame indices to interpolate to, ``(M,)``.

    Returns:
        Interpolated 4×4 poses, shape ``(M, 4, 4)`` as a float tensor.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)

    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_mat = slerp_func_rot(tgt_indices).as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Batch SE(3) inverse: ``T`` has shape ``(B, 4, 4)``."""
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, t)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).expand(
        T.shape[0], -1, -1).clone()
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    """Compute relative camera poses w.r.t. the first frame.

    Args:
        c2ws_mat: Camera-to-world matrices, ``(F, 4, 4)``.
        framewise: If ``True``, compute frame-to-frame relative poses.
        normalize_trans: Normalise translations to unit max-norm.

    Returns:
        Relative poses, ``(F, 4, 4)``.
    """
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device,
                                  dtype=c2ws_mat.dtype)
    if framewise:
        rel_fw = torch.bmm(SE3_inverse(relative_poses[:-1]),
                           relative_poses[1:])
        relative_poses[1:] = rel_fw
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute Plücker ray embeddings for each pixel.

    Args:
        c2ws_mat: Camera-to-world matrices, ``(F, 4, 4)``.
        Ks: Intrinsics ``[fx, fy, cx, cy]``, ``(F, 4)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Plücker embeddings, ``(F, H, W, 6)``.
    """
    n_frames = c2ws_mat.shape[0]
    device, dtype = c2ws_mat.device, c2ws_mat.dtype

    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) + 0.5
    grid_xy = grid_xy.unsqueeze(0).expand(n_frames, -1, -1)  # (F, H*W, 2)

    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # each (F, 1)

    i = grid_xy[..., 0]  # (F, H*W)
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # (F, H*W, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # (F, H*W, 3)
    rays_o = c2ws_mat[:, :3, 3].unsqueeze(1).expand_as(rays_d)  # (F, H*W, 3)

    plucker = torch.cat([rays_o, rays_d], dim=-1)  # (F, H*W, 6)
    return plucker.reshape(n_frames, height, width, 6)


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    """Transform intrinsics for a resize-then-centre-crop pipeline.

    Args:
        Ks: Intrinsics ``[fx, fy, cx, cy]``, ``(F, 4)``.
        height_org / width_org: Original image resolution.
        height_resize / width_resize: Resolution after resize.
        height_final / width_final: Resolution after centre crop.

    Returns:
        Transformed intrinsics, same shape as *Ks*.
    """
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_r = fx * scale_x
    fy_r = fy * scale_y
    cx_r = cx * scale_x
    cy_r = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    Ks_out = torch.zeros_like(Ks)
    Ks_out[:, 0:1] = fx_r
    Ks_out[:, 1:2] = fy_r
    Ks_out[:, 2:3] = cx_r - crop_offset_x
    Ks_out[:, 3:4] = cy_r - crop_offset_y
    return Ks_out


# =========================================================================
# Main entry point
# =========================================================================


def compute_cam_plucker_emb(
    poses_path: str,
    intrinsics_path: str,
    num_frames: int,
    height: int,
    width: int,
    vae_stride: tuple[int, int, int],
    patch_size: tuple[int, int, int] = (1, 2, 2),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    original_height: int = 480,
    original_width: int = 832,
) -> torch.Tensor:
    """Compute Plücker-ray camera embeddings from pose / intrinsic files.

    This is a **standalone** helper that mirrors the camera preparation logic in
    the original ``lingbot-world`` codebase.  It loads ``poses.npy`` and
    ``intrinsics.npy``, down-samples to the latent temporal resolution,
    computes framewise-relative Plücker rays, and reshapes them into the
    ``[B, 6, F_lat, H_lat, W_lat]`` tensor expected by the
    :class:`~fastvideo.models.dits.lingbotworld.LingbotWorldTransformer3DModel`.

    Args:
        poses_path: Path to ``poses.npy`` — shape ``(F, 4, 4)`` (OpenCV c2w).
        intrinsics_path: Path to ``intrinsics.npy`` — shape ``(F, 4)``
            storing ``[fx, fy, cx, cy]`` (for the original resolution).
        num_frames: Number of *video* frames (before VAE temporal stride).
        height: Target video height in pixels.
        width: Target video width in pixels.
        vae_stride: ``(temporal, spatial_h, spatial_w)`` VAE downsampling.
        patch_size: Transformer patch size ``(pt, ph, pw)``.
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        original_height: Height the intrinsics were calibrated for.
        original_width: Width the intrinsics were calibrated for.

    Returns:
        ``cam_plucker_emb`` — shape ``[1, C, F_lat, H_lat, W_lat]`` where
        ``C = 6 * stride_h * stride_w``.
    """
    # --- Load & truncate poses to 4n+1 boundary -------------------------
    c2ws = np.load(poses_path)  # (F_file, 4, 4)
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    num_frames = min(num_frames, len_c2ws)
    c2ws = c2ws[:num_frames]

    # --- Load & transform intrinsics ------------------------------------
    Ks = torch.from_numpy(np.load(intrinsics_path)).float()  # (F_file, 4)
    Ks = get_Ks_transformed(
        Ks,
        height_org=original_height,
        width_org=original_width,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
    )
    Ks = Ks[0]  # (4,) — assume intrinsics are the same for all frames

    # --- Temporal down-sample: video frames → latent frames -------------
    lat_f = (num_frames - 1) // vae_stride[0] + 1
    lat_h = height // vae_stride[1]
    lat_w = width // vae_stride[2]

    len_c2ws = len(c2ws)
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, lat_f),
    )

    # --- Relative poses (frame-wise) ------------------------------------
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks_expanded = Ks.unsqueeze(0).expand(len(c2ws_infer), -1)  # (lat_f, 4)

    # --- Plücker embeddings at full pixel resolution --------------------
    c2ws_infer = c2ws_infer.to(device)
    Ks_expanded = Ks_expanded.to(device)
    plucker = get_plucker_embeddings(
        c2ws_infer, Ks_expanded, height, width)  # (lat_f, H, W, 6)

    # --- Patchify to latent spatial resolution --------------------------
    # plucker: (F_lat, H, W, 6) → (F_lat, 6, H, W) for reshape
    plucker = plucker.permute(0, 3, 1, 2)  # (F_lat, 6, H, W)
    # group spatial dims into (lat_h, stride_h, lat_w, stride_w)
    stride_h = vae_stride[1]
    stride_w = vae_stride[2]
    plucker = plucker.reshape(
        lat_f, 6, lat_h, stride_h, lat_w, stride_w,
    )
    # → (F_lat, lat_h, lat_w, 6 * stride_h * stride_w)
    plucker = plucker.permute(0, 2, 4, 1, 3, 5).reshape(
        lat_f * lat_h * lat_w, 6 * stride_h * stride_w,
    )
    # reshape to [1, C, F_lat, H_lat, W_lat] via intermediate form
    plucker = plucker.unsqueeze(0)  # (1, F_lat*H_lat*W_lat, C)
    plucker = plucker.reshape(
        1, lat_f, lat_h, lat_w, 6 * stride_h * stride_w,
    ).permute(0, 4, 1, 2, 3)  # (1, C, F_lat, H_lat, W_lat)

    return plucker.to(dtype=dtype, device=device)
