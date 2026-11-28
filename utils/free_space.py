"""Fit COLMAP-aligned depth once and penalise opacity in front of surfaces."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def build_aligned_priors(cameras, triangulated_dir):
    """Build one aligned depth map per training camera."""
    folder = Path(triangulated_dir)
    views = {
        Path(view.name).stem: view
        for view in read_extrinsics_binary(str(folder / "images.bin")).values()
    } # image_name -> view extrinsics
    intrinsics = read_intrinsics_binary(str(folder / "cameras.bin"))
    points = read_colmap_points(folder / "points3D.txt") # point_id -> (x, y, z)

    priors = {}
    for camera in cameras:
        view = views[camera.image_name]
        # inverse depths from the depth predictor, stored inside the camera object
        all_pred_depth = camera.depth_mono[0].detach().cpu().numpy().astype(np.float64)
        all_pred_depth /= max(all_pred_depth.max(), 1e-6)
        anchor_pred_depth, anchor_triag_depth = get_depth_pairs(camera, view, intrinsics[view.camera_id], points, all_pred_depth)
        aligned_pred_depth, valid = fit_aligned_depth(all_pred_depth, anchor_pred_depth, anchor_triag_depth, camera.image_name)

        device = camera.world_view_transform.device
        priors[camera.image_name] = (
            torch.as_tensor(aligned_pred_depth, device=device), torch.as_tensor(valid, device=device)
        )
    return priors

def read_colmap_points(path):
    """Read the world-space XYZ for each COLMAP point ID."""
    points = {}
    with path.open() as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                point_id, x, y, z, _ = line.split(maxsplit=4)
                points[int(point_id)] = (float(x), float(y), float(z))
    return points


def get_depth_pairs(camera, view, calibration, points, inv_depth):
    """Pair each tracked pixel's monocular value with its camera-space depth."""
    height, width = inv_depth.shape
    tracked = view.point3D_ids >= 0 # points that COLMAP assigned to a 3D point
    xyz = np.array([points[i] for i in view.point3D_ids[tracked]]).reshape(-1, 3) # [points, 3]
    transform = camera.world_view_transform.detach().cpu().numpy()
    z = xyz @ transform[:3, 2] + transform[3, 2]

    # scale from colmap depth map coordinates to predictor depth map coordinates
    scale = np.array([width / calibration.width, height / calibration.height])
    u, v = np.rint(view.xys[tracked] * scale - 0.5).astype(int).T # adjust for pixel center
    visible = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return inv_depth[v[visible], u[visible]], z[visible]


def fit_aligned_depth(all_pred_depth, anchor_pred_depth, anchor_triag_depth, image_name):
    """Fit the scale (a) and shift (b), then convert the depth map using the linear transform"""
    # fit a * anchor_pred_depth + b to 1 / anchor_triag_depth to find relationship between predictor and actual depths
    
    a, b = least_squares(
        lambda ab: (ab[0] * anchor_pred_depth + ab[1]) * anchor_triag_depth - 1,
        x0=[0.0, np.median(1 / anchor_triag_depth)],
        bounds=([0, -np.inf], [np.inf, np.inf]),
        loss = "soft_l1",
        f_scale=0.08,
    ).x

    aligned_anchor_depth = 1 / (a * anchor_pred_depth + b).clip(1e-6)
    fit_error = np.abs(aligned_anchor_depth - anchor_triag_depth) / anchor_triag_depth
    if np.median(fit_error) > 0.25:
        raise ValueError(f"Poor depth alignment for {image_name}")

    inverse = a * all_pred_depth + b
    aligned_depth = (1 / inverse.clip(1e-6)).astype(np.float32)
    
    low, high = np.percentile(anchor_triag_depth, [2, 98])
    kernel = np.ones((5, 5), np.uint8)
    span = (cv2.dilate(aligned_depth, kernel) - cv2.erode(aligned_depth, kernel)) / aligned_depth

    valid = (
        (inverse > 0)
        & (aligned_depth >= 0.75 * low)
        & (aligned_depth <= 1.4 * high)
        & (span < 0.12)
    )
    return aligned_depth, valid

def free_space_loss(camera, centres, opacity, priors):
    """
        For the gaussians where diff = (depth(gaussian) - depth(prior)) / depth(prior) is more than 15%, 
        Sum their losses = opacity * (diff - 15%)
        Normalize by dividing by the number of gaussians that has a valid prior in the first place
    """
    depth, valid = priors[camera.image_name]
    height, width = depth.shape
    with torch.no_grad():
        transform = camera.world_view_transform
        projection = camera.full_proj_transform
        z = centres @ transform[:3, 2] + transform[3, 2]
        clip = centres @ projection[:3] + projection[3] # [N,4] (x, y, z, w)
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6) # [N, 2] (x/w, y/w)
        u = ((ndc[:, 0] + 1) * width / 2 - 0.5).round().long()
        v = ((ndc[:, 1] + 1) * height / 2 - 0.5).round().long()
        inside = (z > camera.znear) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v = u.clamp(0, width - 1), v.clamp(0, height - 1)
        eligible = inside & valid[v, u]
        
        gap = ((depth[v, u] - z) / depth[v, u] - 0.15).clamp(0, 1)
        gap = torch.where(eligible, gap, 0.0)

    return (opacity.reshape(-1) * gap).sum() / eligible.sum().clamp_min(1)

def aligned_depth_loss(camera, render_pkg, priors):
    aligned_pred_depth, valid = priors[camera.image_name]
    alpha = render_pkg["alpha"].squeeze()
    depth = render_pkg["depth"].squeeze() / alpha.clamp_min(1e-4)

    use = valid & (alpha.detach() > 0.65)
    error = depth[use] / aligned_pred_depth[use] - 1 

    return F.smooth_l1_loss(
        error, torch.zeros_like(error), beta=0.03, reduction="sum"
    ) / use.sum().clamp_min(1)