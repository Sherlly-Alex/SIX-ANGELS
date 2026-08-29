#!/usr/bin/env python3
"""Official RGB-D helpers for material sorting (no monocular depth net).

Uses Server-published aligned depth:
  /head_camera/aligned_depth_to_color/image_raw  (mono16, millimeters)

Capabilities
------------
1. Robust depth sampling (center patch / bbox interior) + validity gate
2. Pixel -> camera -> world back-projection
3. Shelf layer ID from world Z (boards at 0.403 / 0.732 / 1.061 m)
4. "Left of white bar" place proposal from a packaging_box detection
5. Coarse local height map (camera frustum -> world XY bins) for later nav
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np

# Competition shelf boards (world Z, meters) — L1 / L2 / L3 from bottom.
SHELF_LAYER_Z = {1: 0.403, 2: 0.732, 3: 1.061}
BOX_HALF_Z = 0.095
PACKAGING_HALF_Z = 0.117
PACKAGING_LEFT_Y_OFFSET = 0.238
SHELF_PLACE_X = -2.68

DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.5
DEFAULT_DEPTH_SCALE = 0.001  # uint16 millimeters -> meters


@dataclass
class DepthSample:
    depth_m: float
    valid_ratio: float
    n_valid: int
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _as_depth_f32(depth_img) -> np.ndarray:
    return np.asarray(depth_img, dtype=np.float32)


def sample_depth_patch_m(
    depth_img,
    u,
    v,
    radius: int = 4,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    min_valid_ratio: float = 0.25,
) -> DepthSample:
    """Median depth in a square patch around (u, v). Rejects GS holes / outliers."""
    depth = _as_depth_f32(depth_img)
    h, w = depth.shape[:2]
    u, v = int(u), int(v)
    if not (0 <= u < w and 0 <= v < h):
        return DepthSample(0.0, 0.0, 0, False, "oob")

    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)
    patch = depth[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    n_total = int(patch.size)
    if n_total <= 0:
        return DepthSample(0.0, 0.0, 0, False, "empty")

    raw = patch[finite & (patch > 0)]
    if raw.size == 0:
        return DepthSample(0.0, 0.0, 0, False, "all_zero")

    meters = raw * float(depth_scale)
    in_range = (meters >= float(depth_min)) & (meters <= float(depth_max))
    valid = meters[in_range]
    ratio = float(valid.size) / float(n_total)
    if valid.size == 0:
        return DepthSample(0.0, ratio, 0, False, "out_of_range")
    if ratio < float(min_valid_ratio):
        return DepthSample(float(np.median(valid)), ratio, int(valid.size), False, "low_valid_ratio")
    return DepthSample(float(np.median(valid)), ratio, int(valid.size), True, "ok")


def sample_depth_in_bbox_m(
    depth_img,
    x0,
    y0,
    x1,
    y1,
    inner_frac: float = 0.5,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    min_valid_ratio: float = 0.2,
) -> DepthSample:
    """Median depth inside the central `inner_frac` of a detection bbox.

    More stable than a tiny center patch when GS depth has holes at the silhouette.
    """
    depth = _as_depth_f32(depth_img)
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return DepthSample(0.0, 0.0, 0, False, "bad_bbox")

    bw = x1 - x0
    bh = y1 - y0
    mx = (1.0 - float(inner_frac)) * 0.5
    my = (1.0 - float(inner_frac)) * 0.5
    ix0 = x0 + int(bw * mx)
    ix1 = x1 - int(bw * mx)
    iy0 = y0 + int(bh * my)
    iy1 = y1 - int(bh * my)
    if ix1 <= ix0 or iy1 <= iy0:
        return sample_depth_patch_m(
            depth_img,
            (x0 + x1) // 2,
            (y0 + y1) // 2,
            radius=6,
            depth_scale=depth_scale,
            depth_min=depth_min,
            depth_max=depth_max,
            min_valid_ratio=min_valid_ratio,
        )

    patch = depth[iy0:iy1, ix0:ix1]
    finite = np.isfinite(patch)
    n_total = int(patch.size)
    if n_total <= 0:
        return DepthSample(0.0, 0.0, 0, False, "empty")

    raw = patch[finite & (patch > 0)]
    if raw.size == 0:
        return DepthSample(0.0, 0.0, 0, False, "empty")

    meters = raw * float(depth_scale)
    in_range = (meters >= float(depth_min)) & (meters <= float(depth_max))
    valid = meters[in_range]
    ratio = float(valid.size) / float(n_total)
    if valid.size == 0:
        return DepthSample(0.0, ratio, 0, False, "out_of_range")
    if ratio < float(min_valid_ratio):
        return DepthSample(float(np.median(valid)), ratio, int(valid.size), False, "low_valid_ratio")
    return DepthSample(float(np.median(valid)), ratio, int(valid.size), True, "ok")


def pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    k_matrix,
    use_pixel_center: bool = True,
) -> np.ndarray:
    if not np.isfinite(depth_m) or float(depth_m) <= 0.0:
        raise ValueError(f"invalid depth: {depth_m}")
    k = np.asarray(k_matrix, dtype=float)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    uu = float(u) + (0.5 if use_pixel_center else 0.0)
    vv = float(v) + (0.5 if use_pixel_center else 0.0)
    z = float(depth_m)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.array([x, y, z], dtype=float)


def transform_point(tmat, point_xyz: Iterable[float]) -> np.ndarray:
    t = np.asarray(tmat, dtype=float)
    p = np.asarray(list(point_xyz), dtype=float)
    homo = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    out = t @ homo
    return out[:3].copy()


def pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    k_matrix,
    t_cam_world: np.ndarray,
    use_pixel_center: bool = True,
) -> np.ndarray:
    p_cam = pixel_to_camera(u, v, depth_m, k_matrix, use_pixel_center=use_pixel_center)
    return transform_point(t_cam_world, p_cam)


def classify_shelf_layer(
    z_world: float,
    object_half_z: float = BOX_HALF_Z,
    tol: float = 0.12,
) -> Optional[int]:
    """Map a world Z to shelf layer 1/2/3, or None if not near any board."""
    board_z = float(z_world) - float(object_half_z)
    best_layer = None
    best_err = None
    for layer, z in SHELF_LAYER_Z.items():
        err = abs(board_z - float(z))
        if best_err is None or err < best_err:
            best_err = err
            best_layer = int(layer)
    if best_err is None or best_err > float(tol):
        return None
    return best_layer


def object_half_z_for_class(class_name: str) -> float:
    if class_name in ("packaging_box", "material_box"):
        if class_name == "packaging_box":
            return float(PACKAGING_HALF_Z)
        return float(BOX_HALF_Z)
    return 0.09


def place_left_of_packaging(
    packaging_world: Iterable[float],
    place_x: float = SHELF_PLACE_X,
    y_offset: float = PACKAGING_LEFT_Y_OFFSET,
    object_half_z: float = BOX_HALF_Z,
) -> np.ndarray:
    """Propose task-3 place pose: left of white bar, same shelf layer height."""
    p = np.asarray(list(packaging_world), dtype=float)
    layer = classify_shelf_layer(float(p[2]), object_half_z=PACKAGING_HALF_Z, tol=0.2)
    if layer is None:
        z = float(p[2])
    else:
        z = float(SHELF_LAYER_Z[layer]) + float(object_half_z)
    return np.array([float(place_x), float(p[1]) - float(y_offset), z], dtype=float)


def empty_shelf_layers(occupied_layers: Iterable[int]) -> list:
    occ = {int(x) for x in occupied_layers}
    return [layer for layer in (1, 2, 3) if layer not in occ]


def frame_depth_quality(
    depth_img,
    *,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    stride: int = 4,
) -> dict:
    """Cheap whole-frame depth validity stats for map gating."""
    depth = _as_depth_f32(depth_img)
    if depth.ndim != 2 or depth.size == 0:
        return {"n_sampled": 0, "n_valid": 0, "valid_ratio": 0.0, "ok": False}

    step = max(1, int(stride))
    sampled = depth[::step, ::step]
    finite = np.isfinite(sampled) & (sampled > 0)
    n_sampled = int(sampled.size)
    if n_sampled == 0:
        return {"n_sampled": 0, "n_valid": 0, "valid_ratio": 0.0, "ok": False}

    meters = sampled * float(depth_scale)
    n_valid = int(np.sum(finite & (meters >= float(depth_min)) & (meters <= float(depth_max))))
    ratio = float(n_valid) / float(n_sampled)
    return {
        "n_sampled": n_sampled,
        "n_valid": n_valid,
        "valid_ratio": ratio,
        "ok": ratio >= 0.05,
    }


def depth_frame_to_world_points(
    depth_img,
    k_matrix,
    t_cam_world: np.ndarray,
    *,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    stride: int = 4,
) -> np.ndarray:
    """Back-project a strided depth frame to world XYZ points (N, 3)."""
    depth = _as_depth_f32(depth_img)
    h, w = depth.shape[:2]
    k = np.asarray(k_matrix, dtype=float)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx == 0.0 or fy == 0.0:
        return np.zeros((0, 3), dtype=np.float64)

    step = max(1, int(stride))
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)
    d_raw = depth[vv, uu]
    ok = np.isfinite(d_raw) & (d_raw > 0)
    d_m = d_raw * float(depth_scale)
    ok &= (d_m >= float(depth_min)) & (d_m <= float(depth_max))
    if not np.any(ok):
        return np.zeros((0, 3), dtype=np.float64)

    uu = uu[ok].astype(np.float64) + 0.5
    vv = vv[ok].astype(np.float64) + 0.5
    d_m = d_m[ok].astype(np.float64)
    x_c = (uu - cx) * d_m / fx
    y_c = (vv - cy) * d_m / fy
    z_c = d_m
    ones = np.ones_like(z_c)
    pts_c = np.stack([x_c, y_c, z_c, ones], axis=0)
    t = np.asarray(t_cam_world, dtype=float)
    pts_w = (t @ pts_c)[:3, :].T
    return np.ascontiguousarray(pts_w.copy())


def build_local_height_map(
    depth_img,
    k_matrix,
    t_cam_world: np.ndarray,
    x_range=(-3.2, 0.4),
    y_range=(0.2, 2.8),
    resolution: float = 0.05,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    stride: int = 4,
) -> dict:
    """Coarse max-Z height map in world XY from one RGB-D frame.

    Returns dict with grid (H,W), origin_xy, resolution, hit counts, and
    frame quality. Intended for adaptive standoff / local obstacle checks —
    not full SLAM.
    """
    xs = np.arange(x_range[0], x_range[1], resolution)
    ys = np.arange(y_range[0], y_range[1], resolution)
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    counts = np.zeros_like(grid, dtype=np.int32)
    quality = frame_depth_quality(
        depth_img,
        depth_scale=depth_scale,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
    )
    pts_w = depth_frame_to_world_points(
        depth_img,
        k_matrix,
        t_cam_world,
        depth_scale=depth_scale,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
    )
    if pts_w.size == 0:
        return {
            "grid": grid,
            "counts": counts,
            "origin_xy": (float(x_range[0]), float(y_range[0])),
            "resolution": float(resolution),
            "n_points": 0,
            "quality": quality,
        }

    ix = np.floor((pts_w[:, 0] - float(x_range[0])) / float(resolution)).astype(int)
    iy = np.floor((pts_w[:, 1] - float(y_range[0])) / float(resolution)).astype(int)
    inside = (
        (ix >= 0)
        & (iy >= 0)
        & (ix < grid.shape[1])
        & (iy < grid.shape[0])
        & np.isfinite(pts_w[:, 2])
    )
    zw = pts_w[inside, 2]
    for i, j, z in zip(iy[inside], ix[inside], zw):
        prev = grid[i, j]
        if not np.isfinite(prev) or float(z) > float(prev):
            grid[i, j] = float(z)
        counts[i, j] += 1

    return {
        "grid": grid,
        "counts": counts,
        "origin_xy": (float(x_range[0]), float(y_range[0])),
        "resolution": float(resolution),
        "n_points": int(np.sum(inside)),
        "quality": quality,
    }


def localize_detection(
    depth_img,
    k_matrix,
    t_cam_world: np.ndarray,
    u: int,
    v: int,
    w: int,
    h: int,
    class_name: str,
    prefer_bbox: bool = True,
) -> Optional[dict]:
    """Full localize one 2D detection -> world XYZ + shelf layer + depth QA."""
    if prefer_bbox:
        sample = sample_depth_in_bbox_m(
            depth_img,
            int(u - w // 2),
            int(v - h // 2),
            int(u + w // 2),
            int(v + h // 2),
        )
    else:
        sample = sample_depth_patch_m(depth_img, u, v, radius=6)
    if not sample.ok:
        return None

    world = pixel_to_world(float(u), float(v), sample.depth_m, k_matrix, t_cam_world)
    half_z = object_half_z_for_class(class_name)
    layer = classify_shelf_layer(float(world[2]), object_half_z=half_z)
    out = {
        "class": class_name,
        "world": world,
        "depth_m": float(sample.depth_m),
        "depth_valid_ratio": float(sample.valid_ratio),
        "shelf_layer": layer,
        "depth_ok": True,
    }
    if class_name == "packaging_box":
        out["place_left_world"] = place_left_of_packaging(world)
    return out
