"""Optional RGB-D / Canny height refine for shelf layers.

Geometric board heights remain the source of truth. This module only proposes a
world-Z offset. Visual failure always returns applied_delta=0 (non-fatal).

Feature flags (default off):
  MATERIAL_ENABLE_LAYER_REFINE=0
  MATERIAL_ENABLE_CANNY_REFINE=0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from shelf_geometry import ShelfGeometry

# Reject wild estimates before soft clamp (meters).
MAX_APPLY_DELTA_Z_M = 0.05
REJECT_ABS_DELTA_Z_M = 0.08
MIN_CONFIDENCE = 0.45
MIN_DEPTH_POINTS = 40
EDGE_TO_BOARD_TOP_Z = 0.0  # observed_feature board_front_edge → board top; calibrate later
MAX_FRAME_AGE_S = 0.5
MAX_RGB_DEPTH_SKEW_S = 0.1


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def layer_refine_enabled() -> bool:
    return env_flag("MATERIAL_ENABLE_LAYER_REFINE", False)


def canny_refine_enabled() -> bool:
    return env_flag("MATERIAL_ENABLE_CANNY_REFINE", False)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class RefineResult:
    raw_delta_z_world_m: float
    applied_delta_z_world_m: float
    source: str  # "none" | "rgbd" | "canny_hough"
    confidence: float
    reason: str
    observed_feature: str = "none"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def no_refine(reason: str, **diag: Any) -> RefineResult:
    return RefineResult(
        raw_delta_z_world_m=0.0,
        applied_delta_z_world_m=0.0,
        source="none",
        confidence=0.0,
        reason=reason,
        observed_feature="none",
        diagnostics=dict(diag),
    )


def accept_delta(
    raw: float,
    *,
    source: str,
    confidence: float,
    observed_feature: str,
    reason: str = "ok",
    **diag: Any,
) -> RefineResult:
    raw = float(raw)
    if confidence < MIN_CONFIDENCE:
        return no_refine("low confidence", raw_delta_z=raw, confidence=confidence, **diag)
    if abs(raw) > REJECT_ABS_DELTA_Z_M:
        return no_refine(
            "delta outside plausible range",
            raw_delta_z=raw,
            confidence=confidence,
            **diag,
        )
    applied = float(np.clip(raw, -MAX_APPLY_DELTA_Z_M, MAX_APPLY_DELTA_Z_M))
    return RefineResult(
        raw_delta_z_world_m=raw,
        applied_delta_z_world_m=applied,
        source=source,
        confidence=float(confidence),
        reason=reason,
        observed_feature=observed_feature,
        diagnostics=dict(diag),
    )


def refine_layer_height(
    layer_id: int,
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    camera_intrinsics: CameraIntrinsics | None,
    T_world_from_camera: np.ndarray | None,
    geom: ShelfGeometry,
    *,
    enable_canny: bool | None = None,
    frame_age_s: float | None = None,
    rgb_depth_skew_s: float | None = None,
    depth_scale: float = 1.0,
) -> RefineResult:
    """Estimate world-Z correction for ``layer_id``.

    ``T_world_from_camera`` is a 4x4 transform mapping camera-frame points to world.
    Depth must be aligned to RGB; values are multiplied by ``depth_scale`` to meters
    (use 0.001 if the image is uint16 millimeters).
    """
    if enable_canny is None:
        enable_canny = canny_refine_enabled()

    if frame_age_s is not None and frame_age_s > MAX_FRAME_AGE_S:
        return no_refine("frame too old", frame_age_s=frame_age_s)
    if rgb_depth_skew_s is not None and rgb_depth_skew_s > MAX_RGB_DEPTH_SKEW_S:
        return no_refine("rgb/depth skew too large", rgb_depth_skew_s=rgb_depth_skew_s)

    board_z = geom.board_z(int(layer_id))

    rgbd = _refine_rgbd(
        layer_id=int(layer_id),
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        T_world_from_camera=T_world_from_camera,
        geom=geom,
        board_z=board_z,
        depth_scale=depth_scale,
    )
    if rgbd.source == "rgbd":
        return rgbd

    if enable_canny:
        canny = _refine_canny(
            layer_id=int(layer_id),
            rgb=rgb,
            camera_intrinsics=camera_intrinsics,
            T_world_from_camera=T_world_from_camera,
            geom=geom,
            board_z=board_z,
        )
        if canny.source == "canny_hough":
            return canny
        return no_refine(
            f"rgbd_failed={rgbd.reason}; canny_failed={canny.reason}",
            rgbd_reason=rgbd.reason,
            canny_reason=canny.reason,
        )

    return no_refine(rgbd.reason or "rgbd unavailable", **dict(rgbd.diagnostics))


def _as_T(T: np.ndarray | None) -> np.ndarray | None:
    if T is None:
        return None
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None
    return T


def _refine_rgbd(
    *,
    layer_id: int,
    depth: np.ndarray | None,
    camera_intrinsics: CameraIntrinsics | None,
    T_world_from_camera: np.ndarray | None,
    geom: ShelfGeometry,
    board_z: float,
    depth_scale: float,
) -> RefineResult:
    T = _as_T(T_world_from_camera)
    if depth is None or camera_intrinsics is None or T is None:
        return no_refine("rgbd inputs missing")

    depth = np.asarray(depth)
    if depth.ndim != 2:
        return no_refine("depth must be HxW")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        return no_refine("invalid depth_scale")

    h, w = depth.shape
    if camera_intrinsics.width and abs(camera_intrinsics.width - w) > 2:
        return no_refine("depth/intrinsics width mismatch")
    if camera_intrinsics.height and abs(camera_intrinsics.height - h) > 2:
        return no_refine("depth/intrinsics height mismatch")

    # Sample a vertical band around the projected shelf front at board_z.
    fx, fy, cx, cy = (
        camera_intrinsics.fx,
        camera_intrinsics.fy,
        camera_intrinsics.cx,
        camera_intrinsics.cy,
    )
    shelf_x, shelf_y = geom.shelf_xy
    # Front edge sample points in world near board top.
    ys = np.linspace(shelf_y - 0.35, shelf_y + 0.35, 15)
    xs = np.full_like(ys, shelf_x + 0.02)  # slightly in front of shelf body
    zs = np.full_like(ys, board_z + EDGE_TO_BOARD_TOP_Z)

    T_cam_from_world = np.linalg.inv(T)
    edge_z_samples = []
    for x, y, z in zip(xs, ys, zs):
        pw = np.array([x, y, z, 1.0], dtype=float)
        pc = T_cam_from_world @ pw
        if pc[2] <= 0.05:
            continue
        u = int(round(fx * pc[0] / pc[2] + cx))
        v = int(round(fy * pc[1] / pc[2] + cy))
        if not (2 <= u < w - 2 and 8 <= v < h - 8):
            continue
        # Depth discontinuity along a short vertical neighborhood (front edge cue).
        col = depth[:, u].astype(float) * float(depth_scale)
        col[~np.isfinite(col)] = 0.0
        col[col <= 0.05] = np.nan
        window = col[max(0, v - 12) : min(h, v + 13)]
        if np.sum(np.isfinite(window)) < 6:
            continue
        # Largest absolute gradient in window → candidate edge row.
        finite_idx = np.where(np.isfinite(window))[0]
        vals = window[finite_idx]
        if vals.size < 4:
            continue
        grads = np.abs(np.diff(vals))
        k = int(np.argmax(grads))
        v_edge = max(0, v - 12) + int(finite_idx[k])
        d = col[v_edge]
        if not np.isfinite(d) or d <= 0.05:
            continue
        # Back-project edge pixel to world Z.
        x_c = (u - cx) / fx * d
        y_c = (v_edge - cy) / fy * d
        p_c = np.array([x_c, y_c, d, 1.0])
        p_w = T @ p_c
        edge_z_samples.append(float(p_w[2]))

    if len(edge_z_samples) < 5:
        return no_refine(
            "insufficient front-edge depth samples",
            samples=len(edge_z_samples),
        )

    z_obs = float(np.median(edge_z_samples))
    # Convert observed front-edge Z to board-top equivalent.
    z_board_equiv = z_obs + EDGE_TO_BOARD_TOP_Z
    raw = z_board_equiv - board_z
    residual = float(np.median(np.abs(np.asarray(edge_z_samples) - z_obs)))
    if residual > 0.04:
        return no_refine(
            "edge z residual too large",
            residual=residual,
            samples=len(edge_z_samples),
            raw_delta_z=raw,
        )
    # Prefer this layer over neighbors.
    for other in (layer_id - 1, layer_id + 1):
        if 1 <= other <= geom.num_boards:
            if abs(z_board_equiv - geom.board_z(other)) + 0.01 < abs(raw):
                return no_refine(
                    "closer to neighboring board",
                    neighbor=other,
                    raw_delta_z=raw,
                )

    conf = float(np.clip(0.35 + 0.02 * len(edge_z_samples) - 5.0 * residual, 0.0, 0.95))
    return accept_delta(
        raw,
        source="rgbd",
        confidence=conf,
        observed_feature="board_front_edge",
        samples=len(edge_z_samples),
        residual=residual,
        z_obs=z_obs,
        board_z=board_z,
    )


def project_board_front_v(
    board_z: float,
    geom: ShelfGeometry,
    camera_intrinsics: CameraIntrinsics,
    T_world_from_camera: np.ndarray,
    *,
    y: float | None = None,
) -> float | None:
    """Project shelf front point at ``board_z`` to image row v."""
    T_cam_from_world = np.linalg.inv(T_world_from_camera)
    yy = geom.shelf_xy[1] if y is None else float(y)
    pw = np.array([geom.shelf_xy[0] + 0.02, yy, board_z, 1.0])
    pc = T_cam_from_world @ pw
    if pc[2] <= 0.05:
        return None
    v = camera_intrinsics.fy * pc[1] / pc[2] + camera_intrinsics.cy
    return float(v)


def _refine_canny(
    *,
    layer_id: int,
    rgb: np.ndarray | None,
    camera_intrinsics: CameraIntrinsics | None,
    T_world_from_camera: np.ndarray | None,
    geom: ShelfGeometry,
    board_z: float,
) -> RefineResult:
    T = _as_T(T_world_from_camera)
    if rgb is None or camera_intrinsics is None or T is None:
        return no_refine("camera calibration unavailable")

    try:
        import cv2
    except ImportError:
        return no_refine("opencv unavailable")

    rgb = np.asarray(rgb)
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb[:, :, 0]
    elif rgb.ndim == 2:
        gray = rgb
    else:
        return no_refine("unsupported rgb shape")

    v0 = project_board_front_v(board_z, geom, camera_intrinsics, T)
    if v0 is None:
        return no_refine("board projection behind camera")

    # Expected angle from two projected endpoints.
    v_l = project_board_front_v(
        board_z, geom, camera_intrinsics, T, y=geom.shelf_xy[1] - 0.3
    )
    v_r = project_board_front_v(
        board_z, geom, camera_intrinsics, T, y=geom.shelf_xy[1] + 0.3
    )
    expected_angle_deg = 0.0
    if v_l is not None and v_r is not None:
        # Horizontal-ish in image; small roll → small angle.
        expected_angle_deg = math.degrees(math.atan2((v_r - v_l), 100.0))

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=40, minLineLength=40, maxLineGap=12
    )
    if lines is None:
        return no_refine("no hough lines", expected_v=v0)

    best = None
    best_score = 1e9
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 40:
            continue
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalize to [-90, 90]
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        v_mid = 0.5 * (y1 + y2)
        if abs(v_mid - v0) > 35:
            continue
        if abs(ang - expected_angle_deg) > 18:
            continue
        score = abs(v_mid - v0) + 0.5 * abs(ang - expected_angle_deg)
        if score < best_score:
            best_score = score
            best = v_mid

    if best is None:
        return no_refine("no line near projected board", expected_v=v0)

    # 1D search Δz to match projected row to detected row.
    candidates = np.linspace(-MAX_APPLY_DELTA_Z_M, MAX_APPLY_DELTA_Z_M, 101)
    best_dz = 0.0
    best_err = 1e9
    for dz in candidates:
        v_proj = project_board_front_v(board_z + float(dz), geom, camera_intrinsics, T)
        if v_proj is None:
            continue
        err = abs(best - v_proj)
        if err < best_err:
            best_err = err
            best_dz = float(dz)

    if best_err > 8.0:  # pixels
        return no_refine(
            "canny delta search residual too large",
            pixel_err=best_err,
            detected_v=best,
            expected_v=v0,
        )

    conf = float(np.clip(0.55 - 0.03 * best_err, 0.0, 0.85))
    return accept_delta(
        best_dz,
        source="canny_hough",
        confidence=conf,
        observed_feature="board_front_edge",
        detected_v=best,
        expected_v=v0,
        pixel_err=best_err,
        expected_angle_deg=expected_angle_deg,
    )
