#!/usr/bin/env python3
"""Rolling local height / occupancy map from head RGB-D (exploration).

Formal Server has no LiDAR — only head RGB-D + odometry.  This module keeps a
robot-centered 2.5D map for near-field queries.  It does **not** replace the
prior layered occupancy grid used by A*.

Enable later with env ``MATERIAL_LOCAL_MAP=1`` (default off).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from perception.depth_geometry import (  # type: ignore
        DEPTH_MAX_M,
        DEPTH_MIN_M,
        DEFAULT_DEPTH_SCALE,
        depth_frame_to_world_points,
        frame_depth_quality,
    )
except ImportError:  # script / flat path layout
    from depth_geometry import (  # type: ignore
        DEPTH_MAX_M,
        DEPTH_MIN_M,
        DEFAULT_DEPTH_SCALE,
        depth_frame_to_world_points,
        frame_depth_quality,
    )


def local_map_enabled(default: bool = False) -> bool:
    raw = os.environ.get("MATERIAL_LOCAL_MAP")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def local_map_hz(default: float = 0.5) -> float:
    """Max integrate / FK rate when local map is enabled (default 0.5 Hz)."""
    hz = _env_float("MATERIAL_LOCAL_MAP_HZ", default)
    if hz <= 0.0:
        return float(default)
    return min(30.0, max(0.1, float(hz)))


def local_map_min_interval_s(default_hz: float = 0.5) -> float:
    return 1.0 / local_map_hz(default_hz)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class ClearanceResult:
    clear: bool
    distance_m: float
    hit_xy: Optional[Tuple[float, float]]
    hit_z: Optional[float]
    reason: str = ""


class RollingLocalHeightMap:
    """Robot-centered rolling max-Z map with hit counts and age decay."""

    def __init__(
        self,
        *,
        resolution: float = 0.05,
        forward_m: float = 2.5,
        back_m: float = 0.5,
        side_m: float = 1.5,
        floor_z: float = 0.02,
        obstacle_z_above_floor: float = 0.08,
        min_hits: int = 2,
        max_age_s: float = 8.0,
        min_frame_valid_ratio: float = 0.05,
        depth_scale: float = DEFAULT_DEPTH_SCALE,
        depth_min: float = DEPTH_MIN_M,
        depth_max: float = DEPTH_MAX_M,
        stride: int = 4,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        self.resolution = float(resolution)
        self.forward_m = float(forward_m)
        self.back_m = float(back_m)
        self.side_m = float(side_m)
        self.floor_z = float(floor_z)
        self.obstacle_z_above_floor = float(obstacle_z_above_floor)
        self.min_hits = max(1, int(min_hits))
        self.max_age_s = float(max_age_s)
        self.min_frame_valid_ratio = float(min_frame_valid_ratio)
        self.depth_scale = float(depth_scale)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.stride = max(1, int(stride))

        self.origin_xy = (0.0, 0.0)
        self._width = 1
        self._height = 1
        self.height_z = np.full((1, 1), np.nan, dtype=np.float32)
        self.hit_count = np.zeros((1, 1), dtype=np.int32)
        self.last_update_s = np.full((1, 1), np.nan, dtype=np.float64)
        self.robot_pose_xyyaw: Optional[Tuple[float, float, float]] = None
        self.last_integrate_s: Optional[float] = None
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.last_reject_reason = ""
        self._allocate_window(0.0, 0.0)

    @classmethod
    def from_env(cls) -> "RollingLocalHeightMap":
        return cls(
            resolution=_env_float("MATERIAL_LOCAL_MAP_RES", 0.05),
            max_age_s=_env_float("MATERIAL_LOCAL_MAP_MAX_AGE_S", 8.0),
            min_hits=max(1, int(_env_float("MATERIAL_LOCAL_MAP_MIN_HITS", 2))),
            stride=max(1, int(_env_float("MATERIAL_LOCAL_MAP_STRIDE", 8))),
        )

    def reset(self) -> None:
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.last_reject_reason = ""
        self.last_integrate_s = None
        self.robot_pose_xyyaw = None
        self._allocate_window(0.0, 0.0)

    def seed_pose(self, robot_pose_xyyaw: Sequence[float]) -> None:
        """Allocate / attach the window to a pose without depth (tests / warmup)."""
        if len(robot_pose_xyyaw) < 3:
            raise ValueError("robot_pose_xyyaw must be (x, y, yaw)")
        x, y, yaw = map(float, robot_pose_xyyaw[:3])
        self._ensure_window_for_robot(x, y, yaw)

    def _allocate_window(self, cx: float, cy: float) -> None:
        x0 = float(cx) - self.back_m
        y0 = float(cy) - self.side_m
        width = max(1, int(math.ceil((self.forward_m + self.back_m) / self.resolution)))
        height = max(1, int(math.ceil((2.0 * self.side_m) / self.resolution)))
        self.origin_xy = (x0, y0)
        self._width = width
        self._height = height
        self.height_z = np.full((height, width), np.nan, dtype=np.float32)
        self.hit_count = np.zeros((height, width), dtype=np.int32)
        self.last_update_s = np.full((height, width), np.nan, dtype=np.float64)

    def _decay(self, now_s: float) -> None:
        if not np.isfinite(now_s) or self.max_age_s <= 0.0:
            return
        age = now_s - self.last_update_s
        stale = np.isfinite(self.last_update_s) & (age > self.max_age_s)
        if not np.any(stale):
            return
        self.height_z[stale] = np.nan
        self.hit_count[stale] = 0
        self.last_update_s[stale] = np.nan

    def integrate_depth(
        self,
        depth_img,
        k_matrix,
        t_cam_world: np.ndarray,
        robot_pose_xyyaw: Sequence[float],
        now_s: float,
    ) -> dict:
        """Fuse one RGB-D frame. Returns a small status dict."""
        if len(robot_pose_xyyaw) < 3:
            raise ValueError("robot_pose_xyyaw must be (x, y, yaw)")
        x, y, yaw = (
            float(robot_pose_xyyaw[0]),
            float(robot_pose_xyyaw[1]),
            float(robot_pose_xyyaw[2]),
        )
        quality = frame_depth_quality(
            depth_img,
            depth_scale=self.depth_scale,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            stride=self.stride,
        )
        if (
            not quality.get("ok")
            or float(quality.get("valid_ratio", 0.0)) < self.min_frame_valid_ratio
        ):
            self.frames_rejected += 1
            self.last_reject_reason = "low_depth_quality"
            return {
                "accepted": False,
                "reason": self.last_reject_reason,
                "quality": quality,
            }

        self._ensure_window_for_robot(x, y, yaw)
        self._decay(float(now_s))

        pts = depth_frame_to_world_points(
            depth_img,
            k_matrix,
            t_cam_world,
            depth_scale=self.depth_scale,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            stride=self.stride,
        )
        if pts.size == 0:
            self.frames_rejected += 1
            self.last_reject_reason = "no_points"
            return {
                "accepted": False,
                "reason": self.last_reject_reason,
                "quality": quality,
            }

        z_min = self.floor_z + self.obstacle_z_above_floor
        pts = pts[pts[:, 2] >= z_min]
        if pts.size == 0:
            self.frames_rejected += 1
            self.last_reject_reason = "all_floor"
            return {
                "accepted": False,
                "reason": self.last_reject_reason,
                "quality": quality,
            }

        ox, oy = self.origin_xy
        ix = np.floor((pts[:, 0] - ox) / self.resolution).astype(np.int32)
        iy = np.floor((pts[:, 1] - oy) / self.resolution).astype(np.int32)
        inside = (ix >= 0) & (ix < self._width) & (iy >= 0) & (iy < self._height)
        ix, iy = ix[inside], iy[inside]
        zw = pts[inside, 2]
        n = int(ix.size)
        for i, j, z in zip(ix, iy, zw):
            self.hit_count[j, i] += 1
            prev = self.height_z[j, i]
            if not np.isfinite(prev) or z > prev:
                self.height_z[j, i] = float(z)
            self.last_update_s[j, i] = float(now_s)

        self.frames_accepted += 1
        self.last_integrate_s = float(now_s)
        self.last_reject_reason = ""
        return {
            "accepted": True,
            "reason": "ok",
            "n_points": n,
            "quality": quality,
        }

    def _ensure_window_for_robot(self, x: float, y: float, yaw: float) -> None:
        if self.robot_pose_xyyaw is None:
            self._allocate_window(x, y)
            self.robot_pose_xyyaw = (float(x), float(y), float(yaw))
            return
        ox, oy = self.origin_xy
        margin = 0.4
        x_min = ox + margin
        x_max = ox + self._width * self.resolution - margin
        y_min = oy + margin
        y_max = oy + self._height * self.resolution - margin
        if x_min <= x <= x_max and y_min <= y <= y_max:
            self.robot_pose_xyyaw = (float(x), float(y), float(yaw))
            return
        desired_ox = x - self.back_m
        desired_oy = y - self.side_m
        shift_x = int(round((desired_ox - ox) / self.resolution))
        shift_y = int(round((desired_oy - oy) / self.resolution))
        self._shift_window(shift_x, shift_y)
        self.robot_pose_xyyaw = (float(x), float(y), float(yaw))

    def _shift_window(self, shift_cells_x: int, shift_cells_y: int) -> None:
        if shift_cells_x == 0 and shift_cells_y == 0:
            return
        ox, oy = self.origin_xy
        new_origin = (
            ox + shift_cells_x * self.resolution,
            oy + shift_cells_y * self.resolution,
        )
        new_h = np.full_like(self.height_z, np.nan)
        new_c = np.zeros_like(self.hit_count)
        new_t = np.full_like(self.last_update_s, np.nan)
        src_x0 = max(0, shift_cells_x)
        src_y0 = max(0, shift_cells_y)
        src_x1 = min(self._width, self._width + shift_cells_x)
        src_y1 = min(self._height, self._height + shift_cells_y)
        dst_x0 = max(0, -shift_cells_x)
        dst_y0 = max(0, -shift_cells_y)
        w = src_x1 - src_x0
        h = src_y1 - src_y0
        if w > 0 and h > 0:
            new_h[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = self.height_z[
                src_y0:src_y1, src_x0:src_x1
            ]
            new_c[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = self.hit_count[
                src_y0:src_y1, src_x0:src_x1
            ]
            new_t[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = self.last_update_s[
                src_y0:src_y1, src_x0:src_x1
            ]
        self.origin_xy = new_origin
        self.height_z = new_h
        self.hit_count = new_c
        self.last_update_s = new_t

    def occupied_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.height_z)
            & (self.hit_count >= self.min_hits)
            & (self.height_z >= self.floor_z + self.obstacle_z_above_floor)
        )

    def cell_index(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        ox, oy = self.origin_xy
        ix = int(math.floor((float(x) - ox) / self.resolution))
        iy = int(math.floor((float(y) - oy) / self.resolution))
        if 0 <= ix < self._width and 0 <= iy < self._height:
            return ix, iy
        return None

    def height_at(self, x: float, y: float) -> Optional[float]:
        idx = self.cell_index(x, y)
        if idx is None:
            return None
        ix, iy = idx
        if self.hit_count[iy, ix] < self.min_hits:
            return None
        z = float(self.height_z[iy, ix])
        return z if math.isfinite(z) else None

    def forward_clearance(
        self,
        robot_pose_xyyaw: Sequence[float],
        *,
        width_m: float = 0.40,
        max_range_m: float = 1.20,
        sample_step_m: Optional[float] = None,
    ) -> ClearanceResult:
        """First occupied cell along a forward corridor in robot heading."""
        if len(robot_pose_xyyaw) < 3:
            raise ValueError("robot_pose_xyyaw must be (x, y, yaw)")
        x0, y0, yaw = map(float, robot_pose_xyyaw[:3])
        step = (
            float(sample_step_m)
            if sample_step_m
            else max(self.resolution * 0.5, 0.025)
        )
        half_w = 0.5 * float(width_m)
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = -s, c
        dist = 0.0
        while dist <= float(max_range_m) + 1e-9:
            cx = x0 + dist * c
            cy = y0 + dist * s
            for lat in (-half_w, 0.0, half_w):
                qx = cx + lat * lx
                qy = cy + lat * ly
                z = self.height_at(qx, qy)
                if z is not None:
                    return ClearanceResult(
                        clear=False,
                        distance_m=float(dist),
                        hit_xy=(float(qx), float(qy)),
                        hit_z=float(z),
                        reason="occupied",
                    )
            dist += step
        return ClearanceResult(
            clear=True,
            distance_m=float(max_range_m),
            hit_xy=None,
            hit_z=None,
            reason="clear",
        )

    def suggested_standoff(
        self,
        robot_pose_xyyaw: Sequence[float],
        *,
        desired_clearance_m: float = 0.55,
        width_m: float = 0.40,
        max_range_m: float = 1.50,
        min_standoff_m: float = 0.35,
        max_standoff_m: float = 0.90,
    ) -> float:
        """Suggest base standoff from the first forward obstacle (clamped)."""
        result = self.forward_clearance(
            robot_pose_xyyaw,
            width_m=width_m,
            max_range_m=max_range_m,
        )
        if result.clear or result.hit_xy is None:
            return float(np.clip(desired_clearance_m, min_standoff_m, max_standoff_m))
        suggested = min(float(result.distance_m), float(desired_clearance_m))
        return float(np.clip(suggested, min_standoff_m, max_standoff_m))

    def snapshot(self) -> dict:
        return {
            "origin_xy": self.origin_xy,
            "resolution": self.resolution,
            "size_wh": (self._width, self._height),
            "height_z": self.height_z.copy(),
            "hit_count": self.hit_count.copy(),
            "occupied": self.occupied_mask().copy(),
            "robot_pose_xyyaw": self.robot_pose_xyyaw,
            "frames_accepted": self.frames_accepted,
            "frames_rejected": self.frames_rejected,
            "last_reject_reason": self.last_reject_reason,
        }

    def to_debug_ascii(self, max_cols: int = 40) -> str:
        occ = self.occupied_mask()
        h, w = occ.shape
        step = max(1, int(math.ceil(w / max_cols)))
        rows = []
        for j in range(h - 1, -1, -step):
            chars = []
            for i in range(0, w, step):
                block = occ[max(0, j - step + 1) : j + 1, i : i + step]
                chars.append("#" if np.any(block) else ".")
            rows.append("".join(chars))
        return "\n".join(rows)


def integrate_points_for_tests(
    local_map: RollingLocalHeightMap,
    points_xyz: Iterable[Sequence[float]],
    now_s: float,
) -> int:
    """Test helper: write world points directly (bypass camera)."""
    pts = np.asarray(list(points_xyz), dtype=float)
    if pts.size == 0:
        return 0
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if local_map.robot_pose_xyyaw is None:
        raise RuntimeError("call seed_pose or integrate_depth before writing points")
    ox, oy = local_map.origin_xy
    z_min = local_map.floor_z + local_map.obstacle_z_above_floor
    pts = pts[pts[:, 2] >= z_min]
    ix = np.floor((pts[:, 0] - ox) / local_map.resolution).astype(np.int32)
    iy = np.floor((pts[:, 1] - oy) / local_map.resolution).astype(np.int32)
    inside = (
        (ix >= 0)
        & (ix < local_map._width)
        & (iy >= 0)
        & (iy < local_map._height)
    )
    n = 0
    for i, j, z in zip(ix[inside], iy[inside], pts[inside, 2]):
        local_map.hit_count[j, i] += 1
        prev = local_map.height_z[j, i]
        if not np.isfinite(prev) or z > prev:
            local_map.height_z[j, i] = float(z)
        local_map.last_update_s[j, i] = float(now_s)
        n += 1
    return n


__all__ = [
    "ClearanceResult",
    "RollingLocalHeightMap",
    "integrate_points_for_tests",
    "local_map_enabled",
    "local_map_hz",
    "local_map_min_interval_s",
]
