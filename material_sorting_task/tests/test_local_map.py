"""Unit tests for RGB-D local height map (exploration)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from perception.depth_geometry import (  # noqa: E402
    build_local_height_map,
    depth_frame_to_world_points,
    frame_depth_quality,
)
from perception.local_map import (  # noqa: E402
    RollingLocalHeightMap,
    integrate_points_for_tests,
    local_map_enabled,
)


def _toy_depth_and_K(depth_m: float = 1.0, h: int = 48, w: int = 64):
    depth = np.full((h, w), int(depth_m * 1000), dtype=np.uint16)
    K = np.array([[200.0, 0.0, w / 2.0], [0.0, 200.0, h / 2.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    # Camera at world (0,0,1), looking along +X (rotate so cam Z -> world X)
    # Simpler: identity with camera at z=1 looking along +Z into a wall of
    # constant depth — points land near z_cam = depth.
    T[2, 3] = 0.0
    return depth, K, T


class TestDepthGeometryHelpers:
    def test_frame_quality_and_points(self):
        depth, K, T = _toy_depth_and_K(1.2)
        q = frame_depth_quality(depth, stride=2)
        assert q["ok"]
        assert q["valid_ratio"] > 0.5
        pts = depth_frame_to_world_points(depth, K, T, stride=4)
        assert pts.shape[1] == 3
        assert pts.shape[0] > 10
        # With identity extrinsics, z ≈ depth
        assert abs(float(np.median(pts[:, 2])) - 1.2) < 0.05

    def test_build_local_height_map_has_quality(self):
        depth, K, T = _toy_depth_and_K(1.0)
        out = build_local_height_map(
            depth,
            K,
            T,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            resolution=0.1,
            stride=4,
        )
        assert out["n_points"] > 0
        assert "quality" in out
        assert out["quality"]["ok"]


class TestRollingLocalHeightMap:
    def test_clearance_and_standoff(self):
        m = RollingLocalHeightMap(
            resolution=0.05,
            forward_m=2.0,
            back_m=0.5,
            side_m=1.0,
            min_hits=2,
            max_age_s=10.0,
        )
        pose = (0.0, 0.0, 0.0)  # facing +X
        m.seed_pose(pose)
        # Wall of points at x=0.80, z=0.4 — need 2 hits
        pts = []
        for y in np.linspace(-0.2, 0.2, 9):
            pts.append((0.80, float(y), 0.40))
            pts.append((0.80, float(y), 0.40))
        n = integrate_points_for_tests(m, pts, now_s=1.0)
        assert n > 0
        clr = m.forward_clearance(pose, width_m=0.3, max_range_m=1.5)
        assert not clr.clear
        assert 0.7 <= clr.distance_m <= 0.9
        stand = m.suggested_standoff(pose, desired_clearance_m=0.55)
        assert 0.35 <= stand <= 0.90

    def test_decay_clears_old_cells(self):
        m = RollingLocalHeightMap(min_hits=1, max_age_s=2.0, resolution=0.05)
        pose = (0.0, 0.0, 0.0)
        m.seed_pose(pose)
        integrate_points_for_tests(m, [(0.5, 0.0, 0.3), (0.5, 0.0, 0.3)], now_s=1.0)
        assert m.height_at(0.5, 0.0) is not None
        # Trigger decay via empty-ish integrate path: call _decay directly
        m._decay(5.0)
        assert m.height_at(0.5, 0.0) is None

    def test_rejects_bad_depth_frame(self):
        m = RollingLocalHeightMap(min_frame_valid_ratio=0.2)
        depth = np.zeros((32, 32), dtype=np.uint16)
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 100.0
        K[0, 2] = K[1, 2] = 16.0
        status = m.integrate_depth(depth, K, np.eye(4), (0.0, 0.0, 0.0), now_s=0.0)
        assert status["accepted"] is False
        assert m.frames_rejected == 1

    def test_env_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("MATERIAL_LOCAL_MAP", raising=False)
        assert local_map_enabled() is False
        monkeypatch.setenv("MATERIAL_LOCAL_MAP", "1")
        assert local_map_enabled() is True
