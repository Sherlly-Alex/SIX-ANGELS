"""Layered occupancy grid tests.

Hard acceptance: the chassis layer must be cell-for-cell identical to the
historical single-layer grid from ``build_material_scene_grid``.  The arm
layer must contain walls and the shelf but omit the table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.occupancy_grid import (
    ARM_Z_MAX,
    ARM_Z_MIN,
    CHASSIS_Z_MAX,
    CHASSIS_Z_MIN,
    ObstacleVolume,
    build_layered_scene_grid,
    build_material_scene_grid,
    scene_static_obstacle_volumes,
    scene_static_obstacles,
)

LAYOUT_JSON = TASK_DIR / "material_competition_layout.json"


def _scene():
    return json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))["scene"]


class TestObstacleVolumes:

    def test_volume_kinds_and_counts(self):
        vols = scene_static_obstacle_volumes(_scene())
        kinds = [v.kind for v in vols]
        assert kinds.count("table") == 1
        assert kinds.count("shelf") == 1
        assert kinds.count("wall") == 4

    def test_table_does_not_intersect_arm_band(self):
        table = next(v for v in scene_static_obstacle_volumes(_scene()) if v.kind == "table")
        assert table.z_max <= ARM_Z_MIN
        assert not table.intersects_z(ARM_Z_MIN, ARM_Z_MAX)
        assert table.intersects_z(CHASSIS_Z_MIN, CHASSIS_Z_MAX)

    def test_shelf_and_walls_span_both_bands(self):
        for v in scene_static_obstacle_volumes(_scene()):
            if v.kind in ("shelf", "wall"):
                assert v.intersects_z(CHASSIS_Z_MIN, CHASSIS_Z_MAX)
                assert v.intersects_z(ARM_Z_MIN, ARM_Z_MAX)

    def test_legacy_xy_wrapper_matches_chassis_filter(self):
        scene = _scene()
        legacy = scene_static_obstacles(scene)
        filtered = [
            v.as_xy()
            for v in scene_static_obstacle_volumes(scene)
            if v.intersects_z(CHASSIS_Z_MIN, CHASSIS_Z_MAX)
        ]
        assert legacy == filtered


class TestLayeredGridIdentity:

    def test_chassis_matches_legacy_grid_cellwise(self):
        scene = _scene()
        legacy = build_material_scene_grid(scene=scene)
        layered = build_layered_scene_grid(scene=scene)
        assert legacy.origin_x == pytest.approx(layered.origin_x)
        assert legacy.origin_y == pytest.approx(layered.origin_y)
        assert legacy.resolution == pytest.approx(layered.resolution)
        assert legacy.width == layered.width
        assert legacy.height == layered.height
        np.testing.assert_array_equal(legacy._grid, layered.chassis._grid)

    def test_build_material_scene_grid_is_chassis_alias(self):
        """``build_material_scene_grid`` must remain a chassis-layer alias."""
        scene = _scene()
        g = build_material_scene_grid(scene=scene)
        layered = build_layered_scene_grid(scene=scene)
        np.testing.assert_array_equal(g._grid, layered.chassis._grid)

    def test_arm_layer_omits_table(self):
        layered = build_layered_scene_grid(scene=_scene())
        # Table body centre from MJCF / layout.
        tx, ty = -0.54, 2.315
        gx, gy = layered.world_to_grid(tx, ty)
        assert layered.chassis.is_occupied(gx, gy)
        assert layered.arm.is_free(gx, gy)

    def test_arm_layer_keeps_shelf_and_walls(self):
        layered = build_layered_scene_grid(scene=_scene())
        # Shelf body centre.
        sx, sy = layered.world_to_grid(-2.67, 0.78)
        assert layered.chassis.is_occupied(sx, sy)
        assert layered.arm.is_occupied(sx, sy)
        # West wall interior.
        wx, wy = layered.world_to_grid(-2.92, 1.25)
        assert layered.chassis.is_occupied(wx, wy)
        assert layered.arm.is_occupied(wx, wy)

    def test_start_pose_free_on_both_layers(self):
        layered = build_layered_scene_grid(scene=_scene())
        gx, gy = layered.world_to_grid(-0.70, 0.55)
        assert layered.chassis.is_free(gx, gy)
        assert layered.arm.is_free(gx, gy)


class TestDynamicOverlay:

    def test_set_dynamic_marks_by_height(self):
        layered = build_layered_scene_grid(scene=_scene())
        # Floor-level dropped box — chassis only.
        floor_box = ObstacleVolume(
            -1.0, -0.8, 0.4, 0.6, z_min=0.0, z_max=0.19, kind="dynamic",
        )
        # Elevated prop on table — arm band only (z around 0.9).
        elevated = ObstacleVolume(
            -0.70, -0.50, 2.20, 2.40, z_min=0.80, z_max=1.00, kind="dynamic",
        )
        layered.set_dynamic([floor_box, elevated])

        fx, fy = layered.world_to_grid(-0.90, 0.50)
        assert layered.layer("chassis").is_occupied(fx, fy)
        assert layered.layer("arm").is_free(fx, fy)

        ex, ey = layered.world_to_grid(-0.60, 2.30)
        # Static table occupies chassis; dynamic elevated occupies arm.
        assert layered.layer("chassis").is_occupied(ex, ey)  # table
        assert layered.layer("arm").is_occupied(ex, ey)  # dynamic prop

    def test_clear_dynamic_restores_static(self):
        layered = build_layered_scene_grid(scene=_scene())
        before = layered.layer("chassis")._grid.copy()
        layered.set_dynamic([
            ObstacleVolume(-1.0, -0.8, 0.4, 0.6, 0.0, 0.2, kind="dynamic"),
        ])
        layered.clear_dynamic()
        np.testing.assert_array_equal(layered.layer("chassis")._grid, before)


class TestMergeCache:
    """Layer merges are cached; the cache must track overlay changes."""

    def _vol(self, x):
        return ObstacleVolume(x, x + 0.2, 0.4, 0.6, 0.0, 0.2, kind="dynamic")

    def test_repeated_calls_reuse_one_object(self):
        layered = build_layered_scene_grid(scene=_scene())
        assert layered.layer("chassis") is layered.layer("chassis")
        layered.set_dynamic([self._vol(-1.0)])
        assert layered.layer("chassis") is layered.layer("chassis")

    def test_no_overlay_reuses_static_layer(self):
        layered = build_layered_scene_grid(scene=_scene())
        assert layered.planning_grid() is layered.chassis

    def test_cache_invalidated_by_overlay_change(self):
        layered = build_layered_scene_grid(scene=_scene())
        gx, gy = layered.world_to_grid(-0.90, 0.50)
        assert layered.layer("chassis").is_free(gx, gy)
        layered.set_dynamic([self._vol(-1.0)])
        assert layered.layer("chassis").is_occupied(gx, gy)
        layered.clear_dynamic()
        assert layered.layer("chassis").is_free(gx, gy)

    def test_identical_set_dynamic_is_a_noop(self):
        """Per-tick refreshes must not thrash the cache."""
        layered = build_layered_scene_grid(scene=_scene())
        vols = [self._vol(-1.0)]
        layered.set_dynamic(vols)
        first = layered.layer("chassis")
        layered.set_dynamic(list(vols))
        assert layered.layer("chassis") is first
        assert layered.dynamic_volumes == tuple(vols)

    def test_different_volumes_rebuild(self):
        layered = build_layered_scene_grid(scene=_scene())
        layered.set_dynamic([self._vol(-1.0)])
        first = layered.layer("chassis")
        layered.set_dynamic([self._vol(-1.4)])
        assert layered.layer("chassis") is not first
        old_x, old_y = layered.world_to_grid(-0.90, 0.50)
        new_x, new_y = layered.world_to_grid(-1.30, 0.50)
        assert layered.layer("chassis").is_free(old_x, old_y)
        assert layered.layer("chassis").is_occupied(new_x, new_y)
