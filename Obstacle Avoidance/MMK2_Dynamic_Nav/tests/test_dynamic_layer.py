"""Unit tests for DynamicLayer (grid-level evidence tracking only).

No MuJoCo or DISCOVERSE dependencies are used here.
"""
import sys
import os

_module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import numpy as np

from perception.dynamic_layer import DynamicLayer


class DummyConfig:
    static_match_tolerance = 0.3  # meters


def _make_layer(height=10, width=10, resolution=0.05):
    # static_dist_map: large values everywhere (far from any static obstacle)
    static_dist_map = np.full((height, width), 100.0, dtype=np.float32)
    return DynamicLayer(static_dist_map, resolution, DummyConfig())


def test_initial_state():
    layer = _make_layer()
    assert layer.version == 0
    assert not np.any(layer.occupied_mask())
    print("[PASS] test_initial_state")


def test_single_hit_not_enough():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    assert not layer.is_occupied(5, 5)
    assert layer.version == 0
    print("[PASS] test_single_hit_not_enough")


def test_consecutive_hits_occupy():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.is_occupied(5, 5)
    assert layer.version == 1
    assert layer.timestamps[5, 5] == 0.1
    print("[PASS] test_consecutive_hits_occupy")


def test_repeated_hits_no_version_change():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.version == 1
    layer.mark_hit(5, 5, timestamp=0.2)
    layer.mark_hit(5, 5, timestamp=0.3)
    assert layer.version == 1
    assert layer.is_occupied(5, 5)
    print("[PASS] test_repeated_hits_no_version_change")


def test_single_miss_not_enough():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.is_occupied(5, 5)
    layer.mark_miss(5, 5)
    assert layer.is_occupied(5, 5)
    assert layer.version == 1
    print("[PASS] test_single_miss_not_enough")


def test_consecutive_misses_clear():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.is_occupied(5, 5)
    layer.mark_miss(5, 5)
    layer.mark_miss(5, 5)
    assert not layer.is_occupied(5, 5)
    assert layer.version == 2
    print("[PASS] test_consecutive_misses_clear")


def test_expire_removes_old_cells():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.is_occupied(5, 5)
    layer.expire(current_time=2.2, decay_time=2.0)
    assert not layer.is_occupied(5, 5)
    assert layer.version == 2
    print("[PASS] test_expire_removes_old_cells")


def test_continuous_observation_prevents_expire():
    layer = _make_layer()
    layer.mark_hit(5, 5, timestamp=0.0)
    layer.mark_hit(5, 5, timestamp=0.1)
    assert layer.is_occupied(5, 5)
    # Refresh timestamp just before expiry threshold
    layer.mark_hit(5, 5, timestamp=2.0)
    layer.expire(current_time=3.9, decay_time=2.0)
    assert layer.is_occupied(5, 5)
    assert layer.version == 1
    print("[PASS] test_continuous_observation_prevents_expire")


def test_should_track_near_static():
    # Cell is 2 cells away from a static obstacle -> 2 * 0.05 = 0.1m < tolerance
    static_dist_map = np.zeros((10, 10), dtype=np.float32)
    static_dist_map[5, 5] = 2.0
    resolution = 0.05
    tolerance = 0.3
    assert not DynamicLayer.should_track(5, 5, static_dist_map, resolution, tolerance)
    print("[PASS] test_should_track_near_static")


def test_should_track_far_from_static():
    static_dist_map = np.zeros((10, 10), dtype=np.float32)
    static_dist_map[5, 5] = 10.0  # 10 cells = 0.5m
    resolution = 0.05
    tolerance = 0.3
    assert DynamicLayer.should_track(5, 5, static_dist_map, resolution, tolerance)
    print("[PASS] test_should_track_far_from_static")


class MockGrid:
    """Minimal grid exposing world_to_map for use with DynamicLayer.update."""

    def __init__(self, resolution=0.05):
        self.resolution = resolution

    def world_to_map(self, x, y):
        mx = int(x / self.resolution)
        my = int(y / self.resolution)
        return mx, my


def test_update_marks_free_cells():
    """Cells between sensor and endpoint receive miss evidence after update."""
    layer = _make_layer(height=20, width=20)
    grid = MockGrid()
    pose = [0.0, 0.0, 0.0]
    ranges = np.array([0.2])  # 0.2 m = 4 cells at 0.05 res
    angles = np.array([0.0])

    layer.update(ranges, angles, pose, timestamp=0.0, grid=grid)

    # Endpoint at cell (4, 0): hit_count=1 (should_track returns True with
    # our dummy static_dist_map), miss_count was reset to 0 by mark_hit.
    assert layer.hit_count[0, 4] == 1
    assert layer.miss_count[0, 4] == 0

    # Intermediate cells (0,0) through (3,0): miss_count=1 each
    for mx in range(4):
        assert layer.miss_count[0, mx] == 1, f"mx={mx} miss={layer.miss_count[0, mx]}"

    print("[PASS] test_update_marks_free_cells")


def test_update_marks_hit_endpoint():
    """Endpoint cell becomes occupied after two consecutive update calls."""
    layer = _make_layer(height=20, width=20)
    grid = MockGrid()
    pose = [0.0, 0.0, 0.0]
    ranges = np.array([0.2])
    angles = np.array([0.0])

    layer.update(ranges, angles, pose, timestamp=0.0, grid=grid)
    assert not layer.is_occupied(4, 0)

    layer.update(ranges, angles, pose, timestamp=0.1, grid=grid)
    assert layer.is_occupied(4, 0)
    assert layer.hit_count[0, 4] >= 2

    print("[PASS] test_update_marks_hit_endpoint")


if __name__ == "__main__":
    test_initial_state()
    test_single_hit_not_enough()
    test_consecutive_hits_occupy()
    test_repeated_hits_no_version_change()
    test_single_miss_not_enough()
    test_consecutive_misses_clear()
    test_expire_removes_old_cells()
    test_continuous_observation_prevents_expire()
    test_should_track_near_static()
    test_should_track_far_from_static()
    test_update_marks_free_cells()
    test_update_marks_hit_endpoint()
    print("[ALL PASS] DynamicLayer unit tests")
