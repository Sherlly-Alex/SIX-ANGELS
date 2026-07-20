"""Unit tests for PlanningGrid using mock static grid and dynamic layer."""
import sys
import os

_module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import numpy as np

from planning.planning_grid import PlanningGrid


class MockStaticGrid:
    """Minimal static grid stand-in for PlanningGrid tests."""

    def __init__(self, width, height, resolution=0.05, origin=None):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin if origin is not None else [0.0, 0.0]
        # log_odds convention: >0.5 occupied, near 0 unknown, <-0.5 free
        self.log_odds = np.full((height, width), -1.0, dtype=np.float32)

    def world_to_map(self, x, y):
        mx = int(np.floor((x - self.origin[0]) / self.resolution))
        my = int(np.floor((y - self.origin[1]) / self.resolution))
        return mx, my

    def map_to_world(self, mx, my):
        x = self.origin[0] + (mx + 0.5) * self.resolution
        y = self.origin[1] + (my + 0.5) * self.resolution
        return x, y

    def is_occupied(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return True
        return self.log_odds[my, mx] > 0.5

    def is_free(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.log_odds[my, mx] < -0.5

    def is_unknown(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return np.abs(self.log_odds[my, mx]) <= 0.1


class MockDynamicLayer:
    """Minimal dynamic layer stand-in for PlanningGrid tests."""

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.version = 0

    def is_occupied(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.grid[my, mx] > 0

    def occupied_mask(self):
        return self.grid > 0

    def set_occupied(self, mx, my):
        if 0 <= mx < self.width and 0 <= my < self.height:
            if not self.grid[my, mx] > 0:
                self.grid[my, mx] = 1.0
                self.version += 1

    def clear(self, mx, my):
        if 0 <= mx < self.width and 0 <= my < self.height:
            if self.grid[my, mx] > 0:
                self.grid[my, mx] = 0.0
                self.version += 1


def _make_grids(allow_unknown=False):
    static_grid = MockStaticGrid(20, 20)
    dynamic_layer = MockDynamicLayer(20, 20)
    planning_grid = PlanningGrid(static_grid, dynamic_layer, allow_unknown=allow_unknown)
    return planning_grid, static_grid, dynamic_layer


def test_static_occupied_blocked():
    pg, static_grid, _ = _make_grids()
    static_grid.log_odds[10, 10] = 1.0
    assert pg.is_occupied(10, 10)
    print("[PASS] test_static_occupied_blocked")


def test_dynamic_occupied_blocked():
    pg, _, dynamic_layer = _make_grids()
    dynamic_layer.set_occupied(10, 10)
    assert pg.is_occupied(10, 10)
    print("[PASS] test_dynamic_occupied_blocked")


def test_unknown_blocked_default():
    pg, static_grid, _ = _make_grids()
    static_grid.log_odds[10, 10] = 0.0
    assert pg.is_occupied(10, 10)
    print("[PASS] test_unknown_blocked_default")


def test_unknown_allowed():
    pg, static_grid, _ = _make_grids(allow_unknown=True)
    static_grid.log_odds[10, 10] = 0.0
    assert not pg.is_occupied(10, 10)
    print("[PASS] test_unknown_allowed")


def test_out_of_bounds_occupied():
    pg, _, _ = _make_grids()
    assert pg.is_occupied(-1, 10)
    assert pg.is_occupied(10, -1)
    assert pg.is_occupied(20, 10)
    assert pg.is_occupied(10, 20)
    print("[PASS] test_out_of_bounds_occupied")


def test_inflation_cost_near_obstacle():
    pg, static_grid, _ = _make_grids()
    static_grid.log_odds[10, 10] = 1.0
    # Adjacent cell is 1 cell away -> dist=1, min_clearance=2 default -> inf
    cost = pg.get_inflation_cost(11, 10, inflation_radius=5.0, min_clearance=2.0)
    assert cost == float("inf")
    print("[PASS] test_inflation_cost_near_obstacle")


def test_inflation_cost_far_from_obstacle():
    pg, static_grid, _ = _make_grids()
    static_grid.log_odds[10, 10] = 1.0
    # Cell (0, 0) is ~14 cells away from the obstacle, well above inflation_radius=5
    cost = pg.get_inflation_cost(0, 0, inflation_radius=5.0, min_clearance=2.0)
    assert cost == 1.0
    print("[PASS] test_inflation_cost_far_from_obstacle")


def test_inflation_cost_gradient():
    pg, static_grid, _ = _make_grids()
    static_grid.log_odds[10, 10] = 1.0
    # Choose a cell at intermediate distance (e.g., 3 cells away)
    cost = pg.get_inflation_cost(13, 10, inflation_radius=5.0, min_clearance=2.0)
    assert 1.0 < cost < float("inf")
    print("[PASS] test_inflation_cost_gradient")


def test_distance_transform_cache():
    pg, _, _ = _make_grids()
    d1 = pg.compute_distance_transform()
    d2 = pg.compute_distance_transform()
    assert d1 is d2
    print("[PASS] test_distance_transform_cache")


def test_distance_transform_invalidation():
    pg, _, dynamic_layer = _make_grids()
    d1 = pg.compute_distance_transform()
    dynamic_layer.set_occupied(10, 10)
    d2 = pg.compute_distance_transform()
    assert d2 is not d1
    assert d2[10, 10] == 0.0
    print("[PASS] test_distance_transform_invalidation")


if __name__ == "__main__":
    test_static_occupied_blocked()
    test_dynamic_occupied_blocked()
    test_unknown_blocked_default()
    test_unknown_allowed()
    test_out_of_bounds_occupied()
    test_inflation_cost_near_obstacle()
    test_inflation_cost_far_from_obstacle()
    test_inflation_cost_gradient()
    test_distance_transform_cache()
    test_distance_transform_invalidation()
    print("[ALL PASS] PlanningGrid unit tests")
