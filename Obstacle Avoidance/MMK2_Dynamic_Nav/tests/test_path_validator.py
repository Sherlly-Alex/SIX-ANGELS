"""Unit tests for PathValidator using mock planning grids."""
import sys
import os

_module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import numpy as np

from planning.path_validator import PathValidator


class MockConfig:
    path_inflation_radius = 0.5
    path_min_clearance = 0.2
    map_resolution = 0.05


class MockGrid:
    """Grid where get_inflation_cost returns inf for cells in block_set."""

    def __init__(self, block_set=None, resolution=0.05, origin=(0.0, 0.0)):
        self.block_set = set(block_set) if block_set is not None else set()
        self.resolution = resolution
        self.origin = origin

    def world_to_map(self, x, y):
        mx = int(np.floor((x - self.origin[0]) / self.resolution))
        my = int(np.floor((y - self.origin[1]) / self.resolution))
        return mx, my

    def get_inflation_cost(self, mx, my, inflation_radius, min_clearance=0.2):
        if (mx, my) in self.block_set:
            return float("inf")
        return 1.0


def _make_validator():
    return PathValidator(MockConfig())


def test_clear_segment():
    pv = _make_validator()
    grid = MockGrid()
    assert not pv.check_segment([0.0, 0.0], [1.0, 0.0], grid)
    print("[PASS] test_clear_segment")


def test_blocked_segment():
    pv = _make_validator()
    # Block a cell along x=0.5 (mx=10 at resolution 0.05)
    grid = MockGrid(block_set={(10, 0)})
    assert pv.check_segment([0.0, 0.0], [1.0, 0.0], grid)
    print("[PASS] test_blocked_segment")


def test_partial_segment_check():
    pv = _make_validator()
    grid = MockGrid(block_set={(30, 0)})
    path = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]
    # Lookahead of 1.0m stops before the blocked cell at 1.5m
    blocked = pv.confirm_blocked(path, 0, grid, lookahead=1.0)
    # No blocked within 1.0m
    assert not blocked
    print("[PASS] test_partial_segment_check")


def test_consecutive_confirmation():
    pv = _make_validator()
    grid = MockGrid(block_set={(10, 0)})
    path = [[0.0, 0.0], [1.0, 0.0]]
    # First two calls are not confirmed (need 3 consecutive)
    assert not pv.confirm_blocked(path, 0, grid)
    assert not pv.confirm_blocked(path, 0, grid)
    # Third call confirms
    assert pv.confirm_blocked(path, 0, grid)
    print("[PASS] test_consecutive_confirmation")


def test_reset_clears_counter():
    pv = _make_validator()
    grid = MockGrid(block_set={(10, 0)})
    path = [[0.0, 0.0], [1.0, 0.0]]
    pv.confirm_blocked(path, 0, grid)
    pv.confirm_blocked(path, 0, grid)
    pv.reset()
    assert pv.consecutive_blocked == 0
    # After reset, first blocked call is not yet confirmed
    assert not pv.confirm_blocked(path, 0, grid)
    print("[PASS] test_reset_clears_counter")


def test_endpoint_included():
    pv = _make_validator()
    # Block exactly the endpoint cell (mx=20, my=0 for x=1.0)
    grid = MockGrid(block_set={(20, 0)})
    assert pv.check_segment([0.0, 0.0], [1.0, 0.0], grid)
    print("[PASS] test_endpoint_included")


if __name__ == "__main__":
    test_clear_segment()
    test_blocked_segment()
    test_partial_segment_check()
    test_consecutive_confirmation()
    test_reset_clears_counter()
    test_endpoint_included()
    print("[ALL PASS] PathValidator unit tests")
