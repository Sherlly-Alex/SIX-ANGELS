"""Unit tests for LocalGoalSelector."""

import sys
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from planning.local_goal_selector import select_local_goal


def test_selects_ahead_on_straight_path():
    """On a straight 5m path, lookahead 1.2m from wp=0 should pick ~(1.2, 0)."""
    path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]
    goal = select_local_goal((0, 0, 0), path, 0, lookahead_distance=1.2)
    dist = np.hypot(goal[0], goal[1])
    assert abs(dist - 1.2) < 0.05, f"Expected ~1.2, got {dist:.2f}"
    assert goal[1] == 0.0
    print("[PASS] test_selects_ahead_on_straight_path")


def test_falls_back_to_path_end():
    """If remaining path < lookahead, return last path point."""
    path = [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]]
    goal = select_local_goal((0, 0, 0), path, 0, lookahead_distance=2.0)
    assert goal[0] == 0.6
    assert goal[1] == 0.0
    print("[PASS] test_falls_back_to_path_end")


def test_index_advances():
    """Starting from wp=1 should skip the first segment."""
    path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    goal = select_local_goal((1, 0, 0), path, 1, lookahead_distance=0.5)
    # From wp=1 at (1,0), remaining: 1→2 (1.0m) + 2→3 (1.0m)
    # Lookahead 0.5 should land at ~(1.5, 0)
    assert abs(goal[0] - 1.5) < 0.05, f"Expected ~1.5, got {goal[0]:.2f}"
    assert goal[1] == 0.0
    print("[PASS] test_index_advances")


def test_empty_path_returns_current():
    path = []
    goal = select_local_goal((3.0, -2.0, 0.5), path, 0)
    assert goal[0] == 3.0
    assert goal[1] == -2.0
    print("[PASS] test_empty_path_returns_current")


def test_single_waypoint():
    path = [[0.0, 0.0]]
    goal = select_local_goal((0, 0, 0), path, 0)
    assert goal[0] == 0.0
    assert goal[1] == 0.0
    print("[PASS] test_single_waypoint")


def test_higher_index_moves_goal_forward():
    """A larger waypoint index must produce a local goal further along the path."""
    path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    # Robot is near the middle of the path
    robot = (2.1, 0.0, 0.0)

    # Simulate: index should advance past the first two waypoints
    idx_low = 0
    idx_high = 2

    goal_low = select_local_goal(robot, path, idx_low,
                                 lookahead_distance=1.2)
    goal_high = select_local_goal(robot, path, idx_high,
                                  lookahead_distance=1.2)
    # Starting from idx=0 accumulates from (0,0), so goal_low is ~1.2
    # Starting from idx=2 accumulates from (2,0), so goal_high is ~3.2
    assert goal_high[0] > goal_low[0] + 1.0, \
        f"idx_high goal {goal_high[0]} should be well ahead of idx_low {goal_low[0]}"
    # idx_high must never produce a point behind idx_low
    assert goal_high[0] > goal_low[0]
    print("[PASS] test_higher_index_moves_goal_forward")


def test_index_never_exceeds_path_end():
    """Even with index near the path end, result is capped at final point."""
    path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    # Index at last waypoint
    goal = select_local_goal((2, 0, 0), path, 2, lookahead_distance=5.0)
    assert goal[0] == 2.0
    assert goal[1] == 0.0
    print("[PASS] test_index_never_exceeds_path_end")


if __name__ == "__main__":
    test_selects_ahead_on_straight_path()
    test_falls_back_to_path_end()
    test_index_advances()
    test_empty_path_returns_current()
    test_single_waypoint()
    test_higher_index_moves_goal_forward()
    test_index_never_exceeds_path_end()
    print("[ALL PASS] LocalGoalSelector tests")
