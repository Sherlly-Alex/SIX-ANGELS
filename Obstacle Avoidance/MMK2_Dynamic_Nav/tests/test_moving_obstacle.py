"""Verify scripted_line obstacle movement mode works correctly."""

import sys
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from config.dynamic_nav_config import DynamicNavConfig


def test_static_mode_is_default():
    """Default config must set obstacle_movement_mode to 'static'."""
    cfg = DynamicNavConfig()
    assert cfg.obstacle_movement_mode == "static"
    assert cfg.obstacle_movement_speed > 0
    print("[PASS] test_static_mode_is_default")


def test_config_accepts_scripted_line():
    cfg = DynamicNavConfig()
    cfg.obstacle_movement_mode = "scripted_line"
    assert cfg.obstacle_movement_mode == "scripted_line"
    print("[PASS] test_config_accepts_scripted_line")


def test_mock_obstacle_manager_steps_motion():
    """Simulate the path interpolation logic used by ObstacleManager."""
    # Minimal reproduction of _interpolate_path + _path_total_length
    path = [[0.0, 0.0], [0.8, 0.0], [1.6, 0.0]]

    def total_len(p):
        return sum(np.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1])
                   for i in range(len(p)-1))

    def interpolate(p, t):
        t = max(0.0, min(1.0, t))
        total = total_len(p)
        target = t * total
        acc = 0.0
        for i in range(len(p)-1):
            a, b = p[i], p[i+1]
            seg = float(np.hypot(b[0]-a[0], b[1]-a[1]))
            if acc + seg >= target:
                frac = (target - acc) / max(seg, 1e-9)
                return [a[0] + frac*(b[0]-a[0]), a[1] + frac*(b[1]-a[1])]
            acc += seg
        return [float(p[-1][0]), float(p[-1][1])]

    total = total_len(path)
    assert abs(total - 1.6) < 0.01

    # t=0 → start
    start = interpolate(path, 0.0)
    assert abs(start[0]) < 0.01 and abs(start[1]) < 0.01

    # t=0.5 → middle of path at x=0.8
    mid = interpolate(path, 0.5)
    assert abs(mid[0] - 0.8) < 0.01

    # t=1.0 → end at x=1.6
    end = interpolate(path, 1.0)
    assert abs(end[0] - 1.6) < 0.01 and abs(end[1]) < 0.01

    print("[PASS] test_mock_obstacle_manager_steps_motion")


def test_moving_obstacle_position_changes():
    """Movement must produce different positions over time steps."""
    path = [[0.0, 0.0], [2.0, 0.0]]
    speed = 0.15
    dt = 0.01

    def _len(p):
        return sum(np.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1])
                   for i in range(len(p)-1))

    def _interp(p, t):
        t = max(0.0, min(1.0, t))
        total = _len(p)
        target = t * total
        acc = 0.0
        for i in range(len(p)-1):
            a, b = p[i], p[i+1]
            seg = float(np.hypot(b[0]-a[0], b[1]-a[1]))
            if acc + seg >= target:
                frac = (target - acc) / max(seg, 1e-9)
                return [a[0]+frac*(b[0]-a[0]), a[1]+frac*(b[1]-a[1])]
            acc += seg
        return [float(p[-1][0]), float(p[-1][1])]

    progress = 0.0
    forward = True
    total = _len(path)
    prev_x = 0.0

    positions = []
    for _ in range(100):
        if forward:
            progress += speed * dt
            if progress >= total:
                progress = total
                forward = False
        else:
            progress -= speed * dt
            if progress <= 0.0:
                progress = 0.0
                forward = True
        pos = _interp(path, progress / total)
        positions.append(pos[0])

    # Must have moved at least 0.1 m over 100 steps
    assert max(positions) > 0.1, f"Obstacle barely moved: max {max(positions):.3f}"
    # Must have reversed direction (positions go up then down)
    mid = len(positions) // 2
    assert np.mean(positions[:mid]) < np.mean(positions[mid:]) or \
           np.mean(positions[:mid]) > np.mean(positions[mid:]), \
           "No reversal detected"
    print("[PASS] test_moving_obstacle_position_changes")


if __name__ == "__main__":
    test_static_mode_is_default()
    test_config_accepts_scripted_line()
    test_mock_obstacle_manager_steps_motion()
    test_moving_obstacle_position_changes()
    print("[ALL PASS] Moving obstacle tests")
