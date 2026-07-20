"""Unit tests for EmergencyChecker."""
import sys
import os

_module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import numpy as np

from planning.emergency_checker import EmergencyChecker


def _make_checker():
    return EmergencyChecker(emergency_distance=0.35, front_half_angle=np.pi / 3)


def test_no_emergency_far():
    checker = _make_checker()
    ranges = np.full(360, 2.0)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    assert not checker.check(ranges, angles)
    print("[PASS] test_no_emergency_far")


def test_emergency_close():
    checker = _make_checker()
    ranges = np.full(360, 2.0)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    # Find a ray near angle 0 and set it close
    idx = np.argmin(np.abs(angles))
    ranges[idx] = 0.2
    assert checker.check(ranges, angles)
    print("[PASS] test_emergency_close")


def test_nan_inf_filtered():
    checker = _make_checker()
    ranges = np.full(360, np.nan)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    assert not checker.check(ranges, angles)

    ranges = np.full(360, np.inf)
    assert not checker.check(ranges, angles)
    print("[PASS] test_nan_inf_filtered")


def test_side_obstacle_ignored():
    checker = _make_checker()
    ranges = np.full(360, 2.0)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    # Obstacle at 90 degrees, outside front_half_angle=60 degrees
    idx = np.argmin(np.abs(angles - np.pi / 2))
    ranges[idx] = 0.1
    assert not checker.check(ranges, angles)
    print("[PASS] test_side_obstacle_ignored")


if __name__ == "__main__":
    test_no_emergency_far()
    test_emergency_close()
    test_nan_inf_filtered()
    test_side_obstacle_ignored()
    print("[ALL PASS] EmergencyChecker unit tests")
