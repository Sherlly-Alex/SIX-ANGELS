"""Phase-D navigation metric gates."""

import math
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.nav_metrics import evaluate_nav_metrics, summarize_metrics
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    SpeedLimits,
)
from navigation.occupancy_grid import build_material_scene_grid


def _limits():
    return SpeedLimits(
        max_linear=0.3, max_angular=0.6,
        max_linear_accel=2.0, max_angular_accel=3.0,
        emergency_clearance=0.1, max_deceleration=0.5,
    )


def _goal(x, y, yaw=0.0):
    return NavigationGoal(
        x=x, y=y, yaw=yaw,
        position_tolerance=0.06, yaw_tolerance=0.03,
        safety_radius=0.5, segment=NavigationSegment.NAV_SHELF,
        source_tag="tel_test",
    )


def test_evaluate_passes_clean_stream():
    rows = [
        {
            "status": "navigating",
            "footprint_min_clearance": 0.25,
            "path_length": 1.2,
            "planned_straight": 1.0,
            "straight_distance": 0.4,
        },
        {
            "status": "final_aligning",
            "footprint_min_clearance": 0.18,
            "path_length": 1.2,
            "planned_straight": 1.0,
            "straight_distance": 0.05,
        },
    ]
    report = evaluate_nav_metrics(rows)
    assert report["ok"]
    assert "PASS" in summarize_metrics(report)


def test_evaluate_flags_zero_clearance_and_detour():
    rows = [
        {
            "status": "navigating",
            "footprint_min_clearance": 0.0,
            "path_length": 3.0,
            "planned_straight": 1.0,
            "straight_distance": 0.2,
        },
    ]
    report = evaluate_nav_metrics(rows, max_detour_ratio=2.0)
    assert not report["ok"]
    assert report["clear_violations"] == 1
    assert report["detour_violations"] == 1


def test_evaluate_does_not_flag_shrinking_remaining_distance():
    """Live straight_distance shrinks near the goal; planned chord must win."""
    rows = [
        {
            "status": "navigating",
            "footprint_min_clearance": 0.2,
            "path_length": 0.9,
            "planned_straight": 0.8,
            "straight_distance": 0.2,  # would look like 4.5× without planned
        },
    ]
    report = evaluate_nav_metrics(rows, max_detour_ratio=2.0)
    assert report["ok"]
    assert report["max_detour_ratio_seen"] == pytest.approx(0.9 / 0.8)


def test_controller_telemetry_populated_on_update():
    grid = build_material_scene_grid()
    ctrl = NavigationController(grid, _limits())
    assert ctrl.set_goal(_goal(-1.5, 0.70), -0.70, 0.55)
    cmd = ctrl.update(-0.70, 0.55, 0.0, 0.05)
    tel = ctrl.telemetry
    assert tel.status == NavigationStatus.NAVIGATING.value
    assert tel.footprint_min_clearance > 0.0
    assert math.isfinite(tel.lookahead)
    assert math.isfinite(tel.kappa)
    assert math.isfinite(tel.path_length)
    assert tel.straight_distance > 0.5
    assert tel.segment == NavigationSegment.NAV_SHELF.value
    assert math.isfinite(cmd.linear_x) and math.isfinite(cmd.angular_z)


def test_idle_update_still_records_telemetry():
    grid = build_material_scene_grid()
    ctrl = NavigationController(grid, _limits())
    ctrl.update(0.0, 0.0, 0.0, 0.05)
    assert ctrl.telemetry.status == NavigationStatus.IDLE.value


def test_evaluate_ignores_wrong_case_status():
    """Uppercase status strings must not silently skip / falsely trip gates."""
    rows = [
        {
            "status": "NAVIGATING",
            "footprint_min_clearance": 0.0,
            "path_length": 9.0,
            "straight_distance": 1.0,
        },
    ]
    report = evaluate_nav_metrics(rows)
    assert report["ok"]
    assert report["navigating_ticks"] == 0
