from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from executors.transfer_support import TransferMotion
from navigation.carried_envelope import HeldObjectGeometry
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    VelocityCommand,
)


def odometry(x: float, y: float, yaw: float):
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    position = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=position, orientation=orientation)
        )
    )


class FakeNavigation:
    """NavigationController double with a scripted path and command."""

    def __init__(self, path, *, command=None, status=NavigationStatus.NAVIGATING):
        self._path = list(path)
        self._command = command or VelocityCommand(0.0, 0.0)
        self._status = status
        self.reset_count = 0
        self.goal = None

    def reset(self) -> None:
        self.reset_count += 1

    def set_footprint_mode(self, mode, *, payload_z=None) -> None:
        del payload_z

    def set_goal(self, goal, x: float, y: float) -> bool:
        self.goal = (goal, x, y)
        return True

    @property
    def path(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._path)

    @property
    def status(self) -> NavigationStatus:
        return self._status

    @property
    def telemetry(self):
        return SimpleNamespace(
            planned_straight=1.0,
            path_length=1.0,
            status=self._status.value,
            segment="",
            x=0.0,
            y=0.0,
            yaw=0.0,
            dist_err=0.0,
            yaw_err=0.0,
            cmd_lin=0.0,
            cmd_ang=0.0,
            footprint_min_clearance=0.3,
            footprint_mode="transit_carry",
        )

    def update(self, x, y, yaw, dt, obs=None) -> VelocityCommand:
        del x, y, yaw, dt, obs
        return self._command


SHELF_FRONT_X = -2.465  # task1_full.SHELF_FRONT_X (calibrated scene front)
START_POSE = (-1.30, 0.85, math.pi)  # facing west toward the shelf
HELD = HeldObjectGeometry((0.70, 0.0, 0.90), 0.08, source="test")


def navigation_goal() -> NavigationGoal:
    return NavigationGoal(
        x=-1.60,
        y=0.85,
        yaw=math.pi,
        position_tolerance=0.05,
        yaw_tolerance=0.05,
        safety_radius=0.0,
        segment=NavigationSegment.NAV_SHELF,
        source_tag="held_geometry_test",
    )


class HeldObjectGeometryTests(unittest.TestCase):
    def test_valid_geometry_normalizes_and_freezes(self) -> None:
        geometry = HeldObjectGeometry([0.7, "0.0", 0.9], "0.08")
        self.assertEqual(geometry.center_base, (0.7, 0.0, 0.9))
        self.assertEqual(geometry.half_width_m, 0.08)
        with self.assertRaises(AttributeError):
            geometry.center_base = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_rejects_malformed_geometry(self) -> None:
        with self.assertRaises(ValueError):
            HeldObjectGeometry((0.7, 0.0), 0.08)
        with self.assertRaises(ValueError):
            HeldObjectGeometry((0.7, float("nan"), 0.9), 0.08)
        with self.assertRaises(ValueError):
            HeldObjectGeometry((0.7, 0.0, 0.9), 0.0)
        with self.assertRaises(ValueError):
            HeldObjectGeometry((0.7, 0.0, 0.9), -0.1)


class TransferHeldGeometryTests(unittest.TestCase):
    def _transfer(self, path, *, command=None) -> TransferMotion:
        transfer = TransferMotion()
        transfer._navigation = FakeNavigation(path, command=command)
        return transfer

    def test_without_geometry_keeps_historical_opt_in_behaviour(self) -> None:
        # The path sweeps the carried box through the shelf; without the
        # measured geometry the base-only A* check accepts it unchanged.
        transfer = self._transfer([(-1.60, 0.85), (-1.90, 0.85), (-2.20, 0.85)])
        started = transfer.begin_navigation(
            navigation_goal(),
            odometry(*START_POSE),
        )
        self.assertTrue(started)
        self.assertIsNone(transfer._held_geometry)

    def test_begin_rejects_path_that_sweeps_measured_box_through_shelf(self) -> None:
        transfer = self._transfer([(-1.60, 0.85), (-1.90, 0.85), (-2.20, 0.85)])
        started = transfer.begin_navigation(
            navigation_goal(),
            odometry(*START_POSE),
            held_geometry=HELD,
        )
        self.assertFalse(started)
        self.assertIsNone(transfer.goal)
        self.assertIsNone(transfer._held_geometry)

    def test_tick_gate_stops_unsafe_command_with_measured_envelope(self) -> None:
        transfer = self._transfer(
            [(-1.40, 0.85)],
            command=VelocityCommand(0.30, 0.0),
        )
        started = transfer.begin_navigation(
            navigation_goal(),
            odometry(*START_POSE),
            held_geometry=HELD,
        )
        self.assertTrue(started)

        # At x=-1.95 the held box centre (-2.65) is inside the shelf front.
        status, command, detail = transfer.tick_navigation(
            odometry(-1.95, 0.85, math.pi),
            1.0,
        )
        self.assertIs(status, NavigationStatus.EMERGENCY_STOP)
        self.assertEqual(command, (0.0, 0.0))
        self.assertIn("carried envelope guard", detail)
        self.assertIsNone(transfer.goal)

    def test_tick_gate_passes_safe_command(self) -> None:
        transfer = self._transfer(
            [(-1.40, 0.85)],
            command=VelocityCommand(0.10, 0.0),
        )
        started = transfer.begin_navigation(
            navigation_goal(),
            odometry(*START_POSE),
            held_geometry=HELD,
        )
        self.assertTrue(started)

        status, command, detail = transfer.tick_navigation(
            odometry(*START_POSE),
            1.0,
        )
        self.assertIs(status, NavigationStatus.NAVIGATING)
        self.assertEqual(command, (0.10, 0.0))
        self.assertIn("NAV_TEL", detail)


if __name__ == "__main__":
    unittest.main()
