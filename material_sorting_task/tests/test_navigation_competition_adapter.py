from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from executors.base import TargetObservation
from executors.transfer_support import TransferMotion
from navigation.competition_adapter import (
    format_nav_telemetry,
    goal_reached_event,
    refresh_dynamic_overlay,
)
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    SpeedLimits,
)
from navigation.occupancy_grid import build_layered_scene_grid
from navigation.robot_geometry import FootprintMode


def odometry(x: float, y: float, yaw: float):
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            )
        )
    )


def goal() -> NavigationGoal:
    return NavigationGoal(
        -0.18,
        1.55,
        math.pi / 2.0,
        0.08,
        0.05,
        0.65,
        NavigationSegment.NAV_TABLE,
        "competition_adapter_test",
    )


class CompetitionNavigationAdapterTests(unittest.TestCase):
    def test_public_detections_build_overlay_and_exclude_active_target(self) -> None:
        grid = build_layered_scene_grid()
        observations = {
            "pink": TargetObservation("pink", (-1.00, 2.20, 0.834), 1.0, score=0.9),
            "packaging_box": TargetObservation(
                "packaging_box", (-2.20, 1.20, 0.53), 1.0, score=0.8
            ),
        }

        count = refresh_dynamic_overlay(
            grid,
            observations,
            exclude_color="pink",
            robot_xy=(-0.70, 0.55),
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(grid.dynamic_volumes), 1)

    def test_transfer_selects_carry_mode_before_planning(self) -> None:
        motion = TransferMotion()

        started = motion.begin_navigation(
            goal(),
            odometry(-0.70, 0.55, math.pi / 2.0),
            footprint_mode=FootprintMode.TRANSIT_CARRY,
        )

        self.assertTrue(started)
        self.assertEqual(motion._navigation.footprint_mode, FootprintMode.TRANSIT_CARRY)
        self.assertEqual(
            motion._navigation.safety_footprint_mode,
            FootprintMode.TRANSIT_CARRY,
        )

    def test_terminal_entry_does_not_downgrade_carry_envelope(self) -> None:
        grid = build_layered_scene_grid()
        nav = NavigationController(
            grid,
            SpeedLimits(0.20, 0.65, 0.35, 1.20, 0.20, 0.50),
            pos_tolerance=0.08,
            yaw_tolerance=0.05,
            emergency_distance=0.20,
            footprint_mode=FootprintMode.TRANSIT_CARRY,
        )
        target = goal()
        self.assertTrue(nav.set_goal(target, -0.18, 1.70))

        nav.update(-0.18, 1.70, math.pi / 2.0, 0.05)

        self.assertEqual(nav.footprint_mode, FootprintMode.TRANSIT_CARRY)
        self.assertEqual(nav.safety_footprint_mode, FootprintMode.TRANSIT_CARRY)
        self.assertEqual(nav.telemetry.footprint_mode, "transit_carry")

    def test_goal_event_and_telemetry_match_phase_d_contract(self) -> None:
        target = goal()
        self.assertEqual(goal_reached_event(target), "NAV_GOAL_REACHED segment=nav_table")
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_navigation(
                target,
                odometry(-0.18, 1.55, math.pi / 2.0),
            )
        )
        _status, _command, detail = motion.tick_navigation(
            odometry(-0.18, 1.55, math.pi / 2.0),
            0.05,
        )
        self.assertIn("NAV_GOAL_REACHED segment=nav_table", detail)
        self.assertIn("NAV_TEL phase=transfer", detail)
        self.assertIn("footprint=transit_stowed", detail)
        self.assertIn(
            "NAV_TEL phase=manual",
            format_nav_telemetry(motion._navigation.telemetry, phase="manual"),
        )

    def test_formal_client_has_no_private_layout_or_ground_truth_topic(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
        paths = [root / "client_task.py", root / "executors", root / "navigation"]
        source = []
        for path in paths:
            files = [path] if path.is_file() else path.rglob("*.py")
            for file in files:
                source.append(file.read_text(encoding="utf-8"))
        joined = "\n".join(source)
        self.assertNotIn("/material/task_layout", joined)
        self.assertNotIn("/material/gt_objects", joined)


if __name__ == "__main__":
    unittest.main()
