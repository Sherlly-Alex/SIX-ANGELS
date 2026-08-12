from __future__ import annotations

import unittest
from types import SimpleNamespace

# Load the executor package first, matching the formal client import order.
# The existing shelf package re-exports executor-dependent tracker classes.
import executors  # noqa: F401
from control_types import ArmCommand
from shelf.placement_feedback import (
    CompliantSlideLoweringController,
    PlacementContactMonitor,
)


JOINT_NAMES = [
    "slide_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    *(f"left_arm_joint{index}" for index in range(1, 7)),
    "left_arm_eef_gripper_joint",
    *(f"right_arm_joint{index}" for index in range(1, 7)),
    "right_arm_eef_gripper_joint",
]


def joint_state(
    *,
    effort_value: float = 0.0,
    efforts_available: bool = True,
    slide_position: float = 0.0,
):
    efforts = [0.0] * len(JOINT_NAMES)
    for name in (
        "slide_joint",
        "left_arm_joint6",
        "right_arm_joint6",
    ):
        efforts[JOINT_NAMES.index(name)] = effort_value
    return SimpleNamespace(
        name=JOINT_NAMES,
        position=[slide_position, *([0.0] * (len(JOINT_NAMES) - 1))],
        velocity=[0.0] * len(JOINT_NAMES),
        effort=efforts if efforts_available else [],
    )


class PlacementContactMonitorTests(unittest.TestCase):
    def test_uses_official_joint_state_effort_for_bilateral_contact(self) -> None:
        monitor = PlacementContactMonitor()

        ready = False
        for index in range(10):
            ready, _ = monitor.prepare_baseline(index * 0.05, joint_state())
        self.assertTrue(ready)
        self.assertTrue(monitor.available)

        confirmed = False
        for now_s in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            confirmed, _ = monitor.observe(
                now_s,
                joint_state(effort_value=2.0),
                motion_settled=True,
            )

        self.assertTrue(confirmed)
        self.assertTrue(monitor.contact_confirmed)

    def test_falls_back_when_official_effort_array_is_missing(self) -> None:
        monitor = PlacementContactMonitor()

        ready, detail = monitor.prepare_baseline(
            0.0,
            joint_state(efforts_available=False),
        )

        self.assertTrue(ready)
        self.assertFalse(monitor.available)
        self.assertIn("geometry_fallback=true", detail)

    def test_does_not_confirm_during_motion(self) -> None:
        monitor = PlacementContactMonitor()
        for index in range(10):
            monitor.prepare_baseline(index * 0.05, joint_state())

        for now_s in (0.50, 0.60, 0.70, 0.80, 0.90):
            confirmed, _ = monitor.observe(
                now_s,
                joint_state(effort_value=3.0),
                motion_settled=False,
            )

        self.assertFalse(confirmed)


class CompliantSlideLoweringControllerTests(unittest.TestCase):
    def test_uses_fast_approach_then_completes_at_geometry_target(self) -> None:
        controller = CompliantSlideLoweringController()
        command = ArmCommand(
            spine_position=0.700,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.0,
        )
        controller.plan(
            command,
            0.755,
            joint_state(slide_position=command.spine_position),
        )

        phases: set[str] = set()
        completed = False
        for index in range(240):
            now_s = index * 0.05
            command, completed, _ = controller.update(
                now_s,
                joint_state(slide_position=command.spine_position),
            )
            phases.add(controller.phase)
            if completed:
                break

        self.assertTrue(completed)
        self.assertIn("fast_approach", phases)
        self.assertEqual(controller.completion_reason, "geometry_target")
        self.assertAlmostEqual(command.spine_position, 0.755, places=6)
        self.assertLess(now_s, 10.0)


if __name__ == "__main__":
    unittest.main()
