from __future__ import annotations

import unittest
from types import SimpleNamespace

# Match the formal Client import order.  The shelf package re-exports tracker
# classes that depend on executors, so importing executors first avoids a
# package-initialisation cycle in standalone unittest runs.
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
    slide_position: float = 0.700,
    effort_value: float = 0.0,
    efforts_available: bool = True,
):
    efforts = [0.0] * len(JOINT_NAMES)
    for name in (
        "slide_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        *(f"right_arm_joint{index}" for index in range(1, 7)),
    ):
        efforts[JOINT_NAMES.index(name)] = effort_value
    positions = [0.0] * len(JOINT_NAMES)
    positions[JOINT_NAMES.index("slide_joint")] = slide_position
    return SimpleNamespace(
        name=JOINT_NAMES,
        position=positions,
        velocity=[0.0] * len(JOINT_NAMES),
        effort=efforts if efforts_available else [],
    )


class PlacementContactMonitorTests(unittest.TestCase):
    def _ready_monitor(self) -> PlacementContactMonitor:
        monitor = PlacementContactMonitor()
        for index in range(10):
            monitor.prepare_baseline(index * 0.05, joint_state())
        self.assertTrue(monitor.ready)
        return monitor

    def test_confirms_settled_slide_and_bilateral_arm_support(self) -> None:
        monitor = self._ready_monitor()
        confirmed = False
        for now_s in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            confirmed, _ = monitor.observe(
                now_s,
                joint_state(effort_value=2.0),
                motion_settled=True,
                contact_enabled=True,
            )
        self.assertTrue(confirmed)

    def test_rejects_effort_before_final_contact_window(self) -> None:
        monitor = self._ready_monitor()
        for now_s in (0.50, 0.60, 0.70, 0.80, 0.90):
            confirmed, _ = monitor.observe(
                now_s,
                joint_state(effort_value=3.0),
                motion_settled=True,
                contact_enabled=False,
            )
        self.assertFalse(confirmed)
        self.assertFalse(monitor.contact_candidate)

    def test_rejects_single_arm_or_slide_only_transient(self) -> None:
        monitor = self._ready_monitor()
        for now_s in (0.50, 0.60, 0.70, 0.80):
            state = joint_state()
            state.effort[JOINT_NAMES.index("slide_joint")] = 3.0
            state.effort[JOINT_NAMES.index("left_arm_joint6")] = 3.0
            confirmed, _ = monitor.observe(
                now_s,
                state,
                motion_settled=True,
                contact_enabled=True,
            )
        self.assertFalse(confirmed)

    def test_candidate_is_cleared_before_another_descent_step(self) -> None:
        monitor = self._ready_monitor()
        for now_s in (0.50, 0.55, 0.60):
            monitor.observe(
                now_s,
                joint_state(effort_value=3.0),
                motion_settled=True,
                contact_enabled=True,
            )
        self.assertTrue(monitor.contact_candidate)
        monitor.clear_candidate()
        self.assertFalse(monitor.contact_candidate)

    def test_falls_back_when_effort_is_unavailable(self) -> None:
        monitor = PlacementContactMonitor()
        ready, detail = monitor.prepare_baseline(
            0.0, joint_state(efforts_available=False)
        )
        self.assertTrue(ready)
        self.assertFalse(monitor.available)
        self.assertIn("geometry_fallback=true", detail)


class CompliantSlideLoweringControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hold = ArmCommand(
            spine_position=0.700,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )

    def test_uses_fast_approach_then_reaches_geometry_target(self) -> None:
        controller = CompliantSlideLoweringController()
        command = controller.plan(
            self.hold,
            0.755,
            joint_state(slide_position=self.hold.spine_position),
        )
        phases: set[str] = set()
        completed = False
        for index in range(320):
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
        self.assertIn("fine_descent", phases)
        self.assertEqual(controller.completion_reason, "geometry_target")
        self.assertAlmostEqual(command.spine_position, 0.755, places=6)

    def test_effort_missing_uses_original_geometry_target(self) -> None:
        controller = CompliantSlideLoweringController()
        command = controller.plan(
            self.hold,
            0.740,
            joint_state(
                slide_position=self.hold.spine_position,
                efforts_available=False,
            ),
        )
        completed = False
        for index in range(200):
            command, completed, _ = controller.update(
                index * 0.05,
                joint_state(
                    slide_position=command.spine_position,
                    efforts_available=False,
                ),
            )
            if completed:
                break
        self.assertTrue(completed)
        self.assertEqual(controller.completion_reason, "geometry_fallback")
        self.assertAlmostEqual(command.spine_position, 0.740, places=6)

    def test_support_effort_can_finish_inside_final_window(self) -> None:
        controller = CompliantSlideLoweringController()
        command = controller.plan(
            self.hold,
            0.712,
            joint_state(slide_position=self.hold.spine_position),
        )
        completed = False
        for index in range(200):
            now_s = index * 0.05
            effort = 3.0 if controller.phase in {"observe", "contact_confirm"} else 0.0
            command, completed, _ = controller.update(
                now_s,
                joint_state(
                    slide_position=command.spine_position,
                    effort_value=effort,
                ),
            )
            if completed:
                break
        self.assertTrue(completed)
        self.assertEqual(controller.completion_reason, "effort_contact")
        self.assertTrue(controller.contact_detected)


if __name__ == "__main__":
    unittest.main()
