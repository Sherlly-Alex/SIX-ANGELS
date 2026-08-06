from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from control_types import ArmCommand
from executors import build_task_executors
from executors.base import TargetObservation
from executors.task1_full import (
    Task1IntegratedExecutor,
    shelf_observation_stand,
    target_delta_in_heading,
)
from executors.task2 import Task2IntegratedExecutor
from executors.transfer_support import TransferMotion, stand_from_held_center
from navigation.navigation_types import NavigationStatus
from shelf.manipulation import ArmRetractController
from shelf.state_tracker import ShelfStateTracker


def observation(label: str, xyz, stamp: float, score: float = 0.9) -> TargetObservation:
    return TargetObservation(
        color=label,
        position_world=tuple(float(value) for value in xyz),
        received_at_s=float(stamp),
        score=float(score),
    )


class ShelfStateTrackerTests(unittest.TestCase):
    def test_infers_unique_empty_layer_and_filters_carried_color(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(3):
            stamp = 10.0 + index
            result = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                    # A carried task-1 box can be visible in front of the
                    # camera and must not become shelf occupancy evidence.
                    "pink": observation("pink", (-2.55, 0.78, 1.166), stamp),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.colored_class_id, "brown")
        self.assertEqual(result.colored_layer, 2)
        self.assertEqual(result.white_obstacle_layer, 1)
        self.assertEqual(result.empty_layer, 3)
        self.assertEqual(result.layer_contents, ("packaging_box", "brown", "EMPTY"))
        self.assertAlmostEqual(result.empty_place_world[2], 1.166, places=3)

    def test_fails_closed_when_two_semantics_vote_for_same_layer(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(4):
            stamp = 20.0 + index
            result = tracker.update(
                {
                    "yellow": observation("yellow", (-2.55, 0.78, 0.508), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNone(result)


class IntegratedExecutorWiringTests(unittest.TestCase):
    def test_task12_mode_uses_one_shared_memory_instance(self) -> None:
        executors = build_task_executors("task12_full")
        self.assertIsInstance(executors[1], Task1IntegratedExecutor)
        self.assertIsInstance(executors[2], Task2IntegratedExecutor)
        self.assertIs(executors[1]._memory, executors[2]._memory)

    def test_place_stand_preserves_held_object_transform(self) -> None:
        stand = stand_from_held_center(
            (-2.63, 0.778, 0.837),
            (0.65, 0.02, 0.95),
            3.141592653589793,
        )
        self.assertAlmostEqual(stand[0], -1.98, places=6)
        self.assertAlmostEqual(stand[1], 0.798, places=6)

    def test_shelf_scan_stand_keeps_carried_center_outside_front(self) -> None:
        held = (0.70, -0.025, 0.984)
        stand = shelf_observation_stand(
            held,
            shelf_front_x=-2.465,
            shelf_y=0.85,
            center_clearance_m=0.18,
        )
        # Facing west, the held center is base + R(pi) * held.
        carried_center_x = stand[0] - held[0]
        carried_center_y = stand[1] - held[1]
        self.assertAlmostEqual(carried_center_x, -2.285, places=6)
        self.assertAlmostEqual(carried_center_y, 0.85, places=6)
        self.assertGreater(carried_center_x, -2.465)

    def test_straight_shelf_delta_is_forward_when_target_is_west(self) -> None:
        forward, lateral = target_delta_in_heading(
            (-1.30, 0.825, 3.141592653589793),
            (-1.585, 0.825),
            3.141592653589793,
        )
        self.assertAlmostEqual(forward, 0.285, places=6)
        self.assertAlmostEqual(lateral, 0.0, places=6)

    def test_transfer_advance_holds_heading_and_finishes_by_odometry(self) -> None:
        motion = TransferMotion()
        self.assertTrue(motion.begin_advance(_odom(-1.30, 0.85, 3.141592653589793), 0.28))
        done, command, _detail = motion.tick_advance(
            _odom(-1.40, 0.85, 3.141592653589793)
        )
        self.assertFalse(done)
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0, places=6)
        done, command, _detail = motion.tick_advance(
            _odom(-1.58, 0.85, 3.141592653589793)
        )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0))

    def test_transfer_lateral_alignment_rotates_drives_then_restores_yaw(self) -> None:
        motion = TransferMotion()
        final_yaw = math.pi
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 1.00),
                final_yaw,
                _odom(-1.30, 0.85, final_yaw),
                0.0,
            )
        )

        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, final_yaw), 0.05
        )
        self.assertEqual(status.value, "navigating")
        self.assertEqual(command[0], 0.0)
        self.assertIn("rotating toward shelf-front", detail)

        status, command, _detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi / 2.0), 0.20
        )
        self.assertEqual(status.value, "navigating")
        self.assertGreater(command[0], 0.0)

        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.00, math.pi / 2.0), 2.0
        )
        self.assertEqual(status.value, "navigating")
        self.assertEqual(command[0], 0.0)
        self.assertIn("restoring shelf-facing yaw", detail)

        status, command, _detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.00, final_yaw), 3.0
        )
        self.assertEqual(status, NavigationStatus.GOAL_REACHED)
        self.assertEqual(command, (0.0, 0.0))

    def test_arm_retract_targets_neutral_posture_and_waits_for_stability(self) -> None:
        controller = ArmRetractController()
        hold = ArmCommand(
            spine_position=0.14,
            head_positions=(0.12, 0.20),
            left_arm_positions=(0.4, 0.3, 0.2, 0.1, -0.1, -0.2),
            left_gripper_position=1.0,
            right_arm_positions=(-0.4, -0.3, -0.2, -0.1, 0.1, 0.2),
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.14,
            head=(0.12, 0.20),
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )
        command = controller.plan(hold, feedback)
        self.assertAlmostEqual(command.spine_position, 0.14, places=6)
        self.assertEqual(command.left_arm_positions, hold.left_arm_positions)

        command, reached, detail = controller.update(0.0, feedback)
        self.assertFalse(reached)
        self.assertIn("transport retract", detail)
        self.assertLess(max(abs(value) for value in command.left_arm_positions), 0.4)

        neutral = _arm_joint_state(
            slide=0.10,
            head=(0.12, 0.20),
            left=(0.0,) * 6,
            right=(0.0,) * 6,
        )
        _command, reached, _detail = controller.update(0.10, neutral)
        self.assertFalse(reached)
        _command, reached, _detail = controller.update(0.70, neutral)
        self.assertFalse(reached)
        _command, reached, _detail = controller.update(1.30, neutral)
        self.assertTrue(reached)


def _odom(x: float, y: float, yaw: float):
    half = 0.5 * yaw
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(half),
                    w=math.cos(half),
                ),
            )
        )
    )


def _arm_joint_state(*, slide: float, head, left, right):
    names = [
        "slide_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        "left_arm_eef_gripper_joint",
        *(f"right_arm_joint{index}" for index in range(1, 7)),
        "right_arm_eef_gripper_joint",
    ]
    positions = [slide, *head, *left, 1.0, *right, 1.0]
    return SimpleNamespace(
        name=names,
        position=positions,
        velocity=[0.0] * len(names),
        effort=[0.0] * len(names),
    )


if __name__ == "__main__":
    unittest.main()
