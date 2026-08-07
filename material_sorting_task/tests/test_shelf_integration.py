from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from control_types import ArmCommand
from desktop_grasp.pregrasp_core import SPINE_MIN
from executors import build_task_executors
from executors.base import ExecutionContext, StageStatus, TargetObservation, TaskStage
from executors.task1_full import (
    Task1IntegratedExecutor,
    shelf_observation_stand,
    target_delta_in_heading,
)
from executors.task2 import Task2IntegratedExecutor
from executors.transfer_support import TransferMotion, stand_from_held_center
from navigation.carried_envelope import CarriedEnvelopeChecker
from navigation.navigation_types import NavigationGoal, NavigationSegment
from navigation.navigation_types import NavigationStatus
from shelf.manipulation import (
    ArmRetractController,
    HeldTransportController,
    ShelfOpenPregraspController,
)
from shelf.state_tracker import ShelfStateTracker
from shelf.target_center import StableTargetCenterTracker
from shelf.task_memory import CompetitionTaskMemory


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


class StableTargetCenterTrackerTests(unittest.TestCase):
    def test_locks_full_object_center_and_rejects_one_spatial_outlier(self) -> None:
        tracker = StableTargetCenterTracker()
        tracker.reset(accept_after_s=10.5)
        samples = [
            # This pre-settle frame must not enter the fresh observation set.
            (10.4, (-2.74, 0.90, 0.84)),
            (10.6, (-2.671, 0.814, 0.842)),
            (10.8, (-2.669, 0.816, 0.844)),
            (11.0, (-2.672, 0.815, 0.843)),
            # A colour/depth edge fit in the same broad shelf ROI.
            (11.2, (-2.742, 0.875, 0.842)),
            (11.4, (-2.670, 0.813, 0.841)),
            (11.6, (-2.673, 0.817, 0.845)),
            (11.8, (-2.668, 0.815, 0.843)),
        ]
        result = None
        for stamp, center in samples:
            result = tracker.update(
                observation("brown", center, stamp),
                now_s=stamp,
                reference_layer_z=0.837,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.sample_count, 6)
        self.assertAlmostEqual(result.center_world[0], -2.6705, places=3)
        self.assertAlmostEqual(result.center_world[1], 0.815, places=3)
        self.assertAlmostEqual(result.center_world[2], 0.843, places=3)
        self.assertLess(result.max_axis_deviation[1], 0.003)

    def test_ignores_stale_fast_duplicate_and_wrong_layer_frames(self) -> None:
        tracker = StableTargetCenterTracker()
        tracker.reset(accept_after_s=20.0)

        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 19.9),
            now_s=20.0,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 20.0),
            now_s=21.0,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 20.1),
            now_s=20.1,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 1.16), 20.3),
            now_s=20.3,
            reference_layer_z=0.84,
        )

        self.assertEqual(tracker.sample_count, 1)


class IntegratedExecutorWiringTests(unittest.TestCase):
    def test_task12_mode_uses_one_shared_memory_instance(self) -> None:
        executors = build_task_executors("task12_full")
        self.assertIsInstance(executors[1], Task1IntegratedExecutor)
        self.assertIsInstance(executors[2], Task2IntegratedExecutor)
        self.assertIs(executors[1]._memory, executors[2]._memory)

    def test_task2_uses_narrow_shelf_pregrasp_controller(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        self.assertIsInstance(executor._pregrasp, ShelfOpenPregraspController)
        self.assertAlmostEqual(executor._pregrasp.half_width, 0.18, places=6)

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

    def test_transfer_retreat_corrects_explicit_heading_before_translation(self) -> None:
        motion = TransferMotion()
        start_yaw = math.pi - 0.12
        self.assertTrue(
            motion.begin_retreat(
                _odom(-1.28, 0.78, start_yaw),
                0.30,
                heading_yaw=math.pi,
            )
        )
        done, command, _detail = motion.tick_retreat(
            _odom(-1.28, 0.78, start_yaw)
        )
        self.assertFalse(done)
        self.assertEqual(command[0], 0.0)
        self.assertGreater(command[1], 0.0)

        done, command, _detail = motion.tick_retreat(
            _odom(-1.28, 0.78, math.pi)
        )
        self.assertFalse(done)
        self.assertLess(command[0], 0.0)
        done, command, _detail = motion.tick_retreat(
            _odom(-0.98, 0.78, math.pi)
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

    def test_task2_raises_to_physical_top_before_table_navigation(self) -> None:
        class RecordingSlideHold:
            def __init__(self) -> None:
                self.target_slide = None
                self.command = None

            @property
            def planned(self) -> bool:
                return self.target_slide is not None

            def plan(self, hold_command, target_slide, _joint_states):
                self.target_slide = float(target_slide)
                self.command = hold_command
                return hold_command

            def update(self, _now_s, _joint_states):
                return self.command, False, "waiting for physical top"

        memory = CompetitionTaskMemory(
            task1_origin_world=(-0.20, 2.30, 0.84),
            task1_color="brown",
        )
        executor = Task2IntegratedExecutor(memory)
        hold = ArmCommand(
            spine_position=0.714,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )
        context = ExecutionContext(
            now_s=10.0,
            instruction={},
            task_index=2,
            attempt=1,
            odometry=_odom(-2.10, 0.85, math.pi),
            joint_states=_arm_joint_state(
                slide=hold.spine_position,
                head=hold.head_positions,
                left=hold.left_arm_positions,
                right=hold.right_arm_positions,
            ),
        )
        executor.enter_stage(TaskStage.TRANSPORT, context)
        executor._phase = "lift_for_table_transport"
        executor._held_arm_command = hold
        executor._held_center_base = (0.82, 0.0, 0.58)
        recording_slide = RecordingSlideHold()
        executor._slide_hold = recording_slide

        result = executor.tick(TaskStage.TRANSPORT, context)

        self.assertEqual(recording_slide.target_slide, SPINE_MIN)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertFalse(result.controls_base)
        self.assertIn("maximum transport height", result.message)

    def test_held_transport_starts_from_preloaded_command_and_keeps_grippers(self) -> None:
        class RecordingKdl:
            def __init__(self) -> None:
                self.calls = []

            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                self.calls.append(
                    (T_left.copy(), T_right.copy(), tuple(ref_pos), target_height)
                )
                return [list(ref_pos)]

        kdl = RecordingKdl()
        controller = HeldTransportController(kdl=kdl)
        hold = ArmCommand(
            spine_position=SPINE_MIN,
            head_positions=(0.11, 0.42),
            left_arm_positions=(0.3, -0.2, 0.4, 0.1, -0.3, 0.2),
            left_gripper_position=0.83,
            right_arm_positions=(-0.3, 0.2, -0.4, -0.1, 0.3, -0.2),
            right_gripper_position=0.79,
        )

        first = controller.plan(hold, (0.76, 0.03, 1.34), 0.118)

        self.assertEqual(first, hold)
        self.assertEqual(controller.waypoint_count, 4)
        self.assertEqual(len(kdl.calls), controller.waypoint_count)
        self.assertAlmostEqual(kdl.calls[0][3], SPINE_MIN, places=6)
        self.assertAlmostEqual(first.left_gripper_position, 0.83, places=6)
        self.assertAlmostEqual(first.right_gripper_position, 0.79, places=6)
        self.assertEqual(controller.target_center_base, (0.5, 0.0, 1.34))

    def test_carried_envelope_rejects_extended_direct_turn(self) -> None:
        checker = CarriedEnvelopeChecker()
        motion = TransferMotion()
        start = (-1.28, 0.78, math.pi)
        old_goal = NavigationGoal(
            -0.18,
            1.09,
            math.pi / 2.0,
            0.08,
            0.07,
            0.0,
            NavigationSegment.NAV_TABLE,
            "old_direct_table_mid",
        )
        self.assertTrue(motion.begin_navigation(old_goal, _odom(*start)))
        check = checker.check_path(
            start,
            motion.navigation_path,
            old_goal.yaw,
            (0.75, 0.0, 1.34),
            0.118,
        )
        self.assertFalse(check.safe)

    def test_extended_hold_segmented_task2_routes_pass_carried_envelope(self) -> None:
        checker = CarriedEnvelopeChecker()
        held = (0.631, -0.014, 1.34)
        half_width = 0.118
        start = (-1.28, 0.78, math.pi)
        # Both randomized task-1 source slots.  The base first reverses east
        # while still facing west, turns only west -> north at the destination
        # column, then advances north.  The arms and box remain fully extended.
        for place in ((-1.0, 2.20, 0.84), (-0.18, 2.20, 0.84)):
            stand_x, stand_y = stand_from_held_center(
                place, held, math.pi / 2.0
            )
            reverse_end = (stand_x, start[1])
            reverse = checker.check_fixed_heading_translation(
                start,
                reverse_end,
                held,
                half_width,
            )
            self.assertTrue(reverse.safe, msg=f"reverse: {reverse.detail}")

            rotation_pose = (stand_x, start[1], math.pi)
            rotation = checker.check_rotation(
                rotation_pose,
                math.pi / 2.0,
                held,
                half_width,
            )
            self.assertTrue(rotation.safe, msg=f"rotation: {rotation.detail}")

            entry_y = min(1.35, stand_y - 0.25)
            advance = checker.check_fixed_heading_translation(
                (stand_x, start[1], math.pi / 2.0),
                (stand_x, entry_y),
                held,
                half_width,
            )
            self.assertTrue(advance.safe, msg=f"advance: {advance.detail}")

            entry_pose = (stand_x, entry_y, math.pi / 2.0)
            final_check = checker.check_fixed_heading_translation(
                entry_pose,
                (stand_x, stand_y),
                held,
                half_width,
            )
            self.assertTrue(
                final_check.safe,
                msg=f"final placement: {final_check.detail}",
            )


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
