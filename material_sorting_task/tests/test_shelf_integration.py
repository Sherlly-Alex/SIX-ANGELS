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
from executors.task3 import Task3IntegratedExecutor
from executors.transfer_support import TransferMotion, stand_from_held_center
from navigation.carried_envelope import CarriedEnvelopeChecker
from navigation.navigation_types import NavigationGoal, NavigationSegment
from navigation.navigation_types import NavigationStatus
from shelf.manipulation import (
    ArmRetractController,
    HeldTransportController,
    ReleaseSpreadController,
    ShelfOpenPregraspController,
    SlideHoldController,
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
        # Occupied-object centers come from their own RGB-D observations;
        # only the empty slot uses calibrated shelf geometry.
        self.assertEqual(result.task2_target_center_world, (-2.55, 0.81, 0.837))
        self.assertEqual(
            result.task3_packaging_box_center_world,
            (-2.54, 0.78, 0.530),
        )
        self.assertAlmostEqual(result.empty_place_world[2], 1.166, places=3)
        self.assertAlmostEqual(result.empty_shelf_center_world[0], -2.63, places=3)
        self.assertAlmostEqual(result.empty_shelf_center_world[1], 0.778, places=3)

    def test_desktop_material_box_is_not_part_of_shelf_state(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(3):
            stamp = 30.0 + index
            result = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                    # This is a desktop obstacle and must not create another
                    # shelf layer or enter the shared shelf-state cache.
                    "material_box": observation(
                        "material_box", (-0.54, 2.30, 0.833), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.layer_contents, ("packaging_box", "brown", "EMPTY"))

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

    def test_shared_memory_keeps_only_three_shelf_centers(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        state = None
        for index in range(3):
            stamp = 40.0 + index
            state = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        assert state is not None
        memory = CompetitionTaskMemory()
        memory.record_shelf_state(state)
        self.assertEqual(memory.require_empty_shelf_center(), state.empty_shelf_center_world)
        self.assertEqual(memory.require_task2_target_center(), state.task2_target_center_world)
        self.assertEqual(
            memory.require_task3_packaging_box_center(),
            state.task3_packaging_box_center_world,
        )
        self.assertIsNone(getattr(memory, "material_box_center_world", None))

    def test_task2_uses_narrow_shelf_pregrasp_controller(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        self.assertIsInstance(executor._pregrasp, ShelfOpenPregraspController)
        self.assertAlmostEqual(executor._pregrasp.half_width, 0.18, places=6)
        self.assertAlmostEqual(
            executor._transfer.LATERAL_POSITION_TOLERANCE_M,
            executor.SHELF_ALIGNMENT_Y_TOLERANCE_M,
            places=6,
        )

    def test_task2_final_pregrasp_refreshes_far_staging_odometry(self) -> None:
        class RecordingPregrasp:
            def __init__(self, command) -> None:
                self.planned = False
                self.command = command
                self.plan_odometry = None

            def reset(self) -> None:
                self.planned = False

            def plan(self, _target, odometry, _joints):
                self.planned = True
                self.plan_odometry = odometry
                return self.command

            def update(self, _now_s, _joints):
                return self.command, False, "final shelf pregrasp moving"

        class CompletedAdvance:
            def tick_advance(self, _odometry):
                return True, (0.0, 0.0), "straight advance complete"

        command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )
        context = ExecutionContext(
            now_s=5.0,
            instruction={"target_color": "pink"},
            task_index=2,
            attempt=1,
            odometry=_odom(-1.90, 0.778, math.pi),
            joint_states=_arm_joint_state(
                slide=command.spine_position,
                head=command.head_positions,
                left=command.left_arm_positions,
                right=command.right_arm_positions,
            ),
        )
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        executor.enter_stage(TaskStage.ALIGN_FOR_PICK, context)
        executor._locked_target_world = (-2.63, 0.778, 0.50)
        executor._coarse_target_world = executor._locked_target_world
        executor._phase = "approach_pick"
        executor._motion_started = True
        stale_odometry = _odom(-1.50, 0.778, math.pi)
        executor._arm_reference_odometry = stale_odometry
        pregrasp = RecordingPregrasp(command)
        executor._pregrasp = pregrasp
        executor._transfer = CompletedAdvance()

        result = executor._tick_align_for_pick(context)

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIs(pregrasp.plan_odometry, context.odometry)
        self.assertIs(executor._arm_reference_odometry, context.odometry)
        self.assertIsNot(pregrasp.plan_odometry, stale_odometry)

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

    def test_task1_observation_stand_is_aligned_with_empty_slot_y(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        held = (0.70, -0.025, 0.984)
        target_y = executor._shelf_observation_target_y(SimpleNamespace())
        scan_stand = shelf_observation_stand(
            held,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=target_y,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        final_stand = stand_from_held_center(
            (-2.63, target_y, 0.837), held, executor.SHELF_YAW
        )
        self.assertAlmostEqual(target_y, 0.778, places=3)
        self.assertAlmostEqual(scan_stand[1], final_stand[1], places=6)

    def test_task1_transport_commands_direct_left_turn(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.0,
        )
        executor._held_center_base = (0.70, -0.02, 0.98)
        executor._phase = "turn_left_to_shelf"
        context = ExecutionContext(
            now_s=1.0,
            instruction={"target_color": "yellow"},
            task_index=0,
            attempt=1,
            odometry=_odom(-0.21, 0.75, math.pi / 2.0),
        )

        result = executor._tick_transport(context)

        self.assertAlmostEqual(executor.TABLE_RETREAT_M, 0.80, places=6)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertTrue(result.controls_base)
        self.assertEqual(result.base_linear_x, 0.0)
        self.assertAlmostEqual(
            result.base_angular_z,
            executor.SHELF_LEFT_TURN_MAX_SPEED_RADPS,
            places=6,
        )
        self.assertIn("LEFT", result.message)

    def test_slide_hold_ignores_contact_arm_velocity_after_slide_settles(
        self,
    ) -> None:
        controller = SlideHoldController()
        hold = ArmCommand(
            spine_position=0.40,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.40,
            head=(0.0, 0.0),
            left=(0.07,) * 6,
            right=(0.07,) * 6,
        )
        feedback.velocity = [
            0.0 if name == "slide_joint" else 0.05
            for name in feedback.name
        ]
        controller.plan(hold, 0.40, feedback)

        _command, reached, _detail = controller.update(0.0, feedback)
        self.assertFalse(reached)
        _command, reached, detail = controller.update(0.60, feedback)

        self.assertTrue(reached)
        self.assertIn("slide_vel=0.000", detail)
        self.assertIn("max_vel=0.050", detail)

    def test_task3_transport_targets_final_safe_left_row_directly(self) -> None:
        executor = Task3IntegratedExecutor(CompetitionTaskMemory())
        held = (0.58, 0.002, 1.104)
        target_y = executor._shelf_observation_target_y(SimpleNamespace())
        scan_stand = shelf_observation_stand(
            held,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=target_y,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        final_stand = stand_from_held_center(
            (executor.TASK3_RELEASE_X, executor.TASK3_SAFE_RELEASE_Y, 1.164),
            held,
            executor.SHELF_YAW,
        )
        self.assertAlmostEqual(target_y, executor.TASK3_SHELF_TURN_Y, places=6)
        self.assertAlmostEqual(scan_stand[1], executor.TASK3_SHELF_TURN_Y + held[1], places=6)
        self.assertAlmostEqual(final_stand[1], executor.TASK3_SAFE_RELEASE_Y + held[1], places=6)
        self.assertAlmostEqual(scan_stand[1], final_stand[1], places=6)
        self.assertAlmostEqual(target_y, 0.540, places=6)
        self.assertFalse(
            executor.FORCE_SHELF_FACING_TURN_BEFORE_NAVIGATION
        )
        self.assertAlmostEqual(
            executor.SHELF_SCAN_YAW_TOLERANCE_RAD, 0.120, places=6
        )

    def test_task3_packaging_left_target_has_physical_y_clearance(self) -> None:
        executor = Task3IntegratedExecutor(CompetitionTaskMemory())
        context = ExecutionContext(
            now_s=10.0,
            instruction={
                "task": 3,
                "target_color": "brown",
                "direction": "left",
            },
            task_index=2,
            attempt=1,
            target_observations={
                "packaging_box": observation(
                    "packaging_box", (-2.646, 0.778, 0.851), 10.0
                )
            },
        )

        target = executor._task3_place_from_rgbd(context)

        self.assertEqual(target[:2], (executor.TASK3_RELEASE_X, 0.540))
        self.assertAlmostEqual(target[2], 0.829, places=6)
        cuboid_gap_y = (
            0.778 - target[1] - (executor.PACKAGING_HALF_Z + 0.080)
        )
        self.assertGreaterEqual(cuboid_gap_y, 0.040)

    def test_task3_navigation_uses_stable_support_xy_directly(self) -> None:
        executor = Task3IntegratedExecutor(CompetitionTaskMemory())
        pose = _odom(-0.70, 0.55, math.pi / 2.0)

        def make_context(stamp: float, target_x: float) -> ExecutionContext:
            return ExecutionContext(
                now_s=stamp,
                instruction={
                    "task": 3,
                    "target_color": "brown",
                    "place_type": "packaging_left",
                },
                task_index=2,
                attempt=1,
                odometry=pose,
                target_observations={
                    "brown": observation(
                        "brown", (target_x, 2.30, 1.004), stamp
                    ),
                    "material_box": observation(
                        "material_box", (-0.54, 2.30, 0.833), stamp
                    ),
                },
            )

        first = make_context(0.0, -0.40)
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, first)
        for index, target_x in enumerate(
            (-0.40, -0.53, -0.55, -0.54, -0.54, -0.54)
        ):
            executor.tick(
                TaskStage.NAVIGATE_TO_PICK,
                make_context(index * 0.12, target_x),
            )

        self.assertIsNotNone(executor.goal)
        self.assertIsNotNone(executor._navigation_goal)
        self.assertAlmostEqual(executor.goal.x, -0.54, places=6)
        self.assertAlmostEqual(executor.goal.y, 1.68, places=6)
        self.assertAlmostEqual(executor._navigation_goal.x, -0.54, places=6)
        self.assertAlmostEqual(executor._navigation_goal.y, 1.38, places=6)

        executor.enter_stage(
            TaskStage.ALIGN_FOR_PICK, make_context(1.0, -0.54)
        )
        self.assertEqual(executor._task3_pick_phase, "face_target")

    def test_task3_safe_turn_and_lateral_skip_share_y_tolerance(self) -> None:
        executor = Task3IntegratedExecutor(CompetitionTaskMemory())
        self.assertAlmostEqual(
            executor.SHELF_TURN_POSITION_TOLERANCE_M,
            executor.TASK3_SHELF_Y_TOLERANCE_M,
            places=6,
        )
        self.assertAlmostEqual(
            executor._transfer.LATERAL_POSITION_TOLERANCE_M,
            executor.TASK3_SHELF_Y_TOLERANCE_M,
            places=6,
        )

    def test_task2_navigation_prefers_fused_y_over_live_drift(self) -> None:
        memory = CompetitionTaskMemory(
            task2_target_center_world=(-2.65, 0.778, 0.837)
        )
        executor = Task2IntegratedExecutor(memory)
        context = SimpleNamespace(
            instruction={"target_color": "brown"},
            target_observations={
                "brown": observation(
                    "brown", (-2.65, 0.958, 0.837), 10.0
                )
            },
        )
        color, target = executor._task2_target(context)
        self.assertEqual(color, "brown")
        self.assertAlmostEqual(target[0], -2.63, places=6)
        self.assertAlmostEqual(target[1], 0.778, places=6)

    def test_already_y_aligned_motion_skips_lateral_rotation(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 0.778),
                math.pi,
                _odom(-1.30, 0.778, math.pi),
                0.0,
            )
        )
        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.778, math.pi), 0.05
        )
        self.assertEqual(status, NavigationStatus.GOAL_REACHED)
        self.assertEqual(command, (0.0, 0.0))
        self.assertIn("shelf-facing yaw restored", detail)

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

    def test_transfer_advance_honors_stricter_completion_tolerance(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_advance(
                _odom(0.0, 0.0, 0.0),
                0.30,
                completion_tolerance_m=0.005,
            )
        )
        done, command, _detail = motion.tick_advance(
            _odom(0.291, 0.0, 0.0)
        )
        self.assertFalse(done)
        self.assertGreater(command[0], 0.0)
        done, command, _detail = motion.tick_advance(
            _odom(0.296, 0.0, 0.0)
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

    def test_task2_retracts_arms_after_table_retreat_before_end_navigation(self) -> None:
        memory = CompetitionTaskMemory(
            task1_origin_world=(-0.20, 2.30, 0.84),
            task1_color="brown",
        )
        executor = Task2IntegratedExecutor(memory)
        hold = ArmCommand(
            spine_position=0.14,
            head_positions=(0.12, 0.20),
            left_arm_positions=(0.4, 0.3, 0.2, 0.1, -0.1, -0.2),
            left_gripper_position=0.0,
            right_arm_positions=(-0.4, -0.3, -0.2, -0.1, 0.1, 0.2),
            right_gripper_position=0.0,
        )
        context = ExecutionContext(
            now_s=10.0,
            instruction={},
            task_index=2,
            attempt=1,
            odometry=_odom(-0.70, 0.55, math.pi / 2.0),
            joint_states=_arm_joint_state(
                slide=hold.spine_position,
                head=hold.head_positions,
                left=hold.left_arm_positions,
                right=hold.right_arm_positions,
            ),
        )
        executor.enter_stage(TaskStage.RETURN_TO_END, context)
        executor._held_arm_command = hold
        # Simulate that the base retreat has already completed.  The first
        # return tick must command the retract controller and must not start
        # end-zone navigation yet.
        executor._phase = "retract_arms"
        result = executor.tick(TaskStage.RETURN_TO_END, context)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIn("retracting arms", result.message)
        self.assertFalse(result.controls_base)
        self.assertIsNotNone(result.arm_command)
        assert result.arm_command is not None
        self.assertLess(
            max(abs(value) for value in result.arm_command.left_arm_positions),
            max(abs(value) for value in hold.left_arm_positions),
        )

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

    def test_relative_release_keeps_achieved_center_and_spine(self) -> None:
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
        controller = ReleaseSpreadController(kdl=kdl)
        hold = ArmCommand(
            spine_position=0.804,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.1,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(-0.1,) * 6,
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.804,
            head=hold.head_positions,
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )

        command = controller.plan_from_held(
            hold,
            (0.897, -0.036, 0.498),
            feedback,
            half_width=0.114,
        )

        self.assertEqual(controller.target_base, (0.897, -0.036, 0.498))
        self.assertAlmostEqual(command.spine_position, 0.804, places=6)
        self.assertEqual(len(kdl.calls), 1)
        left_target, right_target, _reference, target_height = kdl.calls[0]
        self.assertAlmostEqual(target_height, 0.804, places=6)
        self.assertAlmostEqual(
            left_target[1, 3] - right_target[1, 3],
            2.0 * 0.114,
            places=6,
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
