from __future__ import annotations

import math
import unittest

from control_types import ArmCommand
from executors import build_task_executors
from executors.base import ExecutionContext, TaskStage
from executors.task3 import Task3IntegratedExecutor
from shelf.state_tracker import ShelfState
from shelf.task3_geometry import (
    TASK3_LEFT_CENTER_OFFSET_M,
    task3_safe_release_target,
    task3_scoring_target,
)
from shelf.task_memory import CompetitionTaskMemory
from shelf_geometry import load_shelf_geometry


class Task3GeometryTests(unittest.TestCase):
    def test_left_target_uses_measured_packaging_y_and_detected_layer(self) -> None:
        geometry = load_shelf_geometry()
        target = task3_scoring_target(
            (-2.54, 0.778, 0.530),
            1,
            geometry=geometry,
        )
        self.assertAlmostEqual(
            target[0], geometry.shelf_xy[0] - 0.05, places=6
        )
        self.assertAlmostEqual(
            target[1], 0.778 - TASK3_LEFT_CENTER_OFFSET_M, places=6
        )
        self.assertAlmostEqual(
            target[2], geometry.object_center_z_on_board(1, half_z=0.095), places=6
        )

    def test_safe_release_stays_outward_and_preserves_left_clearance(self) -> None:
        scoring = (-2.68, 0.540, 0.498)
        release = task3_safe_release_target(scoring, place_radius_m=0.24)
        self.assertAlmostEqual(release[0], -2.51, places=6)
        self.assertAlmostEqual(release[1], 0.540, places=6)
        self.assertEqual(release[2], scoring[2])
        # Server task 3 uses a 0.24 m XY placement radius.  Keep explicit
        # margin instead of relying on a point right at its boundary.
        self.assertLess(math.dist(release[:2], scoring[:2]), 0.18)

    def test_safe_release_adapts_to_instruction_radius_and_opening_yaw(self) -> None:
        scoring = (1.0, 2.0, 0.8)
        release = task3_safe_release_target(
            scoring,
            place_radius_m=0.12,
            opening_yaw=math.pi / 2.0,
        )
        # Radius 0.12 with 0.04 margin bounds the preferred 0.17 offset to 0.08.
        self.assertAlmostEqual(release[0], 1.0, places=6)
        self.assertAlmostEqual(release[1], 2.08, places=6)
        self.assertLess(math.dist(release[:2], scoring[:2]), 0.12)


class Task3IntegrationWiringTests(unittest.TestCase):
    def _memory(self) -> CompetitionTaskMemory:
        state = ShelfState(
            empty_layer=3,
            colored_layer=2,
            colored_class_id="brown",
            white_obstacle_layer=1,
            layer_contents=("packaging_box", "brown", "EMPTY"),
            layer_centers_world=(
                (-2.54, 0.778, 0.530),
                (-2.55, 0.810, 0.837),
                (-2.63, 0.778, 1.166),
            ),
            confidence=0.95,
            frames_used=7,
        )
        memory = CompetitionTaskMemory()
        memory.record_shelf_state(state)
        return memory

    def test_executor_uses_task1_grasp_and_task3_orientation(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        self.assertEqual(executor.task_id, 3)
        self.assertEqual(executor.SOURCE_ORIENTATION, "yaw90")
        self.assertEqual(executor.TASK3_LATERAL_TIMEOUT_S, 35.0)
        self.assertEqual(executor.TASK3_LATERAL_POSITION_TOLERANCE_M, 0.015)
        # This is an intentional task-3 calibration knob.  Keep the wiring
        # test independent of the selected safe height (currently 0.09 m in
        # the baseline; remote trials may use 0.12 m) while guarding against
        # an accidentally unsafe or unusable value.
        self.assertGreaterEqual(executor.TASK3_INSERT_CLEARANCE_M, 0.05)
        self.assertLessEqual(executor.TASK3_INSERT_CLEARANCE_M, 0.20)
        self.assertEqual(executor.TASK3_ARM_INSERTION_M, 0.160)
        self.assertEqual(executor.TASK3_ARM_INSERT_TIMEOUT_S, 20.0)
        self.assertTrue(executor._held_insert.allow_extension)
        self.assertEqual(executor._held_insert.max_translation_m, 0.22)

    def test_first_attempt_uses_dynamic_white_cube_reference(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._task3_table_reference_world = (0.25, 1.80, 0.82)
        center = executor._source_center_from_observation(
            (0.31, 1.97, 1.01),
            first_attempt=True,
        )
        self.assertEqual(center, (0.25, 1.80, 1.01))

    def test_transport_keeps_post_lift_arm_pose_without_compaction(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        hold = ArmCommand(
            spine_position=0.18,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.1,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(-0.1,) * 6,
            right_gripper_position=1.0,
        )
        held_center = (0.72, 0.01, 1.10)
        executor._held_arm_command = hold
        executor._held_center_base = held_center

        executor.enter_stage(
            TaskStage.TRANSPORT,
            ExecutionContext(
                now_s=12.0,
                instruction={"task": 3, "target_color": "pink"},
                task_index=2,
                attempt=1,
            ),
        )

        self.assertIs(executor._held_arm_command, hold)
        self.assertEqual(executor._held_center_base, held_center)
        self.assertEqual(executor._phase, "retreat_table")

    def test_first_attempt_rejects_target_far_from_dynamic_reference(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._task3_table_reference_world = (0.25, 1.80, 0.82)
        center = executor._source_center_from_observation(
            (-0.40, 2.20, 1.01),
            first_attempt=True,
        )
        self.assertIsNone(center)

    def test_retry_tracks_current_target_without_snapping_to_initial_slot(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._task3_table_reference_world = (0.25, 1.80, 0.82)
        center = executor._source_center_from_observation(
            (-1.20, 1.10, 0.95),
            first_attempt=False,
        )
        self.assertEqual(center, (-1.20, 1.10, 0.95))

    def test_executor_derives_release_from_shared_shelf_snapshot(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._ensure_task3_place_target(
            ExecutionContext(
                now_s=10.0,
                instruction={
                    "task": 3,
                    "target_color": "pink",
                    "place_type": "shelf_prop_side",
                    "direction": "left",
                    "place_world": [-2.68, 0.540, 0.498],
                    "place_radius": 0.24,
                },
                task_index=2,
                attempt=1,
            )
        )
        self.assertEqual(executor._task3_white_layer, 1)
        self.assertAlmostEqual(executor._task3_scoring_place[1], 0.540, places=6)
        self.assertAlmostEqual(executor._place_world[0], -2.510, places=6)
        self.assertAlmostEqual(executor._place_world[1], 0.540, places=6)
        self.assertEqual(executor._task3_place_radius_m, 0.24)

    def test_executor_uses_each_rounds_formal_place_radius(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._ensure_task3_place_target(
            ExecutionContext(
                now_s=10.0,
                instruction={
                    "task": 3,
                    "target_color": "yellow",
                    "place_type": "shelf_prop_side",
                    "direction": "left",
                    "place_world": [-2.68, 0.540, 0.498],
                    "place_radius": 0.10,
                },
                task_index=2,
                attempt=1,
            )
        )
        # 0.10 radius minus 0.04 safety margin -> 0.06 outward release.
        self.assertAlmostEqual(executor._place_world[0], -2.62, places=6)
        self.assertEqual(executor._task3_place_radius_m, 0.10)

    def test_task123_full_wires_integrated_task3_without_changing_task12(self) -> None:
        executors = build_task_executors("task123_full")
        self.assertIsInstance(executors[3], Task3IntegratedExecutor)
        self.assertIs(executors[1]._memory, executors[3]._memory)
        self.assertEqual(build_task_executors("task12_full")[3].task_id, 3)
        self.assertNotIsInstance(
            build_task_executors("task12_full")[3], Task3IntegratedExecutor
        )


if __name__ == "__main__":
    unittest.main()
