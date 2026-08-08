from __future__ import annotations

import unittest

from executors import build_task_executors
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

    def test_safe_release_stays_outward_and_toward_shelf_centre(self) -> None:
        scoring = (-2.68, 0.540, 0.498)
        release = task3_safe_release_target(scoring)
        self.assertAlmostEqual(release[0], -2.62, places=6)
        self.assertAlmostEqual(release[1], 0.580, places=6)
        self.assertEqual(release[2], scoring[2])


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
        self.assertAlmostEqual(executor.TABLE_BOX_CENTER_Z_M, 1.004, places=6)

    def test_executor_derives_release_from_shared_shelf_snapshot(self) -> None:
        executor = Task3IntegratedExecutor(self._memory())
        executor._ensure_task3_place_target()
        self.assertEqual(executor._task3_white_layer, 1)
        self.assertAlmostEqual(executor._task3_scoring_place[1], 0.540, places=6)
        self.assertAlmostEqual(executor._place_world[0], -2.620, places=6)
        self.assertAlmostEqual(executor._place_world[1], 0.580, places=6)

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
