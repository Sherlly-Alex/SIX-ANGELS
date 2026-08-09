from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from executors.base import ExecutionContext, StageStatus, TargetObservation, TaskStage
from executors.task1 import Task1NavigationExecutor
from navigation.navigation_types import NavigationStatus, VelocityCommand


class FakeNavigationController:
    def __init__(self) -> None:
        self.status = NavigationStatus.IDLE
        self.goal = None

    def reset(self) -> None:
        self.status = NavigationStatus.IDLE
        self.goal = None

    def set_goal(self, goal, robot_x: float, robot_y: float) -> bool:
        self.goal = goal
        self.status = NavigationStatus.NAVIGATING
        return True

    def update(self, robot_x, robot_y, robot_yaw, dt, obs=None):
        if math.hypot(robot_x - self.goal.x, robot_y - self.goal.y) <= 0.08:
            self.status = NavigationStatus.GOAL_REACHED
            return VelocityCommand(0.0, 0.0)
        return VelocityCommand(0.10, 0.20)


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


def context(
    now_s: float,
    pose,
    *,
    with_target: bool = True,
    target_position=(-0.18, 2.20, 0.834),
    target_orientation=None,
) -> ExecutionContext:
    observations = {}
    if with_target:
        observations["brown"] = TargetObservation(
            color="brown",
            position_world=target_position,
            received_at_s=now_s,
            orientation=target_orientation,
        )
    return ExecutionContext(
        now_s=now_s,
        instruction={
            "task": 1,
            "target_color": "brown",
            "place_type": "shelf_point",
            "place_world": [-2.68, 0.778, 1.166],
        },
        task_index=0,
        attempt=1,
        odometry=pose,
        target_observations=observations,
    )


class Task1NavigationExecutorTests(unittest.TestCase):
    def test_waits_safely_for_stable_target(self) -> None:
        executor = Task1NavigationExecutor()
        initial = context(
            0.0,
            odometry(-0.70, 0.55, math.pi / 2.0),
            with_target=False,
        )
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        result = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertFalse(result.controls_base)
        self.assertIn("waiting", result.message)

    def test_navigates_to_detected_pick_stand_then_blocks_before_arm(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        moving = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        self.assertEqual(moving.status, StageStatus.RUNNING)
        self.assertTrue(moving.controls_base)
        self.assertIsNotNone(executor.goal)
        self.assertAlmostEqual(executor.goal.x, -0.22)
        self.assertAlmostEqual(executor.goal.y, 1.55)
        self.assertAlmostEqual(executor._navigation.goal.y, 1.25)
        self.assertEqual(executor._navigation_phase, "route_to_preturn")

        at_goal = context(0.05, odometry(-0.22, 1.55, math.pi / 2.0))
        reached = executor.tick(TaskStage.NAVIGATE_TO_PICK, at_goal)
        self.assertEqual(reached.status, StageStatus.SUCCEEDED)
        self.assertFalse(reached.controls_base)

        executor.enter_stage(TaskStage.ACQUIRE_TARGET, at_goal)
        blocked = executor.tick(TaskStage.ACQUIRE_TARGET, at_goal)
        self.assertEqual(blocked.status, StageStatus.BLOCKED)
        self.assertFalse(blocked.controls_base)
        self.assertIn("arm/perception handoff is not implemented", blocked.message)

    def test_right_slot_preturn_hands_off_to_fixed_north_approach(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)
        executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)

        preturn = context(0.05, odometry(-0.22, 1.25, math.pi / 2.0))
        starting_final = executor.tick(TaskStage.NAVIGATE_TO_PICK, preturn)
        self.assertEqual(starting_final.status, StageStatus.RUNNING)
        self.assertEqual(executor._navigation_phase, "final_north_approach")

        midway = context(0.10, odometry(-0.22, 1.35, math.pi / 2.0))
        advancing = executor.tick(TaskStage.NAVIGATE_TO_PICK, midway)
        self.assertEqual(advancing.status, StageStatus.RUNNING)
        self.assertGreater(advancing.base_linear_x, 0.0)
        self.assertEqual(executor._table_motion._advance_start[2], math.pi / 2.0)

        final = context(0.15, odometry(-0.22, 1.55, math.pi / 2.0))
        reached = executor.tick(TaskStage.NAVIGATE_TO_PICK, final)
        self.assertEqual(reached.status, StageStatus.SUCCEEDED)
        self.assertIn("strict table arm-handoff", reached.message)

    def test_route_is_captured_once_exact_northbound_lane_is_reached(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)
        executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)

        lane = context(0.05, odometry(-0.207, 0.979, math.pi / 2.0))
        captured = executor.tick(TaskStage.NAVIGATE_TO_PICK, lane)
        self.assertEqual(captured.status, StageStatus.RUNNING)
        self.assertEqual(executor._navigation_phase, "final_north_approach")
        self.assertIn("captured the exact northbound", captured.message)
        self.assertAlmostEqual(executor._table_motion._advance_start[2], math.pi / 2.0)

    def test_snaps_biased_yaw90_detection_to_safe_right_source_slot(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation = FakeNavigationController()
        initial = context(
            0.0,
            odometry(-0.70, 0.55, math.pi / 2.0),
            target_position=(-0.223, 2.321, 0.834),
            target_orientation="yaw90",
        )
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        result = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._locked_target_world, (-0.18, 2.20, 0.834))
        self.assertEqual(executor._locked_target_orientation, "yaw0")
        self.assertAlmostEqual(executor.goal.x, -0.22)
        self.assertAlmostEqual(executor.goal.y, 1.55)

    def test_rejects_detection_outside_both_table_source_slots(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation = FakeNavigationController()
        initial = context(
            0.0,
            odometry(-0.70, 0.55, math.pi / 2.0),
            target_position=(-0.54, 2.30, 0.834),
            target_orientation="yaw90",
        )
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        result = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertFalse(result.controls_base)
        self.assertIsNone(executor.goal)
        self.assertIn("outside both calibrated", result.message)


if __name__ == "__main__":
    unittest.main()
