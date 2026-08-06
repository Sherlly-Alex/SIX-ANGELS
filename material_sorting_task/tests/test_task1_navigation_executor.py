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
) -> ExecutionContext:
    observations = {}
    if with_target:
        observations["brown"] = TargetObservation(
            color="brown",
            position_world=(-0.18, 2.20, 0.834),
            received_at_s=now_s,
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
        self.assertAlmostEqual(executor.goal.x, -0.18)
        self.assertAlmostEqual(executor.goal.y, 1.64)

        at_goal = context(0.05, odometry(-0.18, 1.64, math.pi / 2.0))
        reached = executor.tick(TaskStage.NAVIGATE_TO_PICK, at_goal)
        self.assertEqual(reached.status, StageStatus.SUCCEEDED)
        self.assertFalse(reached.controls_base)

        executor.enter_stage(TaskStage.ACQUIRE_TARGET, at_goal)
        blocked = executor.tick(TaskStage.ACQUIRE_TARGET, at_goal)
        self.assertEqual(blocked.status, StageStatus.BLOCKED)
        self.assertFalse(blocked.controls_base)
        self.assertIn("arm/perception handoff is not implemented", blocked.message)


if __name__ == "__main__":
    unittest.main()
