from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from desktop_grasp.pregrasp_core import OpenPregraspController
from executors.base import (
    ArmCommand,
    ExecutionContext,
    StageStatus,
    TargetObservation,
    TaskStage,
)
from executors.task1 import Task1PregraspExecutor
from navigation.navigation_types import NavigationStatus, VelocityCommand


ARM_COMMAND = ArmCommand(
    spine_position=0.4,
    head_positions=(0.0, 0.45),
    left_arm_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    left_gripper_position=1.0,
    right_arm_positions=(-0.1, -0.2, -0.3, -0.4, -0.5, -0.6),
    right_gripper_position=1.0,
)


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


class FakePregraspController:
    def __init__(self) -> None:
        self.planned = False
        self.plan_target = None
        self.update_count = 0

    def reset(self) -> None:
        self.planned = False
        self.update_count = 0

    def plan(self, target_world, odometry, joint_states):
        self.plan_target = target_world
        self.planned = True
        return ARM_COMMAND

    def update(self, now_s, joint_states):
        self.update_count += 1
        return ARM_COMMAND, self.update_count >= 2, "fake feedback"


class FakeKdl:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.ref = None
        self.height = None

    def inverse_kinematics(
        self,
        *,
        T_left,
        T_right,
        ref_pos,
        target_height,
    ):
        self.left = T_left
        self.right = T_right
        self.ref = ref_pos
        self.height = target_height
        return [np.array([target_height, *([0.0] * 12)], dtype=float)]


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


def joint_states(*, slide: float = 0.0):
    names = [
        "slide_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        "left_arm_eef_gripper_joint",
        *(f"right_arm_joint{index}" for index in range(1, 7)),
        "right_arm_eef_gripper_joint",
    ]
    positions = [slide, 0.0, 0.0, *([0.0] * 6), 1.0, *([0.0] * 6), 1.0]
    return SimpleNamespace(
        name=names,
        position=positions,
        velocity=[0.0] * len(names),
        effort=[0.0] * len(names),
    )


def context(now_s: float, pose, joints=None, *, unsafe_collision=False):
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
        joint_states=joints or joint_states(),
        target_observations={
            "brown": TargetObservation(
                color="brown",
                position_world=(-0.18, 2.20, 0.851),
                received_at_s=now_s,
                orientation="yaw0",
                score=0.9,
            )
        },
        unsafe_collision=unsafe_collision,
    )


class OpenPregraspControllerTests(unittest.TestCase):
    def test_plans_calibrated_open_pose_at_navigation_standoff(self) -> None:
        kdl = FakeKdl()
        controller = OpenPregraspController(kdl=kdl)

        controller.plan(
            (-0.18, 2.20, 0.834),
            odometry(-0.18, 1.64, math.pi / 2.0),
            joint_states(),
        )

        self.assertIsNotNone(controller.target_base)
        self.assertAlmostEqual(controller.target_base[0], 0.56, places=6)
        self.assertAlmostEqual(controller.target_base[1], 0.0, places=6)
        self.assertAlmostEqual(kdl.left[0, 3], 0.48, places=6)
        self.assertAlmostEqual(kdl.left[1, 3], 0.225, places=6)
        self.assertAlmostEqual(kdl.right[1, 3], -0.225, places=6)
        self.assertAlmostEqual(kdl.left[2, 3], 0.854, places=6)


class Task1PregraspExecutorTests(unittest.TestCase):
    def test_navigates_then_holds_open_pregrasp_before_contact(self) -> None:
        pregrasp = FakePregraspController()
        executor = Task1PregraspExecutor(pregrasp_controller=pregrasp)
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        moving = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        self.assertTrue(moving.controls_base)
        at_goal = context(0.05, odometry(-0.18, 1.64, math.pi / 2.0))
        reached = executor.tick(TaskStage.NAVIGATE_TO_PICK, at_goal)
        self.assertEqual(reached.status, StageStatus.SUCCEEDED)

        executor.enter_stage(TaskStage.ACQUIRE_TARGET, at_goal)
        acquired = executor.tick(TaskStage.ACQUIRE_TARGET, at_goal)
        self.assertEqual(acquired.status, StageStatus.SUCCEEDED)

        executor.enter_stage(TaskStage.ALIGN_FOR_PICK, at_goal)
        moving_arms = executor.tick(TaskStage.ALIGN_FOR_PICK, at_goal)
        self.assertEqual(moving_arms.status, StageStatus.RUNNING)
        self.assertTrue(moving_arms.controls_arm)
        self.assertEqual(pregrasp.plan_target[2], 0.834)

        aligned = executor.tick(
            TaskStage.ALIGN_FOR_PICK,
            context(0.10, odometry(-0.18, 1.64, math.pi / 2.0)),
        )
        self.assertEqual(aligned.status, StageStatus.SUCCEEDED)
        self.assertEqual(aligned.arm_command, ARM_COMMAND)

        executor.enter_stage(TaskStage.GRASP, at_goal)
        blocked = executor.tick(TaskStage.GRASP, at_goal)
        self.assertEqual(blocked.status, StageStatus.BLOCKED)
        self.assertEqual(blocked.arm_command, ARM_COMMAND)
        self.assertIn("inward grasp", blocked.message)

    def test_server_collision_blocks_before_arm_motion(self) -> None:
        executor = Task1PregraspExecutor(
            pregrasp_controller=FakePregraspController()
        )
        collision = context(
            0.0,
            odometry(-0.18, 1.64, math.pi / 2.0),
            unsafe_collision=True,
        )
        executor.enter_stage(TaskStage.ACQUIRE_TARGET, collision)

        result = executor.tick(TaskStage.ACQUIRE_TARGET, collision)

        self.assertEqual(result.status, StageStatus.BLOCKED)
        self.assertIn("unsafe collision", result.message)


if __name__ == "__main__":
    unittest.main()
