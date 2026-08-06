from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from desktop_grasp.pregrasp_core import (
    ContactGraspController,
    OpenPregraspController,
)
from executors.base import (
    ArmCommand,
    ExecutionContext,
    StageStatus,
    TargetObservation,
    TaskStage,
)
from executors.task1 import Task1ContactExecutor, Task1PregraspExecutor
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


class FakeContactController:
    def __init__(self) -> None:
        self.planned = False
        self.half_width = 0.118
        self.plan_target = None
        self.plan_orientation = None
        self.update_count = 0
        self.updates_since_plan = 0
        self.tighten_offsets = []

    def reset(self) -> None:
        self.planned = False
        self.update_count = 0
        self.updates_since_plan = 0
        self.tighten_offsets = []

    def plan(self, target_world, orientation, odometry, joint_states):
        self.plan_target = target_world
        self.plan_orientation = orientation
        self.planned = True
        return ARM_COMMAND

    def update(self, now_s, joint_states):
        self.update_count += 1
        self.updates_since_plan += 1
        return ARM_COMMAND, self.updates_since_plan >= 2, "fake contact feedback"

    def tighten(self, target_world, inward_offset, odometry, joint_states):
        self.tighten_offsets.append(inward_offset)
        self.half_width = 0.118 - inward_offset
        self.updates_since_plan = 0
        return ARM_COMMAND


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


def context(
    now_s: float,
    pose,
    joints=None,
    *,
    unsafe_collision=False,
    grasp_confirmed=False,
    target_orientation="yaw90",
):
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
                orientation=target_orientation,
                score=0.9,
            )
        },
        grasp_confirmed=grasp_confirmed,
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

    def test_contact_pose_uses_calibrated_task1_lateral_width(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)

        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.64, math.pi / 2.0),
            joint_states(),
        )

        self.assertAlmostEqual(controller.half_width, 0.118, places=6)
        self.assertAlmostEqual(controller.ARM_POSITION_TOL, 0.14, places=6)
        self.assertAlmostEqual(kdl.left[0, 3], 0.58, places=6)
        self.assertAlmostEqual(kdl.left[1, 3], 0.118, places=6)
        self.assertAlmostEqual(kdl.right[1, 3], -0.118, places=6)

        controller.tighten(
            (-0.18, 2.20, 0.834),
            0.001,
            odometry(-0.18, 1.64, math.pi / 2.0),
            joint_states(),
        )
        self.assertAlmostEqual(controller.ARM_POSITION_TOL, 0.24, places=6)


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


class Task1ContactExecutorTests(unittest.TestCase):
    def _reach_contact_stage(self):
        pregrasp = FakePregraspController()
        contact_controller = FakeContactController()
        executor = Task1ContactExecutor(
            pregrasp_controller=pregrasp,
            contact_controller=contact_controller,
        )
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)
        executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        at_goal = context(0.05, odometry(-0.18, 1.64, math.pi / 2.0))
        self.assertEqual(
            executor.tick(TaskStage.NAVIGATE_TO_PICK, at_goal).status,
            StageStatus.SUCCEEDED,
        )
        executor.enter_stage(TaskStage.ACQUIRE_TARGET, at_goal)
        self.assertEqual(
            executor.tick(TaskStage.ACQUIRE_TARGET, at_goal).status,
            StageStatus.SUCCEEDED,
        )
        executor.enter_stage(TaskStage.ALIGN_FOR_PICK, at_goal)
        executor.tick(TaskStage.ALIGN_FOR_PICK, at_goal)
        self.assertEqual(
            executor.tick(TaskStage.ALIGN_FOR_PICK, at_goal).status,
            StageStatus.SUCCEEDED,
        )
        executor.enter_stage(TaskStage.GRASP, at_goal)
        return executor, contact_controller, at_goal

    def test_freezes_on_stable_server_contact_and_blocks_before_lift(self) -> None:
        executor, contact_controller, at_goal = self._reach_contact_stage()

        approaching = executor.tick(TaskStage.GRASP, at_goal)
        self.assertEqual(approaching.status, StageStatus.RUNNING)
        self.assertEqual(contact_controller.plan_orientation, "yaw0")
        self.assertEqual(contact_controller.plan_target[2], 0.834)
        updates_before_contact = contact_controller.update_count

        first_contact = executor.tick(
            TaskStage.GRASP,
            context(
                0.10,
                odometry(-0.18, 1.64, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        self.assertEqual(first_contact.status, StageStatus.RUNNING)
        updates_at_contact = contact_controller.update_count
        self.assertEqual(updates_at_contact, updates_before_contact)

        confirming = executor.tick(
            TaskStage.GRASP,
            context(
                0.25,
                odometry(-0.18, 1.64, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        self.assertEqual(confirming.status, StageStatus.RUNNING)
        self.assertEqual(contact_controller.update_count, updates_at_contact)

        confirmed = executor.tick(
            TaskStage.GRASP,
            context(
                0.41,
                odometry(-0.18, 1.64, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        self.assertEqual(confirmed.status, StageStatus.SUCCEEDED)
        self.assertEqual(confirmed.arm_command, ARM_COMMAND)

        executor.enter_stage(TaskStage.LIFT, at_goal)
        blocked = executor.tick(TaskStage.LIFT, at_goal)
        self.assertEqual(blocked.status, StageStatus.BLOCKED)
        self.assertIn("lift", blocked.message)
        self.assertEqual(blocked.arm_command, ARM_COMMAND)

    def test_dropped_contact_resumes_bounded_inward_motion(self) -> None:
        executor, contact_controller, at_goal = self._reach_contact_stage()
        executor.tick(TaskStage.GRASP, at_goal)
        executor.tick(
            TaskStage.GRASP,
            context(
                0.10,
                odometry(-0.18, 1.64, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        updates_at_contact = contact_controller.update_count

        resumed = executor.tick(
            TaskStage.GRASP,
            context(
                0.20,
                odometry(-0.18, 1.64, math.pi / 2.0),
                grasp_confirmed=False,
            ),
        )

        self.assertEqual(resumed.status, StageStatus.RUNNING)
        self.assertGreater(contact_controller.update_count, updates_at_contact)

    def test_settled_pose_searches_inward_in_bounded_millimeter_steps(self) -> None:
        executor, contact_controller, at_goal = self._reach_contact_stage()

        executor.tick(TaskStage.GRASP, at_goal)
        search = executor.tick(
            TaskStage.GRASP,
            context(0.60, odometry(-0.18, 1.64, math.pi / 2.0)),
        )

        self.assertEqual(search.status, StageStatus.RUNNING)
        self.assertEqual(contact_controller.tighten_offsets, [0.001])
        self.assertIn("1/4 mm", search.message)

        # Each tightened pose must settle again before the next millimeter;
        # the total search is capped at the desktop-grasp module's 4 mm bound.
        now_s = 1.20
        for expected_mm in (2, 3, 4):
            executor.tick(
                TaskStage.GRASP,
                context(now_s, odometry(-0.18, 1.64, math.pi / 2.0)),
            )
            now_s += 0.10
            result = executor.tick(
                TaskStage.GRASP,
                context(now_s, odometry(-0.18, 1.64, math.pi / 2.0)),
            )
            self.assertAlmostEqual(
                contact_controller.tighten_offsets[-1],
                expected_mm / 1000.0,
            )
            now_s += 0.60

        self.assertEqual(len(contact_controller.tighten_offsets), 4)
        executor.tick(
            TaskStage.GRASP,
            context(now_s, odometry(-0.18, 1.64, math.pi / 2.0)),
        )
        self.assertEqual(len(contact_controller.tighten_offsets), 4)

    def test_unsafe_structure_collision_holds_contact_command(self) -> None:
        executor, _contact_controller, at_goal = self._reach_contact_stage()
        executor.tick(TaskStage.GRASP, at_goal)

        result = executor.tick(
            TaskStage.GRASP,
            context(
                0.10,
                odometry(-0.18, 1.64, math.pi / 2.0),
                unsafe_collision=True,
            ),
        )

        self.assertEqual(result.status, StageStatus.BLOCKED)
        self.assertIn("unsafe collision", result.message)
        self.assertEqual(result.arm_command, ARM_COMMAND)


if __name__ == "__main__":
    unittest.main()
