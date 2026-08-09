from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from desktop_grasp.pregrasp_core import (
    ContactGraspController,
    OpenPregraspController,
    SlideLiftController,
)
from executors.base import (
    ArmCommand,
    ExecutionContext,
    StageStatus,
    TargetObservation,
    TaskStage,
)
from executors.task1 import (
    Task1ContactExecutor,
    Task1LiftExecutor,
    Task1PregraspExecutor,
)
from navigation.navigation_types import NavigationStatus, VelocityCommand


ARM_COMMAND = ArmCommand(
    spine_position=0.4,
    head_positions=(0.0, 0.45),
    left_arm_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    left_gripper_position=1.0,
    right_arm_positions=(-0.1, -0.2, -0.3, -0.4, -0.5, -0.6),
    right_gripper_position=1.0,
)

LIFTED_ARM_COMMAND = ArmCommand(
    spine_position=0.25,
    head_positions=ARM_COMMAND.head_positions,
    left_arm_positions=ARM_COMMAND.left_arm_positions,
    left_gripper_position=ARM_COMMAND.left_gripper_position,
    right_arm_positions=ARM_COMMAND.right_arm_positions,
    right_gripper_position=ARM_COMMAND.right_gripper_position,
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


class FakeLiftController:
    def __init__(self) -> None:
        self.planned = False
        self.actual_lift_m = 0.15
        self.plan_command = None
        self.update_count = 0

    def reset(self) -> None:
        self.planned = False
        self.update_count = 0

    def plan(self, hold_command, joint_states):
        self.plan_command = hold_command
        self.planned = True
        return LIFTED_ARM_COMMAND

    def update(self, now_s, joint_states):
        self.update_count += 1
        return (
            LIFTED_ARM_COMMAND,
            self.update_count >= 2,
            "fake lift feedback",
        )

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


def joint_states(
    *,
    slide: float = 0.0,
    left_arm=None,
    right_arm=None,
):
    names = [
        "slide_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        "left_arm_eef_gripper_joint",
        *(f"right_arm_joint{index}" for index in range(1, 7)),
        "right_arm_eef_gripper_joint",
    ]
    left = [0.0] * 6 if left_arm is None else list(left_arm)
    right = [0.0] * 6 if right_arm is None else list(right_arm)
    positions = [slide, 0.0, 0.0, *left, 1.0, *right, 1.0]
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
            odometry(-0.18, 1.55, math.pi / 2.0),
            joint_states(),
        )

        self.assertIsNotNone(controller.target_base)
        self.assertAlmostEqual(controller.target_base[0], 0.65, places=6)
        self.assertAlmostEqual(controller.target_base[1], 0.0, places=6)
        self.assertAlmostEqual(kdl.left[0, 3], 0.57, places=6)
        self.assertAlmostEqual(kdl.left[1, 3], 0.225, places=6)
        self.assertAlmostEqual(kdl.right[1, 3], -0.225, places=6)
        self.assertAlmostEqual(kdl.left[2, 3], 0.854, places=6)

    def test_contact_pose_uses_calibrated_task1_lateral_width(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)

        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            joint_states(),
        )

        self.assertAlmostEqual(controller.half_width, 0.118, places=6)
        self.assertAlmostEqual(controller.ARM_POSITION_TOL, 0.24, places=6)
        self.assertAlmostEqual(kdl.left[0, 3], 0.67, places=6)
        self.assertAlmostEqual(kdl.left[1, 3], 0.118, places=6)
        self.assertAlmostEqual(kdl.right[1, 3], -0.118, places=6)

        controller.tighten(
            (-0.18, 2.20, 0.834),
            0.001,
            odometry(-0.18, 1.55, math.pi / 2.0),
            joint_states(),
        )
        self.assertAlmostEqual(controller.ARM_POSITION_TOL, 0.24, places=6)

    def test_contact_converges_when_box_blocks_joint_by_point_153_rad(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)
        expected_slide = 1.32163718 - 0.854
        blocked_feedback = joint_states(
            slide=expected_slide,
            right_arm=[0.153, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            blocked_feedback,
        )

        reached = False
        for tick in range(30):
            _command, reached, _detail = controller.update(
                tick * 0.05,
                blocked_feedback,
            )

        self.assertTrue(reached)


class SlideLiftControllerTests(unittest.TestCase):
    def test_lift_changes_only_spine_and_preserves_arm_preload(self) -> None:
        controller = SlideLiftController(lift_height=0.15)
        feedback = joint_states(
            slide=ARM_COMMAND.spine_position,
            left_arm=ARM_COMMAND.left_arm_positions,
            right_arm=ARM_COMMAND.right_arm_positions,
        )

        command = controller.plan(ARM_COMMAND, feedback)

        self.assertAlmostEqual(controller.target_slide, 0.25, places=6)
        self.assertAlmostEqual(controller.actual_lift_m, 0.15, places=6)
        self.assertEqual(command.left_arm_positions, ARM_COMMAND.left_arm_positions)
        self.assertEqual(command.right_arm_positions, ARM_COMMAND.right_arm_positions)
        self.assertEqual(
            command.left_gripper_position,
            ARM_COMMAND.left_gripper_position,
        )
        self.assertEqual(
            command.right_gripper_position,
            ARM_COMMAND.right_gripper_position,
        )

        reached = False
        for tick in range(30):
            feedback = joint_states(
                slide=command.spine_position,
                left_arm=command.left_arm_positions,
                right_arm=command.right_arm_positions,
            )
            command, reached, _detail = controller.update(tick * 0.20, feedback)
            if reached:
                break

        self.assertTrue(reached)
        self.assertAlmostEqual(command.spine_position, 0.25, places=6)
        self.assertEqual(command.left_arm_positions, ARM_COMMAND.left_arm_positions)
        self.assertEqual(command.right_arm_positions, ARM_COMMAND.right_arm_positions)


class Task1PregraspExecutorTests(unittest.TestCase):
    def test_navigates_then_holds_open_pregrasp_before_contact(self) -> None:
        pregrasp = FakePregraspController()
        executor = Task1PregraspExecutor(pregrasp_controller=pregrasp)
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)

        moving = executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        self.assertTrue(moving.controls_base)
        at_goal = context(0.05, odometry(-0.18, 1.55, math.pi / 2.0))
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
            context(0.10, odometry(-0.18, 1.55, math.pi / 2.0)),
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
            odometry(-0.18, 1.55, math.pi / 2.0),
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
        at_goal = context(0.05, odometry(-0.18, 1.55, math.pi / 2.0))
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
                odometry(-0.18, 1.55, math.pi / 2.0),
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
                odometry(-0.18, 1.55, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        self.assertEqual(confirming.status, StageStatus.RUNNING)
        self.assertEqual(contact_controller.update_count, updates_at_contact)

        confirmed = executor.tick(
            TaskStage.GRASP,
            context(
                0.41,
                odometry(-0.18, 1.55, math.pi / 2.0),
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
                odometry(-0.18, 1.55, math.pi / 2.0),
                grasp_confirmed=True,
            ),
        )
        updates_at_contact = contact_controller.update_count

        resumed = executor.tick(
            TaskStage.GRASP,
            context(
                0.20,
                odometry(-0.18, 1.55, math.pi / 2.0),
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
            context(0.60, odometry(-0.18, 1.55, math.pi / 2.0)),
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
                context(now_s, odometry(-0.18, 1.55, math.pi / 2.0)),
            )
            now_s += 0.10
            result = executor.tick(
                TaskStage.GRASP,
                context(now_s, odometry(-0.18, 1.55, math.pi / 2.0)),
            )
            self.assertAlmostEqual(
                contact_controller.tighten_offsets[-1],
                expected_mm / 1000.0,
            )
            now_s += 0.60

        self.assertEqual(len(contact_controller.tighten_offsets), 4)
        executor.tick(
            TaskStage.GRASP,
            context(now_s, odometry(-0.18, 1.55, math.pi / 2.0)),
        )
        self.assertEqual(len(contact_controller.tighten_offsets), 4)

    def test_unsafe_structure_collision_holds_contact_command(self) -> None:
        executor, _contact_controller, at_goal = self._reach_contact_stage()
        executor.tick(TaskStage.GRASP, at_goal)

        result = executor.tick(
            TaskStage.GRASP,
            context(
                0.10,
                odometry(-0.18, 1.55, math.pi / 2.0),
                unsafe_collision=True,
            ),
        )

        self.assertEqual(result.status, StageStatus.BLOCKED)
        self.assertIn("unsafe collision", result.message)
        self.assertEqual(result.arm_command, ARM_COMMAND)


class Task1LiftExecutorTests(unittest.TestCase):
    def _reach_grasp_stage(self):
        contact_controller = FakeContactController()
        lift_controller = FakeLiftController()
        executor = Task1LiftExecutor(
            pregrasp_controller=FakePregraspController(),
            contact_controller=contact_controller,
            lift_controller=lift_controller,
        )
        executor._navigation = FakeNavigationController()
        initial = context(0.0, odometry(-0.70, 0.55, math.pi / 2.0))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)
        executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        at_goal = context(0.05, odometry(-0.18, 1.55, math.pi / 2.0))
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
        return executor, contact_controller, lift_controller

    def test_settled_max_preload_lifts_without_server_contact_confirmation(self) -> None:
        executor, contact_controller, lift_controller = self._reach_grasp_stage()
        result = None
        now_s = 0.10
        for _ in range(20):
            result = executor.tick(
                TaskStage.GRASP,
                context(now_s, odometry(-0.18, 1.55, math.pi / 2.0)),
            )
            if result.status is StageStatus.SUCCEEDED:
                break
            now_s += 0.35

        self.assertIsNotNone(result)
        self.assertEqual(result.status, StageStatus.SUCCEEDED)
        self.assertEqual(contact_controller.tighten_offsets, [0.001, 0.002, 0.003, 0.004])
        self.assertIn("without Server", result.message)

        lift_context = context(
            now_s + 0.05,
            odometry(-0.18, 1.55, math.pi / 2.0),
        )
        executor.enter_stage(TaskStage.LIFT, lift_context)
        lifting = executor.tick(TaskStage.LIFT, lift_context)
        self.assertEqual(lifting.status, StageStatus.RUNNING)
        lifted = executor.tick(
            TaskStage.LIFT,
            context(now_s + 0.10, odometry(-0.18, 1.55, math.pi / 2.0)),
        )
        self.assertEqual(lifted.status, StageStatus.SUCCEEDED)
        self.assertEqual(lift_controller.plan_command, ARM_COMMAND)
        self.assertEqual(lifted.arm_command, LIFTED_ARM_COMMAND)

        transport_context = context(
            now_s + 0.15,
            odometry(-0.18, 1.55, math.pi / 2.0),
        )
        executor.enter_stage(TaskStage.TRANSPORT, transport_context)
        held = executor.tick(TaskStage.TRANSPORT, transport_context)
        self.assertEqual(held.status, StageStatus.BLOCKED)
        self.assertEqual(held.arm_command, LIFTED_ARM_COMMAND)
        self.assertIn("transport", held.message)

    def test_unsafe_collision_blocks_lift_and_holds_preload(self) -> None:
        executor, _contact_controller, _lift_controller = self._reach_grasp_stage()
        executor._held_arm_command = ARM_COMMAND
        collision = context(
            1.0,
            odometry(-0.18, 1.55, math.pi / 2.0),
            unsafe_collision=True,
        )
        executor.enter_stage(TaskStage.LIFT, collision)

        result = executor.tick(TaskStage.LIFT, collision)

        self.assertEqual(result.status, StageStatus.BLOCKED)
        self.assertEqual(result.arm_command, ARM_COMMAND)
        self.assertIn("unsafe collision", result.message)


if __name__ == "__main__":
    unittest.main()
