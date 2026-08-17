from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from desktop_grasp.pregrasp_core import (
    COMPLIANT_ENTRY_CLEARANCE_M,
    COMPLIANT_ENTRY_TRAVEL_M,
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

    def track_inward_offset(
        self, target_world, inward_offset, odometry, joint_states
    ):
        self.tighten_offsets.append(inward_offset)
        self.half_width = 0.118 - inward_offset
        return ARM_COMMAND


class FakeCompliantContactController(FakeContactController):
    def __init__(self) -> None:
        super().__init__()
        self.compliance_enabled = True
        self.bilateral_aligned = False
        self.any_contact = False
        self.preload_effort_limit_reached = False
        self.hard_effort_limit_exceeded = False
        self.diagnostic_summary = "compliance=fake"

    def reset(self) -> None:
        super().reset()
        self.compliance_enabled = True
        self.bilateral_aligned = False
        self.any_contact = False

    def prepare_compliance(self, now_s, joint_states):
        return True, self.diagnostic_summary

    def observe_server_contact(self, confirmed):
        return None

    def retry_compliance(self):
        self.bilateral_aligned = False

    def abandon_compliance(self, reason):
        self.compliance_enabled = False

    def tighten(self, target_world, inward_offset, odometry, joint_states):
        command = super().tighten(
            target_world,
            inward_offset,
            odometry,
            joint_states,
        )
        if inward_offset >= COMPLIANT_ENTRY_CLEARANCE_M - 1e-9:
            self.any_contact = True
            self.bilateral_aligned = True
        return command

    def track_inward_offset(
        self, target_world, inward_offset, odometry, joint_states
    ):
        command = super().track_inward_offset(
            target_world,
            inward_offset,
            odometry,
            joint_states,
        )
        if inward_offset >= COMPLIANT_ENTRY_CLEARANCE_M - 1e-9:
            self.any_contact = True
            self.bilateral_aligned = True
        return command


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
    left_effort=None,
    right_effort=None,
    left_velocity=None,
    right_velocity=None,
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
    left_tau = [0.0] * 6 if left_effort is None else list(left_effort)
    right_tau = [0.0] * 6 if right_effort is None else list(right_effort)
    left_vel = [0.0] * 6 if left_velocity is None else list(left_velocity)
    right_vel = [0.0] * 6 if right_velocity is None else list(right_velocity)
    positions = [slide, 0.0, 0.0, *left, 1.0, *right, 1.0]
    return SimpleNamespace(
        name=names,
        position=positions,
        velocity=[0.0, 0.0, 0.0, *left_vel, 0.0, *right_vel, 0.0],
        effort=[0.0, 0.0, 0.0, *left_tau, 0.0, *right_tau, 0.0],
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

    def test_compliant_contact_starts_ten_mm_outside_box_surface(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)
        feedback = joint_states()
        ready = False
        for tick in range(9):
            ready, _ = controller.prepare_compliance(tick * 0.05, feedback)
        self.assertTrue(ready)

        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )

        # yaw0 at this robot yaw has a 0.120 m physical lateral half extent.
        # The fast pose must therefore stop at 0.130 m, not at the old
        # 0.118 m preload pose.
        self.assertAlmostEqual(controller.half_width, 0.130, places=6)
        self.assertAlmostEqual(kdl.left[1, 3], 0.130, places=6)
        self.assertAlmostEqual(kdl.right[1, 3], -0.130, places=6)

        controller.track_inward_offset(
            (-0.18, 2.20, 0.834),
            COMPLIANT_ENTRY_CLEARANCE_M,
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )
        self.assertAlmostEqual(controller.half_width, 0.120, places=6)

        controller.track_inward_offset(
            (-0.18, 2.20, 0.834),
            COMPLIANT_ENTRY_TRAVEL_M,
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )
        self.assertAlmostEqual(controller.half_width, 0.118, places=6)

    def test_contact_pose_adds_symmetric_five_degree_toe_in(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)

        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            joint_states(),
        )

        # Left and right tool-forward axes must turn toward the centre line.
        self.assertLess(kdl.left[1, 0], -0.08)
        self.assertGreater(kdl.right[1, 0], 0.08)

    def test_continuous_retarget_preserves_action_and_locked_wrists(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)
        feedback = joint_states()
        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )
        controller._action_vector[3] = 0.27
        controller._left_wrist.locked_position = 0.04
        controller._right_wrist.locked_position = -0.05

        command = controller.track_inward_offset(
            (-0.18, 2.20, 0.834),
            0.0002,
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )

        self.assertAlmostEqual(controller.half_width, 0.1178, places=6)
        self.assertAlmostEqual(command.left_arm_positions[0], 0.27, places=6)
        self.assertAlmostEqual(command.left_arm_positions[5], 0.04, places=6)
        self.assertAlmostEqual(command.right_arm_positions[5], -0.05, places=6)

    def test_effort_contact_follows_then_locks_both_wrists(self) -> None:
        kdl = FakeKdl()
        controller = ContactGraspController(kdl=kdl)
        expected_slide = 1.32163718 - 0.854
        baseline = joint_states(slide=expected_slide)

        ready = False
        for tick in range(9):
            ready, _detail = controller.prepare_compliance(
                tick * 0.05,
                baseline,
            )
        self.assertTrue(ready)
        self.assertTrue(controller.compliance_enabled)

        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            baseline,
        )
        contact_feedback = joint_states(
            slide=expected_slide,
            left_arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
            right_arm=[0.0, 0.0, 0.0, 0.0, 0.0, -0.04],
            left_effort=[0.0, 0.0, 0.0, 0.0, 0.0, 1.2],
            right_effort=[0.0, 0.0, 0.0, 0.0, 0.0, -1.2],
        )
        command = None
        for tick in range(50):
            command, _settled, _detail = controller.update(
                0.50 + tick * 0.05,
                contact_feedback,
            )
            if controller.bilateral_aligned:
                break

        self.assertTrue(controller.bilateral_aligned)
        self.assertAlmostEqual(command.left_arm_positions[5], 0.04, places=6)
        self.assertAlmostEqual(command.right_arm_positions[5], -0.04, places=6)

        tightened = controller.tighten(
            (-0.18, 2.20, 0.834),
            0.001,
            odometry(-0.18, 1.55, math.pi / 2.0),
            contact_feedback,
        )
        self.assertAlmostEqual(tightened.left_arm_positions[5], 0.04, places=6)
        self.assertAlmostEqual(tightened.right_arm_positions[5], -0.04, places=6)

    def test_wrist_alignment_survives_effort_release_after_contact(self) -> None:
        controller = ContactGraspController(kdl=FakeKdl())
        expected_slide = 1.32163718 - 0.854
        baseline = joint_states(slide=expected_slide)
        for tick in range(9):
            controller.prepare_compliance(tick * 0.05, baseline)
        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            baseline,
        )

        contact = joint_states(
            slide=expected_slide,
            left_arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
            right_arm=[0.0, 0.0, 0.0, 0.0, 0.0, -0.04],
            left_effort=[0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            right_effort=[0.0, 0.0, 0.0, 0.0, 0.0, -0.8],
        )
        now_s = 0.50
        for _ in range(20):
            controller.update(now_s, contact)
            now_s += 0.05
            if controller._left_wrist.contact_seen and controller._right_wrist.contact_seen:
                break
        self.assertTrue(controller.bilateral_contact_seen)

        # Surface alignment unloads joint 6 back near its baseline.  Contact
        # remains latched and both wrists must still lock from angle + velocity.
        relaxed = joint_states(
            slide=expected_slide,
            left_arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
            right_arm=[0.0, 0.0, 0.0, 0.0, 0.0, -0.04],
        )
        for _ in range(20):
            controller.update(now_s, relaxed)
            now_s += 0.05
            if controller.bilateral_aligned:
                break

        self.assertTrue(controller.bilateral_aligned)
        self.assertLess(
            controller._left_wrist.latest_effort_delta,
            controller._left_wrist.effort_threshold,
        )

    def test_retry_preserves_an_already_locked_wrist(self) -> None:
        controller = ContactGraspController(kdl=FakeKdl())
        expected_slide = 1.32163718 - 0.854
        feedback = joint_states(slide=expected_slide)
        for tick in range(9):
            controller.prepare_compliance(tick * 0.05, feedback)
        controller.plan(
            (-0.18, 2.20, 0.834),
            "yaw0",
            odometry(-0.18, 1.55, math.pi / 2.0),
            feedback,
        )
        controller._left_wrist.contact_seen = True
        controller._left_wrist.aligned = True
        controller._left_wrist.locked_position = 0.04
        controller._right_wrist.contact_seen = False

        controller.retry_compliance()

        self.assertTrue(controller._left_wrist.contact_seen)
        self.assertTrue(controller._left_wrist.aligned)
        self.assertEqual(controller._left_wrist.locked_position, 0.04)
        self.assertFalse(controller._right_wrist.contact_seen)

    def test_missing_effort_keeps_legacy_contact_available(self) -> None:
        controller = ContactGraspController(kdl=FakeKdl())
        feedback = joint_states()
        feedback.effort = []

        ready, detail = controller.prepare_compliance(0.0, feedback)

        self.assertTrue(ready)
        self.assertFalse(controller.compliance_enabled)
        self.assertIn("legacy_no_effort", detail)

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

    def test_retries_when_both_contacts_latched_but_one_wrist_unaligned(self) -> None:
        contact_controller = FakeCompliantContactController()
        # Reproduce the remote trace: both wrists once latched contact, but
        # only one completed the angle/velocity alignment debounce.
        contact_controller.any_contact = True
        contact_controller.bilateral_contact_seen = True
        contact_controller.bilateral_aligned = False
        executor = Task1ContactExecutor(
            pregrasp_controller=FakePregraspController(),
            contact_controller=contact_controller,
        )
        executor._locked_target_world = (-0.18, 2.20, 0.834)
        executor._held_arm_command = ARM_COMMAND
        executor._contact_search_used_m = executor.COMPLIANT_SOFT_MAX_M
        executor._compliance_wait_since_s = 0.0

        result = executor._tick_compliant_contact_search(
            context(2.10, odometry(-0.18, 1.55, math.pi / 2.0)),
            ARM_COMMAND,
            True,
            "fake settled pose",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._compliance_retry_count, 1)
        self.assertAlmostEqual(
            executor._contact_search_used_m,
            executor.COMPLIANT_SOFT_MAX_M - executor.COMPLIANT_RETRY_BACKOFF_M,
        )
        self.assertIn("incomplete bilateral wrist alignment", result.message)


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

    def test_compliant_alignment_adds_two_millimetres_locked_preload(self) -> None:
        contact_controller = FakeCompliantContactController()
        executor = Task1LiftExecutor(
            pregrasp_controller=FakePregraspController(),
            contact_controller=contact_controller,
            lift_controller=FakeLiftController(),
        )
        grasp_context = context(
            0.0,
            odometry(-0.18, 1.55, math.pi / 2.0),
        )
        executor._locked_target_world = (-0.18, 2.20, 0.834)
        executor._held_arm_command = ARM_COMMAND
        executor.enter_stage(TaskStage.GRASP, grasp_context)

        result = None
        for tick in range(100):
            result = executor.tick(
                TaskStage.GRASP,
                context(
                    tick * 0.30,
                    odometry(-0.18, 1.55, math.pi / 2.0),
                ),
            )
            if result.status is StageStatus.SUCCEEDED:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result.status, StageStatus.SUCCEEDED)
        offsets = contact_controller.tighten_offsets
        self.assertGreater(len(offsets), 8)
        self.assertAlmostEqual(offsets[-1], COMPLIANT_ENTRY_TRAVEL_M, places=6)
        self.assertTrue(
            all(later > earlier for earlier, later in zip(offsets, offsets[1:]))
        )
        self.assertLessEqual(
            max(later - earlier for earlier, later in zip(offsets, offsets[1:])),
            0.0002 + 1e-9,
        )
        self.assertIn("locked wrists", result.message)

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
