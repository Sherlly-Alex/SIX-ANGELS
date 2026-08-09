"""Integrated Task 3 executor.

Task 3:
    dynamic instruction target_color
    -> teacher Task-1 table pick / dual-arm grasp / lift
    -> teacher shelf transport and shelf recognition
    -> live RGB-D packaging_box reference
    -> place on packaging-box left side
    -> release / retreat / return

No GT object pose and no fixed yellow target are used here.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import desktop_grasp.pregrasp_core as pregrasp_core
from control_types import ArmCommand

from desktop_grasp.pregrasp_core import (
    ContactGraspController,
    PregraspInputError,
    PregraspPlanningError,
    SlideLiftController,
    _joint_maps,
)
from executors.base import (
    ExecutionContext,
    StageResult,
    StageStatus,
    TaskStage,
)
from executors.base import PlaceholderTaskExecutor
from executors.task1_full import (
    Task1IntegratedExecutor,
    stand_from_held_center,
)
from shelf.manipulation import ReleaseSpreadController
from shelf.task_memory import CompetitionTaskMemory
from shelf.target_center import StableTargetCenterTracker


class Task3Executor(PlaceholderTaskExecutor):
    task_id = 3
    name = "task3_table_top_to_shelf_prop_side"


class Task3StableOpenPregraspController(
    pregrasp_core.OpenPregraspController
):
    """Task3 open pregrasp with independent measured-seeded arm IK.

    The target geometry is identical to OpenPregraspController.  Only IK
    branch selection differs: each arm is solved independently using that
    arm's current measured joints as its 7-element reference pose.
    """

    # Exact high-support pregrasp geometry from the validated Task3 runner:
    # grip_half=0.130, pre_grasp_fwd=-0.045, grasp_z=-0.010.  The generic
    # Task1 open pose uses a 0.225 m half width and +0.020 m hand-Z offset;
    # at the raised white support that route leaves the left arm physically
    # stalled.  Keep this adjustment strictly inside Task3.
    PREGRASP_HALF_WIDTH_M = 0.130
    PREGRASP_X_OFFSET_M = -0.045
    PREGRASP_Z_OFFSET_M = -0.010

    def plan(self, target_world, odometry, joint_states):
        adjusted_target = (
            float(target_world[0]),
            float(target_world[1]),
            float(target_world[2])
            + self.PREGRASP_Z_OFFSET_M
            - pregrasp_core.HAND_Z_OFFSET,
        )
        return self._plan_pose(
            adjusted_target,
            odometry,
            joint_states,
            center_backoff_x=-self.PREGRASP_X_OFFSET_M,
            half_width=self.PREGRASP_HALF_WIDTH_M,
        )

    def _plan_pose(
        self,
        target_world,
        odometry,
        joint_states,
        *,
        center_backoff_x,
        half_width,
    ):
        if not all(math.isfinite(float(value)) for value in target_world):
            raise PregraspInputError(
                "Task3 target_world contains non-finite values"
            )
        if not math.isfinite(float(center_backoff_x)):
            raise PregraspInputError(
                "Task3 center_backoff_x is non-finite"
            )
        if (
            not math.isfinite(float(half_width))
            or float(half_width) <= 0.0
        ):
            raise PregraspInputError(
                "Task3 half_width must be finite and positive"
            )

        positions, _velocities = pregrasp_core._joint_maps(joint_states)
        robot_pose = pregrasp_core._odometry_pose(odometry)
        box_center_base = pregrasp_core._world_to_base(
            target_world,
            robot_pose,
        )
        self._target_base = tuple(
            float(value) for value in box_center_base
        )

        arm_center_base = box_center_base + np.array(
            [
                -float(center_backoff_x),
                0.0,
                pregrasp_core.HAND_Z_OFFSET,
            ],
            dtype=float,
        )

        left_target = arm_center_base + np.array(
            [0.0, float(half_width), 0.0],
            dtype=float,
        )
        right_target = arm_center_base + np.array(
            [0.0, -float(half_width), 0.0],
            dtype=float,
        )

        slide_target = float(
            np.clip(
                pregrasp_core.SPINE_REFERENCE_Z
                - arm_center_base[2],
                pregrasp_core.SPINE_MIN,
                pregrasp_core.SPINE_MAX,
            )
        )

        # Important difference from the generic dual-arm planner:
        # seed each arm from its own currently measured configuration.
        left_ref = np.array(
            [
                positions["slide_joint"],
                *(
                    positions[f"left_arm_joint{index}"]
                    for index in range(1, 7)
                ),
            ],
            dtype=float,
        )
        right_ref = np.array(
            [
                positions["slide_joint"],
                *(
                    positions[f"right_arm_joint{index}"]
                    for index in range(1, 7)
                ),
            ],
            dtype=float,
        )

        left_solutions = self._kdl.inverse_kinematics(
            T_left=pregrasp_core._make_transform(
                left_target,
                pregrasp_core.LEFT_A_ROT,
            ),
            T_right=None,
            ref_pos=left_ref,
            target_height=slide_target,
        )
        if left_solutions is None or len(left_solutions) == 0:
            raise PregraspPlanningError(
                "Task3 independent left-arm pregrasp IK failed; "
                f"target_base={np.round(box_center_base, 3).tolist()}, "
                f"left_target={np.round(left_target, 3).tolist()}"
            )

        right_solutions = self._kdl.inverse_kinematics(
            T_left=None,
            T_right=pregrasp_core._make_transform(
                right_target,
                pregrasp_core.RIGHT_A_ROT,
            ),
            ref_pos=right_ref,
            target_height=slide_target,
        )
        if right_solutions is None or len(right_solutions) == 0:
            raise PregraspPlanningError(
                "Task3 independent right-arm pregrasp IK failed; "
                f"target_base={np.round(box_center_base, 3).tolist()}, "
                f"right_target={np.round(right_target, 3).tolist()}"
            )

        left_joints = np.asarray(left_solutions[0], dtype=float)
        right_joints = np.asarray(right_solutions[0], dtype=float)

        if left_joints.shape != (7,) or not np.all(
            np.isfinite(left_joints)
        ):
            raise PregraspPlanningError(
                "Task3 left-arm IK returned invalid result "
                f"shape={left_joints.shape}"
            )

        if right_joints.shape != (7,) or not np.all(
            np.isfinite(right_joints)
        ):
            raise PregraspPlanningError(
                "Task3 right-arm IK returned invalid result "
                f"shape={right_joints.shape}"
            )

        self._target_vector = np.array(
            [
                slide_target,
                pregrasp_core.HEAD_TARGET[0],
                pregrasp_core.HEAD_TARGET[1],
                *left_joints[1:7],
                pregrasp_core.GRIPPER_OPEN,
                *right_joints[1:7],
                pregrasp_core.GRIPPER_OPEN,
            ],
            dtype=float,
        )

        # Start exactly from measured feedback, preserving the existing
        # OpenPregraspController ramp/update implementation.
        self._action_vector = np.array(
            [
                positions["slide_joint"],
                positions.get("head_yaw_joint", 0.0),
                positions.get("head_pitch_joint", 0.0),
                *(
                    positions[f"left_arm_joint{index}"]
                    for index in range(1, 7)
                ),
                pregrasp_core.GRIPPER_OPEN,
                *(
                    positions[f"right_arm_joint{index}"]
                    for index in range(1, 7)
                ),
                pregrasp_core.GRIPPER_OPEN,
            ],
            dtype=float,
        )

        self._last_update_s = None
        self._stable_since_s = None

        return self.command()


class Task3DirectHugController(
    pregrasp_core.OpenPregraspController
):
    """Reproduce the previously successful Task3 direct symmetric hug.

    Exact old geometry in base frame:
      grasp_center = box_center + (-0.045, 0.0, -0.010)
      left  = grasp_center + (0, +0.085, 0)
      right = grasp_center + (0, -0.085, 0)

    Each arm is solved independently from its measured configuration,
    matching the old set_arm() branch-selection behaviour.
    """

    HOLD_HALF_WIDTH_M = 0.080
    GRASP_X_OFFSET_M = -0.045
    GRASP_Z_OFFSET_M = -0.010

    # Old Task3 final squeeze was intentionally slower than normal arm motion.
    COMMAND_RATE_PER_S = 0.45
    COMMAND_TOLERANCE = 0.020
    SETTLE_TIME_S = 2.8

    def __init__(self):
        super().__init__()
        self._grasp_center_base = None

    @property
    def grasp_center_base(self):
        return self._grasp_center_base

    def plan(
        self,
        target_world,
        odometry,
        joint_states,
    ):
        if not all(math.isfinite(float(v)) for v in target_world):
            raise PregraspInputError(
                "Task3 direct-hug target contains non-finite values"
            )

        positions, _velocities = pregrasp_core._joint_maps(joint_states)
        robot_pose = pregrasp_core._odometry_pose(odometry)

        box_center_base = pregrasp_core._world_to_base(
            target_world,
            robot_pose,
        )
        self._target_base = tuple(
            float(v) for v in box_center_base
        )

        grasp_center = box_center_base + np.array(
            [
                self.GRASP_X_OFFSET_M,
                0.0,
                self.GRASP_Z_OFFSET_M,
            ],
            dtype=float,
        )
        self._grasp_center_base = tuple(
            float(v) for v in grasp_center
        )

        left_target = grasp_center + np.array(
            [0.0, self.HOLD_HALF_WIDTH_M, 0.0],
            dtype=float,
        )
        right_target = grasp_center + np.array(
            [0.0, -self.HOLD_HALF_WIDTH_M, 0.0],
            dtype=float,
        )

        slide_target = float(
            np.clip(
                pregrasp_core.SPINE_REFERENCE_Z - grasp_center[2],
                pregrasp_core.SPINE_MIN,
                pregrasp_core.SPINE_MAX,
            )
        )

        # Exactly like the old set_arm(): solve each side from its
        # own measured 6-joint configuration.
        left_ref = np.array(
            [
                positions["slide_joint"],
                *(
                    positions[f"left_arm_joint{i}"]
                    for i in range(1, 7)
                ),
            ],
            dtype=float,
        )
        right_ref = np.array(
            [
                positions["slide_joint"],
                *(
                    positions[f"right_arm_joint{i}"]
                    for i in range(1, 7)
                ),
            ],
            dtype=float,
        )

        left_solutions = self._kdl.inverse_kinematics(
            T_left=pregrasp_core._make_transform(
                left_target,
                pregrasp_core.LEFT_A_ROT,
            ),
            T_right=None,
            ref_pos=left_ref,
            target_height=slide_target,
        )
        if left_solutions is None or len(left_solutions) == 0:
            raise PregraspPlanningError(
                "Task3 old-hug left IK failed: "
                f"box_base={np.round(box_center_base,3).tolist()}, "
                f"left={np.round(left_target,3).tolist()}"
            )

        right_solutions = self._kdl.inverse_kinematics(
            T_left=None,
            T_right=pregrasp_core._make_transform(
                right_target,
                pregrasp_core.RIGHT_A_ROT,
            ),
            ref_pos=right_ref,
            target_height=slide_target,
        )
        if right_solutions is None or len(right_solutions) == 0:
            raise PregraspPlanningError(
                "Task3 old-hug right IK failed: "
                f"box_base={np.round(box_center_base,3).tolist()}, "
                f"right={np.round(right_target,3).tolist()}"
            )

        left_joints = np.asarray(left_solutions[0], dtype=float)
        right_joints = np.asarray(right_solutions[0], dtype=float)

        if left_joints.shape != (7,) or not np.all(np.isfinite(left_joints)):
            raise PregraspPlanningError(
                f"Task3 old-hug left IK invalid: {left_joints.shape}"
            )
        if right_joints.shape != (7,) or not np.all(np.isfinite(right_joints)):
            raise PregraspPlanningError(
                f"Task3 old-hug right IK invalid: {right_joints.shape}"
            )

        head_yaw = positions.get("head_yaw_joint", 0.0)
        head_pitch = positions.get("head_pitch_joint", 0.0)

        self._target_vector = np.array(
            [
                slide_target,
                head_yaw,
                head_pitch,
                *left_joints[1:7],
                pregrasp_core.GRIPPER_OPEN,
                *right_joints[1:7],
                pregrasp_core.GRIPPER_OPEN,
            ],
            dtype=float,
        )

        # Begin from measured feedback.
        self._action_vector = np.array(
            [
                positions["slide_joint"],
                head_yaw,
                head_pitch,
                *(
                    positions[f"left_arm_joint{i}"]
                    for i in range(1, 7)
                ),
                pregrasp_core.GRIPPER_OPEN,
                *(
                    positions[f"right_arm_joint{i}"]
                    for i in range(1, 7)
                ),
                pregrasp_core.GRIPPER_OPEN,
            ],
            dtype=float,
        )

        self._last_update_s = None
        self._stable_since_s = None
        return self.command()

    def update(
        self,
        now_s,
        joint_states,
    ):
        if self._target_vector is None or self._action_vector is None:
            raise PregraspPlanningError(
                "Task3 direct-hug update before plan"
            )

        positions, velocities = pregrasp_core._joint_maps(joint_states)
        now = float(now_s)

        if self._last_update_s is None:
            dt = 0.05
        else:
            dt = min(
                0.20,
                max(0.01, now - self._last_update_s),
            )
        self._last_update_s = now

        diff = np.abs(
            self._target_vector - self._action_vector
        )
        ratios = diff / (float(np.max(diff)) + 1e-6)

        # Same slow-spine weighting used by the existing controllers.
        ratios[0] *= pregrasp_core.SLIDE_COMMAND_RATIO

        steps = (
            ratios
            * self.COMMAND_RATE_PER_S
            * dt
        )

        self._action_vector += np.sign(
            self._target_vector - self._action_vector
        ) * np.minimum(diff, steps)

        measured = np.array(
            [
                positions["slide_joint"],
                *(
                    positions[f"left_arm_joint{i}"]
                    for i in range(1, 7)
                ),
                *(
                    positions[f"right_arm_joint{i}"]
                    for i in range(1, 7)
                ),
            ],
            dtype=float,
        )

        target = np.concatenate(
            (
                self._target_vector[0:1],
                self._target_vector[3:9],
                self._target_vector[10:16],
            )
        )
        commanded = np.concatenate(
            (
                self._action_vector[0:1],
                self._action_vector[3:9],
                self._action_vector[10:16],
            )
        )

        errors = np.abs(measured - target)

        slide_error = float(errors[0])
        left_error = float(np.max(errors[1:7]))
        right_error = float(np.max(errors[7:13]))
        command_error = float(
            np.max(np.abs(commanded - target))
        )

        max_velocity = max(
            abs(
                float(
                    velocities.get(
                        f"{side}_arm_joint{i}",
                        0.0,
                    )
                )
            )
            for side in ("left", "right")
            for i in range(1, 7)
        )

        # IMPORTANT:
        # The old successful hug did NOT require measured joints to reach
        # the unconstrained IK goal.  Physical contact is supposed to stop
        # the arms short and create preload.  We only require the commanded
        # ramp itself to have reached the target.
        command_settled = (
            command_error <= self.COMMAND_TOLERANCE
        )

        detail = (
            f"box_base={tuple(round(v,3) for v in self._target_base)}, "
            f"grasp_center={tuple(round(v,3) for v in self._grasp_center_base)}, "
            f"half={self.HOLD_HALF_WIDTH_M:.3f}, "
            f"cmd_err={command_error:.3f}, "
            f"slide_err={slide_error:.3f}, "
            f"left_residual={left_error:.3f}, "
            f"right_residual={right_error:.3f}, "
            f"max_vel={max_velocity:.3f}"
        )

        return self.command(), command_settled, detail


class Task3ReferenceContactController(ContactGraspController):
    """Task-1 contact motion with the validated old Task-3 hold width."""

    HOLD_HALF_WIDTH_M = 0.080
    CENTER_BACKOFF_X_M = 0.0

    def plan(
        self,
        target_world,
        orientation,
        odometry,
        joint_states,
    ):
        self._orientation = str(orientation)
        self._half_width = self.HOLD_HALF_WIDTH_M
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=self.CENTER_BACKOFF_X_M,
            half_width=self._half_width,
        )

    def tighten(
        self,
        target_world,
        inward_offset,
        odometry,
        joint_states,
    ):
        if self._orientation is None:
            raise PregraspPlanningError(
                "Task3 contact tighten requested before initial plan"
            )
        offset = float(inward_offset)
        if not math.isfinite(offset) or offset < 0.0:
            raise PregraspInputError(
                "Task3 inward_offset must be finite and non-negative"
            )
        self._half_width = max(self.HOLD_HALF_WIDTH_M - offset, 0.01)
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=self.CENTER_BACKOFF_X_M,
            half_width=self._half_width,
        )

    def contact_metrics(self, joint_states):
        """Return measured bilateral IK residuals and joint speed."""

        if self._target_vector is None:
            return None
        positions, velocities = _joint_maps(joint_states)
        left_error = max(
            abs(
                float(positions[f"left_arm_joint{index}"])
                - float(self._target_vector[2 + index])
            )
            for index in range(1, 7)
        )
        right_error = max(
            abs(
                float(positions[f"right_arm_joint{index}"])
                - float(self._target_vector[9 + index])
            )
            for index in range(1, 7)
        )
        max_velocity = max(
            abs(float(velocities.get(f"{side}_arm_joint{index}", 0.0)))
            for side in ("left", "right")
            for index in range(1, 7)
        )
        return left_error, right_error, max_velocity


class Task3GripperOnlyReleaseController:
    """Open both grippers without sweeping either arm farther into the shelf."""

    SETTLE_S = 0.8

    def __init__(self) -> None:
        self._command: ArmCommand | None = None
        self._started_s: float | None = None

    @property
    def planned(self) -> bool:
        return self._command is not None

    def reset(self) -> None:
        self._command = None
        self._started_s = None

    def plan(
        self,
        target_world,
        odometry,
        joint_states,
        *,
        half_width=0.18,
    ):
        # Freeze every measured arm joint at its current shallow-place pose.
        # Only the two gripper actuators open, so there is no forward IK sweep
        # capable of shoving the box deeper or striking the shelf frame.
        positions, _velocities = _joint_maps(joint_states)
        self._command = ArmCommand(
            spine_position=float(positions["slide_joint"]),
            head_positions=(
                float(positions.get("head_yaw_joint", 0.0)),
                float(positions.get("head_pitch_joint", 0.0)),
            ),
            left_arm_positions=tuple(
                float(positions[f"left_arm_joint{index}"])
                for index in range(1, 7)
            ),
            left_gripper_position=float(pregrasp_core.GRIPPER_OPEN),
            right_arm_positions=tuple(
                float(positions[f"right_arm_joint{index}"])
                for index in range(1, 7)
            ),
            right_gripper_position=float(pregrasp_core.GRIPPER_OPEN),
        )
        self._started_s = None
        return self._command

    def update(self, now_s, joint_states):
        if self._command is None:
            raise PregraspPlanningError(
                "Task3 gripper-only release update called before plan"
            )
        now = float(now_s)
        if self._started_s is None:
            self._started_s = now
        elapsed = max(0.0, now - self._started_s)
        reached = elapsed >= self.SETTLE_S
        return (
            self._command,
            reached,
            f"grippers_open_with_arms_fixed; settle={elapsed:.2f}/{self.SETTLE_S:.2f}s",
        )


class Task3IntegratedExecutor(Task1IntegratedExecutor):
    """Reuse the proven Task-1 manipulation chain for formal Task 3."""

    task_id = 3
    name = "task3_integrated_table_top_to_packaging_left"

    # Reuse Task1's contact controller, but feed it Task3's physical box
    # orientation.  The white-support target is yaw90 in the competition
    # scene; Task1's yaw0 default leaves the arms about 42 mm too far apart
    # and can lift empty.
    SOURCE_ORIENTATION = "yaw90"

    # With a near-edge grip the box remains far in front of the base.  Reverse
    # all the way to the safe turn row before allowing any yaw motion; the
    # shorter Task1 retreat lets the planned arc sweep the carried box back
    # through the source table and stalls the chassis.
    TABLE_RETREAT_M = 0.80
    # After the straight table retreat, navigate directly to the shelf-front
    # placement row.  Starting from the north-facing retreat pose makes the
    # navigation controller enter its rotate-in-place gate for the south-west
    # route, then follow the single diagonal cleanly.  The inherited forced
    # west turn was redundant and immediately followed by another path turn;
    # in GS that arc stopped at (-0.899, 0.635) with 5 cm carry clearance.
    FORCE_SHELF_FACING_TURN_BEFORE_NAVIGATION = False
    # Task3 is centred on the raised white support, not beside the east wall.
    # Navigate directly to the real 0.620 m dual-arm working distance.
    TABLE_STANDOFF_M = 0.620
    TABLE_WALL_CLEARANCE_OFFSET_M = 0.0

    # Task3 starts on the coloured block resting on the white material_box,
    # between Task1's two calibrated table slots. This broad gate accepts
    # only client RGB-D observations in that source region.
    TASK3_SOURCE_ROI = (
        (-0.90, -0.10),
        (2.00, 2.55),
        (0.80, 1.25),
    )
    TASK3_SUPPORT_COLOR = "material_box"
    TASK3_SOURCE_MAX_AGE_S = 0.75
    TASK3_SOURCE_ACQUIRE_TIMEOUT_S = 12.0
    # On a client-only Task3 retry the YOLO process starts after the referee
    # stage clock.  Leave enough time for model loading and the first frames.
    TARGET_WAIT_TIMEOUT_S = 90.0
    # In a Task3-only diagnostic the detector process starts at the same time
    # as this executor and needs several seconds to load its model.  Stay
    # still first so the initial table view is not lost, then scan in short
    # increments separated by stationary collection windows.
    TASK3_DETECTOR_STARTUP_HOLD_S = 12.0
    TASK3_SCAN_TURN_S = 1.0
    TASK3_SCAN_HOLD_S = 2.0
    # The first navigation goal is made from the earliest usable detector
    # frame.  After the multi-frame lock, refine the stand so the target is
    # centred between both arms and remains inside their shared workspace.
    # Keep Task1's stable dual-arm sequence, but reproduce the validated old
    # Task3 grasp geometry.  The inherited controller adds +20 mm in approach
    # X; shifting its target 65 mm toward the robot leaves the actual hand
    # centre 45 mm behind the detected box centre (JY_YELLOW_GRASP_FWD=-0.045).
    # Its +20 mm hand-Z offset plus -30 mm here yields the old -10 mm Z.
    TASK3_PICK_STANDOFF_M = 0.620
    TASK3_PICK_STAND_TOLERANCE_M = 0.015
    TASK3_SHALLOW_GRIP_OFFSET_M = 0.045
    TASK3_GRASP_Z_OFFSET_M = -0.010
    TASK3_SOURCE_CENTER_Z = 1.004
    TASK3_REQUIRED_CONTACT_SEARCH_M = 0.004

    # Unlike Task1's formal lift executor, Task3 may not lift merely because
    # the bounded IK target stopped moving.  The shallow grasp must first be
    # reported as stable bilateral target contact.
    REQUIRE_SERVER_CONTACT = True
    ALLOW_SETTLED_MAX_SEARCH = False
    CONTACT_TIMEOUT_S = 25.0

    @classmethod
    def _calibrated_table_source(
        cls,
        observation_world: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        """Accept a live RGB-D source point in Task3's support-table ROI."""

        try:
            point = tuple(float(value) for value in observation_world)
        except (TypeError, ValueError):
            return None
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        if not all(
            limits[0] <= value <= limits[1]
            for value, limits in zip(point, cls.TASK3_SOURCE_ROI)
        ):
            return None
        return point

    def _task1_compat_context(
        self,
        context: ExecutionContext,
    ) -> ExecutionContext:
        """Adapt Task3 to the inherited Task1 manipulation machinery.

        The inherited Task1 executor requires the instruction task to match
        the executor task_id and the place type to be shelf_point. Task3
        reuses that proven mechanical chain while the referee remains Task3.

        Preserve the real Task3 target_color and all sensor/runtime data.
        """
        instruction = dict(context.instruction)
        instruction["task"] = self.task_id
        instruction["place_type"] = "shelf_point"
        return replace(context, instruction=instruction)

    def tick(
        self,
        stage: TaskStage,
        context: ExecutionContext,
    ) -> StageResult:
        compat_context = self._task1_compat_context(context)

        if stage is TaskStage.NAVIGATE_TO_PICK:
            target_color = str(
                context.instruction.get("target_color", "")
            ).strip().lower()
            support_observation = context.target_observations.get(
                self.TASK3_SUPPORT_COLOR
            )
            if support_observation is not None:
                support_age_s = max(
                    0.0,
                    float(context.now_s)
                    - float(support_observation.received_at_s),
                )
                support_point = tuple(
                    float(value)
                    for value in support_observation.position_world
                )
                if (
                    support_age_s <= self.TASK3_SOURCE_MAX_AGE_S
                    and len(support_point) == 3
                    and all(math.isfinite(value) for value in support_point)
                    and self.TASK3_SOURCE_ROI[0][0]
                    <= support_point[0]
                    <= self.TASK3_SOURCE_ROI[0][1]
                    and self.TASK3_SOURCE_ROI[1][0]
                    <= support_point[1]
                    <= self.TASK3_SOURCE_ROI[1][1]
                ):
                    self._task3_support_world = support_point
            observation = context.target_observations.get(target_color)
            estimate = self._task3_source_tracker.update(
                observation,
                now_s=context.now_s,
            )
            if (
                estimate is not None
                and self._task3_source_estimate is None
            ):
                self._task3_source_estimate = estimate

            # A normal full run sees the table target immediately.  On a
            # client-only Task3 retry the base can begin facing away from it;
            # actively scan in yaw until the first fresh in-ROI frame exists,
            # then let the inherited navigator move while this tracker keeps
            # collecting samples.
            usable_observation = False
            if observation is not None:
                age_s = max(
                    0.0,
                    float(context.now_s) - float(observation.received_at_s),
                )
                usable_observation = (
                    age_s <= self.TARGET_MAX_AGE_S
                    and self._calibrated_table_source(
                        observation.position_world
                    )
                    is not None
                )
            if self._goal is None and not usable_observation:
                waited_s = max(
                    0.0,
                    float(context.now_s) - self._stage_started_s,
                )
                if waited_s >= self.TARGET_WAIT_TIMEOUT_S:
                    return StageResult.blocked(
                        f"Task3 timed out actively scanning for {target_color!r}"
                    )
                if waited_s < self.TASK3_DETECTOR_STARTUP_HOLD_S:
                    return StageResult.running(
                        f"Task3 holding initial view while detector starts for "
                        f"{target_color!r}; waited={waited_s:.1f}s",
                        base_command=(0.0, 0.0),
                    )
                scan_cycle_s = self.TASK3_SCAN_TURN_S + self.TASK3_SCAN_HOLD_S
                scan_phase_s = (
                    waited_s - self.TASK3_DETECTOR_STARTUP_HOLD_S
                ) % scan_cycle_s
                if scan_phase_s >= self.TASK3_SCAN_TURN_S:
                    return StageResult.running(
                        f"Task3 paused to collect {target_color!r} RGB-D frames",
                        base_command=(0.0, 0.0),
                    )
                return StageResult.running(
                    f"Task3 stepping yaw scan for first {target_color!r} "
                    "RGB-D frames",
                    base_command=(0.0, 0.40),
                )
            if self._goal is None and self._task3_source_estimate is None:
                return StageResult.running(
                    f"Task3 holding for stable {target_color!r} source centre "
                    f"before navigation; {self._task3_source_tracker.status()}",
                    base_command=(0.0, 0.0),
                )

            # Feed the inherited table navigator the stable multi-frame centre,
            # preferring the static white support XY used by manipulation.
            # This makes its one navigation goal equal the real grasp stand
            # instead of correcting a noisy first detector frame afterwards.
            if self._goal is None:
                assert observation is not None
                assert self._task3_source_estimate is not None
                stable_xy = (
                    self._task3_support_world[:2]
                    if self._task3_support_world is not None
                    else self._task3_source_estimate.center_world[:2]
                )
                stable_observation = replace(
                    observation,
                    position_world=(
                        float(stable_xy[0]),
                        float(stable_xy[1]),
                        self.TASK3_SOURCE_CENTER_Z,
                    ),
                )
                stable_observations = dict(compat_context.target_observations)
                stable_observations[target_color] = stable_observation
                compat_context = replace(
                    compat_context, target_observations=stable_observations
                )

        if stage is TaskStage.ACQUIRE_TARGET:
            return self._tick_task3_acquire_target(context)

        if stage is TaskStage.ALIGN_FOR_PICK:
            return self._tick_task3_align_for_pick(compat_context)

        if stage is TaskStage.GRASP:
            return self._tick_task3_direct_hug(context)

        if stage is TaskStage.LIFT:
            result = super().tick(stage, compat_context)
            if result.status is not StageStatus.SUCCEEDED:
                if result.status is StageStatus.RUNNING:
                    return result
                # The inherited Task1 lift may time out on arm-pose residuals
                # even though the randomized Task3 block visibly rose with the
                # grippers. Physical RGB-D displacement is the stronger proof:
                # accept it before propagating a mechanical timeout.
                target_color = str(
                    context.instruction.get("target_color", "")
                ).strip().lower()
                observation = context.target_observations.get(target_color)
                baseline_z = getattr(
                    self, "_task3_pre_lift_observed_z", None
                )
                if observation is not None and baseline_z is not None:
                    age_s = max(
                        0.0,
                        float(context.now_s)
                        - float(observation.received_at_s),
                    )
                    observed_z = float(observation.position_world[2])
                    rise = observed_z - float(baseline_z)
                    if (
                        age_s <= self.TASK3_SOURCE_MAX_AGE_S
                        and rise >= 0.065
                    ):
                        self._capture_held_center(compat_context)
                        self._task3_lift_verified = True
                        self._task3_transport_retreat_seen = False
                        self._task3_left_turn_active = False
                        self._task3_left_turn_done = False
                        return StageResult.succeeded(
                            "Task3 RGB-D lift proof overrode inherited arm "
                            f"residual timeout: {target_color} z={observed_z:.3f}, "
                            f"rise={rise:.3f} m; real grasp confirmed",
                            arm_command=(
                                result.arm_command or self._held_arm_command
                            ),
                        )
                return result

            target_color = str(
                context.instruction.get("target_color", "")
            ).strip().lower()
            observation = context.target_observations.get(target_color)
            elapsed = max(
                0.0,
                float(context.now_s) - float(self._stage_started_s),
            )

            if observation is not None:
                age_s = max(
                    0.0,
                    float(context.now_s) - float(observation.received_at_s),
                )
                observed = tuple(
                    float(value) for value in observation.position_world
                )
                baseline_z = getattr(
                    self, "_task3_pre_lift_observed_z", None
                )
                rise = (
                    observed[2] - float(baseline_z)
                    if baseline_z is not None
                    else float("-inf")
                )

                if (
                    baseline_z is not None
                    and age_s <= self.TASK3_SOURCE_MAX_AGE_S
                    and rise >= 0.045
                ):
                    self._task3_lift_verified = True
                    self._task3_transport_retreat_seen = False
                    self._task3_left_turn_active = False
                    self._task3_left_turn_done = False
                    return replace(
                        result,
                        message=(
                            "Task3 RGB-D lift proof passed: "
                            f"{target_color} z={observed[2]:.3f}, "
                            f"rise={rise:.3f} m; real grasp confirmed"
                        ),
                    )

                if baseline_z is None:
                    detail = (
                        f"{target_color} z={observed[2]:.3f}, "
                        "pre-lift RGB-D baseline missing"
                    )
                else:
                    detail = (
                        f"{target_color} pre_z={baseline_z:.3f}, "
                        f"post_z={observed[2]:.3f}, "
                        f"rise={rise:.3f} m"
                    )
            else:
                detail = f"no fresh {target_color!r} observation"

            if elapsed >= 12.0:
                return StageResult.blocked(
                    "Task3 lift finished mechanically but RGB-D did not "
                    f"confirm the box was lifted: {detail}",
                    arm_command=result.arm_command,
                )

            return StageResult.running(
                "Task3 holding lifted pose while verifying real object lift; "
                + detail,
                base_command=(0.0, 0.0),
                arm_command=result.arm_command,
            )

        if stage is TaskStage.TRANSPORT:
            if not getattr(self, "_task3_lift_verified", False):
                return StageResult.blocked(
                    "Task3 refused transport because real RGB-D lift proof "
                    "was not obtained",
                    arm_command=self._held_arm_command,
                )
            # The parent now plans directly from the cleared table retreat to
            # the shelf scan stand.  No extra forced turn or intermediate
            # waypoint is needed, so the carry footprint follows one route.
            return super().tick(stage, compat_context)

        if stage is TaskStage.PLACE:
            return self._tick_task3_place(compat_context)

        result = super().tick(stage, compat_context)
        if (
            stage is TaskStage.NAVIGATE_TO_PICK
            and result.status is StageStatus.RUNNING
        ):
            pose = self._odometry_pose(context.odometry)
            if pose is not None:
                return replace(
                    result,
                    message=(
                        f"{result.message}; "
                        f"pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}), "
                        f"cmd=({result.base_linear_x:.3f},"
                        f"{result.base_angular_z:.3f})"
                    ),
                )
        return result

    def _tick_task3_direct_hug(
        self,
        context: ExecutionContext,
    ) -> StageResult:
        """Run the previously successful direct symmetric Task3 hug."""

        if context.unsafe_collision:
            return StageResult.blocked(
                "Task3 direct hug stopped on unsafe collision",
                arm_command=self._held_arm_command,
            )

        if self._locked_target_world is None:
            return StageResult.blocked(
                "Task3 direct hug has no locked RGB-D box centre",
                arm_command=self._held_arm_command,
            )

        if not self._task3_direct_hug.planned:
            try:
                self._held_arm_command = self._task3_direct_hug.plan(
                    self._locked_target_world,
                    context.odometry,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"Task3 direct old-style hug planning failed: {exc}",
                    arm_command=self._held_arm_command,
                )

        try:
            command, command_settled, detail = (
                self._task3_direct_hug.update(
                    context.now_s,
                    context.joint_states,
                )
            )
        except (PregraspInputError, PregraspPlanningError) as exc:
            return StageResult.blocked(
                f"Task3 direct old-style hug control failed: {exc}",
                arm_command=self._held_arm_command,
            )

        self._held_arm_command = command

        elapsed = max(
            0.0,
            float(context.now_s) - float(self._stage_started_s),
        )

        # Reproduce the old Task3 state-6 behaviour:
        # finish commanding the hug, then leave the bilateral preload
        # settled for about 2.8 s.  Do NOT demand zero measured joint
        # residual: contact with the box is expected to stop the arms short.
        if (
            not command_settled
            or elapsed < self._task3_direct_hug.SETTLE_TIME_S
        ):
            if elapsed >= self.CONTACT_TIMEOUT_S:
                return StageResult.blocked(
                    "Task3 direct old-style hug timed out: "
                    f"{detail}",
                    arm_command=command,
                )
            return StageResult.running(
                "Task3 old-style direct symmetric hug; "
                f"settle={elapsed:.1f}/"
                f"{self._task3_direct_hug.SETTLE_TIME_S:.1f}s; "
                f"{detail}",
                arm_command=command,
            )

        # Save a fresh RAW detector Z immediately before lift.
        # Lift itself still has to prove >=45 mm real object rise.
        target_color = str(
            context.instruction.get("target_color", "")
        ).strip().lower()

        observation = context.target_observations.get(target_color)

        if observation is None:
            if elapsed >= self.CONTACT_TIMEOUT_S:
                return StageResult.blocked(
                    "Task3 direct hug settled but no fresh RGB-D "
                    "pre-lift observation was available",
                    arm_command=command,
                )
            return StageResult.running(
                "Task3 direct hug settled; holding preload while "
                f"waiting for fresh {target_color!r} pre-lift RGB-D",
                arm_command=command,
            )

        age_s = max(
            0.0,
            float(context.now_s)
            - float(observation.received_at_s),
        )

        if age_s > self.TASK3_SOURCE_MAX_AGE_S:
            return StageResult.running(
                "Task3 direct hug settled; holding preload while "
                f"waiting for fresh {target_color!r} RGB-D "
                f"(age={age_s:.2f}s)",
                arm_command=command,
            )

        observed = tuple(
            float(v) for v in observation.position_world
        )
        self._task3_pre_lift_observed_z = observed[2]
        self._task3_lift_verified = False

        return StageResult.succeeded(
            "Task3 OLD direct symmetric hug settled; "
            f"{detail}; "
            f"pre_lift_z={observed[2]:.3f}; "
            "proceeding to real RGB-D lift proof",
            arm_command=command,
        )


    def _tick_straight_advance(
        self,
        context: ExecutionContext,
        target_xy: tuple[float, float],
        *,
        action: str,
    ):
        """Task3-only final shelf-entry tolerance.

        The inherited Task1 helper requires about 15 mm final forward error.
        During Task3 the carried box can already be correctly seated while
        physical shelf/object contact prevents the chassis from closing the
        last few centimetres.  Accept <=50 mm only for the final Task3 shelf
        entry; all Task1/Task2 behaviour remains unchanged.
        """

        if action == "entering the recognized empty shelf layer":
            # The shallow Task3 grip flexes under shelf contact, so the frozen
            # held-center transform no longer predicts the real box X/Y to
            # centimetre accuracy.  Stop from the carried box itself before
            # contact can shove it deeper or sideways into packaging_box.
            if self._place_world is not None:
                target_color = str(
                    context.instruction.get("target_color", "")
                ).strip().lower()
                observation = context.target_observations.get(target_color)
                if observation is not None:
                    age_s = max(
                        0.0,
                        float(context.now_s)
                        - float(observation.received_at_s),
                    )
                    observed = tuple(
                        float(value)
                        for value in observation.position_world
                    )
                    target_clearance_z = (
                        float(self._place_world[2]) + self.SHELF_CLEARANCE_M
                    )
                    x_error = observed[0] - float(self._place_world[0])
                    y_error = observed[1] - float(self._place_world[1])
                    z_error = observed[2] - target_clearance_z
                    if (
                        age_s <= self.TASK3_SOURCE_MAX_AGE_S
                        and -self.TASK3_LIVE_ENTRY_INSIDE_X_TOLERANCE_M
                        <= x_error
                        <= self.TASK3_LIVE_ENTRY_OUTSIDE_X_TOLERANCE_M
                        and abs(y_error)
                        <= self.TASK3_LIVE_ENTRY_Y_TOLERANCE_M
                        and abs(z_error)
                        <= self.TASK3_LIVE_ENTRY_Z_TOLERANCE_M
                    ):
                        self._task3_live_entry_detail = (
                            f"live_box=({observed[0]:.3f},"
                            f"{observed[1]:.3f},{observed[2]:.3f}), "
                            f"errors=({x_error:+.3f},"
                            f"{y_error:+.3f},{z_error:+.3f})"
                        )
                        self._transfer.reset()
                        self._motion_started = False
                        return True, None

            pose = self._odometry_pose(context.odometry)
            if pose is not None:
                robot_x, robot_y, _robot_yaw = pose
                dx = float(target_xy[0]) - robot_x
                dy = float(target_xy[1]) - robot_y

                forward_m = (
                    math.cos(self.SHELF_YAW) * dx
                    + math.sin(self.SHELF_YAW) * dy
                )
                lateral_m = (
                    -math.sin(self.SHELF_YAW) * dx
                    + math.cos(self.SHELF_YAW) * dy
                )

                if (
                    -0.03 <= forward_m <= 0.050
                    and abs(lateral_m) <= 0.09
                ):
                    self._transfer.reset()
                    self._motion_started = False
                    return True, None

        return super()._tick_straight_advance(
            context,
            target_xy,
            action=action,
        )


    def _task3_verified_shallow_contact(
        self,
        result: StageResult,
        context: ExecutionContext,
        actual_target,
    ) -> StageResult | None:
        """Accept a settled shallow grasp when Task3-only has no contact bit."""

        if (
            result.status is not StageStatus.RUNNING
            or self._contact_search_used_m
              < self.TASK3_REQUIRED_CONTACT_SEARCH_M - 1e-9
            or actual_target is None
        ):
            return None
        metrics = self._contact.contact_metrics(context.joint_states)
        if metrics is None:
            return None
        left_error, right_error, max_velocity = metrics
        if not (
            0.035 <= left_error <= 0.180
            and 0.035 <= right_error <= 0.180
            and abs(left_error - right_error) <= 0.080
            and max_velocity <= 0.050
        ):
            return None

        target_color = str(
            context.instruction.get("target_color", "")
        ).strip().lower()
        observation = context.target_observations.get(target_color)
        if observation is None:
            return None
        age_s = max(
            0.0,
            float(context.now_s) - float(observation.received_at_s),
        )
        observed = tuple(float(value) for value in observation.position_world)
        xy_displacement = math.hypot(
            observed[0] - float(actual_target[0]),
            observed[1] - float(actual_target[1]),
        )
        if age_s > self.TASK3_SOURCE_MAX_AGE_S:
            return None
        if xy_displacement > 0.080:
            return None

        # Save RAW RGB-D Z immediately before lift.  Different classes/backends
        # have a systematic absolute-Z bias, so lift proof must compare
        # pre-lift and post-lift observations from the same detector.
        self._task3_pre_lift_observed_z = observed[2]

        return StageResult.succeeded(
            "Task3 contact pose settled; allowing diagnostic lift: "
            f"arm_residuals=({left_error:.3f}/{right_error:.3f} m), "
            f"xy_response={xy_displacement:.3f} m, "
            f"pre_lift_z={observed[2]:.3f}",
            arm_command=result.arm_command,
        )

    def _tick_task3_align_for_pick(
        self,
        context: ExecutionContext,
    ) -> StageResult:
        """Centre the final RGB-D lock with a bounded short local motion."""

        if context.unsafe_collision:
            return StageResult.blocked(
                "Task3 pick alignment stopped on unsafe collision",
                arm_command=self._held_arm_command,
            )
        if self._locked_target_world is None:
            return StageResult.blocked(
                "Task3 pick alignment has no locked RGB-D target"
            )

        pose = self._odometry_pose(context.odometry)
        if pose is None:
            return StageResult.running("Task3 waiting for odometry during pick alignment")

        elapsed = max(0.0, float(context.now_s) - self._task3_pick_align_started_s)
        if elapsed > 75.0 and self._task3_pick_phase != "open_pregrasp":
            return StageResult.blocked(
                f"Task3 short pick-stand refinement timed out after {elapsed:.1f}s"
            )

        robot_x, robot_y, robot_yaw = pose
        target_x, target_y, _target_z = self._locked_target_world
        stand_x = float(target_x)
        stand_y = float(target_y) - self.TASK3_PICK_STANDOFF_M
        dx = stand_x - robot_x
        dy = stand_y - robot_y
        distance = math.hypot(dx, dy)

        if self._task3_pick_phase == "rotate_to_stand":
            if distance <= self.TASK3_PICK_STAND_TOLERANCE_M:
                self._task3_pick_phase = "face_target"
                return StageResult.running("Task3 reached refined stand; preparing final yaw")
            heading = math.atan2(dy, dx)
            heading_error = self._wrap_angle(heading - robot_yaw)
            if abs(heading_error) <= 0.06:
                self._task3_pick_phase = "drive_to_stand"
                return StageResult.running("Task3 aligned for short straight pick correction")
            angular = math.copysign(
                max(0.08, min(0.55, 1.4 * abs(heading_error))),
                heading_error,
            )
            return StageResult.running(
                f"Task3 rotating toward refined pick stand; "
                f"yaw_err={heading_error:.3f}",
                base_command=(0.0, angular),
            )

        if self._task3_pick_phase == "drive_to_stand":
            if distance <= self.TASK3_PICK_STAND_TOLERANCE_M:
                self._task3_pick_phase = "face_target"
                return StageResult.running("Task3 reached refined stand; preparing final yaw")
            heading = math.atan2(dy, dx)
            heading_error = self._wrap_angle(heading - robot_yaw)
            if abs(heading_error) > 0.20:
                self._task3_pick_phase = "rotate_to_stand"
                return StageResult.running("Task3 correcting heading before short advance")
            linear = min(0.18, max(0.06, 0.9 * distance))
            angular = max(-0.22, min(0.22, 1.2 * heading_error))
            return StageResult.running(
                f"Task3 driving straight to refined pick stand; "
                f"remaining={distance:.3f}m",
                base_command=(linear, angular),
            )

        if self._task3_pick_phase == "face_target":
            # Coarse navigation uses the stable far-view lock, but final
            # centring MUST use the current RGB-D XY.  Otherwise the base can
            # be perfectly centred on an old point while the real box is
            # visibly several centimetres to one side.
            target_color = str(
                context.instruction.get("target_color", "")
            ).strip().lower()
            observation = context.target_observations.get(target_color)

            # Near the table the coloured box can leave the reliable RGB-D
            # view.  Final centring is therefore best-effort:
            #   fresh RGB-D -> refine with it
            #   missing/stale RGB-D -> fall back to the already validated
            #   multi-frame far-view lock instead of waiting forever.
            use_live = False
            age_s = float("inf")

            if observation is not None:
                age_s = max(
                    0.0,
                    float(context.now_s)
                    - float(observation.received_at_s),
                )
                if age_s <= self.TASK3_SOURCE_MAX_AGE_S:
                    observed = tuple(
                        float(value)
                        for value in observation.position_world
                    )
                    live_x = observed[0]
                    live_y = observed[1]
                    use_live = True

            if not use_live:
                live_x = float(target_x)
                live_y = float(target_y)

            # Reject a clearly unrelated/noisy near-field detection.
            aim_source = "LIVE" if use_live else "LOCKED_FALLBACK"
            lock_shift = math.hypot(
                live_x - float(target_x),
                live_y - float(target_y),
            )
            if lock_shift > 0.100:
                return StageResult.running(
                    "Task3 rejecting implausible live final-centre sample; "
                    f"lock_shift={lock_shift:.3f}m",
                    base_command=(0.0, 0.0),
                )

            # Light low-pass filtering prevents yaw0/yaw90 cuboid estimates
            # from making the base twitch between adjacent RGB-D centres.
            previous = getattr(self, "_task3_live_aim_xy", None)
            if previous is None:
                aim_x, aim_y = live_x, live_y
            else:
                alpha = 0.35
                aim_x = (1.0 - alpha) * float(previous[0]) + alpha * live_x
                aim_y = (1.0 - alpha) * float(previous[1]) + alpha * live_y
            self._task3_live_aim_xy = (aim_x, aim_y)

            rel_x = aim_x - robot_x
            rel_y = aim_y - robot_y
            distance = math.hypot(rel_x, rel_y)
            target_heading = math.atan2(rel_y, rel_x)
            yaw_error = self._wrap_angle(target_heading - robot_yaw)

            forward = (
                math.cos(robot_yaw) * rel_x
                + math.sin(robot_yaw) * rel_y
            )
            lateral = (
                -math.sin(robot_yaw) * rel_x
                + math.cos(robot_yaw) * rel_y
            )
            range_error = distance - self.TASK3_PICK_STANDOFF_M

            # First point the chassis centreline through the LIVE box centre.
            if abs(yaw_error) > 0.008 or abs(lateral) > 0.005:
                angular = math.copysign(
                    max(0.035, min(0.18, 1.8 * abs(yaw_error))),
                    yaw_error,
                )
                return StageResult.running(
                    "Task3 LIVE fine-centring target before pregrasp; "
                    f"forward={forward:.3f}, lateral={lateral:.3f}, "
                    f"yaw_err={yaw_error:.4f}, "
                    f"lock_shift={lock_shift:.3f}, source={aim_source}",
                    base_command=(0.0, angular),
                )

            # Then recover the intended ~0.620 m arm working distance using
            # the same LIVE centre, rather than the stale far-view centre.
            if abs(range_error) > self.TASK3_PICK_STAND_TOLERANCE_M:
                linear = math.copysign(
                    max(0.025, min(0.070, 0.8 * abs(range_error))),
                    range_error,
                )
                return StageResult.running(
                    "Task3 LIVE range refinement before pregrasp; "
                    f"distance={distance:.3f}, "
                    f"range_err={range_error:.3f}, "
                    f"lateral={lateral:.3f}",
                    base_command=(linear, 0.0),
                )

            # Final accepted RGB-D XY becomes the manipulation centre.
            # Preserve the calibrated/frozen Z so near-field depth bias does
            # not change grasp height.
            locked_z = float(self._locked_target_world[2])
            self._locked_target_world = (
                float(aim_x),
                float(aim_y),
                locked_z,
            )

            self._pregrasp.reset()
            self._stage_started_s = float(context.now_s)
            self._task3_pick_phase = "open_pregrasp"
            return StageResult.running(
                "Task3 LIVE target centred between both arms; "
                f"forward={forward:.3f}, lateral={lateral:.3f}, "
                f"distance={distance:.3f}, "
                f"xy_shift={lock_shift:.3f}, source={aim_source}; "
                "freezing corrected XY for pregrasp",
                base_command=(0.0, 0.0),
            )

        if self._task3_pick_phase == "open_pregrasp":
            return super().tick(TaskStage.ALIGN_FOR_PICK, context)

        return StageResult.blocked(
            f"Task3 invalid pick-alignment phase {self._task3_pick_phase!r}"
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


    # Competition geometry.
    COLORED_HALF_Z = 0.095
    PACKAGING_HALF_Z = 0.117

    # Official left-side displacement is -0.238 world Y.
    TASK3_LEFT_DY = 0.238

    # Keep the released object slightly away from the shelf side wall.
    # This matches the previously successful Task3 safe-release idea.
    TASK3_SAFE_Y_OFFSET = 0.040

    # Task1's integrated shelf insertion is already proven around x=-2.64.
    # Its existing SHELF_PLACE_DEPTH_BIAS_M then provides the deeper insertion.
    # Leave only part of the box supported by the shelf. The visual pusher
    # completes the move from this outside release to the official final x.
    TASK3_RELEASE_X = -2.560
    TASK3_SAFE_RELEASE_Y = 0.540
    # Route directly to the final packaging-left row.  Using the same Y for
    # transport and placement avoids a rotate-drive-rotate shelf-front detour.
    TASK3_SHELF_TURN_Y = TASK3_SAFE_RELEASE_Y
    TASK3_LIVE_ENTRY_OUTSIDE_X_TOLERANCE_M = 0.040
    TASK3_LIVE_ENTRY_INSIDE_X_TOLERANCE_M = 0.180
    TASK3_LIVE_ENTRY_Y_TOLERANCE_M = 0.060
    TASK3_LIVE_ENTRY_Z_TOLERANCE_M = 0.120
    # Hand the last shelf-facing correction to the explicit lateral/final
    # alignment controller.  The direct navigator reaches this safe staging
    # point within 6 cm and 0.12 rad; demanding 1 cm/0.01 rad makes it rotate
    # the long carry footprint beside transient RGB-D overlays.
    TASK3_SHELF_Y_TOLERANCE_M = 0.060
    SHELF_TURN_POSITION_TOLERANCE_M = TASK3_SHELF_Y_TOLERANCE_M
    SHELF_SCAN_YAW_TOLERANCE_RAD = 0.120
    # Match the Master post-release sequence: back out 0.40 m, then return
    # exactly 0.40 m with open arms.  This only brings the pusher back to the
    # release plane; it does not add the old extra 0.05 m shove.
    TASK3_RELEASE_BACKOFF_M = 0.40
    TASK3_POST_RELEASE_PUSH_M = 0.40
    TASK3_RELEASE_SETTLE_S = 1.0
    # Master task-3 release calibration: open only 40 mm beyond the achieved
    # grasp and clamp the resulting per-arm half-width to a safe local range.
    TASK3_RELEASE_SPREAD_M = 0.040
    TASK3_RELEASE_MIN_HALF_WIDTH_M = 0.110
    TASK3_RELEASE_MAX_HALF_WIDTH_M = 0.140
    # The old master branch used a fixed 0.45 m base push.  Keep the safer
    # RGB-D closed loop already present in this executor and request only a
    # short correction after the shallow release.
    TASK3_ENABLE_PUSH = True
    TASK3_PUSHER_HALF_WIDTH_M = 0.065
    # Only give the released box a gentle final nudge.  The shelf placement
    # itself already carries almost all of the box over the board.
    TASK3_PUSH_DESIRED_M = 0.010
    TASK3_FINAL_CENTER_X = -2.680
    TASK3_PUSH_X_TOLERANCE_M = 0.004
    TASK3_PUSH_OVERSHOOT_M = 0.004
    TASK3_PUSH_MAX_DISTANCE_M = 0.020
    TASK3_PUSH_TIMEOUT_S = 10.0
    TASK3_PUSH_LOST_TIMEOUT_S = 1.0
    TASK3_PUSH_BASE_SPEED_MPS = 0.008
    TASK3_PUSH_MAX_SPEED_MPS = 0.010

    def _shelf_observation_target_y(self, context: ExecutionContext) -> float:
        """Align the shelf observation stand with the packaging-left target."""

        del context
        return float(self.TASK3_SAFE_RELEASE_Y)

    # packaging_box is a static shelf obstacle.  It is often visible during
    # the safe turn but occluded by the carried box at the final scan stand;
    # retain that validated RGB-D layer measurement for the current run.
    PACKAGING_MAX_AGE_S = 120.0

    def __init__(self, memory: CompetitionTaskMemory) -> None:
        super().__init__(memory)

        # Task3's raised white support needs the validated narrower, lower
        # pregrasp route; Task1 and Task2 keep their own controllers unchanged.
        self._pregrasp = Task3StableOpenPregraspController()
        self._task3_direct_hug = Task3DirectHugController()
        # Task 3 follows the master branch's relative release strategy.  It
        # spreads around the achieved held centre and never recomputes a
        # world-frame shelf IK target.
        self._release = ReleaseSpreadController()
        self._task3_pusher = ReleaseSpreadController()

        # Preserve Task1's contact-search state machine while using the old
        # Task3 reference's proven 85 mm half-width.
        self._contact = Task3ReferenceContactController()
        self._lift = SlideLiftController(lift_height=0.10)
        self._navigation._timeout = 120.0
        # Task3's carried box is intentionally held at the near edge, so the
        # shelf-front three-part lateral manoeuvre is kept cautious.  Give
        # only this executor enough time to finish and accept the measured
        # 0.08 rad shelf-facing yaw instead of failing at 90 s by a hair.
        self._transfer.LATERAL_TIMEOUT_S = 120.0
        self._transfer.LATERAL_POSITION_TOLERANCE_M = (
            self.TASK3_SHELF_Y_TOLERANCE_M
        )
        self._transfer.LATERAL_YAW_TOLERANCE_RAD = 0.08
        self._task3_source_tracker = StableTargetCenterTracker(
            window_size=12,
            required_samples=5,
            required_inliers=4,
            min_sample_interval_s=0.10,
            min_collection_duration_s=0.45,
            max_observation_age_s=self.TASK3_SOURCE_MAX_AGE_S,
            max_axis_deviation=(0.060, 0.050, 0.050),
            shelf_roi=self.TASK3_SOURCE_ROI,
            layer_z_gate_m=0.20,
        )
        self._task3_source_estimate = None
        self._task3_support_world: tuple[float, float, float] | None = None
        self._task3_grasp_target_world: tuple[float, float, float] | None = None
        self._task3_pick_phase = "idle"
        self._task3_pick_align_started_s = 0.0
        self._task3_place_locked = False
        self._task3_packaging_world: tuple[float, float, float] | None = None
        self._task3_place_phase = "idle"
        self._task3_place_phase_started_s = 0.0
        self._task3_push_started_s = 0.0
        self._task3_push_start_x: float | None = None
        self._task3_push_target_x: float | None = None
        self._task3_push_last_seen_s = 0.0
        self._task3_push_confirm_count = 0

    def reset(self) -> None:
        super().reset()
        self._task3_direct_hug.reset()
        self._task3_pusher.reset()
        self._task3_source_tracker.reset()
        self._task3_source_estimate = None
        self._task3_support_world = None
        self._task3_grasp_target_world = None
        self._task3_pick_phase = "idle"
        self._task3_pick_align_started_s = 0.0
        self._task3_place_locked = False
        self._task3_packaging_world = None
        self._task3_place_phase = "idle"
        self._task3_place_phase_started_s = 0.0
        self._task3_push_started_s = 0.0
        self._task3_push_start_x = None
        self._task3_push_target_x = None
        self._task3_push_last_seen_s = 0.0
        self._task3_push_confirm_count = 0

    def enter_stage(
        self,
        stage: TaskStage,
        context: ExecutionContext,
    ) -> None:
        super().enter_stage(stage, self._task1_compat_context(context))
        if stage is TaskStage.NAVIGATE_TO_PICK:
            self._task3_source_tracker.reset(
                accept_after_s=float(context.now_s),
            )
            self._task3_source_estimate = None
        elif stage is TaskStage.ALIGN_FOR_PICK:
            self._transfer.reset()
            self._motion_started = False
            self._task3_pick_phase = "face_target"
            self._task3_pick_align_started_s = float(context.now_s)
            self._task3_live_aim_xy = None
        elif stage is TaskStage.GRASP:
            self._task3_direct_hug.reset()
            self._task3_lift_verified = False
            self._task3_pre_lift_observed_z = None
        elif stage is TaskStage.PLACE:
            self._task3_place_phase = "parent_release"
            self._task3_place_phase_started_s = float(context.now_s)
            self._task3_push_started_s = 0.0
            self._task3_push_start_x = None
            self._task3_push_target_x = None
            self._task3_push_last_seen_s = float(context.now_s)
            self._task3_push_confirm_count = 0

    def _tick_task3_acquire_target(
        self,
        context: ExecutionContext,
    ) -> StageResult:
        """Reacquire the instructed block after stopping by its support."""

        target_color = str(
            context.instruction.get("target_color", "")
        ).strip().lower()
        if not target_color:
            return StageResult.blocked("Task3 instruction has no target_color")

        observation = context.target_observations.get(target_color)

        # Prefer the stable source centre acquired from the farther initial
        # view.  Near-field RGB-D depth/centroid geometry drifts strongly as
        # the robot approaches the white support and must not overwrite it.
        estimate = self._task3_source_estimate

        if estimate is None:
            fresh_estimate = self._task3_source_tracker.update(
                observation,
                now_s=context.now_s,
            )
            if fresh_estimate is not None:
                self._task3_source_estimate = fresh_estimate
                estimate = fresh_estimate
        if estimate is None:
            waited_s = max(
                0.0,
                float(context.now_s) - self._stage_started_s,
            )
            detail = self._task3_source_tracker.status()
            if waited_s >= self.TASK3_SOURCE_ACQUIRE_TIMEOUT_S:
                return StageResult.blocked(
                    f"Task3 timed out reacquiring {target_color!r} above "
                    f"material_box: {detail}"
                )
            return StageResult.running(
                f"Task3 collecting fresh {target_color!r} RGB-D centre above "
                f"material_box; {detail}"
            )

        center = tuple(float(value) for value in estimate.center_world)
        # Task3 source sits on the fixed white material_box top.
        # Near-field RGB-D Z drifted upward by ~4 cm in testing, so retain
        # dynamic RGB-D X/Y but use the validated source centre height.
        center = (
            center[0],
            center[1],
            self.TASK3_SOURCE_CENTER_Z,
        )
        support = context.target_observations.get(self.TASK3_SUPPORT_COLOR)
        if support is not None:
            support_age = max(
                0.0,
                float(context.now_s) - float(support.received_at_s),
            )
            if support_age <= self.TASK3_SOURCE_MAX_AGE_S:
                support_center = tuple(
                    float(value) for value in support.position_world
                )
                xy_error = math.hypot(
                    center[0] - support_center[0],
                    center[1] - support_center[1],
                )
                z_clearance = center[2] - support_center[2]
                if xy_error > 0.30 or z_clearance < 0.04:
                    return StageResult.blocked(
                        "Task3 RGB-D target is inconsistent with material_box: "
                        f"xy_error={xy_error:.3f}, "
                        f"z_clearance={z_clearance:.3f}"
                    )

        # The target-color centroid can jump to a tabletop instance or drift
        # badly when the raised block fills the near-field image. The white
        # support is static and its far-view RGB-D centre is substantially more
        # stable. Task3's block is centred on that support in every randomized
        # scene, so use the cached support X/Y for manipulation geometry while
        # still using target_color to select and verify the carried object.
        if self._task3_support_world is not None:
            center = (
                float(self._task3_support_world[0]),
                float(self._task3_support_world[1]),
                self.TASK3_SOURCE_CENTER_Z,
            )

        pose = self._odometry_pose(context.odometry)
        if pose is None:
            return StageResult.running(
                "Task3 waiting for odometry before computing shallow grip point"
            )
        approach_dx = center[0] - pose[0]
        approach_dy = center[1] - pose[1]
        approach_norm = math.hypot(approach_dx, approach_dy)
        if approach_norm <= 1e-6:
            return StageResult.blocked(
                "Task3 cannot compute shallow grip direction at zero range"
            )
        shallow_center = (
            center[0]
            - self.TASK3_SHALLOW_GRIP_OFFSET_M * approach_dx / approach_norm,
            center[1]
            - self.TASK3_SHALLOW_GRIP_OFFSET_M * approach_dy / approach_norm,
            center[2] + self.TASK3_GRASP_Z_OFFSET_M,
        )
        self._locked_target_world = center
        self._task3_grasp_target_world = shallow_center
        self._locked_target_orientation = estimate.orientation or "yaw0"

        return StageResult.succeeded(
            f"Task3 locked fresh {target_color!r} RGB-D source centre: "
            f"center={tuple(round(value, 3) for value in center)}, "
            f"shallow_grip={tuple(round(value, 3) for value in shallow_center)}, "
            f"orientation={self._locked_target_orientation}, "
            f"samples={estimate.sample_count}"
        )

    def _task3_place_from_rgbd(
        self,
        context: ExecutionContext,
    ) -> tuple[float, float, float]:
        """Build Task3 place target from live packaging_box RGB-D center."""

        observation = context.target_observations.get("packaging_box")
        if observation is None:
            raise RuntimeError(
                "Task3 has no live packaging_box RGB-D observation"
            )

        age = max(
            0.0,
            float(context.now_s) - float(observation.received_at_s),
        )
        if age > self.PACKAGING_MAX_AGE_S:
            raise RuntimeError(
                f"Task3 packaging_box RGB-D observation is stale: {age:.2f}s"
            )

        point = tuple(
            float(value) for value in observation.position_world
        )
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            raise RuntimeError(
                "Task3 packaging_box RGB-D center is invalid"
            )

        px, py, pz = point

        # Fail closed if the detected white obstacle is not actually in
        # the shelf workspace.
        if not (-3.10 <= px <= -2.20):
            raise RuntimeError(
                f"Task3 packaging_box x outside shelf ROI: {px:.3f}"
            )
        if not (0.30 <= py <= 1.30):
            raise RuntimeError(
                f"Task3 packaging_box y outside shelf ROI: {py:.3f}"
            )
        if not (0.30 <= pz <= 1.40):
            raise RuntimeError(
                f"Task3 packaging_box z outside shelf ROI: {pz:.3f}"
            )

        # packaging_box and colored box sit on the same shelf board.
        # Convert packaging center Z to colored-box center Z.
        target_z = (
            pz
            - self.PACKAGING_HALF_Z
            + self.COLORED_HALF_Z
        )

        # "Left" for the official west-facing shelf is negative world Y.
        # The calibrated shelf centre is y=0.778 and the official left offset is
        # 0.238 m, so y=0.540 leaves about 4 cm between the two cuboids.
        # Do not bias this back toward the packaging box: y=0.600 overlaps
        # their physical Y extents even though it remains inside referee radius.
        target_y = self.TASK3_SAFE_RELEASE_Y

        self._task3_packaging_world = (px, py, pz)

        return (
            self.TASK3_RELEASE_X,
            target_y,
            target_z,
        )

    def _tick_align_for_place(
        self,
        context: ExecutionContext,
    ) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked(
                "Task3 lost held object center before shelf placement"
            )

        # Lock Task3's real packaging-left target before the parent is allowed
        # to plan clearance or enter the shelf.  The previous ordering let the
        # parent finish a Task1 empty-layer approach and replaced the target
        # only afterward.
        if not self._task3_place_locked:
            # Formal T1->T2->T3 already has a validated shelf state from Task1.
            # Reuse it instead of forcing Task3 to rediscover two occupied
            # layers while carrying the object at the shelf.
            state = self._shelf_state
            if state is None:
                try:
                    state = self._memory.require_shelf_state()
                except Exception:
                    state = None

            # Keep live recognition only as a fallback for standalone runs.
            if state is None:
                state = self._update_shelf_state(context)

            if state is None:
                return StageResult.running(
                    "Task3 waiting for existing Task1 shelf memory or one fresh "
                    "shelf-state estimate; refusing redundant semantic scan",
                    base_command=(0.0, 0.0),
                    arm_command=self._held_arm_command,
                )

            self._shelf_state = state
            self._memory.record_shelf_state(state)
            try:
                self._place_world = self._task3_place_from_rgbd(context)
                target_held_z = self._place_world[2] + self.SHELF_CLEARANCE_M
                target_slide = (
                    self._held_arm_command.spine_position
                    + self._held_center_base[2]
                    - target_held_z
                )
                self._slide_hold.reset()
                self._slide_start = self._held_arm_command.spine_position
                self._held_arm_command = self._slide_hold.plan(
                    self._held_arm_command,
                    target_slide,
                    context.joint_states,
                )
            except RuntimeError as exc:
                return StageResult.blocked(
                    str(exc), arm_command=self._held_arm_command
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"Task3 could not plan packaging-left clearance: {exc}",
                    arm_command=self._held_arm_command,
                )

            self._final_place_stand = stand_from_held_center(
                self._place_world,
                self._held_center_base,
                self.SHELF_YAW,
            )
            self._final_place_stand = (
                self._final_place_stand[0],
                max(0.50, min(0.98, self._final_place_stand[1])),
            )
            self._phase = "clearance"
            self._motion_started = False
            self._transfer.reset()
            self._task3_place_locked = True

        result = super()._tick_align_for_place(context)
        if result.status is StageStatus.SUCCEEDED:
            px, py, pz = self._task3_packaging_world
            tx, ty, tz = self._place_world
            return StageResult.succeeded(
                "Task3 entered shallow packaging-left release pose: "
                f"packaging=({px:.3f},{py:.3f},{pz:.3f}), "
                f"release=({tx:.3f},{ty:.3f},{tz:.3f})",
                arm_command=self._held_arm_command,
            )
        return result

    def _tick_task3_place(self, context: ExecutionContext) -> StageResult:
        """Shallow release, back out, then RGB-D closed-loop centre push."""

        if self._held_arm_command is None or self._place_world is None:
            return StageResult.blocked("Task3 placement has no held object/target")

        phase = self._task3_place_phase
        if phase == "parent_release":
            # Reproduce master's task-3 placement locally instead of calling
            # Task1's two-step shelf release.  First lower vertically, then
            # spread around the *achieved held centre*.  This keeps both arms
            # outside the shelf-depth IK path and avoids the Task1 0.18 m
            # final spread.
            if self._phase == "lower":
                if not self._slide_hold.planned:
                    target_slide = (
                        self._held_arm_command.spine_position
                        + self._held_center_base[2]
                        - self._place_world[2]
                    )
                    try:
                        self._slide_start = self._held_arm_command.spine_position
                        self._held_arm_command = self._slide_hold.plan(
                            self._held_arm_command,
                            target_slide,
                            context.joint_states,
                        )
                    except (PregraspInputError, PregraspPlanningError) as exc:
                        return StageResult.blocked(
                            f"Task3 could not plan master vertical lowering: {exc}",
                            arm_command=self._held_arm_command,
                        )
                lower_result = self._tick_slide(
                    context,
                    "lowering task-3 box vertically onto the shelf board",
                )
                if lower_result is not None:
                    return lower_result
                self._phase = "release"
                self._phase_started_s = float(context.now_s)

            if self._phase == "release":
                if not self._release.planned:
                    # Task3 is held by the dedicated direct-hug controller,
                    # not Task1/Task2's contact controller.  Use the actual
                    # calibrated shallow-grasp width that produced the held
                    # command (0.080 m per side).
                    grasp_half = self._task3_direct_hug.HOLD_HALF_WIDTH_M
                    if grasp_half is None or not math.isfinite(float(grasp_half)):
                        return StageResult.blocked(
                            "Task3 master relative release has no measured grasp width",
                            arm_command=self._held_arm_command,
                        )
                    release_half = min(
                        self.TASK3_RELEASE_MAX_HALF_WIDTH_M,
                        max(
                            self.TASK3_RELEASE_MIN_HALF_WIDTH_M,
                            float(grasp_half) + self.TASK3_RELEASE_SPREAD_M,
                        ),
                    )
                    try:
                        self._held_arm_command = self._release.plan_from_held(
                            self._held_arm_command,
                            self._held_center_base,
                            context.joint_states,
                            half_width=release_half,
                        )
                    except (PregraspInputError, PregraspPlanningError) as exc:
                        return StageResult.blocked(
                            f"Task3 master relative release planning failed: {exc}",
                            arm_command=self._held_arm_command,
                        )
                try:
                    command, reached, detail = self._release.update(
                        context.now_s,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"Task3 master relative release control failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
                self._held_arm_command = command
                if not reached:
                    if float(context.now_s) - self._phase_started_s >= self.PLACE_TIMEOUT_S:
                        return StageResult.blocked(
                            f"Task3 master relative release timed out: {detail}",
                            arm_command=command,
                        )
                    return StageResult.running(
                        f"Task3 master local symmetric release; {detail}",
                        arm_command=command,
                    )
            self._task3_place_phase = "back_after_release"
            self._task3_place_phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()
            return StageResult.running(
                "Task3 shallow release complete; backing out before centre push",
                arm_command=self._held_arm_command,
            )

        if phase == "back_after_release":
            if not self._motion_started:
                if not self._transfer.begin_retreat(
                    context.odometry,
                    self.TASK3_RELEASE_BACKOFF_M,
                    heading_yaw=self.SHELF_YAW,
                ):
                    return StageResult.running(
                        "Task3 waiting to start post-release back-out",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"Task3 backing out after shallow release; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._task3_place_phase = "prepare_open_pusher"
            self._task3_place_phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()
            self._task3_pusher.reset()
            if not self.TASK3_ENABLE_PUSH:
                return StageResult.succeeded(
                    "Task3 shallow placement complete: grippers opened in "
                    "place and arms backed clear; visual push intentionally "
                    "disabled",
                    arm_command=self._held_arm_command,
                )

        if self._task3_place_phase == "prepare_open_pusher":
            if self._held_center_base is None:
                return StageResult.blocked(
                    "Task3 cannot form the master outside pusher without the held centre",
                    arm_command=self._held_arm_command,
                )
            try:
                if not self._task3_pusher.planned:
                    # Master strategy: after the 0.40 m base retreat, reshape
                    # the already-extracted arms around their achieved held
                    # centre.  Do not solve another world-frame shelf target,
                    # which would extend the arms back into the rack before
                    # the explicit base push starts.
                    self._held_arm_command = self._task3_pusher.plan_from_held(
                        self._held_arm_command,
                        self._held_center_base,
                        context.joint_states,
                        half_width=self.TASK3_PUSHER_HALF_WIDTH_M,
                    )
                command, reached, detail = self._task3_pusher.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"Task3 could not prepare open centre pusher: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if not reached:
                if (
                    float(context.now_s) - self._task3_place_phase_started_s
                    >= 15.0
                ):
                    return StageResult.blocked(
                        f"Task3 open centre pusher timed out: {detail}",
                        arm_command=command,
                    )
                return StageResult.running(
                    f"Task3 forming open centre pusher; {detail}",
                    arm_command=command,
                )
            pose = self._odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "Task3 waiting for odometry before visual push",
                    arm_command=command,
                )
            self._task3_place_phase = "fixed_gentle_push"
            self._task3_push_started_s = float(context.now_s)
            self._task3_push_start_x = pose[0]
            self._task3_push_target_x = None
            self._task3_push_last_seen_s = float(context.now_s)
            self._task3_push_confirm_count = 0
            return StageResult.running(
                "Task3 open centre pusher ready; gently returning 0.40 m to the release plane",
                arm_command=command,
            )

        if self._task3_place_phase == "fixed_gentle_push":
            pose = self._odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "Task3 gentle push waiting for odometry",
                    base_command=(0.0, 0.0),
                    arm_command=self._held_arm_command,
                )
            if not self._motion_started:
                if not self._transfer.begin_advance(
                    context.odometry,
                    self.TASK3_POST_RELEASE_PUSH_M,
                    heading_yaw=self.SHELF_YAW,
                    completion_tolerance_m=0.004,
                ):
                    return StageResult.running(
                        "Task3 waiting to start the 0.40 m gentle return",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
                self._task3_push_start_x = pose[0]
            done, base_command, detail = self._transfer.tick_advance(context.odometry)
            if done:
                return StageResult.succeeded(
                    "Task3 gentle 0.40 m return reached the release plane; push complete",
                    base_command=(0.0, 0.0),
                    arm_command=self._held_arm_command,
                )
            # Limit the whole return and slow down again for the final 6 cm,
            # where the open hands can first touch the released box.
            traveled = max(0.0, self._task3_push_start_x - pose[0])
            remaining = max(0.0, self.TASK3_POST_RELEASE_PUSH_M - traveled)
            speed = 0.060 if remaining > 0.060 else max(0.008, min(0.020, 0.40 * remaining))
            return StageResult.running(
                f"Task3 gentle 0.40 m return; remaining={remaining:.3f} m; {detail}",
                base_command=(speed, base_command[1]),
                arm_command=self._held_arm_command,
            )

        if self._task3_place_phase == "visual_push":
            now_s = float(context.now_s)
            elapsed = max(0.0, now_s - self._task3_push_started_s)
            if elapsed >= self.TASK3_PUSH_TIMEOUT_S:
                return StageResult.blocked(
                    "Task3 visual push timed out before final x confirmation",
                    arm_command=self._held_arm_command,
                )
            pose = self._odometry_pose(context.odometry)
            if pose is None or self._task3_push_start_x is None:
                return StageResult.running(
                    "Task3 visual push waiting for odometry",
                    arm_command=self._held_arm_command,
                )
            traveled = max(0.0, self._task3_push_start_x - pose[0])
            if traveled >= self.TASK3_PUSH_MAX_DISTANCE_M:
                return StageResult.blocked(
                    f"Task3 visual push exceeded {traveled:.3f} m safety limit",
                    arm_command=self._held_arm_command,
                )

            target_color = str(
                context.instruction.get("target_color", "")
            ).strip().lower()
            observation = context.target_observations.get(target_color)
            center_x = None
            if observation is not None:
                age_s = max(0.0, now_s - float(observation.received_at_s))
                point = tuple(float(value) for value in observation.position_world)
                if (
                    age_s <= self.TASK3_PUSH_LOST_TIMEOUT_S
                    and len(point) == 3
                    and -3.10 <= point[0] <= -2.20
                    and 0.30 <= point[1] <= 1.30
                    and abs(point[2] - self._place_world[2]) <= 0.20
                ):
                    center_x = point[0]
                    self._task3_push_last_seen_s = now_s

            if center_x is None:
                self._task3_push_confirm_count = 0
                return StageResult.running(
                    f"Task3 visual push paused for fresh {target_color!r} frame",
                    base_command=(0.0, 0.0),
                    arm_command=self._held_arm_command,
                )

            if self._task3_push_target_x is None:
                # World X decreases while advancing into the shelf.  Never
                # command deeper than the validated final-centre bound, and
                # never ask for more than the small desired correction from
                # the first post-release RGB-D observation.
                self._task3_push_target_x = max(
                    self.TASK3_FINAL_CENTER_X,
                    center_x - self.TASK3_PUSH_DESIRED_M,
                )
            target_x = self._task3_push_target_x
            error_x = center_x - target_x
            if abs(error_x) <= self.TASK3_PUSH_X_TOLERANCE_M:
                self._task3_push_confirm_count += 1
            else:
                self._task3_push_confirm_count = 0
            if self._task3_push_confirm_count >= 3:
                return StageResult.succeeded(
                    f"Task3 RGB-D push confirmed final centre x={center_x:.3f}; "
                    f"short-push target={target_x:.3f}",
                    arm_command=self._held_arm_command,
                )
            if error_x < -self.TASK3_PUSH_OVERSHOOT_M:
                return StageResult.succeeded(
                    f"Task3 overshoot protection stopped push at x={center_x:.3f}",
                    arm_command=self._held_arm_command,
                )

            speed = min(
                self.TASK3_PUSH_MAX_SPEED_MPS,
                self.TASK3_PUSH_BASE_SPEED_MPS + 0.010 * int(elapsed / 1.5),
            )
            return StageResult.running(
                f"Task3 RGB-D centre push; x={center_x:.3f}, "
                f"short-push target={target_x:.3f}, "
                f"traveled={traveled:.3f}m",
                base_command=(speed, 0.0),
                arm_command=self._held_arm_command,
            )

        return StageResult.blocked(
            f"Task3 invalid placement phase {self._task3_place_phase!r}",
            arm_command=self._held_arm_command,
        )
