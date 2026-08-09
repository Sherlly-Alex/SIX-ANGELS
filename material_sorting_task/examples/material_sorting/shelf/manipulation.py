"""ROS-free arm helpers used by shelf and table placement executors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from control_types import ArmCommand
from desktop_grasp.pregrasp_core import (
    COMMAND_RATE_PER_S,
    FEEDBACK_POS_TOL,
    FEEDBACK_STABLE_TIME,
    FEEDBACK_VEL_TOL,
    GRIPPER_OPEN,
    GRASP_BACKOFF_X,
    HAND_Z_OFFSET,
    LEFT_A_ROT,
    LIFT_ARM_POSITION_TOL,
    LIFT_SLIDE_COMMAND_RATIO,
    OpenPregraspController,
    PREGRASP_BACKOFF_X,
    PregraspInputError,
    PregraspPlanningError,
    RIGHT_A_ROT,
    SPINE_MAX,
    SPINE_MIN,
    _joint_maps,
    _make_transform,
)
from mmk2_kdl import MMK2Kdl


class ShelfOpenPregraspController(OpenPregraspController):
    """Open around a shelf box without the desktop controller's wide sweep."""

    def __init__(self, *, half_width: float = 0.18, kdl=None) -> None:
        super().__init__(kdl=kdl)
        self.half_width = float(half_width)
        if not math.isfinite(self.half_width) or self.half_width <= 0.0:
            raise ValueError("shelf pregrasp half_width must be finite and positive")

    def plan(
        self,
        target_world: tuple[float, float, float],
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=PREGRASP_BACKOFF_X,
            half_width=self.half_width,
        )


class HeldTransportController:
    """Bring a preloaded dual-arm grasp closer without releasing the box.

    The grasp used by the competition client keeps both grippers open and
    holds the box through lateral arm preload.  Starting a new controller from
    measured joints would discard that preload.  This controller therefore
    starts from the last commanded grasp pose, preserves both gripper commands
    and the maximum-height spine command, then follows small synchronized IK
    waypoints that translate the box centre toward the robot.
    """

    # 0.50 m is the least-invasive compact pose that still clears the east
    # wall on the right-hand table route with the carried-envelope checker.
    TARGET_CENTER_X_M = 0.50
    TARGET_CENTER_Y_M = 0.0
    COMPACT_WAYPOINT_COUNT = 4
    COMMAND_RATE_PER_S = 0.45
    ARM_POSITION_TOL = 0.16
    MAX_JOINT_WAYPOINT_DELTA = 0.75
    STABLE_TIME_S = 0.30
    VELOCITY_TOL = 0.03

    def __init__(
        self,
        *,
        target_center_x_m: float = TARGET_CENTER_X_M,
        target_center_y_m: float = TARGET_CENTER_Y_M,
        allow_extension: bool = False,
        max_translation_m: float = 0.30,
        kdl: MMK2Kdl | None = None,
    ) -> None:
        self.target_center_x_m = float(target_center_x_m)
        self.target_center_y_m = float(target_center_y_m)
        self.allow_extension = bool(allow_extension)
        self.max_translation_m = float(max_translation_m)
        if (
            not math.isfinite(self.target_center_x_m)
            or self.target_center_x_m <= 0.0
            or not math.isfinite(self.target_center_y_m)
            or not math.isfinite(self.max_translation_m)
            or self.max_translation_m <= 0.0
        ):
            raise ValueError("transport target center/translation limit is invalid")
        if self.COMPACT_WAYPOINT_COUNT < 1:
            raise ValueError("transport compact waypoint count must be positive")
        self._kdl = kdl or MMK2Kdl()
        self._targets: list[np.ndarray] = []
        self._centers: list[tuple[float, float, float]] = []
        self._action_vector: np.ndarray | None = None
        self._stage_index = 0
        self._last_update_s: float | None = None
        self._stable_since_s: float | None = None

    @property
    def planned(self) -> bool:
        return bool(self._targets) and self._action_vector is not None

    @property
    def target_center_base(self) -> tuple[float, float, float] | None:
        return None if not self._centers else self._centers[-1]

    @property
    def waypoint_count(self) -> int:
        return len(self._targets)

    def reset(self) -> None:
        self._targets.clear()
        self._centers.clear()
        self._action_vector = None
        self._stage_index = 0
        self._last_update_s = None
        self._stable_since_s = None

    def plan(
        self,
        hold_command: ArmCommand,
        held_center_base: tuple[float, float, float],
        half_width_m: float,
        *,
        target_center_base: tuple[float, float, float] | None = None,
    ) -> ArmCommand:
        self.reset()
        start_center = np.asarray(held_center_base, dtype=float)
        half_width = float(half_width_m)
        if start_center.shape != (3,) or not np.all(np.isfinite(start_center)):
            raise PregraspInputError("held center is invalid for transport compaction")
        if not math.isfinite(half_width) or half_width <= 0.0:
            raise PregraspInputError("held half-width is invalid for transport compaction")
        if target_center_base is None:
            final_center = np.array(
                [
                    (
                        self.target_center_x_m
                        if self.allow_extension
                        else min(float(start_center[0]), self.target_center_x_m)
                    ),
                    self.target_center_y_m,
                    float(start_center[2]),
                ],
                dtype=float,
            )
        else:
            final_center = np.asarray(target_center_base, dtype=float)
            if final_center.shape != (3,) or not np.all(np.isfinite(final_center)):
                raise PregraspInputError("explicit held target center is invalid")
        if not self.allow_extension and final_center[0] > start_center[0] + 0.02:
            raise PregraspPlanningError(
                "held transport controller refused an outward extension"
            )
        translation_m = float(np.linalg.norm(final_center - start_center))
        if translation_m > self.max_translation_m + 1e-9:
            raise PregraspPlanningError(
                "held-object translation exceeds safety bound: "
                f"requested={translation_m:.3f}m, limit={self.max_translation_m:.3f}m"
            )
        steps = self.COMPACT_WAYPOINT_COUNT
        reference = np.array(
            [
                hold_command.spine_position,
                *hold_command.left_arm_positions,
                *hold_command.right_arm_positions,
            ],
            dtype=float,
        )
        if reference.shape != (13,) or not np.all(np.isfinite(reference)):
            raise PregraspInputError("held ArmCommand is invalid")

        previous_joints = reference.copy()
        for index in range(1, steps + 1):
            fraction = index / steps
            center = start_center + fraction * (final_center - start_center)
            arm_center = center + np.array(
                [-GRASP_BACKOFF_X, 0.0, HAND_Z_OFFSET], dtype=float
            )
            left_target = arm_center + np.array([0.0, half_width, 0.0])
            right_target = arm_center + np.array([0.0, -half_width, 0.0])
            solutions = self._kdl.inverse_kinematics(
                T_left=_make_transform(left_target, LEFT_A_ROT),
                T_right=_make_transform(right_target, RIGHT_A_ROT),
                ref_pos=previous_joints,
                target_height=float(hold_command.spine_position),
            )
            if solutions is None or len(solutions) == 0:
                raise PregraspPlanningError(
                    "dual-arm IK failed for compact transport center="
                    f"{np.round(center, 3).tolist()}"
                )
            joints = np.asarray(solutions[0], dtype=float)
            if joints.shape != (13,) or not np.all(np.isfinite(joints)):
                raise PregraspPlanningError(
                    "compact transport IK returned invalid joints"
                )
            joint_delta = float(np.max(np.abs(joints[1:] - previous_joints[1:])))
            if joint_delta > self.MAX_JOINT_WAYPOINT_DELTA:
                raise PregraspPlanningError(
                    "compact transport IK changed branch; "
                    f"max waypoint joint delta={joint_delta:.3f} rad"
                )
            target = np.array(
                [
                    hold_command.spine_position,
                    *hold_command.head_positions,
                    *joints[1:7],
                    hold_command.left_gripper_position,
                    *joints[7:13],
                    hold_command.right_gripper_position,
                ],
                dtype=float,
            )
            self._targets.append(target)
            self._centers.append(tuple(float(value) for value in center))
            previous_joints = joints

        self._action_vector = np.array(
            [
                hold_command.spine_position,
                *hold_command.head_positions,
                *hold_command.left_arm_positions,
                hold_command.left_gripper_position,
                *hold_command.right_arm_positions,
                hold_command.right_gripper_position,
            ],
            dtype=float,
        )
        return self.command()

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if not self.planned or self._action_vector is None:
            raise PregraspPlanningError("held transport update called before plan")
        positions, velocities = _joint_maps(joint_states)
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("control time is non-finite")
        dt = (
            0.05
            if self._last_update_s is None
            else min(0.20, max(0.01, now - self._last_update_s))
        )
        self._last_update_s = now
        target = self._targets[self._stage_index]
        diff = target - self._action_vector
        max_difference = float(np.max(np.abs(diff)))
        if max_difference > 0.0:
            ratios = np.abs(diff) / (max_difference + 1e-9)
            steps = ratios * self.COMMAND_RATE_PER_S * dt
            self._action_vector += np.sign(diff) * np.minimum(np.abs(diff), steps)

        measured = np.array(
            [
                positions["slide_joint"],
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
            ],
            dtype=float,
        )
        target_feedback = np.concatenate((target[0:1], target[3:9], target[10:16]))
        commanded_feedback = np.concatenate(
            (
                self._action_vector[0:1],
                self._action_vector[3:9],
                self._action_vector[10:16],
            )
        )
        measured_velocity = np.array(
            [
                velocities.get("slide_joint", 0.0),
                *(velocities.get(f"left_arm_joint{index}", 0.0) for index in range(1, 7)),
                *(velocities.get(f"right_arm_joint{index}", 0.0) for index in range(1, 7)),
            ],
            dtype=float,
        )
        errors = np.abs(measured - target_feedback)
        command_error = float(np.max(np.abs(commanded_feedback - target_feedback)))
        max_velocity = float(np.max(np.abs(measured_velocity)))
        stable_now = (
            errors[0] <= FEEDBACK_POS_TOL
            and float(np.max(errors[1:7])) <= self.ARM_POSITION_TOL
            and float(np.max(errors[7:13])) <= self.ARM_POSITION_TOL
            and command_error <= FEEDBACK_POS_TOL
            and max_velocity <= self.VELOCITY_TOL
        )
        if stable_now:
            if self._stable_since_s is None:
                self._stable_since_s = now
        else:
            self._stable_since_s = None

        stage_reached = (
            self._stable_since_s is not None
            and now - self._stable_since_s >= self.STABLE_TIME_S
        )
        if stage_reached and self._stage_index + 1 < len(self._targets):
            self._stage_index += 1
            self._stable_since_s = None
            stage_reached = False
        reached = stage_reached and self._stage_index + 1 == len(self._targets)
        center = self._centers[self._stage_index]
        detail = (
            f"compact_stage={self._stage_index + 1}/{len(self._targets)}, "
            f"center_base={tuple(round(value, 3) for value in center)}, "
            f"left_err={float(np.max(errors[1:7])):.3f}, "
            f"right_err={float(np.max(errors[7:13])):.3f}, "
            f"cmd_err={command_error:.3f}, max_vel={max_velocity:.3f}"
        )
        return self.command(), reached, detail

    def command(self) -> ArmCommand:
        if self._action_vector is None:
            raise PregraspPlanningError("held transport command requested before plan")
        values = self._action_vector
        return ArmCommand(
            spine_position=float(values[0]),
            head_positions=(float(values[1]), float(values[2])),
            left_arm_positions=tuple(float(value) for value in values[3:9]),
            left_gripper_position=float(values[9]),
            right_arm_positions=tuple(float(value) for value in values[10:16]),
            right_gripper_position=float(values[16]),
        )


class ReleaseSpreadController(OpenPregraspController):
    """Move the two open grippers away from a placed object."""

    MAX_RELATIVE_RELEASE_JOINT_DELTA = 0.75

    def plan(
        self,
        target_world: tuple[float, float, float],
        odometry: Any,
        joint_states: Any,
        *,
        half_width: float = 0.18,
    ) -> ArmCommand:
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=float(half_width),
        )

    def plan_from_held(
        self,
        hold_command: ArmCommand,
        held_center_base: tuple[float, float, float],
        joint_states: Any,
        *,
        half_width: float,
    ) -> ArmCommand:
        """Spread around the already-reached base-frame centre.

        Task 3 inserts the box with a bounded arm trajectory before lowering
        it using only the spine.  Recomputing the centre from world odometry
        after contact can shift the requested pose outside IK reach.  This
        path keeps that achieved centre and height, seeds IK from measured
        joints, and changes only the symmetric lateral spacing.
        """

        center = np.asarray(held_center_base, dtype=float)
        width = float(half_width)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise PregraspInputError("held release centre is invalid")
        if not math.isfinite(width) or width <= 0.0:
            raise PregraspInputError("held release half-width is invalid")
        positions, _velocities = _joint_maps(joint_states)
        reference = np.array(
            [
                positions["slide_joint"],
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
            ],
            dtype=float,
        )
        arm_center = center + np.array(
            [-GRASP_BACKOFF_X, 0.0, HAND_Z_OFFSET], dtype=float
        )
        left_target = arm_center + np.array([0.0, width, 0.0], dtype=float)
        right_target = arm_center + np.array([0.0, -width, 0.0], dtype=float)
        solutions = self._kdl.inverse_kinematics(
            T_left=_make_transform(left_target, LEFT_A_ROT),
            T_right=_make_transform(right_target, RIGHT_A_ROT),
            ref_pos=reference,
            target_height=float(hold_command.spine_position),
        )
        if solutions is None or len(solutions) == 0:
            raise PregraspPlanningError(
                "dual-arm relative release IK failed for held_center_base="
                f"{np.round(center, 3).tolist()}"
            )
        joints = np.asarray(solutions[0], dtype=float)
        if joints.shape != (13,) or not np.all(np.isfinite(joints)):
            raise PregraspPlanningError(
                "relative release IK returned invalid joints"
            )
        joint_delta = float(np.max(np.abs(joints[1:] - reference[1:])))
        if joint_delta > self.MAX_RELATIVE_RELEASE_JOINT_DELTA:
            raise PregraspPlanningError(
                "relative release IK changed branch; "
                f"max joint delta={joint_delta:.3f} rad"
            )

        self._target_base = tuple(float(value) for value in center)
        self._target_vector = np.array(
            [
                hold_command.spine_position,
                *hold_command.head_positions,
                *joints[1:7],
                GRIPPER_OPEN,
                *joints[7:13],
                GRIPPER_OPEN,
            ],
            dtype=float,
        )
        self._action_vector = np.array(
            [
                positions["slide_joint"],
                positions.get("head_yaw_joint", hold_command.head_positions[0]),
                positions.get("head_pitch_joint", hold_command.head_positions[1]),
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                hold_command.left_gripper_position,
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
                hold_command.right_gripper_position,
            ],
            dtype=float,
        )
        self._last_update_s = None
        self._stable_since_s = None
        return self.command()


class ArmRetractController:
    """Ramp both arms to the verified neutral transport posture.

    This action is intentionally run only after the base has retreated from a
    shelf/table.  The zero arm-joint posture is the existing reference
    transport posture; opening the grippers alone is not enough because the
    release IK pose can still leave both wrists extended into the obstacle.
    """

    TRANSPORT_SPINE = 0.10
    TRANSPORT_ARM = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self) -> None:
        self._target_vector: np.ndarray | None = None
        self._action_vector: np.ndarray | None = None
        self._last_update_s: float | None = None
        self._stable_since_s: float | None = None

    @property
    def planned(self) -> bool:
        return self._target_vector is not None

    def reset(self) -> None:
        self._target_vector = None
        self._action_vector = None
        self._last_update_s = None
        self._stable_since_s = None

    def plan(self, hold_command: ArmCommand, joint_states: Any) -> ArmCommand:
        positions, _velocities = _joint_maps(joint_states)
        values = np.array(
            [
                positions["slide_joint"],
                positions.get("head_yaw_joint", hold_command.head_positions[0]),
                positions.get("head_pitch_joint", hold_command.head_positions[1]),
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                GRIPPER_OPEN,
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
                GRIPPER_OPEN,
            ],
            dtype=float,
        )
        target = np.array(
            [
                self.TRANSPORT_SPINE,
                # Keep the camera pose unchanged; only retract the arms and
                # lower the spine to the neutral transport height.
                values[1],
                values[2],
                *self.TRANSPORT_ARM,
                GRIPPER_OPEN,
                *self.TRANSPORT_ARM,
                GRIPPER_OPEN,
            ],
            dtype=float,
        )
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise PregraspInputError("joint_states contain invalid retract positions")
        if not np.all(np.isfinite(target)):
            raise PregraspInputError("transport retract target is invalid")
        self._action_vector = values
        self._target_vector = target
        self._last_update_s = None
        self._stable_since_s = None
        return self.command()

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if self._target_vector is None or self._action_vector is None:
            raise PregraspPlanningError("arm retract update called before plan")
        positions, velocities = _joint_maps(joint_states)
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("control time is non-finite")
        dt = (
            0.05
            if self._last_update_s is None
            else min(0.20, max(0.01, now - self._last_update_s))
        )
        self._last_update_s = now

        diff = self._target_vector - self._action_vector
        max_step = COMMAND_RATE_PER_S * dt
        self._action_vector += np.sign(diff) * np.minimum(np.abs(diff), max_step)

        measured = np.array(
            [
                positions["slide_joint"],
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
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
        measured_velocity = np.array(
            [
                velocities.get("slide_joint", 0.0),
                *(velocities.get(f"left_arm_joint{index}", 0.0) for index in range(1, 7)),
                *(velocities.get(f"right_arm_joint{index}", 0.0) for index in range(1, 7)),
            ],
            dtype=float,
        )
        errors = np.abs(measured - target)
        slide_error = float(errors[0])
        left_error = float(np.max(errors[1:7]))
        right_error = float(np.max(errors[7:13]))
        command_error = float(np.max(np.abs(commanded - target)))
        max_velocity = float(np.max(np.abs(measured_velocity)))
        stable_now = (
            slide_error <= FEEDBACK_POS_TOL
            and left_error <= FEEDBACK_POS_TOL
            and right_error <= FEEDBACK_POS_TOL
            and command_error <= FEEDBACK_POS_TOL
            and max_velocity <= FEEDBACK_VEL_TOL
        )
        reached = False
        if stable_now:
            if self._stable_since_s is None:
                self._stable_since_s = now
            reached = now - self._stable_since_s >= FEEDBACK_STABLE_TIME
        else:
            self._stable_since_s = None
        detail = (
            f"transport retract; slide_err={slide_error:.3f}, "
            f"left_err={left_error:.3f}, right_err={right_error:.3f}, "
            f"cmd_err={command_error:.3f}, max_vel={max_velocity:.3f}"
        )
        return self.command(), reached, detail

    def command(self) -> ArmCommand:
        if self._action_vector is None:
            raise PregraspPlanningError("arm retract command requested before plan")
        values = self._action_vector
        return ArmCommand(
            spine_position=float(values[0]),
            head_positions=(float(values[1]), float(values[2])),
            left_arm_positions=tuple(float(value) for value in values[3:9]),
            left_gripper_position=float(values[9]),
            right_arm_positions=tuple(float(value) for value in values[10:16]),
            right_gripper_position=float(values[16]),
        )


class SlideHoldController:
    """Move the spine to an absolute target while preserving both arm poses."""

    def __init__(self) -> None:
        self._target_vector: np.ndarray | None = None
        self._action_vector: np.ndarray | None = None
        self._last_update_s: float | None = None
        self._stable_since_s: float | None = None

    @property
    def planned(self) -> bool:
        return self._target_vector is not None

    @property
    def target_slide(self) -> float | None:
        return None if self._target_vector is None else float(self._target_vector[0])

    def reset(self) -> None:
        self._target_vector = None
        self._action_vector = None
        self._last_update_s = None
        self._stable_since_s = None

    def plan(
        self,
        hold_command: ArmCommand,
        target_slide: float,
        joint_states: Any,
    ) -> ArmCommand:
        _joint_maps(joint_states)
        target = float(np.clip(float(target_slide), SPINE_MIN, SPINE_MAX))
        values = np.array(
            [
                hold_command.spine_position,
                *hold_command.head_positions,
                *hold_command.left_arm_positions,
                hold_command.left_gripper_position,
                *hold_command.right_arm_positions,
                hold_command.right_gripper_position,
            ],
            dtype=float,
        )
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise PregraspInputError("held ArmCommand is invalid")
        self._action_vector = values
        self._target_vector = values.copy()
        self._target_vector[0] = target
        self._last_update_s = None
        self._stable_since_s = None
        return self.command()

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if self._target_vector is None or self._action_vector is None:
            raise PregraspPlanningError("slide-hold update called before plan")
        positions, velocities = _joint_maps(joint_states)
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("control time is non-finite")
        dt = 0.05 if self._last_update_s is None else min(
            0.20, max(0.01, now - self._last_update_s)
        )
        self._last_update_s = now

        slide_diff = float(self._target_vector[0] - self._action_vector[0])
        max_step = COMMAND_RATE_PER_S * LIFT_SLIDE_COMMAND_RATIO * dt
        if abs(slide_diff) > 0.0:
            self._action_vector[0] += math.copysign(
                min(abs(slide_diff), max_step), slide_diff
            )

        measured_left = np.array(
            [positions[f"left_arm_joint{index}"] for index in range(1, 7)],
            dtype=float,
        )
        measured_right = np.array(
            [positions[f"right_arm_joint{index}"] for index in range(1, 7)],
            dtype=float,
        )
        slide_error = abs(positions["slide_joint"] - self._target_vector[0])
        left_error = float(np.max(np.abs(measured_left - self._target_vector[3:9])))
        right_error = float(np.max(np.abs(measured_right - self._target_vector[10:16])))
        command_error = abs(self._action_vector[0] - self._target_vector[0])
        # This controller only commands the slide.  The arm joints are
        # deliberately held at the previously commanded grasp pose, and the
        # simulator can report small arm velocity transients while that pose
        # is being maintained.  Including those unrelated velocities in the
        # slide settle condition can make a correctly placed box time out
        # indefinitely (the observed ``max_vel`` spikes came from the arms,
        # while the slide error was already within tolerance).
        slide_velocity = abs(velocities.get("slide_joint", 0.0))
        max_velocity = max(
            slide_velocity,
            *(abs(velocities.get(f"left_arm_joint{index}", 0.0)) for index in range(1, 7)),
            *(abs(velocities.get(f"right_arm_joint{index}", 0.0)) for index in range(1, 7)),
        )
        stable_now = (
            slide_error <= FEEDBACK_POS_TOL
            and left_error <= LIFT_ARM_POSITION_TOL
            and right_error <= LIFT_ARM_POSITION_TOL
            and command_error <= FEEDBACK_POS_TOL
            and slide_velocity <= FEEDBACK_VEL_TOL
        )
        reached = False
        if stable_now:
            if self._stable_since_s is None:
                self._stable_since_s = now
            reached = now - self._stable_since_s >= FEEDBACK_STABLE_TIME
        else:
            self._stable_since_s = None
        detail = (
            f"slide_target={self._target_vector[0]:.3f}, slide_err={slide_error:.3f}, "
            f"left_err={left_error:.3f}, right_err={right_error:.3f}, "
            f"cmd_err={command_error:.3f}, slide_vel={slide_velocity:.3f}, "
            f"max_vel={max_velocity:.3f}"
        )
        return self.command(), reached, detail

    def command(self) -> ArmCommand:
        if self._action_vector is None:
            raise PregraspPlanningError("slide-hold command requested before plan")
        values = self._action_vector
        return ArmCommand(
            spine_position=float(values[0]),
            head_positions=(float(values[1]), float(values[2])),
            left_arm_positions=tuple(float(value) for value in values[3:9]),
            left_gripper_position=float(values[9]),
            right_arm_positions=tuple(float(value) for value in values[10:16]),
            right_gripper_position=float(values[16]),
        )


__all__ = [
    "ArmRetractController",
    "HeldTransportController",
    "ReleaseSpreadController",
    "ShelfOpenPregraspController",
    "SlideHoldController",
]
