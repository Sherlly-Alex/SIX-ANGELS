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
    LIFT_ARM_POSITION_TOL,
    LIFT_SLIDE_COMMAND_RATIO,
    OpenPregraspController,
    PREGRASP_BACKOFF_X,
    PregraspInputError,
    PregraspPlanningError,
    SPINE_MAX,
    SPINE_MIN,
    _joint_maps,
)


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


class ReleaseSpreadController(OpenPregraspController):
    """Move the two open grippers away from a placed object."""

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
        max_velocity = max(
            abs(velocities.get("slide_joint", 0.0)),
            *(abs(velocities.get(f"left_arm_joint{index}", 0.0)) for index in range(1, 7)),
            *(abs(velocities.get(f"right_arm_joint{index}", 0.0)) for index in range(1, 7)),
        )
        stable_now = (
            slide_error <= FEEDBACK_POS_TOL
            and left_error <= LIFT_ARM_POSITION_TOL
            and right_error <= LIFT_ARM_POSITION_TOL
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
            f"slide_target={self._target_vector[0]:.3f}, slide_err={slide_error:.3f}, "
            f"left_err={left_error:.3f}, right_err={right_error:.3f}, "
            f"cmd_err={command_error:.3f}, max_vel={max_velocity:.3f}"
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
    "ReleaseSpreadController",
    "ShelfOpenPregraspController",
    "SlideHoldController",
]
