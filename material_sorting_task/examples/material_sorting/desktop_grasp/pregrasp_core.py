"""ROS-free dual-arm pregrasp and bounded contact planning/control.

This module extracts the calibrated first phases from
``manual_dual_arm_pregrasp.py``.  ``OpenPregraspController`` stops outside the
box; ``ContactGraspController`` moves inward to the calibrated bilateral
contact pose.  Neither controller performs compliant squeeze or lift motions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from control_types import ArmCommand
from mmk2_kdl import MMK2Kdl


BOX_WIDTH_Y = 0.16
PREGRASP_BACKOFF_X = 0.08
SIDE_CLEARANCE = 0.145
HAND_Z_OFFSET = 0.02
GRIPPER_OPEN = 1.0
GRASP_BACKOFF_X = -0.02
GRASP_INITIAL_PRELOAD = 0.002
BOX_HALF_EXTENTS_BY_ORIENTATION = {
    "yaw0": np.array([0.12, 0.08], dtype=float),
    "yaw90": np.array([0.08, 0.12], dtype=float),
}
SPINE_REFERENCE_Z = 1.32163718
SPINE_MIN = -0.04
SPINE_MAX = 0.87
HEAD_TARGET = (0.0, 0.45)

FEEDBACK_POS_TOL = 0.03
GRASP_CONTACT_POS_TOL = 0.14
FEEDBACK_VEL_TOL = 0.01
FEEDBACK_STABLE_TIME = 0.50
COMMAND_RATE_PER_S = 1.20
SLIDE_COMMAND_RATIO = 0.30

LEFT_A_ROT = np.array(
    [
        [0.99890619, 0.04294831, 0.01848963],
        [-0.0203026, 0.04216758, 0.99890425],
        [0.04212158, -0.99818703, 0.04299342],
    ],
    dtype=float,
)
RIGHT_A_ROT = np.array(
    [
        [0.99890619, -0.04294831, 0.01848963],
        [0.0203026, 0.04216758, -0.99890425],
        [0.04212158, 0.99818703, 0.04299342],
    ],
    dtype=float,
)


class PregraspInputError(ValueError):
    """Required odometry or joint feedback is missing or invalid."""


class PregraspPlanningError(RuntimeError):
    """The requested open-pregrasp pose has no valid IK solution."""


def _make_transform(position_base, rotation_base):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation_base, dtype=float)
    transform[:3, 3] = np.asarray(position_base, dtype=float)
    return transform


def _joint_maps(joint_states: Any) -> tuple[dict[str, float], dict[str, float]]:
    if joint_states is None:
        raise PregraspInputError("joint_states have not been received")
    try:
        names = tuple(str(name) for name in joint_states.name)
        positions = tuple(float(value) for value in joint_states.position)
        velocities = tuple(float(value) for value in joint_states.velocity)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PregraspInputError(f"invalid joint_states: {exc}") from exc
    position_map = {
        name: positions[index]
        for index, name in enumerate(names)
        if index < len(positions)
    }
    velocity_map = {
        name: velocities[index]
        for index, name in enumerate(names)
        if index < len(velocities)
    }
    required = {
        "slide_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        *(f"right_arm_joint{index}" for index in range(1, 7)),
    }
    missing = sorted(required - set(position_map))
    if missing:
        raise PregraspInputError(
            "joint_states missing required positions: " + ", ".join(missing)
        )
    values = tuple(position_map[name] for name in required)
    if not all(math.isfinite(value) for value in values):
        raise PregraspInputError("joint_states contain non-finite positions")
    return position_map, velocity_map


def _odometry_pose(odometry: Any) -> tuple[float, float, float]:
    try:
        position = odometry.pose.pose.position
        orientation = odometry.pose.pose.orientation
        x = float(position.x)
        y = float(position.y)
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PregraspInputError(f"invalid odometry: {exc}") from exc
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise PregraspInputError("odometry contains a non-finite pose")
    return x, y, yaw


def _world_to_base(
    point_world: tuple[float, float, float],
    robot_pose: tuple[float, float, float],
) -> np.ndarray:
    robot_x, robot_y, robot_yaw = robot_pose
    delta_x = float(point_world[0]) - robot_x
    delta_y = float(point_world[1]) - robot_y
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)
    return np.array(
        [
            cos_yaw * delta_x - sin_yaw * delta_y,
            sin_yaw * delta_x + cos_yaw * delta_y,
            float(point_world[2]),
        ],
        dtype=float,
    )


def _oriented_grasp_half_width(orientation: str, robot_yaw: float) -> float:
    """Project a calibrated world-aligned box onto the robot lateral axis."""

    if orientation not in BOX_HALF_EXTENTS_BY_ORIENTATION:
        raise PregraspInputError(
            f"unsupported box orientation {orientation!r}; expected yaw0 or yaw90"
        )
    half_extents = BOX_HALF_EXTENTS_BY_ORIENTATION[orientation]
    base_y_world = np.array(
        [-math.sin(float(robot_yaw)), math.cos(float(robot_yaw))],
        dtype=float,
    )
    lateral_half_extent = float(np.dot(np.abs(base_y_world), half_extents))
    return max(lateral_half_extent - GRASP_INITIAL_PRELOAD, 0.01)


class OpenPregraspController:
    """Plan and ramp one open dual-arm pose using measured joint feedback."""

    ARM_POSITION_TOL = FEEDBACK_POS_TOL

    def __init__(self, kdl: MMK2Kdl | None = None) -> None:
        self._kdl = kdl or MMK2Kdl()
        self._target_vector: np.ndarray | None = None
        self._action_vector: np.ndarray | None = None
        self._last_update_s: float | None = None
        self._stable_since_s: float | None = None
        self._target_base: tuple[float, float, float] | None = None

    @property
    def planned(self) -> bool:
        return self._target_vector is not None

    @property
    def target_base(self) -> tuple[float, float, float] | None:
        return self._target_base

    def reset(self) -> None:
        self._target_vector = None
        self._action_vector = None
        self._last_update_s = None
        self._stable_since_s = None
        self._target_base = None

    def plan(
        self,
        target_world: tuple[float, float, float],
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        pregrasp_half_width = BOX_WIDTH_Y * 0.5 + SIDE_CLEARANCE
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=PREGRASP_BACKOFF_X,
            half_width=pregrasp_half_width,
        )

    def _plan_pose(
        self,
        target_world: tuple[float, float, float],
        odometry: Any,
        joint_states: Any,
        *,
        center_backoff_x: float,
        half_width: float,
    ) -> ArmCommand:
        if not all(math.isfinite(float(value)) for value in target_world):
            raise PregraspInputError("target_world contains non-finite values")
        if not math.isfinite(float(center_backoff_x)):
            raise PregraspInputError("center_backoff_x is non-finite")
        if not math.isfinite(float(half_width)) or float(half_width) <= 0.0:
            raise PregraspInputError("half_width must be finite and positive")
        positions, _velocities = _joint_maps(joint_states)
        robot_pose = _odometry_pose(odometry)
        box_center_base = _world_to_base(target_world, robot_pose)
        self._target_base = tuple(float(value) for value in box_center_base)

        arm_center_base = box_center_base + np.array(
            [-float(center_backoff_x), 0.0, HAND_Z_OFFSET],
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
                SPINE_REFERENCE_Z - arm_center_base[2],
                SPINE_MIN,
                SPINE_MAX,
            )
        )

        ref_pos = np.array(
            [
                positions["slide_joint"],
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
            ],
            dtype=float,
        )
        solutions = self._kdl.inverse_kinematics(
            T_left=_make_transform(left_target, LEFT_A_ROT),
            T_right=_make_transform(right_target, RIGHT_A_ROT),
            ref_pos=ref_pos,
            target_height=slide_target,
        )
        if solutions is None or len(solutions) == 0:
            raise PregraspPlanningError(
                "dual-arm IK failed for target_base="
                f"{np.round(box_center_base, 3).tolist()}"
            )
        joints = np.asarray(solutions[0], dtype=float)
        if joints.shape != (13,) or not np.all(np.isfinite(joints)):
            raise PregraspPlanningError(
                f"dual-arm IK returned invalid shape/value: {joints.shape}"
            )

        self._target_vector = np.array(
            [
                joints[0],
                HEAD_TARGET[0],
                HEAD_TARGET[1],
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
                positions.get("head_yaw_joint", 0.0),
                positions.get("head_pitch_joint", 0.0),
                *(positions[f"left_arm_joint{index}"] for index in range(1, 7)),
                GRIPPER_OPEN,
                *(positions[f"right_arm_joint{index}"] for index in range(1, 7)),
                GRIPPER_OPEN,
            ],
            dtype=float,
        )
        self._last_update_s = None
        self._stable_since_s = None
        return self.command()

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if self._target_vector is None or self._action_vector is None:
            raise PregraspPlanningError("open-pregrasp update called before plan")
        positions, velocities = _joint_maps(joint_states)
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("control time is non-finite")
        if self._last_update_s is None:
            dt = 0.05
        else:
            dt = min(0.20, max(0.01, now - self._last_update_s))
        self._last_update_s = now

        diff = np.abs(self._target_vector - self._action_vector)
        ratios = diff / (float(np.max(diff)) + 1e-6)
        ratios[0] *= SLIDE_COMMAND_RATIO
        steps = ratios * COMMAND_RATE_PER_S * dt
        self._action_vector += np.sign(
            self._target_vector - self._action_vector
        ) * np.minimum(diff, steps)

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
            and left_error <= self.ARM_POSITION_TOL
            and right_error <= self.ARM_POSITION_TOL
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
            f"target_base={tuple(round(value, 3) for value in self._target_base)}; "
            f"arm_tol={self.ARM_POSITION_TOL:.3f}, "
            f"slide_err={slide_error:.3f}, left_err={left_error:.3f}, "
            f"right_err={right_error:.3f}, cmd_err={command_error:.3f}, "
            f"max_vel={max_velocity:.3f}"
        )
        return self.command(), reached, detail

    def command(self) -> ArmCommand:
        if self._action_vector is None:
            raise PregraspPlanningError("open-pregrasp command requested before plan")
        values = self._action_vector
        return ArmCommand(
            spine_position=float(values[0]),
            head_positions=(float(values[1]), float(values[2])),
            left_arm_positions=tuple(float(value) for value in values[3:9]),
            left_gripper_position=float(values[9]),
            right_arm_positions=tuple(float(value) for value in values[10:16]),
            right_gripper_position=float(values[16]),
        )


class ContactGraspController(OpenPregraspController):
    """Move both open grippers inward to the calibrated task contact pose."""

    ARM_POSITION_TOL = GRASP_CONTACT_POS_TOL

    def __init__(self, kdl: MMK2Kdl | None = None) -> None:
        super().__init__(kdl=kdl)
        self._half_width: float | None = None
        self._orientation: str | None = None

    @property
    def half_width(self) -> float | None:
        return self._half_width

    @property
    def orientation(self) -> str | None:
        return self._orientation

    def reset(self) -> None:
        super().reset()
        self._half_width = None
        self._orientation = None

    def plan(
        self,
        target_world: tuple[float, float, float],
        orientation: str,
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        robot_pose = _odometry_pose(odometry)
        self._orientation = str(orientation)
        self._half_width = _oriented_grasp_half_width(
            self._orientation,
            robot_pose[2],
        )
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=self._half_width,
        )

    def tighten(
        self,
        target_world: tuple[float, float, float],
        inward_offset: float,
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        """Replan a bounded inward contact-search step from measured joints."""

        if self._orientation is None:
            raise PregraspPlanningError("contact tighten requested before initial plan")
        offset = float(inward_offset)
        if not math.isfinite(offset) or offset < 0.0:
            raise PregraspInputError("inward_offset must be finite and non-negative")
        robot_pose = _odometry_pose(odometry)
        nominal_half_width = _oriented_grasp_half_width(
            self._orientation,
            robot_pose[2],
        )
        self._half_width = max(nominal_half_width - offset, 0.01)
        return self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=self._half_width,
        )


__all__ = [
    "ContactGraspController",
    "OpenPregraspController",
    "PregraspInputError",
    "PregraspPlanningError",
]
