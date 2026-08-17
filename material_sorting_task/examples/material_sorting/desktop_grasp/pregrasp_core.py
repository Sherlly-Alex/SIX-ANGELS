"""ROS-free dual-arm pregrasp and bounded contact planning/control.

This module extracts the calibrated first phases from
``manual_dual_arm_pregrasp.py``.  ``OpenPregraspController`` stops outside the
box; ``ContactGraspController`` moves inward to the calibrated bilateral
contact pose.  The contact controller also provides a position-interface
compatible wrist-compliance layer: actuator effort detects first contact,
joint 6 follows the measured wrist angle while the pads settle, and the
achieved angles are then locked before bounded preload and lift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
# With effort feedback available, the ordinary position-controlled approach
# stops this far outside the measured box surface.  The remaining motion is
# generated continuously by ``track_inward_offset`` so contact is observable
# and the wrist can follow the surface before it is locked.
COMPLIANT_ENTRY_CLEARANCE_M = 0.010
COMPLIANT_ENTRY_TRAVEL_M = GRASP_INITIAL_PRELOAD + COMPLIANT_ENTRY_CLEARANCE_M
# A unilateral wrist lock indicates that the estimated box centre is slightly
# biased across the robot's lateral axis.  One bounded recentering step lets
# the opposite hand close the residual gap while relieving the locked hand.
COMPLIANT_LATERAL_RECENTER_STEP_M = 0.004
COMPLIANT_LATERAL_RECENTER_MAX_M = 0.004
BOX_HALF_EXTENTS_BY_ORIENTATION = {
    "yaw0": np.array([0.12, 0.08], dtype=float),
    "yaw90": np.array([0.08, 0.12], dtype=float),
}
SPINE_REFERENCE_Z = 1.32163718
SPINE_MIN = -0.04
SPINE_MAX = 0.87
HEAD_TARGET = (0.0, 0.45)

FEEDBACK_POS_TOL = 0.03
SQUEEZE_CONTACT_POS_TOL = 0.24
LIFT_HEIGHT = 0.15
LIFT_SLIDE_COMMAND_RATIO = 0.05
LIFT_ARM_POSITION_TOL = 0.24
FEEDBACK_VEL_TOL = 0.01
FEEDBACK_STABLE_TIME = 0.50
COMMAND_RATE_PER_S = 1.20
SLIDE_COMMAND_RATIO = 0.30

# Client-side compliant-contact parameters.  The Server exposes a fixed-gain
# position controller (joint-6 kp=350), so this is deliberately an admittance-
# like position-target strategy rather than direct torque control.
TOE_IN_ANGLE_RAD = math.radians(5.0)
WRIST_BASELINE_TIME_S = 0.40
WRIST_BASELINE_MIN_SAMPLES = 8
WRIST_EFFORT_FILTER_ALPHA = 0.25
WRIST_MIN_EFFORT_DELTA = 0.35
WRIST_EFFORT_NOISE_MULTIPLIER = 5.0
WRIST_FREE_ANGLE_LIMIT_RAD = math.radians(8.0)
WRIST_CONTACT_MIN_ROTATION_RAD = math.radians(1.5)
WRIST_CONTACT_CONFIRM_TIME_S = 0.15
WRIST_ALIGN_VELOCITY_RAD_S = 0.05
WRIST_ALIGN_STABLE_TIME_S = 0.20
WRIST_PRELOAD_SOFT_LIMIT_DELTA = 2.5
WRIST_ABSOLUTE_EFFORT_LIMIT = 6.0

# ArmCommand's flattened representation is
# [slide, head(2), left(6), left_gripper, right(6), right_gripper].
LEFT_WRIST_VECTOR_INDEX = 8
RIGHT_WRIST_VECTOR_INDEX = 15

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


def _base_yaw_rotation(angle: float) -> np.ndarray:
    """Return a rotation around the base-frame vertical axis."""

    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


# The left hand starts on +Y and therefore toes inward with negative base yaw;
# the right hand mirrors it.  Applying the correction in the base frame keeps
# the grasp face vertically oriented instead of rolling it around a tool axis.
LEFT_CONTACT_ROT = _base_yaw_rotation(-TOE_IN_ANGLE_RAD) @ LEFT_A_ROT
RIGHT_CONTACT_ROT = _base_yaw_rotation(TOE_IN_ANGLE_RAD) @ RIGHT_A_ROT


@dataclass
class _WristContactState:
    """Filtered feedback and latch state for one compliant wrist."""

    name: str
    vector_index: int
    baseline_positions: list[float] = field(default_factory=list)
    baseline_efforts: list[float] = field(default_factory=list)
    baseline_position: float | None = None
    baseline_effort: float | None = None
    effort_threshold: float = WRIST_MIN_EFFORT_DELTA
    filtered_effort: float | None = None
    nominal_command: float | None = None
    contact_candidate_since_s: float | None = None
    contact_seen: bool = False
    aligned_since_s: float | None = None
    aligned: bool = False
    locked_position: float | None = None
    latest_effort_delta: float = 0.0
    latest_angle_delta: float = 0.0
    latest_velocity: float = 0.0

    def reset(self) -> None:
        self.baseline_positions.clear()
        self.baseline_efforts.clear()
        self.baseline_position = None
        self.baseline_effort = None
        self.effort_threshold = WRIST_MIN_EFFORT_DELTA
        self.filtered_effort = None
        self.nominal_command = None
        self.contact_candidate_since_s = None
        self.contact_seen = False
        self.aligned_since_s = None
        self.aligned = False
        self.locked_position = None
        self.latest_effort_delta = 0.0
        self.latest_angle_delta = 0.0
        self.latest_velocity = 0.0


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


def _effort_map(joint_states: Any) -> dict[str, float] | None:
    """Return finite JointState effort values, or ``None`` when unavailable.

    The reference Server fills ``JointState.effort`` from MuJoCo's
    ``jointactuatorfrc`` sensors.  Some test doubles and older images omit the
    array, so compliance must fail open to the validated legacy position-only
    grasp instead of blocking the competition sequence.
    """

    if joint_states is None:
        return None
    try:
        names = tuple(str(name) for name in joint_states.name)
        efforts = tuple(float(value) for value in joint_states.effort)
    except (AttributeError, TypeError, ValueError):
        return None
    if not efforts or len(efforts) != len(names):
        return None
    if not all(math.isfinite(value) for value in efforts):
        return None
    return {name: efforts[index] for index, name in enumerate(names)}


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

    def __init__(
        self,
        kdl: MMK2Kdl | None = None,
        *,
        command_rate_per_s: float | None = None,
    ) -> None:
        self._kdl = kdl or MMK2Kdl()
        self._command_rate_per_s = float(
            COMMAND_RATE_PER_S
            if command_rate_per_s is None
            else command_rate_per_s
        )
        if not math.isfinite(self._command_rate_per_s) or self._command_rate_per_s <= 0.0:
            raise ValueError("command_rate_per_s must be finite and positive")
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

    @property
    def command_rate_per_s(self) -> float:
        return self._command_rate_per_s

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
        center_lateral_offset_m: float = 0.0,
        left_rotation: np.ndarray = LEFT_A_ROT,
        right_rotation: np.ndarray = RIGHT_A_ROT,
    ) -> ArmCommand:
        if not all(math.isfinite(float(value)) for value in target_world):
            raise PregraspInputError("target_world contains non-finite values")
        if not math.isfinite(float(center_backoff_x)):
            raise PregraspInputError("center_backoff_x is non-finite")
        if not math.isfinite(float(center_lateral_offset_m)):
            raise PregraspInputError("center_lateral_offset_m is non-finite")
        if not math.isfinite(float(half_width)) or float(half_width) <= 0.0:
            raise PregraspInputError("half_width must be finite and positive")
        left_rotation = np.asarray(left_rotation, dtype=float)
        right_rotation = np.asarray(right_rotation, dtype=float)
        if (
            left_rotation.shape != (3, 3)
            or right_rotation.shape != (3, 3)
            or not np.all(np.isfinite(left_rotation))
            or not np.all(np.isfinite(right_rotation))
        ):
            raise PregraspInputError("grasp rotations must be finite 3x3 matrices")
        positions, _velocities = _joint_maps(joint_states)
        robot_pose = _odometry_pose(odometry)
        box_center_base = _world_to_base(target_world, robot_pose)
        self._target_base = tuple(float(value) for value in box_center_base)

        arm_center_base = box_center_base + np.array(
            [-float(center_backoff_x), float(center_lateral_offset_m), HAND_Z_OFFSET],
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
            T_left=_make_transform(left_target, left_rotation),
            T_right=_make_transform(right_target, right_rotation),
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
        steps = ratios * self._command_rate_per_s * dt
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
    """Move inward, let both wrists settle on the box, then lock preload.

    ``joint6`` is still driven by the Server's position controller.  During
    contact this controller continuously updates its target to the measured
    wrist angle, which removes the fixed-orientation restoring error while the
    box surface rotates the pad.  Once bilateral settling is observed, the
    achieved angles become fixed targets again.  Missing effort feedback
    disables this enhancement and retains the former bounded-position grasp.
    """

    # Contact with the box can stop the measured joints before they reach the
    # unconstrained IK solution.  This tolerance does not declare success; it
    # only permits the hard-bounded inward search to advance.  The contact-only
    # executor still uses Server bilateral contact as its sole success signal;
    # lift-only may instead require the maximum bounded preload to settle.
    ARM_POSITION_TOL = SQUEEZE_CONTACT_POS_TOL

    def __init__(self, kdl: MMK2Kdl | None = None) -> None:
        super().__init__(kdl=kdl)
        self.ARM_POSITION_TOL = SQUEEZE_CONTACT_POS_TOL
        self._half_width: float | None = None
        self._orientation: str | None = None
        self._left_wrist = _WristContactState(
            "left_arm_joint6", LEFT_WRIST_VECTOR_INDEX
        )
        self._right_wrist = _WristContactState(
            "right_arm_joint6", RIGHT_WRIST_VECTOR_INDEX
        )
        self._baseline_started_s: float | None = None
        self._baseline_ready = False
        self._compliance_available = False
        self._compliance_abandoned_reason: str | None = None
        self._server_contact = False
        self._lateral_center_bias_m = 0.0

    @property
    def half_width(self) -> float | None:
        return self._half_width

    @property
    def orientation(self) -> str | None:
        return self._orientation

    @property
    def compliance_enabled(self) -> bool:
        return (
            self._baseline_ready
            and self._compliance_available
            and self._compliance_abandoned_reason is None
        )

    @property
    def bilateral_aligned(self) -> bool:
        return self.compliance_enabled and all(
            wrist.aligned for wrist in (self._left_wrist, self._right_wrist)
        )

    @property
    def any_contact(self) -> bool:
        return self.compliance_enabled and any(
            wrist.contact_seen for wrist in (self._left_wrist, self._right_wrist)
        )

    @property
    def bilateral_contact_seen(self) -> bool:
        """Whether both wrists have independently latched first contact."""

        return self.compliance_enabled and all(
            wrist.contact_seen for wrist in (self._left_wrist, self._right_wrist)
        )

    @property
    def preload_effort_limit_reached(self) -> bool:
        return self.compliance_enabled and any(
            wrist.latest_effort_delta >= WRIST_PRELOAD_SOFT_LIMIT_DELTA
            for wrist in (self._left_wrist, self._right_wrist)
        )

    @property
    def hard_effort_limit_exceeded(self) -> bool:
        if not self.compliance_enabled:
            return False
        return any(
            wrist.filtered_effort is not None
            and abs(wrist.filtered_effort) >= WRIST_ABSOLUTE_EFFORT_LIMIT
            for wrist in (self._left_wrist, self._right_wrist)
        )

    @property
    def diagnostic_summary(self) -> str:
        if self._compliance_abandoned_reason is not None:
            return "compliance=legacy_fallback:" + self._compliance_abandoned_reason
        if not self._baseline_ready:
            samples = min(
                len(self._left_wrist.baseline_efforts),
                len(self._right_wrist.baseline_efforts),
            )
            return (
                "compliance=baseline; "
                f"samples={samples}/{WRIST_BASELINE_MIN_SAMPLES}"
            )
        if not self._compliance_available:
            return "compliance=legacy_no_effort"
        left = self._wrist_detail(self._left_wrist)
        right = self._wrist_detail(self._right_wrist)
        return (
            "compliance=active; "
            f"center_bias_y={self._lateral_center_bias_m * 1000.0:+.1f}mm; "
            f"left[{left}]; right[{right}]"
        )

    def reset(self) -> None:
        super().reset()
        self.ARM_POSITION_TOL = SQUEEZE_CONTACT_POS_TOL
        self._half_width = None
        self._orientation = None
        self._left_wrist.reset()
        self._right_wrist.reset()
        self._baseline_started_s = None
        self._baseline_ready = False
        self._compliance_available = False
        self._compliance_abandoned_reason = None
        self._server_contact = False
        self._lateral_center_bias_m = 0.0

    def prepare_compliance(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[bool, str]:
        """Collect stationary wrist effort before starting inward motion."""

        if self._baseline_ready:
            return True, self.diagnostic_summary
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("compliance baseline time is non-finite")
        positions, _velocities = _joint_maps(joint_states)
        efforts = _effort_map(joint_states)
        if efforts is None or any(
            wrist.name not in efforts for wrist in (self._left_wrist, self._right_wrist)
        ):
            self._baseline_ready = True
            self._compliance_available = False
            return True, self.diagnostic_summary
        if self._baseline_started_s is None:
            self._baseline_started_s = now
        for wrist in (self._left_wrist, self._right_wrist):
            wrist.baseline_positions.append(float(positions[wrist.name]))
            wrist.baseline_efforts.append(float(efforts[wrist.name]))
        elapsed = max(0.0, now - self._baseline_started_s)
        sample_count = min(
            len(self._left_wrist.baseline_efforts),
            len(self._right_wrist.baseline_efforts),
        )
        if (
            elapsed < WRIST_BASELINE_TIME_S
            or sample_count < WRIST_BASELINE_MIN_SAMPLES
        ):
            return False, self.diagnostic_summary

        for wrist in (self._left_wrist, self._right_wrist):
            positions_array = np.asarray(wrist.baseline_positions, dtype=float)
            efforts_array = np.asarray(wrist.baseline_efforts, dtype=float)
            wrist.baseline_position = float(np.median(positions_array))
            wrist.baseline_effort = float(np.median(efforts_array))
            median_deviation = float(
                np.median(np.abs(efforts_array - wrist.baseline_effort))
            )
            wrist.effort_threshold = max(
                WRIST_MIN_EFFORT_DELTA,
                WRIST_EFFORT_NOISE_MULTIPLIER * 1.4826 * median_deviation,
            )
            wrist.filtered_effort = wrist.baseline_effort
        self._baseline_ready = True
        self._compliance_available = True
        return True, self.diagnostic_summary

    def observe_server_contact(self, confirmed: bool) -> None:
        self._server_contact = bool(confirmed)

    def abandon_compliance(self, reason: str) -> None:
        """Return to the validated bounded position-only contact search."""

        self._compliance_abandoned_reason = str(reason).strip() or "unspecified"
        for wrist in (self._left_wrist, self._right_wrist):
            if wrist.locked_position is not None and self._target_vector is not None:
                self._target_vector[wrist.vector_index] = wrist.locked_position
                if self._action_vector is not None:
                    self._action_vector[wrist.vector_index] = wrist.locked_position

    def retry_compliance(self) -> None:
        """Retry only wrists that have not completed surface alignment.

        A wrist that is already aligned has a valid measured surface angle.
        Keeping its lock prevents a one-sided retry from throwing away the
        successful side and making the grasp oscillate between two contacts.
        """

        if not self.compliance_enabled:
            return
        self._server_contact = False
        left_aligned = self._left_wrist.aligned
        right_aligned = self._right_wrist.aligned
        if left_aligned != right_aligned:
            # The left hand approaches from +Y and the right from -Y.  Shift
            # toward the locked hand: it gives that hand clearance and moves
            # the opposite hand inward toward the box centre.
            direction = 1.0 if left_aligned else -1.0
            self._lateral_center_bias_m = float(np.clip(
                self._lateral_center_bias_m
                + direction * COMPLIANT_LATERAL_RECENTER_STEP_M,
                -COMPLIANT_LATERAL_RECENTER_MAX_M,
                COMPLIANT_LATERAL_RECENTER_MAX_M,
            ))
        for wrist in (self._left_wrist, self._right_wrist):
            if wrist.aligned and wrist.locked_position is not None:
                continue
            wrist.contact_candidate_since_s = None
            wrist.contact_seen = False
            wrist.aligned_since_s = None
            wrist.aligned = False
            wrist.locked_position = None
            wrist.latest_effort_delta = 0.0
            wrist.latest_angle_delta = 0.0
            wrist.latest_velocity = 0.0

    def plan(
        self,
        target_world: tuple[float, float, float],
        orientation: str,
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        robot_pose = _odometry_pose(odometry)
        self._orientation = str(orientation)
        nominal_half_width = _oriented_grasp_half_width(
            self._orientation,
            robot_pose[2],
        )
        # The old implementation sent the arms directly to the nominal
        # contact/preload pose and only made the final 4 mm compliant.  When
        # effort feedback is usable, stop the fast segment 10 mm outside the
        # physical box face instead.  The executor then traverses the full
        # remaining gap continuously at the bounded compliant speed.
        self._half_width = nominal_half_width
        if self.compliance_enabled:
            self._half_width += COMPLIANT_ENTRY_TRAVEL_M
        command = self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=self._half_width,
            center_lateral_offset_m=self._lateral_center_bias_m,
            left_rotation=LEFT_CONTACT_ROT,
            right_rotation=RIGHT_CONTACT_ROT,
        )
        if self._target_vector is not None:
            self._left_wrist.nominal_command = float(
                self._target_vector[LEFT_WRIST_VECTOR_INDEX]
            )
            self._right_wrist.nominal_command = float(
                self._target_vector[RIGHT_WRIST_VECTOR_INDEX]
            )
        return command

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        command, pose_settled, detail = super().update(now_s, joint_states)
        if not self.compliance_enabled:
            return command, pose_settled, f"{detail}; {self.diagnostic_summary}"

        positions, velocities = _joint_maps(joint_states)
        efforts = _effort_map(joint_states)
        if efforts is None:
            # A transient missing sample must not erase already collected
            # baselines or unlock a wrist.  The executor will fall back after
            # its bounded wait if feedback never returns.
            return command, pose_settled, f"{detail}; compliance=waiting_effort"

        now = float(now_s)
        for wrist in (self._left_wrist, self._right_wrist):
            self._update_wrist_compliance(
                wrist,
                now,
                float(positions[wrist.name]),
                float(velocities.get(wrist.name, 0.0)),
                float(efforts[wrist.name]),
                pose_settled=pose_settled,
            )
        return self.command(), pose_settled, (
            f"{detail}; {self.diagnostic_summary}"
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
        # Once inward search begins, physical box contact can prevent the
        # unconstrained IK joint target from being reached exactly.  Reuse the
        # standalone desktop grasp's bounded-contact tolerance so all four
        # millimeter steps remain reachable as state-machine transitions.
        self.ARM_POSITION_TOL = SQUEEZE_CONTACT_POS_TOL
        self._half_width = max(nominal_half_width - offset, 0.01)
        command = self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=self._half_width,
            center_lateral_offset_m=self._lateral_center_bias_m,
            left_rotation=LEFT_CONTACT_ROT,
            right_rotation=RIGHT_CONTACT_ROT,
        )
        self._preserve_locked_wrists()
        return self.command() if self._action_vector is not None else command

    def track_inward_offset(
        self,
        target_world: tuple[float, float, float],
        inward_offset: float,
        odometry: Any,
        joint_states: Any,
    ) -> ArmCommand:
        """Retarget a moving inward pose without restarting arm interpolation.

        ``tighten()`` deliberately starts a fresh bounded pose transition and
        is retained for the legacy stepped search and one-shot retry backoff.
        The compliant search calls this method every control tick instead: IK
        is refreshed for the continuously changing half-width, while the
        current action, interpolation clock and wrist contact latches survive.
        """

        if self._orientation is None:
            raise PregraspPlanningError(
                "continuous contact tracking requested before initial plan"
            )
        offset = float(inward_offset)
        if not math.isfinite(offset) or offset < 0.0:
            raise PregraspInputError("inward_offset must be finite and non-negative")

        previous_action = (
            None if self._action_vector is None else self._action_vector.copy()
        )
        previous_update_s = self._last_update_s
        robot_pose = _odometry_pose(odometry)
        nominal_half_width = _oriented_grasp_half_width(
            self._orientation,
            robot_pose[2],
        )
        entry_half_width = nominal_half_width
        if self.compliance_enabled:
            entry_half_width += COMPLIANT_ENTRY_TRAVEL_M
        self.ARM_POSITION_TOL = SQUEEZE_CONTACT_POS_TOL
        self._half_width = max(entry_half_width - offset, 0.01)
        self._plan_pose(
            target_world,
            odometry,
            joint_states,
            center_backoff_x=GRASP_BACKOFF_X,
            half_width=self._half_width,
            center_lateral_offset_m=self._lateral_center_bias_m,
            left_rotation=LEFT_CONTACT_ROT,
            right_rotation=RIGHT_CONTACT_ROT,
        )

        # _plan_pose initializes interpolation from measured joints.  Restore
        # the already-issued action so a high-rate moving target remains smooth
        # rather than repeatedly snapping back to feedback on every tick.
        if previous_action is not None:
            self._action_vector = previous_action
        self._last_update_s = previous_update_s
        self._stable_since_s = None
        self._preserve_locked_wrists()
        return self.command()

    def _update_wrist_compliance(
        self,
        wrist: _WristContactState,
        now_s: float,
        position: float,
        velocity: float,
        effort: float,
        *,
        pose_settled: bool,
    ) -> None:
        assert wrist.baseline_position is not None
        assert wrist.baseline_effort is not None
        if wrist.filtered_effort is None:
            wrist.filtered_effort = effort
        else:
            wrist.filtered_effort = (
                WRIST_EFFORT_FILTER_ALPHA * effort
                + (1.0 - WRIST_EFFORT_FILTER_ALPHA) * wrist.filtered_effort
            )
        wrist.latest_effort_delta = abs(
            wrist.filtered_effort - wrist.baseline_effort
        )
        wrist.latest_angle_delta = abs(position - wrist.baseline_position)
        wrist.latest_velocity = abs(velocity)

        effort_evidence = wrist.latest_effort_delta >= wrist.effort_threshold
        # Require a settled arm pose to start an effort-only contact latch, so
        # acceleration torque cannot masquerade as box contact.  Once the
        # debounce has started, preserve it across the next half-millimetre IK
        # step as long as the effort evidence remains present.
        contact_evidence = self._server_contact or (
            effort_evidence
            and (
                pose_settled
                or wrist.contact_candidate_since_s is not None
            )
        )
        if not wrist.contact_seen:
            if contact_evidence:
                if wrist.contact_candidate_since_s is None:
                    wrist.contact_candidate_since_s = now_s
                elif (
                    now_s - wrist.contact_candidate_since_s
                    >= WRIST_CONTACT_CONFIRM_TIME_S
                ):
                    wrist.contact_seen = True
            else:
                wrist.contact_candidate_since_s = None

        if wrist.contact_seen and not wrist.aligned:
            nominal = (
                position if wrist.nominal_command is None else wrist.nominal_command
            )
            followed = float(
                np.clip(
                    position,
                    nominal - WRIST_FREE_ANGLE_LIMIT_RAD,
                    nominal + WRIST_FREE_ANGLE_LIMIT_RAD,
                )
            )
            assert self._target_vector is not None
            assert self._action_vector is not None
            self._target_vector[wrist.vector_index] = followed
            self._action_vector[wrist.vector_index] = followed

            angle_aligned = (
                wrist.latest_angle_delta >= WRIST_CONTACT_MIN_ROTATION_RAD
                or self._server_contact
            )
            # Effort is required to latch first contact above.  It is not
            # required to remain high while the free wrist rotates into full
            # surface contact: that rotation can unload joint 6 even though
            # the pad is still touching the box.  After contact_seen, use the
            # measured angle and low velocity to identify a settled face, while
            # the separate soft/absolute effort limits continue to protect the
            # subsequent preload.
            stable = (
                angle_aligned
                and wrist.latest_velocity <= WRIST_ALIGN_VELOCITY_RAD_S
            )
            if stable:
                if wrist.aligned_since_s is None:
                    wrist.aligned_since_s = now_s
                elif now_s - wrist.aligned_since_s >= WRIST_ALIGN_STABLE_TIME_S:
                    wrist.aligned = True
                    wrist.locked_position = followed
                    self._target_vector[wrist.vector_index] = followed
                    self._action_vector[wrist.vector_index] = followed
            else:
                wrist.aligned_since_s = None
        elif wrist.aligned and wrist.locked_position is not None:
            assert self._target_vector is not None
            assert self._action_vector is not None
            self._target_vector[wrist.vector_index] = wrist.locked_position
            self._action_vector[wrist.vector_index] = wrist.locked_position

    def _preserve_locked_wrists(self) -> None:
        if self._target_vector is None or self._action_vector is None:
            return
        for wrist in (self._left_wrist, self._right_wrist):
            if wrist.locked_position is None:
                continue
            self._target_vector[wrist.vector_index] = wrist.locked_position
            self._action_vector[wrist.vector_index] = wrist.locked_position

    @staticmethod
    def _wrist_detail(wrist: _WristContactState) -> str:
        return (
            f"contact={wrist.contact_seen}, aligned={wrist.aligned}, "
            f"angle={math.degrees(wrist.latest_angle_delta):.1f}deg, "
            f"effort_delta={wrist.latest_effort_delta:.2f}, "
            f"effort_threshold={wrist.effort_threshold:.2f}, "
            f"velocity={wrist.latest_velocity:.3f}"
        )


class SlideLiftController:
    """Raise the spine while preserving the established dual-arm preload."""

    def __init__(self, lift_height: float = LIFT_HEIGHT) -> None:
        self._lift_height = float(lift_height)
        if not math.isfinite(self._lift_height) or self._lift_height <= 0.0:
            raise ValueError("lift_height must be finite and positive")
        self._target_vector: np.ndarray | None = None
        self._action_vector: np.ndarray | None = None
        self._last_update_s: float | None = None
        self._stable_since_s: float | None = None
        self._actual_lift_m = 0.0

    @property
    def planned(self) -> bool:
        return self._target_vector is not None

    @property
    def actual_lift_m(self) -> float:
        return self._actual_lift_m

    @property
    def target_slide(self) -> float | None:
        if self._target_vector is None:
            return None
        return float(self._target_vector[0])

    def reset(self) -> None:
        self._target_vector = None
        self._action_vector = None
        self._last_update_s = None
        self._stable_since_s = None
        self._actual_lift_m = 0.0

    def plan(self, hold_command: ArmCommand, joint_states: Any) -> ArmCommand:
        _joint_maps(joint_states)
        start_slide = float(hold_command.spine_position)
        if not math.isfinite(start_slide):
            raise PregraspInputError("held spine command is non-finite")
        target_slide = float(
            np.clip(start_slide - self._lift_height, SPINE_MIN, SPINE_MAX)
        )
        self._actual_lift_m = start_slide - target_slide
        if self._actual_lift_m <= 1e-6:
            raise PregraspPlanningError(
                f"lift unavailable at held slide={start_slide:.3f}"
            )

        values = np.array(
            [
                start_slide,
                *hold_command.head_positions,
                *hold_command.left_arm_positions,
                hold_command.left_gripper_position,
                *hold_command.right_arm_positions,
                hold_command.right_gripper_position,
            ],
            dtype=float,
        )
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise PregraspInputError("held arm command is invalid")
        self._action_vector = values
        self._target_vector = values.copy()
        self._target_vector[0] = target_slide
        self._last_update_s = None
        self._stable_since_s = None
        return self.command()

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if self._target_vector is None or self._action_vector is None:
            raise PregraspPlanningError("slide-lift update called before plan")
        positions, velocities = _joint_maps(joint_states)
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("control time is non-finite")
        if self._last_update_s is None:
            dt = 0.05
        else:
            dt = min(0.20, max(0.01, now - self._last_update_s))
        self._last_update_s = now

        slide_diff = float(self._target_vector[0] - self._action_vector[0])
        max_step = COMMAND_RATE_PER_S * LIFT_SLIDE_COMMAND_RATIO * dt
        self._action_vector[0] += math.copysign(
            min(abs(slide_diff), max_step),
            slide_diff,
        ) if abs(slide_diff) > 0.0 else 0.0

        measured_left = np.array(
            [positions[f"left_arm_joint{index}"] for index in range(1, 7)],
            dtype=float,
        )
        measured_right = np.array(
            [positions[f"right_arm_joint{index}"] for index in range(1, 7)],
            dtype=float,
        )
        target_left = self._target_vector[3:9]
        target_right = self._target_vector[10:16]
        slide_error = abs(positions["slide_joint"] - self._target_vector[0])
        left_error = float(np.max(np.abs(measured_left - target_left)))
        right_error = float(np.max(np.abs(measured_right - target_right)))
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
            f"lift={self._actual_lift_m:.3f}m, "
            f"slide_target={self._target_vector[0]:.3f}, "
            f"slide_err={slide_error:.3f}, left_err={left_error:.3f}, "
            f"right_err={right_error:.3f}, cmd_err={command_error:.3f}, "
            f"max_vel={max_velocity:.3f}"
        )
        return self.command(), reached, detail

    def command(self) -> ArmCommand:
        if self._action_vector is None:
            raise PregraspPlanningError("slide-lift command requested before plan")
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
    "ContactGraspController",
    "OpenPregraspController",
    "PregraspInputError",
    "PregraspPlanningError",
    "SlideLiftController",
]
