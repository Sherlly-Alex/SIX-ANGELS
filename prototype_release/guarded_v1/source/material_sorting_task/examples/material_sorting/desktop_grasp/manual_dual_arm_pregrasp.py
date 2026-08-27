"""Manual target -> optional base navigation -> dual-arm grasp and lift.

The target can come from the semantic RGB-D pipeline or be supplied manually:
1. give a target in robot base frame and only move the arms, or
2. give a target in world frame and let the robot first drive to a stand-off pose.

Examples:

    python3 examples/material_sorting/desktop_grasp/manual_dual_arm_pregrasp.py \
      --target-frame base --target 0.55 0.0 0.824

    python3 examples/material_sorting/desktop_grasp/manual_dual_arm_pregrasp.py \
      --target-frame world --target -0.54 2.30 0.824

Use --pregrasp-only if you only want to move to the open pre-grasp pose.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from mmk2_kdl import MMK2Kdl


# Edit here: default target box center in WORLD frame, unit: meter.
TARGET_WORLD = np.array([-0.54, 2.30, 0.824], dtype=float)

# Colored box size: 24 x 16 x 19 cm, unit: meter.
BOX_LENGTH_X = 0.24
BOX_WIDTH_Y = 0.16
BOX_HEIGHT_Z = 0.19

# Fallback profile used when a direct target has no orientation metadata.
# Semantic targets select the calibrated half-width from their yaw label.
# The grippers stop outside the box during pre-grasp; they do not close.
PREGRASP_BACKOFF_X = 0.08
SIDE_CLEARANCE = 0.145
HAND_Z_OFFSET = 0.02
GRIPPER_OPEN = 1.0

# Grasp parameters.  The finger collision meshes span roughly +/-3 cm along
# base X in this wrist orientation, so placing the endpoint 2 cm in front of
# the box center gives both fingers a broad overlap with the side face.
GRASP_BACKOFF_X = -0.02
# Fallback value used when semantic orientation is unavailable.
GRASP_HALF_WIDTH = 0.118
# Box half extents in WORLD X/Y.  The arms close along base Y, so the grasp
# width must be the projection of these extents onto the robot's lateral axis.
BOX_HALF_EXTENTS_BY_ORIENTATION = {
    "yaw0": np.array([0.12, 0.08], dtype=float),
    "yaw90": np.array([0.08, 0.12], dtype=float),
}
GRASP_INITIAL_PRELOAD = 0.002
TASK_SOURCE_ORIENTATION = {
    1: "yaw0",
    2: "yaw90",
    3: "yaw90",
}
# The two arms provide the lateral hug force.  Keep each gripper open so its
# upper and lower fingers both remain on the 0.19 m-high box side face.
GRIPPER_CONTACT = GRIPPER_OPEN
GRIPPER_HOLD = GRIPPER_CONTACT
LIFT_HEIGHT = 0.15
LIFT_SLIDE_RATIO = 0.05
TRANSPORT_BACKOFF_X = -0.02
# The box is stable after lift but slips when the arm configuration retracts.
# Hold the lift pose directly and skip the transport/retract motion.
ENABLE_TRANSPORT_RETRACT = False
# Enable the bounded post-grasp squeeze by default.  The CLI still exposes
# --no-compliant-squeeze for a one-off diagnostic run.
DEFAULT_COMPLIANT_SQUEEZE = True
TARGET_INFO_TOPIC = "/material/target_info"

# JointState.effort is a tendon-transmission generalized effort, not a fingertip
# normal force in Newtons. It is kept for contact diagnostics only; stable grasp
# is produced by a bounded compliant squeeze after the original pose is reached.
GRIPPER_HOLD_SETTLE_TIME = 1.0
COMPLIANT_MIN_SQUEEZE = 0.003
COMPLIANT_SQUEEZE_STEP = 0.001
COMPLIANT_MAX_SQUEEZE = 0.004
COMPLIANT_EFFORT_DELTA_LIMIT = 30.0
COMPLIANT_SQUEEZE_INTERVAL = 0.30
HOLD_RETIGHTEN_STEP = 0.001
# Keep the transport preload fixed after retract. Repeated open-loop tightening
# made the two-arm contact asymmetric and pushed the box downward.
HOLD_RETIGHTEN_MAX = 0.0
HOLD_RETIGHTEN_INTERVAL = 3.0
HOLD_MIN_EFFORT_RATIO = 0.75
# Simple base navigation parameters used only in world-frame mode.
DEFAULT_STAND_OFF = 0.70
DEFAULT_APPROACH_YAW = math.pi / 2.0
NAV_POS_TOL = 0.015
NAV_YAW_TOL = 0.008
NAV_MAX_LIN = 0.25
NAV_MAX_ANG = 0.80
FEEDBACK_POS_TOL = 0.03
GRASP_CONTACT_POS_TOL = 0.14
# During compliant preload, the box can intentionally prevent the joints from
# reaching the exact IK target.  Allow the settled contact pose to advance to
# the next squeeze step; lift/transport retain their stricter tolerances.
# The open-finger preload can settle against the box with one arm roughly
# 0.21 rad from the unconstrained IK target.  Treat that bounded, low-velocity
# contact as settled so the state machine can finish squeezing and lift.
SQUEEZE_CONTACT_POS_TOL = 0.24
LIFT_HOLD_POS_TOL = 0.085
# With a box held between both arms, one arm can settle against contact before
# matching the exact lift IK joint target.  Require the slide/command to settle
# but allow this bounded joint error so transport can start.
LIFT_CONTACT_POS_TOL = 0.24
TRANSPORT_HOLD_POS_TOL = 0.10
# During retract, the held box can keep one arm from reaching its unconstrained
# IK target. Treat a settled contact pose as valid and add a small preload
# before the backward motion so the open fingers do not lose the box.
TRANSPORT_CONTACT_POS_TOL = 0.24
TRANSPORT_PRELOAD_STEP = 0.003
FEEDBACK_VEL_TOL = 0.01
FEEDBACK_STABLE_TIME = 0.5
FEEDBACK_LOG_PERIOD = 2.0

# Spine/hand z reference copied from the existing client_task_1.1.py behavior.
SPINE_REFERENCE_Z = 1.32163718
SPINE_MIN = -0.04
SPINE_MAX = 0.87

# End-effector rotation matrices copied from the existing dual-arm hug grasp.
LEFT_A_ROT = np.array([
    [0.99890619, 0.04294831, 0.01848963],
    [-0.0203026, 0.04216758, 0.99890425],
    [0.04212158, -0.99818703, 0.04299342],
])
RIGHT_A_ROT = np.array([
    [0.99890619, -0.04294831, 0.01848963],
    [0.0203026, 0.04216758, -0.99890425],
    [0.04212158, 0.99818703, 0.04299342],
])


def move_towards(current, target, max_step):
    diff = target - current
    if abs(diff) <= max_step:
        return target
    return current + math.copysign(max_step, diff)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def oriented_grasp_half_width(orientation, robot_yaw):
    """Project the box's world XY half extents onto the robot's base-Y axis."""
    half_extents = BOX_HALF_EXTENTS_BY_ORIENTATION[orientation]
    base_y_world = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
    lateral_half_extent = float(np.dot(np.abs(base_y_world), half_extents))
    return max(lateral_half_extent - GRASP_INITIAL_PRELOAD, 0.01)


def make_transform(position_base, rotation_base):
    transform = np.eye(4)
    transform[:3, :3] = rotation_base
    transform[:3, 3] = np.asarray(position_base, dtype=float)
    return transform


class ManualDualArmPregrasp(Node):
    def __init__(
        self,
        target,
        target_frame,
        auto_nav,
        stand_off,
        approach_yaw,
        do_grasp,
        lift_height,
        transport_backoff_x,
        target_topic,
        compliant_squeeze,
        squeeze_step,
        max_squeeze,
        effort_delta_limit,
        grasp_half_width,
        hand_z_offset,
        grasp_backoff_x,
        target_orientation,
    ):
        super().__init__("manual_dual_arm_pregrasp")
        self.kdl = MMK2Kdl()
        self.target = np.asarray(target, dtype=float)
        self.target_topic = target_topic
        self.target_received = not bool(target_topic)
        self.target_frame = "world" if target_topic else target_frame
        self.auto_nav = bool(auto_nav and self.target_frame == "world")
        self.stand_off = float(stand_off)
        self.approach_yaw = approach_yaw
        self.do_grasp = bool(do_grasp)
        self.lift_height = float(lift_height)
        self.transport_backoff_x = float(transport_backoff_x)
        self.compliant_squeeze = bool(compliant_squeeze)
        self.squeeze_step = max(float(squeeze_step), 0.0)
        requested_squeeze = max(float(max_squeeze), 0.0)
        self.max_squeeze = (
            max(requested_squeeze, COMPLIANT_MIN_SQUEEZE)
            if self.compliant_squeeze else 0.0
        )
        self.effort_delta_limit = max(float(effort_delta_limit), 0.0)
        self.grasp_half_width = max(float(grasp_half_width), 0.01)
        self.target_orientation = target_orientation
        self.target_orientation_source = "cli" if target_orientation else None
        self.hand_z_offset = float(hand_z_offset)
        self.grasp_backoff_x = float(grasp_backoff_x)
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None
        self.jvel = {}
        self.jeffort = {}
        self.nav_goal_xy = None
        self.nav_goal_yaw = None
        self.nav_done = not self.auto_nav
        self.nav_logged_done = False
        self.phase = "pregrasp"
        self.pregrasp_planned = False
        self.grasp_planned = False
        self.lift_planned = False
        self.transport_planned = False
        self.squeeze_planned = False
        self.finished_logged = False
        self.last_feedback_log_time = 0.0
        self.feedback_stable_since = None
        self.hold_half_width = self.grasp_half_width
        self.squeeze_baseline_effort = None
        self.squeeze_next_time = 0.0
        self.hold_retighten_used = 0.0
        self.hold_retighten_planned = False
        self.hold_next_time = 0.0
        self.hold_gripper_effort_reference = None
        self.gripper_hold_ready_time = 0.0
        self.hold_center_base = None
        self.tc = np.zeros(19)
        self.tc[11] = GRIPPER_OPEN
        self.tc[18] = GRIPPER_OPEN
        self.action = self.tc.copy()

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(
            Float64MultiArray,
            "/spine_forward_position_controller/commands",
            5,
        )
        self.head_pub = self.create_publisher(
            Float64MultiArray,
            "/head_forward_position_controller/commands",
            5,
        )
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray,
            "/left_arm_forward_position_controller/commands",
            5,
        )
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray,
            "/right_arm_forward_position_controller/commands",
            5,
        )

        self.create_subscription(
            Odometry,
            "/slamware_ros_sdk_server_node/odom",
            self.odom_cb,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        if self.target_topic:
            self.create_subscription(PointStamped, self.target_topic, self.target_cb, 5)
            self.create_subscription(String, TARGET_INFO_TOPIC, self.target_info_cb, 5)

        self.dt = 1.0 / 24.0
        self.timer = self.create_timer(self.dt, self.tick)
        waiting_for = "odom + joint_states"
        if self.target_topic:
            waiting_for += f" + target on {self.target_topic}"
        self.get_logger().info(
            f"manual_dual_arm_pregrasp up; waiting for {waiting_for} ..."
        )
        if self.target_topic:
            self.get_logger().info(
                f"target will be received in world frame from {self.target_topic}; "
                f"auto_nav={self.auto_nav}; stand_off={self.stand_off:.2f}; "
                f"do_grasp={self.do_grasp}"
            )
        else:
            self.get_logger().info(
                f"target_{self.target_frame}={np.round(self.target, 3)}; "
                f"auto_nav={self.auto_nav}; stand_off={self.stand_off:.2f}; "
                f"do_grasp={self.do_grasp}"
            )

    def target_cb(self, msg):
        if self.target_received:
            return
        if msg.header.frame_id and msg.header.frame_id != "world":
            self.get_logger().error(
                f"target topic frame must be 'world', got {msg.header.frame_id!r}"
            )
            return
        target = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        if not np.all(np.isfinite(target)):
            self.get_logger().error(f"ignoring non-finite target: {target}")
            return
        self.target = target
        self.target_frame = "world"
        self.target_received = True
        self.nav_goal_xy = None
        self.nav_goal_yaw = None
        self.nav_done = not self.auto_nav
        self.get_logger().info(
            f"target locked from {self.target_topic}: world={np.round(self.target, 3)}"
        )

    def target_info_cb(self, msg):
        """Receive the semantic node's locked target orientation metadata."""
        if not self.target_topic or self.target_orientation_source == "cli":
            return
        try:
            info = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        detected_orientation = info.get("orientation")
        try:
            task_id = int(info.get("task"))
        except (TypeError, ValueError):
            task_id = None
        orientation = TASK_SOURCE_ORIENTATION.get(task_id, detected_orientation)
        if orientation not in BOX_HALF_EXTENTS_BY_ORIENTATION:
            return
        try:
            info_target = np.asarray(info.get("target_world", []), dtype=float)
        except (TypeError, ValueError):
            return
        if self.target_received and (
            info_target.shape != (3,)
            or not np.all(np.isfinite(info_target))
            or float(np.linalg.norm(info_target - self.target)) > 0.08
        ):
            return
        if orientation != self.target_orientation:
            if (
                detected_orientation in BOX_HALF_EXTENTS_BY_ORIENTATION
                and detected_orientation != orientation
            ):
                self.get_logger().warning(
                    f"overriding target_info orientation {detected_orientation} "
                    f"with task-{task_id} source-slot orientation {orientation}"
                )
            self.target_orientation = orientation
            self.target_orientation_source = "target_info"
            reference_yaw = (
                self.base_yaw if self.approach_yaw is None else self.approach_yaw
            )
            nominal_half_width = oriented_grasp_half_width(
                orientation, reference_yaw
            )
            self.get_logger().info(
                f"target orientation locked: {orientation}; "
                f"nominal_grasp_half_width={nominal_half_width:.4f} m"
            )

    def odom_cb(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.base_xy = np.array([position.x, position.y], dtype=float)
        self.base_yaw = Rotation.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]).as_euler("xyz")[2]

    def joint_state_cb(self, msg):
        self.jpos = {
            name: msg.position[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.position)
        }
        self.jvel = {
            name: msg.velocity[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.velocity)
        }
        self.jeffort = {
            name: msg.effort[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.effort)
        }
    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def left_arm_meas(self):
        return np.array([
            self.jpos.get(f"left_arm_joint{index + 1}", self.tc[5 + index])
            for index in range(6)
        ])

    @property
    def right_arm_meas(self):
        return np.array([
            self.jpos.get(f"right_arm_joint{index + 1}", self.tc[12 + index])
            for index in range(6)
        ])

    @property
    def slide_vel(self):
        return float(self.jvel.get("slide_joint", 0.0))

    @property
    def left_arm_vel(self):
        return np.array([
            self.jvel.get(f"left_arm_joint{index + 1}", 0.0)
            for index in range(6)
        ])

    @property
    def right_arm_vel(self):
        return np.array([
            self.jvel.get(f"right_arm_joint{index + 1}", 0.0)
            for index in range(6)
        ])

    @property
    def arm_effort_meas(self):
        return np.array([
            *[self.jeffort.get(f"left_arm_joint{index + 1}", 0.0) for index in range(6)],
            *[self.jeffort.get(f"right_arm_joint{index + 1}", 0.0) for index in range(6)],
        ])
    @property
    def gripper_effort_meas(self):
        return np.array([
            abs(self.jeffort.get("left_arm_eef_gripper_joint", 0.0)),
            abs(self.jeffort.get("right_arm_eef_gripper_joint", 0.0)),
        ])


    def world_to_base(self, point_world):
        delta = np.asarray(point_world, dtype=float) - np.array([
            self.base_xy[0],
            self.base_xy[1],
            0.0,
        ])
        cos_yaw = math.cos(-self.base_yaw)
        sin_yaw = math.sin(-self.base_yaw)
        return np.array([
            cos_yaw * delta[0] - sin_yaw * delta[1],
            sin_yaw * delta[0] + cos_yaw * delta[1],
            delta[2],
        ])

    def target_base(self):
        if self.target_frame == "base":
            return self.target.copy()
        return self.world_to_base(self.target)

    def setup_nav_goal(self):
        if self.nav_goal_xy is not None:
            return
        if self.approach_yaw is None:
            direction = self.target[:2] - self.base_xy
            self.nav_goal_yaw = math.atan2(direction[1], direction[0])
        else:
            self.nav_goal_yaw = float(self.approach_yaw)
        forward = np.array([
            math.cos(self.nav_goal_yaw),
            math.sin(self.nav_goal_yaw),
        ])
        self.nav_goal_xy = self.target[:2] - self.stand_off * forward
        self.get_logger().info(
            f"nav_goal_xy={np.round(self.nav_goal_xy, 3)} "
            f"nav_goal_yaw={self.nav_goal_yaw:.3f} rad"
        )

    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def nav_step(self):
        self.setup_nav_goal()
        delta = self.nav_goal_xy - self.base_xy
        dist = float(np.linalg.norm(delta))

        if dist > NAV_POS_TOL:
            target_yaw = math.atan2(delta[1], delta[0])
            yaw_err = normalize_angle(target_yaw - self.base_yaw)
            linear_x = min(NAV_MAX_LIN, 0.65 * dist) if abs(yaw_err) < 0.55 else 0.0
            angular_z = float(np.clip(1.8 * yaw_err, -NAV_MAX_ANG, NAV_MAX_ANG))
            self.publish_cmd_vel(linear_x, angular_z)
            return False

        yaw_err = normalize_angle(self.nav_goal_yaw - self.base_yaw)
        if abs(yaw_err) > NAV_YAW_TOL:
            angular_z = float(np.clip(1.8 * yaw_err, -NAV_MAX_ANG, NAV_MAX_ANG))
            self.publish_cmd_vel(0.0, angular_z)
            return False

        self.publish_cmd_vel(0.0, 0.0)
        if not self.nav_logged_done:
            self.nav_logged_done = True
            self.get_logger().info(
                f"base aligned: base=({self.base_xy[0]:.3f}, {self.base_xy[1]:.3f}) "
                f"yaw={self.base_yaw:.3f}"
            )
        return True

    def current_ref_pos13(self):
        ref = np.zeros(13)
        ref[0] = float(self.slide_meas)
        ref[1:7] = self.left_arm_meas
        ref[7:13] = self.right_arm_meas
        return ref

    def seed_action_from_feedback(self):
        self.action[2] = float(self.slide_meas)
        self.action[5:11] = self.left_arm_meas
        self.action[12:18] = self.right_arm_meas
        if self.phase in {"pregrasp", "grasp"}:
            self.action[11] = GRIPPER_OPEN
            self.action[18] = GRIPPER_OPEN
        else:
            self.action[11] = self.jpos.get(
                "left_arm_eef_gripper_joint", self.action[11]
            )
            self.action[18] = self.jpos.get(
                "right_arm_eef_gripper_joint", self.action[18]
            )

    def plan_arm_pose(
        self,
        label,
        center_backoff_x,
        half_width,
        gripper_value,
        z_offset=HAND_Z_OFFSET,
        seed_from_feedback=False,
        center_base=None,
    ):
        box_center_base = (
            self.target_base()
            if center_base is None else np.asarray(center_base, dtype=float).copy()
        )
        if seed_from_feedback:
            self.seed_action_from_feedback()

        arm_center_base = box_center_base + np.array([
            -center_backoff_x,
            0.0,
            z_offset,
        ])
        left_target_base = arm_center_base + np.array([0.0, half_width, 0.0])
        right_target_base = arm_center_base + np.array([0.0, -half_width, 0.0])

        slide_target = np.clip(
            SPINE_REFERENCE_Z - arm_center_base[2],
            SPINE_MIN,
            SPINE_MAX,
        )

        left_transform = make_transform(left_target_base, LEFT_A_ROT)
        right_transform = make_transform(right_target_base, RIGHT_A_ROT)
        solutions = self.kdl.inverse_kinematics(
            T_left=left_transform,
            T_right=right_transform,
            ref_pos=self.current_ref_pos13(),
            target_height=float(slide_target),
        )
        if not solutions:
            self.get_logger().error(
                f"{label} IK failed. Move the base closer or adjust target. "
                f"box_center_base={np.round(box_center_base, 3)} "
                f"left={np.round(left_target_base, 3)} right={np.round(right_target_base, 3)}"
            )
            return False

        joints = np.asarray(solutions[0], dtype=float)
        self.tc[2] = float(joints[0])
        self.tc[3] = 0.0
        self.tc[4] = 0.45
        self.tc[5:11] = joints[1:7]
        self.tc[11] = float(gripper_value)
        self.tc[12:18] = joints[7:13]
        self.tc[18] = float(gripper_value)

        self.get_logger().info(f"{label}: box_center_base={np.round(box_center_base, 3)}")
        self.get_logger().info(
            f"{label}: left_base={np.round(left_target_base, 3)} "
            f"right_base={np.round(right_target_base, 3)} "
            f"half_width={half_width:.3f} gripper={gripper_value:.3f} slide={self.tc[2]:.3f}"
        )
        return True

    def plan_pregrasp(self):
        pregrasp_half_width = BOX_WIDTH_Y * 0.5 + SIDE_CLEARANCE
        ok = self.plan_arm_pose(
            "pregrasp",
            PREGRASP_BACKOFF_X,
            pregrasp_half_width,
            GRIPPER_OPEN,
            z_offset=self.hand_z_offset,
            seed_from_feedback=True,
            center_base=self.hold_center_base,
        )
        if ok:
            self.get_logger().info("Pre-grasp IK solved; moving with grippers open.")
        return ok

    def plan_grasp(self):
        if self.target_orientation in BOX_HALF_EXTENTS_BY_ORIENTATION:
            robot_yaw = self.base_yaw if self.target_frame == "world" else 0.0
            self.grasp_half_width = oriented_grasp_half_width(
                self.target_orientation, robot_yaw
            )
            self.hold_half_width = self.grasp_half_width
            self.get_logger().info(
                f"orientation-aware grasp: box={self.target_orientation}, "
                f"base_yaw={robot_yaw:.3f} rad, "
                f"half_width={self.grasp_half_width:.4f} m"
            )
        ok = self.plan_arm_pose(
            "grasp",
            self.grasp_backoff_x,
            self.grasp_half_width,
            GRIPPER_OPEN,
            z_offset=self.hand_z_offset,
            seed_from_feedback=False,
        )
        if ok:
            self.hold_center_base = self.target_base()
            self.get_logger().info(
                "Grasp IK solved; moving both arms inward with grippers open."
            )
        return ok

    def plan_lift(self):
        # Raising the spine moves both end effectors together.  Preserve the
        # arm pose established by compliant contact instead of resolving IK at
        # a new height, which can redistribute pressure and start box slip.
        start_slide = float(self.tc[2])
        target_slide = float(np.clip(
            start_slide - self.lift_height,
            SPINE_MIN,
            SPINE_MAX,
        ))
        actual_lift = start_slide - target_slide
        if actual_lift <= 1e-6:
            self.get_logger().error(
                f"lift unavailable at slide={start_slide:.3f}; "
                f"requested={self.lift_height:.3f} m"
            )
            return False
        self.tc[2] = target_slide
        self.tc[11] = GRIPPER_HOLD
        self.tc[18] = GRIPPER_HOLD
        self.get_logger().info(
            f"lift: slide-only {start_slide:.3f}->{target_slide:.3f}; "
            f"actual_height={actual_lift:.3f} m; "
            f"half_width={self.hold_half_width:.3f}; gripper={GRIPPER_HOLD:.3f}"
        )
        return True

    def plan_transport(self):
        transport_half_width = max(
            self.hold_half_width - TRANSPORT_PRELOAD_STEP,
            0.01,
        )
        ok = self.plan_arm_pose(
            "transport",
            self.transport_backoff_x,
            transport_half_width,
            GRIPPER_HOLD,
            z_offset=self.hand_z_offset + self.lift_height,
            seed_from_feedback=False,
            center_base=self.hold_center_base,
        )
        if ok:
            self.hold_half_width = transport_half_width
            self.get_logger().info(
                f"Transport IK solved; moving the held box backward by "
                f"{self.transport_backoff_x - self.grasp_backoff_x:.3f} m "
                f"with transport preload half_width={transport_half_width:.3f} m."
            )
        return ok
    def smooth_and_publish(self):
        diff = np.abs(self.action - self.tc)
        ratios = diff / (np.max(diff) + 1e-6)
        ratios[2] *= LIFT_SLIDE_RATIO if self.phase == "lift" else 0.3
        max_step = 1.2 * self.dt
        for index in range(2, 19):
            self.action[index] = move_towards(
                self.action[index],
                self.tc[index],
                ratios[index] * max_step,
            )

        self.publish_cmd_vel(0.0, 0.0)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.left_arm_pub.publish(Float64MultiArray(
            data=[float(value) for value in self.action[5:11]] + [float(self.action[11])]
        ))
        self.right_arm_pub.publish(Float64MultiArray(
            data=[float(value) for value in self.action[12:18]] + [float(self.action[18])]
        ))

    def converged(self):
        if self.jpos is None:
            return False

        measured = np.concatenate((
            np.array([self.slide_meas], dtype=float),
            self.left_arm_meas,
            self.right_arm_meas,
        ))
        target = np.concatenate((
            np.array([self.tc[2]], dtype=float),
            self.tc[5:11],
            self.tc[12:18],
        ))
        measured_velocity = np.concatenate((
            np.array([self.slide_vel], dtype=float),
            self.left_arm_vel,
            self.right_arm_vel,
        ))
        commanded = np.concatenate((
            np.array([self.action[2]], dtype=float),
            self.action[5:11],
            self.action[12:18],
        ))

        errors = np.abs(measured - target)
        command_error = float(np.max(np.abs(commanded - target)))
        slide_error = float(errors[0])
        left_arm_max = float(np.max(errors[1:7]))
        right_arm_max = float(np.max(errors[7:13]))
        max_velocity = float(np.max(np.abs(measured_velocity)))
        if self.phase == "grasp":
            arm_tolerance = GRASP_CONTACT_POS_TOL
        elif self.phase == "squeeze":
            arm_tolerance = SQUEEZE_CONTACT_POS_TOL
        elif self.phase == "gripper_hold":
            arm_tolerance = GRASP_CONTACT_POS_TOL
        elif self.phase == "lift":
            arm_tolerance = LIFT_CONTACT_POS_TOL
        elif self.phase in {"transport", "hold"}:
            arm_tolerance = TRANSPORT_CONTACT_POS_TOL
        else:
            arm_tolerance = FEEDBACK_POS_TOL
        position_ok = (
            slide_error <= FEEDBACK_POS_TOL
            and left_arm_max <= arm_tolerance
            and right_arm_max <= arm_tolerance
        )
        stable_now = (
            position_ok
            and command_error <= FEEDBACK_POS_TOL
            and max_velocity <= FEEDBACK_VEL_TOL
        )

        now = time.monotonic()
        if stable_now:
            if self.feedback_stable_since is None:
                self.feedback_stable_since = now
            if now - self.feedback_stable_since >= FEEDBACK_STABLE_TIME:
                self.feedback_stable_since = None
                return True
        else:
            self.feedback_stable_since = None

        if now - self.last_feedback_log_time >= FEEDBACK_LOG_PERIOD:
            self.last_feedback_log_time = now
            self.get_logger().warning(
                f"waiting for measured joints: phase={self.phase}, "
                f"arm_tol={arm_tolerance:.3f}, slide_error={slide_error:.3f}, "
                f"left_arm_max={left_arm_max:.3f}, "
                f"right_arm_max={right_arm_max:.3f}, "
                f"max_velocity={max_velocity:.4f}, "
                f"command_error={command_error:.3f}"
            )
        return False

    def start_compliant_squeeze(self):
        self.phase = "squeeze"
        self.squeeze_planned = False
        self.squeeze_baseline_effort = self.arm_effort_meas.copy()
        self.squeeze_next_time = time.monotonic()
        self.get_logger().info(
            "Partial gripper contact completed; starting bounded arm preload: "
            f"step={self.squeeze_step:.3f} m, max={self.max_squeeze:.3f} m, "
            f"effort_delta_limit={self.effort_delta_limit:.2f}."
        )

    def begin_gripper_hold(self):
        self.phase = "gripper_hold"
        self.tc[11] = GRIPPER_HOLD
        self.tc[18] = GRIPPER_HOLD
        self.gripper_hold_ready_time = time.monotonic() + GRIPPER_HOLD_SETTLE_TIME
        self.get_logger().info(
            f"Grasp pose reached; keeping both grippers open at {GRIPPER_HOLD:.2f} "
            "so all four fingers remain spread on the box side faces. "
            "Gripper effort is recorded as a transmission diagnostic, not Newton force."
        )

    def current_effort_delta(self):
        if self.squeeze_baseline_effort is None:
            return 0.0
        return float(np.max(np.abs(self.arm_effort_meas - self.squeeze_baseline_effort)))

    def hold_gripper_effort_ratio(self):
        if self.hold_gripper_effort_reference is None:
            return 1.0
        reference = np.asarray(self.hold_gripper_effort_reference, dtype=float)
        valid = reference > 1e-6
        if not np.any(valid):
            return 1.0
        current = self.gripper_effort_meas
        return float(np.min(current[valid] / reference[valid]))

    def begin_lift(self, reason):
        self.hold_gripper_effort_reference = self.gripper_effort_meas.copy()
        self.phase = "lift"
        self.lift_planned = False
        self.get_logger().info(
            f"Compliant squeeze complete ({reason}); holding half_width="
            f"{self.hold_half_width:.3f} m and planning lift pose."
        )

    def plan_squeeze_step(self):
        squeeze_used = self.grasp_half_width - self.hold_half_width
        squeeze_remaining = self.max_squeeze - squeeze_used
        if squeeze_remaining <= 1e-6:
            return False
        step = min(self.squeeze_step, squeeze_remaining)
        target_half_width = self.hold_half_width - step
        ok = self.plan_arm_pose(
            "squeeze",
            self.grasp_backoff_x,
            target_half_width,
            GRIPPER_HOLD,
            z_offset=self.hand_z_offset,
            seed_from_feedback=True,
            center_base=self.hold_center_base,
        )
        if ok:
            self.hold_half_width = target_half_width
            self.squeeze_planned = True
            self.squeeze_next_time = time.monotonic() + COMPLIANT_SQUEEZE_INTERVAL
            self.get_logger().info(
                f"Compliant squeeze step applied; half_width="
                f"{self.hold_half_width:.3f} m."
            )
        return ok

    def plan_hold_retighten(self):
        remaining = HOLD_RETIGHTEN_MAX - self.hold_retighten_used
        if remaining <= 1e-6:
            return False
        step = min(HOLD_RETIGHTEN_STEP, remaining)
        target_half_width = self.hold_half_width - step
        ok = self.plan_arm_pose(
            "hold_retighten",
            self.transport_backoff_x,
            target_half_width,
            GRIPPER_HOLD,
            z_offset=self.hand_z_offset + self.lift_height,
            seed_from_feedback=True,
            center_base=self.hold_center_base,
        )
        if ok:
            self.hold_half_width = target_half_width
            self.hold_retighten_used += step
            self.hold_retighten_planned = True
            self.get_logger().info(
                f"Periodic hold re-tighten applied; half_width={self.hold_half_width:.3f} m."
            )
        return ok

    def tick(self):
        if not self.target_received:
            return
        if self.jpos is None:
            return
        if self.target_frame == "world" and self.base_xy is None:
            return
        if self.auto_nav and not self.nav_done:
            self.nav_done = self.nav_step()
            if not self.nav_done:
                return

        if self.phase == "pregrasp":
            if not self.pregrasp_planned:
                self.pregrasp_planned = self.plan_pregrasp()
                if not self.pregrasp_planned:
                    rclpy.shutdown()
                    return
            self.smooth_and_publish()
            if self.converged():
                if self.do_grasp:
                    self.phase = "grasp"
                    self.get_logger().info("Pre-grasp pose reached; planning grasp pose.")
                elif not self.finished_logged:
                    self.finished_logged = True
                    self.get_logger().info("Pre-grasp pose reached. Press Ctrl+C when done.")
            return

        if self.phase == "grasp":
            if not self.grasp_planned:
                self.grasp_planned = self.plan_grasp()
                if not self.grasp_planned:
                    rclpy.shutdown()
                    return
            self.smooth_and_publish()
            if self.converged() and not self.finished_logged:
                self.begin_gripper_hold()
            return

        if self.phase == "gripper_hold":
            self.smooth_and_publish()
            if time.monotonic() < self.gripper_hold_ready_time:
                return
            efforts = self.gripper_effort_meas
            self.get_logger().info(
                f"gripper transmission effort diagnostic={np.round(efforts, 3)}; "
                f"bounded squeeze={self.max_squeeze:.3f} m."
            )
            if self.compliant_squeeze and self.max_squeeze > 0.0 and self.squeeze_step > 0.0:
                self.start_compliant_squeeze()
            else:
                self.get_logger().warning(
                    "Compliant squeeze disabled; lifting without the anti-slip preload."
                )
                self.begin_lift("compliant squeeze disabled")
            return

        if self.phase == "squeeze":
            effort_delta = self.current_effort_delta()
            if effort_delta >= self.effort_delta_limit:
                self.begin_lift(f"contact effort delta={effort_delta:.2f}")
                return
            if self.squeeze_planned:
                self.smooth_and_publish()
                if self.converged() and time.monotonic() >= self.squeeze_next_time:
                    self.squeeze_planned = False
                return
            if self.grasp_half_width - self.hold_half_width >= self.max_squeeze - 1e-6:
                self.begin_lift("maximum squeeze reached")
                return
            if not self.plan_squeeze_step():
                self.begin_lift("squeeze IK unavailable")
            return

        if self.phase == "lift":
            if not self.lift_planned:
                self.lift_planned = self.plan_lift()
                if not self.lift_planned:
                    rclpy.shutdown()
                    return
            self.smooth_and_publish()
            if self.converged() and not self.finished_logged:
                if ENABLE_TRANSPORT_RETRACT:
                    self.phase = "transport"
                    self.get_logger().info(
                        "Lift pose reached; planning transport retract pose."
                    )
                else:
                    self.finished_logged = True
                    self.phase = "hold"
                    self.hold_next_time = time.monotonic() + HOLD_RETIGHTEN_INTERVAL
                    self.get_logger().info(
                        "Lift pose reached; transport retract disabled. "
                        "Holding the box at the lift pose. Press Ctrl+C when done."
                    )
            return

        if self.phase == "transport":
            if not self.transport_planned:
                self.transport_planned = self.plan_transport()
                if not self.transport_planned:
                    rclpy.shutdown()
                    return
            self.smooth_and_publish()
            if not self.finished_logged and self.converged():
                self.finished_logged = True
                self.phase = "hold"
                self.hold_next_time = time.monotonic() + HOLD_RETIGHTEN_INTERVAL
                self.get_logger().info(
                    "Transport retract pose reached; holding the box at the fixed transport preload. "
                    "Press Ctrl+C when done."
                )
            return

        if self.phase == "hold":
            now = time.monotonic()
            if self.hold_retighten_planned:
                self.smooth_and_publish()
                if self.converged():
                    self.hold_retighten_planned = False
                    self.hold_next_time = now + HOLD_RETIGHTEN_INTERVAL
                return
            effort_ratio = self.hold_gripper_effort_ratio()
            if (
                effort_ratio < HOLD_MIN_EFFORT_RATIO
                and self.hold_retighten_used < HOLD_RETIGHTEN_MAX - 1e-6
            ):
                self.hold_next_time = now
                self.get_logger().warning(
                    f"hold gripper effort ratio={effort_ratio:.2f}; recovering preload."
                )
            self.smooth_and_publish()
            if now < self.hold_next_time:
                return
            if self.hold_retighten_used >= HOLD_RETIGHTEN_MAX - 1e-6:
                return
            if not self.plan_hold_retighten():
                self.hold_next_time = now + HOLD_RETIGHTEN_INTERVAL
            return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=TARGET_WORLD.tolist(),
        help="target box center, unit: meter",
    )
    parser.add_argument(
        "--target-frame",
        choices=("world", "base"),
        default="world",
        help="coordinate frame of --target; use base to bypass world-to-base conversion",
    )
    parser.add_argument(
        "--target-topic",
        default=None,
        help="PointStamped topic containing a world-frame box center; overrides --target",
    )
    parser.add_argument(
        "--no-auto-nav",
        action="store_true",
        help="disable base navigation before IK in world-frame mode",
    )
    parser.add_argument(
        "--stand-off",
        type=float,
        default=DEFAULT_STAND_OFF,
        help="base-to-target distance after auto-nav, unit: meter",
    )
    parser.add_argument(
        "--approach-yaw",
        type=float,
        default=DEFAULT_APPROACH_YAW,
        help="final robot yaw in world frame, rad; default is +90 degrees",
    )
    parser.add_argument(
        "--pregrasp-only",
        action="store_true",
        help="stop at open pre-grasp pose; do not move inward or lift",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=LIFT_HEIGHT,
        help="lift distance after the dual-arm grasp, unit: meter",
    )
    parser.add_argument(
        "--transport-backoff-x",
        type=float,
        default=TRANSPORT_BACKOFF_X,
        help="arm center backoff during transport, unit: meter",
    )
    parser.add_argument(
        "--grasp-half-width",
        type=float,
        default=None,
        help="fallback grasp half-width; detected yaw selects the calibrated value when available",
    )
    parser.add_argument(
        "--target-orientation",
        choices=tuple(BOX_HALF_EXTENTS_BY_ORIENTATION),
        default=None,
        help="optional target yaw label for direct/manual targets (yaw0 or yaw90)",
    )
    parser.add_argument(
        "--hand-z-offset",
        type=float,
        default=HAND_Z_OFFSET,
        help="end-effector height relative to the detected box center, unit: meter",
    )
    parser.add_argument(
        "--grasp-backoff-x",
        type=float,
        default=GRASP_BACKOFF_X,
        help="grasp arm center uses arm_x = box_x - value, unit: meter",
    )
    squeeze_group = parser.add_mutually_exclusive_group()
    squeeze_group.add_argument(
        "--compliant-squeeze",
        dest="compliant_squeeze",
        action="store_true",
        help="enable the gradual post-grasp squeeze stage",
    )
    squeeze_group.add_argument(
        "--no-compliant-squeeze",
        dest="compliant_squeeze",
        action="store_false",
        help="skip the gradual post-grasp squeeze stage",
    )
    parser.set_defaults(compliant_squeeze=DEFAULT_COMPLIANT_SQUEEZE)
    parser.add_argument(
        "--squeeze-step",
        type=float,
        default=COMPLIANT_SQUEEZE_STEP,
        help="inward distance per compliant squeeze step, unit: meter",
    )
    parser.add_argument(
        "--max-squeeze",
        type=float,
        default=COMPLIANT_MAX_SQUEEZE,
        help="maximum total inward squeeze after grasp, unit: meter",
    )
    parser.add_argument(
        "--effort-delta-limit",
        type=float,
        default=COMPLIANT_EFFORT_DELTA_LIMIT,
        help="joint-effort change that stops additional inward squeezing",
    )
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = ManualDualArmPregrasp(
        np.array(args.target, dtype=float),
        args.target_frame,
        not args.no_auto_nav,
        args.stand_off,
        args.approach_yaw,
        not args.pregrasp_only,
        args.lift_height,
        args.transport_backoff_x,
        args.target_topic,
        args.compliant_squeeze,
        args.squeeze_step,
        args.max_squeeze,
        args.effort_delta_limit,
        GRASP_HALF_WIDTH if args.grasp_half_width is None else args.grasp_half_width,
        args.hand_z_offset,
        args.grasp_backoff_x,
        args.target_orientation,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
