"""Move a robot that is already carrying a box to the shelf approach pose.

The node deliberately controls only ``/cmd_vel``. It does not publish arm or
gripper commands. Use it only when another controller continuously republishes
the cached grasp command; stopping the grasp publisher without a handoff can
release the preload that keeps the box held.

Default route:
1. Reverse 0.35 m along the current robot heading.
2. Drive to 1.5 m east of the shelf front edge.
3. Rotate to face the shelf, which is west in the world frame.

Example:

    python3 examples/material_sorting/desktop_grasp/manual_dual_arm_to_shelf.py

The shelf coordinates can be overridden when the scene layout changes.
"""

import argparse
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation


SHELF_FRONT_X = -2.47
SHELF_CENTER_Y = 0.778
SHELF_DISTANCE = 1.50
BACKOFF_DISTANCE = 0.35

POSITION_TOLERANCE = 0.04
YAW_TOLERANCE = 0.03
BACKOFF_SPEED = 0.12
NAV_MAX_LINEAR = 0.20
NAV_MAX_ANGULAR = 0.65
YAW_KP = 1.8
BACKOFF_YAW_KP = 2.0


def normalize_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class ManualDualArmToShelf(Node):
    """Navigate the mobile base while leaving the carried grasp untouched."""

    def __init__(
        self,
        shelf_front_x,
        shelf_y,
        shelf_distance,
        backoff_distance,
        position_tolerance,
        yaw_tolerance,
    ):
        super().__init__("manual_dual_arm_to_shelf")
        self.shelf_front_x = float(shelf_front_x)
        self.shelf_y = float(shelf_y)
        self.shelf_distance = max(float(shelf_distance), 0.0)
        self.backoff_distance = max(float(backoff_distance), 0.0)
        self.position_tolerance = max(float(position_tolerance), 0.005)
        self.yaw_tolerance = max(float(yaw_tolerance), 0.005)
        self.shelf_goal_xy = np.array(
            [self.shelf_front_x + self.shelf_distance, self.shelf_y],
            dtype=float,
        )
        self.shelf_goal_yaw = math.pi

        self.base_xy = None
        self.base_yaw = None
        self.phase = "wait_odom"
        self.backoff_start_xy = None
        self.backoff_heading = None
        self.last_log_time = 0.0

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.create_subscription(
            Odometry,
            "/slamware_ros_sdk_server_node/odom",
            self.odom_cb,
            10,
        )
        self.timer = self.create_timer(1.0 / 24.0, self.tick)
        self.get_logger().info(
            "manual_dual_arm_to_shelf up; waiting for odometry. "
            f"shelf_goal=({self.shelf_goal_xy[0]:.3f}, "
            f"{self.shelf_goal_xy[1]:.3f}), yaw=pi, "
            f"backoff={self.backoff_distance:.2f} m"
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

    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        self.publish_cmd_vel(0.0, 0.0)

    def start_backoff(self):
        self.backoff_start_xy = self.base_xy.copy()
        self.backoff_heading = float(self.base_yaw)
        if self.backoff_distance <= self.position_tolerance:
            self.phase = "transit"
            self.get_logger().info("backoff skipped; starting shelf transit.")
            return
        self.phase = "backoff"
        self.get_logger().info(
            f"backoff started from ({self.base_xy[0]:.3f}, "
            f"{self.base_xy[1]:.3f}), heading={self.backoff_heading:.3f} rad"
        )

    def backoff_step(self):
        forward = np.array([
            math.cos(self.backoff_heading),
            math.sin(self.backoff_heading),
        ])
        displacement = self.base_xy - self.backoff_start_xy
        progress = float(np.dot(-displacement, forward))
        heading_error = normalize_angle(self.backoff_heading - self.base_yaw)
        if progress < self.backoff_distance - self.position_tolerance:
            angular_z = float(np.clip(
                BACKOFF_YAW_KP * heading_error,
                -NAV_MAX_ANGULAR,
                NAV_MAX_ANGULAR,
            ))
            self.publish_cmd_vel(-BACKOFF_SPEED, angular_z)
            return

        self.stop_robot()
        self.phase = "transit"
        self.get_logger().info(
            f"backoff complete: moved {progress:.3f} m; "
            f"starting transit to ({self.shelf_goal_xy[0]:.3f}, "
            f"{self.shelf_goal_xy[1]:.3f})"
        )

    def transit_step(self):
        delta = self.shelf_goal_xy - self.base_xy
        distance = float(np.linalg.norm(delta))
        if distance > self.position_tolerance:
            target_yaw = math.atan2(delta[1], delta[0])
            yaw_error = normalize_angle(target_yaw - self.base_yaw)
            linear_x = (
                min(NAV_MAX_LINEAR, 0.55 * distance)
                if abs(yaw_error) < 0.55
                else 0.0
            )
            angular_z = float(np.clip(
                YAW_KP * yaw_error,
                -NAV_MAX_ANGULAR,
                NAV_MAX_ANGULAR,
            ))
            self.publish_cmd_vel(linear_x, angular_z)
            return

        self.stop_robot()
        self.phase = "align"
        self.get_logger().info(
            f"shelf position reached: base=({self.base_xy[0]:.3f}, "
            f"{self.base_xy[1]:.3f}); aligning to shelf"
        )

    def align_step(self):
        yaw_error = normalize_angle(self.shelf_goal_yaw - self.base_yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            angular_z = float(np.clip(
                YAW_KP * yaw_error,
                -NAV_MAX_ANGULAR,
                NAV_MAX_ANGULAR,
            ))
            self.publish_cmd_vel(0.0, angular_z)
            return

        self.stop_robot()
        self.phase = "done"
        self.get_logger().info(
            f"shelf approach complete: base=({self.base_xy[0]:.3f}, "
            f"{self.base_xy[1]:.3f}), yaw={self.base_yaw:.3f} rad; "
            "base motion complete; the external hold controller must remain active"
        )

    def tick(self):
        if self.base_xy is None or self.base_yaw is None:
            self.stop_robot()
            return
        if self.phase == "wait_odom":
            self.start_backoff()
        elif self.phase == "backoff":
            self.backoff_step()
        elif self.phase == "transit":
            self.transit_step()
        elif self.phase == "align":
            self.align_step()
        else:
            self.stop_robot()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Back off, drive to the shelf front, and face the shelf."
    )
    parser.add_argument(
        "--shelf-front-x",
        type=float,
        default=SHELF_FRONT_X,
        help="world x of the shelf front edge, unit: meter",
    )
    parser.add_argument(
        "--shelf-y",
        type=float,
        default=SHELF_CENTER_Y,
        help="world y of the shelf center line, unit: meter",
    )
    parser.add_argument(
        "--shelf-distance",
        type=float,
        default=SHELF_DISTANCE,
        help="distance from shelf front edge to robot base, unit: meter",
    )
    parser.add_argument(
        "--backoff-distance",
        type=float,
        default=BACKOFF_DISTANCE,
        help="initial reverse distance after grasp, unit: meter",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=POSITION_TOLERANCE,
        help="position arrival tolerance, unit: meter",
    )
    parser.add_argument(
        "--yaw-tolerance",
        type=float,
        default=YAW_TOLERANCE,
        help="final heading tolerance, unit: rad",
    )
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = ManualDualArmToShelf(
        args.shelf_front_x,
        args.shelf_y,
        args.shelf_distance,
        args.backoff_distance,
        args.position_tolerance,
        args.yaw_tolerance,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
