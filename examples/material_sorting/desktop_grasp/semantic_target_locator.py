#!/usr/bin/env python3
"""Select one semantic target from RGB-D detections and publish its world center.

Inputs:
  /material/instruction  std_msgs/String, structured JSON from the server
  /material/detections   vision_msgs/Detection3DArray, frame_id=world
  /material/current_task std_msgs/Int32, optional runtime task switch

Outputs:
  /material/target_world geometry_msgs/PointStamped, frame_id=world
  /material/target_info  std_msgs/String, JSON diagnostics
"""

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection3DArray


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from instruction_parser import InstructionParseError, parse_instruction_message
from desktop_grasp.target_metadata import dominant_orientation, infer_box_orientation


TABLE_BOX_CENTER_Z = 0.739 + 0.095
TABLE_TOP_BOX_CENTER_Z = 1.004
SHELF_BOX_CENTER_Z = np.array([0.403, 0.732, 1.061], dtype=float) + 0.095 + 0.010

SOURCE_ROI = {
    1: ((-1.35, 0.15), (1.75, 2.60), (0.60, 1.20)),
    2: ((-3.10, -2.20), (0.30, 1.30), (0.30, 1.40)),
    3: ((-0.90, -0.20), (1.90, 2.60), (0.80, 1.30)),
}


def point_in_roi(point, roi):
    return all(lo <= value <= hi for value, (lo, hi) in zip(point, roi))


class SemanticTargetLocator(Node):
    def __init__(
        self,
        task_id,
        sample_count,
        min_score,
        max_deviation,
        target_topic,
        use_z_snap,
        use_source_roi,
        task3_x_offset,
        task3_y_offset,
        task1_side_offset,
        task1_center_y,
    ):
        super().__init__("semantic_target_locator")
        self.task_id = int(task_id)
        self.sample_count = max(3, int(sample_count))
        self.min_score = float(min_score)
        self.max_deviation = float(max_deviation)
        self.use_z_snap = bool(use_z_snap)
        self.use_source_roi = bool(use_source_roi)
        self.task3_x_offset = float(task3_x_offset)
        self.task3_y_offset = float(task3_y_offset)
        self.task1_side_offset = float(task1_side_offset)
        self.task1_center_y = None if task1_center_y is None else float(task1_center_y)

        self.tasks = {}
        self.selected_task = None
        self.instruction_signature = None
        self.samples = deque(maxlen=max(30, self.sample_count * 3))
        self.orientation_samples = deque(maxlen=max(30, self.sample_count * 3))
        self.locked_point = None
        self.locked_info = None
        self.last_wait_log = -1.0

        target_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_pub = self.create_publisher(PointStamped, target_topic, target_qos)
        self.info_pub = self.create_publisher(String, "/material/target_info", target_qos)
        self.create_subscription(String, "/material/instruction", self.instruction_cb, 5)
        self.create_subscription(Detection3DArray, "/material/detections", self.detections_cb, 10)
        self.create_subscription(Int32, "/material/current_task", self.current_task_cb, 5)
        self.create_timer(0.25, self.republish_locked_target)

        self.get_logger().info(
            f"semantic_target_locator up; task={self.task_id}; samples={self.sample_count}; "
            f"min_score={self.min_score:.2f}; max_deviation={self.max_deviation:.3f}; "
            f"z_snap={self.use_z_snap}; source_roi={self.use_source_roi}; "
            f"task1_side_offset={self.task1_side_offset:.3f}; "
            f"task1_center_y={self.task1_center_y}; "
            f"task3_x_offset={self.task3_x_offset:+.3f}; "
            f"task3_y_offset={self.task3_y_offset:+.3f}; "
            f"target_topic={target_topic}"
        )

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def reset_target(self, reason):
        self.samples.clear()
        self.orientation_samples.clear()
        self.locked_point = None
        self.locked_info = None
        self.get_logger().info(f"target reset: {reason}")

    def current_task_cb(self, msg):
        task_id = int(msg.data)
        if task_id not in (1, 2, 3):
            self.get_logger().warn(f"ignoring invalid task id: {task_id}")
            return
        if task_id != self.task_id:
            self.task_id = task_id
            self.selected_task = self.tasks.get(task_id)
            self.reset_target(f"current task changed to {task_id}")
            self.log_selected_task()

    def instruction_cb(self, msg):
        try:
            parsed = parse_instruction_message(msg.data)
        except InstructionParseError as exc:
            self.get_logger().error(f"instruction parse failed: {exc}")
            return

        tasks = {task.task: task for task in parsed if task.task is not None}
        signature = tuple(
            (task_id, task.target_color, task.target_body, task.place_world)
            for task_id, task in sorted(tasks.items())
        )
        self.tasks = tasks
        self.selected_task = self.tasks.get(self.task_id)
        if signature != self.instruction_signature:
            self.instruction_signature = signature
            self.reset_target("new server instruction set")
            self.log_selected_task()

    def log_selected_task(self):
        if self.selected_task is None:
            self.get_logger().warn(f"task {self.task_id} is not present in /material/instruction")
            return
        self.get_logger().info(
            f"task={self.task_id} target_color={self.selected_task.target_color} "
            f"target_body={self.selected_task.target_body} "
            f"place_world={self.selected_task.place_world}"
        )

    def detections_cb(self, msg):
        if self.locked_point is not None or self.selected_task is None:
            return
        if msg.header.frame_id and msg.header.frame_id != "world":
            self.get_logger().warn(
                f"ignoring detections in frame {msg.header.frame_id!r}; expected 'world'"
            )
            return

        wanted_color = self.selected_task.target_color
        candidates = []
        for detection in msg.detections:
            if not detection.results:
                continue
            result = max(detection.results, key=lambda item: float(item.hypothesis.score))
            if result.hypothesis.class_id.strip().lower() != wanted_color:
                continue
            score = float(result.hypothesis.score)
            if score < self.min_score:
                continue
            position = result.pose.pose.position
            point = np.array([position.x, position.y, position.z], dtype=float)
            if not np.all(np.isfinite(point)):
                continue
            if self.use_source_roi and not point_in_roi(point, SOURCE_ROI[self.task_id]):
                continue
            box_orientation = detection.bbox.center.orientation
            orientation = infer_box_orientation(
                detection.bbox.size.x,
                detection.bbox.size.y,
                box_orientation.z,
                box_orientation.w,
            )
            candidates.append((score, point, orientation))

        if not candidates:
            self.log_waiting(wanted_color)
            return

        _, best_point, best_orientation = max(candidates, key=lambda item: item[0])
        self.samples.append(best_point)
        self.orientation_samples.append(best_orientation)
        self.try_lock_target(msg)

    def log_waiting(self, wanted_color):
        now = self.now()
        if now - self.last_wait_log >= 2.0:
            self.get_logger().info(
                f"waiting for target detection: task={self.task_id} color={wanted_color} "
                f"samples={len(self.samples)}/{self.sample_count}"
            )
            self.last_wait_log = now

    def try_lock_target(self, detection_msg):
        if len(self.samples) < self.sample_count:
            return

        points = np.asarray(list(self.samples)[-self.sample_count:], dtype=float)
        median = np.median(points, axis=0)
        residual = np.max(np.abs(points - median), axis=1)
        inliers = points[residual <= self.max_deviation]
        required_inliers = max(3, int(np.ceil(0.75 * self.sample_count)))
        if len(inliers) < required_inliers:
            self.log_unstable(points, median, len(inliers), required_inliers)
            return

        center = np.median(inliers, axis=0)
        axis_deviation = np.max(np.abs(inliers - center), axis=0)
        if float(np.max(axis_deviation)) > self.max_deviation:
            self.log_unstable(points, center, len(inliers), required_inliers)
            return

        raw_center = center.copy()
        orientation = dominant_orientation(
            list(self.orientation_samples)[-self.sample_count:]
        )
        if self.use_z_snap:
            center[2] = self.snap_center_z(center[2])
        center_offset = np.zeros(3, dtype=float)
        if self.task_id == 1:
            instruction = self.selected_task.instruction
            if "\u5de6\u4fa7" in instruction:
                center_offset[0] = -self.task1_side_offset
            elif "\u53f3\u4fa7" in instruction:
                center_offset[0] = self.task1_side_offset
            else:
                center_offset[0] = (
                    -self.task1_side_offset if center[0] < -0.59
                    else self.task1_side_offset
                )
            if self.task1_center_y is not None:
                center_offset[1] = self.task1_center_y - center[1]
        elif self.task_id == 3:
            center_offset[0] = self.task3_x_offset
            center_offset[1] = self.task3_y_offset
        center += center_offset

        self.locked_point = center
        self.locked_info = {
            "task": self.task_id,
            "target_color": self.selected_task.target_color,
            "target_body": self.selected_task.target_body,
            "target_world": [float(value) for value in center],
            "raw_world_median": [float(value) for value in raw_center],
            "place_world": self.selected_task.place_world,
            "sample_count": int(len(inliers)),
            "max_axis_deviation": [float(value) for value in axis_deviation],
            "z_snapped": self.use_z_snap,
            "center_offset": [float(value) for value in center_offset],
            "orientation": orientation,
            "task3_x_offset": self.task3_x_offset if self.task_id == 3 else 0.0,
            "task3_y_offset": self.task3_y_offset if self.task_id == 3 else 0.0,
        }
        self.publish_target(detection_msg.header.stamp)
        self.get_logger().info(
            f"target locked: task={self.task_id} color={self.selected_task.target_color} "
            f"raw={np.round(raw_center, 3)} center={np.round(center, 3)} "
            f"deviation={np.round(axis_deviation, 4)} n={len(inliers)}"
        )

    def log_unstable(self, points, center, inlier_count, required_inliers):
        now = self.now()
        if now - self.last_wait_log >= 2.0:
            deviation = np.max(np.abs(points - center), axis=0)
            self.get_logger().warn(
                f"target not stable: samples={len(points)} inliers={inlier_count}/"
                f"{required_inliers} axis_deviation={np.round(deviation, 4)}"
            )
            self.last_wait_log = now

    def snap_center_z(self, measured_z):
        if self.task_id == 1:
            return TABLE_BOX_CENTER_Z
        if self.task_id == 3:
            return TABLE_TOP_BOX_CENTER_Z
        return float(SHELF_BOX_CENTER_Z[np.argmin(np.abs(SHELF_BOX_CENTER_Z - measured_z))])

    def publish_target(self, stamp=None):
        if self.locked_point is None:
            return
        msg = PointStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.point.x = float(self.locked_point[0])
        msg.point.y = float(self.locked_point[1])
        msg.point.z = float(self.locked_point[2])
        self.target_pub.publish(msg)
        self.info_pub.publish(String(data=json.dumps(self.locked_info, ensure_ascii=False)))

    def republish_locked_target(self):
        if self.locked_point is not None:
            self.publish_target()


def parse_args():
    parser = argparse.ArgumentParser(description="semantic RGB-D target selector")
    parser.add_argument("--task", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--max-deviation", type=float, default=0.05, help="meters")
    parser.add_argument("--target-topic", default="/material/target_world")
    parser.add_argument("--task1-side-offset", type=float, default=0.0, help="meters")
    parser.add_argument("--task1-center-y", type=float, default=None,
                        help="optional legacy world-Y calibration; omit for RGB-D center output")
    parser.add_argument("--task3-x-offset", type=float, default=0.0, help="meters")
    parser.add_argument("--task3-y-offset", type=float, default=0.0, help="meters")
    parser.add_argument("--no-z-snap", action="store_true")
    parser.add_argument("--no-source-roi", action="store_true")
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = SemanticTargetLocator(
        task_id=args.task,
        sample_count=args.samples,
        min_score=args.min_score,
        max_deviation=args.max_deviation,
        target_topic=args.target_topic,
        use_z_snap=not args.no_z_snap,
        use_source_roi=not args.no_source_roi,
        task3_x_offset=args.task3_x_offset,
        task3_y_offset=args.task3_y_offset,
        task1_side_offset=args.task1_side_offset,
        task1_center_y=args.task1_center_y,
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
