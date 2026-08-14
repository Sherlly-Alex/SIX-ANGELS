#!/usr/bin/env python3
"""Formal ROS 2 entry point for the DG-202612 competition client.

This file owns process lifecycle and the public ROS interface.  The pure
``CompetitionController`` schedules task 1, task 2, and task 3; task-specific
motion remains behind the executor interfaces in ``executors/``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
import math
import os
from statistics import median

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, Int32, String
from vision_msgs.msg import Detection3DArray

from competition_controller import (
    CompetitionController,
    ControllerState,
    ExecutionContext,
)
from executors import build_task_executors
from executors.base import ArmCommand, TargetObservation, TaskStage
from desktop_grasp.target_metadata import dominant_orientation, infer_box_orientation
from instruction_parser import (
    InstructionParseError,
    InstructionValidationError,
    parse_instruction_message,
    validate_instruction,
)
from task_orchestration import parse_gameinfo, sorted_instructions


class ClientPhase(Enum):
    WAITING_FOR_SERVER = auto()
    READY = auto()
    RUNNING = auto()
    FINISHED = auto()
    SAFE_HOLD = auto()


class CompetitionClient(Node):
    """Own the formal subscriptions and keep the robot safe during startup."""

    MAX_BASE_LINEAR_MPS = 0.22
    MAX_BASE_ANGULAR_RADPS = 0.70

    def __init__(self) -> None:
        super().__init__("six_angels_material_sorting_client")

        self.phase = ClientPhase.WAITING_FOR_SERVER
        self.instructions: list[dict] = []
        self.odom_received = False
        self.joints_received = False
        self.latest_odometry: Odometry | None = None
        self.latest_joint_states: JointState | None = None
        self.referee_taskinfo = ""
        self.referee_gameinfo: dict = {}
        self.score = 0
        self._target_histories: dict[
            str,
            deque[tuple[float, float, float]],
        ] = {}
        self._target_orientation_histories: dict[
            str,
            deque[str | None],
        ] = {}
        self._target_quality_histories: dict[
            str,
            deque[str | None],
        ] = {}
        self.target_observations: dict[str, TargetObservation] = {}
        self.grasp_confirmed = False
        self.unsafe_collision = False
        self._last_wait_log_ns = 0
        self._last_progress_log_ns = 0
        self._last_controller_serial = -1
        self._last_task2_detection_reset_key: tuple[int, int, str] | None = None
        self._last_task3_detection_reset_key: tuple[int, int, str] | None = None

        self.execution_mode = (
            os.environ.get("MATERIAL_EXECUTION_MODE", "stub").strip().lower()
        )
        try:
            dry_run_ticks = int(
                os.environ.get("MATERIAL_DRY_RUN_TICKS_PER_STAGE", "2")
            )
            executors = build_task_executors(
                self.execution_mode,
                dry_run_ticks_per_stage=dry_run_ticks,
            )
        except (TypeError, ValueError) as exc:
            self.execution_mode = "stub"
            executors = build_task_executors("stub")
            self.get_logger().error(
                f"invalid executor configuration ({exc}); falling back to safe stub mode"
            )
        self.controller = CompetitionController(
            executors,
            referee_driven=self.execution_mode != "dry_run",
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.shelf_empty_check_pub = self.create_publisher(
            Bool, "/material/shelf_recognition_enable", 10
        )
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
            String, "/material/instruction", self._instruction_cb, 5
        )
        self.create_subscription(
            String, "/referee/taskinfo", self._taskinfo_cb, 5
        )
        self.create_subscription(
            String, "/referee/gameinfo", self._gameinfo_cb, 5
        )
        self.create_subscription(Int32, "/referee/score", self._score_cb, 5)
        self.create_subscription(
            Bool,
            "/material/grasp_confirmed",
            self._grasp_confirmed_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/material/unsafe_collision",
            self._unsafe_collision_cb,
            10,
        )
        self.create_subscription(
            Odometry, "/slamware_ros_sdk_server_node/odom", self._odom_cb, 10
        )
        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        self.create_subscription(
            Detection3DArray,
            "/material/detections",
            self._detections_cb,
            10,
        )

        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            "client started; waiting for instruction, odometry and joint states; "
            f"execution_mode={self.execution_mode}"
        )
        if self.execution_mode == "dry_run":
            self.get_logger().warning(
                "dry_run is scheduling-only: all three tasks advance without robot motion "
                "or Server scoring"
            )
        elif self.execution_mode == "nav_only":
            self.get_logger().warning(
                "nav_only enables real task-1 base motion to the detected table-side "
                "pick stand, then stops and blocks before arm motion"
            )
        elif self.execution_mode == "pregrasp_only":
            self.get_logger().warning(
                "pregrasp_only enables task-1 base navigation and real dual-arm "
                "open pregrasp, then holds and blocks before inward grasp contact"
            )
        elif self.execution_mode == "contact_only":
            self.get_logger().warning(
                "contact_only enables task-1 navigation, open pregrasp and "
                "bilateral target contact, then holds and blocks before "
                "squeeze or lift"
            )
        elif self.execution_mode == "lift_only":
            self.get_logger().warning(
                "lift_only enables task-1 navigation, open pregrasp, the "
                "bounded 4 mm arm preload and a 0.15 m slide lift without "
                "requiring Server bilateral-contact confirmation; it then "
                "holds and blocks before transport"
            )
        elif self.execution_mode == "task12_full":
            self.get_logger().warning(
                "task12_full enables integrated task-1 table pick, stable "
                "shelf-state recognition and empty-layer placement, followed by "
                "task-2 shelf pick and return to task-1's saved table origin; "
                "task 3 remains fail-closed"
            )
        elif self.execution_mode == "task123_full":
            self.get_logger().warning(
                "task123_full enables integrated task-1/task-2 execution and "
                "task-3 top-box pick, measured packaging-box left placement, "
                "and safe return; task transitions remain referee-driven"
            )
        else:
            self.get_logger().info(
                "formal mode is referee-driven; placeholder executors fail closed and keep "
                "the robot stopped until real task actions are connected"
            )

    def _instruction_cb(self, msg: String) -> None:
        try:
            parsed = parse_instruction_message(msg.data)
            for task in parsed:
                validate_instruction(task, require_execution_ready=True)
            instructions = sorted_instructions([task.to_dict() for task in parsed])
            task_ids = [task.get("task") for task in instructions]
            if task_ids != [1, 2, 3]:
                raise InstructionValidationError(
                    f"expected tasks [1, 2, 3], received {task_ids}"
                )
            instructions_changed = self.controller.configure(instructions)
        except (
            InstructionParseError,
            InstructionValidationError,
            RuntimeError,
            ValueError,
        ) as exc:
            self.phase = ClientPhase.SAFE_HOLD
            self.controller.stop(f"instruction rejected: {exc}")
            self.get_logger().error(f"instruction rejected: {exc}")
            return

        self.instructions = instructions
        if instructions_changed:
            self._last_task2_detection_reset_key = None
            self._last_task3_detection_reset_key = None
            self.get_logger().info(
                "instructions accepted: "
                + ", ".join(
                    f"T{task['task']}={task['target_color']}->{task['place_type']}"
                    for task in instructions
                )
            )

    def _taskinfo_cb(self, msg: String) -> None:
        self.referee_taskinfo = msg.data

    def _gameinfo_cb(self, msg: String) -> None:
        self.referee_gameinfo = parse_gameinfo(msg.data)

    def _score_cb(self, msg: Int32) -> None:
        self.score = int(msg.data)

    def _grasp_confirmed_cb(self, msg: Bool) -> None:
        self.grasp_confirmed = bool(msg.data)

    def _unsafe_collision_cb(self, msg: Bool) -> None:
        self.unsafe_collision = bool(msg.data)

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odometry = msg
        self.odom_received = True

    def _joints_cb(self, msg: JointState) -> None:
        self.latest_joint_states = msg
        self.joints_received = True

    def _detections_cb(self, msg: Detection3DArray) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        # Preserve the RGB-D frame timestamp.  Using callback time made old
        # points look fresh after the robot entered task 2, even while the
        # rolling median still contained pre-staging shelf views.
        try:
            header_stamp = (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
            )
        except (AttributeError, TypeError, ValueError):
            header_stamp = 0.0
        received_at_s = (
            header_stamp
            if math.isfinite(header_stamp)
            and header_stamp > 0.0
            and abs(now_s - header_stamp) <= 30.0
            else now_s
        )
        for detection in msg.detections:
            try:
                if not detection.results:
                    continue
                result = max(
                    detection.results,
                    key=lambda item: float(item.hypothesis.score),
                )
                color = str(result.hypothesis.class_id).strip().lower()
                score = float(result.hypothesis.score)
                position = result.pose.pose.position
                point = (
                    float(position.x),
                    float(position.y),
                    float(position.z),
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                # One malformed detector result must never take down the
                # command publisher or leave the previous velocity latched.
                continue
            if color not in {
                "pink",
                "yellow",
                "brown",
                "material_box",
                "packaging_box",
                "shelf_obstacle",
                "shelf_empty",
            }:
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            history = self._target_histories.setdefault(color, deque(maxlen=7))
            history.append(point)
            try:
                bbox_orientation = detection.bbox.center.orientation
                orientation = infer_box_orientation(
                    detection.bbox.size.x,
                    detection.bbox.size.y,
                    bbox_orientation.z,
                    bbox_orientation.w,
                )
            except (AttributeError, TypeError, ValueError):
                # Position-only detections are still useful for navigation.
                # A missing/malformed bbox orientation must not interrupt the
                # ROS callback; the pregrasp planner will use its safe yaw-0
                # fallback for that observation.
                orientation = None
            try:
                bbox_size = (
                    float(detection.bbox.size.x),
                    float(detection.bbox.size.y),
                    float(detection.bbox.size.z),
                )
            except (AttributeError, TypeError, ValueError):
                bbox_size = (0.0, 0.0, 0.0)
            full_fit = (
                orientation in {"yaw0", "yaw90"}
                and all(math.isfinite(value) and value > 1e-4 for value in bbox_size)
            )
            quality = "mask_cloud_cuboid" if full_fit else "bbox_depth_center"
            orientation_history = self._target_orientation_histories.setdefault(
                color,
                deque(maxlen=7),
            )
            orientation_history.append(orientation)
            quality_history = self._target_quality_histories.setdefault(
                color,
                deque(maxlen=7),
            )
            quality_history.append(quality)
            if len(history) < 3:
                continue
            samples = tuple(history)
            stable_position = tuple(
                float(median(sample[axis] for sample in samples))
                for axis in range(3)
            )
            quality_values = tuple(quality_history)
            if quality_values and all(value == "mask_cloud_cuboid" for value in quality_values):
                stable_quality = "mask_cloud_cuboid"
            elif quality_values and all(value == "bbox_depth_center" for value in quality_values):
                stable_quality = "bbox_depth_center"
            else:
                # Do not let a median assembled from mixed full-fit and
                # surface-fallback frames pass the task-2 quality gate.
                stable_quality = "mixed"
            self.target_observations[color] = TargetObservation(
                color=color,
                position_world=stable_position,
                received_at_s=received_at_s,
                orientation=dominant_orientation(orientation_history),
                score=score,
                quality=stable_quality,
            )

    def _reset_target_histories(self, colors: list[str]) -> None:
        """Drop pre-task-2 frames so a new shelf lock starts from fresh RGB-D data."""

        for color in colors:
            self._target_histories.pop(color, None)
            self._target_orientation_histories.pop(color, None)
            self._target_quality_histories.pop(color, None)
            self.target_observations.pop(color, None)

    def _refresh_task2_detection_epoch(self) -> None:
        """Clear the target colour once when task 2 starts its arm staging.

        Task 1 and the early task-2 navigation intentionally share the same
        detector stream.  Without an epoch boundary, the seven-sample client
        median can combine old table/shelf-edge frames with the final shelf
        view, making a biased but apparently stable centre.
        """

        if self.controller.task_index != 1 or self.controller.stage is not TaskStage.ALIGN_FOR_PICK:
            return
        target_color = (
            str(self.instructions[1].get("target_color", "")).strip().lower()
            if len(self.instructions) > 1
            else ""
        )
        if not target_color:
            return
        key = (int(self.controller.task_index), int(self.controller.attempt), target_color)
        if key == self._last_task2_detection_reset_key:
            return
        self._reset_target_histories([target_color])
        self._last_task2_detection_reset_key = key
        self.get_logger().info(
            f"task 2 detection epoch reset for {target_color}; waiting for fresh RGB-D frames"
        )

    def _refresh_task3_detection_epoch(self) -> None:
        """Mark task 3 start without discarding its target observations.

        The task-3 top box stays in the scene while tasks 1 and 2 run.  Its
        rolling RGB-D median is therefore useful as soon as the robot turns
        back toward the table.  Unlike task 2's shelf target, do not clear
        this colour at the task boundary; the task-3 executor keeps fusing
        the retained observation and any newer frames.
        """

        if self.controller.task_index != 2 or self.controller.stage not in {
            TaskStage.NAVIGATE_TO_PICK,
            TaskStage.ACQUIRE_TARGET,
        }:
            return
        target_color = (
            str(self.instructions[2].get("target_color", "")).strip().lower()
            if len(self.instructions) > 2
            else ""
        )
        if not target_color:
            return
        key = (int(self.controller.task_index), int(self.controller.attempt), target_color)
        if key == self._last_task3_detection_reset_key:
            return
        self._last_task3_detection_reset_key = key
        self.get_logger().info(
            f"task 3 retaining existing {target_color} RGB-D history; "
            "continuing centre tracking from navigation"
        )

    def _publish_base_command(self, linear_x: float, angular_z: float) -> None:
        if not rclpy.ok():
            return
        linear = float(linear_x)
        angular = float(angular_z)
        if not (math.isfinite(linear) and math.isfinite(angular)):
            linear = 0.0
            angular = 0.0
        linear = max(-self.MAX_BASE_LINEAR_MPS, min(self.MAX_BASE_LINEAR_MPS, linear))
        angular = max(
            -self.MAX_BASE_ANGULAR_RADPS,
            min(self.MAX_BASE_ANGULAR_RADPS, angular),
        )
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        try:
            self.cmd_vel_pub.publish(command)
        except Exception:
            # Ctrl+C can invalidate the context between the check above and
            # publish().  Suppress only that shutdown race; preserve genuine
            # publishing errors while ROS is still active.
            if rclpy.ok():
                raise

    def _publish_stop(self) -> None:
        self._publish_base_command(0.0, 0.0)

    def _publish_shelf_empty_check(self, enabled: bool) -> None:
        if not rclpy.ok():
            return
        try:
            self.shelf_empty_check_pub.publish(Bool(data=bool(enabled)))
        except Exception:
            if rclpy.ok():
                raise

    def _publish_arm_command(self, command: ArmCommand) -> None:
        if not rclpy.ok():
            return
        values = (
            command.spine_position,
            *command.head_positions,
            *command.left_arm_positions,
            command.left_gripper_position,
            *command.right_arm_positions,
            command.right_gripper_position,
        )
        if len(command.head_positions) != 2:
            raise ValueError("ArmCommand head_positions must contain 2 values")
        if len(command.left_arm_positions) != 6:
            raise ValueError("ArmCommand left_arm_positions must contain 6 values")
        if len(command.right_arm_positions) != 6:
            raise ValueError("ArmCommand right_arm_positions must contain 6 values")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("ArmCommand contains non-finite values")
        try:
            self.spine_pub.publish(
                Float64MultiArray(data=[float(command.spine_position)])
            )
            self.head_pub.publish(
                Float64MultiArray(data=[float(value) for value in command.head_positions])
            )
            self.left_arm_pub.publish(
                Float64MultiArray(
                    data=[
                        *[float(value) for value in command.left_arm_positions],
                        float(command.left_gripper_position),
                    ]
                )
            )
            self.right_arm_pub.publish(
                Float64MultiArray(
                    data=[
                        *[float(value) for value in command.right_arm_positions],
                        float(command.right_gripper_position),
                    ]
                )
            )
        except Exception:
            if rclpy.ok():
                raise

    def _missing_inputs(self) -> list[str]:
        missing = []
        if len(self.instructions) != 3:
            missing.append("instructions")
        if not self.odom_received:
            missing.append("odometry")
        if not self.joints_received:
            missing.append("joint_states")
        if self.execution_mode in {
            "nav_only",
            "pregrasp_only",
            "contact_only",
            "lift_only",
            "task12_full",
            "task123_full",
        } and self.instructions:
            target_color = (
                str(self.instructions[0].get("target_color", "")).strip().lower()
            )
            if target_color not in self.target_observations:
                missing.append(f"detection:{target_color}")
        return missing

    def tick(self) -> None:
        """Feed ROS observations into the non-blocking competition controller."""
        if self.phase in (ClientPhase.SAFE_HOLD, ClientPhase.FINISHED):
            self._publish_stop()
            self._publish_shelf_empty_check(False)
            snapshot = self.controller.snapshot()
            if snapshot.controls_arm and snapshot.arm_command is not None:
                self._publish_arm_command(snapshot.arm_command)
            return

        self._refresh_task2_detection_epoch()
        self._refresh_task3_detection_epoch()
        missing = self._missing_inputs()
        self.controller.set_inputs_ready(not missing)
        if missing:
            self._publish_stop()
            self._publish_shelf_empty_check(False)
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_wait_log_ns >= 5_000_000_000:
                self.get_logger().info("waiting for: " + ", ".join(missing))
                self._last_wait_log_ns = now_ns
            return

        if self.phase is ClientPhase.WAITING_FOR_SERVER:
            self.phase = ClientPhase.READY
            self.get_logger().info(
                "client inputs ready; starting three-task competition controller"
            )

        now_s = self.get_clock().now().nanoseconds * 1e-9
        task_index = min(
            self.controller.task_index,
            max(0, len(self.instructions) - 1),
        )
        instruction = self.instructions[task_index] if self.instructions else {}
        snapshot = self.controller.tick(
            ExecutionContext(
                now_s=now_s,
                instruction=instruction,
                task_index=task_index,
                attempt=self.controller.attempt,
                odometry=self.latest_odometry,
                joint_states=self.latest_joint_states,
                target_observations=dict(self.target_observations),
                referee_gameinfo=self.referee_gameinfo,
                referee_taskinfo=self.referee_taskinfo,
                score=self.score,
                grasp_confirmed=self.grasp_confirmed,
                unsafe_collision=self.unsafe_collision,
            )
        )

        self._publish_shelf_empty_check(
            snapshot.requests_shelf_recognition
        )

        if snapshot.controls_base:
            self._publish_base_command(
                snapshot.base_linear_x,
                snapshot.base_angular_z,
            )
        else:
            self._publish_stop()
        if snapshot.controls_arm and snapshot.arm_command is not None:
            self._publish_arm_command(snapshot.arm_command)

        if snapshot.transition_serial != self._last_controller_serial:
            stage = snapshot.stage.value if snapshot.stage is not None else "-"
            line = (
                f"controller={snapshot.state.value} task={snapshot.task_id} "
                f"attempt={snapshot.attempt} stage={stage} score={self.score}: "
                f"{snapshot.message}"
            )
            if snapshot.state is ControllerState.BLOCKED:
                self.get_logger().warning(line)
            else:
                self.get_logger().info(line)
            self._last_controller_serial = snapshot.transition_serial
            self._last_progress_log_ns = int(now_s * 1e9)
        elif (
            snapshot.state is ControllerState.EXECUTING_STAGE
            and (snapshot.controls_base or snapshot.controls_arm)
            and int(now_s * 1e9) - self._last_progress_log_ns >= 2_000_000_000
        ):
            stage = snapshot.stage.value if snapshot.stage is not None else "-"
            self.get_logger().info(
                f"progress task={snapshot.task_id} stage={stage}: {snapshot.message}"
            )
            self._last_progress_log_ns = int(now_s * 1e9)

        if snapshot.state is ControllerState.FINISHED:
            self.phase = ClientPhase.FINISHED
        elif snapshot.state is ControllerState.SAFE_HOLD:
            self.phase = ClientPhase.SAFE_HOLD
        else:
            self.phase = ClientPhase.RUNNING

    def stop(self) -> None:
        self.controller.stop("client shutdown")
        self.phase = ClientPhase.SAFE_HOLD
        self._publish_stop()
        self._publish_shelf_empty_check(False)
        snapshot = self.controller.snapshot()
        if snapshot.controls_arm and snapshot.arm_command is not None:
            self._publish_arm_command(snapshot.arm_command)


def main() -> None:
    rclpy.init()
    node = CompetitionClient()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
