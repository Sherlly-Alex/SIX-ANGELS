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
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection3DArray

from competition_controller import (
    CompetitionController,
    ControllerState,
    ExecutionContext,
)
from executors import build_task_executors
from executors.base import TargetObservation
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
        self.target_observations: dict[str, TargetObservation] = {}
        self._last_wait_log_ns = 0
        self._last_controller_serial = -1

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

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odometry = msg
        self.odom_received = True

    def _joints_cb(self, msg: JointState) -> None:
        self.latest_joint_states = msg
        self.joints_received = True

    def _detections_cb(self, msg: Detection3DArray) -> None:
        received_at_s = self.get_clock().now().nanoseconds * 1e-9
        for detection in msg.detections:
            try:
                if not detection.results:
                    continue
                result = detection.results[0]
                color = str(result.hypothesis.class_id).strip().lower()
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
            if color not in {"pink", "yellow", "brown"}:
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            history = self._target_histories.setdefault(color, deque(maxlen=7))
            history.append(point)
            if len(history) < 3:
                continue
            samples = tuple(history)
            stable_position = tuple(
                float(median(sample[axis] for sample in samples))
                for axis in range(3)
            )
            self.target_observations[color] = TargetObservation(
                color=color,
                position_world=stable_position,
                received_at_s=received_at_s,
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

    def _missing_inputs(self) -> list[str]:
        missing = []
        if len(self.instructions) != 3:
            missing.append("instructions")
        if not self.odom_received:
            missing.append("odometry")
        if not self.joints_received:
            missing.append("joint_states")
        if self.execution_mode == "nav_only" and self.instructions:
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
            return

        missing = self._missing_inputs()
        self.controller.set_inputs_ready(not missing)
        if missing:
            self._publish_stop()
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
            )
        )

        if snapshot.controls_base:
            self._publish_base_command(
                snapshot.base_linear_x,
                snapshot.base_angular_z,
            )
        else:
            self._publish_stop()

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
