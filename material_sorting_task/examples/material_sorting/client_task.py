#!/usr/bin/env python3
"""Formal ROS 2 entry point for the DG-202612 competition client.

This file owns process lifecycle and the public ROS interface. Motion planning
and manipulation should be added behind ``CompetitionClient.tick`` while the
topic names and startup behavior remain stable.
"""

from __future__ import annotations

from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String

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

    def __init__(self) -> None:
        super().__init__("six_angels_material_sorting_client")

        self.phase = ClientPhase.WAITING_FOR_SERVER
        self.instructions: list[dict] = []
        self.odom_received = False
        self.joints_received = False
        self.referee_taskinfo = ""
        self.referee_gameinfo: dict = {}
        self.score = 0
        self._last_wait_log_ns = 0

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

        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            "client started; waiting for instruction, odometry and joint states"
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
        except (InstructionParseError, InstructionValidationError) as exc:
            self.phase = ClientPhase.SAFE_HOLD
            self.get_logger().error(f"instruction rejected: {exc}")
            return

        self.instructions = instructions
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

    def _odom_cb(self, _msg: Odometry) -> None:
        self.odom_received = True

    def _joints_cb(self, _msg: JointState) -> None:
        self.joints_received = True

    def _publish_stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def _missing_inputs(self) -> list[str]:
        missing = []
        if len(self.instructions) != 3:
            missing.append("instructions")
        if not self.odom_received:
            missing.append("odometry")
        if not self.joints_received:
            missing.append("joint_states")
        return missing

    def tick(self) -> None:
        """Advance the top-level state machine without moving before readiness."""
        if self.phase in (ClientPhase.SAFE_HOLD, ClientPhase.FINISHED):
            self._publish_stop()
            return

        missing = self._missing_inputs()
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
                "client inputs ready; manipulation state machine is the next implementation step"
            )

        # Keep a safe stationary command until navigation/manipulation is connected.
        self._publish_stop()

    def stop(self) -> None:
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
