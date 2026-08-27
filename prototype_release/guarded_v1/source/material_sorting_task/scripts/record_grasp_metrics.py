#!/usr/bin/env python3
"""Record passive compliant-grasp telemetry to CSV.

The recorder publishes no commands.  It combines high-rate ``/joint_states``
samples with state annotations parsed from the competition Client's ``/rosout``
messages.  Effort columns are MuJoCo joint-actuator generalized effort; they are
not fingertip force in newtons.

Run this in a second shell while ``scripts/run_client.sh`` is active::

    python3 scripts/record_grasp_metrics.py --output /tmp/grasp_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
from typing import Any


LEFT_WRIST = "left_arm_joint6"
RIGHT_WRIST = "right_arm_joint6"

CSV_FIELDS = (
    "elapsed_s",
    "ros_time_s",
    "task_id",
    "stage",
    "controller_state",
    "score",
    "left_wrist_position_rad",
    "right_wrist_position_rad",
    "left_wrist_velocity_rad_s",
    "right_wrist_velocity_rad_s",
    "left_wrist_effort",
    "right_wrist_effort",
    "left_baseline_position_rad",
    "right_baseline_position_rad",
    "left_baseline_effort",
    "right_baseline_effort",
    "left_effort_filtered",
    "right_effort_filtered",
    "left_effort_delta",
    "right_effort_delta",
    "left_angle_delta_deg",
    "right_angle_delta_deg",
    "left_effort_threshold",
    "right_effort_threshold",
    "reported_left_contact",
    "reported_right_contact",
    "reported_left_aligned",
    "reported_right_aligned",
    "reported_left_angle_deg",
    "reported_right_angle_deg",
    "reported_left_effort_delta",
    "reported_right_effort_delta",
    "inward_offset_mm",
    "inward_limit_mm",
    "approach_speed_mm_s",
    "retry_count",
)

_STATE_RE = re.compile(
    r"controller=(?P<state>[a-z_]+)\s+task=(?P<task>\d+)\s+"
    r"attempt=\d+\s+stage=(?P<stage>[a-z_\-]+)\s+score=(?P<score>-?\d+)"
)
_PROGRESS_RE = re.compile(
    r"progress\s+task=(?P<task>\d+)\s+stage=(?P<stage>[a-z_]+):"
)
_WRIST_RE = re.compile(
    r"(?P<side>left|right)\[contact=(?P<contact>True|False),\s*"
    r"aligned=(?P<aligned>True|False),\s*angle=(?P<angle>[+\-0-9.eE]+)deg,\s*"
    r"effort_delta=(?P<effort>[+\-0-9.eE]+)"
)
_OFFSET_RE = re.compile(
    r"offset=(?P<offset>[+\-0-9.eE]+)/(?P<limit>[+\-0-9.eE]+)\s*mm"
)
_SPEED_RE = re.compile(r"speed=(?P<speed>[+\-0-9.eE]+)\s*mm/s")
_RETRY_RE = re.compile(r"retry=(?P<retry>\d+)/\d+")


def _blank_or(value: Any) -> Any:
    return "" if value is None else value


class ClientLogState:
    """Latest task/stage plus sparse compliant-controller annotations."""

    def __init__(self) -> None:
        self.task_id = 0
        self.stage = ""
        self.controller_state = ""
        self.score: int | None = None
        self.reset_grasp_annotations()

    def reset_grasp_annotations(self) -> None:
        self.reported_left_contact: bool | None = None
        self.reported_right_contact: bool | None = None
        self.reported_left_aligned: bool | None = None
        self.reported_right_aligned: bool | None = None
        self.reported_left_angle_deg: float | None = None
        self.reported_right_angle_deg: float | None = None
        self.reported_left_effort_delta: float | None = None
        self.reported_right_effort_delta: float | None = None
        self.inward_offset_mm: float | None = None
        self.inward_limit_mm: float | None = None
        self.approach_speed_mm_s: float | None = None
        self.retry_count = 0

    def update(self, message: str) -> bool:
        """Parse one Client log message; return true on grasp-epoch change."""

        previous_epoch = (self.task_id, self.stage)
        state_match = _STATE_RE.search(message)
        if state_match:
            self.controller_state = state_match.group("state")
            self.task_id = int(state_match.group("task"))
            self.stage = state_match.group("stage")
            self.score = int(state_match.group("score"))
        else:
            progress_match = _PROGRESS_RE.search(message)
            if progress_match:
                self.task_id = int(progress_match.group("task"))
                self.stage = progress_match.group("stage")

        epoch_changed = previous_epoch != (self.task_id, self.stage)
        if epoch_changed:
            self.reset_grasp_annotations()

        for match in _WRIST_RE.finditer(message):
            side = match.group("side")
            setattr(self, f"reported_{side}_contact", match.group("contact") == "True")
            setattr(self, f"reported_{side}_aligned", match.group("aligned") == "True")
            setattr(self, f"reported_{side}_angle_deg", float(match.group("angle")))
            setattr(
                self,
                f"reported_{side}_effort_delta",
                float(match.group("effort")),
            )

        offset_match = _OFFSET_RE.search(message)
        if offset_match:
            self.inward_offset_mm = float(offset_match.group("offset"))
            self.inward_limit_mm = float(offset_match.group("limit"))
        speed_match = _SPEED_RE.search(message)
        if speed_match:
            self.approach_speed_mm_s = float(speed_match.group("speed"))
        retry_match = _RETRY_RE.search(message)
        if retry_match:
            self.retry_count = int(retry_match.group("retry"))
        return epoch_changed

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "controller_state": self.controller_state,
            "score": _blank_or(self.score),
            "reported_left_contact": _blank_or(self.reported_left_contact),
            "reported_right_contact": _blank_or(self.reported_right_contact),
            "reported_left_aligned": _blank_or(self.reported_left_aligned),
            "reported_right_aligned": _blank_or(self.reported_right_aligned),
            "reported_left_angle_deg": _blank_or(self.reported_left_angle_deg),
            "reported_right_angle_deg": _blank_or(self.reported_right_angle_deg),
            "reported_left_effort_delta": _blank_or(
                self.reported_left_effort_delta
            ),
            "reported_right_effort_delta": _blank_or(
                self.reported_right_effort_delta
            ),
            "inward_offset_mm": _blank_or(self.inward_offset_mm),
            "inward_limit_mm": _blank_or(self.inward_limit_mm),
            "approach_speed_mm_s": _blank_or(self.approach_speed_mm_s),
            "retry_count": self.retry_count,
        }


class EffortBaseline:
    """Independent high-rate baseline/filter for visualization only."""

    def __init__(
        self,
        baseline_seconds: float = 0.40,
        min_samples: int = 8,
        alpha: float = 0.25,
        minimum_threshold: float = 0.35,
    ) -> None:
        self.baseline_seconds = float(baseline_seconds)
        self.min_samples = int(min_samples)
        self.alpha = float(alpha)
        self.minimum_threshold = float(minimum_threshold)
        self.reset()

    def reset(self) -> None:
        self.positions: dict[str, list[float]] = {LEFT_WRIST: [], RIGHT_WRIST: []}
        self.efforts: dict[str, list[float]] = {LEFT_WRIST: [], RIGHT_WRIST: []}
        self.baseline_position: dict[str, float] = {}
        self.baseline_effort: dict[str, float] = {}
        self.filtered_effort: dict[str, float] = {}
        self.threshold: dict[str, float] = {}
        self.ready = False

    def update(
        self,
        elapsed_s: float,
        positions: dict[str, float],
        efforts: dict[str, float],
    ) -> dict[str, Any]:
        if not self.ready:
            for name in (LEFT_WRIST, RIGHT_WRIST):
                self.positions[name].append(positions[name])
                self.efforts[name].append(efforts[name])
            enough_time = (
                max(len(values) for values in self.efforts.values()) >= 2
                and self.baseline_seconds <= 0.0
            )
            # Sampling begins only after the Client has entered GRASP.  Use a
            # conservative 20 Hz lower bound for the requested baseline window
            # so ROS and simulation clock epochs never have to be mixed here.
            required_for_time = max(2, math.ceil(self.baseline_seconds * 20.0))
            sample_count = min(len(values) for values in self.efforts.values())
            enough_time = enough_time or sample_count >= required_for_time
            enough_samples = sample_count >= self.min_samples
            if enough_time and enough_samples:
                for name in (LEFT_WRIST, RIGHT_WRIST):
                    base_position = statistics.median(self.positions[name])
                    base_effort = statistics.median(self.efforts[name])
                    deviations = [abs(value - base_effort) for value in self.efforts[name]]
                    mad = statistics.median(deviations)
                    self.baseline_position[name] = float(base_position)
                    self.baseline_effort[name] = float(base_effort)
                    self.filtered_effort[name] = float(base_effort)
                    self.threshold[name] = max(
                        self.minimum_threshold,
                        5.0 * 1.4826 * float(mad),
                    )
                self.ready = True

        result: dict[str, Any] = {}
        for prefix, name in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
            if not self.ready:
                for suffix in (
                    "baseline_position_rad",
                    "baseline_effort",
                    "effort_filtered",
                    "effort_delta",
                    "angle_delta_deg",
                    "effort_threshold",
                ):
                    result[f"{prefix}_{suffix}"] = ""
                continue
            filtered = (
                self.alpha * efforts[name]
                + (1.0 - self.alpha) * self.filtered_effort[name]
            )
            self.filtered_effort[name] = filtered
            result[f"{prefix}_baseline_position_rad"] = self.baseline_position[name]
            result[f"{prefix}_baseline_effort"] = self.baseline_effort[name]
            result[f"{prefix}_effort_filtered"] = filtered
            result[f"{prefix}_effort_delta"] = abs(
                filtered - self.baseline_effort[name]
            )
            result[f"{prefix}_angle_delta_deg"] = math.degrees(
                abs(positions[name] - self.baseline_position[name])
            )
            result[f"{prefix}_effort_threshold"] = self.threshold[name]
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/grasp_metrics.csv"),
        help="CSV output path (default: /tmp/grasp_metrics.csv)",
    )
    parser.add_argument("--client-node", default="six_angels_material_sorting_client")
    parser.add_argument("--baseline-seconds", type=float, default=0.40)
    parser.add_argument("--baseline-min-samples", type=int, default=8)
    parser.add_argument("--effort-alpha", type=float, default=0.25)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional recording duration in seconds; 0 records until Ctrl+C",
    )
    return parser


def main() -> int:
    args, ros_args = build_parser().parse_known_args()
    try:
        import rclpy
        from rcl_interfaces.msg import Log
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise SystemExit(
            "ROS 2 Python packages are unavailable; run this script inside "
            f"the offline Client container ({exc})"
        ) from exc

    class GraspMetricsRecorder(Node):
        def __init__(self) -> None:
            super().__init__("grasp_metrics_recorder")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            self._handle = args.output.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
            self._writer.writeheader()
            self._rows = 0
            self._first_time_s: float | None = None
            self._log_state = ClientLogState()
            self._baseline = EffortBaseline(
                args.baseline_seconds,
                args.baseline_min_samples,
                args.effort_alpha,
            )
            self._baseline_epoch: tuple[int, str] | None = None
            self.create_subscription(
                JointState,
                "/joint_states",
                self._joint_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(Log, "/rosout", self._log_callback, 100)
            if args.duration > 0.0:
                self.create_timer(args.duration, self._finish)
            self.get_logger().info(
                f"passive grasp metrics recording to {args.output}; "
                "effort is joint-actuator generalized effort, not fingertip force"
            )

        def _now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _finish(self) -> None:
            self.get_logger().info("requested metrics duration completed")
            rclpy.shutdown()

        def _log_callback(self, msg: Any) -> None:
            if args.client_node and args.client_node not in str(msg.name):
                return
            self._log_state.update(str(msg.msg))

        @staticmethod
        def _map(names: tuple[str, ...], values: Any) -> dict[str, float]:
            try:
                converted = tuple(float(value) for value in values)
            except (TypeError, ValueError):
                return {}
            return {
                name: converted[index]
                for index, name in enumerate(names)
                if index < len(converted) and math.isfinite(converted[index])
            }

        def _joint_callback(self, msg: Any) -> None:
            names = tuple(str(name) for name in msg.name)
            positions = self._map(names, msg.position)
            velocities = self._map(names, msg.velocity)
            efforts = self._map(names, msg.effort)
            if not all(
                name in positions and name in efforts
                for name in (LEFT_WRIST, RIGHT_WRIST)
            ):
                return

            now_s = self._now_s()
            if self._first_time_s is None:
                self._first_time_s = now_s
            elapsed_s = now_s - self._first_time_s
            epoch = (self._log_state.task_id, self._log_state.stage)
            if epoch != self._baseline_epoch:
                self._baseline.reset()
                self._baseline_epoch = epoch

            derived: dict[str, Any] = {}
            if self._log_state.stage == "grasp":
                derived = self._baseline.update(elapsed_s, positions, efforts)

            stamp = msg.header.stamp
            ros_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
            row.update(self._log_state.as_row())
            row.update(derived)
            row.update(
                {
                    "elapsed_s": elapsed_s,
                    "ros_time_s": ros_time_s,
                    "left_wrist_position_rad": positions[LEFT_WRIST],
                    "right_wrist_position_rad": positions[RIGHT_WRIST],
                    "left_wrist_velocity_rad_s": velocities.get(LEFT_WRIST, ""),
                    "right_wrist_velocity_rad_s": velocities.get(RIGHT_WRIST, ""),
                    "left_wrist_effort": efforts[LEFT_WRIST],
                    "right_wrist_effort": efforts[RIGHT_WRIST],
                }
            )
            self._writer.writerow(row)
            self._rows += 1
            if self._rows % 20 == 0:
                self._handle.flush()

        def close(self) -> None:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()
                self.get_logger().info(
                    f"saved {self._rows} joint samples to {args.output}"
                )

    rclpy.init(args=ros_args)
    node = GraspMetricsRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
