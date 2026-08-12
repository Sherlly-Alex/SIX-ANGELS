#!/usr/bin/env python3
"""Measure acceleration during the three official placement events.

The monitor is read-only.  It samples /joint_states and uses the official
/referee/gameinfo ``step=place`` event to close a rolling placement window.
Because the referee confirms after the physical release, the default 45 s
window contains the preceding placement motion without requiring private
Server topics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


ARM_JOINTS = tuple(
    f"{side}_arm_joint{index}"
    for side in ("left", "right")
    for index in range(1, 7)
)


class PlacementAccelerationMonitor(Node):
    def __init__(self, window_s: float) -> None:
        super().__init__("placement_acceleration_monitor")
        self.window_s = float(window_s)
        self.previous_t: float | None = None
        self.previous_v: dict[str, float] = {}
        self.history: deque[tuple[float, dict[str, float]]] = deque()
        self.results: dict[int, list[float]] = {}
        self.closed_tasks: set[int] = set()
        self.last_event: tuple[int, str] | None = None
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 30)
        self.create_subscription(String, "/referee/gameinfo", self._gameinfo_callback, 10)
        self.create_timer(1.0, self._trim_history)

    @staticmethod
    def _time(msg: JointState) -> float:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        return stamp if stamp > 0.0 else time.monotonic()

    def _joint_callback(self, msg: JointState) -> None:
        now = self._time(msg)
        velocity = {
            str(name): float(value)
            for name, value in zip(msg.name, msg.velocity)
            if str(name) in ARM_JOINTS and math.isfinite(float(value))
        }
        acceleration: dict[str, float] = {}
        if self.previous_t is not None:
            dt = now - self.previous_t
            if 0.001 <= dt <= 0.2:
                for name, value in velocity.items():
                    if name in self.previous_v:
                        value_a = abs((value - self.previous_v[name]) / dt)
                        if math.isfinite(value_a) and value_a <= 100.0:
                            acceleration[name] = value_a
        if acceleration:
            self.history.append((now, acceleration))
        self.previous_t = now
        self.previous_v = velocity
        self._trim_history()

    def _trim_history(self) -> None:
        if self.history:
            newest = self.history[-1][0]
            while self.history and newest - self.history[0][0] > self.window_s:
                self.history.popleft()

    def _gameinfo_callback(self, msg: String) -> None:
        text = str(msg.data)
        task_match = re.search(r"(?:task|任务|浠诲姟)\s*[:=]?\s*(\d+)", text, re.I)
        step_match = re.search(r"(?:step|步骤|姝ラ)\s*[:=]?\s*([A-Za-z_]+)", text, re.I)
        if not task_match or not step_match:
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                return
            task_match = payload.get("task") or payload.get("task_ordinal")
            step_match = payload.get("step")
            task = int(task_match) if task_match is not None else -1
            step = str(step_match or "").lower()
        else:
            task = int(task_match.group(1))
            step = step_match.group(1).lower()
        event = (task, step)
        if event == self.last_event or step != "place" or task in self.closed_tasks:
            self.last_event = event
            return
        self.last_event = event
        values: dict[str, list[float]] = defaultdict(list)
        for _, sample in self.history:
            for name, acceleration in sample.items():
                values[name].append(acceleration)
        self.results[task] = [value for samples in values.values() for value in samples]
        self.closed_tasks.add(task)
        print(f"[PLACEMENT_ACCEL] task={task} step=place window={self.window_s:.1f}s", flush=True)
        self._print_task(task, values)

    @staticmethod
    def _p95(values: list[float]) -> float:
        values = sorted(values)
        return values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else 0.0

    @staticmethod
    def _p995(values: list[float]) -> float:
        values = sorted(values)
        return values[min(len(values) - 1, math.ceil(0.995 * len(values)) - 1)] if values else 0.0

    def _print_task(self, task: int, values: dict[str, list[float]]) -> None:
        print(f"=== 任务{task}放置加速度（rad/s²）===")
        all_values: list[float] = []
        for name in ARM_JOINTS:
            samples = values.get(name, [])
            if not samples:
                continue
            all_values.extend(samples)
            print(
                f"{name}: mean={sum(samples)/len(samples):.4f}, "
                f"P95={self._p95(samples):.4f}, "
                f"max_filtered(P99.5)={self._p995(samples):.4f}, n={len(samples)}"
            )
        if all_values:
            print(f"任务{task}双臂最大P95: {self._p95(all_values):.4f} rad/s²")
        else:
            print("未获得该任务的有效双臂加速度样本。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=120.0, help="每个任务回看窗口（秒）；覆盖任务3推送后确认延迟")
    args = parser.parse_args()
    rclpy.init()
    node = PlacementAccelerationMonitor(args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("监测结束；已输出已确认的任务放置数据。")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
