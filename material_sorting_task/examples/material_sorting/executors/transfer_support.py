"""Shared non-ROS base-motion helpers for pick/transport/place executors."""

from __future__ import annotations

import math
from typing import Any

from navigation.navigation_controller import NavigationController
from navigation.navigation_types import NavigationGoal, NavigationStatus, SpeedLimits
from navigation.occupancy_grid import build_material_scene_grid


def odometry_pose(odometry: Any) -> tuple[float, float, float] | None:
    if odometry is None:
        return None
    try:
        position = odometry.pose.pose.position
        orientation = odometry.pose.pose.orientation
        x = float(position.x)
        y = float(position.y)
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)
    except (AttributeError, TypeError, ValueError):
        return None
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        return None
    return x, y, yaw


def world_to_base(
    point_world: tuple[float, float, float],
    robot_pose: tuple[float, float, float],
) -> tuple[float, float, float]:
    robot_x, robot_y, robot_yaw = robot_pose
    dx = float(point_world[0]) - robot_x
    dy = float(point_world[1]) - robot_y
    c = math.cos(-robot_yaw)
    s = math.sin(-robot_yaw)
    return (
        c * dx - s * dy,
        s * dx + c * dy,
        float(point_world[2]),
    )


def stand_from_held_center(
    place_world: tuple[float, float, float],
    held_center_base: tuple[float, float, float],
    place_yaw: float,
) -> tuple[float, float]:
    """Return the base XY that maps the held center onto ``place_world``."""

    held_x, held_y = held_center_base[:2]
    c = math.cos(float(place_yaw))
    s = math.sin(float(place_yaw))
    held_world_x = c * held_x - s * held_y
    held_world_y = s * held_x + c * held_y
    return (
        float(place_world[0]) - held_world_x,
        float(place_world[1]) - held_world_y,
    )


class TransferMotion:
    """Collision-aware navigation plus short straight-line retreat motions."""

    def __init__(self) -> None:
        limits = SpeedLimits(
            max_linear=0.18,
            max_angular=0.55,
            max_linear_accel=0.30,
            max_angular_accel=1.0,
            emergency_clearance=0.20,
            max_deceleration=0.50,
        )
        self._navigation = NavigationController(
            build_material_scene_grid(),
            limits,
            pos_tolerance=0.07,
            yaw_tolerance=0.06,
            lookahead_distance=0.40,
            timeout=90.0,
            emergency_distance=0.20,
        )
        self._goal: NavigationGoal | None = None
        self._last_tick_s: float | None = None
        self._retreat_start: tuple[float, float, float] | None = None
        self._retreat_distance_m = 0.0

    @property
    def goal(self) -> NavigationGoal | None:
        return self._goal

    def reset(self) -> None:
        self._navigation.reset()
        self._goal = None
        self._last_tick_s = None
        self._retreat_start = None
        self._retreat_distance_m = 0.0

    def begin_navigation(self, goal: NavigationGoal, odometry: Any) -> bool:
        pose = odometry_pose(odometry)
        if pose is None:
            return False
        self._navigation.reset()
        self._goal = goal
        self._last_tick_s = None
        return self._navigation.set_goal(goal, pose[0], pose[1])

    def tick_navigation(
        self,
        odometry: Any,
        now_s: float,
    ) -> tuple[NavigationStatus, tuple[float, float], str]:
        pose = odometry_pose(odometry)
        if pose is None:
            return NavigationStatus.IDLE, (0.0, 0.0), "waiting for valid odometry"
        if self._goal is None:
            return NavigationStatus.FAILED, (0.0, 0.0), "navigation goal was not started"
        now = float(now_s)
        dt = 0.05 if self._last_tick_s is None else min(
            0.20, max(0.01, now - self._last_tick_s)
        )
        self._last_tick_s = now
        command = self._navigation.update(*pose, dt, obs=None)
        status = self._navigation.status
        detail = (
            f"goal=({self._goal.x:.2f}, {self._goal.y:.2f}, "
            f"{self._goal.yaw:.2f}); nav_status={status.value}"
        )
        return status, (command.linear_x, command.angular_z), detail

    def begin_retreat(self, odometry: Any, distance_m: float) -> bool:
        pose = odometry_pose(odometry)
        distance = float(distance_m)
        if pose is None or not math.isfinite(distance) or distance <= 0.0:
            return False
        self._retreat_start = pose
        self._retreat_distance_m = distance
        return True

    def tick_retreat(self, odometry: Any) -> tuple[bool, tuple[float, float], str]:
        pose = odometry_pose(odometry)
        if pose is None:
            return False, (0.0, 0.0), "waiting for valid odometry"
        if self._retreat_start is None:
            return False, (0.0, 0.0), "retreat was not started"
        start_x, start_y, yaw_ref = self._retreat_start
        dx = pose[0] - start_x
        dy = pose[1] - start_y
        reversed_m = -(dx * math.cos(yaw_ref) + dy * math.sin(yaw_ref))
        remaining = self._retreat_distance_m - reversed_m
        if remaining <= 0.015:
            return True, (0.0, 0.0), f"retreat complete ({reversed_m:.3f} m)"
        yaw_error = math.atan2(
            math.sin(yaw_ref - pose[2]),
            math.cos(yaw_ref - pose[2]),
        )
        linear = -min(0.12, max(0.04, 0.65 * remaining))
        angular = max(-0.25, min(0.25, 1.0 * yaw_error))
        return (
            False,
            (linear, angular),
            f"retreating straight; remaining={max(0.0, remaining):.3f} m",
        )


__all__ = [
    "TransferMotion",
    "odometry_pose",
    "stand_from_held_center",
    "world_to_base",
]
