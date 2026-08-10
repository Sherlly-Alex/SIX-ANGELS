"""Shared non-ROS base-motion helpers for pick/transport/place executors."""

from __future__ import annotations

import math
from typing import Any, Mapping

from navigation.competition_adapter import (
    format_nav_telemetry,
    goal_reached_event,
    refresh_dynamic_overlay,
)
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import NavigationGoal, NavigationStatus, SpeedLimits
from navigation.occupancy_grid import build_layered_scene_grid
from navigation.robot_geometry import FootprintMode


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

    LATERAL_POSITION_TOLERANCE_M = 0.035
    LATERAL_X_TOLERANCE_M = 0.18
    LATERAL_YAW_TOLERANCE_RAD = 0.06
    LATERAL_TIMEOUT_S = 30.0

    def __init__(self, speed_limits: SpeedLimits | None = None) -> None:
        limits = speed_limits or SpeedLimits(
            max_linear=0.18,
            max_angular=0.55,
            max_linear_accel=0.30,
            max_angular_accel=1.0,
            emergency_clearance=0.20,
            max_deceleration=0.50,
        )
        self._navigation_grid = build_layered_scene_grid()
        self._navigation = NavigationController(
            self._navigation_grid,
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
        self._advance_start: tuple[float, float, float] | None = None
        self._advance_distance_m = 0.0
        self._lateral_target: tuple[float, float] | None = None
        self._lateral_final_yaw = 0.0
        self._lateral_heading = 0.0
        self._lateral_phase = "idle"
        self._lateral_start_s: float | None = None
        self._lateral_position_tolerance_m = self.LATERAL_POSITION_TOLERANCE_M
        self._lateral_yaw_tolerance_rad = self.LATERAL_YAW_TOLERANCE_RAD
        self._lateral_timeout_s = self.LATERAL_TIMEOUT_S

    @property
    def goal(self) -> NavigationGoal | None:
        return self._goal

    @property
    def navigation_path(self) -> tuple[tuple[float, float], ...]:
        return self._navigation.path

    def reset(self) -> None:
        self._navigation.reset()
        self._goal = None
        self._last_tick_s = None
        self._retreat_start = None
        self._retreat_distance_m = 0.0
        self._advance_start = None
        self._advance_distance_m = 0.0
        self._lateral_target = None
        self._lateral_final_yaw = 0.0
        self._lateral_heading = 0.0
        self._lateral_phase = "idle"
        self._lateral_start_s = None
        self._lateral_position_tolerance_m = self.LATERAL_POSITION_TOLERANCE_M
        self._lateral_yaw_tolerance_rad = self.LATERAL_YAW_TOLERANCE_RAD
        self._lateral_timeout_s = self.LATERAL_TIMEOUT_S

    def begin_navigation(
        self,
        goal: NavigationGoal,
        odometry: Any,
        *,
        footprint_mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        observations: Mapping[str, Any] | None = None,
        exclude_color: str | None = None,
        payload_z: float | None = None,
    ) -> bool:
        pose = odometry_pose(odometry)
        if pose is None:
            return False
        self._navigation.reset()
        refresh_dynamic_overlay(
            self._navigation_grid,
            observations,
            exclude_color=exclude_color,
            robot_xy=(pose[0], pose[1]),
        )
        self._navigation.set_footprint_mode(
            footprint_mode,
            payload_z=payload_z,
        )
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
            f"{self._goal.yaw:.2f}); nav_status={status.value}; "
            f"{format_nav_telemetry(self._navigation.telemetry, phase='transfer')}"
        )
        if status is NavigationStatus.GOAL_REACHED:
            detail = f"{detail}; {goal_reached_event(self._goal)}"
        return status, (command.linear_x, command.angular_z), detail

    def begin_lateral_alignment(
        self,
        target_xy: tuple[float, float],
        final_yaw: float,
        odometry: Any,
        now_s: float,
        *,
        position_tolerance_m: float | None = None,
        yaw_tolerance_rad: float | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Start a bounded shelf-front lateral alignment.

        The generic planner is deliberately not used here.  The robot is
        already at a safe shelf-front x coordinate, so a full path-to-pose
        request can repeatedly turn while trying to satisfy a tiny y offset
        and the final shelf-facing yaw at the same time.  This helper makes
        the motion explicit: rotate in place toward the lateral direction,
        drive only along y while staying at the shelf-front x, then rotate in
        place to the final shelf-facing yaw.
        """

        pose = odometry_pose(odometry)
        try:
            target_x = float(target_xy[0])
            target_y = float(target_xy[1])
            target_yaw = _wrap_to_pi(float(final_yaw))
            start_s = float(now_s)
            tolerance = (
                self.LATERAL_POSITION_TOLERANCE_M
                if position_tolerance_m is None
                else float(position_tolerance_m)
            )
            yaw_tolerance = (
                self.LATERAL_YAW_TOLERANCE_RAD
                if yaw_tolerance_rad is None
                else float(yaw_tolerance_rad)
            )
            timeout = (
                self.LATERAL_TIMEOUT_S
                if timeout_s is None
                else float(timeout_s)
            )
        except (TypeError, ValueError, IndexError):
            return False
        if (
            pose is None
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not math.isfinite(yaw_tolerance)
            or yaw_tolerance <= 0.0
            or not math.isfinite(timeout)
            or timeout <= 0.0
            or not all(
                math.isfinite(value)
                for value in (target_x, target_y, target_yaw, start_s)
            )
        ):
            return False
        if abs(target_x - pose[0]) > self.LATERAL_X_TOLERANCE_M:
            # This controller intentionally does not make a forward/backward
            # correction near the shelf.  Let the caller fail safely instead
            # of silently moving the carried object into the shelf front.
            return False

        self._lateral_target = (target_x, target_y)
        self._lateral_final_yaw = target_yaw
        self._lateral_position_tolerance_m = tolerance
        self._lateral_yaw_tolerance_rad = yaw_tolerance
        self._lateral_timeout_s = timeout
        self._lateral_heading = (
            math.pi / 2.0 if target_y >= pose[1] else -math.pi / 2.0
        )
        self._lateral_phase = (
            "rotate_final"
            if abs(target_y - pose[1]) <= self._lateral_position_tolerance_m
            else "rotate_lateral"
        )
        self._lateral_start_s = start_s
        return True

    def tick_lateral_alignment(
        self,
        odometry: Any,
        now_s: float,
    ) -> tuple[NavigationStatus, tuple[float, float], str]:
        """Advance the bounded shelf-front lateral alignment by one tick."""

        pose = odometry_pose(odometry)
        if pose is None:
            return NavigationStatus.NAVIGATING, (0.0, 0.0), (
                "lateral alignment waiting for valid odometry"
            )
        if self._lateral_target is None or self._lateral_start_s is None:
            return NavigationStatus.FAILED, (0.0, 0.0), (
                "lateral alignment was not started"
            )
        elapsed = max(0.0, float(now_s) - self._lateral_start_s)
        if elapsed > self._lateral_timeout_s:
            self._lateral_phase = "failed"
            return NavigationStatus.FAILED, (0.0, 0.0), (
                f"lateral alignment timed out after {elapsed:.1f}s "
                f"(limit={self._lateral_timeout_s:.1f}s)"
            )

        target_x, target_y = self._lateral_target
        if self._lateral_phase == "rotate_lateral":
            yaw_error = _wrap_to_pi(self._lateral_heading - pose[2])
            if abs(yaw_error) <= self._lateral_yaw_tolerance_rad:
                self._lateral_phase = "drive_lateral"
            else:
                angular = max(-0.35, min(0.35, 1.4 * yaw_error))
                return NavigationStatus.NAVIGATING, (0.0, angular), (
                    "lateral alignment rotating toward shelf-front direction; "
                    f"yaw_err={yaw_error:.3f}"
                )

        if self._lateral_phase == "drive_lateral":
            y_error = target_y - pose[1]
            if abs(y_error) <= self._lateral_position_tolerance_m:
                self._lateral_phase = "rotate_final"
            else:
                # Keep x close to the recorded shelf-front line by applying a
                # small heading correction while moving along y.  Do not move
                # forward until the lateral heading is reasonably aligned.
                x_error = target_x - pose[0]
                heading_offset = max(
                    -0.25,
                    min(0.25, -math.sin(self._lateral_heading) * 0.8 * x_error),
                )
                desired_yaw = self._lateral_heading + heading_offset
                yaw_error = _wrap_to_pi(desired_yaw - pose[2])
                angular = max(-0.30, min(0.30, 1.2 * yaw_error))
                linear = min(0.09, max(0.035, 0.55 * abs(y_error)))
                if abs(yaw_error) > 0.18:
                    linear = 0.0
                return NavigationStatus.NAVIGATING, (linear, angular), (
                    "lateral alignment driving along shelf front; "
                    f"y_err={y_error:.3f}, x_err={x_error:.3f}"
                )

        if self._lateral_phase == "rotate_final":
            yaw_error = _wrap_to_pi(self._lateral_final_yaw - pose[2])
            if abs(yaw_error) <= self._lateral_yaw_tolerance_rad:
                self._lateral_phase = "done"
                return NavigationStatus.GOAL_REACHED, (0.0, 0.0), (
                    "lateral alignment complete; shelf-facing yaw restored"
                )
            angular = max(-0.35, min(0.35, 1.4 * yaw_error))
            return NavigationStatus.NAVIGATING, (0.0, angular), (
                "lateral alignment restoring shelf-facing yaw; "
                f"yaw_err={yaw_error:.3f}"
            )

        if self._lateral_phase == "done":
            return NavigationStatus.GOAL_REACHED, (0.0, 0.0), (
                "lateral alignment complete"
            )
        return NavigationStatus.FAILED, (0.0, 0.0), "lateral alignment failed"

    def begin_retreat(
        self,
        odometry: Any,
        distance_m: float,
        *,
        heading_yaw: float | None = None,
    ) -> bool:
        """Start a reverse segment and optionally hold an explicit heading."""

        pose = odometry_pose(odometry)
        distance = float(distance_m)
        yaw_ref = pose[2] if pose is not None and heading_yaw is None else heading_yaw
        if (
            pose is None
            or not math.isfinite(distance)
            or distance <= 0.0
            or yaw_ref is None
            or not math.isfinite(float(yaw_ref))
        ):
            return False
        self._retreat_start = (pose[0], pose[1], float(yaw_ref))
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
        if abs(yaw_error) > 0.08:
            linear = 0.0
        return (
            False,
            (linear, angular),
            f"retreating straight; remaining={max(0.0, remaining):.3f} m",
        )

    def begin_advance(
        self,
        odometry: Any,
        distance_m: float,
        *,
        heading_yaw: float | None = None,
    ) -> bool:
        """Start a short forward motion at the current or explicit heading."""

        pose = odometry_pose(odometry)
        distance = float(distance_m)
        yaw_ref = pose[2] if pose is not None and heading_yaw is None else heading_yaw
        if (
            pose is None
            or not math.isfinite(distance)
            or distance <= 0.0
            or yaw_ref is None
            or not math.isfinite(float(yaw_ref))
        ):
            return False
        self._advance_start = (pose[0], pose[1], float(yaw_ref))
        self._advance_distance_m = distance
        return True

    def tick_advance(self, odometry: Any) -> tuple[bool, tuple[float, float], str]:
        """Advance along the recorded heading without turning toward a waypoint."""

        pose = odometry_pose(odometry)
        if pose is None:
            return False, (0.0, 0.0), "waiting for valid odometry"
        if self._advance_start is None:
            return False, (0.0, 0.0), "straight advance was not started"
        start_x, start_y, yaw_ref = self._advance_start
        dx = pose[0] - start_x
        dy = pose[1] - start_y
        advanced_m = dx * math.cos(yaw_ref) + dy * math.sin(yaw_ref)
        remaining = self._advance_distance_m - advanced_m
        if remaining <= 0.015:
            return True, (0.0, 0.0), f"straight advance complete ({advanced_m:.3f} m)"
        yaw_error = math.atan2(
            math.sin(yaw_ref - pose[2]),
            math.cos(yaw_ref - pose[2]),
        )
        linear = min(0.10, max(0.035, 0.55 * remaining))
        angular = max(-0.20, min(0.20, 0.8 * yaw_error))
        if abs(yaw_error) > 0.08:
            linear = 0.0
        return (
            False,
            (linear, angular),
            f"advancing straight; remaining={max(0.0, remaining):.3f} m",
        )


__all__ = [
    "TransferMotion",
    "odometry_pose",
    "stand_from_held_center",
    "world_to_base",
]


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
