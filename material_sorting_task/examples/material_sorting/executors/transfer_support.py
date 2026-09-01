"""Shared non-ROS base-motion helpers for pick/transport/place executors."""

from __future__ import annotations

import math
from typing import Any, Mapping

from navigation.carried_envelope import CarriedEnvelopeChecker, HeldObjectGeometry
from navigation.competition_adapter import (
    format_nav_telemetry,
    goal_reached_event,
    refresh_dynamic_overlay,
)
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import NavigationGoal, NavigationStatus, SpeedLimits
from navigation.occupancy_grid import LayeredGrid, build_layered_scene_grid
from navigation.robot_geometry import FootprintMode
from navigation.task3_lateral_safety import (
    Task3LateralGuardParams,
    guard_task3_lateral_cmd,
)


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
    # Task-1 guided shelf approach. Limited to the small terminal error left
    # by direct A*: larger errors keep the established lateral fallback.
    GUIDED_MAX_LATERAL_M = 0.055
    GUIDED_MAX_INITIAL_YAW_ERROR_RAD = 0.06
    GUIDED_POSITION_TOLERANCE_M = 0.015
    GUIDED_YAW_TOLERANCE_RAD = 0.06
    GUIDED_TIMEOUT_S = 35.0
    GUIDED_GATE_BUFFER_M = 0.05
    GUIDED_MAX_YAW_OFFSET_RAD = 0.20
    GUIDED_MAX_LINEAR_MPS = 0.085
    GUIDED_MAX_ANGULAR_RPS = 0.24
    GUIDED_CROSS_TRACK_GAIN = 4.0

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
        self._guided_start: tuple[float, float, float] | None = None
        self._guided_target: tuple[float, float] | None = None
        self._guided_final_yaw = 0.0
        self._guided_forward_m = 0.0
        self._guided_lateral_m = 0.0
        self._guided_start_s: float | None = None
        self._guided_held_geometry: HeldObjectGeometry | None = None
        self._guided_min_clearance_m: float | None = None
        self._lateral_target: tuple[float, float] | None = None
        self._lateral_final_yaw = 0.0
        self._lateral_heading = 0.0
        self._lateral_travel_heading = 0.0
        self._lateral_drive_sign = 1.0
        self._lateral_phase = "idle"
        self._lateral_start_s: float | None = None
        self._lateral_position_tolerance_m = self.LATERAL_POSITION_TOLERANCE_M
        self._lateral_yaw_tolerance_rad = self.LATERAL_YAW_TOLERANCE_RAD
        self._lateral_timeout_s = self.LATERAL_TIMEOUT_S
        # Measured carried-object envelope for A* transport segments.
        # None keeps the historical generic TRANSIT_CARRY behaviour.
        self._held_geometry: HeldObjectGeometry | None = None
        self._carried_checker = CarriedEnvelopeChecker()
        self._held_path_clearance_m: float | None = None
        self._held_min_clearance_m: float | None = None
        self._last_navigation_failure_detail: str | None = None
        self._lateral_held_geometry: HeldObjectGeometry | None = None
        self._lateral_predictive_guard = False
        self._lateral_guard_params = Task3LateralGuardParams()
        self._lateral_guard_block_count = 0
        self._lateral_min_clearance_m: float | None = None

    @property
    def goal(self) -> NavigationGoal | None:
        return self._goal

    @property
    def navigation_path(self) -> tuple[tuple[float, float], ...]:
        return self._navigation.path

    @property
    def navigation_grid(self) -> LayeredGrid:
        """Read-only layered grid used by executor candidate hooks.

        Scheduler stand candidates are re-validated on this same grid so
        the executor-side footprint/clearance check can never disagree
        with the controller that will actually drive the segment.
        """
        return self._navigation_grid

    @property
    def last_navigation_failure_detail(self) -> str | None:
        return self._last_navigation_failure_detail

    def reset(self) -> None:
        self._navigation.reset()
        self._goal = None
        self._last_tick_s = None
        self._retreat_start = None
        self._retreat_distance_m = 0.0
        self._advance_start = None
        self._advance_distance_m = 0.0
        self._guided_start = None
        self._guided_target = None
        self._guided_final_yaw = 0.0
        self._guided_forward_m = 0.0
        self._guided_lateral_m = 0.0
        self._guided_start_s = None
        self._guided_held_geometry = None
        self._guided_min_clearance_m = None
        self._lateral_target = None
        self._lateral_final_yaw = 0.0
        self._lateral_heading = 0.0
        self._lateral_travel_heading = 0.0
        self._lateral_drive_sign = 1.0
        self._lateral_phase = "idle"
        self._lateral_start_s = None
        self._lateral_position_tolerance_m = self.LATERAL_POSITION_TOLERANCE_M
        self._lateral_yaw_tolerance_rad = self.LATERAL_YAW_TOLERANCE_RAD
        self._lateral_timeout_s = self.LATERAL_TIMEOUT_S
        self._held_geometry = None
        self._held_path_clearance_m = None
        self._held_min_clearance_m = None
        self._last_navigation_failure_detail = None
        self._lateral_held_geometry = None
        self._lateral_predictive_guard = False
        self._lateral_guard_params = Task3LateralGuardParams()
        self._lateral_guard_block_count = 0
        self._lateral_min_clearance_m = None

    def begin_navigation(
        self,
        goal: NavigationGoal,
        odometry: Any,
        *,
        footprint_mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        observations: Mapping[str, Any] | None = None,
        exclude_color: str | None = None,
        payload_z: float | None = None,
        held_geometry: HeldObjectGeometry | None = None,
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
        self._held_geometry = held_geometry
        self._held_path_clearance_m = None
        self._held_min_clearance_m = None
        self._last_navigation_failure_detail = None
        if not self._navigation.set_goal(goal, pose[0], pose[1]):
            self._goal = None
            self._last_navigation_failure_detail = "base planner found no path"
            return False
        if held_geometry is not None:
            # The measured box/arms swept envelope must also survive the
            # planned path.  The base-only A* grid cannot see the payload.
            safety = self._carried_checker.check_path(
                pose,
                self._navigation.path,
                goal.yaw,
                held_geometry.center_base,
                held_geometry.half_width_m,
            )
            if not safety.safe:
                self._last_navigation_failure_detail = safety.detail
                self._navigation.reset()
                self._goal = None
                self._held_geometry = None
                return False
            self._held_path_clearance_m = float(safety.clearance_m)
            self._held_min_clearance_m = float(safety.clearance_m)
        return True

    def tick_navigation(
        self,
        odometry: Any,
        now_s: float,
        *,
        linear_scale: float = 1.0,
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
        if linear_scale == 1.0:
            # Preserve the exact verified call path (and compatible injected
            # navigation implementations) while the experiment is disabled.
            command = self._navigation.update(*pose, dt, obs=None)
        else:
            command = self._navigation.update(
                *pose,
                dt,
                obs=None,
                linear_scale=linear_scale,
            )
        status = self._navigation.status
        if self._held_geometry is not None:
            # Fail-closed per-tick payload gate: any commanded sweep of the
            # measured box through the shelf/perimeter walls stops motion.
            safety = self._carried_checker.check_command(
                pose,
                (command.linear_x, command.angular_z),
                self._held_geometry.center_base,
                self._held_geometry.half_width_m,
            )
            if not safety.safe:
                self._navigation.reset()
                self._goal = None
                self._held_geometry = None
                return (
                    NavigationStatus.EMERGENCY_STOP,
                    (0.0, 0.0),
                    f"carried envelope guard stopped motion: {safety.detail}",
                )
            clearance = float(safety.clearance_m)
            self._held_min_clearance_m = min(
                clearance,
                self._held_min_clearance_m
                if self._held_min_clearance_m is not None
                else clearance,
            )
        detail = (
            f"goal=({self._goal.x:.2f}, {self._goal.y:.2f}, "
            f"{self._goal.yaw:.2f}); nav_status={status.value}; "
            f"{format_nav_telemetry(self._navigation.telemetry, phase='transfer')}"
        )
        if self._held_geometry is not None:
            detail += (
                "; measured_carried_guard=active "
                f"source={self._held_geometry.source or 'unknown'} "
                f"half_width={self._held_geometry.half_width_m:.3f}m "
                f"path_clearance={self._held_path_clearance_m:.3f}m "
                f"minimum_clearance={self._held_min_clearance_m:.3f}m"
            )
        if status is NavigationStatus.GOAL_REACHED:
            detail = f"{detail}; {goal_reached_event(self._goal)}"
        return status, (command.linear_x, command.angular_z), detail

    def check_held_command(
        self,
        odometry: Any,
        command: tuple[float, float],
        held_geometry: HeldObjectGeometry,
    ) -> tuple[bool, str]:
        """Check and describe a manually generated carried-object command.

        Task 3 deliberately uses bounded retreat, turn, straight and lateral
        controllers instead of A* for its constrained transfer corridor.  They
        still need the same measured payload sweep check and telemetry as an
        A* navigation tick.
        """

        pose = odometry_pose(odometry)
        if pose is None:
            return False, "measured carried guard waiting for valid odometry"
        safety = self._carried_checker.check_command(
            pose,
            (float(command[0]), float(command[1])),
            held_geometry.center_base,
            held_geometry.half_width_m,
        )
        clearance = float(safety.clearance_m)
        detail = (
            "measured_carried_guard=active "
            f"source={held_geometry.source or 'unknown'} "
            f"half_width={held_geometry.half_width_m:.3f}m "
            f"path_clearance={clearance:.3f}m "
            f"minimum_clearance={clearance:.3f}m; {safety.detail}"
        )
        return safety.safe, detail

    @property
    def guided_object_center_clearance_m(self) -> float:
        """Return the payload-safe object-centre clearance for the outside Gate."""

        box_radius = math.hypot(
            self._carried_checker.BOX_HALF_FORWARD_M,
            self._carried_checker.BOX_HALF_LATERAL_M,
        ) + self._carried_checker.BOX_EXTRA_RADIUS_M
        return (
            box_radius
            + self._carried_checker.clearance_m
            + self.GUIDED_GATE_BUFFER_M
        )

    def begin_guided_advance(
        self,
        target_xy: tuple[float, float],
        final_yaw: float,
        odometry: Any,
        now_s: float,
        *,
        held_geometry: HeldObjectGeometry,
    ) -> bool:
        """Start a bounded zero-end-slope S-curve to an outside-shelf Gate."""

        pose = odometry_pose(odometry)
        try:
            target_x = float(target_xy[0])
            target_y = float(target_xy[1])
            yaw_ref = _wrap_to_pi(float(final_yaw))
            start_s = float(now_s)
        except (TypeError, ValueError, IndexError):
            return False
        if pose is None or not all(
            math.isfinite(value)
            for value in (target_x, target_y, yaw_ref, start_s)
        ):
            return False

        dx = target_x - pose[0]
        dy = target_y - pose[1]
        c = math.cos(yaw_ref)
        s = math.sin(yaw_ref)
        forward_m = c * dx + s * dy
        lateral_m = -s * dx + c * dy
        initial_yaw_error = _wrap_to_pi(yaw_ref - pose[2])
        if (
            forward_m <= 0.10
            or forward_m > 1.20
            or abs(lateral_m) > self.GUIDED_MAX_LATERAL_M
            or abs(initial_yaw_error) > self.GUIDED_MAX_INITIAL_YAW_ERROR_RAD
        ):
            return False

        # Sample the exact planned curve with the existing measured payload
        # envelope. The shelf is still outside the payload at this Gate.
        start_check = self._carried_checker.check_pose(
            pose, held_geometry.center_base, held_geometry.half_width_m
        )
        if not start_check.safe:
            return False
        best_clearance = float(start_check.clearance_m)
        samples = max(
            1, int(math.ceil(forward_m / self._carried_checker.PATH_SAMPLE_M))
        )
        for sample in range(1, samples + 1):
            u = sample / samples
            progress = forward_m * u
            smooth = 3.0 * u * u - 2.0 * u * u * u
            lateral = lateral_m * smooth
            slope = (lateral_m / forward_m) * 6.0 * u * (1.0 - u)
            sample_pose = (
                pose[0] + c * progress - s * lateral,
                pose[1] + s * progress + c * lateral,
                _wrap_to_pi(yaw_ref + math.atan(slope)),
            )
            check = self._carried_checker.check_pose(
                sample_pose, held_geometry.center_base, held_geometry.half_width_m
            )
            if not check.safe:
                return False
            best_clearance = min(best_clearance, float(check.clearance_m))

        self._guided_start = pose
        self._guided_target = (target_x, target_y)
        self._guided_final_yaw = yaw_ref
        self._guided_forward_m = forward_m
        self._guided_lateral_m = lateral_m
        self._guided_start_s = start_s
        self._guided_held_geometry = held_geometry
        self._guided_min_clearance_m = best_clearance
        return True

    def tick_guided_advance(
        self,
        odometry: Any,
        now_s: float,
    ) -> tuple[NavigationStatus, tuple[float, float], str]:
        """Track the outside-shelf S-curve and stop at its transition Gate."""

        pose = odometry_pose(odometry)
        if pose is None:
            return NavigationStatus.NAVIGATING, (0.0, 0.0), (
                "guided shelf approach waiting for valid odometry"
            )
        if (
            self._guided_start is None
            or self._guided_target is None
            or self._guided_start_s is None
            or self._guided_held_geometry is None
        ):
            return NavigationStatus.FAILED, (0.0, 0.0), (
                "guided shelf approach was not started"
            )
        elapsed = max(0.0, float(now_s) - self._guided_start_s)
        if elapsed > self.GUIDED_TIMEOUT_S:
            return NavigationStatus.FAILED, (0.0, 0.0), (
                f"guided shelf approach timed out after {elapsed:.1f}s "
                f"(limit={self.GUIDED_TIMEOUT_S:.1f}s)"
            )

        start_x, start_y, _start_yaw = self._guided_start
        c = math.cos(self._guided_final_yaw)
        s = math.sin(self._guided_final_yaw)
        dx = pose[0] - start_x
        dy = pose[1] - start_y
        progress = c * dx + s * dy
        lateral = -s * dx + c * dy
        u = max(0.0, min(1.0, progress / self._guided_forward_m))
        smooth = 3.0 * u * u - 2.0 * u * u * u
        reference_lateral = self._guided_lateral_m * smooth
        slope = (
            (self._guided_lateral_m / self._guided_forward_m)
            * 6.0 * u * (1.0 - u)
        )
        cross_track_error = reference_lateral - lateral
        yaw_offset = max(
            -self.GUIDED_MAX_YAW_OFFSET_RAD,
            min(
                self.GUIDED_MAX_YAW_OFFSET_RAD,
                math.atan(slope + self.GUIDED_CROSS_TRACK_GAIN * cross_track_error),
            ),
        )
        desired_yaw = _wrap_to_pi(self._guided_final_yaw + yaw_offset)
        yaw_error = _wrap_to_pi(desired_yaw - pose[2])

        target_x, target_y = self._guided_target
        goal_dx = target_x - pose[0]
        goal_dy = target_y - pose[1]
        forward_error = c * goal_dx + s * goal_dy
        lateral_error = -s * goal_dx + c * goal_dy
        final_yaw_error = _wrap_to_pi(self._guided_final_yaw - pose[2])
        if (
            -self.GUIDED_POSITION_TOLERANCE_M <= forward_error <= 0.020
            and abs(lateral_error) <= self.GUIDED_POSITION_TOLERANCE_M
            and abs(final_yaw_error) <= self.GUIDED_YAW_TOLERANCE_RAD
        ):
            return NavigationStatus.GOAL_REACHED, (0.0, 0.0), (
                "guided shelf approach reached outside transition Gate; "
                f"forward_err={forward_error:.3f}m, "
                f"lateral_err={lateral_error:.3f}m, "
                f"yaw_err={final_yaw_error:.3f}rad, "
                f"minimum_clearance={self._guided_min_clearance_m:.3f}m"
            )
        if forward_error < -self.GUIDED_POSITION_TOLERANCE_M:
            return NavigationStatus.FAILED, (0.0, 0.0), (
                "guided shelf approach overshot its outside transition Gate; "
                f"forward_err={forward_error:.3f}m, "
                f"lateral_err={lateral_error:.3f}m"
            )

        linear = max(
            0.0,
            min(
                self.GUIDED_MAX_LINEAR_MPS,
                0.55 * max(0.0, forward_error),
            ),
        )
        if abs(yaw_error) > 0.13:
            linear = 0.0
        else:
            linear *= max(0.20, math.cos(yaw_error))
        angular = max(
            -self.GUIDED_MAX_ANGULAR_RPS,
            min(self.GUIDED_MAX_ANGULAR_RPS, 1.6 * yaw_error),
        )
        command = (linear, angular)
        safety = self._carried_checker.check_command(
            pose,
            command,
            self._guided_held_geometry.center_base,
            self._guided_held_geometry.half_width_m,
        )
        clearance = float(safety.clearance_m)
        self._guided_min_clearance_m = min(
            clearance,
            self._guided_min_clearance_m
            if self._guided_min_clearance_m is not None
            else clearance,
        )
        if not safety.safe:
            return NavigationStatus.EMERGENCY_STOP, (0.0, 0.0), (
                "guided shelf approach carried-envelope guard stopped motion: "
                f"{safety.detail}"
            )
        return NavigationStatus.NAVIGATING, command, (
            "guided shelf approach following bounded S-curve outside shelf; "
            f"forward_err={forward_error:.3f}m, "
            f"lateral_err={lateral_error:.3f}m, "
            f"yaw_offset={yaw_offset:.3f}rad, "
            f"minimum_clearance={self._guided_min_clearance_m:.3f}m"
        )

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
        drive_in_reverse: bool = False,
        held_geometry: HeldObjectGeometry | None = None,
        travel_face_yaw: float | None = None,
        predictive_guard: bool = False,
        large_yaw_threshold_rad: float | None = None,
        guard_params: Task3LateralGuardParams | None = None,
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

        params = guard_params or Task3LateralGuardParams()
        large_yaw = (
            params.large_yaw_threshold_rad
            if large_yaw_threshold_rad is None
            else float(large_yaw_threshold_rad)
        )
        if (
            held_geometry is not None or travel_face_yaw is not None
        ) and abs(_wrap_to_pi(target_yaw - pose[2])) > large_yaw:
            # Task-3 shelf-front alignment must not start an unconstrained
            # in-place spin when the pre-place yaw is already lost.
            return False

        self._lateral_target = (target_x, target_y)
        self._lateral_final_yaw = target_yaw
        self._lateral_position_tolerance_m = tolerance
        self._lateral_yaw_tolerance_rad = yaw_tolerance
        self._lateral_timeout_s = timeout
        self._lateral_held_geometry = held_geometry
        self._lateral_predictive_guard = bool(predictive_guard) and held_geometry is not None
        self._lateral_guard_params = params
        self._lateral_guard_block_count = 0
        self._lateral_min_clearance_m = None
        if travel_face_yaw is not None:
            # Task-3 carry-safe travel: keep one face yaw (north) and reverse
            # when the Y error is south, so the held box never sweeps into
            # the south wall during a left correction.
            face_yaw = _wrap_to_pi(float(travel_face_yaw))
            moving_plus_y = target_y >= pose[1]
            self._lateral_travel_heading = (
                math.pi / 2.0 if moving_plus_y else -math.pi / 2.0
            )
            self._lateral_heading = face_yaw
            face_y_component = math.sin(face_yaw)
            if abs(face_y_component) < 0.5:
                return False
            desired_y_sign = 1.0 if moving_plus_y else -1.0
            self._lateral_drive_sign = (
                1.0 if (desired_y_sign * face_y_component) > 0.0 else -1.0
            )
        else:
            self._lateral_travel_heading = (
                math.pi / 2.0 if target_y >= pose[1] else -math.pi / 2.0
            )
            self._lateral_drive_sign = -1.0 if drive_in_reverse else 1.0
            self._lateral_heading = _wrap_to_pi(
                self._lateral_travel_heading
                + (math.pi if drive_in_reverse else 0.0)
            )
        if abs(target_y - pose[1]) <= self._lateral_position_tolerance_m:
            self._lateral_phase = "rotate_final"
        elif abs(_wrap_to_pi(self._lateral_heading - pose[2])) <= yaw_tolerance:
            self._lateral_phase = "drive_lateral"
        else:
            self._lateral_phase = "rotate_lateral"
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
                return self._emit_lateral_command(
                    pose,
                    (0.0, angular),
                    "lateral alignment rotating toward shelf-front direction; "
                    f"yaw_err={yaw_error:.3f}",
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
                heading_limit = (
                    self._lateral_yaw_tolerance_rad
                    if self._lateral_predictive_guard
                    else 0.25
                )
                heading_offset = max(
                    -heading_limit,
                    min(
                        heading_limit,
                        -math.sin(self._lateral_travel_heading) * 0.8 * x_error,
                    ),
                )
                desired_yaw = self._lateral_heading + heading_offset
                yaw_error = _wrap_to_pi(desired_yaw - pose[2])
                angular = max(-0.30, min(0.30, 1.2 * yaw_error))
                max_speed = self._lateral_guard_params.lateral_max_speed_mps
                linear = self._lateral_drive_sign * min(
                    max_speed, max(0.035, 0.55 * abs(y_error))
                )
                if abs(yaw_error) > self._lateral_guard_params.large_yaw_threshold_rad:
                    linear = 0.0
                return self._emit_lateral_command(
                    pose,
                    (linear, angular),
                    "lateral alignment driving along shelf front; "
                    f"y_err={y_error:.3f}, x_err={x_error:.3f}",
                )

        if self._lateral_phase == "rotate_final":
            yaw_error = _wrap_to_pi(self._lateral_final_yaw - pose[2])
            if abs(yaw_error) <= self._lateral_yaw_tolerance_rad:
                self._lateral_phase = "done"
                return NavigationStatus.GOAL_REACHED, (0.0, 0.0), (
                    "lateral alignment complete; shelf-facing yaw restored"
                )
            angular = max(-0.35, min(0.35, 1.4 * yaw_error))
            return self._emit_lateral_command(
                pose,
                (0.0, angular),
                "lateral alignment restoring shelf-facing yaw; "
                f"yaw_err={yaw_error:.3f}",
            )

        if self._lateral_phase == "done":
            return NavigationStatus.GOAL_REACHED, (0.0, 0.0), (
                "lateral alignment complete"
            )
        return NavigationStatus.FAILED, (0.0, 0.0), "lateral alignment failed"

    def _emit_lateral_command(
        self,
        pose: tuple[float, float, float],
        command: tuple[float, float],
        detail: str,
    ) -> tuple[NavigationStatus, tuple[float, float], str]:
        """Apply the optional predictive carry guard before a lateral command."""

        if not self._lateral_predictive_guard or self._lateral_held_geometry is None:
            return NavigationStatus.NAVIGATING, command, detail
        guarded = guard_task3_lateral_cmd(
            pose,
            command,
            self._lateral_held_geometry,
            checker=self._carried_checker,
            params=self._lateral_guard_params,
        )
        clearance = float(guarded.clearance_m)
        self._lateral_min_clearance_m = min(
            clearance,
            self._lateral_min_clearance_m
            if self._lateral_min_clearance_m is not None
            else clearance,
        )
        if guarded.blocked:
            self._lateral_guard_block_count += 1
            if (
                self._lateral_guard_block_count
                >= self._lateral_guard_params.max_consecutive_guard_blocks
            ):
                self._lateral_phase = "failed"
                return (
                    NavigationStatus.EMERGENCY_STOP,
                    (0.0, 0.0),
                    "TASK3_LATERAL_BLOCKED reason=predictive_guard "
                    f"{guarded.reason}",
                )
            return (
                NavigationStatus.NAVIGATING,
                (0.0, 0.0),
                f"{detail}; predictive_guard_hold={guarded.reason}",
            )
        self._lateral_guard_block_count = 0
        suffix = (
            f"; predictive_guard=slowed clearance={clearance:.3f}m"
            if guarded.slowed
            else f"; predictive_guard=ok clearance={clearance:.3f}m"
        )
        return (
            NavigationStatus.NAVIGATING,
            (guarded.linear_x, guarded.angular_z),
            detail + suffix,
        )

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
