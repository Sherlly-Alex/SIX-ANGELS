"""Navigation controller — integrates N2–N4 modules to drive the Client.

Provides a single ``update()`` call per control tick that returns the
``NavigationStatus`` and the ``(linear_x, angular_z)`` command for ``/cmd_vel``.

The controller:
1. Plans a global A* path when a goal is set.
2. Follows the path via lookahead point + heading/speed PID.
3. Applies the ``SpeedLimiter`` for kinematic / obstacle‑braking safety.
4. Checks ``EmergencyChecker`` on every tick — immediately zero‑speeds.
5. Re‑plans on consecutive path blockage and fails safely on timeout.

No ROS2 dependency.  No exploration.
"""
from __future__ import annotations

import math
import time as _time
from typing import List, Optional, Sequence, Tuple

from navigation.emergency_checker import EmergencyChecker
from navigation.global_planner import GlobalPlanner, NoPathError
from navigation.local_avoidance import LocalAvoidance
from navigation.local_goal_selector import select_local_goal
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    ObstacleObservation,
    SpeedLimits,
    VelocityCommand,
)
from navigation.occupancy_grid import OccupancyGrid
from navigation.path_validator import PathValidator
from navigation.speed_limiter import SpeedLimiter


class NavigationController:
    """Waypoint‑following controller with integrated safety."""

    # seconds after which a single‑segment plan is abandoned
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        grid: OccupancyGrid,
        speed_limits: SpeedLimits,
        *,
        pos_tolerance: float = 0.06,
        yaw_tolerance: float = 0.03,
        lookahead_distance: float = 1.0,
        timeout: float = DEFAULT_TIMEOUT,
        emergency_distance: float = 0.28,
    ):
        self._grid = grid
        self._planner = GlobalPlanner(grid)
        self._speed_limiter = SpeedLimiter(speed_limits)
        # Official eval has no LiDAR; slightly shorter grid estop reduces
        # false stops at shelf / table approach docks (was 0.35 m).
        self._emergency = EmergencyChecker(emergency_distance=float(emergency_distance))
        self._avoidance = LocalAvoidance()
        # Must match GlobalPlanner.plan_goal default (0.15 m hardware clearance).
        self._min_clearance = 0.15
        self._path_validator = PathValidator(min_clearance=self._min_clearance)
        self._pos_tol = float(pos_tolerance)
        self._yaw_tol = float(yaw_tolerance)
        self._lookahead = float(lookahead_distance)
        self._timeout = float(timeout)

        # mutable state
        self._status = NavigationStatus.IDLE
        self._path: List[Tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._goal: Optional[NavigationGoal] = None
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._goal_yaw: float = 0.0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> NavigationStatus:
        return self._status

    @property
    def path(self) -> tuple[tuple[float, float], ...]:
        """Return the current immutable path for external safety validation."""

        return tuple((float(x), float(y)) for x, y in self._path)

    def set_goal(self, goal: NavigationGoal, robot_x: float, robot_y: float) -> bool:
        """Plan a global path to *goal* from the robot's current position.

        Returns ``True`` on success, ``False`` when no path exists (status
        becomes ``FAILED``).
        """
        self._goal = goal
        self._goal_x = goal.x
        self._goal_y = goal.y
        self._goal_yaw = goal.yaw

        try:
            self._path = self._planner.plan_goal(
                robot_x, robot_y, goal, min_clearance=self._min_clearance,
            )
        except NoPathError:
            self._status = NavigationStatus.FAILED
            return False

        self._waypoint_idx = 0
        self._start_time = _time.time()
        self._speed_limiter.reset()
        self._path_validator.reset()
        self._status = NavigationStatus.NAVIGATING
        return True

    def update(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        dt: float,
        obs: Optional[ObstacleObservation] = None,
    ) -> VelocityCommand:
        """Advance one control tick.

        Parameters
        ----------
        robot_x, robot_y, robot_yaw:
            Current odometry pose.
        dt:
            Control period (s).
        obs:
            Optional LiDAR observation for emergency / avoidance.

        Returns
        -------
        VelocityCommand
            Safe ``(linear_x, angular_z)`` for ``/cmd_vel``.
        """
        if self._status in (NavigationStatus.IDLE, NavigationStatus.GOAL_REACHED,
                            NavigationStatus.FAILED):
            return VelocityCommand(0.0, 0.0)

        # --- non‑finite pose → safe zero ---
        if not (math.isfinite(robot_x) and math.isfinite(robot_y) and math.isfinite(robot_yaw)):
            return VelocityCommand(0.0, 0.0)

        # --- emergency latch: once triggered, stay stopped until reset ---
        if self._status == NavigationStatus.EMERGENCY_STOP:
            return VelocityCommand(0.0, 0.0)

        # One-tick REPLANNING marker, then resume normal following.
        if self._status == NavigationStatus.REPLANNING:
            self._status = NavigationStatus.NAVIGATING

        # --- emergency check ---
        if self._emergency.is_emergency(obs, self._grid, robot_x, robot_y, robot_yaw):
            self._status = NavigationStatus.EMERGENCY_STOP
            return VelocityCommand(0.0, 0.0)

        # --- timeout ---
        if _time.time() - self._start_time > self._timeout:
            self._status = NavigationStatus.FAILED
            return VelocityCommand(0.0, 0.0)

        # --- goal‑reached check ---
        dist_to_goal = math.hypot(robot_x - self._goal_x, robot_y - self._goal_y)
        if dist_to_goal <= self._pos_tol:
            # position reached — switch to final yaw alignment
            yaw_err = _wrap_to_pi(self._goal_yaw - robot_yaw)
            if abs(yaw_err) <= self._yaw_tol:
                self._status = NavigationStatus.GOAL_REACHED
                return VelocityCommand(0.0, 0.0)
            raw_lin = 0.0
            raw_ang = 2.0 * yaw_err
            candidate = VelocityCommand(linear_x=raw_lin, angular_z=raw_ang)
            return self._speed_limiter.limit(candidate, dt, 100.0, 0.0)

        # --- path blocked → replan ---
        if self._path_validator.confirm_blocked(
            self._path, self._waypoint_idx, self._grid, lookahead=2.5,
        ):
            self._status = NavigationStatus.BLOCKED
            if self._goal is not None:
                try:
                    self._path = self._planner.plan_goal(
                        robot_x, robot_y, self._goal,
                        min_clearance=self._min_clearance,
                    )
                    self._waypoint_idx = 0
                    self._start_time = _time.time()
                    self._path_validator.reset()
                    self._speed_limiter.reset()
                    self._status = NavigationStatus.REPLANNING
                except NoPathError:
                    self._status = NavigationStatus.FAILED
                    return VelocityCommand(0.0, 0.0)
            else:
                self._status = NavigationStatus.FAILED
                return VelocityCommand(0.0, 0.0)

        # --- advance waypoint ---
        # A differential-drive robot cuts A* grid-cell corners and may never
        # pass within the final-position tolerance of an intermediate cell.
        # Select the nearest waypoint ahead of the current index and keep the
        # index monotonic so the lookahead cannot pull the robot backwards.
        if self._path:
            start_idx = min(self._waypoint_idx, len(self._path) - 1)
            nearest_idx = min(
                range(start_idx, len(self._path)),
                key=lambda index: math.hypot(
                    robot_x - self._path[index][0],
                    robot_y - self._path[index][1],
                ),
            )
            self._waypoint_idx = max(self._waypoint_idx, nearest_idx)
            waypoint_tolerance = max(
                self._pos_tol,
                1.5 * self._grid.resolution,
            )
            while self._waypoint_idx + 1 < len(self._path):
                wx, wy = self._path[self._waypoint_idx]
                if math.hypot(robot_x - wx, robot_y - wy) > waypoint_tolerance:
                    break
                self._waypoint_idx += 1

        # --- local goal ---
        lg_x, lg_y, _lg_yaw = select_local_goal(
            robot_x, robot_y, self._path,
            lookahead_distance=self._lookahead,
            closest_index=self._waypoint_idx,
        )

        # --- velocity command ---
        dx = lg_x - robot_x
        dy = lg_y - robot_y
        target_yaw = math.atan2(dy, dx)
        yaw_error = _wrap_to_pi(target_yaw - robot_yaw)
        path_deviation = math.hypot(dx, dy) * abs(math.sin(yaw_error)) if math.hypot(dx, dy) > 1e-6 else 0.0

        # proportional speed / heading control
        dist_to_target = math.hypot(dx, dy)
        raw_lin = min(0.3, 0.8 * dist_to_target)
        raw_ang = 2.0 * yaw_error

        # obstacle distance (minimum from grid distance transform)
        gx, gy = self._grid.world_to_grid(robot_x, robot_y)
        obs_dist = 100.0
        if gx >= 0 and gy >= 0:
            obs_dist = float(self._grid.distance_transform()[gy, gx]) * self._grid.resolution

        candidate = VelocityCommand(linear_x=raw_lin, angular_z=raw_ang)

        # --- local avoidance (N4) — angular repulsion from dynamic obstacles ---
        candidate, _static_fallback = self._avoidance.adjust(
            candidate, obs, self._grid, robot_x, robot_y, robot_yaw,
        )

        return self._speed_limiter.limit(candidate, dt, obs_dist, path_deviation)

    def reset(self) -> None:
        """Clear internal state for a fresh navigation segment."""
        self._status = NavigationStatus.IDLE
        self._path = []
        self._waypoint_idx = 0
        self._goal = None
        self._speed_limiter.reset()
        self._path_validator.reset()


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
