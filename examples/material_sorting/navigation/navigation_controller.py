"""Navigation controller — integrates N2–N4 modules to drive the Client.

Provides a single ``update()`` call per control tick that returns the
``NavigationStatus`` and the ``(linear_x, angular_z)`` command for ``/cmd_vel``.

The controller:
1. Plans a global A* path when a goal is set (chassis-layer planning surface).
2. Follows the path via lookahead point + heading/speed control.
3. Uses explicit terminal-positioning and final-yaw states near the goal.
4. Applies the ``SpeedLimiter`` for kinematic / obstacle‑braking safety.
5. Checks ``EmergencyChecker`` (forward ray + oriented footprint) every tick —
   soft-latches to zero and clears after consecutive free ticks.
6. Re‑plans on consecutive path blockage and fails safely on timeout.

No ROS2 dependency.  No exploration.
"""
from __future__ import annotations

import math
import time as _time
from typing import List, Optional, Tuple, Union

from navigation.emergency_checker import EmergencyChecker
from navigation.footprint_checker import FootprintChecker
from navigation.global_planner import GlobalPlanner, NoPathError
from navigation.local_avoidance import LocalAvoidance
from navigation.local_goal_selector import select_local_goal
from navigation.navigation_types import (
    NavigationGoal,
    NavigationStatus,
    NavigationTelemetry,
    ObstacleObservation,
    SpeedLimits,
    VelocityCommand,
)
from navigation.occupancy_grid import ARM_Z_MIN, LayeredGrid, OccupancyGrid
from navigation.path_validator import PathValidator
from navigation.path_smoother import smooth_path
from navigation.robot_geometry import FootprintMode
from navigation.speed_limiter import SpeedLimiter

GridLike = Union[OccupancyGrid, LayeredGrid]


class NavigationController:
    """Waypoint‑following controller with integrated safety."""

    # seconds after which a single‑segment plan is abandoned
    DEFAULT_TIMEOUT = 120.0
    # Hardware chassis half-width is 0.20 m; keep 2 cm margin.
    DEFAULT_MIN_CLEARANCE = 0.22

    def __init__(
        self,
        grid: GridLike,
        speed_limits: SpeedLimits,
        *,
        pos_tolerance: float = 0.06,
        yaw_tolerance: float = 0.03,
        lookahead_distance: float = 1.0,
        terminal_positioning_radius: float = 0.18,
        terminal_linear_max: float = 0.10,
        terminal_heading_gate: float = 0.35,
        terminal_turn_max: float = 0.5,
        timeout: float = DEFAULT_TIMEOUT,
        emergency_distance: float = 0.28,
        soft_estop_clear_ticks: int = 10,
        footprint_mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        max_brake_ticks: int = 8,
        enable_footprint_path_check: bool = True,
        heading_gate_enter: float = 1.0,
        heading_gate_exit: float = 0.25,
        rot_gain: float = 2.0,
        estop_fail_ticks: int = 160,
    ):
        if isinstance(grid, LayeredGrid):
            self._grid: GridLike = grid
            planning = grid.planning_grid()
        else:
            self._grid = grid
            planning = grid

        self._planner = GlobalPlanner(planning)
        self._speed_limiter = SpeedLimiter(speed_limits)
        self._footprint = FootprintChecker()
        self._footprint_mode = footprint_mode
        # The envelope used by the live emergency / predictive-sweep gate.
        # It is deliberately tracked separately from the path-planning mode so
        # telemetry can prove that the reported clearance uses the same
        # geometry as the command authority.
        self._safety_footprint_mode = footprint_mode
        # Official eval has no LiDAR; slightly shorter grid estop reduces
        # false stops at shelf / table approach docks (was 0.35 m).
        self._emergency = EmergencyChecker(
            emergency_distance=float(emergency_distance),
            footprint_checker=self._footprint,
            footprint_mode=footprint_mode,
        )
        self._avoidance = LocalAvoidance()
        self._min_clearance = float(self.DEFAULT_MIN_CLEARANCE)
        self._path_validator = PathValidator(
            min_clearance=self._min_clearance,
            footprint_checker=(
                self._footprint if enable_footprint_path_check else None
            ),
            footprint_mode=footprint_mode,
        )
        self._pos_tol = float(pos_tolerance)
        self._yaw_tol = float(yaw_tolerance)
        self._active_pos_tol = self._pos_tol
        self._active_yaw_tol = self._yaw_tol
        self._lookahead = float(lookahead_distance)
        self._terminal_radius = float(terminal_positioning_radius)
        self._terminal_linear_max = float(terminal_linear_max)
        self._terminal_heading_gate = float(terminal_heading_gate)
        self._terminal_turn_max = float(terminal_turn_max)
        self._timeout = float(timeout)
        self._soft_estop_clear_ticks = max(1, int(soft_estop_clear_ticks))
        self._estop_free_ticks = 0
        # Consecutive predictive-brake ticks tolerated before escalating to a
        # replan.  Braking is only useful against something that moves away.
        self._max_brake_ticks = max(1, int(max_brake_ticks))
        self._brake_ticks = 0
        # Heading gate for reverse goals (phase B).  Plan §B2 proposed 1.9 /
        # 0.35; with the real-time server loop the plant is clean, but a
        # 60-100° corner turn under 1.9 still arcs forward while rotating, and
        # on the right-side stand that drift pushes the arm envelope to within
        # 5 cm of the east wall where the predictive brake deadlocks.  A
        # tighter 1.0 / 0.25 gate (still hysteresis, still plan-shaped) forces
        # a pure in-place reorient for any turn beyond ~57°, which the plant
        # executes with negligible drift.
        self._heading_gate_enter = float(heading_gate_enter)
        self._heading_gate_exit = float(heading_gate_exit)
        self._rot_gain = float(rot_gain)
        self._rotating_in_place = False
        # Frozen carrot heading while rotating in place.  On L-docks the
        # pure-pursuit carrot can jump from the staging leg (west/east) to the
        # approach leg (north) mid-spin, flipping the P target and thrashing
        # the base into the table/shelf band.  Latch the enter heading until
        # the exit gate is met.
        self._rotate_target_yaw: Optional[float] = None
        # Soft-estop escalation: a pose-collision latch that never clears (robot
        # wedged) must surface as FAILED rather than hang forever.  The plan
        # forbids reversing out, so the client retries from the phase entry.
        self._estop_fail_ticks = max(1, int(estop_fail_ticks))
        self._estop_latch_ticks = 0
        # Replan-without-progress guard: after several replans that never move
        # the robot (predictive-sweep blocked near an obstacle), fail instead of
        # looping.
        self._replan_count = 0
        self._replan_unstick_threshold = 5

        if not (math.isfinite(self._pos_tol) and self._pos_tol > 0.0):
            raise ValueError("pos_tolerance must be finite and > 0")
        if not (math.isfinite(self._yaw_tol) and self._yaw_tol > 0.0):
            raise ValueError("yaw_tolerance must be finite and > 0")
        if not (math.isfinite(self._terminal_radius) and self._terminal_radius > self._pos_tol):
            raise ValueError("terminal_positioning_radius must be finite and > pos_tolerance")
        if not (math.isfinite(self._terminal_linear_max) and self._terminal_linear_max > 0.0):
            raise ValueError("terminal_linear_max must be finite and > 0")
        if not (math.isfinite(self._terminal_turn_max) and self._terminal_turn_max > 0.0):
            raise ValueError("terminal_turn_max must be finite and > 0")
        if not (
            math.isfinite(self._terminal_heading_gate)
            and 0.0 < self._terminal_heading_gate <= math.pi
        ):
            raise ValueError("terminal_heading_gate must be finite and in (0, pi]")

        # mutable state
        self._status = NavigationStatus.IDLE
        self._path: List[Tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._goal: Optional[NavigationGoal] = None
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._goal_yaw: float = 0.0
        self._start_time: float = 0.0
        self._telemetry = NavigationTelemetry(
            status=NavigationStatus.IDLE.value,
            x=0.0, y=0.0, yaw=0.0,
            goal_x=0.0, goal_y=0.0, goal_yaw=0.0,
            dist_err=0.0, yaw_err=0.0,
            cmd_lin=0.0, cmd_ang=0.0,
            footprint_min_clearance=0.0,
            rotate_in_place=False,
            lookahead=0.0,
            kappa=0.0,
            path_deviation=0.0,
            footprint_mode=footprint_mode.value,
            segment="",
        )
        self._last_lookahead = 0.0
        self._last_kappa = 0.0
        self._last_path_deviation = 0.0
        self._planned_path_length = 0.0
        self._planned_straight = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> NavigationStatus:
        return self._status

    @property
    def telemetry(self) -> NavigationTelemetry:
        """Latest per-tick navigation snapshot (safe to log / serialize)."""
        return self._telemetry

    @property
    def footprint_mode(self) -> FootprintMode:
        return self._footprint_mode

    @property
    def safety_footprint_mode(self) -> FootprintMode:
        """Envelope currently used by emergency and sweep collision gates."""
        return self._safety_footprint_mode

    @property
    def path(self) -> List[Tuple[float, float]]:
        """The smoothed path currently being followed (empty when idle)."""
        return list(self._path)

    def set_footprint_mode(
        self,
        mode: FootprintMode,
        *,
        payload_z: Optional[float] = None,
    ) -> None:
        """Switch the oriented envelope used by path / emergency checks.

        Callers should set ``TRANSIT_STOWED`` for empty-handed transit,
        ``TRANSIT_CARRY`` while transporting a box, and ``DOCKING`` (chassis
        only) once inside the terminal radius.

        *payload_z* is the world height (m) of the lowest carried / stowed arm
        geometry.  The arm layer omits the table, which is only sound while
        that geometry stays inside the arm height band; below it the checker
        falls back to the chassis layer so the table is treated as a hazard.
        """
        self._footprint_mode = mode
        self._safety_footprint_mode = mode
        self._emergency.set_footprint_mode(mode)
        self._path_validator.set_footprint_mode(mode)
        if payload_z is not None:
            self._footprint.set_arm_layer_enabled(float(payload_z) >= ARM_Z_MIN)

    def set_goal(self, goal: NavigationGoal, robot_x: float, robot_y: float) -> bool:
        """Plan a global path to *goal* from the robot's current position.

        Returns ``True`` on success, ``False`` when no path exists (status
        becomes ``FAILED``).
        """
        self._goal = goal
        self._goal_x = goal.x
        self._goal_y = goal.y
        self._goal_yaw = goal.yaw

        goal_pos_tol = float(goal.position_tolerance)
        goal_yaw_tol = float(goal.yaw_tolerance)
        if not (math.isfinite(goal_pos_tol) and goal_pos_tol > 0.0):
            self._status = NavigationStatus.FAILED
            return False
        if not (math.isfinite(goal_yaw_tol) and goal_yaw_tol > 0.0):
            self._status = NavigationStatus.FAILED
            return False
        self._active_pos_tol = goal_pos_tol
        self._active_yaw_tol = goal_yaw_tol

        # Refresh the A* surface so dynamic overlays (if any) are visible.
        self._refresh_planner_grid()

        try:
            raw_path = self._planner.plan_goal(
                robot_x, robot_y, goal, min_clearance=self._min_clearance,
            )
        except NoPathError:
            # The robot is likely inside the A* inflation band around an
            # obstacle.  A* cannot plan from there; surface FAILED so the client
            # retries (the plan forbids reversing out of the band).
            self._status = NavigationStatus.FAILED
            return False

        self._path = smooth_path(
            raw_path, self._grid,
            footprint=self._footprint,
            # Must match PathValidator / emergency envelope: CHASSIS-only
            # smoothing can cut corners that TRANSIT_CARRY later rejects,
            # producing replan↔same-shortcut livelock on carry segments.
            mode=self._footprint_mode,
            # Force the arrival heading: the approach lane makes the robot arrive
            # already facing the goal yaw, so no terminal in-place turn is needed
            # at the stand pose (where the chassis is close to the table/shelf).
            approach_dir=(math.cos(goal.yaw), math.sin(goal.yaw)),
        )
        self._waypoint_idx = 0
        self._start_time = _time.time()
        self._speed_limiter.reset()
        self._path_validator.reset()
        self._estop_free_ticks = 0
        self._estop_latch_ticks = 0
        self._brake_ticks = 0
        self._replan_count = 0
        self._rotating_in_place = False
        self._rotate_target_yaw = None
        self._planned_path_length = self._path_length()
        self._planned_straight = math.hypot(goal.x - robot_x, goal.y - robot_y)
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
            out = VelocityCommand(0.0, 0.0)
            self._record_telemetry(robot_x, robot_y, robot_yaw, out)
            return out

        # --- non‑finite pose → safe zero ---
        if not (math.isfinite(robot_x) and math.isfinite(robot_y) and math.isfinite(robot_yaw)):
            out = VelocityCommand(0.0, 0.0)
            self._record_telemetry(robot_x, robot_y, robot_yaw, out)
            return out

        # --- soft emergency latch ---
        if self._status == NavigationStatus.EMERGENCY_STOP:
            if self._emergency.is_clear(obs, self._grid, robot_x, robot_y, robot_yaw):
                self._estop_free_ticks += 1
                if self._estop_free_ticks >= self._soft_estop_clear_ticks:
                    self._estop_free_ticks = 0
                    self._estop_latch_ticks = 0
                    self._speed_limiter.reset()
                    self._status = (
                        NavigationStatus.NAVIGATING
                        if self._goal is not None and self._path
                        else NavigationStatus.IDLE
                    )
                else:
                    out = VelocityCommand(0.0, 0.0)
                    self._record_telemetry(robot_x, robot_y, robot_yaw, out)
                    return out
            else:
                # Still flagged at rest: hold zero velocity.  A latch that never
                # clears (robot wedged against an obstacle) escalates to FAILED
                # instead of hanging — the plan forbids reversing out, so the
                # client retries the segment from its phase entry.
                self._estop_free_ticks = 0
                self._estop_latch_ticks += 1
                if self._estop_latch_ticks >= self._estop_fail_ticks:
                    self._status = NavigationStatus.FAILED
                out = VelocityCommand(0.0, 0.0)
                self._record_telemetry(robot_x, robot_y, robot_yaw, out)
                return out

        # One-tick REPLANNING marker, then resume normal following.
        if self._status == NavigationStatus.REPLANNING:
            self._status = NavigationStatus.NAVIGATING

        # --- timeout ---
        if _time.time() - self._start_time > self._timeout:
            self._status = NavigationStatus.FAILED
            out = VelocityCommand(0.0, 0.0)
            self._record_telemetry(robot_x, robot_y, robot_yaw, out)
            return out

        # --- terminal manoeuvre state machine ---
        dist_to_goal = math.hypot(robot_x - self._goal_x, robot_y - self._goal_y)
        yaw_err = _wrap_to_pi(self._goal_yaw - robot_yaw)

        terminal_entry_radius = max(
            self._terminal_radius,
            2.0 * self._active_pos_tol,
        )
        # Keep the live safety authority identical to the active navigation
        # envelope, including while terminal positioning or turning.  A carry
        # envelope may be close to a table/shelf exactly when the final yaw is
        # corrected; silently downgrading to chassis-only ``DOCKING`` there
        # would miss an arm/payload collision.  Callers may explicitly select
        # ``DOCKING`` only after they have independently certified that arms
        # are stowed and no payload protrudes beyond the chassis.
        self._safety_footprint_mode = self._footprint_mode
        self._emergency.set_footprint_mode(self._safety_footprint_mode)

        if self._status == NavigationStatus.FINAL_ALIGNING:
            cmd = self._update_final_alignment(dist_to_goal, yaw_err, dt)
            return self._gate_emergency(cmd, obs, robot_x, robot_y, robot_yaw)

        if self._status == NavigationStatus.FINAL_POSITIONING:
            cmd = self._update_final_positioning(
                robot_x, robot_y, robot_yaw, dist_to_goal, yaw_err, dt, obs,
            )
            return self._gate_emergency(cmd, obs, robot_x, robot_y, robot_yaw)

        if dist_to_goal <= self._active_pos_tol:
            if abs(yaw_err) <= self._active_yaw_tol:
                self._status = NavigationStatus.GOAL_REACHED
                out = VelocityCommand(0.0, 0.0)
                self._record_telemetry(robot_x, robot_y, robot_yaw, out)
                return out
            cmd = self._enter_terminal_state(NavigationStatus.FINAL_ALIGNING)
            self._record_telemetry(robot_x, robot_y, robot_yaw, cmd)
            return cmd

        if dist_to_goal <= terminal_entry_radius:
            # Continuous decelerate into terminal if already headed at the goal.
            position_yaw = math.atan2(self._goal_y - robot_y, self._goal_x - robot_x)
            if abs(_wrap_to_pi(position_yaw - robot_yaw)) <= self._terminal_heading_gate:
                self._status = NavigationStatus.FINAL_POSITIONING
                cmd = self._update_final_positioning(
                    robot_x, robot_y, robot_yaw, dist_to_goal, yaw_err, dt, obs,
                )
                return self._gate_emergency(cmd, obs, robot_x, robot_y, robot_yaw)
            cmd = self._enter_terminal_state(NavigationStatus.FINAL_POSITIONING)
            self._record_telemetry(robot_x, robot_y, robot_yaw, cmd)
            return cmd

        # --- path blocked → replan ---
        # Validate from the live pose, not from a path index: the smoother can
        # collapse a route to two points, and then `path[waypoint_idx:]` holds
        # no forward segment at all, so the segment being driven right now
        # would never be checked.
        remaining = self._remaining_route(robot_x, robot_y)
        if len(remaining) > 1 and self._path_validator.confirm_blocked(
            remaining, 0, self._grid, lookahead=2.5,
        ):
            self._status = NavigationStatus.BLOCKED
            if not self._replan(robot_x, robot_y):
                out = VelocityCommand(0.0, 0.0)
                self._record_telemetry(robot_x, robot_y, robot_yaw, out)
                return out

        # --- advance waypoint (plan §B1: monotonic nearest + 6 cm gate) ---
        # The 6 cm advance must NOT fire on path[0] at set_goal (the robot is
        # always within pos_tol of its own start) — that jumps the index past
        # the staging corner and pure-pursuit aims straight at the goal, which
        # is exactly the SE→table overshoot on short L-docks.  Skip the gate
        # while the nearest waypoint is still the path origin.
        if self._path:
            nearest = self._nearest_path_index(robot_x, robot_y)
            self._waypoint_idx = max(self._waypoint_idx, nearest)
            if (
                self._waypoint_idx > 0
                and self._waypoint_idx < len(self._path)
            ):
                wx, wy = self._path[self._waypoint_idx]
                if math.hypot(robot_x - wx, robot_y - wy) < self._pos_tol:
                    self._waypoint_idx = min(
                        self._waypoint_idx + 1, len(self._path) - 1,
                    )

        # --- local goal with adaptive lookahead (plan §C3) ---
        prev_lin = self._speed_limiter.prev_linear or 0.0
        lookahead = _clamp(1.6 * abs(prev_lin), 0.35, 0.9)
        lg_x, lg_y, _lg_yaw = select_local_goal(
            robot_x, robot_y, self._path,
            lookahead_distance=lookahead,
            closest_index=self._waypoint_idx if self._path else None,
            project_from_pose=True,
        )

        dx = lg_x - robot_x
        dy = lg_y - robot_y
        target_yaw = math.atan2(dy, dx)
        dist_to_target = math.hypot(dx, dy)
        yaw_error = _wrap_to_pi(target_yaw - robot_yaw)

        was_rotating = self._rotating_in_place
        if self._rotating_in_place:
            # Hold the enter-time carrot heading until exit; recomputing from a
            # moving pure-pursuit carrot mid-spin thrash-flips the P target on
            # L-docks (staging → approach) and drifts the base into obstacles.
            if self._rotate_target_yaw is not None:
                yaw_error = _wrap_to_pi(self._rotate_target_yaw - robot_yaw)
            if abs(yaw_error) < self._heading_gate_exit:
                self._rotating_in_place = False
                self._rotate_target_yaw = None
                # Kill residual spin so the accel ramp cannot coast past the
                # exit gate and re-enter rotate on the next tick (limit-cycle
                # that was spinning the base forever on L-docks).
                self._speed_limiter.reset()
        elif abs(yaw_error) > self._heading_gate_enter:
            self._rotating_in_place = True
            self._rotate_target_yaw = target_yaw

        if self._rotating_in_place:
            # Plan §B2/C2: stop forward motion and rotate with a plain P law.
            # The speed limiter clamps the rate to max_angular; the hysteresis
            # band absorbs any coast-through on the clean real-time plant.
            raw_lin = 0.0
            raw_ang = self._rot_gain * yaw_error
            kappa = 0.0
            # Kill residual cruise velocity immediately on enter: the accel
            # ramp would otherwise keep a few cm/s of forward motion while
            # omega is large, which is what drifts the base off the stand
            # path into the table/shelf band on short L-docks.
            if not was_rotating:
                self._speed_limiter.reset()
        else:
            raw_lin = min(0.36, 1.0 * dist_to_target) * max(0.0, math.cos(yaw_error)) ** 2
            # Pure-pursuit curvature: kappa = 2 sin(e) / L ; omega = kappa * v.
            L = max(lookahead, 0.2)
            kappa = 2.0 * math.sin(yaw_error) / L
            if abs(raw_lin) < 1e-4:
                # cos^2 killed linear (heading ≈ ±90°+).  Pure-pursuit omega
                # would also be zero, freezing the robot while the limiter's
                # accel memory still coasts it sideways.  Fall back to the
                # same P heading law the rotating branch uses so we reorient
                # without residual translation.
                raw_ang = self._rot_gain * yaw_error
                kappa = 0.0
                self._speed_limiter.reset()
            else:
                raw_ang = kappa * raw_lin
                # Curvature speed limit against max_angular (plan §C3) so a tight
                # arc never demands more angular rate than the limiter allows.
                max_ang = self._speed_limiter.max_angular
                if abs(kappa) > 1e-6:
                    raw_lin = min(raw_lin, abs(max_ang / kappa))

        path_deviation = self._lateral_path_error(robot_x, robot_y)
        self._last_lookahead = float(lookahead)
        self._last_kappa = float(kappa)
        self._last_path_deviation = float(path_deviation)

        planning = _planning_surface(self._grid)
        gx, gy = planning.world_to_grid(robot_x, robot_y)
        obs_dist = 100.0
        if gx >= 0 and gy >= 0:
            obs_dist = float(planning.distance_transform()[gy, gx]) * planning.resolution

        candidate = VelocityCommand(linear_x=raw_lin, angular_z=raw_ang)
        candidate, _static_fallback = self._avoidance.adjust(
            candidate, obs, planning, robot_x, robot_y, robot_yaw,
        )
        limited = self._speed_limiter.limit(candidate, dt, obs_dist, path_deviation)
        return self._gate_emergency(limited, obs, robot_x, robot_y, robot_yaw)

    def _path_length(self) -> float:
        if len(self._path) < 2:
            return 0.0
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self._path, self._path[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    def _record_telemetry(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        cmd: VelocityCommand,
    ) -> None:
        dist_err = (
            math.hypot(robot_x - self._goal_x, robot_y - self._goal_y)
            if self._goal is not None else 0.0
        )
        yaw_err = (
            _wrap_to_pi(self._goal_yaw - robot_yaw)
            if self._goal is not None else 0.0
        )
        straight = (
            math.hypot(self._goal_x - robot_x, self._goal_y - robot_y)
            if self._goal is not None else 0.0
        )
        clear = 0.0
        if math.isfinite(robot_x) and math.isfinite(robot_y) and math.isfinite(robot_yaw):
            clear = float(self._footprint.min_clearance(
                self._grid, robot_x, robot_y, robot_yaw,
                self._safety_footprint_mode,
            ))
        self._telemetry = NavigationTelemetry(
            status=self._status.value,
            x=float(robot_x), y=float(robot_y), yaw=float(robot_yaw),
            goal_x=float(self._goal_x), goal_y=float(self._goal_y),
            goal_yaw=float(self._goal_yaw),
            dist_err=float(dist_err), yaw_err=float(yaw_err),
            cmd_lin=float(cmd.linear_x), cmd_ang=float(cmd.angular_z),
            footprint_min_clearance=clear,
            rotate_in_place=bool(self._rotating_in_place),
            lookahead=float(self._last_lookahead),
            kappa=float(self._last_kappa),
            path_deviation=float(self._last_path_deviation),
            footprint_mode=self._safety_footprint_mode.value,
            path_length=float(self._planned_path_length),
            straight_distance=float(straight),
            planned_straight=float(self._planned_straight),
            segment=(self._goal.segment.value if self._goal is not None else ""),
        )

    def _nearest_path_index(self, robot_x: float, robot_y: float) -> int:
        """Return the path index closest to the robot (O(n), n is small)."""
        if not self._path:
            return 0
        best_i = 0
        best_d = float("inf")
        for i, (px, py) in enumerate(self._path):
            d = (px - robot_x) ** 2 + (py - robot_y) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _lateral_path_error(self, robot_x: float, robot_y: float) -> float:
        """Perpendicular distance from the robot to the current path segment."""
        if len(self._path) < 2:
            return 0.0
        i = min(self._waypoint_idx, len(self._path) - 2)
        ax, ay = self._path[i]
        bx, by = self._path[i + 1]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return math.hypot(robot_x - ax, robot_y - ay)
        t = ((robot_x - ax) * abx + (robot_y - ay) * aby) / ab2
        t = max(0.0, min(1.0, t))
        px = ax + t * abx
        py = ay + t * aby
        return math.hypot(robot_x - px, robot_y - py)

    def _gate_emergency(
        self,
        cmd: VelocityCommand,
        obs: Optional[ObstacleObservation],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> VelocityCommand:
        """Latch on pose/ray danger; brake one tick on predictive sweep hits."""
        if self._emergency.is_emergency(
            obs, self._grid, robot_x, robot_y, robot_yaw,
        ):
            self._status = NavigationStatus.EMERGENCY_STOP
            self._estop_free_ticks = 0
            self._speed_limiter.reset()
            out = VelocityCommand(0.0, 0.0)
            self._record_telemetry(robot_x, robot_y, robot_yaw, out)
            return out
        if self._emergency.would_collide_command(
            self._grid, robot_x, robot_y, robot_yaw,
            cmd.linear_x, cmd.angular_z,
        ):
            # Predictive brake: zero this tick without latching so reverse-goal
            # arcs and corner cuts do not kill the segment.
            self._speed_limiter.reset()
            self._brake_ticks += 1
            if self._brake_ticks >= self._max_brake_ticks:
                # Braking alone cannot clear a static obstruction: the robot is
                # stopped, so next tick recomputes the identical command and
                # brakes again forever.  Escalate to a replan; if none exists,
                # latch the (soft) estop so the caller sees a real failure.
                self._brake_ticks = 0
                self._replan_count += 1
                self._status = NavigationStatus.BLOCKED
                if self._replan_count >= self._replan_unstick_threshold:
                    # Repeated replans without progress: the robot is wedged near
                    # an obstacle (the predictive sweep blocks every command).
                    # The plan forbids reversing out, so fail and let the client
                    # retry from the phase entry.
                    self._status = NavigationStatus.FAILED
                elif not self._replan(robot_x, robot_y):
                    self._status = NavigationStatus.EMERGENCY_STOP
                    self._estop_free_ticks = 0
            out = VelocityCommand(0.0, 0.0)
            self._record_telemetry(robot_x, robot_y, robot_yaw, out)
            return out
        self._brake_ticks = 0
        self._replan_count = 0
        self._record_telemetry(robot_x, robot_y, robot_yaw, cmd)
        return cmd

    def _remaining_route(self, robot_x: float, robot_y: float) -> List[Tuple[float, float]]:
        """Live pose followed by the not-yet-passed waypoints."""
        if not self._path:
            return []
        tail = self._path[min(self._waypoint_idx, len(self._path) - 1):]
        return [(robot_x, robot_y)] + list(tail)

    def _replan(self, robot_x: float, robot_y: float) -> bool:
        """Re-plan to the active goal from ``(robot_x, robot_y)``.

        Sets ``REPLANNING`` and returns ``True`` on success; sets ``FAILED``
        and returns ``False`` when no route exists.
        """
        if self._goal is None:
            self._status = NavigationStatus.FAILED
            return False
        try:
            self._refresh_planner_grid()
            raw_path = self._planner.plan_goal(
                robot_x, robot_y, self._goal,
                min_clearance=self._min_clearance,
            )
        except NoPathError:
            self._status = NavigationStatus.FAILED
            return False
        self._path = smooth_path(
            raw_path, self._grid,
            footprint=self._footprint,
            # Must match PathValidator / emergency envelope: CHASSIS-only
            # smoothing can cut corners that TRANSIT_CARRY later rejects,
            # producing replan↔same-shortcut livelock on carry segments.
            mode=self._footprint_mode,
            approach_dir=(math.cos(self._goal_yaw), math.sin(self._goal_yaw)),
        )
        self._waypoint_idx = 0
        self._start_time = _time.time()
        self._path_validator.reset()
        self._speed_limiter.reset()
        self._planned_path_length = self._path_length()
        self._planned_straight = math.hypot(
            self._goal_x - robot_x, self._goal_y - robot_y,
        )
        self._status = NavigationStatus.REPLANNING
        return True

    def _refresh_planner_grid(self) -> None:
        """Point the A* planner at a fresh planning surface (dynamic overlays)."""
        self._planner = GlobalPlanner(_planning_surface(self._grid))

    def _enter_terminal_state(self, status: NavigationStatus) -> VelocityCommand:
        """Enter a terminal state through a one-tick zero-speed barrier.

        Resetting the limiter is intentional: otherwise its acceleration memory
        carries a positive linear command from path following into final yaw
        alignment.  The robot then drives a short arc while supposedly rotating
        in place, leaves the position tolerance, and chatters between position
        and heading control.
        """
        self._status = status
        self._speed_limiter.reset()
        return VelocityCommand(0.0, 0.0)

    def _update_final_alignment(
        self,
        dist_to_goal: float,
        yaw_err: float,
        dt: float,
    ) -> VelocityCommand:
        """Rotate in place after XY positioning has converged."""
        exit_tolerance = max(
            1.5 * self._active_pos_tol,
            self._active_pos_tol + 0.02,
        )
        if dist_to_goal > exit_tolerance:
            return self._enter_terminal_state(NavigationStatus.FINAL_POSITIONING)

        if abs(yaw_err) <= self._active_yaw_tol:
            if dist_to_goal <= self._active_pos_tol:
                self._status = NavigationStatus.GOAL_REACHED
                return VelocityCommand(0.0, 0.0)
            return self._enter_terminal_state(NavigationStatus.FINAL_POSITIONING)

        candidate = VelocityCommand(
            linear_x=0.0,
            angular_z=_clamp(
                self._rot_gain * yaw_err,
                -self._terminal_turn_max, self._terminal_turn_max,
            ),
        )
        return self._speed_limiter.limit(candidate, dt, 100.0, 0.0)

    def _update_final_positioning(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        dist_to_goal: float,
        yaw_err: float,
        dt: float,
        obs: Optional[ObstacleObservation],
    ) -> VelocityCommand:
        """Correct the last few centimetres before enforcing final yaw.

        A differential-drive base cannot correct a lateral XY error while
        simultaneously holding an unrelated final heading (for example pi).
        This state therefore points at the remaining displacement, translates
        slowly, and only then hands over to ``FINAL_ALIGNING``.
        """
        if dist_to_goal <= self._active_pos_tol:
            if abs(yaw_err) <= self._active_yaw_tol:
                self._status = NavigationStatus.GOAL_REACHED
                return VelocityCommand(0.0, 0.0)
            return self._enter_terminal_state(NavigationStatus.FINAL_ALIGNING)

        dx = self._goal_x - robot_x
        dy = self._goal_y - robot_y
        position_yaw = math.atan2(dy, dx)
        position_yaw_err = _wrap_to_pi(position_yaw - robot_yaw)

        # Drive an arc instead of "turn-then-drive": correcting the last few
        # centimetres while simultaneously rotating to the final heading keeps
        # the manoeuvre short, so the chassis stays clear of the table/shelf
        # edge throughout.  Forward motion is cosine-scaled and the heading rate
        # is bounded by terminal_turn_max.
        raw_lin = min(self._terminal_linear_max, 0.8 * dist_to_goal)
        raw_lin *= max(0.0, math.cos(position_yaw_err))
        if raw_lin < 1e-6:
            raw_lin = 0.0
        raw_ang = _clamp(
            self._rot_gain * position_yaw_err,
            -self._terminal_turn_max, self._terminal_turn_max,
        )
        candidate = VelocityCommand(
            linear_x=raw_lin,
            angular_z=raw_ang,
        )

        planning = _planning_surface(self._grid)
        candidate, _static_fallback = self._avoidance.adjust(
            candidate, obs, planning, robot_x, robot_y, robot_yaw,
        )
        gx, gy = planning.world_to_grid(robot_x, robot_y)
        obs_dist = 100.0
        if gx >= 0 and gy >= 0:
            obs_dist = (
                float(planning.distance_transform()[gy, gx])
                * planning.resolution
            )
        # FINAL_POSITIONING has no active path polyline to project onto (the
        # robot is converging on the goal xy, not tracking remaining
        # waypoints).  Use the geometric chord * |sin(e)| form here; B3's
        # ``_lateral_path_error`` applies only while a path is followed.
        path_deviation = dist_to_goal * abs(math.sin(position_yaw_err))
        return self._speed_limiter.limit(candidate, dt, obs_dist, path_deviation)

    def reset(self) -> None:
        """Clear internal state for a fresh navigation segment."""
        self._status = NavigationStatus.IDLE
        self._path = []
        self._waypoint_idx = 0
        self._goal = None
        self._active_pos_tol = self._pos_tol
        self._active_yaw_tol = self._yaw_tol
        self._estop_free_ticks = 0
        self._estop_latch_ticks = 0
        self._brake_ticks = 0
        self._replan_count = 0
        self._rotating_in_place = False
        self._rotate_target_yaw = None
        self._planned_path_length = 0.0
        self._planned_straight = 0.0
        self._speed_limiter.reset()
        self._path_validator.reset()
        self._safety_footprint_mode = self._footprint_mode
        self._emergency.set_footprint_mode(self._safety_footprint_mode)


def _planning_surface(grid: GridLike) -> OccupancyGrid:
    if isinstance(grid, LayeredGrid):
        return grid.planning_grid()
    return grid


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
