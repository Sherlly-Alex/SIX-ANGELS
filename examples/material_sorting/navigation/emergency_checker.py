"""Emergency stop checker — front danger → immediate zero speed.

Three detection modes, selected automatically based on the available input:

**LiDAR mode** (``ObstacleObservation.valid=True``):
    Projects world-frame obstacle points into the robot frame and checks whether
    any front‑hemisphere point lies inside ``emergency_distance``.

**Static‑grid mode** (``ObstacleObservation.valid=False`` or no observation):
    Samples the occupancy grid along the robot's forward axis and checks
    whether any cell within ``emergency_distance`` is occupied.  Used as the
    latch authority only when no oriented ``FootprintChecker`` is configured.

**Footprint mode** (optional ``FootprintChecker``):
    Rejects poses (and optional swept velocity commands) whose oriented
    chassis / arm envelope collides with the layered grid.  When a footprint
    checker is present it is the **latch authority** for the no‑LiDAR case —
    the forward ray stays available via ``check_grid`` as a cheap diagnostic /
    fast path, but is *not* OR‑ed into the soft‑estop latch.  Latching on the
    ray alone false‑triggers at legitimate table/shelf stand poses (the
    obstacle sits in the cone while the footprint is still free), after which
    soft‑estop cannot clear and the segment dies.

Both LiDAR and footprint predicates return a boolean emergency flag.
The caller is responsible for issuing a zero‑velocity command and for any
soft‑latch clear logic.  Predictive sweeps go through ``would_collide_command``
and must brake without latching.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

from navigation.footprint_checker import FootprintChecker
from navigation.navigation_types import ObstacleObservation
from navigation.obstacle_adapter import is_expired
from navigation.occupancy_grid import LayeredGrid, OccupancyGrid
from navigation.robot_geometry import FootprintMode

GridLike = Union[OccupancyGrid, LayeredGrid]


class EmergencyChecker:
    """Front‑danger and oriented-footprint detection for emergency stop.

    Parameters
    ----------
    emergency_distance:
        Minimum safe distance (m) for the forward-ray / LiDAR cone.
    front_half_angle:
        Half‑width (rad) of the front cone for LiDAR‑mode filtering.
    grid_sample_step:
        Step size (m) for static‑grid forward sampling.
    footprint_checker:
        Optional oriented-footprint checker.  When set, ``is_emergency`` also
        rejects colliding chassis / arm envelopes, and the forward ray no
        longer participates in the soft‑estop latch (see module docstring).
    footprint_mode:
        Initial envelope mode (updated via ``set_footprint_mode``).
    command_horizon:
        Seconds of forward integration used when a candidate command is
        supplied to ``is_emergency``.
    """

    def __init__(
        self,
        emergency_distance: float = 0.35,
        front_half_angle: float = math.pi / 3,
        grid_sample_step: float = 0.05,
        obs_expiry: float = 2.0,
        footprint_checker: Optional[FootprintChecker] = None,
        footprint_mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        command_horizon: float = 0.35,
    ):
        self._emergency_distance = float(emergency_distance)
        self._front_half_angle = float(front_half_angle)
        self._grid_sample_step = float(grid_sample_step)
        self._obs_expiry = float(obs_expiry)
        self._footprint = footprint_checker
        self._footprint_mode = footprint_mode
        self._command_horizon = float(command_horizon)
        self.last_reason: str = ""

    def set_footprint_mode(self, mode: FootprintMode) -> None:
        self._footprint_mode = mode

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def check(
        self,
        obs: ObstacleObservation,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """Return ``True`` if a LiDAR obstacle is too close in front.

        When *obs* is invalid (degraded mode) ``False`` is returned; the
        caller should fall back to the static‑grid check.
        """
        if not obs.valid:
            return False
        return self._lidar_danger(obs, robot_x, robot_y, robot_yaw)

    def check_grid(
        self,
        grid: OccupancyGrid,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """Return ``True`` if a static‑grid cell ahead is occupied.

        This is the fallback when no LiDAR is available.  With an active
        footprint checker, prefer ``is_emergency`` / ``is_pose_free`` for
        latching — the ray remains a fast diagnostic only.
        """
        return self._grid_danger(grid, robot_x, robot_y, robot_yaw)

    def is_emergency(
        self,
        obs: Optional[ObstacleObservation],
        grid: GridLike,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """Unified latch-worthy emergency check (LiDAR / ray / footprint pose).

        Predictive command sweeps that would collide are reported via
        ``would_collide_command`` instead — callers should brake that tick
        without entering the emergency latch, otherwise reverse-goal arcs
        instantly soft-kill every navigation segment.

        With a configured footprint checker the forward ray is **not** a latch
        source (see module docstring); without one, ray fallback is kept.
        """
        self.last_reason = ""
        planning = _planning_surface(grid)

        if obs is not None and obs.valid and not is_expired(obs, self._obs_expiry):
            if self._lidar_danger(obs, robot_x, robot_y, robot_yaw):
                self.last_reason = "lidar"
                return True
        elif self._footprint is None and self._grid_danger(
            planning, robot_x, robot_y, robot_yaw,
        ):
            # Legacy / tests without oriented footprints: keep the ray latch.
            self.last_reason = "forward_ray"
            return True

        if self._footprint is not None:
            if not self._footprint.is_pose_free(
                grid, robot_x, robot_y, robot_yaw, self._footprint_mode,
            ):
                self.last_reason = "footprint_pose"
                return True
        return False

    def would_collide_command(
        self,
        grid: GridLike,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        linear_x: float,
        angular_z: float,
    ) -> bool:
        """True when the proposed ``(v, ω)`` sweep would hit the footprint map.

        Intended as a one-tick brake, not a latched emergency.
        """
        if self._footprint is None:
            return False
        if abs(linear_x) < 1e-6 and abs(angular_z) < 1e-6:
            return False
        hit = not self._footprint.is_command_free(
            grid, robot_x, robot_y, robot_yaw,
            linear_x, angular_z, self._footprint_mode,
            horizon=self._command_horizon,
        )
        if hit:
            self.last_reason = "footprint_sweep"
        return hit

    def is_clear(
        self,
        obs: Optional[ObstacleObservation],
        grid: GridLike,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """True when neither LiDAR nor footprint pose report danger at rest."""
        return not self.is_emergency(obs, grid, robot_x, robot_y, robot_yaw)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _lidar_danger(
        self,
        obs: ObstacleObservation,
        rx: float,
        ry: float,
        ryaw: float,
    ) -> bool:
        em = self._emergency_distance
        ha = self._front_half_angle
        for px, py in obs.points:
            dx = px - rx
            dy = py - ry
            dist = math.hypot(dx, dy)
            if dist > em:
                continue
            angle = math.atan2(dy, dx) - ryaw
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) < ha:
                return True
        return False

    def _grid_danger(
        self,
        grid: OccupancyGrid,
        rx: float,
        ry: float,
        ryaw: float,
    ) -> bool:
        step = self._grid_sample_step
        em = self._emergency_distance
        n = max(1, int(em / step))
        for i in range(n + 1):
            d = i * step
            x = rx + d * math.cos(ryaw)
            y = ry + d * math.sin(ryaw)
            gx, gy = grid.world_to_grid(x, y)
            # Out-of-bounds samples are ignored (map edge ≠ obstacle). Only
            # in-bounds occupied cells trigger emergency.
            if gx < 0 or gy < 0:
                continue
            if grid.is_occupied(gx, gy):
                return True
        return False


def _planning_surface(grid: GridLike) -> OccupancyGrid:
    if isinstance(grid, LayeredGrid):
        return grid.planning_grid()
    return grid
