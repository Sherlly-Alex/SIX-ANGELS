"""Emergency stop checker — front danger → immediate zero speed.

Two detection modes, selected automatically based on the available input:

**LiDAR mode** (``ObstacleObservation.valid=True``):
    Projects world-frame obstacle points into the robot frame and checks whether
    any front‑hemisphere point lies inside ``emergency_distance``.

**Static‑grid mode** (``ObstacleObservation.valid=False`` or no observation):
    Samples the occupancy grid along the robot's forward axis and checks
    whether any cell within ``emergency_distance`` is occupied.

Both modes return a boolean emergency flag.  The caller is responsible for
issuing a zero‑velocity command and triggering replanning.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from navigation.navigation_types import ObstacleObservation
from navigation.obstacle_adapter import is_expired
from navigation.occupancy_grid import OccupancyGrid


class EmergencyChecker:
    """Front‑danger detection for emergency stop.

    Parameters
    ----------
    emergency_distance:
        Minimum safe distance (m).  An obstacle inside this radius triggers
        emergency stop.
    front_half_angle:
        Half‑width (rad) of the front cone for LiDAR‑mode filtering.
    grid_sample_step:
        Step size (m) for static‑grid forward sampling.
    """

    def __init__(
        self,
        emergency_distance: float = 0.35,
        front_half_angle: float = math.pi / 3,
        grid_sample_step: float = 0.05,
        obs_expiry: float = 2.0,
    ):
        self._emergency_distance = float(emergency_distance)
        self._front_half_angle = float(front_half_angle)
        self._grid_sample_step = float(grid_sample_step)
        self._obs_expiry = float(obs_expiry)

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

        This is the fallback when no LiDAR is available.
        """
        return self._grid_danger(grid, robot_x, robot_y, robot_yaw)

    def is_emergency(
        self,
        obs: Optional[ObstacleObservation],
        grid: OccupancyGrid,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """Unified emergency check: uses LiDAR if valid and fresh, else the
        static grid.  Expired LiDAR observations are treated as degraded.
        """
        if obs is not None and obs.valid and not is_expired(obs, self._obs_expiry):
            return self._lidar_danger(obs, robot_x, robot_y, robot_yaw)
        return self._grid_danger(grid, robot_x, robot_y, robot_yaw)

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
            angle = (angle + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π)
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
