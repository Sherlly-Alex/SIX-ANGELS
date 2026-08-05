"""Local avoidance — velocity redirection and dynamic‑obstacle clearance.

Two operating modes, chosen by the type of obstacle input provided:

**LiDAR mode** (*obstacle_adapter* supplies ``ObstacleObservation(valid=True)``):
    Finds the nearest forward‑hemisphere dynamic obstacle point and computes a
    lateral repulsion that is blended into the angular velocity command.
    Expired observations (older than ``expiry``) are discarded.

**Static‑only mode** (``ObstacleObservation(valid=False)`` or ``None``):
    Relies exclusively on the static occupancy grid and its associated
    inflation layer.  No velocity redirection is applied — the path validator
    and speed limiter already provide static‑collision safety.

The module does **not** implement a DWB controller, Client integration, or
exploration mapping.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from navigation.navigation_types import ObstacleObservation, VelocityCommand
from navigation.occupancy_grid import OccupancyGrid
from navigation.obstacle_adapter import is_expired


class LocalAvoidance:
    """Applies obstacle‑aware velocity adjustment.

    Parameters
    ----------
    repulsion_gain:
        Scaling factor for the angular repulsion from the nearest obstacle.
    max_repulsion_angle:
        Maximum angular velocity (rad/s) contributed by repulsion.
    front_half_angle:
        Only obstacles within this half‑angle (rad) of the robot heading
        are considered for repulsion.
    min_obstacle_dist:
        Obstacles farther than this (m) are ignored for repulsion.
    expiry:
        Dynamic observation expiry time (s).
    """

    def __init__(
        self,
        repulsion_gain: float = 2.0,
        max_repulsion_angle: float = 0.6,
        front_half_angle: float = math.pi / 2,
        min_obstacle_dist: float = 1.5,
        expiry: float = 2.0,
    ):
        self._repulsion_gain = float(repulsion_gain)
        self._max_repulsion_angle = float(max_repulsion_angle)
        self._front_half_angle = float(front_half_angle)
        self._min_obstacle_dist = float(min_obstacle_dist)
        self._expiry = float(expiry)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def adjust(
        self,
        candidate: VelocityCommand,
        obs: Optional[ObstacleObservation],
        grid: OccupancyGrid,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> Tuple[VelocityCommand, bool]:
        """Return ``(adjusted_velocity, static_only_fallback)``.

        * LiDAR available (*obs*.valid=True, not expired):
          computes angular repulsion away from the nearest forward obstacle.
        * LiDAR degraded or absent:
          ``static_only_fallback=True``; the candidate velocity is returned
          unchanged (static‑grid safety is handled by the path validator and
          speed limiter).

        Parameters
        ----------
        candidate:
            Pre‑validated velocity (already through SpeedLimiter).
        obs:
            Latest obstacle observation, or ``None`` / invalid for static‑only.
        grid:
            Static occupancy grid (used for mode detection only).
        robot_x, robot_y, robot_yaw:
            Current robot pose.
        """
        if obs is None or not obs.valid or is_expired(obs, self._expiry):
            return (candidate, True)

        # find the nearest in‑cone dynamic point
        nearest_dist, nearest_angle = self._nearest_dynamic(
            obs, robot_x, robot_y, robot_yaw
        )
        if nearest_dist is None or nearest_dist > self._min_obstacle_dist:
            return (candidate, False)

        # repulsion: turn away from the nearest obstacle
        # obstacle on left (angle > 0) → steer right (angular < 0)
        sign = -1.0 if nearest_angle > 0 else 1.0
        # stronger repulsion when closer
        strength = 1.0 - min(nearest_dist / self._min_obstacle_dist, 1.0)
        repulsion = sign * strength * self._repulsion_gain * self._max_repulsion_angle
        repulsion = max(-self._max_repulsion_angle, min(repulsion, self._max_repulsion_angle))

        # Blend repulsion into angular; do not re-clamp the total yaw rate to
        # max_repulsion_angle (that would starve legitimate heading control).
        new_ang = candidate.angular_z + repulsion
        # Scale linear speed down when a forward obstacle is close.
        new_lin = candidate.linear_x * (1.0 - 0.85 * strength)

        adjusted = VelocityCommand(linear_x=new_lin, angular_z=new_ang)
        return (adjusted, False)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _nearest_dynamic(
        self,
        obs: ObstacleObservation,
        rx: float,
        ry: float,
        ryaw: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(nearest_dist, relative_angle)`` of the closest in‑cone
        dynamic point, or ``(None, None)`` if none qualifies."""
        best_dist = float("inf")
        best_angle: Optional[float] = None
        for px, py in obs.points:
            dx = px - rx
            dy = py - ry
            dist = math.hypot(dx, dy)
            if dist > self._min_obstacle_dist:
                continue
            angle = math.atan2(dy, dx) - ryaw
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) > self._front_half_angle:
                continue
            if dist < best_dist:
                best_dist = dist
                best_angle = angle
        if best_angle is None:
            return (None, None)
        return (best_dist, best_angle)
