"""Obstacle adapter — LiDAR-to-ObstacleObservation conversion and degrade.

When a LiDAR scan is available, converts polar ranges/angles into world-frame
Cartesian obstacle points using the robot pose.  When no scan is available, the
adapter explicitly produces an invalid ``ObstacleObservation`` so that downstream
modules can degrade safely to static-grid-only mode.

Non‑finite robot pose returns an invalid observation; non‑finite scan angles
are silently skipped during ray projection.

Expired dynamic obstacles are identified by their ``timestamp`` field; callers
should discard observations older than a configurable expiry period.
"""
from __future__ import annotations

import math
import time as _time
from typing import Optional, Sequence, Tuple

from navigation.navigation_types import ObstacleObservation


class ObstacleAdapter:
    """Converts sensor input to ``ObstacleObservation``.

    The adapter is stateless; expiry decisions are made by callers comparing
    timestamps.
    """

    DEFAULT_MAX_RANGE = 30.0  # rays beyond this are treated as no-return

    def __init__(self, max_range: float = DEFAULT_MAX_RANGE):
        self._max_range = float(max_range)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def from_lidar(
        self,
        ranges: Sequence[float],
        angles: Sequence[float],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        *,
        timestamp: Optional[float] = None,
        frame_id: str = "lidar",
        rospy_timestamp: Optional[float] = None,
    ) -> ObstacleObservation:
        """Convert a full LiDAR scan into world-frame obstacle points.

        Each valid (finite, within max_range) ray produces one world point:
        ``(robot_x + r*cos(yaw+angle), robot_y + r*sin(yaw+angle))``.
        Non‑finite angles and ranges are silently skipped during projection.
        A non‑finite robot pose causes an invalid (``valid=False``) result.

        Parameters
        ----------
        ranges, angles:
            Per-ray range (m) and angle (rad, counter‑clockwise from robot +X).
        robot_x, robot_y, robot_yaw:
            Robot world pose at the time of the scan.
        timestamp:
            Sensor timestamp (s).  Defaults to ``time.time()``.
        rospy_timestamp:
            Alias for *timestamp* (backward compat).  *timestamp* takes
            precedence.
        frame_id:
            Coordinate frame label (e.g. ``"lidar"``).

        Returns
        -------
        ObstacleObservation
        """
        ts = (
            timestamp
            if timestamp is not None
            else (rospy_timestamp if rospy_timestamp is not None else _time.time())
        )

        if not (math.isfinite(robot_x) and math.isfinite(robot_y) and math.isfinite(robot_yaw)):
            return ObstacleObservation(
                frame_id=frame_id,
                ranges=(),
                angles=(),
                points=(),
                timestamp=float(ts),
                valid=False,
            )

        n = min(len(ranges), len(angles))
        raw_ranges = tuple(float(ranges[i]) for i in range(n))
        raw_angles = tuple(float(angles[i]) for i in range(n))

        points: list[Tuple[float, float]] = []
        for i in range(n):
            r = raw_ranges[i]
            a = raw_angles[i]
            if not (math.isfinite(r) and math.isfinite(a) and 0 < r < self._max_range):
                continue
            world_a = robot_yaw + a
            points.append(
                (robot_x + r * math.cos(world_a), robot_y + r * math.sin(world_a))
            )

        return ObstacleObservation(
            frame_id=frame_id,
            ranges=raw_ranges,
            angles=raw_angles,
            points=tuple(points),
            timestamp=float(ts),
            valid=True,
        )

    @staticmethod
    def degraded(timestamp: Optional[float] = None) -> ObstacleObservation:
        """Return an invalid observation for no-LiDAR (static-grid-only) mode.

        Downstream modules recognise ``valid=False`` and fall back to the
        static occupancy grid.
        """
        return ObstacleObservation(
            frame_id="degraded",
            ranges=(),
            angles=(),
            points=(),
            timestamp=float(timestamp if timestamp is not None else _time.time()),
            valid=False,
        )


def is_expired(obs: ObstacleObservation, expiry: float, now: Optional[float] = None) -> bool:
    """Return ``True`` when *obs* is older than *expiry* seconds.

    Invalid observations (valid=False) are never expired — they convey the
    absence of LiDAR, not stale data.
    """
    if not obs.valid:
        return False
    now_ts = now if now is not None else _time.time()
    return (now_ts - obs.timestamp) > expiry
