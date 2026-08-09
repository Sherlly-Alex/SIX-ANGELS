"""Path validation — line-segment collision checking against the static grid.

Samples waypoint segments at a configurable resolution and checks each sample
point against the inflation-cost map.  Consecutive blocked confirmations are
accumulated so that transient false-positives do not trigger an immediate
replan.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from navigation.occupancy_grid import OccupancyGrid


class PathValidator:
    """Check whether a global-path segment is blocked.

    Uses the same grid and inflation-cost definition as the A* planner so that
    the "blocked" predicate is internally consistent.
    """

    def __init__(
        self,
        *,
        inflation_radius: float = 1.0,
        min_clearance: float = 0.15,
        sample_step: float = 0.05,
        confirm_threshold: int = 3,
    ):
        self._inflation_radius = float(inflation_radius)
        self._min_clearance = float(min_clearance)
        self._sample_step = float(sample_step)
        self._confirm_threshold = int(confirm_threshold)
        self._consecutive_blocked: int = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def check_segment(
        self,
        p1: Sequence[float],
        p2: Sequence[float],
        grid: OccupancyGrid,
    ) -> bool:
        """Return ``True`` if any point between *p1* and *p2* is blocked.

        The segment is sampled uniformly; both endpoints are included.
        Non‑finite endpoint coordinates are treated as *blocked* (unsafe
        input cannot produce a safe result).
        """
        if len(p1) < 2 or len(p2) < 2:
            return True
        if not (_isfinite(p1[0]) and _isfinite(p1[1])):
            return True
        if not (_isfinite(p2[0]) and _isfinite(p2[1])):
            return True
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(length / self._sample_step)))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            gx, gy = grid.world_to_grid(x, y)
            if gx < 0 or gy < 0:
                return True
            cost = grid.inflation_cost(
                gx, gy,
                inflation_radius=self._inflation_radius,
                min_clearance=self._min_clearance,
            )
            if math.isinf(cost):
                return True
        return False

    def path_blocked(
        self,
        path: Sequence[Sequence[float]],
        start_index: int,
        grid: OccupancyGrid,
        lookahead: float = 2.5,
    ) -> bool:
        """Walk *lookahead* metres forward from *start_index* on *path*.

        Returns ``True`` when any segment of that prefix is blocked.
        Non‑finite waypoint coordinates are also treated as blocked.
        Empty or single‑point paths are considered invalid (blocked).
        """
        if len(path) <= 1:
            return True
        accumulated = 0.0
        for i in range(start_index, len(path) - 1):
            a = path[i]
            b = path[i + 1]
            if not (_isfinite(a[0]) and _isfinite(a[1])):
                return True
            if not (_isfinite(b[0]) and _isfinite(b[1])):
                return True
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            remaining = lookahead - accumulated
            if remaining <= 0:
                break
            if seg_len > remaining:
                t = remaining / seg_len
                partial_end = (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                )
                return self.check_segment(a, partial_end, grid)
            if self.check_segment(a, b, grid):
                return True
            accumulated += seg_len
        return False

    def confirm_blocked(
        self,
        path: Sequence[Sequence[float]],
        start_index: int,
        grid: OccupancyGrid,
        lookahead: float = 2.5,
    ) -> bool:
        """Same as ``path_blocked``, but requires *confirm_threshold* consecutive
        blocked calls before reporting ``True``."""
        blocked = self.path_blocked(path, start_index, grid, lookahead)
        if blocked:
            self._consecutive_blocked += 1
        else:
            self._consecutive_blocked = 0
        return self._consecutive_blocked >= self._confirm_threshold

    def reset(self) -> None:
        """Reset the consecutive-blocked counter (e.g. after replan)."""
        self._consecutive_blocked = 0

    @property
    def consecutive_blocked(self) -> int:
        return self._consecutive_blocked


def _isfinite(v: float) -> bool:
    return math.isfinite(v)
