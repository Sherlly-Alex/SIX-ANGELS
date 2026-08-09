"""Local-goal selector — picks a lookahead point along the global A* path.

Given the robot's current position and a global waypoint list, walks forward
from the closest waypoint, accumulating Euclidean distances, and returns a
linearly-interpolated point at exactly *lookahead_distance* ahead.

If the remaining path is shorter than the lookahead distance the final goal
is returned.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


def select_local_goal(
    current_x: float,
    current_y: float,
    global_path: Sequence[Sequence[float]],
    *,
    lookahead_distance: float = 1.2,
    closest_index: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Return ``(x, y, yaw)`` of the lookahead target on *global_path*.

    Parameters
    ----------
    current_x, current_y:
        Robot's world position (used to find the nearest waypoint if
        *closest_index* is ``None``).
    global_path:
        Ordered waypoints ``[x, y]`` or ``[x, y, yaw]``.
    lookahead_distance:
        Desired distance (m) ahead along the path.
    closest_index:
        If given, start walking from this waypoint index.  When ``None`` the
        Euclidean nearest waypoint is used (robust to robot drift).

    Returns
    -------
    (x, y, yaw)
        World coordinates and heading toward the next path segment.
        Non‑finite inputs (NaN / Inf) produce a safe fallback of the robot's
        current position with zero heading.
    """
    # guard non‑finite robot pose
    if not (_isfinite(current_x) and _isfinite(current_y)):
        return (0.0, 0.0, 0.0)

    # guard non‑finite path points
    if global_path:
        for wp in global_path:
            if len(wp) >= 2 and not (_isfinite(wp[0]) and _isfinite(wp[1])):
                return (float(current_x), float(current_y), 0.0)

    if not global_path:
        return (float(current_x), float(current_y), 0.0)

    n = len(global_path)

    # find the nearest waypoint if not given
    if closest_index is None:
        best_sq = float("inf")
        best_i = 0
        for i, wp in enumerate(global_path):
            sq = (wp[0] - current_x) ** 2 + (wp[1] - current_y) ** 2
            if sq < best_sq:
                best_sq = sq
                best_i = i
        start_idx = best_i
    else:
        start_idx = max(0, min(closest_index, n - 1))

    accumulated = 0.0
    for i in range(start_idx, n - 1):
        a = global_path[i]
        b = global_path[i + 1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if accumulated + seg_len >= lookahead_distance or seg_len < 1e-9:
            remaining = lookahead_distance - accumulated
            if seg_len < 1e-9:
                yaw = _yaw_toward(a, b) if accumulated == 0 else 0.0
                return (float(a[0]), float(a[1]), yaw)
            t = remaining / seg_len
            lx = a[0] + t * (b[0] - a[0])
            ly = a[1] + t * (b[1] - a[1])
            yaw = _yaw_toward(a, b)
            return (lx, ly, yaw)
        accumulated += seg_len

    # path too short — return the final point
    final = global_path[-1]
    yaw = _yaw_toward(global_path[-2], final) if n >= 2 else 0.0
    return (float(final[0]), float(final[1]), yaw)


def _yaw_toward(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Heading (rad) from *a* towards *b*."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _isfinite(v: float) -> bool:
    return math.isfinite(v)
