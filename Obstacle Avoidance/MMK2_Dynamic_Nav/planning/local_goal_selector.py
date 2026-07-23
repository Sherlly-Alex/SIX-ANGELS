"""Local goal selector — picks a lookahead point along the global A* path."""

from typing import List, Optional

import numpy as np


def select_local_goal(
    current_pose: tuple,
    global_path: list,
    current_waypoint_index: int,
    lookahead_distance: float = 1.2,
) -> List[float]:
    """Return [x, y] of a point on *global_path* ahead of the waypoint.

    Walks forward from *current_waypoint_index*, accumulating Euclidean
    segment lengths.  When the accumulated distance exceeds
    *lookahead_distance*, it picks a linearly-interpolated point on the
    current segment.  If the remaining path is shorter than the lookahead,
    the final path point is returned.

    Parameters
    ----------
    current_pose : (x, y, yaw) or (x, y)
        Ignored for selection; provided for interface compatibility.
    global_path : list of [x, y] or [x, y, yaw]
    current_waypoint_index : int
        Where to start walking along the path.
    lookahead_distance : float
        Target distance (m) ahead along the path.

    Returns
    -------
    list of [x, y]
        Local goal coordinates.  Never returns None.
    """
    if global_path is None or len(global_path) == 0:
        return [float(current_pose[0]), float(current_pose[1])]

    path = global_path
    idx = max(0, min(current_waypoint_index, len(path) - 1))

    accumulated = 0.0
    for i in range(idx, len(path) - 1):
        a = path[i]
        b = path[i + 1]
        seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if accumulated + seg_len >= lookahead_distance:
            remaining = lookahead_distance - accumulated
            if seg_len < 1e-9:
                return [float(a[0]), float(a[1])]
            t = remaining / seg_len
            return [a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1])]
        accumulated += seg_len

    return [float(path[-1][0]), float(path[-1][1])]
