"""Path-dist critic — penalises deviation from the A* global path."""

from typing import Any

import numpy as np

from ..critic import TrajectoryCritic


class PathDistCritic(TrajectoryCritic):
    """Cost = average shortest distance of trajectory poses to global path."""

    name: str = "path_dist"
    weight: float = 1.0

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def prepare(self, context: Any) -> None:
        pass

    def score(self, trajectory, context: Any) -> float:
        global_path = context.get("global_path")
        if global_path is None or len(global_path) < 2:
            return 0.0

        path_pts = np.asarray(global_path)
        if path_pts.ndim == 2 and path_pts.shape[1] >= 2:
            path_pts = path_pts[:, :2]

        total_dist = 0.0
        n = trajectory.poses.shape[0]
        for i in range(n):
            px = float(trajectory.poses[i, 0])
            py = float(trajectory.poses[i, 1])
            dx = path_pts[:, 0] - px
            dy = path_pts[:, 1] - py
            dists = np.hypot(dx, dy)
            total_dist += float(np.min(dists))
        return total_dist / n
