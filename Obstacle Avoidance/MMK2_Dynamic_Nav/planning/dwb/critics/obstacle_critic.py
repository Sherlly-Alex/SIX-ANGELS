"""Obstacle critic — penalises trajectories that are near obstacles."""

from typing import Any

import numpy as np

from ..critic import TrajectoryCritic


class ObstacleCritic(TrajectoryCritic):
    """Reject trajectories inside unsafe cells; soft-cost others by proximity."""

    name: str = "obstacle"
    weight: float = 3.0

    def __init__(self, weight: float = 3.0):
        self.weight = weight

    def prepare(self, context: Any) -> None:
        pass

    def score(self, trajectory, context: Any) -> float:
        grid = context.get("planning_grid")
        robot_radius = context.get("robot_radius", 0.22)
        safety_margin = context.get("safety_margin", 0.15)
        safe_dist = robot_radius + safety_margin
        safe_cells = safe_dist / grid.resolution

        total_cost = 0.0
        for i in range(trajectory.poses.shape[0]):
            x = float(trajectory.poses[i, 0])
            y = float(trajectory.poses[i, 1])
            mx, my = grid.world_to_map(x, y)
            if mx < 0:
                return float("inf")

            cost = grid.get_inflation_cost(
                mx, my,
                int(np.ceil(safe_cells)),
                min_clearance=0.0,
            )
            if not np.isfinite(cost):
                return float("inf")
            # Accumulate the excess over the baseline (1.0 = completely clear)
            total_cost += cost - 1.0

        return total_cost / max(1, trajectory.poses.shape[0])
