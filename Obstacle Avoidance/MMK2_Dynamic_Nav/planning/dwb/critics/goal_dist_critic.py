"""Goal-dist critic — penalises distance of trajectory end from local goal."""

from typing import Any

import numpy as np

from ..critic import TrajectoryCritic


class GoalDistCritic(TrajectoryCritic):
    """Cost = Euclidean distance from last trajectory pose to local goal."""

    name: str = "goal_dist"
    weight: float = 1.2

    def __init__(self, weight: float = 1.2):
        self.weight = weight

    def prepare(self, context: Any) -> None:
        pass

    def score(self, trajectory, context: Any) -> float:
        local_goal = context.get("local_goal")
        if local_goal is None:
            return 0.0

        end = trajectory.poses[-1]
        return float(np.hypot(end[0] - local_goal[0],
                              end[1] - local_goal[1]))
