"""Prefer-forward critic — favours higher linear velocity, punishes pure spin."""

from typing import Any

from ..critic import TrajectoryCritic


class PreferForwardCritic(TrajectoryCritic):
    """Cost = (max_speed - v) plus extra penalty for low-v high-|w| spin."""

    name: str = "prefer_forward"
    weight: float = 1.0

    def __init__(self, weight: float = 1.0, max_speed: float = 0.30,
                 min_progress_speed: float = 0.06,
                 spin_yaw_threshold: float = 0.25,
                 spin_penalty: float = 2.0):
        self.weight = weight
        self.max_speed = max_speed
        self.min_progress_speed = min_progress_speed
        self.spin_yaw_threshold = spin_yaw_threshold
        self.spin_penalty = spin_penalty

    def prepare(self, context: Any) -> None:
        pass

    def score(self, trajectory, context: Any) -> float:
        if self.max_speed <= 0.0:
            return 0.0
        v = float(trajectory.linear_vel)
        w = float(trajectory.angular_vel)
        cost = max(0.0, self.max_speed - v)
        # Strongly discourage in-place spinning while still far from goal.
        if v < self.min_progress_speed and abs(w) > self.spin_yaw_threshold:
            cost += self.spin_penalty * abs(w)
            cost += self.spin_penalty * (self.min_progress_speed - v)
        return cost
