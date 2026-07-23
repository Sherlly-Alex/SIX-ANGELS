"""Oscillation critic — penalises frequent left/right yaw reversals."""

from typing import Any

from ..critic import TrajectoryCritic


class OscillationCritic(TrajectoryCritic):
    """Cost for flipping angular-velocity sign relative to the last command."""

    name: str = "oscillation"
    weight: float = 1.5

    def __init__(self, weight: float = 1.5, deadband: float = 0.05):
        self.weight = weight
        self.deadband = deadband

    def prepare(self, context: Any) -> None:
        pass

    def score(self, trajectory, context: Any) -> float:
        if context is None:
            return 0.0
        last_w = float(context.get("last_angular_vel", 0.0) or 0.0)
        w = float(trajectory.angular_vel)
        if abs(last_w) < self.deadband or abs(w) < self.deadband:
            return 0.0
        if last_w * w < 0.0:
            return abs(w) + abs(last_w)
        return 0.0
