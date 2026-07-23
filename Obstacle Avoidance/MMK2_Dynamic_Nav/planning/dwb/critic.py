"""Abstract critic interface for DWB trajectory scoring."""

from abc import ABC, abstractmethod
from typing import Any


class TrajectoryCritic(ABC):
    """Base class for all DWB trajectory critics.

    Each critic evaluates a candidate trajectory and returns a non-negative
    cost.  An illegal trajectory must return ``float("inf")``.
    """

    name: str = "base"
    weight: float = 1.0

    def prepare(self, context: Any) -> None:
        """Called once per control cycle to cache environment data."""
        pass

    @abstractmethod
    def score(self, trajectory, context: Any) -> float:
        """Return non-negative cost for *trajectory*.

        Return ``float("inf")`` to mark the trajectory as invalid.
        """
        ...
