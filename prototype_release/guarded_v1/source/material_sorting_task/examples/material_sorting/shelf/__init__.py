"""Shelf perception, shared task memory, and manipulation helpers.

The modules in this package are deliberately ROS-free.  The formal
``client_task.py`` remains the only owner of subscriptions and publishers.
"""

from shelf.manipulation import (
    HeldTransportController,
    ReleaseSpreadController,
    ShelfOpenPregraspController,
    SlideHoldController,
)
from shelf.state_tracker import ShelfState, ShelfStateTracker
from shelf.task_memory import CompetitionTaskMemory
from shelf.target_center import StableTargetCenterTracker, TargetCenterEstimate

__all__ = [
    "CompetitionTaskMemory",
    "HeldTransportController",
    "ReleaseSpreadController",
    "ShelfState",
    "ShelfStateTracker",
    "ShelfOpenPregraspController",
    "SlideHoldController",
    "StableTargetCenterTracker",
    "TargetCenterEstimate",
]
