"""Shelf perception, shared task memory, and manipulation helpers.

The modules in this package are deliberately ROS-free.  The formal
``client_task.py`` remains the only owner of subscriptions and publishers.
"""

from shelf.manipulation import ReleaseSpreadController, SlideHoldController
from shelf.state_tracker import ShelfState, ShelfStateTracker
from shelf.task_memory import CompetitionTaskMemory

__all__ = [
    "CompetitionTaskMemory",
    "ReleaseSpreadController",
    "ShelfState",
    "ShelfStateTracker",
    "SlideHoldController",
]
