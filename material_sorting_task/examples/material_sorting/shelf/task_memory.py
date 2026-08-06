"""Shared in-process memory for the three formal task executors."""

from __future__ import annotations

from dataclasses import dataclass

from shelf.state_tracker import ShelfState


@dataclass
class CompetitionTaskMemory:
    """Information that must survive the referee handoff between tasks.

    Executors receive the same instance from ``build_task_executors``.  The
    top-level controller may reset an individual executor between attempts,
    but a successful task-1 origin and shelf scan must remain available to
    task 2.
    """

    task1_origin_world: tuple[float, float, float] | None = None
    task1_color: str | None = None
    shelf_state: ShelfState | None = None

    def record_task1_origin(
        self,
        world_xyz: tuple[float, float, float],
        color: str,
    ) -> None:
        point = tuple(float(value) for value in world_xyz)
        if len(point) != 3:
            raise ValueError("task1 origin must contain exactly three coordinates")
        self.task1_origin_world = point
        self.task1_color = str(color).strip().lower()

    def record_shelf_state(self, state: ShelfState) -> None:
        self.shelf_state = state

    def require_task1_origin(self) -> tuple[float, float, float]:
        if self.task1_origin_world is None:
            raise RuntimeError("task 1 original table coordinate is unavailable")
        return self.task1_origin_world

    def require_shelf_state(self) -> ShelfState:
        if self.shelf_state is None:
            raise RuntimeError("task 1 has not produced a stable shelf-state result")
        return self.shelf_state


__all__ = ["CompetitionTaskMemory"]
