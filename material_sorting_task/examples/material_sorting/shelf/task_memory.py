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
    tasks 2 and 3.
    """

    task1_origin_world: tuple[float, float, float] | None = None
    task1_color: str | None = None
    shelf_state: ShelfState | None = None
    # These are the only shelf-state coordinates shared across tasks.  The
    # desktop material_box and desktop source coordinates are intentionally
    # not part of this cache.
    shelf_empty_center_world: tuple[float, float, float] | None = None
    task2_target_center_world: tuple[float, float, float] | None = None
    task3_packaging_box_center_world: tuple[float, float, float] | None = None

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
        self.shelf_empty_center_world = state.empty_shelf_center_world
        self.task2_target_center_world = state.task2_target_center_world
        self.task3_packaging_box_center_world = (
            state.task3_packaging_box_center_world
        )

    def clear_shelf_state(self) -> None:
        """Discard the previous scene before task 1 starts a fresh epoch.

        The new epoch spans retreat, direct shelf navigation and any
        conditional stationary scan.  A complete stable state may therefore
        be recorded during transport; arriving at the pre-place stand must not
        clear it again.
        """

        self.shelf_state = None
        self.shelf_empty_center_world = None
        self.task2_target_center_world = None
        self.task3_packaging_box_center_world = None

    def require_task1_origin(self) -> tuple[float, float, float]:
        if self.task1_origin_world is None:
            raise RuntimeError("task 1 original table coordinate is unavailable")
        return self.task1_origin_world

    def require_shelf_state(self) -> ShelfState:
        if self.shelf_state is None:
            raise RuntimeError("task 1 has not produced a stable shelf-state result")
        return self.shelf_state

    def require_empty_shelf_center(self) -> tuple[float, float, float]:
        if self.shelf_empty_center_world is None:
            raise RuntimeError("task 1 empty shelf center is unavailable")
        return self.shelf_empty_center_world

    def require_task2_target_center(self) -> tuple[float, float, float]:
        if self.task2_target_center_world is None:
            raise RuntimeError("task 2 shelf target center is unavailable")
        return self.task2_target_center_world

    def require_task3_packaging_box_center(self) -> tuple[float, float, float]:
        if self.task3_packaging_box_center_world is None:
            raise RuntimeError("task 3 packaging-box center is unavailable")
        return self.task3_packaging_box_center_world


__all__ = ["CompetitionTaskMemory"]
