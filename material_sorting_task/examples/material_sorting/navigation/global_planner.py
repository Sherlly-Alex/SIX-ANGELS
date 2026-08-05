"""Deterministic global path planner (A* on an ``OccupancyGrid``).

The planner uses 8-connected A* with a quadratic inflation-cost gradient so
paths naturally stay away from obstacle boundaries.  It is deterministic for a
given grid and performs no randomisation.

All path points are returned in world coordinates (m).  A ``NoPathError`` is
raised when the planner cannot connect start to goal.

The planner does **not** wire into the ROS2 Client, handle dynamic obstacles, or
perform local trajectory generation.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

from navigation.navigation_types import NavigationGoal
from navigation.occupancy_grid import OccupancyGrid

# 8-connected neighbours  (dx, dy, base_cost)
_NEIGHBOURS: List[Tuple[int, int, float]] = [
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (-1, -1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
]


class NoPathError(RuntimeError):
    """Raised when the global planner cannot find a collision-free path."""


class GlobalPlanner:
    """A* global planner on a static occupancy grid."""

    def __init__(self, grid: OccupancyGrid):
        self._grid = grid

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def plan_path(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        *,
        inflation_radius: float = 1.0,
        min_clearance: float = 0.15,
        cost_weight: float = 4.0,
    ) -> List[Tuple[float, float]]:
        """Return a list of ``(x, y)`` waypoints from start to goal.

        Parameters
        ----------
        start_x, start_y:
            World coordinates of the start pose.
        goal_x, goal_y:
            World coordinates of the goal pose.
        inflation_radius:
            Distance (m) beyond which inflation cost returns to 1.0 (baseline).
        min_clearance:
            Distance (m) within which a cell is considered impassable.
        cost_weight:
            Multiplier for the quadratic-cost gradient.

        Returns
        -------
        list of (float, float)
            Ordered waypoints in world coordinates, from start (exclusive) to
            goal (inclusive).  If start and goal are effectively at the same
            position the list contains the single goal waypoint.

        Raises
        ------
        NoPathError
            If no collision-free path exists.
        """
        if not self._validate_finite(start_x, start_y, goal_x, goal_y):
            raise NoPathError("start or goal coordinates are non-finite")

        sgx, sgy = self._grid.world_to_grid(start_x, start_y)
        ggx, ggy = self._grid.world_to_grid(goal_x, goal_y)

        if sgx < 0 or sgy < 0:
            raise NoPathError("start is outside grid bounds")
        if ggx < 0 or ggy < 0:
            raise NoPathError("goal is outside grid bounds")

        if self._grid.is_occupied(sgx, sgy):
            raise NoPathError("start cell is occupied")
        if self._grid.is_occupied(ggx, ggy):
            raise NoPathError("goal cell is occupied")

        # same-cell short-circuit
        if (sgx, sgy) == (ggx, ggy):
            return [(goal_x, goal_y)]

        # distance transform for inflation costs
        dist_map = self._grid.distance_transform()

        came_from: dict = {}
        g_score: dict = {(sgx, sgy): 0.0}
        open_set: list = []
        push_counter = 0
        heapq.heappush(open_set, (self._heuristic(sgx, sgy, ggx, ggy), push_counter, sgx, sgy))

        max_iters = self._grid.width * self._grid.height

        while open_set:
            _f, _counter, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == (ggx, ggy):
                return self._reconstruct(came_from, (ggx, ggy))

            max_iters -= 1
            if max_iters <= 0:
                break

            current_g = g_score[(cx, cy)]
            for dx, dy, base_cost in _NEIGHBOURS:
                nx, ny = cx + dx, cy + dy
                if not self._grid.is_free(nx, ny):
                    continue

                # soft inflation cost from the original (non-inflated) grid
                inflate = self._grid.inflation_cost(
                    nx, ny,
                    inflation_radius=inflation_radius,
                    min_clearance=min_clearance,
                    cost_weight=cost_weight,
                )
                if math.isinf(inflate):
                    continue

                step_cost = base_cost * inflate
                tentative_g = current_g + step_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self._heuristic(nx, ny, ggx, ggy)
                    push_counter += 1
                    heapq.heappush(open_set, (f, push_counter, nx, ny))

        raise NoPathError("no collision-free path found")

    def plan_goal(
        self,
        start_x: float,
        start_y: float,
        goal: NavigationGoal,
        *,
        min_clearance: float = 0.15,
    ) -> List[Tuple[float, float]]:
        """Convenience: plan to a ``NavigationGoal``.

        The per-goal ``safety_radius`` is *not* used as the obstacle clearance
        (it encodes the standoff from the target, not a hardware limit). A
        fixed ``min_clearance`` of 0.15 m is used for safety — the caller may
        override.
        """
        return self.plan_path(
            start_x=start_x,
            start_y=start_y,
            goal_x=goal.x,
            goal_y=goal.y,
            min_clearance=min_clearance,
        )

    def plan_segments(
        self,
        start_x: float,
        start_y: float,
        goals: Sequence[NavigationGoal],
    ) -> List[List[Tuple[float, float]]]:
        """Plan a chain of segments through a sequence of ``NavigationGoal`` s.

        Each segment plans from the previous goal's position (or *start* for the
        first segment) to the next goal.  Returns one waypoint list per
        segment.

        Raises ``NoPathError`` from inside the first failing segment.
        """
        x, y = float(start_x), float(start_y)
        segments = []
        for goal in goals:
            path = self.plan_goal(x, y, goal)
            segments.append(path)
            x, y = goal.x, goal.y
        return segments

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(gx: int, gy: int, goal_gx: int, goal_gy: int) -> float:
        dx = abs(gx - goal_gx)
        dy = abs(gy - goal_gy)
        # Octile distance for 8-connected grid
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    def _reconstruct(
        self, came_from: dict, goal_node: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        path: List[Tuple[int, int]] = []
        node = goal_node
        while node in came_from:
            path.append(node)
            node = came_from[node]
        # path is reversed (goal → start); flip and convert to world
        path.append(node)  # add the start cell
        world_path = []
        for gx, gy in reversed(path):
            world_path.append(self._grid.grid_to_world(gx, gy))
        return world_path

    @staticmethod
    def _validate_finite(*values: float) -> bool:
        return all(math.isfinite(v) for v in values)
