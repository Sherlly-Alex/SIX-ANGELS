"""Oriented-footprint collision checks against a layered occupancy grid.

The checker splits the robot envelope into two bands:

* chassis rectangle → queried against the ``chassis`` layer (walls, shelf,
  table, floor-level dynamics);
* arm / carry rectangle → queried against the ``arm`` layer (walls, shelf,
  elevated dynamics).  The table is absent from the arm layer so carry
  envelopes can overfly table stands.

When only a legacy single-layer ``OccupancyGrid`` is supplied the checker
falls back to querying that grid for both bands, which is strictly more
conservative than the layered path (table is then treated as an arm hazard).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from navigation.occupancy_grid import LayeredGrid, OccupancyGrid
from navigation.robot_geometry import (
    CHASSIS,
    FootprintMode,
    OrientedRect,
    rect_for_mode,
    sample_rect_points,
)

GridLike = Union[OccupancyGrid, LayeredGrid]


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_samples(
    yaw0: float,
    yaw1: float,
    step: float,
    direction: Optional[float] = None,
) -> List[float]:
    """Inclusive samples from *yaw0* to *yaw1*.

    *direction* (sign of the commanded angular velocity) selects the arc when
    given.  Without it the shortest arc is used, except for a half-turn where
    ``_wrap_to_pi`` cannot tell the two equal-length arcs apart — pass the
    commanded sign there or the sweep may check the arc the robot never takes.
    """
    delta = _wrap_to_pi(yaw1 - yaw0)
    if abs(delta) < 1e-9:
        return [yaw0]
    if direction is not None and direction != 0.0:
        # Re-project onto the requested turn direction (0, 2π) or (-2π, 0).
        if direction > 0.0 and delta < 0.0:
            delta += 2.0 * math.pi
        elif direction < 0.0 and delta > 0.0:
            delta -= 2.0 * math.pi
    n = max(1, int(math.ceil(abs(delta) / step)))
    return [yaw0 + delta * (i / n) for i in range(n + 1)]


class FootprintChecker:
    """Pose / rotation / swept-command collision checks."""

    def __init__(
        self,
        *,
        sample_step: float = 0.05,
        rotation_step: float = 0.15,
        circumradius_margin: float = 0.02,
        arm_layer_enabled: bool = True,
    ):
        if sample_step <= 0.0:
            raise ValueError("sample_step must be > 0")
        if rotation_step <= 0.0:
            raise ValueError("rotation_step must be > 0")
        self._sample_step = float(sample_step)
        self._rotation_step = float(rotation_step)
        self._circumradius_margin = float(circumradius_margin)
        self._arm_layer_enabled = bool(arm_layer_enabled)

    @property
    def arm_layer_enabled(self) -> bool:
        return self._arm_layer_enabled

    def set_arm_layer_enabled(self, enabled: bool) -> None:
        """Choose which layer the arm / carry rectangle is queried against.

        The arm layer omits the table because it only spans z >= ``ARM_Z_MIN``.
        That is only true while the spine holds the arms (and any payload)
        above the tabletop.  With a low spine the payload really would strike
        the table, so callers must disable the arm layer and fall back to the
        chassis layer, which still contains the table.
        """
        self._arm_layer_enabled = bool(enabled)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_pose_free(
        self,
        grid: GridLike,
        x: float,
        y: float,
        yaw: float,
        mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
    ) -> bool:
        """True when the chassis and arm envelopes are free at ``(x, y, yaw)``.

        Uses a circumradius fast-path on each layer's distance transform, then
        falls back to dense rectangle sampling when the fast-path cannot prove
        clearance.
        """
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return False
        chassis_grid, arm_grid = self._resolve_layers(grid)
        if not self._rect_free(chassis_grid, x, y, yaw, CHASSIS):
            return False
        arm_rect = self._arm_rect(grid, mode)
        if arm_rect is None:
            return True
        return self._rect_free(arm_grid, x, y, yaw, arm_rect)

    def is_rotation_free(
        self,
        grid: GridLike,
        x: float,
        y: float,
        yaw0: float,
        yaw1: float,
        mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        *,
        direction: Optional[float] = None,
    ) -> bool:
        """True when every yaw sample between *yaw0* and *yaw1* is free.

        Covers the in-place rotate sweep that the legacy forward-ray emergency
        checker was blind to.  Pass *direction* (the sign of the commanded
        angular velocity) for half-turns, where the shortest-arc guess is
        ambiguous.

        Runtime soft‑estop / predictive brake go through
        ``is_command_free`` → ``swept_poses`` (which already samples yaw for
        nonzero ``ω``).  ``is_rotation_free`` remains the public pure‑yaw
        helper for unit tests and offline audits — it is not dead, just not
        on the hot control path.
        """
        for yaw in _yaw_samples(yaw0, yaw1, self._rotation_step, direction):
            if not self.is_pose_free(grid, x, y, yaw, mode):
                return False
        return True

    def swept_poses(
        self,
        x: float,
        y: float,
        yaw: float,
        linear_x: float,
        angular_z: float,
        *,
        horizon: float = 0.4,
        dt: float = 0.05,
    ) -> List[Tuple[float, float, float]]:
        """Integrate a constant ``(v, ω)`` command for up to *horizon* seconds.

        Returns the starting pose plus every predicted pose at multiples of
        *dt*.  Non-finite inputs yield an empty list.
        """
        if not all(math.isfinite(v) for v in (x, y, yaw, linear_x, angular_z, horizon, dt)):
            return []
        if horizon <= 0.0 or dt <= 0.0:
            return [(x, y, yaw)]
        poses: List[Tuple[float, float, float]] = [(x, y, yaw)]
        t = 0.0
        cx, cy, cyaw = x, y, yaw
        while t + dt <= horizon + 1e-12:
            cyaw = cyaw + angular_z * dt
            cx = cx + linear_x * math.cos(cyaw) * dt
            cy = cy + linear_x * math.sin(cyaw) * dt
            poses.append((cx, cy, cyaw))
            t += dt
        return poses

    def is_command_free(
        self,
        grid: GridLike,
        x: float,
        y: float,
        yaw: float,
        linear_x: float,
        angular_z: float,
        mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
        *,
        horizon: float = 0.4,
        dt: float = 0.05,
    ) -> bool:
        """True when every pose along the swept command is free."""
        for px, py, pyaw in self.swept_poses(
            x, y, yaw, linear_x, angular_z, horizon=horizon, dt=dt,
        ):
            if not self.is_pose_free(grid, px, py, pyaw, mode):
                return False
        return True

    def min_clearance(
        self,
        grid: GridLike,
        x: float,
        y: float,
        yaw: float,
        mode: FootprintMode = FootprintMode.TRANSIT_STOWED,
    ) -> float:
        """Approximate minimum distance (m) from the envelope to any obstacle.

        Returns 0.0 when the pose is colliding.  Useful for telemetry.
        """
        if not self.is_pose_free(grid, x, y, yaw, mode):
            return 0.0
        chassis_grid, arm_grid = self._resolve_layers(grid)
        c_clear = self._rect_clearance(chassis_grid, x, y, yaw, CHASSIS)
        arm_rect = self._arm_rect(grid, mode)
        if arm_rect is None:
            return c_clear
        a_clear = self._rect_clearance(arm_grid, x, y, yaw, arm_rect)
        return min(c_clear, a_clear)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _resolve_layers(self, grid: GridLike) -> Tuple[OccupancyGrid, OccupancyGrid]:
        if isinstance(grid, LayeredGrid):
            chassis = grid.layer("chassis")
            if not self._arm_layer_enabled:
                # Spine too low for the arm band: the table is a real hazard
                # for the payload, so query the chassis layer for both bands.
                return chassis, chassis
            return chassis, grid.layer("arm")
        # Legacy single-layer path: arm checks reuse the same grid (strict).
        return grid, grid

    def _arm_rect(self, grid: GridLike, mode: FootprintMode) -> Optional[OrientedRect]:
        """Arm rectangle needing a separate query, or ``None`` when redundant.

        A legacy single-layer ``OccupancyGrid`` cannot distinguish table height
        from wall / shelf height, so checking a long arm envelope against it
        would forbid every table approach; only a ``LayeredGrid`` gets the arm
        query.  Modes whose arm rect equals the chassis are already covered by
        the chassis check.
        """
        if not isinstance(grid, LayeredGrid):
            return None
        arm_rect = rect_for_mode(mode)
        if arm_rect == CHASSIS:
            return None
        return arm_rect

    def _rect_free(
        self,
        grid: OccupancyGrid,
        x: float,
        y: float,
        yaw: float,
        rect: OrientedRect,
    ) -> bool:
        gx, gy = grid.world_to_grid(x, y)
        if gx < 0 or gy < 0:
            return False
        # Circumradius fast-path: if the distance transform at the robot
        # centre already exceeds the oriented-rect circumradius, every corner
        # is free and the expensive rasterisation can be skipped.
        dist_cells = float(grid.distance_transform()[gy, gx])
        need = (rect.circumradius + self._circumradius_margin) / grid.resolution
        if dist_cells >= need:
            return True
        for px, py in sample_rect_points(
            rect, x, y, yaw, step=self._sample_step,
        ):
            sgx, sgy = grid.world_to_grid(px, py)
            if sgx < 0 or sgy < 0 or grid.is_occupied(sgx, sgy):
                return False
        return True

    def _rect_clearance(
        self,
        grid: OccupancyGrid,
        x: float,
        y: float,
        yaw: float,
        rect: OrientedRect,
    ) -> float:
        min_cells = float("inf")
        for px, py in sample_rect_points(
            rect, x, y, yaw, step=self._sample_step,
        ):
            sgx, sgy = grid.world_to_grid(px, py)
            if sgx < 0 or sgy < 0:
                return 0.0
            if grid.is_occupied(sgx, sgy):
                return 0.0
            min_cells = min(min_cells, float(grid.distance_transform()[sgy, sgx]))
        if not math.isfinite(min_cells):
            return 0.0
        return min_cells * grid.resolution
