"""Static 2-D occupancy grid built from scene layout data.

Coordinate convention (consistent with ``navigation_types`` and the Server):

- +X east, +Y north.
- Grid origin is the world coordinate of cell ``(0, 0)`` (the south-west corner
  of the grid extent).
- ``grid_to_world`` returns the **centre** of the cell (``+0.5`` offset).
- ``world_to_grid`` uses ``math.floor`` so that coordinates just outside the
  origin are correctly mapped to negative indices (out-of-bounds).
- Units: metres (m) for world coordinates, integer cell indices for grid.

All obstacle positions are derived from scene layout data (zones, fixed props).
Structural dimensions that the JSON layout does not explicitly encode (shelf
depth, wall thickness) are annotated with references to the canonical MJCF XML
(``material_competition.xml``).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover – scipy is a runtime dependency
    distance_transform_edt = None


class OccupancyGrid:
    """A 2-D binary occupancy grid.

    ``0`` = free, ``1`` = occupied. The grid is constructed by marking obstacle
    rectangles in world coordinates. Once built, it supports world / grid
    coordinate conversion, inflation (for robot footprint expansion), and a
    Euclidean distance transform for soft-cost gradients used by A*/DWA.
    """

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        resolution: float,
        width: int,
        height: int,
    ):
        if resolution <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"resolution={resolution} width={width} height={height} must be > 0"
            )
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)

        self._grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)
        self._dist_map: np.ndarray | None = None

    # ------------------------------------------------------------------
    # coordinate transforms
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world ``(x, y)`` to the closest grid cell ``(gx, gy)``.

        Uses ``math.floor`` so that world coordinates just short of the origin
        (negative) map to out-of-bounds indices rather than being truncated to
        ``(0, 0)`` by ``int()``.

        Returns ``(-1, -1)`` when the world point lies outside the grid extent.
        """
        gx = math.floor((x - self.origin_x) / self.resolution)
        gy = math.floor((y - self.origin_y) / self.resolution)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return gx, gy
        return -1, -1

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the **centre** of grid cell ``(gx, gy)`` in world coordinates."""
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    # ------------------------------------------------------------------
    # obstacle marking
    # ------------------------------------------------------------------

    def mark_rectangle(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """Mark every cell that overlaps the world-aligned rectangle.

        The first overlapping cell is obtained via ``math.floor`` on the lower
        bound and the last via ``math.ceil - 1`` on the upper bound, so that
        small rectangles just inside a cell boundary are correctly captured and
        rectangles entirely outside the grid are silently ignored.

        Invalidates the cached distance transform.
        """
        gx_start = math.floor((x_min - self.origin_x) / self.resolution)
        gx_end = math.ceil((x_max - self.origin_x) / self.resolution) - 1
        gy_start = math.floor((y_min - self.origin_y) / self.resolution)
        gy_end = math.ceil((y_max - self.origin_y) / self.resolution) - 1

        gx_low = max(0, gx_start)
        gx_high = min(self.width - 1, gx_end)
        gy_low = max(0, gy_start)
        gy_high = min(self.height - 1, gy_end)

        if gx_low > gx_high or gy_low > gy_high:
            return  # entirely outside the grid

        self._grid[gy_low : gy_high + 1, gx_low : gx_high + 1] = 1
        self._dist_map = None

    # ------------------------------------------------------------------
    # cell queries
    # ------------------------------------------------------------------

    def is_occupied(self, gx: int, gy: int) -> bool:
        """True when the cell is explicitly occupied or out of bounds."""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True
        return bool(self._grid[gy, gx])

    def is_free(self, gx: int, gy: int) -> bool:
        """True when the cell is inside bounds and **not** occupied."""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return bool(self._grid[gy, gx] == 0)

    # ------------------------------------------------------------------
    # inflation
    # ------------------------------------------------------------------

    def build_inflated(self, radius: float) -> OccupancyGrid:
        """Return a **new** grid whose occupied cells are enlarged by *radius*.

        Inflation uses the Euclidean distance transform: every cell whose
        distance to the nearest obstacle is ≤ *radius* becomes occupied.
        The output grid has the same origin, resolution and dimensions.
        """
        dist = self.distance_transform()
        inflated_grid = dist <= (radius / self.resolution)
        new = OccupancyGrid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            width=self.width,
            height=self.height,
        )
        new._grid = inflated_grid.astype(np.uint8)
        return new

    # ------------------------------------------------------------------
    # distance transform
    # ------------------------------------------------------------------

    def distance_transform(self) -> np.ndarray:
        """Euclidean distance (in cells) from each free cell to the nearest
        occupied cell.  Occupied cells report distance 0; outer bounds are
        treated as occupied.  Cached (invalidated on ``mark_rectangle``)."""
        if self._dist_map is not None:
            return self._dist_map
        if distance_transform_edt is None:  # pragma: no cover
            raise RuntimeError("scipy is required for distance_transform")
        padded = np.ones((self.height + 2, self.width + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = self._grid
        dist = distance_transform_edt(padded == 0, return_distances=True)
        self._dist_map = dist[1:-1, 1:-1]
        return self._dist_map

    # ------------------------------------------------------------------
    # inflation-cost lookup (for A* soft gradient)
    # ------------------------------------------------------------------

    def inflation_cost(
        self,
        gx: int,
        gy: int,
        inflation_radius: float = 1.0,
        min_clearance: float = 0.2,
        cost_weight: float = 4.0,
    ) -> float:
        """Quadratic-gradient inflation cost for cell ``(gx, gy)``.

        *  ``inf``  when the cell is inside *min_clearance* (impassable).
        *  ``1.0``  when distance ≥ *inflation_radius* (baseline).
        *  Quadratic transition between the two thresholds.

        All radii are in metres; they are converted to cell units internally.
        """
        if self.is_occupied(gx, gy):
            return float("inf")
        dist_map = self.distance_transform()
        dist_cells = float(dist_map[gy, gx])
        min_cells = min_clearance / self.resolution
        inflate_cells = inflation_radius / self.resolution
        if dist_cells <= min_cells:
            return float("inf")
        if dist_cells >= inflate_cells:
            return 1.0
        t = (dist_cells - min_cells) / (inflate_cells - min_cells)
        return 1.0 + cost_weight * (1.0 - t) ** 2

    def __repr__(self) -> str:
        return (
            f"OccupancyGrid(origin=({self.origin_x:.2f},{self.origin_y:.2f}), "
            f"res={self.resolution:.2f}m, {self.width}×{self.height})"
        )


# ------------------------------------------------------------------
# scene-derived obstacle rectangles
# ------------------------------------------------------------------

def scene_static_obstacles(
    scene: Mapping[str, Any],
) -> list[Tuple[float, float, float, float]]:
    """Return ``(xmin, xmax, ymin, ymax)`` obstacle rectangles derived from
    the *scene* dictionary (the ``"scene"`` key of a layout / task layout).

    Primary bounds come from the layout's zone definitions
    (``material_competition_layout.json``).  Structural dimensions that the
    JSON does not explicitly encode (shelf depth, wall thickness) are
    annotated with references to the canonical MJCF
    ``material_competition.xml`` so that every constant is traceable.

    Raises ``KeyError`` when a required zone field is absent — no hardcoded
    coordinate fallback is permitted.

    All rectangles are 2-D floor-level projections (z ≈ 0); objects resting
    *on* the shelf or table at z > 0.5 m are excluded.
    """
    _EPS = 1e-9
    obstacles: list[Tuple[float, float, float, float]] = []

    # ---------- table (MJCF body material_table, lines 105-111) ----------
    tpz = scene.get("table_place_zone")
    if tpz is None:
        raise KeyError("scene.table_place_zone is required for table obstacle")
    tx = tpz.get("x")
    ty = tpz.get("y")
    if tx is None or ty is None or len(tx) < 2 or len(ty) < 2:
        raise KeyError("scene.table_place_zone.x/.y must each have 2 bounds")
    table_xmin, table_xmax = float(tx[0]), float(tx[1])
    table_ymin, table_ymax = float(ty[0]), float(ty[1])
    obstacles.append((table_xmin, table_xmax, table_ymin, table_ymax))

    # ---------- shelf (MJCF body shelf, lines 90-102) ----------
    # picking_zone defines the reachable area east of the shelf.
    # The shelf body (posts + back panel) is west of picking_zone.x[0].
    # MJCF: shelf body centre = (-2.67, 0.778), posts at (±0.185, ±0.388),
    #       back panel at x=-2.865 (body centre - 0.195).
    # The shelf front posts are at x ≈ -2.485, back panel at x ≈ -2.865.
    # picking_zone x[0] = -2.45 is just east of the front posts.
    pz = scene.get("picking_zone")
    if pz is None:
        raise KeyError("scene.picking_zone is required for shelf obstacle")
    px = pz.get("x")
    py = pz.get("y")
    if px is None or py is None or len(px) < 2 or len(py) < 2:
        raise KeyError("scene.picking_zone.x/.y must each have 2 bounds")
    # shelf body x extends from (picking_zone xmin - 0.42) to (picking_zone
    # xmin - 0.02), covering the back panel through the front posts.
    shelf_xmin = float(px[0]) - 0.42  # MJCF: back panel world x ≈ -2.865
    shelf_xmax = float(px[0]) - 0.02  # MJCF: front posts world x ≈ -2.47
    # shelf y: the posts are inset from the zone y bounds
    shelf_ymin = float(py[0]) + 0.39   # MJCF: post centres y ≈ 0.39
    shelf_ymax = float(py[1]) - 0.23   # MJCF: post centres y ≈ 1.19
    obstacles.append((shelf_xmin, shelf_xmax, shelf_ymin, shelf_ymax))

    # ---------- perimeter walls (MJCF lines 54-58) ----------
    # All wall centre coords from MJCF perimeter_walls body:
    #   west_wall   pos="-2.93 1.25 1.0"  size="0.03 1.65 1.0"
    #   east_wall   pos="0.43  1.25 1.0"  size="0.03 1.65 1.0"
    #   south_wall  pos="-1.25 -0.40 1.0" size="1.68 0.03 1.0"
    #   north_wall  pos="-1.25 2.90 1.0" size="1.68 0.03 1.0"
    # Wall thickness: 0.06 m (2 × half-size 0.03).
    # Wall positions are anchored to the union of zone extents plus the MJCF-
    # documented centre-to-edge margins so that the wall always encloses the
    # scene.
    all_x = [table_xmin, table_xmax, float(px[0]), float(px[1])]
    all_y = [table_ymin, table_ymax, float(py[0]), float(py[1])]

    ez = scene.get("end_zone")
    if ez is not None:
        ex = ez.get("x")
        ey = ez.get("y")
        if ex is None or ey is None:
            raise KeyError("scene.end_zone present but missing x/y bounds")
        if len(ex) < 2 or len(ey) < 2:
            raise KeyError("scene.end_zone.x/.y must each have 2 bounds")
        all_x.extend((float(ex[0]), float(ex[1])))
        all_y.extend((float(ey[0]), float(ey[1])))

    cx_min = min(all_x)
    cx_max = max(all_x)
    cy_min = min(all_y)
    cy_max = max(all_y)

    # west wall: MJCF centre x=-2.93, half-size 0.03 → world x∈[-2.96, -2.90]
    ww_left = cx_min - 0.51  # matches MJCF west_wall left edge (-2.96)
    ww_right = ww_left + 0.06
    obstacles.append((ww_left, ww_right, cy_min - 0.08 - _EPS, cy_max + 0.08))
    # east wall: MJCF centre x=0.43, half-size 0.03 → world x∈[0.40, 0.46]
    ew_right = cx_max + 0.14  # matches MJCF east_wall right edge (0.46)
    ew_left = ew_right - 0.06
    obstacles.append((ew_left, ew_right, cy_min - 0.08 - _EPS, cy_max + 0.08))
    # south wall: MJCF centre y=-0.40, half-size 0.03 → world y∈[-0.43, -0.37]
    sw_bottom = cy_min - 0.08  # matches MJCF south_wall bottom edge (-0.43)
    sw_top = sw_bottom + 0.06
    obstacles.append((ww_left - 0.2 - _EPS, ew_right + 0.2, sw_bottom, sw_top))
    # north wall: MJCF centre y=2.90, half-size 0.03 → world y∈[2.87, 2.93]
    nw_top = cy_max + 0.08  # matches MJCF north_wall top edge (2.93)
    nw_bottom = nw_top - 0.06
    obstacles.append((ww_left - 0.2 - _EPS, ew_right + 0.2, nw_bottom, nw_top))

    return obstacles


def build_material_scene_grid(
    scene: Optional[Mapping[str, Any]] = None,
    *,
    resolution: float = 0.05,
    margin: float = 0.3,
) -> OccupancyGrid:
    """Construct an ``OccupancyGrid`` covering the material-sorting scene.

    If *scene* is ``None`` the function falls back to the committed
    ``material_competition_layout.json`` (fixed layout).  Callers should
    prefer to pass the ``"scene"`` key from a concrete task layout so that
    all obstacle bounds are centrally sourced and multi-seed layouts are
    supported.
    """
    if scene is None:
        import json
        from pathlib import Path

        layout_json = (
            Path(__file__).resolve().parents[1]
            / "material_competition_layout.json"
        )
        scene = json.loads(layout_json.read_text(encoding="utf-8")).get("scene", {})

    obstacles = scene_static_obstacles(scene)
    xs = [r[0] for r in obstacles] + [r[1] for r in obstacles]
    ys = [r[2] for r in obstacles] + [r[3] for r in obstacles]
    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin

    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1

    grid = OccupancyGrid(
        origin_x=x_min,
        origin_y=y_min,
        resolution=resolution,
        width=width,
        height=height,
    )
    for rect in obstacles:
        grid.mark_rectangle(*rect)
    return grid
