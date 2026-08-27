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

Height-aware planning uses :class:`LayeredGrid`::

- ``chassis`` layer — z ∈ [0, 1.60]: walls + shelf + table (matches the
  historical single-layer grid cell-for-cell).
- ``arm`` layer — z ∈ [0.75, 1.60]: walls + shelf only.  The tabletop sits at
  z ≤ 0.739 so arm envelopes fly over it; walls and the shelf span both
  height bands and therefore appear in both layers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover – scipy is a runtime dependency
    distance_transform_edt = None


# Layer height bands (m).  Intersected against each ObstacleVolume's z-range.
CHASSIS_Z_MIN = 0.0
CHASSIS_Z_MAX = 1.60
ARM_Z_MIN = 0.75
ARM_Z_MAX = 1.60


@dataclass(frozen=True)
class ObstacleVolume:
    """World-aligned axis AABB with an explicit vertical span.

    ``kind`` is a stable tag used by tests and debug dumps
    (``"table"`` / ``"shelf"`` / ``"wall"`` / ``"dynamic"``…).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    kind: str = "static"

    def intersects_z(self, z_lo: float, z_hi: float) -> bool:
        """True when this volume overlaps the half-open height band
        ``[z_lo, z_hi]`` (degenerate zero-thickness bands are empty)."""
        return self.z_max > z_lo and self.z_min < z_hi

    def as_xy(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


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

    def has_any_occupied(self) -> bool:
        """True when at least one cell is marked occupied."""
        return bool(self._grid.any())

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
# scene-derived obstacle rectangles (with height bands)
# ------------------------------------------------------------------

def scene_static_obstacle_volumes(
    scene: Mapping[str, Any],
) -> list[ObstacleVolume]:
    """Return height-tagged obstacle volumes derived from *scene*.

    Primary planar bounds come from the layout's zone definitions
    (``material_competition_layout.json``).  Structural dimensions that the
    JSON does not explicitly encode (shelf depth, wall thickness, z extents)
    are annotated with references to the canonical MJCF
    ``material_competition.xml`` so that every constant is traceable.

    Raises ``KeyError`` when a required zone field is absent — no hardcoded
    coordinate fallback is permitted.
    """
    _EPS = 1e-9
    volumes: list[ObstacleVolume] = []

    # ---------- table (MJCF body material_table, lines 105-111) ----------
    # Top slab: size 0.83 0.40 0.02 at z=0.719 → z ∈ [0.699, 0.739].
    # Four legs: size 0.03 0.03 0.349 at z=0.349 → z ∈ [0, 0.698].
    # Above z=0.739 the table volume is empty — arm envelopes fly over it.
    tpz = scene.get("table_place_zone")
    if tpz is None:
        raise KeyError("scene.table_place_zone is required for table obstacle")
    tx = tpz.get("x")
    ty = tpz.get("y")
    if tx is None or ty is None or len(tx) < 2 or len(ty) < 2:
        raise KeyError("scene.table_place_zone.x/.y must each have 2 bounds")
    table_xmin, table_xmax = float(tx[0]), float(tx[1])
    table_ymin, table_ymax = float(ty[0]), float(ty[1])
    volumes.append(ObstacleVolume(
        table_xmin, table_xmax, table_ymin, table_ymax,
        z_min=0.0, z_max=0.739, kind="table",
    ))

    # ---------- shelf (MJCF body shelf, lines 90-102) ----------
    # Posts: size 0.02 0.02 1.025 at z=1.025 → z ∈ [0, 2.05].
    # Back panel: size 0.01 0.40 1.025 at z=1.025 → same.
    # Boards reach up to z=2.034.  Spans both chassis and arm bands.
    pz = scene.get("picking_zone")
    if pz is None:
        raise KeyError("scene.picking_zone is required for shelf obstacle")
    px = pz.get("x")
    py = pz.get("y")
    if px is None or py is None or len(px) < 2 or len(py) < 2:
        raise KeyError("scene.picking_zone.x/.y must each have 2 bounds")
    shelf_xmin = float(px[0]) - 0.42  # MJCF: back panel world x ≈ -2.865
    shelf_xmax = float(px[0]) - 0.02  # MJCF: front posts world x ≈ -2.47
    shelf_ymin = float(py[0]) + 0.39   # MJCF: post centres y ≈ 0.39
    shelf_ymax = float(py[1]) - 0.23   # MJCF: post centres y ≈ 1.19
    volumes.append(ObstacleVolume(
        shelf_xmin, shelf_xmax, shelf_ymin, shelf_ymax,
        z_min=0.0, z_max=2.05, kind="shelf",
    ))

    # ---------- perimeter walls (MJCF lines 54-58) ----------
    # Wall half-height 1.0 centred at z=1.0 → z ∈ [0, 2.0].  Spans both bands.
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
    volumes.append(ObstacleVolume(
        ww_left, ww_right, cy_min - 0.08 - _EPS, cy_max + 0.08,
        z_min=0.0, z_max=2.0, kind="wall",
    ))
    # east wall: MJCF centre x=0.43, half-size 0.03 → world x∈[0.40, 0.46]
    ew_right = cx_max + 0.14  # matches MJCF east_wall right edge (0.46)
    ew_left = ew_right - 0.06
    volumes.append(ObstacleVolume(
        ew_left, ew_right, cy_min - 0.08 - _EPS, cy_max + 0.08,
        z_min=0.0, z_max=2.0, kind="wall",
    ))
    # south wall: MJCF centre y=-0.40, half-size 0.03 → world y∈[-0.43, -0.37]
    sw_bottom = cy_min - 0.08  # matches MJCF south_wall bottom edge (-0.43)
    sw_top = sw_bottom + 0.06
    volumes.append(ObstacleVolume(
        ww_left - 0.2 - _EPS, ew_right + 0.2, sw_bottom, sw_top,
        z_min=0.0, z_max=2.0, kind="wall",
    ))
    # north wall: MJCF centre y=2.90, half-size 0.03 → world y∈[2.87, 2.93]
    nw_top = cy_max + 0.08  # matches MJCF north_wall top edge (2.93)
    nw_bottom = nw_top - 0.06
    volumes.append(ObstacleVolume(
        ww_left - 0.2 - _EPS, ew_right + 0.2, nw_bottom, nw_top,
        z_min=0.0, z_max=2.0, kind="wall",
    ))

    return volumes


def scene_static_obstacles(
    scene: Mapping[str, Any],
) -> list[Tuple[float, float, float, float]]:
    """Return ``(xmin, xmax, ymin, ymax)`` obstacle rectangles for the
    historical single-layer (chassis-band) grid.

    Backward-compatible wrapper over :func:`scene_static_obstacle_volumes`.
    Volumes that do not intersect the chassis height band are dropped so
    callers that only consume XY never see arm-only hazards.
    """
    return [
        v.as_xy()
        for v in scene_static_obstacle_volumes(scene)
        if v.intersects_z(CHASSIS_Z_MIN, CHASSIS_Z_MAX)
    ]


def _empty_matching_grid(template: OccupancyGrid) -> OccupancyGrid:
    return OccupancyGrid(
        origin_x=template.origin_x,
        origin_y=template.origin_y,
        resolution=template.resolution,
        width=template.width,
        height=template.height,
    )


def _mark_volumes(
    grid: OccupancyGrid,
    volumes: Sequence[ObstacleVolume],
    z_lo: float,
    z_hi: float,
) -> None:
    for vol in volumes:
        if vol.intersects_z(z_lo, z_hi):
            grid.mark_rectangle(*vol.as_xy())


def _extent_from_volumes(
    volumes: Sequence[ObstacleVolume],
    *,
    resolution: float,
    margin: float,
) -> Tuple[float, float, float, float, int, int]:
    xs = [v.x_min for v in volumes] + [v.x_max for v in volumes]
    ys = [v.y_min for v in volumes] + [v.y_max for v in volumes]
    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin
    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1
    return x_min, x_max, y_min, y_max, width, height


def _load_default_scene() -> Mapping[str, Any]:
    import json
    from pathlib import Path

    layout_json = (
        Path(__file__).resolve().parents[1]
        / "material_competition_layout.json"
    )
    return json.loads(layout_json.read_text(encoding="utf-8")).get("scene", {})


@dataclass
class LayeredGrid:
    """Paired chassis / arm occupancy layers sharing origin and resolution.

    ``chassis`` is cell-for-cell identical to the historical single-layer grid
    produced by :func:`build_material_scene_grid`.  ``arm`` omits the table
    (z ≤ 0.739) so arm envelopes can safely overfly table stands.

    Optional dynamic overlays (boxes / props) are marked into a separate pair
    of layers and OR-ed at query time so they can be cleared and rebuilt on
    every replan without touching the static geometry.
    """

    chassis: OccupancyGrid
    arm: OccupancyGrid

    def __post_init__(self) -> None:
        if (
            self.chassis.origin_x != self.arm.origin_x
            or self.chassis.origin_y != self.arm.origin_y
            or self.chassis.resolution != self.arm.resolution
            or self.chassis.width != self.arm.width
            or self.chassis.height != self.arm.height
        ):
            raise ValueError("chassis and arm layers must share geometry")
        self._dyn_chassis = _empty_matching_grid(self.chassis)
        self._dyn_arm = _empty_matching_grid(self.arm)
        self._dyn_volumes: Tuple[ObstacleVolume, ...] = ()
        self._dyn_version = 0
        self._merge_cache: dict = {}

    # ---------- geometry passthrough ----------

    @property
    def origin_x(self) -> float:
        return self.chassis.origin_x

    @property
    def origin_y(self) -> float:
        return self.chassis.origin_y

    @property
    def resolution(self) -> float:
        return self.chassis.resolution

    @property
    def width(self) -> int:
        return self.chassis.width

    @property
    def height(self) -> int:
        return self.chassis.height

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return self.chassis.world_to_grid(x, y)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return self.chassis.grid_to_world(gx, gy)

    # ---------- planning surface (chassis, with dynamic overlay) ----------

    def planning_grid(self) -> OccupancyGrid:
        """Return a chassis-band grid that includes current dynamic overlays.

        A* and the path validator consume this surface.  **Read-only**: the
        result is cached (and is the static layer itself when no dynamic
        overlay is set), so mutating it would corrupt the static geometry.
        Callers that need to scribble on a grid must build their own.
        """
        return self._merged("chassis")

    def layer(self, name: str) -> OccupancyGrid:
        """Return ``"chassis"`` or ``"arm"`` merged with its dynamic overlay.

        Read-only, for the same reason as :meth:`planning_grid`.
        """
        if name in ("chassis", "arm"):
            return self._merged(name)
        raise ValueError(f"unknown layer name: {name!r}")

    def _merged(self, name: str) -> OccupancyGrid:
        """Cached union of a static layer and its dynamic overlay.

        Merging allocates a grid and discards the cached distance transform, so
        an uncached implementation re-runs the EDT on every footprint query —
        dozens of times per control tick.
        """
        if name == "chassis":
            static, dyn = self.chassis, self._dyn_chassis
        else:
            static, dyn = self.arm, self._dyn_arm
        entry = self._merge_cache.get(name)
        if entry is not None and entry[0] == self._dyn_version:
            return entry[1]
        if not dyn.has_any_occupied():
            merged = static  # no overlay: reuse the static layer and its EDT
        else:
            merged = _empty_matching_grid(static)
            merged._grid = np.maximum(static._grid, dyn._grid)
            merged._dist_map = None
        self._merge_cache[name] = (self._dyn_version, merged)
        return merged

    # ---------- dynamic overlay ----------

    def clear_dynamic(self) -> None:
        """Wipe the dynamic overlay on both layers."""
        if self._dyn_volumes == ():
            return
        self._dyn_chassis = _empty_matching_grid(self.chassis)
        self._dyn_arm = _empty_matching_grid(self.arm)
        self._dyn_volumes = ()
        self._dyn_version += 1

    def mark_dynamic(self, volumes: Sequence[ObstacleVolume]) -> None:
        """OR *volumes* into the dynamic overlay, selecting layers by z-span."""
        if not volumes:
            return
        _mark_volumes(self._dyn_chassis, volumes, CHASSIS_Z_MIN, CHASSIS_Z_MAX)
        _mark_volumes(self._dyn_arm, volumes, ARM_Z_MIN, ARM_Z_MAX)
        self._dyn_volumes = self._dyn_volumes + tuple(volumes)
        self._dyn_version += 1

    def set_dynamic(self, volumes: Sequence[ObstacleVolume]) -> None:
        """Replace the dynamic overlay with *volumes* (clear then mark).

        Re-setting an identical volume set is a no-op, so per-tick refreshes
        from a client do not invalidate the merge cache for free.
        """
        incoming = tuple(volumes)
        if incoming == self._dyn_volumes:
            return
        self.clear_dynamic()
        self.mark_dynamic(incoming)

    @property
    def dynamic_volumes(self) -> Tuple[ObstacleVolume, ...]:
        return self._dyn_volumes

    def __repr__(self) -> str:
        return (
            f"LayeredGrid(origin=({self.origin_x:.2f},{self.origin_y:.2f}), "
            f"res={self.resolution:.2f}m, {self.width}×{self.height})"
        )


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

    The returned grid is the chassis-band projection and is cell-for-cell
    identical to the pre-layered implementation.
    """
    return build_layered_scene_grid(
        scene, resolution=resolution, margin=margin,
    ).chassis


def build_layered_scene_grid(
    scene: Optional[Mapping[str, Any]] = None,
    *,
    resolution: float = 0.05,
    margin: float = 0.3,
) -> LayeredGrid:
    """Construct a :class:`LayeredGrid` for the material-sorting scene.

    ``chassis`` matches :func:`build_material_scene_grid`.  ``arm`` omits the
    table so that arm / carry envelopes can fly over table stands while still
    colliding with walls and the shelf.
    """
    if scene is None:
        scene = _load_default_scene()

    volumes = scene_static_obstacle_volumes(scene)
    x_min, _x_max, y_min, _y_max, width, height = _extent_from_volumes(
        volumes, resolution=resolution, margin=margin,
    )

    chassis = OccupancyGrid(
        origin_x=x_min,
        origin_y=y_min,
        resolution=resolution,
        width=width,
        height=height,
    )
    arm = _empty_matching_grid(chassis)
    _mark_volumes(chassis, volumes, CHASSIS_Z_MIN, CHASSIS_Z_MAX)
    _mark_volumes(arm, volumes, ARM_Z_MIN, ARM_Z_MAX)
    return LayeredGrid(chassis=chassis, arm=arm)