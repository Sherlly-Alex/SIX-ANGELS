"""Shelf structure AABB for dual-arm collision planning (Client-side).

Distinguish:
  shelf_structure_xy — MJCF shelf body origin (posts/boards/back), default (-2.67, 0.778)
  shelf_object_xy    — movable box / prop centres on the shelf (often -2.63, 0.778)

Collision AABBs **must** use ``shelf_structure_xy``. Target object centres stay on
``shelf_object_xy`` / detection — never mix the two as the structure origin.

Production path: structure XY + verified relative size table. MJCF is not parsed
at runtime for body-local chains; tests parse ``material_competition.xml`` and
assert table constants match.

Opening frame: +X = outward (robot side), +Y = lateral, +Z = up.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from shelf_geometry import (
    DEFAULT_SHELF_OPENING_YAW,
    DEFAULT_SHELF_XY,
    ShelfGeometry,
    load_shelf_geometry,
)

_DEFAULT_LAYOUT = Path(__file__).resolve().parent / "material_competition_layout.json"
_DEFAULT_MJCF = Path(__file__).resolve().parent / "mjcf" / "material_competition.xml"

# --- Verified relative size table (MJCF shelf body geom half-sizes / offsets) ---
# board geom size="0.18 0.40 0.01" → depth 0.36, width 0.80, thickness 0.02
# posts at (±0.185, ±0.388); back at x=-0.195
SHELF_STRUCTURE_XY = (-2.67, 0.778)  # MJCF <body name="shelf" pos=...>
SHELF_OBJECT_XY_DEFAULT = DEFAULT_SHELF_XY  # typical box centre (-2.63, 0.778)
MJCF_SHELF_BODY_XY = SHELF_STRUCTURE_XY  # alias
MJCF_BOARD_HALF = (0.18, 0.40, 0.01)
MJCF_POST_HALF = (0.02, 0.02, 1.025)
MJCF_POST_XY = (
    (-0.185, -0.388),
    (0.185, -0.388),
    (-0.185, 0.388),
    (0.185, 0.388),
)
MJCF_BACK_HALF = (0.01, 0.40, 1.025)
MJCF_BACK_POS = (-0.195, 0.0, 1.025)

DEFAULT_SHELF_DEPTH = 2.0 * MJCF_BOARD_HALF[0]  # 0.36
DEFAULT_SHELF_WIDTH = 2.0 * MJCF_BOARD_HALF[1]  # 0.80
DEFAULT_BOARD_THICKNESS = 2.0 * MJCF_BOARD_HALF[2]  # 0.02


class ShelfCollisionGeometryError(ValueError):
    """Missing or inconsistent shelf collision parameters."""


@dataclass(frozen=True)
class AABB:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self) -> None:
        if not (self.xmin <= self.xmax and self.ymin <= self.ymax and self.zmin <= self.zmax):
            raise ShelfCollisionGeometryError(f"invalid AABB bounds: {self}")


@dataclass(frozen=True)
class ShelfCollisionGeometry:
    """``center_xy`` is always the **structure** origin (not object centre)."""

    center_xy: tuple[float, float]
    opening_yaw: float
    width: float
    depth: float
    board_thickness: float
    upright_boxes: tuple[AABB, ...]
    board_boxes: tuple[AABB, ...]
    back_boxes: tuple[AABB, ...]


def _rot_offset(opening_yaw: float, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(opening_yaw), math.sin(opening_yaw)
    return (c * lx - s * ly, s * lx + c * ly)


def _aabb_from_center(
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    opening_yaw: float,
) -> AABB:
    """Axis-aligned world box from opening-frame half extents (yaw about Z only)."""
    c, s = abs(math.cos(opening_yaw)), abs(math.sin(opening_yaw))
    ext_x = c * hx + s * hy
    ext_y = s * hx + c * hy
    return AABB(
        xmin=cx - ext_x,
        xmax=cx + ext_x,
        ymin=cy - ext_y,
        ymax=cy + ext_y,
        zmin=cz - hz,
        zmax=cz + hz,
    )


def build_shelf_collision_geometry(
    *,
    structure_xy: tuple[float, float],
    opening_yaw: float,
    board_surfaces_z: Sequence[float],
    width: float = DEFAULT_SHELF_WIDTH,
    depth: float = DEFAULT_SHELF_DEPTH,
    board_thickness: float = DEFAULT_BOARD_THICKNESS,
) -> ShelfCollisionGeometry:
    """Build AABBs anchored at shelf **structure** origin (MJCF body)."""
    if width <= 0 or depth <= 0 or board_thickness <= 0:
        raise ShelfCollisionGeometryError("width/depth/board_thickness must be > 0")
    if len(board_surfaces_z) < 1:
        raise ShelfCollisionGeometryError("board_surfaces_z must be non-empty")

    ox, oy = float(structure_xy[0]), float(structure_xy[1])
    yaw = float(opening_yaw)
    hx_board = depth * 0.5
    hy_board = width * 0.5
    hz_board = board_thickness * 0.5

    boards: list[AABB] = []
    for z_top in board_surfaces_z:
        z_c = float(z_top)
        boards.append(_aabb_from_center(ox, oy, z_c, hx_board, hy_board, hz_board, yaw))

    uprights: list[AABB] = []
    phx, phy, phz = MJCF_POST_HALF
    for lx, ly in MJCF_POST_XY:
        dx, dy = _rot_offset(yaw, lx, ly)
        uprights.append(
            _aabb_from_center(ox + dx, oy + dy, phz, phx, phy, phz, yaw)
        )

    bx, by, bz = MJCF_BACK_POS
    bhx, bhy, bhz = MJCF_BACK_HALF
    bdx, bdy = _rot_offset(yaw, bx, by)
    backs = (
        _aabb_from_center(ox + bdx, oy + bdy, bz, bhx, bhy, bhz, yaw),
    )

    return ShelfCollisionGeometry(
        center_xy=(ox, oy),
        opening_yaw=yaw,
        width=float(width),
        depth=float(depth),
        board_thickness=float(board_thickness),
        upright_boxes=tuple(uprights),
        board_boxes=tuple(boards),
        back_boxes=backs,
    )


def resolve_shelf_structure_xy(scene: Mapping | None) -> tuple[float, float]:
    """Structure origin from layout.scene or MJCF default — never object centre."""
    scene = scene or {}
    raw = scene.get("shelf_structure_xy")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ShelfCollisionGeometryError(
                "scene.shelf_structure_xy must be [x, y]"
            )
        return (float(raw[0]), float(raw[1]))
    return SHELF_STRUCTURE_XY


def load_shelf_collision_geometry(
    layout_path: str | Path | None = None,
    layout: Mapping | None = None,
    geom: ShelfGeometry | None = None,
) -> ShelfCollisionGeometry:
    """Load collision model on structure XY + relative table; no MJCF runtime parse."""
    if layout is None and geom is None:
        path = Path(layout_path) if layout_path is not None else _DEFAULT_LAYOUT
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)
    if geom is None:
        geom = load_shelf_geometry(layout_path=layout_path, layout=layout)
    scene = (layout or {}).get("scene") or {} if layout is not None else {}

    width = float(scene["shelf_width"]) if scene.get("shelf_width") is not None else DEFAULT_SHELF_WIDTH
    depth = float(scene["shelf_depth"]) if scene.get("shelf_depth") is not None else DEFAULT_SHELF_DEPTH
    thickness = (
        float(scene["board_thickness"])
        if scene.get("board_thickness") is not None
        else DEFAULT_BOARD_THICKNESS
    )
    if width <= 0 or depth <= 0 or thickness <= 0:
        raise ShelfCollisionGeometryError(
            "layout shelf_width/depth/board_thickness must be > 0 when provided"
        )

    structure_xy = resolve_shelf_structure_xy(scene)

    return build_shelf_collision_geometry(
        structure_xy=structure_xy,
        opening_yaw=geom.opening_yaw,
        board_surfaces_z=geom.board_surfaces_z,
        width=width,
        depth=depth,
        board_thickness=thickness,
    )


def parse_mjcf_shelf_structure(mjcf_path: str | Path | None = None) -> dict[str, object]:
    """Parse shelf body/geom fields from competition MJCF for test assertions."""
    path = Path(mjcf_path) if mjcf_path is not None else _DEFAULT_MJCF
    text = path.read_text(encoding="utf-8")
    body_m = re.search(
        r'<body\s+name="shelf"\s+pos="([^"]+)"',
        text,
    )
    if not body_m:
        raise ShelfCollisionGeometryError(f"shelf body not found in {path}")
    body_xyz = tuple(float(x) for x in body_m.group(1).split())
    board_line = re.search(r'name="shelf_board_L1"[^\n]+', text)
    if not board_line:
        raise ShelfCollisionGeometryError("shelf_board_L1 not found")
    line = board_line.group(0)
    size_m = re.search(r'size="([^"]+)"', line)
    pos_m = re.search(r'pos="([^"]+)"', line)
    if not size_m or not pos_m:
        raise ShelfCollisionGeometryError("shelf_board_L1 missing size/pos")
    board_half = tuple(float(x) for x in size_m.group(1).split())
    board_pos = tuple(float(x) for x in pos_m.group(1).split())

    posts: list[tuple[float, float]] = []
    post_half = MJCF_POST_HALF
    for name in ("shelf_post_bl", "shelf_post_br", "shelf_post_fl", "shelf_post_fr"):
        m = re.search(rf'name="{name}"[^\n]+', text)
        if not m:
            raise ShelfCollisionGeometryError(f"{name} not found")
        pm = re.search(r'pos="([^"]+)"', m.group(0))
        sm = re.search(r'size="([^"]+)"', m.group(0))
        if not pm or not sm:
            raise ShelfCollisionGeometryError(f"{name} missing pos/size")
        p = tuple(float(x) for x in pm.group(1).split())
        posts.append((p[0], p[1]))
        post_half = tuple(float(x) for x in sm.group(1).split())

    back_m = re.search(r'name="shelf_back_panel"[^\n]+', text)
    if not back_m:
        raise ShelfCollisionGeometryError("shelf_back_panel not found")
    bsize = re.search(r'size="([^"]+)"', back_m.group(0))
    bpos = re.search(r'pos="([^"]+)"', back_m.group(0))
    if not bsize or not bpos:
        raise ShelfCollisionGeometryError("shelf_back_panel missing size/pos")
    back_half = tuple(float(x) for x in bsize.group(1).split())
    back_pos = tuple(float(x) for x in bpos.group(1).split())

    return {
        "body_xy": (body_xyz[0], body_xyz[1]),
        "board_half": board_half,
        "board_pos_z": board_pos[2],
        "post_xy": tuple(posts),
        "post_half": post_half,
        "back_half": back_half,
        "back_pos": back_pos,
    }


def mjcf_relative_table_assertions() -> dict[str, float]:
    """Module table snapshot (pair with ``parse_mjcf_shelf_structure`` in tests)."""
    return {
        "depth": DEFAULT_SHELF_DEPTH,
        "width": DEFAULT_SHELF_WIDTH,
        "board_thickness": DEFAULT_BOARD_THICKNESS,
        "board_half_x": MJCF_BOARD_HALF[0],
        "board_half_y": MJCF_BOARD_HALF[1],
        "board_half_z": MJCF_BOARD_HALF[2],
        "post_x": abs(MJCF_POST_XY[0][0]),
        "post_y": abs(MJCF_POST_XY[0][1]),
        "structure_x": SHELF_STRUCTURE_XY[0],
        "structure_y": SHELF_STRUCTURE_XY[1],
        "object_x": SHELF_OBJECT_XY_DEFAULT[0],
        "object_y": SHELF_OBJECT_XY_DEFAULT[1],
        "default_opening_yaw": DEFAULT_SHELF_OPENING_YAW,
    }
