"""Dynamic obstacle overlay for the layered occupancy grid.

The formal Client path receives dynamic obstacles through Client-side
perception detections, including the white fixed-prop classes.  It must not
consume a Server-private random-layout or ground-truth topic.  The
``task_layout`` helpers below remain offline fixture utilities for geometry
tests only; production Client code passes detections only.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from navigation.occupancy_grid import ObstacleVolume

# Movable boxes are 24 × 16 × 19 cm (half 0.12 × 0.08 × 0.095).
_BOX_HALF_XY = (0.12, 0.08)
_BOX_HALF_Z = 0.095


def _rotated_half_extents(
    half_size: Sequence[float],
    euler: Sequence[float],
) -> Tuple[float, float, float]:
    """World-axis half extents of a box with body half extents and XYZ euler.

    ``prop_packaging_box`` is rotated pi/2 about X, which swaps its Y and Z
    extents; taking ``half_size`` verbatim would model the wrong shape.
    """
    hx, hy, hz = (float(v) for v in half_size[:3])
    rx, ry, rz = (float(v) for v in list(euler[:3]) + [0.0] * (3 - len(euler[:3])))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz @ Ry @ Rx; the AABB half extent along each world axis is the
    # absolute row of R dotted with the body half extents.
    rows = (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
    return tuple(
        abs(r[0]) * hx + abs(r[1]) * hy + abs(r[2]) * hz for r in rows
    )  # type: ignore[return-value]


def volumes_from_task_layout(
    task_layout: Optional[Mapping[str, Any]],
) -> List[ObstacleVolume]:
    """Build volumes for fixed white props in an offline fixture layout.

    This function is intentionally not used by the formal Client navigation
    path.  Under ``MATERIAL_RANDOMIZE=1`` the static layout JSON is not
    authoritative and stamping it would create phantom obstacles.
    """
    if not task_layout:
        return []
    out: List[ObstacleVolume] = []
    for prop in task_layout.get("fixed_props", []) or []:
        pos = prop.get("world_position")
        half_size = prop.get("half_size")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue
        if not isinstance(half_size, (list, tuple)) or len(half_size) < 3:
            continue
        euler = prop.get("euler") or (0.0, 0.0, 0.0)
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            hx, hy, hz = _rotated_half_extents(half_size, euler)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, z, hx, hy, hz)):
            continue
        out.append(ObstacleVolume(
            x - hx, x + hx, y - hy, y + hy,
            z_min=z - hz, z_max=z + hz,
            kind="prop",
        ))
    return out


def volumes_from_detections(
    detections: Iterable[Tuple[str, Sequence[float], float]],
    *,
    exclude_color: Optional[str] = None,
) -> List[ObstacleVolume]:
    """Build volumes from ``(color, xyz, score)`` detection tuples.

    ``exclude_color`` skips the currently targeted box so its own stand pose
    is never marked occupied.
    """
    out: List[ObstacleVolume] = []
    hx, hy = _BOX_HALF_XY
    for color, xyz, _score in detections:
        if color == "shelf_empty":
            continue
        if exclude_color is not None and color == exclude_color:
            continue
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            continue
        try:
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        out.append(ObstacleVolume(
            x - hx, x + hx, y - hy, y + hy,
            z_min=z - _BOX_HALF_Z, z_max=z + _BOX_HALF_Z,
            kind="box",
        ))
    return out


def drop_volumes_containing(
    volumes: Iterable[ObstacleVolume],
    x: float,
    y: float,
    *,
    margin: float = 0.0,
) -> List[ObstacleVolume]:
    """Drop volumes whose XY footprint covers ``(x, y)``.

    Stale or mislocalised detections can otherwise stamp an obstacle onto the
    robot's own pose, which latches the emergency stop with no way out.
    """
    out: List[ObstacleVolume] = []
    for vol in volumes:
        if (
            vol.x_min - margin <= x <= vol.x_max + margin
            and vol.y_min - margin <= y <= vol.y_max + margin
        ):
            continue
        out.append(vol)
    return out


def build_nav_overlay(
    task_layout: Optional[Mapping[str, Any]] = None,
    detections: Optional[Iterable[Tuple[str, Sequence[float], float]]] = None,
    *,
    exclude_color: Optional[str] = None,
    robot_xy: Optional[Tuple[float, float]] = None,
    robot_margin: float = 0.25,
) -> List[ObstacleVolume]:
    """Convenience: props + detections for a navigation segment."""
    vols = volumes_from_task_layout(task_layout)
    if detections is not None:
        vols.extend(volumes_from_detections(detections, exclude_color=exclude_color))
    if robot_xy is not None:
        vols = drop_volumes_containing(
            vols, robot_xy[0], robot_xy[1], margin=robot_margin,
        )
    return vols
