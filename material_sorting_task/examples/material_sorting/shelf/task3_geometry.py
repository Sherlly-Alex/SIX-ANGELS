"""Geometry helpers for task-3 placement beside the shelf packaging box.

The reference packaging-box centre comes from the client-side RGB-D shelf
state.  This module only combines that measured Y coordinate with the
calibrated shelf depth and the detected occupied layer; it never reads Server
ground truth or the instruction's place_world field.
"""

from __future__ import annotations

import math
from typing import Iterable

from shelf_geometry import ShelfGeometry, load_shelf_geometry


TASK3_LEFT_CENTER_OFFSET_M = 0.238
TASK3_SHELF_DEPTH_OFFSET_M = 0.050
TASK3_SAFE_RELEASE_REAR_M = 0.060
TASK3_SAFE_RELEASE_CENTER_INSET_M = 0.040
TASK3_BOX_HALF_Z_M = 0.095


def _finite_point(point: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in point)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("packaging center must contain three finite coordinates")
    return values


def task3_scoring_target(
    packaging_center_world: Iterable[float],
    white_obstacle_layer: int,
    *,
    geometry: ShelfGeometry | None = None,
    left_center_offset_m: float = TASK3_LEFT_CENTER_OFFSET_M,
    shelf_depth_offset_m: float = TASK3_SHELF_DEPTH_OFFSET_M,
) -> tuple[float, float, float]:
    """Compute the nominal colored-box centre left of the white cuboid.

    The shelf faces +X, while the robot approaches it facing west (yaw pi).
    Therefore the robot/world left direction is -Y.  The measured packaging
    X is intentionally not used because a camera can lock the visible front
    face instead of the cuboid centre; shelf depth is calibrated separately.
    """

    center = _finite_point(packaging_center_world)
    layer = int(white_obstacle_layer)
    geom = geometry or load_shelf_geometry()
    left_offset = float(left_center_offset_m)
    depth_offset = float(shelf_depth_offset_m)
    if not math.isfinite(left_offset) or left_offset <= 0.0:
        raise ValueError("left_center_offset_m must be finite and positive")
    if not math.isfinite(depth_offset) or depth_offset < 0.0:
        raise ValueError("shelf_depth_offset_m must be finite and non-negative")
    return (
        float(geom.shelf_xy[0] - depth_offset),
        float(center[1] - left_offset),
        float(
            geom.object_center_z_on_board(
                layer,
                half_z=TASK3_BOX_HALF_Z_M,
            )
        ),
    )


def task3_safe_release_target(
    scoring_target_world: Iterable[float],
    *,
    rear_offset_m: float = TASK3_SAFE_RELEASE_REAR_M,
    center_inset_m: float = TASK3_SAFE_RELEASE_CENTER_INSET_M,
) -> tuple[float, float, float]:
    """Move the release point slightly outward and toward shelf centre.

    The returned point remains within the Server task-3 scoring radius while
    leaving extra clearance from the shelf back and side panel.  A later
    visual push is deliberately optional; the first formal implementation can
    release at this bounded safe point and then retreat.
    """

    target = _finite_point(scoring_target_world)
    rear = float(rear_offset_m)
    inset = float(center_inset_m)
    if not math.isfinite(rear) or rear < 0.0:
        raise ValueError("rear_offset_m must be finite and non-negative")
    if not math.isfinite(inset) or inset < 0.0:
        raise ValueError("center_inset_m must be finite and non-negative")
    return (float(target[0] + rear), float(target[1] + inset), target[2])


__all__ = [
    "TASK3_BOX_HALF_Z_M",
    "TASK3_LEFT_CENTER_OFFSET_M",
    "TASK3_SAFE_RELEASE_CENTER_INSET_M",
    "TASK3_SAFE_RELEASE_REAR_M",
    "TASK3_SHELF_DEPTH_OFFSET_M",
    "task3_safe_release_target",
    "task3_scoring_target",
]
