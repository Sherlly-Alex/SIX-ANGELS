"""Oriented rectangular footprints for the MMK2 chassis and arm envelopes.

All extents are body-frame metres relative to ``base_link`` / robot root
(``+X`` forward, ``+Y`` left).  Numbers are derived from the MJCF collision
geoms in ``mmk2.xml`` and endpoint FK under ``INIT_ARM_*`` / carry poses;
``scripts/dump_robot_envelope.py`` and the accompanying unit tests re-check
the envelopes against MuJoCo FK so the constants cannot silently drift.

These footprints are used by the layered collision checker.  They are **not**
fed into A* as an inflation radius — that would make the 0.65 m table
standoff unplannable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class OrientedRect:
    """Axis-aligned rectangle in the robot body frame.

    Attributes
    ----------
    front:
        Extent along +X (m).
    rear:
        Extent along -X (m).
    half_width:
        Half extent along ±Y (m).
    """

    front: float
    rear: float
    half_width: float

    def __post_init__(self) -> None:
        if self.front < 0.0 or self.rear < 0.0 or self.half_width < 0.0:
            raise ValueError(
                f"OrientedRect extents must be >= 0 "
                f"(front={self.front}, rear={self.rear}, "
                f"half_width={self.half_width})"
            )

    @property
    def circumradius(self) -> float:
        """Radius of the smallest circle centred at the origin that covers
        every corner of the rectangle."""
        return math.hypot(max(self.front, self.rear), self.half_width)

    def vertices_body(self) -> Tuple[Tuple[float, float], ...]:
        """Counter-clockwise body-frame vertices starting at front-left."""
        return (
            (self.front, self.half_width),
            (-self.rear, self.half_width),
            (-self.rear, -self.half_width),
            (self.front, -self.half_width),
        )

    def vertices_world(
        self, x: float, y: float, yaw: float
    ) -> Tuple[Tuple[float, float], ...]:
        """Rotate ``vertices_body`` into the world frame at pose ``(x, y, yaw)``."""
        c = math.cos(yaw)
        s = math.sin(yaw)
        out = []
        for bx, by in self.vertices_body():
            out.append((x + c * bx - s * by, y + s * bx + c * by))
        return tuple(out)


class FootprintMode(Enum):
    """Which body-frame envelope the navigation stack should enforce."""

    CHASSIS = "chassis"
    TRANSIT_STOWED = "transit_stowed"
    TRANSIT_CARRY = "transit_carry"
    DOCKING = "docking"


# MJCF mmk2.xml agv_link collision boxes:
#   deck half-extents 0.21 × 0.20, centre at x=-0.015 →
#     body-frame x ∈ [-0.225, 0.195], y ∈ [-0.20, 0.20]
#   caster spheres (radius 0.06) at x=0.13045 / -0.15755
# Rounded up to a conservative chassis footprint:
CHASSIS = OrientedRect(front=0.22, rear=0.23, half_width=0.20)

# INIT_ARM_* FK endpoint ≈ (0.25, ±0.22) in base_link under slide≈0.18; the
# legacy comment claiming 0.41 m was a rough upper bound.  TRANSIT_STOWED is
# sized for the milder TRANSIT_ARM_* tuck (~0.27 m forward, ±0.12 m lateral).
# Applied while navigating empty-handed.
TRANSIT_STOWED = OrientedRect(front=0.32, rear=0.23, half_width=0.18)

# Carry envelope keeps a larger front for the held box (INIT reach + box).
# NAV_TABLE intentionally does not force TRANSIT_ARM — arms stay in hold pose.
TRANSIT_CARRY = OrientedRect(front=0.60, rear=0.23, half_width=0.24)

# Terminal docking uses chassis-only: arms are either retracting or the
# stand-pose geometry already absorbs arm reach via the KnownScene standoff.
DOCKING = CHASSIS

_MODE_TO_RECT = {
    FootprintMode.CHASSIS: CHASSIS,
    FootprintMode.TRANSIT_STOWED: TRANSIT_STOWED,
    FootprintMode.TRANSIT_CARRY: TRANSIT_CARRY,
    FootprintMode.DOCKING: DOCKING,
}


def rect_for_mode(mode: FootprintMode) -> OrientedRect:
    """Return the body-frame rectangle associated with *mode*."""
    try:
        return _MODE_TO_RECT[mode]
    except KeyError as exc:  # pragma: no cover – enum is exhaustive
        raise ValueError(f"unknown footprint mode: {mode!r}") from exc


def sample_rect_points(
    rect: OrientedRect,
    x: float,
    y: float,
    yaw: float,
    *,
    step: float = 0.05,
) -> Sequence[Tuple[float, float]]:
    """Dense world-frame sample points covering the oriented rectangle.

    Used by the footprint checker to query occupancy cells.  *step* should
    match the occupancy-grid resolution so every overlapping cell is hit.
    """
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    c = math.cos(yaw)
    s = math.sin(yaw)
    points = []
    # Inclusive loops over the body-frame AABB.
    nx = max(1, int(math.ceil((rect.front + rect.rear) / step)))
    ny = max(1, int(math.ceil((2.0 * rect.half_width) / step)))
    for ix in range(nx + 1):
        bx = -rect.rear + (rect.front + rect.rear) * (ix / nx)
        for iy in range(ny + 1):
            by = -rect.half_width + (2.0 * rect.half_width) * (iy / ny)
            points.append((x + c * bx - s * by, y + s * bx + c * by))
    return points


def contains_point_body(rect: OrientedRect, bx: float, by: float) -> bool:
    """True when body-frame point ``(bx, by)`` lies inside *rect*."""
    return (-rect.rear <= bx <= rect.front) and (abs(by) <= rect.half_width)


def covers_points(
    rect: OrientedRect,
    points: Iterable[Tuple[float, float]],
) -> bool:
    """True when every body-frame point in *points* lies inside *rect*."""
    return all(contains_point_body(rect, float(px), float(py)) for px, py in points)
