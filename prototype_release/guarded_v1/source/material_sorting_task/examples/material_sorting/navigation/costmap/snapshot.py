"""Immutable data models for the scheduler-facing world costmap.

The live :class:`~navigation.costmap.world_costmap.WorldCostmap` is mutable,
but every planning/ranking cycle consumes one versioned snapshot.  Dynamic
obstacle evidence is frozen in the snapshot so a later perception update
cannot change the meaning of an already scored candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from navigation.occupancy_grid import ObstacleVolume


@dataclass(frozen=True)
class AABB:
    """World-aligned 3-D bounding box in metres."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 1.60

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("AABB bounds must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("AABB XY bounds must have positive area")
        if self.z_min >= self.z_max:
            raise ValueError("AABB Z bounds must have positive height")

    @classmethod
    def from_volume(cls, volume: ObstacleVolume) -> "AABB":
        return cls(
            x_min=float(volume.x_min),
            x_max=float(volume.x_max),
            y_min=float(volume.y_min),
            y_max=float(volume.y_max),
            z_min=float(volume.z_min),
            z_max=float(volume.z_max),
        )

    def to_volume(self, *, kind: str = "dynamic") -> ObstacleVolume:
        return ObstacleVolume(
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
            kind=kind,
        )

    def distance_xy(self, x: float, y: float) -> float:
        """Euclidean distance to the XY rectangle (zero inside)."""
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        return math.hypot(dx, dy)


@dataclass(frozen=True, init=False)
class DynamicObstacle:
    """Time-bounded perception evidence used by the dynamic costmap layer.

    ``bounds`` is the preferred constructor input.  ``volume`` is accepted as
    a compatibility convenience for existing ``ObstacleVolume`` producers.
    Confidence is deliberately retained instead of being baked into occupancy:
    lethal geometry and soft risk can therefore use different thresholds.
    """

    bounds: AABB
    confidence: float
    observed_at_s: float
    expires_at_s: float
    source: str
    obstacle_id: str
    label: str
    kind: str

    def __init__(
        self,
        bounds: Optional[AABB | ObstacleVolume] = None,
        confidence: float = 1.0,
        observed_at_s: float = 0.0,
        expires_at_s: float = float("inf"),
        source: str = "perception",
        obstacle_id: str = "",
        label: str = "",
        kind: str = "dynamic",
        *,
        volume: Optional[ObstacleVolume] = None,
    ) -> None:
        selected = bounds if bounds is not None else volume
        if selected is None:
            raise ValueError("DynamicObstacle requires bounds or volume")
        if isinstance(selected, ObstacleVolume):
            if kind == "dynamic" and selected.kind:
                kind = selected.kind
            selected = AABB.from_volume(selected)
        if not isinstance(selected, AABB):
            raise TypeError("bounds must be AABB or ObstacleVolume")

        confidence = float(confidence)
        observed_at_s = float(observed_at_s)
        expires_at_s = float(expires_at_s)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("dynamic obstacle confidence must be within [0, 1]")
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if math.isnan(expires_at_s) or expires_at_s <= observed_at_s:
            raise ValueError("expires_at_s must be later than observed_at_s")

        object.__setattr__(self, "bounds", selected)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "observed_at_s", observed_at_s)
        object.__setattr__(self, "expires_at_s", expires_at_s)
        object.__setattr__(self, "source", str(source))
        object.__setattr__(self, "obstacle_id", str(obstacle_id))
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "kind", str(kind or "dynamic"))

    @property
    def volume(self) -> ObstacleVolume:
        return self.bounds.to_volume(kind=self.kind)

    def active_at(self, timestamp_s: float, *, min_confidence: float = 0.0) -> bool:
        return (
            math.isfinite(timestamp_s)
            and self.observed_at_s <= timestamp_s < self.expires_at_s
            and self.confidence >= min_confidence
        )


@dataclass(frozen=True)
class PathMetrics:
    """Complete deterministic metrics for one planned navigation path."""

    reachable: bool
    path: Tuple[Tuple[float, float], ...] = ()
    path_length_m: float = 0.0
    straight_distance_m: float = 0.0
    detour_ratio: float = 1.0
    min_clearance_m: float = 0.0
    inflation_cost_integral: float = 0.0
    heading_change_rad: float = 0.0
    turn_count: int = 0
    dynamic_risk: float = 0.0
    carried_envelope_safe: bool = True
    failure_reason: str = ""

    @classmethod
    def unreachable(cls, reason: str) -> "PathMetrics":
        return cls(reachable=False, failure_reason=str(reason))

    def finite(self) -> bool:
        numeric = (
            self.path_length_m,
            self.straight_distance_m,
            self.detour_ratio,
            self.min_clearance_m,
            self.inflation_cost_integral,
            self.heading_change_rad,
            self.dynamic_risk,
        )
        return all(math.isfinite(float(value)) for value in numeric)

    # Duck-typing aliases used by telemetry/replay/learning adapters.  The
    # canonical names above retain explicit units.
    @property
    def path_length(self) -> float:
        return self.path_length_m

    @property
    def min_clearance(self) -> float:
        return self.min_clearance_m

    @property
    def inflation_integral(self) -> float:
        return self.inflation_cost_integral

    @property
    def obstacle_integral(self) -> float:
        return self.inflation_cost_integral

    @property
    def heading_change(self) -> float:
        return self.heading_change_rad


def freeze_path(path: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """Return a finite, immutable copy of a path.

    Raises ``ValueError`` instead of letting a NaN silently enter critic
    features.
    """
    result = []
    for point in path:
        if len(point) < 2:
            raise ValueError("path point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("path coordinates must be finite")
        result.append((x, y))
    return tuple(result)


__all__ = ["AABB", "DynamicObstacle", "PathMetrics", "freeze_path"]
