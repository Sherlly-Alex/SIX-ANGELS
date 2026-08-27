"""Versioned layered world costmap for deterministic macro-action scoring.

This module deliberately wraps the navigation primitives already used by the
project.  It adds lifecycle and snapshot semantics; it does not introduce a
second planner or a ROS dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

from navigation.carried_envelope import CarriedEnvelopeChecker
from navigation.dynamic_overlay import volumes_from_detections
from navigation.footprint_checker import FootprintChecker
from navigation.global_planner import GlobalPlanner, NoPathError
from navigation.occupancy_grid import (
    LayeredGrid,
    ObstacleVolume,
    OccupancyGrid,
    build_layered_scene_grid,
)
from navigation.robot_geometry import FootprintMode

from .snapshot import AABB, DynamicObstacle, PathMetrics, freeze_path


PoseTuple = Tuple[float, float, float]


def _finite_pose(pose: Sequence[float]) -> PoseTuple:
    if len(pose) < 3:
        raise ValueError("pose must contain x, y and yaw")
    result = (float(pose[0]), float(pose[1]), float(pose[2]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("pose must be finite")
    return result


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _coerce_footprint_mode(mode: FootprintMode | str) -> FootprintMode:
    if isinstance(mode, FootprintMode):
        return mode
    try:
        return FootprintMode(str(mode))
    except ValueError as exc:
        raise ValueError(f"unknown footprint mode: {mode!r}") from exc


@dataclass(frozen=True)
class WorldCostmapSnapshot:
    """One coherent, read-only-by-contract view of the world.

    The contained ``LayeredGrid`` is private to this snapshot's dynamic layer;
    subsequent calls to :class:`WorldCostmap` never mutate it.  As with the
    project's existing ``LayeredGrid.planning_grid()``, returned grid objects
    are read-only by contract and must not be scribbled on by callers.
    """

    version: int
    timestamp_s: float
    layered_grid: LayeredGrid
    dynamic_obstacles: Tuple[DynamicObstacle, ...]
    lethal_confidence: float

    @property
    def active_obstacles(self) -> Tuple[DynamicObstacle, ...]:
        return self.dynamic_obstacles

    @property
    def lethal_obstacles(self) -> Tuple[DynamicObstacle, ...]:
        return tuple(
            obstacle
            for obstacle in self.dynamic_obstacles
            if obstacle.confidence >= self.lethal_confidence
        )

    def planning_grid(self) -> OccupancyGrid:
        return self.layered_grid.planning_grid()

    def layer(self, name: str) -> OccupancyGrid:
        return self.layered_grid.layer(name)

    def plan_path(
        self,
        start_pose: Sequence[float],
        goal_pose: Sequence[float],
        *,
        footprint_mode: FootprintMode | str = FootprintMode.TRANSIT_STOWED,
        inflation_radius: float = 1.0,
        min_clearance: float = 0.22,
        cost_weight: float = 4.0,
        held_center_base: Optional[Tuple[float, float, float]] = None,
        held_half_width_m: Optional[float] = None,
        carried_checker: Optional[CarriedEnvelopeChecker] = None,
    ) -> PathMetrics:
        """Plan once and return path plus all deterministic critic features."""
        try:
            start = _finite_pose(start_pose)
            goal = _finite_pose(goal_pose)
            inflation_radius = float(inflation_radius)
            min_clearance = float(min_clearance)
            cost_weight = float(cost_weight)
            mode = _coerce_footprint_mode(footprint_mode)
        except (TypeError, ValueError) as exc:
            return PathMetrics.unreachable(str(exc))
        if not all(
            math.isfinite(value)
            for value in (inflation_radius, min_clearance, cost_weight)
        ):
            return PathMetrics.unreachable("planner parameters must be finite")
        if inflation_radius <= 0.0 or min_clearance < 0.0 or cost_weight < 0.0:
            return PathMetrics.unreachable("invalid planner clearance/cost parameters")

        planner = GlobalPlanner(self.planning_grid())
        try:
            path = planner.plan_path(
                start[0],
                start[1],
                goal[0],
                goal[1],
                inflation_radius=inflation_radius,
                min_clearance=min_clearance,
                cost_weight=cost_weight,
            )
            frozen_path = freeze_path(path)
        except (NoPathError, TypeError, ValueError) as exc:
            return PathMetrics.unreachable(str(exc))

        return self.evaluate_path(
            start,
            frozen_path,
            goal_yaw=goal[2],
            footprint_mode=mode,
            inflation_radius=inflation_radius,
            cost_weight=cost_weight,
            held_center_base=held_center_base,
            held_half_width_m=held_half_width_m,
            carried_checker=carried_checker,
        )

    def evaluate_path(
        self,
        start_pose: Sequence[float],
        path_xy: Sequence[Tuple[float, float]],
        *,
        goal_yaw: float,
        footprint_mode: FootprintMode | str = FootprintMode.TRANSIT_STOWED,
        inflation_radius: float = 1.0,
        cost_weight: float = 4.0,
        held_center_base: Optional[Tuple[float, float, float]] = None,
        held_half_width_m: Optional[float] = None,
        carried_checker: Optional[CarriedEnvelopeChecker] = None,
    ) -> PathMetrics:
        """Measure a supplied path and enforce footprint/carried hard safety."""
        try:
            start = _finite_pose(start_pose)
            path = freeze_path(path_xy)
            goal_yaw = float(goal_yaw)
            mode = _coerce_footprint_mode(footprint_mode)
            inflation_radius = float(inflation_radius)
            cost_weight = float(cost_weight)
        except (TypeError, ValueError) as exc:
            return PathMetrics.unreachable(str(exc))
        if not path:
            return PathMetrics.unreachable("navigation path is empty")
        if not all(
            math.isfinite(value)
            for value in (goal_yaw, inflation_radius, cost_weight)
        ):
            return PathMetrics.unreachable("path metric inputs must be finite")

        points = [(start[0], start[1])]
        points.extend(path)
        # Avoid zero-length duplicates without changing the exported planner path.
        metric_points = [points[0]]
        for point in points[1:]:
            if math.hypot(
                point[0] - metric_points[-1][0],
                point[1] - metric_points[-1][1],
            ) > 1e-12:
                metric_points.append(point)

        path_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(metric_points, metric_points[1:])
        )
        goal_xy = path[-1]
        straight = math.hypot(goal_xy[0] - start[0], goal_xy[1] - start[1])
        detour = path_length / straight if straight > 1e-9 else 1.0

        checker = FootprintChecker()
        current_yaw = start[2]
        min_footprint_clearance = float("inf")
        heading_change = 0.0
        turn_count = 0
        segment_headings = []
        for a, b in zip(metric_points, metric_points[1:]):
            segment_headings.append(math.atan2(b[1] - a[1], b[0] - a[0]))
        headings = segment_headings + [goal_yaw]

        # Check each in-place heading transition and every A* sample pose.
        for index, heading in enumerate(headings):
            pivot = metric_points[min(index, len(metric_points) - 1)]
            delta = abs(_wrap_to_pi(heading - current_yaw))
            heading_change += delta
            if delta > 0.15:
                turn_count += 1
            if not checker.is_rotation_free(
                self.layered_grid,
                pivot[0],
                pivot[1],
                current_yaw,
                heading,
                mode,
            ):
                return PathMetrics.unreachable(
                    f"footprint rotation collision near ({pivot[0]:.3f}, {pivot[1]:.3f})"
                )
            current_yaw = heading

        pose_headings = [start[2]]
        pose_headings.extend(segment_headings)
        while len(pose_headings) < len(metric_points):
            pose_headings.append(goal_yaw)
        for (x, y), yaw in zip(metric_points, pose_headings):
            clearance = checker.min_clearance(self.layered_grid, x, y, yaw, mode)
            if clearance <= 0.0:
                return PathMetrics.unreachable(
                    f"footprint collision near ({x:.3f}, {y:.3f})"
                )
            min_footprint_clearance = min(min_footprint_clearance, clearance)

        if not math.isfinite(min_footprint_clearance):
            min_footprint_clearance = 0.0

        grid = self.planning_grid()
        inflation_integral = 0.0
        for a, b in zip(metric_points, metric_points[1:]):
            segment_length = math.hypot(b[0] - a[0], b[1] - a[1])
            gx, gy = grid.world_to_grid(b[0], b[1])
            if gx < 0 or gy < 0:
                return PathMetrics.unreachable("path leaves costmap bounds")
            cell_cost = grid.inflation_cost(
                gx,
                gy,
                inflation_radius=inflation_radius,
                min_clearance=0.0,
                cost_weight=cost_weight,
            )
            if not math.isfinite(cell_cost):
                return PathMetrics.unreachable("path intersects lethal cost")
            inflation_integral += segment_length * cell_cost

        dynamic_risk = self._dynamic_path_risk(metric_points, path_length)

        envelope_safe = True
        if (held_center_base is None) != (held_half_width_m is None):
            return PathMetrics.unreachable(
                "held_center_base and held_half_width_m must be provided together"
            )
        if held_center_base is not None and held_half_width_m is not None:
            try:
                carry = carried_checker or CarriedEnvelopeChecker()
                carry_result = carry.check_path(
                    start,
                    path,
                    goal_yaw,
                    held_center_base,
                    float(held_half_width_m),
                )
            except (TypeError, ValueError) as exc:
                return PathMetrics.unreachable(str(exc))
            envelope_safe = bool(carry_result.safe)
            if not envelope_safe:
                return PathMetrics.unreachable(carry_result.detail)
            min_footprint_clearance = min(
                min_footprint_clearance,
                max(0.0, float(carry_result.clearance_m)),
            )

        metrics = PathMetrics(
            reachable=True,
            path=path,
            path_length_m=path_length,
            straight_distance_m=straight,
            detour_ratio=detour,
            min_clearance_m=min_footprint_clearance,
            inflation_cost_integral=inflation_integral,
            heading_change_rad=heading_change,
            turn_count=turn_count,
            dynamic_risk=dynamic_risk,
            carried_envelope_safe=envelope_safe,
        )
        if not metrics.finite():
            return PathMetrics.unreachable("non-finite path metrics")
        return metrics

    def _dynamic_path_risk(
        self,
        points: Sequence[Tuple[float, float]],
        path_length: float,
    ) -> float:
        if not self.dynamic_obstacles or not points:
            return 0.0
        weighted = 0.0
        denominator = max(path_length, 1e-9)
        pairs = list(zip(points, points[1:]))
        if not pairs:
            pairs = [(points[0], points[0])]
        for a, b in pairs:
            ds = math.hypot(b[0] - a[0], b[1] - a[1])
            if ds <= 0.0:
                ds = 1.0 if path_length <= 1e-9 else 0.0
            local = 0.0
            for obstacle in self.dynamic_obstacles:
                distance = obstacle.bounds.distance_xy(b[0], b[1])
                local += obstacle.confidence * math.exp(-distance / 0.40)
            weighted += ds * local
        if path_length <= 1e-9:
            denominator = 1.0
        return weighted / denominator


class WorldCostmap:
    """Own dynamic evidence and produce atomic scheduler-facing snapshots."""

    def __init__(
        self,
        layered_grid: Optional[LayeredGrid] = None,
        *,
        scene: Optional[Mapping[str, object]] = None,
        resolution: float = 0.05,
        margin: float = 0.3,
        dynamic_ttl_s: float = 1.0,
        lethal_confidence: float = 0.50,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if layered_grid is not None and scene is not None:
            raise ValueError("provide layered_grid or scene, not both")
        dynamic_ttl_s = float(dynamic_ttl_s)
        lethal_confidence = float(lethal_confidence)
        if not math.isfinite(dynamic_ttl_s) or dynamic_ttl_s <= 0.0:
            raise ValueError("dynamic_ttl_s must be finite and > 0")
        if not math.isfinite(lethal_confidence) or not 0.0 <= lethal_confidence <= 1.0:
            raise ValueError("lethal_confidence must be within [0, 1]")
        self._base = layered_grid or build_layered_scene_grid(
            scene, resolution=resolution, margin=margin,
        )
        self._base_dynamic = tuple(self._base.dynamic_volumes)
        self._dynamic_ttl_s = dynamic_ttl_s
        self._lethal_confidence = lethal_confidence
        self._clock = clock
        self._obstacles: Tuple[DynamicObstacle, ...] = ()
        self._version = 0
        self._lock = threading.RLock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def dynamic_ttl_s(self) -> float:
        return self._dynamic_ttl_s

    @property
    def lethal_confidence(self) -> float:
        return self._lethal_confidence

    def clear_dynamic(self, *, source: Optional[str] = None) -> None:
        with self._lock:
            if source is None:
                incoming: Tuple[DynamicObstacle, ...] = ()
            else:
                incoming = tuple(item for item in self._obstacles if item.source != source)
            if incoming != self._obstacles:
                self._obstacles = incoming
                self._version += 1

    def set_dynamic(
        self,
        obstacles: Iterable[DynamicObstacle | ObstacleVolume],
        *,
        observed_at_s: Optional[float] = None,
        ttl_s: Optional[float] = None,
        source: str = "perception",
    ) -> None:
        """Atomically replace dynamic evidence.

        Bare ``ObstacleVolume`` objects receive confidence 1.0 and the supplied
       /default TTL, which makes this a direct upgrade path for the existing
        ``dynamic_overlay`` producers.
        """
        now = self._now(observed_at_s)
        ttl = self._ttl(ttl_s)
        converted = tuple(
            self._coerce_obstacle(item, now=now, ttl=ttl, source=source, index=index)
            for index, item in enumerate(obstacles)
        )
        converted = self._sorted(converted)
        with self._lock:
            if converted != self._obstacles:
                self._obstacles = converted
                self._version += 1

    def update_dynamic(
        self,
        obstacles: Iterable[DynamicObstacle | ObstacleVolume],
        *,
        observed_at_s: Optional[float] = None,
        ttl_s: Optional[float] = None,
        source: str = "perception",
    ) -> None:
        """Upsert evidence by ``(source, obstacle_id/signature)``."""
        now = self._now(observed_at_s)
        ttl = self._ttl(ttl_s)
        incoming = tuple(
            self._coerce_obstacle(item, now=now, ttl=ttl, source=source, index=index)
            for index, item in enumerate(obstacles)
        )
        with self._lock:
            merged = {self._obstacle_key(item): item for item in self._obstacles}
            for item in incoming:
                merged[self._obstacle_key(item)] = item
            updated = self._sorted(tuple(merged.values()))
            if updated != self._obstacles:
                self._obstacles = updated
                self._version += 1

    def replace_dynamic_source(
        self,
        obstacles: Iterable[DynamicObstacle | ObstacleVolume],
        *,
        source: str,
        observed_at_s: Optional[float] = None,
        ttl_s: Optional[float] = None,
    ) -> None:
        """Atomically replace one producer's evidence and preserve all others."""
        source_name = str(source).strip()
        if not source_name:
            raise ValueError("dynamic obstacle source must be non-empty")
        now = self._now(observed_at_s)
        ttl = self._ttl(ttl_s)
        converted = tuple(
            self._coerce_obstacle(
                item,
                now=now,
                ttl=ttl,
                source=source_name,
                index=index,
            )
            for index, item in enumerate(obstacles)
        )
        if any(item.source != source_name for item in converted):
            raise ValueError("all replacement obstacles must match source")
        with self._lock:
            retained = tuple(
                item for item in self._obstacles if item.source != source_name
            )
            updated = self._sorted(retained + converted)
            if updated != self._obstacles:
                self._obstacles = updated
                self._version += 1

    def observe_detections(
        self,
        detections: Iterable[Tuple[str, Sequence[float], float]],
        *,
        observed_at_s: Optional[float] = None,
        ttl_s: Optional[float] = None,
        source: str = "detector",
        exclude_color: Optional[str] = None,
        replace_source: bool = True,
    ) -> int:
        """Convert existing detection tuples into confidence/TTL evidence.

        Invalid detections are ignored at this untrusted input boundary.  The
        return value is the number of accepted observations.
        """
        now = self._now(observed_at_s)
        ttl = self._ttl(ttl_s)
        accepted = []
        for index, detection in enumerate(detections):
            try:
                color, xyz, score = detection
                score = float(score)
            except (TypeError, ValueError):
                continue
            if exclude_color is not None and color == exclude_color:
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            volumes = volumes_from_detections([(str(color), xyz, score)])
            if not volumes:
                continue
            accepted.append(DynamicObstacle(
                volume=volumes[0],
                confidence=score,
                observed_at_s=now,
                expires_at_s=now + ttl,
                source=source,
                obstacle_id=f"{source}:{color}:{index}",
                label=str(color),
                kind="box",
            ))

        with self._lock:
            retained = (
                tuple(item for item in self._obstacles if item.source != source)
                if replace_source
                else self._obstacles
            )
            if replace_source:
                updated = self._sorted(retained + tuple(accepted))
            else:
                merged = {self._obstacle_key(item): item for item in retained}
                for item in accepted:
                    merged[self._obstacle_key(item)] = item
                updated = self._sorted(tuple(merged.values()))
            if updated != self._obstacles:
                self._obstacles = updated
                self._version += 1
        return len(accepted)

    def snapshot(self, *, now_s: Optional[float] = None) -> WorldCostmapSnapshot:
        timestamp = self._now(now_s)
        with self._lock:
            version = self._version
            active = tuple(
                item
                for item in self._obstacles
                if item.active_at(timestamp, min_confidence=0.0)
            )
        # A new LayeredGrid owns a new mutable overlay while safely sharing the
        # never-mutated static arrays.  Later live updates cannot alter it.
        grid = LayeredGrid(chassis=self._base.chassis, arm=self._base.arm)
        lethal_volumes = list(self._base_dynamic)
        lethal_volumes.extend(
            item.volume
            for item in active
            if item.confidence >= self._lethal_confidence
        )
        grid.set_dynamic(lethal_volumes)
        return WorldCostmapSnapshot(
            version=version,
            timestamp_s=timestamp,
            layered_grid=grid,
            dynamic_obstacles=self._sorted(active),
            lethal_confidence=self._lethal_confidence,
        )

    def _now(self, supplied: Optional[float]) -> float:
        value = float(self._clock() if supplied is None else supplied)
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value

    def _ttl(self, supplied: Optional[float]) -> float:
        value = self._dynamic_ttl_s if supplied is None else float(supplied)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("TTL must be finite and > 0")
        return value

    @staticmethod
    def _coerce_obstacle(
        item: DynamicObstacle | ObstacleVolume,
        *,
        now: float,
        ttl: float,
        source: str,
        index: int,
    ) -> DynamicObstacle:
        if isinstance(item, DynamicObstacle):
            return item
        if not isinstance(item, ObstacleVolume):
            raise TypeError("dynamic obstacle must be DynamicObstacle or ObstacleVolume")
        return DynamicObstacle(
            volume=item,
            confidence=1.0,
            observed_at_s=now,
            expires_at_s=now + ttl,
            source=source,
            obstacle_id=f"{source}:{index}",
            kind=item.kind or "dynamic",
        )

    @staticmethod
    def _obstacle_key(item: DynamicObstacle) -> tuple:
        identity = item.obstacle_id or (
            item.label,
            item.bounds.x_min,
            item.bounds.x_max,
            item.bounds.y_min,
            item.bounds.y_max,
            item.bounds.z_min,
            item.bounds.z_max,
        )
        return (item.source, identity)

    @classmethod
    def _sorted(
        cls, obstacles: Iterable[DynamicObstacle],
    ) -> Tuple[DynamicObstacle, ...]:
        return tuple(sorted(obstacles, key=lambda item: repr(cls._obstacle_key(item))))


__all__ = [
    "AABB",
    "DynamicObstacle",
    "PathMetrics",
    "WorldCostmap",
    "WorldCostmapSnapshot",
]
