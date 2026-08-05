"""KnownSceneProvider: centralized source of navigation goals and scene geometry.

All coordinates are read from the competition ``material_competition_layout.json``
and from the concrete ``task_layout`` produced by ``material_sorting_server``
(fixed or randomized). This keeps known coordinates out of the Client and
controllers. No coordinate is hardcoded as a fallback; missing layout fields
raise instead of silently substituting scene constants.

Coordinate convention:
- +X east, +Y north.
- ``yaw = 0`` faces +X (east); ``yaw = pi/2`` faces +Y (north);
  ``yaw = pi`` faces -X (west).
- Units: meters (m) and radians (rad).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from navigation.navigation_types import NavigationGoal, NavigationSegment


def _freeze(value: Any) -> Any:
    """Return a recursively read-only view of a JSON-like value.

    Mappings become ``MappingProxyType`` (item assignment raises), sequences
    become ``tuple`` (immutable). Scalars are returned unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


class KnownSceneProvider:
    """Provides pick, place and end-zone goals derived from a task layout.

    The provider does not run ROS2, A* or collision checking; it only turns the
    known scene geometry into navigation targets. The input layout is deep-copied
    on construction so the caller's object is never mutated, and the ``scene``
    property exposes only a recursively read-only view.
    """

    # Robot / task geometry parameters. These are not scene coordinates; they
    # describe how far the robot stops short of a target in the corresponding
    # direction and double as the per-goal safety_radius.
    SHELF_APPROACH_STANDOFF = 0.90  # m, robot stops east of a shelf target
    TABLE_APPROACH_STANDOFF = 0.65  # m, robot stops south of a table target

    DEFAULT_POSITION_TOLERANCE = 0.06
    DEFAULT_YAW_TOLERANCE = 0.03

    def __init__(
        self,
        layout_path: Optional[str] = None,
        task_layout: Optional[Dict[str, Any]] = None,
    ):
        if task_layout is not None:
            # Deep-copy so the caller's mapping/nested structures are isolated
            # from any internal reads/writes.
            self._layout: Dict[str, Any] = copy.deepcopy(task_layout)
        elif layout_path is not None:
            self._layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        else:
            raise ValueError("Either layout_path or task_layout must be provided")

        self._scene: Dict[str, Any] = dict(self._layout.get("scene", {}))
        self._movable: Mapping[str, Dict[str, Any]] = {
            b["body"]: dict(b) for b in self._layout.get("movable_boxes", [])
        }
        self._fixed: Mapping[str, Dict[str, Any]] = {
            p["body"]: dict(p) for p in self._layout.get("fixed_props", [])
        }

    @property
    def scene(self) -> Mapping[str, Any]:
        """Read-only, recursively frozen scene metadata."""
        return _freeze(self._scene)

    @property
    def end_zone(self) -> Tuple[float, float, float, float]:
        """End zone rectangle as ``(xmin, xmax, ymin, ymax)`` in m.

        Raises ``KeyError`` if ``scene.end_zone`` (or its x/y bounds) is absent;
        coordinates must come from the layout, never from a hardcoded fallback.
        """
        zone = self._scene.get("end_zone")
        if zone is None:
            raise KeyError("scene.end_zone is required to derive the end-zone goal")
        xs = zone.get("x")
        ys = zone.get("y")
        if xs is None or ys is None:
            raise KeyError("scene.end_zone.x and scene.end_zone.y are required")
        if len(xs) < 2 or len(ys) < 2:
            raise ValueError("scene.end_zone.x/.y must each provide at least 2 bounds")
        return (float(xs[0]), float(xs[1]), float(ys[0]), float(ys[1]))

    def end_goal(self) -> NavigationGoal:
        """Return the navigation goal for the end zone.

        The goal is the center of the configured end zone, facing north. No
        per-target standoff applies, so ``safety_radius`` is 0.0.
        """
        xmin, xmax, ymin, ymax = self.end_zone
        return NavigationGoal(
            x=(xmin + xmax) / 2.0,
            y=(ymin + ymax) / 2.0,
            yaw=math.pi / 2.0,
            position_tolerance=self.DEFAULT_POSITION_TOLERANCE,
            yaw_tolerance=self.DEFAULT_YAW_TOLERANCE,
            safety_radius=0.0,
            segment=NavigationSegment.NAV_END,
            source_tag="layout_derived",
        )

    def _zone_bounds(self, zone_name: str) -> Optional[Tuple[float, float, float, float]]:
        """Return ``(xmin, xmax, ymin, ymax)`` for a scene zone, or None.

        Zone lookups are data-driven from the layout; no hardcoded fallback.
        """
        zone = self._scene.get(zone_name)
        if not isinstance(zone, dict):
            return None
        xs, ys = zone.get("x"), zone.get("y")
        if not isinstance(xs, (list, tuple)) or not isinstance(ys, (list, tuple)):
            return None
        if len(xs) < 2 or len(ys) < 2:
            return None
        return (float(xs[0]), float(xs[1]), float(ys[0]), float(ys[1]))

    @staticmethod
    def _point_in_zone(
        world_xyz: Sequence[float],
        bounds: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        if bounds is None or world_xyz is None or len(world_xyz) < 2:
            return False
        xmin, xmax, ymin, ymax = bounds
        x, y = float(world_xyz[0]), float(world_xyz[1])
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def is_shelf_location(self, world_xyz: Sequence[float]) -> bool:
        """True when a world point lies inside the shelf picking zone."""
        return self._point_in_zone(world_xyz, self._zone_bounds("picking_zone"))

    def is_table_location(self, world_xyz: Sequence[float]) -> bool:
        """True when a world point lies inside the table delivery zone."""
        return self._point_in_zone(world_xyz, self._zone_bounds("delivery_zone"))

    def location_of(self, world_xyz: Sequence[float], max_dist: float = 0.5) -> str | None:
        """Return the layout location (``"shelf"`` / ``"table_side"`` / ``"table_top"``)
        of the nearest movable box, or ``None`` when nothing is within *max_dist*.

        Data-driven from the layout box positions — no hardcoded coordinates.
        Used to disambiguate a perception lock that sits between zones.
        """
        if world_xyz is None or len(world_xyz) < 2:
            return None
        x, y = float(world_xyz[0]), float(world_xyz[1])
        best_loc: str | None = None
        best_d = float("inf")
        for box in self._movable.values():
            wp = box.get("world_position")
            if not isinstance(wp, (list, tuple)) or len(wp) < 2:
                continue
            d = math.hypot(x - float(wp[0]), y - float(wp[1]))
            if d < best_d:
                best_d = d
                best_loc = box.get("location")
        if best_loc is None or best_d > float(max_dist):
            return None
        return best_loc

    def _find_target_box(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve a task instruction to the corresponding movable box dict."""
        body = task.get("target_body")
        if body and body in self._movable:
            return dict(self._movable[body])

        color = task.get("target_color")
        if color:
            for box in self._movable.values():
                if box.get("color") == color:
                    return dict(box)

        raise KeyError(
            f"target not found: target_body={body!r} target_color={color!r}"
        )

    @staticmethod
    def _stand_pose_and_clearance(
        world_position: Sequence[float],
        location: str,
    ) -> Tuple[float, float, float, float]:
        """Compute a stand pose ``(x, y, yaw, safety_radius)`` for a target.

        - Shelf targets: stop to the east, facing west.
        - Table targets: stop to the south, facing north.
        ``safety_radius`` is the approach standoff baked into the stand pose.
        """
        x, y = float(world_position[0]), float(world_position[1])
        if location == "shelf":
            standoff = KnownSceneProvider.SHELF_APPROACH_STANDOFF
            return (x + standoff, y, math.pi, standoff)
        standoff = KnownSceneProvider.TABLE_APPROACH_STANDOFF
        return (x, y - standoff, math.pi / 2.0, standoff)

    def pick_goal(self, task: Mapping[str, Any]) -> NavigationGoal:
        """Return the pick-stand goal for the task's target object."""
        box = self._find_target_box(task)
        pos = box.get("world_position")
        if pos is None or len(pos) < 2:
            raise ValueError(f"target box missing world_position: {box}")

        location = box.get("location", "shelf")
        x, y, yaw, safety_radius = self._stand_pose_and_clearance(pos, location)
        segment = (
            NavigationSegment.NAV_SHELF
            if location == "shelf"
            else NavigationSegment.NAV_TABLE
        )
        return NavigationGoal(
            x=x,
            y=y,
            yaw=yaw,
            position_tolerance=self.DEFAULT_POSITION_TOLERANCE,
            yaw_tolerance=self.DEFAULT_YAW_TOLERANCE,
            safety_radius=safety_radius,
            segment=segment,
            source_tag="task_derived",
        )

    def place_goal(self, task: Mapping[str, Any]) -> NavigationGoal:
        """Return the place-stand goal for the task's place target."""
        place_world = task.get("place_world")
        if place_world is None or len(place_world) < 2:
            raise ValueError("task missing valid place_world")

        place_type = task.get("place_type", "table_point")
        if place_type in ("shelf_point", "shelf_prop_side"):
            x = place_world[0] + self.SHELF_APPROACH_STANDOFF
            y = place_world[1]
            yaw = math.pi
            safety_radius = self.SHELF_APPROACH_STANDOFF
            segment = NavigationSegment.NAV_SHELF
        else:
            # table_point and fallback
            x = place_world[0]
            y = place_world[1] - self.TABLE_APPROACH_STANDOFF
            yaw = math.pi / 2.0
            safety_radius = self.TABLE_APPROACH_STANDOFF
            segment = NavigationSegment.NAV_TABLE

        return NavigationGoal(
            x=x,
            y=y,
            yaw=yaw,
            position_tolerance=self.DEFAULT_POSITION_TOLERANCE,
            yaw_tolerance=self.DEFAULT_YAW_TOLERANCE,
            safety_radius=safety_radius,
            segment=segment,
            source_tag="task_derived",
        )

    def goals_for_task(self, task: Mapping[str, Any]) -> Tuple[NavigationGoal, ...]:
        """Convenience helper returning (pick_goal, place_goal, end_goal)."""
        return (self.pick_goal(task), self.place_goal(task), self.end_goal())
