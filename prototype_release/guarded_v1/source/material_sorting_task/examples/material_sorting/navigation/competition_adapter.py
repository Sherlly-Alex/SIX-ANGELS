"""Thin adapter between the executor architecture and the complete v3 stack.

The navigation algorithms are vendored unchanged from the authoritative
``D:/local_discoverse`` v3 source.  This module only translates this project's
stable ``TargetObservation`` mapping into the public detection overlay contract
and formats controller evidence for the existing executor log path.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from navigation.dynamic_overlay import build_nav_overlay
from navigation.navigation_types import NavigationGoal, NavigationTelemetry
from navigation.occupancy_grid import LayeredGrid


def refresh_dynamic_overlay(
    grid: LayeredGrid,
    observations: Optional[Mapping[str, Any]],
    *,
    exclude_color: Optional[str],
    robot_xy: tuple[float, float],
) -> int:
    """Replace the layered overlay using only public perception detections."""

    detections = []
    for key, observation in (observations or {}).items():
        try:
            label = str(getattr(observation, "color", key)).strip().lower()
            xyz = tuple(float(value) for value in observation.position_world)
            score = float(getattr(observation, "score", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue
        if len(xyz) < 3 or not all(math.isfinite(value) for value in xyz[:3]):
            continue
        detections.append((label, xyz[:3], score))
    volumes = build_nav_overlay(
        detections=detections,
        exclude_color=(
            str(exclude_color).strip().lower() if exclude_color else None
        ),
        robot_xy=(float(robot_xy[0]), float(robot_xy[1])),
    )
    grid.set_dynamic(volumes)
    return len(volumes)


def goal_reached_event(goal: NavigationGoal) -> str:
    return f"NAV_GOAL_REACHED segment={goal.segment.value}"


def format_nav_telemetry(
    telemetry: NavigationTelemetry,
    *,
    phase: str,
) -> str:
    """Format telemetry for the v3 phase-D log parser."""

    straight = float(telemetry.planned_straight)
    path = float(telemetry.path_length)
    detour = path / straight if straight > 1e-9 else 1.0
    return (
        f"NAV_TEL phase={phase} status={telemetry.status} "
        f"segment={telemetry.segment} "
        f"x={telemetry.x:.3f} y={telemetry.y:.3f} yaw={telemetry.yaw:.3f} "
        f"dist={telemetry.dist_err:.3f} yaw_err={telemetry.yaw_err:.3f} "
        f"cmd=({telemetry.cmd_lin:.3f},{telemetry.cmd_ang:.3f}) "
        f"clear={telemetry.footprint_min_clearance:.3f} "
        f"footprint={telemetry.footprint_mode} "
        f"path={path:.3f} planned_straight={straight:.3f} detour={detour:.3f}"
    )


__all__ = [
    "format_nav_telemetry",
    "goal_reached_event",
    "refresh_dynamic_overlay",
]
