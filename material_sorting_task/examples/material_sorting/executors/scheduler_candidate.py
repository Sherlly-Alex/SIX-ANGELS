"""Shared opt-in helpers for executor-side scheduler candidate application.

The v2 engine ranks centre/left/right stand candidates on its own costmap.
An executor that opts in must repeat every safety check on its own layered
grid and replan through its own validated navigation controller; a rejected
selection raises so the engine fails closed.  These helpers keep that
contract uniform across task executors without coupling them to scheduler
internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from navigation.footprint_checker import FootprintChecker
from navigation.occupancy_grid import LayeredGrid
from navigation.robot_geometry import FootprintMode


class CandidateApplicationStatus(Enum):
    """Executor acknowledgement for one scheduler candidate offer."""

    APPLIED = "applied"
    AUDIT_ONLY = "audit_only"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class CandidateStandPolicy:
    """Hard corridor/clearance bounds a candidate stand must satisfy.

    These values mirror Task1NavigationExecutor's validated
    SCHEDULER_CANDIDATE_* constants.  The forward bound is intentionally
    tighter than the lateral bound: pushing a pick/place stand forward can
    enter the shelf/table emergency band, while a lateral shift stays on
    the already-cleared stand row.
    """

    max_lateral_m: float = 0.15
    max_forward_m: float = 0.10
    min_clearance_m: float = 0.22

    def __post_init__(self) -> None:
        for name in ("max_lateral_m", "max_forward_m", "min_clearance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


def candidate_goal_pose(selected: Any) -> tuple[float, float, float]:
    """Extract a finite navigation goal pose from a ranked selection."""
    candidate = getattr(selected, "candidate", selected)
    if candidate is None or not bool(getattr(candidate, "is_navigation", False)):
        raise ValueError("executor rejected a non-navigation scheduler candidate")
    goal_pose = getattr(candidate, "goal_pose", None)
    if goal_pose is None or not all(
        math.isfinite(float(value)) for value in goal_pose
    ):
        raise ValueError(
            "executor rejected a scheduler candidate without a finite goal pose"
        )
    return tuple(float(value) for value in goal_pose)  # type: ignore[return-value]


def stand_errors_in_heading(
    candidate_xy: tuple[float, float],
    nominal_pose: tuple[float, float, float],
) -> tuple[float, float]:
    """Forward/lateral displacement of a candidate inside the goal frame."""
    dx = float(candidate_xy[0]) - float(nominal_pose[0])
    dy = float(candidate_xy[1]) - float(nominal_pose[1])
    heading = float(nominal_pose[2])
    c = math.cos(heading)
    s = math.sin(heading)
    forward = c * dx + s * dy
    lateral = -s * dx + c * dy
    return forward, lateral


def validate_candidate_stand(
    candidate_xy: tuple[float, float],
    nominal_pose: tuple[float, float, float],
    *,
    policy: CandidateStandPolicy | None = None,
) -> None:
    """Enforce the stand corridor; raises ValueError with measured errors."""
    policy = policy or CandidateStandPolicy()
    forward_error, lateral_error = stand_errors_in_heading(
        candidate_xy, nominal_pose
    )
    if abs(lateral_error) > policy.max_lateral_m:
        raise ValueError(
            "stand lateral error " + format(lateral_error, "+.3f") + " m exceeds "
            + format(policy.max_lateral_m, ".2f") + " m corridor"
        )
    if abs(forward_error) > policy.max_forward_m:
        raise ValueError(
            "stand forward error " + format(forward_error, "+.3f") + " m exceeds "
            + format(policy.max_forward_m, ".2f") + " m corridor"
        )


def stand_clearance_m(grid: LayeredGrid, x: float, y: float) -> float | None:
    """Nearest-obstacle distance on the planning surface, or None."""
    planning = grid.planning_grid()
    gx, gy = planning.world_to_grid(float(x), float(y))
    if gx < 0 or gy < 0:
        return None
    try:
        distance_cells = planning.distance_transform()[gy, gx]
    except IndexError:
        return None
    value = float(distance_cells)
    if not math.isfinite(value):
        return None
    return value * planning.resolution


def validate_collision_free(
    grid: LayeredGrid,
    x: float,
    y: float,
    yaw: float,
    footprint_mode: FootprintMode,
    *,
    policy: CandidateStandPolicy | None = None,
) -> None:
    """Enforce oriented footprint freedom and the minimum stand clearance."""
    policy = policy or CandidateStandPolicy()
    checker = FootprintChecker()
    if not checker.is_pose_free(grid, float(x), float(y), float(yaw), footprint_mode):
        raise ValueError("stand pose is not collision-free on the layered grid")
    clearance = stand_clearance_m(grid, float(x), float(y))
    display = "unavailable" if clearance is None else f"{clearance:.3f} m"
    if clearance is None or clearance < policy.min_clearance_m:
        raise ValueError(
            "stand clearance " + display + " is below "
            + format(policy.min_clearance_m, ".2f") + " m"
        )


def validate_scheduler_stand(
    candidate_xy: tuple[float, float],
    nominal_pose: tuple[float, float, float],
    grid: LayeredGrid,
    footprint_mode: FootprintMode,
    *,
    policy: CandidateStandPolicy | None = None,
) -> tuple[float, float]:
    """Executor-side hard filter for a 2-D stand candidate.

    Returns the validated world XY.  Any corridor, footprint or clearance
    violation raises ValueError; the caller replans through its own
    controller and must treat that replan failure as fail-closed.
    """
    policy = policy or CandidateStandPolicy()
    x, y = (float(value) for value in candidate_xy)
    validate_candidate_stand((x, y), nominal_pose, policy=policy)
    validate_collision_free(
        grid, x, y, float(nominal_pose[2]), footprint_mode, policy=policy
    )
    return x, y


__all__ = [
    "CandidateApplicationStatus",
    "CandidateStandPolicy",
    "candidate_goal_pose",
    "stand_clearance_m",
    "stand_errors_in_heading",
    "validate_candidate_stand",
    "validate_collision_free",
    "validate_scheduler_stand",
]
