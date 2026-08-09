"""Navigation data models and interface contracts.

All spatial quantities use the world/odom coordinate convention observed by the
Server and Client:

- +X points east, +Y points north.
- ``yaw = 0`` means the robot faces +X (east).
- ``yaw = pi/2`` means the robot faces +Y (north).
- ``yaw = pi`` means the robot faces -X (west).

Units are explicitly meters (m), radians (rad), seconds (s), meters per second
(m/s) and radians per second (rad/s). Navigation algorithms consuming these
objects must validate that coordinates are finite.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple


class NavigationSegment(Enum):
    """One of the three navigation segments handled in this stage."""

    NAV_SHELF = "nav_shelf"
    NAV_TABLE = "nav_table"
    NAV_END = "nav_end"


class NavigationStatus(Enum):
    """High-level state returned by the navigation controller."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    FINAL_POSITIONING = "final_positioning"
    FINAL_ALIGNING = "final_aligning"
    GOAL_REACHED = "goal_reached"
    BLOCKED = "blocked"
    REPLANNING = "replanning"
    FAILED = "failed"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class Pose2D:
    """2-D planar pose in the world/odom frame.

    Attributes:
        x: East coordinate in m.
        y: North coordinate in m.
        yaw: Heading in rad, zero = east, CCW positive.
    """

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class NavigationGoal:
    """A navigation target for one of the three segments.

    Attributes:
        x: Target east coordinate in m.
        y: Target north coordinate in m.
        yaw: Target heading in rad.
        position_tolerance: Goal-arrival tolerance in m. The robot is
            considered to have reached the goal once its planar distance to
            ``(x, y)`` is within this value. This is an arrival threshold,
            NOT a safety standoff from obstacles.
        yaw_tolerance: Goal-arrival heading tolerance in rad.
        safety_radius: Minimum clearance (m) this stand goal maintains
            between itself and its associated target/hazard. For pick and
            place goals this equals the approach standoff already baked into
            ``(x, y)``; for zone-center goals (e.g. the end zone) it is 0.0
            because no single target standoff applies. Path planners may use
            it as a per-goal inflation hint. It is distinct from
            ``position_tolerance`` (arrival threshold).
        segment: Which logical segment this goal belongs to.
        source_tag: Human-readable origin of the goal, e.g.
            ``layout_derived``, ``task_derived`` or ``config_derived``.
    """

    x: float
    y: float
    yaw: float
    position_tolerance: float
    yaw_tolerance: float
    safety_radius: float
    segment: NavigationSegment
    source_tag: str


@dataclass(frozen=True)
class VelocityCommand:
    """Differential-drive velocity command actually published by the Client.

    Only ``linear.x`` and ``angular.z`` are used by the current Server/MMK2
    controller. Lateral velocities are reserved for future holonomic extensions.
    """

    linear_x: float
    angular_z: float


@dataclass(frozen=True)
class ObstacleObservation:
    """A single snapshot of obstacle evidence, optional real-time input.

    When LiDAR is unavailable the navigation stack falls back to the static
    occupancy grid derived from ``KnownScene``. All numeric fields are in m or
    rad and the caller is responsible for finite/NaN filtering.

    Sequence fields are stored as ``tuple`` so that the frozen dataclass cannot
    be mutated through a mutable ``list`` keep-alive reference.
    """

    frame_id: str
    ranges: Sequence[float]
    angles: Sequence[float]
    points: Sequence[Tuple[float, float]]
    timestamp: float
    valid: bool

    def __post_init__(self):
        object.__setattr__(self, "ranges", tuple(self.ranges))
        object.__setattr__(self, "angles", tuple(self.angles))
        object.__setattr__(self, "points", tuple(
            (float(x), float(y)) for x, y in self.points
        ))


@dataclass(frozen=True)
class RobotFootprint:
    """Convex polygon describing the robot footprint in its body frame.

    Vertices must be ordered counter-clockwise and are given as ``(x, y)`` in m
    relative to the robot centre. The footprint is used for collision inflation
    and path validity checks.

    Sequence fields are stored as ``tuple`` so that the frozen dataclass cannot
    be mutated through a mutable ``list`` keep-alive reference.
    """

    vertices: Sequence[Tuple[float, float]]

    def __post_init__(self):
        object.__setattr__(self, "vertices", tuple(
            (float(x), float(y)) for x, y in self.vertices
        ))


@dataclass(frozen=True)
class SpeedLimits:
    """Kinematic and safety speed limits.

    All values are in m/s, rad/s, m/s^2 or rad/s^2. The emergency clearance
    together with ``max_deceleration`` is used to bound stopping distance.
    """

    max_linear: float
    max_angular: float
    max_linear_accel: float
    max_angular_accel: float
    emergency_clearance: float
    max_deceleration: float


@dataclass(frozen=True)
class NavigationTelemetry:
    """Per-tick snapshot of navigation internals for regression / logs.

    Fields match the handoff document §9 list, plus the phase-A/C metrics
    (``footprint_min_clearance``, ``rotate_in_place``, ``lookahead``, ``kappa``).

    ``path_length`` / ``planned_straight`` are frozen at ``set_goal`` so the
    phase-D detour gate (path ≤ 2× chord) is not polluted by shrinking
    remaining distance as the robot approaches the goal.
    """

    status: str
    x: float
    y: float
    yaw: float
    goal_x: float
    goal_y: float
    goal_yaw: float
    dist_err: float
    yaw_err: float
    cmd_lin: float
    cmd_ang: float
    footprint_min_clearance: float
    rotate_in_place: bool
    lookahead: float
    kappa: float
    path_deviation: float
    footprint_mode: str
    path_length: float = 0.0
    straight_distance: float = 0.0
    planned_straight: float = 0.0
    segment: str = ""
