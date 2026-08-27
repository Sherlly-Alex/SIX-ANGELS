"""Path smoother — string pulling + Chaikin corner rounding.

Operates on world-frame waypoint lists produced by A*.  Visibility tests use
the oriented footprint checker so shortcuts stay chassis-safe.  Deterministic,
no external dependencies.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

from navigation.footprint_checker import FootprintChecker
from navigation.occupancy_grid import LayeredGrid, OccupancyGrid
from navigation.robot_geometry import FootprintMode

GridLike = Union[OccupancyGrid, LayeredGrid]
Point = Tuple[float, float]


def smooth_path(
    path: Sequence[Sequence[float]],
    grid: GridLike,
    *,
    footprint: Optional[FootprintChecker] = None,
    mode: FootprintMode = FootprintMode.CHASSIS,
    sample_step: float = 0.05,
    chaikin_iterations: int = 1,
    approach_dir: Optional[Tuple[float, float]] = None,
    approach_len: float = 0.60,
) -> List[Point]:
    """Return a shortened, softly rounded path.

    1. String-pull: greedily skip waypoints whose connecting segment is
       footprint-free (or inflation-free when no checker is supplied).
    2. Optional Chaikin rounding to remove the remaining 45° A* staircases.
    3. Optional *approach_dir*: append a short straight "approach lane"
       behind the final waypoint along that unit direction (AFTER Chaikin so
       the lane stays geometrically straight).  The robot arrives already
       facing the required final yaw, so no large terminal in-place turn is
       needed next to the table/shelf.  Skipped when the lane is not
       footprint-free or would force a reverse hop (graceful fallback).
    """
    if len(path) <= 2:
        pts = [(float(p[0]), float(p[1])) for p in path]
        if approach_dir is not None:
            checker = footprint or FootprintChecker(sample_step=sample_step)
            pts = _with_approach_lane(pts, grid, checker, mode, sample_step,
                                      approach_dir, approach_len)
        return pts

    pts = [(float(p[0]), float(p[1])) for p in path]
    checker = footprint or FootprintChecker(sample_step=sample_step)

    # --- string pulling ---
    out: List[Point] = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        farthest = i + 1
        for j in range(len(pts) - 1, i, -1):
            if _segment_free(checker, grid, pts[i], pts[j], mode, sample_step):
                farthest = j
                break
        out.append(pts[farthest])
        i = farthest

    # --- Chaikin (keep endpoints fixed) BEFORE the approach lane ---
    # The approach lane must stay geometrically straight at the goal yaw; if
    # Chaikin runs after it, the first rounded connector can pull south/sideways
    # of the robot and pure-pursuit then cuts a diagonal that overshoots the
    # stand into the table/shelf band (observed on right-side table docks).
    for _ in range(max(0, int(chaikin_iterations))):
        if len(out) < 3:
            break
        rounded: List[Point] = [out[0]]
        for k in range(len(out) - 1):
            ax, ay = out[k]
            bx, by = out[k + 1]
            q = (0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by)
            r = (0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by)
            if k > 0:
                rounded.append(q)
            rounded.append(r)
        rounded[-1] = out[-1]
        # Drop any Chaikin point that broke footprint clearance; fall back.
        if all(
            checker.is_pose_free(
                grid, x, y, _yaw_of(rounded, idx), mode,
            )
            for idx, (x, y) in enumerate(rounded)
        ):
            out = rounded
        else:
            break

    if approach_dir is not None:
        out = _with_approach_lane(out, grid, checker, mode, sample_step,
                                  approach_dir, approach_len)
    return out


def _with_approach_lane(
    out: List[Point],
    grid: GridLike,
    checker: FootprintChecker,
    mode: FootprintMode,
    sample_step: float,
    approach_dir: Tuple[float, float],
    approach_len: float,
) -> List[Point]:
    """Insert a straight approach segment aligned with *approach_dir* before
    the final waypoint (the goal).

    When the robot sits far off the approach axis, also insert a *staging*
    waypoint that shares the robot's along-track coordinate with the goal's
    cross-track coordinate.  That forces an L-shaped approach (slide, then
    drive in) so pure-pursuit cannot collapse the dock into a diagonal that
    overshoots the stand into the table/shelf band.  Returns the input
    unchanged when the lane is not footprint-free.
    """
    if len(out) < 1 or approach_len <= 0.0:
        return out
    gx, gy = out[-1]
    dx, dy = float(approach_dir[0]), float(approach_dir[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return out
    dx, dy = dx / norm, dy / norm

    rx, ry = out[0]
    # Signed distance of the robot behind the goal along the approach axis.
    behind = (gx - rx) * dx + (gy - ry) * dy
    if behind < 0.20:
        # Robot is already at/past the goal along the approach axis — a forced
        # lane would require a reverse hop; skip and let terminal alignment win.
        return out
    lane_len = min(approach_len, max(0.10, behind - 0.15))
    lane = (gx - lane_len * dx, gy - lane_len * dy)
    lane_yaw = math.atan2(dy, dx)

    # Lane start must be free at the arrival heading.
    if not checker.is_pose_free(grid, lane[0], lane[1], lane_yaw, mode):
        return out
    # The final approach segment itself must be free.
    if not _segment_free(checker, grid, lane, (gx, gy), mode, sample_step):
        return out

    # Staging point: project the robot onto the approach axis through the goal
    # so pure-pursuit first cancels cross-track error, then drives in.
    # stage sits on the axis at the robot's along-track coordinate.
    stage: Optional[Point] = None
    # Lateral offset of the robot from the approach axis.
    # cross = | (robot - goal) × approach |  (2-D cross product magnitude).
    cross = abs((rx - gx) * dy - (ry - gy) * dx)
    if cross > 0.10:
        cand = (gx - behind * dx, gy - behind * dy)
        cand_behind = (gx - cand[0]) * dx + (gy - cand[1]) * dy
        if cand_behind > lane_len + 0.05:
            stage_yaw = math.atan2(lane[1] - cand[1], lane[0] - cand[0])
            if (
                checker.is_pose_free(grid, cand[0], cand[1], stage_yaw, mode)
                and _segment_free(checker, grid, cand, lane, mode, sample_step)
            ):
                stage = cand

    # Build the tail: […prev, (stage?), lane, goal].
    tail: List[Point] = []
    if stage is not None:
        tail.append(stage)
    tail.append(lane)
    tail.append((gx, gy))

    if len(out) >= 2:
        # Connector from the previous waypoint to the first tail point must be free.
        if not _segment_free(checker, grid, out[-2], tail[0], mode, sample_step):
            # Try without the stage (lane only).
            if stage is not None:
                if not _segment_free(checker, grid, out[-2], lane, mode, sample_step):
                    return out
                return out[:-1] + [lane, (gx, gy)]
            return out
        return out[:-1] + tail
    return tail


def _yaw_of(pts: Sequence[Point], idx: int) -> float:
    if idx + 1 < len(pts):
        return math.atan2(pts[idx + 1][1] - pts[idx][1], pts[idx + 1][0] - pts[idx][0])
    if idx > 0:
        return math.atan2(pts[idx][1] - pts[idx - 1][1], pts[idx][0] - pts[idx - 1][0])
    return 0.0


def _segment_free(
    checker: FootprintChecker,
    grid: GridLike,
    a: Point,
    b: Point,
    mode: FootprintMode,
    sample_step: float,
) -> bool:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return True
    yaw = math.atan2(dy, dx)
    n = max(1, int(math.ceil(length / sample_step)))
    for i in range(n + 1):
        t = i / n
        x = a[0] + t * dx
        y = a[1] + t * dy
        if not checker.is_pose_free(grid, x, y, yaw, mode):
            return False
    return True
