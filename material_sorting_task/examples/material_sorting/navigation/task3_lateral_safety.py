"""Task-3 shelf-front 1-D lateral safety layer.

The global A* stack is unchanged.  After the existing task-3 code computes
the qzhRL packaging-left stand, this module keeps that Y as the parking
target whether or not the carry envelope calls the sweep safe.  Safety
is applied to *how* the chassis travels (north face, reverse south,
swept-volume and predictive checks), not by substituting a different
stop.  A geometrically impossible left packing gap is still rejected.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from navigation.carried_envelope import (
    CarriedEnvelopeChecker,
    EnvelopeCheck,
    HeldObjectGeometry,
)


@dataclass(frozen=True)
class Task3LateralGuardParams:
    """Concentrated task-3 lateral-guard knobs.  Units and sources below.

    enabled:
        Dimensionless flag.  Default True so competition builds keep the
        guard on.  False restores the full pre-change lateral *behaviour*
        (legacy target, rotate_lateral to ±π/2, drive_lateral, rotate_final),
        not merely the unprojected target Y.  It does not skip motion, does
        not force a north-facing travel pose, and does not run the
        predictive carry guard.
    carry_clearance_margin_m:
        Metres.  Same 0.02 m disc clearance already used by
        ``CarriedEnvelopeChecker.DEFAULT_CLEARANCE_M``.  In the left-gap
        packing test this is the face-to-face clearance on *each* side of
        the coloured box (shelf south face and white south face).
    lateral_deadband_m:
        Metres.  Reuses ``Task3IntegratedExecutor.TASK3_LATERAL_POSITION_TOLERANCE_M``
        (0.015 m), the already-validated place-alignment success band.
    lateral_max_speed_mps:
        Metres/second.  Reuses the existing shelf-front drive cap of 0.09 m/s
        in ``TransferMotion.tick_lateral_alignment``.
    yaw_tolerance_rad:
        Radians.  Reuses ``LATERAL_YAW_TOLERANCE_RAD`` / shelf-turn tolerance
        0.06 rad.
    large_yaw_threshold_rad:
        Radians.  Reuses the existing drive-lateral freeze threshold 0.18 rad
        (``tick_lateral_alignment`` zeros linear when heading error exceeds
        this).  A larger error at shelf-front must not start an unconstrained
        in-place spin.
    collision_sample_step_m:
        Metres.  Reuses ``CarriedEnvelopeChecker.PATH_SAMPLE_M`` (0.04 m),
        which is already matched to the 0.05 m occupancy grid.
    prediction_horizon_s / prediction_step_s:
        Seconds.  Reuse ``CarriedEnvelopeChecker.check_command`` defaults
        (0.8 s horizon, 0.1 s step).
    min_prop_center_separation_m:
        Metres.  Reuses ``TASK3_MIN_PROP_CENTER_SEPARATION_M`` (0.150 m).
        This is the legacy *target-inset floor* (centre-to-centre).  It is
        smaller than ``r_white + r_place`` (0.197 m) so it is not a
        non-overlap packing constraint and must not be added on top of the
        object radii in ``evaluate_left_placement_feasibility``.
    place_object_half_y_m:
        Metres.  Coloured-box half-width along shelf Y from the layout
        yaw0 size 0.08 m.  The packing test uses the full width ``2 * r``.
    white_object_half_y_m:
        Metres.  Packaging-box planar half-extent 0.1170 m from
        ``material_competition_layout.json``.
    place_radius_margin_m:
        Metres.  Reuses ``TASK3_SAFE_RELEASE_RADIUS_MARGIN_M`` (0.040 m).
        Scoring-radius budget for ``I_place`` only; not a shelf-fit margin.
    workspace_s_min_m / workspace_s_max_m:
        Metres.  Picking-zone Y ``[-0.02, 1.42]`` from the layout, used only
        to bound the 1-D scan.
    max_consecutive_guard_blocks:
        Count.  Three rejected predictive ticks (~0.15 s at 20 Hz) is enough
        to declare LATERAL_BLOCKED without chatter.  Derived from the
        existing 0.05–0.20 s control dt clamp.
    """

    enabled: bool = True
    carry_clearance_margin_m: float = 0.02
    lateral_deadband_m: float = 0.015
    lateral_max_speed_mps: float = 0.09
    yaw_tolerance_rad: float = 0.06
    large_yaw_threshold_rad: float = 0.18
    collision_sample_step_m: float = 0.04
    prediction_horizon_s: float = 0.8
    prediction_step_s: float = 0.1
    min_prop_center_separation_m: float = 0.150
    place_object_half_y_m: float = 0.08
    white_object_half_y_m: float = 0.1170
    place_radius_margin_m: float = 0.040
    workspace_s_min_m: float = -0.02
    workspace_s_max_m: float = 1.42
    max_consecutive_guard_blocks: int = 3


@dataclass(frozen=True)
class SafeLateralTarget:
    feasible: bool
    current_s: float
    legacy_s: float
    safe_target_s: float | None
    safe_min_s: float | None
    safe_max_s: float | None
    place_min_s: float | None
    place_max_s: float | None
    was_clipped: bool
    placement_feasible: bool
    motion_safe: bool
    skip_motion: bool
    yaw_error_rad: float
    min_clearance_m: float
    reason: str
    latency_s: float


@dataclass(frozen=True)
class GuardedLateralCommand:
    linear_x: float
    angular_z: float
    allowed: bool
    slowed: bool
    blocked: bool
    clearance_m: float
    reason: str


def compute_safe_lateral_target(
    current_pose: tuple[float, float, float],
    legacy_target_xy: tuple[float, float],
    yaw_ref: float,
    held_geometry: HeldObjectGeometry | None,
    *,
    checker: CarriedEnvelopeChecker | None = None,
    params: Task3LateralGuardParams | None = None,
    white_object_y: float | None = None,
    place_radius_m: float | None = None,
    scoring_target_xy: tuple[float, float] | None = None,
    release_xy: tuple[float, float] | None = None,
    travel_face_yaw: float | None = None,
) -> SafeLateralTarget:
    """Keep the qzhRL shelf-front Y; envelope checks are telemetry only.

    ``s`` is world Y because the shelf opens along +X and the existing
    task-3 lateral controller only adjusts Y while holding X and the
    pre-place yaw.  ``safe_target_s`` is always ``legacy_s`` when the
    plan is feasible.
    """

    started = time.perf_counter()
    cfg = params or Task3LateralGuardParams()
    yaw_error = _wrap_to_pi(float(yaw_ref) - float(current_pose[2]))
    current_s = float(current_pose[1])
    legacy_s = float(legacy_target_xy[1])
    n_ref = float(legacy_target_xy[0])

    def _done(
        *,
        feasible: bool,
        target: float | None,
        safe_min: float | None,
        safe_max: float | None,
        place_min: float | None,
        place_max: float | None,
        clipped: bool,
        placement_feasible: bool,
        motion_safe: bool,
        skip: bool,
        clearance: float,
        reason: str,
    ) -> SafeLateralTarget:
        return SafeLateralTarget(
            feasible=feasible,
            current_s=current_s,
            legacy_s=legacy_s,
            safe_target_s=target,
            safe_min_s=safe_min,
            safe_max_s=safe_max,
            place_min_s=place_min,
            place_max_s=place_max,
            was_clipped=clipped,
            placement_feasible=placement_feasible,
            motion_safe=motion_safe,
            skip_motion=skip,
            yaw_error_rad=yaw_error,
            min_clearance_m=clearance,
            reason=reason,
            latency_s=max(0.0, time.perf_counter() - started),
        )

    if not cfg.enabled:
        return _done(
            feasible=True,
            target=legacy_s,
            safe_min=legacy_s,
            safe_max=legacy_s,
            place_min=legacy_s,
            place_max=legacy_s,
            clipped=False,
            placement_feasible=True,
            motion_safe=True,
            skip=False,
            clearance=float("inf"),
            reason="guard_disabled",
        )

    values = (
        current_pose[0],
        current_pose[1],
        current_pose[2],
        n_ref,
        legacy_s,
        yaw_ref,
        yaw_error,
    )
    if not all(math.isfinite(value) for value in values):
        return _done(
            feasible=False,
            target=None,
            safe_min=None,
            safe_max=None,
            place_min=None,
            place_max=None,
            clipped=False,
            placement_feasible=False,
            motion_safe=False,
            skip=False,
            clearance=float("-inf"),
            reason="non_finite_input",
        )
    if held_geometry is None:
        return _done(
            feasible=False,
            target=None,
            safe_min=None,
            safe_max=None,
            place_min=None,
            place_max=None,
            clipped=False,
            placement_feasible=False,
            motion_safe=False,
            skip=False,
            clearance=float("-inf"),
            reason="carried_envelope_unavailable",
        )
    if abs(yaw_error) > cfg.large_yaw_threshold_rad:
        return _done(
            feasible=False,
            target=None,
            safe_min=None,
            safe_max=None,
            place_min=None,
            place_max=None,
            clipped=False,
            placement_feasible=False,
            motion_safe=False,
            skip=False,
            clearance=float("-inf"),
            reason="yaw_error_too_large",
        )

    envelope = checker or CarriedEnvelopeChecker(
        clearance_m=cfg.carry_clearance_margin_m,
    )
    placement = evaluate_left_placement_feasibility(
        white_object_y=white_object_y,
        checker=envelope,
        params=cfg,
    )
    if not placement.feasible:
        return _done(
            feasible=False,
            target=None,
            safe_min=None,
            safe_max=None,
            place_min=None,
            place_max=None,
            clipped=False,
            placement_feasible=False,
            motion_safe=False,
            skip=False,
            clearance=placement.clearance_m,
            reason="placement_left_infeasible",
        )

    place_min, place_max = _place_interval(
        legacy_s,
        cfg,
        place_radius_m=place_radius_m,
        scoring_target_xy=scoring_target_xy,
        release_xy=release_xy,
    )
    # Parking Y is the verified qzhRL stand in every feasible case.
    # Scoring-radius I_place and the carry-safe interval are telemetry;
    # they must not rewrite a stop that subsequent placement already uses.
    interval = _safe_interval_containing(
        n_ref,
        current_s,
        float(yaw_ref),
        held_geometry,
        envelope,
        cfg,
    )
    if interval is None:
        safe_min = None
        safe_max = None
        corridor_clearance = float("-inf")
    else:
        safe_min, safe_max, corridor_clearance = interval

    skip_motion = abs(legacy_s - current_s) <= cfg.lateral_deadband_m
    legacy_pose = envelope.check_pose(
        (n_ref, legacy_s, float(yaw_ref)),
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    min_clearance = min(corridor_clearance, legacy_pose.clearance_m)
    if skip_motion:
        return _done(
            feasible=True,
            target=legacy_s,
            safe_min=safe_min,
            safe_max=safe_max,
            place_min=place_min,
            place_max=place_max,
            clipped=False,
            placement_feasible=True,
            motion_safe=True,
            skip=True,
            clearance=min_clearance,
            reason="already_at_legacy_target",
        )

    segment = is_lateral_motion_safe(
        (float(current_pose[0]), current_s, float(yaw_ref)),
        (n_ref, legacy_s, float(yaw_ref)),
        held_geometry,
        checker=envelope,
        params=cfg,
        travel_face_yaw=travel_face_yaw,
    )
    min_clearance = min(min_clearance, segment.clearance_m)
    return _done(
        feasible=True,
        target=legacy_s,
        safe_min=safe_min,
        safe_max=safe_max,
        place_min=place_min,
        place_max=place_max,
        clipped=False,
        placement_feasible=True,
        motion_safe=segment.safe,
        skip=False,
        clearance=min_clearance,
        reason="legacy_target_safe" if segment.safe else "legacy_stop_kept",
    )


def is_lateral_motion_safe(
    start_pose: tuple[float, float, float],
    target_pose: tuple[float, float, float],
    held_geometry: HeldObjectGeometry,
    *,
    checker: CarriedEnvelopeChecker | None = None,
    params: Task3LateralGuardParams | None = None,
    travel_face_yaw: float | None = None,
) -> EnvelopeCheck:
    """Check the frozen-yaw slide and the actual rotate-drive-restore sweep.

    The runtime ``check_command`` horizon only looks ~0.8 s ahead and does
    not cover a full 90° restore.  This planner check samples the whole
    carried-envelope yaw sweep: start yaw → ``travel_face_yaw`` at the
    start xy, translation at that heading, then ``travel_face_yaw`` →
    target yaw at the *target* xy (the restore near the left/right bound).
    """

    cfg = params or Task3LateralGuardParams()
    envelope = checker or CarriedEnvelopeChecker(
        clearance_m=cfg.carry_clearance_margin_m,
    )
    start = tuple(float(value) for value in start_pose)
    target = tuple(float(value) for value in target_pose)
    if not all(math.isfinite(value) for value in (*start, *target)):
        return EnvelopeCheck(False, float("-inf"), "non-finite lateral segment")

    frozen_start = (start[0], start[1], start[2])
    frozen_end_xy = (target[0], target[1])
    frozen = envelope.check_fixed_heading_translation(
        frozen_start,
        frozen_end_xy,
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    if not frozen.safe:
        return frozen

    face_yaw = (
        start[2]
        if travel_face_yaw is None
        else _wrap_to_pi(float(travel_face_yaw))
    )
    if abs(_wrap_to_pi(face_yaw - start[2])) <= cfg.yaw_tolerance_rad:
        # Already at the travel face: translation at this heading is the
        # commanded motion, then a small restore to target yaw.
        travel = envelope.check_fixed_heading_translation(
            (start[0], start[1], face_yaw),
            frozen_end_xy,
            held_geometry.center_base,
            held_geometry.half_width_m,
        )
        if not travel.safe:
            return travel
        restore = envelope.check_rotation(
            (target[0], target[1], face_yaw),
            target[2],
            held_geometry.center_base,
            held_geometry.half_width_m,
        )
        if not restore.safe:
            return restore
        return _worse(frozen, travel, restore)

    rotate = envelope.check_rotation(
        (start[0], start[1], start[2]),
        face_yaw,
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    if not rotate.safe:
        return rotate
    travel = envelope.check_fixed_heading_translation(
        (start[0], start[1], face_yaw),
        frozen_end_xy,
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    if not travel.safe:
        return travel
    restore = envelope.check_rotation(
        (target[0], target[1], face_yaw),
        target[2],
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    if not restore.safe:
        return restore
    return _worse(frozen, rotate, travel, restore)


def guard_task3_lateral_cmd(
    pose: tuple[float, float, float],
    command: tuple[float, float],
    held_geometry: HeldObjectGeometry,
    *,
    checker: CarriedEnvelopeChecker | None = None,
    params: Task3LateralGuardParams | None = None,
) -> GuardedLateralCommand:
    """Last-cycle predictive filter for a candidate ``(v, ω)`` pair."""

    cfg = params or Task3LateralGuardParams()
    linear, angular = float(command[0]), float(command[1])
    if not all(math.isfinite(value) for value in (*pose, linear, angular)):
        return GuardedLateralCommand(
            0.0, 0.0, False, False, True, float("-inf"), "non_finite_command"
        )
    envelope = checker or CarriedEnvelopeChecker(
        clearance_m=cfg.carry_clearance_margin_m,
    )
    predicted = envelope.check_command(
        pose,
        (linear, angular),
        held_geometry.center_base,
        held_geometry.half_width_m,
        horizon_s=cfg.prediction_horizon_s,
        step_s=cfg.prediction_step_s,
    )
    if not predicted.safe:
        return GuardedLateralCommand(
            0.0,
            0.0,
            False,
            False,
            True,
            predicted.clearance_m,
            predicted.detail,
        )
    soft = 2.0 * cfg.carry_clearance_margin_m
    if predicted.clearance_m < soft:
        return GuardedLateralCommand(
            0.5 * linear,
            0.5 * angular,
            True,
            True,
            False,
            predicted.clearance_m,
            f"soft_margin {predicted.detail}",
        )
    return GuardedLateralCommand(
        linear,
        angular,
        True,
        False,
        False,
        predicted.clearance_m,
        predicted.detail,
    )


@dataclass(frozen=True)
class PlacementFeasibility:
    feasible: bool
    left_free_space_m: float
    required_place_width_m: float
    clearance_m: float
    reason: str


def left_placement_required_width_m(params: Task3LateralGuardParams) -> float:
    """Width the coloured box needs between the shelf south face and white.

    Geometry, not a sum of every named margin:

    * coloured width = ``2 * place_object_half_y_m``
    * one ``carry_clearance_margin_m`` to the shelf south face
    * one ``carry_clearance_margin_m`` to the white south face

    ``min_prop_center_separation_m`` and ``place_radius_margin_m`` are
    intentionally omitted: the former is a centre-to-centre target inset
    already smaller than the two radii, and the latter is an ``I_place``
    scoring budget.
    """

    return (
        2.0 * float(params.place_object_half_y_m)
        + 2.0 * float(params.carry_clearance_margin_m)
    )


def left_placement_critical_white_y(
    shelf_ymin_m: float,
    params: Task3LateralGuardParams | None = None,
) -> float:
    """Smallest white-object centre Y that still leaves a left packing gap."""

    cfg = params or Task3LateralGuardParams()
    return (
        float(shelf_ymin_m)
        + float(cfg.white_object_half_y_m)
        + left_placement_required_width_m(cfg)
    )


def evaluate_left_placement_feasibility(
    *,
    white_object_y: float | None,
    checker: CarriedEnvelopeChecker,
    params: Task3LateralGuardParams,
) -> PlacementFeasibility:
    """Return whether the white prop's left gap can hold the coloured box.

    ``left_free_space`` is the face-to-face gap:

        (white_y - r_white) - shelf_ymin

    ``required_place_width`` is the coloured width plus clearance on both
    faces.  Clearance is counted once per face here, not subtracted from
    the gap *and* added to the requirement.
    """

    required = left_placement_required_width_m(params)
    if white_object_y is None:
        return PlacementFeasibility(
            True, float("inf"), required, float("inf"), "white_y_not_provided"
        )
    white_y = float(white_object_y)
    if not math.isfinite(white_y):
        return PlacementFeasibility(
            False, float("-inf"), required, float("-inf"), "white_y_invalid"
        )
    shelf = checker.obstacle_bounds("shelf")
    if shelf is None:
        return PlacementFeasibility(
            False, float("-inf"), required, float("-inf"), "shelf_geometry_unavailable"
        )
    shelf_ymin = float(shelf[2])
    white_left_edge = white_y - params.white_object_half_y_m
    left_free = white_left_edge - shelf_ymin
    if left_free + 1e-9 < required:
        return PlacementFeasibility(
            False,
            left_free,
            required,
            left_free - required,
            "placement_left_infeasible",
        )
    return PlacementFeasibility(
        True, left_free, required, left_free - required, "placement_feasible"
    )


def format_task3_lateral_log(prefix: str, result: SafeLateralTarget) -> str:
    """One-line TASK3_LATERAL / DONE / BLOCKED telemetry."""

    def _fmt(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "nan"
        return f"{value:.4f}"

    return (
        f"{prefix} current_s={_fmt(result.current_s)} "
        f"legacy_s={_fmt(result.legacy_s)} "
        f"safe_min={_fmt(result.safe_min_s)} "
        f"safe_max={_fmt(result.safe_max_s)} "
        f"final_s={_fmt(result.safe_target_s)} "
        f"clipped={int(result.was_clipped)} "
        f"skip={int(result.skip_motion)} "
        f"placement_feasible={int(result.placement_feasible)} "
        f"yaw_error={_fmt(result.yaw_error_rad)} "
        f"min_clearance={_fmt(result.min_clearance_m)} "
        f"motion_safe={int(result.motion_safe)} "
        f"reason={result.reason}"
    )


def _safe_interval_containing(
    n_ref: float,
    current_s: float,
    yaw_ref: float,
    held_geometry: HeldObjectGeometry,
    checker: CarriedEnvelopeChecker,
    params: Task3LateralGuardParams,
) -> tuple[float, float, float] | None:
    step = max(1e-3, float(params.collision_sample_step_m))
    s_min = min(params.workspace_s_min_m, current_s)
    s_max = max(params.workspace_s_max_m, current_s)
    samples = int(math.ceil((s_max - s_min) / step))
    current_check = checker.check_pose(
        (n_ref, current_s, yaw_ref),
        held_geometry.center_base,
        held_geometry.half_width_m,
    )
    if not current_check.safe:
        return None

    def _safe_at(s_value: float) -> EnvelopeCheck:
        return checker.check_pose(
            (n_ref, s_value, yaw_ref),
            held_geometry.center_base,
            held_geometry.half_width_m,
        )

    low = current_s
    high = current_s
    min_clearance = current_check.clearance_m
    probe = current_s - step
    while probe >= s_min - 1e-9:
        check = _safe_at(probe)
        if not check.safe:
            break
        low = probe
        min_clearance = min(min_clearance, check.clearance_m)
        probe -= step
    probe = current_s + step
    while probe <= s_max + 1e-9:
        check = _safe_at(probe)
        if not check.safe:
            break
        high = probe
        min_clearance = min(min_clearance, check.clearance_m)
        probe += step
    del samples
    return low, high, min_clearance


def _place_interval(
    legacy_s: float,
    params: Task3LateralGuardParams,
    *,
    place_radius_m: float | None,
    scoring_target_xy: tuple[float, float] | None,
    release_xy: tuple[float, float],
) -> tuple[float, float]:
    slack = params.lateral_deadband_m
    if place_radius_m is not None:
        radius = float(place_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            radius = params.lateral_deadband_m
        slack = max(slack, max(0.0, radius - params.place_radius_margin_m))
        if scoring_target_xy is not None and release_xy is not None:
            used = math.hypot(
                float(release_xy[0]) - float(scoring_target_xy[0]),
                float(release_xy[1]) - float(scoring_target_xy[1]),
            )
            remaining = max(
                0.0, radius - used - params.place_radius_margin_m
            )
            slack = max(params.lateral_deadband_m, remaining)
    return legacy_s - slack, legacy_s + slack


def _nearest_segment_safe_target(
    current_s: float,
    requested_s: float,
    n_ref: float,
    yaw_ref: float,
    current_pose: tuple[float, float, float],
    held_geometry: HeldObjectGeometry,
    checker: CarriedEnvelopeChecker,
    params: Task3LateralGuardParams,
    final_min: float,
    final_max: float,
    travel_face_yaw: float | None = None,
) -> tuple[float, EnvelopeCheck] | None:
    step = max(1e-3, float(params.collision_sample_step_m))
    direction = -1.0 if requested_s > current_s else 1.0
    s_value = requested_s
    best: tuple[float, EnvelopeCheck] | None = None
    while (direction > 0.0 and s_value <= current_s + 1e-9) or (
        direction < 0.0 and s_value >= current_s - 1e-9
    ):
        if final_min - 1e-9 <= s_value <= final_max + 1e-9:
            segment = is_lateral_motion_safe(
                (float(current_pose[0]), current_s, yaw_ref),
                (n_ref, s_value, yaw_ref),
                held_geometry,
                checker=checker,
                params=params,
                travel_face_yaw=travel_face_yaw,
            )
            if segment.safe:
                best = (s_value, segment)
                break
        s_value += direction * step
    return best


def _project(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _worse(*checks: EnvelopeCheck) -> EnvelopeCheck:
    best = checks[0]
    for check in checks[1:]:
        if check.clearance_m < best.clearance_m:
            best = check
    return best


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "GuardedLateralCommand",
    "PlacementFeasibility",
    "SafeLateralTarget",
    "Task3LateralGuardParams",
    "compute_safe_lateral_target",
    "evaluate_left_placement_feasibility",
    "format_task3_lateral_log",
    "left_placement_critical_white_y",
    "left_placement_required_width_m",
    "guard_task3_lateral_cmd",
    "is_lateral_motion_safe",
]
