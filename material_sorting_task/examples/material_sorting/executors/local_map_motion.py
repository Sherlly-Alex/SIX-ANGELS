"""Local-map motion helpers (fail-open, APPLY-gated)."""

from __future__ import annotations

import math
from typing import Any, Mapping

from executors.base import ExecutionContext

# Tier-2: corridor clear → faster straight / nav segments; near hit → slower
# only when MATERIAL_LOCAL_MAP_SPEED_MODE=full.
LOCAL_MAP_CLEAR_LINEAR_BOOST = 1.40
LOCAL_MAP_NEAR_HIT_LINEAR_SCALE = 0.52
LOCAL_MAP_NEAR_HIT_MAX_DIST_M = 0.38
LOCAL_MAP_MID_HIT_LINEAR_SCALE = 0.85
LOCAL_MAP_SPEED_APPLY_MAX_DIST_M = 1.00

MAP_STANDOFF_APPLY_MAX_DIST_M = 1.00
MAP_STANDOFF_OBSTACLE_MARGIN_M = 0.10


def map_standoff_m(
    context: ExecutionContext,
    fallback_m: float,
    *,
    max_standoff_m: float,
) -> float:
    """Return a farther-only standoff from fresh, APPLY-gated advice."""

    fallback = float(fallback_m)
    advice = getattr(context, "local_map_advice", None)
    data: Mapping[str, Any] | None = advice if isinstance(advice, Mapping) else None
    if data is None:
        return fallback
    if not data.get("enabled") or not data.get("apply") or not data.get("fresh"):
        return fallback
    if data.get("clear") is not False:
        return fallback
    try:
        hit_dist = float(data.get("distance_m"))
        suggested = float(data.get("suggested_standoff_m"))
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(hit_dist) or not math.isfinite(suggested):
        return fallback
    if hit_dist >= MAP_STANDOFF_APPLY_MAX_DIST_M:
        return fallback
    dist_based = hit_dist + MAP_STANDOFF_OBSTACLE_MARGIN_M
    return min(float(max_standoff_m), max(fallback, suggested, dist_based))


def map_linear_scale(context: ExecutionContext | None) -> float:
    """Return a linear-speed multiplier for the current control tick."""

    if context is None:
        return 1.0
    advice = getattr(context, "local_map_advice", None)
    try:
        from perception.local_map_sidecar import local_map_linear_scale
    except ImportError:
        try:
            from local_map_sidecar import local_map_linear_scale  # type: ignore
        except ImportError:
            return 1.0
    return float(local_map_linear_scale(advice))
