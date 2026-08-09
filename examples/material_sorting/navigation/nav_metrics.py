"""Phase-D navigation metric gates (handoff / plan §D).

``path_length`` / ``planned_straight`` describe the committed plan
(frozen at ``set_goal`` / replan).  Pass criteria:

1. Every sample with an active nav status has ``footprint_min_clearance > 0``.
2. While ``status == navigating`` and ``planned_straight`` is meaningful,
   ``path_length / planned_straight ≤ max_detour_ratio`` (default 2.0).
   Falls back to live ``straight_distance`` only when ``planned_straight``
   is missing (legacy JSONL).
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_ACTIVE = frozenset({
    "navigating",
    "replanning",
    "blocked",
    "final_positioning",
    "final_aligning",
    "emergency_stop",
})


def _as_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if is_dataclass(sample) and not isinstance(sample, type):
        return asdict(sample)
    raise TypeError(f"unsupported telemetry sample type: {type(sample)!r}")


def evaluate_nav_metrics(
    samples: Sequence[Any] | Iterable[Any],
    *,
    max_detour_ratio: float = 2.0,
    min_straight_m: float = 0.30,
    clearance_eps: float = 1e-9,
) -> Dict[str, Any]:
    """Evaluate phase-D gates over a telemetry stream.

    Returns a dict with ``ok`` (bool), ``failures`` (list[str]), and summary
    counters for logging / CI.
    """
    rows: List[Mapping[str, Any]] = [_as_mapping(s) for s in samples]
    failures: List[str] = []

    clear_violations = 0
    min_clear = float("inf")
    detour_violations = 0
    max_detour = 0.0
    navigating_ticks = 0
    # Deduplicate: same planned path should fail once, not once per tick.
    seen_detour_keys = set()

    for i, row in enumerate(rows):
        status = str(row.get("status", ""))
        clear = float(row.get("footprint_min_clearance", 0.0) or 0.0)
        if clear < min_clear:
            min_clear = clear

        if status in _ACTIVE and clear <= clearance_eps:
            clear_violations += 1
            if len(failures) < 8:
                failures.append(
                    f"[{i}] {status}: footprint_min_clearance={clear:.4f} <= 0"
                )

        if status != "navigating":
            continue
        navigating_ticks += 1
        path_len = float(row.get("path_length", 0.0) or 0.0)
        planned = float(row.get("planned_straight", 0.0) or 0.0)
        straight = (
            planned if planned > 0.0
            else float(row.get("straight_distance", 0.0) or 0.0)
        )
        if straight < min_straight_m:
            continue
        ratio = path_len / straight
        if ratio > max_detour:
            max_detour = ratio
        if ratio > max_detour_ratio + 1e-9:
            key = (round(path_len, 3), round(straight, 3))
            if key in seen_detour_keys:
                continue
            seen_detour_keys.add(key)
            detour_violations += 1
            if len(failures) < 8:
                failures.append(
                    f"[{i}] navigating detour path/planned_straight="
                    f"{path_len:.3f}/{straight:.3f}={ratio:.2f} > {max_detour_ratio}"
                )

    if not rows:
        failures.append("empty telemetry stream")

    ok = not failures
    return {
        "ok": ok,
        "failures": failures,
        "n_samples": len(rows),
        "navigating_ticks": navigating_ticks,
        "clear_violations": clear_violations,
        "detour_violations": detour_violations,
        "min_footprint_clearance": (0.0 if math.isinf(min_clear) else min_clear),
        "max_detour_ratio_seen": max_detour,
        "max_detour_ratio_limit": max_detour_ratio,
    }


def summarize_metrics(report: Mapping[str, Any]) -> str:
    """One-line human summary."""
    flag = "PASS" if report.get("ok") else "FAIL"
    return (
        f"NAV_METRICS_{flag} n={report.get('n_samples', 0)} "
        f"nav_ticks={report.get('navigating_ticks', 0)} "
        f"min_clear={float(report.get('min_footprint_clearance', 0.0)):.4f} "
        f"max_detour={float(report.get('max_detour_ratio_seen', 0.0)):.2f} "
        f"clear_viol={report.get('clear_violations', 0)} "
        f"detour_viol={report.get('detour_violations', 0)}"
    )
