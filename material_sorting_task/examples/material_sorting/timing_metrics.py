"""Small ROS-free timing helpers for state-machine performance telemetry."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping


def percentile95(values: list[float] | tuple[float, ...]) -> float | None:
    """Return the nearest-rank P95 used by the competition timing logs."""

    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    return finite[max(0, math.ceil(0.95 * len(finite)) - 1)]


class TimingRecorder:
    """Record one active phase and retain samples across executor retries."""

    def __init__(self, scope: str) -> None:
        self.scope = str(scope)
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.active_name: str | None = None
        self.active_started_s: float | None = None
        self.active_metadata: dict[str, Any] = {}

    def reset_active(self) -> None:
        self.active_name = None
        self.active_started_s = None
        self.active_metadata = {}

    def begin(
        self,
        name: str,
        now_s: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.reset_active()
        self.active_name = str(name)
        self.active_started_s = float(now_s)
        self.active_metadata = dict(metadata or {})

    def transition(
        self,
        name: str,
        now_s: float,
        *,
        outcome: str = "completed",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        event = self.finish(now_s, outcome=outcome)
        self.begin(name, now_s, metadata)
        return event

    def finish(self, now_s: float, *, outcome: str = "completed") -> dict[str, Any] | None:
        if self.active_name is None or self.active_started_s is None:
            return None
        duration = max(0.0, float(now_s) - self.active_started_s)
        name = self.active_name
        self.samples[name].append(duration)
        event: dict[str, Any] = {
            "event": "timing",
            "scope": self.scope,
            "name": name,
            "duration_s": round(duration, 6),
            "outcome": str(outcome),
            "count": len(self.samples[name]),
            "mean_s": round(sum(self.samples[name]) / len(self.samples[name]), 6),
            "p95_s": round(percentile95(self.samples[name]) or 0.0, 6),
            **self.active_metadata,
        }
        print("[TIMING] " + json.dumps(event, ensure_ascii=False, sort_keys=True))
        self.reset_active()
        return event

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, values in self.samples.items():
            if not values:
                continue
            result[name] = {
                "count": len(values),
                "mean_s": sum(values) / len(values),
                "p95_s": percentile95(values) or 0.0,
            }
        return result


__all__ = ["TimingRecorder", "percentile95"]
