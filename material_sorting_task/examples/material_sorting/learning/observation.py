"""Stable, allow-listed observation encoding for scheduler learning.

Only public Client-side state and deterministic candidate evaluations are read.
Unknown mapping keys are deliberately ignored so Server-private truth or the
semantic-audit side channel cannot accidentally enter a training observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import DEFAULT_MAX_CANDIDATES


OBSERVATION_SCHEMA_VERSION = "scheduler-observation-v1"

GLOBAL_FEATURE_NAMES = (
    "task_ordinal",
    "attempt",
    "step_progress",
    "remaining_time_s",
    "robot_x_m",
    "robot_y_m",
    "robot_yaw_rad",
    "payload_code",
    "referee_allowed",
    "recovery_count",
)

CANDIDATE_FEATURE_NAMES = (
    "expected_score",
    "success_probability",
    "expected_time_s",
    "path_length_m",
    "min_clearance_m",
    "obstacle_integral",
    "heading_change_rad",
    "dynamic_risk",
    "perception_uncertainty",
    "manipulation_difficulty",
    "irreversible_risk",
    "recovery_cost",
    "utility",
    "valid",
    "action_mask",
    "slot_present",
)

_GLOBAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "task_ordinal": ("task_ordinal", "task_id", "task"),
    "attempt": ("attempt",),
    "step_progress": ("step_progress", "progress"),
    "remaining_time_s": ("remaining_time_s", "remaining_s"),
    "robot_x_m": ("robot_x_m", "robot_x", "x"),
    "robot_y_m": ("robot_y_m", "robot_y", "y"),
    "robot_yaw_rad": ("robot_yaw_rad", "robot_yaw", "yaw"),
    "payload_code": ("payload_code", "footprint_mode_code"),
    "referee_allowed": ("referee_allowed", "execution_allowed"),
    "recovery_count": ("recovery_count",),
}

_CANDIDATE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "expected_score": ("expected_score",),
    "success_probability": ("success_probability",),
    "expected_time_s": ("expected_time_s",),
    "path_length_m": ("path_length_m", "path_length"),
    "min_clearance_m": ("min_clearance_m", "min_clearance"),
    "obstacle_integral": (
        "obstacle_integral",
        "inflation_integral",
        "inflation_cost_integral",
    ),
    "heading_change_rad": ("heading_change_rad", "heading_change"),
    "dynamic_risk": ("dynamic_risk",),
    "perception_uncertainty": ("perception_uncertainty",),
    "manipulation_difficulty": ("manipulation_difficulty",),
    "irreversible_risk": ("irreversible_risk",),
    "recovery_cost": ("recovery_cost",),
    "utility": ("utility",),
}


def observation_schema_hash(max_candidates: int = DEFAULT_MAX_CANDIDATES) -> str:
    """Return the hash persisted beside trained policies."""

    payload = {
        "version": OBSERVATION_SCHEMA_VERSION,
        "max_candidates": int(max_candidates),
        "global": GLOBAL_FEATURE_NAMES,
        "candidate": CANDIDATE_FEATURE_NAMES,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(obj: Any, aliases: Sequence[str], default: Any = 0.0) -> Any:
    for name in aliases:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if not isinstance(obj, Mapping) and hasattr(obj, name):
            return getattr(obj, name)
    return default


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return float(np.clip(number, -1.0e6, 1.0e6))


def _candidate_source(value: Any) -> tuple[Any, Any, Any]:
    """Return (evaluation, candidate, path_metrics) using duck typing."""

    evaluation = value
    candidate = _read(value, ("candidate",), value)
    path_metrics = _read(value, ("path_metrics",), None)
    if path_metrics is None:
        path_metrics = _read(candidate, ("path_metrics",), None)
    return evaluation, candidate, path_metrics


class ObservationBuilder:
    """Encode world state and candidate slots into a fixed float32 vector."""

    def __init__(self, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = int(max_candidates)

    @property
    def size(self) -> int:
        return len(GLOBAL_FEATURE_NAMES) + (
            self.max_candidates * len(CANDIDATE_FEATURE_NAMES)
        )

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    @property
    def schema_hash(self) -> str:
        return observation_schema_hash(self.max_candidates)

    def build(
        self,
        world_state: Mapping[str, Any] | Any,
        candidates: Sequence[Any],
        action_mask: Sequence[bool] | None = None,
    ) -> np.ndarray:
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"received {len(candidates)} candidates for "
                f"{self.max_candidates} observation slots"
            )
        if action_mask is None:
            mask = [True] * len(candidates) + [False] * (
                self.max_candidates - len(candidates)
            )
        else:
            if len(action_mask) != self.max_candidates:
                raise ValueError(
                    f"action_mask must contain {self.max_candidates} entries"
                )
            mask = [bool(value) for value in action_mask]

        result = np.zeros(self.size, dtype=np.float32)
        offset = 0
        for name in GLOBAL_FEATURE_NAMES:
            value = _read(world_state, _GLOBAL_ALIASES[name], 0.0)
            result[offset] = _finite_float(value)
            offset += 1

        width = len(CANDIDATE_FEATURE_NAMES)
        for slot, wrapped in enumerate(candidates):
            evaluation, candidate, path_metrics = _candidate_source(wrapped)
            base = len(GLOBAL_FEATURE_NAMES) + slot * width
            for feature_index, name in enumerate(CANDIDATE_FEATURE_NAMES):
                if name == "valid":
                    value = _read(evaluation, ("valid",), True)
                elif name == "action_mask":
                    value = mask[slot]
                elif name == "slot_present":
                    value = 1.0
                elif name == "utility":
                    value = _read(evaluation, _CANDIDATE_ALIASES[name], 0.0)
                elif name in {
                    "path_length_m",
                    "min_clearance_m",
                    "obstacle_integral",
                    "heading_change_rad",
                    "dynamic_risk",
                }:
                    value = _read(path_metrics, _CANDIDATE_ALIASES[name], None)
                    if value is None:
                        value = _read(candidate, _CANDIDATE_ALIASES[name], 0.0)
                else:
                    value = _read(candidate, _CANDIDATE_ALIASES[name], 0.0)
                result[base + feature_index] = _finite_float(value)

        # This is both an invariant and a last boundary against corrupt logs.
        if not bool(np.all(np.isfinite(result))):  # pragma: no cover - defensive
            raise ValueError("observation encoder produced a non-finite value")
        return result


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "OBSERVATION_SCHEMA_VERSION",
    "ObservationBuilder",
    "observation_schema_hash",
]
