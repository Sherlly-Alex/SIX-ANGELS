"""Construction and validation of hard scheduler action masks."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .action_space import DEFAULT_MAX_CANDIDATES


class InvalidActionMask(ValueError):
    """Raised when a mask cannot safely constrain a discrete policy."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def candidate_is_selectable(candidate: Any) -> bool:
    """Return whether a candidate/evaluation passed deterministic hard gates."""

    valid = _field(candidate, "valid", True)
    if valid is not True and valid != 1:
        return False
    if _field(candidate, "rejection_reasons", ()):
        return False
    utility = _field(candidate, "utility", None)
    if utility is not None and not _finite_number(utility):
        return False
    inner = _field(candidate, "candidate", candidate)
    metadata = _field(inner, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        if metadata.get("hard_constraint_passed") is False:
            return False
        if metadata.get("selectable") is False:
            return False
    hard_constraints = _field(inner, "hard_constraints", {}) or {}
    if isinstance(hard_constraints, Mapping) and any(
        not bool(allowed) for allowed in hard_constraints.values()
    ):
        return False
    return True


def build_action_mask(
    candidates: Sequence[Any] | Iterable[Any],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> np.ndarray:
    """Build a fixed-width mask; unused candidate slots are always false."""

    values = tuple(candidates)
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if len(values) > max_candidates:
        raise ValueError(
            f"received {len(values)} candidates for {max_candidates} mask entries"
        )
    mask = np.zeros(max_candidates, dtype=np.bool_)
    for index, candidate in enumerate(values):
        mask[index] = candidate_is_selectable(candidate)
    return mask


def validate_action_mask(
    mask: Sequence[bool] | np.ndarray,
    action_count: int,
    *,
    require_any: bool = True,
) -> np.ndarray:
    """Normalize a mask while rejecting NaN, non-binary and wrong-shape input."""

    if action_count <= 0:
        raise InvalidActionMask("action_count must be positive")
    try:
        raw = np.asarray(mask)
    except Exception as exc:  # defensive boundary around user/model data
        raise InvalidActionMask(f"action mask is not array-like: {exc}") from exc
    if raw.shape != (action_count,):
        raise InvalidActionMask(
            f"action mask shape must be ({action_count},), got {raw.shape}"
        )
    if raw.dtype == np.bool_:
        result = raw.astype(np.bool_, copy=True)
    else:
        try:
            numeric = raw.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidActionMask("action mask must contain boolean or 0/1 values") from exc
        if not np.all(np.isfinite(numeric)):
            raise InvalidActionMask("action mask contains NaN or infinity")
        if not np.all((numeric == 0.0) | (numeric == 1.0)):
            raise InvalidActionMask("numeric action mask entries must be exactly 0 or 1")
        result = numeric.astype(np.bool_)
    if require_any and not bool(np.any(result)):
        raise InvalidActionMask("action mask contains no selectable action")
    return result


def masked_argmax(values: Sequence[float], mask: Sequence[bool]) -> int:
    """Return a finite masked argmax with deterministic lowest-index tie break."""

    scores = np.asarray(values, dtype=np.float64)
    safe_mask = validate_action_mask(mask, int(scores.size), require_any=True)
    if scores.shape != safe_mask.shape:
        raise ValueError("scores must be a one-dimensional vector matching mask")
    selectable = safe_mask & np.isfinite(scores)
    if not bool(np.any(selectable)):
        raise ValueError("no selectable action has a finite score")
    safe_scores = np.where(selectable, scores, -np.inf)
    return int(np.argmax(safe_scores))


__all__ = [
    "InvalidActionMask",
    "build_action_mask",
    "candidate_is_selectable",
    "masked_argmax",
    "validate_action_mask",
]
