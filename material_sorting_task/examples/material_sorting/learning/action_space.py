"""Finite macro-action slots used by the scheduling learner.

The scheduler produces a variable number of safe, semantically meaningful
``CandidateAction`` objects.  Learning algorithms, on the other hand, need a
fixed discrete action space.  ``ActionCatalog`` bridges the two without
allowing a policy to synthesize motor commands: action ``i`` can only select
candidate slot ``i`` from the current scheduler snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MAX_CANDIDATES = 12


class MacroActionKind(str, Enum):
    """Canonical action categories used for logging and offline analysis."""

    NAVIGATE = "navigate"
    OBSERVE = "observe"
    SCAN = "scan"
    REPLAN = "replan"
    RETRY_ALIGNMENT = "retry_alignment"
    RECOVER = "recover"
    ABORT_ATTEMPT = "abort_attempt"
    HOLD = "hold"
    OTHER = "other"


def candidate_action_id(candidate: Any) -> str:
    """Return a stable action id from a candidate or evaluation wrapper."""

    if hasattr(candidate, "candidate"):
        candidate = candidate.candidate
    if isinstance(candidate, Mapping):
        value = candidate.get("action_id")
    else:
        value = getattr(candidate, "action_id", None)
    if value is None or not str(value).strip():
        raise ValueError("every macro-action candidate requires a non-empty action_id")
    return str(value)


def candidate_action_type(candidate: Any) -> str:
    """Return the scheduler action type while accepting evaluation wrappers."""

    if hasattr(candidate, "candidate"):
        candidate = candidate.candidate
    if isinstance(candidate, Mapping):
        value = candidate.get("action_type", MacroActionKind.OTHER.value)
    else:
        value = getattr(candidate, "action_type", MacroActionKind.OTHER.value)
    if isinstance(value, Enum):
        value = value.value
    return str(value)


@dataclass(frozen=True)
class ActionSlot:
    """One immutable candidate-to-discrete-index binding."""

    index: int
    action_id: str
    action_type: str
    candidate: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class ActionCatalog:
    """Snapshot-local mapping between discrete indices and candidates."""

    candidates: tuple[Any, ...]
    max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __init__(
        self,
        candidates: Sequence[Any] | Iterable[Any],
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        values = tuple(candidates)
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if len(values) > max_candidates:
            raise ValueError(
                f"received {len(values)} candidates for {max_candidates} action slots"
            )
        ids = tuple(candidate_action_id(candidate) for candidate in values)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate action_id values must be unique per snapshot")
        object.__setattr__(self, "candidates", values)
        object.__setattr__(self, "max_candidates", int(max_candidates))

    @property
    def n(self) -> int:
        return self.max_candidates

    @property
    def slots(self) -> tuple[ActionSlot, ...]:
        return tuple(
            ActionSlot(
                index=index,
                action_id=candidate_action_id(candidate),
                action_type=candidate_action_type(candidate),
                candidate=candidate,
            )
            for index, candidate in enumerate(self.candidates)
        )

    def contains(self, action: Any) -> bool:
        return isinstance(action, int) and not isinstance(action, bool) and (
            0 <= action < self.max_candidates
        )

    def resolve(self, action: int) -> Any:
        if not self.contains(action):
            raise IndexError(f"action index {action!r} is outside [0, {self.n})")
        if action >= len(self.candidates):
            raise IndexError(f"action slot {action} is empty in the current snapshot")
        return self.candidates[action]

    def index_of(self, action_id: str) -> int | None:
        wanted = str(action_id)
        for slot in self.slots:
            if slot.action_id == wanted:
                return slot.index
        return None


class DiscreteMacroActionSpace:
    """Small Gym-compatible fallback for deployments without Gymnasium."""

    def __init__(self, n: int, seed: int | None = None) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self._rng = random.Random(seed)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng.seed(seed)
        return [seed]

    def contains(self, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and (
            0 <= value < self.n
        )

    def sample(self, mask: Sequence[bool] | None = None) -> int:
        choices = list(range(self.n))
        if mask is not None:
            if len(mask) != self.n:
                raise ValueError(f"mask must contain exactly {self.n} entries")
            choices = [index for index, allowed in enumerate(mask) if bool(allowed)]
        if not choices:
            raise ValueError("cannot sample from an empty action mask")
        return self._rng.choice(choices)


__all__ = [
    "ActionCatalog",
    "ActionSlot",
    "DEFAULT_MAX_CANDIDATES",
    "DiscreteMacroActionSpace",
    "MacroActionKind",
    "candidate_action_id",
    "candidate_action_type",
]
