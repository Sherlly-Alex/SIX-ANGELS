"""Gymnasium-style environment for constrained macro-action scheduling.

Gymnasium is optional.  The production Client can import this module with only
NumPy installed; when Gymnasium is present, standard ``Env`` and ``spaces`` are
used.  A simulator/backend is injected explicitly, keeping training code out of
the formal control path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import uuid

import numpy as np

from .action_mask import (
    InvalidActionMask,
    build_action_mask,
    validate_action_mask,
)
from .action_space import (
    ActionCatalog,
    DEFAULT_MAX_CANDIDATES,
    DiscreteMacroActionSpace,
)
from .observation import ObservationBuilder
from .reward import RewardBreakdown, RewardConfig, RewardEvent, SchedulingReward


try:  # optional integration; formal runtime does not require Gymnasium
    import gymnasium as _gym
    from gymnasium import spaces as _gym_spaces
except ImportError:  # pragma: no cover - branch depends on deployment image
    _gym = None
    _gym_spaces = None


class _FallbackEnv:
    metadata: Mapping[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Any:
        del seed, options

    def close(self) -> None:
        return None


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return array.shape == self.shape and bool(np.all(np.isfinite(array)))


_EnvBase = _gym.Env if _gym is not None else _FallbackEnv


@dataclass(frozen=True)
class SchedulingSnapshot:
    """All Client-visible state needed to select the next macro action."""

    candidates: tuple[Any, ...]
    public_state: Mapping[str, Any] = field(default_factory=dict)
    action_mask: Sequence[bool] | None = None
    episode_id: str | None = None

    def __init__(
        self,
        candidates: Sequence[Any],
        public_state: Mapping[str, Any] | None = None,
        action_mask: Sequence[bool] | None = None,
        episode_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "public_state", dict(public_state or {}))
        object.__setattr__(
            self,
            "action_mask",
            None if action_mask is None else tuple(action_mask),
        )
        object.__setattr__(self, "episode_id", episode_id)


@dataclass(frozen=True)
class SchedulingTransition:
    snapshot: SchedulingSnapshot
    events: tuple[RewardEvent | Mapping[str, Any], ...] = ()
    elapsed_s: float = 0.0
    path_length_m: float = 0.0
    obstacle_cost: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SchedulingBackend(Protocol):
    """Minimal simulator adapter expected by ``SchedulingEnv``."""

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> SchedulingSnapshot:
        ...

    def step(self, candidate: Any) -> SchedulingTransition:
        ...


def _coerce_snapshot(value: Any) -> SchedulingSnapshot:
    if isinstance(value, SchedulingSnapshot):
        return value
    if isinstance(value, Mapping):
        return SchedulingSnapshot(
            candidates=value.get("candidates", ()),
            public_state=value.get("public_state", {}),
            action_mask=value.get("action_mask"),
            episode_id=value.get("episode_id"),
        )
    raise TypeError("backend reset/snapshot must return SchedulingSnapshot or mapping")


def _coerce_transition(value: Any) -> SchedulingTransition:
    if isinstance(value, SchedulingTransition):
        return value
    if isinstance(value, Mapping):
        return SchedulingTransition(
            snapshot=_coerce_snapshot(value.get("snapshot", value)),
            events=tuple(value.get("events", ())),
            elapsed_s=float(value.get("elapsed_s", 0.0)),
            path_length_m=float(value.get("path_length_m", 0.0)),
            obstacle_cost=float(value.get("obstacle_cost", 0.0)),
            terminated=bool(value.get("terminated", False)),
            truncated=bool(value.get("truncated", False)),
            info=dict(value.get("info", {})),
        )
    raise TypeError("backend step must return SchedulingTransition or mapping")


class SchedulingEnv(_EnvBase):
    """Maskable discrete environment; policies select candidates, never motors."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: SchedulingBackend,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        reward_config: RewardConfig | None = None,
        require_safe_action: bool = True,
    ) -> None:
        if backend is None:
            raise ValueError("a scheduling simulation backend is required")
        self.backend = backend
        self.max_candidates = int(max_candidates)
        self.require_safe_action = bool(require_safe_action)
        self.observation_builder = ObservationBuilder(self.max_candidates)
        self.reward_model = SchedulingReward(reward_config)
        if _gym_spaces is not None:
            self.action_space = _gym_spaces.Discrete(self.max_candidates)
            self.observation_space = _gym_spaces.Box(
                low=-1.0e6,
                high=1.0e6,
                shape=self.observation_builder.shape,
                dtype=np.float32,
            )
        else:
            self.action_space = DiscreteMacroActionSpace(self.max_candidates)
            self.observation_space = _FallbackBox(
                -1.0e6, 1.0e6, self.observation_builder.shape, np.float32
            )
        self._snapshot: SchedulingSnapshot | None = None
        self._catalog: ActionCatalog | None = None
        self._mask = np.zeros(self.max_candidates, dtype=np.bool_)
        self._last_observation = np.zeros(
            self.observation_builder.shape, dtype=np.float32
        )

    def _install_snapshot(self, value: Any) -> np.ndarray:
        snapshot = _coerce_snapshot(value)
        catalog = ActionCatalog(snapshot.candidates, self.max_candidates)
        if snapshot.action_mask is None:
            mask = build_action_mask(snapshot.candidates, max_candidates=self.max_candidates)
        else:
            mask = validate_action_mask(
                snapshot.action_mask,
                self.max_candidates,
                require_any=self.require_safe_action,
            )
            intrinsic = build_action_mask(
                snapshot.candidates, max_candidates=self.max_candidates
            )
            if bool(np.any(mask & ~intrinsic)):
                raise InvalidActionMask(
                    "backend mask enables an empty or deterministically invalid candidate"
                )
            mask &= intrinsic
        if self.require_safe_action and not bool(np.any(mask)):
            raise InvalidActionMask("snapshot contains no safe macro action")
        observation = self.observation_builder.build(
            snapshot.public_state, snapshot.candidates, mask
        )
        self._snapshot = snapshot
        self._catalog = catalog
        self._mask = mask
        self._last_observation = observation
        return observation.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if _gym is not None:
            super().reset(seed=seed)
        elif hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        snapshot = _coerce_snapshot(self.backend.reset(seed=seed, options=options))
        episode_id = snapshot.episode_id or uuid.uuid4().hex
        self.reward_model.reset(episode_id)
        observation = self._install_snapshot(snapshot)
        return observation, {
            "action_mask": self.action_masks(),
            "episode_id": episode_id,
            "schema_hash": self.observation_builder.schema_hash,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._catalog is None:
            raise RuntimeError("reset() must be called before step()")
        action_valid = (
            isinstance(action, (int, np.integer))
            and not isinstance(action, (bool, np.bool_))
            and 0 <= int(action) < self.max_candidates
            and bool(self._mask[int(action)])
        )
        if not action_valid:
            breakdown = self.reward_model.score(invalid_action=True)
            return (
                self._last_observation.copy(),
                breakdown.total,
                False,
                False,
                {
                    "action_mask": self.action_masks(),
                    "invalid_action": True,
                    "reward_breakdown": breakdown,
                },
            )

        selected = self._catalog.resolve(int(action))
        transition = _coerce_transition(self.backend.step(selected))
        breakdown = self.reward_model.score(
            transition.events,
            elapsed_s=transition.elapsed_s,
            path_length_m=transition.path_length_m,
            obstacle_cost=transition.obstacle_cost,
        )
        observation = self._install_snapshot(transition.snapshot)
        info = dict(transition.info)
        info.update(
            {
                "action_mask": self.action_masks(),
                "selected_action_index": int(action),
                "reward_breakdown": breakdown,
                "transition_elapsed_s": transition.elapsed_s,
                "transition_path_length_m": transition.path_length_m,
                "transition_obstacle_cost": transition.obstacle_cost,
            }
        )
        # ActionCatalog is snapshot-local; record the id before replacement.
        try:
            from .action_space import candidate_action_id

            info["selected_action_id"] = candidate_action_id(selected)
        except ValueError:  # impossible after ActionCatalog construction
            pass
        return (
            observation,
            breakdown.total,
            transition.terminated,
            transition.truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        """Interface recognized by sb3-contrib's MaskablePPO."""

        return self._mask.copy()


__all__ = [
    "SchedulingBackend",
    "SchedulingEnv",
    "SchedulingSnapshot",
    "SchedulingTransition",
]
