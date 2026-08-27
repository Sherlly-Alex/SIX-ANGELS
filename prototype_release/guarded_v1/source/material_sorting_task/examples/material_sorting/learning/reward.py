"""Referee-aligned scheduling rewards with one-shot event deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping


DEFAULT_EVENT_REWARDS: Mapping[str, float] = {
    "referee_task_success": 100.0,
    "grasp_stable": 20.0,
    "place_confirmed": 10.0,
    "key_step_completed": 3.0,
    "replan": -2.0,
    "local_recovery": -5.0,
    "grasp_failed": -20.0,
    "object_dropped": -40.0,
    "attempt_failed": -100.0,
    "collision_or_safety_violation": -500.0,
}


@dataclass(frozen=True)
class RewardConfig:
    event_rewards: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVENT_REWARDS)
    )
    time_penalty_per_s: float = 0.02
    path_penalty_per_m: float = 0.10
    obstacle_cost_weight: float = 0.01
    invalid_action_penalty: float = 25.0

    def __post_init__(self) -> None:
        values = (
            self.time_penalty_per_s,
            self.path_penalty_per_m,
            self.obstacle_cost_weight,
            self.invalid_action_penalty,
            *self.event_rewards.values(),
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("reward coefficients must be finite")
        if any(
            value < 0.0
            for value in (
                self.time_penalty_per_s,
                self.path_penalty_per_m,
                self.obstacle_cost_weight,
                self.invalid_action_penalty,
            )
        ):
            raise ValueError("penalty magnitudes must be non-negative")


@dataclass(frozen=True)
class RewardEvent:
    """An externally observed event; ``event_id`` must be stable on retries."""

    name: str
    event_id: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward event name must be non-empty")
        if not self.event_id:
            raise ValueError("reward event_id/step_run_id must be non-empty")
        if not math.isfinite(float(self.scale)):
            raise ValueError("reward event scale must be finite")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    event_reward: float
    time_penalty: float
    path_penalty: float
    obstacle_penalty: float
    invalid_action_penalty: float
    accepted_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    unknown_events: tuple[str, ...]


class RewardLedger:
    """Episode-local exactly-once ledger for physical/referee events."""

    def __init__(self) -> None:
        self.episode_id = ""
        self._seen: set[str] = set()

    def reset(self, episode_id: str) -> None:
        if not str(episode_id):
            raise ValueError("episode_id must be non-empty")
        self.episode_id = str(episode_id)
        self._seen.clear()

    def accept(self, event: RewardEvent) -> bool:
        if not self.episode_id:
            raise RuntimeError("reward ledger must be reset before use")
        # A single physical/referee event id must never be paid under two
        # different labels.  This also closes a common reward-hacking route.
        if event.event_id in self._seen:
            return False
        self._seen.add(event.event_id)
        return True


def coerce_reward_event(value: RewardEvent | Mapping[str, Any]) -> RewardEvent:
    if isinstance(value, RewardEvent):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("reward events must be RewardEvent objects or mappings")
    event_id = value.get("event_id", value.get("step_run_id", ""))
    return RewardEvent(
        name=str(value.get("name", value.get("event_type", ""))),
        event_id=str(event_id),
        scale=float(value.get("scale", 1.0)),
    )


class SchedulingReward:
    """Compute shaped reward while preserving event exactly-once semantics."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.ledger = RewardLedger()

    def reset(self, episode_id: str) -> None:
        self.ledger.reset(episode_id)

    def score(
        self,
        events: Iterable[RewardEvent | Mapping[str, Any]] = (),
        *,
        elapsed_s: float = 0.0,
        path_length_m: float = 0.0,
        obstacle_cost: float = 0.0,
        invalid_action: bool = False,
    ) -> RewardBreakdown:
        magnitudes = (elapsed_s, path_length_m, obstacle_cost)
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in magnitudes):
            raise ValueError("elapsed_s, path_length_m and obstacle_cost must be finite >= 0")

        event_reward = 0.0
        accepted: list[str] = []
        duplicates: list[str] = []
        unknown: list[str] = []
        for raw_event in events:
            event = coerce_reward_event(raw_event)
            if not self.ledger.accept(event):
                duplicates.append(event.event_id)
                continue
            coefficient = self.config.event_rewards.get(event.name)
            if coefficient is None:
                unknown.append(event.name)
                continue
            event_reward += float(coefficient) * float(event.scale)
            accepted.append(event.event_id)

        time_penalty = self.config.time_penalty_per_s * float(elapsed_s)
        path_penalty = self.config.path_penalty_per_m * float(path_length_m)
        obstacle_penalty = self.config.obstacle_cost_weight * float(obstacle_cost)
        invalid_penalty = self.config.invalid_action_penalty if invalid_action else 0.0
        total = (
            event_reward
            - time_penalty
            - path_penalty
            - obstacle_penalty
            - invalid_penalty
        )
        return RewardBreakdown(
            total=float(total),
            event_reward=float(event_reward),
            time_penalty=float(time_penalty),
            path_penalty=float(path_penalty),
            obstacle_penalty=float(obstacle_penalty),
            invalid_action_penalty=float(invalid_penalty),
            accepted_event_ids=tuple(accepted),
            duplicate_event_ids=tuple(duplicates),
            unknown_events=tuple(unknown),
        )


__all__ = [
    "DEFAULT_EVENT_REWARDS",
    "RewardBreakdown",
    "RewardConfig",
    "RewardEvent",
    "RewardLedger",
    "SchedulingReward",
    "coerce_reward_event",
]
