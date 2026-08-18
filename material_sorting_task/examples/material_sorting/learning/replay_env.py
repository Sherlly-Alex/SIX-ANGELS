"""Maskable contextual-bandit environment backed by validated replay JSONL."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .event_replay import REPLAY_DATASET_SCHEMA_VERSION
from .observation import (
    CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    observation_schema_hash,
)
from .action_space import DiscreteMacroActionSpace


try:
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
except ImportError:  # pragma: no cover - deployment dependent
    _gym = None
    _spaces = None


class _FallbackEnv:
    metadata: Mapping[str, Any] = {}


class _FallbackBox:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


_EnvBase = _gym.Env if _gym is not None else _FallbackEnv


def load_replay_dataset(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load a dataset only after revalidating its fixed learning boundary."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: record is not an object")
        if value.get("dataset_schema_version") != REPLAY_DATASET_SCHEMA_VERSION:
            raise ValueError(f"line {line_number}: replay dataset schema mismatch")
        try:
            maximum = int(value["max_candidates"])
            selected = int(value["selected_action_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: invalid slot metadata") from exc
        observation = value.get("observation")
        mask = value.get("action_mask")
        action_ids = value.get("candidate_action_ids")
        utilities = value.get("candidate_utilities")
        if not all(
            isinstance(item, list)
            for item in (observation, mask, action_ids, utilities)
        ):
            raise ValueError(f"line {line_number}: replay arrays are missing")
        if not (
            len(mask) == len(action_ids) == len(utilities) == maximum
            and 0 <= selected < maximum
            and all(isinstance(item, bool) for item in mask)
            and mask[selected]
        ):
            raise ValueError(f"line {line_number}: replay mask/slot mismatch")
        observation_array = np.asarray(observation, dtype=np.float32)
        if (
            observation_array.ndim != 1
            or observation_array.size
            != len(GLOBAL_FEATURE_NAMES) + maximum * len(CANDIDATE_FEATURE_NAMES)
            or not bool(np.all(np.isfinite(observation_array)))
            or value.get("observation_schema_hash")
            != observation_schema_hash(maximum)
        ):
            raise ValueError(f"line {line_number}: observation validation failed")
        for index, allowed in enumerate(mask):
            utility = utilities[index]
            if allowed:
                try:
                    finite = math.isfinite(float(utility))
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if not finite or not isinstance(action_ids[index], str):
                    raise ValueError(
                        f"line {line_number}: enabled slot lacks action/utility"
                    )
            elif utility is not None:
                raise ValueError(
                    f"line {line_number}: masked slot exposes a training utility"
                )
        record = dict(value)
        record["observation"] = observation_array
        record["action_mask"] = np.asarray(mask, dtype=np.bool_)
        records.append(record)
    if not records:
        raise ValueError("replay dataset is empty")
    maximums = {int(record["max_candidates"]) for record in records}
    shapes = {tuple(record["observation"].shape) for record in records}
    if len(maximums) != 1 or len(shapes) != 1:
        raise ValueError("replay dataset mixes action or observation schemas")
    return tuple(records)


class ReplayBanditEnv(_EnvBase):
    """Learn candidate ranking from exact production decision snapshots."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: str | Path | Sequence[Mapping[str, Any]],
        *,
        episode_length: int = 256,
    ) -> None:
        if isinstance(dataset, (str, Path)):
            records = load_replay_dataset(dataset)
        else:
            records = tuple(dict(item) for item in dataset)
        if not records:
            raise ValueError("replay dataset is empty")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        self.records = records
        self.max_candidates = int(records[0]["max_candidates"])
        self.episode_length = min(int(episode_length), len(records))
        observation_shape = tuple(np.asarray(records[0]["observation"]).shape)
        if _spaces is not None:
            self.action_space = _spaces.Discrete(self.max_candidates)
            self.observation_space = _spaces.Box(
                low=-1.0e6,
                high=1.0e6,
                shape=observation_shape,
                dtype=np.float32,
            )
        else:
            self.action_space = DiscreteMacroActionSpace(self.max_candidates)
            self.observation_space = _FallbackBox(observation_shape)
        self._order = list(range(len(records)))
        self._cursor = 0
        self._current: Mapping[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if _gym is not None:
            super().reset(seed=seed)
        rng = random.Random(seed)
        self._order = list(range(len(self.records)))
        rng.shuffle(self._order)
        self._cursor = 0
        self._current = self.records[self._order[0]]
        return np.asarray(self._current["observation"], dtype=np.float32).copy(), {
            "action_mask": self.action_masks(),
            "source_sha256": self._current.get("source_sha256"),
        }

    def action_masks(self) -> np.ndarray:
        if self._current is None:
            return np.zeros(self.max_candidates, dtype=np.bool_)
        return np.asarray(self._current["action_mask"], dtype=np.bool_).copy()

    def step(self, action: int):
        if self._current is None:
            raise RuntimeError("reset() must be called before step()")
        mask = self.action_masks()
        valid_action = (
            isinstance(action, (int, np.integer))
            and not isinstance(action, (bool, np.bool_))
            and 0 <= int(action) < self.max_candidates
            and bool(mask[int(action)])
        )
        heuristic_index = int(self._current["selected_action_index"])
        source_sha256 = self._current.get("source_sha256")
        if valid_action:
            utilities = self._current["candidate_utilities"]
            chosen_utility = float(utilities[int(action)])
            best_utility = max(
                float(value)
                for value, allowed in zip(utilities, mask)
                if allowed
            )
            utility_regret = max(0.0, best_utility - chosen_utility)
            reward = 1.0 - utility_regret
        else:
            utility_regret = math.inf
            reward = -100.0

        self._cursor += 1
        terminated = self._cursor >= self.episode_length
        if not terminated:
            self._current = self.records[self._order[self._cursor]]
        observation = np.asarray(self._current["observation"], dtype=np.float32).copy()
        return observation, reward, terminated, False, {
            "invalid_action": not valid_action,
            "heuristic_action_index": heuristic_index,
            "exact_match": valid_action and int(action) == heuristic_index,
            "utility_regret": utility_regret,
            "source_sha256": source_sha256,
        }


def build_replay_env() -> ReplayBanditEnv:
    """Factory for train_maskable_ppo's module:function CLI boundary."""

    path = os.environ.get("MATERIAL_SCHEDULER_REPLAY_DATASET", "").strip()
    if not path:
        raise RuntimeError("MATERIAL_SCHEDULER_REPLAY_DATASET is required")
    episode_length = int(
        os.environ.get("MATERIAL_SCHEDULER_REPLAY_EPISODE_LENGTH", "256")
    )
    return ReplayBanditEnv(path, episode_length=episode_length)


__all__ = ["ReplayBanditEnv", "build_replay_env", "load_replay_dataset"]
