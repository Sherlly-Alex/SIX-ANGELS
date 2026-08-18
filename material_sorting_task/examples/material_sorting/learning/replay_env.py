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
from .domain_randomization import DomainRandomizationConfig, DomainRandomizer
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
        randomization_config: DomainRandomizationConfig | None = None,
        best_reward: float = 1.0,
        utility_regret_scale: float = 1.0,
        invalid_action_reward: float = -100.0,
    ) -> None:
        if isinstance(dataset, (str, Path)):
            records = load_replay_dataset(dataset)
        else:
            records = tuple(dict(item) for item in dataset)
        if not records:
            raise ValueError("replay dataset is empty")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        for name, value in (
            ("best_reward", best_reward),
            ("utility_regret_scale", utility_regret_scale),
            ("invalid_action_reward", invalid_action_reward),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if utility_regret_scale < 0.0:
            raise ValueError("utility_regret_scale must be non-negative")
        self.records = records
        self.max_candidates = int(records[0]["max_candidates"])
        self.episode_length = min(int(episode_length), len(records))
        self.best_reward = float(best_reward)
        self.utility_regret_scale = float(utility_regret_scale)
        self.invalid_action_reward = float(invalid_action_reward)
        self._randomizer = (
            None
            if randomization_config is None
            else DomainRandomizer(randomization_config)
        )
        self._slot_rng = random.Random()
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

    def _materialize(self, source: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(source)
        record["observation"] = np.asarray(
            source["observation"], dtype=np.float32
        ).copy()
        record["action_mask"] = np.asarray(
            source["action_mask"], dtype=np.bool_
        ).copy()
        record["candidate_utilities"] = list(source["candidate_utilities"])
        if self._randomizer is None:
            return record

        sample = self._randomizer.sample()
        observation = record["observation"]
        mask = record["action_mask"]
        utilities = record["candidate_utilities"]
        global_indices = {
            name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)
        }
        observation[global_indices["robot_x_m"]] += sample.pose_dx_m
        observation[global_indices["robot_y_m"]] += sample.pose_dy_m
        observation[global_indices["robot_yaw_rad"]] += sample.yaw_delta_rad

        width = len(CANDIDATE_FEATURE_NAMES)
        base = len(GLOBAL_FEATURE_NAMES)
        feature_indices = {
            name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)
        }
        manipulation_delta = abs(sample.depth_scale - 1.0) + abs(
            sample.friction_scale - 1.0
        )
        uncertainty_delta = abs(sample.detection_delta) + (
            0.25 if sample.detection_dropout else 0.0
        )
        dynamic_delta = 0.15 if sample.dynamic_obstacle_present else 0.0
        for slot in np.flatnonzero(mask):
            slot = int(slot)
            offset = base + slot * width
            utility = float(utilities[slot])
            old_time = float(observation[offset + feature_indices["expected_time_s"]])
            new_time = old_time / max(0.1, sample.speed_scale) + sample.message_latency_s
            observation[offset + feature_indices["expected_time_s"]] = new_time
            utility -= 0.20 * (new_time - old_time)
            observation[
                offset + feature_indices["perception_uncertainty"]
            ] += uncertainty_delta
            utility -= 2.0 * uncertainty_delta
            observation[
                offset + feature_indices["manipulation_difficulty"]
            ] += manipulation_delta
            utility -= 1.50 * manipulation_delta
            observation[offset + feature_indices["dynamic_risk"]] += dynamic_delta
            utility -= 1.50 * dynamic_delta
            observation[offset + feature_indices["utility"]] = utility
            utilities[slot] = utility

        allowed = [int(index) for index in np.flatnonzero(mask)]
        if sample.planner_failure and len(allowed) > 1:
            failed_slot = self._slot_rng.choice(allowed)
            mask[failed_slot] = False
            utilities[failed_slot] = None
            offset = base + failed_slot * width
            observation[offset + feature_indices["valid"]] = 0.0
            observation[offset + feature_indices["action_mask"]] = 0.0
            observation[offset + feature_indices["utility"]] = 0.0
        if not bool(np.any(mask)) or not bool(np.all(np.isfinite(observation))):
            raise RuntimeError("domain randomization produced an invalid replay state")
        return record

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if _gym is not None:
            super().reset(seed=seed)
        rng = random.Random(seed)
        if self._randomizer is not None:
            self._randomizer.reset(seed)
        self._slot_rng.seed(seed)
        self._order = list(range(len(self.records)))
        rng.shuffle(self._order)
        self._cursor = 0
        self._current = self._materialize(self.records[self._order[0]])
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
        source_sha256 = self._current.get("source_sha256")
        utilities = self._current["candidate_utilities"]
        allowed_indices = [int(index) for index in np.flatnonzero(mask)]
        heuristic_index = max(
            allowed_indices,
            key=lambda index: (float(utilities[index]), -index),
        )
        if valid_action:
            chosen_utility = float(utilities[int(action)])
            best_utility = max(
                float(value)
                for value, allowed in zip(utilities, mask)
                if allowed
            )
            utility_regret = max(0.0, best_utility - chosen_utility)
            reward = self.best_reward - self.utility_regret_scale * utility_regret
        else:
            utility_regret = math.inf
            reward = self.invalid_action_reward

        self._cursor += 1
        terminated = self._cursor >= self.episode_length
        if not terminated:
            self._current = self._materialize(
                self.records[self._order[self._cursor]]
            )
        observation = np.asarray(self._current["observation"], dtype=np.float32).copy()
        return observation, reward, terminated, False, {
            "invalid_action": not valid_action,
            "heuristic_action_index": heuristic_index,
            "exact_match": valid_action and int(action) == heuristic_index,
            "utility_regret": utility_regret,
            "source_sha256": source_sha256,
        }


def load_replay_training_config(path: str | Path) -> dict[str, Any]:
    """Load the versioned, provenance-hashed replay training configuration."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        "scheduler-replay-training-v1"
    ):
        raise ValueError("replay training config schema mismatch")
    allowed = {
        "schema_version",
        "episode_length",
        "randomize",
        "domain_randomization",
        "reward",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown replay training config keys: {sorted(unknown)}")
    if not isinstance(value.get("domain_randomization"), dict):
        raise ValueError("domain_randomization must be an object")
    if not isinstance(value.get("reward"), dict):
        raise ValueError("reward must be an object")
    if not isinstance(value.get("randomize"), bool):
        raise ValueError("randomize must be boolean")
    return value


def build_replay_env() -> ReplayBanditEnv:
    """Factory for train_maskable_ppo's module:function CLI boundary."""

    path = os.environ.get("MATERIAL_SCHEDULER_REPLAY_DATASET", "").strip()
    if not path:
        raise RuntimeError("MATERIAL_SCHEDULER_REPLAY_DATASET is required")
    config_path = os.environ.get("MATERIAL_SCHEDULER_REPLAY_CONFIG", "").strip()
    if not config_path:
        config_path = str(
            Path(__file__).resolve().parent
            / "configs"
            / "replay_training_v1.json"
        )
    config = load_replay_training_config(config_path)
    randomization = DomainRandomizationConfig(**config["domain_randomization"])
    reward = config["reward"]
    return ReplayBanditEnv(
        path,
        episode_length=int(config["episode_length"]),
        randomization_config=(randomization if bool(config["randomize"]) else None),
        best_reward=float(reward["best_reward"]),
        utility_regret_scale=float(reward["utility_regret_scale"]),
        invalid_action_reward=float(reward["invalid_action_reward"]),
    )


__all__ = [
    "ReplayBanditEnv",
    "build_replay_env",
    "load_replay_dataset",
    "load_replay_training_config",
]
