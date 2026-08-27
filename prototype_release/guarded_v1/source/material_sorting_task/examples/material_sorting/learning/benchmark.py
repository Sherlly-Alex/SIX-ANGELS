"""Paired-seed benchmark for Heuristic versus constrained RL scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import coerce_discrete_action
from .observation import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES


@dataclass(frozen=True)
class EpisodeBenchmark:
    seed: int
    completed: bool
    success: bool
    steps: int
    return_value: float
    elapsed_s: float | None
    path_length_m: float | None
    recoveries: float | None
    safety_violations: int
    policy_errors: int
    masked_action_violations: int
    inference_ms: tuple[float, ...]
    avoidable_failures: int = 0
    unavoidable_failures: int = 0
    oracle_misses: int = 0


@dataclass(frozen=True)
class MetricImprovement:
    metric: str
    pair_count: int
    heuristic_mean: float
    rl_mean: float
    relative_improvement: float
    bootstrap_lower_95: float
    improved: bool


@dataclass(frozen=True)
class BenchmarkReport:
    seeds: tuple[int, ...]
    heuristic: tuple[EpisodeBenchmark, ...]
    rl: tuple[EpisodeBenchmark, ...]
    improvements: tuple[MetricImprovement, ...]
    rl_inference_p95_ms: float
    model_sha256: str | None
    limits: Mapping[str, float | int]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "seeds": list(self.seeds),
            "heuristic": [asdict(item) for item in self.heuristic],
            "rl": [asdict(item) for item in self.rl],
            "improvements": [asdict(item) for item in self.improvements],
            "rl_inference_p95_ms": self.rl_inference_p95_ms,
            "model_sha256": self.model_sha256,
            "limits": dict(self.limits),
            "failures": list(self.failures),
        }


class ObservationHeuristicPolicy:
    """Recover the deterministic utility argmax from a fixed observation."""

    def predict(self, observation, *, action_masks, deterministic=True):
        del deterministic
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        mask = np.asarray(action_masks, dtype=np.bool_).reshape(-1)
        width = len(CANDIDATE_FEATURE_NAMES)
        offset = len(GLOBAL_FEATURE_NAMES)
        utility_index = CANDIDATE_FEATURE_NAMES.index("utility")
        if values.size != offset + mask.size * width:
            raise ValueError("observation shape does not match action mask")
        allowed = np.flatnonzero(mask)
        if allowed.size == 0:
            raise ValueError("heuristic benchmark received an empty action mask")
        return int(
            max(
                allowed,
                key=lambda index: (
                    values[offset + int(index) * width + utility_index],
                    -int(index),
                ),
            )
        )


def _run_policy(
    env: Any,
    policy: Any,
    seeds: Sequence[int],
    *,
    max_steps_per_episode: int,
) -> tuple[EpisodeBenchmark, ...]:
    results: list[EpisodeBenchmark] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        total_return = 0.0
        elapsed = 0.0
        path_length = 0.0
        elapsed_observed = False
        path_observed = False
        recoveries: float | None = None
        safety_violations = 0
        policy_errors = 0
        masked_violations = 0
        avoidable_failures = 0
        unavoidable_failures = 0
        oracle_misses = 0
        inference_samples: list[float] = []
        completed = False
        success = False
        steps = 0
        for steps in range(1, max_steps_per_episode + 1):
            mask = np.asarray(env.action_masks(), dtype=np.bool_)
            started = time.perf_counter()
            try:
                prediction = policy.predict(
                    observation, action_masks=mask, deterministic=True
                )
                wall_ms = (time.perf_counter() - started) * 1000.0
                reported_ms = getattr(prediction, "inference_ms", wall_ms)
                inference_ms = max(wall_ms, float(reported_ms))
                if not math.isfinite(inference_ms) or inference_ms < 0.0:
                    raise ValueError("policy inference time is invalid")
                action = coerce_discrete_action(prediction)
            except Exception:
                policy_errors += 1
                break
            inference_samples.append(inference_ms)
            if not 0 <= action < mask.size or not bool(mask[action]):
                masked_violations += 1
                break
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if "transition_elapsed_s" in info:
                elapsed += float(info["transition_elapsed_s"])
                elapsed_observed = True
            if "transition_path_length_m" in info:
                path_length += float(info["transition_path_length_m"])
                path_observed = True
            if "recovery_count" in info:
                recoveries = float(info["recovery_count"])
            elif "recovery_increment" in info:
                recoveries = (recoveries or 0.0) + float(info["recovery_increment"])
            safety_violations += int(bool(info.get("safety_violation", False)))
            avoidable_failures += int(bool(info.get("avoidable_failure", False)))
            unavoidable_failures += int(
                info.get("failure_reason") == "unavoidable_potential_outcome"
            )
            oracle_misses += int(bool(info.get("oracle_miss", False)))
            if terminated or truncated:
                completed = True
                success = bool(info.get("success", False))
                break
        results.append(
            EpisodeBenchmark(
                seed=int(seed),
                completed=completed,
                success=success,
                steps=steps,
                return_value=total_return,
                elapsed_s=elapsed if elapsed_observed else None,
                path_length_m=path_length if path_observed else None,
                recoveries=recoveries,
                safety_violations=safety_violations,
                policy_errors=policy_errors,
                masked_action_violations=masked_violations,
                inference_ms=tuple(inference_samples),
                avoidable_failures=avoidable_failures,
                unavoidable_failures=unavoidable_failures,
                oracle_misses=oracle_misses,
            )
        )
    return tuple(results)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _metric_improvement(
    name: str,
    heuristic: Sequence[EpisodeBenchmark],
    rl: Sequence[EpisodeBenchmark],
    *,
    minimum_relative_improvement: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> MetricImprovement:
    pairs = [
        (float(left), float(right))
        for h_episode, r_episode in zip(heuristic, rl)
        if (left := getattr(h_episode, name)) is not None
        and (right := getattr(r_episode, name)) is not None
    ]
    if not pairs:
        return MetricImprovement(name, 0, 0.0, 0.0, 0.0, 0.0, False)
    heuristic_mean = sum(left for left, _ in pairs) / len(pairs)
    rl_mean = sum(right for _, right in pairs) / len(pairs)
    differences = [left - right for left, right in pairs]
    relative = (heuristic_mean - rl_mean) / max(abs(heuristic_mean), 1.0e-9)
    rng = random.Random(bootstrap_seed)
    bootstrapped = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(bootstrap_samples)
    )
    lower = bootstrapped[max(0, math.floor(0.025 * len(bootstrapped)))]
    return MetricImprovement(
        metric=name,
        pair_count=len(pairs),
        heuristic_mean=heuristic_mean,
        rl_mean=rl_mean,
        relative_improvement=relative,
        bootstrap_lower_95=lower,
        improved=(
            relative >= minimum_relative_improvement and lower > 0.0
        ),
    )


def benchmark_policies(
    heuristic_env: Any,
    rl_env: Any,
    rl_policy: Any,
    *,
    seeds: Sequence[int],
    max_steps_per_episode: int = 500,
    max_inference_p95_ms: float = 25.0,
    minimum_relative_improvement: float = 0.02,
    bootstrap_samples: int = 2000,
    expected_model_sha256: str | None = None,
) -> BenchmarkReport:
    """Run paired blind seeds and enforce the PR-11 pre-release criteria."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    if max_steps_per_episode <= 0 or bootstrap_samples <= 0:
        raise ValueError("step and bootstrap counts must be positive")
    if not math.isfinite(max_inference_p95_ms) or max_inference_p95_ms <= 0.0:
        raise ValueError("max_inference_p95_ms must be finite and positive")
    if (
        not math.isfinite(minimum_relative_improvement)
        or minimum_relative_improvement < 0.0
    ):
        raise ValueError("minimum_relative_improvement must be non-negative")

    heuristic = _run_policy(
        heuristic_env,
        ObservationHeuristicPolicy(),
        seed_values,
        max_steps_per_episode=max_steps_per_episode,
    )
    rl = _run_policy(
        rl_env,
        rl_policy,
        seed_values,
        max_steps_per_episode=max_steps_per_episode,
    )
    failures: list[str] = []
    model_sha256 = getattr(rl_policy, "model_sha256", None)
    if model_sha256 is not None:
        model_sha256 = str(model_sha256).lower()
    if expected_model_sha256 is not None and model_sha256 != str(
        expected_model_sha256
    ).lower():
        failures.append("benchmark policy model SHA256 is not the approved model")
    for name, episodes in (("heuristic", heuristic), ("rl", rl)):
        if sum(item.completed for item in episodes) != len(seed_values):
            failures.append(f"{name} has incomplete episodes")
        if sum(item.policy_errors for item in episodes):
            failures.append(f"{name} has policy errors")
        if sum(item.masked_action_violations for item in episodes):
            failures.append(f"{name} selected a masked action")
        if sum(item.safety_violations for item in episodes):
            failures.append(f"{name} has safety violations")
    heuristic_successes = sum(item.success for item in heuristic)
    rl_successes = sum(item.success for item in rl)
    if rl_successes < heuristic_successes:
        failures.append(
            f"rl successes {rl_successes} below heuristic {heuristic_successes}"
        )
    rl_inference = [value for item in rl for value in item.inference_ms]
    rl_p95 = _percentile(rl_inference, 0.95)
    if rl_p95 > max_inference_p95_ms:
        failures.append(
            f"rl inference p95 {rl_p95:.6f} ms exceeds {max_inference_p95_ms:.6f} ms"
        )
    improvements = tuple(
        _metric_improvement(
            name,
            heuristic,
            rl,
            minimum_relative_improvement=minimum_relative_improvement,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=100 + index,
        )
        for index, name in enumerate(("elapsed_s", "path_length_m", "recoveries"))
    )
    if not any(item.improved for item in improvements):
        failures.append("no paired metric has a significant bounded improvement")
    return BenchmarkReport(
        seeds=seed_values,
        heuristic=heuristic,
        rl=rl,
        improvements=improvements,
        rl_inference_p95_ms=rl_p95,
        model_sha256=model_sha256,
        limits={
            "max_steps_per_episode": max_steps_per_episode,
            "max_inference_p95_ms": max_inference_p95_ms,
            "minimum_relative_improvement": minimum_relative_improvement,
            "bootstrap_samples": bootstrap_samples,
        },
        failures=tuple(failures),
    )


__all__ = [
    "BenchmarkReport",
    "EpisodeBenchmark",
    "MetricImprovement",
    "ObservationHeuristicPolicy",
    "benchmark_policies",
]
