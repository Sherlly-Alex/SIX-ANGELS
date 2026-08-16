"""Deterministic offline evaluation for a constrained scheduler policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from scheduler.policies.rl import RLPolicy


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    completed_episodes: int
    successful_episodes: int
    policy_errors: int
    masked_action_violations: int
    mean_return: float
    mean_steps: float


def evaluate_policy(
    env: Any,
    policy: Any,
    *,
    episodes: int = 20,
    max_steps_per_episode: int = 500,
    seed: int = 0,
) -> EvaluationSummary:
    """Evaluate without updating model weights or bypassing action masks."""

    if episodes <= 0 or max_steps_per_episode <= 0:
        raise ValueError("episodes and max_steps_per_episode must be positive")
    if not callable(getattr(env, "action_masks", None)):
        raise TypeError("env must expose action_masks()")
    returns: list[float] = []
    steps_taken: list[int] = []
    successful = 0
    completed = 0
    policy_errors = 0
    masked_violations = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        step_count = 0
        for step_count in range(1, max_steps_per_episode + 1):
            mask = env.action_masks()
            try:
                prediction = policy.predict(
                    observation, action_masks=mask, deterministic=True
                )
                action = int(
                    getattr(
                        prediction,
                        "action_index",
                        prediction[0] if isinstance(prediction, tuple) else prediction,
                    )
                )
            except Exception:
                policy_errors += 1
                break
            if not 0 <= action < len(mask) or not bool(mask[action]):
                masked_violations += 1
                break
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                completed += 1
                if bool(final_info.get("success", False)):
                    successful += 1
                break
        returns.append(total_reward)
        steps_taken.append(step_count)

    return EvaluationSummary(
        episodes=episodes,
        completed_episodes=completed,
        successful_episodes=successful,
        policy_errors=policy_errors,
        masked_action_violations=masked_violations,
        mean_return=sum(returns) / len(returns),
        mean_steps=sum(steps_taken) / len(steps_taken),
    )


def _load_factory(spec: str) -> Callable[[], Any]:
    if ":" not in spec:
        raise ValueError("environment factory must use 'module:function' syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"environment factory {spec!r} is not callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a masked scheduler policy")
    parser.add_argument("--env-factory", required=True, help="module:function")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    env = _load_factory(args.env_factory)()
    schema_hash = getattr(getattr(env, "observation_builder", None), "schema_hash", None)
    policy = RLPolicy(
        model_path=Path(args.model), expected_schema_hash=schema_hash
    )
    if not policy.load():
        raise RuntimeError(policy.last_error or "could not load policy")
    summary = evaluate_policy(
        env, policy, episodes=args.episodes, seed=args.seed
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = ["EvaluationSummary", "evaluate_policy"]
