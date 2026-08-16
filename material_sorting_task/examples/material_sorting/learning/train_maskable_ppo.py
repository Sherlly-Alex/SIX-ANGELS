"""Offline MaskablePPO training entry point.

This module is safe to import in the formal Client image.  Gym/SB3 packages are
resolved only when ``train_maskable_ppo`` is called; no model is downloaded and
the caller must inject a scheduling simulation environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from .observation import OBSERVATION_SCHEMA_VERSION, observation_schema_hash


class TrainingDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    total_timesteps: int
    observation_schema_hash: str


def _maskable_ppo_class() -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise TrainingDependencyError(
            "offline RL training requires optional packages 'gymnasium', "
            "'stable-baselines3' and 'sb3-contrib'. Install them only in the "
            "training environment; the formal heuristic Client does not need them."
        ) from exc
    return MaskablePPO


def train_maskable_ppo(
    env: Any,
    output_path: str | Path,
    *,
    total_timesteps: int = 10_000,
    seed: int = 0,
    learning_rate: float = 3.0e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    verbose: int = 1,
) -> TrainingResult:
    """Train against an injected env and persist the exact feature schema."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if not callable(getattr(env, "action_masks", None)):
        raise TypeError("env must expose action_masks() for hard action masking")
    max_candidates = int(getattr(getattr(env, "action_space", None), "n", 0))
    if max_candidates <= 0:
        raise TypeError("env must expose a positive discrete action_space.n")
    model_class = _maskable_ppo_class()
    model = model_class(
        "MlpPolicy",
        env,
        seed=int(seed),
        learning_rate=float(learning_rate),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        verbose=int(verbose),
    )
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False)

    requested = Path(output_path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(requested))
    model_path = requested if requested.suffix == ".zip" else Path(f"{requested}.zip")
    schema_hash = observation_schema_hash(max_candidates)
    metadata_path = Path(f"{model_path}.metadata.json")
    metadata = {
        "algorithm": "MaskablePPO",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_hash": schema_hash,
        "max_candidates": max_candidates,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "learning_rate": float(learning_rate),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        total_timesteps=int(total_timesteps),
        observation_schema_hash=schema_hash,
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
    parser = argparse.ArgumentParser(
        description="Train the optional masked scheduler policy offline"
    )
    parser.add_argument("--env-factory", required=True, help="module:function")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    env = _load_factory(args.env_factory)()
    result = train_maskable_ppo(
        env, args.output, total_timesteps=args.timesteps, seed=args.seed
    )
    print(json.dumps({key: str(value) for key, value in result.__dict__.items()}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "TrainingDependencyError",
    "TrainingResult",
    "train_maskable_ppo",
]
