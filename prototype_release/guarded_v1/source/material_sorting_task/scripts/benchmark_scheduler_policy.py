#!/usr/bin/env python3
"""Run paired-seed Heuristic versus RL scheduler benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.benchmark import benchmark_policies
from scheduler.policies.rl import RLPolicy


def _factory(spec: str):
    if ":" not in spec:
        raise ValueError("environment factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("environment factory is not callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark constrained RL scheduling")
    parser.add_argument("--env-factory", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-inference-p95-ms", type=float, default=25.0)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    metadata_path = Path(f"{args.model}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training_seed = int(metadata["training_config"]["seed"])
    blind_seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    if training_seed in blind_seeds:
        raise ValueError("blind benchmark seeds overlap the training seed")
    factory = _factory(args.env_factory)
    heuristic_env = factory()
    rl_env = factory()
    schema_hash = getattr(
        getattr(rl_env, "observation_builder", None), "schema_hash", None
    )
    policy = RLPolicy(
        model_path=args.model,
        expected_sha256=args.model_sha256,
        expected_schema_hash=schema_hash,
    )
    if not policy.load():
        raise RuntimeError(policy.last_error or "could not load policy")
    report = benchmark_policies(
        heuristic_env,
        rl_env,
        policy,
        seeds=blind_seeds,
        max_steps_per_episode=args.max_steps,
        max_inference_p95_ms=args.max_inference_p95_ms,
        minimum_relative_improvement=args.minimum_relative_improvement,
        bootstrap_samples=args.bootstrap_samples,
        expected_model_sha256=args.model_sha256,
    )
    payload = json.dumps(report.to_json_dict(), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
