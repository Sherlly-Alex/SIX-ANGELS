#!/usr/bin/env python3
"""Fail-closed RL-1 offline pipeline: split, train, package-check, held-out gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "material_sorting"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, env: dict[str, str], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run(
            command, cwd=ROOT, env=env, text=True,
            stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"step failed ({result.returncode}); see {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the optional RL-1 scheduler pipeline")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="discount factor; replay snapshots are independent contextual-bandit decisions",
    )
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args(argv)
    if (
        args.timesteps <= 0
        or not 0.0 <= args.gamma <= 1.0
        or not 0.0 <= args.gae_lambda <= 1.0
        or not args.device.strip()
        or not args.code_revision.strip()
    ):
        raise SystemExit(
            "timesteps must be positive; gamma and gae-lambda must be within "
            "[0, 1]; device and code revision must be non-empty"
        )
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    if not dataset.is_file():
        raise SystemExit(f"dataset does not exist: {dataset}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXAMPLE) + os.pathsep + env.get("PYTHONPATH", "")
    split_dir = output / "split"
    model = output / "model" / "scheduler_policy.zip"
    model.parent.mkdir()
    logs = output / "logs"
    logs.mkdir()
    try:
        _run(
            [sys.executable, str(ROOT / "scripts" / "split_replay_dataset.py"),
             "--dataset", str(dataset), "--output-dir", str(split_dir),
             "--seed", str(args.seed)],
            env=env, output=logs / "split.log"
        )
        train_env = dict(env)
        train_env["MATERIAL_SCHEDULER_REPLAY_DATASET"] = str(split_dir / "train.jsonl")
        _run([sys.executable, str(ROOT / "scripts" / "train_scheduler_policy.py"),
              "--env-factory", "learning.replay_env:build_replay_env", "--output", str(model),
              "--timesteps", str(args.timesteps), "--seed", str(args.seed), "--code-revision", args.code_revision,
              "--gamma", str(args.gamma), "--gae-lambda", str(args.gae_lambda), "--device", args.device,
              "--provenance", str(split_dir / "train.jsonl"), "--provenance", str(split_dir / "split_manifest.json")],
             env=train_env, output=logs / "train.log")
        _run([sys.executable, str(ROOT / "scripts" / "validate_scheduler_model.py"), "--model", str(model),
              "--output", str(output / "model_package_acceptance.json")], env=env, output=logs / "package.log")
        _run([sys.executable, str(ROOT / "scripts" / "validate_rl1_heldout.py"), "--model", str(model),
              "--split-dir", str(split_dir), "--output", str(output / "heldout_acceptance.json"),
              "--seed", str(args.seed)], env=env, output=logs / "heldout.log")
        heldout = json.loads((output / "heldout_acceptance.json").read_text(encoding="utf-8"))
        package_report = output / "model_package_acceptance.json"
        payload = {"passed": bool(heldout.get("passed")), "failures": heldout.get("failures", []),
                   "pipeline": "rl-1", "dataset_sha256": _sha256(dataset), "code_revision": args.code_revision,
                   "seed": args.seed, "timesteps": args.timesteps,
                   "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                   "device": args.device,
                   "next_allowed_mode": "rl_shadow" if heldout.get("passed") else "heuristic",
                   "rl_guarded_authorized": False,
                   "split_manifest_sha256": _sha256(split_dir / "split_manifest.json"),
                   "model_sha256": heldout.get("model_sha256"),
                   "model_package_report": str(package_report),
                   "heldout_report": str(output / "heldout_acceptance.json"),
                   "split_manifest": str(split_dir / "split_manifest.json")}
    except Exception as exc:
        payload = {"passed": False, "failures": [f"{type(exc).__name__}: {exc}"],
                   "pipeline": "rl-1", "next_allowed_mode": "heuristic",
                   "rl_guarded_authorized": False}
    (output / "rl1_acceptance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
