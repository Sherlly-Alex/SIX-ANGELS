#!/usr/bin/env python3
"""Validate a MaskablePPO package on validation and test replay sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import re

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "material_sorting"
sys.path.insert(0, str(EXAMPLE))
from learning.heldout_gate import PolicySummary, assess_heldout_gate
from learning.model_package import validate_model_package
from learning.replay_env import ReplayBanditEnv, load_replay_dataset
from learning.observation import observation_schema_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _OraclePolicy:
    def __init__(self, env: ReplayBanditEnv) -> None:
        self.env = env

    def predict(self, observation, *, action_masks, deterministic=True):
        del observation, deterministic
        utilities = self.env._current["candidate_utilities"]
        return max((int(index) for index, allowed in enumerate(action_masks) if allowed),
                   key=lambda index: (float(utilities[index]), -index))


class _SelectedPolicy:
    def __init__(self, env: ReplayBanditEnv) -> None:
        self.env = env

    def predict(self, observation, *, action_masks, deterministic=True):
        del observation, deterministic
        return int(self.env._current["selected_action_index"])


def _run(env: ReplayBanditEnv, policy: object, episodes: int, seed: int) -> PolicySummary:
    from learning.evaluate_policy import evaluate_policy
    value = evaluate_policy(
        env, policy, episodes=episodes,
        max_steps_per_episode=env.episode_length, seed=seed
    )
    return PolicySummary(value.episodes, value.completed_episodes, value.policy_errors,
                         value.masked_action_violations, value.mean_return)


def validate_split_manifest(split_dir: Path) -> tuple[dict, dict[str, tuple[dict, ...]]]:
    """Validate declarations against all actual JSONL session records."""
    manifest_path = split_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "scheduler-replay-session-split-v1":
        raise ValueError("split manifest schema mismatch")
    files = manifest.get("files", {})
    names = ("train", "validation", "test")
    for name in names:
        try:
            raw_path = files[name]["path"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name} manifest file path is missing") from exc
        path = (split_dir / raw_path).resolve()
        if path.parent != split_dir.resolve() or path.name != f"{name}.jsonl":
            raise ValueError(f"{name} manifest file path escapes split directory")
        if _sha256(path) != files[name]["sha256"]:
            raise ValueError(f"{name} split SHA256 mismatch")
    split_data = {name: load_replay_dataset(split_dir / files[name]["path"]) for name in names}
    actual_keys = {
        name: {(str(record.get("source_sha256", "")).lower(), int(record.get("session_index", -1)))
               for record in values}
        for name, values in split_data.items()
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", source) or index < 1
           for keys in actual_keys.values() for source, index in keys):
        raise ValueError("split data contains malformed session key")
    for name in names:
        declared = {(str(item["source_sha256"]).lower(), int(item["session_index"]))
                    for item in manifest["sessions"][name]}
        if declared != actual_keys[name]:
            raise ValueError(f"{name} session manifest does not match JSONL")
        if len(split_data[name]) != int(manifest["record_counts"][name]):
            raise ValueError(f"{name} record count does not match manifest")
        if len(actual_keys[name]) != int(manifest["session_counts"][name]):
            raise ValueError(f"{name} session count does not match manifest")
    if (actual_keys["train"] & actual_keys["validation"] or
            actual_keys["train"] & actual_keys["test"] or
            actual_keys["validation"] & actual_keys["test"]):
        raise ValueError("session leakage across split data")
    return manifest, split_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    manifest_path = args.split_dir / "split_manifest.json"
    failures: list[str] = []
    try:
        manifest, split_data = validate_split_manifest(args.split_dir)
        files = manifest["files"]
        train_path = args.split_dir / files["train"]["path"]
        report = validate_model_package(args.model, expected_provenance_sha256=_sha256(train_path))
        if not report.passed:
            raise ValueError("model package: " + "; ".join(report.failures))
        from scheduler.policies.rl import RLPolicy
        maximum = int(split_data["validation"][0]["max_candidates"])
        policy = RLPolicy(model_path=args.model,
                          expected_sha256=report.model_sha256,
                          expected_schema_hash=observation_schema_hash(maximum))
        if not policy.load():
            raise RuntimeError(policy.last_error or "RLPolicy failed to load")
        validation_env = ReplayBanditEnv(split_data["validation"])
        test_env = ReplayBanditEnv(split_data["test"])
        validation_summary = _run(validation_env, policy, args.episodes, args.seed)
        test_summary = _run(test_env, policy, args.episodes, args.seed + 1000)
        validation_oracle = _run(validation_env, _OraclePolicy(validation_env), args.episodes, args.seed)
        test_oracle = _run(test_env, _OraclePolicy(test_env), args.episodes, args.seed + 1000)
        validation_baseline = _run(validation_env, _SelectedPolicy(validation_env), args.episodes, args.seed)
        test_baseline = _run(test_env, _SelectedPolicy(test_env), args.episodes, args.seed + 1000)
        gate = assess_heldout_gate(validation_summary, test_summary,
                                    validation_baseline_return=validation_baseline.mean_return,
                                    test_baseline_return=test_baseline.mean_return,
                                    validation_oracle_return=validation_oracle.mean_return,
                                    test_oracle_return=test_oracle.mean_return,
                                    tolerance=args.tolerance)
        payload = {"passed": gate["passed"], "failures": gate["failures"], "gate": gate,
                   "model_sha256": report.model_sha256, "manifest_sha256": _sha256(manifest_path)}
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        payload = {"passed": False, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
