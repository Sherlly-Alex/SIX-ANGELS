#!/usr/bin/env python3
"""CLI for the RL-2 success-first pipeline. Invoked by rl2ctl.sh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
PROJECT = SCRIPT_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

def _dump(path: Path | None, payload: object) -> int:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if isinstance(payload, dict) and payload.get("passed") is False:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RL-2 success-first controller")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-sim")
    generate.add_argument("--output-root", required=True, type=Path)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--episodes-per-worker", type=int, default=334)
    generate.add_argument("--seed-start", type=int, default=41000)
    generate.add_argument("--profile-config", type=Path)

    status = sub.add_parser("status")
    status.add_argument("--output-root", required=True, type=Path)

    validate_sim = sub.add_parser("validate-sim")
    validate_sim.add_argument("--input-root", required=True, type=Path)
    validate_sim.add_argument("--expected-workers", type=int, default=8)
    validate_sim.add_argument("--minimum-decisions", type=int, default=24000)
    validate_sim.add_argument("--output", type=Path)

    collect = sub.add_parser("collect-official")
    collect.add_argument("--output-root", required=True, type=Path)
    collect.add_argument("--mode", choices=("shadow", "guarded", "heuristic"), default="shadow")
    collect.add_argument("--seeds", nargs="+", type=int, required=True)
    collect.add_argument("--model", type=Path)
    collect.add_argument("--model-sha256")
    collect.add_argument("--approval", type=Path)
    collect.add_argument("--approval-sha256")
    collect.add_argument("--dry-run", action="store_true")

    dataset = sub.add_parser("build-dataset")
    dataset.add_argument("--simulation-root", required=True, type=Path)
    dataset.add_argument("--official-root", required=True, type=Path)
    dataset.add_argument("--output", required=True, type=Path)
    dataset.add_argument("--manifest", required=True, type=Path)
    dataset.add_argument("--coverage-json", required=True, type=Path)
    dataset.add_argument("--coverage-csv", required=True, type=Path)
    dataset.add_argument("--failures", required=True, type=Path)
    dataset.add_argument("--minimum-total", type=int, default=30000)
    dataset.add_argument("--minimum-simulation", type=int, default=24000)
    dataset.add_argument("--minimum-official", type=int, default=6000)

    audit = sub.add_parser("audit-coverage")
    audit.add_argument("inputs", nargs="+", type=Path)
    audit.add_argument("--coverage-json", required=True, type=Path)
    audit.add_argument("--coverage-csv", required=True, type=Path)
    audit.add_argument("--failures", required=True, type=Path)
    audit.add_argument("--manifest", type=Path)

    identifiability = sub.add_parser("audit-identifiability")
    identifiability.add_argument("--simulation-config", required=True, type=Path)
    identifiability.add_argument("--seed-start", type=int, default=80000)
    identifiability.add_argument("--episodes", type=int, default=500)
    identifiability.add_argument(
        "--minimum-oracle-success-gain", type=float, default=0.10
    )
    identifiability.add_argument("--output", required=True, type=Path)

    train = sub.add_parser("train-matrix")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--training-only-dataset", type=Path)
    train.add_argument("--dataset-manifest", required=True, type=Path)
    train.add_argument("--output-root", required=True, type=Path)
    train.add_argument("--workers", type=int, default=6)
    train.add_argument("--timesteps", type=int, default=150000)
    train.add_argument("--gamma", type=float, default=0.0)
    train.add_argument("--train-sessions", type=int, default=3)
    train.add_argument("--validation-sessions", type=int, default=1)
    train.add_argument("--test-sessions", type=int, default=1)
    train.add_argument("--seeds", nargs="+", type=int, required=True)
    train.add_argument("--reward-configs", nargs="+", default=["baseline", "success_time"])
    train.add_argument("--code-revision", default="rl2")

    validate_train = sub.add_parser("validate-training")
    validate_train.add_argument("--input-root", required=True, type=Path)
    validate_train.add_argument("--expected-models", type=int, default=6)
    validate_train.add_argument("--output", type=Path)

    bench = sub.add_parser("benchmark-matrix")
    bench.add_argument("--models-root", required=True, type=Path)
    bench.add_argument("--baseline-model", required=True, type=Path)
    bench.add_argument("--seed-start", type=int, default=50000)
    bench.add_argument("--episodes", type=int, default=500)
    bench.add_argument("--workers", type=int, default=6)
    bench.add_argument("--max-inference-p95-ms", type=float, default=25.0)
    bench.add_argument("--simulation-config", type=Path)
    bench.add_argument("--output-root", required=True, type=Path)

    select = sub.add_parser("select-candidate")
    select.add_argument("--benchmark-root", required=True, type=Path)
    select.add_argument("--success-first", action="store_true")
    select.add_argument("--require-no-success-regression", action="store_true")
    select.add_argument("--minimum-success-improvement", type=float, default=0.02)
    select.add_argument("--minimum-elapsed-improvement", type=float, default=0.05)
    select.add_argument("--maximum-return-regression", type=float, default=0.02)
    select.add_argument("--maximum-path-regression", type=float, default=0.02)
    select.add_argument("--maximum-recovery-regression", type=float, default=0.02)
    select.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "generate-sim":
        from learning.sim_collect import generate_simulation_matrix

        return _dump(
            args.output_root / "generate_status.json",
            generate_simulation_matrix(
                output_root=args.output_root,
                workers=args.workers,
                episodes_per_worker=args.episodes_per_worker,
                seed_start=args.seed_start,
                profile_config=args.profile_config,
            ),
        )
    if args.command == "status":
        from learning.sim_collect import simulation_status

        payload = simulation_status(args.output_root)
        train_status = args.output_root / "training_matrix_acceptance.json"
        if train_status.is_file():
            payload = json.loads(train_status.read_text(encoding="utf-8"))
        collect_status = args.output_root / "collect_status.json"
        if collect_status.is_file():
            payload = json.loads(collect_status.read_text(encoding="utf-8"))
        return _dump(None, payload)
    if args.command == "validate-sim":
        from learning.sim_collect import validate_simulation_matrix

        report = validate_simulation_matrix(
            args.input_root,
            expected_workers=args.expected_workers,
            minimum_decisions=args.minimum_decisions,
        )
        return _dump(args.output, report)
    if args.command == "collect-official":
        from rl2_official_runner import collect_official_matrix, official_docker_runner

        if args.dry_run:
            runner = lambda seed: {"passed": True, "seed": seed, "dry_run": True}
        else:
            runner = official_docker_runner(
                project=PROJECT,
                output_root=args.output_root,
                mode=args.mode,
                model=args.model,
                model_sha256=args.model_sha256,
                approval=args.approval,
                approval_sha256=args.approval_sha256,
            )
        report = collect_official_matrix(args.seeds, runner=runner, fail_fast=True)
        return _dump(args.output_root / "collect_status.json", report)
    if args.command == "build-dataset":
        from learning.rl2_pipeline import build_rl2_dataset

        report = build_rl2_dataset(
            simulation_root=args.simulation_root,
            official_root=args.official_root,
            output=args.output,
            manifest=args.manifest,
            coverage_json=args.coverage_json,
            coverage_csv=args.coverage_csv,
            failures=args.failures,
            minimum_total=args.minimum_total,
            minimum_simulation=args.minimum_simulation,
            minimum_official=args.minimum_official,
        )
        return 0 if report.get("passed") else 1
    if args.command == "audit-coverage":
        from learning.coverage_audit import (
            CoverageThresholds,
            audit_coverage,
            write_coverage_artifacts,
        )

        report = audit_coverage(args.inputs, thresholds=CoverageThresholds())
        write_coverage_artifacts(
            report,
            json_path=args.coverage_json,
            csv_path=args.coverage_csv,
            failures_path=args.failures,
            manifest_path=args.manifest,
        )
        return 0 if report.passed else 1
    if args.command == "audit-identifiability":
        from learning.rl2_pipeline import audit_simulation_identifiability
        from learning.simulation_backend import build_project_sim_env

        if args.episodes <= 0:
            raise ValueError("episodes must be positive")
        os.environ["MATERIAL_SCHEDULER_SIM_CONFIG"] = str(
            args.simulation_config
        )
        report = audit_simulation_identifiability(
            env_factory=build_project_sim_env,
            seeds=range(args.seed_start, args.seed_start + args.episodes),
            minimum_oracle_success_gain=args.minimum_oracle_success_gain,
        )
        return _dump(args.output, report)
    if args.command == "train-matrix":
        from learning.rl2_pipeline import train_matrix

        report = train_matrix(
            dataset=args.dataset,
            dataset_manifest=args.dataset_manifest,
            training_only_dataset=args.training_only_dataset,
            output_root=args.output_root,
            workers=args.workers,
            timesteps=args.timesteps,
            gamma=args.gamma,
            train_sessions=args.train_sessions,
            validation_sessions=args.validation_sessions,
            test_sessions=args.test_sessions,
            seeds=args.seeds,
            reward_configs=args.reward_configs,
            code_revision=args.code_revision,
        )
        return 0 if report.get("passed") else 1
    if args.command == "validate-training":
        from learning.rl2_pipeline import validate_training_matrix

        report = validate_training_matrix(
            args.input_root, expected_models=args.expected_models
        )
        return _dump(args.output, report)
    if args.command == "benchmark-matrix":
        from learning.rl2_pipeline import benchmark_matrix
        from learning.simulation_backend import build_project_sim_env

        os.environ["MATERIAL_SCHEDULER_SIM_CONFIG"] = str(
            args.simulation_config
            or EXAMPLE_DIR
            / "learning"
            / "configs"
            / "project_simulation_v2.json"
        )
        models = {"rl1": args.baseline_model}
        for model in sorted(args.models_root.rglob("scheduler_policy.zip")):
            models[model.parent.parent.name if model.parent.name == "model" else model.parent.name] = model
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        reports = benchmark_matrix(
            models=models,
            env_factory=build_project_sim_env,
            seeds=seeds,
            max_inference_p95_ms=args.max_inference_p95_ms,
            workers=args.workers,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        output = args.output_root / "benchmark_matrix.json"
        return _dump(output, reports)
    if args.command == "select-candidate":
        from learning.rl2_pipeline import copy_selected_model, select_rl2_candidate

        reports_path = args.benchmark_root / "benchmark_matrix.json"
        reports = json.loads(reports_path.read_text(encoding="utf-8"))
        selection = select_rl2_candidate(
            reports,
            success_first=args.success_first,
            require_no_success_regression=args.require_no_success_regression,
            minimum_success_improvement=args.minimum_success_improvement,
            minimum_elapsed_improvement=args.minimum_elapsed_improvement,
            maximum_return_regression=args.maximum_return_regression,
            maximum_path_regression=args.maximum_path_regression,
            maximum_recovery_regression=args.maximum_recovery_regression,
        )
        if selection.get("selected_model"):
            copy_selected_model(
                selection["selected_model"], args.benchmark_root / "selected"
            )
            selection["selected_model"] = str(
                args.benchmark_root / "selected" / "scheduler_policy.zip"
            )
        output = args.output or (args.benchmark_root / "rl2_selection.json")
        return _dump(output, selection)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
