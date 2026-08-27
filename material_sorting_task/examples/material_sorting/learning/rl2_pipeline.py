"""RL-2 success-first pipeline: dataset, official collection, training and gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import random
from typing import Any, Callable, Mapping, Sequence

REWARD_CONFIGS = {
    "baseline": Path(__file__).resolve().parent / "configs" / "replay_training_v1.json",
    "success_time": Path(__file__).resolve().parent
    / "configs"
    / "replay_training_success_time.json",
    "contextual_success": Path(__file__).resolve().parent
    / "configs"
    / "replay_training_contextual_success.json",
    "contextual_costaware": Path(__file__).resolve().parent
    / "configs"
    / "replay_training_contextual_costaware.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_jsonl(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def build_rl2_dataset(
    *,
    simulation_root: str | Path,
    official_root: str | Path,
    output: str | Path,
    manifest: str | Path,
    coverage_json: str | Path,
    coverage_csv: str | Path,
    failures: str | Path,
    minimum_total: int = 30000,
    minimum_simulation: int = 24000,
    minimum_official: int = 6000,
) -> dict[str, Any]:
    from .coverage_audit import (
        CoverageThresholds,
        audit_coverage,
        write_coverage_artifacts,
    )
    from .event_replay import replay_event_logs

    sim_files = discover_jsonl(Path(simulation_root))
    official_files = discover_jsonl(Path(official_root))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path, source in (
        *((item, "project-sim") for item in sim_files),
        *((item, "shadow") for item in official_files),
    ):
        text = path.read_text(encoding="utf-8", errors="replace")
        first = next((line for line in text.splitlines() if line.strip()), "")
        kind = "unknown"
        if first:
            try:
                sample = json.loads(first)
            except json.JSONDecodeError:
                sample = {}
            if sample.get("dataset_schema_version"):
                kind = "replay"
            elif sample.get("event_type"):
                kind = "events"
        if kind == "events":
            _summary, replay_records = replay_event_logs([path], min_decisions=0)
            payloads = [item.to_json_dict() for item in replay_records]
        elif kind == "replay":
            payloads = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            continue
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            payload = dict(payload)
            payload.setdefault("dataset_source", source)
            key = (
                str(payload.get("source_sha256", "")),
                str(payload.get("decision_id", "")),
            )
            if key in seen and key != ("", ""):
                continue
            seen.add(key)
            records.append(payload)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    report = audit_coverage(
        [output_path],
        thresholds=CoverageThresholds(
            minimum_total=int(minimum_total),
            minimum_simulation=int(minimum_simulation),
            minimum_official=int(minimum_official),
        ),
    )
    write_coverage_artifacts(
        report,
        json_path=Path(coverage_json),
        csv_path=Path(coverage_csv),
        failures_path=Path(failures),
        manifest_path=Path(manifest),
        dataset_path=output_path,
        extra_manifest={
            "simulation_root": str(simulation_root),
            "official_root": str(official_root),
            "record_count": len(records),
        },
    )
    payload = report.to_json_dict()
    payload["dataset"] = str(output_path)
    payload["dataset_sha256"] = _sha256(output_path) if output_path.is_file() else None
    return payload


def collect_official_matrix(
    seeds: Sequence[int],
    *,
    runner: Callable[[int], Mapping[str, Any]],
    fail_fast: bool = True,
) -> dict[str, Any]:
    seed_values = [int(seed) for seed in seeds]
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    runs: list[dict[str, Any]] = []
    stopped_at = None
    for seed in seed_values:
        report = dict(runner(int(seed)))
        report["seed"] = int(seed)
        runs.append(report)
        if not bool(report.get("passed", False)):
            stopped_at = int(seed)
            if fail_fast:
                break
    passed = stopped_at is None and all(bool(item.get("passed")) for item in runs)
    return {
        "passed": passed,
        "stopped_at_seed": stopped_at,
        "completed_seeds": [item["seed"] for item in runs],
        "runs": runs,
        "fail_fast": bool(fail_fast),
        "promotion_allowed": False,
    }


def _wait_for_log(path: Path, pattern: str, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and pattern in path.read_text(encoding="utf-8", errors="replace"):
            return True
        time.sleep(1.0)
    return False


def official_docker_runner(
    *,
    project: Path,
    output_root: Path,
    mode: str,
    model: Path | None = None,
    model_sha256: str | None = None,
    approval: Path | None = None,
    approval_sha256: str | None = None,
    server_ready_timeout_s: float = 180.0,
    client_timeout_s: float = 1800.0,
    expected_score: int = 160,
) -> Callable[[int], dict[str, Any]]:
    """Build a fail-closed per-seed official-Server runner.

    The runner always stops containers before returning. A non-160 score is a
    failed seed, never a reason to continue the batch.
    """

    scripts = project / "material_sorting_task" / "scripts"
    ctl = scripts / "competitionctl.sh"

    def _run(seed: int) -> dict[str, Any]:
        run_name = f"v2_multiseed_{seed}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        current = output_root / "current"
        current.mkdir(parents=True, exist_ok=True)
        # Keep a stable tail path for operators: current/client.log.
        env = os.environ.copy()
        env["PROJECT"] = str(project)
        env["MATERIAL_ARTIFACT_ROOT"] = str(output_root)
        if model is not None:
            env["MATERIAL_RL_MODEL_RELATIVE_PATH"] = str(model)
            if model_sha256:
                env["MATERIAL_RL_MODEL_SHA256"] = model_sha256
        if approval is not None:
            env["MATERIAL_RL_APPROVAL_RELATIVE_PATH"] = str(approval)
            if approval_sha256:
                env["MATERIAL_RL_APPROVAL_SHA256"] = approval_sha256
        subprocess.run(
            ["bash", str(ctl), "stop"],
            check=False,
            env=env,
            cwd=str(project),
        )
        server_log = run_dir / f"server_{run_name}.log"
        client_log = run_dir / f"client_{run_name}.log"
        # Detached server: reuse competitionctl env but avoid -it.
        subprocess.run(
            ["bash", str(ctl), "server-detached", run_name, str(seed)],
            check=False,
            env=env,
            cwd=str(project),
        )
        if not _wait_for_log(server_log, "material", timeout_s=server_ready_timeout_s):
            subprocess.run(["bash", str(ctl), "stop"], check=False, env=env, cwd=str(project))
            return {
                "passed": False,
                "failures": ["server did not become ready"],
                "client_log": str(client_log),
                "server_log": str(server_log),
            }
        client_returncode: int | None = None
        timed_out = False
        try:
            client = subprocess.run(
                ["bash", str(ctl), "client", run_name, mode],
                check=False,
                env=env,
                cwd=str(project),
                timeout=client_timeout_s,
            )
            client_returncode = client.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            subprocess.run(
                ["bash", str(ctl), "stop"], check=False, env=env, cwd=str(project)
            )
        if client_log.is_file():
            shutil.copy2(client_log, current / "client.log")
        text = (
            client_log.read_text(encoding="utf-8", errors="replace")
            if client_log.is_file()
            else ""
        )
        finished = f"controller=finished" in text and f"score={expected_score}" in text
        failures = []
        if timed_out:
            failures.append(f"client timeout after {client_timeout_s:.0f}s")
        if client_returncode not in {0, None} and not finished:
            failures.append(f"client exit={client_returncode}")
        if not finished:
            failures.append(f"missing controller=finished score={expected_score}")
        for name in ("controller=blocked", "controller=safe_hold", "executor error", "unsafe collision"):
            if name in text.casefold() or name in text:
                failures.append(name)
        return {
            "passed": not failures,
            "failures": failures,
            "client_log": str(client_log),
            "server_log": str(server_log),
            "score": expected_score if finished else None,
        }

    return _run


def validate_training_matrix(
    input_root: str | Path,
    *,
    expected_models: int = 6,
    release_assets: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(input_root)
    models = sorted(root.rglob("scheduler_policy.zip"))
    hashes = []
    failures: list[str] = []
    if len(models) != int(expected_models):
        failures.append(f"expected_models={expected_models} found={len(models)}")
    release = Path(release_assets) if release_assets is not None else None
    schema_hashes: set[str] = set()
    for model in models:
        if release is not None and release.resolve() in model.resolve().parents:
            failures.append(f"model written under release assets: {model}")
        metadata = Path(f"{model}.metadata.json")
        if not metadata.is_file():
            failures.append(f"missing metadata for {model}")
            continue
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        schema_hashes.add(str(payload.get("observation_schema_hash", "")))
        digest = _sha256(model)
        hashes.append(digest)
        if str(payload.get("model_sha256", "")).lower() != digest.lower():
            failures.append(f"metadata hash mismatch for {model}")
        gamma = payload.get("training_config", {}).get("gamma")
        if gamma not in {0, 0.0}:
            failures.append(f"gamma must be 0.0 for {model}")
    if len(set(hashes)) != len(hashes):
        failures.append("duplicate model SHA256 values")
    if len(schema_hashes) > 1:
        failures.append("observation schema hash mismatch across candidates")
    return {
        "passed": not failures,
        "expected_models": int(expected_models),
        "found_models": len(models),
        "model_sha256": hashes,
        "failures": failures,
        "promotion_allowed": False,
    }


def train_matrix(
    *,
    dataset: str | Path,
    dataset_manifest: str | Path,
    training_only_dataset: str | Path | None = None,
    output_root: str | Path,
    workers: int = 6,
    timesteps: int = 150000,
    gamma: float = 0.0,
    train_sessions: int = 3,
    validation_sessions: int = 1,
    test_sessions: int = 1,
    seeds: Sequence[int],
    reward_configs: Sequence[str],
    code_revision: str,
    python_executable: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise RuntimeError("refusing to train: dataset coverage did not pass")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_rl1_pipeline.py"
    jobs = []
    for reward_name in reward_configs:
        if reward_name not in REWARD_CONFIGS:
            raise ValueError(f"unknown reward config {reward_name}")
        for seed in seeds:
            jobs.append((str(reward_name), int(seed)))
    exe = python_executable or sys.executable

    def _train(job: tuple[str, int]) -> dict[str, Any]:
        reward_name, seed = job
        out = root / f"{reward_name}_seed{seed}"
        if out.exists() and any(out.iterdir()):
            return {
                "reward": reward_name,
                "seed": seed,
                "output": str(out),
                "returncode": 2,
                "passed": False,
                "failure": "refusing to overwrite non-empty output directory",
            }
        env = os.environ.copy()
        env["MATERIAL_SCHEDULER_REPLAY_CONFIG"] = str(REWARD_CONFIGS[reward_name])
        command = [
            exe,
            str(script),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(out),
            "--timesteps",
            str(int(timesteps)),
            "--seed",
            str(seed),
            "--gamma",
            str(gamma),
            "--train-sessions",
            str(int(train_sessions)),
            "--validation-sessions",
            str(int(validation_sessions)),
            "--test-sessions",
            str(int(test_sessions)),
            "--code-revision",
            str(code_revision),
        ]
        if training_only_dataset is not None:
            command.extend(
                ["--training-only-dataset", str(training_only_dataset)]
            )
        log_dir = root / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{reward_name}_seed{seed}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return {
            "reward": reward_name,
            "seed": seed,
            "output": str(out),
            "returncode": result.returncode,
            "passed": result.returncode == 0,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_train, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    report = {
        "passed": all(item["passed"] for item in results) and len(results) == len(jobs),
        "jobs": results,
        "promotion_allowed": False,
    }
    validation = validate_training_matrix(root, expected_models=len(jobs))
    report["validate_training"] = validation
    report["passed"] = bool(report["passed"] and validation["passed"])
    _write_json(root / "training_matrix_acceptance.json", report)
    return report


def _success_rate(episodes: Sequence[Any]) -> float:
    if not episodes:
        return 0.0
    return sum(bool(item.success) for item in episodes) / len(episodes)


def _mean_metric(episodes: Sequence[Any], name: str) -> float | None:
    values = [getattr(item, name) for item in episodes if getattr(item, name) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _episode_rows(episodes: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(item.seed),
            "completed": bool(item.completed),
            "success": bool(item.success),
            "return_value": float(item.return_value),
            "elapsed_s": item.elapsed_s,
            "path_length_m": item.path_length_m,
            "recoveries": item.recoveries,
            "safety_violations": int(item.safety_violations),
            "policy_errors": int(item.policy_errors),
            "masked_action_violations": int(item.masked_action_violations),
            "avoidable_failures": int(getattr(item, "avoidable_failures", 0)),
            "unavoidable_failures": int(
                getattr(item, "unavoidable_failures", 0)
            ),
            "oracle_misses": int(getattr(item, "oracle_misses", 0)),
        }
        for item in episodes
    ]


def _bootstrap_lower(values: Sequence[float], *, seed: int, samples: int = 2000) -> float:
    if not values:
        return 0.0
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(int(samples))
    )
    return float(draws[max(0, int(0.025 * len(draws)))])


def _paired_gate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float] | None:
    left = {
        int(item["seed"]): item
        for item in reference.get("episode_results", [])
        if isinstance(item, Mapping) and "seed" in item
    }
    right = {
        int(item["seed"]): item
        for item in candidate.get("episode_results", [])
        if isinstance(item, Mapping) and "seed" in item
    }
    seeds = sorted(set(left).intersection(right))
    if not seeds:
        return None
    success_delta = [
        float(bool(right[seed].get("success")))
        - float(bool(left[seed].get("success")))
        for seed in seeds
    ]
    candidate_wins = sum(
        bool(right[seed].get("success"))
        and not bool(left[seed].get("success"))
        for seed in seeds
    )
    candidate_losses = sum(
        bool(left[seed].get("success"))
        and not bool(right[seed].get("success"))
        for seed in seeds
    )
    elapsed_delta = []
    for seed in seeds:
        before = left[seed]
        after = right[seed]
        if not bool(before.get("success")) or not bool(after.get("success")):
            continue
        base_elapsed = before.get("elapsed_s")
        candidate_elapsed = after.get("elapsed_s")
        if base_elapsed in {None, 0} or candidate_elapsed is None:
            continue
        elapsed_delta.append(
            (float(base_elapsed) - float(candidate_elapsed)) / abs(float(base_elapsed))
        )
    return {
        "paired_seed_count": float(len(seeds)),
        "candidate_win_count": float(candidate_wins),
        "candidate_loss_count": float(candidate_losses),
        "net_rescue_count": float(candidate_wins - candidate_losses),
        "success_delta_lower_95": _bootstrap_lower(success_delta, seed=8124),
        "elapsed_improvement_lower_95": _bootstrap_lower(
            elapsed_delta, seed=8125
        ),
    }


def select_rl2_candidate(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    success_first: bool = True,
    require_no_success_regression: bool = True,
    minimum_success_improvement: float = 0.02,
    minimum_elapsed_improvement: float = 0.05,
    maximum_return_regression: float = 0.02,
    maximum_path_regression: float = 0.02,
    maximum_recovery_regression: float = 0.02,
) -> dict[str, Any]:
    del success_first
    heuristic = reports.get("heuristic") or {}
    baseline = reports.get("rl1") or reports.get("baseline") or {}
    heuristic_success = float(heuristic.get("success_rate", 0.0))
    baseline_success = float(baseline.get("success_rate", heuristic_success))
    ranked: list[dict[str, Any]] = []
    for name, report in reports.items():
        if name in {"heuristic", "rl1", "baseline"}:
            continue
        failures = list(report.get("failures") or [])
        success_rate = float(report.get("success_rate", 0.0))
        if require_no_success_regression and (
            success_rate < heuristic_success or success_rate < baseline_success
        ):
            failures.append(
                f"success_rate {success_rate:.4f} below heuristic "
                f"{heuristic_success:.4f} or rl1 {baseline_success:.4f}"
            )
        success_reference = max(heuristic_success, baseline_success)
        if (
            success_reference < 1.0
            and success_rate + 1.0e-12
            < success_reference + minimum_success_improvement
        ):
            failures.append(
                f"success improvement below {minimum_success_improvement:.4f}"
            )
        if report.get("policy_errors") or report.get("masked_action_violations"):
            failures.append("mask, safety or policy error present")
        if report.get("safety_violations"):
            failures.append("safety violations present")
        if report.get("completed_episodes") != report.get("episodes"):
            failures.append("incomplete episodes present")
        elapsed = report.get("elapsed_s")
        reference_elapsed = heuristic.get("elapsed_s", baseline.get("elapsed_s"))
        if (
            elapsed is not None
            and reference_elapsed
            and reference_elapsed > 0
            and (reference_elapsed - elapsed) / reference_elapsed
            < minimum_elapsed_improvement
        ):
            failures.append("elapsed improvement below 5%")
        for metric, limit in (
            ("path_length_m", maximum_path_regression),
            ("recoveries", maximum_recovery_regression),
        ):
            candidate_value = report.get(metric)
            reference = heuristic.get(metric, baseline.get(metric))
            if candidate_value is None or reference in {None, 0}:
                continue
            if reference > 0 and (candidate_value - reference) / abs(reference) > limit:
                failures.append(f"{metric} regression exceeds {limit}")
        candidate_return = report.get("return_value")
        reference_return = heuristic.get(
            "return_value", baseline.get("return_value")
        )
        if (
            candidate_return is not None
            and reference_return not in {None, 0}
            and candidate_return
            < reference_return - abs(reference_return) * maximum_return_regression
        ):
            failures.append(
                f"return_value regression exceeds {maximum_return_regression}"
            )
        paired = _paired_gate(heuristic, report)
        paired_baseline = _paired_gate(baseline, report) if baseline else None
        if paired is not None:
            if success_reference < 1.0 and paired["success_delta_lower_95"] <= 0.0:
                failures.append("paired success bootstrap lower bound is not positive")
            if paired["elapsed_improvement_lower_95"] <= 0.0:
                failures.append("paired elapsed bootstrap lower bound is not positive")
        ranked.append(
            {
                "name": name,
                "passed": not failures,
                "failures": failures,
                "success_rate": success_rate,
                "elapsed_s": report.get("elapsed_s"),
                "paired_gate": paired,
                "paired_vs_heuristic": paired,
                "paired_vs_rl1": paired_baseline,
                "report": report,
            }
        )
    ranked.sort(
        key=lambda item: (
            not item["passed"],
            -float(item["success_rate"]),
            float(item["elapsed_s"] or 1.0e9),
        )
    )
    selected = next((item for item in ranked if item["passed"]), None)
    return {
        "selected_model": None if selected is None else selected["report"].get("model_path"),
        "selected_name": None if selected is None else selected["name"],
        "promotion_allowed": False,
        "effective_policy": "heuristic" if selected is None else "rl_shadow",
        "candidates": ranked,
        "passed": selected is not None,
    }


def summarize_episode_group(name: str, episodes: Sequence[Any], **extra: Any) -> dict[str, Any]:
    from .benchmark import _percentile

    return {
        "name": name,
        "episodes": len(episodes),
        "completed_episodes": sum(bool(item.completed) for item in episodes),
        "success_rate": _success_rate(episodes),
        "elapsed_s": _mean_metric(episodes, "elapsed_s"),
        "path_length_m": _mean_metric(episodes, "path_length_m"),
        "recoveries": _mean_metric(episodes, "recoveries"),
        "return_value": _mean_metric(episodes, "return_value"),
        "policy_errors": sum(int(item.policy_errors) for item in episodes),
        "masked_action_violations": sum(
            int(item.masked_action_violations) for item in episodes
        ),
        "safety_violations": sum(int(item.safety_violations) for item in episodes),
        "avoidable_failures": sum(
            int(getattr(item, "avoidable_failures", 0)) for item in episodes
        ),
        "unavoidable_failures": sum(
            int(getattr(item, "unavoidable_failures", 0)) for item in episodes
        ),
        "oracle_misses": sum(
            int(getattr(item, "oracle_misses", 0)) for item in episodes
        ),
        "inference_p95_ms": _percentile(
            [value for item in episodes for value in item.inference_ms], 0.95
        ),
        "episode_results": _episode_rows(episodes),
        **extra,
    }


def benchmark_matrix(
    *,
    models: Mapping[str, str | Path],
    env_factory: Callable[[], Any],
    seeds: Sequence[int],
    max_steps_per_episode: int = 500,
    max_inference_p95_ms: float = 25.0,
    workers: int = 1,
) -> dict[str, Any]:
    """Paired-seed comparison of Heuristic, RL-1 and RL-2 candidates."""

    from .benchmark import ObservationHeuristicPolicy, _run_policy
    from scheduler.policies.rl import RLPolicy

    seed_values = tuple(int(seed) for seed in seeds)
    heuristic_env = env_factory()
    heuristic = _run_policy(
        heuristic_env,
        ObservationHeuristicPolicy(),
        seed_values,
        max_steps_per_episode=max_steps_per_episode,
    )
    reports: dict[str, dict[str, Any]] = {
        "heuristic": summarize_episode_group("heuristic", heuristic)
    }
    def _benchmark_model(item: tuple[str, str | Path]) -> tuple[str, dict[str, Any]]:
        name, model_path = item
        path = Path(model_path)
        env = env_factory()
        schema_hash = getattr(
            getattr(env, "observation_builder", None), "schema_hash", None
        )
        policy = RLPolicy(
            model_path=path,
            expected_schema_hash=schema_hash,
        )
        if not policy.load():
            return name, {
                "name": name,
                "success_rate": 0.0,
                "failures": [policy.last_error or "could not load policy"],
                "policy_errors": 1,
                "masked_action_violations": 0,
                "model_path": str(path),
            }
        episodes = _run_policy(
            env,
            policy,
            seed_values,
            max_steps_per_episode=max_steps_per_episode,
        )
        summary = summarize_episode_group(name, episodes, model_path=str(path))
        summary["model_sha256"] = policy.model_sha256
        failures: list[str] = []
        if summary["inference_p95_ms"] > max_inference_p95_ms:
            failures.append(
                f"inference p95 {summary['inference_p95_ms']:.3f} ms exceeds "
                f"{max_inference_p95_ms:.3f} ms"
            )
        if summary["policy_errors"]:
            failures.append("policy_errors>0")
        if summary["masked_action_violations"]:
            failures.append("masked_action_violations>0")
        summary["failures"] = failures
        return name, summary
    model_items = list(models.items())
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(model_items)))) as pool:
        futures = [pool.submit(_benchmark_model, item) for item in model_items]
        for future in as_completed(futures):
            name, summary = future.result()
            reports[name] = summary
    return reports


def audit_simulation_identifiability(
    *,
    env_factory: Callable[[], Any],
    seeds: Sequence[int],
    minimum_oracle_success_gain: float = 0.10,
) -> dict[str, Any]:
    """Prove that public-context actions can improve over the heuristic.

    The oracle is an offline simulator audit only.  Its private labels are
    never exposed through the observation or used by formal runtime code.
    """

    import numpy as np

    from .benchmark import ObservationHeuristicPolicy, _run_policy

    class _PrivateOutcomeOracle:
        def __init__(self, env: Any) -> None:
            self.env = env

        def predict(
            self, observation: Any, *, action_masks: Any, deterministic: bool = True
        ) -> int:
            del observation, deterministic
            backend = self.env.backend
            labels = backend.counterfactual_outcome_probabilities()
            mask = np.asarray(action_masks, dtype=np.bool_)
            allowed = [int(index) for index in np.flatnonzero(mask)]
            return max(allowed, key=lambda index: (float(labels[index]), -index))

    seed_values = tuple(int(seed) for seed in seeds)
    heuristic_env = env_factory()
    oracle_env = env_factory()
    heuristic = _run_policy(
        heuristic_env,
        ObservationHeuristicPolicy(),
        seed_values,
        max_steps_per_episode=500,
    )
    oracle = _run_policy(
        oracle_env,
        _PrivateOutcomeOracle(oracle_env),
        seed_values,
        max_steps_per_episode=500,
    )
    heuristic_by_seed = {item.seed: item for item in heuristic}
    oracle_by_seed = {item.seed: item for item in oracle}
    wins = sum(
        oracle_by_seed[seed].success and not heuristic_by_seed[seed].success
        for seed in seed_values
    )
    losses = sum(
        heuristic_by_seed[seed].success and not oracle_by_seed[seed].success
        for seed in seed_values
    )
    heuristic_success = _success_rate(heuristic)
    oracle_success = _success_rate(oracle)
    failures: list[str] = []
    if oracle_success < heuristic_success + float(minimum_oracle_success_gain):
        failures.append(
            "oracle success gain below "
            f"{float(minimum_oracle_success_gain):.4f}"
        )
    if wins <= losses:
        failures.append("oracle net rescue count is not positive")
    avoidable = sum(int(item.avoidable_failures) for item in heuristic)
    if avoidable <= 0:
        failures.append("heuristic has no attributable avoidable failures")
    if any(item.safety_violations for item in heuristic + oracle):
        failures.append("safety violation present")
    return {
        "passed": not failures,
        "episodes": len(seed_values),
        "seed_start": seed_values[0] if seed_values else None,
        "seed_end": seed_values[-1] if seed_values else None,
        "heuristic_success_rate": heuristic_success,
        "oracle_success_rate": oracle_success,
        "oracle_success_gain": oracle_success - heuristic_success,
        "candidate_win_count": wins,
        "candidate_loss_count": losses,
        "net_rescue_count": wins - losses,
        "heuristic_avoidable_failures": avoidable,
        "heuristic_oracle_misses": sum(
            int(item.oracle_misses) for item in heuristic
        ),
        "private_labels_in_observation": False,
        "failures": failures,
    }


def copy_selected_model(model_path: str | Path, destination: str | Path) -> Path:
    source = Path(model_path)
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "scheduler_policy.zip"
    shutil.copy2(source, target)
    metadata = Path(f"{source}.metadata.json")
    if metadata.is_file():
        shutil.copy2(metadata, Path(f"{target}.metadata.json"))
    return target


__all__ = [
    "REWARD_CONFIGS",
    "audit_simulation_identifiability",
    "benchmark_matrix",
    "build_rl2_dataset",
    "collect_official_matrix",
    "copy_selected_model",
    "official_docker_runner",
    "select_rl2_candidate",
    "summarize_episode_group",
    "train_matrix",
    "validate_training_matrix",
]
