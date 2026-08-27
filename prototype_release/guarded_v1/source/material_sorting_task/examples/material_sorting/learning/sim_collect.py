"""Offline Heuristic simulation collector for RL-2 coverage.

Workers never start ROS or the official Server. Each shard uses an isolated
seed range and output directory so a failed worker can be rerun alone.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback
from typing import Any, Mapping
import uuid

from scheduler.events import EventLog, JsonlEventSink

from .event_replay import replay_event_logs
from .observation import OBSERVATION_SCHEMA_VERSION, ObservationBuilder
from .simulation_backend import (
    DEFAULT_PROJECT_SIMULATION_CONFIG_PATH,
    ProjectSchedulingSimulationBackend,
    load_project_simulation_config,
)


def _scene_tags(sample: Any, payload_code: int, clearance: float) -> dict[str, bool]:
    latency = float(getattr(sample, "message_latency_s", 0.0) or 0.0)
    return {
        "carrying": int(payload_code) >= 2,
        "dynamic_obstacle": bool(getattr(sample, "dynamic_obstacle_present", False)),
        "planner_failure": bool(getattr(sample, "planner_failure", False)),
        "perception_dropout": bool(getattr(sample, "detection_dropout", False)),
        "high_latency": latency >= 0.08,
        "low_clearance": float(clearance) <= 0.25,
    }


def collect_simulation_shard(
    *,
    output_dir: str | Path,
    seed_start: int,
    episodes: int,
    config_path: str | Path | None = None,
    worker_id: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / f"scheduler_sim_w{worker_id:02d}.jsonl"
    replay_path = out / f"replay_sim_w{worker_id:02d}.jsonl"
    if events_path.exists():
        events_path.unlink()
    config = load_project_simulation_config(
        config_path or DEFAULT_PROJECT_SIMULATION_CONFIG_PATH
    )
    backend = ProjectSchedulingSimulationBackend(config)
    builder = ObservationBuilder(config.max_candidates)
    seeds = list(range(int(seed_start), int(seed_start) + int(episodes)))
    outcome_labels_by_decision: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        snapshot = backend.reset(seed=seed)
        log = EventLog(
            [JsonlEventSink(events_path)],
            clock=lambda: 0.0,
            session_id=f"project-sim-{seed}",
        )
        log.emit(
            "scheduler_started",
            "project simulation shard",
            details={
                "engine": "v2",
                "policy_mode": "heuristic",
                "dataset_source": "project-sim",
                "seed": seed,
                "worker_id": worker_id,
            },
        )
        while True:
            stage = config.stages[backend._stage_index]
            sample = backend._samples[backend._stage_index]
            evaluations = snapshot.candidates
            mask = tuple(item.valid for item in evaluations) + (False,) * (
                config.max_candidates - len(evaluations)
            )
            public_state = dict(snapshot.public_state)
            clearance = min(
                (
                    float(item.path_metrics.min_clearance_m)
                    for item in evaluations
                    if item.path_metrics is not None
                ),
                default=1.0,
            )
            tags = _scene_tags(sample, stage.payload_code, clearance)
            public_state["scene_tags"] = tags
            observation = builder.build(public_state, evaluations, mask)
            valid = [item for item in evaluations if item.valid]
            if not valid:
                break
            selected = max(valid, key=lambda item: (item.utility, item.action_id))
            decision_id = uuid.uuid4().hex
            outcome_probabilities = list(
                backend.counterfactual_outcome_probabilities()
            )
            potential_successes = list(
                backend.counterfactual_potential_successes()
            )
            navigable = [
                (index, probability)
                for index, (item, probability) in enumerate(
                    zip(evaluations, outcome_probabilities)
                )
                if item.valid
                and item.candidate.action_type == "navigate"
                and probability is not None
            ]
            oracle_index = max(
                navigable,
                key=lambda pair: (float(pair[1]), evaluations[pair[0]].utility),
                default=(None, None),
            )[0]
            outcome_labels_by_decision[decision_id] = {
                "candidate_outcome_probabilities": outcome_probabilities,
                "candidate_potential_successes": potential_successes,
                "oracle_action_index": oracle_index,
                "simulation_schema_version": config.schema_version,
                "outcome_model": config.outcome_model,
            }
            correlation = {
                "task_id": stage.task_id,
                "attempt": 1,
                "step_id": stage.step_id,
                "task_run_id": f"sim-{seed}-task{stage.task_id}",
                "attempt_run_id": f"sim-{seed}-attempt",
                "step_run_id": (
                    f"sim-{seed}-step-{backend._stage_index}-"
                    f"{backend._replan_counts[backend._stage_index]}"
                ),
                "decision_id": decision_id,
            }
            log.emit(
                "candidates_evaluated",
                timestamp_s=float(seed) + backend._stage_index,
                **correlation,
                details={
                    "dataset_source": "project-sim",
                    "scene_tags": tags,
                    "costmap_version": None,
                    "max_candidates": config.max_candidates,
                    "observation": observation.tolist(),
                    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                    "observation_schema_hash": builder.schema_hash,
                    "action_mask": list(mask),
                    "candidates": [
                        {
                            "action_id": item.action_id,
                            "valid": item.valid,
                            "utility": item.utility if item.valid else None,
                            "critics": dict(item.critic_scores),
                            "rejections": list(item.rejection_reasons),
                        }
                        for item in evaluations
                    ],
                },
            )
            log.emit(
                "action_selected",
                timestamp_s=float(seed) + backend._stage_index,
                action_id=selected.action_id,
                message="deterministic_best",
                **correlation,
                details={
                    "source": "heuristic",
                    "dataset_source": "project-sim",
                    "scene_tags": tags,
                    "switched": False,
                    "policy_suggestion": None,
                },
            )
            transition = backend.step(selected.candidate)
            if transition.terminated:
                break
            snapshot = transition.snapshot
    summary, records = replay_event_logs(
        [events_path], min_decisions=0, require_training_ready=False
    )
    scene_tags_by_decision: dict[str, dict[str, bool]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        details = event.get("details")
        decision_id = str(event.get("decision_id", ""))
        if (
            decision_id
            and isinstance(details, Mapping)
            and isinstance(details.get("scene_tags"), Mapping)
        ):
            scene_tags_by_decision[decision_id] = {
                str(name): bool(value)
                for name, value in details["scene_tags"].items()
            }
    annotated = []
    for record in records:
        payload = record.to_json_dict()
        payload["dataset_source"] = "project-sim"
        tags = scene_tags_by_decision.get(str(payload.get("decision_id", "")))
        if tags is not None:
            payload["scene_tags"] = tags
        labels = outcome_labels_by_decision.get(
            str(payload.get("decision_id", ""))
        )
        if labels is not None:
            payload.update(labels)
        annotated.append(payload)
    replay_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in annotated
        ),
        encoding="utf-8",
    )
    status = {
        "passed": bool(summary.passed and summary.invalid_selections == 0),
        "worker_id": worker_id,
        "seed_start": int(seed_start),
        "episodes": int(episodes),
        "seeds": seeds,
        "events_path": str(events_path),
        "replay_path": str(replay_path),
        "training_ready_decisions": int(summary.training_ready_decisions),
        "outcome_model": config.outcome_model,
        "outcome_labeled_decisions": sum(
            "candidate_outcome_probabilities" in item for item in annotated
        ),
        "malformed_count": int(summary.malformed_lines),
        "invalid_action_count": int(summary.invalid_selections),
        "unpaired_count": int(
            summary.unpaired_candidate_events + summary.unpaired_selection_events
        ),
        "failures": list(summary.failures),
    }
    (out / "worker_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def _worker_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return collect_simulation_shard(
            output_dir=payload["output_dir"],
            seed_start=int(payload["seed_start"]),
            episodes=int(payload["episodes"]),
            config_path=payload.get("config_path"),
            worker_id=int(payload["worker_id"]),
        )
    except Exception as exc:
        return {
            "passed": False,
            "worker_id": int(payload.get("worker_id", -1)),
            "failures": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
        }


def generate_simulation_matrix(
    *,
    output_root: str | Path,
    workers: int = 8,
    episodes_per_worker: int = 334,
    seed_start: int = 41000,
    profile_config: str | Path | None = None,
) -> dict[str, Any]:
    if workers <= 0 or episodes_per_worker <= 0:
        raise ValueError("workers and episodes_per_worker must be positive")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for worker_id in range(int(workers)):
        jobs.append(
            {
                "output_dir": str(root / f"worker_{worker_id:02d}"),
                "seed_start": int(seed_start) + worker_id * int(episodes_per_worker),
                "episodes": int(episodes_per_worker),
                "config_path": None if profile_config is None else str(profile_config),
                "worker_id": worker_id,
            }
        )
    results: list[dict[str, Any]] = [{} for _ in jobs]
    if workers == 1:
        results[0] = _worker_entry(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(_worker_entry, job): job["worker_id"] for job in jobs}
            for future in as_completed(futures):
                worker_id = futures[future]
                results[worker_id] = future.result()
    seeds = [seed for item in results for seed in item.get("seeds", [])]
    duplicate_seed_count = len(seeds) - len(set(seeds))
    training_ready = sum(int(item.get("training_ready_decisions", 0)) for item in results)
    malformed = sum(int(item.get("malformed_count", 0)) for item in results)
    invalid = sum(int(item.get("invalid_action_count", 0)) for item in results)
    report = {
        "passed": all(bool(item.get("passed")) for item in results) and duplicate_seed_count == 0,
        "worker_count": len(results),
        "duplicate_seed_count": duplicate_seed_count,
        "malformed_count": malformed,
        "invalid_action_count": invalid,
        "training_ready_decisions": training_ready,
        "workers": results,
    }
    (root / "generate_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_simulation_matrix(
    input_root: str | Path,
    *,
    expected_workers: int = 8,
    minimum_decisions: int = 24000,
) -> dict[str, Any]:
    root = Path(input_root)
    worker_dirs = sorted(path for path in root.glob("worker_*") if path.is_dir())
    statuses = []
    seeds: list[int] = []
    training_ready = 0
    malformed = 0
    invalid = 0
    outcome_labeled = 0
    contextual_decisions = 0
    failures: list[str] = []
    for directory in worker_dirs:
        status_path = directory / "worker_status.json"
        if not status_path.is_file():
            failures.append(f"missing {status_path}")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        statuses.append(status)
        seeds.extend(int(seed) for seed in status.get("seeds", []))
        training_ready += int(status.get("training_ready_decisions", 0))
        malformed += int(status.get("malformed_count", 0))
        invalid += int(status.get("invalid_action_count", 0))
        outcome_labeled += int(status.get("outcome_labeled_decisions", 0))
        if status.get("outcome_model") == "contextual_latent":
            contextual_decisions += int(status.get("training_ready_decisions", 0))
        if not status.get("passed"):
            failures.extend(status.get("failures") or [f"worker {status.get('worker_id')} failed"])
    duplicate_seed_count = len(seeds) - len(set(seeds))
    if len(worker_dirs) != int(expected_workers):
        failures.append(
            f"worker_count={len(worker_dirs)} expected={expected_workers}"
        )
    if duplicate_seed_count:
        failures.append(f"duplicate_seed_count={duplicate_seed_count}")
    if malformed:
        failures.append(f"malformed_count={malformed}")
    if invalid:
        failures.append(f"invalid_action_count={invalid}")
    if training_ready < int(minimum_decisions):
        failures.append(
            f"training_ready_decisions={training_ready} below {minimum_decisions}"
        )
    if contextual_decisions and outcome_labeled < contextual_decisions:
        failures.append(
            f"outcome_labeled_decisions={outcome_labeled} below "
            f"contextual_decisions={contextual_decisions}"
        )
    report = {
        "passed": not failures,
        "worker_count": len(worker_dirs),
        "duplicate_seed_count": duplicate_seed_count,
        "malformed_count": malformed,
        "invalid_action_count": invalid,
        "training_ready_decisions": training_ready,
        "outcome_labeled_decisions": outcome_labeled,
        "contextual_decisions": contextual_decisions,
        "failures": failures,
        "workers": statuses,
    }
    return report


def simulation_status(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    generate = root / "generate_status.json"
    if generate.is_file():
        return json.loads(generate.read_text(encoding="utf-8"))
    return validate_simulation_matrix(root, expected_workers=len(list(root.glob("worker_*"))))


__all__ = [
    "collect_simulation_shard",
    "generate_simulation_matrix",
    "simulation_status",
    "validate_simulation_matrix",
]
