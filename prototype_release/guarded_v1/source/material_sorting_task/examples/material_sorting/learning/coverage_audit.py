"""Coverage audit for RL-2 training-ready scheduler decisions.

The auditor writes four artifacts and never plots. Thresholds default to the
RL-2 success-first matrix; tests may inject smaller floors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .event_replay import (
    REPLAY_DATASET_SCHEMA_VERSION,
    replay_event_logs,
)
from .observation import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES


TASK_IDS = (1, 2, 3)
STAGE_IDS = ("navigate_to_pick", "transport", "return_to_end")
ACTION_FAMILIES = ("center", "left", "right", "replan")
SOURCE_FAMILIES = ("official", "shadow", "guarded", "project-sim")
SCENE_TAGS = (
    "carrying",
    "dynamic_obstacle",
    "planner_failure",
    "perception_dropout",
    "high_latency",
    "low_clearance",
)

ACTION_ID_RE = re.compile(
    r"task(?P<task>[123]):(?P<step>navigate_to_pick|transport|return_to_end)"
    r":(?:stand|sim|recovery):(?P<action>[a-z_]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoverageThresholds:
    minimum_total: int = 30_000
    minimum_simulation: int = 24_000
    minimum_official: int = 6_000
    minimum_task_stage: int = 2_500
    minimum_task_stage_action: int = 200
    minimum_action_family: int = 1_500
    minimum_scene_tag: int = 1_000
    maximum_malformed: int = 0
    maximum_schema_mix: int = 0
    maximum_unpaired: int = 0
    maximum_invalid_action: int = 0


@dataclass
class CoverageReport:
    passed: bool
    thresholds: dict[str, int]
    training_ready_decisions: int
    source_counts: dict[str, int]
    task_stage_counts: dict[str, int]
    task_stage_action_counts: dict[str, int]
    action_family_counts: dict[str, int]
    candidate_available_counts: dict[str, int]
    masked_counts: dict[str, int]
    selected_counts: dict[str, int]
    success_counts: dict[str, int]
    failure_counts: dict[str, int]
    fallback_counts: dict[str, int]
    scene_tag_counts: dict[str, int]
    metric_summaries: dict[str, float]
    malformed_count: int
    schema_mix_count: int
    unpaired_count: int
    invalid_action_count: int
    duplicate_count: int
    non_training_ready_count: int
    failures: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def parse_action_id(action_id: str | None) -> tuple[int | None, str | None, str | None]:
    if not action_id:
        return None, None, None
    match = ACTION_ID_RE.search(str(action_id))
    if match is None:
        return None, None, None
    family = match.group("action").casefold()
    if family in {"stand_center", "sim_center"}:
        family = "center"
    if family not in ACTION_FAMILIES and family not in {"rescan", "safe_retreat"}:
        family = family
    return int(match.group("task")), match.group("step"), family


def classify_source(path: Path, record: Mapping[str, Any]) -> str:
    explicit = str(
        record.get("dataset_source")
        or record.get("source_family")
        or ""
    ).strip().casefold()
    if explicit in SOURCE_FAMILIES:
        return explicit
    text = str(path).replace("\\", "/").casefold()
    if "project-sim" in text or "simulation" in text:
        return "project-sim"
    if "guarded" in text:
        return "guarded"
    if "shadow" in text:
        return "shadow"
    if "official" in text:
        return "official"
    return "official"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _observation_feature(
    observation: Sequence[Any], name: str, *, slot: int | None = None
) -> float | None:
    values = list(observation or ())
    if slot is None:
        if name not in GLOBAL_FEATURE_NAMES:
            return None
        index = GLOBAL_FEATURE_NAMES.index(name)
        return _finite(values[index]) if index < len(values) else None
    if name not in CANDIDATE_FEATURE_NAMES:
        return None
    offset = len(GLOBAL_FEATURE_NAMES) + int(slot) * len(CANDIDATE_FEATURE_NAMES)
    index = offset + CANDIDATE_FEATURE_NAMES.index(name)
    return _finite(values[index]) if index < len(values) else None


def _scene_tags(record: Mapping[str, Any]) -> dict[str, bool | None]:
    explicit = record.get("scene_tags")
    if isinstance(explicit, Mapping):
        return {
            name: None if explicit.get(name) is None else bool(explicit.get(name))
            for name in SCENE_TAGS
        }
    observation = record.get("observation") or ()
    payload = _observation_feature(observation, "payload_code")
    selected = int(record.get("selected_action_index") or 0)
    dynamic = _observation_feature(observation, "dynamic_risk", slot=selected)
    uncertainty = _observation_feature(
        observation, "perception_uncertainty", slot=selected
    )
    latency = _observation_feature(observation, "expected_time_s", slot=selected)
    clearance = _observation_feature(observation, "min_clearance_m", slot=selected)
    planner = record.get("planner_failure")
    return {
        "carrying": None if payload is None else payload >= 1.5,
        "dynamic_obstacle": None if dynamic is None else dynamic >= 0.4,
        "planner_failure": None if planner is None else bool(planner),
        "perception_dropout": None if uncertainty is None else uncertainty >= 0.3,
        "high_latency": None if latency is None else latency >= 8.0,
        "low_clearance": None if clearance is None else clearance <= 0.25,
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any] | None]]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError:
        yield 0, None
        return
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            yield line_number, None
            continue
        if not isinstance(value, Mapping):
            yield line_number, None
            continue
        yield line_number, value


def _load_replay_records(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    schemas: set[str] = set()
    seen: set[tuple[str, str]] = set()
    event_paths: list[Path] = []
    for path in paths:
        sample_kind = None
        for _line_number, value in _iter_jsonl(path):
            if value is None:
                stats["malformed"] += 1
                continue
            if value.get("dataset_schema_version") == REPLAY_DATASET_SCHEMA_VERSION:
                sample_kind = "replay"
                schemas.add(str(value.get("dataset_schema_version")))
                key = (
                    str(value.get("source_sha256", "")),
                    str(value.get("decision_id", "")),
                )
                if key in seen and key != ("", ""):
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
                record = dict(value)
                record["_source_path"] = str(path)
                records.append(record)
            elif value.get("event_type") or value.get("event_schema_version"):
                sample_kind = "events"
                break
            else:
                stats["malformed"] += 1
        if sample_kind == "events":
            event_paths.append(path)
    if event_paths:
        summary, replay_records = replay_event_logs(event_paths, min_decisions=0)
        stats["malformed"] += int(summary.malformed_lines)
        stats["unpaired"] += int(
            summary.unpaired_candidate_events + summary.unpaired_selection_events
        )
        stats["invalid_action"] += int(summary.invalid_selections)
        stats["non_training_ready"] += int(summary.legacy_decisions)
        if not summary.passed:
            stats["replay_failures"] += len(summary.failures)
        for item in replay_records:
            payload = item.to_json_dict()
            payload["_source_path"] = item.source_file
            key = (item.source_sha256, item.decision_id)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            records.append(payload)
    stats["schema_mix"] = max(0, len(schemas) - 1) if records else 0
    return records, stats


def audit_coverage(
    paths: Sequence[str | Path],
    *,
    thresholds: CoverageThresholds | None = None,
) -> CoverageReport:
    limits = thresholds or CoverageThresholds()
    files = [Path(path) for path in paths]
    records, stats = _load_replay_records(files)
    source_counts: Counter[str] = Counter()
    task_stage: Counter[str] = Counter()
    task_stage_action: Counter[str] = Counter()
    action_family: Counter[str] = Counter()
    available: Counter[str] = Counter()
    masked: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    fallbacks: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    available_combos: set[str] = set()
    returns: list[float] = []
    elapsed: list[float] = []
    path_lengths: list[float] = []
    recoveries: list[float] = []

    for record in records:
        source = classify_source(Path(record.get("_source_path", "")), record)
        source_counts[source] += 1
        action_id = str(record.get("selected_action_id", ""))
        task_id, stage, family = parse_action_id(action_id)
        if task_id is None:
            task_id = int(_observation_feature(record.get("observation") or (), "task_ordinal") or 0) or None
        if family in ACTION_FAMILIES:
            action_family[family] += 1
            selected[family] += 1
        if task_id in TASK_IDS and stage in STAGE_IDS:
            task_key = f"task{task_id}:{stage}"
            task_stage[task_key] += 1
            if family in ACTION_FAMILIES:
                combo = f"{task_key}:{family}"
                task_stage_action[combo] += 1
        mask = list(record.get("action_mask") or ())
        action_ids = list(record.get("candidate_action_ids") or ())
        for index, allowed in enumerate(mask):
            candidate_id = action_ids[index] if index < len(action_ids) else None
            _task, _stage, cand_family = parse_action_id(
                None if candidate_id is None else str(candidate_id)
            )
            if cand_family not in ACTION_FAMILIES:
                continue
            if allowed:
                available[cand_family] += 1
                if _task in TASK_IDS and _stage in STAGE_IDS:
                    available_combos.add(f"task{_task}:{_stage}:{cand_family}")
            else:
                masked[cand_family] += 1
        reason = str(record.get("selection_reason", "")).casefold()
        source_name = str(record.get("selection_source", "")).casefold()
        if "fallback" in reason or source_name == "heuristic" and "rl_" in reason:
            fallbacks[source] += 1
        tags = _scene_tags(record)
        for name, flag in tags.items():
            if flag is True:
                scene_counts[name] += 1
            elif flag is None:
                scene_counts[f"{name}:unknown"] += 1
        utility = None
        utilities = record.get("candidate_utilities") or []
        index = record.get("selected_action_index")
        if isinstance(index, int) and 0 <= index < len(utilities):
            utility = _finite(utilities[index])
        if utility is not None:
            returns.append(utility)
        elapsed_value = _observation_feature(
            record.get("observation") or (), "expected_time_s", slot=index if isinstance(index, int) else 0
        )
        if elapsed_value is not None:
            elapsed.append(elapsed_value)
        path_value = _observation_feature(
            record.get("observation") or (), "path_length_m", slot=index if isinstance(index, int) else 0
        )
        if path_value is not None:
            path_lengths.append(path_value)
        recovery = _observation_feature(record.get("observation") or (), "recovery_count")
        if recovery is not None:
            recoveries.append(recovery)
        success_p = _observation_feature(
            record.get("observation") or (),
            "success_probability",
            slot=index if isinstance(index, int) else 0,
        )
        if success_p is not None and success_p >= 0.5:
            successes[source] += 1
        else:
            failures[source] += 1

    gap_failures: list[str] = []
    total = len(records)
    simulation = int(source_counts.get("project-sim", 0))
    official = int(
        source_counts.get("official", 0)
        + source_counts.get("shadow", 0)
        + source_counts.get("guarded", 0)
    )
    if total < limits.minimum_total:
        gap_failures.append(
            f"training-ready total missing {limits.minimum_total - total} (have {total})"
        )
    if simulation < limits.minimum_simulation:
        gap_failures.append(
            f"project-sim missing {limits.minimum_simulation - simulation} (have {simulation})"
        )
    if official < limits.minimum_official:
        gap_failures.append(
            f"official/shadow missing {limits.minimum_official - official} (have {official})"
        )
    for task_id in TASK_IDS:
        for stage in STAGE_IDS:
            key = f"task{task_id}:{stage}"
            have = int(task_stage.get(key, 0))
            if have < limits.minimum_task_stage:
                gap_failures.append(
                    f"{key} missing {limits.minimum_task_stage - have} (have {have})"
                )
    for combo in sorted(available_combos):
        have = int(task_stage_action.get(combo, 0))
        if have < limits.minimum_task_stage_action:
            gap_failures.append(
                f"{combo} selected missing {limits.minimum_task_stage_action - have} (have {have})"
            )
    for family in ACTION_FAMILIES:
        have = int(action_family.get(family, 0))
        if have < limits.minimum_action_family:
            gap_failures.append(
                f"{family} safe selections missing {limits.minimum_action_family - have} (have {have})"
            )
    for tag in SCENE_TAGS:
        have = int(scene_counts.get(tag, 0))
        if have < limits.minimum_scene_tag:
            gap_failures.append(
                f"{tag} scenarios missing {limits.minimum_scene_tag - have} (have {have})"
            )
    malformed = int(stats.get("malformed", 0))
    schema_mix = int(stats.get("schema_mix", 0))
    unpaired = int(stats.get("unpaired", 0))
    invalid_action = int(stats.get("invalid_action", 0))
    if malformed > limits.maximum_malformed:
        gap_failures.append(f"malformed={malformed}")
    if schema_mix > limits.maximum_schema_mix:
        gap_failures.append(f"schema_mix={schema_mix}")
    if unpaired > limits.maximum_unpaired:
        gap_failures.append(f"unpaired={unpaired}")
    if invalid_action > limits.maximum_invalid_action:
        gap_failures.append(f"invalid_action={invalid_action}")

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    report = CoverageReport(
        passed=not gap_failures,
        thresholds=asdict(limits),
        training_ready_decisions=total,
        source_counts=dict(source_counts),
        task_stage_counts=dict(task_stage),
        task_stage_action_counts=dict(task_stage_action),
        action_family_counts=dict(action_family),
        candidate_available_counts=dict(available),
        masked_counts=dict(masked),
        selected_counts=dict(selected),
        success_counts=dict(successes),
        failure_counts=dict(failures),
        fallback_counts=dict(fallbacks),
        scene_tag_counts=dict(scene_counts),
        metric_summaries={
            "mean_selected_utility": _mean(returns),
            "mean_expected_time_s": _mean(elapsed),
            "mean_path_length_m": _mean(path_lengths),
            "mean_recovery_count": _mean(recoveries),
        },
        malformed_count=malformed,
        schema_mix_count=schema_mix,
        unpaired_count=unpaired,
        invalid_action_count=invalid_action,
        duplicate_count=int(stats.get("duplicate", 0)),
        non_training_ready_count=int(stats.get("non_training_ready", 0)),
        failures=gap_failures,
    )
    return report


def write_coverage_artifacts(
    report: CoverageReport,
    *,
    json_path: Path,
    csv_path: Path,
    failures_path: Path,
    manifest_path: Path | None = None,
    dataset_path: Path | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["summary", "passed", int(report.passed)])
        writer.writerow(
            ["summary", "training_ready_decisions", report.training_ready_decisions]
        )
        for section, mapping in (
            ("source", report.source_counts),
            ("task_stage", report.task_stage_counts),
            ("task_stage_action", report.task_stage_action_counts),
            ("action_family", report.action_family_counts),
            ("scene", report.scene_tag_counts),
        ):
            for key, value in sorted(mapping.items()):
                writer.writerow([section, key, value])
    failures_path.write_text(
        "\n".join(report.failures) + ("\n" if report.failures else ""),
        encoding="utf-8",
    )
    if manifest_path is not None:
        manifest = {
            "passed": report.passed,
            "training_ready_decisions": report.training_ready_decisions,
            "source_counts": report.source_counts,
            "failures": report.failures,
        }
        if dataset_path is not None and dataset_path.is_file():
            digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            manifest["dataset"] = str(dataset_path)
            manifest["dataset_sha256"] = digest
            manifest["dataset_bytes"] = dataset_path.stat().st_size
        if extra_manifest:
            manifest.update(dict(extra_manifest))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


__all__ = [
    "CoverageReport",
    "CoverageThresholds",
    "audit_coverage",
    "classify_source",
    "parse_action_id",
    "write_coverage_artifacts",
]
