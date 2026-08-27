"""Safe offline replay and dataset export for scheduler EventLog JSONL.

The replay consumes only the allow-listed observation vector recorded beside
candidate evaluations.  It never reads referee scores, Server layout truth or
semantic-audit payloads into the learning dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .observation import (
    CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OBSERVATION_SCHEMA_VERSION,
    observation_schema_hash,
)


REPLAY_DATASET_SCHEMA_VERSION = "scheduler-replay-v2"


@dataclass(frozen=True)
class ReplayRecord:
    source_file: str
    source_sha256: str
    session_index: int
    session_id: str
    task_run_id: str
    attempt_run_id: str
    step_run_id: str
    decision_id: str
    evaluation_event_id: str
    selection_event_id: str
    evaluation_sequence: int
    selection_sequence: int
    timestamp_s: float
    observation_schema_version: str
    observation_schema_hash: str
    max_candidates: int
    observation: tuple[float, ...]
    action_mask: tuple[bool, ...]
    candidate_action_ids: tuple[str | None, ...]
    candidate_utilities: tuple[float | None, ...]
    selected_action_index: int
    selected_action_id: str
    selection_source: str
    selection_reason: str
    costmap_version: int | None

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset_schema_version"] = REPLAY_DATASET_SCHEMA_VERSION
        value["observation"] = list(self.observation)
        value["action_mask"] = list(self.action_mask)
        return value


@dataclass(frozen=True)
class ReplaySummary:
    files: int
    sessions: int
    candidate_events: int
    selection_events: int
    paired_decisions: int
    training_ready_decisions: int
    legacy_decisions: int
    no_safe_candidate_decisions: int
    invalid_selections: int
    malformed_lines: int
    unpaired_candidate_events: int
    unpaired_selection_events: int
    heuristic_selection_count: int
    hysteresis_selection_count: int
    rl_selection_count: int
    maximum_heuristic_regret: float
    mean_heuristic_regret: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["failures"] = list(self.failures)
        return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _event_details(event: Mapping[str, Any]) -> Mapping[str, Any]:
    details = event.get("details")
    return details if isinstance(details, Mapping) else {}


def _candidate_table(
    event: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, int]] | None:
    raw = _event_details(event).get("candidates")
    if not isinstance(raw, list):
        return None
    candidates: list[Mapping[str, Any]] = []
    indices: dict[str, int] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            return None
        action_id = value.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            return None
        if action_id in indices:
            return None
        candidates.append(value)
        indices[action_id] = index
    return candidates, indices


def _build_record(
    candidate_event: Mapping[str, Any],
    selection_event: Mapping[str, Any],
    *,
    source_file: str,
    source_sha256: str,
    session_index: int,
) -> tuple[ReplayRecord | None, str | None, float | None, bool]:
    """Return (record, structural_error, heuristic_regret, no_safe)."""

    table = _candidate_table(candidate_event)
    if table is None:
        return None, "candidate payload is malformed", None, False
    candidates, indices = table
    selected_id = selection_event.get("action_id")
    valid_candidates = [item for item in candidates if item.get("valid") is True]
    if selected_id is None:
        if valid_candidates:
            return None, "null selection while valid candidates exist", None, False
        return None, None, None, True
    if not isinstance(selected_id, str) or selected_id not in indices:
        return None, "selected action is absent from candidate slots", None, False
    selected = candidates[indices[selected_id]]
    if selected.get("valid") is not True:
        return None, "selected action is masked or invalid", None, False

    selected_utility = _finite_float(selected.get("utility"))
    valid_utilities = [
        utility
        for item in valid_candidates
        if (utility := _finite_float(item.get("utility"))) is not None
    ]
    if selected_utility is None or len(valid_utilities) != len(valid_candidates):
        return None, "valid candidate utility is non-finite", None, False
    regret = max(valid_utilities) - selected_utility
    if regret < -1.0e-9:
        return None, "selected utility exceeds replay maximum", None, False
    regret = max(0.0, regret)

    details = _event_details(candidate_event)
    observation = details.get("observation")
    action_mask = details.get("action_mask")
    max_candidates = details.get("max_candidates")
    schema_version = details.get("observation_schema_version")
    schema_hash = details.get("observation_schema_hash")
    # Old EventLogs remain audit-valid but cannot become training data.
    if any(
        value is None
        for value in (
            observation,
            action_mask,
            max_candidates,
            schema_version,
            schema_hash,
        )
    ):
        return None, None, regret, False
    correlation_names = (
        "session_id",
        "task_run_id",
        "attempt_run_id",
        "step_run_id",
        "decision_id",
    )
    candidate_correlation_values = [
        candidate_event.get(name) for name in correlation_names
    ]
    selection_correlation_values = [
        selection_event.get(name) for name in correlation_names
    ]
    if not any(candidate_correlation_values) and not any(selection_correlation_values):
        # Logs produced before scheduler-event-v2 remain audit-only.
        return None, None, regret, False
    correlation: dict[str, str] = {}
    for name in correlation_names:
        candidate_value = candidate_event.get(name)
        selection_value = selection_event.get(name)
        if (
            not isinstance(candidate_value, str)
            or not candidate_value.strip()
            or selection_value != candidate_value
        ):
            return None, f"correlation mismatch for {name}", regret, False
        correlation[name] = candidate_value
    evaluation_event_id = candidate_event.get("event_id")
    selection_event_id = selection_event.get("event_id")
    if (
        not isinstance(evaluation_event_id, str)
        or not evaluation_event_id.strip()
        or not isinstance(selection_event_id, str)
        or not selection_event_id.strip()
        or evaluation_event_id == selection_event_id
    ):
        return None, "event_id correlation is missing or duplicated", regret, False
    try:
        maximum = int(max_candidates)
    except (TypeError, ValueError):
        return None, "max_candidates is invalid", regret, False
    if maximum <= 0 or maximum < len(candidates):
        return None, "max_candidates cannot contain candidate slots", regret, False
    if schema_version != OBSERVATION_SCHEMA_VERSION:
        return None, "observation schema version mismatch", regret, False
    if schema_hash != observation_schema_hash(maximum):
        return None, "observation schema hash mismatch", regret, False
    if not isinstance(observation, list) or not isinstance(action_mask, list):
        return None, "observation or action_mask is not an array", regret, False
    expected_observation_size = len(GLOBAL_FEATURE_NAMES) + maximum * len(
        CANDIDATE_FEATURE_NAMES
    )
    values = tuple(_finite_float(value) for value in observation)
    if len(values) != expected_observation_size or any(
        value is None for value in values
    ):
        return None, "observation shape or finiteness check failed", regret, False
    if len(action_mask) != maximum or any(
        not isinstance(value, bool) for value in action_mask
    ):
        return None, "action_mask shape or type check failed", regret, False
    expected_mask = tuple(item.get("valid") is True for item in candidates) + (
        False,
    ) * (maximum - len(candidates))
    if tuple(action_mask) != expected_mask:
        return None, "action_mask disagrees with candidate validity", regret, False
    selected_index = indices[selected_id]
    if not action_mask[selected_index]:
        return None, "selected action is disabled by action_mask", regret, False

    selection_details = _event_details(selection_event)
    costmap_version = details.get("costmap_version")
    if costmap_version is not None:
        try:
            costmap_version = int(costmap_version)
        except (TypeError, ValueError):
            return None, "costmap_version is invalid", regret, False
    timestamp = _finite_float(candidate_event.get("timestamp_s"))
    if timestamp is None:
        return None, "candidate timestamp is non-finite", regret, False
    return (
        ReplayRecord(
            source_file=source_file,
            source_sha256=source_sha256,
            session_index=session_index,
            session_id=correlation["session_id"],
            task_run_id=correlation["task_run_id"],
            attempt_run_id=correlation["attempt_run_id"],
            step_run_id=correlation["step_run_id"],
            decision_id=correlation["decision_id"],
            evaluation_event_id=evaluation_event_id,
            selection_event_id=selection_event_id,
            evaluation_sequence=int(candidate_event.get("sequence", 0)),
            selection_sequence=int(selection_event.get("sequence", 0)),
            timestamp_s=timestamp,
            observation_schema_version=str(schema_version),
            observation_schema_hash=str(schema_hash),
            max_candidates=maximum,
            observation=tuple(float(value) for value in values if value is not None),
            action_mask=tuple(action_mask),
            candidate_action_ids=tuple(indices) + (None,) * (maximum - len(indices)),
            candidate_utilities=tuple(
                _finite_float(item.get("utility")) if item.get("valid") is True else None
                for item in candidates
            )
            + (None,) * (maximum - len(candidates)),
            selected_action_index=selected_index,
            selected_action_id=selected_id,
            selection_source=str(selection_details.get("source", "unknown")),
            selection_reason=str(selection_event.get("message", "")),
            costmap_version=costmap_version,
        ),
        None,
        regret,
        False,
    )


def replay_event_logs(
    paths: Sequence[str | Path],
    *,
    min_decisions: int = 1,
    require_training_ready: bool = False,
) -> tuple[ReplaySummary, tuple[ReplayRecord, ...]]:
    """Audit scheduler decisions and return safe imitation-learning records."""

    if min_decisions < 0:
        raise ValueError("min_decisions must be non-negative")
    records: list[ReplayRecord] = []
    failures: list[str] = []
    counters = {
        "sessions": 0,
        "candidate_events": 0,
        "selection_events": 0,
        "paired_decisions": 0,
        "legacy_decisions": 0,
        "no_safe_candidate_decisions": 0,
        "invalid_selections": 0,
        "malformed_lines": 0,
        "unpaired_candidate_events": 0,
        "unpaired_selection_events": 0,
    }
    sources = {"heuristic": 0, "hysteresis": 0, "rl": 0}
    regrets: list[float] = []
    for raw_path in paths:
        path = Path(raw_path)
        pending: Mapping[str, Any] | None = None
        correlated_pending: dict[str, Mapping[str, Any]] = {}
        session_index = 0
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            lines = raw_text.splitlines()
        except OSError as exc:
            failures.append(f"{path}: {type(exc).__name__}")
            continue
        source_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                counters["malformed_lines"] += 1
                continue
            if not isinstance(event, Mapping):
                counters["malformed_lines"] += 1
                continue
            event_type = event.get("event_type")
            if event_type == "scheduler_started":
                if pending is not None:
                    counters["unpaired_candidate_events"] += 1
                    pending = None
                if correlated_pending:
                    counters["unpaired_candidate_events"] += len(
                        correlated_pending
                    )
                    correlated_pending.clear()
                session_index += 1
                counters["sessions"] += 1
            elif event_type == "candidates_evaluated":
                counters["candidate_events"] += 1
                decision_id = event.get("decision_id")
                if decision_id is not None:
                    if not isinstance(decision_id, str) or not decision_id.strip():
                        counters["invalid_selections"] += 1
                        failures.append(
                            f"{path}:{line_number}: candidate decision_id is invalid"
                        )
                        continue
                    if decision_id in correlated_pending:
                        counters["unpaired_candidate_events"] += 1
                    correlated_pending[decision_id] = event
                else:
                    if pending is not None:
                        counters["unpaired_candidate_events"] += 1
                    pending = event
            elif event_type == "action_selected":
                counters["selection_events"] += 1
                decision_id = event.get("decision_id")
                if decision_id is not None:
                    if not isinstance(decision_id, str) or not decision_id.strip():
                        counters["invalid_selections"] += 1
                        failures.append(
                            f"{path}:{line_number}: selection decision_id is invalid"
                        )
                        continue
                    candidate = correlated_pending.pop(decision_id, None)
                else:
                    candidate, pending = pending, None
                if candidate is None:
                    counters["unpaired_selection_events"] += 1
                    continue
                counters["paired_decisions"] += 1
                record, error, regret, no_safe = _build_record(
                    candidate,
                    event,
                    source_file=path.name,
                    source_sha256=source_sha256,
                    session_index=max(1, session_index),
                )
                if error is not None:
                    counters["invalid_selections"] += 1
                    failures.append(f"{path}:{line_number}: {error}")
                    continue
                if no_safe:
                    counters["no_safe_candidate_decisions"] += 1
                    continue
                if regret is not None:
                    regrets.append(regret)
                selection_source = str(_event_details(event).get("source", ""))
                if selection_source in sources:
                    sources[selection_source] += 1
                if record is None:
                    counters["legacy_decisions"] += 1
                else:
                    records.append(record)
        if pending is not None:
            counters["unpaired_candidate_events"] += 1
        counters["unpaired_candidate_events"] += len(correlated_pending)

    if counters["paired_decisions"] < min_decisions:
        failures.append(
            f"paired_decisions={counters['paired_decisions']} below {min_decisions}"
        )
    for name in (
        "invalid_selections",
        "malformed_lines",
        "unpaired_candidate_events",
        "unpaired_selection_events",
    ):
        if counters[name]:
            failures.append(f"{name}={counters[name]}")
    if require_training_ready and len(records) < min_decisions:
        failures.append(
            f"training_ready_decisions={len(records)} below {min_decisions}"
        )
    summary = ReplaySummary(
        files=len(paths),
        sessions=counters["sessions"],
        candidate_events=counters["candidate_events"],
        selection_events=counters["selection_events"],
        paired_decisions=counters["paired_decisions"],
        training_ready_decisions=len(records),
        legacy_decisions=counters["legacy_decisions"],
        no_safe_candidate_decisions=counters["no_safe_candidate_decisions"],
        invalid_selections=counters["invalid_selections"],
        malformed_lines=counters["malformed_lines"],
        unpaired_candidate_events=counters["unpaired_candidate_events"],
        unpaired_selection_events=counters["unpaired_selection_events"],
        heuristic_selection_count=sources["heuristic"],
        hysteresis_selection_count=sources["hysteresis"],
        rl_selection_count=sources["rl"],
        maximum_heuristic_regret=max(regrets, default=0.0),
        mean_heuristic_regret=(sum(regrets) / len(regrets) if regrets else 0.0),
        failures=tuple(failures),
    )
    return summary, tuple(records)


def write_replay_dataset(path: str | Path, records: Iterable[ReplayRecord]) -> None:
    """Write only validated, allow-listed training records as JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record.to_json_dict(), sort_keys=True, separators=(",", ":"))
                + "\n"
            )


__all__ = [
    "REPLAY_DATASET_SCHEMA_VERSION",
    "ReplayRecord",
    "ReplaySummary",
    "replay_event_logs",
    "write_replay_dataset",
]
