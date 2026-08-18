#!/usr/bin/env python3
"""Validate one official full-task remote run from saved text artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


FINISHED_RE = re.compile(
    r"controller=finished\s+task=3\s+attempt=\d+\s+stage=-\s+score=(\d+)"
)
TASK_SCORE_RE = re.compile(
    r"controller=waiting_for_referee\s+task=(\d+).*?score=(\d+)"
)
FATAL_PATTERNS = {
    "controller_blocked": re.compile(r"controller=blocked\b"),
    "controller_safe_hold": re.compile(r"controller=safe_hold\b"),
    "executor_error": re.compile(r"executor error", re.IGNORECASE),
    "unsafe_collision": re.compile(r"unsafe collision", re.IGNORECASE),
}
MEASURED_CARRY_RE = re.compile(
    r"measured_carried_guard=active\s+"
    r"source=(task[123])\s+"
    r"half_width=([0-9]+(?:\.[0-9]+)?)m\s+"
    r"path_clearance=([0-9]+(?:\.[0-9]+)?)m\s+"
    r"minimum_clearance=([0-9]+(?:\.[0-9]+)?)m"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def validate_run(
    client_text: str,
    server_text: str,
    *,
    expected_score: int = 160,
    scheduler_events_text: str | None = None,
    max_interval_p95_ms: float = 65.0,
    max_interval_p99_ms: float = 125.0,
    max_execution_p95_ms: float = 50.0,
    max_deadline_miss_rate: float = 0.01,
    require_measured_carry: bool = False,
    require_candidate_application: bool = False,
    min_applied_candidates: int = 1,
    min_noncenter_applied: int = 0,
) -> dict[str, object]:
    """Return a stable machine-readable acceptance report."""

    failures: list[str] = []
    fatal_counts = {
        name: len(pattern.findall(client_text))
        for name, pattern in FATAL_PATTERNS.items()
    }
    for name, count in fatal_counts.items():
        if count:
            failures.append(f"{name}={count}")

    finished_matches = FINISHED_RE.findall(client_text)
    final_score = int(finished_matches[-1]) if finished_matches else None
    if final_score is None:
        failures.append("missing controller=finished task=3")
    elif final_score != expected_score:
        failures.append(
            f"final_score={final_score}, expected_score={expected_score}"
        )

    task_scores: dict[int, int] = {}
    for task_id, score in TASK_SCORE_RE.findall(client_text):
        task_scores[int(task_id)] = int(score)
    expected_cumulative = {1: 40, 2: 100}
    for task_id, score in expected_cumulative.items():
        if task_scores.get(task_id) != score:
            failures.append(
                f"task{task_id}_cumulative_score={task_scores.get(task_id)!r}, "
                f"expected={score}"
            )

    server_all_done = "all_tasks_done" in server_text
    if not server_all_done:
        failures.append("server missing all_tasks_done")
    server_total_visible = bool(
        re.search(rf"all_tasks_done[^\r\n]*\b{expected_score}\b", server_text)
        or re.search(rf"\b{expected_score}\s*=.*40.*60.*60", server_text)
    )
    if not server_total_visible:
        failures.append(f"server missing total-score evidence {expected_score}")

    runtime_health = None
    if scheduler_events_text is not None:
        runtime_health = validate_runtime_health(
            scheduler_events_text,
            max_interval_p95_ms=max_interval_p95_ms,
            max_interval_p99_ms=max_interval_p99_ms,
            max_execution_p95_ms=max_execution_p95_ms,
            max_deadline_miss_rate=max_deadline_miss_rate,
        )
        failures.extend(
            f"runtime_health: {message}"
            for message in runtime_health["failures"]
        )

    measured_carry = None
    if require_measured_carry:
        measured_carry = validate_measured_carry(client_text)
        failures.extend(
            f"measured_carry: {message}"
            for message in measured_carry["failures"]
        )

    candidate_applications = None
    if require_candidate_application:
        if scheduler_events_text is None:
            candidate_applications = {
                "passed": False,
                "failures": ["scheduler EventLog is required"],
            }
        else:
            candidate_applications = validate_candidate_applications(
                scheduler_events_text,
                min_applied_candidates=min_applied_candidates,
                min_noncenter_applied=min_noncenter_applied,
            )
        failures.extend(
            f"candidate_applications: {message}"
            for message in candidate_applications["failures"]
        )

    return {
        "passed": not failures,
        "expected_score": expected_score,
        "final_score": final_score,
        "task_cumulative_scores": {
            str(task_id): score for task_id, score in sorted(task_scores.items())
        },
        "server_all_tasks_done": server_all_done,
        "fatal_counts": fatal_counts,
        "runtime_health": runtime_health,
        "measured_carry": measured_carry,
        "candidate_applications": candidate_applications,
        "failures": failures,
    }


def validate_candidate_applications(
    events_text: str,
    *,
    min_applied_candidates: int = 1,
    min_noncenter_applied: int = 0,
) -> dict[str, object]:
    """Prove that ranked candidates reached opt-in executors in one session."""

    if min_applied_candidates < 0 or min_noncenter_applied < 0:
        raise ValueError("candidate application minimums must be non-negative")
    events: list[dict[str, object]] = []
    malformed = 0
    for raw in events_text.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        events.append(event)
    starts = [
        index
        for index, event in enumerate(events)
        if str(event.get("event_type", "")) == "scheduler_started"
    ]
    failures: list[str] = []
    if not starts:
        failures.append("missing scheduler_started session boundary")
        session = events
    else:
        session = events[starts[-1] :]
    terminal_index = None
    for index, event in enumerate(session):
        details = event.get("details")
        if (
            str(event.get("event_type", "")) == "scheduler_transition"
            and isinstance(details, dict)
            and str(details.get("state", "")) == "finished"
        ):
            terminal_index = index
            break
    if terminal_index is None:
        failures.append("missing scheduler finished event boundary")
        evaluated = session
    else:
        evaluated = session[: terminal_index + 1]

    statuses: dict[str, int] = {}
    applied = 0
    noncenter_applied = 0
    invalid_records = 0
    applied_actions: set[str] = set()
    for event in evaluated:
        if str(event.get("event_type", "")) != "candidate_application":
            continue
        details = event.get("details")
        if not isinstance(details, dict):
            invalid_records += 1
            continue
        status = str(details.get("application_status", ""))
        action_id = details.get("action_id")
        if status not in {"applied", "audit_only", "too_late"} or not isinstance(
            action_id, str
        ) or not action_id.strip():
            invalid_records += 1
            continue
        statuses[status] = statuses.get(status, 0) + 1
        if status != "applied":
            continue
        try:
            offset = float(details.get("lateral_offset_m"))
        except (TypeError, ValueError, OverflowError):
            invalid_records += 1
            continue
        if not math.isfinite(offset):
            invalid_records += 1
            continue
        applied += 1
        applied_actions.add(action_id)
        if abs(offset) > 1.0e-6:
            noncenter_applied += 1
    if malformed:
        failures.append(f"malformed_event_lines={malformed}")
    if invalid_records:
        failures.append(f"invalid_candidate_application_records={invalid_records}")
    if applied < min_applied_candidates:
        failures.append(
            f"applied_count={applied} below {min_applied_candidates}"
        )
    if noncenter_applied < min_noncenter_applied:
        failures.append(
            f"noncenter_applied_count={noncenter_applied} below "
            f"{min_noncenter_applied}"
        )
    return {
        "passed": not failures,
        "event_count": sum(statuses.values()),
        "status_counts": dict(sorted(statuses.items())),
        "applied_count": applied,
        "noncenter_applied_count": noncenter_applied,
        "unique_applied_action_count": len(applied_actions),
        "terminal_event_found": terminal_index is not None,
        "limits": {
            "min_applied_candidates": min_applied_candidates,
            "min_noncenter_applied": min_noncenter_applied,
        },
        "failures": failures,
    }


def validate_measured_carry(client_text: str) -> dict[str, object]:
    """Require positive Task1/Task3 evidence from the opt-in carry guard."""

    failures: list[str] = []
    enabled = "measured_carry_guard=True" in client_text
    if not enabled:
        failures.append("startup flag measured_carry_guard=True is missing")

    observations: dict[str, list[float]] = {}
    for source, _half_width, _path_clearance, minimum_clearance in (
        MEASURED_CARRY_RE.findall(client_text)
    ):
        observations.setdefault(source, []).append(float(minimum_clearance))

    required_sources = {"task1", "task3"}
    missing_sources = sorted(required_sources - observations.keys())
    if missing_sources:
        failures.append(
            "missing active guard telemetry for " + ", ".join(missing_sources)
        )

    minimum_clearance_by_source = {
        source: min(values) for source, values in sorted(observations.items())
    }
    minimum_allowed_m = 0.02
    for source in sorted(required_sources & observations.keys()):
        clearance = minimum_clearance_by_source[source]
        if not math.isfinite(clearance) or clearance < minimum_allowed_m:
            failures.append(
                f"{source} minimum_clearance_m={clearance:.3f} below "
                f"{minimum_allowed_m:.3f}"
            )

    guard_stops = client_text.count("carried envelope guard stopped motion")
    if guard_stops:
        failures.append(f"carried_envelope_guard_stops={guard_stops}")

    return {
        "passed": not failures,
        "enabled": enabled,
        "required_sources": sorted(required_sources),
        "observed_sources": sorted(observations),
        "minimum_clearance_m": minimum_clearance_by_source,
        "minimum_allowed_m": minimum_allowed_m,
        "guard_stop_count": guard_stops,
        "failures": failures,
    }


def validate_runtime_health(
    events_text: str,
    *,
    max_interval_p95_ms: float,
    max_interval_p99_ms: float,
    max_execution_p95_ms: float,
    max_deadline_miss_rate: float,
) -> dict[str, object]:
    limits = (
        max_interval_p95_ms,
        max_interval_p99_ms,
        max_execution_p95_ms,
        max_deadline_miss_rate,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in limits):
        raise ValueError("runtime-health limits must be finite and non-negative")
    parsed_events: list[dict[str, object]] = []
    malformed_lines = 0
    for raw in events_text.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            malformed_lines += 1
            continue
        parsed_events.append(event)

    failures: list[str] = []
    session_starts = [
        index
        for index, event in enumerate(parsed_events)
        if str(event.get("event_type", "")) == "scheduler_started"
    ]
    if not session_starts:
        failures.append("missing scheduler_started session boundary")
        session_events = parsed_events
    else:
        session_events = parsed_events[session_starts[-1] :]

    terminal_index = None
    for index, event in enumerate(session_events):
        details = event.get("details")
        if (
            str(event.get("event_type", "")) == "scheduler_transition"
            and isinstance(details, dict)
            and str(details.get("state", "")) == "finished"
        ):
            terminal_index = index
            break
    if terminal_index is None:
        failures.append("missing scheduler finished event boundary")
        evaluation_events = session_events
    else:
        evaluation_events = session_events[: terminal_index + 1]

    unexpected_stale_events = sum(
        1
        for event in evaluation_events
        if str(event.get("event_type", "")) in {"input_stale", "safety_stop"}
    )
    records: list[dict[str, object]] = []
    for event in evaluation_events:
        event_type = str(event.get("event_type", ""))
        details = event.get("details")
        if event_type != "control_loop_health" or not isinstance(details, dict):
            continue
        try:
            if int(details.get("sample_count", 0)) >= 400:
                records.append(details)
        except (TypeError, ValueError):
            malformed_lines += 1

    if malformed_lines:
        failures.append(f"malformed_event_lines={malformed_lines}")
    if unexpected_stale_events:
        failures.append(f"unexpected_stale_or_safety_events={unexpected_stale_events}")
    if not records:
        failures.append("missing full-window control_loop_health report")
        return {
            "passed": False,
            "report_count": 0,
            "terminal_event_found": terminal_index is not None,
            "failures": failures,
        }

    def maximum(name: str) -> float:
        return max(float(record.get(name, math.inf)) for record in records)

    max_p95 = maximum("interval_p95_ms")
    max_p99 = maximum("interval_p99_ms")
    max_exec_p95 = maximum("execution_p95_ms")
    latest = records[-1]
    try:
        interval_total = int(latest["total_interval_count"])
        sample_total = int(latest["total_sample_count"])
        interval_misses = int(latest["interval_deadline_misses"])
        execution_misses = int(latest["execution_deadline_misses"])
        interval_miss_rate = interval_misses / max(1, interval_total)
        execution_miss_rate = execution_misses / max(1, sample_total)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        interval_miss_rate = math.inf
        execution_miss_rate = math.inf
        failures.append("missing cumulative loop counters")

    checks = (
        (max_p95, max_interval_p95_ms, "interval_p95_ms"),
        (max_p99, max_interval_p99_ms, "interval_p99_ms"),
        (max_exec_p95, max_execution_p95_ms, "execution_p95_ms"),
        (interval_miss_rate, max_deadline_miss_rate, "interval_deadline_miss_rate"),
        (execution_miss_rate, max_deadline_miss_rate, "execution_deadline_miss_rate"),
    )
    for observed, limit, name in checks:
        if not math.isfinite(observed) or observed > limit:
            failures.append(f"{name}={observed:.6f} exceeds {limit:.6f}")

    return {
        "passed": not failures,
        "report_count": len(records),
        "session_event_count": len(session_events),
        "evaluated_event_count": len(evaluation_events),
        "terminal_event_found": terminal_index is not None,
        "max_interval_p95_ms": max_p95,
        "max_interval_p99_ms": max_p99,
        "max_execution_p95_ms": max_exec_p95,
        "interval_deadline_miss_rate": interval_miss_rate,
        "execution_deadline_miss_rate": execution_miss_rate,
        "limits": {
            "max_interval_p95_ms": max_interval_p95_ms,
            "max_interval_p99_ms": max_interval_p99_ms,
            "max_execution_p95_ms": max_execution_p95_ms,
            "max_deadline_miss_rate": max_deadline_miss_rate,
        },
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate saved Client/Server logs for a full 160-point run."
    )
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--expected-score", type=int, default=160)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--max-interval-p95-ms", type=float, default=65.0)
    parser.add_argument("--max-interval-p99-ms", type=float, default=125.0)
    parser.add_argument("--max-execution-p95-ms", type=float, default=50.0)
    parser.add_argument("--max-deadline-miss-rate", type=float, default=0.01)
    parser.add_argument(
        "--require-measured-carry",
        action="store_true",
        help="require positive Task1/Task3 measured carry-guard telemetry",
    )
    parser.add_argument(
        "--require-candidate-application",
        action="store_true",
        help="require ranked candidates to be applied by opt-in executors",
    )
    parser.add_argument("--min-applied-candidates", type=int, default=1)
    parser.add_argument("--min-noncenter-applied", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_run(
        _read(args.client),
        _read(args.server),
        expected_score=args.expected_score,
        scheduler_events_text=_read(args.events) if args.events is not None else None,
        max_interval_p95_ms=args.max_interval_p95_ms,
        max_interval_p99_ms=args.max_interval_p99_ms,
        max_execution_p95_ms=args.max_execution_p95_ms,
        max_deadline_miss_rate=args.max_deadline_miss_rate,
        require_measured_carry=args.require_measured_carry,
        require_candidate_application=args.require_candidate_application,
        min_applied_candidates=args.min_applied_candidates,
        min_noncenter_applied=args.min_noncenter_applied,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
