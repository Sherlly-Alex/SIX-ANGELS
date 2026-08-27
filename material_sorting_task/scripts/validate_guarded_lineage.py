#!/usr/bin/env python3
"""Trace applied scheduler candidates back to RL selections in Guarded logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


RL_LINEAGE = frozenset({"rl_direct", "rl_latched", "rl_lineage"})


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event["_raw"] = json.dumps(event, ensure_ascii=False, sort_keys=True)
            events.append(event)
    return events


def _source(event: dict[str, Any]) -> str | None:
    details = event.get("details")
    if not isinstance(details, dict):
        details = {}
    value = details.get("source", event.get("source"))
    return str(value) if value else None


def validate_file(path: Path) -> dict[str, Any]:
    events = _read_events(path)
    selections = [event for event in events if event.get("event_type") == "action_selected"]
    applications = []
    for event in events:
        details = event.get("details")
        if not isinstance(details, dict):
            details = {}
        if (
            event.get("event_type") == "candidate_application"
            and details.get("application_status") == "applied"
        ):
            applications.append(event)

    records: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()
    for application in applications:
        details = application.get("details")
        if not isinstance(details, dict):
            details = {}
        action_id = details.get("action_id")
        sequence = int(application.get("sequence", 0))
        step_run_id = application.get("step_run_id")

        history = []
        for selection in selections:
            if selection.get("step_run_id") != step_run_id:
                continue
            if int(selection.get("sequence", 0)) > sequence:
                continue
            if action_id and str(action_id) not in selection["_raw"]:
                continue
            history.append(selection)
        history.sort(key=lambda item: int(item.get("sequence", 0)))
        sources = [source for source in map(_source, history) if source]
        last_source = sources[-1] if sources else None

        if last_source == "rl":
            classification = "rl_direct"
        elif last_source == "hysteresis" and "rl" in sources:
            classification = "rl_latched"
        elif last_source == "heuristic":
            classification = "heuristic"
        elif "rl" in sources:
            classification = "rl_lineage"
        else:
            classification = "unresolved"
        classifications[classification] += 1
        records.append(
            {
                "task_id": application.get("task_id"),
                "step_id": application.get("step_id"),
                "step_run_id": step_run_id,
                "action_id": action_id,
                "classification": classification,
                "source_counts": dict(Counter(sources)),
            }
        )

    rl_origin = sum(classifications[name] for name in RL_LINEAGE)
    return {
        "events_path": str(path),
        "applied_count": len(applications),
        "rl_origin_applied": rl_origin,
        "lineage_summary": dict(classifications),
        "applications": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", type=Path)
    parser.add_argument("--minimum-applied", type=int, default=1)
    parser.add_argument("--minimum-rl-origin-applied", type=int, default=1)
    parser.add_argument("--require-all-applied-from-rl", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [validate_file(path) for path in args.events]
    failures: list[str] = []
    for result in results:
        path = result["events_path"]
        applied = int(result["applied_count"])
        rl_origin = int(result["rl_origin_applied"])
        if applied < args.minimum_applied:
            failures.append(f"{path}: applied_count={applied} below {args.minimum_applied}")
        if rl_origin < args.minimum_rl_origin_applied:
            failures.append(
                f"{path}: rl_origin_applied={rl_origin} below "
                f"{args.minimum_rl_origin_applied}"
            )
        if args.require_all_applied_from_rl and rl_origin != applied:
            failures.append(
                f"{path}: rl_origin_applied={rl_origin} does not equal applied_count={applied}"
            )

    report = {
        "schema_version": "guarded-lineage-acceptance-v1",
        "passed": not failures,
        "files": len(results),
        "applied_count": sum(int(item["applied_count"]) for item in results),
        "rl_origin_applied": sum(int(item["rl_origin_applied"]) for item in results),
        "results": results,
        "failures": failures,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
