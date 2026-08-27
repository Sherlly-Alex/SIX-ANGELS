#!/usr/bin/env python3
"""Validate selective input-drop recovery and terminal stale handling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        events.append(value)
    return events


def validate(
    events: Iterable[dict[str, Any]],
    *,
    expect_recovered: Iterable[str],
    expect_terminal: Iterable[str],
) -> dict[str, Any]:
    required_recovered = set(expect_recovered)
    required_terminal = set(expect_terminal)
    observed_stale: set[str] = set()
    observed_recovered: set[str] = set()
    observed_terminal: set[str] = set()
    pending_stale: set[str] = set()
    loop_reports = 0
    max_interval_p99_ms = 0.0

    for event in events:
        event_type = str(event.get("event_type", ""))
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        sources = {
            str(value)
            for value in details.get("stale_inputs", [])
            if str(value)
        }
        if event_type == "input_stale":
            observed_stale.update(sources)
            pending_stale = set(sources)
        elif event_type == "input_recovered":
            observed_recovered.update(pending_stale)
            pending_stale.clear()
        elif event_type == "safety_stop" and event.get("failure_code") == "input_stale":
            observed_terminal.update(sources or pending_stale)
            pending_stale.clear()
        elif event_type == "control_loop_health":
            loop_reports += 1
            try:
                max_interval_p99_ms = max(
                    max_interval_p99_ms,
                    float(details.get("interval_p99_ms", 0.0)),
                )
            except (TypeError, ValueError):
                pass

    errors: list[str] = []
    missing_stale = (required_recovered | required_terminal) - observed_stale
    missing_recovered = required_recovered - observed_recovered
    missing_terminal = required_terminal - observed_terminal
    if missing_stale:
        errors.append(f"missing input_stale evidence: {sorted(missing_stale)}")
    if missing_recovered:
        errors.append(f"missing input_recovered evidence: {sorted(missing_recovered)}")
    if missing_terminal:
        errors.append(f"missing terminal INPUT_STALE evidence: {sorted(missing_terminal)}")
    if loop_reports < 1:
        errors.append("missing control_loop_health event")

    return {
        "passed": not errors,
        "errors": errors,
        "observed_stale": sorted(observed_stale),
        "observed_recovered": sorted(observed_recovered),
        "observed_terminal": sorted(observed_terminal),
        "control_loop_report_count": loop_reports,
        "max_interval_p99_ms": max_interval_p99_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-recovered", nargs="*", default=[])
    parser.add_argument("--expect-terminal", nargs="*", default=[])
    args = parser.parse_args()
    try:
        report = validate(
            load_events(args.events),
            expect_recovered=args.expect_recovered,
            expect_terminal=args.expect_terminal,
        )
    except (OSError, ValueError) as exc:
        report = {"passed": False, "errors": [str(exc)]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
