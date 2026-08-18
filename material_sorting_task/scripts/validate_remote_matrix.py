#!/usr/bin/env python3
"""Aggregate full-score acceptance over multiple randomized-layout seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from validate_remote_run import validate_run


def validate_matrix(
    root: Path,
    seeds: list[int],
    *,
    expected_score: int = 160,
    require_events: bool = False,
    require_measured_carry: bool = False,
    max_interval_p95_ms: float = 65.0,
    max_interval_p99_ms: float = 125.0,
    max_execution_p95_ms: float = 50.0,
    max_deadline_miss_rate: float = 0.01,
) -> dict[str, object]:
    seed_values = [int(seed) for seed in seeds]
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    results: dict[str, object] = {}
    failures: list[str] = []
    for seed in seed_values:
        run_name = f"v2_multiseed_{seed}"
        run_dir = root / run_name
        client = run_dir / f"client_{run_name}.log"
        server = run_dir / f"server_{run_name}.log"
        events = run_dir / f"scheduler_{run_name}.jsonl"
        required_paths = [client, server]
        if require_events:
            required_paths.append(events)
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            report: dict[str, object] = {
                "passed": False,
                "failures": [f"missing artifact: {path}" for path in missing],
            }
        else:
            report = validate_run(
                client.read_text(encoding="utf-8", errors="replace"),
                server.read_text(encoding="utf-8", errors="replace"),
                expected_score=expected_score,
                scheduler_events_text=(
                    events.read_text(encoding="utf-8", errors="replace")
                    if require_events
                    else None
                ),
                max_interval_p95_ms=max_interval_p95_ms,
                max_interval_p99_ms=max_interval_p99_ms,
                max_execution_p95_ms=max_execution_p95_ms,
                max_deadline_miss_rate=max_deadline_miss_rate,
                require_measured_carry=require_measured_carry,
            )
        results[str(seed)] = report
        if not report["passed"]:
            failures.append(str(seed))
    return {
        "passed": not failures and len(results) == len(seed_values),
        "expected_score_per_seed": expected_score,
        "require_events": bool(require_events),
        "require_measured_carry": bool(require_measured_carry),
        "seed_count": len(seed_values),
        "passed_seed_count": len(seed_values) - len(failures),
        "failed_seeds": failures,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--expected-score", default=160, type=int)
    parser.add_argument("--require-events", action="store_true")
    parser.add_argument("--require-measured-carry", action="store_true")
    parser.add_argument("--max-interval-p95-ms", type=float, default=65.0)
    parser.add_argument("--max-interval-p99-ms", type=float, default=125.0)
    parser.add_argument("--max-execution-p95-ms", type=float, default=50.0)
    parser.add_argument("--max-deadline-miss-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_matrix(
        args.root,
        args.seeds,
        expected_score=args.expected_score,
        require_events=args.require_events,
        require_measured_carry=args.require_measured_carry,
        max_interval_p95_ms=args.max_interval_p95_ms,
        max_interval_p99_ms=args.max_interval_p99_ms,
        max_execution_p95_ms=args.max_execution_p95_ms,
        max_deadline_miss_rate=args.max_deadline_miss_rate,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
