#!/usr/bin/env python3
"""Validate one official full-task remote run from saved text artifacts."""

from __future__ import annotations

import argparse
import json
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def validate_run(
    client_text: str,
    server_text: str,
    *,
    expected_score: int = 160,
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

    return {
        "passed": not failures,
        "expected_score": expected_score,
        "final_score": final_score,
        "task_cumulative_scores": {
            str(task_id): score for task_id, score in sorted(task_scores.items())
        },
        "server_all_tasks_done": server_all_done,
        "fatal_counts": fatal_counts,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate saved Client/Server logs for a full 160-point run."
    )
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--expected-score", type=int, default=160)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_run(
        _read(args.client),
        _read(args.server),
        expected_score=args.expected_score,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
