#!/usr/bin/env python3
"""Audit scheduler JSONL files and export safe offline-learning records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.event_replay import replay_event_logs, write_replay_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay scheduler EventLog files without Server-private truth"
    )
    parser.add_argument("events", nargs="+", type=Path)
    parser.add_argument("--min-decisions", type=int, default=1)
    parser.add_argument("--require-training-ready", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, records = replay_event_logs(
        args.events,
        min_decisions=args.min_decisions,
        require_training_ready=args.require_training_ready,
    )
    if args.dataset is not None and summary.passed:
        write_replay_dataset(args.dataset, records)
    payload = json.dumps(summary.to_json_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
