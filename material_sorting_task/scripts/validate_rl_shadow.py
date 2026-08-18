#!/usr/bin/env python3
"""Validate that a recorded RL-shadow run never controlled the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.shadow_gate import validate_rl_shadow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RL-shadow EventLogs")
    parser.add_argument("events", nargs="+", type=Path)
    parser.add_argument("--min-suggestions", type=int, default=1000)
    parser.add_argument("--max-inference-p95-ms", type=float, default=25.0)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = validate_rl_shadow(
        args.events,
        min_suggestions=args.min_suggestions,
        max_inference_p95_ms=args.max_inference_p95_ms,
        max_fallback_rate=args.max_fallback_rate,
        expected_model_sha256=args.expected_model_sha256,
    )
    payload = json.dumps(summary.to_json_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
