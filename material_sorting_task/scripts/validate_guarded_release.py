#!/usr/bin/env python3
"""Validate the frozen Guarded manifest, model and production schema chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.observation import observation_schema_hash
from learning.promotion import validate_guarded_approval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = validate_guarded_approval(
        args.approval,
        expected_manifest_sha256=args.approval_sha256,
        model_path=args.model,
        expected_model_sha256=args.model_sha256,
        expected_schema_hash=observation_schema_hash(args.max_candidates),
    ).to_json_dict()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
