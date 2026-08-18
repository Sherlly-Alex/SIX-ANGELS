#!/usr/bin/env python3
"""Validate scheduler model, schema, training config and provenance hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.model_package import validate_model_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a scheduler model package")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-provenance-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_model_package(
        args.model,
        expected_model_sha256=args.expected_model_sha256,
        expected_provenance_sha256=args.expected_provenance_sha256,
    )
    payload = json.dumps(report.to_json_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
