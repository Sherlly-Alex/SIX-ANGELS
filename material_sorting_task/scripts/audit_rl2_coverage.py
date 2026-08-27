#!/usr/bin/env python3
"""Write RL-2 coverage artifacts without generating figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.coverage_audit import (
    CoverageThresholds,
    audit_coverage,
    write_coverage_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit RL-2 scheduler dataset coverage")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--coverage-json", required=True, type=Path)
    parser.add_argument("--coverage-csv", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--minimum-total", type=int, default=30000)
    parser.add_argument("--minimum-simulation", type=int, default=24000)
    parser.add_argument("--minimum-official", type=int, default=6000)
    args = parser.parse_args(argv)
    report = audit_coverage(
        args.inputs,
        thresholds=CoverageThresholds(
            minimum_total=args.minimum_total,
            minimum_simulation=args.minimum_simulation,
            minimum_official=args.minimum_official,
        ),
    )
    write_coverage_artifacts(
        report,
        json_path=args.coverage_json,
        csv_path=args.coverage_csv,
        failures_path=args.failures,
        manifest_path=args.manifest,
    )
    print(f"passed={str(report.passed).lower()} training_ready={report.training_ready_decisions}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
