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
) -> dict[str, object]:
    results: dict[str, object] = {}
    failures: list[str] = []
    for seed in seeds:
        run_name = f"v2_multiseed_{seed}"
        run_dir = root / run_name
        client = run_dir / f"client_{run_name}.log"
        server = run_dir / f"server_{run_name}.log"
        missing = [str(path) for path in (client, server) if not path.is_file()]
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
            )
        results[str(seed)] = report
        if not report["passed"]:
            failures.append(str(seed))
    return {
        "passed": not failures and len(results) == len(seeds),
        "expected_score_per_seed": expected_score,
        "seed_count": len(seeds),
        "passed_seed_count": len(seeds) - len(failures),
        "failed_seeds": failures,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--expected-score", default=160, type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_matrix(
        args.root,
        args.seeds,
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
