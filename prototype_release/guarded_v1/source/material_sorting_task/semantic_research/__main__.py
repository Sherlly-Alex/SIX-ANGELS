"""CLI: python -m semantic_research --dataset ... --rows-out ... --metrics-out ..."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import evaluate_path
from .logger import write_evaluation
from .regex_adapter import predict_from_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline regex semantic evaluation")
    default_data = Path(__file__).resolve().parent / "data" / "text_eval.jsonl"
    parser.add_argument("--dataset", type=Path, default=default_data)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="test",
        help="Only evaluate records with this split (empty = all)",
    )
    args = parser.parse_args(argv)

    from .evaluator import load_jsonl, evaluate

    records = load_jsonl(args.dataset)
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    result = evaluate(records, predict_from_text)
    write_evaluation(
        result,
        rows_path=args.rows_out,
        metrics_path=args.metrics_out,
    )
    metrics = result["metrics"]
    print(
        f"n={metrics['n']} complete={metrics['complete_match_rate']:.3f} "
        f"p50_ms={metrics['latency_ms']['p50']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
