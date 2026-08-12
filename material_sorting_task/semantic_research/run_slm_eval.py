"""Run offline SLM evaluation against the shared JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import evaluate, load_jsonl
from .logger import write_evaluation
from .slm_parser import default_weight_path, predict_from_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline local SLM evaluation")
    default_data = Path(__file__).resolve().parent / "data" / "text_eval.jsonl"
    parser.add_argument("--dataset", type=Path, default=default_data)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=default_weight_path())
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--tag",
        default="",
        help="If set, only rows containing this tag (e.g. standard / oral)",
    )
    parser.add_argument(
        "--exclude-tag",
        default="",
        help="Exclude rows with this tag; use to avoid scoring in-context demos.",
    )
    args = parser.parse_args(argv)

    records = load_jsonl(args.dataset)
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    if args.tag:
        records = [r for r in records if args.tag in (r.get("tags") or [])]
    if args.exclude_tag:
        records = [
            r for r in records if args.exclude_tag not in (r.get("tags") or [])
        ]

    def predictor(text: str):
        return predict_from_text(
            text,
            weight_path=args.weights,
            timeout_s=args.timeout_s,
        )

    result = evaluate(records, predictor)
    write_evaluation(
        result, rows_path=args.rows_out, metrics_path=args.metrics_out
    )

    # Extra SLM-oriented rates from row errors.
    rows = result["rows"]
    n = max(len(rows), 1)
    timeout_n = sum(
        1
        for row in rows
        if any(str(e).startswith("timeout:") for e in row["prediction"]["errors"])
    )
    json_fail_n = sum(
        1
        for row in rows
        if any(str(e).startswith("json_format_error") for e in row["prediction"]["errors"])
    )
    illegal_n = sum(
        1
        for row in rows
        if any(str(e).startswith("illegal_enum") for e in row["prediction"]["errors"])
    )
    missing_w = sum(
        1
        for row in rows
        if any(str(e).startswith("weights_missing") for e in row["prediction"]["errors"])
    )
    extra = {
        "timeout_rate": timeout_n / n,
        "json_format_failure_rate": json_fail_n / n,
        "illegal_enum_rate": illegal_n / n,
        "weights_missing_rate": missing_w / n,
        "hallucination_proxy_rate": illegal_n / n,
        "weights": str(args.weights),
        "weights_present": args.weights.is_file(),
        "note": "Default run does not download or load SLM weights.",
    }
    metrics_path = Path(args.metrics_out)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["slm"] = extra
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(extra, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
