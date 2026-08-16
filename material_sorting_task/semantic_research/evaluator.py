"""Read-only comparison of research predictions against comparable gold slots."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .schema import COMPARABLE_SLOTS, SemanticPrediction

Predictor = Callable[[str], SemanticPrediction]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            records.append(obj)
    return records


def gold_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    gold = record.get("gold")
    if isinstance(gold, Mapping):
        return {
            slot: gold.get(slot)
            for slot in COMPARABLE_SLOTS
        }
    # Allow Server-like structured fields without place_world leakage.
    return {
        "target_color": record.get("target_color"),
        "place_type": record.get("place_type"),
        "direction": record.get("direction"),
        "reference_kind": record.get("reference_kind") or record.get("ref_prop"),
    }


def compare_prediction(
    prediction: SemanticPrediction,
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    slot_hits: dict[str, bool] = {}
    missing = 0
    conflicts = 0
    for slot in COMPARABLE_SLOTS:
        predicted = getattr(prediction, slot)
        expected = gold.get(slot)
        if predicted is None:
            missing += 1
            slot_hits[slot] = expected is None and not prediction.errors
        elif expected is None:
            conflicts += 1
            slot_hits[slot] = False
        elif predicted != expected:
            conflicts += 1
            slot_hits[slot] = False
        else:
            slot_hits[slot] = True

    complete_match = (
        not prediction.errors
        and all(slot_hits.values())
        and all(
            getattr(prediction, slot) == gold.get(slot)
            for slot in COMPARABLE_SLOTS
        )
    )
    return {
        "slot_hits": slot_hits,
        "missing": missing,
        "conflicts": conflicts,
        "complete_match": complete_match,
        "has_errors": bool(prediction.errors),
    }


def evaluate(
    records: Sequence[Mapping[str, Any]],
    predictor: Predictor,
    *,
    parser_filter: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    slot_correct = {slot: 0 for slot in COMPARABLE_SLOTS}
    slot_total = {slot: 0 for slot in COMPARABLE_SLOTS}
    complete = 0
    missing_total = 0
    conflict_total = 0
    error_total = 0
    latencies_ms: list[float] = []

    for record in records:
        text = str(record.get("text") or record.get("instruction") or "")
        gold = gold_from_record(record)
        started = time.perf_counter()
        prediction = predictor(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        if parser_filter and prediction.parser_name != parser_filter:
            continue
        comparison = compare_prediction(prediction, gold)
        for slot, hit in comparison["slot_hits"].items():
            slot_total[slot] += 1
            if hit:
                slot_correct[slot] += 1
        complete += int(comparison["complete_match"])
        missing_total += int(comparison["missing"])
        conflict_total += int(comparison["conflicts"])
        error_total += int(comparison["has_errors"])
        rows.append(
            {
                "id": record.get("id"),
                "text": text,
                "gold": gold,
                "prediction": prediction.to_dict(),
                "comparison": comparison,
                "latency_ms": elapsed_ms,
                "tags": list(record.get("tags") or []),
                "split": record.get("split"),
            }
        )

    n = max(len(rows), 1)
    latencies_ms_sorted = sorted(latencies_ms) if latencies_ms else [0.0]

    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return values[idx]

    metrics = {
        "n": len(rows),
        "slot_accuracy": {
            slot: (slot_correct[slot] / slot_total[slot] if slot_total[slot] else 0.0)
            for slot in COMPARABLE_SLOTS
        },
        "complete_match_rate": complete / n,
        "missing_rate": missing_total / (n * len(COMPARABLE_SLOTS)),
        "conflict_rate": conflict_total / (n * len(COMPARABLE_SLOTS)),
        "error_rate": error_total / n,
        "latency_ms": {
            "mean": sum(latencies_ms_sorted) / len(latencies_ms_sorted),
            "p50": _percentile(latencies_ms_sorted, 0.50),
            "p95": _percentile(latencies_ms_sorted, 0.95),
        },
    }
    return {"metrics": metrics, "rows": rows}


def evaluate_path(
    dataset_path: str | Path,
    predictor: Predictor,
) -> dict[str, Any]:
    return evaluate(load_jsonl(dataset_path), predictor)
