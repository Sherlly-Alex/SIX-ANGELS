"""Write offline evaluation artifacts to explicitly provided paths only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def write_metrics(path: str | Path, metrics: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def write_evaluation(
    result: Mapping[str, Any],
    *,
    rows_path: str | Path,
    metrics_path: str | Path,
) -> tuple[Path, Path]:
    rows = list(result.get("rows") or [])
    metrics = dict(result.get("metrics") or {})
    return write_jsonl(rows_path, rows), write_metrics(metrics_path, metrics)
