"""Train TF-IDF + LogisticRegression slot classifiers for offline research."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluator import gold_from_record, load_jsonl
from .ml_parser import _NONE, _default_model_path, slot_feature_text
from .schema import COMPARABLE_SLOTS


def _encode_label(value: Any) -> str:
    if value is None:
        return _NONE
    return str(value)


def train_bundle(
    records: list[dict[str, Any]],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sklearn is required for train_ml; install requirements-research.txt"
        ) from exc

    rng = random.Random(seed)
    rows = list(records)
    rng.shuffle(rows)
    texts = [str(r.get("text") or r.get("instruction") or "") for r in rows]
    golds = [gold_from_record(r) for r in rows]

    classifiers: dict[str, Any] = {}
    vectorizers: dict[str, Any] = {}
    label_maps: dict[str, dict[str, str]] = {}
    distributions: dict[str, dict[str, int]] = {}
    for slot in COMPARABLE_SLOTS:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=1,
        )
        x = vectorizer.fit_transform([slot_feature_text(text, slot) for text in texts])
        y = [_encode_label(g.get(slot)) for g in golds]
        distributions[slot] = dict(Counter(y))
        # LogisticRegression needs >=2 classes; inject a sentinel if needed.
        classes = sorted(set(y))
        if len(classes) < 2:
            y = list(y) + ([_NONE] if classes[0] != _NONE else ["__other__"])
            # Fit on duplicated matrix row for the sentinel.
            import numpy as np
            from scipy.sparse import vstack

            x_fit = vstack([x, x[0]])
            y_fit = y
        else:
            x_fit = x
            y_fit = y
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            multi_class="auto",
        )
        clf.fit(x_fit, y_fit)
        classifiers[slot] = clf
        vectorizers[slot] = vectorizer
        label_maps[slot] = {label: label for label in clf.classes_}

    return {
        "vectorizers": vectorizers,
        "classifiers": classifiers,
        "label_maps": label_maps,
        "meta": {
            "seed": seed,
            "n_train": len(rows),
            "distributions": distributions,
            "slots": list(COMPARABLE_SLOTS),
        },
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train offline ML slot parser")
    default_data = Path(__file__).resolve().parent / "data" / "text_eval.jsonl"
    parser.add_argument("--dataset", type=Path, default=default_data)
    parser.add_argument("--out", type=Path, default=_default_model_path())
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits used for fitting. Default is train only; "
        "never include test. Use val only for explicit model-selection runs.",
    )
    args = parser.parse_args(argv)

    allowed = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "test" in allowed:
        raise SystemExit(
            "refusing to train on split=test; omit test from --splits "
            "(default is train only; val is for model selection only)"
        )
    records = [r for r in load_jsonl(args.dataset) if r.get("split") in allowed]
    if any(r.get("split") == "test" for r in records):
        raise SystemExit("internal error: test records leaked into training set")
    if len(records) < 3:
        raise SystemExit(f"need at least 3 records to train, got {len(records)}")

    bundle = train_bundle(records, seed=args.seed)
    bundle["meta"]["train_splits"] = sorted(allowed)
    bundle["meta"]["includes_test"] = False
    save_bundle(bundle, args.out)
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(bundle["meta"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} n={bundle['meta']['n_train']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
