"""Offline ML slot parser (research only). Soft-fails without sklearn/weights."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import COMPARABLE_SLOTS, SemanticPrediction

_PARSER_NAME = "ml_tfidf_lr"
_NONE = "__none__"
_DESTINATION_MARKER = re.compile(
    r"放到|放入|放进|置于|复位到|移至|移动到|归还到|摆在|放在"
)


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "ml_slots.joblib"


def load_model(model_path: str | Path | None = None) -> Any | None:
    path = Path(model_path) if model_path else _default_model_path()
    if not path.is_file():
        return None
    try:
        import joblib
    except Exception:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def slot_feature_text(text: str, slot: str) -> str:
    """Return the clause relevant to one learnable semantic slot.

    This is deterministic feature selection, not a rule parser: labels still
    come solely from the per-slot ML classifier.  It prevents source-side
    wording such as ``桌面右侧`` from becoming evidence for a destination-side
    ``direction`` label.
    """
    match = _DESTINATION_MARKER.search(text)
    if match is None:
        return text
    pickup = text[: match.start()]
    destination = text[match.end() :]
    return pickup if slot == "target_color" else destination


def apply_explicit_consistency_guard(
    text: str,
    slots: dict[str, str | None],
) -> dict[str, str | None]:
    """Remove only predictions contradicted by explicit wording.

    This optional research guard is intentionally separate from the ML model.
    It does not infer missing fields, and it never runs in the competition
    control path.  Its purpose is to report a transparent ``ML + guard``
    baseline for clear ambiguity cases.
    """
    guarded = dict(slots)
    match = _DESTINATION_MARKER.search(text)
    pickup = text[: match.start()] if match else text
    destination = text[match.end() :] if match else ""

    color_count = sum(
        bool(re.search(pattern, pickup))
        for pattern in (r"粉(?:红)?色", r"黄色|黄颜色", r"褐色|棕色|咖啡色")
    )
    if color_count != 1:
        guarded["target_color"] = None

    has_left = bool(re.search(r"左边|左侧|左方", destination))
    has_right = bool(re.search(r"右边|右侧|右方", destination))
    if has_left == has_right:
        guarded["direction"] = None
        if has_left:
            guarded["place_type"] = None
    if not (has_left or has_right):
        guarded["reference_kind"] = None
    elif re.search(r"白色长方体|包装盒|长方体包装盒", destination):
        guarded["reference_kind"] = "packaging_box"
    return guarded


def predict_from_text(
    text: str,
    *,
    model_path: str | Path | None = None,
    model: Any | None = None,
    apply_guard: bool = False,
) -> SemanticPrediction:
    """Predict comparable slots. Never raises into formal code."""
    bundle = model
    if bundle is None:
        bundle = load_model(model_path)
    if bundle is None:
        return SemanticPrediction.failure(
            _PARSER_NAME,
            f"model_missing:{model_path or _default_model_path()}",
        )

    try:
        vectorizers = bundle.get("vectorizers")
        vectorizer = bundle.get("vectorizer")
        classifiers = bundle["classifiers"]
        label_maps = bundle.get("label_maps") or {}
    except Exception as exc:
        return SemanticPrediction.failure(_PARSER_NAME, f"model_corrupt: {exc}")

    if not isinstance(text, str) or not text.strip():
        return SemanticPrediction.failure(_PARSER_NAME, "empty_text")

    try:
        slots: dict[str, str | None] = {}
        confidences: list[float] = []
        for slot in COMPARABLE_SLOTS:
            clf = classifiers[slot]
            slot_vectorizer = vectorizers.get(slot) if vectorizers else vectorizer
            if slot_vectorizer is None:
                raise ValueError(f"missing_vectorizer:{slot}")
            features = slot_vectorizer.transform([slot_feature_text(text.strip(), slot)])
            raw = clf.predict(features)[0]
            if hasattr(clf, "predict_proba"):
                proba = float(max(clf.predict_proba(features)[0]))
            else:
                proba = 1.0
            confidences.append(proba)
            label = label_maps.get(slot, {}).get(raw, raw)
            slots[slot] = None if label in (None, _NONE) else str(label)
        if apply_guard:
            slots = apply_explicit_consistency_guard(text.strip(), slots)
        return SemanticPrediction(
            target_color=slots["target_color"],
            place_type=slots["place_type"],
            direction=slots["direction"],
            reference_kind=slots["reference_kind"],
            confidence=sum(confidences) / max(len(confidences), 1),
            parser_name=(f"{_PARSER_NAME}_guarded" if apply_guard else _PARSER_NAME),
            errors=(),
        )
    except Exception as exc:
        return SemanticPrediction.failure(_PARSER_NAME, f"inference_error: {exc}")


def model_size_bytes(model_path: str | Path | None = None) -> int | None:
    path = Path(model_path) if model_path else _default_model_path()
    if not path.is_file():
        return None
    return path.stat().st_size
