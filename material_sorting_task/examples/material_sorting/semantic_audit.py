"""Best-effort, research-only audit of Server instruction text.

The competition path trusts structured Server JSON.  This module compares
text-only research parsers against that already-accepted JSON asynchronously;
it never returns data to the controller and never rejects a task.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_SLOTS = ("target_color", "place_type", "direction", "reference_kind")


def _enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _positive_float_env(name: str, default: float) -> float:
    """Read an optional research timeout without risking Client startup."""
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) and value > 0.0 else float(default)


def _research_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalised_truth(instruction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_color": instruction.get("target_color"),
        "place_type": instruction.get("place_type"),
        "direction": instruction.get("direction"),
        "reference_kind": instruction.get("ref_prop")
        or instruction.get("ref_prop_body"),
    }


def compare_prediction(prediction: Any, truth: Mapping[str, Any]) -> list[str]:
    """Return text-observable slot disagreements; errors are reported too."""
    errors = [str(item) for item in getattr(prediction, "errors", ())]
    if errors:
        return ["unavailable=" + "|".join(errors)]
    mismatches = [
        slot
        for slot in _SLOTS
        if getattr(prediction, slot, None) != truth.get(slot)
    ]
    return mismatches


class SemanticAudit:
    """Run optional research parsers without influencing robot control."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log
        self._enabled = _enabled("MATERIAL_SEMANTIC_AUDIT")
        self._use_ml = self._enabled and _enabled("MATERIAL_SEMANTIC_AUDIT_ML", True)
        self._use_slm = self._enabled and _enabled("MATERIAL_SEMANTIC_AUDIT_SLM")
        self._ml_guard = _enabled("MATERIAL_SEMANTIC_AUDIT_ML_GUARD", True)
        self._ml_model = os.environ.get("MATERIAL_SEMANTIC_AUDIT_ML_MODEL")
        self._slm_timeout_s = _positive_float_env(
            "MATERIAL_SEMANTIC_AUDIT_SLM_TIMEOUT_S",
            30.0,
        )
        self._slm_weights = os.environ.get("MATERIAL_SEMANTIC_AUDIT_SLM_WEIGHTS")
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-audit")
            if self._enabled
            else None
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def describe(self) -> str:
        if not self._enabled:
            return "semantic audit disabled"
        parsers = ["regex"]
        if self._use_ml:
            parsers.append("ml+guard" if self._ml_guard else "ml")
        if self._use_slm:
            parsers.append("slm")
        return "semantic audit enabled (log-only): " + ", ".join(parsers)

    def submit(self, instructions: Sequence[Mapping[str, Any]]) -> None:
        if not self._enabled or self._executor is None:
            return
        snapshot = [dict(item) for item in instructions]
        self._executor.submit(self._run, snapshot)

    def _run(self, instructions: Sequence[Mapping[str, Any]]) -> None:
        root = _research_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from semantic_research.regex_adapter import predict_from_text as regex_predict
        except Exception as exc:
            self._log(f"SEM_AUDIT unavailable parser=regex error={type(exc).__name__}:{exc}")
            return

        parsers: list[tuple[str, Callable[[str], Any]]] = [("regex", regex_predict)]
        if self._use_ml:
            try:
                from semantic_research.ml_parser import predict_from_text as ml_predict

                parsers.append(
                    (
                        "ml+guard" if self._ml_guard else "ml",
                        lambda text: ml_predict(
                            text,
                            model_path=self._ml_model,
                            apply_guard=self._ml_guard,
                        ),
                    )
                )
            except Exception as exc:
                self._log(f"SEM_AUDIT unavailable parser=ml error={type(exc).__name__}:{exc}")
        if self._use_slm:
            try:
                from semantic_research.slm_parser import predict_from_text as slm_predict

                parsers.append(
                    (
                        "slm",
                        lambda text: slm_predict(
                            text,
                            weight_path=self._slm_weights,
                            timeout_s=self._slm_timeout_s,
                        ),
                    )
                )
            except Exception as exc:
                self._log(f"SEM_AUDIT unavailable parser=slm error={type(exc).__name__}:{exc}")

        for item in instructions:
            task = item.get("task", "?")
            text = str(item.get("instruction") or "")
            truth = _normalised_truth(item)
            for parser_name, predict in parsers:
                try:
                    prediction = predict(text)
                    differences = compare_prediction(prediction, truth)
                    state = "MATCH" if not differences else "DIFF"
                    self._log(
                        f"SEM_AUDIT task={task} parser={parser_name} state={state} "
                        f"details={','.join(differences) if differences else '-'}"
                    )
                except Exception as exc:  # no research failure may affect control
                    self._log(
                        f"SEM_AUDIT task={task} parser={parser_name} state=ERROR "
                        f"error={type(exc).__name__}:{exc}"
                    )
