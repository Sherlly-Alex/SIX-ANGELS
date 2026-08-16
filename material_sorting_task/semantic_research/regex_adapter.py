"""Adapt the formal Chinese text parser into the research SemanticPrediction."""

from __future__ import annotations

import sys
from pathlib import Path

from .schema import SemanticPrediction

_PARSER_NAME = "regex_text_adapter"

# Formal parser lives under examples/material_sorting; keep research isolated
# from client_task while still reusing the single dictionary implementation.
_FORMAL_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
if str(_FORMAL_DIR) not in sys.path:
    sys.path.insert(0, str(_FORMAL_DIR))


def predict_from_text(text: str) -> SemanticPrediction:
    """Map parse_instruction_text output to research slots. Never raises."""
    try:
        from instruction_parser import InstructionParseError, parse_instruction_text
    except Exception as exc:  # pragma: no cover - import environment issues
        return SemanticPrediction.failure(_PARSER_NAME, f"import_error: {exc}")

    try:
        task = parse_instruction_text(text)
    except InstructionParseError as exc:
        return SemanticPrediction.failure(_PARSER_NAME, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return SemanticPrediction.failure(_PARSER_NAME, f"unexpected: {exc}")

    errors = list(task.warnings)
    return SemanticPrediction(
        target_color=task.target_color,
        place_type=task.place_type,
        direction=task.direction,
        reference_kind=task.ref_prop,
        confidence=1.0 if task.semantic_valid else 0.5,
        parser_name=_PARSER_NAME,
        errors=tuple(errors),
    )
