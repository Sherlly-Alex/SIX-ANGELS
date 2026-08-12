"""Research-only prediction schema for text-observable slots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


COMPARABLE_SLOTS: tuple[str, ...] = (
    "target_color",
    "place_type",
    "direction",
    "reference_kind",
)


@dataclass(frozen=True)
class SemanticPrediction:
    """Slots that can be observed from Chinese instruction text.

    Intentionally omits target_body / place_world / place_radius — those are
    Server execution ground truth and must not be guessed by offline parsers.
    """

    target_color: str | None
    place_type: str | None
    direction: str | None
    reference_kind: str | None
    confidence: float | None
    parser_name: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticPrediction":
        errors = data.get("errors") or ()
        return cls(
            target_color=data.get("target_color"),
            place_type=data.get("place_type"),
            direction=data.get("direction"),
            reference_kind=data.get("reference_kind"),
            confidence=data.get("confidence"),
            parser_name=str(data.get("parser_name") or "unknown"),
            errors=tuple(errors),
        )

    @classmethod
    def failure(cls, parser_name: str, *errors: str) -> "SemanticPrediction":
        return cls(
            target_color=None,
            place_type=None,
            direction=None,
            reference_kind=None,
            confidence=None,
            parser_name=parser_name,
            errors=tuple(errors),
        )
