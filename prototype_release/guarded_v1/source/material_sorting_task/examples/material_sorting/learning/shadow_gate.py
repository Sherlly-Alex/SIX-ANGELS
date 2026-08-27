"""Offline acceptance gate for RL-shadow scheduler EventLogs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .event_replay import replay_event_logs


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# These are deliberate policy-quality or safety checks.  In shadow mode they
# mean "do not offer an RL suggestion for this decision", not that the model
# or runtime failed.  Keep the list explicit and fail closed for every other
# reason so a new model/runtime fault cannot be hidden by the audit gate.
_SAFETY_ABSTAIN_REASONS = frozenset(
    {
        "rl_dynamic_risk_guard",
        "rl_success_probability_guard",
        "rl_irreversible_risk_guard",
        "rl_utility_regret_guard",
        "safety_lower_bound",
        "safety_lower_bound_missing",
    }
)


# Expected non-blocking Shadow states. They are visible in audit output but
# are not worker/model failures: Heuristic remains the executed policy.
_ASYNC_ABSTAIN_REASONS = frozenset(
    {
        "submitted",
        "inference_pending",
        "inference_result_stale",
        "inference_result_expired",
    }
)


@dataclass(frozen=True)
class ShadowGateSummary:
    files: int
    shadow_sessions: int
    eligible_decisions: int
    suggestion_count: int
    agreement_count: int
    disagreement_count: int
    fallback_count: int
    fallback_rate: float
    runtime_fallback_count: int
    runtime_fallback_rate: float
    safety_abstain_count: int
    safety_abstain_rate: float
    async_abstain_count: int
    async_abstain_rate: float
    runtime_fallback_reasons: Mapping[str, int]
    safety_abstain_reasons: Mapping[str, int]
    async_abstain_reasons: Mapping[str, int]
    masked_suggestion_violations: int
    actual_rl_takeovers: int
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    inference_max_ms: float
    model_sha256: tuple[str, ...]
    replay_training_ready_decisions: int
    limits: Mapping[str, float | int]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["model_sha256"] = list(self.model_sha256)
        value["failures"] = list(self.failures)
        return value


def _details(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("details")
    return value if isinstance(value, Mapping) else {}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def validate_rl_shadow(
    paths: Sequence[str | Path],
    *,
    min_suggestions: int = 1000,
    max_inference_p95_ms: float = 25.0,
    max_fallback_rate: float = 0.01,
    expected_model_sha256: str | None = None,
) -> ShadowGateSummary:
    """Prove that RL suggested only safe slots and never controlled output."""

    if min_suggestions <= 0:
        raise ValueError("min_suggestions must be positive")
    if not math.isfinite(max_inference_p95_ms) or max_inference_p95_ms <= 0.0:
        raise ValueError("max_inference_p95_ms must be finite and positive")
    if not math.isfinite(max_fallback_rate) or not 0.0 <= max_fallback_rate <= 1.0:
        raise ValueError("max_fallback_rate must lie in [0, 1]")
    if expected_model_sha256 is not None and not _SHA256_RE.fullmatch(
        expected_model_sha256
    ):
        raise ValueError("expected_model_sha256 must be 64 hexadecimal characters")

    replay, _ = replay_event_logs(
        paths,
        min_decisions=min_suggestions,
        require_training_ready=True,
    )
    failures = [f"replay: {item}" for item in replay.failures]
    shadow_sessions = 0
    eligible = 0
    suggestions = 0
    agreements = 0
    disagreements = 0
    runtime_fallbacks = 0
    safety_abstains = 0
    async_abstains = 0
    runtime_fallback_reasons: dict[str, int] = {}
    safety_abstain_reasons: dict[str, int] = {}
    async_abstain_reasons: dict[str, int] = {}
    masked_violations = 0
    actual_takeovers = 0
    inference_samples: list[float] = []
    model_hashes: set[str] = set()

    for raw_path in paths:
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            failures.append(f"{path}: {type(exc).__name__}")
            continue
        in_shadow_session = False
        pending: Mapping[str, Any] | None = None
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # replay gate already reports malformed input
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("event_type")
            if event_type == "scheduler_started":
                in_shadow_session = _details(event).get("policy_mode") == "rl_shadow"
                if in_shadow_session:
                    shadow_sessions += 1
                pending = None
                continue
            if not in_shadow_session:
                continue
            if event_type == "candidates_evaluated":
                pending = event
                continue
            if event_type != "action_selected" or pending is None:
                continue

            candidate_values = _details(pending).get("candidates")
            pending = None
            if not isinstance(candidate_values, list):
                continue  # replay gate reports the structural error
            valid_ids = {
                str(item.get("action_id"))
                for item in candidate_values
                if isinstance(item, Mapping) and item.get("valid") is True
            }
            if not valid_ids:
                continue
            eligible += 1
            event_details = _details(event)
            actual_source = str(event_details.get("source", ""))
            if actual_source == "rl":
                actual_takeovers += 1
                failures.append(
                    f"{path}:{line_number}: RL controlled an rl_shadow selection"
                )
            elif actual_source not in {"heuristic", "hysteresis"}:
                failures.append(
                    f"{path}:{line_number}: unexpected actual source {actual_source!r}"
                )

            suggestion = event_details.get("policy_suggestion")
            policy_reason = event_details.get("policy_decision_reason")
            if suggestion is None:
                reason_key = str(policy_reason or "missing_policy_reason")
                if reason_key in _SAFETY_ABSTAIN_REASONS:
                    safety_abstains += 1
                    safety_abstain_reasons[reason_key] = (
                        safety_abstain_reasons.get(reason_key, 0) + 1
                    )
                elif reason_key in _ASYNC_ABSTAIN_REASONS:
                    async_abstains += 1
                    async_abstain_reasons[reason_key] = (
                        async_abstain_reasons.get(reason_key, 0) + 1
                    )
                else:
                    runtime_fallbacks += 1
                    runtime_fallback_reasons[reason_key] = (
                        runtime_fallback_reasons.get(reason_key, 0) + 1
                    )
                if policy_reason == "accepted":
                    failures.append(
                        f"{path}:{line_number}: accepted policy has no suggestion"
                    )
                continue
            suggestion_id = str(suggestion)
            if suggestion_id not in valid_ids:
                masked_violations += 1
                failures.append(
                    f"{path}:{line_number}: policy suggestion is masked or absent"
                )
                continue
            if policy_reason != "accepted":
                failures.append(
                    f"{path}:{line_number}: suggestion reason is {policy_reason!r}"
                )
                continue
            try:
                inference_ms = float(event_details.get("policy_inference_ms"))
            except (TypeError, ValueError, OverflowError):
                inference_ms = math.nan
            if not math.isfinite(inference_ms) or inference_ms < 0.0:
                failures.append(
                    f"{path}:{line_number}: policy inference time is invalid"
                )
                continue
            model_hash = event_details.get("policy_model_sha256")
            if not isinstance(model_hash, str) or not _SHA256_RE.fullmatch(model_hash):
                failures.append(
                    f"{path}:{line_number}: policy model SHA256 is missing or invalid"
                )
                continue
            suggestions += 1
            inference_samples.append(inference_ms)
            model_hashes.add(model_hash.lower())
            if suggestion_id == event.get("action_id"):
                agreements += 1
            else:
                disagreements += 1

    runtime_fallback_rate = runtime_fallbacks / max(1, eligible)
    safety_abstain_rate = safety_abstains / max(1, eligible)
    async_abstain_rate = async_abstains / max(1, eligible)
    inference_p95 = _percentile(inference_samples, 0.95)
    if shadow_sessions == 0:
        failures.append("no scheduler session declares policy_mode=rl_shadow")
    if suggestions < min_suggestions:
        failures.append(f"suggestion_count={suggestions} below {min_suggestions}")
    if runtime_fallback_rate > max_fallback_rate:
        failures.append(
            "runtime_fallback_rate="
            f"{runtime_fallback_rate:.6f} exceeds {max_fallback_rate:.6f}"
        )
    if inference_p95 > max_inference_p95_ms:
        failures.append(
            f"inference_p95_ms={inference_p95:.6f} exceeds "
            f"{max_inference_p95_ms:.6f}"
        )
    if len(model_hashes) != 1:
        failures.append(f"model_hash_count={len(model_hashes)}, expected 1")
    if expected_model_sha256 is not None and model_hashes != {
        expected_model_sha256.lower()
    }:
        failures.append("observed model SHA256 does not match the approved model")

    return ShadowGateSummary(
        files=len(paths),
        shadow_sessions=shadow_sessions,
        eligible_decisions=eligible,
        suggestion_count=suggestions,
        agreement_count=agreements,
        disagreement_count=disagreements,
        # Retain the old keys as aliases for backward-compatible callers.  A
        # fallback now deliberately means a model/runtime failure; safety
        # abstentions are reported separately and never authorize control.
        fallback_count=runtime_fallbacks,
        fallback_rate=runtime_fallback_rate,
        runtime_fallback_count=runtime_fallbacks,
        runtime_fallback_rate=runtime_fallback_rate,
        safety_abstain_count=safety_abstains,
        safety_abstain_rate=safety_abstain_rate,
        async_abstain_count=async_abstains,
        async_abstain_rate=async_abstain_rate,
        runtime_fallback_reasons=dict(sorted(runtime_fallback_reasons.items())),
        safety_abstain_reasons=dict(sorted(safety_abstain_reasons.items())),
        async_abstain_reasons=dict(sorted(async_abstain_reasons.items())),
        masked_suggestion_violations=masked_violations,
        actual_rl_takeovers=actual_takeovers,
        inference_p50_ms=_percentile(inference_samples, 0.50),
        inference_p95_ms=inference_p95,
        inference_p99_ms=_percentile(inference_samples, 0.99),
        inference_max_ms=max(inference_samples, default=0.0),
        model_sha256=tuple(sorted(model_hashes)),
        replay_training_ready_decisions=replay.training_ready_decisions,
        limits={
            "min_suggestions": min_suggestions,
            "max_inference_p95_ms": max_inference_p95_ms,
            "max_runtime_fallback_rate": max_fallback_rate,
        },
        failures=tuple(dict.fromkeys(failures)),
    )


__all__ = ["ShadowGateSummary", "validate_rl_shadow"]
