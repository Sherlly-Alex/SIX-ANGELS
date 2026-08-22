#!/usr/bin/env python3
"""Validate one official-Client rl_guarded scheduler EventLog."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_POLICY_FAILURE_REASONS = frozenset(
    {
        "inference_timeout",
        "inference_timeout_quarantined",
        "policy_unavailable",
        "invalid_index",
        "masked_action",
        "policy_error",
    }
)


def _details(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("details")
    return value if isinstance(value, Mapping) else {}


def validate_guarded_log(
    path: str | Path,
    *,
    expected_model_sha256: str,
    minimum_rl_takeovers: int = 1,
    maximum_inference_p95_ms: float = 25.0,
) -> dict[str, Any]:
    if not _SHA256_RE.fullmatch(expected_model_sha256):
        raise ValueError("expected_model_sha256 must be 64 hexadecimal characters")
    if minimum_rl_takeovers <= 0:
        raise ValueError("minimum_rl_takeovers must be positive")
    if not math.isfinite(maximum_inference_p95_ms) or maximum_inference_p95_ms <= 0:
        raise ValueError("maximum_inference_p95_ms must be finite and positive")

    sessions = 0
    in_guarded = False
    sources: Counter[str] = Counter()
    policy_reasons: Counter[str] = Counter()
    model_hashes: set[str] = set()
    inference_samples: list[float] = []
    rl_takeovers = 0
    failures: list[str] = []

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        details = _details(event)
        if event.get("event_type") == "scheduler_started":
            in_guarded = details.get("policy_mode") == "rl_guarded"
            if in_guarded:
                sessions += 1
            continue
        if not in_guarded or event.get("event_type") != "action_selected":
            continue

        source = str(details.get("source", "missing"))
        sources[source] += 1
        reason_value = details.get("policy_decision_reason")
        reason = None if reason_value is None else str(reason_value)
        if reason is not None:
            policy_reasons[reason] += 1
        model_hash = details.get("policy_model_sha256")
        if isinstance(model_hash, str) and model_hash:
            model_hashes.add(model_hash.lower())

        if reason in _POLICY_FAILURE_REASONS:
            failures.append(f"line {line_number}: policy failure {reason}")
        elif reason not in {None, "accepted"}:
            failures.append(f"line {line_number}: unexpected policy reason {reason}")

        if reason == "accepted":
            try:
                inference_ms = float(details.get("policy_inference_ms"))
            except (TypeError, ValueError):
                inference_ms = math.nan
            if not math.isfinite(inference_ms) or inference_ms < 0:
                failures.append(f"line {line_number}: invalid inference time")
            else:
                inference_samples.append(inference_ms)

        if source != "rl":
            continue
        rl_takeovers += 1
        if reason != "accepted":
            failures.append(f"line {line_number}: RL source was not accepted")
        if str(details.get("policy_suggestion")) != str(event.get("action_id")):
            failures.append(
                f"line {line_number}: selected action differs from RL suggestion"
            )
        if model_hash != expected_model_sha256:
            failures.append(f"line {line_number}: wrong model SHA256")

    ordered = sorted(inference_samples)
    inference_p95_ms = (
        ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0
    )
    if sessions != 1:
        failures.append(f"guarded session count={sessions}, expected 1")
    if rl_takeovers < minimum_rl_takeovers:
        failures.append(
            f"rl_takeovers={rl_takeovers} below {minimum_rl_takeovers}"
        )
    if model_hashes != {expected_model_sha256.lower()}:
        failures.append(f"model hashes do not match: {sorted(model_hashes)}")
    if inference_p95_ms > maximum_inference_p95_ms:
        failures.append(
            f"inference p95={inference_p95_ms:.6f} ms exceeds "
            f"{maximum_inference_p95_ms:.6f} ms"
        )

    failures = list(dict.fromkeys(failures))
    return {
        "passed": not failures,
        "failures": failures,
        "guarded_sessions": sessions,
        "rl_takeovers": rl_takeovers,
        "source_counts": dict(sources),
        "policy_reason_counts": dict(policy_reasons),
        "model_sha256": sorted(model_hashes),
        "inference_sample_count": len(inference_samples),
        "inference_p95_ms": inference_p95_ms,
        "limits": {
            "minimum_rl_takeovers": minimum_rl_takeovers,
            "maximum_inference_p95_ms": maximum_inference_p95_ms,
            "policy_failures_allowed": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--minimum-rl-takeovers", type=int, default=1)
    parser.add_argument("--maximum-inference-p95-ms", type=float, default=25.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate_guarded_log(
        args.events,
        expected_model_sha256=args.expected_model_sha256,
        minimum_rl_takeovers=args.minimum_rl_takeovers,
        maximum_inference_p95_ms=args.maximum_inference_p95_ms,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
