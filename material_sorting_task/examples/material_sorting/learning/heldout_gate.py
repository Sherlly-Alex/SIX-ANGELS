"""Fail-closed held-out acceptance for the RL-1 scheduler candidate.

This module keeps the gate independent from SB3 so boundary tests can run in
the normal test environment.  The real evaluator supplies MaskablePPO policy
summaries produced by :func:`evaluate_policy`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class PolicySummary:
    episodes: int
    completed_episodes: int
    policy_errors: int
    masked_action_violations: int
    mean_return: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicySummary":
        result = cls(
            episodes=int(value["episodes"]),
            completed_episodes=int(value["completed_episodes"]),
            policy_errors=int(value["policy_errors"]),
            masked_action_violations=int(value["masked_action_violations"]),
            mean_return=float(value["mean_return"]),
        )
        if result.episodes <= 0 or not math.isfinite(result.mean_return):
            raise ValueError("invalid policy summary")
        return result

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_heldout_gate(
    validation: PolicySummary,
    test: PolicySummary,
    *,
    validation_baseline_return: float,
    test_baseline_return: float,
    validation_oracle_return: float,
    test_oracle_return: float,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Apply honest RL-1 criteria: safe, complete, oracle-near, baseline-safe."""

    if tolerance < 0 or not all(
        math.isfinite(float(item))
        for item in (
            validation_baseline_return,
            test_baseline_return,
            validation_oracle_return,
            test_oracle_return,
            tolerance,
        )
    ):
        raise ValueError("gate thresholds must be finite")
    failures: list[str] = []
    for name, summary, baseline, oracle in (
        ("validation", validation, validation_baseline_return, validation_oracle_return),
        ("test", test, test_baseline_return, test_oracle_return),
    ):
        if summary.completed_episodes != summary.episodes:
            failures.append(f"{name}: completed episodes do not equal requested episodes")
        if summary.policy_errors:
            failures.append(f"{name}: policy_errors={summary.policy_errors}")
        if summary.masked_action_violations:
            failures.append(f"{name}: masked_action_violations={summary.masked_action_violations}")
        # The oracle is an upper bound for this replay reward; exceeding it is
        # evidence of an evaluator/model mismatch, not a success claim.
        if summary.mean_return > oracle + tolerance:
            failures.append(f"{name}: return exceeds oracle bound")
        if summary.mean_return + tolerance < baseline:
            failures.append(f"{name}: return is below selected-action baseline")
    return {
        "passed": not failures,
        "failures": failures,
        "criteria": {
            "completed_equals_episodes": True,
            "policy_errors": 0,
            "masked_action_violations": 0,
            "oracle_tolerance": tolerance,
            "baseline_not_worse_than_tolerance": tolerance,
        },
        "validation": validation.to_json_dict(),
        "test": test.to_json_dict(),
        "baselines": {"validation": validation_baseline_return, "test": test_baseline_return},
        "oracles": {"validation": validation_oracle_return, "test": test_oracle_return},
    }


__all__ = ["PolicySummary", "assess_heldout_gate"]
