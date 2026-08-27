"""Deterministic multi-critic utility for safe macro-action ranking."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from navigation.costmap import PathMetrics
from scheduler.candidate_generator import CandidateAction


@dataclass(frozen=True)
class UtilityWeights:
    """Weights for weighted critic contributions.

    Positive weights are used for both rewards and penalties; penalty critics
    negate their weighted feature.  Keeping weights non-negative makes the
    configured intent auditable.
    """

    expected_reward: float = 1.0
    success_probability: float = 5.0
    expected_time: float = 0.20
    path_length: float = 0.40
    obstacle_cost: float = 0.25
    dynamic_risk: float = 1.50
    heading_change: float = 0.10
    perception_uncertainty: float = 2.0
    manipulation_difficulty: float = 1.50
    irreversible_risk: float = 4.0
    recovery_cost: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"utility weight {name} must be finite and >= 0")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Auditable result of applying hard constraints and all critics."""

    candidate: CandidateAction
    valid: bool
    utility: float
    path_metrics: Optional[PathMetrics] = None
    critic_scores: Mapping[str, float] = field(default_factory=dict)
    rejection_reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "utility", float(self.utility))
        object.__setattr__(
            self,
            "critic_scores",
            MappingProxyType({
                str(name): float(value)
                for name, value in self.critic_scores.items()
            }),
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(reason) for reason in self.rejection_reasons),
        )
        if self.valid:
            if not math.isfinite(self.utility):
                raise ValueError("valid candidate utility must be finite")
            if self.rejection_reasons:
                raise ValueError("valid candidate cannot have rejection reasons")
        elif self.utility != float("-inf"):
            raise ValueError("invalid candidate utility must be -inf")

    @property
    def action_id(self) -> str:
        return self.candidate.action_id

    @property
    def action(self) -> CandidateAction:
        return self.candidate

    @property
    def safety_lower_bound(self) -> Optional[float]:
        """Conservative guard input; absent for non-navigation actions."""
        if self.path_metrics is None:
            return None
        if not self.valid or not self.path_metrics.reachable:
            return float("-inf")
        return self.path_metrics.min_clearance_m


# Name used by learning/replay adapters; both names have exactly one schema.
ScoredCandidate = CandidateEvaluation


_METADATA_HARD_KEYS = (
    "referee_allowed",
    "step_allowed",
    "collision_free",
    "resource_available",
    "ik_reachable",
    "irreversible_allowed",
    "payload_envelope_safe",
)


def _rejected(
    candidate: CandidateAction,
    reasons: Sequence[str],
    path_metrics: Optional[PathMetrics],
) -> CandidateEvaluation:
    # Stable de-duplication helps trace comparisons and makes tests deterministic.
    unique = tuple(dict.fromkeys(str(reason) for reason in reasons if reason))
    return CandidateEvaluation(
        candidate=candidate,
        valid=False,
        utility=float("-inf"),
        path_metrics=path_metrics,
        rejection_reasons=unique or ("candidate rejected",),
    )


def _hard_rejections(
    candidate: CandidateAction,
    constraints: Optional[Mapping[str, bool]],
) -> list[str]:
    reasons = []
    combined = dict(candidate.hard_constraints)
    nested = candidate.metadata.get("hard_constraints")
    if isinstance(nested, Mapping):
        combined.update({str(name): bool(value) for name, value in nested.items()})
    for name in _METADATA_HARD_KEYS:
        if name in candidate.metadata:
            combined[name] = bool(candidate.metadata[name])
    if constraints:
        combined.update({str(name): bool(value) for name, value in constraints.items()})
    for name in sorted(combined):
        if not combined[name]:
            reasons.append(f"hard constraint failed: {name}")
    return reasons


def evaluate_candidate(
    candidate: CandidateAction,
    *,
    weights: UtilityWeights = UtilityWeights(),
    path_metrics: Optional[PathMetrics] = None,
    hard_constraints: Optional[Mapping[str, bool]] = None,
    min_clearance_m: float = 0.0,
    require_path: bool = False,
) -> CandidateEvaluation:
    """Hard-filter and score one candidate without side effects.

    Any NaN/Inf is rejected before arithmetic, including a non-finite final
    sum caused by extreme-but-finite inputs.  Safety constraints never become
    tunable penalties: a single false constraint yields ``utility=-inf``.
    """
    if not isinstance(candidate, CandidateAction):
        raise TypeError("candidate must be CandidateAction")
    try:
        min_clearance_m = float(min_clearance_m)
    except (TypeError, ValueError):
        min_clearance_m = float("nan")
    reasons = _hard_rejections(candidate, hard_constraints)
    if not math.isfinite(min_clearance_m) or min_clearance_m < 0.0:
        reasons.append("minimum clearance threshold is invalid")

    feature_names = (
        "expected_score",
        "success_probability",
        "expected_time_s",
        "perception_uncertainty",
        "manipulation_difficulty",
        "irreversible_risk",
        "recovery_cost",
    )
    for name in feature_names:
        if not math.isfinite(float(getattr(candidate, name))):
            reasons.append(f"non-finite candidate feature: {name}")
    if math.isfinite(candidate.success_probability) and not 0.0 <= candidate.success_probability <= 1.0:
        reasons.append("success_probability outside [0, 1]")
    for name in (
        "expected_time_s",
        "perception_uncertainty",
        "manipulation_difficulty",
        "irreversible_risk",
        "recovery_cost",
    ):
        value = float(getattr(candidate, name))
        if math.isfinite(value) and value < 0.0:
            reasons.append(f"negative penalty feature: {name}")

    if candidate.is_navigation:
        if candidate.goal_pose is None:
            reasons.append("navigation candidate has no goal pose")
        elif not all(math.isfinite(value) for value in candidate.goal_pose):
            reasons.append("navigation goal pose is non-finite")
        if require_path and path_metrics is None:
            reasons.append("navigation candidate has no path evaluation")

    if path_metrics is not None:
        if not path_metrics.reachable:
            reasons.append(path_metrics.failure_reason or "path is unreachable")
        if not path_metrics.finite():
            reasons.append("path metrics contain NaN/Inf")
        if not path_metrics.carried_envelope_safe:
            reasons.append("carried envelope is unsafe")
        if (
            path_metrics.reachable
            and path_metrics.finite()
            and path_metrics.min_clearance_m < min_clearance_m
        ):
            reasons.append(
                f"path clearance {path_metrics.min_clearance_m:.3f} m below "
                f"required {min_clearance_m:.3f} m"
            )
    if reasons:
        return _rejected(candidate, reasons, path_metrics)

    path_length = path_metrics.path_length_m if path_metrics is not None else 0.0
    obstacle_cost = (
        path_metrics.inflation_cost_integral if path_metrics is not None else 0.0
    )
    dynamic_risk = path_metrics.dynamic_risk if path_metrics is not None else 0.0
    heading_change = path_metrics.heading_change_rad if path_metrics is not None else 0.0

    critics = {
        "expected_reward": weights.expected_reward * candidate.expected_score,
        "success_probability": weights.success_probability * candidate.success_probability,
        "expected_time": -weights.expected_time * candidate.expected_time_s,
        "path_length": -weights.path_length * path_length,
        "obstacle_cost": -weights.obstacle_cost * obstacle_cost,
        "dynamic_risk": -weights.dynamic_risk * dynamic_risk,
        "heading_change": -weights.heading_change * heading_change,
        "perception_uncertainty": (
            -weights.perception_uncertainty * candidate.perception_uncertainty
        ),
        "manipulation_difficulty": (
            -weights.manipulation_difficulty * candidate.manipulation_difficulty
        ),
        "irreversible_risk": (
            -weights.irreversible_risk * candidate.irreversible_risk
        ),
        "recovery_cost": -weights.recovery_cost * candidate.recovery_cost,
    }
    if not all(math.isfinite(value) for value in critics.values()):
        return _rejected(candidate, ["critic contribution is non-finite"], path_metrics)
    utility = math.fsum(critics.values())
    if not math.isfinite(utility):
        return _rejected(candidate, ["utility sum is non-finite"], path_metrics)
    return CandidateEvaluation(
        candidate=candidate,
        valid=True,
        utility=utility,
        path_metrics=path_metrics,
        critic_scores=critics,
    )


def rank_candidates(
    candidates: Sequence[CandidateAction],
    *,
    weights: UtilityWeights = UtilityWeights(),
    path_metrics_by_id: Optional[Mapping[str, PathMetrics]] = None,
    constraints_by_id: Optional[Mapping[str, Mapping[str, bool]]] = None,
    min_clearance_m: float = 0.0,
    require_path: bool = False,
) -> Tuple[CandidateEvaluation, ...]:
    """Return all evaluations best-first with a stable action-id tie break."""
    path_metrics_by_id = path_metrics_by_id or {}
    constraints_by_id = constraints_by_id or {}
    evaluations = [
        evaluate_candidate(
            candidate,
            weights=weights,
            path_metrics=path_metrics_by_id.get(candidate.action_id),
            hard_constraints=constraints_by_id.get(candidate.action_id),
            min_clearance_m=min_clearance_m,
            require_path=require_path,
        )
        for candidate in candidates
    ]
    evaluations.sort(
        key=lambda item: (
            0 if item.valid else 1,
            -item.utility if item.valid else 0.0,
            item.action_id,
        )
    )
    return tuple(evaluations)


__all__ = [
    "CandidateEvaluation",
    "ScoredCandidate",
    "UtilityWeights",
    "evaluate_candidate",
    "rank_candidates",
]
