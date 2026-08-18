"""Real-time macro-action ranking with deterministic and guarded-RL modes.

The service selects only from caller-provided, finite candidates.  Collision,
resource and referee constraints are applied by the deterministic evaluator
before an optional learned policy sees an action mask.  It never publishes an
actuator command; executors remain the sole motion backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from navigation.costmap import WorldCostmap, WorldCostmapSnapshot
from scheduler.candidate_generator import CandidateAction
from scheduler.policies.heuristic import HeuristicPolicy
from scheduler.utility import CandidateEvaluation


@dataclass(frozen=True)
class DecisionConfig:
    policy_mode: str = "heuristic"
    minimum_action_hold_s: float = 0.75
    switch_utility_margin: float = 0.25
    candidate_stability_frames: int = 2
    max_candidates: int = 8
    rl_max_utility_regret: float = 0.25

    def __post_init__(self) -> None:
        mode = str(self.policy_mode).strip().casefold()
        if mode not in {"heuristic", "rl_shadow", "rl_guarded"}:
            raise ValueError(
                "policy_mode must be heuristic, rl_shadow, or rl_guarded"
            )
        object.__setattr__(self, "policy_mode", mode)
        for name in (
            "minimum_action_hold_s",
            "switch_utility_margin",
            "rl_max_utility_regret",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if int(self.candidate_stability_frames) <= 0:
            raise ValueError("candidate_stability_frames must be positive")
        if int(self.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive")
        object.__setattr__(
            self, "candidate_stability_frames", int(self.candidate_stability_frames)
        )
        object.__setattr__(self, "max_candidates", int(self.max_candidates))


@dataclass(frozen=True)
class DecisionOutcome:
    selected: CandidateEvaluation | None
    evaluations: tuple[CandidateEvaluation, ...]
    action_mask: tuple[bool, ...]
    source: str
    reason: str
    costmap_version: int | None
    policy_suggestion: CandidateEvaluation | None = None
    switched: bool = False
    policy_decision_reason: str | None = None
    policy_inference_ms: float | None = None
    policy_model_sha256: str | None = None

    @property
    def action_id(self) -> str | None:
        return None if self.selected is None else self.selected.action_id


class SchedulerDecisionService:
    """Rank safe alternatives and apply timeout/fallback/hysteresis guards."""

    def __init__(
        self,
        *,
        config: DecisionConfig | None = None,
        heuristic_policy: HeuristicPolicy | None = None,
        rl_policy: Any = None,
        policy_guard: Any = None,
        event_log: Any = None,
    ) -> None:
        self.config = config or DecisionConfig()
        self.heuristic_policy = heuristic_policy or HeuristicPolicy()
        self.rl_policy = rl_policy
        self._policy_guard = policy_guard
        self._event_log = event_log
        self._current_action_id: str | None = None
        self._selected_at_s: float | None = None
        self._pending_action_id: str | None = None
        self._pending_frames = 0

    def decide(
        self,
        candidates: Sequence[CandidateAction],
        *,
        now_s: float,
        world_state: Mapping[str, Any] | Any = None,
        costmap: WorldCostmap | WorldCostmapSnapshot | None = None,
        start_pose: Sequence[float] | Any = None,
        constraints: Mapping[str, bool] | Mapping[str, Mapping[str, bool]] | None = None,
        footprint_mode: Any = "transit_stowed",
        held_center_base: tuple[float, float, float] | None = None,
        held_half_width_m: float | None = None,
        force_switch: bool = False,
    ) -> DecisionOutcome:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        candidate_tuple = tuple(candidates)
        if len(candidate_tuple) > self.config.max_candidates:
            raise ValueError(
                f"received {len(candidate_tuple)} candidates; maximum is "
                f"{self.config.max_candidates}"
            )

        snapshot = (
            costmap.snapshot(now_s=now)
            if isinstance(costmap, WorldCostmap)
            else costmap
        )
        ranked = self.heuristic_policy.rank(
            candidate_tuple,
            costmap=snapshot,
            start_pose=start_pose,
            constraints=constraints,
            now_s=now,
            footprint_mode=footprint_mode,
            held_center_base=held_center_base,
            held_half_width_m=held_half_width_m,
        )
        # Learned action indices must retain CandidateGenerator's stable slot
        # order.  The deterministic policy may sort for ranking, so map its
        # evaluations back onto the original candidate order before building
        # the observation and mask.
        evaluated_by_id = {item.action_id: item for item in ranked}
        evaluations = tuple(
            evaluated_by_id[item.action_id] for item in candidate_tuple
        )
        heuristic_best = next((item for item in ranked if item.valid), None)
        mask = tuple(item.valid for item in evaluations) + (False,) * (
            self.config.max_candidates - len(evaluations)
        )
        version = None if snapshot is None else int(snapshot.version)
        self._emit_candidates(now, evaluations, mask, version, world_state)

        if heuristic_best is None:
            self._reset_selection()
            outcome = DecisionOutcome(
                selected=None,
                evaluations=evaluations,
                action_mask=mask,
                source="none",
                reason="no_safe_candidate",
                costmap_version=version,
            )
            self._emit_selection(now, outcome)
            return outcome

        (
            preferred,
            source,
            reason,
            suggestion,
            policy_reason,
            policy_inference_ms,
            policy_model_sha256,
        ) = self._policy_preference(
            evaluations,
            mask,
            heuristic_best,
            world_state,
        )
        if preferred is None or not preferred.valid:
            preferred = heuristic_best
            source = "heuristic"
            reason = "policy_returned_no_safe_candidate"
        if (
            source == "rl"
            and heuristic_best.utility - preferred.utility
            > self.config.rl_max_utility_regret
        ):
            preferred = heuristic_best
            source = "heuristic"
            reason = "rl_utility_regret_guard"

        selected, switched, hysteresis_reason = self._apply_hysteresis(
            preferred,
            evaluations,
            now,
            force_switch=force_switch,
        )
        if hysteresis_reason:
            reason = hysteresis_reason
            source = "hysteresis"
        outcome = DecisionOutcome(
            selected=selected,
            evaluations=evaluations,
            action_mask=mask,
            source=source,
            reason=reason,
            costmap_version=version,
            policy_suggestion=suggestion,
            switched=switched,
            policy_decision_reason=policy_reason,
            policy_inference_ms=policy_inference_ms,
            policy_model_sha256=policy_model_sha256,
        )
        self._emit_selection(now, outcome)
        return outcome

    def _policy_preference(
        self,
        evaluations: tuple[CandidateEvaluation, ...],
        mask: tuple[bool, ...],
        heuristic_best: CandidateEvaluation,
        world_state: Mapping[str, Any] | Any,
    ) -> tuple[
        CandidateEvaluation | None,
        str,
        str,
        CandidateEvaluation | None,
        str | None,
        float | None,
        str | None,
    ]:
        if self.config.policy_mode == "heuristic":
            return (
                heuristic_best,
                "heuristic",
                "deterministic_best",
                None,
                None,
                None,
                None,
            )

        try:
            from learning.observation import ObservationBuilder
            from scheduler.policies.guard import PolicyGuard

            if self._policy_guard is None:
                self._policy_guard = PolicyGuard()
            builder = ObservationBuilder(self.config.max_candidates)
            observation = builder.build(world_state or {}, evaluations, mask)
            decision = self._policy_guard.select(
                self.rl_policy,
                observation,
                candidates=evaluations,
                action_mask=mask,
                heuristic_best=heuristic_best,
            )
            suggestion = decision.selected
            if suggestion is not None and not isinstance(
                suggestion, CandidateEvaluation
            ):
                suggestion = None
            if self.config.policy_mode == "rl_shadow":
                return (
                    heuristic_best,
                    "heuristic",
                    f"rl_shadow:{decision.reason}",
                    suggestion,
                    decision.reason,
                    decision.inference_ms,
                    decision.model_sha256,
                )
            selected = suggestion if suggestion is not None else heuristic_best
            return (
                selected,
                decision.source,
                decision.reason,
                suggestion,
                decision.reason,
                decision.inference_ms,
                decision.model_sha256,
            )
        except Exception as exc:
            return (
                heuristic_best,
                "heuristic",
                f"rl_fallback:{type(exc).__name__}",
                None,
                f"policy_exception:{type(exc).__name__}",
                None,
                None,
            )

    def _apply_hysteresis(
        self,
        preferred: CandidateEvaluation,
        evaluations: tuple[CandidateEvaluation, ...],
        now_s: float,
        *,
        force_switch: bool,
    ) -> tuple[CandidateEvaluation, bool, str | None]:
        by_id = {item.action_id: item for item in evaluations}
        current = by_id.get(self._current_action_id)
        if current is None or not current.valid:
            previous = self._current_action_id
            self._commit(preferred, now_s)
            return preferred, previous != preferred.action_id, None
        if preferred.action_id == current.action_id:
            self._pending_action_id = None
            self._pending_frames = 0
            return current, False, None
        if not force_switch:
            held_for = (
                math.inf
                if self._selected_at_s is None
                else max(0.0, now_s - self._selected_at_s)
            )
            if held_for < self.config.minimum_action_hold_s:
                return current, False, "minimum_action_hold"
            if preferred.utility < current.utility + self.config.switch_utility_margin:
                return current, False, "switch_utility_margin"
            if self._pending_action_id == preferred.action_id:
                self._pending_frames += 1
            else:
                self._pending_action_id = preferred.action_id
                self._pending_frames = 1
            if self._pending_frames < self.config.candidate_stability_frames:
                return current, False, "candidate_not_stable"
        self._commit(preferred, now_s)
        return preferred, True, None

    def _commit(self, evaluation: CandidateEvaluation, now_s: float) -> None:
        self._current_action_id = evaluation.action_id
        self._selected_at_s = now_s
        self._pending_action_id = None
        self._pending_frames = 0

    def _reset_selection(self) -> None:
        self._current_action_id = None
        self._selected_at_s = None
        self._pending_action_id = None
        self._pending_frames = 0

    def _emit_candidates(
        self,
        now_s: float,
        evaluations: tuple[CandidateEvaluation, ...],
        action_mask: tuple[bool, ...],
        costmap_version: int | None,
        world_state: Mapping[str, Any] | Any,
    ) -> None:
        details = {
            "costmap_version": costmap_version,
            "candidates": [
                {
                    "action_id": item.action_id,
                    "valid": item.valid,
                    "utility": item.utility if math.isfinite(item.utility) else None,
                    "critics": dict(item.critic_scores),
                    "rejections": list(item.rejection_reasons),
                }
                for item in evaluations
            ],
        }
        # Persist the exact allow-listed vector seen by a future RL policy.
        # ObservationBuilder ignores unknown world-state keys, so neither
        # Server-private truth nor semantic-audit data can leak into replay.
        try:
            from learning.observation import (
                OBSERVATION_SCHEMA_VERSION,
                ObservationBuilder,
            )

            builder = ObservationBuilder(self.config.max_candidates)
            observation = builder.build(
                world_state or {}, evaluations, action_mask
            )
            details.update(
                {
                    "observation": observation.tolist(),
                    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                    "observation_schema_hash": builder.schema_hash,
                    "max_candidates": self.config.max_candidates,
                    "action_mask": list(action_mask),
                }
            )
        except Exception as exc:
            # Telemetry is audit-only. A serialization dependency failure may
            # make this sample ineligible for training but never changes the
            # deterministic action selected below.
            details["observation_error"] = type(exc).__name__
        self._emit("candidates_evaluated", now_s, details=details)

    def _emit_selection(self, now_s: float, outcome: DecisionOutcome) -> None:
        self._emit(
            "action_selected",
            now_s,
            action_id=outcome.action_id,
            message=outcome.reason,
            details={
                "source": outcome.source,
                "switched": outcome.switched,
                "costmap_version": outcome.costmap_version,
                "policy_suggestion": (
                    None
                    if outcome.policy_suggestion is None
                    else outcome.policy_suggestion.action_id
                ),
                "policy_decision_reason": outcome.policy_decision_reason,
                "policy_inference_ms": outcome.policy_inference_ms,
                "policy_model_sha256": outcome.policy_model_sha256,
            },
        )

    def _emit(self, event: str, now_s: float, **fields: Any) -> None:
        if self._event_log is None:
            return
        try:
            self._event_log.emit(event, timestamp_s=now_s, **fields)
        except Exception:
            # Audit failures cannot alter the selected safe action.
            pass

    def close(self) -> None:
        close = getattr(self._policy_guard, "close", None)
        if callable(close):
            close()


__all__ = [
    "DecisionConfig",
    "DecisionOutcome",
    "SchedulerDecisionService",
]
