"""Safety guard that makes learned scheduling strictly fail-closed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from learning.action_mask import (
    InvalidActionMask,
    candidate_is_selectable,
    validate_action_mask,
)
from learning.action_space import candidate_action_id

from .rl import PolicyPrediction, RLPolicy


@dataclass(frozen=True)
class PolicyGuardConfig:
    inference_timeout_s: float = 0.025
    minimum_safety_lower_bound: float = 0.0
    require_safety_lower_bound: bool = False
    quarantine_after_timeout: bool = True
    consecutive_timeouts_before_quarantine: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.inference_timeout_s) or self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        if not math.isfinite(self.minimum_safety_lower_bound):
            raise ValueError("minimum_safety_lower_bound must be finite")
        if (
            isinstance(self.consecutive_timeouts_before_quarantine, bool)
            or not isinstance(self.consecutive_timeouts_before_quarantine, int)
            or self.consecutive_timeouts_before_quarantine <= 0
        ):
            raise ValueError(
                "consecutive_timeouts_before_quarantine must be a positive integer"
            )


@dataclass(frozen=True)
class GuardDecision:
    selected: Any | None
    action_index: int | None
    source: str
    reason: str
    inference_ms: float = 0.0
    rl_action_index: int | None = None
    model_sha256: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.source != "rl"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _action_index(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, PolicyPrediction):
        value = value.action_index
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "action_index"):
        value = value.action_index
    array = np.asarray(value)
    if array.size != 1:
        return None, "non_scalar_action"
    try:
        scalar_value = array.reshape(-1)[0]
        if isinstance(scalar_value, (bool, np.bool_)):
            return None, "boolean_action"
        scalar = float(scalar_value)
    except (TypeError, ValueError, OverflowError):
        return None, "non_numeric_action"
    if not math.isfinite(scalar):
        return None, "non_finite_action"
    index = int(scalar)
    if scalar != float(index):
        return None, "non_integral_action"
    return index, None


class PolicyGuard:
    """Run an optional policy under timeout, mask and safety-floor checks."""

    def __init__(self, config: PolicyGuardConfig | None = None) -> None:
        self.config = config or PolicyGuardConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scheduler-rl-policy"
        )
        self._quarantined_reason: str | None = None
        self._consecutive_faults = 0

    @property
    def quarantined(self) -> bool:
        return self._quarantined_reason is not None

    def reset_quarantine(self) -> None:
        self._quarantined_reason = None
        self._consecutive_faults = 0

    def _record_fault(self, reason: str = "inference_timeout_quarantined") -> None:
        self._consecutive_faults += 1
        if (
            self.config.quarantine_after_timeout
            and self._consecutive_faults
            >= self.config.consecutive_timeouts_before_quarantine
        ):
            self._quarantined_reason = reason

    def _record_timeout(self) -> None:
        self._record_fault("inference_timeout_quarantined")

    def _clear_faults(self) -> None:
        self._consecutive_faults = 0

    @staticmethod
    def _find_heuristic_index(
        candidates: Sequence[Any], heuristic_best: Any | None, mask: np.ndarray | None
    ) -> int | None:
        if isinstance(heuristic_best, (int, np.integer)) and not isinstance(
            heuristic_best, (bool, np.bool_)
        ):
            index = int(heuristic_best)
            if 0 <= index < len(candidates) and (
                mask is None or bool(mask[index])
            ):
                return index
        if heuristic_best is not None:
            try:
                wanted = candidate_action_id(heuristic_best)
            except ValueError:
                wanted = None
            if wanted is not None:
                for index, candidate in enumerate(candidates):
                    try:
                        matches = candidate_action_id(candidate) == wanted
                    except ValueError:
                        matches = False
                    if matches and (mask is None or bool(mask[index])):
                        return index
        best_index: int | None = None
        best_utility = -math.inf
        for index, candidate in enumerate(candidates):
            if mask is not None and not bool(mask[index]):
                continue
            if not candidate_is_selectable(candidate):
                continue
            utility = _field(candidate, "utility", 0.0)
            try:
                utility_value = float(utility)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(utility_value) and utility_value > best_utility:
                best_index = index
                best_utility = utility_value
        return best_index

    def _fallback(
        self,
        reason: str,
        candidates: Sequence[Any],
        heuristic_best: Any | None,
        mask: np.ndarray | None,
        *,
        inference_ms: float = 0.0,
        rl_action_index: int | None = None,
        model_sha256: str | None = None,
    ) -> GuardDecision:
        index = self._find_heuristic_index(candidates, heuristic_best, mask)
        selected = candidates[index] if index is not None else heuristic_best
        return GuardDecision(
            selected=selected,
            action_index=index,
            source="heuristic",
            reason=reason,
            inference_ms=float(inference_ms),
            rl_action_index=rl_action_index,
            model_sha256=model_sha256,
        )

    def _safety_value(self, candidate: Any) -> float | None:
        direct = _field(candidate, "safety_lower_bound", None)
        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError, OverflowError):
                return math.nan
        critics = _field(candidate, "critic_scores", {}) or {}
        if isinstance(critics, Mapping):
            for key in ("safety_lower_bound", "safety_margin", "min_clearance"):
                if key in critics:
                    try:
                        return float(critics[key])
                    except (TypeError, ValueError, OverflowError):
                        return math.nan
        return None

    @staticmethod
    def _raw_metric(candidate: Any, *names: str) -> float | None:
        inner = _field(candidate, "candidate", candidate)
        sources = (candidate, inner, _field(candidate, "path_metrics", None))
        for name in names:
            for source in sources:
                if source is None:
                    continue
                value = _field(source, name, None)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    return math.nan
                return number if math.isfinite(number) else math.nan
        return None

    def _prefer_heuristic_over_rl(
        self, selected: Any, heuristic_best: Any | None
    ) -> str | None:
        """Reject an RL action that is less successful, riskier, or weaker."""

        if heuristic_best is None:
            return None
        selected_success = self._raw_metric(selected, "success_probability")
        heuristic_success = self._raw_metric(heuristic_best, "success_probability")
        if (
            selected_success is not None
            and heuristic_success is not None
            and math.isfinite(selected_success)
            and math.isfinite(heuristic_success)
            and selected_success < heuristic_success
        ):
            return "rl_success_probability_guard"
        selected_dynamic = self._raw_metric(selected, "dynamic_risk")
        heuristic_dynamic = self._raw_metric(heuristic_best, "dynamic_risk")
        if (
            selected_dynamic is not None
            and heuristic_dynamic is not None
            and math.isfinite(selected_dynamic)
            and math.isfinite(heuristic_dynamic)
            and selected_dynamic > heuristic_dynamic
        ):
            return "rl_dynamic_risk_guard"
        selected_irreversible = self._raw_metric(selected, "irreversible_risk")
        heuristic_irreversible = self._raw_metric(heuristic_best, "irreversible_risk")
        if (
            selected_irreversible is not None
            and heuristic_irreversible is not None
            and math.isfinite(selected_irreversible)
            and math.isfinite(heuristic_irreversible)
            and selected_irreversible > heuristic_irreversible
        ):
            return "rl_irreversible_risk_guard"
        return None

    def accept_or_fallback(
        self,
        rl_action: Any,
        heuristic_best: Any | None,
        *,
        candidates: Sequence[Any],
        action_mask: Sequence[bool],
        safety_lower_bounds: Sequence[float] | None = None,
        inference_ms: float = 0.0,
    ) -> GuardDecision:
        """Validate a completed inference result and select or fall back."""

        try:
            mask = validate_action_mask(
                action_mask, len(action_mask), require_any=True
            )
        except InvalidActionMask:
            return self._fallback(
                "invalid_action_mask", candidates, heuristic_best, None
            )
        if len(candidates) > mask.size or bool(np.any(mask[len(candidates) :])):
            return self._fallback(
                "mask_enables_empty_slot", candidates, heuristic_best, None
            )
        index, error = _action_index(rl_action)
        if error is not None:
            self._record_fault("policy_error_quarantined")
            return self._fallback(
                error,
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
            )
        assert index is not None
        model_sha256 = _field(rl_action, "model_sha256", None)
        if model_sha256 is not None:
            model_sha256 = str(model_sha256)
        if not 0 <= index < len(candidates):
            self._record_fault("policy_error_quarantined")
            return self._fallback(
                "action_out_of_range",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        if not bool(mask[index]):
            self._record_fault("policy_error_quarantined")
            return self._fallback(
                "masked_action",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        candidate = candidates[index]
        if not candidate_is_selectable(candidate):
            self._record_fault("policy_error_quarantined")
            return self._fallback(
                "candidate_invalid",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        if safety_lower_bounds is not None:
            try:
                bounds = np.asarray(safety_lower_bounds, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                bounds = np.asarray([], dtype=np.float64)
            safety_value = (
                float(bounds[index]) if bounds.shape == mask.shape else math.nan
            )
        else:
            safety_value = self._safety_value(candidate)
        if safety_value is None and self.config.require_safety_lower_bound:
            return self._fallback(
                "safety_lower_bound_missing",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        if safety_value is not None and (
            not math.isfinite(safety_value)
            or safety_value < self.config.minimum_safety_lower_bound
        ):
            return self._fallback(
                "safety_lower_bound",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        comparison_reason = self._prefer_heuristic_over_rl(candidate, heuristic_best)
        if comparison_reason is not None:
            return self._fallback(
                comparison_reason,
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
                rl_action_index=index,
                model_sha256=model_sha256,
            )
        self._clear_faults()
        return GuardDecision(
            selected=candidate,
            action_index=index,
            source="rl",
            reason="accepted",
            inference_ms=float(inference_ms),
            rl_action_index=index,
            model_sha256=model_sha256,
        )

    def select(
        self,
        policy: RLPolicy | Any | None,
        observation: Sequence[float] | np.ndarray,
        *,
        candidates: Sequence[Any],
        action_mask: Sequence[bool],
        heuristic_best: Any | None = None,
        safety_lower_bounds: Sequence[float] | None = None,
        deterministic: bool = True,
    ) -> GuardDecision:
        """Run policy inference with a strict wall-clock deadline."""

        try:
            mask = validate_action_mask(
                action_mask, len(action_mask), require_any=True
            )
        except InvalidActionMask:
            return self._fallback(
                "invalid_action_mask", candidates, heuristic_best, None
            )
        if len(candidates) > mask.size or bool(np.any(mask[len(candidates) :])):
            return self._fallback(
                "mask_enables_empty_slot", candidates, heuristic_best, None
            )
        if policy is None or (
            isinstance(policy, RLPolicy) and not policy.configured
        ):
            return self._fallback(
                "model_missing", candidates, heuristic_best, mask
            )
        if self._quarantined_reason is not None:
            return self._fallback(
                self._quarantined_reason, candidates, heuristic_best, mask
            )

        started = time.perf_counter()
        future = self._executor.submit(
            policy.predict,
            np.asarray(observation, dtype=np.float32),
            action_masks=mask,
            deterministic=bool(deterministic),
        )
        try:
            prediction = future.result(timeout=self.config.inference_timeout_s)
        except FutureTimeout:
            future.cancel()
            self._record_timeout()
            return self._fallback(
                "inference_timeout",
                candidates,
                heuristic_best,
                mask,
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            self._record_fault("policy_error_quarantined")
            message = str(exc).lower()
            reason = (
                "non_finite_action"
                if "nan" in message or "infinite" in message
                else "policy_error"
            )
            return self._fallback(
                reason,
                candidates,
                heuristic_best,
                mask,
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        reported_ms = _field(prediction, "inference_ms", elapsed_ms)
        try:
            inference_ms = max(elapsed_ms, float(reported_ms))
        except (TypeError, ValueError, OverflowError):
            inference_ms = elapsed_ms
        if inference_ms > self.config.inference_timeout_s * 1000.0:
            self._record_timeout()
            return self._fallback(
                "inference_timeout",
                candidates,
                heuristic_best,
                mask,
                inference_ms=inference_ms,
            )
        return self.accept_or_fallback(
            prediction,
            heuristic_best,
            candidates=candidates,
            action_mask=mask,
            safety_lower_bounds=safety_lower_bounds,
            inference_ms=inference_ms,
        )

    choose = select

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "GuardDecision",
    "PolicyGuard",
    "PolicyGuardConfig",
]
