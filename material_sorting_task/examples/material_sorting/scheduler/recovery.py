"""Deterministic, bounded recovery decisions for structured action failures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .models import ActionResult, ActionStatus, FailureCode, ScheduledAction


class RecoveryLevel(Enum):
    ACTION_LOCAL = 0
    STEP_LOCAL = 1
    RETREAT_AND_RETRY = 2
    ATTEMPT_END = 3
    SAFE_HOLD = 4


@dataclass(frozen=True)
class RecoveryPolicy:
    """Finite round-robin recovery sequence for one or more failures."""

    recoveries: tuple[str, ...]
    max_recoveries: int
    level: RecoveryLevel = RecoveryLevel.STEP_LOCAL

    def __post_init__(self) -> None:
        sequence = tuple(str(item).strip() for item in self.recoveries)
        if any(not item for item in sequence):
            raise ValueError("recovery names must be non-empty")
        if self.max_recoveries < 0:
            raise ValueError("max_recoveries cannot be negative")
        if self.max_recoveries > 0 and not sequence:
            raise ValueError("a positive recovery budget requires a recovery sequence")
        object.__setattr__(self, "recoveries", sequence)

    def action_for(self, recovery_count: int) -> str | None:
        if recovery_count < 0:
            raise ValueError("recovery_count cannot be negative")
        if recovery_count >= self.max_recoveries or not self.recoveries:
            return None
        return self.recoveries[recovery_count % len(self.recoveries)]


@dataclass(frozen=True)
class RecoveryDecision:
    failure_code: FailureCode
    recovery_would_help: bool
    next_recovery: str | None
    max_recoveries: int
    recovery_count: int
    level: RecoveryLevel
    exhausted: bool = False
    reason: str = ""

    @property
    def recovery_name(self) -> str | None:
        return self.next_recovery


DEFAULT_RECOVERY_POLICIES: Mapping[FailureCode, RecoveryPolicy] = MappingProxyType(
    {
        FailureCode.INPUT_STALE: RecoveryPolicy(("wait_for_fresh_input",), 1),
        FailureCode.TARGET_LOST: RecoveryPolicy(("stationary_rescan", "adjust_observation_stand"), 2),
        FailureCode.NAV_NO_PATH: RecoveryPolicy(("replan", "alternate_legal_stand"), 2),
        FailureCode.NAV_STUCK: RecoveryPolicy(("clear_local_path", "retreat_and_replan"), 2),
        FailureCode.ALIGNMENT_FAILED: RecoveryPolicy(("realign", "alternate_alignment"), 2),
        FailureCode.IK_FAILED: RecoveryPolicy(("replan_ik", "adjust_stand"), 2),
        FailureCode.SINGLE_SIDE_CONTACT: RecoveryPolicy(
            ("short_contact_wait", "one_millimetre_backoff"),
            2,
            RecoveryLevel.ACTION_LOCAL,
        ),
        FailureCode.GRASP_NOT_CONFIRMED: RecoveryPolicy(("reacquire_and_regrasp",), 1),
        FailureCode.EFFORT_SOFT_LIMIT: RecoveryPolicy(
            ("bounded_backoff",),
            1,
            RecoveryLevel.ACTION_LOCAL,
        ),
        FailureCode.PLACEMENT_UNCERTAIN: RecoveryPolicy(("verify_placement",), 1),
    }
)


# Structured failures that must escalate to SAFE_HOLD instead of recovery.
# BLOCKED results carrying one of these codes are fail-closed in the v2
# engine; every other code keeps the historical BLOCKED / referee semantics.
FATAL_SAFETY_FAILURE_CODES: frozenset[FailureCode] = frozenset(
    {
        FailureCode.EFFORT_HARD_LIMIT,
        FailureCode.UNSAFE_COLLISION,
        FailureCode.RESOURCE_CONFLICT,
        FailureCode.COMMAND_NON_FINITE,
        FailureCode.COMMAND_EXPIRED,
        FailureCode.REFEREE_DESYNC,
        FailureCode.INTERNAL_ERROR,
    }
)


class RecoveryClassifier:
    """Map a failure code and consumed budget to the next finite recovery."""

    def __init__(
        self,
        policies: Mapping[FailureCode, RecoveryPolicy] | None = None,
    ) -> None:
        self.policies = MappingProxyType(dict(DEFAULT_RECOVERY_POLICIES if policies is None else policies))

    def classify(
        self,
        failure_code: FailureCode | None,
        context: Any = None,
        *,
        recovery_count: int = 0,
    ) -> RecoveryDecision:
        del context  # Reserved for a future deterministic context predicate.
        code = failure_code or FailureCode.INTERNAL_ERROR
        policy = self.policies.get(code)
        if policy is None:
            return RecoveryDecision(
                code,
                False,
                None,
                0,
                recovery_count,
                RecoveryLevel.SAFE_HOLD,
                exhausted=False,
                reason=f"{code.value} has no safe recovery policy",
            )
        next_recovery = policy.action_for(recovery_count)
        exhausted = next_recovery is None
        return RecoveryDecision(
            code,
            not exhausted,
            next_recovery,
            policy.max_recoveries,
            recovery_count,
            policy.level if not exhausted else RecoveryLevel.ATTEMPT_END,
            exhausted=exhausted,
            reason=(
                f"recovery budget exhausted for {code.value}"
                if exhausted
                else f"start {next_recovery} for {code.value}"
            ),
        )


RecoveryFactory = Callable[[str], ScheduledAction | None]


@dataclass
class RecoverableStep:
    """Wrap an action with deterministic, bounded recovery accounting.

    A recovery factory is optional.  Without one, the wrapper returns a
    ``RUNNING`` result carrying ``recovery_requested`` metadata and retries the
    action on the next tick; this supports schedulers that execute recoveries
    externally.  With a factory, the returned recovery action is ticked until
    it succeeds before the original action is retried.
    """

    action: ScheduledAction
    classifier: RecoveryClassifier = field(default_factory=RecoveryClassifier)
    recovery_factory: RecoveryFactory | Mapping[str, Any] | None = None
    release_resources: Callable[[], Any] | None = None
    restart_action: Callable[[], Any] | None = None
    max_total_recoveries: int = 8
    recovery_count: int = 0
    active_recovery_name: str | None = field(default=None, init=False)
    active_recovery: ScheduledAction | None = field(default=None, init=False)
    _counts_by_failure: Counter[FailureCode] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        if self.max_total_recoveries < 0:
            raise ValueError("max_total_recoveries cannot be negative")

    def tick(self, context: Any) -> ActionResult:
        if self.active_recovery is not None:
            return self._tick_recovery(context)

        try:
            result = self.action.tick(context)
        except Exception as exc:
            return ActionResult.fatal(
                FailureCode.INTERNAL_ERROR,
                f"action raised during recoverable step: {exc}",
            )
        if result.status is not ActionStatus.RETRYABLE_FAILURE:
            self.active_recovery_name = None
            return result

        failure = result.failure_code or FailureCode.INTERNAL_ERROR
        used = self._counts_by_failure[failure]
        decision = self.classifier.classify(
            failure,
            context,
            recovery_count=used,
        )
        if self.recovery_count >= self.max_total_recoveries:
            decision = RecoveryDecision(
                failure,
                False,
                None,
                self.max_total_recoveries,
                used,
                RecoveryLevel.ATTEMPT_END,
                exhausted=True,
                reason="step-wide recovery budget exhausted",
            )
        if not decision.recovery_would_help or decision.next_recovery is None:
            return ActionResult.attempt_failed(
                FailureCode.RECOVERY_EXHAUSTED,
                decision.reason,
                metadata={
                    "original_failure": failure.value,
                    "recovery_count": self.recovery_count,
                },
            )

        try:
            self.action.cancel("entering recovery")
            if self.release_resources is not None:
                self.release_resources()
        except Exception as exc:
            return ActionResult.fatal(
                FailureCode.INTERNAL_ERROR,
                f"failed to enter recovery safely: {exc}",
            )

        self.recovery_count += 1
        self._counts_by_failure[failure] += 1
        self.active_recovery_name = decision.next_recovery
        try:
            self.active_recovery = self._make_recovery(decision.next_recovery)
        except Exception as exc:
            self.active_recovery_name = None
            return ActionResult.fatal(
                FailureCode.INTERNAL_ERROR,
                f"failed to build recovery {decision.next_recovery!r}: {exc}",
            )
        return ActionResult.running(
            decision.reason,
            metadata={
                "recovery_requested": decision.next_recovery,
                "recovery_count": self.recovery_count,
                "original_failure": failure.value,
            },
        )

    def _make_recovery(self, name: str) -> ScheduledAction | None:
        factory = self.recovery_factory
        if factory is None:
            return None
        if isinstance(factory, Mapping):
            entry = factory.get(name)
            if entry is None:
                return None
            return entry() if callable(entry) and not hasattr(entry, "tick") else entry
        return factory(name)

    def _tick_recovery(self, context: Any) -> ActionResult:
        recovery = self.active_recovery
        assert recovery is not None
        try:
            result = recovery.tick(context)
        except Exception as exc:
            return self._recovery_failed(f"recovery raised: {exc}")
        if result.status is ActionStatus.RUNNING:
            return result
        if result.status is not ActionStatus.SUCCEEDED:
            return self._recovery_failed(
                f"recovery {self.active_recovery_name!r} failed: {result.message}"
            )

        completed = self.active_recovery_name
        self.active_recovery = None
        self.active_recovery_name = None
        try:
            if self.restart_action is not None:
                self.restart_action()
        except Exception as exc:
            return ActionResult.fatal(
                FailureCode.INTERNAL_ERROR,
                f"failed to restart action after recovery: {exc}",
            )
        return ActionResult.running(
            f"recovery {completed!r} completed; retrying step",
            metadata={"recovery_completed": completed},
        )

    def _recovery_failed(self, message: str) -> ActionResult:
        self.active_recovery = None
        self.active_recovery_name = None
        return ActionResult.attempt_failed(
            FailureCode.RECOVERY_EXHAUSTED,
            message,
            metadata={"recovery_count": self.recovery_count},
        )

    def cancel(self, reason: str) -> None:
        if self.active_recovery is not None:
            self.active_recovery.cancel(reason)
        self.action.cancel(reason)
        self.active_recovery = None
        self.active_recovery_name = None

    def reset(self) -> None:
        self.recovery_count = 0
        self.active_recovery = None
        self.active_recovery_name = None
        self._counts_by_failure.clear()
        reset = getattr(self.action, "reset", None)
        if callable(reset):
            reset()


__all__ = [
    "DEFAULT_RECOVERY_POLICIES",
    "FATAL_SAFETY_FAILURE_CODES",
    "RecoverableStep",
    "RecoveryClassifier",
    "RecoveryDecision",
    "RecoveryLevel",
    "RecoveryPolicy",
]
