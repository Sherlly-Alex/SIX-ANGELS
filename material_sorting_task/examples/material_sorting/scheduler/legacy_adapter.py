"""Adapters from existing task executors to scheduler-v2 actions.

``LegacyStageAction`` exposes one validated executor stage through the
enter/tick/cancel action contract.  ``RecoverableStageAction`` adds the
bounded, deterministic recovery layer that the v2 engine uses once an
executor starts reporting structured ``FailureCode`` values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from executors.base import ExecutionContext, StageResult, StageStatus, TaskExecutor, TaskStage
from scheduler.models import FailureCode
from scheduler.recovery import (
    FATAL_SAFETY_FAILURE_CODES,
    RecoveryClassifier,
    RecoveryDecision,
    RecoveryLevel,
)


@dataclass
class LegacyStageAction:
    """Expose one legacy stage through an enter/tick/cancel action contract."""

    executor: TaskExecutor
    stage: TaskStage
    entered: bool = False

    def enter(self, context: ExecutionContext) -> None:
        if self.entered:
            return
        self.executor.enter_stage(self.stage, context)
        self.entered = True

    def tick(self, context: ExecutionContext) -> StageResult:
        if not self.entered:
            raise RuntimeError(f"stage {self.stage.value} ticked before enter")
        return self.executor.tick(self.stage, context)

    def cancel(self, reason: str) -> None:
        self.executor.cancel(reason)
        self.entered = False


RecoveryActionFactory = Callable[[str], Any | None]


@dataclass
class RecoverableStageAction:
    """Bounded, deterministic recovery around one legacy stage action.

    This is the StageResult counterpart of scheduler.recovery.RecoverableStep:
    it reuses the same finite round-robin ``RecoveryClassifier`` policy
    table but speaks the executor native enter/tick/cancel contract, so the
    engine can keep publishing arm holds and zero base velocity throughout
    a recovery.

    Without a recovery factory the selected recovery is a bounded step
    re-entry: the inner action is cancelled, the stage is re-entered on the
    following tick and the executor replans from its own entry logic (Nav2
    level L2 semantics).  Executors that provide finer-grained recovery
    motions expose ``build_recovery_action(name)``; the engine passes that
    callable here.  Irreversible steps must never be wrapped.
    """

    action: LegacyStageAction
    classifier: RecoveryClassifier = field(default_factory=RecoveryClassifier)
    recovery_factory: RecoveryActionFactory | None = None
    restart_action: Callable[[], None] | None = None
    max_total_recoveries: int = 8

    _recovery_count: int = field(default=0, init=False)
    _counts_by_failure: Counter[FailureCode] = field(default_factory=Counter, init=False)
    _active_recovery_name: str | None = field(default=None, init=False)
    _active_recovery: Any = field(default=None, init=False)
    _restart_pending: bool = field(default=False, init=False)
    _restart_note: str = field(default="", init=False)
    _context: ExecutionContext | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.max_total_recoveries < 0:
            raise ValueError("max_total_recoveries cannot be negative")

    def enter(self, context: ExecutionContext) -> None:
        self._context = context
        self.action.enter(context)

    def tick(self, context: ExecutionContext) -> StageResult:
        self._context = context
        if self._restart_pending:
            self._restart_pending = False
            self._restart_inner()
            return StageResult.running(
                f"bounded recovery re-entered the step: {self._restart_note}",
                metadata={"recovery_completed": self._restart_note},
            )
        if self._active_recovery is not None:
            return self._tick_recovery(context)

        result = self.action.tick(context)
        if (
            not isinstance(result, StageResult)
            or result.status is not StageStatus.RETRYABLE_FAILURE
        ):
            # Malformed results pass through untouched so the engine
            # raises its canonical StageResult validation error.
            return result

        # Preserve malformed codes for the engine's canonical validation
        # boundary instead of laundering them into RECOVERY_EXHAUSTED.
        if not isinstance(result.failure_code, FailureCode):
            return result
        code = result.failure_code
        if code in FATAL_SAFETY_FAILURE_CODES:
            return StageResult.fatal(
                code,
                result.message,
                arm_command=result.arm_command,
                metadata=result.metadata,
            )
        used = self._counts_by_failure[code]
        decision = self.classifier.classify(code, context, recovery_count=used)
        if self._recovery_count >= self.max_total_recoveries:
            decision = RecoveryDecision(
                code,
                False,
                None,
                self.max_total_recoveries,
                used,
                RecoveryLevel.ATTEMPT_END,
                exhausted=True,
                reason="stage-wide recovery budget exhausted",
            )
        if not decision.recovery_would_help or decision.next_recovery is None:
            return StageResult.blocked(
                f"no bounded recovery for {code.value}: {decision.reason}",
                failure_code=FailureCode.RECOVERY_EXHAUSTED,
                metadata={
                    "original_failure": code.value,
                    "recovery_count": self._recovery_count,
                },
            )

        self.action.cancel("entering bounded recovery")
        self._recovery_count += 1
        self._counts_by_failure[code] += 1
        name = decision.next_recovery
        self._active_recovery_name = name
        recovery = self._make_recovery(name, context)
        if recovery is None:
            self._restart_pending = True
            self._restart_note = name
            return StageResult.running(
                f"bounded recovery {name!r}: re-entering the step "
                f"(recovery {self._recovery_count}/{self.max_total_recoveries})",
                metadata={
                    "recovery_requested": name,
                    "recovery_count": self._recovery_count,
                    "original_failure": code.value,
                    "recovery_kind": "reenter_step",
                },
            )
        self._active_recovery = recovery
        return StageResult.running(
            f"bounded recovery {name!r}: running executor-provided recovery "
            f"action (recovery {self._recovery_count}/{self.max_total_recoveries})",
            metadata={
                "recovery_requested": name,
                "recovery_count": self._recovery_count,
                "original_failure": code.value,
                "recovery_kind": "explicit_action",
            },
        )

    def _tick_recovery(self, context: ExecutionContext) -> StageResult:
        recovery = self._active_recovery
        assert recovery is not None
        try:
            result = recovery.tick(context)
        except Exception as exc:
            self._discard_active_recovery("recovery action raised")
            return StageResult.blocked(
                f"recovery action raised: {exc}",
                failure_code=FailureCode.INTERNAL_ERROR,
                metadata={"recovery_count": self._recovery_count},
            )
        # A recovery action is subject to the exact same orchestration
        # boundary as the primary action.  Pass malformed results/codes
        # through for canonical engine validation, and never downgrade a
        # hard safety code to ordinary RECOVERY_EXHAUSTED.
        if not isinstance(result, StageResult):
            return result
        if not isinstance(result.status, StageStatus):
            return result
        if result.failure_code is not None and not isinstance(
            result.failure_code, FailureCode
        ):
            return result
        if result.failure_code in FATAL_SAFETY_FAILURE_CODES:
            self._discard_active_recovery("fatal recovery result")
            return StageResult.fatal(
                result.failure_code,
                result.message,
                arm_command=result.arm_command,
                metadata=result.metadata,
            )
        if result.status is StageStatus.RUNNING:
            return result
        completed = self._active_recovery_name
        if result.status is StageStatus.SUCCEEDED:
            self._active_recovery = None
            self._active_recovery_name = None
            self._restart_pending = True
            self._restart_note = completed or "step"
            return StageResult.running(
                f"bounded recovery {completed!r} completed; retrying the step",
                metadata={
                    "recovery_completed": completed,
                    "recovery_count": self._recovery_count,
                },
            )
        self._discard_active_recovery("recovery action failed")
        return StageResult.blocked(
            f"bounded recovery {completed!r} failed: {result.message}",
            failure_code=FailureCode.RECOVERY_EXHAUSTED,
            metadata={"recovery_count": self._recovery_count},
        )

    def _make_recovery(self, name: str, context: ExecutionContext) -> Any:
        if self.recovery_factory is None:
            return None
        recovery = self.recovery_factory(name)
        if recovery is None:
            return None
        enter = getattr(recovery, "enter", None)
        if callable(enter):
            enter(context)
        return recovery

    def _discard_active_recovery(self, reason: str) -> None:
        recovery = self._active_recovery
        if recovery is not None:
            cancel = getattr(recovery, "cancel", None)
            if callable(cancel):
                try:
                    cancel(reason)
                except Exception:
                    pass
        self._active_recovery = None
        self._active_recovery_name = None

    def _restart_inner(self) -> None:
        # The wrapper always owns the re-entry; a supplied restart_action
        # is an additional notification hook, never a substitute for it.
        if self._context is not None:
            self.action.enter(self._context)
        if self.restart_action is not None:
            self.restart_action()

    def cancel(self, reason: str) -> None:
        if self._active_recovery is not None:
            try:
                self._active_recovery.cancel(reason)
            except Exception:
                pass
        self.action.cancel(reason)
        self._active_recovery = None
        self._active_recovery_name = None
        self._restart_pending = False
        self._restart_note = ""

    def reset(self) -> None:
        self._recovery_count = 0
        self._counts_by_failure.clear()
        self._active_recovery = None
        self._active_recovery_name = None
        self._restart_pending = False
        self._restart_note = ""


__all__ = ["LegacyStageAction", "RecoverableStageAction", "RecoveryActionFactory"]
