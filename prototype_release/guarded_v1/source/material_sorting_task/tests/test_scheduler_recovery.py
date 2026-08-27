from __future__ import annotations

import unittest

from scheduler.models import ActionResult, ActionStatus, FailureCode
from scheduler.recovery import (
    RecoverableStep,
    RecoveryClassifier,
    RecoveryPolicy,
)


class _AlwaysNoPath:
    def __init__(self) -> None:
        self.cancel_count = 0

    def tick(self, context):
        return ActionResult.retryable_failure(FailureCode.NAV_NO_PATH, "no path")

    def cancel(self, reason: str) -> None:
        self.cancel_count += 1


class _ScriptedAction:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.cancel_count = 0

    def tick(self, context):
        return self.results.pop(0)

    def cancel(self, reason: str) -> None:
        self.cancel_count += 1


class SchedulerRecoveryTests(unittest.TestCase):
    def test_classifier_round_robin_is_finite(self) -> None:
        classifier = RecoveryClassifier(
            {FailureCode.NAV_NO_PATH: RecoveryPolicy(("replan",), 2)}
        )

        first = classifier.classify(FailureCode.NAV_NO_PATH, recovery_count=0)
        second = classifier.classify(FailureCode.NAV_NO_PATH, recovery_count=1)
        exhausted = classifier.classify(FailureCode.NAV_NO_PATH, recovery_count=2)

        self.assertEqual(first.next_recovery, "replan")
        self.assertEqual(second.next_recovery, "replan")
        self.assertTrue(exhausted.exhausted)
        self.assertFalse(exhausted.recovery_would_help)

    def test_recoverable_step_exhausts_retry_budget(self) -> None:
        action = _AlwaysNoPath()
        classifier = RecoveryClassifier(
            {FailureCode.NAV_NO_PATH: RecoveryPolicy(("replan",), 2)}
        )
        step = RecoverableStep(action, classifier=classifier)

        first = step.tick(None)
        second = step.tick(None)
        terminal = step.tick(None)

        self.assertIs(first.status, ActionStatus.RUNNING)
        self.assertIs(second.status, ActionStatus.RUNNING)
        self.assertIs(terminal.status, ActionStatus.ATTEMPT_FAILED)
        self.assertIs(terminal.failure_code, FailureCode.RECOVERY_EXHAUSTED)
        self.assertEqual(action.cancel_count, 2)

    def test_explicit_recovery_runs_before_original_action_retry(self) -> None:
        action = _ScriptedAction(
            [
                ActionResult.retryable_failure(FailureCode.TARGET_LOST),
                ActionResult.succeeded("target found"),
            ]
        )
        recovery = _ScriptedAction(
            [ActionResult.running("scanning"), ActionResult.succeeded("scan complete")]
        )
        classifier = RecoveryClassifier(
            {FailureCode.TARGET_LOST: RecoveryPolicy(("rescan",), 1)}
        )
        restarted: list[bool] = []
        step = RecoverableStep(
            action,
            classifier=classifier,
            recovery_factory={"rescan": recovery},
            restart_action=lambda: restarted.append(True),
        )

        self.assertIs(step.tick(None).status, ActionStatus.RUNNING)  # recovery requested
        self.assertEqual(step.tick(None).message, "scanning")
        completed = step.tick(None)
        final = step.tick(None)

        self.assertEqual(completed.metadata["recovery_completed"], "rescan")
        self.assertEqual(restarted, [True])
        self.assertIs(final.status, ActionStatus.SUCCEEDED)

    def test_fatal_failure_is_never_recovered(self) -> None:
        action = _ScriptedAction(
            [ActionResult.fatal(FailureCode.EFFORT_HARD_LIMIT, "hard effort")]
        )
        result = RecoverableStep(action).tick(None)

        self.assertIs(result.status, ActionStatus.FATAL)
        self.assertIs(result.failure_code, FailureCode.EFFORT_HARD_LIMIT)


if __name__ == "__main__":
    unittest.main()
