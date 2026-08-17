from __future__ import annotations

import unittest

from competition_controller import CompetitionController, ControllerState
from executors.base import ExecutionContext, StageResult, StageStatus, TaskStage
from scheduler.events import EventLog, MemoryEventSink
from scheduler.legacy_adapter import LegacyStageAction, RecoverableStageAction
from scheduler.models import FailureCode
from scheduler.recovery import FATAL_SAFETY_FAILURE_CODES


TASKS = [
    {
        "task": task_id,
        "instruction": f"task {task_id}",
        "target_color": color,
        "place_type": place_type,
        "place_world": [float(task_id), 0.0, 0.5],
    }
    for task_id, color, place_type in (
        (1, "pink", "shelf_point"),
        (2, "yellow", "table_point"),
        (3, "brown", "shelf_prop_side"),
    )
]


def execution_context(controller: CompetitionController) -> ExecutionContext:
    index = min(controller.task_index, len(TASKS) - 1)
    return ExecutionContext(
        now_s=float(controller.snapshot().transition_serial),
        instruction=TASKS[index],
        task_index=index,
        attempt=controller.attempt,
        referee_gameinfo={"attempt": 0, "raw": ""},
        referee_taskinfo="",
    )


class _ScriptedExecutor:
    task_id = 1
    name = "scripted"

    def __init__(self, results) -> None:
        self.results = list(results)
        self.enter_count = 0
        self.cancel_count = 0
        self.reset_count = 0
        self.stage_history: list[TaskStage] = []
        self.last_stage: TaskStage | None = None

    def reset(self) -> None:
        self.reset_count += 1

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        self.enter_count += 1
        self.last_stage = stage
        self.stage_history.append(stage)

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        self.last_stage = stage
        if not self.results:
            return StageResult.succeeded("scripted task complete")
        return self.results.pop(0)

    def cancel(self, reason: str) -> None:
        self.cancel_count += 1


class _SuccessExecutor(_ScriptedExecutor):
    """Completes every stage immediately; used for tasks 2 and 3."""

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        self.last_stage = stage
        return StageResult.succeeded("scripted success")


def build_controller(executor, *, scheduler_mode="v2", event_sink=None):
    return CompetitionController(
        {1: executor, 2: _SuccessExecutor([]), 3: _SuccessExecutor([])},
        referee_driven=False,
        scheduler_mode=scheduler_mode,
        event_sink=event_sink,
    )


class StageRecoveryBridgeTests(unittest.TestCase):
    def test_v2_recovers_retryable_navigation_failure_with_bounded_retry(self) -> None:
        executor = _ScriptedExecutor(
            [
                StageResult.retryable_failure(FailureCode.NAV_NO_PATH, "no path"),
                StageResult.retryable_failure(FailureCode.NAV_NO_PATH, "no path again"),
                StageResult.succeeded("replanned and reached"),
            ]
        )
        sink = MemoryEventSink()
        controller = build_controller(executor, event_sink=EventLog([sink]))
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(500):
            controller.tick(execution_context(controller))
            if controller.state is ControllerState.FINISHED:
                break
        else:
            self.fail("controller did not finish")

        # 10 stages entered once each, plus two bounded step re-entries.
        self.assertEqual(executor.enter_count, 12)
        recovery_events = [
            event for event in sink.events if event.type == "step_recovery"
        ]
        self.assertEqual(len(recovery_events), 2)
        self.assertEqual(
            [event.details.get("recovery_kind") for event in recovery_events],
            ["reenter_step", "reenter_step"],
        )

    def test_recovery_budget_exhaustion_fails_closed(self) -> None:
        executor = _ScriptedExecutor(
            [StageResult.retryable_failure(FailureCode.NAV_NO_PATH, "no path")] * 10
        )
        controller = build_controller(executor)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(500):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.BLOCKED:
                break
        else:
            self.fail("controller did not block")

        # Initial entry plus exactly two classified NAV_NO_PATH recoveries.
        self.assertEqual(executor.enter_count, 3)
        self.assertIn("no bounded recovery", controller.snapshot().message)

    def test_fatal_structured_failure_enters_safe_hold(self) -> None:
        executor = _ScriptedExecutor(
            [StageResult.fatal(FailureCode.UNSAFE_COLLISION, "hard collision")]
        )
        controller = build_controller(executor)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(100):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.SAFE_HOLD:
                break
        else:
            self.fail("controller did not enter safe hold")

        self.assertIn("fatal structured failure", controller.snapshot().message)

    def test_retryable_fatal_code_cannot_be_laundered_by_recovery(self) -> None:
        executor = _ScriptedExecutor(
            [
                StageResult.retryable_failure(
                    FailureCode.UNSAFE_COLLISION,
                    "collision was incorrectly marked retryable",
                )
            ]
        )
        controller = build_controller(executor)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(100):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.SAFE_HOLD:
                break
        else:
            self.fail("fatal retryable code did not enter safe hold")

        self.assertEqual(executor.enter_count, 1)
        self.assertIn("unsafe_collision", controller.snapshot().message)

    def test_invalid_failure_code_reaches_engine_validation_boundary(self) -> None:
        executor = _ScriptedExecutor(
            [StageResult.retryable_failure("nav_no_path", "untyped code")]
        )
        controller = build_controller(executor)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(100):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.SAFE_HOLD:
                break
        else:
            self.fail("invalid failure code did not enter safe hold")

        self.assertEqual(executor.enter_count, 1)
        self.assertIn("failure_code must be", controller.snapshot().message)

    def test_legacy_retryable_failure_keeps_blocked_semantics(self) -> None:
        executor = _ScriptedExecutor(
            [StageResult.retryable_failure(FailureCode.NAV_STUCK, "stuck")]
        )
        controller = build_controller(executor, scheduler_mode="legacy")
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(100):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.BLOCKED:
                break
        else:
            self.fail("legacy controller did not block")

        # Legacy has no recovery layer: exactly one stage entry.
        self.assertEqual(executor.enter_count, 1)
        self.assertEqual(executor.cancel_count, 1)

    def test_irreversible_stage_never_restarts(self) -> None:
        # Stages 0..6 succeed; stage 7 is PLACE (irreversible) and fails.
        executor = _ScriptedExecutor(
            [StageResult.succeeded("ok")] * 7
            + [StageResult.retryable_failure(FailureCode.NAV_STUCK, "stuck")],
        )
        controller = build_controller(executor)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(500):
            snapshot = controller.tick(execution_context(controller))
            if snapshot.state is ControllerState.BLOCKED:
                break
        else:
            self.fail("controller did not block")

        # Seven completed stages + one PLACE entry; no re-entry for PLACE.
        self.assertEqual(executor.enter_count, 8)
        self.assertIs(executor.last_stage, TaskStage.PLACE)
        self.assertIn("without a recovery path", controller.snapshot().message)


class RecoverableStageActionTests(unittest.TestCase):
    def test_explicit_recovery_runs_before_step_retry(self) -> None:
        action_results = [
            StageResult.retryable_failure(FailureCode.TARGET_LOST, "lost"),
            StageResult.succeeded("target found"),
        ]
        recovery_results = [
            StageResult.running("scanning"),
            StageResult.succeeded("scan complete"),
        ]
        action = LegacyStageAction(
            _ScriptedExecutor(action_results),
            TaskStage.NAVIGATE_TO_PICK,
        )
        recovery = LegacyStageAction(
            _ScriptedExecutor(recovery_results),
            TaskStage.ACQUIRE_TARGET,
        )
        restarted: list[bool] = []
        wrapped = RecoverableStageAction(
            action=action,
            recovery_factory=lambda name: recovery,
            restart_action=lambda: restarted.append(True),
        )
        context = execution_context(build_controller(_SuccessExecutor([])))
        wrapped.enter(context)

        first = wrapped.tick(context)
        second = wrapped.tick(context)
        third = wrapped.tick(context)
        fourth = wrapped.tick(context)
        final = wrapped.tick(context)

        self.assertIs(first.status, StageStatus.RUNNING)
        self.assertEqual(first.metadata["recovery_requested"], "stationary_rescan")
        self.assertEqual(second.message, "scanning")
        self.assertIn("completed", third.message)
        self.assertIn("re-entered", fourth.message)
        self.assertEqual(restarted, [True])
        self.assertIs(final.status, StageStatus.SUCCEEDED)
        self.assertEqual(action.executor.cancel_count, 1)

    def test_failed_recovery_fails_closed_without_restart(self) -> None:
        action = LegacyStageAction(
            _ScriptedExecutor(
                [StageResult.retryable_failure(FailureCode.TARGET_LOST, "lost")],
            ),
            TaskStage.NAVIGATE_TO_PICK,
        )
        recovery = LegacyStageAction(
            _ScriptedExecutor([StageResult.blocked("scan failed")]),
            TaskStage.ACQUIRE_TARGET,
        )
        wrapped = RecoverableStageAction(
            action=action,
            recovery_factory=lambda name: recovery,
        )
        context = execution_context(build_controller(_SuccessExecutor([])))
        wrapped.enter(context)

        wrapped.tick(context)  # requests recovery
        terminal = wrapped.tick(context)  # recovery fails immediately

        self.assertIs(terminal.status, StageStatus.BLOCKED)
        self.assertIs(terminal.failure_code, FailureCode.RECOVERY_EXHAUSTED)

    def test_fatal_code_from_explicit_recovery_is_preserved(self) -> None:
        action = LegacyStageAction(
            _ScriptedExecutor(
                [StageResult.retryable_failure(FailureCode.TARGET_LOST, "lost")],
            ),
            TaskStage.NAVIGATE_TO_PICK,
        )
        recovery = LegacyStageAction(
            _ScriptedExecutor(
                [StageResult.fatal(FailureCode.UNSAFE_COLLISION, "recovery collision")]
            ),
            TaskStage.ACQUIRE_TARGET,
        )
        wrapped = RecoverableStageAction(
            action=action,
            recovery_factory=lambda name: recovery,
        )
        context = execution_context(build_controller(_SuccessExecutor([])))
        wrapped.enter(context)

        wrapped.tick(context)  # requests and enters explicit recovery
        terminal = wrapped.tick(context)

        self.assertIs(terminal.status, StageStatus.BLOCKED)
        self.assertIs(terminal.failure_code, FailureCode.UNSAFE_COLLISION)
        self.assertEqual(terminal.message, "recovery collision")


class StageResultStructureTests(unittest.TestCase):
    def test_retryable_failure_carries_code_and_immutable_metadata(self) -> None:
        result = StageResult.retryable_failure(
            FailureCode.NAV_NO_PATH,
            "no path",
            metadata={"recovery_count": 1},
        )
        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.NAV_NO_PATH)
        self.assertEqual(result.metadata["recovery_count"], 1)
        with self.assertRaises(TypeError):
            result.metadata["extra"] = True  # type: ignore[index]

    def test_fatal_keeps_legacy_safe_blocked_status(self) -> None:
        result = StageResult.fatal(FailureCode.EFFORT_HARD_LIMIT, "hard effort")
        self.assertIs(result.status, StageStatus.BLOCKED)
        self.assertIs(result.failure_code, FailureCode.EFFORT_HARD_LIMIT)

    def test_fatal_safety_code_membership(self) -> None:
        self.assertIn(FailureCode.UNSAFE_COLLISION, FATAL_SAFETY_FAILURE_CODES)
        self.assertIn(FailureCode.EFFORT_HARD_LIMIT, FATAL_SAFETY_FAILURE_CODES)
        self.assertNotIn(FailureCode.NAV_NO_PATH, FATAL_SAFETY_FAILURE_CODES)
        self.assertNotIn(FailureCode.TARGET_LOST, FATAL_SAFETY_FAILURE_CODES)


if __name__ == "__main__":
    unittest.main()
