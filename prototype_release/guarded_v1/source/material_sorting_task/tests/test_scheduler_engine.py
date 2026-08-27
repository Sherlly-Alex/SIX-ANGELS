from __future__ import annotations

import math
import unittest

from competition_controller import CompetitionController, ControllerState
from control_types import ArmCommand
from executors import build_task_executors
from executors.base import ExecutionContext, StageResult, TASK_STAGE_SEQUENCE
from scheduler.plans import TerminalPolicy, build_executor_task_plans
from scheduler.models import Resource
from scheduler.events import EventLog, MemoryEventSink


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


def context(
    controller: CompetitionController,
    *,
    task_ordinal: int | None = None,
    attempts_completed: int = 0,
    taskinfo: str = "",
    unsafe_collision: bool = False,
) -> ExecutionContext:
    index = min(controller.task_index, len(TASKS) - 1)
    gameinfo = {"attempt": attempts_completed, "raw": ""}
    if task_ordinal is not None:
        gameinfo["task_ordinal"] = task_ordinal
    return ExecutionContext(
        now_s=float(controller.snapshot().transition_serial),
        instruction=TASKS[index],
        task_index=index,
        attempt=controller.attempt,
        referee_gameinfo=gameinfo,
        referee_taskinfo=taskinfo,
        unsafe_collision=unsafe_collision,
    )


def run_until(controller, state, *, task_ordinal=None, limit=500):
    for _ in range(limit):
        controller.tick(context(controller, task_ordinal=task_ordinal))
        if controller.state is state:
            return
    raise AssertionError(f"did not reach {state.value}: {controller.state.value}")


class SchedulerEngineTests(unittest.TestCase):
    def test_v2_assigns_unique_task_attempt_and_step_run_scopes(self) -> None:
        sink = MemoryEventSink()
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
            scheduler_mode="v2",
            event_sink=EventLog([sink], session_id="scope-session"),
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.FINISHED)

        scoped = [event for event in sink.events if event.task_run_id]
        task_runs = {
            task_id: {event.task_run_id for event in scoped if event.task_id == task_id}
            for task_id in (1, 2, 3)
        }
        self.assertTrue(all(len(values) == 1 for values in task_runs.values()))
        self.assertEqual(len(set.union(*task_runs.values())), 3)
        for task_id in (1, 2, 3):
            attempt_runs = {
                event.attempt_run_id
                for event in scoped
                if event.task_id == task_id and event.attempt_run_id
            }
            self.assertEqual(len(attempt_runs), 1)
        step_runs = {
            event.step_run_id
            for event in scoped
            if event.task_id == 1 and event.step_run_id
        }
        self.assertEqual(len(step_runs), len(TASK_STAGE_SEQUENCE))
        self.assertTrue(all(event.session_id == "scope-session" for event in scoped))

    def test_v2_dry_run_executes_all_versioned_plans(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.FINISHED)

        self.assertEqual(controller.task_index, 3)
        for task_id in (1, 2, 3):
            self.assertEqual(executors[task_id].stage_history, list(TASK_STAGE_SEQUENCE))

    def test_v2_matches_legacy_dry_run_trace(self) -> None:
        legacy = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
            scheduler_mode="legacy",
        )
        v2 = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
            scheduler_mode="v2",
        )
        for controller in (legacy, v2):
            controller.configure(TASKS)
            controller.set_inputs_ready(True)

        for _ in range(500):
            legacy_snapshot = legacy.tick(context(legacy))
            v2_snapshot = v2.tick(context(v2))
            comparable_legacy = (
                legacy_snapshot.state,
                legacy_snapshot.task_index,
                legacy_snapshot.task_id,
                legacy_snapshot.attempt,
                legacy_snapshot.stage,
                legacy_snapshot.safe_stop,
                legacy_snapshot.message,
                legacy_snapshot.transition_serial,
            )
            comparable_v2 = (
                v2_snapshot.state,
                v2_snapshot.task_index,
                v2_snapshot.task_id,
                v2_snapshot.attempt,
                v2_snapshot.stage,
                v2_snapshot.safe_stop,
                v2_snapshot.message,
                v2_snapshot.transition_serial,
            )
            self.assertEqual(comparable_v2, comparable_legacy)
            if legacy.state is ControllerState.FINISHED:
                break
        else:
            self.fail("controllers did not finish")

    def test_shadow_observes_without_double_ticking_executor(self) -> None:
        class CountingExecutor:
            task_id = 1
            name = "counting"

            def __init__(self):
                self.ticks = 0

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                self.ticks += 1
                return StageResult.running("count")

            def cancel(self, reason):
                pass

        executors = build_task_executors("stub")
        counting = CountingExecutor()
        executors[1] = counting
        controller = CompetitionController(
            executors,
            referee_driven=True,
            scheduler_mode="shadow",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(8):
            controller.tick(context(controller, task_ordinal=1))

        self.assertEqual(counting.ticks, 5)
        self.assertTrue(controller.shadow_healthy)

    def test_v2_collision_input_enters_safe_hold(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        snapshot = controller.tick(context(controller, unsafe_collision=True))

        self.assertEqual(snapshot.state, ControllerState.SAFE_HOLD)
        self.assertTrue(snapshot.safe_stop)
        self.assertIn("unsafe collision", snapshot.message)

    def test_v2_rejects_non_finite_base_command(self) -> None:
        class InvalidExecutor:
            task_id = 1
            name = "invalid"

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                return StageResult.running(base_command=(math.nan, 0.0))

            def cancel(self, reason):
                pass

        executors = build_task_executors("stub")
        executors[1] = InvalidExecutor()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.SAFE_HOLD)

        self.assertRegex(controller.snapshot().message, r"non-finite|NaN|infinity")

    def test_v2_rejects_external_manipulator_resource_conflict(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        controller._backend.resource_manager.acquire(
            {Resource.RIGHT_ARM},
            owner="external-owner",
        )

        run_until(controller, ControllerState.SAFE_HOLD)

        self.assertIn("resource conflict", controller.snapshot().message)

    def test_v2_rejects_malformed_full_arm_command(self) -> None:
        command = ArmCommand(
            spine_position=0.4,
            head_positions=(0.0,),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )

        class InvalidExecutor:
            task_id = 1
            name = "malformed_arm"

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                return StageResult.running(arm_command=command)

            def cancel(self, reason):
                pass

        executors = build_task_executors("stub")
        executors[1] = InvalidExecutor()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.SAFE_HOLD)

        self.assertIn("ArmCommand", controller.snapshot().message)

    def test_v2_does_not_consume_a_desynchronised_referee_jump(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=True,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        run_until(controller, ControllerState.WAITING_FOR_REFEREE, task_ordinal=1)

        controller.tick(context(controller, task_ordinal=3))

        self.assertEqual(controller.state, ControllerState.WAITING_FOR_REFEREE)
        self.assertEqual(controller.task_id, 1)

    def test_close_releases_active_resources_and_enters_safe_hold(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=5),
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        for _ in range(3):
            controller.tick(context(controller))
        self.assertTrue(controller._backend.resource_manager.owners)

        controller.close()

        self.assertEqual(controller.state, ControllerState.SAFE_HOLD)
        self.assertFalse(controller._backend.resource_manager.owners)
        self.assertFalse(controller._backend.base_command_lease.snapshot(10.0).active)

    def test_persistent_arm_hold_keeps_full_manipulator_lease(self) -> None:
        command = ArmCommand(
            spine_position=0.4,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )

        class HoldingExecutor:
            task_id = 1
            name = "holding"

            def __init__(self):
                self.ticks = 0

            def reset(self):
                self.ticks = 0

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                self.ticks += 1
                if self.ticks == 1:
                    return StageResult.running("hold", arm_command=command)
                return StageResult.succeeded("advance", arm_command=command)

            def cancel(self, reason):
                pass

        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        executors[1] = HoldingExecutor()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(80):
            snapshot = controller.tick(context(controller))
            backend = controller._backend
            if snapshot.controls_arm and backend._active_action is None:
                break
        else:
            self.fail("scheduler never reached a cross-stage arm hold")

        self.assertIsNotNone(backend._arm_hold_owner)
        self.assertTrue(
            backend.resource_manager.owns(
                backend._arm_hold_owner,
                backend._manipulator_resources(),
            )
        )

        # Entering the next stage transfers ownership from the persistent hold
        # to the active stage without dropping the command.
        controller.tick(context(controller))
        self.assertIsNone(backend._arm_hold_owner)
        self.assertTrue(
            backend.resource_manager.owns(
                backend._active_owner,
                backend._manipulator_resources(),
            )
        )

    def test_v2_rejects_non_stage_result(self) -> None:
        class InvalidExecutor:
            task_id = 1
            name = "invalid_result"

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                return object()

            def cancel(self, reason):
                pass

        executors = build_task_executors("stub")
        executors[1] = InvalidExecutor()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.SAFE_HOLD)

        self.assertIn("StageResult", controller.snapshot().message)

    def test_v2_cancel_exception_fails_closed(self) -> None:
        class InvalidExecutor:
            task_id = 1
            name = "cancel_error"

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                return StageResult.succeeded("done")

            def cancel(self, reason):
                raise RuntimeError("cancel boom")

        executors = build_task_executors("stub")
        executors[1] = InvalidExecutor()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.SAFE_HOLD)

        self.assertIn("cancel boom", controller.snapshot().message)
        self.assertFalse(controller._backend.resource_manager.owners)

    def test_terminal_cleanup_policy_is_plan_metadata(self) -> None:
        plans = build_executor_task_plans()

        self.assertEqual(
            plans[1].terminal_policy,
            TerminalPolicy.STOP_IMMEDIATELY,
        )
        self.assertEqual(
            plans[3].terminal_policy,
            TerminalPolicy.COMPLETE_ACTIVE_SEQUENCE,
        )
        self.assertTrue(plans[3].stages[-1].cleanup)
        full_manipulator = {"spine", "head", "left_arm", "right_arm", "grippers"}
        for plan in plans.values():
            for spec in plan.stages:
                if spec.allows_arm:
                    self.assertTrue(full_manipulator.issubset(spec.resources))

    def test_terminal_after_irreversible_step_skips_to_explicit_cleanup(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        controller = CompetitionController(
            executors,
            referee_driven=True,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        for _ in range(200):
            controller.tick(context(controller, task_ordinal=3))
            if controller.stage is TASK_STAGE_SEQUENCE[7]:  # PLACE
                break
        else:
            self.fail("task 3 did not reach PLACE")

        for _ in range(20):
            snapshot = controller.tick(
                context(
                    controller,
                    task_ordinal=3,
                    taskinfo="全部任务结束",
                )
            )
            if snapshot.state is ControllerState.FINISHED:
                break

        self.assertEqual(snapshot.state, ControllerState.FINISHED)
        self.assertIn(TASK_STAGE_SEQUENCE[7], executors[3].stage_history)
        self.assertNotIn(TASK_STAGE_SEQUENCE[8], executors[3].stage_history)
        self.assertIn(TASK_STAGE_SEQUENCE[9], executors[3].stage_history)

    def test_terminal_before_irreversible_step_stops_without_cleanup(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=5)
        controller = CompetitionController(
            executors,
            referee_driven=True,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        controller.tick(context(controller, task_ordinal=3))
        controller.tick(context(controller, task_ordinal=3))
        controller.tick(context(controller, task_ordinal=3))

        snapshot = controller.tick(
            context(
                controller,
                task_ordinal=3,
                taskinfo="全部任务结束",
            )
        )

        self.assertEqual(snapshot.state, ControllerState.FINISHED)
        self.assertNotIn(TASK_STAGE_SEQUENCE[-1], executors[3].stage_history)

    def test_unknown_scheduler_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported scheduler_mode"):
            CompetitionController(
                build_task_executors("stub"),
                scheduler_mode="experimental",
            )


if __name__ == "__main__":
    unittest.main()
