from __future__ import annotations

import unittest

from competition_controller import (
    CompetitionController,
    ControllerState,
    ExecutionContext,
)
from executors import build_task_executors
from executors.base import TASK_STAGE_SEQUENCE, ArmCommand, StageResult


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
) -> ExecutionContext:
    index = min(controller.task_index, len(TASKS) - 1)
    gameinfo = {"attempt": attempts_completed, "raw": ""}
    if task_ordinal is not None:
        gameinfo["task_ordinal"] = task_ordinal
    return ExecutionContext(
        now_s=0.0,
        instruction=TASKS[index],
        task_index=index,
        attempt=controller.attempt,
        referee_gameinfo=gameinfo,
        referee_taskinfo=taskinfo,
    )


def run_until(
    controller: CompetitionController,
    target: ControllerState,
    *,
    task_ordinal: int | None = None,
    limit: int = 300,
) -> None:
    for _ in range(limit):
        controller.tick(context(controller, task_ordinal=task_ordinal))
        if controller.state is target:
            return
    raise AssertionError(
        f"controller did not reach {target.value}; current={controller.state.value}"
    )


class CompetitionControllerTests(unittest.TestCase):
    def test_contact_only_builds_bounded_task1_executor(self) -> None:
        executors = build_task_executors("contact_only")

        self.assertEqual(executors[1].name, "task1_contact_only")
        self.assertEqual(executors[2].task_id, 2)
        self.assertEqual(executors[3].task_id, 3)

    def test_lift_only_builds_task1_slide_lift_executor(self) -> None:
        executors = build_task_executors("lift_only")

        self.assertEqual(executors[1].name, "task1_lift_only")
        self.assertEqual(executors[2].task_id, 2)
        self.assertEqual(executors[3].task_id, 3)

    def test_last_arm_command_persists_through_safe_hold(self) -> None:
        command = ArmCommand(
            spine_position=0.4,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )

        class ArmExecutor:
            task_id = 1
            name = "arm"

            def reset(self) -> None:
                pass

            def enter_stage(self, stage, execution_context) -> None:
                pass

            def tick(self, stage, execution_context):
                return StageResult.running(arm_command=command)

            def cancel(self, reason: str) -> None:
                pass

        executors = build_task_executors("stub")
        executors[1] = ArmExecutor()
        controller = CompetitionController(executors, referee_driven=True)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)
        for _ in range(4):
            snapshot = controller.tick(context(controller, task_ordinal=1))

        self.assertTrue(snapshot.controls_arm)
        self.assertEqual(snapshot.arm_command, command)
        controller.stop("test safe hold")
        held = controller.snapshot()
        self.assertEqual(held.state, ControllerState.SAFE_HOLD)
        self.assertTrue(held.controls_arm)
        self.assertEqual(held.arm_command, command)

    def test_executor_exception_enters_safe_hold_and_clears_command(self) -> None:
        class BrokenExecutor:
            task_id = 1
            name = "broken"

            def reset(self) -> None:
                pass

            def enter_stage(self, stage, execution_context) -> None:
                pass

            def tick(self, stage, execution_context):
                raise RuntimeError("synthetic motion failure")

            def cancel(self, reason: str) -> None:
                pass

        executors = build_task_executors("stub")
        executors[1] = BrokenExecutor()
        controller = CompetitionController(executors, referee_driven=True)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(4):
            snapshot = controller.tick(context(controller, task_ordinal=1))

        self.assertEqual(snapshot.state, ControllerState.SAFE_HOLD)
        self.assertTrue(snapshot.safe_stop)
        self.assertFalse(snapshot.controls_base)
        self.assertEqual(snapshot.base_linear_x, 0.0)
        self.assertEqual(snapshot.base_angular_z, 0.0)
        self.assertIn("synthetic motion failure", snapshot.message)

    def test_executor_base_command_is_exposed_in_snapshot(self) -> None:
        class CommandExecutor:
            task_id = 1
            name = "command"

            def reset(self) -> None:
                pass

            def enter_stage(self, stage, execution_context) -> None:
                pass

            def tick(self, stage, execution_context):
                return StageResult.running(base_command=(0.12, -0.25))

            def cancel(self, reason: str) -> None:
                pass

        executors = build_task_executors("stub")
        executors[1] = CommandExecutor()
        controller = CompetitionController(executors, referee_driven=True)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(4):
            snapshot = controller.tick(context(controller, task_ordinal=1))

        self.assertTrue(snapshot.controls_base)
        self.assertFalse(snapshot.safe_stop)
        self.assertAlmostEqual(snapshot.base_linear_x, 0.12)
        self.assertAlmostEqual(snapshot.base_angular_z, -0.25)

    def test_repeated_identical_instructions_do_not_reset_execution(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
        )
        self.assertTrue(controller.configure(TASKS))
        controller.set_inputs_ready(True)
        controller.tick(context(controller))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)

        self.assertFalse(controller.configure(TASKS))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertEqual(controller.task_id, 1)

    def test_dry_run_schedules_all_three_tasks(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        controller = CompetitionController(executors, referee_driven=False)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.FINISHED)

        self.assertEqual(controller.task_index, 3)
        for task_id in (1, 2, 3):
            self.assertEqual(executors[task_id].stage_history, list(TASK_STAGE_SEQUENCE))

    def test_stub_executor_blocks_and_keeps_safe_stop(self) -> None:
        controller = CompetitionController(
            build_task_executors("stub"),
            referee_driven=True,
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.BLOCKED, task_ordinal=1)

        snapshot = controller.snapshot()
        self.assertEqual(snapshot.task_id, 1)
        self.assertTrue(snapshot.safe_stop)
        self.assertIn("not implemented", snapshot.message)

    def test_formal_mode_waits_for_referee_before_retry_or_next_task(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=True,
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        run_until(controller, ControllerState.WAITING_FOR_REFEREE, task_ordinal=1)
        self.assertEqual(controller.task_id, 1)

        controller.tick(context(controller, task_ordinal=1, attempts_completed=0))
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_REFEREE)

        controller.tick(context(controller, task_ordinal=1, attempts_completed=1))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertEqual(controller.task_id, 1)
        self.assertEqual(controller.attempt, 2)

        run_until(controller, ControllerState.WAITING_FOR_REFEREE, task_ordinal=1)
        controller.tick(context(controller, task_ordinal=2, attempts_completed=0))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertEqual(controller.task_id, 2)

    def test_formal_mode_follows_referee_through_all_three_tasks(self) -> None:
        executors = build_task_executors("dry_run", dry_run_ticks_per_stage=1)
        controller = CompetitionController(executors, referee_driven=True)
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for task_ordinal in (1, 2, 3):
            run_until(
                controller,
                ControllerState.WAITING_FOR_REFEREE,
                task_ordinal=task_ordinal,
            )
            self.assertEqual(controller.task_id, task_ordinal)
            if task_ordinal < 3:
                controller.tick(
                    context(
                        controller,
                        task_ordinal=task_ordinal + 1,
                        attempts_completed=0,
                    )
                )
                self.assertEqual(controller.state, ControllerState.STARTING_TASK)
            else:
                controller.tick(
                    context(
                        controller,
                        task_ordinal=3,
                        attempts_completed=1,
                        taskinfo="\u5168\u90e8\u4efb\u52a1\u7ed3\u675f",
                    )
                )

        self.assertEqual(controller.state, ControllerState.FINISHED)
        for task_id in (1, 2, 3):
            self.assertEqual(executors[task_id].stage_history, list(TASK_STAGE_SEQUENCE))

    def test_referee_can_finish_the_long_lived_client(self) -> None:
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=True,
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        controller.tick(
            context(
                controller,
                task_ordinal=3,
                taskinfo="\u5168\u90e8\u4efb\u52a1\u7ed3\u675f",
            )
        )

        self.assertEqual(controller.state, ControllerState.FINISHED)
        self.assertTrue(controller.snapshot().safe_stop)


if __name__ == "__main__":
    unittest.main()
