"""Pure three-task orchestration for the formal competition client.

The controller deliberately contains no ROS imports.  ``client_task.py`` owns
ROS lifecycle and feeds observations into this state machine.  In formal mode,
the referee is the source of truth for attempt settlement, task progression,
and scoring; the client never awards itself points or advances a scored task
without referee confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from executors.base import (
    TASK_STAGE_SEQUENCE,
    ExecutionContext,
    StageStatus,
    TaskExecutor,
    TaskStage,
)


class ControllerState(Enum):
    WAITING_FOR_INPUTS = "waiting_for_inputs"
    STARTING_TASK = "starting_task"
    EXECUTING_STAGE = "executing_stage"
    WAITING_FOR_REFEREE = "waiting_for_referee"
    BLOCKED = "blocked"
    FINISHED = "finished"
    SAFE_HOLD = "safe_hold"


@dataclass(frozen=True)
class ControllerSnapshot:
    state: ControllerState
    task_index: int
    task_id: int | None
    attempt: int
    stage: TaskStage | None
    safe_stop: bool
    controls_base: bool
    base_linear_x: float
    base_angular_z: float
    message: str
    transition_serial: int


class CompetitionController:
    """Schedule task 1, task 2, and task 3 in one long-lived process."""

    def __init__(
        self,
        executors: Mapping[int, TaskExecutor],
        *,
        referee_driven: bool = True,
        max_attempts: int = 3,
    ) -> None:
        missing = {1, 2, 3} - set(executors)
        if missing:
            raise ValueError(f"missing task executors: {sorted(missing)}")
        self.executors = dict(executors)
        self.referee_driven = bool(referee_driven)
        self.max_attempts = max(1, int(max_attempts))

        self.instructions: list[dict] = []
        self.inputs_ready = False
        self.state = ControllerState.WAITING_FOR_INPUTS
        self.task_index = 0
        self.attempt = 1
        self.stage_index = 0
        self._stage_entered = False
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._message = "waiting for validated instructions and robot inputs"
        self._transition_serial = 0
        self._wait_referee_attempts_completed = 0

    @property
    def task_id(self) -> int | None:
        if self.task_index < 0 or self.task_index >= len(self.instructions):
            return None
        value = self.instructions[self.task_index].get("task")
        return int(value) if value is not None else None

    @property
    def stage(self) -> TaskStage | None:
        if self.state not in (
            ControllerState.STARTING_TASK,
            ControllerState.EXECUTING_STAGE,
        ):
            return None
        if self.stage_index < 0 or self.stage_index >= len(TASK_STAGE_SEQUENCE):
            return None
        return TASK_STAGE_SEQUENCE[self.stage_index]

    def configure(self, instructions: Sequence[Mapping]) -> bool:
        """Load the three validated instructions.

        Repeated publication of the same instruction list is idempotent.  A
        changed list is rejected after execution starts because resetting the
        client would desynchronise it from the long-lived Server session.
        """
        normalized = sorted(
            (dict(item) for item in instructions),
            key=lambda item: int(item.get("task", 0)),
        )
        task_ids = [item.get("task") for item in normalized]
        if task_ids != [1, 2, 3]:
            raise ValueError(f"expected task ids [1, 2, 3], received {task_ids}")
        if normalized == self.instructions:
            return False
        if self.instructions and self.state is not ControllerState.WAITING_FOR_INPUTS:
            raise RuntimeError("instructions changed after task execution started")

        self.instructions = normalized
        self.task_index = 0
        self.attempt = 1
        self.stage_index = 0
        self._stage_entered = False
        self._transition(
            ControllerState.WAITING_FOR_INPUTS,
            "three validated task instructions configured",
        )
        return True

    def set_inputs_ready(self, ready: bool) -> None:
        self.inputs_ready = bool(ready)

    def tick(self, context: ExecutionContext) -> ControllerSnapshot:
        """Advance at most one visible state transition per control cycle."""
        if self.state in (
            ControllerState.FINISHED,
            ControllerState.SAFE_HOLD,
            ControllerState.BLOCKED,
        ):
            return self.snapshot()

        if self.referee_driven and self._referee_finished(context):
            self._transition(ControllerState.FINISHED, "referee reported all tasks finished")
            return self.snapshot()

        if self.state is ControllerState.WAITING_FOR_INPUTS:
            if not self.instructions or not self.inputs_ready:
                return self.snapshot()
            self._sync_start_from_referee(context)
            self._transition(
                ControllerState.STARTING_TASK,
                f"starting task {self.task_id} attempt {self.attempt}",
            )
            return self.snapshot()

        if self.state is ControllerState.STARTING_TASK:
            executor = self._executor()
            executor.reset()
            self.stage_index = 0
            self._stage_entered = False
            self._controls_base = False
            self._transition(
                ControllerState.EXECUTING_STAGE,
                f"task {self.task_id} entering {self.stage.value}",
            )
            return self.snapshot()

        if self.state is ControllerState.EXECUTING_STAGE:
            executor = self._executor()
            stage = self.stage
            if stage is None:
                self.stop("invalid empty execution stage")
                return self.snapshot()
            if not self._stage_entered:
                try:
                    executor.enter_stage(stage, context)
                except Exception as exc:
                    self.stop(
                        f"task {self.task_id} failed to enter stage={stage.value}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return self.snapshot()
                self._stage_entered = True
                self._controls_base = False
                self._bump_message(
                    f"task {self.task_id} attempt {self.attempt} stage={stage.value}"
                )
                return self.snapshot()

            try:
                result = executor.tick(stage, context)
            except Exception as exc:
                self.stop(
                    f"task {self.task_id} executor error at stage={stage.value}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return self.snapshot()
            self._controls_base = bool(result.controls_base)
            self._base_linear_x = (
                float(result.base_linear_x) if self._controls_base else 0.0
            )
            self._base_angular_z = (
                float(result.base_angular_z) if self._controls_base else 0.0
            )
            if result.status is StageStatus.RUNNING:
                if result.message:
                    message_changed = result.message != self._message
                    self._message = result.message
                    if result.controls_base and message_changed:
                        self._transition_serial += 1
                return self.snapshot()
            if result.status is StageStatus.BLOCKED:
                executor.cancel(result.message)
                self._controls_base = False
                self._transition(ControllerState.BLOCKED, result.message)
                return self.snapshot()
            if result.status is StageStatus.FAILED:
                executor.cancel(result.message)
                self._controls_base = False
                self._finish_local_attempt(context, succeeded=False, message=result.message)
                return self.snapshot()

            if self.stage_index + 1 < len(TASK_STAGE_SEQUENCE):
                self.stage_index += 1
                self._stage_entered = False
                self._controls_base = False
                self._bump_message(
                    f"task {self.task_id} entering {self.stage.value}: {result.message}"
                )
                return self.snapshot()

            executor.cancel("task action sequence complete")
            self._controls_base = False
            self._finish_local_attempt(context, succeeded=True, message=result.message)
            return self.snapshot()

        if self.state is ControllerState.WAITING_FOR_REFEREE:
            self._tick_waiting_for_referee(context)
            return self.snapshot()

        self.stop(f"unhandled controller state {self.state.value}")
        return self.snapshot()

    def stop(self, reason: str = "client stop requested") -> None:
        task_id = self.task_id
        if task_id in self.executors:
            try:
                self.executors[task_id].cancel(reason)
            except Exception:
                # The ROS owner still publishes a zero base command.  A
                # broken executor cancel hook must not crash the Client.
                pass
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._transition(ControllerState.SAFE_HOLD, reason)

    def snapshot(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self.state,
            task_index=self.task_index,
            task_id=self.task_id,
            attempt=self.attempt,
            stage=self.stage,
            safe_stop=not self._controls_base,
            controls_base=self._controls_base,
            base_linear_x=self._base_linear_x,
            base_angular_z=self._base_angular_z,
            message=self._message,
            transition_serial=self._transition_serial,
        )

    def _executor(self) -> TaskExecutor:
        task_id = self.task_id
        if task_id not in self.executors:
            raise RuntimeError(f"no executor for task id {task_id}")
        return self.executors[task_id]

    def _finish_local_attempt(
        self,
        context: ExecutionContext,
        *,
        succeeded: bool,
        message: str,
    ) -> None:
        if self.referee_driven:
            self._wait_referee_attempts_completed = self._referee_attempts_completed(context)
            outcome = "completed" if succeeded else "failed"
            self._transition(
                ControllerState.WAITING_FOR_REFEREE,
                f"task {self.task_id} local sequence {outcome}; waiting for Server referee: {message}",
            )
            return

        if succeeded:
            self._advance_to_next_task("dry-run task sequence completed")
            return
        if self.attempt < self.max_attempts:
            self.attempt += 1
            self._transition(
                ControllerState.STARTING_TASK,
                f"dry-run retry task {self.task_id} attempt {self.attempt}: {message}",
            )
        else:
            self._advance_to_next_task(
                f"dry-run task {self.task_id} exhausted {self.max_attempts} attempts"
            )

    def _tick_waiting_for_referee(self, context: ExecutionContext) -> None:
        if self._referee_finished(context):
            self._transition(ControllerState.FINISHED, "referee reported all tasks finished")
            return

        ordinal = self._referee_task_ordinal(context)
        completed = self._referee_attempts_completed(context)
        current_ordinal = self.task_index + 1

        if ordinal is not None and ordinal > current_ordinal:
            self.task_index = min(ordinal - 1, len(self.instructions) - 1)
            self.attempt = max(1, completed + 1)
            self.stage_index = 0
            self._stage_entered = False
            self._transition(
                ControllerState.STARTING_TASK,
                f"referee advanced to task {self.task_id}; starting attempt {self.attempt}",
            )
            return

        if completed > self._wait_referee_attempts_completed:
            self.attempt = completed + 1
            if self.attempt <= self.max_attempts:
                self.stage_index = 0
                self._stage_entered = False
                self._transition(
                    ControllerState.STARTING_TASK,
                    f"referee settled attempt; retrying task {self.task_id} attempt {self.attempt}",
                )
            else:
                self._message = (
                    f"referee reports {completed} attempts settled for task {self.task_id}; "
                    "waiting for task progression"
                )

    def _advance_to_next_task(self, message: str) -> None:
        self.task_index += 1
        self.attempt = 1
        self.stage_index = 0
        self._stage_entered = False
        if self.task_index >= len(self.instructions):
            self._transition(ControllerState.FINISHED, message)
        else:
            self._transition(
                ControllerState.STARTING_TASK,
                f"{message}; advancing to task {self.task_id}",
            )

    def _sync_start_from_referee(self, context: ExecutionContext) -> None:
        if not self.referee_driven:
            return
        ordinal = self._referee_task_ordinal(context)
        if ordinal is not None and 1 <= ordinal <= len(self.instructions):
            self.task_index = ordinal - 1
        completed = self._referee_attempts_completed(context)
        self.attempt = min(self.max_attempts, max(1, completed + 1))

    @staticmethod
    def _referee_task_ordinal(context: ExecutionContext) -> int | None:
        value = context.referee_gameinfo.get("task_ordinal")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _referee_attempts_completed(context: ExecutionContext) -> int:
        value = context.referee_gameinfo.get("attempt", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _referee_finished(context: ExecutionContext) -> bool:
        task_text = context.referee_taskinfo.casefold()
        game_text = str(context.referee_gameinfo.get("raw", "")).casefold()
        return (
            "\u5168\u90e8\u4efb\u52a1\u7ed3\u675f" in task_text
            or "all tasks finished" in task_text
            or "all_tasks_done" in game_text
        )

    def _transition(self, state: ControllerState, message: str) -> None:
        self.state = state
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._message = message
        self._transition_serial += 1

    def _bump_message(self, message: str) -> None:
        self._message = message
        self._transition_serial += 1


__all__ = [
    "CompetitionController",
    "ControllerSnapshot",
    "ControllerState",
    "ExecutionContext",
]
