"""No-motion executor for validating the three-task scheduling chain."""

from __future__ import annotations

from executors.base import ExecutionContext, StageResult, TaskStage


class DryRunTaskExecutor:
    """Completes every stage after a configurable number of timer ticks.

    This executor never controls the robot.  It exists only to verify task
    ordering, state transitions, logging, and process lifetime.
    """

    name = "dry_run"

    def __init__(self, task_id: int, ticks_per_stage: int = 2) -> None:
        self.task_id = int(task_id)
        self.ticks_per_stage = max(1, int(ticks_per_stage))
        self.active_stage: TaskStage | None = None
        self.ticks = 0
        self.stage_history: list[TaskStage] = []

    def reset(self) -> None:
        self.active_stage = None
        self.ticks = 0

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        self.active_stage = stage
        self.ticks = 0
        self.stage_history.append(stage)

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if stage is not self.active_stage:
            return StageResult.failed(
                f"dry-run stage mismatch: active={self.active_stage}, requested={stage}"
            )
        self.ticks += 1
        if self.ticks < self.ticks_per_stage:
            return StageResult.running(
                f"dry-run task {self.task_id} stage={stage.value} "
                f"tick={self.ticks}/{self.ticks_per_stage}"
            )
        return StageResult.succeeded(
            f"dry-run task {self.task_id} stage={stage.value} complete"
        )

    def cancel(self, reason: str) -> None:
        self.active_stage = None
        self.ticks = 0
