"""Adapters from existing task executors to scheduler-v2 actions."""

from __future__ import annotations

from dataclasses import dataclass

from executors.base import ExecutionContext, StageResult, TaskExecutor, TaskStage


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


__all__ = ["LegacyStageAction"]
