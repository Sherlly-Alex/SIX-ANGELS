"""Versioned task-stage plans for the scheduler v2 compatibility path.

The concrete motion implementations remain in ``executors/``.  These plans
make ordering, resource intent and terminal-cleanup policy explicit without
changing a validated physical trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from executors.base import TASK_STAGE_SEQUENCE, TaskStage


class TerminalPolicy(Enum):
    """How a running plan reacts to a terminal referee event."""

    STOP_IMMEDIATELY = "stop_immediately"
    COMPLETE_ACTIVE_SEQUENCE = "complete_active_sequence"


@dataclass(frozen=True)
class LegacyStageSpec:
    """Metadata around one existing executor stage."""

    stage: TaskStage
    resources: frozenset[str]
    timeout_s: float | None = None
    recovery_policy: str | None = None
    irreversible: bool = False
    cleanup: bool = False

    @property
    def id(self) -> str:
        return self.stage.value

    @property
    def allows_base(self) -> bool:
        return "base" in self.resources

    @property
    def allows_arm(self) -> bool:
        return bool(
            self.resources
            & {"spine", "head", "left_arm", "right_arm", "grippers"}
        )


@dataclass(frozen=True)
class ExecutorTaskPlan:
    """One task's ordered compatibility plan."""

    task_id: int
    stages: tuple[LegacyStageSpec, ...]
    terminal_policy: TerminalPolicy = TerminalPolicy.STOP_IMMEDIATELY
    version: str = "executor-v1"

    def __post_init__(self) -> None:
        if self.task_id <= 0:
            raise ValueError("task_id must be positive")
        if not self.stages:
            raise ValueError("task plan must contain at least one stage")
        stage_values = [item.stage for item in self.stages]
        if len(stage_values) != len(set(stage_values)):
            raise ValueError(f"task {self.task_id} contains duplicate stages")

    def stage_at(self, index: int) -> LegacyStageSpec | None:
        if index < 0 or index >= len(self.stages):
            return None
        return self.stages[index]


_STAGE_RESOURCES: Mapping[TaskStage, frozenset[str]] = MappingProxyType(
    {
        # A complete ArmCommand is a persistent position hold: the controller
        # keeps publishing the last safe target across stage/task boundaries.
        # Navigation therefore owns the full manipulator as well as the base,
        # even before the executor emits a new arm target in this stage.
        TaskStage.NAVIGATE_TO_PICK: frozenset(
            {
                "base",
                "perception",
                "spine",
                "head",
                "left_arm",
                "right_arm",
                "grippers",
            }
        ),
        # Integrated Task 2/3 acquisition keeps publishing a complete held
        # ArmCommand while scanning/staging.  Complete position frames require
        # the full manipulator lease even when most fields are holds.
        TaskStage.ACQUIRE_TARGET: frozenset(
            {"perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.ALIGN_FOR_PICK: frozenset(
            {"base", "perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.GRASP: frozenset(
            {"perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.LIFT: frozenset(
            {"spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.TRANSPORT: frozenset(
            {"base", "perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.ALIGN_FOR_PLACE: frozenset(
            {"base", "perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.PLACE: frozenset(
            {"perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.VERIFY_PLACE: frozenset(
            {"perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
        TaskStage.RETURN_TO_END: frozenset(
            {"base", "perception", "spine", "head", "left_arm", "right_arm", "grippers"}
        ),
    }
)


def _plan(
    task_id: int,
    *,
    terminal_policy: TerminalPolicy = TerminalPolicy.STOP_IMMEDIATELY,
) -> ExecutorTaskPlan:
    specs = []
    for stage in TASK_STAGE_SEQUENCE:
        specs.append(
            LegacyStageSpec(
                stage=stage,
                resources=_STAGE_RESOURCES[stage],
                recovery_policy=f"{stage.value}_bounded_recovery",
                irreversible=stage is TaskStage.PLACE,
                cleanup=stage is TaskStage.RETURN_TO_END,
            )
        )
    return ExecutorTaskPlan(
        task_id=task_id,
        stages=tuple(specs),
        terminal_policy=terminal_policy,
    )


def build_executor_task_plans() -> dict[int, ExecutorTaskPlan]:
    """Return independent plans preserving the current physical stage order.

    Task 3 completes its already-running local sequence after the referee has
    accepted the terminal placement.  This is expressed as plan metadata
    rather than a scheduler check for ``task_id == 3``.
    """

    return {
        1: _plan(1),
        2: _plan(2),
        3: _plan(3, terminal_policy=TerminalPolicy.COMPLETE_ACTIVE_SEQUENCE),
    }


__all__ = [
    "ExecutorTaskPlan",
    "LegacyStageSpec",
    "TerminalPolicy",
    "build_executor_task_plans",
]
