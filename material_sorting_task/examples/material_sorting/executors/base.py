"""Common contracts for task-specific competition executors.

The top-level competition controller owns task ordering and attempt accounting.
An executor owns the concrete work for one task.  A stage must return
``FAILED`` only after the current competition attempt has been physically
settled (for example, after returning to the end zone or after a scored drop).
Transient perception, planning, or grasp errors should be recovered inside the
executor and continue returning ``RUNNING``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class TaskStage(Enum):
    """Ordered stages shared by all three competition tasks."""

    NAVIGATE_TO_PICK = "navigate_to_pick"
    ACQUIRE_TARGET = "acquire_target"
    ALIGN_FOR_PICK = "align_for_pick"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    ALIGN_FOR_PLACE = "align_for_place"
    PLACE = "place"
    VERIFY_PLACE = "verify_place"
    RETURN_TO_END = "return_to_end"


TASK_STAGE_SEQUENCE: tuple[TaskStage, ...] = tuple(TaskStage)


class StageStatus(Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TargetObservation:
    """Stable world-frame target position produced by the perception node."""

    color: str
    position_world: tuple[float, float, float]
    received_at_s: float


@dataclass(frozen=True)
class ExecutionContext:
    """Read-only runtime inputs supplied by ``client_task.py``."""

    now_s: float
    instruction: Mapping[str, Any]
    task_index: int
    attempt: int
    odometry: Any = None
    joint_states: Any = None
    target_observations: Mapping[str, TargetObservation] = field(default_factory=dict)
    referee_gameinfo: Mapping[str, Any] = field(default_factory=dict)
    referee_taskinfo: str = ""
    score: int = 0


@dataclass(frozen=True)
class StageResult:
    """One executor update returned to the orchestration layer.

    ``controls_base`` must be true only when the command in ``base_linear_x``
    and ``base_angular_z`` is valid for this control cycle.  ``client_task.py``
    owns the ROS publisher.  When no command is supplied it publishes zero
    velocity as a safety fallback.
    """

    status: StageStatus
    message: str = ""
    controls_base: bool = False
    base_linear_x: float = 0.0
    base_angular_z: float = 0.0

    @classmethod
    def running(
        cls,
        message: str = "",
        *,
        base_command: tuple[float, float] | None = None,
    ) -> "StageResult":
        if base_command is None:
            return cls(StageStatus.RUNNING, message, False, 0.0, 0.0)
        linear_x, angular_z = base_command
        return cls(
            StageStatus.RUNNING,
            message,
            True,
            float(linear_x),
            float(angular_z),
        )

    @classmethod
    def succeeded(cls, message: str = "") -> "StageResult":
        return cls(StageStatus.SUCCEEDED, message, False, 0.0, 0.0)

    @classmethod
    def failed(cls, message: str) -> "StageResult":
        return cls(StageStatus.FAILED, message, False, 0.0, 0.0)

    @classmethod
    def blocked(cls, message: str) -> "StageResult":
        return cls(StageStatus.BLOCKED, message, False, 0.0, 0.0)


class TaskExecutor(Protocol):
    """Interface implemented by each task-specific executor."""

    task_id: int
    name: str

    def reset(self) -> None:
        """Reset internal software state without resetting the physical scene."""

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        """Prepare a newly entered stage."""

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        """Advance the active stage by one non-blocking control cycle."""

    def cancel(self, reason: str) -> None:
        """Stop executor-owned commands and enter a safe state."""


class PlaceholderTaskExecutor:
    """Safe placeholder used until a real task executor is connected."""

    task_id = 0
    name = "placeholder"

    def __init__(self) -> None:
        self.active_stage: TaskStage | None = None

    def reset(self) -> None:
        self.active_stage = None

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        self.active_stage = stage

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        return StageResult.blocked(
            f"task {self.task_id} executor is not implemented at stage={stage.value}"
        )

    def cancel(self, reason: str) -> None:
        self.active_stage = None
