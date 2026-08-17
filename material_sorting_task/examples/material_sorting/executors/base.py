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
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from control_types import ArmCommand


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
    RETRYABLE_FAILURE = "retryable_failure"


@dataclass(frozen=True)
class TargetObservation:
    """Stable world-frame target position produced by the perception node."""

    color: str
    position_world: tuple[float, float, float]
    received_at_s: float
    orientation: str | None = None
    score: float = 0.0
    # ``mask_cloud_cuboid`` means the detector had enough same-colour RGB-D
    # points to fit the complete box.  ``bbox_depth_center`` is the visible
    # surface fallback and is deliberately not suitable for a final shelf
    # grasp lock.  Older producers may leave this unset.
    quality: str | None = None


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
    grasp_confirmed: bool = False
    unsafe_collision: bool = False


@dataclass(frozen=True)
class StageResult:
    """One executor update returned to the orchestration layer.

    ``controls_base`` must be true only when the command in ``base_linear_x``
    and ``base_angular_z`` is valid for this control cycle.  ``client_task.py``
    owns the ROS publisher.  When no command is supplied it publishes zero
    velocity as a safety fallback.

    ``failure_code`` carries the structured ``scheduler.models.FailureCode``
    when an executor opts into the v2 recovery bridge.  It is deliberately
    typed as ``Any`` here: importing the scheduler package from this module
    would create an import cycle (scheduler.engine imports executors.base).
    The scheduler engine performs the real type validation at the
    orchestration boundary; legacy/shadow controllers ignore the field, so
    annotating a failure can never change legacy behaviour.
    """

    status: StageStatus
    message: str = ""
    controls_base: bool = False
    base_linear_x: float = 0.0
    base_angular_z: float = 0.0
    controls_arm: bool = False
    arm_command: ArmCommand | None = None
    failure_code: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def running(
        cls,
        message: str = "",
        *,
        base_command: tuple[float, float] | None = None,
        arm_command: ArmCommand | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        linear_x, angular_z = (0.0, 0.0) if base_command is None else base_command
        return cls(
            status=StageStatus.RUNNING,
            message=message,
            controls_base=base_command is not None,
            base_linear_x=float(linear_x),
            base_angular_z=float(angular_z),
            controls_arm=arm_command is not None,
            arm_command=arm_command,
            metadata=metadata or {},
        )

    @classmethod
    def succeeded(
        cls,
        message: str = "",
        *,
        arm_command: ArmCommand | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        return cls(
            status=StageStatus.SUCCEEDED,
            message=message,
            controls_arm=arm_command is not None,
            arm_command=arm_command,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        message: str,
        *,
        failure_code: Any = None,
    ) -> "StageResult":
        return cls(
            status=StageStatus.FAILED,
            message=message,
            failure_code=failure_code,
        )

    @classmethod
    def blocked(
        cls,
        message: str,
        *,
        arm_command: ArmCommand | None = None,
        failure_code: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        return cls(
            status=StageStatus.BLOCKED,
            message=message,
            controls_arm=arm_command is not None,
            arm_command=arm_command,
            failure_code=failure_code,
            metadata=metadata or {},
        )

    @classmethod
    def retryable_failure(
        cls,
        failure_code: Any,
        message: str = "",
        *,
        arm_command: ArmCommand | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        """Structured, bounded-recoverable failure.

        Legacy and shadow controllers treat this exactly like the historical
        BLOCKED sites it replaces, so a newly annotated executor failure
        cannot change the validated legacy trace.
        """
        return cls(
            status=StageStatus.RETRYABLE_FAILURE,
            message=message,
            controls_arm=arm_command is not None,
            arm_command=arm_command,
            failure_code=failure_code,
            metadata=metadata or {},
        )

    @classmethod
    def fatal(
        cls,
        failure_code: Any,
        message: str = "",
        *,
        arm_command: ArmCommand | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        """Hard safety failure that must end in SAFE_HOLD, never recovery.

        The status stays BLOCKED so the legacy controller keeps its existing
        safe-stop behaviour; the v2 engine reads the structured code and
        escalates the same result to SAFE_HOLD.
        """
        return cls(
            status=StageStatus.BLOCKED,
            message=message,
            controls_arm=arm_command is not None,
            arm_command=arm_command,
            failure_code=failure_code,
            metadata=metadata or {},
        )


def apply_detection_epoch_decisions(
    decisions: Mapping[str, str],
    *,
    reset: Callable[[list[str]], None],
    log: Callable[[str], None],
) -> list[str]:
    """Apply one executor's detection-epoch policy without ROS imports.

    Returns the colours that were reset.  Unknown actions are logged and
    ignored so a malformed executor policy can never clear production
    detection history.
    """
    reset_colors = []
    for color, action in decisions.items():
        color = str(color).strip().lower()
        if not color:
            continue
        if action == "reset":
            reset_colors.append(color)
        elif action == "keep":
            log(f"task detection epoch retains {color} RGB-D history")
        else:
            log(
                f"invalid detection epoch action {action!r} for {color!r}; ignored"
            )
    if reset_colors:
        reset(sorted(set(reset_colors)))
    return reset_colors


def resolve_executor_for_task_index(
    executors: Mapping[int, Any],
    instructions: list[Mapping[str, Any]],
    task_index: int,
) -> Any:
    """Resolve the active executor through the instruction's formal task id.

    Controller ``task_index`` is zero-based while executor dictionaries are
    keyed by Server task ids (1/2/3).  Keeping that conversion here makes the
    ROS client testable without importing rclpy and prevents index/id mixups.
    """
    index = int(task_index)
    if index < 0 or index >= len(instructions):
        raise IndexError(f"task index {index} is outside configured instructions")
    instruction = instructions[index]
    try:
        task_id = int(instruction["task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("active instruction has no valid task id") from exc
    try:
        return executors[task_id]
    except KeyError as exc:
        raise KeyError(f"no executor configured for task id {task_id}") from exc


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
