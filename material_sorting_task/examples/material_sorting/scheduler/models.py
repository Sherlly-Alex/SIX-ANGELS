"""ROS-free contracts shared by the task scheduler.

The classes in this module intentionally describe *what* an action did rather
than how a concrete executor is implemented.  That keeps the scheduler usable
in unit tests, replay tools and training environments without importing ROS or
MuJoCo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable


class ActionStatus(Enum):
    """Terminal and non-terminal outcomes of one scheduled action update."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    ATTEMPT_FAILED = "attempt_failed"
    BLOCKED = "blocked"
    CANCELED = "canceled"
    FATAL = "fatal"


class FailureCode(Enum):
    """Machine-readable failures; recovery must never parse message strings."""

    INPUT_STALE = "input_stale"
    TARGET_LOST = "target_lost"
    NAV_NO_PATH = "nav_no_path"
    NAV_STUCK = "nav_stuck"
    ALIGNMENT_FAILED = "alignment_failed"
    IK_FAILED = "ik_failed"
    SINGLE_SIDE_CONTACT = "single_side_contact"
    GRASP_NOT_CONFIRMED = "grasp_not_confirmed"
    EFFORT_SOFT_LIMIT = "effort_soft_limit"
    EFFORT_HARD_LIMIT = "effort_hard_limit"
    OBJECT_DROPPED = "object_dropped"
    OBJECT_RELEASED = "object_released"
    PLACEMENT_UNCERTAIN = "placement_uncertain"
    UNSAFE_COLLISION = "unsafe_collision"
    REFEREE_DESYNC = "referee_desync"
    RESOURCE_CONFLICT = "resource_conflict"
    COMMAND_NON_FINITE = "command_non_finite"
    COMMAND_EXPIRED = "command_expired"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    ACTION_TIMEOUT = "action_timeout"
    INTERNAL_ERROR = "internal_error"


class Resource(Enum):
    """Exclusively owned scheduler resources."""

    BASE = "base"
    SPINE = "spine"
    HEAD = "head"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    GRIPPERS = "grippers"
    PERCEPTION = "perception"


class ArmCommandMode(Enum):
    """Whether an arm command moves actuators or preserves a safe pose."""

    NONE = "none"
    MOVE = "move"
    HOLD = "hold"

    # Readable compatibility aliases for adapters.
    POSITION = "move"
    HOLD_LAST = "hold"


class PayloadMode(Enum):
    """Coarse payload envelope used by safety and costmap adapters."""

    EMPTY = "empty"
    ARMS_OPEN = "arms_open"
    CARRYING = "carrying"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BaseCommand:
    """Planar velocity command in SI units."""

    linear_x: float = 0.0
    angular_z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear_x", float(self.linear_x))
        object.__setattr__(self, "angular_z", float(self.angular_z))

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.linear_x) and math.isfinite(self.angular_z)

    @classmethod
    def zero(cls) -> "BaseCommand":
        return cls(0.0, 0.0)


@dataclass(frozen=True)
class CommandFrame:
    """Commands emitted by exactly one scheduler step.

    ``arm_command`` deliberately uses ``Any``.  Concrete actuator command
    types live in the executor layer and importing them here would couple the
    scheduler to the current robot implementation.
    """

    owner_step_id: str
    base_command: BaseCommand | tuple[float, float] | None = None
    arm_command: Any = None
    arm_mode: ArmCommandMode = ArmCommandMode.NONE
    valid_until_s: float | None = None
    resources: frozenset[Resource] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.owner_step_id or not self.owner_step_id.strip():
            raise ValueError("owner_step_id must be non-empty")
        base = self.base_command
        if base is not None and not isinstance(base, BaseCommand):
            try:
                linear_x, angular_z = base
            except (TypeError, ValueError) as exc:
                raise TypeError("base_command must be BaseCommand or a pair") from exc
            base = BaseCommand(linear_x, angular_z)
            object.__setattr__(self, "base_command", base)
        object.__setattr__(self, "resources", frozenset(self.resources))
        if self.valid_until_s is not None:
            object.__setattr__(self, "valid_until_s", float(self.valid_until_s))
        if self.arm_command is not None and self.arm_mode is ArmCommandMode.NONE:
            object.__setattr__(self, "arm_mode", ArmCommandMode.MOVE)

    @property
    def required_resources(self) -> frozenset[Resource]:
        required = set(self.resources)
        if self.base_command is not None:
            required.add(Resource.BASE)
        if self.arm_command is not None or self.arm_mode is not ArmCommandMode.NONE:
            if not any(
                resource in required
                for resource in (
                    Resource.SPINE,
                    Resource.HEAD,
                    Resource.LEFT_ARM,
                    Resource.RIGHT_ARM,
                    Resource.GRIPPERS,
                )
            ):
                required.update(
                    {
                        Resource.SPINE,
                        Resource.HEAD,
                        Resource.LEFT_ARM,
                        Resource.RIGHT_ARM,
                        Resource.GRIPPERS,
                    }
                )
        return frozenset(required)

    @classmethod
    def stopped(cls, owner_step_id: str, *, valid_until_s: float | None = None) -> "CommandFrame":
        return cls(
            owner_step_id=owner_step_id,
            base_command=BaseCommand.zero(),
            valid_until_s=valid_until_s,
            resources=frozenset({Resource.BASE}),
        )


@dataclass(frozen=True)
class ActionResult:
    """One non-blocking update from a scheduled action."""

    status: ActionStatus
    failure_code: FailureCode | None = None
    message: str = ""
    command: CommandFrame | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def terminal(self) -> bool:
        return self.status is not ActionStatus.RUNNING

    @property
    def failure(self) -> FailureCode | None:
        """Short compatibility alias for policy and replay adapters."""

        return self.failure_code

    @classmethod
    def running(
        cls,
        message: str = "",
        *,
        command: CommandFrame | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        return cls(ActionStatus.RUNNING, message=message, command=command, metadata=metadata or {})

    @classmethod
    def succeeded(
        cls,
        message: str = "",
        *,
        command: CommandFrame | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        return cls(ActionStatus.SUCCEEDED, message=message, command=command, metadata=metadata or {})

    @classmethod
    def retryable_failure(
        cls,
        failure_code: FailureCode,
        message: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        return cls(
            ActionStatus.RETRYABLE_FAILURE,
            failure_code=failure_code,
            message=message,
            metadata=metadata or {},
        )

    @classmethod
    def attempt_failed(
        cls,
        failure_code: FailureCode,
        message: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        return cls(
            ActionStatus.ATTEMPT_FAILED,
            failure_code=failure_code,
            message=message,
            metadata=metadata or {},
        )

    @classmethod
    def fatal(cls, failure_code: FailureCode, message: str = "") -> "ActionResult":
        return cls(ActionStatus.FATAL, failure_code=failure_code, message=message)

    @classmethod
    def blocked(cls, failure_code: FailureCode, message: str = "") -> "ActionResult":
        return cls(ActionStatus.BLOCKED, failure_code=failure_code, message=message)

    @classmethod
    def canceled(cls, message: str = "") -> "ActionResult":
        return cls(ActionStatus.CANCELED, message=message)


@runtime_checkable
class ScheduledAction(Protocol):
    """Minimal action interface understood by scheduler and recovery layers."""

    def tick(self, context: Any) -> ActionResult:
        """Advance at most one control update."""

    def cancel(self, reason: str) -> None:
        """Stop action-owned output."""


@dataclass(frozen=True)
class StepSpec:
    """Static definition of a node in a task plan."""

    id: str
    action_factory: Callable[[], ScheduledAction] | None = None
    resources: frozenset[Resource] = field(default_factory=frozenset)
    timeout_s: float = 30.0
    next_on_success: str | None = None
    recovery_policy: str | None = None
    irreversible: bool = False
    cleanup: bool = False
    legacy_stage: Any = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("step id must be non-empty")
        timeout_s = float(self.timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and greater than zero")
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "resources", frozenset(self.resources))
        if any(not isinstance(resource, Resource) for resource in self.resources):
            raise TypeError("StepSpec resources must contain Resource values")
        if self.action_factory is not None and not callable(self.action_factory):
            raise TypeError("action_factory must be callable")


@dataclass(frozen=True)
class TaskPlan:
    """Validated, ordered step graph for one formal competition task."""

    task_id: int
    entry_step: str
    steps: Mapping[str, StepSpec] | Iterable[StepSpec]

    def __post_init__(self) -> None:
        task_id = int(self.task_id)
        if task_id <= 0:
            raise ValueError("task_id must be positive")
        object.__setattr__(self, "task_id", task_id)

        if isinstance(self.steps, Mapping):
            ordered: dict[str, StepSpec] = {}
            for key, step in self.steps.items():
                if not isinstance(step, StepSpec):
                    raise TypeError("all task plan values must be StepSpec")
                if str(key) != step.id:
                    raise ValueError(f"step mapping key {key!r} does not match id {step.id!r}")
                ordered[step.id] = step
        else:
            ordered = {}
            for step in self.steps:
                if not isinstance(step, StepSpec):
                    raise TypeError("all task plan entries must be StepSpec")
                if step.id in ordered:
                    raise ValueError(f"duplicate step id: {step.id}")
                ordered[step.id] = step

        if not ordered:
            raise ValueError("task plan must contain at least one step")
        if self.entry_step not in ordered:
            raise ValueError(f"entry step {self.entry_step!r} does not exist")
        for step in ordered.values():
            if step.next_on_success is not None and step.next_on_success not in ordered:
                raise ValueError(
                    f"step {step.id!r} points to missing successor {step.next_on_success!r}"
                )
        object.__setattr__(self, "steps", MappingProxyType(ordered))

    def step(self, step_id: str) -> StepSpec:
        return self.steps[step_id]

    @property
    def ordered_steps(self) -> tuple[StepSpec, ...]:
        return tuple(self.steps.values())


@dataclass(frozen=True)
class RefereeSnapshot:
    """Canonical referee state after parsing both referee topics."""

    task_ordinal: int | None = None
    task_total: int | None = None
    task_id: int | None = None
    attempts_completed: int = 0
    step: str | None = None
    score: int | None = None
    all_tasks_done: bool = False
    raw_gameinfo: str = ""
    raw_taskinfo: str = ""

    @property
    def attempt(self) -> int:
        """Compatibility alias used by the legacy controller."""

        return self.attempts_completed


@dataclass(frozen=True)
class WorldState:
    """Read-only scheduler input assembled by an outer adapter each tick."""

    now_s: float
    instruction: Mapping[str, Any] = field(default_factory=dict)
    odometry: Any = None
    joint_states: Any = None
    target_observations: Mapping[str, Any] = field(default_factory=dict)
    referee: RefereeSnapshot = field(default_factory=RefereeSnapshot)
    score: int = 0
    grasp_confirmed: bool = False
    unsafe_collision: bool = False
    input_ages_s: Mapping[str, float] = field(default_factory=dict)
    payload_mode: PayloadMode = PayloadMode.EMPTY

    def __post_init__(self) -> None:
        now_s = float(self.now_s)
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        object.__setattr__(self, "now_s", now_s)
        object.__setattr__(self, "instruction", MappingProxyType(dict(self.instruction)))
        object.__setattr__(
            self,
            "target_observations",
            MappingProxyType(dict(self.target_observations)),
        )
        object.__setattr__(self, "input_ages_s", MappingProxyType(dict(self.input_ages_s)))


__all__ = [
    "ActionResult",
    "ActionStatus",
    "ArmCommandMode",
    "BaseCommand",
    "CommandFrame",
    "FailureCode",
    "PayloadMode",
    "RefereeSnapshot",
    "Resource",
    "ScheduledAction",
    "StepSpec",
    "TaskPlan",
    "WorldState",
]
