"""Plan-driven scheduler compatible with the existing task executors.

``SchedulerEngine`` deliberately keeps ROS and policy learning outside the
control loop.  It executes immutable task plans, validates every actuator
command at the orchestration boundary, and still treats the Server referee as
the sole authority for formal task progression.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Any, Callable, Mapping, Sequence

from executors.base import (
    ArmCommand,
    ExecutionContext,
    StageResult,
    StageStatus,
    TaskExecutor,
    TaskStage,
)
from executors.scheduler_candidate import CandidateApplicationStatus
from scheduler.legacy_adapter import LegacyStageAction, RecoverableStageAction
from scheduler.models import (
    ArmCommandMode,
    BaseCommand,
    CommandFrame,
    FailureCode,
    Resource,
)
from scheduler.recovery import FATAL_SAFETY_FAILURE_CODES, RecoveryClassifier
from scheduler.plans import ExecutorTaskPlan, TerminalPolicy, build_executor_task_plans
from scheduler.referee import RefereeGateway, RefereeUpdate
from scheduler.resources import (
    BaseCommandLease,
    CommandValidationError,
    CommandValidator,
    ResourceConflictError,
    ResourceManager,
)
from scheduler.safety import SafetySupervisor


class EngineState(Enum):
    WAITING_FOR_INPUTS = "waiting_for_inputs"
    STARTING_TASK = "starting_task"
    EXECUTING_STAGE = "executing_stage"
    WAITING_FOR_REFEREE = "waiting_for_referee"
    BLOCKED = "blocked"
    FINISHED = "finished"
    SAFE_HOLD = "safe_hold"


@dataclass(frozen=True)
class EngineSnapshot:
    state: EngineState
    task_index: int
    task_id: int | None
    attempt: int
    stage: TaskStage | None
    safe_stop: bool
    controls_base: bool
    base_linear_x: float
    base_angular_z: float
    controls_arm: bool
    arm_command: ArmCommand | None
    message: str
    transition_serial: int


class SchedulerEngine:
    """Execute versioned plans while preserving the validated executor API.

    ``state_enum`` and ``snapshot_factory`` are dependency-injection points
    used by :mod:`competition_controller` so callers receive its long-standing
    public enum/dataclass types in every scheduler mode.
    """

    def __init__(
        self,
        executors: Mapping[int, TaskExecutor],
        *,
        plans: Mapping[int, ExecutorTaskPlan] | None = None,
        referee_driven: bool = True,
        max_attempts: int = 3,
        state_enum: type[Enum] = EngineState,
        snapshot_factory: Callable[..., Any] = EngineSnapshot,
        event_sink: Any = None,
        resource_manager: ResourceManager | None = None,
        command_validator: CommandValidator | None = None,
        base_command_lease: BaseCommandLease | None = None,
        safety_supervisor: SafetySupervisor | None = None,
        referee_gateway: RefereeGateway | None = None,
        decision_service: Any = None,
        candidate_provider: Any = None,
        decision_period_s: float = 0.25,
        referee_desync_limit: int = 20,
        stage_recovery_budget: int = 8,
        candidate_initial_wait_s: float = 0.10,
    ) -> None:
        missing = {1, 2, 3} - set(executors)
        if missing:
            raise ValueError(f"missing task executors: {sorted(missing)}")
        resolved_plans = dict(build_executor_task_plans() if plans is None else plans)
        missing_plans = {1, 2, 3} - set(resolved_plans)
        if missing_plans:
            raise ValueError(f"missing task plans: {sorted(missing_plans)}")

        self.executors = dict(executors)
        self.plans = resolved_plans
        self.referee_driven = bool(referee_driven)
        self.max_attempts = max(1, int(max_attempts))
        self._states = state_enum
        self._snapshot_factory = snapshot_factory
        self._event_sink = event_sink
        self.resource_manager = resource_manager or ResourceManager()
        self.command_validator = command_validator or CommandValidator(
            self.resource_manager
        )
        self.base_command_lease = base_command_lease or BaseCommandLease(0.15)
        self.safety_supervisor = safety_supervisor or SafetySupervisor()
        self.referee_gateway = referee_gateway or RefereeGateway()
        self.decision_service = decision_service
        self.candidate_provider = candidate_provider
        self.decision_period_s = float(decision_period_s)
        if not math.isfinite(self.decision_period_s) or self.decision_period_s <= 0.0:
            raise ValueError("decision_period_s must be finite and positive")
        if int(stage_recovery_budget) < 0:
            raise ValueError("stage_recovery_budget cannot be negative")
        self.stage_recovery_budget = int(stage_recovery_budget)
        self.candidate_initial_wait_s = float(candidate_initial_wait_s)
        if (
            not math.isfinite(self.candidate_initial_wait_s)
            or self.candidate_initial_wait_s < 0.0
        ):
            raise ValueError("candidate_initial_wait_s must be finite and non-negative")
        self._decision_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="scheduler-costmap")
            if decision_service is not None and candidate_provider is not None
            else None
        )
        self._decision_future: Future | None = None
        self._decision_lifecycle_lock = threading.RLock()
        self._last_decision_submit_s: float | None = None
        self.last_decision: Any = None
        self.last_candidate_application: str | None = None
        self._closing = False
        self.referee_desync_limit = max(1, int(referee_desync_limit))

        self.instructions: list[dict] = []
        self.inputs_ready = False
        self.state = self._states.WAITING_FOR_INPUTS
        self.task_index = 0
        self.attempt = 1
        self.stage_index = 0
        self._active_action: LegacyStageAction | RecoverableStageAction | None = None
        self._active_owner: str | None = None
        self._stage_started_s: float | None = None
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._arm_command: ArmCommand | None = None
        self._arm_hold_owner: str | None = None
        self._message = "waiting for validated instructions and robot inputs"
        self._transition_serial = 0
        self._wait_referee_attempts_completed = 0
        self._referee_update: RefereeUpdate | None = None
        self._last_consistent_referee_update: RefereeUpdate | None = None
        self._referee_desync_count = 0
        self._terminal_referee_pending = False

    @property
    def task_id(self) -> int | None:
        if self.task_index < 0 or self.task_index >= len(self.instructions):
            return None
        value = self.instructions[self.task_index].get("task")
        return int(value) if value is not None else None

    @property
    def stage_spec(self):
        plan = self._plan()
        return None if plan is None else plan.stage_at(self.stage_index)

    @property
    def stage(self) -> TaskStage | None:
        if self.state not in (
            self._states.STARTING_TASK,
            self._states.EXECUTING_STAGE,
        ):
            return None
        spec = self.stage_spec
        return None if spec is None else spec.stage

    def configure(self, instructions: Sequence[Mapping]) -> bool:
        if self._closing:
            raise RuntimeError("scheduler is closed")
        normalized = sorted(
            (dict(item) for item in instructions),
            key=lambda item: int(item.get("task", 0)),
        )
        task_ids = [item.get("task") for item in normalized]
        if task_ids != [1, 2, 3]:
            raise ValueError(f"expected task ids [1, 2, 3], received {task_ids}")
        if normalized == self.instructions:
            return False
        if self.instructions and self.state is not self._states.WAITING_FOR_INPUTS:
            raise RuntimeError("instructions changed after task execution started")

        for executor in self.executors.values():
            configure = getattr(executor, "configure_instructions", None)
            if callable(configure):
                configure(normalized)

        self.instructions = normalized
        self.task_index = 0
        self.attempt = 1
        self.stage_index = 0
        self._active_action = None
        self._release_active_resources()
        self._stage_started_s = None
        self._release_arm_hold_lease()
        self._arm_command = None
        self.referee_gateway.reset()
        self._referee_update = None
        self._last_consistent_referee_update = None
        self._referee_desync_count = 0
        self._terminal_referee_pending = False
        self.last_decision = None
        self.last_candidate_application = None
        self._last_decision_submit_s = None
        self._transition(
            self._states.WAITING_FOR_INPUTS,
            "three validated task instructions configured",
        )
        return True

    def set_inputs_ready(self, ready: bool) -> None:
        self.inputs_ready = bool(ready)

    def tick(self, context: ExecutionContext):
        """Advance at most one externally visible transition."""
        if self._closing:
            return self.snapshot()
        if self.state in (
            self._states.FINISHED,
            self._states.SAFE_HOLD,
            self._states.BLOCKED,
        ):
            return self.snapshot()

        violation = self.safety_supervisor.check(context)
        if violation.must_stop:
            self.stop(violation.message or "safety supervisor stop")
            return self.snapshot()

        self._referee_update = self.referee_gateway.observe(
            context.referee_gameinfo,
            context.referee_taskinfo,
        )
        if self._referee_update.desynchronised:
            self._referee_desync_count += 1
            self._emit_structured_event(
                "referee_desync",
                context,
                details={
                    "reasons": list(self._referee_update.desync_reasons),
                    "count": self._referee_desync_count,
                },
            )
            # One inconsistent topic frame is logged and tolerated.  A
            # repeated inconsistency is fail-closed because task ownership can
            # no longer be proven from the Server topics.
            if self._referee_desync_count >= self.referee_desync_limit:
                self.stop(
                    "referee topics remained desynchronised: "
                    + "; ".join(self._referee_update.desync_reasons)
                )
                return self.snapshot()
        else:
            self._referee_desync_count = 0
            self._last_consistent_referee_update = self._referee_update

        if self.referee_driven and self._referee_finished(context):
            if self._may_finish_active_action_then_cleanup():
                self._terminal_referee_pending = True
            else:
                self._finish("referee reported all tasks finished")
                return self.snapshot()

        if self.state is self._states.WAITING_FOR_INPUTS:
            if not self.instructions or not self.inputs_ready:
                return self.snapshot()
            if (
                self.referee_driven
                and self._referee_update is not None
                and self._referee_update.desynchronised
                and self._last_consistent_referee_update is None
            ):
                return self.snapshot()
            self._sync_start_from_referee(context)
            self._transition(
                self._states.STARTING_TASK,
                f"starting task {self.task_id} attempt {self.attempt}",
            )
            return self.snapshot()

        if self.state is self._states.STARTING_TASK:
            executor = self._executor()
            try:
                executor.reset()
            except Exception as exc:
                self.stop(
                    f"task {self.task_id} reset failed: {type(exc).__name__}: {exc}"
                )
                return self.snapshot()
            self.stage_index = 0
            self._active_action = None
            self._stage_started_s = None
            self._controls_base = False
            stage = self.stage
            if stage is None:
                self.stop(f"task {self.task_id} has an empty or invalid plan")
                return self.snapshot()
            self._transition(
                self._states.EXECUTING_STAGE,
                f"task {self.task_id} entering {stage.value}",
            )
            return self.snapshot()

        if self.state is self._states.EXECUTING_STAGE:
            return self._tick_stage(context)

        if self.state is self._states.WAITING_FOR_REFEREE:
            self._tick_waiting_for_referee(context)
            return self.snapshot()

        self.stop(f"unhandled scheduler state {self.state.value}")
        return self.snapshot()

    def _tick_stage(self, context: ExecutionContext):
        executor = self._executor()
        spec = self.stage_spec
        if spec is None:
            self.stop("invalid empty execution stage")
            return self.snapshot()

        if self._active_action is None:
            action = LegacyStageAction(executor=executor, stage=spec.stage)
            if spec.recovery_policy and not spec.irreversible:
                # Structured retryable failures are consumed by a bounded
                # deterministic recovery wrapper.  Without executor-provided
                # recovery actions the recovery is a bounded step re-entry
                # (Nav2 level L2); irreversible steps are never wrapped.
                action = RecoverableStageAction(
                    action,
                    classifier=RecoveryClassifier(),
                    recovery_factory=self._recovery_factory_for(executor, spec),
                    max_total_recoveries=self.stage_recovery_budget,
                )
            owner = self._owner_id(spec.stage)
            try:
                self._release_arm_hold_lease()
                self.resource_manager.acquire(
                    self._resource_set(spec),
                    owner=owner,
                )
                self._update_decision_sidecar(context, spec, force=True)
                if self.state is self._states.SAFE_HOLD:
                    self._release_owner(owner)
                    return self.snapshot()
                action.enter(context)
            except ResourceConflictError as exc:
                self._release_owner(owner)
                self.stop(
                    f"task {self.task_id} resource conflict at "
                    f"stage={spec.stage.value}: {exc}"
                )
                return self.snapshot()
            except Exception as exc:
                self._release_owner(owner)
                self.stop(
                    f"task {self.task_id} failed to enter stage={spec.stage.value}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return self.snapshot()
            self._active_action = action
            self._active_owner = owner
            self._stage_started_s = float(context.now_s)
            self._controls_base = False
            self._bump_message(
                f"task {self.task_id} attempt {self.attempt} stage={spec.stage.value}"
            )
            return self.snapshot()

        self._update_decision_sidecar(context, spec)
        if self.state is self._states.SAFE_HOLD:
            return self.snapshot()
        if self._should_wait_for_initial_candidate(context, spec):
            self.base_command_lease.revoke()
            self._controls_base = False
            self._base_linear_x = 0.0
            self._base_angular_z = 0.0
            self._message = (
                f"task {self.task_id} waiting up to "
                f"{self.candidate_initial_wait_s:.2f}s for initial safe candidate"
            )
            return self.snapshot()

        if (
            spec.timeout_s is not None
            and self._stage_started_s is not None
            and float(context.now_s) - self._stage_started_s > spec.timeout_s
        ):
            cancel_error = self._cancel_action(
                f"stage timeout after {spec.timeout_s:.2f}s"
            )
            if cancel_error:
                self.stop(f"executor cancel failed after timeout: {cancel_error}")
                return self.snapshot()
            self._controls_base = False
            self._finish_local_attempt(
                context,
                succeeded=False,
                message=f"stage {spec.stage.value} exceeded timeout",
            )
            return self.snapshot()

        try:
            result = self._active_action.tick(context)
        except Exception as exc:
            self.stop(
                f"task {self.task_id} executor error at stage={spec.stage.value}: "
                f"{type(exc).__name__}: {exc}"
            )
            return self.snapshot()

        try:
            self._validate_result(spec, result, context)
        except (CommandValidationError, ValueError, TypeError) as exc:
            self.stop(
                f"task {self.task_id} invalid command at stage={spec.stage.value}: "
                f"{exc}"
            )
            return self.snapshot()

        self._controls_base = bool(result.controls_base)
        if self._controls_base:
            leased = self.base_command_lease.resolve(float(context.now_s))
            self._base_linear_x = leased.linear_x
            self._base_angular_z = leased.angular_z
        else:
            self.base_command_lease.revoke()
            self._base_linear_x = 0.0
            self._base_angular_z = 0.0
        if result.controls_arm:
            self._arm_command = result.arm_command

        if result.status is StageStatus.RUNNING:
            recovery_name = result.metadata.get("recovery_requested")
            if recovery_name:
                self._emit_structured_event(
                    "step_recovery",
                    context,
                    message=result.message,
                    details=dict(result.metadata),
                )
            if result.message:
                self._message = result.message
            return self.snapshot()
        if result.status is StageStatus.RETRYABLE_FAILURE:
            # An unwrapped step (no recovery policy or an irreversible
            # stage) cannot retry: fail closed with the legacy BLOCKED
            # semantics and the structured code recorded.
            code = self._result_failure_code(result)
            code_label = self._failure_code_label(code)
            cancel_error = self._cancel_action(result.message)
            if cancel_error:
                self.stop(
                    f"executor cancel failed on retryable failure: {cancel_error}"
                )
                return self.snapshot()
            self._controls_base = False
            self._emit_structured_event(
                "step_failed",
                context,
                message=result.message,
                details={"failure_code": code_label, "retryable": True, "recovered": False},
            )
            self._transition(
                self._states.BLOCKED,
                f"stage {spec.stage.value} failed without a recovery path: "
                f"{code_label}: {result.message}",
            )
            return self.snapshot()
        if result.status is StageStatus.BLOCKED:
            code = self._result_failure_code(result)
            if code in FATAL_SAFETY_FAILURE_CODES:
                # Hard effort limits, collisions and internal errors must
                # end in SAFE_HOLD; they must never wait for a referee.
                cancel_error = self._cancel_action(result.message)
                if cancel_error:
                    self.stop(
                        f"executor cancel failed on fatal failure: {cancel_error}"
                    )
                    return self.snapshot()
                self._emit_structured_event(
                    "step_failed",
                    context,
                    message=result.message,
                    details={"failure_code": self._failure_code_label(code), "fatal": True},
                )
                self.stop(
                    f"task {self.task_id} fatal structured failure {code.value} "
                    f"at stage={spec.stage.value}: {result.message}"
                )
                return self.snapshot()
            cancel_error = self._cancel_action(result.message)
            if cancel_error:
                self.stop(f"executor cancel failed while blocking: {cancel_error}")
                return self.snapshot()
            self._controls_base = False
            self._transition(self._states.BLOCKED, result.message)
            return self.snapshot()
        if result.status is StageStatus.FAILED:
            cancel_error = self._cancel_action(result.message)
            if cancel_error:
                self.stop(f"executor cancel failed after failure: {cancel_error}")
                return self.snapshot()
            self._controls_base = False
            self._emit_structured_event(
                "step_failed",
                context,
                message=result.message,
                details={"failure_code": self._failure_code_label(self._result_failure_code(result))},
            )
            self._finish_local_attempt(context, succeeded=False, message=result.message)
            return self.snapshot()

        plan = self._plan()
        if self._terminal_referee_pending:
            cancel_error = self._cancel_action(
                "terminal referee cleanup transition"
            )
            if cancel_error:
                self.stop(f"executor cancel failed before cleanup: {cancel_error}")
                return self.snapshot()
            cleanup_index = self._next_cleanup_index()
            if cleanup_index is None:
                self._finish("referee terminal cleanup completed")
                return self.snapshot()
            self.stage_index = cleanup_index
            self._controls_base = False
            next_stage = self.stage
            self._bump_message(
                "referee reported all tasks finished; "
                f"entering safe cleanup {next_stage.value}"
            )
            return self.snapshot()
        if plan is not None and self.stage_index + 1 < len(plan.stages):
            cancel_error = self._cancel_action("stage action sequence complete")
            if cancel_error:
                self.stop(f"executor cancel failed at stage completion: {cancel_error}")
                return self.snapshot()
            self.stage_index += 1
            self._controls_base = False
            next_stage = self.stage
            self._bump_message(
                f"task {self.task_id} entering {next_stage.value}: {result.message}"
            )
            return self.snapshot()

        cancel_error = self._cancel_action("task action sequence complete")
        if cancel_error:
            self.stop(f"executor cancel failed at task completion: {cancel_error}")
            return self.snapshot()
        self._controls_base = False
        self._finish_local_attempt(context, succeeded=True, message=result.message)
        return self.snapshot()

    def stop(self, reason: str = "client stop requested") -> None:
        if self._active_action is not None:
            try:
                self._active_action.cancel(reason)
            except Exception:
                pass
            self._active_action = None
        else:
            task_id = self.task_id
            if task_id in self.executors:
                try:
                    self.executors[task_id].cancel(reason)
                except Exception:
                    pass
        self._release_active_resources()
        self._ensure_arm_hold_lease()
        self.base_command_lease.revoke()
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._transition(self._states.SAFE_HOLD, reason)

    def _finish(self, reason: str) -> None:
        if self._active_action is not None:
            try:
                self._active_action.cancel(reason)
            except Exception:
                pass
            self._active_action = None
        self._release_active_resources()
        self._ensure_arm_hold_lease()
        self.base_command_lease.revoke()
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._transition(self._states.FINISHED, reason)

    def snapshot(self):
        return self._snapshot_factory(
            state=self.state,
            task_index=self.task_index,
            task_id=self.task_id,
            attempt=self.attempt,
            stage=self.stage,
            safe_stop=not self._controls_base,
            controls_base=self._controls_base,
            base_linear_x=self._base_linear_x,
            base_angular_z=self._base_angular_z,
            controls_arm=self._arm_command is not None,
            arm_command=self._arm_command,
            message=self._message,
            transition_serial=self._transition_serial,
        )

    def decide_candidates(self, candidates, **kwargs):
        """Rank a bounded macro-action set through the configured policy path.

        Selection is deliberately separate from ``tick``: existing executors
        keep ownership of real motion until a caller explicitly maps the
        returned candidate to a validated executor action.
        """
        if self.decision_service is None:
            raise RuntimeError("scheduler decision service is not configured")
        return self.decision_service.decide(candidates, **kwargs)

    def _decision_key(self) -> tuple[int | None, int, int, str | None]:
        return (
            self.task_id,
            self.attempt,
            self.stage_index,
            None if self.stage is None else self.stage.value,
        )

    def _compute_stage_decision(self, key, context, spec):
        nominal_goal = self._probe_nominal_goal(context, spec)
        try:
            batch = self.candidate_provider.build(
                context,
                spec,
                nominal_goal=nominal_goal,
            )
        except TypeError as exc:
            # Duck-typed providers from earlier batches may not accept the
            # nominal-goal keyword; they keep their own base-goal logic.
            if "nominal_goal" not in str(exc):
                raise
            batch = self.candidate_provider.build(context, spec)
        if batch is None:
            return key, None
        constraints = dict(batch.constraints or {})
        # Stage resources are acquired immediately before the async sidecar is
        # submitted.  During the first submission ``_active_owner`` is not yet
        # published, so verify the deterministic stage owner directly.
        owner = self._active_owner or self._owner_id(spec.stage)
        constraints["resource_available"] = bool(
            owner is not None
            and self.resource_manager.owns(owner, self._resource_set(spec))
        )
        if getattr(spec, "stage", spec) is TaskStage.TRANSPORT:
            held_center_base, held_half_width_m = self._probe_held_geometry(context)
        else:
            held_center_base, held_half_width_m = None, None
        with self._decision_lifecycle_lock:
            if self._closing:
                return key, None
            outcome = self.decision_service.decide(
                batch.candidates,
                now_s=float(context.now_s),
                world_state=batch.world_state,
                costmap=batch.costmap,
                start_pose=batch.start_pose,
                constraints=constraints,
                footprint_mode=batch.footprint_mode,
                held_center_base=held_center_base,
                held_half_width_m=held_half_width_m,
            )
        return key, outcome

    def _probe_nominal_goal(
        self,
        context: ExecutionContext,
        spec: Any,
    ) -> tuple[float, float, float] | None:
        """Best-effort nominal stand from an opt-in executor hook.

        Candidates are offset around the executor's own current stand
        instead of a provider-side approximation.  A missing hook or a hook
        exception falls back to the provider's computed goal, so the
        sidecar keeps its audit-only grading behaviour unchanged.
        """
        stage = getattr(spec, "stage", spec)
        hook = getattr(self._executor(), "scheduler_nominal_goal", None)
        if not callable(hook):
            return None
        try:
            goal = hook(stage, context)
        except Exception:
            return None
        if goal is None:
            return None
        try:
            x, y, yaw = (float(value) for value in goal)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return None
        return (x, y, yaw)

    def _probe_held_geometry(
        self,
        context: ExecutionContext,
    ) -> tuple[tuple[float, float, float] | None, float | None]:
        """Best-effort measured held-object envelope for transport scoring.

        The probe is read-only with respect to executors.  A missing hook, a
        hook exception, or malformed geometry degrades to the generic
        TRANSIT_CARRY footprint for scoring only; it never alters commands.
        """
        hook = getattr(self._executor(), "held_object_geometry", None)
        if not callable(hook):
            return None, None
        try:
            geometry = hook(context)
        except Exception:
            return None, None
        if geometry is None:
            return None, None
        try:
            center = tuple(float(value) for value in geometry.center_base)
            half = float(geometry.half_width_m)
        except (AttributeError, TypeError, ValueError):
            return None, None
        if (
            len(center) != 3
            or not all(math.isfinite(value) for value in center)
            or not math.isfinite(half)
            or half <= 0.0
        ):
            return None, None
        return center, half

    def _update_decision_sidecar(self, context, spec, *, force: bool = False) -> None:
        """Poll/submit costmap decisions without blocking the 20 Hz tick."""
        worker = self._decision_executor
        if worker is None or self._closing:
            return
        future = self._decision_future
        if future is not None and future.done():
            self._decision_future = None
            try:
                key, outcome = future.result()
            except Exception as exc:
                self._emit_structured_event(
                    "step_failed",
                    context,
                    message=f"decision sidecar error: {type(exc).__name__}: {exc}",
                    details={"sidecar_only": True},
                )
            else:
                if key == self._decision_key() and outcome is not None:
                    self.last_decision = outcome
                    self._offer_candidate_to_executor(outcome, context)
            future = None
        if self._decision_future is not None:
            return
        now = float(context.now_s)
        if (
            not force
            and self._last_decision_submit_s is not None
            and now - self._last_decision_submit_s < self.decision_period_s
        ):
            return
        self._last_decision_submit_s = now
        key = self._decision_key()
        self._decision_future = worker.submit(
            self._compute_stage_decision,
            key,
            context,
            spec,
        )

    def _should_wait_for_initial_candidate(self, context, spec) -> bool:
        if self.candidate_initial_wait_s <= 0.0 or self._decision_future is None:
            return False
        if getattr(spec, "stage", spec) not in {
            TaskStage.NAVIGATE_TO_PICK,
            TaskStage.TRANSPORT,
            TaskStage.RETURN_TO_END,
        }:
            return False
        if not callable(getattr(self._executor(), "apply_scheduler_candidate", None)):
            return False
        if self._stage_started_s is None:
            return False
        elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
        return elapsed <= self.candidate_initial_wait_s

    def _offer_candidate_to_executor(self, outcome, context) -> None:
        """Use an opt-in executor hook; absence means audit-only operation."""
        selected = getattr(outcome, "selected", None)
        if selected is None:
            return
        hook = getattr(self._executor(), "apply_scheduler_candidate", None)
        if not callable(hook):
            return
        try:
            application = hook(selected.candidate, outcome, context)
            if isinstance(application, CandidateApplicationStatus):
                status = application.value
            elif application is None:
                status = "unreported"
            else:
                status = str(application)
            self.last_candidate_application = status
            candidate = selected.candidate
            raw_goal_pose = getattr(candidate, "goal_pose", None)
            goal_pose = (
                tuple(float(value) for value in raw_goal_pose)
                if raw_goal_pose is not None
                else None
            )
            metadata = getattr(candidate, "metadata", {})
            lateral_offset = (
                metadata.get("lateral_offset_m")
                if isinstance(metadata, Mapping)
                else None
            )
            self._emit_structured_event(
                "candidate_application",
                context,
                message=f"candidate {selected.action_id} application={status}",
                details={
                    "action_id": selected.action_id,
                    "application_status": status,
                    "goal_pose": goal_pose,
                    "lateral_offset_m": lateral_offset,
                },
            )
        except Exception as exc:
            # An executor that opted into scheduler candidates must fail closed
            # if it cannot accept a validated selection.
            self.stop(
                "executor rejected scheduler candidate: "
                f"{type(exc).__name__}: {exc}"
            )

    def close(self) -> None:
        with self._decision_lifecycle_lock:
            if self._closing:
                return
            self._closing = True
        if self.state not in (
            self._states.FINISHED,
            self._states.SAFE_HOLD,
        ):
            self.stop("scheduler closed")
        self._release_arm_hold_lease()
        self._arm_command = None
        if self._decision_future is not None:
            self._decision_future.cancel()
            self._decision_future = None
        if self._decision_executor is not None:
            self._decision_executor.shutdown(wait=False, cancel_futures=True)
            self._decision_executor = None
        with self._decision_lifecycle_lock:
            close = getattr(self.decision_service, "close", None)
            if callable(close):
                close()

    def _plan(self) -> ExecutorTaskPlan | None:
        task_id = self.task_id
        return None if task_id is None else self.plans.get(task_id)

    def _executor(self) -> TaskExecutor:
        task_id = self.task_id
        if task_id not in self.executors:
            raise RuntimeError(f"no executor for task id {task_id}")
        return self.executors[task_id]

    def _cancel_action(self, reason: str) -> str | None:
        action, self._active_action = self._active_action, None
        self._stage_started_s = None
        error = None
        if action is not None:
            try:
                action.cancel(reason)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                self._release_active_resources()
                self._ensure_arm_hold_lease()
        return error

    def _owner_id(self, stage: TaskStage) -> str:
        return (
            f"task{self.task_id}:attempt{self.attempt}:"
            f"stage{self.stage_index}:{stage.value}"
        )

    @staticmethod
    def _resource_set(spec) -> frozenset[Resource]:
        return frozenset(Resource(value) for value in spec.resources)

    @staticmethod
    def _result_failure_code(result) -> FailureCode | None:
        code = getattr(result, "failure_code", None)
        return code if isinstance(code, FailureCode) else None

    @staticmethod
    def _failure_code_label(code) -> str:
        return "unknown" if code is None else str(code.value)

    @staticmethod
    def _recovery_factory_for(executor: TaskExecutor, spec) -> Any | None:
        """Opt-in per-executor recovery actions; None means step re-entry.

        Executors that provide finer-grained recovery motions expose a
        ``build_recovery_action(name)`` callable returning an entered
        stage action.  The default is the bounded L2 re-entry executed by
        :class:`RecoverableStageAction` itself.
        """
        builder = getattr(executor, "build_recovery_action", None)
        return builder if callable(builder) else None

    def _release_owner(self, owner: str) -> None:
        try:
            self.resource_manager.release(owner)
        except Exception:
            pass

    def _release_active_resources(self) -> None:
        owner, self._active_owner = self._active_owner, None
        if owner is not None:
            self._release_owner(owner)

    @staticmethod
    def _manipulator_resources() -> frozenset[Resource]:
        return frozenset(
            {
                Resource.SPINE,
                Resource.HEAD,
                Resource.LEFT_ARM,
                Resource.RIGHT_ARM,
                Resource.GRIPPERS,
            }
        )

    def _ensure_arm_hold_lease(self) -> None:
        if self._arm_command is None or self._active_owner is not None:
            return
        if self._arm_hold_owner is None:
            owner = "scheduler:persistent_arm_hold"
            try:
                self.resource_manager.acquire(
                    self._manipulator_resources(),
                    owner=owner,
                )
            except ResourceConflictError:
                # Never publish a stale hold target without proven ownership.
                self._arm_command = None
            else:
                self._arm_hold_owner = owner

    def _release_arm_hold_lease(self) -> None:
        owner, self._arm_hold_owner = self._arm_hold_owner, None
        if owner is not None:
            self._release_owner(owner)

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
                self._states.WAITING_FOR_REFEREE,
                f"task {self.task_id} local sequence {outcome}; "
                f"waiting for Server referee: {message}",
            )
            return

        if succeeded:
            self._advance_to_next_task("dry-run task sequence completed")
            return
        if self.attempt < self.max_attempts:
            self.attempt += 1
            self._transition(
                self._states.STARTING_TASK,
                f"dry-run retry task {self.task_id} attempt {self.attempt}: {message}",
            )
        else:
            self._advance_to_next_task(
                f"dry-run task {self.task_id} exhausted {self.max_attempts} attempts"
            )

    def _tick_waiting_for_referee(self, context: ExecutionContext) -> None:
        if self._referee_finished(context):
            self._finish("referee reported all tasks finished")
            return

        ordinal = self._referee_task_ordinal(context)
        completed = self._referee_attempts_completed(context)
        current_ordinal = self.task_index + 1
        if ordinal is not None and ordinal > current_ordinal:
            self.task_index = min(ordinal - 1, len(self.instructions) - 1)
            self.attempt = max(1, completed + 1)
            self.stage_index = 0
            self._active_action = None
            self._stage_started_s = None
            self._transition(
                self._states.STARTING_TASK,
                f"referee advanced to task {self.task_id}; starting attempt {self.attempt}",
            )
            return

        if completed > self._wait_referee_attempts_completed:
            self.attempt = completed + 1
            if self.attempt <= self.max_attempts:
                self.stage_index = 0
                self._active_action = None
                self._stage_started_s = None
                self._transition(
                    self._states.STARTING_TASK,
                    f"referee settled attempt; retrying task {self.task_id} "
                    f"attempt {self.attempt}",
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
        self._active_action = None
        self._stage_started_s = None
        if self.task_index >= len(self.instructions):
            self._finish(message)
        else:
            self._transition(
                self._states.STARTING_TASK,
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

    def _may_finish_active_action_then_cleanup(self) -> bool:
        plan = self._plan()
        spec = self.stage_spec
        irreversible_reached = bool(
            plan is not None
            and any(item.irreversible for item in plan.stages[: self.stage_index + 1])
        )
        return bool(
            self.state is self._states.EXECUTING_STAGE
            and plan is not None
            and plan.terminal_policy is TerminalPolicy.COMPLETE_ACTIVE_SEQUENCE
            and spec is not None
            and (spec.cleanup or irreversible_reached)
        )

    def _next_cleanup_index(self) -> int | None:
        plan = self._plan()
        if plan is None:
            return None
        for index in range(self.stage_index + 1, len(plan.stages)):
            if plan.stages[index].cleanup:
                return index
        return None

    def _validate_result(self, spec, result, context: ExecutionContext) -> None:
        if not isinstance(result, StageResult):
            raise TypeError(
                f"executor must return StageResult, got {type(result).__name__}"
            )
        if not isinstance(result.status, StageStatus):
            raise ValueError(f"executor returned invalid stage status {result.status!r}")
        if result.failure_code is not None and not isinstance(
            result.failure_code, FailureCode
        ):
            raise TypeError(
                "StageResult failure_code must be a scheduler FailureCode value"
            )
        if result.controls_arm and not self._arm_command_is_finite(result.arm_command):
            raise ValueError(
                "ArmCommand must contain finite spine, 2 head, 6 left-arm, "
                "2 gripper, and 6 right-arm values"
            )
        if result.controls_base and not spec.allows_base:
            raise ValueError("base command is outside the stage resource lease")
        if result.controls_arm and not spec.allows_arm:
            raise ValueError("arm command is outside the stage resource lease")
        if result.controls_arm and result.arm_command is None:
            raise ValueError("controls_arm is true without an ArmCommand")
        owner = self._active_owner
        if owner is None:
            raise ValueError("active stage has no resource owner")
        frame = CommandFrame(
            owner_step_id=owner,
            base_command=(
                BaseCommand(result.base_linear_x, result.base_angular_z)
                if result.controls_base
                else None
            ),
            arm_command=result.arm_command if result.controls_arm else None,
            arm_mode=(
                ArmCommandMode.MOVE if result.controls_arm else ArmCommandMode.NONE
            ),
            resources=self._resource_set(spec),
        )
        if result.controls_arm:
            full_manipulator = frozenset(
                {
                    Resource.SPINE,
                    Resource.HEAD,
                    Resource.LEFT_ARM,
                    Resource.RIGHT_ARM,
                    Resource.GRIPPERS,
                }
            )
            if not full_manipulator.issubset(self._resource_set(spec)):
                raise ValueError(
                    "complete ArmCommand is outside the full manipulator "
                    "resource lease"
                )
        self.command_validator.validate(frame, now_s=float(context.now_s))
        violation = self.safety_supervisor.check(context, frame)
        if violation.must_stop:
            raise ValueError(violation.message or "safety command violation")
        if result.controls_base:
            self.base_command_lease.renew(
                owner,
                frame.base_command,
                float(context.now_s),
            )

    @staticmethod
    def _arm_command_is_finite(command: ArmCommand | None) -> bool:
        if command is None:
            return False
        try:
            if len(command.head_positions) != 2:
                return False
            if len(command.left_arm_positions) != 6:
                return False
            if len(command.right_arm_positions) != 6:
                return False
        except (AttributeError, TypeError):
            return False
        values = (
            command.spine_position,
            *command.head_positions,
            *command.left_arm_positions,
            command.left_gripper_position,
            *command.right_arm_positions,
            command.right_gripper_position,
        )
        try:
            return all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _context_referee_task_ordinal(context: ExecutionContext) -> int | None:
        value = context.referee_gameinfo.get("task_ordinal")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _context_referee_attempts_completed(context: ExecutionContext) -> int:
        value = context.referee_gameinfo.get("attempt", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _context_referee_finished(context: ExecutionContext) -> bool:
        task_text = context.referee_taskinfo.casefold()
        game_text = str(context.referee_gameinfo.get("raw", "")).casefold()
        return (
            "\u5168\u90e8\u4efb\u52a1\u7ed3\u675f" in task_text
            or "all tasks finished" in task_text
            or "all_tasks_done" in game_text
        )

    def _referee_task_ordinal(self, context: ExecutionContext) -> int | None:
        if self._referee_update is not None and self._referee_update.desynchronised:
            update = self._last_consistent_referee_update
            return None if update is None else update.task_ordinal
        update = self._referee_update
        if update is not None:
            return update.task_ordinal
        return self._context_referee_task_ordinal(context)

    def _referee_attempts_completed(self, context: ExecutionContext) -> int:
        if self._referee_update is not None and self._referee_update.desynchronised:
            update = self._last_consistent_referee_update
            return 0 if update is None else update.attempts_completed
        update = self._referee_update
        if update is not None:
            return update.attempts_completed
        return self._context_referee_attempts_completed(context)

    def _referee_finished(self, context: ExecutionContext) -> bool:
        # Terminal markers are monotonic and safety-relevant.  Accept a real
        # all-done marker even when another field in the same frame disagrees;
        # progression ordinals/attempts remain frozen to the last consistent
        # snapshot above.
        if self._referee_update is not None and self._referee_update.all_tasks_done:
            return True
        update = (
            self._last_consistent_referee_update
            if self._referee_update is not None and self._referee_update.desynchronised
            else self._referee_update
        )
        if update is not None:
            return update.all_tasks_done
        return self._context_referee_finished(context)

    def _transition(self, state: Enum, message: str) -> None:
        previous = self.state
        self.state = state
        self._controls_base = False
        self._base_linear_x = 0.0
        self._base_angular_z = 0.0
        self._message = message
        self._transition_serial += 1
        self._emit_event(previous, state, message)

    def _bump_message(self, message: str) -> None:
        self._message = message
        self._transition_serial += 1
        self._emit_event(self.state, self.state, message)

    def _emit_event(self, previous: Enum, current: Enum, message: str) -> None:
        """Best-effort observability; event failures never affect robot control."""
        if self._event_sink is None:
            return
        payload = {
            "serial": self._transition_serial,
            "previous_state": previous.value,
            "state": current.value,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "stage": None if self.stage is None else self.stage.value,
            "message": message,
        }
        try:
            emit = getattr(self._event_sink, "emit", None)
            if callable(emit):
                emit(payload)
            elif callable(self._event_sink):
                self._event_sink(payload)
        except Exception:
            pass

    def _emit_structured_event(
        self,
        event_type: str,
        context: ExecutionContext,
        *,
        message: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        try:
            emit = getattr(self._event_sink, "emit", None)
            if callable(emit):
                emit(
                    event_type,
                    message,
                    timestamp_s=float(context.now_s),
                    task_id=self.task_id,
                    attempt=self.attempt,
                    step_id=None if self.stage is None else self.stage.value,
                    details=dict(details or {}),
                )
        except Exception:
            pass


__all__ = ["EngineSnapshot", "EngineState", "SchedulerEngine"]
