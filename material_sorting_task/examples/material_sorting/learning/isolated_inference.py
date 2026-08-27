"""Fail-closed, process-isolated inference transport for scheduler policies.

The competition controller must remain responsive even if optional policy
inference blocks, crashes or runs out of GPU resources.  This module contains
no ROS, actuator or executor code: a child process sees only an immutable
observation/action-mask request and returns only a discrete candidate slot.

It is deliberately an opt-in transport.  Existing heuristic, Shadow and
Guarded call paths remain unchanged until a later adapter consumes its results
with a fresh action-mask and costmap-version check.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
from queue import Empty, Full
import time
from typing import Any, Protocol, Sequence
import uuid

import numpy as np


@dataclass(frozen=True)
class PolicyWorkerConfig:
    """Serializable model configuration for the unprivileged child process."""

    model_path: str
    expected_sha256: str
    expected_schema_hash: str
    device: str = "cpu"
    observation_size: int = 138
    action_count: int = 8

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path must be non-empty")
        if not str(self.expected_sha256).strip():
            raise ValueError("expected_sha256 must be non-empty")
        if not str(self.expected_schema_hash).strip():
            raise ValueError("expected_schema_hash must be non-empty")
        if not str(self.device).strip():
            raise ValueError("device must be non-empty")
        if int(self.observation_size) <= 0 or int(self.action_count) <= 0:
            raise ValueError("worker observation_size and action_count must be positive")


@dataclass(frozen=True)
class InferenceRequest:
    """One immutable, non-actuating policy query."""

    request_id: str
    signature: str
    observation: tuple[float, ...]
    action_mask: tuple[bool, ...]
    submitted_at_s: float

    @classmethod
    def build(
        cls,
        *,
        signature: str,
        observation: Sequence[float],
        action_mask: Sequence[bool],
        submitted_at_s: float | None = None,
    ) -> "InferenceRequest":
        values = tuple(float(value) for value in observation)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("observation must be finite and non-empty")
        mask = tuple(bool(value) for value in action_mask)
        if not mask or not any(mask):
            raise ValueError("action_mask must contain an allowed action")
        token = str(signature).strip()
        if not token:
            raise ValueError("signature must be non-empty")
        now = time.monotonic() if submitted_at_s is None else float(submitted_at_s)
        if not math.isfinite(now):
            raise ValueError("submitted_at_s must be finite")
        return cls(
            request_id=uuid.uuid4().hex,
            signature=token,
            observation=values,
            action_mask=mask,
            submitted_at_s=now,
        )


@dataclass(frozen=True)
class InferenceResult:
    """A child-process result.  It has no authority to choose an action."""

    request_id: str
    signature: str
    action_index: int | None
    inference_ms: float
    completed_at_s: float
    model_sha256: str | None = None
    error: str | None = None


class InferenceTransport(Protocol):
    """Minimal non-blocking transport used by :class:`InferenceSupervisor`."""

    def submit(self, request: InferenceRequest) -> bool: ...

    def poll(self) -> tuple[InferenceResult, ...]: ...

    def is_alive(self) -> bool: ...

    def restart(self) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class InferenceSupervisorConfig:
    """Parent-side timeout and fault-isolation policy."""

    deadline_s: float = 0.025
    stale_after_s: float = 0.25
    isolate_after_consecutive_faults: int = 3

    def __post_init__(self) -> None:
        for name in ("deadline_s", "stale_after_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if int(self.isolate_after_consecutive_faults) <= 0:
            raise ValueError("isolate_after_consecutive_faults must be positive")
        object.__setattr__(
            self,
            "isolate_after_consecutive_faults",
            int(self.isolate_after_consecutive_faults),
        )


@dataclass(frozen=True)
class SupervisorPoll:
    """One parent-side poll result.

    ``ready`` means only that the immutable policy response is fresh and
    signature-matched.  The caller must still re-run ActionMask and safety
    checks before it can ever influence a scheduler decision.
    """

    status: str
    result: InferenceResult | None = None
    reason: str | None = None


class InferenceSupervisor:
    """Own one in-flight query and fail closed without blocking the caller."""

    def __init__(
        self,
        transport: InferenceTransport,
        *,
        config: InferenceSupervisorConfig | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.transport = transport
        self.config = config or InferenceSupervisorConfig()
        self._clock = clock
        self._pending: InferenceRequest | None = None
        self._consecutive_faults = 0
        self._isolated_reason: str | None = None

    @property
    def isolated(self) -> bool:
        return self._isolated_reason is not None

    @property
    def pending(self) -> InferenceRequest | None:
        return self._pending

    @property
    def consecutive_faults(self) -> int:
        return self._consecutive_faults

    def submit(
        self,
        *,
        signature: str,
        observation: Sequence[float],
        action_mask: Sequence[bool],
        now_s: float | None = None,
    ) -> SupervisorPoll:
        """Queue an optional suggestion and return immediately."""

        if self.isolated:
            return SupervisorPoll("unavailable", reason=self._isolated_reason)
        if self._pending is not None:
            return SupervisorPoll("pending", reason="inference_pending")
        now = self._now(now_s)
        request = InferenceRequest.build(
            signature=signature,
            observation=observation,
            action_mask=action_mask,
            submitted_at_s=now,
        )
        if not self.transport.is_alive() and not self.transport.restart():
            return self._fault("worker_unavailable")
        if not self.transport.submit(request):
            return self._fault("worker_queue_full")
        self._pending = request
        return SupervisorPoll("submitted")

    def poll(
        self,
        *,
        expected_signature: str,
        now_s: float | None = None,
    ) -> SupervisorPoll:
        """Consume at most one matching result; this method never waits."""

        now = self._now(now_s)
        pending = self._pending
        for result in self.transport.poll():
            if pending is None or result.request_id != pending.request_id:
                continue
            self._pending = None
            if result.signature != str(expected_signature):
                return SupervisorPoll("abstain", reason="inference_result_stale")
            if now - pending.submitted_at_s > self.config.stale_after_s:
                return SupervisorPoll("abstain", reason="inference_result_expired")
            if result.error:
                return self._fault("worker_error")
            if result.action_index is None:
                return self._fault("worker_invalid_result")
            self._consecutive_faults = 0
            return SupervisorPoll("ready", result=result)

        pending = self._pending
        if pending is None:
            return SupervisorPoll("idle")
        if now - pending.submitted_at_s > self.config.deadline_s:
            self._pending = None
            return self._fault("inference_timeout")
        return SupervisorPoll("pending", reason="inference_pending")

    def reset_isolation(self) -> bool:
        """Explicit operator action only; a fault never re-enables itself."""

        if not self.transport.restart():
            return False
        self._pending = None
        self._consecutive_faults = 0
        self._isolated_reason = None
        return True

    def close(self) -> None:
        self._pending = None
        self.transport.close()

    def _fault(self, reason: str) -> SupervisorPoll:
        self._consecutive_faults += 1
        if self._consecutive_faults >= self.config.isolate_after_consecutive_faults:
            self._isolated_reason = f"{reason}_quarantined"
            self.transport.close()
        return SupervisorPoll("fallback", reason=reason)

    def _now(self, value: float | None) -> float:
        now = float(self._clock() if value is None else value)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        return now


def _worker_main(
    requests: Any,
    results: Any,
    readiness: Any,
    worker_config: PolicyWorkerConfig,
) -> None:
    """Child entry point.  It owns model/GPU state and cannot command a robot."""

    from scheduler.policies.rl import RLPolicy

    policy = RLPolicy(
        model_path=worker_config.model_path,
        expected_sha256=worker_config.expected_sha256,
        expected_schema_hash=worker_config.expected_schema_hash,
        device=worker_config.device,
    )
    try:
        # Loading and warmup are a startup concern, never a control-loop
        # deadline concern. If they fail, the parent leaves Heuristic active.
        policy.warmup(
            observation_size=worker_config.observation_size,
            action_count=worker_config.action_count,
        )
    except Exception as exc:
        readiness.put((False, f"{type(exc).__name__}: {exc}"))
        return
    readiness.put((True, None))
    while True:
        request = requests.get()
        if request is None:
            return
        started = time.perf_counter()
        try:
            prediction = policy.predict(
                np.asarray(request.observation, dtype=np.float32),
                action_masks=np.asarray(request.action_mask, dtype=np.bool_),
                deterministic=True,
            )
            result = InferenceResult(
                request_id=request.request_id,
                signature=request.signature,
                action_index=int(prediction.action_index),
                inference_ms=max(
                    (time.perf_counter() - started) * 1000.0,
                    float(prediction.inference_ms),
                ),
                completed_at_s=time.monotonic(),
                model_sha256=prediction.model_sha256,
            )
        except Exception as exc:
            result = InferenceResult(
                request_id=request.request_id,
                signature=request.signature,
                action_index=None,
                inference_ms=(time.perf_counter() - started) * 1000.0,
                completed_at_s=time.monotonic(),
                error=f"{type(exc).__name__}: {exc}",
            )
        results.put(result)


class ProcessPolicyTransport:
    """Bounded local IPC transport with a separately-owned RL process."""

    def __init__(
        self,
        worker_config: PolicyWorkerConfig,
        *,
        start_method: str | None = None,
    ) -> None:
        self.worker_config = worker_config
        self._context = mp.get_context(start_method)
        self._requests: Any | None = None
        self._results: Any | None = None
        self._readiness: Any | None = None
        self._process: Any | None = None
        self.startup_error: str | None = None

    def start(self, *, startup_timeout_s: float = 10.0) -> bool:
        if self.is_alive():
            return True
        self.close()
        self.startup_error = None
        try:
            self._requests = self._context.Queue(maxsize=1)
            self._results = self._context.Queue(maxsize=2)
            self._readiness = self._context.Queue(maxsize=1)
            self._process = self._context.Process(
                target=_worker_main,
                args=(
                    self._requests,
                    self._results,
                    self._readiness,
                    self.worker_config,
                ),
                name="scheduler-rl-inference",
                daemon=True,
            )
            self._process.start()
            ready, error = self._readiness.get(timeout=float(startup_timeout_s))
            if bool(ready) and self.is_alive():
                return True
            self.startup_error = str(error or "worker exited during startup")
        except Empty:
            self.startup_error = "worker startup timed out"
        except Exception as exc:
            self.startup_error = f"{type(exc).__name__}: {exc}"
        if self.startup_error is not None:
            self.close()
            return False
        return False

    def submit(self, request: InferenceRequest) -> bool:
        if not self.is_alive() or self._requests is None:
            return False
        try:
            self._requests.put_nowait(request)
            return True
        except Full:
            return False

    def poll(self) -> tuple[InferenceResult, ...]:
        if self._results is None:
            return ()
        collected: list[InferenceResult] = []
        while True:
            try:
                result = self._results.get_nowait()
            except Empty:
                break
            if isinstance(result, InferenceResult):
                collected.append(result)
        return tuple(collected)

    def is_alive(self) -> bool:
        return bool(
            self._process is not None
            and self._process.pid is not None
            and self._process.exitcode is None
        )

    def restart(self) -> bool:
        return self.start()

    def close(self) -> None:
        process, requests, results, readiness = (
            self._process,
            self._requests,
            self._results,
            self._readiness,
        )
        self._process = None
        self._requests = None
        self._results = None
        self._readiness = None
        if process is None:
            return
        try:
            if process.is_alive() and requests is not None:
                try:
                    requests.put_nowait(None)
                except Full:
                    pass
                process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        finally:
            for queue in (requests, results, readiness):
                close = getattr(queue, "close", None)
                if callable(close):
                    close()


__all__ = [
    "InferenceRequest",
    "InferenceResult",
    "InferenceSupervisor",
    "InferenceSupervisorConfig",
    "InferenceTransport",
    "PolicyWorkerConfig",
    "ProcessPolicyTransport",
    "SupervisorPoll",
]
