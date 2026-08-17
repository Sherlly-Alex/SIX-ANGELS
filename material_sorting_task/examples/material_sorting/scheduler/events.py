"""Structured scheduler events with in-memory and JSON Lines sinks."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol

from .models import FailureCode


class EventType(Enum):
    SCHEDULER_STARTED = "scheduler_started"
    SCHEDULER_TRANSITION = "scheduler_transition"
    REFEREE_CHANGED = "referee_changed"
    REFEREE_DESYNC = "referee_desync"
    STEP_ENTERED = "step_entered"
    STEP_SUCCEEDED = "step_succeeded"
    STEP_FAILED = "step_failed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_FINISHED = "recovery_finished"
    RESOURCE_ACQUIRED = "resource_acquired"
    RESOURCE_RELEASED = "resource_released"
    COMMAND_REJECTED = "command_rejected"
    SAFETY_STOP = "safety_stop"
    CANDIDATES_EVALUATED = "candidates_evaluated"
    ACTION_SELECTED = "action_selected"
    CANDIDATE_APPLICATION = "candidate_application"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass(frozen=True)
class SchedulerEvent:
    """One append-only scheduler audit record."""

    timestamp_s: float
    event_type: EventType | str
    message: str = ""
    task_id: int | None = None
    attempt: int | None = None
    step_id: str | None = None
    action_id: str | None = None
    failure_code: FailureCode | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        timestamp_s = float(self.timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("event timestamp_s must be finite")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.sequence < 0:
            raise ValueError("event sequence cannot be negative")

    @property
    def type(self) -> str:
        return self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "sequence": self.sequence,
            "event_type": self.type,
            "message": self.message,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "step_id": self.step_id,
            "action_id": self.action_id,
            "failure_code": (
                self.failure_code.value if self.failure_code is not None else None
            ),
            "details": _json_safe(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerEvent":
        failure = value.get("failure_code")
        return cls(
            timestamp_s=float(value["timestamp_s"]),
            sequence=int(value.get("sequence", 0)),
            event_type=str(value.get("event_type", value.get("type", "unknown"))),
            message=str(value.get("message", "")),
            task_id=value.get("task_id"),
            attempt=value.get("attempt"),
            step_id=value.get("step_id"),
            action_id=value.get("action_id"),
            failure_code=FailureCode(failure) if failure else None,
            details=dict(value.get("details", {})),
        )


class EventSink(Protocol):
    def write(self, event: SchedulerEvent) -> None:
        """Persist one event."""


class MemoryEventSink:
    """Thread-safe event sink useful for tests, dashboards and shadow mode."""

    def __init__(self, *, max_events: int | None = None) -> None:
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: list[SchedulerEvent] = []
        self._lock = threading.RLock()

    def write(self, event: SchedulerEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self.max_events is not None:
                del self._events[: max(0, len(self._events) - self.max_events)]

    emit = write
    append = write

    @property
    def events(self) -> tuple[SchedulerEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class JsonlEventSink:
    """Append complete UTF-8 JSON records, one event per line."""

    def __init__(self, path: str | Path, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.flush = bool(flush)
        self._lock = threading.RLock()

    def write(self, event: SchedulerEvent) -> None:
        line = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.write("\n")
                if self.flush:
                    stream.flush()

    emit = write
    append = write


class EventLog:
    """Fan out ordered events to any number of sinks."""

    def __init__(
        self,
        sinks: Iterable[EventSink] = (),
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sinks = list(sinks)
        self._clock = clock
        self._sequence = 0
        self._lock = threading.RLock()

    def add_sink(self, sink: EventSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def emit(
        self,
        event: SchedulerEvent | EventType | str | Mapping[str, Any],
        message: str = "",
        **fields: Any,
    ) -> SchedulerEvent:
        with self._lock:
            self._sequence += 1
            if isinstance(event, SchedulerEvent):
                record = replace(event, sequence=self._sequence)
            elif isinstance(event, Mapping):
                # Compatibility with the v2 facade's pre-structured
                # transition payload.  Preserve every unknown field in
                # ``details`` instead of dropping useful replay context.
                payload = dict(event)
                event_type = payload.pop("event_type", EventType.SCHEDULER_TRANSITION)
                record = SchedulerEvent(
                    timestamp_s=float(payload.pop("timestamp_s", self._clock())),
                    event_type=event_type,
                    message=str(payload.pop("message", message)),
                    task_id=payload.pop("task_id", None),
                    attempt=payload.pop("attempt", None),
                    step_id=payload.pop("step_id", payload.pop("stage", None)),
                    sequence=self._sequence,
                    details=payload,
                )
            else:
                record = SchedulerEvent(
                    timestamp_s=float(fields.pop("timestamp_s", self._clock())),
                    event_type=event,
                    message=message,
                    sequence=self._sequence,
                    **fields,
                )
            for sink in tuple(self._sinks):
                sink.write(record)
            return record

    write = emit

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence


# Common spelling variants used by downstream tooling.
InMemoryEventSink = MemoryEventSink
JSONLEventSink = JsonlEventSink


__all__ = [
    "EventLog",
    "EventSink",
    "EventType",
    "InMemoryEventSink",
    "JSONLEventSink",
    "JsonlEventSink",
    "MemoryEventSink",
    "SchedulerEvent",
]
