"""ROS-free runtime health guards for the 20 Hz competition client."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Mapping


class FreshnessState(Enum):
    """State of the required runtime input stream set."""

    STARTUP = "startup"
    FRESH = "fresh"
    STALE_GRACE = "stale_grace"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class FreshnessReport:
    state: FreshnessState
    ages_s: Mapping[str, float]
    missing_inputs: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    stale_for_s: float
    transitioned: bool

    @property
    def motion_allowed(self) -> bool:
        return self.state is FreshnessState.FRESH

    @property
    def terminal(self) -> bool:
        return self.state is FreshnessState.EXHAUSTED


class InputFreshnessWatchdog:
    """Track callback arrival ages and bound recovery from input dropouts.

    Callback arrival time is intentionally monotonic host time rather than a
    ROS header stamp.  That avoids mixing the Server's simulation clock with
    the client's wall clock and detects a publisher that simply stops.
    """

    def __init__(
        self,
        max_ages_s: Mapping[str, float],
        *,
        stale_grace_s: float,
    ) -> None:
        limits = {str(name): float(limit) for name, limit in max_ages_s.items()}
        if not limits:
            raise ValueError("at least one input freshness limit is required")
        if any(not name for name in limits):
            raise ValueError("input freshness names cannot be empty")
        if any(not math.isfinite(limit) or limit <= 0.0 for limit in limits.values()):
            raise ValueError("input freshness limits must be finite and positive")
        grace = float(stale_grace_s)
        if not math.isfinite(grace) or grace <= 0.0:
            raise ValueError("stale_grace_s must be finite and positive")
        self._limits = limits
        self._stale_grace_s = grace
        self._received_at_s: dict[str, float] = {}
        self._armed = False
        self._stale_since_s: float | None = None
        self._state = FreshnessState.STARTUP

    @property
    def limits_s(self) -> Mapping[str, float]:
        return dict(self._limits)

    @property
    def stale_grace_s(self) -> float:
        return self._stale_grace_s

    def observe(self, name: str, now_s: float) -> None:
        key = str(name)
        if key not in self._limits:
            raise KeyError(f"unconfigured freshness input: {key}")
        timestamp = float(now_s)
        if not math.isfinite(timestamp):
            raise ValueError("input observation time must be finite")
        self._received_at_s[key] = timestamp

    def arm(self) -> None:
        """Enable the terminal grace deadline once execution can start."""

        self._armed = True

    def evaluate(self, now_s: float) -> FreshnessReport:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("watchdog evaluation time must be finite")
        ages = {
            name: max(0.0, now - received_at)
            for name, received_at in self._received_at_s.items()
        }
        missing = tuple(name for name in self._limits if name not in ages)
        stale = tuple(
            name
            for name, limit in self._limits.items()
            if name in ages and ages[name] > limit
        )
        previous = self._state
        if missing:
            self._stale_since_s = None
            state = FreshnessState.STARTUP
            stale_for_s = 0.0
        elif stale:
            if self._stale_since_s is None:
                self._stale_since_s = now
            stale_for_s = max(0.0, now - self._stale_since_s)
            state = (
                FreshnessState.EXHAUSTED
                if self._armed and stale_for_s > self._stale_grace_s
                else FreshnessState.STALE_GRACE
            )
        else:
            self._stale_since_s = None
            state = FreshnessState.FRESH
            stale_for_s = 0.0
        self._state = state
        return FreshnessReport(
            state=state,
            ages_s=ages,
            missing_inputs=missing,
            stale_inputs=stale,
            stale_for_s=stale_for_s,
            transitioned=state is not previous,
        )


class InputDropFaultInjector:
    """Explicit opt-in file markers for remote callback-drop validation.

    Production behavior is unchanged when ``marker_directory`` is blank.
    A remote operator can create ``drop_odometry`` or ``drop_joint_states``
    inside the configured container directory, then remove it to restore the
    stream without modifying the ROS graph or restarting the Client.
    """

    INPUT_NAMES = frozenset({"odometry", "joint_states"})

    def __init__(self, marker_directory: str | Path | None = None) -> None:
        raw = "" if marker_directory is None else str(marker_directory).strip()
        self._directory = Path(raw) if raw else None

    @property
    def enabled(self) -> bool:
        return self._directory is not None

    @property
    def marker_directory(self) -> Path | None:
        return self._directory

    def marker_path(self, input_name: str) -> Path:
        name = str(input_name)
        if name not in self.INPUT_NAMES:
            raise KeyError(f"unsupported fault-injection input: {name}")
        if self._directory is None:
            raise RuntimeError("input-drop fault injection is disabled")
        return self._directory / f"drop_{name}"

    def should_drop(self, input_name: str) -> bool:
        if self._directory is None:
            return False
        try:
            return self.marker_path(input_name).is_file()
        except OSError:
            # A broken/unreadable marker path must not disable live inputs.
            return False


@dataclass(frozen=True)
class ControlLoopHealth:
    sample_count: int
    total_sample_count: int
    total_interval_count: int
    interval_p50_ms: float
    interval_p95_ms: float
    interval_p99_ms: float
    interval_max_ms: float
    execution_p95_ms: float
    execution_max_ms: float
    interval_deadline_misses: int
    execution_deadline_misses: int

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


class ControlLoopTelemetry:
    """Rolling cadence and execution-time telemetry for a periodic callback."""

    def __init__(
        self,
        period_s: float,
        *,
        report_period_s: float = 5.0,
        interval_miss_ratio: float = 1.5,
        window_size: int = 400,
    ) -> None:
        self.period_s = float(period_s)
        self.report_period_s = float(report_period_s)
        self.interval_miss_ratio = float(interval_miss_ratio)
        if not math.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError("period_s must be finite and positive")
        if not math.isfinite(self.report_period_s) or self.report_period_s <= 0.0:
            raise ValueError("report_period_s must be finite and positive")
        if not math.isfinite(self.interval_miss_ratio) or self.interval_miss_ratio < 1.0:
            raise ValueError("interval_miss_ratio must be finite and at least 1")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        self._intervals_s: deque[float] = deque(maxlen=int(window_size))
        self._executions_s: deque[float] = deque(maxlen=int(window_size))
        self._last_start_s: float | None = None
        self._last_report_s: float | None = None
        self._interval_deadline_misses = 0
        self._execution_deadline_misses = 0
        self._total_interval_count = 0
        self._total_sample_count = 0

    def begin(self, now_s: float) -> None:
        now = float(now_s)
        if self._last_start_s is not None:
            interval = max(0.0, now - self._last_start_s)
            self._intervals_s.append(interval)
            self._total_interval_count += 1
            if interval > self.period_s * self.interval_miss_ratio:
                self._interval_deadline_misses += 1
        self._last_start_s = now
        if self._last_report_s is None:
            self._last_report_s = now

    def finish(self, started_at_s: float, finished_at_s: float) -> ControlLoopHealth | None:
        duration = max(0.0, float(finished_at_s) - float(started_at_s))
        self._executions_s.append(duration)
        self._total_sample_count += 1
        if duration > self.period_s:
            self._execution_deadline_misses += 1
        if (
            self._last_report_s is None
            or float(finished_at_s) - self._last_report_s < self.report_period_s
        ):
            return None
        self._last_report_s = float(finished_at_s)
        return self.snapshot()

    @staticmethod
    def _percentile(values: tuple[float, ...], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    def snapshot(self) -> ControlLoopHealth:
        intervals = tuple(self._intervals_s)
        executions = tuple(self._executions_s)
        return ControlLoopHealth(
            sample_count=len(executions),
            total_sample_count=self._total_sample_count,
            total_interval_count=self._total_interval_count,
            interval_p50_ms=self._percentile(intervals, 0.50) * 1000.0,
            interval_p95_ms=self._percentile(intervals, 0.95) * 1000.0,
            interval_p99_ms=self._percentile(intervals, 0.99) * 1000.0,
            interval_max_ms=max(intervals, default=0.0) * 1000.0,
            execution_p95_ms=self._percentile(executions, 0.95) * 1000.0,
            execution_max_ms=max(executions, default=0.0) * 1000.0,
            interval_deadline_misses=self._interval_deadline_misses,
            execution_deadline_misses=self._execution_deadline_misses,
        )
