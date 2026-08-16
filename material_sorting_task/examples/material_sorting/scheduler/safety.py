"""Highest-priority, ROS-free scheduler safety checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .models import BaseCommand, CommandFrame, FailureCode
from .resources import _contains_non_finite


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class SafetyViolation:
    """Result of a safety pass; a safe result has no failure code."""

    failure_code: FailureCode | None = None
    message: str = ""
    must_stop: bool = False
    source: str | None = None
    observed_value: float | None = None
    limit: float | None = None

    @property
    def safe(self) -> bool:
        return not self.must_stop

    @property
    def code(self) -> FailureCode | None:
        return self.failure_code

    def __bool__(self) -> bool:
        return self.must_stop

    @classmethod
    def no_violation(cls) -> "SafetyViolation":
        return cls()


class SafetySupervisor:
    """Check collision flags, command finiteness and configured input ages."""

    def __init__(
        self,
        input_age_limits_s: Mapping[str, float] | float | None = None,
        *,
        max_input_age_s: Mapping[str, float] | float | None = None,
    ) -> None:
        if input_age_limits_s is not None and max_input_age_s is not None:
            raise ValueError("supply only one input age limit argument")
        configured = input_age_limits_s if max_input_age_s is None else max_input_age_s
        if configured is None:
            limits: dict[str, float] = {}
        elif isinstance(configured, Mapping):
            limits = {str(name): float(limit) for name, limit in configured.items()}
        else:
            limits = {"*": float(configured)}
        for name, limit in limits.items():
            if not name:
                raise ValueError("input age source names must be non-empty")
            if not math.isfinite(limit) or limit < 0.0:
                raise ValueError("input age limits must be finite and non-negative")
        self.input_age_limits_s = limits

    def check(
        self,
        world: Any = None,
        command: Any = None,
    ) -> SafetyViolation:
        violations = self.violations(world, command)
        return violations[0] if violations else SafetyViolation.no_violation()

    evaluate = check

    def violations(
        self,
        world: Any = None,
        command: Any = None,
    ) -> tuple[SafetyViolation, ...]:
        found: list[SafetyViolation] = []
        if bool(_field(world, "unsafe_collision", False)):
            found.append(
                SafetyViolation(
                    FailureCode.UNSAFE_COLLISION,
                    "unsafe collision signal is active",
                    True,
                    "unsafe_collision",
                )
            )

        if command is not None:
            issue = self._check_command(command)
            if issue is not None:
                found.append(issue)

        ages = _field(world, "input_ages_s", {}) or {}
        if isinstance(ages, Mapping):
            wildcard = self.input_age_limits_s.get("*")
            for source, raw_age in ages.items():
                source_name = str(source)
                limit = self.input_age_limits_s.get(source_name, wildcard)
                if limit is None:
                    continue
                try:
                    age = float(raw_age)
                except (TypeError, ValueError, OverflowError):
                    age = math.inf
                if not math.isfinite(age) or age < 0.0 or age > limit:
                    found.append(
                        SafetyViolation(
                            FailureCode.INPUT_STALE,
                            f"input {source_name!r} age {age!r}s exceeds {limit:.3f}s",
                            True,
                            source_name,
                            age,
                            limit,
                        )
                    )
        return tuple(found)

    @staticmethod
    def _check_command(command: Any) -> SafetyViolation | None:
        if isinstance(command, CommandFrame):
            deadline = command.valid_until_s
            if deadline is not None and not math.isfinite(deadline):
                return SafetyViolation(
                    FailureCode.COMMAND_NON_FINITE,
                    "command validity deadline contains NaN or infinity",
                    True,
                    "command.valid_until_s",
                    deadline,
                )
            base = command.base_command
            if base is not None and not base.is_finite:
                return SafetyViolation(
                    FailureCode.COMMAND_NON_FINITE,
                    "base command contains NaN or infinity",
                    True,
                    "base_command",
                )
            inspected = command.arm_command
        elif isinstance(command, BaseCommand):
            if not command.is_finite:
                return SafetyViolation(
                    FailureCode.COMMAND_NON_FINITE,
                    "base command contains NaN or infinity",
                    True,
                    "base_command",
                )
            return None
        else:
            inspected = command

        if _contains_non_finite(inspected):
            return SafetyViolation(
                FailureCode.COMMAND_NON_FINITE,
                "actuator command contains NaN or infinity",
                True,
                "command",
            )
        return None

    def check_command(self, command: Any) -> SafetyViolation:
        issue = self._check_command(command)
        return issue if issue is not None else SafetyViolation.no_violation()


__all__ = ["SafetySupervisor", "SafetyViolation"]
