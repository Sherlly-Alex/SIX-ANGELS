"""Atomic resource ownership, command validation and short base leases."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .models import (
    ArmCommandMode,
    BaseCommand,
    CommandFrame,
    FailureCode,
    Resource,
)


class ResourceConflictError(RuntimeError):
    """Raised when an owner attempts to use a resource held by another owner."""

    failure_code = FailureCode.RESOURCE_CONFLICT

    def __init__(self, message: str, *, conflicts: Mapping[Resource, str] | None = None) -> None:
        super().__init__(message)
        self.conflicts = dict(conflicts or {})


class CommandValidationError(RuntimeError):
    """A command is unsafe, expired, or exceeds its owner's resource claim."""

    def __init__(self, failure_code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class ResourceLease:
    owner: str
    resources: frozenset[Resource]
    acquired_at_s: float | None = None
    valid_until_s: float | None = None

    def active_at(self, now_s: float) -> bool:
        return self.valid_until_s is None or float(now_s) <= self.valid_until_s


@dataclass(frozen=True)
class _Ownership:
    owner: str
    acquired_at_s: float | None
    valid_until_s: float | None


def _normalise_resources(resources: Iterable[Resource]) -> frozenset[Resource]:
    result = frozenset(resources)
    invalid = [resource for resource in result if not isinstance(resource, Resource)]
    if invalid:
        raise TypeError(f"unknown scheduler resources: {invalid!r}")
    return result


class ResourceManager:
    """Exclusive resource table with all-or-nothing multi-resource acquire."""

    def __init__(self) -> None:
        self._owners: dict[Resource, _Ownership] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        resources: Iterable[Resource],
        *,
        owner: str,
        now_s: float | None = None,
        lease_duration_s: float | None = None,
    ) -> ResourceLease:
        requested = _normalise_resources(resources)
        if not owner or not owner.strip():
            raise ValueError("resource owner must be non-empty")
        now = None if now_s is None else float(now_s)
        if now is not None and not math.isfinite(now):
            raise ValueError("now_s must be finite")
        if lease_duration_s is not None:
            duration = float(lease_duration_s)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("lease_duration_s must be finite and positive")
            if now is None:
                raise ValueError("now_s is required for a timed resource lease")
            valid_until = now + duration
        else:
            valid_until = None

        with self._lock:
            self._expire_locked(now)
            conflicts = {
                resource: ownership.owner
                for resource in requested
                if (ownership := self._owners.get(resource)) is not None
                and ownership.owner != owner
            }
            if conflicts:
                summary = ", ".join(
                    f"{resource.value}={held_by}" for resource, held_by in conflicts.items()
                )
                raise ResourceConflictError(
                    f"owner {owner!r} cannot atomically acquire resources: {summary}",
                    conflicts=conflicts,
                )

            # No mutation occurs until every requested resource passes the
            # conflict check above.  Re-acquire by the same owner is idempotent
            # and renews a timed claim.
            for resource in requested:
                old = self._owners.get(resource)
                self._owners[resource] = _Ownership(
                    owner=owner,
                    acquired_at_s=(old.acquired_at_s if old is not None else now),
                    valid_until_s=valid_until,
                )
            return ResourceLease(owner, requested, now, valid_until)

    def try_acquire(
        self,
        resources: Iterable[Resource],
        *,
        owner: str,
        now_s: float | None = None,
        lease_duration_s: float | None = None,
    ) -> ResourceLease | None:
        try:
            return self.acquire(
                resources,
                owner=owner,
                now_s=now_s,
                lease_duration_s=lease_duration_s,
            )
        except ResourceConflictError:
            return None

    def release(
        self,
        resources: Iterable[Resource] | str | None = None,
        *,
        owner: str | None = None,
    ) -> frozenset[Resource]:
        # ``release("step")`` is a convenient shorthand for releasing every
        # resource owned by that step.
        if isinstance(resources, str):
            if owner is not None:
                raise TypeError("owner was supplied twice")
            owner = resources
            resources = None
        if not owner or not owner.strip():
            raise ValueError("resource owner must be non-empty")
        requested = None if resources is None else _normalise_resources(resources)
        with self._lock:
            targets = (
                tuple(self._owners)
                if requested is None
                else tuple(requested)
            )
            released: set[Resource] = set()
            for resource in targets:
                ownership = self._owners.get(resource)
                if ownership is not None and ownership.owner == owner:
                    del self._owners[resource]
                    released.add(resource)
            return frozenset(released)

    def release_lease(self, lease: ResourceLease) -> frozenset[Resource]:
        return self.release(lease.resources, owner=lease.owner)

    def expire(self, now_s: float) -> frozenset[Resource]:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        with self._lock:
            before = set(self._owners)
            self._expire_locked(now)
            return frozenset(before.difference(self._owners))

    def _expire_locked(self, now_s: float | None) -> None:
        if now_s is None:
            return
        expired = [
            resource
            for resource, ownership in self._owners.items()
            if ownership.valid_until_s is not None and now_s > ownership.valid_until_s
        ]
        for resource in expired:
            del self._owners[resource]

    def owned_resources(self, owner: str, *, now_s: float | None = None) -> frozenset[Resource]:
        with self._lock:
            self._expire_locked(None if now_s is None else float(now_s))
            return frozenset(
                resource
                for resource, ownership in self._owners.items()
                if ownership.owner == owner
            )

    def owns(
        self,
        owner: str,
        resources: Iterable[Resource],
        *,
        now_s: float | None = None,
    ) -> bool:
        requested = _normalise_resources(resources)
        return requested.issubset(self.owned_resources(owner, now_s=now_s))

    def owner_of(self, resource: Resource, *, now_s: float | None = None) -> str | None:
        with self._lock:
            self._expire_locked(None if now_s is None else float(now_s))
            ownership = self._owners.get(resource)
            return None if ownership is None else ownership.owner

    @property
    def owners(self) -> dict[Resource, str]:
        with self._lock:
            return {resource: ownership.owner for resource, ownership in self._owners.items()}

    def clear(self) -> None:
        with self._lock:
            self._owners.clear()


def _contains_non_finite(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> bool:
    if _depth > 12 or value is None or isinstance(value, (str, bytes, bool, Enum)):
        return False
    if isinstance(value, (int, float)):
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return True
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_non_finite(item, _seen=seen, _depth=_depth + 1)
            for item in value.values()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(
            _contains_non_finite(item, _seen=seen, _depth=_depth + 1)
            for item in value
        )
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return _contains_non_finite(attributes, _seen=seen, _depth=_depth + 1)
    # Numpy arrays and similar command containers expose ``tolist`` without
    # requiring numpy as a scheduler dependency.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _contains_non_finite(tolist(), _seen=seen, _depth=_depth + 1)
        except Exception:
            return True
    return False


class CommandValidator:
    """Reject expired, non-finite, or unauthorised command frames."""

    def __init__(self, resource_manager: ResourceManager | None = None) -> None:
        self.resource_manager = resource_manager

    def validate(
        self,
        frame: CommandFrame,
        *,
        owned_resources: Iterable[Resource] | None = None,
        now_s: float | None = None,
    ) -> CommandFrame:
        if not isinstance(frame, CommandFrame):
            raise TypeError("frame must be a CommandFrame")
        if frame.valid_until_s is not None:
            if not math.isfinite(frame.valid_until_s):
                raise CommandValidationError(
                    FailureCode.COMMAND_NON_FINITE,
                    "command validity deadline is non-finite",
                )
            if now_s is not None and float(now_s) > frame.valid_until_s:
                raise CommandValidationError(
                    FailureCode.COMMAND_EXPIRED,
                    f"command owned by {frame.owner_step_id!r} has expired",
                )
        if frame.base_command is not None and not frame.base_command.is_finite:
            raise CommandValidationError(
                FailureCode.COMMAND_NON_FINITE,
                "base command contains NaN or infinity",
            )
        if _contains_non_finite(frame.arm_command):
            raise CommandValidationError(
                FailureCode.COMMAND_NON_FINITE,
                "arm command contains NaN or infinity",
            )
        if frame.arm_mode is ArmCommandMode.MOVE and frame.arm_command is None:
            raise CommandValidationError(
                FailureCode.INTERNAL_ERROR,
                "arm MOVE mode requires an arm command",
            )

        required = frame.required_resources
        if owned_resources is not None:
            owned = _normalise_resources(owned_resources)
        elif self.resource_manager is not None:
            owned = self.resource_manager.owned_resources(frame.owner_step_id, now_s=now_s)
        else:
            raise ValueError("owned_resources or a ResourceManager is required")
        missing = required.difference(owned)
        if missing:
            names = ", ".join(sorted(resource.value for resource in missing))
            raise CommandValidationError(
                FailureCode.RESOURCE_CONFLICT,
                f"owner {frame.owner_step_id!r} emitted commands without resources: {names}",
            )
        return frame


@dataclass(frozen=True)
class BaseLeaseSnapshot:
    owner: str | None
    command: BaseCommand
    valid_until_s: float | None
    active: bool


class BaseCommandLease:
    """A base command automatically resolves to zero unless renewed in time."""

    def __init__(
        self,
        lease_duration_s: float = 0.15,
        *,
        duration_s: float | None = None,
    ) -> None:
        duration = float(lease_duration_s if duration_s is None else duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("base lease duration must be finite and positive")
        self.lease_duration_s = duration
        self._owner: str | None = None
        self._command = BaseCommand.zero()
        self._valid_until_s: float | None = None
        self._lock = threading.RLock()

    def renew(
        self,
        owner: str,
        command: BaseCommand | tuple[float, float],
        now_s: float,
        *,
        duration_s: float | None = None,
    ) -> BaseLeaseSnapshot:
        now = float(now_s)
        duration = self.lease_duration_s if duration_s is None else float(duration_s)
        if not math.isfinite(now) or not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("lease time and duration must be finite; duration must be positive")
        if not owner or not owner.strip():
            raise ValueError("base lease owner must be non-empty")
        if not isinstance(command, BaseCommand):
            command = BaseCommand(*command)
        if not command.is_finite:
            raise CommandValidationError(
                FailureCode.COMMAND_NON_FINITE,
                "base lease command contains NaN or infinity",
            )
        with self._lock:
            if (
                self._owner is not None
                and self._owner != owner
                and self._valid_until_s is not None
                and now <= self._valid_until_s
            ):
                raise ResourceConflictError(
                    f"base lease is still held by {self._owner!r}",
                    conflicts={Resource.BASE: self._owner},
                )
            self._owner = owner
            self._command = command
            self._valid_until_s = now + duration
            return BaseLeaseSnapshot(owner, command, self._valid_until_s, True)

    def resolve(self, now_s: float) -> BaseCommand:
        return self.snapshot(now_s).command

    current = resolve
    command_at = resolve

    def snapshot(self, now_s: float) -> BaseLeaseSnapshot:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        with self._lock:
            active = (
                self._owner is not None
                and self._valid_until_s is not None
                and now <= self._valid_until_s
            )
            if not active:
                return BaseLeaseSnapshot(None, BaseCommand.zero(), None, False)
            return BaseLeaseSnapshot(
                self._owner,
                self._command,
                self._valid_until_s,
                True,
            )

    def revoke(self, *, owner: str | None = None) -> None:
        with self._lock:
            if owner is not None and self._owner not in (None, owner):
                raise ResourceConflictError(
                    f"owner {owner!r} cannot revoke base lease held by {self._owner!r}",
                    conflicts={Resource.BASE: self._owner or ""},
                )
            self._owner = None
            self._command = BaseCommand.zero()
            self._valid_until_s = None

    clear = revoke


__all__ = [
    "BaseCommandLease",
    "BaseLeaseSnapshot",
    "CommandValidationError",
    "CommandValidator",
    "ResourceConflictError",
    "ResourceLease",
    "ResourceManager",
]
