"""Idempotent parser and synchronisation guard for referee topics."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from .models import RefereeSnapshot


_TASK_PROGRESS_RE = re.compile(r"\btask\s*=\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_ATTEMPT_RE = re.compile(r"\battempt\s*=\s*(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"\bstep\s*=\s*([\w-]+)", re.IGNORECASE)
_SCORE_RE = re.compile(r"\bscore\s*=\s*(-?\d+)", re.IGNORECASE)
_TASKINFO_ID_PATTERNS = (
    re.compile(r"^\s*当前任务\s*[:：=]?\s*(\d+)"),
    re.compile(r"^\s*任务\s*(\d+)\s*[:：=]?"),
    re.compile(r"(?:^|[^A-Za-z])task\s*[=:\s]+\s*(\d+)", re.IGNORECASE),
)
_FINISHED_MARKERS = (
    "全部任务结束",
    "所有任务结束",
    "all tasks finished",
    "all_tasks_done",
    "game finished",
)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "done", "finished"}
    return bool(value)


def _parse_gameinfo(gameinfo: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(gameinfo, Mapping):
        data = dict(gameinfo)
        raw = str(data.get("raw", ""))
    else:
        raw = "" if gameinfo is None else str(gameinfo)
        data = {}
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, Mapping):
                data.update(decoded)

    parsed: dict[str, Any] = {"raw": raw}
    task_match = _TASK_PROGRESS_RE.search(raw)
    if task_match:
        parsed["task_ordinal"] = int(task_match.group(1))
        parsed["task_total"] = int(task_match.group(2))
    attempt_match = _ATTEMPT_RE.search(raw)
    if attempt_match:
        parsed["attempts_completed"] = int(attempt_match.group(1))
    step_match = _STEP_RE.search(raw)
    if step_match:
        parsed["step"] = step_match.group(1)
    score_match = _SCORE_RE.search(raw)
    if score_match:
        parsed["score"] = int(score_match.group(1))

    # Explicit structured fields are authoritative over their loose raw form.
    aliases = {
        "task_ordinal": ("task_ordinal", "task_index", "current_task"),
        "task_total": ("task_total", "total_tasks"),
        "attempts_completed": ("attempts_completed", "attempt"),
        "step": ("step",),
        "score": ("score", "total_score"),
    }
    for target, names in aliases.items():
        for name in names:
            if name in data and data[name] is not None:
                parsed[target] = data[name]
                break
    parsed["all_tasks_done"] = any(
        _truthy(data.get(key))
        for key in ("all_tasks_done", "finished", "game_finished")
        if key in data
    ) or any(marker in raw.casefold() for marker in _FINISHED_MARKERS)
    return parsed


def _parse_task_id(taskinfo: str) -> int | None:
    for pattern in _TASKINFO_ID_PATTERNS:
        match = pattern.search(taskinfo)
        if match:
            return int(match.group(1))
    return None


def _taskinfo_finished(taskinfo: str) -> bool:
    normalized = taskinfo.casefold()
    return any(marker in normalized for marker in _FINISHED_MARKERS)


@dataclass(frozen=True)
class RefereeUpdate:
    """One gateway observation with semantic change and desync indicators."""

    snapshot: RefereeSnapshot
    previous: RefereeSnapshot | None
    changed: bool
    desynchronised: bool = False
    desync_reasons: tuple[str, ...] = ()
    observation_serial: int = 0
    revision: int = 0

    @property
    def task_ordinal(self) -> int | None:
        return self.snapshot.task_ordinal

    @property
    def task_total(self) -> int | None:
        return self.snapshot.task_total

    @property
    def task_id(self) -> int | None:
        return self.snapshot.task_id

    @property
    def attempts_completed(self) -> int:
        return self.snapshot.attempts_completed

    @property
    def all_tasks_done(self) -> bool:
        return self.snapshot.all_tasks_done

    @property
    def step(self) -> str | None:
        return self.snapshot.step

    @property
    def score(self) -> int | None:
        return self.snapshot.score

    @property
    def desynchronized(self) -> bool:
        """US spelling alias."""

        return self.desynchronised


class RefereeGateway:
    """Canonicalise referee messages and ignore repeated semantic snapshots.

    The server remains authoritative: this class reports regressions and topic
    disagreement but never invents local task progression.
    """

    def __init__(self) -> None:
        self._last: RefereeSnapshot | None = None
        self._last_consistent: RefereeSnapshot | None = None
        self._last_key: tuple[Any, ...] | None = None
        self._last_state_key: tuple[Any, ...] | None = None
        self._last_desync_reasons: tuple[str, ...] = ()
        self._observation_serial = 0
        self._revision = 0
        self._lock = threading.RLock()

    def observe(
        self,
        gameinfo: str | Mapping[str, Any] | None,
        taskinfo: str | None = "",
    ) -> RefereeUpdate:
        task_text = "" if taskinfo is None else str(taskinfo)
        parsed = _parse_gameinfo(gameinfo)
        task_id = _parse_task_id(task_text)

        ordinal = _optional_int(parsed.get("task_ordinal"))
        total = _optional_int(parsed.get("task_total"))
        attempts = _optional_int(parsed.get("attempts_completed"))
        score = _optional_int(parsed.get("score"))
        step_value = parsed.get("step")
        step = str(step_value) if step_value not in (None, "") else None
        all_done = bool(parsed.get("all_tasks_done")) or _taskinfo_finished(task_text)

        # In this competition task IDs are the ordered slots 1..3.  Using the
        # task topic as a fallback is useful during the first gameinfo tick;
        # when both are available we still retain both to detect disagreement.
        if ordinal is None and task_id is not None:
            ordinal = task_id

        raw_game = str(parsed.get("raw", ""))
        if isinstance(gameinfo, Mapping) and not raw_game:
            raw_game = str(gameinfo.get("raw", ""))
        snapshot = RefereeSnapshot(
            task_ordinal=ordinal,
            task_total=total,
            task_id=task_id,
            attempts_completed=max(0, attempts or 0),
            step=step,
            score=score,
            all_tasks_done=all_done,
            raw_gameinfo=raw_game,
            raw_taskinfo=task_text,
        )

        with self._lock:
            reasons = self._desync_reasons(snapshot, self._last_consistent)
            state_key = (
                snapshot.task_ordinal,
                snapshot.task_total,
                snapshot.task_id,
                snapshot.attempts_completed,
                snapshot.step.casefold() if snapshot.step else None,
                snapshot.score,
                snapshot.all_tasks_done,
            )
            # A repeated regressed snapshot must remain both desynchronised and
            # idempotent; comparing only with the immediately previous sample
            # would otherwise make a one-shot regression disappear.
            if state_key == self._last_state_key:
                reasons = self._last_desync_reasons
            semantic_key = state_key + (reasons,)
            changed = semantic_key != self._last_key
            self._observation_serial += 1
            if changed:
                self._revision += 1
            update = RefereeUpdate(
                snapshot=snapshot,
                previous=self._last,
                changed=changed,
                desynchronised=bool(reasons),
                desync_reasons=reasons,
                observation_serial=self._observation_serial,
                revision=self._revision,
            )
            self._last = snapshot
            if not reasons:
                self._last_consistent = snapshot
            self._last_key = semantic_key
            self._last_state_key = state_key
            self._last_desync_reasons = reasons
            return update

    @staticmethod
    def _desync_reasons(
        current: RefereeSnapshot,
        previous: RefereeSnapshot | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if current.task_ordinal is not None and current.task_ordinal < 1:
            reasons.append("task ordinal is below one")
        if current.task_total is not None and current.task_total < 1:
            reasons.append("task total is below one")
        if (
            current.task_ordinal is not None
            and current.task_total is not None
            and current.task_ordinal > current.task_total
        ):
            reasons.append("task ordinal exceeds task total")
        if (
            current.task_id is not None
            and current.task_ordinal is not None
            and current.task_id != current.task_ordinal
        ):
            reasons.append("gameinfo and taskinfo disagree on current task")

        if previous is not None:
            if previous.all_tasks_done and not current.all_tasks_done:
                reasons.append("referee completion regressed")
            if (
                previous.task_total is not None
                and current.task_total is not None
                and previous.task_total != current.task_total
            ):
                reasons.append("task total changed during the game")
            if previous.task_ordinal is not None and current.task_ordinal is not None:
                if current.task_ordinal < previous.task_ordinal:
                    reasons.append("task ordinal regressed")
                elif current.task_ordinal > previous.task_ordinal + 1:
                    reasons.append("task ordinal skipped a task")
                elif (
                    current.task_ordinal == previous.task_ordinal
                    and current.attempts_completed < previous.attempts_completed
                ):
                    reasons.append("attempt count regressed within the current task")
        return tuple(reasons)

    @property
    def snapshot(self) -> RefereeSnapshot | None:
        with self._lock:
            return self._last

    def reset(self) -> None:
        with self._lock:
            self._last = None
            self._last_consistent = None
            self._last_key = None
            self._last_state_key = None
            self._last_desync_reasons = ()
            self._observation_serial = 0
            self._revision = 0


__all__ = ["RefereeGateway", "RefereeUpdate"]
