"""Pure helpers for multi-task instruction consumption (competition rules).

Formal path uses structured /material/instruction fields: task, place_type,
place_world, target_color, optional ref_prop for shelf_prop_side.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

TABLE_TOP_Z_DEFAULT = 0.747
PLACE_SIDE_DX = 0.22


def sorted_instructions(instructions: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    if not instructions:
        return []
    return sorted(
        [dict(t) for t in instructions],
        key=lambda t: (t.get("task") is None, int(t["task"]) if t.get("task") is not None else 0),
    )


def current_instruction(
    instructions: Sequence[Mapping[str, Any]] | None,
    task_index: int,
) -> dict | None:
    tasks = sorted_instructions(instructions)
    if task_index < 0 or task_index >= len(tasks):
        return None
    return tasks[task_index]


def startup_target_color(
    instructions: Sequence[Mapping[str, Any]] | None,
    gameinfo: Mapping[str, Any] | None,
) -> str:
    """Target colour required for a fresh task-1 start.

    Later tasks first navigate to their calibrated observation stands, where
    their executors wait for a fresh target.  Requiring those detections before
    startup deadlocks recovery when the current camera view cannot see them.
    """

    tasks = sorted_instructions(instructions)
    if not tasks:
        return ""
    ordinal = 1
    if gameinfo:
        try:
            ordinal = int(gameinfo.get("task_ordinal", 1))
        except (TypeError, ValueError):
            ordinal = 1
    index = min(len(tasks) - 1, max(0, ordinal - 1))
    if index > 0:
        return ""
    return str(tasks[index].get("target_color", "")).strip().lower()


def is_shelf_pick_task(ins: Mapping[str, Any] | None) -> bool:
    """Task 2 picks from the shelf; tasks 1/3 pick from the table."""
    if ins is None:
        return False
    task_id = ins.get("task")
    if task_id == 2:
        return True
    if task_id in (1, 3):
        return False
    # Heuristic from Chinese text when task id missing.
    text = str(ins.get("instruction") or "")
    return "货架中" in text or "货架上" in text


def is_shelf_place_type(ins: Mapping[str, Any] | None) -> bool:
    if ins is None:
        return False
    return ins.get("place_type") in ("shelf_point", "shelf_prop_side")


def place_world_from_instruction(
    ins: Mapping[str, Any] | None,
    *,
    props: Mapping[str, Mapping[str, Any]] | None = None,
    table_top_z: float = TABLE_TOP_Z_DEFAULT,
    place_side_dx: float = PLACE_SIDE_DX,
    fallback: Sequence[float] | None = None,
) -> np.ndarray:
    """Resolve place XYZ from formal fields; never require ref_prop for task 2."""
    if fallback is None:
        fallback = (0.15, 2.55, table_top_z + 0.12)
    if not ins:
        return np.asarray(fallback, dtype=float)

    pw = ins.get("place_world")
    if pw is not None and len(pw) >= 2:
        z = float(pw[2]) if len(pw) > 2 else float(table_top_z + 0.12)
        return np.array([float(pw[0]), float(pw[1]), z], dtype=float)

    place_type = ins.get("place_type")
    ref = ins.get("ref_prop")
    if place_type == "shelf_prop_side" and ref and props:
        prop = props.get(ref)
        if prop is not None and "world_position" in prop:
            pp = np.asarray(prop["world_position"], dtype=float)
            direction = ins.get("direction") or "left"
            dx = -float(place_side_dx) if direction == "left" else float(place_side_dx)
            x = float(np.clip(pp[0] + dx, -1.55, 0.32))
            z = float(pp[2]) if len(pp) > 2 else float(table_top_z + 0.12)
            return np.array([x, float(pp[1]), z], dtype=float)

    return np.asarray(fallback, dtype=float)


_GAMEINFO_TASK_RE = re.compile(r"task=(\d+)\s*/\s*(\d+)")
_GAMEINFO_ATTEMPT_RE = re.compile(r"attempt=(\d+)")
_GAMEINFO_STEP_RE = re.compile(r"step=([A-Za-z_-]+)")


def parse_gameinfo(text: str) -> dict[str, Any]:
    """Loose parse of referee.game_info string."""
    out: dict[str, Any] = {"raw": text}
    if not text:
        return out
    m = _GAMEINFO_TASK_RE.search(text)
    if m:
        out["task_ordinal"] = int(m.group(1))  # 1-based current task slot
        out["task_total"] = int(m.group(2))
    a = _GAMEINFO_ATTEMPT_RE.search(text)
    if a:
        out["attempt"] = int(a.group(1))
    step = _GAMEINFO_STEP_RE.search(text)
    if step:
        out["step"] = step.group(1)
    return out


def referee_place_confirmed(
    gameinfo: Mapping[str, Any] | None,
    task_index: int,
) -> bool:
    """True when the official referee reports ``step=place`` for this task.

    The current official offline Server does not publish a
    ``/material/place_confirmed`` topic.  Its supported placement event is the
    ``step=place`` field in ``/referee/gameinfo``.
    """

    if not gameinfo:
        return False
    try:
        ordinal = int(gameinfo.get("task_ordinal", -1))
    except (TypeError, ValueError):
        return False
    step = str(gameinfo.get("step", "")).strip().lower()
    return ordinal == int(task_index) + 1 and step == "place"


def parse_taskinfo_task_id(text: str) -> int | None:
    """Extract task id from '任务N: ...' / 'Task N' / 'task=N' strings."""
    if not text:
        return None
    m = re.match(r"\s*当前任务\s*[:：=]?\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.match(r"\s*任务\s*(\d+)\s*[:：=]?", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[^A-Za-z])(?:task|Task|TASK)\s*[=:\s]+\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None
