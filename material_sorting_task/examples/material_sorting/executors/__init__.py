"""Task executor construction for the formal competition client."""

from __future__ import annotations

from executors.base import TaskExecutor
from executors.dry_run import DryRunTaskExecutor
from executors.task1 import (
    Task1ContactExecutor,
    Task1Executor,
    Task1LiftExecutor,
    Task1NavigationExecutor,
    Task1PregraspExecutor,
)
from executors.task2 import Task2Executor
from executors.task3 import Task3Executor

EXECUTION_MODES = (
    "stub",
    "dry_run",
    "nav_only",
    "pregrasp_only",
    "contact_only",
    "lift_only",
)


def build_task_executors(
    mode: str = "stub",
    *,
    dry_run_ticks_per_stage: int = 2,
) -> dict[int, TaskExecutor]:
    """Build task executors for safe formal or scheduling-only operation."""
    normalized = str(mode).strip().lower()
    if normalized == "stub":
        return {1: Task1Executor(), 2: Task2Executor(), 3: Task3Executor()}
    if normalized == "dry_run":
        return {
            task_id: DryRunTaskExecutor(task_id, dry_run_ticks_per_stage)
            for task_id in (1, 2, 3)
        }
    if normalized == "nav_only":
        return {
            1: Task1NavigationExecutor(),
            2: Task2Executor(),
            3: Task3Executor(),
        }
    if normalized == "pregrasp_only":
        return {
            1: Task1PregraspExecutor(),
            2: Task2Executor(),
            3: Task3Executor(),
        }
    if normalized == "contact_only":
        return {
            1: Task1ContactExecutor(),
            2: Task2Executor(),
            3: Task3Executor(),
        }
    if normalized == "lift_only":
        return {
            1: Task1LiftExecutor(),
            2: Task2Executor(),
            3: Task3Executor(),
        }
    raise ValueError(
        f"unsupported MATERIAL_EXECUTION_MODE={mode!r}; expected one of {EXECUTION_MODES}"
    )


__all__ = ["EXECUTION_MODES", "build_task_executors"]
