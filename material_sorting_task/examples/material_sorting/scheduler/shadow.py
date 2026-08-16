"""Read-only validation of legacy controller traces against v2 task plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scheduler.plans import ExecutorTaskPlan


@dataclass(frozen=True)
class ShadowDivergence:
    transition_serial: int
    reason: str


class ShadowTraceValidator:
    """Validate plan membership without ticking a motion executor twice."""

    def __init__(self, plans: Mapping[int, ExecutorTaskPlan]) -> None:
        self._plans = dict(plans)
        self._seen_serial = -1
        self.divergences: list[ShadowDivergence] = []

    def observe(self, snapshot: Any) -> None:
        serial = int(getattr(snapshot, "transition_serial", -1))
        if serial == self._seen_serial:
            return
        self._seen_serial = serial
        task_id = getattr(snapshot, "task_id", None)
        stage = getattr(snapshot, "stage", None)
        if task_id is None or stage is None:
            return
        plan = self._plans.get(int(task_id))
        if plan is None:
            self._record(serial, f"task {task_id} has no v2 plan")
            return
        if stage not in {spec.stage for spec in plan.stages}:
            self._record(serial, f"stage {stage!r} is absent from task {task_id} plan")

    def _record(self, serial: int, reason: str) -> None:
        self.divergences.append(ShadowDivergence(serial, reason))

    @property
    def healthy(self) -> bool:
        return not self.divergences


__all__ = ["ShadowDivergence", "ShadowTraceValidator"]
