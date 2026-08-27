"""Deterministic macro-action candidates for scheduler policy evaluation.

Candidates are intentionally small, immutable feature records.  The generator
does not decide which action is safe or best; ``utility`` and the costmap do
that using one coherent world snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from navigation.navigation_types import NavigationGoal


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CandidateAction:
    """One bounded scheduler action and its policy-facing prior features.

    Navigation actions have finite ``x/y/yaw``.  Recovery actions such as
    ``rescan`` or ``replan`` intentionally leave them as ``None`` and are
    scored without path critics.
    """

    action_id: str
    action_type: str = "navigate"
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
    expected_score: float = 0.0
    success_probability: float = 1.0
    expected_time_s: float = 0.0
    perception_uncertainty: float = 0.0
    manipulation_difficulty: float = 0.0
    irreversible_risk: float = 0.0
    recovery_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hard_constraints: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_id = str(self.action_id).strip()
        action_type = str(self.action_type).strip().lower()
        if not action_id:
            raise ValueError("candidate action_id must be non-empty")
        if not action_type:
            raise ValueError("candidate action_type must be non-empty")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_type", action_type)
        for name in ("x", "y", "yaw"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else float(value))
        for name in (
            "expected_score",
            "success_probability",
            "expected_time_s",
            "perception_uncertainty",
            "manipulation_difficulty",
            "irreversible_risk",
            "recovery_cost",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        object.__setattr__(
            self,
            "hard_constraints",
            MappingProxyType({
                str(name): bool(allowed)
                for name, allowed in self.hard_constraints.items()
            }),
        )

    @property
    def id(self) -> str:
        return self.action_id

    @property
    def is_navigation(self) -> bool:
        return self.action_type in {"navigate", "observation_stand", "retreat"}

    @property
    def goal_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.x is None or self.y is None or self.yaw is None:
            return None
        return (self.x, self.y, self.yaw)


class CandidateGenerator:
    """Generate a stable centre/left/right stand set around a known goal."""

    DEFAULT_LATERAL_OFFSETS_M = (0.0, 0.08, -0.08)

    def __init__(
        self,
        *,
        lateral_offsets_m: Iterable[float] = DEFAULT_LATERAL_OFFSETS_M,
    ) -> None:
        offsets = tuple(float(value) for value in lateral_offsets_m)
        if not offsets:
            raise ValueError("at least one lateral offset is required")
        if not all(math.isfinite(value) for value in offsets):
            raise ValueError("candidate offsets must be finite")
        if len(set(offsets)) != len(offsets):
            raise ValueError("candidate offsets must be unique")
        self._offsets = offsets

    @property
    def lateral_offsets_m(self) -> Tuple[float, ...]:
        return self._offsets

    def generate(
        self,
        base_goal: NavigationGoal | Sequence[float],
        *,
        task_id: Optional[int] = None,
        step_id: str = "navigate",
        action_type: str = "navigate",
        expected_score: float = 0.0,
        success_probability: float = 1.0,
        expected_time_s: float = 0.0,
        perception_uncertainty: float = 0.0,
        manipulation_difficulty: float = 0.0,
        irreversible_risk: float = 0.0,
        recovery_cost: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
        hard_constraints: Optional[Mapping[str, bool]] = None,
        include_recovery: bool = False,
    ) -> Tuple[CandidateAction, ...]:
        """Generate lateral stand alternatives in the goal's body frame.

        Positive offset is goal-left, negative offset goal-right.  This works
        for shelf (yaw=pi), table (yaw=pi/2) and end-zone goals without
        embedding scene coordinates in the scheduler.
        """
        x, y, yaw, source_tag = self._goal_tuple(base_goal)
        prefix = f"task{task_id}:" if task_id is not None else ""
        step = str(step_id).strip() or "navigate"
        common_metadata = dict(metadata or {})
        common_metadata.update({
            "task_id": task_id,
            "step_id": step,
            "base_goal": (x, y, yaw),
            "source_tag": source_tag,
        })
        candidates = []
        for index, offset in enumerate(self._offsets):
            side = "center" if abs(offset) <= 1e-12 else ("left" if offset > 0.0 else "right")
            candidate_x = x - math.sin(yaw) * offset
            candidate_y = y + math.cos(yaw) * offset
            item_metadata = dict(common_metadata)
            item_metadata.update({
                "lateral_offset_m": offset,
                "candidate_index": index,
                "side": side,
            })
            candidates.append(CandidateAction(
                action_id=f"{prefix}{step}:stand:{side}",
                action_type=action_type,
                x=candidate_x,
                y=candidate_y,
                yaw=yaw,
                expected_score=expected_score,
                success_probability=success_probability,
                expected_time_s=expected_time_s,
                perception_uncertainty=perception_uncertainty,
                manipulation_difficulty=manipulation_difficulty,
                irreversible_risk=irreversible_risk,
                recovery_cost=recovery_cost,
                metadata=item_metadata,
                hard_constraints=hard_constraints or {},
            ))

        if include_recovery:
            recovery_specs = (
                ("rescan", 1.0, 0.15),
                ("replan", 1.5, 0.25),
                ("safe_retreat", 2.0, 0.35),
            )
            for name, extra_time, extra_cost in recovery_specs:
                candidates.append(CandidateAction(
                    action_id=f"{prefix}{step}:recovery:{name}",
                    action_type=name,
                    expected_score=expected_score,
                    success_probability=success_probability,
                    expected_time_s=expected_time_s + extra_time,
                    perception_uncertainty=perception_uncertainty,
                    manipulation_difficulty=manipulation_difficulty,
                    irreversible_risk=irreversible_risk,
                    recovery_cost=recovery_cost + extra_cost,
                    metadata=common_metadata,
                    hard_constraints=hard_constraints or {},
                ))
        return tuple(candidates)

    def generate_navigation_candidates(self, *args: Any, **kwargs: Any) -> Tuple[CandidateAction, ...]:
        return self.generate(*args, **kwargs)

    def generate_around_goal(self, *args: Any, **kwargs: Any) -> Tuple[CandidateAction, ...]:
        return self.generate(*args, **kwargs)

    @staticmethod
    def _goal_tuple(
        goal: NavigationGoal | Sequence[float],
    ) -> Tuple[float, float, float, str]:
        if isinstance(goal, NavigationGoal):
            values = (float(goal.x), float(goal.y), float(goal.yaw))
            source = goal.source_tag
        else:
            if len(goal) < 3:
                raise ValueError("base goal must contain x, y and yaw")
            values = (float(goal[0]), float(goal[1]), float(goal[2]))
            source = "pose"
        if not all(math.isfinite(value) for value in values):
            raise ValueError("base goal must be finite")
        return values[0], values[1], values[2], source


__all__ = ["CandidateAction", "CandidateGenerator"]
