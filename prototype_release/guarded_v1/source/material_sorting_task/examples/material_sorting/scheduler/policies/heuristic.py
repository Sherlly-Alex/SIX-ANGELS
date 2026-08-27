"""Safe deterministic baseline policy for scheduler macro actions."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from navigation.carried_envelope import CarriedEnvelopeChecker
from navigation.costmap import PathMetrics, WorldCostmap, WorldCostmapSnapshot
from navigation.navigation_types import Pose2D
from navigation.robot_geometry import FootprintMode
from scheduler.candidate_generator import CandidateAction
from scheduler.utility import (
    CandidateEvaluation,
    UtilityWeights,
    rank_candidates,
)


class HeuristicPolicy:
    """Rank candidates using hard safety followed by weighted critics.

    If a live ``WorldCostmap`` is supplied, exactly one snapshot is acquired
    for the entire call.  This guarantees deterministic within-cycle
    comparisons even when perception updates concurrently.
    """

    def __init__(
        self,
        *,
        weights: UtilityWeights = UtilityWeights(),
        min_clearance_m: float = 0.02,
    ) -> None:
        min_clearance_m = float(min_clearance_m)
        if not math.isfinite(min_clearance_m) or min_clearance_m < 0.0:
            raise ValueError("min_clearance_m must be finite and >= 0")
        self._weights = weights
        self._min_clearance_m = min_clearance_m

    @property
    def weights(self) -> UtilityWeights:
        return self._weights

    @property
    def min_clearance_m(self) -> float:
        return self._min_clearance_m

    def rank(
        self,
        candidates: Sequence[CandidateAction],
        *,
        costmap: Optional[WorldCostmap | WorldCostmapSnapshot] = None,
        start_pose: Optional[Pose2D | Sequence[float]] = None,
        footprint_mode: FootprintMode | str = FootprintMode.TRANSIT_STOWED,
        constraints: Optional[
            Mapping[str, bool] | Mapping[str, Mapping[str, bool]]
        ] = None,
        now_s: Optional[float] = None,
        held_center_base: Optional[Tuple[float, float, float]] = None,
        held_half_width_m: Optional[float] = None,
        carried_checker: Optional[CarriedEnvelopeChecker] = None,
        inflation_radius: float = 1.0,
        planner_min_clearance_m: float = 0.22,
        planner_cost_weight: float = 4.0,
    ) -> Tuple[CandidateEvaluation, ...]:
        candidate_tuple = tuple(candidates)
        if len({item.action_id for item in candidate_tuple}) != len(candidate_tuple):
            raise ValueError("candidate action_id values must be unique")
        snapshot: Optional[WorldCostmapSnapshot]
        if isinstance(costmap, WorldCostmap):
            snapshot = costmap.snapshot(now_s=now_s)
        elif isinstance(costmap, WorldCostmapSnapshot) or costmap is None:
            snapshot = costmap
        else:
            raise TypeError("costmap must be WorldCostmap or WorldCostmapSnapshot")

        pose = self._pose_tuple(start_pose)
        metrics_by_id: dict[str, PathMetrics] = {}
        require_path = snapshot is not None
        for candidate in candidate_tuple:
            if not candidate.is_navigation:
                continue
            if snapshot is None:
                continue
            if pose is None:
                metrics_by_id[candidate.action_id] = PathMetrics.unreachable(
                    "start pose is required for costmap path evaluation"
                )
                continue
            if candidate.goal_pose is None:
                metrics_by_id[candidate.action_id] = PathMetrics.unreachable(
                    "navigation candidate has no goal pose"
                )
                continue
            metrics_by_id[candidate.action_id] = snapshot.plan_path(
                pose,
                candidate.goal_pose,
                footprint_mode=footprint_mode,
                inflation_radius=inflation_radius,
                min_clearance=planner_min_clearance_m,
                cost_weight=planner_cost_weight,
                held_center_base=held_center_base,
                held_half_width_m=held_half_width_m,
                carried_checker=carried_checker,
            )

        constraints_by_id = self._constraints_by_id(candidate_tuple, constraints)
        return rank_candidates(
            candidate_tuple,
            weights=self._weights,
            path_metrics_by_id=metrics_by_id,
            constraints_by_id=constraints_by_id,
            min_clearance_m=self._min_clearance_m,
            require_path=require_path,
        )

    def select(
        self,
        candidates: Sequence[CandidateAction],
        **kwargs: Any,
    ) -> Optional[CandidateEvaluation]:
        """Return the best safe evaluation, or ``None`` when all are masked."""
        ranked = self.rank(candidates, **kwargs)
        return next((item for item in ranked if item.valid), None)

    def rank_actions(
        self,
        candidates: Sequence[CandidateAction],
        **kwargs: Any,
    ) -> Tuple[CandidateEvaluation, ...]:
        return self.rank(candidates, **kwargs)

    @staticmethod
    def _pose_tuple(
        pose: Optional[Pose2D | Sequence[float]],
    ) -> Optional[Tuple[float, float, float]]:
        if pose is None:
            return None
        if isinstance(pose, Pose2D):
            values = (float(pose.x), float(pose.y), float(pose.yaw))
        else:
            if len(pose) < 3:
                return None
            try:
                values = (float(pose[0]), float(pose[1]), float(pose[2]))
            except (TypeError, ValueError):
                return None
        return values if all(math.isfinite(value) for value in values) else None

    @staticmethod
    def _constraints_by_id(
        candidates: Sequence[CandidateAction],
        constraints: Optional[
            Mapping[str, bool] | Mapping[str, Mapping[str, bool]]
        ],
    ) -> Mapping[str, Mapping[str, bool]]:
        if not constraints:
            return {}
        if all(isinstance(value, Mapping) for value in constraints.values()):
            return {
                str(action_id): dict(value)  # type: ignore[arg-type]
                for action_id, value in constraints.items()
            }
        shared = {str(name): bool(value) for name, value in constraints.items()}
        return {candidate.action_id: shared for candidate in candidates}


__all__ = ["HeuristicPolicy"]
