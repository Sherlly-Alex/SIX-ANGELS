"""Project-specific candidate generation for the v2 scheduling sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from executors.base import ExecutionContext, TaskStage
from navigation.costmap import DynamicObstacle, WorldCostmap
from navigation.dynamic_overlay import volumes_from_detections
from navigation.robot_geometry import FootprintMode
from scheduler.candidate_generator import CandidateAction, CandidateGenerator


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[CandidateAction, ...]
    start_pose: tuple[float, float, float]
    costmap: WorldCostmap
    constraints: Mapping[str, bool]
    footprint_mode: FootprintMode
    world_state: Mapping[str, Any]


class ProjectCandidateProvider:
    """Build navigation alternatives from current odometry and observations.

    The provider is read-only with respect to executors.  It refreshes the
    dynamic costmap, computes finite center/left/right macro-actions, and
    returns a batch for ranking.  Applying the selected candidate requires an
    explicit ``apply_scheduler_candidate`` hook on an executor.
    """

    NAVIGATION_STAGES = frozenset(
        {
            TaskStage.NAVIGATE_TO_PICK,
            TaskStage.TRANSPORT,
            TaskStage.RETURN_TO_END,
        }
    )

    def __init__(
        self,
        *,
        costmap: WorldCostmap | None = None,
        generator: CandidateGenerator | None = None,
        nominal_speed_mps: float = 0.15,
        layout_path: str | Path | None = None,
    ) -> None:
        speed = float(nominal_speed_mps)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("nominal_speed_mps must be finite and positive")
        self.costmap = costmap or WorldCostmap()
        self.generator = generator or CandidateGenerator()
        self.nominal_speed_mps = speed
        path = (
            Path(layout_path)
            if layout_path is not None
            else Path(__file__).resolve().parents[1]
            / "material_competition_layout.json"
        )
        self._end_goal = self._read_end_goal(path)

    def build(self, context: ExecutionContext, stage_spec: Any) -> CandidateBatch | None:
        stage = getattr(stage_spec, "stage", stage_spec)
        if stage not in self.NAVIGATION_STAGES:
            return None
        pose = self._odometry_pose(context.odometry)
        if pose is None:
            return None
        instruction = dict(context.instruction)
        task_id = int(instruction.get("task", context.task_index + 1))
        goal = self._goal_for(
            stage,
            task_id,
            instruction,
            context.target_observations,
            now_s=float(context.now_s),
        )
        if goal is None:
            return None

        target_color = str(instruction.get("target_color", "")).strip().casefold()
        dynamic_obstacles = []
        for index, (key, observation) in enumerate(context.target_observations.items()):
            try:
                color = str(getattr(observation, "color", key)).strip().casefold()
                xyz = tuple(float(value) for value in observation.position_world[:3])
                score = min(1.0, max(0.0, float(getattr(observation, "score", 0.0))))
                observed_at_s = float(observation.received_at_s)
            except (AttributeError, TypeError, ValueError):
                continue
            if color == target_color:
                continue
            expires_at_s = observed_at_s + self.costmap.dynamic_ttl_s
            if (
                len(xyz) != 3
                or not all(math.isfinite(value) for value in xyz)
                or not math.isfinite(observed_at_s)
                or observed_at_s > float(context.now_s) + 0.05
                or expires_at_s <= float(context.now_s)
            ):
                continue
            volumes = volumes_from_detections([(color, xyz, score)])
            if not volumes:
                continue
            dynamic_obstacles.append(
                DynamicObstacle(
                    volume=volumes[0],
                    confidence=score,
                    observed_at_s=observed_at_s,
                    expires_at_s=expires_at_s,
                    source="scheduler_target_observations",
                    obstacle_id=f"target:{color}:{index}",
                    label=color,
                    kind="box",
                )
            )
        self.costmap.replace_dynamic_source(
            dynamic_obstacles,
            source="scheduler_target_observations",
            observed_at_s=float(context.now_s),
        )

        distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        confidence = self._target_confidence(
            target_color,
            context.target_observations,
            now_s=float(context.now_s),
            ttl_s=self.costmap.dynamic_ttl_s,
        )
        footprint = (
            FootprintMode.TRANSIT_CARRY
            if stage is TaskStage.TRANSPORT
            else (
                FootprintMode.DOCKING
                if stage is TaskStage.RETURN_TO_END
                else FootprintMode.TRANSIT_STOWED
            )
        )
        constraints = {
            "referee_allowed": not self._referee_finished(context),
            "step_allowed": stage in self.NAVIGATION_STAGES,
            "resource_available": "base" in getattr(stage_spec, "resources", ()),
            "irreversible_allowed": not bool(
                getattr(stage_spec, "irreversible", False)
            ),
        }
        candidates = self.generator.generate(
            goal,
            task_id=task_id,
            step_id=stage.value,
            expected_score=0.0,
            success_probability=0.55 + 0.40 * confidence,
            expected_time_s=distance / self.nominal_speed_mps,
            perception_uncertainty=1.0 - confidence,
            hard_constraints=constraints,
            metadata={"costmap_sidecar": True},
        )
        world_state = {
            "task_id": task_id,
            "attempt": int(context.attempt),
            "score": int(context.score),
            "target_confidence": confidence,
            "unsafe_collision": bool(context.unsafe_collision),
        }
        return CandidateBatch(
            candidates=candidates,
            start_pose=pose,
            costmap=self.costmap,
            constraints=constraints,
            footprint_mode=footprint,
            world_state=world_state,
        )

    @staticmethod
    def _referee_finished(context: ExecutionContext) -> bool:
        task_text = str(context.referee_taskinfo).casefold()
        raw = str(context.referee_gameinfo.get("raw", "")).casefold()
        return any(
            marker in task_text or marker in raw
            for marker in (
                "全部任务结束",
                "所有任务结束",
                "all tasks finished",
                "all_tasks_done",
            )
        )

    def _goal_for(
        self,
        stage: TaskStage,
        task_id: int,
        instruction: Mapping[str, Any],
        observations: Mapping[str, Any],
        *,
        now_s: float,
    ) -> tuple[float, float, float] | None:
        if stage is TaskStage.RETURN_TO_END:
            return self._end_goal
        if stage is TaskStage.TRANSPORT:
            place = instruction.get("place_world")
            if not isinstance(place, (tuple, list)) or len(place) < 2:
                return None
            try:
                x, y = float(place[0]), float(place[1])
            except (TypeError, ValueError):
                return None
            place_type = str(instruction.get("place_type", "")).casefold()
            if place_type in {"shelf_point", "shelf_prop_side"}:
                return (x + 0.90, y, math.pi)
            return (x, y - 0.65, math.pi / 2.0)

        color = str(instruction.get("target_color", "")).strip().casefold()
        observation = observations.get(color)
        if observation is None:
            return None
        try:
            received_at_s = float(observation.received_at_s)
            x, y = (
                float(observation.position_world[0]),
                float(observation.position_world[1]),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        age_s = now_s - received_at_s
        if (
            not all(math.isfinite(value) for value in (x, y, age_s))
            or age_s < -0.05
            or age_s >= self.costmap.dynamic_ttl_s
        ):
            return None
        if task_id == 2:
            return (x + 0.90, y, math.pi)
        return (x, y - 0.65, math.pi / 2.0)

    @staticmethod
    def _target_confidence(
        color: str,
        observations: Mapping[str, Any],
        *,
        now_s: float,
        ttl_s: float,
    ) -> float:
        observation = observations.get(color)
        if observation is None:
            return 0.0
        try:
            score = float(getattr(observation, "score", 0.0))
            observed_at_s = float(observation.received_at_s)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        age_s = now_s - observed_at_s
        if not math.isfinite(age_s) or age_s < -0.05 or age_s >= ttl_s:
            return 0.0
        return min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0

    @staticmethod
    def _odometry_pose(odometry: Any) -> tuple[float, float, float] | None:
        if odometry is None:
            return None
        try:
            position = odometry.pose.pose.position
            orientation = odometry.pose.pose.orientation
            x, y = float(position.x), float(position.y)
            qx, qy = float(orientation.x), float(orientation.y)
            qz, qw = float(orientation.z), float(orientation.w)
        except (AttributeError, TypeError, ValueError):
            return None
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return (x, y, yaw) if all(math.isfinite(v) for v in (x, y, yaw)) else None

    @staticmethod
    def _read_end_goal(path: Path) -> tuple[float, float, float]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            zone = data["scene"]["end_zone"]
            xs, ys = zone["x"], zone["y"]
            goal = (
                (float(xs[0]) + float(xs[1])) / 2.0,
                (float(ys[0]) + float(ys[1])) / 2.0,
                math.pi / 2.0,
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot derive end-zone goal from {path}: {exc}") from exc
        if not all(math.isfinite(value) for value in goal):
            raise ValueError("end-zone goal contains a non-finite value")
        return goal


__all__ = ["CandidateBatch", "ProjectCandidateProvider"]
