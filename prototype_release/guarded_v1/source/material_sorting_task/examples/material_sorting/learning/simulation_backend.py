"""Project-level deterministic simulator for scheduler policy experiments.

This backend deliberately models macro-action outcomes, not robot dynamics.
It uses the production CandidateAction, PathMetrics, hard-filter and
Multi-Critic schemas so offline policies see the same bounded action contract
as the Client. It never imports ROS, referee Server state or motor controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

from navigation.costmap import PathMetrics
from scheduler.candidate_generator import CandidateAction
from scheduler.utility import CandidateEvaluation, evaluate_candidate

from .domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizationSample,
    DomainRandomizer,
)
from .env import SchedulingEnv, SchedulingSnapshot, SchedulingTransition
from .reward import RewardEvent


SIMULATION_SCHEMA_VERSION = "scheduler-project-sim-v1"
IDENTIFIABLE_SIMULATION_SCHEMA_VERSION = "scheduler-project-sim-v2"
LEGACY_OUTCOME_MODEL = "oracle_visible"
CONTEXTUAL_OUTCOME_MODEL = "contextual_latent"
DEFAULT_PROJECT_SIMULATION_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "project_simulation_v1.json"
)


@dataclass(frozen=True)
class SimulationStage:
    task_id: int
    step_id: str
    payload_code: int
    base_path_length_m: float


DEFAULT_SIMULATION_STAGES = tuple(
    SimulationStage(task_id, step_id, payload, distance)
    for task_id, values in (
        (1, (("navigate_to_pick", 0, 1.20), ("transport", 2, 1.75), ("return_to_end", 0, 1.35))),
        (2, (("navigate_to_pick", 0, 1.55), ("transport", 2, 1.40), ("return_to_end", 0, 1.10))),
        (3, (("navigate_to_pick", 0, 1.30), ("transport", 2, 1.65), ("return_to_end", 0, 1.25))),
    )
    for step_id, payload, distance in values
)


@dataclass(frozen=True)
class ProjectSimulationConfig:
    stages: tuple[SimulationStage, ...] = DEFAULT_SIMULATION_STAGES
    # Keep the simulator on the production SchedulerDecisionConfig schema so
    # replay-trained models can be integrity-checked without adapters.
    max_candidates: int = 8
    minimum_clearance_m: float = 0.22
    remaining_time_s: float = 300.0
    randomization: DomainRandomizationConfig = DomainRandomizationConfig()
    schema_version: str = SIMULATION_SCHEMA_VERSION
    outcome_model: str = LEGACY_OUTCOME_MODEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("simulation requires at least one stage")
        if self.max_candidates < 4:
            raise ValueError("simulation max_candidates must be at least four")
        if (
            not math.isfinite(self.minimum_clearance_m)
            or self.minimum_clearance_m <= 0.0
        ):
            raise ValueError("minimum_clearance_m must be finite and positive")
        if not math.isfinite(self.remaining_time_s) or self.remaining_time_s <= 0.0:
            raise ValueError("remaining_time_s must be finite and positive")
        if self.schema_version not in {
            SIMULATION_SCHEMA_VERSION,
            IDENTIFIABLE_SIMULATION_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported simulation schema version")
        if self.outcome_model not in {
            LEGACY_OUTCOME_MODEL,
            CONTEXTUAL_OUTCOME_MODEL,
        }:
            raise ValueError("unsupported simulation outcome model")
        if (
            self.schema_version == IDENTIFIABLE_SIMULATION_SCHEMA_VERSION
            and self.outcome_model != CONTEXTUAL_OUTCOME_MODEL
        ):
            raise ValueError("v2 simulation schema requires contextual outcomes")
        if (
            self.outcome_model == CONTEXTUAL_OUTCOME_MODEL
            and self.schema_version != IDENTIFIABLE_SIMULATION_SCHEMA_VERSION
        ):
            raise ValueError("contextual outcomes require v2 simulation schema")
        for stage in self.stages:
            if not isinstance(stage, SimulationStage):
                raise TypeError("stages must contain SimulationStage values")
            if stage.task_id not in {1, 2, 3}:
                raise ValueError("simulation task_id must be 1, 2 or 3")
            if not stage.step_id:
                raise ValueError("simulation step_id must be non-empty")
            if not math.isfinite(stage.base_path_length_m) or stage.base_path_length_m <= 0:
                raise ValueError("base_path_length_m must be finite and positive")


class ProjectSchedulingSimulationBackend:
    """Seeded public-state macro simulator used by paired policy benchmarks."""

    def __init__(self, config: ProjectSimulationConfig | None = None) -> None:
        self.config = config or ProjectSimulationConfig()
        self._randomizer = DomainRandomizer(self.config.randomization)
        self._rng = random.Random()
        self._seed = 0
        self._stage_index = 0
        self._elapsed_s = 0.0
        self._recoveries = 0
        self._success = True
        self._replan_counts: list[int] = []
        self._samples: tuple[DomainRandomizationSample, ...] = ()
        self._success_draws: tuple[float, ...] = ()
        self._evaluations: tuple[CandidateEvaluation, ...] = ()

    @staticmethod
    def _side_sign(action_id: str) -> int:
        suffix = str(action_id).rsplit(":", 1)[-1]
        if suffix == "left":
            return 1
        if suffix == "right":
            return -1
        return 0

    def _potential_draw(self, action_id: str) -> float:
        """Return a stable potential-outcome draw for one state/action pair."""

        payload = (
            f"{self._seed}|{self._stage_index}|"
            f"{self._replan_counts[self._stage_index]}|{action_id}"
        ).encode("utf-8")
        raw = hashlib.sha256(payload).digest()[:8]
        return int.from_bytes(raw, "big") / float(1 << 64)

    def _true_success_probability(self, evaluation: CandidateEvaluation) -> float:
        """Private potential-outcome probability, never added to observations."""

        estimated = float(evaluation.candidate.success_probability)
        if self.config.outcome_model == LEGACY_OUTCOME_MODEL:
            return estimated
        if evaluation.candidate.action_type != "navigate":
            return estimated

        stage = self.config.stages[self._stage_index]
        sample = self._samples[self._stage_index]
        side_sign = self._side_sign(evaluation.action_id)
        task_bias = {1: -0.18, 2: 0.22, 3: -0.05}[stage.task_id]
        stage_bias = {
            "navigate_to_pick": -0.12,
            "transport": 0.18,
            "return_to_end": 0.06,
        }.get(stage.step_id, 0.0)
        payload_bias = 0.16 if stage.payload_code >= 2 else -0.04
        context_signal = (
            16.0 * sample.pose_dx_m
            + 4.0 * sample.yaw_delta_rad
            + task_bias
            + stage_bias
            + payload_bias
        )
        preferred_sign = 1 if context_signal >= 0.0 else -1
        interaction_scale = 0.12 if stage.payload_code >= 2 else 0.09
        if side_sign == preferred_sign:
            residual = interaction_scale
        elif side_sign == -preferred_sign:
            residual = -interaction_scale
        else:
            residual = -0.045 if abs(context_signal) >= 0.10 else 0.015
        if sample.detection_dropout and self._replan_counts[self._stage_index] == 0:
            residual -= 0.035 if side_sign == 0 else 0.0
        return min(0.995, max(0.50, estimated + residual))

    def counterfactual_outcome_probabilities(self) -> tuple[float | None, ...]:
        """Training labels for safe candidates; never part of policy input."""

        navigation_values = [
            self._true_success_probability(item)
            for item in self._evaluations
            if item.valid and item.candidate.action_type == "navigate"
        ]
        best_navigation = max(navigation_values, default=0.50)
        sample = self._samples[self._stage_index]
        recovery_needed = bool(
            self._replan_counts[self._stage_index] == 0
            and (sample.detection_dropout or sample.planner_failure)
        )
        values: list[float | None] = []
        for item in self._evaluations:
            if not item.valid:
                values.append(None)
            elif item.candidate.action_type == "replan":
                adjustment = 0.07 if recovery_needed else -0.12
                values.append(min(0.995, max(0.50, best_navigation + adjustment)))
            else:
                values.append(self._true_success_probability(item))
        return tuple(values) + (None,) * (self.config.max_candidates - len(values))

    def counterfactual_potential_successes(self) -> tuple[bool | None, ...]:
        legacy_draw = (
            self._success_draws[self._stage_index]
            if self.config.outcome_model == LEGACY_OUTCOME_MODEL
            else None
        )
        values = [
            (
                (
                    legacy_draw
                    if legacy_draw is not None
                    else self._potential_draw(item.action_id)
                )
                <= self._true_success_probability(item)
            )
            if item.valid and item.candidate.action_type == "navigate"
            else None
            for item in self._evaluations
        ]
        return tuple(values) + (None,) * (self.config.max_candidates - len(values))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SchedulingSnapshot:
        del options
        self._seed = int(seed or 0)
        self._stage_index = 0
        self._elapsed_s = 0.0
        self._recoveries = 0
        self._success = True
        self._replan_counts = [0 for _ in self.config.stages]
        self._randomizer.reset(self._seed)
        self._rng.seed(self._seed ^ 0x5A17)
        self._samples = tuple(
            self._randomizer.sample() for _ in self.config.stages
        )
        self._success_draws = tuple(
            self._rng.random() for _ in self.config.stages
        )
        return self._snapshot()

    def step(self, candidate: Any) -> SchedulingTransition:
        if self._stage_index >= len(self.config.stages):
            raise RuntimeError("simulation episode is already complete")
        action_id = str(getattr(candidate, "action_id", ""))
        selected = next(
            (item for item in self._evaluations if item.action_id == action_id),
            None,
        )
        if selected is None or not selected.valid:
            raise ValueError("backend received an absent or masked candidate")

        stage_index = self._stage_index
        sample = self._samples[stage_index]
        action = selected.candidate
        if action.action_type == "replan":
            elapsed_s = 1.5 + sample.message_latency_s
            path_length_m = 0.0
            obstacle_cost = 0.0
            self._recoveries += 1
            self._replan_counts[stage_index] += 1
            self._elapsed_s += elapsed_s
            replan_count = self._replan_counts[stage_index]
            return SchedulingTransition(
                snapshot=self._snapshot(),
                events=(
                    RewardEvent(
                        "local_recovery",
                        f"sim-{self._seed}:step:{stage_index}:replan:{replan_count}",
                    ),
                ),
                elapsed_s=elapsed_s,
                path_length_m=0.0,
                obstacle_cost=0.0,
                info={
                    "success": False,
                    "recovery_count": self._recoveries,
                    "recovery_increment": 1,
                    "safety_violation": False,
                    "simulation_schema_version": self.config.schema_version,
                    "stage_index": stage_index,
                    "selected_action_id": action_id,
                },
            )
        else:
            metrics = selected.path_metrics
            if metrics is None:
                raise RuntimeError("navigation candidate is missing path metrics")
            elapsed_s = (
                action.expected_time_s
                / max(sample.speed_scale * sample.friction_scale, 0.1)
                + sample.message_latency_s
            )
            path_length_m = metrics.path_length_m
            obstacle_cost = metrics.inflation_cost_integral

        true_probability = self._true_success_probability(selected)
        outcome_draw = (
            self._success_draws[stage_index]
            if self.config.outcome_model == LEGACY_OUTCOME_MODEL
            else self._potential_draw(action_id)
        )
        succeeded = outcome_draw <= true_probability
        safe_navigation = [
            item
            for item in self._evaluations
            if item.valid and item.candidate.action_type == "navigate"
        ]
        oracle = max(
            safe_navigation,
            key=lambda item: (
                self._true_success_probability(item),
                item.utility,
                item.action_id,
            ),
            default=None,
        )
        potential_successes = {
            item.action_id: (
                self._potential_draw(item.action_id)
                <= self._true_success_probability(item)
            )
            for item in safe_navigation
        }
        avoidable_failure = bool(
            not succeeded
            and any(
                value
                for candidate_id, value in potential_successes.items()
                if candidate_id != action_id
            )
        )
        if not succeeded:
            self._success = False
            self._recoveries += 1
        self._elapsed_s += elapsed_s
        event_id = f"sim-{self._seed}:step:{stage_index}"
        events = [RewardEvent("key_step_completed", event_id)]
        self._stage_index += 1
        terminated = self._stage_index >= len(self.config.stages)
        if terminated:
            events.append(
                RewardEvent(
                    "referee_task_success" if self._success else "attempt_failed",
                    f"sim-{self._seed}:episode-result",
                )
            )
        snapshot = self._terminal_snapshot() if terminated else self._snapshot()
        return SchedulingTransition(
            snapshot=snapshot,
            events=tuple(events),
            elapsed_s=elapsed_s,
            path_length_m=path_length_m,
            obstacle_cost=obstacle_cost,
            terminated=terminated,
            info={
                "success": bool(terminated and self._success),
                "recovery_count": self._recoveries,
                "safety_violation": False,
                "simulation_schema_version": self.config.schema_version,
                "stage_index": stage_index,
                "selected_action_id": action_id,
                "estimated_success_probability": float(
                    action.success_probability
                ),
                "true_success_probability": true_probability,
                "oracle_action_id": None if oracle is None else oracle.action_id,
                "oracle_miss": bool(oracle is not None and oracle.action_id != action_id),
                "avoidable_failure": avoidable_failure,
                "failure_reason": "" if succeeded else (
                    "avoidable_action_choice"
                    if avoidable_failure
                    else "unavoidable_potential_outcome"
                ),
            },
        )

    def _snapshot(self) -> SchedulingSnapshot:
        stage = self.config.stages[self._stage_index]
        sample = self._samples[self._stage_index]
        self._evaluations = self._build_evaluations(stage, sample)
        public_state = DomainRandomizer.apply_public_state(
            {
                "task_ordinal": stage.task_id,
                "attempt": 1,
                "step_progress": self._stage_index / len(self.config.stages),
                "remaining_time_s": max(
                    0.0, self.config.remaining_time_s - self._elapsed_s
                ),
                "robot_x_m": -0.70 + 0.12 * self._stage_index,
                "robot_y_m": 0.10 * (stage.task_id - 2),
                "robot_yaw_rad": 0.0,
                "payload_code": stage.payload_code,
                "referee_allowed": 1.0,
                "recovery_count": self._recoveries,
            },
            sample,
        )
        return SchedulingSnapshot(
            self._evaluations,
            public_state,
            episode_id=f"project-sim-{self._seed}",
        )

    def _terminal_snapshot(self) -> SchedulingSnapshot:
        hold = CandidateAction(
            "episode:terminal:hold",
            "hold",
            success_probability=1.0,
            hard_constraints={"referee_allowed": True},
        )
        self._evaluations = (evaluate_candidate(hold),)
        return SchedulingSnapshot(
            self._evaluations,
            {
                "task_ordinal": 3,
                "attempt": 1,
                "step_progress": 1.0,
                "remaining_time_s": max(
                    0.0, self.config.remaining_time_s - self._elapsed_s
                ),
                "payload_code": 0,
                "referee_allowed": 0.0,
                "recovery_count": self._recoveries,
            },
            episode_id=f"project-sim-{self._seed}",
        )

    def _build_evaluations(
        self,
        stage: SimulationStage,
        sample: DomainRandomizationSample,
    ) -> tuple[CandidateEvaluation, ...]:
        replan_count = self._replan_counts[self._stage_index]
        obstacle_side = 1 if sample.pose_dy_m >= 0.0 else -1
        evaluations: list[CandidateEvaluation] = []
        for slot, (side, lateral_m) in enumerate(
            (("center", 0.0), ("left", 0.08), ("right", -0.08))
        ):
            side_sign = 0 if lateral_m == 0.0 else (1 if lateral_m > 0.0 else -1)
            dynamic_blocked = bool(
                sample.dynamic_obstacle_present and side_sign == obstacle_side
            )
            planner_blocked = bool(
                sample.planner_failure and replan_count == 0 and slot == 0
            )
            path_length = (
                stage.base_path_length_m * sample.depth_scale
                + abs(lateral_m) * 1.8
            )
            if side_sign == -obstacle_side:
                path_length = max(0.2, path_length - 0.12)
            clearance = 0.34 - (0.14 if dynamic_blocked else 0.0)
            obstacle_integral = (
                0.75 if sample.dynamic_obstacle_present and side_sign == 0 else 0.12
            )
            metrics = PathMetrics(
                reachable=not planner_blocked,
                path=((0.0, 0.0), (path_length, lateral_m)),
                path_length_m=path_length,
                straight_distance_m=stage.base_path_length_m,
                detour_ratio=path_length / stage.base_path_length_m,
                min_clearance_m=clearance,
                inflation_cost_integral=obstacle_integral,
                heading_change_rad=abs(lateral_m) * 0.8,
                turn_count=int(side_sign != 0),
                dynamic_risk=0.8 if dynamic_blocked else 0.05,
                failure_reason="simulated planner failure" if planner_blocked else "",
            )
            success_probability = min(
                0.995,
                max(
                    0.55,
                    0.96
                    - abs(sample.detection_delta) * 0.35
                    - (
                        0.18
                        if sample.detection_dropout and replan_count == 0
                        else 0.0
                    )
                    - (0.12 if dynamic_blocked else 0.0),
                ),
            )
            candidate = CandidateAction(
                action_id=f"task{stage.task_id}:{stage.step_id}:sim:{side}",
                action_type="navigate",
                x=float(self._stage_index),
                y=lateral_m,
                yaw=sample.yaw_delta_rad,
                expected_score=3.0,
                success_probability=success_probability,
                expected_time_s=path_length / 0.20,
                perception_uncertainty=(
                    abs(sample.detection_delta)
                    + (0.4 if sample.detection_dropout else 0.0)
                ),
                manipulation_difficulty=(0.2 if stage.payload_code else 0.05),
                irreversible_risk=(0.15 if stage.step_id == "transport" else 0.02),
                recovery_cost=0.0,
                hard_constraints={
                    "referee_allowed": True,
                    "collision_free": not dynamic_blocked,
                    "resource_available": True,
                },
            )
            evaluations.append(
                evaluate_candidate(
                    candidate,
                    path_metrics=metrics,
                    min_clearance_m=self.config.minimum_clearance_m,
                    require_path=True,
                )
            )

        replan = CandidateAction(
            action_id=f"task{stage.task_id}:{stage.step_id}:sim:replan",
            action_type="replan",
            expected_score=0.0,
            success_probability=0.98,
            expected_time_s=1.5,
            perception_uncertainty=max(0.0, abs(sample.detection_delta) - 0.1),
            manipulation_difficulty=0.0,
            irreversible_risk=0.0,
            recovery_cost=0.35,
            hard_constraints={
                "referee_allowed": True,
                "resource_available": True,
                "step_allowed": replan_count < 2,
            },
        )
        evaluations.append(evaluate_candidate(replan))
        if not any(item.valid for item in evaluations):  # pragma: no cover
            raise RuntimeError("simulation produced no safe macro action")
        return tuple(evaluations)


def load_project_simulation_config(
    path: str | Path = DEFAULT_PROJECT_SIMULATION_CONFIG_PATH,
) -> ProjectSimulationConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("simulation config root must be an object")
    allowed = {
        "schema_version",
        "max_candidates",
        "minimum_clearance_m",
        "remaining_time_s",
        "stages",
        "domain_randomization",
        "outcome_model",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"unknown simulation config keys: {sorted(unknown)}")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        SIMULATION_SCHEMA_VERSION,
        IDENTIFIABLE_SIMULATION_SCHEMA_VERSION,
    }:
        raise ValueError("simulation config schema version mismatch")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise TypeError("simulation config stages must be an array")
    stages = []
    for raw in raw_stages:
        if not isinstance(raw, Mapping):
            raise TypeError("simulation stage must be an object")
        if set(raw) != {
            "task_id",
            "step_id",
            "payload_code",
            "base_path_length_m",
        }:
            raise ValueError("simulation stage fields do not match the v1 schema")
        stages.append(
            SimulationStage(
                task_id=int(raw["task_id"]),
                step_id=str(raw["step_id"]),
                payload_code=int(raw["payload_code"]),
                base_path_length_m=float(raw["base_path_length_m"]),
            )
        )
    randomization = payload.get("domain_randomization", {})
    if not isinstance(randomization, Mapping):
        raise TypeError("domain_randomization must be an object")
    valid_randomization_fields = set(
        DomainRandomizationConfig.__dataclass_fields__
    )
    unknown_randomization = set(randomization).difference(
        valid_randomization_fields
    )
    if unknown_randomization:
        raise ValueError(
            "unknown domain randomization keys: "
            f"{sorted(unknown_randomization)}"
        )
    return ProjectSimulationConfig(
        stages=tuple(stages),
        max_candidates=int(payload.get("max_candidates", 8)),
        minimum_clearance_m=float(payload.get("minimum_clearance_m", 0.22)),
        remaining_time_s=float(payload.get("remaining_time_s", 300.0)),
        randomization=DomainRandomizationConfig(**dict(randomization)),
        schema_version=str(schema_version),
        outcome_model=str(payload.get("outcome_model", LEGACY_OUTCOME_MODEL)),
    )


def build_project_sim_env() -> SchedulingEnv:
    """Stable CLI factory for training and paired blind benchmarks."""

    config_path = os.environ.get(
        "MATERIAL_SCHEDULER_SIM_CONFIG",
        str(DEFAULT_PROJECT_SIMULATION_CONFIG_PATH),
    )
    config = load_project_simulation_config(config_path)
    return SchedulingEnv(
        ProjectSchedulingSimulationBackend(config),
        max_candidates=config.max_candidates,
    )


__all__ = [
    "DEFAULT_SIMULATION_STAGES",
    "DEFAULT_PROJECT_SIMULATION_CONFIG_PATH",
    "CONTEXTUAL_OUTCOME_MODEL",
    "IDENTIFIABLE_SIMULATION_SCHEMA_VERSION",
    "LEGACY_OUTCOME_MODEL",
    "ProjectSchedulingSimulationBackend",
    "ProjectSimulationConfig",
    "SIMULATION_SCHEMA_VERSION",
    "SimulationStage",
    "build_project_sim_env",
    "load_project_simulation_config",
]
