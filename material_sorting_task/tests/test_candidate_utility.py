"""Candidate generation and deterministic multi-critic policy tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.costmap import PathMetrics, WorldCostmap
from navigation.navigation_types import NavigationGoal, NavigationSegment
from navigation.occupancy_grid import LayeredGrid, OccupancyGrid
from navigation.robot_geometry import FootprintMode
from scheduler.candidate_generator import CandidateAction, CandidateGenerator
from scheduler.policies.heuristic import HeuristicPolicy
from scheduler.utility import UtilityWeights, evaluate_candidate, rank_candidates


def _metrics(**overrides) -> PathMetrics:
    values = dict(
        reachable=True,
        path=((0.0, 0.0), (1.0, 0.0)),
        path_length_m=1.0,
        straight_distance_m=1.0,
        detour_ratio=1.0,
        min_clearance_m=0.20,
        inflation_cost_integral=1.0,
        heading_change_rad=0.0,
        turn_count=0,
        dynamic_risk=0.0,
        carried_envelope_safe=True,
    )
    values.update(overrides)
    return PathMetrics(**values)


def _goal() -> NavigationGoal:
    return NavigationGoal(
        x=1.0,
        y=2.0,
        yaw=math.pi / 2.0,
        position_tolerance=0.06,
        yaw_tolerance=0.03,
        safety_radius=0.65,
        segment=NavigationSegment.NAV_TABLE,
        source_tag="test",
    )


def _empty_world() -> WorldCostmap:
    return WorldCostmap(LayeredGrid(
        chassis=OccupancyGrid(-2.0, -2.0, 0.05, 80, 80),
        arm=OccupancyGrid(-2.0, -2.0, 0.05, 80, 80),
    ))


def test_generator_creates_deterministic_body_frame_offsets():
    candidates = CandidateGenerator().generate(_goal(), task_id=2, step_id="observe")
    assert [item.action_id for item in candidates] == [
        "task2:observe:stand:center",
        "task2:observe:stand:left",
        "task2:observe:stand:right",
    ]
    # At yaw=pi/2, goal-left points toward world -X.
    assert candidates[0].goal_pose == pytest.approx((1.0, 2.0, math.pi / 2.0))
    assert candidates[1].x == pytest.approx(0.92)
    assert candidates[2].x == pytest.approx(1.08)
    assert candidates[1].metadata["lateral_offset_m"] == pytest.approx(0.08)


def test_false_hard_constraint_is_never_a_soft_penalty():
    candidate = CandidateAction(
        "unsafe",
        x=1.0,
        y=0.0,
        yaw=0.0,
        expected_score=1000.0,
        hard_constraints={"collision_free": False},
    )
    result = evaluate_candidate(candidate, path_metrics=_metrics())
    assert not result.valid
    assert result.utility == float("-inf")
    assert "collision_free" in result.rejection_reasons[0]


@pytest.mark.parametrize(
    "field",
    [
        "expected_score",
        "success_probability",
        "expected_time_s",
        "perception_uncertainty",
        "manipulation_difficulty",
        "irreversible_risk",
        "recovery_cost",
    ],
)
def test_non_finite_candidate_features_are_masked(field):
    kwargs = {field: float("nan")}
    candidate = CandidateAction("nan", x=1.0, y=0.0, yaw=0.0, **kwargs)
    result = evaluate_candidate(candidate, path_metrics=_metrics())
    assert not result.valid
    assert result.utility == float("-inf")


def test_non_finite_path_metric_is_masked():
    candidate = CandidateAction("bad_path", x=1.0, y=0.0, yaw=0.0)
    result = evaluate_candidate(
        candidate,
        path_metrics=_metrics(dynamic_risk=float("inf")),
    )
    assert not result.valid
    assert result.utility == float("-inf")


def test_multi_critic_ranking_is_stable_and_auditable():
    safer = CandidateAction(
        "b_safer",
        x=1.0,
        y=0.0,
        yaw=0.0,
        expected_score=10.0,
        success_probability=0.95,
        expected_time_s=2.0,
    )
    risky = CandidateAction(
        "a_risky",
        x=1.0,
        y=0.0,
        yaw=0.0,
        expected_score=10.0,
        success_probability=0.50,
        expected_time_s=2.0,
    )
    ranked = rank_candidates(
        (risky, safer),
        path_metrics_by_id={risky.action_id: _metrics(), safer.action_id: _metrics()},
    )
    assert ranked[0].action_id == "b_safer"
    assert sum(ranked[0].critic_scores.values()) == pytest.approx(ranked[0].utility)
    assert tuple(ranked[0].critic_scores) == (
        "expected_reward",
        "success_probability",
        "expected_time",
        "path_length",
        "obstacle_cost",
        "dynamic_risk",
        "heading_change",
        "perception_uncertainty",
        "manipulation_difficulty",
        "irreversible_risk",
        "recovery_cost",
    )


def test_equal_utilities_tie_break_by_action_id():
    first = CandidateAction("a", action_type="rescan")
    second = CandidateAction("b", action_type="rescan")
    assert [item.action_id for item in rank_candidates((second, first))] == ["a", "b"]


def test_configured_weights_change_only_soft_ranking():
    candidate = CandidateAction(
        "weighted",
        action_type="rescan",
        expected_score=3.0,
        success_probability=0.5,
        expected_time_s=2.0,
    )
    weights = UtilityWeights(
        expected_reward=2.0,
        success_probability=4.0,
        expected_time=1.0,
        path_length=0.0,
        obstacle_cost=0.0,
        dynamic_risk=0.0,
        heading_change=0.0,
        perception_uncertainty=0.0,
        manipulation_difficulty=0.0,
        irreversible_risk=0.0,
        recovery_cost=0.0,
    )
    result = evaluate_candidate(candidate, weights=weights)
    assert result.valid
    assert result.utility == pytest.approx(6.0 + 2.0 - 2.0)


def test_minimal_world_generator_policy_integration_is_pure():
    world = _empty_world()
    candidates = CandidateGenerator(lateral_offsets_m=(0.0, 0.08)).generate(
        (0.8, 0.0, 0.0),
        task_id=1,
        step_id="navigate_pick",
        success_probability=0.9,
    )
    ranked = HeuristicPolicy(min_clearance_m=0.0).rank(
        candidates,
        costmap=world,
        start_pose=(-0.8, 0.0, 0.0),
        footprint_mode=FootprintMode.CHASSIS,
    )
    assert ranked
    assert ranked[0].valid, ranked[0].rejection_reasons
    assert ranked[0].path_metrics is not None
    assert ranked[0].path_metrics.reachable
    # Policy outputs a decision record, never actuator commands.
    assert not hasattr(ranked[0], "linear_x")
    assert not hasattr(ranked[0], "arm_command")
