"""WorldCostmap TTL, snapshot and metric tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.costmap import AABB, DynamicObstacle, WorldCostmap
from navigation.occupancy_grid import LayeredGrid, OccupancyGrid
from navigation.robot_geometry import FootprintMode


def _empty_layers() -> LayeredGrid:
    chassis = OccupancyGrid(-2.0, -2.0, 0.05, 80, 80)
    arm = OccupancyGrid(-2.0, -2.0, 0.05, 80, 80)
    return LayeredGrid(chassis=chassis, arm=arm)


def _obstacle(
    obstacle_id: str,
    x: float,
    confidence: float,
    expires: float = 2.0,
) -> DynamicObstacle:
    return DynamicObstacle(
        bounds=AABB(x, x + 0.20, -0.10, 0.10, 0.0, 0.20),
        confidence=confidence,
        observed_at_s=0.0,
        expires_at_s=expires,
        source="test_detector",
        obstacle_id=obstacle_id,
    )


def test_snapshot_filters_lethal_confidence_but_preserves_evidence():
    world = WorldCostmap(_empty_layers(), lethal_confidence=0.60)
    world.set_dynamic([_obstacle("high", 0.30, 0.9), _obstacle("low", -0.40, 0.4)])
    snapshot = world.snapshot(now_s=1.0)

    assert tuple(item.obstacle_id for item in snapshot.active_obstacles) == ("high", "low")
    assert tuple(item.obstacle_id for item in snapshot.lethal_obstacles) == ("high",)
    high = snapshot.layered_grid.world_to_grid(0.35, 0.0)
    low = snapshot.layered_grid.world_to_grid(-0.35, 0.0)
    assert snapshot.planning_grid().is_occupied(*high)
    assert snapshot.planning_grid().is_free(*low)


def test_ttl_expiry_does_not_mutate_an_older_snapshot():
    world = WorldCostmap(_empty_layers())
    world.set_dynamic([_obstacle("short", 0.20, 1.0, expires=1.0)])
    old = world.snapshot(now_s=0.5)
    cell = old.layered_grid.world_to_grid(0.25, 0.0)
    assert old.planning_grid().is_occupied(*cell)

    expired = world.snapshot(now_s=1.0)
    assert expired.active_obstacles == ()
    assert expired.planning_grid().is_free(*cell)
    assert old.planning_grid().is_occupied(*cell)


def test_snapshot_and_obstacle_models_are_frozen():
    world = WorldCostmap(_empty_layers())
    obstacle = _obstacle("frozen", 0.2, 1.0)
    world.set_dynamic([obstacle])
    snapshot = world.snapshot(now_s=0.5)
    with pytest.raises(FrozenInstanceError):
        snapshot.version = 99
    with pytest.raises(FrozenInstanceError):
        obstacle.confidence = 0.0


def test_detection_adapter_keeps_score_and_ttl():
    world = WorldCostmap(_empty_layers(), lethal_confidence=0.5)
    count = world.observe_detections(
        [("pink", (0.30, 0.0, 0.10), 0.75)],
        observed_at_s=10.0,
        ttl_s=0.5,
    )
    assert count == 1
    live = world.snapshot(now_s=10.25)
    assert live.active_obstacles[0].confidence == pytest.approx(0.75)
    assert live.active_obstacles[0].expires_at_s == pytest.approx(10.5)
    assert world.snapshot(now_s=10.5).active_obstacles == ()


def test_path_metrics_are_finite_and_complete_on_empty_map():
    snapshot = WorldCostmap(_empty_layers()).snapshot(now_s=0.0)
    metrics = snapshot.plan_path(
        (-0.8, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        footprint_mode=FootprintMode.CHASSIS,
        min_clearance=0.22,
    )
    assert metrics.reachable, metrics.failure_reason
    assert metrics.finite()
    assert metrics.path
    assert metrics.path_length_m == pytest.approx(1.6, abs=0.08)
    assert metrics.straight_distance_m == pytest.approx(1.6, abs=0.08)
    assert metrics.detour_ratio >= 1.0
    assert metrics.min_clearance_m > 0.0
    assert metrics.inflation_cost_integral > 0.0


def test_non_finite_goal_is_an_unreachable_metric_not_an_exception():
    snapshot = WorldCostmap(_empty_layers()).snapshot(now_s=0.0)
    metrics = snapshot.plan_path((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0))
    assert not metrics.reachable
    assert metrics.failure_reason


def test_replace_dynamic_source_preserves_other_producers():
    world = WorldCostmap(_empty_layers())
    lidar = DynamicObstacle(
        bounds=AABB(0.1, 0.2, 0.1, 0.2),
        confidence=1.0,
        observed_at_s=0.0,
        expires_at_s=5.0,
        source="lidar",
        obstacle_id="lidar-1",
    )
    vision_old = DynamicObstacle(
        bounds=AABB(0.3, 0.4, 0.1, 0.2),
        confidence=1.0,
        observed_at_s=0.0,
        expires_at_s=5.0,
        source="vision",
        obstacle_id="old",
    )
    vision_new = DynamicObstacle(
        bounds=AABB(0.5, 0.6, 0.1, 0.2),
        confidence=1.0,
        observed_at_s=1.0,
        expires_at_s=5.0,
        source="vision",
        obstacle_id="new",
    )
    world.set_dynamic((lidar, vision_old))

    world.replace_dynamic_source((vision_new,), source="vision", observed_at_s=1.0)

    snapshot = world.snapshot(now_s=2.0)
    assert {(item.source, item.obstacle_id) for item in snapshot.active_obstacles} == {
        ("lidar", "lidar-1"),
        ("vision", "new"),
    }
