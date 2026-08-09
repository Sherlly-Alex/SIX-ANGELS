"""Regression tests for the v3 navigation handoff integration."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from executors.base import ExecutionContext, TargetObservation
from executors.task1 import Task1NavigationExecutor
from executors.transfer_support import (
    TransferMotion,
    navigation_overlay_from_context,
)
from navigation.local_goal_selector import select_local_goal
from navigation.occupancy_grid import LayeredGrid
from navigation.robot_geometry import FootprintMode


def _context(
    now_s: float,
    observations: dict[str, TargetObservation],
    *,
    target_color: str = "brown",
) -> ExecutionContext:
    return ExecutionContext(
        now_s=now_s,
        instruction={"task": 1, "target_color": target_color},
        task_index=0,
        attempt=1,
        target_observations=observations,
    )


def _observation(
    color: str,
    xyz: tuple[float, float, float],
    received_at_s: float,
) -> TargetObservation:
    return TargetObservation(
        color=color,
        position_world=xyz,
        received_at_s=received_at_s,
        score=0.9,
    )


def _odometry(x: float, y: float, yaw: float = 0.0):
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            )
        )
    )


def test_formal_client_uses_perception_without_private_layout_topic() -> None:
    root = Path(__file__).resolve().parents[1]
    client_source = (
        root / "examples/material_sorting/client_task.py"
    ).read_text(encoding="utf-8")
    transfer_source = (
        root / "examples/material_sorting/executors/transfer_support.py"
    ).read_text(encoding="utf-8")
    assert '"/material/detections"' in client_source
    assert "/material/task_layout" not in client_source
    assert "build_nav_overlay" in transfer_source
    assert "target_observations" in transfer_source


def test_overlay_excludes_target_and_drops_stale_observations() -> None:
    context = _context(
        10.0,
        {
            "brown": _observation("brown", (-1.0, 2.2, 0.84), 10.0),
            "packaging_box": _observation(
                "packaging_box", (-2.63, 0.78, 1.18), 9.5
            ),
            "pink": _observation("pink", (-2.63, 0.78, 0.50), 7.0),
        },
    )
    volumes = navigation_overlay_from_context(
        context, (-0.70, 0.55), exclude_target=True
    )
    assert len(volumes) == 1
    volume = volumes[0]
    assert volume.x_min < -2.63 < volume.x_max
    assert volume.y_min < 0.78 < volume.y_max


def test_task_and_transfer_navigation_use_layered_grids() -> None:
    task1 = Task1NavigationExecutor()
    transfer = TransferMotion()
    assert isinstance(task1._nav_grid, LayeredGrid)
    assert isinstance(transfer._nav_grid, LayeredGrid)


def test_transfer_context_selects_carry_safety_envelope() -> None:
    transfer = TransferMotion()
    context = _context(
        1.0,
        {
            "packaging_box": _observation(
                "packaging_box", (-2.63, 0.78, 1.18), 1.0
            )
        },
    )
    transfer._refresh_navigation_context(
        _odometry(-0.70, 0.55),
        context,
        footprint_mode=FootprintMode.TRANSIT_CARRY,
        exclude_target=True,
    )
    assert transfer._navigation.footprint_mode is FootprintMode.TRANSIT_CARRY
    assert transfer._navigation.safety_footprint_mode is FootprintMode.TRANSIT_CARRY
    assert len(transfer._nav_grid.dynamic_volumes) == 1


def test_closest_index_semantics_and_controller_projection_are_both_supported() -> None:
    path = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    explicit = select_local_goal(
        10.0, 10.0, path, closest_index=1, lookahead_distance=0.5
    )
    projected = select_local_goal(
        1.4, 0.0, path, closest_index=1, lookahead_distance=0.5,
        project_from_pose=True,
    )
    assert explicit[0] == 1.5
    assert math.isclose(projected[0], 1.9)
