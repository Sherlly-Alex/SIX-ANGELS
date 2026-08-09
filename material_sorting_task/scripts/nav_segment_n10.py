#!/usr/bin/env python3
"""Local-only N10 controller regression for the three navigation segments.

This is a controller test, not a competition Client acceptance run.  It uses
local random-layout fixtures to derive goals, then integrates the controller's
differential-drive command in a simple kinematic plant.  It proves controller
convergence and empty/carry footprint selection, not perception or grabbing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TASK_DIR = Path(__file__).resolve().parent.parent / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.known_scene import KnownSceneProvider
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import NavigationGoal, NavigationStatus, SpeedLimits
from navigation.occupancy_grid import build_material_scene_grid
from navigation.robot_geometry import FootprintMode

START_XY_YAW = (-0.70, 0.55, math.pi / 2.0)
DT_S = 0.05
MAX_TICKS_PER_SEGMENT = 2_400
SHELF_XY = (-2.63, 0.778)
SHELF_LAYERS = {1: 0.403, 2: 0.732, 3: 1.061}
TABLE_SIDE_X = {"left": -1.00, "right": -0.18}
BOX_HALF_Z = 0.095
BOX_SUPPORT_CLEARANCE = 0.010
PACKAGING_VERTICAL_HALF = 0.1170


def _limits() -> SpeedLimits:
    return SpeedLimits(0.3, 0.6, 0.5, 1.0, 0.15, 0.8)


@dataclass(frozen=True)
class SegmentResult:
    seed: int
    segment: str
    footprint_mode: str
    reached: bool
    ticks: int
    status: str
    final_x: float
    final_y: float
    final_yaw: float
    safety_footprint_mode: str


def _wrap(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _layout_for_seed(seed: int) -> dict:
    layout_path = TASK_DIR / "material_competition_layout.json"
    with open(layout_path, "r", encoding="utf-8") as handle:
        layout = json.load(handle)
    rng = random.Random(seed)
    boxes = [dict(item) for item in layout["movable_boxes"]]
    props = [dict(item) for item in layout["fixed_props"]]
    rng.shuffle(boxes)
    side = rng.choice(("left", "right"))
    shelf_box_layer = rng.choice(tuple(SHELF_LAYERS))
    packaging_layer = rng.choice(
        tuple(layer for layer in SHELF_LAYERS if layer != shelf_box_layer)
    )
    empty_layer = next(
        layer
        for layer in SHELF_LAYERS
        if layer not in (shelf_box_layer, packaging_layer)
    )
    boxes[0].update({
        "location": "table",
        "slot": "table_side",
        "side": side,
        "world_position": [
            TABLE_SIDE_X[side],
            2.20,
            float(layout["scene"]["table_top_z"]) + BOX_HALF_Z,
        ],
        "euler": [0.0, 0.0, 0.0],
    })
    boxes[1].update({
        "location": "table",
        "slot": "table_top",
        "world_position": [-0.54, 2.30, 1.004],
        "euler": [0.0, 0.0, math.pi / 2.0],
    })
    boxes[2].update({
        "location": "shelf",
        "slot": "shelf",
        "shelf_layer": shelf_box_layer,
        "world_position": [
            SHELF_XY[0],
            SHELF_XY[1],
            SHELF_LAYERS[shelf_box_layer] + BOX_HALF_Z + BOX_SUPPORT_CLEARANCE,
        ],
        "euler": [0.0, 0.0, math.pi / 2.0],
    })
    for prop in props:
        if prop.get("prop") == "packaging_box":
            prop.update({
                "location": "shelf",
                "shelf_layer": packaging_layer,
                "world_position": [
                    SHELF_XY[0],
                    SHELF_XY[1],
                    SHELF_LAYERS[packaging_layer]
                    + PACKAGING_VERTICAL_HALF
                    + BOX_SUPPORT_CLEARANCE,
                ],
                "euler": [math.pi / 2.0, 0.0, 0.0],
            })
    meta = {
        "table_side": side,
        "shelf_box_layer": shelf_box_layer,
        "packaging_box_layer": packaging_layer,
        "empty_shelf_layer": empty_layer,
        "seed": seed,
    }
    return {"movable_boxes": boxes, "fixed_props": props,
            "scene": layout["scene"], "random_meta": meta}


def _task1_instruction(task_layout: dict) -> dict:
    target = next(
        item for item in task_layout["movable_boxes"]
        if item.get("slot") == "table_side"
    )
    empty_layer = int(task_layout["random_meta"]["empty_shelf_layer"])
    return {
        "task": 1,
        "target_body": target["body"],
        "target_color": target["color"],
        "place_type": "shelf_point",
        "place_world": [
            -2.68,
            SHELF_XY[1],
            SHELF_LAYERS[empty_layer] + BOX_HALF_Z,
        ],
    }


def _run_segment(*, seed: int, grid, goal: NavigationGoal,
                 footprint_mode: FootprintMode,
                 pose: tuple[float, float, float]) -> tuple[SegmentResult, tuple[float, float, float]]:
    """Run one goal against the deterministic local kinematic plant."""
    controller = NavigationController(grid, _limits())
    controller.set_footprint_mode(footprint_mode)
    x, y, yaw = pose
    if not controller.set_goal(goal, x, y):
        return SegmentResult(seed, goal.segment.value, footprint_mode.value, False, 0,
                             controller.status.value, x, y, yaw,
                             controller.safety_footprint_mode.value), pose
    for tick in range(1, MAX_TICKS_PER_SEGMENT + 1):
        command = controller.update(x, y, yaw, DT_S)
        yaw = _wrap(yaw + command.angular_z * DT_S)
        x += command.linear_x * DT_S * math.cos(yaw)
        y += command.linear_x * DT_S * math.sin(yaw)
        if controller.status in (NavigationStatus.GOAL_REACHED, NavigationStatus.FAILED):
            break
    result = SegmentResult(seed, goal.segment.value, footprint_mode.value,
                           controller.status == NavigationStatus.GOAL_REACHED, tick,
                           controller.status.value, x, y, yaw,
                           controller.safety_footprint_mode.value)
    return result, (x, y, yaw)


def run_seed(seed: int) -> tuple[SegmentResult, ...]:
    """Run table -> shelf(carry) -> end for one local random-layout seed."""
    task_layout = _layout_for_seed(seed)
    provider = KnownSceneProvider(task_layout=task_layout)
    instruction = _task1_instruction(task_layout)
    goals_and_modes = (
        (provider.pick_goal(instruction), FootprintMode.TRANSIT_STOWED),
        (provider.place_goal(instruction), FootprintMode.TRANSIT_CARRY),
        (provider.end_goal(), FootprintMode.TRANSIT_STOWED),
    )
    grid = build_material_scene_grid(scene=provider.scene)
    pose = START_XY_YAW
    results: list[SegmentResult] = []
    for goal, mode in goals_and_modes:
        result, pose = _run_segment(seed=seed, grid=grid, goal=goal,
                                    footprint_mode=mode, pose=pose)
        results.append(result)
        if not result.reached:
            break
    return tuple(results)


def run_n10(seeds: Iterable[int] = range(1, 11)) -> tuple[SegmentResult, ...]:
    results: list[SegmentResult] = []
    for seed in seeds:
        results.extend(run_seed(int(seed)))
    return tuple(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be > 0")
    results = run_n10(range(1, args.count + 1))
    failures = [item for item in results if not item.reached]
    for item in results:
        print("NAV_SEGMENT"
              f" seed={item.seed} segment={item.segment} mode={item.footprint_mode}"
              f" safety_mode={item.safety_footprint_mode} status={item.status} ticks={item.ticks}")
    if failures:
        print(f"NAV_SEGMENT_N{args.count}_FAIL failures={len(failures)}")
        return 1
    print(f"NAV_SEGMENT_N{args.count}_PASS segments={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
