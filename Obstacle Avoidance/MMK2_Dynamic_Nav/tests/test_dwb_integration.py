"""Integration tests for DWB main-loop integration."""

import sys
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from config.dynamic_nav_config import DynamicNavConfig
from planning.dwb.trajectory import RobotState
from planning.dwb.trajectory_generator import StandardTrajectoryGenerator
from planning.dwb.controller import DWBController
from planning.dwb.critics.obstacle_critic import ObstacleCritic
from planning.dwb.critics.path_dist_critic import PathDistCritic
from planning.dwb.critics.goal_dist_critic import GoalDistCritic
from planning.dwb.critics.prefer_forward_critic import PreferForwardCritic
from planning.dwb.critics.oscillation_critic import OscillationCritic
from planning.local_goal_selector import select_local_goal


class _MockGrid:
    def __init__(self):
        self.width = 400
        self.height = 300
        self.resolution = 0.05
        self.origin = (-8.0, -6.0)
        self._blocked = np.zeros((self.height, self.width), dtype=bool)

    def world_to_map(self, x, y):
        mx = int(np.floor((x - self.origin[0]) / self.resolution))
        my = int(np.floor((y - self.origin[1]) / self.resolution))
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return (-1, -1)
        return (mx, my)

    def get_inflation_cost(self, mx, my, inflation_radius,
                           cost_weight=3.0, min_clearance=0.0):
        if mx < 0 or my < 0:
            return float("inf")
        if mx >= self.width or my >= self.height:
            return float("inf")
        if self._blocked[my, mx]:
            return float("inf")
        return 1.0


def _build_dwb():
    cfg = DynamicNavConfig()
    gen = StandardTrajectoryGenerator(cfg)
    critics = [
        ObstacleCritic(weight=cfg.dwb_obstacle_weight),
        PathDistCritic(weight=cfg.dwb_path_dist_weight),
        GoalDistCritic(weight=cfg.dwb_goal_dist_weight),
        PreferForwardCritic(
            weight=cfg.dwb_prefer_forward_weight,
            max_speed=cfg.dwb_max_speed,
            min_progress_speed=cfg.dwb_min_progress_speed,
            spin_yaw_threshold=cfg.dwb_spin_yaw_threshold,
            spin_penalty=cfg.dwb_spin_penalty,
        ),
        OscillationCritic(weight=cfg.dwb_oscillation_weight),
    ]
    ctrl = DWBController(gen, critics)
    return cfg, ctrl


def test_dwb_components_constructable():
    """DWB generator, controller and critics can be constructed."""
    cfg, ctrl = _build_dwb()
    assert ctrl is not None
    assert len(ctrl.critics) == 5
    print("[PASS] test_dwb_components_constructable")


def test_local_goal_selector_output():
    """LocalGoalSelector returns a 2D point on the lookahead."""
    path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    goal = select_local_goal((0, 0, 0), path, 0, lookahead_distance=1.2)
    assert len(goal) == 2
    assert goal[0] > 0.5
    print("[PASS] test_local_goal_selector_output")


def test_dwb_controller_computes_result_with_context():
    """Full DWB pipeline produces a valid result using mock grid + path."""
    cfg, ctrl = _build_dwb()
    grid = _MockGrid()
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    local_goal = select_local_goal((0, 0, 0), path, 0,
                                   lookahead_distance=cfg.dwb_local_goal_distance)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    ctx = {
        "planning_grid": grid,
        "global_path": path,
        "local_goal": local_goal,
        "robot_radius": cfg.dwb_robot_radius,
        "safety_margin": cfg.dwb_safety_margin,
    }
    result = ctrl.compute_best(state, ctx)
    assert result.success
    assert result.linear_vel >= cfg.dwb_min_speed
    assert result.linear_vel <= cfg.dwb_max_speed
    assert abs(result.angular_vel) <= cfg.dwb_max_yaw_rate
    print("[PASS] test_dwb_controller_computes_result_with_context")


def test_dynamic_nav_config_has_dwb_params():
    """Config must carry DWB integration parameters."""
    cfg = DynamicNavConfig()
    assert hasattr(cfg, "dwb_failure_confirm_count")
    assert cfg.dwb_failure_confirm_count >= 1
    assert hasattr(cfg, "dwb_replan_timeout")
    assert cfg.dwb_replan_timeout > 0
    assert hasattr(cfg, "dwb_min_speed")
    assert hasattr(cfg, "dwb_max_speed")
    print("[PASS] test_dynamic_nav_config_has_dwb_params")


def test_path_validator_index_selection_logic():
    """DWB mode uses dwb_waypoint_idx; waypoint uses motion_ctrl idx."""
    # Simulate the branch in run_dynamic_nav.py
    controller_mode = "dwb"
    dwb_waypoint_idx = 3
    motion_ctrl_idx = 0

    if controller_mode == "dwb":
        wp_idx = dwb_waypoint_idx
    else:
        wp_idx = motion_ctrl_idx
    assert wp_idx == 3, f"DWB should use dwb idx=3, got {wp_idx}"

    controller_mode = "waypoint"
    motion_ctrl_idx = 5
    if controller_mode == "dwb":
        wp_idx = dwb_waypoint_idx
    else:
        wp_idx = motion_ctrl_idx
    assert wp_idx == 5, f"Waypoint should use motion_ctrl idx=5, got {wp_idx}"
    print("[PASS] test_path_validator_index_selection_logic")


if __name__ == "__main__":
    test_dwb_components_constructable()
    test_local_goal_selector_output()
    test_dwb_controller_computes_result_with_context()
    test_dynamic_nav_config_has_dwb_params()
    test_path_validator_index_selection_logic()
    print("[ALL PASS] DWB integration tests")
