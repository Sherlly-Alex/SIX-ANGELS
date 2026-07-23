"""Unit tests for DWB base critics (Obstacle, PathDist, GoalDist, PreferForward)."""

import sys
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from config.dynamic_nav_config import DynamicNavConfig
from planning.dwb.trajectory import RobotState, Trajectory
from planning.dwb.trajectory_generator import StandardTrajectoryGenerator
from planning.dwb.controller import DWBController
from planning.dwb.critics.obstacle_critic import ObstacleCritic
from planning.dwb.critics.path_dist_critic import PathDistCritic
from planning.dwb.critics.goal_dist_critic import GoalDistCritic
from planning.dwb.critics.prefer_forward_critic import PreferForwardCritic
from planning.dwb.critics.oscillation_critic import OscillationCritic


# ------------------------------------------------------------------ #
# Lightweight mock PlanningGrid for unit tests
# ------------------------------------------------------------------ #

class _MockGrid:
    """Supports world_to_map, map_to_world, get_inflation_cost, is_occupied.

    400×300 cells at 0.05 m resolution → 20 m × 15 m, origin (-8, -6).
    Blocked cell at map (180, 120) → world ~(1.0, 0.0).
    """

    def __init__(self):
        self.width = 400
        self.height = 300
        self.resolution = 0.05
        self.origin = (-8.0, -6.0)
        blocked = np.zeros((self.height, self.width), dtype=bool)
        blocked[120, 180] = True
        self._blocked = blocked

    def world_to_map(self, x, y):
        mx = int(np.floor((x - self.origin[0]) / self.resolution))
        my = int(np.floor((y - self.origin[1]) / self.resolution))
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return (-1, -1)
        return (mx, my)

    def map_to_world(self, mx, my):
        cx = self.origin[0] + (mx + 0.5) * self.resolution
        cy = self.origin[1] + (my + 0.5) * self.resolution
        return (cx, cy)

    def get_inflation_cost(self, mx, my, inflation_radius,
                           cost_weight=3.0, min_clearance=0.0):
        if mx < 0 or my < 0:
            return float("inf")
        if mx >= self.width or my >= self.height:
            return float("inf")
        if self._blocked[my, mx]:
            return float("inf")
        dist_cells = float(np.hypot(mx - 180, my - 120))
        if dist_cells <= inflation_radius:
            t = dist_cells / max(inflation_radius, 1)
            return 1.0 + cost_weight * (1.0 - t) ** 2
        return 1.0


# ------------------------------------------------------------------ #

def _make_config():
    return DynamicNavConfig()


def _make_state(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0):
    return RobotState(x=x, y=y, yaw=yaw, linear_vel=v, angular_vel=w)


def _make_trajectory(poses, linear_vel=0.15, angular_vel=0.0):
    return Trajectory(
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        poses=np.array(poses, dtype=float),
        total_score=0.0,
        critic_scores={},
        valid=True,
    )


def _collision_context():
    grid = _MockGrid()
    return {
        "planning_grid": grid,
        "global_path": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "local_goal": (2.0, 0.0),
        "robot_radius": 0.22,
        "safety_margin": 0.15,
    }


def _clear_context():
    grid = _MockGrid()
    return {
        "planning_grid": grid,
        "global_path": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "local_goal": (2.0, 0.0),
        "robot_radius": 0.22,
        "safety_margin": 0.15,
    }


# ================================================================== #
# ObstacleCritic tests
# ================================================================== #

def test_obstacle_critic_rejects_collision():
    """Trajectory entering an occupied cell must get inf."""
    critic = ObstacleCritic()
    grid = _MockGrid()
    # Blocked cell at map (180, 120) → world ~(1.025, 0.025)
    wx, wy = grid.map_to_world(180, 120)
    poses = np.array([
        [wx, wy, 0.0],
        [wx, wy, 0.0],
        [wx, wy, 0.0],
    ])
    traj = _make_trajectory(poses)
    ctx = {"planning_grid": grid, "robot_radius": 0.22, "safety_margin": 0.15}
    score = critic.score(traj, ctx)
    assert not np.isfinite(score), f"Collision should give inf, got {score}"
    print("[PASS] test_obstacle_critic_rejects_collision")


def test_obstacle_critic_accepts_clear_path():
    """Trajectory through clear cells must return finite cost."""
    critic = ObstacleCritic()
    grid = _MockGrid()
    poses = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ])
    traj = _make_trajectory(poses)
    ctx = {"planning_grid": grid, "robot_radius": 0.22, "safety_margin": 0.15}
    score = critic.score(traj, ctx)
    assert np.isfinite(score), f"Clear path should be finite, got {score}"
    print("[PASS] test_obstacle_critic_accepts_clear_path")


def test_obstacle_critic_soft_cost_near_obstacle():
    """Soft cost must be positive when trajectory is near (not at) obstacle."""
    critic = ObstacleCritic()
    grid = _MockGrid()
    # Poses near but not at blocked cell (180,120) — world ~ (1.025, 0.025)
    # Map (176, 120): distance 4 cells from blocked cell, within inflation radius (8)
    wx, wy = grid.map_to_world(176, 120)
    poses = np.array([
        [wx, wy, 0.0],
        [wx, wy, 0.0],
    ])
    traj = _make_trajectory(poses)
    ctx = {"planning_grid": grid, "robot_radius": 0.22, "safety_margin": 0.15}
    score = critic.score(traj, ctx)
    assert np.isfinite(score), f"Near-obstacle should be finite, got {score}"
    assert score > 0.0, f"Soft cost should be >0 near obstacle, got {score}"
    print("[PASS] test_obstacle_critic_soft_cost_near_obstacle")


def test_obstacle_critic_closer_is_costlier():
    """Trajectory closer to obstacle must have higher soft cost."""
    critic = ObstacleCritic()
    grid = _MockGrid()
    ctx = {"planning_grid": grid, "robot_radius": 0.22, "safety_margin": 0.15}

    # Far: 8 cells away (at inflation boundary, cost ~1.0 → soft ~0)
    wx_far, wy_far = grid.map_to_world(172, 120)
    far_traj = _make_trajectory(np.array([[wx_far, wy_far, 0.0]]))
    score_far = critic.score(far_traj, ctx)

    # Near: 4 cells away
    wx_near, wy_near = grid.map_to_world(176, 120)
    near_traj = _make_trajectory(np.array([[wx_near, wy_near, 0.0]]))
    score_near = critic.score(near_traj, ctx)

    assert score_near > score_far, \
        f"Near {score_near:.4f} should be > far {score_far:.4f}"
    assert np.isfinite(score_far) and np.isfinite(score_near)
    print("[PASS] test_obstacle_critic_closer_is_costlier")


# ================================================================== #
# PathDistCritic tests
# ================================================================== #

def test_path_dist_critic_penalizes_deviation():
    """Trajectory further from global path must have higher cost."""
    critic = PathDistCritic()
    ctx = {"global_path": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])}

    close_poses = np.array([[0.1, 0.05, 0.0], [0.5, 0.05, 0.0], [1.0, 0.05, 0.0]])
    far_poses = np.array([[0.1, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]])

    score_close = critic.score(_make_trajectory(close_poses), ctx)
    score_far = critic.score(_make_trajectory(far_poses), ctx)
    assert score_close < score_far, \
        f"Close {score_close} should be < far {score_far}"
    print("[PASS] test_path_dist_critic_penalizes_deviation")


def test_path_dist_critic_empty_path():
    """Empty global path must return 0."""
    critic = PathDistCritic()
    ctx = {"global_path": None}
    traj = _make_trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert critic.score(traj, ctx) == 0.0
    print("[PASS] test_path_dist_critic_empty_path")


# ================================================================== #
# GoalDistCritic tests
# ================================================================== #

def test_goal_dist_critic_prefers_closer_endpoint():
    """Trajectory ending closer to local_goal must have lower cost."""
    critic = GoalDistCritic()
    ctx = {"local_goal": (2.0, 0.0)}

    close_end = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    far_end = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    score_close = critic.score(_make_trajectory(close_end), ctx)
    score_far = critic.score(_make_trajectory(far_end), ctx)
    assert score_close < score_far, \
        f"Close {score_close} should be < far {score_far}"
    print("[PASS] test_goal_dist_critic_prefers_closer_endpoint")


# ================================================================== #
# PreferForwardCritic tests
# ================================================================== #

def test_prefer_forward_prefers_higher_speed():
    """Higher linear velocity must yield lower cost."""
    critic = PreferForwardCritic(max_speed=0.3)
    ctx = {}

    score_low = critic.score(_make_trajectory([[0, 0, 0]], linear_vel=0.05), ctx)
    score_high = critic.score(_make_trajectory([[0, 0, 0]], linear_vel=0.25), ctx)
    assert score_high < score_low, \
        f"High speed {score_high} should be < low speed {score_low}"
    print("[PASS] test_prefer_forward_prefers_higher_speed")


def test_prefer_forward_penalizes_spin_in_place():
    """Low-v high-|w| must cost more than progressive forward motion."""
    critic = PreferForwardCritic(max_speed=0.3, min_progress_speed=0.06,
                                 spin_yaw_threshold=0.25, spin_penalty=2.0)
    spin = critic.score(_make_trajectory([[0, 0, 0]], linear_vel=0.0,
                                         angular_vel=0.5), {})
    go = critic.score(_make_trajectory([[0, 0, 0]], linear_vel=0.2,
                                       angular_vel=0.1), {})
    assert spin > go, f"Spin {spin} should cost more than forward {go}"
    print("[PASS] test_prefer_forward_penalizes_spin_in_place")


def test_oscillation_critic_penalizes_sign_flip():
    critic = OscillationCritic(weight=1.5, deadband=0.05)
    ctx = {"last_angular_vel": 0.4}
    flip = critic.score(_make_trajectory([[0, 0, 0]], angular_vel=-0.4), ctx)
    same = critic.score(_make_trajectory([[0, 0, 0]], angular_vel=0.3), ctx)
    assert flip > same, f"Flip {flip} should cost more than same-sign {same}"
    print("[PASS] test_oscillation_critic_penalizes_sign_flip")


# ================================================================== #
# Weights + prepare tests
# ================================================================== #

def test_critics_prepare_and_score_with_weights():
    """Critic weight should be applied by controller (not internally).

    We verify each critic accepts weight constructor arg and returns correct
    polarity.  The controller applies weights, tested separately in TASK-013.
    """
    ctx = _clear_context()
    poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    obs = ObstacleCritic(weight=3.0)
    s = obs.score(_make_trajectory(poses), ctx)
    assert np.isfinite(s) and s >= 0.0
    assert obs.weight == 3.0

    pd = PathDistCritic(weight=1.5)
    s = pd.score(_make_trajectory(poses), ctx)
    assert np.isfinite(s) and s >= 0.0
    assert pd.weight == 1.5

    gd = GoalDistCritic(weight=2.0)
    s = gd.score(_make_trajectory(poses), ctx)
    assert np.isfinite(s) and s >= 0.0
    assert gd.weight == 2.0

    pf = PreferForwardCritic(weight=0.5, max_speed=0.3)
    s = pf.score(_make_trajectory(poses, linear_vel=0.1), ctx)
    assert np.isfinite(s) and s >= 0.0
    assert pf.weight == 0.5

    print("[PASS] test_critics_prepare_and_score_with_weights")


# ================================================================== #
# Combined controller test with real critics
# ================================================================== #

def test_combined_controller_with_real_critics():
    """Wire up real critics + generator + controller; must find a valid result."""
    cfg = _make_config()
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

    state = _make_state(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0)
    ctx = _clear_context()
    ctx["last_angular_vel"] = 0.0
    ctx["last_linear_vel"] = 0.0

    result = ctrl.compute_best(state, ctx)
    assert result.success, "Combined controller failed to find valid trajectory"
    assert result.trajectory is not None
    assert cfg.dwb_min_speed <= result.linear_vel <= cfg.dwb_max_speed
    assert -cfg.dwb_max_yaw_rate <= result.angular_vel <= cfg.dwb_max_yaw_rate

    # Critic scores should be recorded
    cs = result.trajectory.critic_scores
    for name in ["obstacle", "path_dist", "goal_dist",
                 "prefer_forward", "oscillation"]:
        assert name in cs, f"{name} not in critic_scores"

    print("[PASS] test_combined_controller_with_real_critics")


if __name__ == "__main__":
    test_obstacle_critic_rejects_collision()
    test_obstacle_critic_accepts_clear_path()
    test_obstacle_critic_soft_cost_near_obstacle()
    test_obstacle_critic_closer_is_costlier()
    test_path_dist_critic_penalizes_deviation()
    test_path_dist_critic_empty_path()
    test_goal_dist_critic_prefers_closer_endpoint()
    test_prefer_forward_prefers_higher_speed()
    test_prefer_forward_penalizes_spin_in_place()
    test_oscillation_critic_penalizes_sign_flip()
    test_critics_prepare_and_score_with_weights()
    test_combined_controller_with_real_critics()
    print("[ALL PASS] DWB critics tests")
