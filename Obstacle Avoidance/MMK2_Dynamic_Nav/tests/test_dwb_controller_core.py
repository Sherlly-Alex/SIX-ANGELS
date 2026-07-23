"""Unit tests for DWB controller core using mock critics."""

import sys
import os
import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from config.dynamic_nav_config import DynamicNavConfig
from planning.dwb.trajectory import RobotState, Trajectory
from planning.dwb.critic import TrajectoryCritic
from planning.dwb.trajectory_generator import StandardTrajectoryGenerator
from planning.dwb.controller import DWBController


class MockLowCostCritic(TrajectoryCritic):
    """Always returns a low finite cost."""
    name = "mock_low"
    weight = 1.0

    def score(self, trajectory, context):
        return 0.1


class MockVelocityCritic(TrajectoryCritic):
    """Prefers higher linear velocity (lower cost for higher v)."""
    name = "mock_velocity"
    weight = 1.0

    def score(self, trajectory, context):
        return 1.0 - trajectory.linear_vel


class MockInfiniteCritic(TrajectoryCritic):
    """Returns inf for all trajectories — used to test rejection."""
    name = "mock_infinite"
    weight = 1.0

    def score(self, trajectory, context):
        return float("inf")


class MockWeightedCritic(TrajectoryCritic):
    """Returns a fixed cost to test weight application."""
    name = "mock_weighted"
    weight = 5.0

    def score(self, trajectory, context):
        return 2.0


def _make_config():
    return DynamicNavConfig()


def _make_controller(critics):
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    return DWBController(gen, critics)


def test_controller_selects_lowest_cost_trajectory():
    """Controller should select the trajectory with lowest total cost."""
    ctrl = _make_controller([MockLowCostCritic()])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    result = ctrl.compute_best(state, context=None)
    assert result.success, "Should find a valid trajectory"
    assert result.linear_vel >= 0.0
    print("[PASS] test_controller_selects_lowest_cost_trajectory")


def test_controller_applies_critic_weights():
    """Total score should reflect critic weights."""
    critic = MockWeightedCritic()
    ctrl = _make_controller([critic])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    result = ctrl.compute_best(state, context=None)
    assert result.success
    # MockWeightedCritic returns 2.0 with weight 5.0 → total = 10.0
    assert abs(result.trajectory.total_score - 10.0) < 1e-6, \
        f"Expected 10.0, got {result.trajectory.total_score}"
    print("[PASS] test_controller_applies_critic_weights")


def test_infinite_critic_score_rejects_trajectory():
    """Any trajectory with inf critic score must be rejected."""
    ctrl = _make_controller([MockInfiniteCritic()])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    result = ctrl.compute_best(state, context=None)
    assert not result.success, "Should fail when all trajectories are invalid"
    assert result.linear_vel == 0.0
    assert result.angular_vel == 0.0
    print("[PASS] test_infinite_critic_score_rejects_trajectory")


def test_all_invalid_returns_safe_stop():
    """When no valid trajectory exists, return zero velocity and success=False."""
    ctrl = _make_controller([MockInfiniteCritic()])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.15, angular_vel=0.3)
    result = ctrl.compute_best(state, context=None)
    assert not result.success
    assert result.linear_vel == 0.0
    assert result.angular_vel == 0.0
    assert result.trajectory is None
    print("[PASS] test_all_invalid_returns_safe_stop")


def test_result_velocity_within_limits():
    """Output velocities must be within configured limits."""
    cfg = _make_config()
    ctrl = _make_controller([MockLowCostCritic()])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    result = ctrl.compute_best(state, context=None)
    assert result.success
    assert cfg.dwb_min_speed <= result.linear_vel <= cfg.dwb_max_speed
    assert -cfg.dwb_max_yaw_rate <= result.angular_vel <= cfg.dwb_max_yaw_rate
    print("[PASS] test_result_velocity_within_limits")


def test_critic_scores_are_recorded():
    """Each trajectory's critic_scores dict should contain all critic scores."""
    critics = [MockLowCostCritic(), MockVelocityCritic()]
    ctrl = _make_controller(critics)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    result = ctrl.compute_best(state, context=None)
    assert result.success
    assert result.trajectory is not None
    cs = result.trajectory.critic_scores
    assert "mock_low" in cs, "mock_low score not recorded"
    assert "mock_velocity" in cs, "mock_velocity score not recorded"
    assert np.isfinite(cs["mock_low"])
    assert np.isfinite(cs["mock_velocity"])
    print("[PASS] test_critic_scores_are_recorded")


class MockSpinPreferCritic(TrajectoryCritic):
    """Prefers pure spin (v=0, large |w|) unless controller escapes it."""
    name = "mock_spin"
    weight = 1.0

    def score(self, trajectory, context):
        # Lower cost for lower linear velocity and higher |w|
        return trajectory.linear_vel + 0.01 * (0.6 - abs(trajectory.angular_vel))


def test_controller_escapes_spin_when_forward_exists():
    """If best is pure spin but a forward cmd is close in score, pick forward."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    # Spin-trap state: low v, high w — recovery window must open forward samples
    ctrl = DWBController(gen, [MockSpinPreferCritic()])
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.5)
    result = ctrl.compute_best(state, context={})
    assert result.success
    assert result.linear_vel >= cfg.dwb_min_progress_speed - 1e-6, \
        f"Expected escape from spin, got v={result.linear_vel}, w={result.angular_vel}"
    print("[PASS] test_controller_escapes_spin_when_forward_exists")


if __name__ == "__main__":
    test_controller_selects_lowest_cost_trajectory()
    test_controller_applies_critic_weights()
    test_infinite_critic_score_rejects_trajectory()
    test_all_invalid_returns_safe_stop()
    test_result_velocity_within_limits()
    test_critic_scores_are_recorded()
    test_controller_escapes_spin_when_forward_exists()
    print("[ALL PASS] DWB controller core tests")
