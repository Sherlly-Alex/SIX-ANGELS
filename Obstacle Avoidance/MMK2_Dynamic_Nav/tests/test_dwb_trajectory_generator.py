"""Unit tests for DWB trajectory generator."""

import sys
import os
import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)

from config.dynamic_nav_config import DynamicNavConfig
from planning.dwb.trajectory import RobotState
from planning.dwb.trajectory_generator import StandardTrajectoryGenerator


def _make_config():
    return DynamicNavConfig()


def _make_generator():
    return StandardTrajectoryGenerator(_make_config())


def test_dynamic_window_respects_global_limits():
    """Window must never exceed global speed / yaw-rate limits."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    v_min, v_max, w_min, w_max = gen.calc_dynamic_window(state)
    assert v_min >= cfg.dwb_min_speed, f"v_min {v_min} < {cfg.dwb_min_speed}"
    assert v_max <= cfg.dwb_max_speed, f"v_max {v_max} > {cfg.dwb_max_speed}"
    assert w_min >= -cfg.dwb_max_yaw_rate
    assert w_max <= cfg.dwb_max_yaw_rate
    print("[PASS] test_dynamic_window_respects_global_limits")


def test_dynamic_window_respects_acceleration_limits():
    """Window must be constrained by acceleration from current velocity."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    # Start at max speed — window should be limited by acceleration
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=cfg.dwb_max_speed,
                       angular_vel=cfg.dwb_max_yaw_rate)
    v_min, v_max, w_min, w_max = gen.calc_dynamic_window(state)
    dt = cfg.dwb_control_period
    expected_v_min = max(cfg.dwb_min_speed,
                         cfg.dwb_max_speed - cfg.dwb_max_accel * dt)
    expected_v_max = min(cfg.dwb_max_speed,
                         cfg.dwb_max_speed + cfg.dwb_max_accel * dt)
    assert abs(v_min - expected_v_min) < 1e-9, f"v_min {v_min} != {expected_v_min}"
    assert abs(v_max - expected_v_max) < 1e-9, f"v_max {v_max} != {expected_v_max}"
    print("[PASS] test_dynamic_window_respects_acceleration_limits")


def test_sampling_includes_window_bounds():
    """Velocity samples must include the window boundaries."""
    gen = _make_generator()
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    v_min, v_max, w_min, w_max = gen.calc_dynamic_window(state)
    samples = gen.sample_velocities(state)
    assert len(samples) > 0, "No samples generated"

    v_vals = [s[0] for s in samples]
    w_vals = [s[1] for s in samples]
    assert abs(min(v_vals) - v_min) < 1e-6, "v_min boundary not sampled"
    assert abs(max(v_vals) - v_max) < 1e-6, "v_max boundary not sampled"
    assert abs(min(w_vals) - w_min) < 1e-6, "w_min boundary not sampled"
    assert abs(max(w_vals) - w_max) < 1e-6, "w_max boundary not sampled"
    print("[PASS] test_sampling_includes_window_bounds")


def test_straight_motion_prediction():
    """Straight motion (w=0) should produce a line along the heading."""
    gen = _make_generator()
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    poses = gen.predict_trajectory(state, linear_vel=0.2, angular_vel=0.0)
    assert poses.shape[1] == 3
    # All y should be ~0, x should increase
    assert np.all(np.abs(poses[:, 1]) < 1e-6), "y should stay 0 for straight motion"
    assert poses[-1, 0] > poses[0, 0], "x should increase for positive velocity"
    print("[PASS] test_straight_motion_prediction")


def test_turning_motion_prediction():
    """Turning motion (w!=0) should produce curved trajectory."""
    gen = _make_generator()
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    poses = gen.predict_trajectory(state, linear_vel=0.2, angular_vel=0.5)
    # Yaw should change
    assert abs(poses[-1, 2] - poses[0, 2]) > 0.1, "Yaw should change during turn"
    # Y should change (turning left from heading 0)
    assert poses[-1, 1] > 0.0, "y should increase for positive angular velocity"
    print("[PASS] test_turning_motion_prediction")


def test_prediction_length_matches_horizon():
    """Number of predicted poses must match horizon / dt."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=0.0)
    poses = gen.predict_trajectory(state, linear_vel=0.2, angular_vel=0.0)
    expected_steps = max(1, int(np.round(cfg.dwb_prediction_horizon
                                         / cfg.dwb_prediction_dt)))
    expected_len = expected_steps + 1  # include initial pose
    assert poses.shape[0] == expected_len, \
        f"Expected {expected_len} poses, got {poses.shape[0]}"
    print("[PASS] test_prediction_length_matches_horizon")


def test_generated_velocities_within_limits():
    """All generated trajectories must have velocities within config limits."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.1, angular_vel=0.0)
    trajectories = gen.generate(state)
    assert len(trajectories) > 0, "No trajectories generated"
    for traj in trajectories:
        assert cfg.dwb_min_speed <= traj.linear_vel <= cfg.dwb_max_speed, \
            f"v={traj.linear_vel} out of bounds"
        assert -cfg.dwb_max_yaw_rate <= traj.angular_vel <= cfg.dwb_max_yaw_rate, \
            f"w={traj.angular_vel} out of bounds"
    print("[PASS] test_generated_velocities_within_limits")


def test_spin_trap_recovery_opens_forward_window():
    """When spinning at near-zero v, window must allow progress-speed samples."""
    cfg = _make_config()
    gen = StandardTrajectoryGenerator(cfg)
    state = RobotState(x=0.0, y=0.0, yaw=0.0,
                       linear_vel=0.0, angular_vel=cfg.dwb_max_yaw_rate)
    v_min, v_max, w_min, w_max = gen.calc_dynamic_window(state)
    assert v_max >= cfg.dwb_min_progress_speed - 1e-9, \
        f"Recovery window v_max={v_max} too small"
    samples = gen.sample_velocities(state)
    max_v = max(s[0] for s in samples)
    assert max_v >= cfg.dwb_min_progress_speed - 1e-9, \
        f"No progress sample; max_v={max_v}"
    print("[PASS] test_spin_trap_recovery_opens_forward_window")


if __name__ == "__main__":
    test_dynamic_window_respects_global_limits()
    test_dynamic_window_respects_acceleration_limits()
    test_sampling_includes_window_bounds()
    test_straight_motion_prediction()
    test_turning_motion_prediction()
    test_prediction_length_matches_horizon()
    test_generated_velocities_within_limits()
    test_spin_trap_recovery_opens_forward_window()
    print("[ALL PASS] DWB trajectory generator tests")
