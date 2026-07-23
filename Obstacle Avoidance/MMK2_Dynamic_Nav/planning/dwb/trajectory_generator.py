"""Trajectory generator for the DWB-inspired local controller.

Generates candidate (v, w) velocity pairs within the dynamic window and
predicts short-horizon trajectories using a differential-drive motion model.
"""

from typing import List, Tuple

import numpy as np

from .trajectory import RobotState, Trajectory


class StandardTrajectoryGenerator:
    """Sample-based trajectory generator with dynamic window."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------ #
    # Dynamic window
    # ------------------------------------------------------------------ #

    def calc_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """Return (v_min, v_max, w_min, w_max) for the current state."""
        cfg = self.config
        dt = cfg.dwb_control_period

        # Global limits
        v_global_min = cfg.dwb_min_speed
        v_global_max = cfg.dwb_max_speed
        w_global_min = -cfg.dwb_max_yaw_rate
        w_global_max = cfg.dwb_max_yaw_rate

        # Acceleration-limited reachable window
        v_reach_min = state.linear_vel - cfg.dwb_max_accel * dt
        v_reach_max = state.linear_vel + cfg.dwb_max_accel * dt
        w_reach_min = state.angular_vel - cfg.dwb_max_yaw_accel * dt
        w_reach_max = state.angular_vel + cfg.dwb_max_yaw_accel * dt

        # Intersection
        v_min = max(v_global_min, v_reach_min)
        v_max = min(v_global_max, v_reach_max)
        w_min = max(w_global_min, w_reach_min)
        w_max = min(w_global_max, w_reach_max)

        # Spin-trap recovery: if nearly stopped but spinning hard, open the
        # linear window enough to re-accelerate (one-step progress samples).
        min_progress = float(getattr(cfg, "dwb_min_progress_speed", 0.06))
        spin_w = float(getattr(cfg, "dwb_spin_yaw_threshold", 0.25))
        if (state.linear_vel < min_progress
                and abs(state.angular_vel) > spin_w):
            v_max = max(v_max, min(v_global_max, min_progress))
            # Also allow braking the yaw rate toward zero more aggressively.
            w_min = min(w_min, max(w_global_min, -spin_w))
            w_max = max(w_max, min(w_global_max, spin_w))

        return v_min, v_max, w_min, w_max

    # ------------------------------------------------------------------ #
    # Velocity sampling
    # ------------------------------------------------------------------ #

    def sample_velocities(self, state: RobotState) -> List[Tuple[float, float]]:
        """Return a list of (linear_vel, angular_vel) candidates."""
        v_min, v_max, w_min, w_max = self.calc_dynamic_window(state)
        cfg = self.config

        # Ensure valid range
        if v_min > v_max or w_min > w_max:
            return []

        # Number of samples — ensure boundaries are included
        n_v = max(2, int(np.ceil((v_max - v_min) / cfg.dwb_v_resolution)) + 1)
        n_w = max(2, int(np.ceil((w_max - w_min) / cfg.dwb_yaw_rate_resolution)) + 1)

        v_samples = np.linspace(v_min, v_max, n_v)
        w_samples = np.linspace(w_min, w_max, n_w)

        candidates = []
        for v in v_samples:
            for w in w_samples:
                candidates.append((float(v), float(w)))

        return candidates

    # ------------------------------------------------------------------ #
    # Trajectory prediction
    # ------------------------------------------------------------------ #

    def predict_trajectory(self, state: RobotState,
                           linear_vel: float,
                           angular_vel: float) -> np.ndarray:
        """Predict poses using differential-drive model.

        Returns array of shape (N, 3) with columns [x, y, yaw].
        """
        cfg = self.config
        dt = cfg.dwb_prediction_dt
        horizon = cfg.dwb_prediction_horizon
        n_steps = max(1, int(np.round(horizon / dt)))

        poses = np.zeros((n_steps + 1, 3), dtype=np.float64)
        poses[0] = [state.x, state.y, state.yaw]

        x, y, yaw = state.x, state.y, state.yaw
        for i in range(1, n_steps + 1):
            x += linear_vel * np.cos(yaw) * dt
            y += linear_vel * np.sin(yaw) * dt
            yaw += angular_vel * dt
            # Normalize yaw to [-pi, pi]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            poses[i] = [x, y, yaw]

        return poses

    # ------------------------------------------------------------------ #
    # Generate all candidates
    # ------------------------------------------------------------------ #

    def generate(self, state: RobotState) -> List[Trajectory]:
        """Generate all candidate trajectories for the current state."""
        samples = self.sample_velocities(state)
        trajectories = []
        for v, w in samples:
            poses = self.predict_trajectory(state, v, w)
            traj = Trajectory(
                linear_vel=v,
                angular_vel=w,
                poses=poses,
                total_score=0.0,
                critic_scores={},
                valid=True,
            )
            trajectories.append(traj)
        return trajectories
