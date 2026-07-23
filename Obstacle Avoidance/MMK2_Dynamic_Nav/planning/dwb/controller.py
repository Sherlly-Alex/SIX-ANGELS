"""DWB controller — scores candidate trajectories and selects the best."""

from typing import List, Optional

import numpy as np

from .critic import TrajectoryCritic
from .trajectory import DWBResult, RobotState, Trajectory
from .trajectory_generator import StandardTrajectoryGenerator


class DWBController:
    """Score trajectories using injected critics and select the best."""

    def __init__(self, generator: StandardTrajectoryGenerator,
                 critics: List[TrajectoryCritic]):
        self.generator = generator
        self.critics = critics
        self.last_candidates = []  # read-only snapshot for GUI

    def compute_best(self, state: RobotState, context) -> DWBResult:
        """Generate candidates, score them, and return the best result."""
        cfg = self.generator.config
        for critic in self.critics:
            critic.prepare(context)

        trajectories = self.generator.generate(state)
        if not trajectories:
            self.last_candidates = []
            return DWBResult(
                linear_vel=0.0,
                angular_vel=0.0,
                trajectory=None,
                success=False,
            )

        scored: List[Trajectory] = []
        for traj in trajectories:
            total = 0.0
            valid = True
            for critic in self.critics:
                s = critic.score(traj, context)
                traj.critic_scores[critic.name] = s
                if not np.isfinite(s):
                    valid = False
                    break
                total += critic.weight * s
            traj.total_score = total
            traj.valid = valid
            if valid:
                scored.append(traj)

        if not scored:
            self.last_candidates = []
            return DWBResult(
                linear_vel=0.0,
                angular_vel=0.0,
                trajectory=None,
                success=False,
            )

        best = self._select_best(scored, cfg)

        v_out = float(np.clip(best.linear_vel,
                              cfg.dwb_min_speed, cfg.dwb_max_speed))
        w_out = float(np.clip(best.angular_vel,
                              -cfg.dwb_max_yaw_rate, cfg.dwb_max_yaw_rate))

        # Escape pure spin: prefer the best progressive (v >= min_v) candidate
        # within a score margin; otherwise any higher-v alternative.
        min_v = float(getattr(cfg, "dwb_min_progress_speed", 0.06))
        max_spin_w = float(getattr(cfg, "dwb_spin_yaw_threshold", 0.25))
        if v_out < min_v and abs(w_out) > max_spin_w:
            margin = float(getattr(cfg, "dwb_spin_escape_score_margin", 0.8))
            progressive = [t for t in scored
                           if t.linear_vel >= min_v - 1e-9
                           and t.total_score <= best.total_score + margin]
            if progressive:
                progressive.sort(key=lambda t: (t.total_score, -t.linear_vel))
                best = progressive[0]
            else:
                better_v = [t for t in scored
                            if t.linear_vel > v_out + 1e-6
                            and t.total_score <= best.total_score + margin]
                if better_v:
                    better_v.sort(key=lambda t: (-t.linear_vel, t.total_score))
                    best = better_v[0]
            v_out = float(np.clip(best.linear_vel,
                                  cfg.dwb_min_speed, cfg.dwb_max_speed))
            w_out = float(np.clip(best.angular_vel,
                                  -cfg.dwb_max_yaw_rate,
                                  cfg.dwb_max_yaw_rate))

        self.last_candidates = [{
            "linear_vel": t.linear_vel,
            "angular_vel": t.angular_vel,
            "total_score": t.total_score,
            "valid": t.valid,
            "poses": t.poses,
        } for t in trajectories]

        return DWBResult(
            linear_vel=v_out,
            angular_vel=w_out,
            trajectory=best,
            success=True,
        )

    @staticmethod
    def _select_best(scored: List[Trajectory], cfg) -> Trajectory:
        best = scored[0]
        for traj in scored[1:]:
            if traj.total_score < best.total_score:
                best = traj
            elif (abs(traj.total_score - best.total_score) < 1e-9
                  and traj.linear_vel > best.linear_vel):
                best = traj
        return best
