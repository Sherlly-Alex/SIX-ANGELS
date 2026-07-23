"""Manage mocap obstacles in the MuJoCo scene for dynamic navigation.

Supports static (default) and scripted_line movement modes.
"""

import numpy as np


class ObstacleManager:
    """Spawn and manage dynamic obstacles via mocap bodies."""

    OBSTACLE_PRESETS = {
        "small":  {"half_size": (0.20, 0.20, 0.30), "spawn_z": 0.30},
        "medium": {"half_size": (0.35, 0.35, 0.40), "spawn_z": 0.40},
        "large":  {"half_size": (0.50, 0.50, 0.50), "spawn_z": 0.50},
        "long":   {"half_size": (0.70, 0.20, 0.40), "spawn_z": 0.40},
    }

    def __init__(self, mj_model, mj_data, config, rng=None):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.enabled = False
        self.spawned = False

        # Movement state
        mode = getattr(config, "obstacle_movement_mode", "static")
        self.movement_mode = mode if mode in ("static", "scripted_line") else "static"
        self._move_speed = getattr(config, "obstacle_movement_speed", 0.15)
        self._move_path = None      # list of [x, y] waypoints for scripted line
        self._move_progress = 0.0   # m along the path
        self._move_forward = True   # True → advancing, False → reversing
        self._spawn_z = 0.40

    def spawn(self, position, size_key="medium", yaw=0.0):
        """Place the mocap obstacle at *position* (world x, y)."""
        preset = self.OBSTACLE_PRESETS[size_key]
        self._spawn_z = preset["spawn_z"]
        self.mj_data.mocap_pos[0] = [position[0], position[1], self._spawn_z]
        self.spawned = True

    def update(self, current_time, robot_pose, goal, current_path):
        """Try to spawn when conditions are met.

        Called once per simulation step.  Spawns **once**.
        Returns True if an obstacle was spawned, False otherwise.
        """
        if not self.enabled or self.spawned:
            return False
        if current_path is None or len(current_path) == 0:
            return False

        robot_x, robot_y = robot_pose[0], robot_pose[1]
        goal_x, goal_y = goal[0], goal[1]
        min_robot = self.config.obstacle_spawn_min_dist
        min_goal = self.config.obstacle_spawn_min_goal_dist

        candidates = []
        for pt in current_path:
            d_r = np.hypot(pt[0] - robot_x, pt[1] - robot_y)
            d_g = np.hypot(pt[0] - goal_x, pt[1] - goal_y)
            if d_r > min_robot and d_g > min_goal:
                candidates.append(pt)

        if not candidates:
            return False

        idx = self.rng.integers(0, max(1, int(len(candidates) * 0.6)))
        chosen = candidates[idx]

        for _ in range(5):
            pos_x = chosen[0] + self.rng.normal(0, 0.3)
            pos_y = chosen[1] + self.rng.normal(0, 0.3)
            pos_x = float(np.clip(pos_x, -7.0, 7.0))
            pos_y = float(np.clip(pos_y, -5.0, 5.0))
            if np.hypot(pos_x - robot_x, pos_y - robot_y) >= min_robot:
                self.spawn((pos_x, pos_y), size_key="medium")

                # Initialise scripted movement path
                if self.movement_mode == "scripted_line":
                    self._init_scripted_motion(pos_x, pos_y, robot_x, robot_y)

                return True

        return False

    # ------------------------------------------------------------------ #
    # Scripted motion
    # ------------------------------------------------------------------ #

    def _init_scripted_motion(self, start_x, start_y, robot_x, robot_y):
        """Create a short back-and-forth path across the spawn point."""
        dx = robot_y - start_y
        dy = -(robot_x - start_x)
        length = np.hypot(dx, dy) or 1.0
        dx /= length
        dy /= length
        step = 0.8
        self._move_path = [
            [start_x - step * dx, start_y - step * dy],
            [start_x, start_y],
            [start_x + step * dx, start_y + step * dy],
        ]
        self._move_progress = 1.0
        self._move_forward = True

    def is_moving(self):
        return self.movement_mode == "scripted_line"

    def step_motion(self, dt):
        """Advance obstacle along its scripted path by *dt* seconds.

        Must be called once per simulation step after spawning.
        """
        if self._move_path is None or len(self._move_path) < 2:
            return
        if not self.spawned:
            return

        total_len = self._path_total_length()
        if total_len < 1e-9:
            return

        delta = self._move_speed * dt
        if self._move_forward:
            self._move_progress += delta
            if self._move_progress >= total_len:
                self._move_progress = total_len
                self._move_forward = False
        else:
            self._move_progress -= delta
            if self._move_progress <= 0.0:
                self._move_progress = 0.0
                self._move_forward = True

        pos = self._interpolate_path(self._move_progress / total_len)
        self.mj_data.mocap_pos[0] = [pos[0], pos[1], self._spawn_z]

    def get_position(self):
        if self.spawned and len(self.mj_data.mocap_pos) > 0:
            return (float(self.mj_data.mocap_pos[0][0]),
                    float(self.mj_data.mocap_pos[0][1]))
        return None

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _path_total_length(self):
        total = 0.0
        for i in range(len(self._move_path) - 1):
            a, b = self._move_path[i], self._move_path[i + 1]
            total += float(np.hypot(b[0] - a[0], b[1] - a[1]))
        return total

    def _interpolate_path(self, t):
        t = max(0.0, min(1.0, t))
        total = self._path_total_length()
        target = t * total
        accumulated = 0.0
        for i in range(len(self._move_path) - 1):
            a = self._move_path[i]
            b = self._move_path[i + 1]
            seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if accumulated + seg_len >= target:
                if seg_len < 1e-9:
                    return [float(a[0]), float(a[1])]
                frac = (target - accumulated) / seg_len
                return [a[0] + frac * (b[0] - a[0]),
                        a[1] + frac * (b[1] - a[1])]
            accumulated += seg_len
        return [float(self._move_path[-1][0]),
                float(self._move_path[-1][1])]
