"""Manage mocap obstacles in the MuJoCo scene for dynamic navigation.

Phase 1 supports spawning a single obstacle at a position on the current
navigation path during the FOLLOWING_PATH state.
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

    def spawn(self, position, size_key="medium", yaw=0.0):
        """Place the mocap obstacle at *position* (world x, y).

        Parameters
        ----------
        position : (float, float)
            World (x, y) coordinates.
        size_key : str
            Key into ``OBSTACLE_PRESETS``.
        yaw : float
            Rotation around z (radians).  Not used in Phase 1.
        """
        preset = self.OBSTACLE_PRESETS[size_key]
        self.mj_data.mocap_pos[0] = [position[0], position[1], preset["spawn_z"]]
        self.spawned = True

    def update(self, current_time, robot_pose, goal, current_path):
        """Try to spawn when conditions are met.

        Called once per simulation step.  Spawns **once** when:
        - ``enabled`` is True
        - ``spawned`` is False
        - A ``current_path`` is provided
        - At least one path point satisfies the distance constraints

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

        # Pick randomly from the front ~60% of candidate points.
        idx = self.rng.integers(0, max(1, int(len(candidates) * 0.6)))
        chosen = candidates[idx]

        # Sample with rejection: keep the final position inside the map and
        # away from the robot.  A few retries make deterministic tests robust.
        for _ in range(5):
            pos_x = chosen[0] + self.rng.normal(0, 0.3)
            pos_y = chosen[1] + self.rng.normal(0, 0.3)

            pos_x = float(np.clip(pos_x, -7.0, 7.0))
            pos_y = float(np.clip(pos_y, -5.0, 5.0))

            if np.hypot(pos_x - robot_x, pos_y - robot_y) >= min_robot:
                self.spawn((pos_x, pos_y), size_key="medium")
                return True

        return False
