"""Path validation: line-segment collision checking with inflation cost."""
import numpy as np


class PathValidator:
    """Check whether a path or segment is blocked using the same inflation
    cost definition as the A* planner.
    """

    def __init__(self, config):
        self.config = config
        self.consecutive_blocked = 0
        self.confirm_threshold = 3
        self.inflation_radius_cells = (
            config.path_inflation_radius / config.map_resolution
        )
        self.min_clearance_cells = (
            config.path_min_clearance / config.map_resolution
        )

    def check_segment(self, p1, p2, grid, sample_step=0.025):
        """Sample a line segment and return True if any point is blocked."""
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(np.ceil(length / sample_step)))
        for i in range(steps + 1):  # 含端点 t=0 到 t=1
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            mx, my = grid.world_to_map(x, y)
            cost = grid.get_inflation_cost(
                mx, my,
                self.inflation_radius_cells,
                min_clearance=self.min_clearance_cells,
            )
            if not np.isfinite(cost):
                return True
        return False

    def confirm_blocked(self, path, start_index, grid, lookahead=2.5):
        """Check upcoming path segments up to lookahead and confirm blocking
        only after consecutive_blocked reaches confirm_threshold.
        """
        accumulated = 0.0
        blocked = False
        for i in range(start_index, len(path) - 1):
            seg_len = np.hypot(
                path[i + 1][0] - path[i][0],
                path[i + 1][1] - path[i][1],
            )
            remaining = lookahead - accumulated
            if remaining <= 0:
                break
            if seg_len > remaining:
                ratio = remaining / seg_len
                partial_end = [
                    path[i][0] + ratio * (path[i + 1][0] - path[i][0]),
                    path[i][1] + ratio * (path[i + 1][1] - path[i][1]),
                ]
                if self.check_segment(path[i], partial_end, grid):
                    blocked = True
                break
            if self.check_segment(path[i], path[i + 1], grid):
                blocked = True
                break
            accumulated += seg_len

        if blocked:
            self.consecutive_blocked += 1
        else:
            self.consecutive_blocked = 0
        return self.consecutive_blocked >= self.confirm_threshold

    def reset(self):
        """Reset the consecutive blocked counter."""
        self.consecutive_blocked = 0
