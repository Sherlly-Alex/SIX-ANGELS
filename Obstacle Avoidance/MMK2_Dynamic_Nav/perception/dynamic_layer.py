"""Dynamic occupancy layer that only records obstacles not in the static map.

This module is intentionally dependency-free except for NumPy. It does not
perform LiDAR raycasting or coordinate conversion; callers feed grid cells
via mark_hit / mark_miss and the layer tracks evidence over time.
"""
import numpy as np


class DynamicLayer:
    """Track dynamic (non-static) obstacles on a 2D grid.

    Parameters
    ----------
    static_dist_map : np.ndarray
        Precomputed distance transform of the static map. Shape (height, width).
        Values are in grid cells; callers must multiply by resolution to get
        metric distances. A larger value means farther from any static obstacle.
    resolution : float
        Map resolution in meters per grid cell.
    config : object
        Configuration object providing ``static_match_tolerance`` in meters.
        A dynamic hit is only tracked at cells whose distance to the static
        map exceeds this tolerance.
    """

    def __init__(self, static_dist_map, resolution, config):
        if not isinstance(static_dist_map, np.ndarray):
            static_dist_map = np.asarray(static_dist_map)
        self.height, self.width = static_dist_map.shape
        self.resolution = float(resolution)
        self.config = config

        self.static_dist_map = static_dist_map.astype(np.float32)

        # 0 = no evidence, 1 = occupied
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.timestamps = np.zeros((self.height, self.width), dtype=np.float64)
        self.hit_count = np.zeros((self.height, self.width), dtype=np.int32)
        self.miss_count = np.zeros((self.height, self.width), dtype=np.int32)

        self.version = 0
        self.hit_threshold = 2
        self.miss_threshold = 2

    # ------------------------------------------------------------------ #
    # Grid-level cell operations
    # ------------------------------------------------------------------ #

    def mark_hit(self, mx, my, timestamp):
        """Record one hit observation at map cell (mx, my).

        The cell becomes occupied after ``hit_threshold`` consecutive hits.
        ``version`` only increments on 0→1 transitions.
        """
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return

        self.hit_count[my, mx] = min(self.hit_count[my, mx] + 1, self.hit_threshold)
        self.miss_count[my, mx] = 0

        if self.hit_count[my, mx] >= self.hit_threshold:
            was_occupied = self.grid[my, mx] > 0
            self.timestamps[my, mx] = timestamp
            if not was_occupied:
                self.grid[my, mx] = 1.0
                self.version += 1

    def mark_miss(self, mx, my):
        """Record one miss (free-space observation) at map cell (mx, my).

        The cell is cleared after ``miss_threshold`` consecutive misses.
        ``version`` only increments on 1→0 transitions.
        """
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return

        self.miss_count[my, mx] += 1

        if self.miss_count[my, mx] >= self.miss_threshold:
            was_occupied = self.grid[my, mx] > 0
            if was_occupied:
                self.grid[my, mx] = 0.0
                self.hit_count[my, mx] = 0
                self.version += 1

    def expire(self, current_time, decay_time=2.0):
        """Clear occupied cells that have not been refreshed for too long."""
        expired = (current_time - self.timestamps > decay_time) & (self.grid > 0)
        if np.any(expired):
            self.grid[expired] = 0.0
            self.hit_count[expired] = 0
            self.version += 1

    # ------------------------------------------------------------------ #
    # Raytracing update
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Yield (mx, my) cells along the line from (x0, y0) to (x1, y1)
        inclusive of both endpoints.

        Standard Bresenham line algorithm adapted to the project's
        ``[my, mx]`` grid indexing where *y* is the row and *x* the column.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # ------------------------------------------------------------------ #
    # Full LiDAR integration
    # ------------------------------------------------------------------ #

    def update(self, ranges, angles, pose, timestamp, grid):
        """Process a full LiDAR scan: raytrace free space + hit check.

        For each valid ray:
        1.  Compute the world endpoint.
        2.  Convert sensor position and endpoint to map cells.
        3.  Bresenham from sensor to endpoint → ``mark_miss`` on every
            traversed cell (including start, including endpoint).
        4.  At the endpoint, if ``should_track`` returns True, call
            ``mark_hit`` which resets the miss evidence at that cell
            and increments hit evidence towards occupation.

        Parameters
        ----------
        ranges : np.ndarray
            LiDAR range readings (meters).
        angles : np.ndarray
            LiDAR beam angles (radians) in the **sensor** frame.
        pose : (float, float, float)
            Sensor world pose ``[x, y, yaw]``.
        timestamp : float
            Current simulation time (seconds).
        grid : object
            Any object with ``world_to_map(x, y) -> (mx, my)``.
        """
        sensor_mx, sensor_my = grid.world_to_map(pose[0], pose[1])

        for i in range(len(ranges)):
            r = ranges[i]
            if not np.isfinite(r) or r <= 0.05:
                continue

            world_angle = pose[2] + angles[i]
            ex = pose[0] + r * np.cos(world_angle)
            ey = pose[1] + r * np.sin(world_angle)
            emx, emy = grid.world_to_map(ex, ey)
            if emx < 0:
                continue

            if sensor_mx >= 0:
                for cx, cy in self._bresenham(
                        sensor_mx, sensor_my, emx, emy
                ):
                    self.mark_miss(cx, cy)

            if DynamicLayer.should_track(
                    emx, emy, self.static_dist_map,
                    self.resolution,
                    self.config.static_match_tolerance,
            ):
                self.mark_hit(emx, emy, timestamp)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def is_occupied(self, mx, my):
        """Return True if cell (mx, my) is currently occupied."""
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.grid[my, mx] > 0

    def occupied_mask(self):
        """Return boolean mask of occupied cells."""
        return self.grid > 0

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def should_track(mx, my, static_dist_map, resolution, tolerance):
        """Return True if a hit at (mx, my) is far enough from static obstacles.

        Parameters
        ----------
        mx, my : int
            Grid cell indices (column, row).  ``static_dist_map[my, mx]`` is
            used following the project's ``[my, mx]`` indexing convention.
        static_dist_map : np.ndarray
            Static distance transform in grid cells.
        resolution : float
            Meters per grid cell.
        tolerance : float
            Metric distance threshold; cells closer than this to a static
            obstacle are not tracked as dynamic obstacles.
        """
        if not (0 <= mx < static_dist_map.shape[1] and 0 <= my < static_dist_map.shape[0]):
            return False
        return static_dist_map[my, mx] * resolution > tolerance
