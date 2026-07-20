import numpy as np
from config.slam_config import SLAMConfig

class OccupancyGrid:
    def __init__(self, config: SLAMConfig):
        self.resolution = config.map_resolution
        self.width = config.map_width
        self.height = config.map_height
        self.origin = np.array(config.map_origin, dtype=np.float64)

        self.log_odd_lo = config.map_log_odd_lo
        self.log_odd_hi = config.map_log_odd_hi
        self.log_odd_free = config.map_log_odd_free
        self.log_odd_occupied = config.map_log_odd_occupied

        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    def world_to_map(self, x, y):
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)

        if 0 <= mx < self.width and 0 <= my < self.height:
            return mx, my
        return -1, -1

    def map_to_world(self, mx, my):
        wx = self.origin[0] + (mx + 0.5) * self.resolution
        wy = self.origin[1] + (my + 0.5) * self.resolution
        return wx, wy

    def get_occupancy_prob(self):
        clipped = np.clip(self.log_odds, -50, 50)
        prob = 1.0 / (1.0 + np.exp(-clipped))
        return prob

    def get_ros_map_data(self):
        prob = self.get_occupancy_prob()
        data = np.full_like(prob, -1, dtype=np.int8)
        known = np.abs(self.log_odds) > 0.1
        data[known] = (prob[known] * 100).astype(np.int8)
        return data

    def is_occupied(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.log_odds[my, mx] > 0.5

    def is_free(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.log_odds[my, mx] < -0.5

    def is_unknown(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return True
        return abs(self.log_odds[my, mx]) <= 0.1

    def get_occupied_points_around(self, pose, radius):
        cx, cy = pose[0], pose[1]

        min_mx = max(0, int((cx - radius - self.origin[0]) / self.resolution))
        max_mx = min(self.width - 1, int((cx + radius - self.origin[0]) / self.resolution))
        min_my = max(0, int((cy - radius - self.origin[1]) / self.resolution))
        max_my = min(self.height - 1, int((cy + radius - self.origin[1]) / self.resolution))

        if min_mx > max_mx or min_my > max_my:
            return np.array([]).reshape(0, 2)

        sub = self.log_odds[min_my:max_my + 1, min_mx:max_mx + 1]
        occupied = np.argwhere(sub > 0.5)

        if len(occupied) == 0:
            return np.array([]).reshape(0, 2)

        occupied_mx = occupied[:, 1] + min_mx
        occupied_my = occupied[:, 0] + min_my
        wx = self.origin[0] + (occupied_mx + 0.5) * self.resolution
        wy = self.origin[1] + (occupied_my + 0.5) * self.resolution

        return np.column_stack([wx, wy])

    def update_from_scan(self, robot_pose, scan_ranges, scan_angles, max_range):
        if hasattr(self, '_dist_map'):
            self._dist_map = None

        rx, ry, rtheta = robot_pose

        robot_mx, robot_my = self.world_to_map(rx, ry)
        if robot_mx < 0:
            return

        for i in range(len(scan_ranges)):
            r = scan_ranges[i]
            angle = scan_angles[i]

            world_angle = rtheta + angle

            if np.isinf(r) or r >= max_range:
                end_dist = max_range * 0.8
                end_x = rx + np.cos(world_angle) * end_dist
                end_y = ry + np.sin(world_angle) * end_dist
                end_mx, end_my = self.world_to_map(end_x, end_y)
                if end_mx >= 0:
                    self._bresenham_update(robot_mx, robot_my, end_mx, end_my,
                                           mark_end_occupied=False)
            else:
                end_x = rx + np.cos(world_angle) * r
                end_y = ry + np.sin(world_angle) * r
                end_mx, end_my = self.world_to_map(end_x, end_y)
                if end_mx >= 0:
                    self._bresenham_update(robot_mx, robot_my, end_mx, end_my,
                                           mark_end_occupied=True)

    def _bresenham_update(self, x0, y0, x1, y1, mark_end_occupied=True):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.log_odds[y, x] += self.log_odd_free

            if x == x1 and y == y1:
                if mark_end_occupied and 0 <= x < self.width and 0 <= y < self.height:
                    self.log_odds[y, x] -= self.log_odd_free
                    self.log_odds[y, x] += self.log_odd_occupied
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        np.clip(self.log_odds, self.log_odd_lo, self.log_odd_hi, out=self.log_odds)

    def compute_distance_transform(self):
        from scipy.ndimage import distance_transform_edt

        occupied = self.log_odds > 0.5
        dist_map = distance_transform_edt(~occupied).astype(np.float32)

        self._dist_map = dist_map
        return dist_map

    def get_inflation_cost(self, mx, my, inflation_radius, cost_weight=3.0,
                           min_clearance=0.20):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return float('inf')

        if not hasattr(self, '_dist_map') or self._dist_map is None:
            self.compute_distance_transform()

        dist = self._dist_map[my, mx]

        if dist <= min_clearance:
            return float('inf')

        if dist >= inflation_radius:
            return 1.0

        t = (dist - min_clearance) / (inflation_radius - min_clearance)
        return 1.0 + cost_weight * (1.0 - t) ** 2

    def get_coverage(self):
        total = self.width * self.height
        known = np.sum(np.abs(self.log_odds) > 0.1)
        return known / total

    def reset(self):
        self.log_odds.fill(0)
