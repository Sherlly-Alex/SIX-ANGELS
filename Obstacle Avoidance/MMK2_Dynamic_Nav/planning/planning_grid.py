"""PlanningGrid adapter combining a frozen static map and a dynamic layer."""
import numpy as np
from scipy.ndimage import distance_transform_edt


class PlanningGrid:
    """Adapter that exposes the original A* planner interface.

    It merges a frozen static occupancy grid with a DynamicLayer so that
    both static obstacles and newly detected dynamic obstacles are treated
    as occupied during planning and distance-transform computation.
    """

    def __init__(self, static_grid, dynamic_layer, allow_unknown=False):
        self.static_grid = static_grid
        self.dynamic_layer = dynamic_layer
        self.allow_unknown = allow_unknown

        self.width = static_grid.width
        self.height = static_grid.height
        self.resolution = static_grid.resolution
        self.origin = static_grid.origin

        self._dist_map = None
        self._last_dynamic_version = -1

    def world_to_map(self, x, y):
        return self.static_grid.world_to_map(x, y)

    def map_to_world(self, mx, my):
        return self.static_grid.map_to_world(mx, my)

    def is_occupied(self, mx, my):
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return True
        if self.static_grid.is_occupied(mx, my):
            return True
        if self.dynamic_layer.is_occupied(mx, my):
            return True
        if not self.allow_unknown and self.static_grid.is_unknown(mx, my):
            return True
        return False

    def is_free(self, mx, my):
        if self.dynamic_layer.is_occupied(mx, my):
            return False
        if not self.allow_unknown and self.static_grid.is_unknown(mx, my):
            return False
        return self.static_grid.is_free(mx, my)

    def is_unknown(self, mx, my):
        return self.static_grid.is_unknown(mx, my)

    def compute_distance_transform(self):
        if (self._last_dynamic_version == self.dynamic_layer.version
                and self._dist_map is not None):
            return self._dist_map

        occupied = (self.static_grid.log_odds > 0.5)
        occupied = occupied | self.dynamic_layer.occupied_mask()
        if not self.allow_unknown:
            unknown = np.abs(self.static_grid.log_odds) <= 0.1
            occupied = occupied | unknown

        self._dist_map = distance_transform_edt(~occupied).astype(np.float32)
        self._last_dynamic_version = self.dynamic_layer.version
        return self._dist_map

    def get_inflation_cost(self, mx, my, inflation_radius,
                           cost_weight=3.0, min_clearance=0.20):
        """Return inflation cost compatible with the existing A* interface."""
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return float("inf")

        dist_map = self.compute_distance_transform()
        dist = dist_map[my, mx]

        if dist <= min_clearance:
            return float("inf")
        if dist >= inflation_radius:
            return 1.0

        denominator = inflation_radius - min_clearance
        if denominator <= 1e-9:
            return 1.0

        t = (dist - min_clearance) / denominator
        return 1.0 + cost_weight * (1.0 - t) ** 2
