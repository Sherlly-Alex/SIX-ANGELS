from __future__ import annotations

import math
import unittest

import numpy as np

from navigation.navigation_controller import NavigationController
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    SpeedLimits,
)
from navigation.occupancy_grid import build_material_scene_grid


def material_grid_with_cached_distance_map():
    """Build the real scene grid without requiring SciPy in the test runtime."""
    grid = build_material_scene_grid()
    occupied = np.argwhere(grid._grid != 0)
    boundary = []
    for x in range(grid.width):
        boundary.extend(((0, x), (grid.height - 1, x)))
    for y in range(grid.height):
        boundary.extend(((y, 0), (y, grid.width - 1)))
    sites = np.vstack((occupied, np.asarray(boundary, dtype=int)))
    yy, xx = np.indices(grid._grid.shape)
    distance_squared = np.full(grid._grid.shape, np.inf)
    for site_y, site_x in sites:
        distance_squared = np.minimum(
            distance_squared,
            (yy - site_y) ** 2 + (xx - site_x) ** 2,
        )
    grid._dist_map = np.sqrt(distance_squared)
    return grid


class NavigationControllerTests(unittest.TestCase):
    def test_reaches_both_randomized_table_side_pick_stands(self) -> None:
        for target_x in (-1.00, -0.18):
            with self.subTest(target_x=target_x):
                controller = NavigationController(
                    material_grid_with_cached_distance_map(),
                    SpeedLimits(0.20, 0.65, 0.35, 1.20, 0.20, 0.50),
                    pos_tolerance=0.08,
                    yaw_tolerance=0.05,
                    lookahead_distance=0.45,
                    timeout=60.0,
                    emergency_distance=0.20,
                )
                goal = NavigationGoal(
                    x=target_x,
                    y=1.64,
                    yaw=math.pi / 2.0,
                    position_tolerance=0.08,
                    yaw_tolerance=0.05,
                    safety_radius=0.56,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="test",
                )
                x, y, yaw = -0.70, 0.55, math.pi / 2.0
                self.assertTrue(controller.set_goal(goal, x, y))

                for _ in range(400):
                    command = controller.update(x, y, yaw, 0.05, None)
                    self.assertLessEqual(abs(command.linear_x), 0.20 + 1e-9)
                    self.assertLessEqual(abs(command.angular_z), 0.65 + 1e-9)
                    x += command.linear_x * math.cos(yaw) * 0.05
                    y += command.linear_x * math.sin(yaw) * 0.05
                    yaw = (
                        yaw + command.angular_z * 0.05 + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    if controller.status is NavigationStatus.GOAL_REACHED:
                        break

                self.assertEqual(controller.status, NavigationStatus.GOAL_REACHED)
                self.assertLessEqual(math.hypot(x - goal.x, y - goal.y), 0.08)


if __name__ == "__main__":
    unittest.main()
