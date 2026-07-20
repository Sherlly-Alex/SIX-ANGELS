import numpy as np
from collections import deque
import heapq
from config.slam_config import SLAMConfig

class FrontierExplorer:
    def __init__(self, config: SLAMConfig):
        self.min_frontier_size = config.frontier_min_size
        self.reach_threshold = config.frontier_reach_threshold
        self.inflation_radius = getattr(config, 'path_inflation_radius', 0.50)
        self.inflation_cost_weight = getattr(config, 'path_inflation_weight', 5.0)
        self.min_clearance = getattr(config, 'path_min_clearance', 0.15)
        self.frontier_safety_margin = getattr(config, 'frontier_safety_margin', 0.40)
        self.frontier_memory = []
        self.frontier_memory_max = getattr(config, 'frontier_memory_max', 8)
        self.frontier_visit_penalty = getattr(config, 'frontier_visit_penalty', 2.0)
        self.frontier_forget_dist = getattr(config, 'frontier_forget_dist', 1.5)

    def find_frontiers(self, grid):
        log_odds = grid.log_odds
        height, width = log_odds.shape

        dist_map = grid.compute_distance_transform()
        safety_cells = self.frontier_safety_margin / grid.resolution

        visited = np.zeros((height, width), dtype=bool)
        frontiers = []

        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]

        for my in range(height):
            for mx in range(width):
                if visited[my, mx] or not grid.is_free(mx, my):
                    continue

                queue = deque()
                queue.append((mx, my))
                visited[my, mx] = True

                frontier_cells = []

                while queue:
                    cx, cy = queue.popleft()

                    is_frontier = False
                    for dx, dy in neighbors_8:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if not visited[ny, nx] and grid.is_free(nx, ny):
                                visited[ny, nx] = True
                                queue.append((nx, ny))
                            elif grid.is_unknown(nx, ny):
                                is_frontier = True

                    if is_frontier:
                        if dist_map[cy, cx] >= safety_cells:
                            frontier_cells.append((cx, cy))

                if len(frontier_cells) >= self.min_frontier_size:
                    frontiers.append(frontier_cells)

        return frontiers

    def select_target(self, frontiers, robot_pose, grid):
        if not frontiers:
            return None

        rx, ry = robot_pose[0], robot_pose[1]

        dist_map = grid.compute_distance_transform()
        res = grid.resolution
        safe_dist_cells = self.frontier_safety_margin / res

        best_target = None
        best_score = float('inf')

        for frontier in frontiers:
            wx_sum, wy_sum = 0.0, 0.0
            for mx, my in frontier:
                wx, wy = grid.map_to_world(mx, my)
                wx_sum += wx
                wy_sum += wy

            cx = wx_sum / len(frontier)
            cy = wy_sum / len(frontier)

            best_cell = None
            best_cell_dist = float('inf')
            min_obs_dist = float('inf')

            for mx, my in frontier:
                obs_d = dist_map[my, mx] * res
                if obs_d < min_obs_dist:
                    min_obs_dist = obs_d

                wx, wy = grid.map_to_world(mx, my)
                d_to_centroid = np.sqrt((wx - cx)**2 + (wy - cy)**2)
                if obs_d >= safe_dist_cells and d_to_centroid < best_cell_dist:
                    best_cell_dist = d_to_centroid
                    best_cell = (wx, wy)

            if best_cell is None:
                max_obs = 0.0
                for mx, my in frontier:
                    obs_d = dist_map[my, mx] * res
                    wx, wy = grid.map_to_world(mx, my)
                    d_to_centroid = np.sqrt((wx - cx)**2 + (wy - cy)**2)
                    score = d_to_centroid - obs_d * 2.0
                    if score < best_cell_dist:
                        best_cell_dist = score
                        best_cell = (wx, wy)

            if best_cell is None:
                continue

            target_x, target_y = best_cell

            dist = np.sqrt((target_x - rx) ** 2 + (target_y - ry) ** 2)

            safety_bonus = min(min_obs_dist, 2.0)
            effective_dist = dist - safety_bonus

            for mem_x, mem_y in self.frontier_memory:
                mem_dist = np.sqrt((target_x - mem_x) ** 2 + (target_y - mem_y) ** 2)
                if mem_dist < self.frontier_forget_dist:
                    effective_dist += self.frontier_visit_penalty
                    break

            if effective_dist < best_score:
                best_score = effective_dist
                best_target = [target_x, target_y]

        return best_target

    def plan_path(self, start_world, goal_world, grid):
        start_mx, start_my = grid.world_to_map(start_world[0], start_world[1])
        goal_mx, goal_my = grid.world_to_map(goal_world[0], goal_world[1])

        if start_mx < 0 or goal_mx < 0:
            return None

        grid.compute_distance_transform()
        inflation_radius_cells = self.inflation_radius / grid.resolution
        min_clearance_cells = self.min_clearance / grid.resolution

        if grid.is_occupied(goal_mx, goal_my):
            goal_mx, goal_my = self._find_nearest_free(goal_mx, goal_my, grid)
            if goal_mx < 0:
                return None

        open_set = []
        heapq.heappush(open_set, (0.0, (start_mx, start_my)))

        came_from = {}
        g_score = {(start_mx, start_my): 0.0}

        neighbors_8 = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
                       (0, -1, 1.0),                      (0, 1, 1.0),
                       (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)]

        max_iterations = grid.width * grid.height
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            cx, cy = current

            if cx == goal_mx and cy == goal_my:
                return self._reconstruct_path(came_from, current, grid)

            for dx, dy, base_cost in neighbors_8:
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue

                if grid.is_occupied(nx, ny):
                    continue

                inflation = grid.get_inflation_cost(
                    nx, ny, inflation_radius_cells,
                    cost_weight=self.inflation_cost_weight,
                    min_clearance=min_clearance_cells
                )
                if inflation == float('inf'):
                    continue

                if grid.is_unknown(nx, ny):
                    step_cost = base_cost * 5.0 * inflation
                else:
                    step_cost = base_cost * inflation

                tentative_g = g_score[current] + step_cost

                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    h = np.sqrt((nx - goal_mx) ** 2 + (ny - goal_my) ** 2)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def _find_nearest_free(self, mx, my, grid, max_radius=10):
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        if grid.is_free(nx, ny):
                            return nx, ny
        return -1, -1

    def _reconstruct_path(self, came_from, current, grid):
        path = []
        while current in came_from:
            mx, my = current
            wx, wy = grid.map_to_world(mx, my)
            path.append([wx, wy])
            current = came_from[current]

        mx, my = current
        wx, wy = grid.map_to_world(mx, my)
        path.append([wx, wy])

        path.reverse()
        return path

    def get_exploration_target(self, robot_pose, grid):
        frontiers = self.find_frontiers(grid)
        target = self.select_target(frontiers, robot_pose, grid)

        if target is None:
            return None, None

        gmx, gmy = grid.world_to_map(target[0], target[1])
        if gmx < 0 or not grid.is_free(gmx, gmy):
            gmx, gmy = self._find_nearest_free(
                gmx if gmx >= 0 else 0,
                gmy if gmy >= 0 else 0,
                grid, max_radius=20
            )
            if gmx < 0:
                return None, None
            target[0], target[1] = grid.map_to_world(gmx, gmy)

        path = self.plan_path(
            [robot_pose[0], robot_pose[1]],
            target,
            grid
        )

        if path is None:
            return target, None

        self._add_frontier_memory(target)

        return target, path

    def _add_frontier_memory(self, target):
        self.frontier_memory.append((target[0], target[1]))
        if len(self.frontier_memory) > self.frontier_memory_max:
            self.frontier_memory.pop(0)

    def reset_memory(self):
        self.frontier_memory = []
