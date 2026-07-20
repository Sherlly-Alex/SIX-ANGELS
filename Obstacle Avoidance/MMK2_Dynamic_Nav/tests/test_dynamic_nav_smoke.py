"""Headless integration smoke test for dynamic obstacle detection and replan.

Verifies the full pipeline: static mapping → initial A* path → obstacle spawn
→ dynamic-layer detection → path-blocked confirmation → A* replan.
"""

import copy
import os
import sys

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.dirname(_parent_dir))

import mujoco
from config.slam_config import SLAMConfig
from config.dynamic_nav_config import DynamicNavConfig
from core.frontier_exploration import FrontierExplorer
from core.slam_estimator import SLAMEstimator
from perception.dynamic_layer import DynamicLayer
from planning.planning_grid import PlanningGrid
from planning.path_validator import PathValidator
from robot.mmk2_slam_robot import MMK2SlamRobot
from sim.obstacle_manager import ObstacleManager


def test_dynamic_nav_smoke():
    # ---------------------------------------------------------------- #
    # 1. Initialise robot + SLAM
    # ---------------------------------------------------------------- #
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(
        _parent_dir, "scenes", "dynamic_nav_mmk2.xml"
    )
    config.headless = True
    config.enable_render = False
    config.use_ros2 = False
    config.use_gui = False
    config.lidar_enabled = True

    robot = MMK2SlamRobot(config)

    slam = SLAMEstimator(config)
    robot_pose = robot.get_robot_pose_2d()
    slam.set_pose(robot_pose)

    # ---------------------------------------------------------------- #
    # 2. Mapping phase (stationary)
    # ---------------------------------------------------------------- #
    action = robot.init_joint_ctrl.copy()

    for _ in range(10):
        robot.step(action)

    cutoff = config.lidar_cutoff_dist
    mapping_steps = 100

    for step_i in range(mapping_steps):
        robot.step(action)
        ranges, angles = robot.get_lidar_scan()
        if len(ranges) == 0:
            continue
        wl, wr = robot.get_wheel_positions()
        odom_pose = slam.update_odometry(wl, wr)
        slam.process_scan(odom_pose, ranges, angles, cutoff)

    coverage = slam.get_coverage()
    print(f"[INFO] Mapping complete — coverage={coverage:.4f}")

    # ---------------------------------------------------------------- #
    # 3. Freeze static map & create planning wrappers
    # ---------------------------------------------------------------- #
    static_grid = copy.deepcopy(slam.grid)
    static_dist_map = static_grid.compute_distance_transform().copy()

    dyn_cfg = DynamicNavConfig()
    dynamic_layer = DynamicLayer(static_dist_map,
                                 static_grid.resolution,
                                 dyn_cfg)
    planning_grid = PlanningGrid(static_grid, dynamic_layer,
                                 allow_unknown=True)

    # ---------------------------------------------------------------- #
    # 4. Initial A* path
    # ---------------------------------------------------------------- #
    robot_pose = robot.get_robot_pose_2d()
    start = [robot_pose[0], robot_pose[1]]
    goal = [-0.5, -2.0]

    planner = FrontierExplorer(config)
    initial_path = planner.plan_path(start, goal, planning_grid)
    assert initial_path is not None, "ERROR: initial A* path planning failed"
    print(f"[INFO] Initial path planned — {len(initial_path)} waypoints")

    # ---------------------------------------------------------------- #
    # 5. Spawn obstacle on the path
    # ---------------------------------------------------------------- #
    obs_mgr = ObstacleManager(robot.mj_model, robot.mj_data, dyn_cfg)
    obs_mgr.enabled = True
    spawned = obs_mgr.update(0.0, robot_pose, goal, initial_path)
    assert spawned, "ERROR: ObstacleManager failed to spawn obstacle"

    mujoco.mj_forward(robot.mj_model, robot.mj_data)
    print("[INFO] Obstacle spawned and physics forwarded")

    # ---------------------------------------------------------------- #
    # 6. Detection + replan loop
    # ---------------------------------------------------------------- #
    path_validator = PathValidator(dyn_cfg)
    new_path = None
    observed = False

    for step in range(500):
        robot.step(action)
        cur_time = robot.mj_data.time
        latest_pose = robot.get_robot_pose_2d()

        # ------------------------------------------------------------ #
        # Feed dynamic layer from LiDAR endpoints
        # ------------------------------------------------------------ #
        points_2d = robot.get_lidar_points_2d()
        laser_pos = robot.mj_data.site("laser").xpos
        for pt in points_2d:
            wx = laser_pos[0] + pt[0]
            wy = laser_pos[1] + pt[1]
            mx, my = planning_grid.world_to_map(wx, wy)
            if mx < 0:
                continue
            if DynamicLayer.should_track(mx, my, static_dist_map,
                                         planning_grid.resolution,
                                         dyn_cfg.static_match_tolerance):
                dynamic_layer.mark_hit(mx, my, cur_time)

        if not observed and np.any(dynamic_layer.occupied_mask()):
            observed = True
            print(f"[INFO] Dynamic-layer first detection at step {step}")

        if path_validator.confirm_blocked(initial_path, 0, planning_grid):
            print(f"[INFO] PathValidator confirmed blocked at step {step}")
            new_start = [latest_pose[0], latest_pose[1]]
            new_path = planner.plan_path(new_start, goal, planning_grid)
            assert new_path is not None, "ERROR: replanning returned None"
            print(f"[INFO] Replanned path — {len(new_path)} waypoints")
            break

        if step >= 199:
            raise AssertionError(
                "ERROR: obstacle not detected within 200 steps"
            )

    # ---------------------------------------------------------------- #
    # 7. Validate replanning result
    # ---------------------------------------------------------------- #
    assert new_path is not None, "ERROR: no replanned path obtained"

    # The new path must differ from the original.
    differs = (
        len(new_path) != len(initial_path)
        or any(
            not np.allclose(np.array(new_path[i]), np.array(initial_path[i]))
            for i in range(len(new_path))
        )
    )
    assert differs, "ERROR: replanned path is identical to the original"

    print("[ALL PASS] Dynamic nav smoke test complete")
    return True


if __name__ == "__main__":
    test_dynamic_nav_smoke()
