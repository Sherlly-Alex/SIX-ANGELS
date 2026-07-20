#!/usr/bin/env python3
"""MMK2 Dynamic Navigation — state-machine demo with obstacle detection and A* replan.

Run headless validation:
  python run_dynamic_nav.py --headless --max_steps 500
"""

import argparse
import copy
import os
import sys

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
_lidar_path = os.path.join(
    os.path.dirname(_current_dir), "submodules", "MuJoCo-LiDAR"
)
_lidar_path = os.path.normpath(_lidar_path)
if _lidar_path not in sys.path:
    sys.path.insert(0, _lidar_path)

import mujoco
from config.dynamic_nav_config import DynamicNavConfig
from config.slam_config import SLAMConfig
from core.frontier_exploration import FrontierExplorer
from core.slam_estimator import SLAMEstimator
from perception.dynamic_layer import DynamicLayer
from planning.emergency_checker import EmergencyChecker
from planning.planning_grid import PlanningGrid
from planning.path_validator import PathValidator
from robot.mmk2_slam_robot import MMK2SlamRobot
from robot.motion_controller import MotionController
from sim.obstacle_manager import ObstacleManager
from gui.nav_overlay import NavOverlay
from gui.dynamic_nav_viewer import DynamicNavViewer

STATE_MAPPING = "MAPPING"
STATE_WAITING_FOR_GOAL = "WAITING_FOR_GOAL"
STATE_FOLLOWING_PATH = "FOLLOWING_PATH"
STATE_EMERGENCY_STOP = "EMERGENCY_STOP"
STATE_REPLANNING = "REPLANNING"
STATE_WAITING_FOR_CLEARANCE = "WAITING_FOR_CLEARANCE"
STATE_GOAL_REACHED = "GOAL_REACHED"
STATE_SAFE_STOP = "SAFE_STOP"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_x", type=float, default=-0.5)
    parser.add_argument("--goal_y", type=float, default=-2.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--unlimited", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1600,
                        help="Viewer window width in pixels (default: 1600)")
    parser.add_argument("--height", type=int, default=900,
                        help="Viewer window height in pixels (default: 900)")
    args = parser.parse_args()

    goal = [args.goal_x, args.goal_y]

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    # ---------------------------------------------------------------- #
    # 1. Robot + SLAM init
    # ---------------------------------------------------------------- #
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(
        _current_dir, "scenes", "dynamic_nav_mmk2.xml"
    )
    config.render_set["width"] = args.width
    config.render_set["height"] = args.height
    if args.headless:
        config.headless = True
        config.enable_render = False
    config.use_ros2 = False
    config.lidar_enabled = True

    robot = MMK2SlamRobot(config)

    # Set initial camera to show the full room from above
    if not args.headless and hasattr(robot, "free_camera"):
        robot.free_camera.lookat[:] = [-1.5, -2.0, 0.0]  # room center
        robot.free_camera.distance  = 10.0
        robot.free_camera.azimuth   = 90.0
        robot.free_camera.elevation = -50.0

    slam = SLAMEstimator(config)
    init_pose = robot.get_robot_pose_2d()
    slam.set_pose(init_pose)
    start_pos = [init_pose[0], init_pose[1]]

    print(f"[INFO] Robot at ({init_pose[0]:.2f}, {init_pose[1]:.2f})")
    print(f"[INFO] Goal  at ({goal[0]:.2f}, {goal[1]:.2f})")

    # ---------------------------------------------------------------- #
    # 2. MAPPING phase (stationary)
    # ---------------------------------------------------------------- #
    state = STATE_MAPPING
    print(f"[STATE] {state}")

    action = robot.init_joint_ctrl.copy()
    for _ in range(10):
        robot.step(action)

    cutoff = config.lidar_cutoff_dist
    for _ in range(100):
        robot.step(action)
        ranges, angles = robot.get_lidar_scan()
        if len(ranges) == 0:
            continue
        wl, wr = robot.get_wheel_positions()
        odom_pose = slam.update_odometry(wl, wr)
        slam.process_scan(odom_pose, ranges, angles, cutoff)

    coverage = slam.get_coverage()
    print(f"[INFO] Mapping done — coverage={coverage:.4f}")

    # ---------------------------------------------------------------- #
    # 3. Freeze static map  &  create planning / safety modules
    # ---------------------------------------------------------------- #
    static_grid = copy.deepcopy(slam.grid)
    static_dist_map = static_grid.compute_distance_transform().copy()

    dyn_cfg = DynamicNavConfig()
    dynamic_layer = DynamicLayer(static_dist_map,
                                 static_grid.resolution,
                                 dyn_cfg)
    planning_grid = PlanningGrid(static_grid, dynamic_layer,
                                 allow_unknown=True)

    planner = FrontierExplorer(config)
    motion_ctrl = MotionController(config)
    path_validator = PathValidator(dyn_cfg)
    emergency_checker = EmergencyChecker(
        emergency_distance=dyn_cfg.emergency_distance
    )
    obs_mgr = ObstacleManager(robot.mj_model, robot.mj_data, dyn_cfg, rng=rng)

    overlay = None if args.headless else NavOverlay(interval=30)
    viewer = None if args.headless else DynamicNavViewer(config)

    # ---------------------------------------------------------------- #
    # 4. Plan initial path
    # ---------------------------------------------------------------- #
    state = STATE_WAITING_FOR_GOAL
    print(f"[STATE] {state}")

    current_path = planner.plan_path(start_pos, goal, planning_grid)
    if current_path is None:
        print("[ERROR] A* failed initial path")
        print("[STATE] SAFE_STOP")
        return

    state = STATE_FOLLOWING_PATH
    print(f"[STATE] WAITING_FOR_GOAL -> {state}")
    motion_ctrl.set_path(current_path)
    print(f"[INFO] Initial path: {len(current_path)} waypoints")

    # ---------------------------------------------------------------- #
    # 5. State-machine loop
    # ---------------------------------------------------------------- #
    stop_hold_steps = max(
        1, round(dyn_cfg.stop_hold_time
                 / config.timestep / config.decimation)
    )
    emerg_counter = 0
    clearance_timer = 0
    clearance_timeout = 300
    clearance_last_ver = -1
    following_timer = 0
    total_distance = 0.0
    prev_pose = init_pose.copy()
    replan_count = 0

    max_s = None if args.unlimited else args.max_steps
    for gstep in range(max_s if max_s else 999_999_999):
        # ------------------------------------------------------------ #
        # LiDAR + SLAM localisation + dynamic layer (every step)
        # ------------------------------------------------------------ #
        ranges, angles = robot.get_lidar_scan()
        if len(ranges) > 0:
            wl, wr = robot.get_wheel_positions()
            odom_pose = slam.update_odometry(wl, wr)
            slam.process_scan_localization_only(
                odom_pose, ranges, angles, cutoff
            )
            robot_pose = slam.get_pose()

            # Track distance
            dist = np.hypot(
                robot_pose[0] - prev_pose[0],
                robot_pose[1] - prev_pose[1],
            )
            total_distance += dist
            prev_pose = robot_pose.copy()

            # Update dynamic layer (Bresenham raytracing)
            cur_time = robot.mj_data.time
            laser_site = robot.mj_data.site("laser")
            sensor_pose = [
                laser_site.xpos[0], laser_site.xpos[1], robot_pose[2]
            ]
            dynamic_layer.update(
                ranges, angles, sensor_pose, cur_time, planning_grid
            )
            dynamic_layer.expire(cur_time, decay_time=2.0)
        else:
            robot_pose = slam.get_pose()

        # ------------------------------------------------------------ #
        # Emergency check (front LiDAR)
        # ------------------------------------------------------------ #
        if state in (STATE_FOLLOWING_PATH, STATE_EMERGENCY_STOP):
            if len(ranges) > 0 and emergency_checker.check(ranges, angles):
                if state != STATE_EMERGENCY_STOP:
                    print(
                        f"[STATE] {state} -> EMERGENCY_STOP "
                        f"(step {gstep})"
                    )
                state = STATE_EMERGENCY_STOP
                emerg_counter = 0

        # ------------------------------------------------------------ #
        # Path validation (FOLLOWING_PATH only)
        # ------------------------------------------------------------ #
        if state == STATE_FOLLOWING_PATH:
            wp_idx = motion_ctrl.current_waypoint_idx
            if path_validator.confirm_blocked(
                    current_path, wp_idx, planning_grid
            ):
                print(
                    f"[STATE] {state} -> REPLANNING "
                    f"(blocked at wp {wp_idx})"
                )
                state = STATE_REPLANNING

        # ------------------------------------------------------------ #
        # Obstacle spawn (after robot has travelled a bit)
        # ------------------------------------------------------------ #
        if state == STATE_FOLLOWING_PATH:
            following_timer += 1
        if (state == STATE_FOLLOWING_PATH
                and following_timer >= 50
                and not obs_mgr.spawned):
            obs_mgr.enabled = True
            spawned = obs_mgr.update(
                robot.mj_data.time, robot_pose, goal, current_path
            )
            if spawned:
                mujoco.mj_forward(robot.mj_model, robot.mj_data)
                obs_x = robot.mj_data.mocap_pos[0][0]
                obs_y = robot.mj_data.mocap_pos[0][1]
                print(
                    f"[INFO] Obstacle spawned at "
                    f"({obs_x:.2f}, {obs_y:.2f})"
                )
                if viewer is not None:
                    try:
                        viewer.notify_obstacle_spawn((obs_x, obs_y), gstep)
                    except Exception:
                        pass
        # ------------------------------------------------------------ #
        # State actions
        # ------------------------------------------------------------ #
        linear_vel, angular_vel = 0.0, 0.0

        if state == STATE_FOLLOWING_PATH:
            linear_vel, angular_vel, completed = motion_ctrl.follow_path(
                robot_pose, current_path
            )
            if completed:
                print(f"[STATE] FOLLOWING_PATH -> GOAL_REACHED")
                state = STATE_GOAL_REACHED
                linear_vel, angular_vel = 0.0, 0.0

        elif state == STATE_EMERGENCY_STOP:
            linear_vel, angular_vel = 0.0, 0.0
            emerg_counter += 1
            if emerg_counter >= stop_hold_steps:
                print(f"[STATE] EMERGENCY_STOP -> REPLANNING")
                emerg_counter = 0
                state = STATE_REPLANNING

        elif state == STATE_REPLANNING:
            linear_vel, angular_vel = 0.0, 0.0
            old_path = current_path
            new_path = planner.plan_path(
                [robot_pose[0], robot_pose[1]], goal, planning_grid
            )
            if new_path is not None:
                replan_count += 1
                print(
                    f"[STATE] REPLANNING -> FOLLOWING_PATH "
                    f"(#{replan_count}, {len(new_path)} wp)"
                )
                if viewer is not None:
                    try:
                        viewer.notify_replan(old_path, new_path, gstep)
                    except Exception:
                        pass
                current_path = new_path
                motion_ctrl.set_path(current_path)
                path_validator.reset()
                state = STATE_FOLLOWING_PATH
            else:
                clearance_last_ver = dynamic_layer.version
                print(
                    f"[STATE] REPLANNING -> WAITING_FOR_CLEARANCE "
                    f"(ver={clearance_last_ver})"
                )
                state = STATE_WAITING_FOR_CLEARANCE
                clearance_timer = 0

        elif state == STATE_WAITING_FOR_CLEARANCE:
            linear_vel, angular_vel = 0.0, 0.0
            clearance_timer += 1
            if dynamic_layer.version != clearance_last_ver:
                print(
                    f"[STATE] WAITING_FOR_CLEARANCE -> REPLANNING "
                    f"(ver {clearance_last_ver} -> {dynamic_layer.version})"
                )
                state = STATE_REPLANNING
            elif clearance_timer >= clearance_timeout:
                print("[STATE] WAITING_FOR_CLEARANCE -> SAFE_STOP (timeout)")
                state = STATE_SAFE_STOP

        elif state in (STATE_GOAL_REACHED, STATE_SAFE_STOP):
            linear_vel, angular_vel = 0.0, 0.0

        # ------------------------------------------------------------ #
        # Velocity command -> wheel speeds -> step
        # ------------------------------------------------------------ #
        wd = robot.wheel_distance
        wr = robot.wheel_radius
        v_left = (linear_vel - angular_vel * wd / 2.0) / wr
        v_right = (linear_vel + angular_vel * wd / 2.0) / wr

        action = robot.init_joint_ctrl.copy()
        action[0] = np.clip(v_left, -10.0, 10.0)
        action[1] = np.clip(v_right, -10.0, 10.0)
        robot.step(action)

        # ------------------------------------------------------------ #
        # Periodic status
        # ------------------------------------------------------------ #
        if gstep % 50 == 0:
            print(
                f"[{state}] step={gstep:4d}  "
                f"pose=({robot_pose[0]:+.1f}, {robot_pose[1]:+.1f})  "
                f"dist={total_distance:.2f}m  ndyn={np.sum(dynamic_layer.occupied_mask())}"
            )
        if overlay is not None:
            overlay.update(
                gstep, state, robot_pose, total_distance,
                dynamic_layer, replan_count, linear_vel, angular_vel
            )
        if viewer is not None:
            try:
                viewer.update(
                    gstep, slam, robot_pose, ranges, angles,
                    planning_grid, dynamic_layer, current_path, goal,
                    obs_mgr, state, total_distance, replan_count
                )
            except Exception:
                pass

        if state in (STATE_GOAL_REACHED, STATE_SAFE_STOP):
            break

    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print(f"\n[DONE] Final state: {state}")
    print(f"[DONE] Steps: {gstep + 1}")
    print(f"[DONE] Distance: {total_distance:.2f} m")
    print(f"[DONE] Replans: {replan_count}")

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
