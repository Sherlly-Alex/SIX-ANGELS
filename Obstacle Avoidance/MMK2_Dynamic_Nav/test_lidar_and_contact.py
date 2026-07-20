"""Verify LiDAR detects mocap obstacle and mj_data.contact records collision."""
import sys
import os
import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import mujoco
from config.slam_config import SLAMConfig
from robot.mmk2_slam_robot import MMK2SlamRobot


def test_lidar_detection():
    """Place obstacle in front of robot and check LiDAR sees it."""
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(_module_dir, "scenes", "dynamic_nav_mmk2.xml")
    config.headless = True
    config.enable_render = False
    config.use_ros2 = False
    config.use_gui = False

    robot = MMK2SlamRobot(config)

    # Robot starts at [-3, -2, 0], facing +x (yaw=0)
    # Place obstacle directly in front at [-1.5, -2, 0.4] (1.5m ahead)
    robot.mj_data.mocap_pos[0] = [-1.5, -2.0, 0.4]
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    # Step a few times to let physics settle
    action = robot.init_joint_ctrl.copy()
    for _ in range(10):
        robot.step(action)

    # Get LiDAR scan
    ranges, angles = robot.get_lidar_scan()

    # Check front-facing rays (angles near 0) have reduced range
    front_mask = np.abs(angles) < np.deg2rad(15)
    front_ranges = ranges[front_mask]
    valid_front = front_ranges[(front_ranges > 0.05) & np.isfinite(front_ranges)]

    assert len(valid_front) > 0, "No valid front LiDAR readings"
    min_front = float(np.min(valid_front))
    # Obstacle is ~1.5m away, expect some rays to hit at roughly that distance
    assert min_front < 3.0, f"Front min range {min_front} too large, obstacle not detected"
    assert min_front > 0.1, f"Front min range {min_front} suspiciously small"
    print(f"[PASS] LiDAR detects obstacle: min front range = {min_front:.3f}m")

    return True


def test_collision_contact():
    """Drive robot into obstacle and check mj_data.contact."""
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(_module_dir, "scenes", "dynamic_nav_mmk2.xml")
    config.headless = True
    config.enable_render = False
    config.use_ros2 = False
    config.use_gui = False

    robot = MMK2SlamRobot(config)

    # Place obstacle very close to robot (0.3m ahead, within collision range)
    robot.mj_data.mocap_pos[0] = [-2.7, -2.0, 0.4]
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    # Drive forward into the obstacle
    action = robot.init_joint_ctrl.copy()
    wheel_speed = 3.0  # rad/s forward
    action[0] = wheel_speed  # left wheel
    action[1] = wheel_speed  # right wheel

    contact_detected = False
    obstacle_geom_id = mujoco.mj_name2id(
        robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_obstacle_0_geom"
    )

    for step in range(200):
        robot.step(action)
        # Check contacts
        for i in range(robot.mj_data.ncon):
            contact = robot.mj_data.contact[i]
            if contact.geom1 == obstacle_geom_id or contact.geom2 == obstacle_geom_id:
                contact_detected = True
                break
        if contact_detected:
            break

    assert contact_detected, "No contact with obstacle detected after 200 steps"
    print(f"[PASS] Collision contact detected at step {step}")

    return True


if __name__ == "__main__":
    test_lidar_detection()
    test_collision_contact()
    print("[ALL PASS] LiDAR and contact verification complete")
