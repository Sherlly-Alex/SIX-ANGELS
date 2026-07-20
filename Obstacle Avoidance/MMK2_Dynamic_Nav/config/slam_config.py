import os
import numpy as np
from discoverse.robots_env.mmk2_base import MMK2Cfg

_scenes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes")

class SLAMConfig(MMK2Cfg):
    mjcf_file_path = os.path.join(_scenes_dir, "slam_room_mmk2.xml")
    wheel_distance = 0.3265
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720
    }
    use_gaussian_renderer = False

    init_state     = {
        "base_position"    : [-3.0, -2.0, 0.0],
        "base_orientation" : [1.0, 0.0, 0.0, 0.0],
        "slide_qpos"       : [0.0],
        "head_qpos"        : [0.0, 0.0],
        "lft_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lft_gripper_qpos" : [0.0],
        "rgt_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rgt_gripper_qpos" : [0.0],
    }

    lidar_enabled = True
    lidar_site_name = "laser"
    lidar_backend = "cpu"
    lidar_cutoff_dist = 10.0
    lidar_horizontal_resolution = 360
    lidar_horizontal_fov = 2 * np.pi
    lidar_publish_rate = 10

    map_resolution = 0.05
    map_width = 300
    map_height = 220
    map_origin = [-7.5, -5.5]
    map_log_odd_lo = -100
    map_log_odd_hi = 100
    map_log_odd_free = -1.0
    map_log_odd_occupied = 2.0

    odom_wheel_radius = 0.0838
    odom_wheel_distance = 0.3265
    odom_noise_alpha1 = 0.0
    odom_noise_alpha2 = 0.0
    odom_noise_alpha3 = 0.0
    odom_noise_alpha4 = 0.0

    scan_match_max_iter = 50
    scan_match_tolerance = 1e-4
    scan_match_max_translation = 0.5
    scan_match_max_rotation = 0.3
    scan_match_min_score = 0.5
    scan_match_correspondence_dist = 0.3
    scan_match_map_radius = 5.0
    scan_match_keyframe_dist = 0.05
    scan_match_keyframe_rot = 0.05

    path_inflation_radius = 0.80
    path_inflation_weight = 6.0
    robot_radius = 0.22
    path_min_clearance = 0.40

    lidar_collision_dist = 0.5
    lidar_slowdown_dist = 1.0
    lidar_front_arc = np.deg2rad(60)

    frontier_min_size = 5
    frontier_reach_threshold = 0.5
    exploration_step_time = 1.0
    frontier_safety_margin = 0.45
    frontier_visit_penalty = 2.5
    frontier_forget_dist = 1.5
    frontier_memory_max = 8

    trajectory_reach_threshold = 0.2

    max_linear_vel = 0.3
    max_angular_vel = 0.6
    kp_angular = 1.5
    kp_linear = 1.0

    use_ros2 = True
    use_gui = True
    gui_update_rate = 5
