"""Configuration for MMK2 Dynamic Navigation modules."""


class DynamicNavConfig:
    """Central configuration shared across dynamic navigation modules.

    Each attribute is in SI units (meters / seconds) unless noted otherwise.
    """

    static_match_tolerance = 0.15

    path_inflation_radius = 0.80
    path_min_clearance = 0.40

    map_resolution = 0.05

    replan_cooldown = 1.0

    stop_hold_time = 0.2

    emergency_distance = 0.35

    obstacle_spawn_min_dist = 1.0
    obstacle_spawn_min_goal_dist = 0.5
    spawn_clearance = 0.05

    # DWB-inspired local controller parameters
    dwb_min_speed = 0.0
    dwb_max_speed = 0.30
    dwb_max_yaw_rate = 0.60
    dwb_max_accel = 0.30
    dwb_max_yaw_accel = 1.00
    dwb_v_resolution = 0.03
    dwb_yaw_rate_resolution = 0.10
    dwb_control_period = 0.10
    dwb_prediction_dt = 0.10
    dwb_prediction_horizon = 2.0
    dwb_robot_radius = 0.22
    dwb_safety_margin = 0.15
    dwb_local_goal_distance = 1.20

    # DWB critic default weights
    dwb_obstacle_weight = 3.0
    dwb_path_dist_weight = 1.0
    dwb_goal_dist_weight = 1.2
    dwb_prefer_forward_weight = 1.0
    dwb_oscillation_weight = 1.5

    # Anti spin-in-place (low-v / high-w trap)
    dwb_min_progress_speed = 0.06
    dwb_spin_yaw_threshold = 0.25
    dwb_spin_penalty = 2.0
    dwb_spin_escape_score_margin = 0.8

    # DWB integration parameters
    dwb_failure_confirm_count = 5
    dwb_replan_timeout = 1.0

    # Moving obstacle experiment
    obstacle_movement_mode = "static"  # "static" or "scripted_line"
    obstacle_movement_speed = 0.15     # m/s along scripted line
