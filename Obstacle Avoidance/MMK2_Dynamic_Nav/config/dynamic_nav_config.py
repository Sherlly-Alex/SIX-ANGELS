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
