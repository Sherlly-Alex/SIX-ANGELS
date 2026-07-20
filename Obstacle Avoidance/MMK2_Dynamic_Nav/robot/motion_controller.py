import numpy as np
from config.slam_config import SLAMConfig

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class MotionController:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.max_linear_vel = config.max_linear_vel
        self.max_angular_vel = config.max_angular_vel
        self.kp_angular = config.kp_angular
        self.kp_linear = config.kp_linear
        self.reach_threshold = config.trajectory_reach_threshold

        self.current_path = None
        self.current_waypoint_idx = 0

    def move_to_target(self, robot_pose, target):
        dx = target[0] - robot_pose[0]
        dy = target[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)

        target_angle = np.arctan2(dy, dx)
        angle_error = normalize_angle(target_angle - robot_pose[2])

        angular_vel = np.clip(
            self.kp_angular * angle_error,
            -self.max_angular_vel,
            self.max_angular_vel
        )

        linear_vel = np.clip(
            self.kp_linear * distance * np.cos(angle_error),
            0,
            self.max_linear_vel
        )

        if abs(angle_error) > 0.5:
            linear_vel = 0.0

        reached = distance < self.reach_threshold

        return linear_vel, angular_vel, reached

    def follow_path(self, robot_pose, path):
        if path is None or len(path) == 0:
            return 0.0, 0.0, True

        if self.current_waypoint_idx >= len(path):
            return 0.0, 0.0, True

        target = path[self.current_waypoint_idx]

        linear_vel, angular_vel, reached = self.move_to_target(robot_pose, target)

        if reached:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(path):
                return 0.0, 0.0, True

        return linear_vel, angular_vel, False

    def set_path(self, path):
        self.current_path = path
        self.current_waypoint_idx = 0

    def execute_velocity_command(self, robot, linear_vel, angular_vel):
        robot.apply_diff_drive(linear_vel, angular_vel)

    def stop(self, robot):
        robot.apply_diff_drive(0.0, 0.0)

    def reset(self):
        self.current_path = None
        self.current_waypoint_idx = 0
