import numpy as np
from config.slam_config import SLAMConfig

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class DifferentialDriveOdometry:
    def __init__(self, config: SLAMConfig):
        self.wheel_radius = config.odom_wheel_radius
        self.wheel_distance = config.odom_wheel_distance
        self.noise_alpha1 = config.odom_noise_alpha1
        self.noise_alpha2 = config.odom_noise_alpha2
        self.noise_alpha3 = config.odom_noise_alpha3
        self.noise_alpha4 = config.odom_noise_alpha4
        self.pose = np.array([0.0, 0.0, 0.0])
        self.last_wheel_left = None
        self.last_wheel_right = None
        self.trajectory = []

    def update(self, wheel_left_pos, wheel_right_pos):
        if self.last_wheel_left is None:
            self.last_wheel_left = wheel_left_pos
            self.last_wheel_right = wheel_right_pos
            return self.pose.copy()

        d_left = self.wheel_radius * (wheel_left_pos - self.last_wheel_left)
        d_right = self.wheel_radius * (wheel_right_pos - self.last_wheel_right)

        d_left, d_right = self._add_motion_noise(d_left, d_right)

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_distance

        theta = self.pose[2]
        if abs(d_theta) < 1e-6:
            self.pose[0] += d_center * np.cos(theta)
            self.pose[1] += d_center * np.sin(theta)
        else:
            r = d_center / d_theta
            self.pose[0] += r * (np.sin(theta + d_theta) - np.sin(theta))
            self.pose[1] += r * (np.cos(theta) - np.cos(theta + d_theta))

        self.pose[2] = normalize_angle(theta + d_theta)
        self.trajectory.append(self.pose.copy())

        self.last_wheel_left = wheel_left_pos
        self.last_wheel_right = wheel_right_pos

        return self.pose.copy()

    def update_from_velocity(self, linear_vel, angular_vel, dt):
        d_center = linear_vel * dt
        d_theta = angular_vel * dt

        theta = self.pose[2]
        if abs(d_theta) < 1e-6:
            self.pose[0] += d_center * np.cos(theta)
            self.pose[1] += d_center * np.sin(theta)
        else:
            r = d_center / d_theta
            self.pose[0] += r * (np.sin(theta + d_theta) - np.sin(theta))
            self.pose[1] += -r * (np.cos(theta + d_theta) - np.cos(theta))

        self.pose[2] = normalize_angle(theta + d_theta)
        self.trajectory.append(self.pose.copy())

        return self.pose.copy()

    def _add_motion_noise(self, d_left, d_right):
        if (self.noise_alpha1 == 0 and self.noise_alpha2 == 0 and
                self.noise_alpha3 == 0 and self.noise_alpha4 == 0):
            return d_left, d_right

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_distance

        noise_d_theta = self.noise_alpha1 * abs(d_theta) + self.noise_alpha2 * abs(d_center)
        noise_d_center = self.noise_alpha3 * abs(d_center) + self.noise_alpha4 * abs(d_theta)

        d_theta_noisy = d_theta + np.random.normal(0, noise_d_theta)
        d_center_noisy = d_center + np.random.normal(0, noise_d_center)

        d_left_noisy = d_center_noisy - d_theta_noisy * self.wheel_distance / 2
        d_right_noisy = d_center_noisy + d_theta_noisy * self.wheel_distance / 2

        return d_left_noisy, d_right_noisy

    def get_pose(self):
        return self.pose.copy()

    def set_pose(self, pose):
        self.pose = np.array(pose, dtype=np.float64)

    def get_trajectory(self):
        return np.array(self.trajectory)

    def reset(self):
        self.pose = np.array([0.0, 0.0, 0.0])
        self.last_wheel_left = None
        self.last_wheel_right = None
        self.trajectory = []
