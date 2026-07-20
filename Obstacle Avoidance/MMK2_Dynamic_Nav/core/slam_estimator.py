import numpy as np
from config.slam_config import SLAMConfig
from core.occupancy_grid import OccupancyGrid
from core.scan_matching import ICPMatcher, transform_points, normalize_angle
from core.odometry import DifferentialDriveOdometry
from core.frontier_exploration import FrontierExplorer

def ranges_to_points(ranges, angles, max_range):
    valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < max_range)
    r = ranges[valid]
    a = angles[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.column_stack([x, y])

class SLAMEstimator:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.odom = DifferentialDriveOdometry(config)
        self.grid = OccupancyGrid(config)
        self.icp = ICPMatcher(config)
        self.explorer = FrontierExplorer(config)
        self.pose = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.prev_scan_points = None
        self.prev_pose = None
        self.scan_count = 0
        self.icp_scores = []
        self.corrections_applied = 0
        self.no_correction_count = 0

    def update_odometry(self, wheel_left, wheel_right):
        return self.odom.update(wheel_left, wheel_right)

    def update_odometry_from_velocity(self, linear_vel, angular_vel, dt):
        return self.odom.update_from_velocity(linear_vel, angular_vel, dt)

    def process_scan(self, odom_pose, ranges, angles, max_range):
        current_points = ranges_to_points(ranges, angles, max_range)
        n_points = len(current_points)

        icp_score = 0.0
        corrected = False
        estimated_pose = odom_pose.copy()

        if n_points >= 10:
            map_points = self.grid.get_occupied_points_around(
                odom_pose, self.config.scan_match_map_radius
            )

            if len(map_points) >= 10:
                scan_world = transform_points(current_points, odom_pose)
                icp_pose, icp_score = self.icp.match(
                    source_points=scan_world,
                    target_points=map_points,
                    init_pose=np.array([0.0, 0.0, 0.0])
                )

                if icp_score > self.config.scan_match_min_score:
                    estimated_pose = odom_pose.copy()
                    estimated_pose[0] += icp_pose[0]
                    estimated_pose[1] += icp_pose[1]
                    estimated_pose[2] = normalize_angle(odom_pose[2] + icp_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

            elif self.prev_scan_points is not None and len(self.prev_scan_points) >= 10:
                odom_delta = odom_pose - self.pose
                icp_pose, icp_score = self.icp.match(
                    source_points=current_points,
                    target_points=self.prev_scan_points,
                    init_pose=odom_delta
                )

                if icp_score > self.config.scan_match_min_score:
                    estimated_pose = self.pose + icp_pose
                    estimated_pose[2] = normalize_angle(estimated_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

        self.pose = estimated_pose.copy()
        self.trajectory.append(self.pose.copy())
        self.scan_count += 1
        self.icp_scores.append(icp_score)

        if corrected:
            self.no_correction_count = 0
        else:
            self.no_correction_count += 1

        self.grid.update_from_scan(self.pose, ranges, angles, max_range)

        if self.prev_scan_points is None or self.prev_pose is None:
            self.prev_scan_points = current_points
            self.prev_pose = self.pose.copy()
        else:
            dx = self.pose[0] - self.prev_pose[0]
            dy = self.pose[1] - self.prev_pose[1]
            dt = abs(normalize_angle(self.pose[2] - self.prev_pose[2]))
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > self.config.scan_match_keyframe_dist or dt > self.config.scan_match_keyframe_rot:
                self.prev_scan_points = current_points
                self.prev_pose = self.pose.copy()

        return {
            'pose': self.pose.copy(),
            'icp_score': icp_score,
            'corrected': corrected,
            'coverage': self.grid.get_coverage(),
            'n_points': n_points
        }

    def process_scan_localization_only(self, odom_pose, ranges, angles, max_range):
        current_points = ranges_to_points(ranges, angles, max_range)
        n_points = len(current_points)

        icp_score = 0.0
        corrected = False
        estimated_pose = odom_pose.copy()

        if n_points >= 10:
            map_points = self.grid.get_occupied_points_around(
                odom_pose, self.config.scan_match_map_radius
            )
            if len(map_points) >= 20:
                scan_world = transform_points(current_points, odom_pose)
                icp_pose, icp_score = self.icp.match(
                    source_points=scan_world,
                    target_points=map_points,
                    init_pose=np.array([0.0, 0.0, 0.0])
                )
                if icp_score > self.config.scan_match_min_score:
                    estimated_pose = odom_pose.copy()
                    estimated_pose[0] += icp_pose[0]
                    estimated_pose[1] += icp_pose[1]
                    estimated_pose[2] = normalize_angle(odom_pose[2] + icp_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

        self.pose = estimated_pose.copy()
        self.trajectory.append(self.pose.copy())
        self.scan_count += 1
        self.icp_scores.append(icp_score)

        return {
            'pose': self.pose.copy(),
            'icp_score': icp_score,
            'corrected': corrected,
            'n_points': n_points
        }

    def get_exploration_target(self):
        return self.explorer.get_exploration_target(self.pose, self.grid)

    def get_map_prob(self):
        return self.grid.get_occupancy_prob()

    def get_ros_map_data(self):
        return self.grid.get_ros_map_data()

    def get_trajectory(self):
        return np.array(self.trajectory) if self.trajectory else np.array([]).reshape(0, 3)

    def get_odom_trajectory(self):
        return self.odom.get_trajectory()

    def get_pose(self):
        return self.pose.copy()

    def get_coverage(self):
        return self.grid.get_coverage()

    def get_stats(self):
        avg_icp = np.mean(self.icp_scores) if self.icp_scores else 0.0
        return {
            'scan_count': self.scan_count,
            'corrections_applied': self.corrections_applied,
            'avg_icp_score': avg_icp,
            'coverage': self.grid.get_coverage(),
            'trajectory_length': len(self.trajectory)
        }

    def set_pose(self, pose):
        self.pose = np.array(pose, dtype=np.float64)
        self.odom.set_pose(self.pose)

    def reset(self):
        self.odom.reset()
        self.grid.reset()
        self.pose = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.prev_scan_points = None
        self.prev_pose = None
        self.scan_count = 0
        self.icp_scores = []
        self.corrections_applied = 0
        self.no_correction_count = 0
