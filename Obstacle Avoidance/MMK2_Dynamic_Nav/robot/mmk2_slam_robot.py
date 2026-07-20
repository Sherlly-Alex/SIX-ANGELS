import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lidar_path = os.path.join(_project_root, "submodules", "MuJoCo-LiDAR")
if _lidar_path not in sys.path:
    sys.path.insert(0, _lidar_path)

from discoverse.robots_env.mmk2_base import MMK2Base
from config.slam_config import SLAMConfig


class MMK2SlamRobot(MMK2Base):
    wheel_distance = 0.3265
    init_position = [-3.0, -2.0, 0.0]

    def __init__(self, config: SLAMConfig):
        self.config = config
        super().__init__(config)

        self.init_joint_pose[0] = self.init_position[0]
        self.init_joint_pose[1] = self.init_position[1]
        self.init_joint_pose[2] = self.init_position[2]

        if self.config.lidar_enabled:
            self._init_lidar()
        else:
            self.lidar_wrapper = None

        self.key_states = {
            'w': False, 's': False, 'a': False, 'd': False, 'shift': False
        }

    def _init_lidar(self):
        try:
            from mujoco_lidar.lidar_wrapper import MjLidarWrapper
            from mujoco_lidar.scan_gen import create_lidar_single_line

            self.rays_theta, self.rays_phi = create_lidar_single_line(
                horizontal_resolution=self.config.lidar_horizontal_resolution,
                horizontal_fov=self.config.lidar_horizontal_fov
            )

            print(f"[MMK2SlamRobot] LiDAR rays: {len(self.rays_theta)} points")

            self.lidar_wrapper = MjLidarWrapper(
                mj_model=self.mj_model,
                site_name=self.config.lidar_site_name,
                backend=self.config.lidar_backend,
                cutoff_dist=self.config.lidar_cutoff_dist,
                args={
                    'bodyexclude': self.mj_model.body("agv_link").id
                }
            )

            self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
            self.lidar_wrapper.get_hit_points()

            print(f"[MMK2SlamRobot] LiDAR initialized (backend={self.config.lidar_backend}, "
                  f"site={self.config.lidar_site_name})")

        except ImportError as e:
            print(f"[WARNING] Failed to initialize LiDAR: {e}")
            print("  Please ensure MuJoCo-LiDAR submodule is initialized:")
            print("  python scripts/setup_submodules.py --module lidar")
            self.lidar_wrapper = None

    def get_lidar_scan(self):
        if self.lidar_wrapper is None:
            return np.array([]), np.array([])

        self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)

        distances = self.lidar_wrapper.get_distances()

        ranges = distances.copy()
        ranges[ranges >= self.config.lidar_cutoff_dist] = np.inf

        angles = self.rays_theta.copy()

        return ranges, angles

    def get_lidar_points_2d(self):
        if self.lidar_wrapper is None:
            return np.array([]).reshape(0, 2)

        self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
        points_3d = self.lidar_wrapper.get_hit_points()

        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2)

        return points_3d[:, :2]

    def get_robot_pose_2d(self):
        agv_body = self.mj_data.body("agv_link")
        pos = agv_body.xpos.copy()
        quat_wxyz = agv_body.xquat.copy()

        rot = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
        euler = rot.as_euler('zyx')

        return np.array([pos[0], pos[1], euler[0]])

    def get_laser_pose_2d(self):
        laser_site = self.mj_data.site("laser")
        pos = laser_site.xpos.copy()
        mat = laser_site.xmat.reshape(3, 3)
        quat = Rotation.from_matrix(mat).as_quat()
        euler = Rotation.from_quat(quat).as_euler('zyx')

        return np.array([pos[0], pos[1], euler[0]])

    def get_wheel_positions(self):
        return float(self.sensor_wheel_qpos[0]), float(self.sensor_wheel_qpos[1])

    def apply_diff_drive(self, linear_vel, angular_vel):
        wheel_radius = self.wheel_radius
        wheel_distance = self.wheel_distance

        v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
        v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius

        self.mj_data.ctrl[0] = np.clip(v_left, -10.0, 10.0)
        self.mj_data.ctrl[1] = np.clip(v_right, -10.0, 10.0)

    def updateControlFromKeyboard(self, key_states):
        speed_factor = 3.0 if key_states.get('shift', False) else 1.0
        linear_speed = 0.5 * speed_factor
        angular_speed = 2.0 * speed_factor

        linear_vel = 0.0
        angular_vel = 0.0

        if key_states.get('w', False):
            linear_vel = linear_speed
        elif key_states.get('s', False):
            linear_vel = -linear_speed

        if key_states.get('a', False):
            angular_vel = angular_speed
        elif key_states.get('d', False):
            angular_vel = -angular_speed

        self.apply_diff_drive(linear_vel, angular_vel)

    def resetState(self):
        super().resetState()

    def printHelp(self):
        print("\n" + "=" * 60)
        print("       MMK2 SLAM 学习例程 - 键盘控制")
        print("=" * 60)
        print("\n=== 键盘控制 ===")
        print("W / S : 前进 / 后退")
        print("A / D : 左转 / 右转")
        print("Shift : 按住加速")
        print("ESC   : 切换到自由视角")
        print("[ / ] : 切换相机 (overview / top_down)")
        print("H     : 显示此帮助")
        print("=" * 60 + "\n")
