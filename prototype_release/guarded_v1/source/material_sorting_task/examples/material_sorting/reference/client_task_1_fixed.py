#!/usr/bin/env python3
"""任务 baseline client（文件名由 DG-202612 文档规定）。

新任务(3 步搬运,一次一个,循环同一套双臂 hug 抓放例程)：
  1) 桌上 pink → 放到原圆柱所在的货架层
  2) 视觉识别 brown → 放到 pink 原来在桌面上的位置
  3) 桌上 white cube 顶部 yellow → 放到柜里白色长方体左边
  "左边" = 机器人视角(面向目标时的左手边)。抓取目标由 /material/detections
  视觉识别锁定；fixed_world 只保留为临时调试兜底。

坐标系：世界 +X 东 / +Y 北；base 系 X=前 Y=左 Z=上。第 1 步起点看桌面后动态抓 pink,
第 2 步粉色放完退到货架口看货架后抓 brown,第 3 步回起点看桌面后动态抓 yellow。
"""
import math
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray

from discoverse.utils import step_func
from mmk2_kdl import MMK2Kdl

# —— 抓取姿态矩阵（base 系，与参考 baseline 逐字节一致）——
LEFT_A_ROT = np.array([
    [0.99890619, 0.04294831, 0.01848963],
    [-0.0203026, 0.04216758, 0.99890425],
    [0.04212158, -0.99818703, 0.04299342],
])
RIGHT_A_ROT = np.array([
    [0.99890619, -0.04294831, 0.01848963],
    [0.0203026, 0.04216758, -0.99890425],
    [0.04212158, 0.99818703, 0.04299342],
])

# 双臂 hug 参考常量（base 系）
PRE_GRASP_FWD = 0.48240646     # 预抓取前伸
PRE_GRASP_LAT = 0.22475936     # 预抓取左右各半
PRE_GRASP_Z0 = 1.32163718      # 预抓取高度基准(减 slide)
GRASP_OFF = np.array([-0.05, 0.20, -0.01])   # 相对盒中心的前伸/高度偏移(横向半展改为每步 grip_half)
SQUEEZE = 0.10                 # 默认 hug 内收量(放置阶段用,具体任务可用 hold_half 覆盖)
OPEN_MARGIN = 0.10             # 张开位 = grip_half + 此余量(先在盒外张开再合拢)
GRIP_OPEN = 1.0                # 双臂 hug 全程夹爪保持张开(靠双臂挤压夹持)

BOX_HALF_DEPTH = 0.08          # 前表面检测点沿视线补偿半盒深 -> 盒中心(侧视货架盒用)
TOP_Z_OFFSET = 0.05            # 俯视桌面盒:检测点在顶面,下修到盒中心的高度补偿
PLACE_CLEARANCE = 0.12         # 放置:先把盒举到放置面上方这么高,再移动到底盘放置站位
PLACE_OBJ_FWD = 0.52           # held_center_base 未锁定时的兜底前向距离
PLACE_RELEASE_SPREAD = 0.04    # 参照 xmartev baseline:放下后左右臂小幅外展松开
SHELF_PLACE_CLEARANCE = 0.055  # 柜内举高余量必须低于上层板下沿
SHELF_RETREAT_BACK = 0.32      # 柜内放置后先直线倒出柜口,再收臂
TABLE_RETREAT_BACK = 0.28      # 桌面放置后也先后撤,避免桌边直接收臂扫到物体
PINK_SHELF_PLACE_X = -2.64    # pink 放浅一点,避免顶/怼到柜子深处
YELLOW_SHELF_PLACE_X = -2.55  # yellow 再放浅一点,减少进柜深度和手/货架干涉
SHELF_LEFT_CLEAR_Y = 0.585     # yellow 再往左一点,仍避开缩小后的长方体
USE_FIXED_OBJECT_POSES = False  # 抓取目标走 YOLO/视觉检测,不再吸附固定坐标
START_OBSERVE_XY = (-0.70, 0.55)
TABLE_OBSTACLE_X = -0.54       # 面向桌面时,小于该 x 约为左侧,大于该 x 约为右侧
INITIAL_OBSERVE_SLIDE = 0.0    # 桌面视觉判断用正常初始相机姿态,不额外抬升降/低头
INITIAL_OBSERVE_HEAD = (0.0, 0.0)
YAW_WEST = math.pi
YAW_NORTH = math.pi / 2.0
END_XY = np.array([-0.70, 0.55])           # 结束区

# —— 任务表(一次一个,循环同一套 pick-place)。3 步 ——
#   place_world = 放置目标; look_pitch = 抓取时低头看盒的俯仰
TASKS = [
    {
        "name": "pink->shelf_L3",
        "color": "pink",
        "grasp_height": 0.865,                 # 桌面 pink 视觉锁定高度再抬,避免夹爪碰桌面
        "fixed_world": (-1.00, 2.20, 0.834),   # 固定坐标: pink 初始中心
        "pick_view": "top",                    # 俯视桌面盒:检测点在顶面,不做水平视线补偿
        "top_z_offset": 0.095,                 # pink 检测点接近盒顶面,下修半高到中心
        "observe_stand": START_OBSERVE_XY,     # 第一次视觉判断:起点正常初始姿态看完整桌面
        "observe_yaw": YAW_NORTH,
        "observe_initial_posture": True,
        "dynamic_table_pick": True,
        "table_pick_standoff": 0.54,
        "table_pick_x_range": (-1.35, 0.18),
        "table_pick_y_range": (1.55, 1.82),
        "pick_stand": (-1.00, 1.66), "pick_yaw": YAW_NORTH, "look_pitch": -0.50,
        "pick_creep": (-1.00, 1.66),           # 桌子挡底盘,不 creep;停桌边直接伸臂抓(臂够 ~0.57m)
        "grip_half": 0.13,                     # pink 桌面横放后朝北夹 0.24 长边,半宽放大
        "hold_half": 0.115,                    # 横放后搬运半宽按长边设置,避免过度挤压
        "pre_grasp_fwd": 0.065,                # 视觉点略偏外,抓取中心往桌面内侧推进一点
        "grasp_fwd": 0.065,
        "grasp_z": 0.045,                      # 抬高抓取中心,避免夹爪/指根刮桌面
        "place_yaw": YAW_WEST,
        "place_world": (PINK_SHELF_PLACE_X, 0.778, 1.145),  # 原圆柱 L3 层,略放深但避开背板
        "place_clearance": SHELF_PLACE_CLEARANCE,
        "place_obj_x": PLACE_OBJ_FWD,
        "retreat_yaw": YAW_NORTH,              # 抓后朝北向南倒退,再转向柜子
    },
    {
        "name": "brown->pink_table_spot",
        "color": "brown",
        "grasp_height": 0.837,                 # 柜里 brown(L2)中心高度
        "fixed_world": (-2.63, 0.778, 0.837),  # 仅作临时调试兜底;默认不用
        "pick_view": "front",                  # 第二次视觉判断:粉色放完退出来后看完整货架
        "pick_stand": (-1.75, 0.778), "pick_yaw": YAW_WEST, "look_pitch": -0.25,
        "pick_creep": (-1.90, 0.778),          # 手臂抬高后到柜口附近即停,不再硬顶货架
        "creep_tol": 0.05,
        "grip_half": 0.12,                     # 朝西夹 0.24 长边,收窄到接触盒面而不是停在盒外
        "hold_half": 0.100,                    # 更紧一点,避免只由指尖带住 brown
        "slide_z_offset": 0.01,                # brown 抓取再往下,但仍避开层板
        "pre_grasp_fwd": -0.15,                # 预抓取稍微更深,避免只夹前端
        "grasp_fwd": 0.050,                    # 合拢点再往柜内推进一点,提高 brown 抓取稳定性
        "grasp_z": 0.000,                      # 抓取点再往下,增加 brown 夹持接触面积
        "place_yaw": YAW_NORTH,
        "place_world": (-1.00, 2.20, 0.834),   # pink 原桌面位置
        "place_obj_x": PLACE_OBJ_FWD,
        "release_spread": 0.105,               # brown 抓得紧,桌面释放手再放开一点
        "release_delay": 1.3,
        "place_retreat_back": 0.18,            # 脱离后小退即可,避免后退过程重新带起盒子
        "retreat_yaw": YAW_WEST,               # 抓后朝西向东倒退,再转向桌子
    },
    {
        "name": "yellow->packaging_box_left",
        "color": "yellow",
        "grasp_height": 1.004,                 # yellow 叠在白色正方体顶面
        "fixed_world": (-0.54, 2.30, 1.004),   # 固定坐标: yellow 初始中心
        "pick_view": "top",
        "top_z_offset": 0.05,                  # yellow 在白色正方体上,可见面检测点已低于真实顶面
        "observe_stand": START_OBSERVE_XY,     # 第三次视觉判断:brown 放完后回起点正常初始姿态看桌面
        "observe_yaw": YAW_NORTH,
        "observe_initial_posture": True,
        "dynamic_table_pick": True,
        "table_pick_standoff": 0.56,
        "table_pick_x_range": (-1.35, 0.18),
        "table_pick_y_range": (1.55, 1.82),
        "pick_stand": (-0.54, 1.74), "pick_yaw": YAW_NORTH, "look_pitch": -0.45,
        "pick_creep": (-0.54, 1.74),
        "grip_half": 0.10,
        "hold_half": 0.090,                    # yellow 在高处,略压紧提高带起稳定性
        "pre_grasp_fwd": 0.085,
        "grasp_fwd": 0.085,                    # yellow 抓取再明显往桌面内侧/深处一点
        "grasp_z": -0.02,
        "place_yaw": YAW_WEST,
        "place_world": (YELLOW_SHELF_PLACE_X, SHELF_LEFT_CLEAR_Y, 0.545),  # 抬高释放,避免腕部压到 L1 层板
        "place_clearance": SHELF_PLACE_CLEARANCE,
        "place_nav_tol": 0.10,                 # 到货架口物理极限后直接落下,不继续硬顶
        "place_push_sequence": "release_back_hug_push",
        "place_push_back": 0.12,               # 放下并释放后先退出来一点,给推进动作留距离
        "place_push_forward": 0.09,            # 保持小角度 hug 后再向前走,把 yellow 推进去
        "place_push_hug_half": 0.065,          # hug 角度小一点,但不收太多避免 IK 无解
        "place_push_tol": 0.025,
        "release_spread": 0.045,               # yellow 需要充分外展脱离盒面,避免释放后带走
        "place_obj_x": PLACE_OBJ_FWD,
        "retreat_yaw": YAW_NORTH,
    },
]


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TaskPickPlaceClient(Node):
    """3 步搬运 baseline：一次一个目标,循环同一套双臂 hug 抓 + 放。"""

    def __init__(self):
        super().__init__("client_task_1")
        self.kdl = MMK2Kdl()

        # 19-dim 控制向量 tc
        self.tc = np.zeros(19)
        self.tc[11] = GRIP_OPEN
        self.tc[18] = GRIP_OPEN
        self.action = self.tc.copy()
        self.joint_move_ratio = np.ones(19)

        # feedback
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None
        self.det_buf = deque(maxlen=60)
        self.box_world = None
        self.tgt_l = None
        self.tgt_r = None
        self.held_center_base = None
        self.reverse_target = None       # (axis, sign, limit) 直线倒退目标
        self.creep_target = None         # (axis, sign, limit) 直线前进(creep)目标,保持朝向不转向
        self.pick_lock_nav_set = False   # 视觉锁定后是否已设置动态抓取站位导航

        # task loop
        self.task_idx = 0

        # nav / ramp
        self.rate_hz = 24.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin, self.max_ang = 0.45, 1.2
        self.max_lin_acc, self.max_ang_acc = 0.8, 5.0
        self.cur_lin = self.cur_ang = 0.0
        self.des_lin = self.des_ang = 0.0
        self.nav_target = None
        self.nav_done = True
        self.nav_pos_tol = 0.06

        # state machine
        self.state = -1
        self.state_entered = False
        self.delay_until = 0.0

        # io
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Detection3DArray, "/material/detections", self.det_cb, 10)

        self.timer = self.create_timer(self.dt, self.tick)
        self.last_log = 0.0
        self.get_logger().info("client_task_1 (3-step pick-place) up; waiting for odom + joint_states ...")

    @property
    def task(self):
        return TASKS[self.task_idx]

    # ---- callbacks ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}

    def det_cb(self, msg):
        if USE_FIXED_OBJECT_POSES:
            return
        if self.state != 2:
            return
        if self.box_world is not None:
            return
        want = self.task["color"]
        for det in msg.detections:
            if not det.results:
                continue
            if str(det.results[0].hypothesis.class_id) != want:   # 只收当前步的目标颜色
                continue
            pos = det.results[0].pose.pose.position
            self.det_buf.append(np.array([pos.x, pos.y, pos.z]))

    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def larm_meas(self):
        return np.array([self.jpos.get(f"left_arm_joint{i+1}", self.tc[5 + i]) for i in range(6)])

    @property
    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i+1}", self.tc[12 + i]) for i in range(6)])

    # ---- frames ----
    def world_to_base(self, p_world):
        d = np.array(p_world, float) - np.array([self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def set_arm(self, target_base, arm, a_rot):
        T = np.eye(4)
        T[:3, :3] = a_rot
        T[:3, 3] = np.asarray(target_base, float)
        q_ref = self.larm_meas if arm == "l" else self.rarm_meas
        ref = np.zeros(7)
        ref[0] = float(self.tc[2])
        ref[1:] = q_ref
        if arm == "l":
            sols = self.kdl.inverse_kinematics(T_left=T, T_right=None, ref_pos=ref, target_height=float(self.tc[2]))
            if sols:
                self.tc[5:11] = np.asarray(sols[0])[1:7]
                return True
        else:
            sols = self.kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref, target_height=float(self.tc[2]))
            if sols:
                self.tc[12:18] = np.asarray(sols[0])[1:7]
                return True
        self.get_logger().warn(f"IK fail arm={arm} tgt={np.round(target_base,3)}")
        return False

    def set_both_arms(self, center_base, lat):
        """双臂对称设到 center_base ± 横向 lat。"""
        self.tgt_l = center_base + np.array([0.0, lat, 0.0])
        self.tgt_r = center_base + np.array([0.0, -lat, 0.0])
        self.set_arm(self.tgt_l, "l", LEFT_A_ROT)
        self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)

    def hold_half(self):
        """当前任务搬运/放置阶段的双臂半夹距。"""
        return float(self.task.get("hold_half", SQUEEZE))

    def place_obj_x(self):
        return float(self.task.get("place_obj_x", PLACE_OBJ_FWD))

    def grasp_offset(self, key="grasp_fwd"):
        return np.array([
            float(self.task.get(key, self.task.get("grasp_fwd", GRASP_OFF[0]))),
            0.0,
            float(self.task.get("grasp_z", GRASP_OFF[2])),
        ])

    def place_stand_from_goal(self):
        """让放置目标落到当前抱持盒子的 base 系位置,放置阶段不重算抱持姿态。"""
        pw = np.array(self.task["place_world"], dtype=float)
        yaw = self.task["place_yaw"]
        held_xy = (np.array([self.place_obj_x(), 0.0], dtype=float)
                   if self.held_center_base is None else self.held_center_base[:2])
        c, s = math.cos(yaw), math.sin(yaw)
        held_world = np.array([c * held_xy[0] - s * held_xy[1],
                               s * held_xy[0] + c * held_xy[1]])
        return (float(pw[0] - held_world[0]), float(pw[1] - held_world[1]))

    def set_slide_keep_hold(self, target_slide):
        """只动升降,保持双臂关节和抱持姿态;同步更新估计的盒/末端 base 系 z。"""
        new_slide = float(np.clip(target_slide, -0.04, 0.87))
        dz = float(self.tc[2] - new_slide)
        self.tc[2] = new_slide
        if self.tgt_l is not None:
            self.tgt_l = self.tgt_l + np.array([0.0, 0.0, dz])
        if self.tgt_r is not None:
            self.tgt_r = self.tgt_r + np.array([0.0, 0.0, dz])
        if self.held_center_base is not None:
            self.held_center_base = self.held_center_base + np.array([0.0, 0.0, dz])

    def slide_for_held_z(self, z_world):
        if self.held_center_base is None:
            return PRE_GRASP_Z0 - z_world
        return self.tc[2] + float(self.held_center_base[2] - z_world)

    # ---- nav ----
    def set_twist(self, lin, ang):
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def ramp(self):
        dl = np.clip(self.des_lin - self.cur_lin, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -self.max_ang_acc * self.dt, self.max_ang_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def drive_nav(self):
        if self.nav_target is None:
            self.set_twist(0.0, 0.0)
            return
        tx, ty, tyaw = self.nav_target
        d = np.array([tx, ty]) - self.base_xy
        dist = float(np.linalg.norm(d))
        if dist > self.nav_pos_tol:
            bearing = math.atan2(d[1], d[0])
            yaw_err = wrap_to_pi(bearing - self.base_yaw)
            if abs(yaw_err) > 0.10:
                self.set_twist(0.0, 2.0 * yaw_err)
            else:
                self.set_twist(min(0.9 * dist, self.max_lin) * max(0.0, math.cos(yaw_err)), 1.5 * yaw_err)
            self.nav_done = False
        else:
            yaw_err = wrap_to_pi(tyaw - self.base_yaw)
            if abs(yaw_err) > 0.03:
                self.set_twist(0.0, 1.6 * yaw_err)
                self.nav_done = False
            else:
                self.set_twist(0.0, 0.0)
                self.nav_done = True

    def do_reverse(self):
        """沿当前朝向直线倒退(负线速度),到达 limit 停。reverse_target=(axis,sign,limit)。"""
        if len(self.reverse_target) == 4:
            axis, sign, limit, yaw_ref = self.reverse_target
        else:
            axis, sign, limit = self.reverse_target
            yaw_ref = self.task["retreat_yaw"]
        yaw_err = wrap_to_pi(yaw_ref - self.base_yaw)
        cur = self.base_xy[axis]
        if (sign > 0 and cur < limit) or (sign < 0 and cur > limit):
            self.set_twist(-0.35, 1.0 * yaw_err)
            self.nav_done = False
        else:
            self.set_twist(0.0, 0.0)
            self.nav_done = True

    def reverse_target_for_yaw(self, yaw, distance):
        """构造按当前朝向直线倒退的目标,用于抓取/放置后的安全退出。"""
        if abs(wrap_to_pi(yaw - YAW_NORTH)) < 0.10:
            return (1, -1, self.base_xy[1] - distance, yaw)
        if abs(wrap_to_pi(yaw - YAW_WEST)) < 0.10:
            return (0, +1, self.base_xy[0] + distance, yaw)
        raise ValueError(f"unsupported reverse yaw: {yaw}")

    def do_creep(self):
        """保持抓取朝向直线前进逼近盒子,只做微小 yaw 保持,不为对准航点大转向。
        接近目标时按剩余距离减速,避免冲过头撞货架。creep_target=(axis,sign,limit)。"""
        if len(self.creep_target) >= 4:
            axis, sign, limit, yaw_ref = self.creep_target[:4]
        else:
            axis, sign, limit = self.creep_target
            yaw_ref = self.task["pick_yaw"]
        yaw_err = wrap_to_pi(yaw_ref - self.base_yaw)
        cur = self.base_xy[axis]
        remain = (limit - cur) * sign            # 剩余前进距离(>0 未到)
        done_tol = float(self.creep_target[4] if len(self.creep_target) >= 5 else self.task.get("creep_tol", 0.015))
        if remain > done_tol:
            v = float(np.clip(0.8 * remain, 0.05, 0.25))   # 按剩余距离减速,最高 0.25
            self.set_twist(v, 1.0 * yaw_err)
            self.nav_done = False
        else:
            self.set_twist(0.0, 0.0)
            self.nav_done = True

    def configure_dynamic_pick_after_lock(self):
        """桌面随机目标:先在观察位锁定,再按目标坐标生成桌边抓取站位。"""
        if self.pick_lock_nav_set or not self.task.get("dynamic_table_pick") or self.box_world is None:
            return
        t = self.task
        standoff = float(t.get("table_pick_standoff", 0.56))
        x_min, x_max = t.get("table_pick_x_range", (-1.35, 0.18))
        y_min, y_max = t.get("table_pick_y_range", (1.55, 1.82))
        px = float(np.clip(self.box_world[0], x_min, x_max))
        py = float(np.clip(self.box_world[1] - standoff, y_min, y_max))
        t["pick_stand"] = (px, py)
        t["pick_creep"] = (px, py)
        t["pick_yaw"] = YAW_NORTH
        t["retreat_yaw"] = YAW_NORTH
        self.nav_target = (px, py, YAW_NORTH)
        self.nav_pos_tol = float(t.get("dynamic_pick_nav_tol", 0.05))
        self.nav_done = False
        self.pick_lock_nav_set = True
        side = "left" if self.box_world[0] < TABLE_OBSTACLE_X else "right"
        self.get_logger().info(
            f"[vision-pick] {t['color']} center={np.round(self.box_world,3)} "
            f"table_side={side} -> pick_stand=({px:.2f},{py:.2f}) yaw=north")

    # ---- box selection ----
    def lock_box(self):
        """纯视觉锁定目标盒中心。挑最接近本步 grasp_height 的检测簇。
        - front(侧视货架盒):检测点在前表面,沿"机器人->点"视线补偿半盒深得中心。
        - top(俯视桌面盒):检测点在顶面,x/y≈盒中心 x/y(不做水平补偿),z 下修到盒中心。"""
        if USE_FIXED_OBJECT_POSES:
            self.box_world = np.array(self.task["fixed_world"], dtype=float)
            self.get_logger().info(
                f"[fixed] {self.task['color']} center={np.round(self.box_world,3)}")
            return True
        if len(self.det_buf) < 4:
            return False
        gh = self.task["grasp_height"]
        view = self.task.get("pick_view", "front")
        pts = np.array(self.det_buf)
        order = np.argsort(np.abs(pts[:, 2] - gh))
        best = pts[order[0]]
        near = pts[np.linalg.norm(pts - best, axis=1) < 0.15]
        det = np.median(near, axis=0)
        if view == "top":
            self.box_world = np.array([det[0], det[1], det[2] - self.task.get("top_z_offset", TOP_Z_OFFSET)])
        else:
            v = det[:2] - self.base_xy
            v = v / (np.linalg.norm(v) + 1e-9)
            center_xy = det[:2] + v * BOX_HALF_DEPTH
            self.box_world = np.array([center_xy[0], center_xy[1], det[2]])
        self.get_logger().info(
            f"[perception] {self.task['color']}({view}) det={np.round(det,3)} -> center={np.round(self.box_world,3)} (n={len(near)})")
        return True

    # ---- publish ----
    def smooth_and_publish(self):
        dif = np.abs(self.action - self.tc)
        self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
        self.joint_move_ratio[2] *= 0.3
        step = 1.2 * self.dt
        for i in range(2, 19):
            self.action[i] = step_func(self.action[i], self.tc[i], self.joint_move_ratio[i] * step)
        tw = Twist()
        tw.linear.x = float(self.tc[0])
        tw.angular.z = float(self.tc[1])
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    def arm_converged(self):
        return (np.max(np.abs(self.action[2:19] - self.tc[2:19])) < 0.02)

    # ---- state machine ----
    def enter(self, s):
        self.state = s
        self.state_entered = True
        self.delay_until = 0.0
        self.reverse_target = None
        self.creep_target = None
        self.nav_pos_tol = 0.06
        self.state_t0 = self.now()
        if s == 2:
            self.det_buf.clear()
            self.pick_lock_nav_set = False

    def state_done(self):
        if self.state == 2 and self.box_world is None:
            return False
        if self.now() - self.state_t0 > 30.0:
            return True
        if self.reverse_target is not None and not self.nav_done:
            return False
        if self.creep_target is not None and not self.nav_done:
            return False
        if self.nav_target is not None and not self.nav_done:
            return False
        if not self.arm_converged():
            return False
        if self.now() < self.delay_until:
            return False
        return True

    def start_next_task(self):
        """本步完成 -> 清状态,进入下一步(或全部完成回结束区)。"""
        self.box_world = None
        self.det_buf.clear()
        self.tgt_l = self.tgt_r = None
        self.held_center_base = None
        if self.task_idx + 1 < len(TASKS):
            self.task_idx += 1
            self.get_logger().info(f"==== 进入第 {self.task_idx+1} 步: {self.task['name']} ====")
            self.enter(0)
        else:
            self.enter(90)     # 全部完成 -> 回结束区

    def tick(self):
        if self.base_xy is None or self.jpos is None:
            return
        if self.state == -1:
            self.get_logger().info(f"==== 第 1 步: {self.task['name']} ====")
            self.enter(0)
        s = self.state
        t = self.task
        if self.state_entered:
            self.state_entered = False
            # —— pick 阶段 ——
            if s == 0:     # 导航到抓取站位
                stand = t.get("observe_stand", t["pick_stand"])
                yaw = t.get("observe_yaw", t["pick_yaw"])
                self.nav_target = (stand[0], stand[1], yaw); self.nav_done = False
            elif s == 1:   # 升降/低头看盒(俯仰按当前步)
                if t.get("observe_initial_posture"):
                    self.tc[2] = INITIAL_OBSERVE_SLIDE
                    self.tc[3] = INITIAL_OBSERVE_HEAD[0]
                    self.tc[4] = INITIAL_OBSERVE_HEAD[1]
                else:
                    self.tc[2] = float(t.get("observe_slide", 0.15))
                    self.tc[3] = 0.0
                    self.tc[4] = t.get("observe_look_pitch", t["look_pitch"])
            elif s == 2:   # 锁定目标盒
                self.delay_until = self.now() + 2.0
            elif s == 3:   # 预抓取(双臂张开,抬到盒高)
                z = self.box_world[2] + float(t.get("slide_z_offset", 0.0))
                self.tc[2] = float(np.clip(PRE_GRASP_Z0 - z, -0.04, 0.87))
                self.tgt_l = np.array([PRE_GRASP_FWD, PRE_GRASP_LAT, PRE_GRASP_Z0 - self.tc[2]])
                self.tgt_r = np.array([PRE_GRASP_FWD, -PRE_GRASP_LAT, PRE_GRASP_Z0 - self.tc[2]])
                self.set_arm(self.tgt_l, "l", LEFT_A_ROT); self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                self.tc[11] = GRIP_OPEN; self.tc[18] = GRIP_OPEN
            elif s == 4:   # 双臂移到盒两侧(张开 hug)
                b = self.world_to_base(self.box_world)
                gh = t["grip_half"]                             # 该步夹持半宽(按被夹面尺寸)
                self.set_both_arms(b + self.grasp_offset("pre_grasp_fwd"), gh + OPEN_MARGIN)
            elif s == 5:   # base 直线 creep 逼近盒(保持抓取朝向,不转向,避免打歪已对齐的双臂)
                cx, cy = t["pick_creep"]
                if t["pick_yaw"] == YAW_NORTH:            # 朝北:沿 y 前进(y 增大)
                    self.creep_target = (1, +1, cy)
                else:                                     # 朝西:沿 x 前进(x 减小)
                    self.creep_target = (0, -1, cx)
                self.nav_target = None; self.nav_done = False
            elif s == 6:   # creep 后按当前底盘位置重新瞄准盒两侧,合拢到 grip_half 夹住 + 稳定
                b = self.world_to_base(self.box_world)          # 重新计算(底盘已 creep 前移)
                gh = self.hold_half()
                grasp_center = b + self.grasp_offset()
                self.held_center_base = b.copy()                # 后续放置站位按盒真实中心计算
                self.tgt_l = grasp_center + np.array([0.0, gh, 0.0])
                self.tgt_r = grasp_center + np.array([0.0, -gh, 0.0])
                self.set_arm(self.tgt_l, "l", LEFT_A_ROT); self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                self.delay_until = self.now() + 2.0
            elif s == 7:   # 抬起:只动升降,保持抓住时的双臂姿态,避免 IK 换解
                self.set_slide_keep_hold(self.tc[2] - 0.10)
            elif s == 8:   # 直线倒退离开抓取处(保持朝向)
                self.reverse_target = self.reverse_target_for_yaw(t["retreat_yaw"], 0.35)
                self.nav_target = None; self.nav_done = False
            # —— 搬运 + place 阶段 ——
            elif s == 9:   # 中转:到放置站位"正后方"(沿放置朝向反方向偏移),提前对准放置朝向
                px, py = self.place_stand_from_goal(); yaw = t["place_yaw"]
                self.nav_target = (px - 0.5 * math.cos(yaw), py - 0.5 * math.sin(yaw), yaw)
                self.nav_done = False
            elif s == 10:  # 举高:只动升降,保持抱持姿态;底盘先停在中转点
                pw = np.array(t["place_world"], float)
                self.set_slide_keep_hold(self.slide_for_held_z(pw[2] + t.get("place_clearance", PLACE_CLEARANCE)))
                self.nav_target = None
            elif s == 11:  # 底盘直线开到放置站位(盒已举高+收回,不会低位插进货架)
                px, py = self.place_stand_from_goal()
                self.nav_pos_tol = float(t.get("place_nav_tol", 0.06))
                self.nav_target = (px, py, t["place_yaw"]); self.nav_done = False
            elif s == 12:  # 到位稳定:盒还在高位,底盘锁死
                self.nav_target = None
                self.delay_until = self.now() + 0.4
            elif s == 13:  # 先放好:只动升降把盒落到放置面,不重新求手臂 IK
                pw = np.array(t["place_world"], float)
                self.set_slide_keep_hold(self.slide_for_held_z(pw[2]))
                self.nav_target = None
                self.delay_until = self.now() + 1.0
            elif s == 14:  # 放好后先退出一点,给后续推进留距离
                back = float(t.get("place_push_back", 0.0))
                if back > 0.0:
                    self.reverse_target = self.reverse_target_for_yaw(t["place_yaw"], back)
                    self.nav_target = None; self.nav_done = False
                else:
                    self.nav_target = None
                    self.delay_until = self.now() + 0.2
            elif s == 15:  # 退出后把 hug 角度小幅收紧,不要收太小以免 IK 无解
                if t.get("place_push_hug_half") is not None and self.held_center_base is not None:
                    half = float(t.get("place_push_hug_half", self.hold_half()))
                    center = (self.world_to_base(np.array(t["place_world"], float))
                              if t.get("place_push_sequence") == "release_back_hug_push"
                              else self.held_center_base)
                    self.held_center_base = center.copy()
                    self.tgt_l = center + np.array([0.0, half, 0.0])
                    self.tgt_r = center + np.array([0.0, -half, 0.0])
                    self.set_arm(self.tgt_l, "l", LEFT_A_ROT); self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                self.nav_target = None
                self.delay_until = self.now() + 0.4
            elif s == 16:  # 保持这个 hug 角度,底盘往前走把盒推入货架
                fwd = float(t.get("place_push_forward", 0.0))
                if fwd > 0.0:
                    if t["place_yaw"] == YAW_NORTH:
                        self.creep_target = (1, +1, self.base_xy[1] + fwd, t["place_yaw"], float(t.get("place_push_tol", 0.025)))
                    else:
                        self.creep_target = (0, -1, self.base_xy[0] - fwd, t["place_yaw"], float(t.get("place_push_tol", 0.025)))
                    self.nav_target = None; self.nav_done = False
                else:
                    self.nav_target = None
                    self.delay_until = self.now() + 0.2
            elif s == 17:  # 松开:双臂小幅外展脱离盒面,不再抬升拖动物体
                spread = float(t.get("release_spread", PLACE_RELEASE_SPREAD))
                self.tgt_l = self.tgt_l + np.array([0.0, spread, 0.0])
                self.tgt_r = self.tgt_r + np.array([0.0, -spread, 0.0])
                self.set_arm(self.tgt_l, "l", LEFT_A_ROT); self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                self.nav_target = None
                self.delay_until = self.now() + float(t.get("release_delay", 0.8))
            elif s == 18:  # 放置后手臂保持释放姿态,底盘先直线后撤离开柜口/桌边
                if t.get("place_push_sequence") == "release_back_hug_push":
                    spread = float(t.get("release_spread", PLACE_RELEASE_SPREAD))
                    self.tgt_l = self.tgt_l + np.array([0.0, spread, 0.0])
                    self.tgt_r = self.tgt_r + np.array([0.0, -spread, 0.0])
                    self.set_arm(self.tgt_l, "l", LEFT_A_ROT); self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                back = t.get("place_retreat_back", SHELF_RETREAT_BACK if t["place_yaw"] == YAW_WEST else TABLE_RETREAT_BACK)
                self.reverse_target = self.reverse_target_for_yaw(t["place_yaw"], back)
                self.nav_target = None; self.nav_done = False
            elif s == 19:  # 后撤到安全距离后再收臂 + 升降复位,避免近环境大幅 IK/关节跳变
                self.tc[5:11] = 0.0; self.tc[12:18] = 0.0; self.tc[2] = 0.1
                self.nav_target = None
                self.delay_until = self.now() + 0.5
            # —— 收尾 ——
            elif s == 90:  # 全部完成回结束区(此时手已完全收回)
                self.tc[5:11] = 0.0; self.tc[12:18] = 0.0; self.tc[2] = 0.0
                self.nav_target = (END_XY[0], END_XY[1], YAW_NORTH); self.nav_done = False

        if s == 2 and self.box_world is None:
            if self.lock_box():
                self.delay_until = self.now() + 0.5
        if s == 2 and self.box_world is not None:
            self.configure_dynamic_pick_after_lock()

        if self.state_done():
            if s == 19:
                self.start_next_task()
            elif t.get("place_push_sequence") == "release_back_hug_push" and s in {13, 16, 17}:
                self.enter({13: 17, 17: 14, 16: 18}[s])
            elif s < 19:
                self.enter(s + 1)
            elif s == 90:
                self.set_twist(0.0, 0.0)

        if self.reverse_target is not None:
            self.do_reverse()          # 直线倒退(不转向)
        elif self.creep_target is not None:
            self.do_creep()            # 直线前进 creep(保持抓取朝向,不打歪双臂)
        else:
            self.drive_nav()
        self.ramp()
        self.smooth_and_publish()
        if self.now() - self.last_log > 1.5:
            self.get_logger().info(
                f"task{self.task_idx+1} state={self.state} base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} navdone={self.nav_done} "
                f"box={None if self.box_world is None else np.round(self.box_world,2)}")
            self.last_log = self.now()


def main():
    rclpy.init()
    node = TaskPickPlaceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
