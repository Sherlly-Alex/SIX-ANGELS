# MMK2_Dynamic_Nav

基于 DISCOVERSE 仿真平台的 MMK2 机器人动态导航系统，在 MMK2_SLAM 基础上扩展了动态障碍物感知与在线重规划能力。

## 项目位置

本目录位于 DISCOVERSE 项目内：

DISCOVERSE-main/
├── discoverse/                     ← DISCOVERSE 仿真引擎核心
│   └── robots_env/
│       └── mmk2_base.py            ← MMK2 机器人基类（MMK2Cfg / MMK2Base）
├── examples/
│   ├── MMK2_SLAM/                  ← 上游项目：MMK2 2D SLAM 教学示例
│   ├── MMK2_Dynamic_Nav/           ← （本目录）MMK2 动态导航
│   └── tasks_mmk2/                 ← MMK2 操作任务示例
└── submodules/
    └── MuJoCo-LiDAR/               ← LiDAR 仿真模块

## 开发基础

本系统基于 **MMK2_SLAM** 项目构建，复用了其全部 SLAM 核心模块，并在此基础上新增了动态导航功能：

### 从 MMK2_SLAM 继承的模块

| 模块 | 来源 | 说明 |
|------|------|------|
| `config/slam_config.py` | 继承 | 机器人/SLAM 参数配置（继承 `MMK2Cfg`） |
| `core/*` | 复用 | odometry, scan_matching, occupancy_grid, slam_estimator, frontier_exploration |
| `robot/*` | 复用 | mmk2_slam_robot（继承 `MMK2Base`）, motion_controller |
| `scenes/*` | 复用 | MuJoCo 场景 XML（mmk2 模型 + 房间） |

### 本项目新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 动态感知 | `perception/dynamic_layer.py` | 动态层：检测并追踪非静态地图障碍物 |
| 规划网格 | `planning/planning_grid.py` | 融合静态地图 + 动态层的规划适配器 |
| 路径验证 | `planning/path_validator.py` | 线段碰撞检测，确认路径是否被阻塞 |
| 紧急制动 | `planning/emergency_checker.py` | 基于前向 LiDAR 的紧急停车检测 |
| 障碍物管理 | `sim/obstacle_manager.py` | 在 MuJoCo 场景中动态生成 mocap 障碍物 |
| 导航配置 | `config/dynamic_nav_config.py` | 动态导航专用参数 |
| 导航 GUI | `gui/dynamic_nav_viewer.py` | 6 面板 Matplotlib 可视化 |
| 状态覆盖 | `gui/nav_overlay.py` | 控制台状态摘要输出 |

## 项目结构

MMK2_Dynamic_Nav/
├── config/
│   ├── slam_config.py              # SLAM 参数（继承自 MMK2_SLAM）
│   └── dynamic_nav_config.py       # 动态导航参数
├── core/                           # SLAM 核心（复用自 MMK2_SLAM）
│   ├── slam_estimator.py           # 中央协调器
│   ├── odometry.py                 # 差速驱动轮式里程计
│   ├── scan_matching.py            # ICP 扫描匹配
│   ├── occupancy_grid.py           # 占据栅格地图
│   └── frontier_exploration.py     # 前沿探索 + A* 路径规划
├── robot/                          # 机器人层（复用自 MMK2_SLAM）
│   ├── mmk2_slam_robot.py          # 机器人封装（继承 MMK2Base）
│   └── motion_controller.py        # 运动控制器
├── perception/                     # （新增）动态感知
│   └── dynamic_layer.py            # 动态障碍物检测与追踪
├── planning/                       # （新增）规划与安全
│   ├── planning_grid.py            # 融合静态+动态地图的规划网格
│   ├── path_validator.py           # 路径阻塞检测
│   └── emergency_checker.py        # 前向 LiDAR 紧急制动
├── sim/                            # （新增）仿真管理
│   └── obstacle_manager.py         # 动态障碍物生成
├── gui/                            # （新）可视化
│   ├── dynamic_nav_viewer.py       # 6 面板 Matplotlib GUI
│   └── nav_overlay.py              # 控制台状态输出
├── scenes/                         # MuJoCo 场景
│   ├── dynamic_nav_mmk2.xml        # 动态导航场景（含 mocap 障碍物）
│   ├── slam_room_mmk2.xml          # 原始房间场景
│   ├── mmk2_slim.xml               # MMK2 机器人模型
│   └── mmk2_dependencies_slim.xml
├── tests/                          # （新增）单元测试
│   ├── test_dynamic_layer.py
│   ├── test_planning_grid.py
│   ├── test_path_validator.py
│   ├── test_emergency_checker.py
│   └── test_dynamic_nav_smoke.py
├── run_dynamic_nav.py              # 主入口：动态导航状态机
├── test_init.py                    # 基础导入测试
├── test_lidar_and_contact.py       # LiDAR + 接触传感器测试
├── test_mocap_obstacle.py          # mocap 障碍物测试
└── README.md                       # 本文件

## 系统架构

                ┌─────────────────────────────┐
                │      SLAM (MMK2_SLAM)        │
                │  odometry → ICP → occupancy  │
                └─────────────┬───────────────┘
                              │
                ┌─────────────▼───────────────┐
                │      Dynamic Layer           │
                │  Bresenham 射线 + 动态衰减   │
                └─────────────┬───────────────┘
                              │
                ┌─────────────▼───────────────┐
                │      Planning Grid           │
                │  静态地图 + 动态层 = 规划网格 │
                └─────────────┬───────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────▼────────┐ ┌───────▼───────┐ ┌─────────▼────────┐
│  Emergency Check  │ │ Path Validator│ │  A* Path Planner │
│  (前向 LiDAR)     │ │ (线段碰撞)    │ │  (frontier_expl) │
└─────────┬────────┘ └───────┬───────┘ └─────────┬────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                ┌─────────────▼───────────────┐
                │      State Machine           │
                │  MAPPING → FOLLOWING_PATH    │
                │  EMERGENCY_STOP → REPLANNING │
                │  WAITING_FOR_CLEARANCE       │
                │  GOAL_REACHED / SAFE_STOP    │
                └─────────────────────────────┘

## 状态机

MAPPING
   │ (建图 100 帧，覆盖静态场景)
   ▼
WAITING_FOR_GOAL
   │ (A* 规划初始路径)
   ▼
FOLLOWING_PATH  ◄──────── REPLANNING
   │    ▲                    │  ▲
   │    │ (重规划成功)       │  │
   │    └────────────────────┘  │
   │                            │
   ├── 紧急制动 ──► EMERGENCY_STOP
   │                    │ (等待)
   │                    └──────► REPLANNING
   │
   ├── 路径阻塞 ──► REPLANNING
   │                    │
   │                    ├── 成功 ──► FOLLOWING_PATH
   │                    └── 失败 ──► WAITING_FOR_CLEARANCE
   │                                      │
   │                    动态层更新 ────────┘
   │                    │ 超时 ──► SAFE_STOP
   │
   └── 到达目标 ──► GOAL_REACHED

## 快速开始

```bash
# 激活环境
conda activate discoverse

# 初始化 MuJoCo-LiDAR 子模块
git submodule update --init submodules/MuJoCo-LiDAR

# 运行动态导航（默认目标 (-0.5, -2.0)）
python run_dynamic_nav.py

# 无头模式 + 自定义目标
python run_dynamic_nav.py --headless --goal_x 1.0 --goal_y 1.0 --max_steps 500

# 运行单元测试
python -m pytest tests/ -v
命令行参数
参数	默认值
--goal_x	-0.5
--goal_y	-2.0
--headless	False
--max_steps	10000
--unlimited	False
--seed	None
--width	1600
--height	900
技术栈
- 仿真引擎：MuJoCo（通过 DISCOVERSE）
- LiDAR 仿真：MuJoCo-LiDAR 子模块
- SLAM 基础：MMK2_SLAM（轮式里程计 + ICP + 占据栅格地图）
- 动态感知：Bresenham 射线投射 + 时间衰减的动态层
- 路径规划：A*（8 邻域，膨胀代价）
- 可视化：Matplotlib（6 面板 GUI）+ 控制台状态输出
