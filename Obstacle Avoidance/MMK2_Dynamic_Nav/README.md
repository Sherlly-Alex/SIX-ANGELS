# MMK2_Dynamic_Nav

基于 DISCOVERSE 仿真平台的 MMK2 机器人动态导航系统，在 MMK2_SLAM 基础上扩展了动态障碍物感知、在线 A* 重规划、以及 DWB-inspired 局部控制器。

## 项目位置

本目录位于 DISCOVERSE 项目内：

```
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
```

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
| 动态感知 | `perception/dynamic_layer.py` | 动态层：Bresenham 射线 + 命中/空闲确认 + 时间衰减 |
| 规划网格 | `planning/planning_grid.py` | 融合静态地图 + 动态层的规划适配器 |
| 路径验证 | `planning/path_validator.py` | 线段级膨胀代价碰撞检测 |
| 紧急制动 | `planning/emergency_checker.py` | 基于前向 LiDAR 的紧急停车检测 |
| 局部目标 | `planning/local_goal_selector.py` | 弧长累积法选取 A* 路径前方局部目标 |
| DWB 控制器 | `planning/dwb/` | DWB-inspired 局部控制器（轨迹生成 + Critic 评分） |
| DWB Critics | `planning/dwb/critics/` | 障碍/路径偏离/目标距离/前进偏好 四个评分器 |
| 障碍物管理 | `sim/obstacle_manager.py` | 动态 mocap 障碍物生成 + 脚本运动模式 |
| 导航配置 | `config/dynamic_nav_config.py` | 动态导航 + DWB 专用参数 |
| 导航 GUI | `gui/dynamic_nav_viewer.py` | 4 面板 Matplotlib 可视化（含 DWB 轨迹+候选束+critic 分数） |
| 状态覆盖 | `gui/nav_overlay.py` | 控制台状态摘要输出 |

## 项目结构

```
MMK2_Dynamic_Nav/
├── config/
│   ├── slam_config.py              # SLAM 参数（继承自 MMK2_SLAM）
│   └── dynamic_nav_config.py       # 动态导航 + DWB 参数
├── core/                           # SLAM 核心（复用自 MMK2_SLAM）
│   ├── slam_estimator.py
│   ├── odometry.py
│   ├── scan_matching.py
│   ├── occupancy_grid.py
│   └── frontier_exploration.py     # A* 路径规划
├── robot/                          # 机器人层（复用自 MMK2_SLAM）
│   ├── mmk2_slam_robot.py
│   └── motion_controller.py
├── perception/
│   └── dynamic_layer.py            # 动态障碍物检测
├── planning/
│   ├── planning_grid.py            # 融合规划网格
│   ├── path_validator.py           # 路径阻塞检测
│   ├── emergency_checker.py        # 紧急制动
│   ├── local_goal_selector.py      # 局部目标选择
│   └── dwb/                        # DWB 局部控制器
│       ├── trajectory.py           # RobotState / Trajectory / DWBResult
│       ├── trajectory_generator.py # 动态窗口 + 轨迹预测
│       ├── critic.py               # TrajectoryCritic 抽象接口
│       ├── controller.py           # 评分调度 + 最优选择
│       └── critics/
│           ├── obstacle_critic.py      # 障碍碰撞（硬拒绝 + 软膨胀代价）
│           ├── path_dist_critic.py     # 全局路径偏离代价
│           ├── goal_dist_critic.py     # 局部目标距离代价
│           └── prefer_forward_critic.py # 前进速度偏好代价
├── sim/
│   └── obstacle_manager.py         # mocap 障碍物 + 脚本运动
├── gui/
│   ├── dynamic_nav_viewer.py       # 4 面板 Matplotlib + DWB 可视化
│   └── nav_overlay.py              # 控制台状态输出
├── scenes/
│   ├── dynamic_nav_mmk2.xml        # 动态导航场景（含 mocap 障碍物）
│   ├── slam_room_mmk2.xml
│   ├── mmk2_slim.xml
│   └── mmk2_dependencies_slim.xml
├── tests/
│   ├── test_dynamic_layer.py
│   ├── test_planning_grid.py
│   ├── test_path_validator.py
│   ├── test_emergency_checker.py
│   ├── test_dynamic_nav_smoke.py
│   ├── test_dwb_trajectory_generator.py
│   ├── test_dwb_controller_core.py
│   ├── test_dwb_critics.py
│   ├── test_dwb_integration.py
│   ├── test_dwb_viewer_metrics.py
│   ├── test_local_goal_selector.py
│   ├── test_moving_obstacle.py
│   ├── test_lidar_and_contact.py
│   └── test_mocap_obstacle.py
├── run_dynamic_nav.py              # 主入口：状态机 + DWB/waypoint 双模式
├── test_init.py
└── README.md
```

## 系统架构

```
                        ┌──────────────────────────┐
                        │    SLAM (MMK2_SLAM)       │
                        │  odometry → ICP → grid    │
                        └───────────┬──────────────┘
                                    │
                        ┌───────────▼──────────────┐
                        │      Dynamic Layer        │
                        │  Bresenham 射线 + 衰减    │
                        └───────────┬──────────────┘
                                    │
                        ┌───────────▼──────────────┐
                        │      Planning Grid        │
                        │  静态地图 + 动态层融合     │
                        └───────────┬──────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
  ┌─────────▼────────┐  ┌──────────▼──────────┐  ┌─────────▼────────┐
  │  Emergency Check  │  │   Path Validator    │  │  A* Path Planner │
  │  (前向 LiDAR)     │  │   (膨胀代价碰撞)    │  │  (全局路径规划)   │
  └─────────┬────────┘  └──────────┬──────────┘  └─────────┬────────┘
            │                      │                        │
            │              ┌───────▼───────┐                │
            │              │ LocalGoal     │                │
            │              │ Selector      │                │
            │              └───────┬───────┘                │
            │                      │                        │
            │              ┌───────▼────────────────────────┘
            │              │
            │     ┌────────▼────────┐
            │     │  DWB Controller  │  ← Phase 2 新增
            │     │ ┌──────────────┐ │
            │     │ │ TrajGenerator│ │  动态窗口采样
            │     │ │   → poses    │ │
            │     │ └──────┬───────┘ │
            │     │ ┌──────▼───────┐ │
            │     │ │   Critics    │ │  Obstacle / PathDist
            │     │ │   → scores   │ │  GoalDist / Forward
            │     │ └──────┬───────┘ │
            │     │ ┌──────▼───────┐ │
            │     │ │  best (v,w)  │ │
            │     │ └──────────────┘ │
            │     └────────┬────────┘
            │              │
            └──────────────┼───────────────────────┐
                           │                       │
                   ┌───────▼───────┐               │
                   │  State Machine│               │
                   │  MAPPING      │               │
                   │  → FOLLOWING  │◄──────────────┘
                   │  → EMERG_STOP │
                   │  → REPLANNING │
                   │  → GOAL_REACH │
                   └───────────────┘
```

## 状态机

```
MAPPING (stationary scan, 100 frames)
   │
   ▼
WAITING_FOR_GOAL
   │ (A* plans initial path)
   ▼
FOLLOWING_PATH ◄─────────────────┐
   │    ▲                        │
   │    │ (replan success)       │
   │    └────────────────────────┤
   │                             │
   ├── EmergencyChecker ──► EMERGENCY_STOP
   │                              │ (hold)
   │                              └──► REPLANNING
   │
   ├── PathValidator blocked ──► REPLANNING
   │                              │
   │              ┌───────────────┼───────────────┐
   │              ▼               │               ▼
   │          success ──► FOLLOWING_PATH    fail ──► WAITING_FOR_CLEARANCE
   │                                                      │
   │                                   dynamic_layer chg ──┤
   │                                   timeout ──► SAFE_STOP
   │
   └── at goal ──► GOAL_REACHED

Phase 2 extension (--controller dwb):
   DWB failure count ≥ threshold ──► REPLANNING
```

## 控制器模式

项目支持两种局部控制器，通过 `--controller` 切换：

| 模式 | 说明 | 默认 |
|------|------|------|
| `waypoint` | Phase 1 基线：MotionController 跟踪 A* 航点 | ✓ |
| `dwb` | Phase 2 新增：DWB-inspired 轨迹采样 + Critic 评分 | — |

DWB 模式下控制优先级：
1. **EmergencyChecker** （最高优先级，无条件停车）
2. **DWB 成功** → 执行 DWB 速度命令
3. **DWB 失败** → 零速，累计失败次数
4. **连续失败 ≥ 阈值** → 触发 A* 重规划
5. **A* 无解** → WAITING_FOR_CLEARANCE / SAFE_STOP

## 障碍物模式

通过 `--obstacle_mode` 控制障碍物行为：

| 模式 | 说明 |
|------|------|
| `static` | Phase 1 默认：在 A* 路径上生成一个静止障碍物 |
| `scripted_line` | Phase 2 新增：障碍物沿垂直于机器人路径的方向往返运动 |

## 快速开始

```powershell
# 激活环境
conda activate discoverse

# Phase 1: waypoint 控制器 + GUI
python run_dynamic_nav.py

# Phase 1: headless 快速验证
python run_dynamic_nav.py --headless --max_steps 500 --seed 42

# Phase 2: DWB 控制器 + GUI + 不限步数
python run_dynamic_nav.py --controller dwb --unlimited

# Phase 2: DWB + 移动障碍 + Headless
python run_dynamic_nav.py --controller dwb --headless --max_steps 500 --obstacle_mode scripted_line --seed 42

# 运行全部单元测试
Get-ChildItem tests/test_*.py | ForEach-Object { python $_.FullName }
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--goal_x` | -0.5 | 导航目标 X 坐标 |
| `--goal_y` | -2.0 | 导航目标 Y 坐标 |
| `--headless` | False | 无渲染/无 GUI 模式 |
| `--max_steps` | 10000 | 最大仿真步数 |
| `--unlimited` | False | 无限制步数 |
| `--seed` | None | 随机种子（int=可复现） |
| `--controller` | waypoint | 控制器模式：`waypoint` / `dwb` |
| `--obstacle_mode` | static | 障碍物模式：`static` / `scripted_line` |
| `--width` | 1600 | MuJoCo 窗口宽度 |
| `--height` | 900 | MuJoCo 窗口高度 |

## 技术栈

- **仿真引擎**：MuJoCo（通过 DISCOVERSE）
- **LiDAR 仿真**：MuJoCo-LiDAR 子模块
- **SLAM 基础**：MMK2_SLAM（轮式里程计 + ICP + 占据栅格地图）
- **动态感知**：Bresenham 射线投射 + 连续命中/空闲确认 + 时间衰减
- **全局规划**：A*（8 邻域，膨胀代价距离变换）
- **局部控制**：DWB-inspired（动态窗口 + 差速模型 + Critic 插件评分）
- **可视化**：Matplotlib 4 面板 GUI（地图/扫描/轨迹/DWB 状态）+ MuJoCo 3D 渲染
