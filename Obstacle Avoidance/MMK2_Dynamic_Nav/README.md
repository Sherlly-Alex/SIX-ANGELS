# MMK2_Dynamic_Nav

基于 DISCOVERSE 仿真平台的 MMK2 机器人动态导航系统，在 `MMK2_SLAM` 基础上扩展了动态障碍物感知、路径安全检测与在线重规划能力。

---

## 1. 项目位置

本目录位于 DISCOVERSE 项目内：

```text
DISCOVERSE-main/
├── discoverse/                         # DISCOVERSE 仿真引擎核心
│   └── robots_env/
│       └── mmk2_base.py                # MMK2 机器人基类（MMK2Cfg / MMK2Base）
├── examples/
│   ├── MMK2_SLAM/                      # 上游项目：MMK2 2D SLAM 教学示例
│   ├── MMK2_Dynamic_Nav/               # 本项目：MMK2 动态导航
│   └── tasks_mmk2/                     # MMK2 操作任务示例
└── submodules/
    └── MuJoCo-LiDAR/                   # LiDAR 仿真模块
```

---

## 2. 开发基础

本系统基于 **MMK2_SLAM** 项目构建，复用了其 SLAM 核心模块，并在此基础上新增动态障碍物感知、规划网格融合、路径验证、紧急制动和在线重规划功能。

### 2.1 从 MMK2_SLAM 继承的模块

| 模块                      | 来源 | 说明                             |
| ----------------------- | -- | ------------------------------ |
| `config/slam_config.py` | 继承 | 机器人与 SLAM 参数配置，继承 `MMK2Cfg`    |
| `core/*`                | 复用 | 包含里程计、扫描匹配、占据栅格、SLAM 估计器和前沿探索  |
| `robot/*`               | 复用 | 包含 MMK2 机器人封装和运动控制器            |
| `scenes/*`              | 复用 | 包含 MMK2 模型和房间场景的 MuJoCo XML 文件 |

### 2.2 本项目新增模块

| 模块     | 文件                              | 说明                            |
| ------ | ------------------------------- | ----------------------------- |
| 动态感知   | `perception/dynamic_layer.py`   | 检测并追踪不属于静态地图的动态障碍物            |
| 规划网格   | `planning/planning_grid.py`     | 融合静态占据栅格与动态障碍物层               |
| 路径验证   | `planning/path_validator.py`    | 通过线段碰撞检测判断当前路径是否被阻塞           |
| 紧急制动   | `planning/emergency_checker.py` | 根据前向 LiDAR 数据检测近距离碰撞风险        |
| 障碍物管理  | `sim/obstacle_manager.py`       | 在 MuJoCo 场景中生成和控制 mocap 动态障碍物 |
| 导航配置   | `config/dynamic_nav_config.py`  | 管理动态导航相关参数                    |
| 导航 GUI | `gui/dynamic_nav_viewer.py`     | 提供 6 面板 Matplotlib 可视化界面      |
| 状态覆盖   | `gui/nav_overlay.py`            | 输出导航状态、路径状态和安全状态摘要            |

---

## 3. 项目结构

```text
MMK2_Dynamic_Nav/
├── config/
│   ├── slam_config.py                  # SLAM 参数，继承自 MMK2_SLAM
│   └── dynamic_nav_config.py           # 动态导航参数
│
├── core/                               # SLAM 核心模块，复用自 MMK2_SLAM
│   ├── slam_estimator.py               # SLAM 中央协调器
│   ├── odometry.py                     # 差速驱动轮式里程计
│   ├── scan_matching.py                # ICP 激光扫描匹配
│   ├── occupancy_grid.py               # 二维占据栅格地图
│   └── frontier_exploration.py         # 前沿探索与 A* 路径规划
│
├── robot/                              # 机器人控制层，复用自 MMK2_SLAM
│   ├── mmk2_slam_robot.py              # MMK2 机器人封装，继承 MMK2Base
│   └── motion_controller.py            # 运动控制器
│
├── perception/                         # 动态障碍物感知
│   └── dynamic_layer.py                # 动态障碍物检测、记录与时间衰减
│
├── planning/                           # 路径规划与安全检测
│   ├── planning_grid.py                # 静态地图与动态层融合
│   ├── path_validator.py               # 路径阻塞检测
│   └── emergency_checker.py            # 前向 LiDAR 紧急制动
│
├── sim/                                # 仿真环境管理
│   └── obstacle_manager.py             # 动态障碍物生成与控制
│
├── gui/                                # 可视化与状态输出
│   ├── dynamic_nav_viewer.py           # 6 面板 Matplotlib GUI
│   └── nav_overlay.py                  # 控制台导航状态输出
│
├── scenes/                             # MuJoCo 场景文件
│   ├── dynamic_nav_mmk2.xml            # 动态导航场景，包含 mocap 障碍物
│   ├── slam_room_mmk2.xml              # 原始 SLAM 房间场景
│   ├── mmk2_slim.xml                   # MMK2 机器人模型
│   └── mmk2_dependencies_slim.xml      # MMK2 模型依赖
│
├── tests/                              # 单元测试与集成测试
│   ├── test_dynamic_layer.py
│   ├── test_planning_grid.py
│   ├── test_path_validator.py
│   ├── test_emergency_checker.py
│   └── test_dynamic_nav_smoke.py
│
├── run_dynamic_nav.py                  # 主程序：动态导航状态机
├── test_init.py                        # 基础导入测试
├── test_lidar_and_contact.py           # LiDAR 与接触传感器测试
├── test_mocap_obstacle.py              # mocap 动态障碍物测试
└── README.md                           # 项目说明文档
```

---

## 4. 系统架构

```text
┌───────────────────────────────────────┐
│          SLAM（MMK2_SLAM）             │
│                                       │
│  Odometry → ICP → Occupancy Grid      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│             Dynamic Layer             │
│                                       │
│  Bresenham 射线投射 + 动态时间衰减     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│             Planning Grid             │
│                                       │
│     静态地图 + 动态障碍物层            │
└───────────────────┬───────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Emergency    │ │ Path         │ │ A* Path      │
│ Checker      │ │ Validator    │ │ Planner      │
│              │ │              │ │              │
│ 前向 LiDAR   │ │ 线段碰撞检测 │ │ 8 邻域搜索   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
┌───────────────────────────────────────┐
│              State Machine            │
│                                       │
│  MAPPING                              │
│  WAITING_FOR_GOAL                     │
│  FOLLOWING_PATH                       │
│  EMERGENCY_STOP                       │
│  REPLANNING                           │
│  WAITING_FOR_CLEARANCE                │
│  GOAL_REACHED                         │
│  SAFE_STOP                            │
└───────────────────────────────────────┘
```

系统主要数据流如下：

```text
LiDAR / Wheel Encoder
        │
        ▼
SLAM 定位与静态地图构建
        │
        ▼
动态障碍物检测与动态层更新
        │
        ▼
静态地图与动态层融合
        │
        ▼
路径安全检测与 A* 路径规划
        │
        ▼
导航状态机
        │
        ▼
运动控制器
        │
        ▼
MMK2 底盘控制
```

---

## 5. 状态机

```text
MAPPING
   │
   │ 建图约 100 帧，获取静态环境信息
   ▼
WAITING_FOR_GOAL
   │
   │ 使用 A* 规划初始路径
   ▼
FOLLOWING_PATH ◄──────────────────── REPLANNING
   │                                      │
   │                                      │ 重规划成功
   │                                      │
   │                                      └───────────────┐
   │                                                      │
   ├── 检测到近距离障碍物                                 │
   │                                                      │
   ▼                                                      │
EMERGENCY_STOP                                            │
   │                                                      │
   │ 停车并等待短暂安全时间                               │
   ▼                                                      │
REPLANNING ───────────────────────────────────────────────┘
   │
   ├── 重规划成功
   │       │
   │       ▼
   │  FOLLOWING_PATH
   │
   └── 重规划失败
           │
           ▼
WAITING_FOR_CLEARANCE
   │
   ├── 动态障碍物消失或规划网格更新
   │       │
   │       ▼
   │  REPLANNING
   │
   └── 等待超时
           │
           ▼
       SAFE_STOP
```

### 5.1 状态说明

| 状态                      | 说明                   |
| ----------------------- | -------------------- |
| `MAPPING`               | 机器人执行初始建图，尽可能覆盖静态环境  |
| `WAITING_FOR_GOAL`      | 等待目标点设置，并尝试生成初始路径    |
| `FOLLOWING_PATH`        | 按照当前路径依次跟踪路径点        |
| `EMERGENCY_STOP`        | 检测到近距离碰撞风险后立即停止机器人   |
| `REPLANNING`            | 根据最新规划网格重新执行 A* 路径规划 |
| `WAITING_FOR_CLEARANCE` | 当前无可行路径时等待动态障碍物离开    |
| `GOAL_REACHED`          | 机器人到达目标位置            |
| `SAFE_STOP`             | 长时间无法找到安全路径时停止运行     |

### 5.2 主要状态转换条件

| 当前状态                    | 触发条件         | 下一状态                    |
| ----------------------- | ------------ | ----------------------- |
| `MAPPING`               | 达到预设建图帧数     | `WAITING_FOR_GOAL`      |
| `WAITING_FOR_GOAL`      | 初始路径规划成功     | `FOLLOWING_PATH`        |
| `WAITING_FOR_GOAL`      | 初始路径规划失败     | `WAITING_FOR_CLEARANCE` |
| `FOLLOWING_PATH`        | 前方障碍物距离过近    | `EMERGENCY_STOP`        |
| `FOLLOWING_PATH`        | 当前路径被动态障碍物阻塞 | `REPLANNING`            |
| `FOLLOWING_PATH`        | 到达最终目标点      | `GOAL_REACHED`          |
| `EMERGENCY_STOP`        | 完成安全停车等待     | `REPLANNING`            |
| `REPLANNING`            | 新路径规划成功      | `FOLLOWING_PATH`        |
| `REPLANNING`            | 新路径规划失败      | `WAITING_FOR_CLEARANCE` |
| `WAITING_FOR_CLEARANCE` | 地图更新或障碍物离开   | `REPLANNING`            |
| `WAITING_FOR_CLEARANCE` | 等待时间超过阈值     | `SAFE_STOP`             |

---

## 6. 核心模块说明

### 6.1 SLAM 模块

SLAM 模块沿用 `MMK2_SLAM` 项目的实现，主要负责：

1. 根据轮速信息计算机器人轮式里程计；
2. 使用 ICP 对连续 LiDAR 帧进行扫描匹配；
3. 融合里程计与 ICP 结果，估计机器人位姿；
4. 将 LiDAR 数据写入二维占据栅格地图；
5. 为路径规划提供静态环境地图。

整体处理流程如下：

```text
轮速数据
   │
   ▼
Odometry
   │
   ├──────────────┐
   │              │
   ▼              ▼
预测位姿      LiDAR 扫描
   │              │
   └──────┬───────┘
          ▼
      ICP 匹配
          │
          ▼
     位姿修正结果
          │
          ▼
  Occupancy Grid 更新
```

### 6.2 动态障碍物层

`dynamic_layer.py` 用于识别当前 LiDAR 观测中不属于静态地图的障碍物。

主要流程如下：

```text
当前 LiDAR 扫描
        │
        ▼
根据机器人位姿转换到世界坐标系
        │
        ▼
使用 Bresenham 算法遍历激光射线路径
        │
        ▼
与静态占据栅格地图进行比较
        │
        ▼
识别潜在动态障碍物栅格
        │
        ▼
更新时间戳与动态置信度
        │
        ▼
对长时间未观测到的动态栅格执行衰减和清除
```

动态层不会直接修改原始静态地图，而是作为独立图层保存。这样可以避免移动障碍物被永久写入静态占据栅格。

### 6.3 规划网格

`planning_grid.py` 将静态占据栅格与动态障碍物层融合，生成供路径规划使用的最终网格。

```text
静态占据栅格
        │
        ├─────────────┐
        │             │
        ▼             ▼
静态障碍物判断    动态障碍物层
        │             │
        └──────┬──────┘
               ▼
          障碍物膨胀
               │
               ▼
          规划代价网格
```

规划网格主要处理以下内容：

* 静态障碍物；
* 动态障碍物；
* 未知区域；
* 机器人自身尺寸；
* 安全距离；
* 障碍物膨胀代价。

### 6.4 路径验证

`path_validator.py` 用于判断机器人当前正在执行的路径是否仍然安全。

检查过程如下：

```text
机器人当前位置
        │
        ▼
选取当前路径中的后续线段
        │
        ▼
在线段上进行离散采样
        │
        ▼
查询规划网格中的占据状态
        │
        ├── 全部可通行 ──► 路径有效
        │
        └── 存在障碍物 ──► 路径阻塞，触发重规划
```

路径验证可以避免机器人继续沿着已经被动态障碍物堵塞的旧路径行驶。

### 6.5 紧急制动

`emergency_checker.py` 独立于全局路径规划，用于提供最低层级的安全保护。

其基本原理是：

1. 从完整 LiDAR 数据中截取机器人正前方扇形区域；
2. 计算该区域内的最小有效距离；
3. 将最小距离与紧急制动阈值比较；
4. 当距离低于阈值时，立即输出停车指令。

```text
前向 LiDAR 点云
        │
        ▼
过滤无效距离
        │
        ▼
计算前向最小距离
        │
        ├── 距离安全 ──► 继续行驶
        │
        └── 距离过近 ──► EMERGENCY_STOP
```

紧急制动不依赖动态障碍物识别是否成功，因此能够在感知或规划出现短暂误差时提供额外保护。

### 6.6 A* 路径规划

A* 路径规划复用 `frontier_exploration.py` 中的相关实现。

当前规划器采用：

* 二维栅格地图；
* 8 邻域搜索；
* 对角线移动；
* 障碍物膨胀；
* 启发式代价；
* 静态与动态障碍物融合网格。

A* 节点总代价为：

```text
f(n) = g(n) + h(n)
```

其中：

* `g(n)`：起点到当前节点的实际路径代价；
* `h(n)`：当前节点到目标节点的启发式估计代价；
* `f(n)`：节点的综合优先级。

对于 8 邻域搜索，可以使用欧氏距离或对角距离作为启发函数。

---

## 7. 快速开始

### 7.1 激活 Conda 环境

```bash
conda activate discoverse
```

### 7.2 初始化 MuJoCo-LiDAR 子模块

请在 `DISCOVERSE-main` 项目根目录下执行：

```bash
git submodule update --init --recursive submodules/MuJoCo-LiDAR
```

也可以初始化项目中的全部 Git 子模块：

```bash
git submodule update --init --recursive
```

### 7.3 进入项目目录

```bash
cd examples/MMK2_Dynamic_Nav
```

### 7.4 运行动态导航

使用默认目标点 `(-0.5, -2.0)`：

```bash
python run_dynamic_nav.py
```

### 7.5 使用自定义目标点

```bash
python run_dynamic_nav.py --goal_x 1.0 --goal_y 1.0
```

### 7.6 无头模式运行

```bash
python run_dynamic_nav.py \
    --headless \
    --goal_x 1.0 \
    --goal_y 1.0 \
    --max_steps 500
```

Windows PowerShell 中也可以写成单行：

```powershell
python run_dynamic_nav.py --headless --goal_x 1.0 --goal_y 1.0 --max_steps 500
```

### 7.7 无限步数运行

```bash
python run_dynamic_nav.py --unlimited
```

### 7.8 指定随机种子

```bash
python run_dynamic_nav.py --seed 42
```

### 7.9 指定 GUI 分辨率

```bash
python run_dynamic_nav.py --width 1600 --height 900
```

---

## 8. 命令行参数

| 参数            |     默认值 | 说明          |
| ------------- | ------: | ----------- |
| `--goal_x`    |  `-0.5` | 目标点的 X 坐标   |
| `--goal_y`    |  `-2.0` | 目标点的 Y 坐标   |
| `--headless`  | `False` | 是否关闭图形界面    |
| `--max_steps` | `10000` | 最大仿真步数      |
| `--unlimited` | `False` | 是否取消最大步数限制  |
| `--seed`      |  `None` | 随机种子，用于复现实验 |
| `--width`     |  `1600` | GUI 窗口宽度    |
| `--height`    |   `900` | GUI 窗口高度    |

完整示例：

```bash
python run_dynamic_nav.py \
    --goal_x 1.5 \
    --goal_y -1.0 \
    --max_steps 20000 \
    --seed 42 \
    --width 1600 \
    --height 900
```

---

## 9. 测试方法

### 9.1 运行全部单元测试

```bash
python -m pytest tests/ -v
```

### 9.2 运行指定测试文件

测试动态障碍物层：

```bash
python -m pytest tests/test_dynamic_layer.py -v
```

测试规划网格：

```bash
python -m pytest tests/test_planning_grid.py -v
```

测试路径验证器：

```bash
python -m pytest tests/test_path_validator.py -v
```

测试紧急制动模块：

```bash
python -m pytest tests/test_emergency_checker.py -v
```

运行动态导航冒烟测试：

```bash
python -m pytest tests/test_dynamic_nav_smoke.py -v
```

### 9.3 运行基础导入测试

```bash
python test_init.py
```

### 9.4 测试 LiDAR 与接触传感器

```bash
python test_lidar_and_contact.py
```

### 9.5 测试 mocap 动态障碍物

```bash
python test_mocap_obstacle.py
```

---

## 10. 技术栈

| 类别       | 技术                    |
| -------- | --------------------- |
| 仿真引擎     | MuJoCo                |
| 仿真平台     | DISCOVERSE            |
| 机器人平台    | MMK2                  |
| LiDAR 仿真 | MuJoCo-LiDAR          |
| SLAM     | 轮式里程计、ICP、占据栅格地图      |
| 动态感知     | Bresenham 射线投射、动态时间衰减 |
| 路径规划     | A*、8 邻域搜索、障碍物膨胀       |
| 路径安全     | 路径线段碰撞检测              |
| 紧急避障     | 前向 LiDAR 最小距离检测       |
| 动态障碍物    | MuJoCo mocap body     |
| 可视化      | Matplotlib 6 面板 GUI   |
| 状态输出     | 控制台导航状态摘要             |
| 测试框架     | pytest                |

---

## 11. 系统运行流程

系统启动后的完整运行流程如下：

```text
1. 加载 DISCOVERSE 与 MuJoCo 场景
        │
        ▼
2. 初始化 MMK2 机器人、LiDAR 和 SLAM
        │
        ▼
3. 执行初始建图
        │
        ▼
4. 生成或激活动态 mocap 障碍物
        │
        ▼
5. 更新静态地图与动态障碍物层
        │
        ▼
6. 融合生成规划网格
        │
        ▼
7. 使用 A* 规划到目标点的路径
        │
        ▼
8. 运动控制器跟踪路径点
        │
        ▼
9. 持续执行紧急检测与路径验证
        │
        ├── 路径安全 ──► 继续跟踪
        │
        ├── 路径阻塞 ──► 在线重规划
        │
        └── 障碍物过近 ──► 紧急停车
        │
        ▼
10. 到达目标点或进入安全停止状态
```

---

## 12. 动态障碍物处理策略

动态障碍物处理分为三个层次。

### 12.1 感知层

通过当前 LiDAR 观测与静态地图之间的差异，识别潜在动态障碍物。

```text
当前观测为占据
+
静态地图中对应位置为空闲
=
潜在动态障碍物
```

### 12.2 规划层

将动态障碍物写入独立动态层，并与静态地图融合生成规划网格。

```text
Planning Grid = Static Occupancy Grid + Dynamic Layer
```

### 12.3 安全层

即使动态障碍物尚未稳定写入动态层，前向 LiDAR 紧急制动模块仍会直接判断碰撞风险。

因此系统具有以下安全链路：

```text
动态层检测
    │
    ▼
路径阻塞检测
    │
    ▼
A* 在线重规划
    │
    ▼
前向 LiDAR 紧急制动兜底
```

---

## 13. 可视化界面

`dynamic_nav_viewer.py` 提供 6 面板 Matplotlib GUI，用于显示导航过程中的关键信息。

可视化内容可以包括：

1. 静态占据栅格地图；
2. 动态障碍物层；
3. 静态与动态融合后的规划网格；
4. 机器人当前位置与历史轨迹；
5. 当前 A* 路径与目标点；
6. LiDAR 数据、状态机状态或运行指标。

控制台状态输出由 `nav_overlay.py` 提供，可用于显示：

```text
State: FOLLOWING_PATH
Robot Pose: (x, y, yaw)
Goal: (goal_x, goal_y)
Path Length: ...
Current Waypoint: ...
Emergency: False
Path Blocked: False
Replan Count: ...
```

---

## 14. 项目特点

* 基于已有 `MMK2_SLAM` 项目扩展，保留原有 SLAM 学习结构；
* 将静态地图与动态障碍物层分离，避免污染长期地图；
* 支持动态障碍物出现、移动、消失和时间衰减；
* 支持运行过程中检测路径阻塞；
* 支持基于最新地图进行 A* 在线重规划；
* 提供独立的前向 LiDAR 紧急制动机制；
* 通过状态机管理建图、跟踪、停车、重规划和安全停止；
* 提供单元测试、冒烟测试和可视化调试工具；
* 项目位于独立示例目录中，尽量避免修改 DISCOVERSE 核心代码。

---

## 15. 当前限制

当前系统仍存在以下限制：

1. 动态障碍物主要通过静态地图差异进行检测，尚未使用目标级检测模型；
2. 当前动态层主要记录障碍物占据状态，未完整估计障碍物速度和运动轨迹；
3. A* 规划基于二维离散栅格，路径可能存在折线和局部不平滑问题；
4. 当前系统以在线全局重规划为主，尚未集成完整 DWA、DWB 或 TEB 局部规划器；
5. 紧急制动主要根据前向最小距离判断，尚未结合速度和制动距离建立动态阈值；
6. 当障碍物完全封锁通道时，系统只能等待障碍物离开或进入安全停止状态；
7. 当前动态障碍物主要由 MuJoCo mocap body 模拟，与真实行人或车辆运动模型仍有差异。

---

## 16. 后续可扩展方向

后续可以继续增加以下功能：

* 动态障碍物聚类与目标级追踪；
* 障碍物速度估计；
* 卡尔曼滤波或扩展卡尔曼滤波；
* 基于预测轨迹的时空路径规划；
* DWA、DWB 或 TEB 局部规划器；
* 路径平滑与曲率约束；
* 根据机器人速度动态调整紧急停车距离；
* 动态障碍物多场景测试；
* 导航成功率、路径长度和重规划次数统计；
* 不同障碍物速度下的性能评估；
* 与 RRT、RRT*、D* Lite 等规划算法进行对比；
* 保存运行日志并自动生成实验图表。

---

## 17. 说明

本项目主要用于：

* 学习 DISCOVERSE 项目结构；
* 理解 MMK2 机器人仿真接口；
* 学习二维 SLAM 基础实现；
* 学习动态障碍物感知；
* 学习 A* 在线重规划；
* 学习导航状态机和安全机制；
* 为后续集成局部规划器或预测式动态避障算法提供基础。

本项目当前定位为教学与实验示例，不建议在未经进一步安全验证的情况下直接用于真实机器人环境。
