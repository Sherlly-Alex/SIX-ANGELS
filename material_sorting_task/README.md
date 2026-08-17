# SIX ANGELS Material Sorting Client

DG-202612 文旅机器人搬运赛题的参赛 Client 工作区。本仓库只维护参赛端代码；正式
Server、场景随机化和裁判由赛方镜像提供，不在这里修改。

## 当前状态

当前已经整理出指令解析、感知、导航、几何计算和 ROS 2 Client 入口。`client_task.py`
已经接入三任务连续调度状态机，并以 Server 裁判状态作为正式模式下的尝试结算、任务推进
和得分真值。任务 1 已提供显式 `nav_only`、`pregrasp_only`、`contact_only` 和
`lift_only` 分段测试模式。`task12_full` 保留为任务 1/2 回归模式；新增的 `task123_full`
在同一套接口中继续接入任务 3：复用任务 1 的桌面抓取与抬升、任务 1 保存的货架状态，
并使用货架白色长方体的测量中心计算左侧放置位和安全释放位。任务切换仍只由 Server
裁判确认；该整合模式已经通过本地接口和状态机测试，需在 4090 官方镜像中逐段标定验证。

## 目录

```text
examples/material_sorting/
  client_task.py                 正式 Client 入口
  competition_controller.py      三任务连续调度与裁判同步
  scheduler/                     V2 调度内核、TaskPlan、资源/安全、候选策略
  learning/                      可选离散宏动作 RL 训练与运行时护栏
  executors/                     任务 1/2/3 执行器接口和安全占位实现
  instruction_parser.py          结构化指令解析与校验
  task_orchestration.py          三任务编排辅助函数
  navigation/                    底盘导航与版本化全局代价地图
  perception/                    RGB-D / YOLO 感知模块
  desktop_grasp/                 任务 1/3 桌面双臂抓取模块
  mjcf/                          Client 坐标和碰撞计算所需模型
  models/mjcf, models/meshes/    Client 运动学所需资产
  reference/                     赛方示例与本地参考实现，不作为正式入口
scripts/
  run_client.sh                  容器内正式启动脚本
  run_desktop_grasp.sh           桌面抓取联调启动脚本
  setup_env_gpu.sh               ROS 2 / GPU 环境初始化
tests/                            不依赖 ROS 2 的单元测试
docs/                             架构和开发说明
semantic_research/                离线语义旁路（Regex/ML/SLM），不进入正式控制链
```

## 运行

将仓库挂载到赛方 Client 容器的 `/workspace/baseline/material_sorting_task`，然后执行：

```bash
cd /workspace/baseline/material_sorting_task
bash scripts/run_client.sh
```

默认使用 YOLO：

```bash
MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
```

默认权重为 `examples/material_sorting/perception/checkpoints/best.pt`，可通过
`MATERIAL_YOLO_CHECKPOINT` 覆盖。

没有权重时可先用颜色后端检查通信链路：

```bash
MATERIAL_DETECT_BACKEND=color bash scripts/run_client.sh
```

正式运行必须使用：

```text
ROS_DOMAIN_ID=99
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 调度链路测试

默认 `MATERIAL_EXECUTION_MODE=stub`，正式 Client 会在第一个尚未实现的动作阶段安全
阻塞。只验证状态机和三任务顺序时，可以显式启用不控制机器人的 dry-run：

```bash
MATERIAL_EXECUTION_MODE=dry_run \
MATERIAL_DRY_RUN_TICKS_PER_STAGE=2 \
bash scripts/run_client.sh
```

dry-run 只验证任务 1 -> 任务 2 -> 任务 3 的内部调度、日志和进程生命周期，不产生机器人
动作，也不会得到 Server 评分。正式模式的任务切换必须等待 `/referee/taskinfo`、
`/referee/gameinfo` 和 `/referee/score`。

### 调度器 V2 与实时候选评分

调度入口支持三种可回退模式：

```bash
# 正式默认：原控制器
MATERIAL_SCHEDULER_ENGINE=legacy bash scripts/run_client.sh

# 原控制器执行，V2 只校验状态轨迹，不重复 tick 运动执行器
MATERIAL_EXECUTION_MODE=dry_run \
MATERIAL_SCHEDULER_ENGINE=shadow bash scripts/run_client.sh

# 版本化 TaskPlan、裁判网关、资源租约和命令安全边界
MATERIAL_EXECUTION_MODE=dry_run \
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_SCHEDULER_POLICY=heuristic \
bash scripts/run_client.sh
```

`v2` 在导航阶段通过后台单线程以最高 4 Hz 更新 `WorldCostmap`，生成中心/左偏/右偏有限
站位，并先做碰撞、携物包络、资源和裁判硬过滤，再进行确定性 Multi-Critic 评分。评分线程
不阻塞 20 Hz 控制 tick，也不直接发布底盘或机械臂命令。只有显式实现
`apply_scheduler_candidate(...)` 的执行器才会接收通过约束的候选，其余执行器保持该链路为
可观测旁路。当前仅 `nav_only` 的 Task 1 导航执行器实现了该 hook：候选必须再次通过站位
走廊（横向 ≤0.15 m / 纵向 ≤0.10 m）、分层栅格无碰撞、站位净空 ≥0.22 m 和
`NavigationController` 实际重规划四道校验；与已标定名义站位不一致的可选候选记录为
`audit_only` 并继续名义轨迹，非法输入、碰撞/净空和实际重规划失败仍 fail-closed 停车；`legacy`/`shadow`
和未接入 hook 的模式不受影响。因此正式实动仍应保持 `legacy`，先以 `shadow`、
`v2 + dry_run` 收集轨迹，再在 `v2 + nav_only` 上逐段标定候选站位切换：

```bash
MATERIAL_EXECUTION_MODE=nav_only \
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_SCHEDULER_POLICY=heuristic \
MATERIAL_SCHEDULER_EVENT_LOG=/tmp/material_scheduler.jsonl \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

可将结构化状态、候选 Critic 和选择结果写入 JSONL：

```bash
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_EXECUTION_MODE=dry_run \
MATERIAL_SCHEDULER_POLICY=heuristic \
MATERIAL_SCHEDULER_EVENT_LOG=/tmp/material_scheduler.jsonl \
bash scripts/run_client.sh
```

强化学习仅能选择已经通过硬过滤的离散宏动作，不允许输出 `vx/wz` 或关节控制量。推荐先
使用 `rl_shadow`；模型缺失、哈希/schema 不匹配、推理超时、NaN、越界或选择 masked 动作
都会确定性回退到启发式策略：

```bash
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_EXECUTION_MODE=dry_run \
MATERIAL_SCHEDULER_POLICY=rl_shadow \
MATERIAL_SCHEDULER_MODEL=/workspace/models/scheduler_maskable_ppo.zip \
MATERIAL_SCHEDULER_MODEL_SHA256=<approved-sha256> \
MATERIAL_SCHEDULER_EVENT_LOG=/tmp/material_scheduler.jsonl \
bash scripts/run_client.sh
```

`rl_guarded` 已提供运行时护栏，但在离线回放、Shadow、仿真和实机分段验收完成前不要用于
正式比赛。项目不会自动下载模型，Gymnasium、Stable-Baselines3 和 sb3-contrib 也不会进入
默认正式 Client 依赖。

### 任务 1 底盘实动测试

`nav_only` 会读取 `/material/detections` 中任务 1 目标颜色的稳定世界坐标，使用静态场景
A*、限速和急停检查导航到目标前方 0.65 米的桌边抓取站位。视觉坐标会先匹配赛题规定的
左右桌边槽位，校准槽位中心和固定 `yaw0` 姿态；超出合法槽位容差的检测会停车拒绝伸臂，
避免白色固定方块或顶面深度误差被传给抓取 IK。该站距为随机布局保留了
0.20 米静态障碍急停余量。到位后会停车，并在尚未接入的机械臂阶段安全
阻塞：

```bash
MATERIAL_EXECUTION_MODE=nav_only \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

该模式会真实发布 `/cmd_vel`。测试前必须确认只有一个 Client 在运行，并使用新启动的
Server；速度在执行器和 ROS 发布入口处被双重限制为不超过 0.20/0.22 m/s 和
0.65/0.70 rad/s。此模式不会完成或结算任务，测试后需重启 Server 恢复初始场景。

### 任务 1 开放预抓取测试

`pregrasp_only` 包含完整的 `nav_only` 路径。导航到位后，它使用当前 `/joint_states`、
目标世界坐标和双臂 IK，让两个夹爪保持完全张开并运动到方块两侧；到位后持续发布最后
的升降轴、头部和双臂位置命令，并在向内夹取前阻塞：

```bash
MATERIAL_EXECUTION_MODE=pregrasp_only \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

该模式会真实移动底盘、升降轴、头部和双臂，但不会执行向内合拢、柔顺挤压或抬升。
Server 的 `/material/unsafe_collision` 会让执行器立即停止推进并保持最后的机械臂命令。
测试期间不得同时运行 `run_desktop_grasp.sh` 或其他机械臂控制节点。

### 任务 1 双侧接触测试

`contact_only` 继续调用桌面抓取模块的标定逆解：完成开放预抓取后，按照任务 1
货源槽位的固定 `yaw0` 方向让两个张开的夹爪缓慢向内移动。Server 的
`/material/grasp_confirmed` 必须连续为真 0.30 秒；首次检测到双侧接触时会立即冻结
当前命令。如果标定接触位尚未得到双侧反馈，则复用桌面抓取模块的 1 mm 步进、最大
4 mm 有限向内搜索；确认后保持接触姿态，并在确认后的柔顺挤压和抬升前阻塞。

```bash
MATERIAL_EXECUTION_MODE=contact_only \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

该模式会真实接触目标方块，但不会挤压或抬升。检测结果摘要默认每 5 秒输出一次；
可用 `MATERIAL_DETECTION_LOG_PERIOD=0` 完全关闭摘要，或设置其他秒数。状态转换、
接触确认、碰撞和错误日志不受影响。

### 任务 1 抬升测试

`lift_only` 复用完整导航、开放预抓取和 4 mm 有限向内预紧。最终预紧姿态稳定后，不再
等待 Server 的双侧接触布尔量，而是保持双臂和张开的夹爪命令，仅缓慢调整升降轴，将
方块抬高 0.15 米。抬升完成后持续保持，并在搬运前安全阻塞：

```bash
MATERIAL_EXECUTION_MODE=lift_only \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

### 任务 1 + 任务 2 + 任务 3 整合测试

`task12_full` 保留为任务 1/2 回归模式；`task123_full` 启用三任务完整整合。两者继续使用同一个
`client_task.py` 发布底盘、升降柱、头部和双臂命令，不得同时启动旧版
`client_task_1.py`、队友验证脚本或其他控制节点：

```bash
MATERIAL_EXECUTION_MODE=task123_full \
MATERIAL_DETECT_BACKEND=yolo \
MATERIAL_DETECTION_LOG_PERIOD=0 \
bash scripts/run_client.sh
```

建议第一次先固定 Server seed，并准备随时 `Ctrl+C`。任务 1 会依次执行桌面抓取、直线
后撤、货架语义识别、空层放置和返回结束区；裁判推进后，任务 2 才会从已识别层抓取
彩色方块并放回任务 1 保存的桌面原坐标。任务 2 的缓存货架状态只用于粗导航和层位；
抓取前会在货架外采集多帧 RGB-D 目标物体中心、横向对中，再直线靠近，不读取 Server
真实物体坐标。详细接口、目录和停机条件见
[`docs/SHELF_TASK12_INTEGRATION.md`](docs/SHELF_TASK12_INTEGRATION.md)。

该模式仍会响应 `/material/unsafe_collision` 并立即停止推进。

桌面抓取联调（仅任务 1 或 3）见 [docs/DESKTOP_GRASP.md](docs/DESKTOP_GRASP.md)。

## 测试

正式（不发现 `tests/semantic_research`，无需 sklearn）：

```bash
bash scripts/run_formal_tests.sh
python scripts/check_workspace.py
```

研究旁路单测与离线评估（与 `run_client.sh` 互不依赖）：

```bash
bash scripts/run_semantic_research_tests.sh
bash scripts/run_semantic_research_eval.sh
```

语义真值与旁路边界见 [docs/SEMANTIC_PARSING.md](docs/SEMANTIC_PARSING.md)、
[docs/DEPENDENCIES_LICENSES.md](docs/DEPENDENCIES_LICENSES.md)、
[semantic_research/README.md](semantic_research/README.md)。

开发约束和后续实现顺序见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。
导航 v3 的兼容边界、接入结构和本地/SSH 验证步骤见
[docs/NAVIGATION_V3_INTEGRATION.md](docs/NAVIGATION_V3_INTEGRATION.md)。
