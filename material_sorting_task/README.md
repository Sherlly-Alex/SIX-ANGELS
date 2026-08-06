# SIX ANGELS Material Sorting Client

DG-202612 文旅机器人搬运赛题的参赛 Client 工作区。本仓库只维护参赛端代码；正式
Server、场景随机化和裁判由赛方镜像提供，不在这里修改。

## 当前状态

当前已经整理出指令解析、感知、导航、几何计算和 ROS 2 Client 入口。`client_task.py`
已经接入三任务连续调度状态机，并以 Server 裁判状态作为正式模式下的尝试结算、任务推进
和得分真值。任务 1 已提供显式 `nav_only` 底盘实动、`pregrasp_only` 开放预抓取以及
`contact_only` 双侧接触模式：最后一种会从视觉世界坐标导航到随机桌边目标，调用桌面
抓取标定逆解，在 Server 确认双臂同时接触目标后保持。柔顺挤压、抬升、搬运和放置以及
任务 2/3 正式执行器仍未开放，不会把空动作误判为成功。

## 目录

```text
examples/material_sorting/
  client_task.py                 正式 Client 入口
  competition_controller.py      三任务连续调度与裁判同步
  executors/                     任务 1/2/3 执行器接口和安全占位实现
  instruction_parser.py          结构化指令解析与校验
  task_orchestration.py          三任务编排辅助函数
  navigation/                    底盘导航模块
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

桌面抓取联调（仅任务 1 或 3）见 [docs/DESKTOP_GRASP.md](docs/DESKTOP_GRASP.md)。

## 测试

```bash
python3 -m unittest discover -s tests -t .
python3 scripts/check_workspace.py
```

开发约束和后续实现顺序见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。
