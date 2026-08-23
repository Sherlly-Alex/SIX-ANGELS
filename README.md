# SIX-ANGELS

SIX-ANGELS 是 DG-202612 文旅机器人搬运赛题的参赛 Client 工作区。项目把 Server 发布的结构化任务、RGB-D/YOLO 感知、底盘导航、双臂抓取、货架放置和三任务调度连接成一条 ROS 2 控制链。正式运行依赖赛方提供的 Server、场景和裁判镜像；本仓库维护 Client、离线研究工具、测试脚本和必要的参考资产。

## 项目目标

机器人需要连续完成三类搬运任务：

1. 从桌面抓取目标物，识别货架空层并完成放置。
2. 从货架取出彩色物体，返回桌面并放置到任务指定位置。
3. 处理顶部或桌面目标及包装箱，完成左侧放置和结束区返回。

任务切换由 Server 裁判状态确认，Client 不使用本地计时器猜测任务完成，也不读取 Server 私有布局真值代替视觉观测。

## 总体架构

```text
Server /material/instruction JSON
Server /referee/*  RGB-D  YOLO  odom  joint_states
                    |
                    v
              client_task.py
     输入新鲜度、指令校验、命令安全检查
                    |
                    v
     CompetitionController + Scheduler V2
                    |
        +-----------+-----------+
        |                       |
        v                       v
   perception/              executors/
 RGB-D 世界坐标和货架状态       三任务阶段动作
        |                       |
        +-----------+-----------+
                    v
       导航、双臂、升降、夹爪命令
                    |
                    v
            Server 裁判与得分
```

正式控制链的基本原则是：感知提供证据，调度器选择有限的安全宏动作，执行器拥有具体动作，Client 统一发布 ROS 命令，裁判确认任务结果。

## 核心算法与实现

### 1. 结构化任务准入

`instruction_parser.py` 只接受 Server `/material/instruction` 中真实存在且通过校验的 JSON 字段，包括目标、颜色、放置类型、世界坐标和半径等。中文 `instruction` 文本不能补全缺失的执行字段。字段缺失、枚举非法、几何值非有限、任务序列冲突或运行中指令变化时，Client 进入 `SAFE_HOLD`。

### 2. RGB-D/YOLO 目标定位

YOLO 或颜色后端先提供类别和候选框，随后在框内使用颜色掩码、中心深度门、连通域筛选和中值去离群，将像素反投影到世界坐标。`fit_cuboid_center()` 比较 `yaw0` 与 `yaw90` 两种箱体方向，根据点云跨度和外溢误差拟合目标中心；点云不足时只能降级为表面深度估计，并由抓取模块决定是否接受。

### 3. 货架状态与空层确认

`ShelfStateTracker` 独立维护彩色物体、白色包装箱和空层历史，使用 ROI、时间新鲜度和多帧投票形成稳定层位。空层不由“没有检测到物体”直接推断，而由 `ShelfEmptyLayerVerifier` 将三层开口投影到深度图，区分后板 `rear`、层内实体 `foreground` 和近距离遮挡 `occluder`。只有唯一空层、后板可见、最近窗口满足投票且无冲突时，任务一才允许放置。

### 4. 导航与携物安全

导航器使用分层栅格、动态障碍、A* 或分段路径、路径平滑和速度限幅。空载、携物和停靠阶段分别使用不同 robot footprint；携物阶段还检查货物包络。动态观测带时间戳和 TTL，过期数据不会永久阻塞路径。任何碰撞、急停余量不足、目标越界、输入过期或实际重规划失败都 fail-closed 停车。

### 5. 双臂连续柔顺抓取

桌面抓取先根据 RGB-D 世界坐标和目标方向计算双臂开放预抓取姿态，再采用小步连续靠近、腕部反馈检测、双侧接触锁定、有限预紧和慢速抬升。该方案是基于 ROS 位置接口的客户端导纳式柔顺控制，不是底层力矩闭环；预紧、接触搜索和抬升均有时间、位移和工作空间上限。

### 6. Scheduler V2 任务调度

Scheduler V2 不输出 `vx/wz`、关节角或夹爪控制量。它围绕执行器的名义站位生成 `center/left/right/recovery` 有限候选，在同一份世界状态快照上依次执行：

```text
候选生成 -> 裁判/资源过滤 -> 碰撞和包络检查
         -> Multi-Critic 代价评分 -> heuristic 或 RL 建议
         -> 滞回和稳定帧门 -> 执行器二次校验 -> 实际重规划
```

候选评分综合成功概率、路径长度、预计耗时、障碍代价、动态风险、航向变化、感知不确定性和恢复成本。候选决策在后台线程以有限频率运行，不阻塞 20 Hz 控制 tick；同一 `step_run_id + action_id + goal_pose` 只允许应用一次。默认使用确定性 `heuristic`，RL 只能在已生成且通过硬约束的候选中选择。

## 主要创新点

- **结构化 JSON 真值边界**：把任务执行字段与自然语言研究解耦，避免 NLP/ML 误解析直接控制机器人。
- **RGB-D 几何中心拟合**：融合颜色掩码、深度门、连通域和箱体尺寸，降低货架、后板和可见表面深度对抓取位姿的影响。
- **主动观察的空层确认**：对货架候选空层进行独立深度证据确认，遮挡或证据冲突时保持 `UNKNOWN`。
- **执行器拥有动作、调度器只选宏动作**：策略优化被限制在可解释的安全站位，不能绕过导航和运动安全边界。
- **携物包络与实测闭环**：运输阶段同时检查规划路径和逐周期实际命令，避免仅用空载 footprint 造成假安全。
- **可审计的调度与研究隔离**：EventLog 记录 observation、mask、候选和选择结果；研究模型、训练依赖和权重不进入正式 Client。

## 代码与文档导航

| 目录 | 内容 |
| --- | --- |
| [`material_sorting_task/examples/material_sorting/`](material_sorting_task/examples/material_sorting/README.md) | 正式 Client 入口、三任务编排和执行模式。 |
| [`perception/`](material_sorting_task/examples/material_sorting/perception/README.md) | YOLO、RGB-D 世界坐标、货架空层证据。 |
| [`navigation/`](material_sorting_task/examples/material_sorting/navigation/README.md) | 代价地图、路径规划、携物包络和安全跟踪。 |
| [`desktop_grasp/`](material_sorting_task/examples/material_sorting/desktop_grasp/README.md) | 双臂预抓取、接触、柔顺预紧和抬升。 |
| [`shelf/`](material_sorting_task/examples/material_sorting/shelf/README.md) | 货架状态、层位几何和放置反馈。 |
| [`executors/`](material_sorting_task/examples/material_sorting/executors/README.md) | 任务 1/2/3 阶段执行器和动作契约。 |
| [`scheduler/`](material_sorting_task/examples/material_sorting/scheduler/README.md) | V2 调度、候选评分、资源安全和事件日志。 |
| [`learning/`](material_sorting_task/examples/material_sorting/learning/README.md) | EventLog 回放、仿真、离线训练和 RL 门禁。 |
| [`material_sorting_task/scripts/README.md`](material_sorting_task/scripts/README.md) | 启动、随机 seed、测试、部署和远程验收。 |
| [`material_sorting_task/semantic_research/README.md`](material_sorting_task/semantic_research/README.md) | Regex/ML/SLM 离线语义研究旁路。 |
| [`reference/`](material_sorting_task/examples/material_sorting/reference/README.md) | 赛方参考和本地仿真参考代码。 |

## 运行方式

比赛默认配置位于 `material_sorting_task/config/competition_release.env`，包括 ROS Domain `102`、CycloneDDS、`task123_full`、`v2/heuristic` 和 YOLO。远程 4090 已准备好 Server/Client 镜像时，推荐使用统一控制脚本：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_random_$(date +%Y%m%d_%H%M%S)

bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" preflight
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" stop
```

终端一启动 Server，终端二启动 Client：

```bash
# 终端一：不传 seed 使用 release env 中的默认 seed
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN"

# 终端二：与 Server 使用同一个 RUN
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" v2
```

每次使用新随机场景时，由 Server 生成或显式传入新的整数 seed；Client 不设置场景 seed：

```bash
export SEED=$(shuf -i 1-999999999 -n 1)
export RUN="v2_random_$(date +%Y%m%d_%H%M%S)_$SEED"
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN" "$SEED"
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" v2
```

运行日志保存到 `$PROJECT/remote_artifacts/$RUN/`。需要回退调度器时，在新终端执行：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" rollback "${RUN}_legacy"
```

具体执行模式、停机条件、容器挂载和验收参数见 [`material_sorting_task/scripts/README.md`](material_sorting_task/scripts/README.md)。

## 测试

在 `material_sorting_task/` 目录下运行：

```bash
bash scripts/run_formal_tests.sh
python3 scripts/check_workspace.py
```

研究旁路单测和离线评估需要额外依赖，且不属于正式控制链：

```bash
bash scripts/run_semantic_research_tests.sh
bash scripts/run_semantic_research_eval.sh
```

`dry_run` 只验证状态机和日志，不代表机器人动作或 Server 得分；`nav_only`、`pregrasp_only`、`contact_only` 和 `lift_only` 会产生真实动作，测试时只能保留一个 Client 和一个机械臂控制源。

## 测试结果边界

历史远程验收记录曾在固定 seed 和五 seed 随机矩阵中达到三任务总分 `160/160`，并验证候选真实应用、重复应用防护、携物包络和 EventLog 回放。该结果属于历史提交和历史日志证据，不自动代表当前工作区或新 seed 已通过；每次修改代码后必须保存当前 commit、Server/Client 镜像、seed、Client/Server 日志和 Scheduler EventLog，并重新运行验收脚本。

## 开发与发布约束

- 修改阶段契约时同步更新执行器、测试和对应模块 README。
- 正式 Client 不导入 `semantic_research`，研究依赖和模型权重不得混入比赛镜像。
- 新调度策略必须先通过硬约束、离线回放和 Shadow，再讨论正式启用。
- 从干净工作区使用 `scripts/freeze_competition_release.sh` 生成带 commit 和 SHA256 的发布包；部署到新目录，不覆盖当前可运行版本。
- 依赖和模型许可证见 `semantic_research/README.md` 及 `semantic_research/MODEL_MANIFEST.json`。
