# SIX-ANGELS

文旅搬运赛题（DG-202612）机器人项目。仓库维护参赛 **Client**：指令校验、感知、导航、抓取、放置、三任务状态机、调度辅助、可选强化学习和验收脚本。官方 **Server**、场景随机化与裁判由赛方 Docker 镜像提供，不在本仓库修改。

正式比赛只信任 Server 发布的结构化 `/material/instruction` JSON 和 `/referee/*`。Client 不从中文文本猜测执行字段，也不用 NLP / ML / LLM 补全 `target_body`、`place_world` 或 `place_radius`。底盘速度和机械臂关节始终由确定性执行器发出。

---

## 目录

1. [赛题](#赛题)
2. [系统结构](#系统结构)
3. [三任务流程](#三任务流程)
4. [代码布局](#代码布局)
5. [主要模块](#主要模块)
6. [环境与依赖](#环境与依赖)
7. [运行](#运行)
8. [调度与强化学习](#调度与强化学习)
9. [可选功能](#可选功能)
10. [测试与验收](#测试与验收)
11. [文档](#文档)

---

## 赛题

一局长生命周期 Client 连续完成三个任务，默认限时 600 s。中途崩溃或主动退出记 0 分。每个任务最多 3 次尝试，取最高分。裁判用 MuJoCo 地面真值计分，Client 日志不能替代裁判。

| 任务 | 满分 | 内容 |
|---|---|---|
| 1 | 40 | 桌面取指定颜色盒，放到货架空层，返回结束区 |
| 2 | 60 | 从货架取指令颜色盒，放回任务 1 的桌面原位，返回结束区 |
| 3 | 60 | 再从桌面取盒，放到货架白色长方体包装盒左侧，返回结束区 |
| 合计 | 160 | 任务切换只由裁判确认 |

世界系：+X 东、+Y 北。颜色白名单：`pink` / `yellow` / `brown`。官方环境没有 2D 激光。结束区大致为 `x ∈ [-1.15, -0.25]`、`y ∈ [0.10, 1.00]`。

判分点（与 `reference/server/referee.py` 一致）：

- 任务 1：触碰 10 + 夹起 10 + 放置 10 + 无结构碰撞回到结束区 10
- 任务 2 / 3：触碰 20 + 夹起 10 + 放置 20 + 回结束区 10

---

## 系统结构

```text
/material/instruction
/referee/taskinfo, gameinfo, score
/odom, /joint_states
/camera/*, /material/detections
        │
        ▼
 client_task.py          ROS 生命周期、20 Hz 控制循环
        │
        ▼
 CompetitionController   三任务顺序、阶段、重试、恢复、终止
        │
        ├── Task 1 / 2 / 3 执行器
        ├── NavigationController（全局 A*、足迹、限速、急停）
        ├── TransferMotion（直行、转向、货架前列移）
        ├── desktop_grasp + IK
        └── 货架状态 / 柔顺放置
                │
                ▼
        /cmd_vel、升降柱、头部、双臂
```

控制权分层：

| 层 | 职责 |
|---|---|
| Server / 裁判 | 发布任务、结算尝试、给出最终分 |
| `CompetitionController` | 任务与阶段主控 |
| 执行器与安全检查 | 导航、抓取、放置、碰撞与净空 |
| V2 Heuristic | 在已通过硬过滤的宏动作候选中作确定性选择（默认） |
| RL Shadow / Guarded | 只建议或在审批链通过后选择宏动作，不输出速度或关节 |

`client_task.py` 必须撑过整局；任何错误先停底盘。纯逻辑可在无 ROS 的单元测试中运行。`examples/material_sorting/reference/` 是赛方示例，正式 Client 不导入。

三任务共用阶段（`executors/base.py`）：

```text
navigate_to_pick → acquire_target → align_for_pick → grasp → lift
 → transport → align_for_place → place → verify_place → return_to_end
```

控制器状态：`waiting_for_inputs` → `starting_task` → `executing_stage` → `waiting_for_referee`，以及 `blocked` / `safe_hold` / `finished`。

---

## 三任务流程

### 任务 1（40）

1. 等待指令、里程计、关节就绪。
2. YOLO（或颜色后端）给出任务 1 颜色的稳定世界坐标，匹配左右桌边槽位；超出容差则停车，避免把白色固定块或顶面深度误差传给 IK。
3. 静态场景 A* 导航到桌边抓取站。
4. 张开双爪预抓取 → 有界向内预紧（腕部柔顺；官方镜像不提供 `/material/grasp_confirmed`）→ 升降轴抬升。
5. 直线后撤，经货架东侧安全点转西，再到货架外观察站。观察站由携物几何推出，使下层白色包装盒保持可见。
6. 运输过程中融合货架观测。需要一层彩色盒与白色包装盒落在不同层（L1–L3），忽略手中颜色，剩余唯一层作为空层。快照保留三个坐标：空层中心、货架彩色盒中心、白色包装盒中心。
7. 货架前列移对齐空层，直线进入，合规下降放到层板，张爪后撤，收回运输姿态，返回结束区。
8. 将桌面原坐标写入进程记忆，供任务 2 放回。

分段模式：`nav_only`、`pregrasp_only`、`contact_only`、`lift_only`、`task1_full`。

### 任务 2（60）

1. 指令颜色必须与任务 1 货架快照一致。
2. 缓存层位只用于粗导航。在臂预备站张开并放低双臂，用带帧时间戳的新 RGB-D 锁定目标盒三维中心；进入该站时清空该颜色滚动历史。
3. 货架外先横向对齐，再直线靠近到可达距离，执行货架抓取与有界抬升。成功后不再二次 IK、不再向内挤压。
4. 面西沿巷道退到任务 1 桌列，转向后直线进入南桌入口，放到保存的桌面坐标。靠近东墙的最后一段不走网格规划，避免栅格吸附导致扫臂。
5. 返回结束区。

### 任务 3（60）

1. 复用任务 1 的桌面抓取与抬升。
2. 放置位由 Client 测到的白色长方体中心计算左侧目标，不抄 Server 坐标。
3. 导航到货架前观察/对齐站，沿货架行向横向对准左侧站位，再直线进入层板。
4. 浅插入后继续底盘推进一段距离，合规下降、释放、确认。
5. 后撤、收回姿态，返回结束区。

---

## 代码布局

```text
.
├── README.md
├── SEMANTIC_PARSING_IMPLEMENTATION_PLAN.md
├── TASK_SCHEDULING_IMPLEMENTATION_PLAN_0813.md
├── release_assets/rl_guarded/          可选调度策略模型与 Approval
├── prototype_release/                 原型冻结证据
├── prototype_acceptance/               基于冻结包的验收结果
└── material_sorting_task/             Client 工作区
    ├── config/competition_release.env
    ├── Dockerfile
    ├── discoverse/
    ├── docs/
    ├── patches/
    ├── scripts/
    ├── semantic_research/              离线语义研究（正式链不导入）
    ├── tests/
    └── examples/material_sorting/
        ├── client_task.py
        ├── competition_controller.py
        ├── instruction_parser.py
        ├── task_orchestration.py
        ├── runtime_health.py
        ├── semantic_audit.py
        ├── arm_kdl.py / mmk2_kdl.py
        ├── desktop_grasp/
        ├── executors/
        ├── navigation/
        ├── perception/
        ├── shelf/
        ├── scheduler/
        ├── learning/
        ├── mjcf/  models/
        └── reference/
```

挂载到 Client 容器时，工作目录为 `/workspace/baseline/material_sorting_task`。

---

## 主要模块

### 入口与编排

| 路径 | 作用 |
|---|---|
| `client_task.py` | ROS 2 节点：订阅指令/裁判/本体/检测，发布运动命令，写调度事件 JSONL |
| `competition_controller.py` | 无 ROS 的三任务状态机 |
| `instruction_parser.py` | 校验 Server JSON；中文只做冲突检测 |
| `task_orchestration.py` | 任务顺序与共享记忆辅助 |
| `runtime_health.py` | 控制环间隔/执行耗时；输入过期则停底盘并保持手臂 |
| `scene_grounding.py` / `scene_context_adapters.py` | 场景槽位与桌边几何 |

### 执行器

| 路径 | 作用 |
|---|---|
| `executors/base.py` | 阶段、`StageResult`、`ExecutionContext` |
| `executors/task1.py` | 导航、预抓取、接触、抬升（可分段） |
| `executors/task1_full.py` | 运输、货架识别、空层放置、回结束区 |
| `executors/task2.py` | 货架抓取与桌面原位放置 |
| `executors/task3.py` | 桌面再抓与包装盒左侧放置 |
| `executors/transfer_support.py` | 直行、转向、货架前列移 |
| `executors/scheduler_candidate.py` | 执行器侧再次校验后才应用调度候选 |
| `executors/dry_run.py` | 不驱动机器人的状态机演练 |
| `executors/local_map_motion.py` | 将局部地图建议转为有限速度倍率 |

### 导航

`navigation/navigation_controller.py` 负责全局 A*、路径平滑与校验、足迹、限速、急停和动态障碍叠加。`costmap/` 维护版本化世界代价图。携物时 `carried_envelope.py` 把盒子扫掠体并入碰撞检查。直线平移和原地转向在发速度前检查身体、手臂、盒子是否扫到货架或围墙。速度在执行器和 ROS 发布口双重限制。

### 感知

| 路径 | 作用 |
|---|---|
| `perception/box_detect.py`、`backends.py` | YOLO 或颜色检测，稳定世界系目标 |
| `perception/checkpoints/` | 默认 YOLO 权重 |
| `perception/depth_geometry.py` | RGB-D 几何中心 |
| `perception/shelf_empty_confirm.py` | 空层确认 |
| `perception/local_map.py`、`local_map_sidecar.py` | 可选局部占用栅格 |

货架 L1 被遮挡时，包装箱或空层证据只用于触发一次补充观察；任务 2 目标仍必须来自彩色箱的直接 RGB-D。

### 抓取与货架

`desktop_grasp/pregrasp_core.py` 提供标定双臂 IK。`shelf/state_tracker.py` 融合层与颜色；`shelf/task3_geometry.py` 由白色包装盒中心计算左侧放置与安全释放；`shelf/placement_feedback.py` 合规下降；`shelf/manipulation.py` 保持运输姿态。

### 调度

`scheduler/` 在导航阶段后台更新代价图，生成中心 / 左偏 / 右偏等有限站位，先做碰撞、携物包络、资源与裁判硬过滤，再做确定性评分。评分不阻塞 20 Hz 控制循环，也不直接发布 `/cmd_vel`。只有实现了应用接口的执行器才会改站位；与名义轨迹不一致的候选记为 `audit_only`。

策略文件在 `scheduler/policies/`：`heuristic.py`、`rl.py`、`guard.py`。

### 学习

`learning/` 把调度当成受 mask 约束的离散候选排序问题，而不是端到端运动控制。包含回放环境、仿真采集、MaskablePPO 训练、模型打包、Shadow 门与 Promotion。任意推理失败都回退 Heuristic，不重启状态机、不清空携物状态。

---

## 环境与依赖

- Ubuntu、ROS 2 Humble
- Docker、NVIDIA Container Toolkit
- 官方仿真需要 GPU 与 X11
- 镜像：`material_sorting:offline-server`、`material_sorting:offline-client`；Guarded 另用带隔离推理的 Client 镜像

正式通信：

```text
ROS_DOMAIN_ID=102
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

```bash
export PROJECT=/path/to/SIX-ANGELS
cd "$PROJECT"
python3 material_sorting_task/scripts/check_workspace.py
```

研究依赖（sklearn、可选 GGUF）只装在 `semantic_research`，不要打进比赛 Client 镜像。

---

## 运行

优先使用 `material_sorting_task/scripts/competitionctl.sh`，由 `config/competition_release.env` 注入冻结默认值。默认 Client 模式为 **heuristic**。

```bash
export PROJECT=/path/to/SIX-ANGELS
export DISPLAY=:1
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
export RUN=run_$(date +%Y%m%d_%H%M%S)

xhost +SI:localuser:root
cd "$PROJECT"
bash material_sorting_task/scripts/competitionctl.sh stop
bash material_sorting_task/scripts/competitionctl.sh preflight heuristic
bash material_sorting_task/scripts/competitionctl.sh server "$RUN"
```

另一终端（同一 `RUN`）：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" heuristic
```

| 命令 | 含义 |
|---|---|
| `client RUN heuristic` | 默认正式路径 |
| `client RUN legacy` | 绕过 V2 调度扩展，状态机仍主控 |
| `client RUN shadow` | Heuristic 控制，RL 只记建议 |
| `client RUN guarded` | 显式启用已审批的 RL 选动作 |
| `rollback RUN` | 始终用 Heuristic 镜像重启，不要求模型 |

容器内等价入口：`bash scripts/run_client.sh`。无 YOLO 权重时可用 `MATERIAL_DETECT_BACKEND=color` 检查通信。同一时刻只跑一个控制 Client，不要并行桌面抓取脚本或其他发布 `/cmd_vel` 的节点。

`MATERIAL_EXECUTION_MODE`：

| 值 | 用途 |
|---|---|
| `stub` | 未实现阶段安全阻塞 |
| `dry_run` | 不运动，只跑状态机 |
| `nav_only` 等 | 任务 1 分段实动 |
| `task1_full` / `task12_full` / `task123_full` | 整合运行（发布配置为三任务） |

常用环境变量见 `material_sorting_task/config/competition_release.env`。命令行 `export` 覆盖文件。主要项：

| 变量 | 含义 |
|---|---|
| `MATERIAL_EXECUTION_MODE` | 执行范围 |
| `MATERIAL_SCHEDULER_ENGINE` | `legacy` / `v2` |
| `MATERIAL_DETECT_BACKEND` | `yolo` / `color` |
| `MATERIAL_YOLO_CHECKPOINT` | 权重路径 |
| `MATERIAL_SCHEDULER_EVENT_LOG` | 调度 JSONL |
| `MATERIAL_LOCAL_MAP` / `MATERIAL_LOCAL_MAP_APPLY` | 局部建图观测 / 有限应用 |
| `MATERIAL_SEMANTIC_AUDIT` | 语义旁路日志 |
| `MATERIAL_RL_*` | 仅 Guarded：模型路径、SHA256、超时与隔离 |

---

## 调度与强化学习

默认不加载策略网络。强化学习只在已经通过硬过滤的宏动作里排序，不能输出 `vx/wz` 或关节。

训练与门禁入口：

- `scripts/rl2_cli.py`、`scripts/rl2ctl.sh`：仿真采集、官方采集、覆盖审计、训练、盲测
- `scripts/replay_scheduler_events.py`：EventLog 回放为训练数据
- `scripts/train_scheduler_policy.py`：MaskablePPO
- `scripts/validate_scheduler_model.py`、`validate_rl_shadow.py`、`validate_rl_guarded.py`、`validate_guarded_lineage.py`

模型、metadata 与 Approval 在 `release_assets/rl_guarded/`。哈希或审批不一致时运行时拒绝 RL 权限。详细步骤见 `docs/RL2_SUCCESS_FIRST.md`、`docs/RL2_REMOTE_EXECUTION.md`、`docs/GUARDED_PROTOTYPE_DEPLOYMENT.md`。

---

## 可选功能

**局部建图**（默认关闭）：`MATERIAL_LOCAL_MAP=1` 只计算并记录建议；再开 `MATERIAL_LOCAL_MAP_APPLY=1` 才允许把少数运输站位外推或调整速度倍率。数据缺失或过期时保持原参数。设计见 `docs/LOCAL_MAP_STRATEGY.md`。

**语义研究**（默认关闭）：`semantic_research/` 用 Regex、逻辑回归和可选本地 Qwen GGUF 抽取中文槽位，禁止预测放置坐标。正式代码不得导入该包。`MATERIAL_SEMANTIC_AUDIT=1` 只对已接受的 Server JSON 打对比日志。模型安装见 `semantic_research/README.md`。

---

## 测试与验收

正式单测（不含 `tests/semantic_research`，无需研究依赖）：

```bash
cd material_sorting_task
bash scripts/run_formal_tests.sh
python3 scripts/check_workspace.py
```

研究模块：

```bash
bash scripts/run_semantic_research_tests.sh
```

官方 Server 多种子采集：

```bash
python3 -S material_sorting_task/scripts/rl2_cli.py collect-official \
  --output-root "$OUT" --mode heuristic --seeds 20260817 20260818 20260819 20260820 20260821
```

产物：`$OUT/v2_multiseed_<seed>/{client,server,scheduler}_*.{log,jsonl}`。整表结束后才有 `collect_status.json`。

单局验收：

```bash
python3 material_sorting_task/scripts/validate_remote_run.py \
  --client client.log --server server.log --events scheduler.jsonl \
  --output acceptance.json
```

通过要求：任务 1 累计 40、任务 2 累计 100、Client `controller=finished task=3 score=160`、Server `all_tasks_done`、无 `blocked` / `safe_hold` / 执行器错误 / 非道具碰撞。带 `--events` 时还检查控制环满窗：interval p95 / p99、execution p95 与 deadline miss 率。多种子聚合用 `scripts/validate_remote_matrix.py`。局内时长见 Server 日志 `>>>>>> …s: 本局结束`。

验收标准说明：`docs/REMOTE_FULL_SCORE_ACCEPTANCE.md`、`docs/RUNTIME_HEALTH_REMOTE_VALIDATION.md`。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`material_sorting_task/README.md`](material_sorting_task/README.md) | Client 工作区、分段实动与调度训练命令 |
| [`docs/ARCHITECTURE.md`](material_sorting_task/docs/ARCHITECTURE.md) | Client 进程边界 |
| [`docs/DEVELOPMENT.md`](material_sorting_task/docs/DEVELOPMENT.md) | 实现约束 |
| [`docs/SHELF_TASK12_INTEGRATION.md`](material_sorting_task/docs/SHELF_TASK12_INTEGRATION.md) | 任务 1/2 货架流程 |
| [`docs/NAVIGATION_V3_INTEGRATION.md`](material_sorting_task/docs/NAVIGATION_V3_INTEGRATION.md) | 导航接入 |
| [`docs/DESKTOP_GRASP.md`](material_sorting_task/docs/DESKTOP_GRASP.md) | 桌面抓取 |
| [`docs/COMPLIANT_GRASP.md`](material_sorting_task/docs/COMPLIANT_GRASP.md) | 柔顺抓取 |
| [`docs/LOCAL_MAP_STRATEGY.md`](material_sorting_task/docs/LOCAL_MAP_STRATEGY.md) | 局部建图 |
| [`docs/SEMANTIC_PARSING.md`](material_sorting_task/docs/SEMANTIC_PARSING.md) | 语义研究边界 |
| [`docs/RL2_SUCCESS_FIRST.md`](material_sorting_task/docs/RL2_SUCCESS_FIRST.md) | RL 运行边界 |
| [`docs/RL2_REMOTE_EXECUTION.md`](material_sorting_task/docs/RL2_REMOTE_EXECUTION.md) | 远程采集与训练 |
| [`docs/GUARDED_PROTOTYPE_DEPLOYMENT.md`](material_sorting_task/docs/GUARDED_PROTOTYPE_DEPLOYMENT.md) | Guarded 部署 |
| [`docs/REMOTE_FULL_SCORE_ACCEPTANCE.md`](material_sorting_task/docs/REMOTE_FULL_SCORE_ACCEPTANCE.md) | 满分验收 |
| [`semantic_research/README.md`](material_sorting_task/semantic_research/README.md) | Regex / ML / SLM |
