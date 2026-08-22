# SIX-ANGELS

文旅搬运赛题机器人项目。正式比赛链路以 Server 发布的结构化
`/material/instruction` JSON 为唯一执行依据；导航、感知、抓取、放置和裁判同步均不依赖本地 NLP 模型。

## 目录

- `material_sorting_task/examples/material_sorting/`：正式 ROS 2 客户端、控制器和任务执行器。
- `material_sorting_task/examples/material_sorting/scheduler/`：Legacy/Shadow/V2 调度、资源安全、代价评价与策略护栏。
- `material_sorting_task/examples/material_sorting/learning/`：离散宏动作训练、评估与 Guarded 运行时护栏。
- `material_sorting_task/semantic_research/`：离线 Regex、ML 和本地 LLM 研究模块。
- `material_sorting_task/scripts/`：正式测试、研究测试和可选模型安装脚本。
- `SEMANTIC_PARSING_IMPLEMENTATION_PLAN.md`：语义解析分阶段实施与审查记录。
- `TASK_SCHEDULING_IMPLEMENTATION_PLAN_0813.md`：调度架构、当前落地状态和逐步实机放行方案。

## 正式运行原则

正式客户端严格校验 Server JSON 的任务字段，缺失或冲突时拒绝执行；不会从中文文本猜测执行字段，也不会由 ML/LLM 补全 `target_body`、`place_world` 或 `place_radius`。

语义研究模块在 ROS 客户端中默认关闭。启用旁路审计后，Regex/ML/LLM 只对已接受的 Server JSON 做一致性对比，并输出 `SEM_AUDIT` 日志；它们不能修改任务、拒绝任务或阻塞控制器。

## qzhRL 版实现总览

本分支是在已经通过官方 Server 160/160 验收的任务执行链上增加受约束任务调度，而不是
让神经网络直接控制底盘速度或机械臂关节。主要完成内容如下：

- 将 Task 1～3 拆分为导航、目标获取、预抓取、接触、抬升、运输、放置、验证和返航阶段，
  由统一状态机管理尝试次数、超时、裁判分数和安全终止。
- 建立 `WorldCostmap` 全局代价地图，融合静态边界、货架/桌面、动态障碍、机器人足迹与
  携物包络；候选动作必须先通过碰撞、净空、资源和任务阶段硬约束。
- 为每个决策生成中心/左偏/右偏等有限宏动作，使用 Multi-Critic 对路径长度、风险、切换
  代价、任务进度和预计回报评分；执行器应用候选前仍会重新规划和复核。
- 所有候选、action mask、observation schema、策略选择、应用状态和运行健康指标写入
  `scheduler_*.jsonl`，支持确定性回放、数据集生成和远程验收。
- 完成 RGB-D 物块定位、双臂柔顺抓取、货架空层双重确认、Task 3 放置回撤，以及运行时
  输入健康、幂等候选应用、测量携物保护等比赛链路修复。
- 提供 `legacy`、V2 Heuristic、`rl_shadow`、`rl_guarded` 四级运行方式，并保留一键
  Heuristic 回退；任何 RL 异常都不会绕过硬 action mask 或安全执行器。

### 强化学习实现思路

RL 解决的是“当前安全候选中选择哪个宏动作回报最高”，不是端到端运动控制：

1. `ObservationBuilder` 将任务阶段、候选 Critic、代价图摘要、资源状态和历史选择编码为
   固定 schema；无效或不存在的动作由硬 action mask 禁用。
2. 从五场启发式多随机种子 EventLog 回放得到 4,745 条训练可用决策。训练环境按单次候选
   选择建模为 contextual bandit，避免把互不相干的日志记录错误地做跨步信用分配。
3. 使用 Stable-Baselines3 Contrib `MaskablePPO` 学习离散候选排序；模型包同时绑定训练
   配置、数据集、observation schema 和模型 SHA256，运行时不允许静默替换。
4. 候选模型依次通过离线 held-out、100 个盲种子配对仿真、两场官方 Client Shadow、
   Guarded 批准清单和官方 Server 金丝雀门，之后才获得实际选择权限。
5. `rl_guarded` 中每次推理仍由 `PolicyGuard` 检查模型哈希、schema、mask、输出范围和
   安全下界。单次超过 50 ms 立即改用 Heuristic；连续 3 次超时才隔离 RL，成功推理会
   清零计数。正式验收要求有效推理 p95 ≤25 ms、孤立超时 ≤2、隔离事件为 0。

最终官方 Server 恢复金丝雀结果为 160/160、774 次 RL 实际接管、930 次有效推理、
推理 p95 5.17 ms、1 次孤立超时成功恢复、0 次永久隔离。模型只负责安全候选排序，最终
动作始终由确定性安全层和任务执行器批准。

关键实现位置：

- `material_sorting_task/examples/material_sorting/scheduler/`：候选、Critic、代价图与策略护栏。
- `material_sorting_task/examples/material_sorting/learning/`：action mask、训练环境、模型加载与晋级链。
- `material_sorting_task/scripts/replay_scheduler_events.py`：EventLog 回放与训练集生成。
- `material_sorting_task/scripts/validate_rl_guarded.py`：官方 Guarded 运行验收。
- `material_sorting_task/docs/GUARDED_POLICY_PROMOTION.md`：模型晋级和运行时安全边界。

## 比赛冻结版本与一键运行

RL-1 离线候选阶段见
[RL_PHASE1.md](material_sorting_task/docs/RL_PHASE1.md)。当前模型已经依次通过模型包、
100 个盲种子配对仿真、两场官方 Client Shadow 和官方 Server Guarded 金丝雀门；冻结包
因此默认启用 `rl_guarded`。RL 仍只能选择通过硬过滤的离散宏动作，任何批准文件、模型、
schema 或动作掩码异常都会 fail-closed。正式运行的单次推理硬超时为 50 ms：孤立超时
立即回退 Heuristic，连续 3 次超时才隔离 RL；验收仍要求有效推理 p95 不超过 25 ms，且
最多允许 2 次孤立超时、零隔离事件。`rollback` 不依赖 RL 模型，直接
恢复已经独立获得 160 分验收的 `v2 / heuristic`。

比赛默认配置固定在 `material_sorting_task/config/competition_release.env`。当前冻结基线为：

| 项目 | 冻结值 |
|---|---|
| 正式执行模式 | `task123_full` |
| 调度器 | `v2 / rl_guarded`（一键回退 `v2 / heuristic`） |
| 测量搬运保护 | `0`（默认关闭，实验功能不进入比赛链） |
| ROS Domain | `102` |
| 已验收得分 | 官方 Server `160/160` |
| 验收基线提交 | `e3f5284` |
| Guarded 模型 | `364d5cf5...e3322` |
| Guarded 批准清单 | `4aa38963...d59ac` |
| 推理保护 | 50 ms 单次硬超时；连续 3 次才隔离；验收 p95 ≤25 ms |

### 部署选择

推荐使用经过哈希校验的冻结包部署，而不是直接把开发工作区当成比赛目录：

```text
SIX-ANGELS-competition-6cdf5b4.tar.gz
SHA256: EF2ED940E89FB0D83DC439D85340EA94A95D03B450EFCCDBBFCDC1C2A6B21CAF
```

Git 分支不提交模型二进制。若从 `qzhRL版` 克隆源码，必须另外提供冻结包中的
`release_assets/rl_guarded/` 模型、metadata、批准清单和证据文件，Guarded 预检才会
通过；项目不会联网下载或自动替换它们。只运行 V2 Heuristic 时不需要这些 RL 资产。

源码检出与基础检查：

```bash
git clone --branch 'qzhRL版' --single-branch \
  https://github.com/qzhvscode/SIX-ANGELS.git SIX-ANGELS-qzhRL
cd SIX-ANGELS-qzhRL
python3 material_sorting_task/scripts/check_workspace.py
```

正式部署还要求远程机已有以下验收镜像：

- `material_sorting:offline-server`
- `material_sorting:offline-client-rl-shadow-e3f5284`
- `material_sorting:offline-client`（Heuristic 回退）

远程机先设置项目目录并做预检：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" preflight guarded
```

正式运行使用两个终端。终端一启动官方 Server（前台运行并保存日志）：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=competition_$(date +%Y%m%d_%H%M%S)
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN"
```

终端二使用同一个 `RUN` 启动正式 Client：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=competition_YYYYMMDD_HHMMSS
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" guarded
```

运行日志统一写到 `$PROJECT/remote_artifacts/$RUN/`。查看状态或停止容器：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" status
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" stop
```

### 一键冻结、部署与回退

从干净的 Git 工作区生成不可变比赛包和 SHA256。Guarded 冻结还必须显式提供六个已经
通过的模型/证据文件；冻结脚本会校验模型和批准清单哈希、检查四份报告的 `passed=true`，
并将它们一并封装：

```bash
export MATERIAL_FREEZE_MODEL_SOURCE=/path/to/scheduler_policy.zip
export MATERIAL_FREEZE_APPROVAL_SOURCE=/path/to/scheduler_guarded_approval.json
export MATERIAL_FREEZE_BENCHMARK_SOURCE=/path/to/project_sim_blind_report.json
export MATERIAL_FREEZE_SHADOW_SOURCE=/path/to/rl_shadow_acceptance.json
export MATERIAL_FREEZE_GUARDED_ACCEPTANCE_SOURCE=/path/to/guarded_policy_acceptance.json
export MATERIAL_FREEZE_REMOTE_ACCEPTANCE_SOURCE=/path/to/remote_acceptance.json
bash material_sorting_task/scripts/competitionctl.sh freeze
```

部署时必须使用一个新的空目录，不覆盖当前可运行版本：

```bash
bash material_sorting_task/scripts/deploy_competition_release.sh \
  /home/abc123/SIX-ANGELS-competition-COMMIT.tar.gz \
  /home/abc123/polaris/workspace/SIX-ANGELS-release-COMMIT
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-release-COMMIT
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" preflight guarded
```

如果 Guarded 调度现场异常，保留 Server，只在新终端重启 Client 并切换到不需要模型或
批准文件的 V2 Heuristic：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=competition_rollback_$(date +%Y%m%d_%H%M%S)
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" rollback "$RUN"
```

此回退只切换策略，不替换动作代码、不删除原日志。需要进一步回退调度引擎时可显式运行
`client "$RUN" legacy`。若需回退整个代码版本，重新部署上一个已校验归档到另一个目录，
再修改 `PROJECT`；不要覆盖或删除当前目录。

Guarded 比赛完成后可用仓库内正式验收器生成结构化报告：

```bash
python3 "$PROJECT/material_sorting_task/scripts/validate_rl_guarded.py" \
  "$PROJECT/remote_artifacts/$RUN/scheduler_$RUN.jsonl" \
  --expected-model-sha256 364d5cf5e94be08597cd9bde643b1ed132ab347ec520bd8e16d2d24fc68e3322 \
  --minimum-rl-takeovers 1 \
  --maximum-inference-p95-ms 25 \
  --maximum-isolated-timeouts 2 \
  --output "$PROJECT/remote_artifacts/$RUN/guarded_policy_acceptance.json"
```

## 环境与测试

正式代码需要 ROS 2 Humble、项目提供的仿真容器和对应 GPU/渲染环境。研究依赖不要安装进正式比赛镜像：

```bash
cd material_sorting_task
python -m pip install -r semantic_research/requirements-research.txt

# 正式回归测试
bash scripts/run_formal_tests.sh

# 研究模块测试
bash scripts/run_semantic_research_tests.sh
```

## 模型配置（可选）

模型文件不提交到 Git。克隆仓库后按需安装：

```bash
cd material_sorting_task

# 只重训 ML；使用仓库内 train split，不接触 test split
bash scripts/setup_semantic_research.sh --runtime --ml

# 需要本地 LLM 时，再下载约 2.1 GB 的 GGUF 并自动校验 SHA256
bash scripts/setup_semantic_research.sh --slm
```

Windows PowerShell：

```powershell
cd material_sorting_task
.\scripts\setup_semantic_research.ps1 -Runtime -ML
.\scripts\setup_semantic_research.ps1 -SLM
```

当前研究模型配置：

| 项目 | 配置 |
|---|---|
| LLM | Qwen2.5-3B-Instruct-GGUF Q4_K_M |
| 文件 | `semantic_research/artifacts/slm/qwen2.5-3b-instruct-q4_k_m.gguf` |
| 大小 | 2,104,932,768 bytes（约 2.1 GB） |
| 推理后端 | `llama-cpp-python`，CPU 可运行；无独显时速度较慢 |
| 上下文窗口 | 1024 |
| 用途 | 离线语义研究/旁路审计，不参与正式控制 |
| 权重许可 | Qwen Research License，使用或再分发前请阅读许可 |

ML 模型由以下命令从 `train` split 生成：

```bash
PYTHONPATH=. python -m semantic_research.train_ml \
  --dataset semantic_research/data/text_eval.jsonl \
  --splits train --seed 7 \
  --out semantic_research/artifacts/ml_slots_v2.joblib
```

生成文件：

- `semantic_research/artifacts/ml_slots_v2.joblib`
- `semantic_research/artifacts/ml_slots_v2.meta.json`

元数据必须显示 `train_splits=["train"]` 和 `includes_test=false`。

## ROS 旁路审计配置

默认关闭。启用 ML 审计：

```bash
export MATERIAL_SEMANTIC_AUDIT=1
export MATERIAL_SEMANTIC_AUDIT_ML_MODEL=/workspace/baseline/semantic_research/artifacts/ml_slots_v2.joblib
```

调试时才建议启用 CPU LLM：

```bash
export MATERIAL_SEMANTIC_AUDIT_SLM=1
export MATERIAL_SEMANTIC_AUDIT_SLM_WEIGHTS=/workspace/baseline/semantic_research/artifacts/slm/qwen2.5-3b-instruct-q4_k_m.gguf
```

权重下载地址、固定 revision 和 SHA256 记录在
`material_sorting_task/semantic_research/MODEL_MANIFEST.json` 中。

## 模型结果边界

ML/LLM 指标用于研究和错误分析，不代表正式机器人控制正确率。正式比赛仍以结构化 Server JSON、确定性校验和已验证的任务执行代码为准。
