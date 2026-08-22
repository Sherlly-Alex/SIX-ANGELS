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

## 比赛冻结版本与一键运行

RL-1 离线候选阶段见
[RL_PHASE1.md](material_sorting_task/docs/RL_PHASE1.md)。当前模型已经依次通过模型包、
100 个盲种子配对仿真、两场官方 Client Shadow 和官方 Server Guarded 金丝雀门；冻结包
因此默认启用 `rl_guarded`。RL 仍只能选择通过硬过滤的离散宏动作，任何批准文件、模型、
schema、动作掩码或 25 ms 推理门异常都会 fail-closed。`rollback` 不依赖 RL 模型，直接
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
