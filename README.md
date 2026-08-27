# SIX-ANGELS qzhRL 可迭代原型

本分支保存 SIX-ANGELS 三任务满分执行链及其可选的强化学习调度原型。项目的正式主控制器始终是确定性的 `CompetitionController` 状态机；任务调度与 RL 只在状态机允许的有限宏动作候选中提供辅助选择，不能接管任务流程、机械臂关节或底盘速度。

## 控制权边界

| 层级 | 职责 | 正式地位 |
|---|---|---|
| Server / 裁判 | 发布结构化任务并确认最终得分 | 唯一结果真值 |
| `CompetitionController` 状态机 | Task 1–3 顺序、阶段切换、携物状态、重试、恢复和终止 | 正式主控制器 |
| 执行器与安全检查 | 导航、抓取、放置、碰撞/净空复核 | 强制执行层 |
| V2 Heuristic | 在已经通过硬过滤的宏动作候选中作确定性选择 | 正式可用辅助与 RL 回退 |
| RL Shadow | 只记录建议，不改变执行动作 | 实验模式 |
| RL Guarded | 通过模型、Approval、Action Mask 和运行时护栏后选择宏动作 | 后续可迭代原型，非默认模式 |

无参数启动默认使用 `heuristic`。如需完全绕过 V2 调度扩展，可显式使用 `legacy`；无论选择哪种模式，状态机都保持任务流程控制权。Guarded 必须由操作者显式指定，模型缺失、SHA/Schema 不匹配、推理异常、动作非法或 Approval 无效时不得获得控制权。

## 本版完成的工作

- 保留并同步远程机已验证的 Task 1–3 状态机、导航、感知、抓取和放置实现。
- Task 3 使用已经完成多随机种子验证的整体实现，RL 工作没有改写其抓取/放置动作参数。
- 建立有限宏动作候选、Critic 评分、硬 Action Mask、候选幂等应用和 EventLog。
- 支持 `heuristic`、`legacy`、`shadow`、`guarded` 四种显式运行模式。
- 完成隔离式异步 Shadow 推理，避免 RL 推理阻塞机器人控制循环。
- 训练 contextual-success MaskablePPO 候选排序模型，并绑定模型、训练配置和 observation schema。
- 完成模型包、held-out、500 盲种子、五场 Shadow、Approval 和三场 Guarded 金丝雀验收。
- 新增 `validate_guarded_lineage.py`，可将实际应用候选追溯到 RL 或 Hysteresis 锁存来源。

## 强化学习实现思路

RL 学习的是“从当前安全候选中选择哪个离散宏动作”，而不是端到端运动控制。

1. 状态机确定当前任务和阶段，候选生成器只产生中心、左偏、右偏等有限动作。
2. 碰撞、净空、资源、携物包络、阶段合法性和 Action Mask 在推理前硬过滤候选。
3. Observation 编码任务阶段、候选 Critic、代价图摘要、历史选择和恢复上下文。
4. 训练数据由正式 Shadow 回放和项目仿真组成；正式会话用于严格的 train/validation/test 会话隔离，仿真数据只作为 training-only 辅助数据。
5. `MaskablePPO` 学习候选排序；模型包绑定 observation schema、训练配置和 SHA256。
6. Shadow 只旁路建议；Guarded 还必须通过 Approval、运行时延迟、动态风险和执行前复核。
7. 任何 RL 故障都立即回退 Heuristic，不重启状态机、不清空携物状态、不重复应用动作。

关键代码：

- `material_sorting_task/examples/material_sorting/competition_controller.py`
- `material_sorting_task/examples/material_sorting/scheduler/`
- `material_sorting_task/examples/material_sorting/learning/`
- `material_sorting_task/scripts/rl2_cli.py`
- `material_sorting_task/scripts/competitionctl.sh`
- `material_sorting_task/scripts/validate_guarded_lineage.py`

## 冻结模型与验收结果

当前原型资产：

| 项目 | 值 |
|---|---|
| 模型 | `release_assets/rl_guarded/scheduler_policy.zip` |
| 模型 SHA256 | `5340c47b1fbcfaf799667e1b36a2474e7809817abca78e38875f690a222fb785` |
| Approval | `release_assets/rl_guarded/scheduler_guarded_approval.json` |
| Approval SHA256 | `0f92ad4a1a0039c9dbefc54d3710aeba38910b0aaf259a443db7dd9af9a95f0a` |
| Observation Schema | `b9822b0819a43a7fb9289d4452e9d152df5c8d848cb1327309d4d7f979efa2fd` |

已冻结证据：

| 验收阶段 | 结果 |
|---|---|
| 模型包与 held-out | 通过 |
| 500 个新盲种子 | Heuristic `241/500`，RL `303/500` |
| 500 种子推理延迟 | RL p95 `3.698 ms` |
| 恢复次数 | Heuristic `3.684`，RL `2.638`，改善 `28.39%` |
| 五场 Shadow | `2132` 次建议，运行时 fallback `0`，推理 p95 `1.900 ms` |
| 默认 1000 ms Shadow 金丝雀 | 160 分，fallback `0`，控制 p99 `86.767 ms` |
| Guarded `20260917` | 160 分，无安全/控制异常 |
| Guarded `20260918` | 160 分，8/8 应用候选可追溯至 RL，控制 p99 `103.483 ms` |
| Guarded `20260919` | 160 分，8/8 应用候选可追溯至 RL，控制 p99 `116.167 ms` |

这证明 Guarded 已形成可继续研究的样品，但不意味着它应替代正式状态机或成为默认模式。离线结果中耗时和路径没有改善，因此后续迭代重点是保持成功率与安全的同时优化路径、耗时和接管质量。

完整冻结包位于 [`prototype_release/guarded_v1`](prototype_release/guarded_v1)，其中包含：

- 远程源码快照；
- 模型、metadata 和 Approval；
- 训练/模型包验收；
- 500 盲种子报告；
- 五场 Shadow 原始日志与验收；
- 三场 Guarded 原始日志与验收；
- 逐文件 `SHA256SUMS.txt`。

## 仓库结构

- `material_sorting_task/examples/material_sorting/`：正式状态机、执行器、导航、感知与调度辅助。
- `material_sorting_task/examples/material_sorting/learning/`：训练环境、回放、仿真和 Promotion。
- `material_sorting_task/scripts/`：部署、运行、训练和验收入口。
- `material_sorting_task/docs/`：详细架构、部署和复现文档。
- `release_assets/rl_guarded/`：可选 Guarded 模型、metadata 和 Approval。
- `prototype_release/guarded_v1/`：不可变原型证据快照。
- `prototype_acceptance/`：基于冻结包重新生成的模型与 RL 来源验收结果。

## 环境要求

- Ubuntu + ROS 2 Humble
- Docker + NVIDIA Container Toolkit
- 可用 GPU 和 X11 显示环境（官方 Server 仿真需要）
- 已加载镜像：
  - `material_sorting:offline-server`
  - `material_sorting:offline-client`
  - `material_sorting:offline-client-rl-shadow-e3f5284`

基础检查：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-qzhRL
cd "$PROJECT"
python3 material_sorting_task/scripts/check_workspace.py
bash -n material_sorting_task/scripts/competitionctl.sh
```

## 正式运行：状态机 + Heuristic

正式运行默认不启用 RL：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-qzhRL
export DISPLAY=:1
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority

xhost +SI:localuser:root
cd "$PROJECT"
bash material_sorting_task/scripts/competitionctl.sh stop
bash material_sorting_task/scripts/competitionctl.sh preflight heuristic
```

终端一启动 Server：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-qzhRL
export RUN=competition_heuristic_$(date +%Y%m%d_%H%M%S)
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN"
```

终端二使用同一个 `RUN`。省略第三个参数时同样默认 Heuristic：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-qzhRL
export RUN=competition_heuristic_YYYYMMDD_HHMMSS
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" heuristic
```

完全绕过 V2 调度扩展：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" legacy
```

## 可选 Guarded 原型

先验证冻结资产：

```bash
cd "$PROJECT"
sha256sum release_assets/rl_guarded/scheduler_policy.zip
sha256sum release_assets/rl_guarded/scheduler_guarded_approval.json
bash material_sorting_task/scripts/competitionctl.sh preflight guarded
```

只有显式写出 `guarded` 才会启用 RL：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" guarded
```

发现任何异常时停止 Client，并以新 `RUN` 启动 Heuristic；不删除原始日志：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" rollback "${RUN}_heuristic"
```

详细步骤见：

- [Guarded 原型部署](material_sorting_task/docs/GUARDED_PROTOTYPE_DEPLOYMENT.md)
- [Guarded 原型复现与验收](material_sorting_task/docs/GUARDED_PROTOTYPE_REPRODUCTION.md)
- [RL-2 成功率优先设计](material_sorting_task/docs/RL2_SUCCESS_FIRST.md)
- [RL-2 远程执行](material_sorting_task/docs/RL2_REMOTE_EXECUTION.md)
- [RL-2.1 Contextual 结果](material_sorting_task/docs/RL2_1_CONTEXTUAL_OUTCOME.md)

## 安全与证据边界

- Server/裁判状态是最终结果真值，Client 本地完成日志不能替代裁判确认。
- Guarded 只选择有限宏动作，不直接输出底盘速度或机械臂关节命令。
- Action Mask、安全检查和执行器复核不能被 RL 绕过。
- Task 1–3 已验证动作逻辑不因 RL 训练或部署而修改。
- 正式模式保持 Heuristic；Guarded 仅供显式实验和后续迭代。
- 500 盲种子显示成功率和恢复次数改善，但耗时、路径没有改善，不作超出证据的宣传。
