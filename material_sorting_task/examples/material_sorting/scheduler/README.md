# Scheduler V2：任务计划、候选决策与安全护栏

Scheduler V2 是正式 Client 的安全优先调度层。它不直接输出 `vx/wz`、关节角或夹爪命令，而是围绕执行器的名义站位生成有限宏动作，在同一份世界状态快照上执行硬过滤和确定性 Multi-Critic 排序。真正采用候选必须经过执行器自己的二次校验。

## 目录职责

| 文件 | 作用 |
| --- | --- |
| `plans.py` | 三任务共享阶段计划、资源集合、不可逆阶段和终止策略。 |
| `engine.py` | 计划驱动的状态机、阶段 tick、命令校验、资源租约、恢复和候选旁路。 |
| `candidate_generator.py` | 生成 center/left/right/recovery 候选。 |
| `project_candidates.py` | 从当前 odometry、目标观测和任务指令构造项目候选批次。 |
| `decision.py` | 候选评估、可选 RL 选择、滞回和稳定性控制。 |
| `utility.py` | 硬约束过滤和 Multi-Critic utility。 |
| `resources.py`、`safety.py` | 资源冲突、命令边界、输入新鲜度和安全检查。 |
| `referee.py` | Server 裁判状态解析、去重和 desync 检测。 |
| `recovery.py` | 有界恢复策略和失败等级。 |
| `events.py` | 结构化 JSONL 事件日志。 |
| `policies/` | heuristic、RL shadow 和 guarded RL 适配器。 |
| `learning/` | observation、回放、训练和模型包验证，默认不进入正式控制依赖。 |

## 决策流程

```text
ExecutionContext
      |
ProjectCandidateProvider
      | 目标、odometry、动态障碍、footprint
      v
CandidateAction(center/left/right/recovery)
      |
WorldCostmapSnapshot + hard constraints
      |
碰撞 / 路径 / 净空 / 资源 / 裁判过滤
      |
Multi-Critic utility ranking
      |
heuristic 或受 mask 约束的 RL suggestion
      |
minimum hold + utility margin + stability frames
      |
executor.apply_scheduler_candidate()
```

候选生成默认横向偏移为 `0`、`+0.08 m`、`-0.08 m`。评分包含成功概率、预计时间、路径长度、障碍物代价、动态风险、航向变化、感知不确定性、操作难度、不可逆风险和恢复成本。决策默认每 0.25 秒更新一次，运行在单独线程，不阻塞 20 Hz 控制循环。

## 硬约束优先

一个候选只有在以下条件通过后才有有限 utility：

- `referee_allowed`、`step_allowed` 和 `resource_available` 为真；
- 输入和目标观测没有过期，pose 坐标为有限值；
- 规划路径可达，沿途 footprint 无碰撞；
- 携物阶段使用 `TRANSIT_CARRY`，不是空载包络；
- 站位横向/纵向偏移和净空满足执行器要求；
- 不可逆阶段不被错误标记为可恢复。

没有安全候选时结果为 `no_safe_candidate`，执行器不接收动作。即使 RL 模型选择候选，也必须位于 action mask 内；模型缺失、哈希/schema 不一致、超时、NaN、越界或 utility regret 超限时回退 heuristic。

## 滞回与幂等

为避免动态地图导致动作抖动，决策服务默认要求：

- 当前动作至少保持 0.75 秒；
- 新动作 utility 比当前动作高出切换边界；
- 新候选连续稳定至少两帧；
- 同一个 `step_run_id + action_id + goal_pose` 只向执行器应用一次。

## Server 裁判同步

`RefereeGateway` 解析 `gameinfo/taskinfo`，检查任务编号回退、跳过任务、尝试次数回退、完成状态回退以及两个 topic 的任务号冲突。一次异常帧只记录；持续异常超过阈值后 fail-closed，不能由 Client 自行猜测下一任务。

## 恢复与状态

阶段结果可以是 `RUNNING`、`SUCCEEDED`、`RETRYABLE_FAILURE`、`BLOCKED` 或 `FAILED`。可恢复失败通过 `RecoveryClassifier` 和有限预算进入重扫描、重规划、退避或阶段重入；碰撞、命令非法、资源冲突、输入过期和内部错误进入安全停止。`PLACE` 为不可逆阶段，不做普通回滚。

## RL 边界

RL 只学习候选排序，不模拟真实动力学，也不生成底盘或机械臂控制量。正式使用顺序是：

1. 先用 heuristic 产生 EventLog。
2. 回放 EventLog，验证 observation schema、action mask 和数据来源。
3. 离线训练和盲测。
4. `rl_shadow` 中零 takeover 验证。
5. 通过 guarded approval 后才考虑 `rl_guarded`。

默认正式配置仍为 `MATERIAL_SCHEDULER_POLICY=heuristic`。

## 运行与验收

```bash
MATERIAL_EXECUTION_MODE=task123_full \
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_SCHEDULER_POLICY=heuristic \
MATERIAL_SCHEDULER_EVENT_LOG=/workspace/artifacts/scheduler.jsonl \
bash scripts/run_client.sh
```

官方固定种子历史记录为任务一 40/40、任务二 60/60、任务三 60/60，总分 160/160；这只是历史证据，当前代码版本和随机种子测试必须重新保存 Client、Server 和 EventLog。新版本验收还应检查至少一次 `application_status=applied`、不重复应用相同候选，以及至少一次非中心候选应用。

## 验收门与回退

一次完整的 V2 验收至少包含以下证据：

1. Client、Server 和 Scheduler EventLog 使用同一个 `RUN` 和同一个 seed，且来自当前 commit。
2. Server 得分、任务终止事件和 Client fatal/safe-hold 计数通过单轮检查。
3. EventLog 中至少有执行器真实应用的候选，并检查重复的 step/action/goal 记录。
4. 启用 measured-carry 时，Task 1 和 Task 3 都有正向实测包络遥测；关闭时不得把旧日志误算成该证据。
5. 多 seed 矩阵逐 seed 检查事件、周期和执行耗时，不能用平均分掩盖单个 seed 失败。

推荐由 `scripts/validate_remote_run.py` 和 `scripts/validate_remote_matrix.py` 生成 JSON 报告，
不要只根据终端中的总分判断调度器成功。V2 发生输入、候选、时序或安全回归时，可保留 Server
并在新终端用 `competitionctl.sh rollback RUN` 重启 Legacy Client；回退不覆盖原日志。

## 相关验收命令

```bash
python3 scripts/replay_scheduler_events.py run1.jsonl run2.jsonl \
  --min-decisions 1000 --require-training-ready \
  --dataset scheduler_replay.jsonl --output replay_report.json

python3 scripts/validate_remote_run.py \
  --client client.log --server server.log --events scheduler.jsonl \
  --require-candidate-application \
  --reject-duplicate-candidate-applications
```

RL 相关的模型包、项目仿真、成对盲测和 guarded approval 只属于离线发布流程，具体门限和命令见
`learning/README.md`。在模型文件、observation schema、provenance、Shadow 和 approval manifest
全部通过前，正式配置必须保持 `MATERIAL_SCHEDULER_POLICY=heuristic`。

三种策略适配器的加载、动作校验、超时隔离和 Heuristic fallback 见
[`policies/README.md`](policies/README.md)。
