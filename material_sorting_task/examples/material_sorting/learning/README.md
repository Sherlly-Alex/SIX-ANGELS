# 调度学习与离线回放

`learning/` 是 Scheduler V2 的可选离线学习和回放模块。它不进入默认正式 Client 的必需依赖，不下载模型，不直接控制机器人。它学习的是已经通过安全硬过滤的有限宏动作排序。

## 主要模块

| 模块 | 作用 |
| --- | --- |
| `observation.py` | 生成白名单 observation、action mask、schema 版本和哈希。 |
| `event_replay.py` | 读取 Scheduler EventLog 并筛选可训练样本。 |
| `replay_env.py` | 将回放样本包装为受 mask 约束的训练环境。 |
| `simulation_backend.py` | ROS-free 项目仿真后端，使用公共随机化和真实调度契约。 |
| `train_maskable_ppo.py` | 可选 MaskablePPO 训练入口。 |
| `evaluate_policy.py`、`benchmark.py` | 离线评估、盲测和统计。 |
| `model_package.py` | 模型、配置、代码版本和 provenance 哈希打包。 |
| `promotion.py`、`shadow_gate.py` | Shadow 和 guarded promotion 门禁。 |

## 数据边界

训练样本只允许包含：候选特征、有限 utility、固定 observation、action mask 和选择结果。不得泄漏 Server 私有布局真值、裁判内部状态、语义审计字段或直接电机命令。旧版没有精确 observation 的日志只能用于选择一致性审计，不能静默转换为训练数据。

## 标准流程

```text
生产 Heuristic EventLog
        |
  replay / qualification
        |
 observation + mask dataset
        |
  offline training / benchmark
        |
   RL Shadow, zero takeover
        |
 guarded approval manifest
```

RL 策略只能输出候选槽位索引，不能输出 `vx`、`wz`、关节角或夹爪命令。推理超时、模型缺失、哈希/schema 不匹配、NaN、越界或 mask 违规都会回退到 heuristic。

## 训练前检查

```bash
python3 scripts/replay_scheduler_events.py run1.jsonl run2.jsonl \
  --min-decisions 1000 --require-training-ready \
  --dataset scheduler_replay.jsonl --output scheduler_replay_report.json

python3 scripts/validate_scheduler_model.py \
  --model scheduler_maskable_ppo.zip \
  --expected-model-sha256 <approved-model-sha256> \
  --expected-provenance-sha256 <approved-dataset-sha256>
```

正式比赛默认保持 heuristic。只有模型包、回放数据、Shadow、盲测和 approval manifest 都通过，才可讨论 `rl_guarded`。

## 项目级仿真

`simulation_backend.py` 是 ROS-free 的项目仿真后端。它复用生产的 `CandidateAction`、
`PathMetrics`、hard constraint、`SchedulingEnv` 和 utility，覆盖三任务的 pick、transport、
return 共九个宏决策。`project_simulation_v1.json` 固定公开拓扑、候选偏移、RGB-D 噪声/丢失、
速度、摩擦、消息延迟、规划失败和动态障碍等随机化字段；未知字段或 schema 不匹配直接拒绝。

仿真只产生候选、observation、action mask、转移和 reward，不模拟底盘动力学、双臂接触、裁判
评分或 Server 私有真值。相同 seed 的成对环境必须得到相同 observation、mask、转移和成功抽样；
碰撞、净空不足和规划失败会进入 action mask，绕过 mask 的动作还会被后端再次拒绝。

```bash
export MATERIAL_SCHEDULER_SIM_CONFIG=/workspace/baseline/examples/material_sorting/learning/configs/project_simulation_v1.json
python3 scripts/train_scheduler_policy.py \
  --env-factory learning.simulation_backend:build_project_sim_env \
  --output /models/scheduler_project_sim.zip \
  --timesteps 100000 --seed 20260818 \
  --code-revision <git-commit> \
  --provenance "$MATERIAL_SCHEDULER_SIM_CONFIG"

python3 scripts/benchmark_scheduler_policy.py \
  --env-factory learning.simulation_backend:build_project_sim_env \
  --model /models/scheduler_project_sim.zip \
  --model-sha256 <approved-file-hash> \
  --seed-start 30000 --episodes 100 \
  --output /models/project_sim_blind_report.json
```

仿真通过只允许进入 `rl_shadow`；它不能直接授权 `rl_guarded`，也不能替代官方 Server 验收。

## Guarded promotion

`rl_guarded` 必须由一个不可变 approval manifest 绑定三道门：模型包和 schema 完整性、至少 100
个非重叠 seed 的成对盲测，以及至少 1,000 条官方 Client Shadow 建议。Shadow 要求零 takeover、
零 mask 违规、推理 P95 不超过 25 ms 和 fallback rate 不超过 1%；成对盲测还要通过改善幅度和
bootstrap 门。生成清单后，部署时同时提供模型 SHA256、清单路径和清单 SHA256；任何缺失或篡改
都会在策略初始化阶段回退到 Heuristic。

```bash
python3 scripts/approve_guarded_policy.py \
  --model scheduler_maskable_ppo.zip \
  --benchmark scheduler_blind_benchmark.json \
  --shadow rl_shadow_acceptance.json \
  --output scheduler_guarded_approval.json
```

正式比赛默认不启用 `rl_guarded`。

仿真和回放配置的 schema、随机化字段、修改规则和 provenance 要求见
[`configs/README.md`](configs/README.md)。
