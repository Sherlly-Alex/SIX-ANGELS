# RL-1：离线候选模型闭环

RL-1 将已经审计的 scheduler replay 变成可复现、可审查、默认拒绝的离线
候选模型。它不是比赛策略切换，也不代表 RL 已经超过 heuristic。

## 输入与隔离

五个远程 session 的 replay 回放目前产生了 4745 条 training-ready 记录，
文件名为 `scheduler_replay_v2.jsonl`。将它拉回隔离训练机后执行：

```bash
cd /workspace/SIX-ANGELS-v5/material_sorting_task
python3 scripts/run_rl1_pipeline.py \
  --dataset /workspace/artifacts/scheduler_replay_v2.jsonl \
  --output-dir /workspace/artifacts/rl1_20260821 \
  --timesteps 10000 --seed 20260821 \
  --gamma 0 --gae-lambda 1 --device auto \
  --code-revision qzh
```

pipeline 按 `(source_sha256, session_index)` 整组做默认 `3/1/1` 划分，先
校验 `load_replay_dataset`，再只用 `train.jsonl` 训练。它拒绝少于五个
session、空集、重复 decision、输入篡改和非空输出目录；manifest 记录输入
和输出 SHA256、记录数、schema 及 session 归属。

每条 replay 记录都是独立的调度决策快照，不是同一条连续 MDP 轨迹。因此
RL-1 显式使用 `gamma=0`，避免把相邻但无因果关系的记录做跨步信用分配；
`gamma`、`gae_lambda` 和训练设备都会写入模型元数据。通用训练入口仍保留
MaskablePPO 的常规默认值，只有 replay contextual-bandit pipeline 默认采用
`gamma=0`。

训练依赖（gymnasium、stable-baselines3、sb3-contrib、numpy）只允许安装
在隔离训练容器，不得加入比赛镜像。pipeline 不下载网络资源；依赖缺失时
会失败并保持 `next_allowed_mode=heuristic`。

## Held-out 门禁

验证器再次校验 split manifest、模型包、schema 和 provenance，然后在
validation/test 上使用硬 action mask 做确定性评估。两个集合都必须满足：

- `completed_episodes == episodes`；
- `policy_errors == 0`；
- `masked_action_violations == 0`；
- 回报不超过 replay utility oracle（仅允许数值容差）；
- 回报不低于当前 selected-action baseline（仅允许同一容差）。

oracle 是每一步最高 candidate utility，不应描述为 RL 可以超过的目标。通过
只代表候选模型具备进入 `rl_shadow` 的证据；本阶段禁止 `rl_guarded`，也不会
修改 `competition_release.env` 的 `MATERIAL_SCHEDULER_POLICY=heuristic` 默认值。

## 证据与下一步

交付 `rl1_acceptance.json`、`split/split_manifest.json`、三个 JSONL 子集、
`model_package_acceptance.json` 和 `heldout_acceptance.json`。先在 shadow
运行中记录候选动作、应用状态、重复应用和 runtime health；多 seed、回放和
官方 Server 验收保持通过后，另行评审 RL guarded promotion。任何异常、缺
文件、模型 hash 不一致或门禁失败，都回退到 heuristic。

2026-08-21 的第一轮 4090 隔离训练显示：10k steps 的 validation/test
mean return 分别为 `240.218/241.908`，100k steps 反而降至
`229.540/235.627`，均低于 selected-action baseline
`250.330/250.036`；两轮 action mask、模型包和完整 episode 检查均通过。
这说明失败不来自 GPU、模型封装或训练轮数不足，而来自原配置把独立 replay
快照按 `gamma=0.99` 做了虚假的跨记录信用分配。修复后的首要验收实验使用
`gamma=0`，仍执行同一 held-out 门禁，禁止通过放宽 baseline 容差来授权模型。

## 参考思想与取舍

- Stable-Baselines3-Contrib 的 MaskablePPO：评估始终传递 `action_masks`，不
  使用普通 EvalCallback 绕过安全动作掩码。
- Minari：借鉴 episode/session 作为不可拆分的数据单元、版本化 manifest 和
  元数据；本项目保持 JSONL，不引入 HDF5/Minari 依赖。
- RL Baselines3 Zoo：借鉴显式 seed、超参数、评估和 artifact 目录；不引入
  整个实验框架。
- rliable：后续 project-simulation 扩展到 100+ blind seeds 后再考虑 paired
  seed 与 bootstrap CI/IQM；五个生产 session 仅用于防泄漏，不能用于优越性
  声明。
- d3rlpy：只参考 dataset/evaluator/model artifact 分离；动作掩码和正式加载
  链已绑定 MaskablePPO，本阶段不切换算法。
