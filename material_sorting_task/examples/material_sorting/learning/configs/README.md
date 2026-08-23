# 调度学习配置

本目录保存 Scheduler V2 离线回放和项目级仿真的版本化配置。配置只描述公开的随机化、阶段拓扑、观测环境和奖励规则，不包含 Server 私有布局真值、机器人实时控制参数或模型权重。

## 文件职责

| 文件 | 用途 |
| --- | --- |
| `project_simulation_v1.json` | 三任务项目仿真配置，覆盖 pick、transport、return 九个宏决策阶段。 |
| `replay_training_v1.json` | EventLog 回放训练环境的 episode、随机化和 reward 配置。 |

## 项目仿真配置

`project_simulation_v1.json` 的 `schema_version` 固定为 `scheduler-project-sim-v1`，并定义：

- 最大候选数和最小净空；
- Task 1/2/3 的阶段顺序、payload 类型和基础路径长度；
- pose/yaw 噪声、检测噪声、深度比例、速度、摩擦和消息延迟；
- 检测丢失、规划失败和动态障碍概率。

每个 seed 在配置允许的随机化范围内生成完整 episode。成对盲测必须对 Heuristic 和 RL 使用相同 seed、相同公开随机样本和相同成功抽样，才能比较 utility regret 或完成率差异。

## 回放训练配置

`replay_training_v1.json` 描述从生产 EventLog 导出的训练环境，包括 episode 长度、是否随机化和同一组公开噪声参数。奖励只用于候选排序：相对最优安全候选计算 regret，非法动作得到硬惩罚，不模拟动力学，也不把评分真值泄漏给策略。

## 修改规则

1. 新增字段必须同步更新 `simulation_backend.py`、回放环境和对应测试。
2. 未知字段、缺失字段或 schema 版本不匹配必须 fail-closed。
3. 配置改动后重新生成数据集，更新配置哈希和模型 provenance。
4. 仿真通过只代表可以进入 `rl_shadow`，不能直接启用 `rl_guarded` 或替代官方 Server 验收。

示例命令和模型发布门禁见 [`../README.md`](../README.md) 与 [`../../scheduler/README.md`](../../scheduler/README.md)。
