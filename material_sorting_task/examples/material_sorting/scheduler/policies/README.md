# 调度策略与运行时护栏

`policies/` 只负责从已经生成并通过基础约束检查的有限候选中选择策略结果。它不生成底盘速度、关节角或夹爪命令；真实动作仍由 Scheduler 和执行器的安全接口拥有。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `heuristic.py` | 默认确定性策略，使用同一份 costmap snapshot 进行路径评价和 utility 排序。 |
| `rl.py` | 可选 MaskablePPO 适配器，负责模型哈希、schema 元数据、离散动作和 action mask 校验。 |
| `guard.py` | RL 运行时护栏，负责超时、隔离、mask、安全下界和 Heuristic fallback。 |
| `__init__.py` | 策略公共接口导出。 |

## Heuristic 流程

```text
候选 center/left/right/recovery
              |
       同一 WorldCostmapSnapshot
              |
     路径、footprint、携物包络评价
              |
       hard constraints 过滤
              |
        Multi-Critic utility 排序
              |
          选择最高安全候选
```

`HeuristicPolicy.rank()` 在一次调用开始时固定快照，避免并发感知更新导致候选之间使用不同世界状态。不可达路径、非有限 pose、碰撞、净空不足或资源/裁判约束失败的候选不会进入有效排序；全部候选被屏蔽时返回空选择，执行器保持安全状态。

## RL 适配边界

`RLPolicy` 采用惰性加载，只在明确请求时读取本地模型，不下载权重。加载前可以校验模型 SHA256、metadata schema、算法类型和 observation schema。推理输出必须是一个有限、严格为整数的候选槽位索引，并且该索引必须位于 action mask 内；小数、布尔值、越界动作、masked action 和异常输出全部拒绝。

## Guarded fallback

`PolicyGuard` 在独立线程中执行推理并设置默认 25 ms 超时。发生模型缺失、推理异常、超时、schema/哈希不匹配、action mask 无效、安全下界不足或策略被隔离时，返回 `source="heuristic"`，不授予 RL 控制权。超时可以触发 quarantine，后续决策持续使用 Heuristic，直到显式重置。

```text
rl_shadow  : 只记录 RL 建议，Heuristic 仍然选择和执行
rl_guarded : 只有模型包、盲测、Shadow、approval manifest 全部通过后才可请求
heuristic  : 正式默认策略
```

正式配置必须保持 `MATERIAL_SCHEDULER_POLICY=heuristic`，除非已经完成 `learning/README.md` 描述的离线回放、项目仿真、成对盲测、Shadow 和 guarded approval 流程。策略目录的单元测试应覆盖 fallback reason、超时、哈希/schema 漂移、非法离散动作和 masked action。

相关说明：[`../README.md`](../README.md)、[`../../learning/README.md`](../../learning/README.md)、[`../../../../scripts/README.md`](../../../../scripts/README.md)。
