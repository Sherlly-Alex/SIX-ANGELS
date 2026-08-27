# RL 独立推理 Shadow：第一阶段实现说明

## 目的

当前比赛版本的默认策略仍是 V2 Heuristic。实机 Shadow 虽然任务结果正常，但同一
种子已出现比 Heuristic 更高的控制周期 p99 尾部延迟。因此，RL 推理不能继续与
客户端控制、感知和调度线程共享同一进程资源。

本阶段新增 `learning/isolated_inference.py`，只提供一个不含 ROS 和执行器的本地
进程 IPC 通道。它不会发布速度、关节或抓取命令，也不会改写
`CompetitionController`、Task 1/2/3 执行器或启发式候选选择。

## 约束

1. 子进程只收到有限 observation、Action Mask、请求标识和版本签名；只返回离散
   候选槽位、耗时、模型 SHA 或错误。
2. 父进程 `submit()` / `poll()` 都是非阻塞调用，最多保持一个在途请求。
3. 返回结果必须同时匹配 request id 和当前签名；签名应包含任务、阶段、候选槽位
   顺序、Action Mask、costmap version 和 step run id。任何不匹配结果标为 stale。
4. 即便结果匹配，调用方仍必须对当前候选重新执行 Action Mask、成本地图和
   `PolicyGuard.accept_or_fallback()`；子进程没有动作执行权限。
5. 超时、队列满、进程错误、无效动作、模型/SHA/Schema 失败均保留 Heuristic。
   连续三次运行时故障后隔离子进程，必须由显式运维操作才能重新启用。

## 当前状态

基础层已作为仅 Shadow 的可选接入层挂接到 `SchedulerDecisionService`。它必须同时
满足以下环境变量才会启动：

```bash
MATERIAL_SCHEDULER_RL_ENABLED=1
MATERIAL_SCHEDULER_POLICY=rl_shadow
MATERIAL_RL_SHADOW_ISOLATED=1
```

不开启第三项时，既有 Shadow 行为保持不变；默认配置仍是 Heuristic。即使开启，Shadow
返回值也只进入审计字段，不会更改启发式选出的执行动作。Guarded 明确不走此路径。

下一步是以同种子 `heuristic` 与 `rl_shadow + isolated` 做严格 p99 配对验收；全部通过
前不讨论 Guarded，也不改变任务执行器。
