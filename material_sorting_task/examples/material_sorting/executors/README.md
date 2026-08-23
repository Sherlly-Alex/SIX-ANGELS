# 任务执行器与阶段实现

执行器是具体机器人动作的唯一拥有者。Scheduler 负责计划、约束和候选排序，执行器负责把一个阶段转换为 `StageResult`，并通过 `client_task.py` 统一发布命令。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `base.py` | `TaskStage`、`ExecutionContext`、`StageResult` 和执行器公共契约。 |
| `task1.py` | 任务一桌面导航和基础抓取阶段。 |
| `task1_full.py` | 任务一从桌面抓取到货架空层放置的完整流程。 |
| `task2.py` | 任务二货架彩色目标抓取、返回和放置流程。 |
| `task3.py` | 任务三顶部目标和包装箱参考物放置流程。 |
| `scheduler_candidate.py` | 执行器接收 scheduler 候选时的统一校验辅助。 |
| `dry_run.py` | 不驱动机器人但推进阶段的测试执行器。 |
| `transfer_support.py` | 携物转移、对中、回撤和放置支撑动作。 |

## 执行器生命周期

```text
configure_instructions()
        |
reset() -> enter(stage)
        |
tick(context) -> StageResult
        |
RUNNING / SUCCEEDED / RETRYABLE_FAILURE / BLOCKED / FAILED
        |
cancel(reason) / release resources
```

`ExecutionContext` 是只读输入，包含当前时间、任务指令、尝试次数、里程计、关节状态、稳定检测、裁判信息、得分和输入年龄。执行器不得直接读取 Server 私有布局真值，也不得绕过 Client 发布命令。

## 三任务实现重点

### 任务一

桌面目标经过 RGB-D 中心锁定后，完成开放预抓取、双侧接触、有限预紧和抬升；随后携物退离桌面，在货架外完成语义识别、唯一空层确认、横向对中、直线进入、缓慢下降、释放和安全回撤。任务一使用严格空层确认门。

### 任务二

复用任务一保存的货架状态进行粗导航，但在抓取前重新采集多帧 RGB-D 彩色目标中心和方向，不读取 Server 物体坐标。货架外先横向对中，再直线靠近目标；完成抓取后通过分段路径返回桌面原点。

### 任务三

复用任务一保存的货架状态和白色包装箱测量中心，计算左侧放置位和安全释放位。任务三仍使用独立的目标锁定、放置几何和结束区返回逻辑。

## Scheduler 候选接入

只有实现 `apply_scheduler_candidate()` 的执行器才会真正接收候选。接收时需要再次检查：

- 候选是有限的导航 pose，且阶段仍允许改目标；
- 横向/纵向偏移在执行器走廊内；
- 执行器自己的分层栅格无碰撞、净空足够；
- `NavigationController` 重新规划成功；
- 同一 `step_run_id` 内不会反复重置相同目标。

不满足时必须拒绝或 `audit_only`，不能让候选绕过已验证的物理路径。PLACE 等不可逆阶段不接受普通重试。

## 编写新执行器的检查清单

1. 明确每个阶段拥有的资源和是否允许底盘/机械臂控制。
2. 所有输出使用 `StageResult`，不要在执行器里直接发布 ROS 消息。
3. 对输入、姿态、时间、速度和工作空间做有限值检查。
4. 为可恢复失败提供结构化 `FailureCode`，并设置恢复上限。
5. 为真实动作增加 dry-run 或 ROS-free 单元测试。
6. 在随机场景中验证失败时仍能停止或安全回撤。
