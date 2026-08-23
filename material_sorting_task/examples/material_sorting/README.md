# 正式 Client 与任务编排

本目录是材料分拣比赛的正式 ROS 2 Client 代码。它负责把 Server 的结构化任务、感知结果、裁判状态和执行器动作连接起来。正式运行只使用 `/material/instruction` 发布的结构化 JSON，不从中文指令文本猜测执行字段，也不让 NLP/ML/SLM 研究模块修改控制决策。

## 运行入口

| 文件 | 作用 |
| --- | --- |
| `client_task.py` | 唯一正式 ROS 2 Client 入口，拥有 topic 订阅、命令发布、输入新鲜度检查和控制 tick。 |
| `competition_controller.py` | 三任务连续编排、任务阶段切换和 Server 裁判同步。 |
| `instruction_parser.py` | 校验 Server JSON 是否包含可执行字段。 |
| `task_orchestration.py` | 解析和排序三条任务指令。 |
| `runtime_health.py` | 输入新鲜度和 20 Hz 控制循环健康监测。 |
| `control_types.py` | 底盘、机械臂和夹爪控制数据契约。 |
| `executors/` | 任务一、二、三的具体阶段动作。 |
| `perception/` | RGB-D/YOLO 感知节点。 |
| `scheduler/` | 任务计划、候选决策、资源、安全、恢复和策略护栏。 |

## 端到端流程

```text
Server JSON + /referee/* + odom/joint states
                 |
                 v
           client_task.py
       输入新鲜度 / 安全检查
                 |
                 v
      CompetitionController / SchedulerEngine
                 |
       +---------+----------+
       |                    |
       v                    v
  perception/*         executors/*
       |                    |
       +------ detections --+
                 |
                 v
       ROS command topics
```

每个控制周期最多推进一个外部可见状态转换。Client 是唯一的 ROS 命令发布者；检测节点只发布 `/material/detections`，执行器返回结构化 `StageResult`，最终由 Client 统一发布底盘、升降、头部、双臂和夹爪命令。

## 任务阶段

三个任务共享同一套阶段契约：

```text
NAVIGATE_TO_PICK -> ACQUIRE_TARGET -> ALIGN_FOR_PICK -> GRASP -> LIFT
-> TRANSPORT -> ALIGN_FOR_PLACE -> PLACE -> VERIFY_PLACE -> RETURN_TO_END
```

任务切换不是本地计时器触发，而是等待 Server 的 `/referee/taskinfo`、`/referee/gameinfo` 和 `/referee/score`。裁判信息持续不一致、输入过期、命令非法或检测证据不足时，系统保持、阻塞或进入 `SAFE_HOLD`。

## 执行模式

| 模式 | 用途 | 是否驱动机器人 |
| --- | --- | --- |
| `stub` | 验证入口和安全阻塞行为。 | 在未实现阶段安全阻塞。 |
| `dry_run` | 验证阶段顺序、日志和任务生命周期。 | 不发布真实动作。 |
| `nav_only` | 任务一底盘导航联调。 | 发布底盘命令。 |
| `pregrasp_only` | 导航加双臂开放预抓取。 | 发布底盘和机械臂命令。 |
| `contact_only` | 双侧接触和有限预紧。 | 发布真实接触动作。 |
| `lift_only` | 抓取、预紧后抬升。 | 发布真实抬升动作。 |
| `task1_full` | 只验证任务一完整流程。 | 任务二、三安全阻塞。 |
| `task12_full` | 任务一和任务二回归。 | 连续执行两任务。 |
| `task123_full` | 正式三任务连续执行。 | 完整控制链。 |

## 开发约束

1. 不在 Client 中硬编码 Server 私有布局真值来代替感知。
2. 不在执行器外部重复发布机器人命令。
3. 不用固定时间强行推进任务；任务结算以裁判为准。
4. 新的候选、策略或恢复路径必须保留硬约束和 fail-closed 行为。
5. 修改阶段契约后，需要同步更新对应测试和本目录 README。

## 相关模块

- [perception/README.md](perception/README.md)
- [navigation/README.md](navigation/README.md)
- [desktop_grasp/README.md](desktop_grasp/README.md)
- [shelf/README.md](shelf/README.md)
- [executors/README.md](executors/README.md)
- [scheduler/README.md](scheduler/README.md)
- [learning/README.md](learning/README.md)

### 深层板块

- [navigation/costmap/README.md](navigation/costmap/README.md)：版本化代价地图、动态障碍、路径指标和携物包络评价。
- [scheduler/policies/README.md](scheduler/policies/README.md)：Heuristic、RL 适配器、超时隔离和安全回退。
- [learning/configs/README.md](learning/configs/README.md)：项目仿真与 EventLog 回放训练配置。
- [reference/server/README.md](reference/server/README.md)：参考 Server、裁判状态机和计分配置。
