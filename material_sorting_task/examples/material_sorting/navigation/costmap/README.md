# 代价地图与路径评价

`costmap/` 是 Scheduler V2 使用的导航评价层。它把静态分层栅格、带时间戳的动态障碍和机器人/携物 footprint 组合成一个可版本化的世界快照，再为每个候选站位生成统一的 `PathMetrics`。该目录不发布 ROS 命令，也不替代 `navigation_controller.py` 的实时跟踪。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `snapshot.py` | `AABB`、`DynamicObstacle`、`PathMetrics` 和有限路径冻结工具。 |
| `world_costmap.py` | 管理静态/动态地图、生成只读快照、规划路径和计算指标。 |
| `__init__.py` | 导出代价地图公共接口。 |

## 数据生命周期

```text
静态场景 / LayeredGrid
          + RGB-D 动态检测
          + 时间戳、置信度、TTL
                    |
                    v
             WorldCostmap
                    |
           versioned snapshot
                    |
                    v
   planner + footprint + carried envelope
                    |
                    v
              PathMetrics
```

`WorldCostmap` 可以被感知更新线程修改，但每个调度周期只获取一次 `WorldCostmapSnapshot`。候选比较使用同一快照，后续感知更新不会改变已经计算出的候选含义。快照中的动态障碍是不可变记录，过期或低置信度证据不会永久阻塞规划。

## 主要算法

### 动态障碍

`DynamicObstacle` 保存 `AABB`、置信度、观测时间、过期时间、来源和类别。`active_at()` 按时间窗口和最低置信度筛选证据；`lethal_obstacles` 再按致命置信度门限区分硬碰撞和软风险。货架空层的自由空间证据不能直接作为实体障碍写入动态层。

### 路径规划与指标

`WorldCostmapSnapshot.plan_path()` 使用项目已有的 `GlobalPlanner`，对候选的起点、终点、footprint 模式、膨胀半径和最小净空做有限值检查。规划成功后由 `evaluate_path()` 统一计算：

- 路径长度、直线距离和绕行比例；
- 最小 footprint 净空、膨胀代价积分；
- 航向变化量、转弯次数和动态风险；
- 携物阶段的实测货物包络是否安全。

路径中的姿态转向和每个采样点都要重新检查 footprint；路径碰撞、离开地图、非有限指标或携物包络不安全都会返回 `PathMetrics.unreachable(...)`，由上层硬过滤候选。

## 与调度器、导航器的边界

1. Scheduler 使用快照和 `PathMetrics` 做候选过滤、排序和日志记录。
2. 执行器接收候选后还要按自己的走廊、净空和目标一致性规则二次检查。
3. `NavigationController` 负责真实重规划、速度限幅、跟踪和急停；代价地图不直接发布 `/cmd_vel`。
4. `TRANSIT_STOWED`、`TRANSIT_CARRY` 和 `DOCKING` 使用不同机器人几何包络。

## 测试重点

ROS-free 测试应覆盖静态碰撞、动态障碍 TTL、低置信度证据、路径指标有限值、三种 footprint、携物包络和规划失败原因。随机场景验收时，Scheduler EventLog 应保存快照版本、候选路径指标、硬过滤原因和最终应用状态。

相关说明：[`../README.md`](../README.md)、[`../../scheduler/README.md`](../../scheduler/README.md)、[`../navigation_controller.py`](../navigation_controller.py)。
