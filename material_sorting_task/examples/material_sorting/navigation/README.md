# 导航模块：代价地图、路径规划与携物安全

导航模块将世界坐标目标转换为可执行的底盘路径，并在机器人处于空载、携物或靠近放置区时使用不同的几何包络。导航代码由执行器调用，Scheduler 只提供经过约束的候选站位，不绕过导航控制器直接发布速度。

## 文件职责

| 文件/目录 | 作用 |
| --- | --- |
| `navigation_controller.py` | A* / 分段路径、速度限制、跟踪和状态机。 |
| `competition_adapter.py` | 比赛坐标、任务目标和导航接口适配。 |
| `occupancy_grid.py`、`footprint_checker.py` | 栅格占用和机器人 footprint 检查。 |
| `costmap/` | 世界代价地图快照、动态障碍和规划指标。 |
| `carried_envelope.py` | 携带物的机器人包络检查。 |
| `path_smoother.py` | 路径转角和速度连续性处理。 |
| `dynamic_overlay.py` | 将检测到的非目标物体加入动态障碍层。 |
| `robot_geometry.py` | `TRANSIT_STOWED`、`TRANSIT_CARRY`、`DOCKING` 等包络模式。 |

## 基本路径流程

```text
当前位置 + 名义目标 + 目标朝向
              |
       静态场景 / 动态障碍
              |
       LayeredGrid + Costmap
              |
       A* / segment planning
              |
       路径平滑 + 速度限幅
              |
       实时跟踪 + 急停检查
```

目标站位先由执行器根据任务标定和感知结果确定，导航器再检查路径是否在栅格内、是否与静态/动态障碍冲突、是否有足够的急停余量。输入不新鲜、目标越界或无法规划时，控制器停止发布速度。

## Scheduler V2 的导航接入

候选提供器围绕执行器声明的名义站位生成中心、左偏、右偏候选。候选评分使用 `WorldCostmapSnapshot`，并根据阶段选择 footprint：

- `NAVIGATE_TO_PICK`：`TRANSIT_STOWED`。
- `TRANSPORT`：`TRANSIT_CARRY`，检查手持物外形。
- `RETURN_TO_END`：`DOCKING`，检查结束区停靠空间。

Task 1 导航执行器对 scheduler 候选重复四项检查：横向走廊、纵向走廊、分层栅格碰撞和站位净空；通过后仍需 `NavigationController` 实际重规划成功。未达到校准一致性的候选可以记录 `audit_only`，但不能强行改变名义轨迹。

## 关键安全规则

1. 底盘速度在执行器和 ROS 发布入口双重限幅。
2. 携物阶段使用携物 footprint，而不是空载 footprint。
3. 动态障碍有时间戳和 TTL，过期观测不能长期阻塞路径。
4. 任务一放置前的横向对中使用更严格的 1.5 cm 专用容差；其他任务不读取该任务一常量。
5. 任何碰撞、急停余量不足或实际重规划失败都 fail-closed。

## 测试

```bash
MATERIAL_EXECUTION_MODE=nav_only \
MATERIAL_SCHEDULER_ENGINE=v2 \
MATERIAL_SCHEDULER_POLICY=heuristic \
MATERIAL_DETECT_BACKEND=yolo \
bash scripts/run_client.sh
```

`nav_only` 会真实发布 `/cmd_vel`，不能与其他 Client、旧控制脚本或机械臂节点并行运行。测试前使用新 Server，结束后重启场景。

## 接口演进与验证

导航 v3 的接入保持 `NavigationController` 为唯一速度出口：新的 costmap、path smoother、
dynamic overlay 和 footprint 检查通过已有导航接口接入，不能在候选策略或联调脚本中另发
`/cmd_vel`。本地验证先运行 ROS-free 的栅格、几何、路径和平滑测试，再在官方容器中按
`nav_only`、携物运输和返回结束区的顺序逐段验证。任何新版本都要同时检查空载、携物和停靠
三种 footprint，并保留规划失败、急停和实际重规划日志。

代价地图的快照语义、动态障碍 TTL、路径指标和携物包络评价见
[`costmap/README.md`](costmap/README.md)。
