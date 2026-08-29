# 局部建图策略（Local Map Strategy）

> qzhRL 接入状态（2026-08-28）：本文保留输入方案的完整设计，便于后续研究；
> 当前安全移植版启用建图、advice 发布/日志、Task 2 的 farther-only standoff，以及
> Task 2/3 运输导航的实验速度倍率。倍率已前移到导航控制器的局部避障、速度限制与
> 预测碰撞检查之前；没有照抄输入包的事后放大位置。Task 1 和所有放置动作维持原实现。

本文档描述正式 Client 中**可选局部建图（RGB-D rolling height map）**的设计、数据流、
运动接入点与环境变量。供整体技术文档引用；默认**关闭**，开启后遵循 **fail-open**
（异常或数据不足时不改变原有运动）。

---

## 1. 定位与边界

### 1.1 解决什么问题

赛方 Server **无 LiDAR**，仅有头部 RGB-D + 里程计。全局 A* 使用预置场地栅格，对**静态布局**
可靠，但对近场动态遮挡、随机摆放的边缘情况反应较慢。

局部建图在全局栅格之上增加一层**机器人中心的滚动 2.5D 高度图**，用于：

| 能力 | 说明 |
|------|------|
| **Tier-1 近场 standoff** | 前方有可信障碍时，将任务 2 的货架取货 / 桌面入口站位略后退 |
| **Tier-2 运输加速** | 前方走廊开阔时，提高底盘**运输段**线速度 |

局部建图**不替代** A* 路径规划，也不参与手臂规划。

### 1.2 不做什么

以下阶段**刻意不接入**建图调速或 standoff，避免破坏已标定几何：

| 阶段 | 原因 |
|------|------|
| 任务 1 货架**观察站**净空 | 固定 `SHELF_SCAN_CENTER_CLEARANCE_M = 0.75`；推远观察站会导致放置偏侧、手臂蹭架 |
| 任务 1/2/3 **放置链**（横向对中、直进插入、松手） | `map_speed=False` / `linear_scale=1.0` |
| 任务 3 **扫描站位 X** | `align_for_place` 复用扫描站 X，改动会破坏放置 |
| 货架语义扫描（pitch 多档停留） | 固定感知流程，与建图无关 |
| 桌面/货架抓取、抬升、slide | 纯手臂控制 |

### 1.3 与「深度」的关系

深度（Depth）是传感器输入；建图是把多帧深度融合后的**中间表示**：

```text
头部深度图 + 相机内参 K + 里程计 + MMK2 FK（T_cam_world）
        ↓
RollingLocalHeightMap（滚动局部高度图）
        ↓
forward_clearance() → LocalMapAdvice
        ↓
client_task → ExecutionContext.local_map_advice → 执行器
```

不存在「深度模块减速、建图模块加速」两套并行判决；**速度决策只读建图 advice**，
而建图本身由深度融合而来。

---

## 2. 系统架构

```text
┌─────────────────────────────────────────────────────────────────┐
│  box_detect（感知节点）                                          │
│    RGB/Depth 检测管线                                            │
│      └─ MATERIAL_LOCAL_MAP=1 时：                                │
│           LocalMapSidecar.on_tick()                              │
│           发布 /material/local_map_advice (std_msgs/String JSON) │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  client_task.py                                                  │
│    订阅 advice → ExecutionContext.local_map_advice               │
│    每控制周期传入 task1/2/3 执行器                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   task1_full.py         task2.py            task3.py
   · 运输加速            · standoff + 加速    · 仅运输加速
   · 观察站固定 0.75m    · 放置不加速
   · 放置不加速
```

### 2.1 核心模块

| 模块 | 路径 | 职责 |
|------|------|------|
| 高度图 | `perception/local_map.py` | `RollingLocalHeightMap`：融合、衰减、前方通畅查询 |
| Sidecar | `perception/local_map_sidecar.py` | advice 生成、standoff/速度策略、环境变量 |
| 感知融合 | `perception/box_detect.py` | 复用 YOLO 管线深度帧，节流后调用 sidecar |
| 运动入口 | `executors/local_map_motion.py` | `map_linear_scale(context)` |
| Standoff | `executors/task1_full.py` | `_shelf_clearance_m()`（任务 2 调用） |
| 底盘缩放 | `executors/transfer_support.py` | `tick_navigation` / `tick_advance` 的 `linear_scale` |

### 2.2 ROS 接口

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/material/local_map_advice` | `std_msgs/String` | detect → client | JSON 序列化的 `LocalMapAdvice` |

Client **不二次订阅深度**，避免重复解码与 CPU 开销。

---

## 3. 建图构建流程

### 3.1 输入

每帧（经 `MATERIAL_LOCAL_MAP_HZ` 节流）需要：

1. **对齐深度图**（与 YOLO/RGB-D 检测同帧）
2. **相机内参** `K`（`CameraInfo`）
3. **里程计** `(x, y, yaw)`
4. **相机外参** `T_cam_world`（MMK2 正运动学）

任一缺失 → 本 tick 不融合，`fresh=False`，执行器 fail-open。

### 3.2 深度 → 世界点云

`depth_frame_to_world_points()`：

- 按 `MATERIAL_LOCAL_MAP_STRIDE`（默认 8）降采样
- 反投影到相机系，再用 `T_cam_world` 变换到世界系
- 丢弃深度范围外、低于 `floor_z + obstacle_z_above_floor`（默认地面 + 8cm）的点

融合前做帧质量门控：`valid_ratio` 低于阈值则拒绝本帧。

### 3.3 滚动栅格

`RollingLocalHeightMap` 维护机器人周围的局部窗口：

| 参数 | 环境变量 | 默认值 | 含义 |
|------|----------|--------|------|
| 分辨率 | `MATERIAL_LOCAL_MAP_RES` | 0.05 m | 栅格边长 |
| 前向范围 | — | 2.5 m | 机器人前方 |
| 后向范围 | — | 0.5 m | 机器人后方 |
| 侧向范围 | — | ±1.5 m | 左右 |
| 最小命中 | `MATERIAL_LOCAL_MAP_MIN_HITS` | 2 | 格子至少命中次数才算障碍 |
| 最大年龄 | `MATERIAL_LOCAL_MAP_MAX_AGE_S` | 8 s | 超时格子衰减清除 |
| 降采样 | `MATERIAL_LOCAL_MAP_STRIDE` | 8 | 深度步长 |

每个有效点 `(x, y, z)` 写入对应格子：

- `hit_count += 1`
- `height_z = max(旧值, z)`（2.5D：每格记录最高点）
- `last_update_s = now`

机器人移动时窗口随位姿**平移**（`_shift_window`），不无限扩张。

### 3.4 前方通畅查询

`forward_clearance(robot_pose)`：

- 沿机器人 **yaw 方向**向前采样，步长约 2.5 cm，最远 **1.2 m**
- 走廊宽度 **0.4 m**（左 / 中 / 右三条采样线）
- 某格 `hit_count ≥ MIN_HITS` 且高度有效 → 命中障碍，返回 `clear=False, distance_m`
- 全程无命中 → `clear=True`

`suggested_standoff()` 在命中时给出建议后退距离（供 Tier-1 standoff）。

### 3.5 Advice 生成

`LocalMapSidecar.on_tick()` 输出 `LocalMapAdvice`：

```json
{
  "enabled": true,
  "apply": true,
  "fresh": true,
  "clear": true,
  "distance_m": 1.2,
  "suggested_standoff_m": 0.55,
  "frames_accepted": 42,
  "frames_rejected": 3,
  "reason": "ok"
}
```

| 字段 | 含义 |
|------|------|
| `enabled` | `MATERIAL_LOCAL_MAP=1` |
| `apply` | `MATERIAL_LOCAL_MAP_APPLY=1` |
| `fresh` | 本 tick 成功融合，或 2 s 内有成功融合 |
| `clear` | 前方 1.2 m 走廊是否通畅 |
| `distance_m` | 最近障碍距离（`clear=True` 时为最大量程） |
| `suggested_standoff_m` | 建议站位后退量 |
| `reason` | `ok` / `throttled` / `no_depth` / `stale` 等 |

**只有 `enabled ∧ apply ∧ fresh` 时**，Tier-1 / Tier-2 才会改变运动；否则等同未开启。

---

## 4. 运动策略（Tier-1 / Tier-2）

### 4.1 Tier-1：近场 standoff

函数：`_shelf_clearance_m(context, fallback_m, ...)`

**门控条件**（全部满足才推远，否则返回 `fallback_m`）：

1. `allow_local_map=True`（调用点显式允许）
2. advice：`enabled + apply + fresh`
3. `clear=False`（前方有障碍）
4. `distance_m < 1.0 m`

推远公式（简化）：

```text
result = clamp(
    max(fallback, suggested_standoff, distance + margin),
    min=fallback,
    max=SHELF_MAP_STANDOFF_MAX_M  # 或任务 2 专用上限
)
```

**永不比标定 fallback 更近**（fail-open 下限）。

#### 各任务 standoff 接入

| 任务 | 接入点 | fallback | 上限 | 说明 |
|------|--------|----------|------|------|
| 任务 1 观察站 | **不接入** | 0.75 m 固定 | — | 防止放置偏侧 |
| 任务 2 货架取货 | `_shelf_pick_approach_x` | 0.965 m | 1.08 m | 取货前 staging X |
| 任务 2 桌面入口 | `_table_entry_margin_m` | 0.25 m | — | 进桌安全边距 |
| 任务 3 | **不接入** | 固定扫描站 | — | 放置几何敏感 |

### 4.2 Tier-2：运输线速度缩放

函数：`map_linear_scale(context)` → `local_map_linear_scale(advice)`

在 `transfer_support` 中仅缩放 **线速度** `linear_x`，角速度不变：

```python
cmd = (linear_x * scale, angular_z)
```

#### 速度模式 `MATERIAL_LOCAL_MAP_SPEED_MODE`

| 模式 | `clear=True` | `clear=False` | 推荐场景 |
|------|--------------|---------------|----------|
| **`boost_only`（默认）** | ×`CLEAR_BOOST`（默认 1.40） | ×1.0（不减速） | 竞赛提速；A* 仍管静态避障 |
| `full` | ×`CLEAR_BOOST` | 近距 ×0.52，中距 ×0.85 | 保守近场；朝货架运输会被减速 |

`full` 模式历史问题：机器人**面朝货架**运输时，建图正确地将货架判为前方障碍
（`clear=False`），运输后半段持续减速，抵消开阔路段加速。因此当前默认采用 `boost_only`。

#### 加速系数

| 环境变量 | 默认 | 范围 | 说明 |
|----------|------|------|------|
| `MATERIAL_LOCAL_MAP_CLEAR_BOOST` | 1.40 | [1.0, 2.0] | `clear=True` 时的线速度倍率 |

`full` 模式减速常数（代码内，暂无环境变量）：

| 条件 | 倍率 |
|------|------|
| `clear=False` 且 `distance_m ≤ 0.38 m` | 0.52 |
| `clear=False` 且 `0.38 m < distance_m < 1.0 m` | 0.85 |

#### 各任务加速接入

| 任务 | 接入阶段 | 不接入 |
|------|----------|--------|
| 任务 1 | 桌↔货架 A* 运输、返程 | 观察站、扫描、横向对中、直进放置、松手 |
| 任务 2 | 去货架、运输、桌面入口直进 | `advance_table_final` 等最终放置 |
| 任务 3 | 运输段 | 扫描站位、放置、插入、推箱（`map_speed=False`） |

---

## 5. 任务 1 观察站说明

**观察站**（`shelf_scan_stand`）：放货前停在货架正前方，头部扫描识别空层的位置。

- 标定目标：怀里**箱子中心**距货架前沿 **0.75 m**（`SHELF_SCAN_CENTER_CLEARANCE_M`）
- 由 `shelf_observation_stand()` + `stand_from_held_center()` 计算底盘 `(x, y)`
- 此站**只看不放**；后续横向对中 → 直进 → 放入

若 Tier-1 standoff 将 0.75 m 推远（例如 0.88 m），机器人整体后退，但放置链仍按
标准几何计算，曾导致**偏侧放置与手臂蹭架**。当前策略：**任务 1 观察站净空固定，
不受建图 standoff 影响**。

---

## 6. 环境变量一览

### 6.1 开关

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATERIAL_LOCAL_MAP` | `0` | `1` 开启建图与 advice 发布 |
| `MATERIAL_LOCAL_MAP_APPLY` | `0` | `1` advice 真正影响运动（否则仅观测/日志） |

### 6.2 融合与地图

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATERIAL_LOCAL_MAP_HZ` | `0.5` | 最大融合频率；建议竞赛 **`1.0`** 以上 |
| `MATERIAL_LOCAL_MAP_RES` | `0.05` | 栅格分辨率 (m) |
| `MATERIAL_LOCAL_MAP_MAX_AGE_S` | `8.0` | 格子衰减时间 (s) |
| `MATERIAL_LOCAL_MAP_MIN_HITS` | `2` | 障碍确认最小命中次数 |
| `MATERIAL_LOCAL_MAP_STRIDE` | `8` | 深度降采样步长 |

### 6.3 运动策略

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATERIAL_LOCAL_MAP_SPEED_MODE` | `boost_only` | `full` 启用近场减速 |
| `MATERIAL_LOCAL_MAP_CLEAR_BOOST` | `1.40` | 开阔走廊线速度倍率 |

### 6.4 推荐竞赛配置

```bash
export MATERIAL_LOCAL_MAP=1
export MATERIAL_LOCAL_MAP_APPLY=1
export MATERIAL_LOCAL_MAP_HZ=1.0
export MATERIAL_LOCAL_MAP_SPEED_MODE=boost_only
export MATERIAL_LOCAL_MAP_CLEAR_BOOST=1.45
```

Docker Client 示例：

```bash
docker run ... \
  -e ROS_DOMAIN_ID=88 \
  -e MATERIAL_EXECUTION_MODE=task123_full \
  -e MATERIAL_LOCAL_MAP=1 \
  -e MATERIAL_LOCAL_MAP_APPLY=1 \
  -e MATERIAL_LOCAL_MAP_HZ=1.0 \
  -e MATERIAL_LOCAL_MAP_SPEED_MODE=boost_only \
  -e MATERIAL_LOCAL_MAP_CLEAR_BOOST=1.45 \
  ...
```

---

## 7. Fail-open 原则

以下任一情况，执行器行为与**未开启建图**一致：

- `MATERIAL_LOCAL_MAP=0` 或 sidecar 初始化失败
- `MATERIAL_LOCAL_MAP_APPLY=0`
- advice 为 `None`
- `fresh=False`（无深度、无里程计、节流、质量差、异常）
- 调用点 `allow_local_map=False` 或 `map_speed=False`
- standoff：命中距离 ≥ 1.0 m，或 `clear=True`
- 速度：`boost_only` 且 `clear=False`

Sidecar **不向控制环抛异常**；tick 内异常会将 `apply` 置 false 并标记 `reason`。

---

## 8. 预期收益与局限

### 8.1 能节省的时间

建图主要缩短 **底盘运输段**（桌↔货架、任务间转场）。开阔走廊上约 **30–40%**
线速度提升（取决于 `CLEAR_BOOST` 与路段占比）。

### 8.2 无法缩短的时间

- 货架 pitch 扫描（默认 3 档 × 3 s dwell）
- 放置横向对中与直进（ deliberately 不用建图）
- 抓取、抬升、分阶段松手

因此**整局任务**通常只能缩短约 **10–15%**，而非成倍提升。

### 8.3 已知设计取舍

| 取舍 | 说明 |
|------|------|
| 任务 1 观察站不 standoff | 优先放置精度，牺牲近场「更安全但更慢」的站位 |
| 任务 3 不 standoff | `align_for_place` 与扫描站 X 耦合 |
| 默认 `boost_only` | 避免朝货架运输时误减速 |
| 不替代 A* | 动态地图仅近场提示，全局路径仍靠预置栅格 |

---

## 9. 调试与日志

### 9.1 Client 日志

`client_task.py` 每 5 s 打印一次 relay 摘要（`MATERIAL_LOCAL_MAP=1` 时）：

```text
local_map relay apply=True fresh=True clear=True dist=1.2 standoff=0.55 acc=42 rej=3 reason=ok
```

### 9.2 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 开了 MAP 但没加速 | `APPLY=0` 或 `fresh=False` | 检查 `HZ`、深度/里程计、`reason` |
| 比不开还慢 | `SPEED_MODE=full` 且运输面朝障碍 | 改为 `boost_only` |
| `frames_rejected` 高 | 深度质量差或 stride 过大 | 降低 `STRIDE` 或检查相机 |
| 任务 1 放置偏侧 | 观察站被推远（旧版） | 确认已固定 0.75 m 观察站 |

### 9.3 单元测试

```bash
cd material_sorting_task
python -m pytest tests/test_local_map.py tests/test_local_map_sidecar.py -q
```

---

## 10. 与全局导航的关系

| | 全局 A* 栅格 | 局部滚动建图 |
|--|-------------|-------------|
| 数据来源 | `material_competition_layout.json` 等预置布局 | 实时头部 RGB-D |
| 更新 | 静态 | 滚动窗口 + 时间衰减 |
| 用途 | 路径规划、静态障碍 | 近场通畅、运输加速、任务 2 standoff |
| 是否互相替代 | 否 | 否 |

---

## 11. 版本记录（摘要）

| 阶段 | 内容 |
|------|------|
| 初版 | `box_detect` 深度融合 + `/material/local_map_advice` + fail-open sidecar |
| Tier-1 | `_shelf_clearance_m` 任务 2 standoff |
| Tier-2 | `map_linear_scale` 运输段加速；`transfer_support.linear_scale` |
| 任务 1 修正 | 观察站固定 0.75 m；放置链收紧横向容差；放置不参与调速 |
| 速度策略 | 默认 `boost_only`；`CLEAR_BOOST` 1.28 → 1.40；环境变量可配 |

---

## 附录 A：建图量化参数总表

本节汇总**仅与局部建图模块相关**的可引用指标，供最终技术文档摘录。任务几何、
放置精度、全局导航限速等由其他模块文档维护。

### A.1 滚动地图几何与融合

| 参数 | 符号 / 代码名 | 默认值 | 单位 | 可调 | 说明 |
|------|---------------|--------|------|------|------|
| 地图类型 | — | 2.5D 高度图 | — | 否 | 每格记录该处最高点 \(z\) |
| 栅格分辨率 | `MATERIAL_LOCAL_MAP_RES` | **0.05** | m | 是 | 单格边长 |
| 前向窗口 | `forward_m` | **2.5** | m | 否 | 机器人前方覆盖 |
| 后向窗口 | `back_m` | **0.5** | m | 否 | 机器人后方覆盖 |
| 侧向半宽 | `side_m` | **1.5** | m | 否 | 左右各 1.5 m |
| 地图平面覆盖（约） | — | **3.0 × 3.0** | m | — | (前+后) × (2×侧向) |
| 栅格规模（约） | — | **60 × 60** | 格 | — | 3.0 m ÷ 0.05 m |
| 地面参考高度 | `floor_z` | **0.02** | m | 否 | 世界系 \(z\) 基准 |
| 障碍高度阈值 | `obstacle_z_above_floor` | **+0.08** | m | 否 | 高于地面 8 cm 视为障碍 |
| 障碍确认最小命中 | `MATERIAL_LOCAL_MAP_MIN_HITS` | **2** | 次 | 是 | 格子命中次数不足则忽略 |
| 格子最大存活时间 | `MATERIAL_LOCAL_MAP_MAX_AGE_S` | **8.0** | s | 是 | 超时未更新则衰减清除 |
| 深度降采样步长 | `MATERIAL_LOCAL_MAP_STRIDE` | **8** | px | 是 | 融合前像素步长 |
| 深度有效下限 | `DEPTH_MIN_M` | **0.15** | m | 否 | 近距过滤 |
| 深度有效上限 | `DEPTH_MAX_M` | **3.5** | m | 否 | 远距过滤 |
| 深度量化比例 | `DEFAULT_DEPTH_SCALE` | **1×10⁻³** | m/unit | 否 | uint16 毫米 → 米 |
| 帧质量门控 | `min_frame_valid_ratio` | **0.05** | — | 否 | 有效像素占比低于 5% 拒帧 |
| 融合频率下限 | `local_map_hz` clamp | **0.1** | Hz | 是 | 环境变量过小则抬升 |
| 融合频率上限 | `local_map_hz` clamp | **30.0** | Hz | 是 | 环境变量过大则截断 |
| 默认融合频率 | `MATERIAL_LOCAL_MAP_HZ` | **0.5** | Hz | 是 | 未设置时的默认值 |
| 推荐竞赛融合频率 | — | **1.0** | Hz | 是 | 降低 `fresh=False` 占比 |

### A.2 前方通畅检测与 standoff 建议

`forward_clearance()` 与 `suggested_standoff()` 输出写入 advice，供 Tier-1 / Tier-2 消费。

| 参数 | 代码默认值 | 单位 | 说明 |
|------|------------|------|------|
| 前向采样最大距离 | **1.2** | m | `forward_clearance.max_range_m` |
| 前向采样步长 | **≈0.025** | m | `max(res×0.5, 0.025)` |
| 检测走廊宽度 | **0.40** | m | 左 / 中 / 右三条采样线 |
| standoff 查询最大距离 | **1.50** | m | `suggested_standoff.max_range_m` |
| 期望净空（无障碍时） | **0.55** | m | `desired_clearance_m` |
| 建议 standoff 下限 | **0.35** | m | `min_standoff_m` |
| 建议 standoff 上限 | **0.90** | m | `max_standoff_m` |
| 无障碍时 `clear` | **True** | — | `distance_m` 取最大量程 **1.2** m |
| 有障碍时 `clear` | **False** | — | `distance_m` 为首次命中距离 |

### A.3 Advice 门控与刷新

| 参数 | 数值 | 说明 |
|------|------|------|
| Advice 有效窗口 | **2.0 s** | 本 tick 融合成功，或距上次成功融合 ≤2 s → `fresh=True` |
| 运动生效条件 | 三者同时为真 | `enabled ∧ apply ∧ fresh` |
| Client 日志周期 | **5.0 s** | `client_task` relay 摘要打印间隔 |
| ROS 话题 | `/material/local_map_advice` | `std_msgs/String` JSON |

**Advice 字段（量化语义）**

| 字段 | 类型 | 运动相关取值 |
|------|------|--------------|
| `clear` | bool / null | `True` → 可加速；`False` → standoff 候选 |
| `distance_m` | float / null | 最近障碍距离；通畅时为 **1.2** m |
| `suggested_standoff_m` | float / null | Tier-1 建议后退量，clamp 在 **[0.35, 0.90]** m |
| `frames_accepted` | int | 累计成功融合帧数（调试） |
| `frames_rejected` | int | 累计拒帧数（调试） |

### A.4 Tier-1：建图 standoff 策略参数

仅当 advice 满足 **§A.3 运动生效条件** 且 **`clear=False`** 且 **`distance_m < 1.0 m`**
时，`_shelf_clearance_m()` 可能推远站位；否则返回各调用点自身的 fallback（任务参数，
见任务文档）。

| 参数 | 代码名 | 数值 | 单位 | 说明 |
|------|--------|------|------|------|
| standoff 生效最大障碍距离 | `SHELF_MAP_STANDOFF_APPLY_MAX_DIST_M` | **1.00** | m | ≥此距离不推远 |
| 障碍安全余量 | `SHELF_MAP_STANDOFF_OBSTACLE_MARGIN_M` | **+0.10** | m | `max(..., distance + margin)` |
| 通用 standoff 上限 | `SHELF_MAP_STANDOFF_MAX_M` | **0.88** | m | 默认 cap |
| 任务 2 货架取货 standoff 上限 | `SHELF_PICK_MAP_STANDOFF_MAX_M` | **1.08** | m | 任务 2 专用 cap |
| 任务 2 桌面入口 standoff 上限 | `TABLE_ENTRY_MAP_MARGIN_MAX_M` | **0.42** | m | 任务 2 专用 cap |
| 推远下限 | — | **fallback** | m | 永不比标定 fallback **更近** |

**建图 standoff 接入范围（任务维度）**

| 任务 | Tier-1 standoff |
|------|-----------------|
| 任务 1 | **不接入** |
| 任务 2 | 货架取货 staging、桌面入口 |
| 任务 3 | **不接入** |

### A.5 Tier-2：建图运输加速参数

| 参数 | 环境变量 / 代码名 | 默认值 | 范围 | 说明 |
|------|-------------------|--------|------|------|
| 速度模式 | `MATERIAL_LOCAL_MAP_SPEED_MODE` | **`boost_only`** | `boost_only` \| `full` | 当前竞赛默认 |
| 开阔加速倍率 | `MATERIAL_LOCAL_MAP_CLEAR_BOOST` | **1.40** | **[1.0, 2.0]** | `clear=True` 时 |
| 推荐竞赛倍率 | — | **1.45** | — | §6.4 |
| 有障碍倍率（boost_only） | — | **1.0** | — | 不额外减速 |
| 有近距障碍倍率（full） | `LOCAL_MAP_NEAR_HIT_LINEAR_SCALE` | **0.52** | — | `distance_m ≤ 0.38 m` |
| 有中距障碍倍率（full） | `LOCAL_MAP_MID_HIT_LINEAR_SCALE` | **0.85** | — | **0.38 ~ 1.0 m** |
| 近距障碍分界 | `LOCAL_MAP_NEAR_HIT_MAX_DIST_M` | **0.38** | m | full 模式 |
| 速度缩放最大生效距离 | `LOCAL_MAP_SPEED_APPLY_MAX_DIST_M` | **1.00** | m | full 模式减速上界 |
| 缩放作用量 | — | **linear_x** | — | 角速度 **不变** |

**Tier-2 接入范围（阶段维度）**

| 任务 | 接入 | 不接入 |
|------|------|--------|
| 任务 1 | 桌↔货架 A* 运输、返程 | 观察站、扫描、放置链 |
| 任务 2 | 去货架、运输、桌面入口直进 | 最终桌面放置 |
| 任务 3 | 运输段 | 扫描站位、放置 / 插入 / 推箱 |

### A.6 环境变量速查（建图专用）

| 变量 | 默认 | 竞赛推荐 | 作用 |
|------|------|----------|------|
| `MATERIAL_LOCAL_MAP` | `0` | `1` | 总开关 |
| `MATERIAL_LOCAL_MAP_APPLY` | `0` | `1` | advice 是否影响运动 |
| `MATERIAL_LOCAL_MAP_HZ` | `0.5` | `1.0` | 融合频率 (Hz) |
| `MATERIAL_LOCAL_MAP_RES` | `0.05` | — | 栅格分辨率 (m) |
| `MATERIAL_LOCAL_MAP_MAX_AGE_S` | `8.0` | — | 格子衰减 (s) |
| `MATERIAL_LOCAL_MAP_MIN_HITS` | `2` | — | 障碍确认次数 |
| `MATERIAL_LOCAL_MAP_STRIDE` | `8` | — | 深度降采样 (px) |
| `MATERIAL_LOCAL_MAP_SPEED_MODE` | `boost_only` | `boost_only` | 速度策略 |
| `MATERIAL_LOCAL_MAP_CLEAR_BOOST` | `1.40` | `1.45` | 开阔加速倍率 |

### A.7 建图维度预期效果

| 指标 | 量化范围 | 备注 |
|------|----------|------|
| 开阔运输段线速度提升 | **+30% ~ +40%** | `CLEAR_BOOST` ≈ 1.40，相对原命令 |
| 整局任务时间缩短 | **约 10% ~ 15%** | 仅运输段受益；手臂 / 扫描 / 放置不在此列 |
| 放置链建图影响 | **0** | 放置阶段 `map_speed=False` / `linear_scale=1.0` |
| 单元测试 | **21 项** | `test_local_map.py` + `test_local_map_sidecar.py` |

### A.8 Fail-open 触发条件（建图失效 ≡ 未开启）

| 条件 | 运动结果 |
|------|----------|
| `MATERIAL_LOCAL_MAP=0` 或 sidecar 初始化失败 | 无 advice |
| `MATERIAL_LOCAL_MAP_APPLY=0` | 仅观测 / 日志 |
| `fresh=False` | Tier-1 / Tier-2 均不生效 |
| `clear=True` | Tier-1 不推远 |
| `distance_m ≥ 1.0 m` | Tier-1 不推远 |
| `boost_only` 且 `clear=False` | Tier-2 倍率 **1.0** |
| 调用点 `map_speed=False` | Tier-2 倍率 **1.0** |

---

## 12. 相关文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Client 总体架构
- [SHELF_TASK12_INTEGRATION.md](./SHELF_TASK12_INTEGRATION.md) — 任务 1/2 货架流程与观察站
- [NAVIGATION_V3_INTEGRATION.md](./NAVIGATION_V3_INTEGRATION.md) — 全局 A* 导航

---

*文档对应当前 `material_sorting_task` 代码树；若修改 `local_map.py` /
`local_map_sidecar.py` 中常量或任务接入点，请同步更新本文档 §4、§6 与附录 A。*
