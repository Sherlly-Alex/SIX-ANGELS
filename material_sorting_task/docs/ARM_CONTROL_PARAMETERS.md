# 双臂控制与柔顺抓取参数说明

本文档对应当前 `SIX-ANGELS-upload` 版本，汇总双臂预抓取、连续柔顺接触、腕部贴合、夹持预紧和抬升过程的控制逻辑及参数。

## 1. 控制方案概述

当前抓取不是末端六维力传感器闭环力控，而是基于 ROS 2 位置控制接口实现的客户端导纳式柔顺控制：

1. 根据 RGB-D 目标中心和目标朝向计算双臂预抓取位姿。
2. 双臂移动到物体两侧，末端保持约 5° 内八姿态。
3. 记录左右腕部第 6 关节的位置、速度和 `effort` 基线。
4. 双臂以连续速度向内靠近，不再采用“每隔一段时间跳进固定距离”的方式。
5. 通过第 6 关节转角、速度和执行器 `effort` 相对基线的变化判断接触。
6. 哪一侧先接触，哪一侧先停止继续内移，并允许该侧腕部在有限角度内随物体表面转动贴合。
7. 两侧均接触并稳定后，锁定实际贴合角度。
8. 继续施加有限的夹持预紧量，然后保持夹持并抬升。

任务 1、任务 2 和任务 3 的桌面/货架双臂抓取均复用这套核心控制逻辑，具体任务执行器负责提供目标中心、目标高度、导航位置和后续搬运状态。

## 2. 几何与预抓取参数

参数定义位置：`examples/material_sorting/desktop_grasp/pregrasp_core.py`

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `BOX_WIDTH_Y` | 0.16 m | 默认物块横向宽度 |
| `PREGRASP_BACKOFF_X` | 0.08 m | 预抓取时手臂相对物体的纵向后撤量 |
| `SIDE_CLEARANCE` | 0.145 m | 预抓取时每侧额外横向安全间隙 |
| `HAND_Z_OFFSET` | 0.02 m | 手部接触中心相对物块中心的高度补偿 |
| `GRIPPER_OPEN` | 1.0 | 抓夹完全张开指令 |
| `GRASP_BACKOFF_X` | -0.02 m | 接触抓取阶段的纵向补偿 |
| `GRASP_INITIAL_PRELOAD` | 0.002 m | 初始夹持预紧量，2 mm |
| `TOE_IN_ANGLE_RAD` | 5° | 左右末端的内八角度，使接触面更容易贴合箱体 |

目标朝向对应的平面半尺寸为：

| 方向 | X 半尺寸 | Y 半尺寸 |
|---|---:|---:|
| `yaw0` | 0.12 m | 0.08 m |
| `yaw90` | 0.08 m | 0.12 m |

## 3. 连续柔顺接触参数

参数定义位置：`examples/material_sorting/executors/task1.py`

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `COMPLIANT_APPROACH_SPEED_M_S` | 0.0020 m/s | 尚未检测到接触时的连续向内速度，即 2.0 mm/s |
| `COMPLIANT_CONTACT_SPEED_M_S` | 0.0005 m/s | 任意一侧接触后的低速搜索速度，即 0.5 mm/s |
| `COMPLIANT_PRELOAD_SPEED_M_S` | 0.0010 m/s | 双侧贴合后施加预紧的速度，即 1.0 mm/s |
| `COMPLIANT_DT_MAX_S` | 0.10 s | 单次控制周期用于积分的最大时间，避免卡顿后指令突跳 |
| `COMPLIANT_SOFT_MAX_M` | 0.004 m | 常规软接触搜索最大内移量，4 mm |
| `COMPLIANT_POST_ALIGN_PRELOAD_M` | 0.002 m | 双侧贴合并锁腕后的追加预紧量，2 mm |
| `COMPLIANT_ABSOLUTE_MAX_M` | 0.006 m | 搜索加预紧的绝对内移上限，6 mm |
| `COMPLIANT_SINGLE_SIDE_WAIT_S` | 2.0 s | 仅单侧接触时允许等待另一侧接触的最长时间 |
| `COMPLIANT_RETRY_BACKOFF_M` | 0.001 m | 单侧接触失败后重试前的回退量，1 mm |
| `COMPLIANT_MAX_RETRIES` | 1 | 最大重新接触次数 |

连续内移量按控制周期计算：

```text
本周期内移量 = 当前接触阶段速度 × min(实际控制周期, 0.10 s)
```

因此控制目标会随每个控制周期平滑更新，而不是每 0.25 秒突然内收 0.5 mm。

## 4. 腕部接触与贴合参数

参数定义位置：`examples/material_sorting/desktop_grasp/pregrasp_core.py`

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `WRIST_BASELINE_TIME_S` | 0.40 s | 接触搜索前采集腕部反馈基线的时间 |
| `WRIST_BASELINE_MIN_SAMPLES` | 8 | 建立基线所需的最少样本数 |
| `WRIST_EFFORT_FILTER_ALPHA` | 0.25 | `effort` 低通滤波新样本权重 |
| `WRIST_MIN_EFFORT_DELTA` | 0.35 | 判定接触所需的最小 `effort` 增量 |
| `WRIST_EFFORT_NOISE_MULTIPLIER` | 5.0 | 根据基线噪声自适应放大接触阈值的倍数 |
| `WRIST_FREE_ANGLE_LIMIT_RAD` | ±8° | 接触贴合阶段腕部允许偏离标称角度的范围 |
| `WRIST_CONTACT_MIN_ROTATION_RAD` | 1.5° | 用于确认有效表面接触的最小腕部转角变化 |
| `WRIST_CONTACT_CONFIRM_TIME_S` | 0.15 s | 接触条件持续成立的确认时间 |
| `WRIST_ALIGN_VELOCITY_RAD_S` | 0.05 rad/s | 认为腕部已基本停止转动的速度阈值 |
| `WRIST_ALIGN_STABLE_TIME_S` | 0.20 s | 腕部低速状态需要持续的稳定时间 |
| `WRIST_PRELOAD_SOFT_LIMIT_DELTA` | 2.5 | 预紧阶段允许的软 `effort` 增量上限 |
| `WRIST_ABSOLUTE_EFFORT_LIMIT` | 6.0 | 腕部执行器 `effort` 的绝对安全上限 |

接触判断综合使用：

- 第 6 关节相对基线的转角变化；
- 第 6 关节执行器 `effort` 相对基线的变化；
- 第 6 关节运动速度；
- 条件连续成立的时间。

这里的 `effort` 来自 `/joint_states` 中的关节执行器广义力，不等同于指尖接触力，也不能直接按牛顿解释。系统没有公开独立的末端接触、六维力/力矩或指尖压力话题。

## 5. 位置跟踪与抬升参数

参数定义位置：`examples/material_sorting/desktop_grasp/pregrasp_core.py`

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `SPINE_REFERENCE_Z` | 1.32163718 m | 升降柱高度换算参考值 |
| `SPINE_MIN` | -0.04 m | 升降柱指令下限 |
| `SPINE_MAX` | 0.87 m | 升降柱指令上限 |
| `FEEDBACK_POS_TOL` | 0.03 | 常规关节/升降柱到位容差 |
| `SQUEEZE_CONTACT_POS_TOL` | 0.24 | 接触与挤压阶段的手臂位置容差 |
| `FEEDBACK_VEL_TOL` | 0.01 | 判定静止的最大关节速度 |
| `FEEDBACK_STABLE_TIME` | 0.50 s | 到位条件需要保持的时间 |
| `COMMAND_RATE_PER_S` | 1.20 | 关节空间指令插值速率；旋转关节主要可理解为 rad/s |
| `SLIDE_COMMAND_RATIO` | 0.30 | 常规动作中升降柱相对关节插值速率的比例 |
| `LIFT_HEIGHT` | 0.15 m | 抓取完成后的默认抬升高度 |
| `LIFT_SLIDE_COMMAND_RATIO` | 0.05 | 抬升时升降柱指令速率比例 |
| `LIFT_ARM_POSITION_TOL` | 0.24 | 保持夹持并抬升时的双臂位置容差 |

## 6. Server 端执行器参数

参数定义位置：`examples/material_sorting/models/mjcf/mobile_chassis/mmk2/mmk2_control.xml`

| 执行器 | `kp` | 其他限制 |
|---|---:|---|
| 左右臂关节 1～3 | 1000 | Server 固定位置控制增益 |
| 左右臂关节 4～6 | 350 | Server 固定位置控制增益 |
| 左右抓夹 | 3 | 控制范围 0～1，力范围 -1～1 |

这些是仿真 Server 中的位置执行器参数。当前 Client 不能在抓取过程中动态降低或恢复这些 `kp`，所以客户端通过改变位置目标、限制腕部角度和监测 `effort` 来实现近似柔顺，而不是真正的关节力矩控制。

## 7. 参数调节建议

建议一次只改一个参数，并在多个随机种子下对比抓取成功率、接触峰值和物块滑落情况。

### 抓取容易滑落

优先尝试：

1. 小幅增加 `COMPLIANT_POST_ALIGN_PRELOAD_M`，例如每次增加 0.0005 m；
2. 观察左右腕部峰值 `effort` 是否仍明显低于软上限；
3. 必要时再小幅增加 `GRASP_INITIAL_PRELOAD`。

不要先提高连续靠近速度。速度过大会加剧冲击和物块弹开。

### 接触冲击过大或物块被弹开

优先尝试：

1. 降低 `COMPLIANT_APPROACH_SPEED_M_S`；
2. 降低 `COMPLIANT_PRELOAD_SPEED_M_S`；
3. 减小 `COMPLIANT_POST_ALIGN_PRELOAD_M`；
4. 检查目标中心和物块横向半宽估计是否准确。

### 经常只有单臂接触

优先检查目标中心与底盘横向对齐误差，而不是直接增大夹持量。还可依次尝试：

1. 适当增大 `COMPLIANT_SINGLE_SIDE_WAIT_S`；
2. 检查两侧 IK 目标是否关于物体中心对称；
3. 检查 RGB-D 中心估计是否受到颜色、遮挡或深度空洞影响；
4. 最后才小幅增加 `COMPLIANT_SOFT_MAX_M`。

### 误判接触

可小幅提高 `WRIST_MIN_EFFORT_DELTA` 或 `WRIST_CONTACT_MIN_ROTATION_RAD`。阈值过高会造成已经接触却继续挤压，因此每次调整后都应检查抓取曲线。

## 8. 安全边界

- `COMPLIANT_SOFT_MAX_M` 是常规搜索边界；
- `COMPLIANT_ABSOLUTE_MAX_M` 是搜索和预紧的绝对位移边界；
- `WRIST_PRELOAD_SOFT_LIMIT_DELTA` 是相对基线的软 `effort` 边界；
- `WRIST_ABSOLUTE_EFFORT_LIMIT` 是绝对 `effort` 安全边界；
- 单侧长时间接触会触发回退/重试，超过次数后阻塞当前任务，避免持续挤压。

这些限制不应同时大幅放宽。若需要增强夹持，优先以 0.5 mm 量级调整追加预紧，并结合量化曲线验证。

