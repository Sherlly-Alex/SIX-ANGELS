# Airbot Play 眼在手外 RGB-D 到达物块上方说明

本文说明 `examples/planning/airbot_reach_point.py` 当前实现的整体框架：固定外部深度相机 `eye_side` 如何参与坐标估计，脚本如何估计机械臂基座和夹爪位置，以及如何把绿色物块上方作为点到点规划目标。

## 1. 任务目标

当前脚本不是完整抓取策略，而是一个可调试的到达规划 baseline：

```text
固定外部相机 eye_side 获取 RGB-D
通过彩色基座标记估计 arm_base 在 world 中的位姿
通过 FK 得到夹爪在 arm_base 中的位置
把夹爪位置转换到估计 world 坐标
把物块上方点转换到 arm_base 坐标
用位置 IK 求关节目标
在关节空间规划并执行轨迹
输出到达误差、碰撞诊断和可选相机数据
```

核心实现文件：

```text
examples/planning/airbot_reach_point.py
```

相关模型和场景：

```text
models/mjcf/manipulator/robot_airbot_play_eye_side_qz_lab3.xml
models/mjcf/scene/qz_lab3.xml
```

## 2. 总体数据流

脚本的数据流可以按下面理解：

```text
qz_lab3 eye_side 相机参数
        |
        v
RGB-D 图像 + 深度图
        |
        v
彩色基座标记分割和深度反投影
        |
        v
arm_base 位姿估计
        |
        +------------------+
        |                  |
        v                  v
夹爪 FK: gripper_arm_base  目标: block top + clearance
        |                  |
        v                  v
gripper_est_world          target_arm_base
        |                  |
        +--------+---------+
                 v
            IK + 关节空间规划
                 |
                 v
            MuJoCo 位置控制执行
```

代码中对应的主要函数：

| 模块 | 函数或常量 | 作用 |
| --- | --- | --- |
| 相机配置 | `DOCUMENTED_EYE_SIDE_*` | 记录 qz_lab3 文档里的 `eye_side` 位姿和视场参数 |
| 模型选择 | `make_cfg()` | 在需要 `eye_side` 或 `global_depth` 时加载带外部相机和 RGB-D 标记的 MJCF |
| 物块目标 | `configure_target_block()` | 放置可见物块，并计算物块上方目标点 |
| 深度反投影 | `depth_pixel_to_world_with_camera_pose()` | 把深度像素从相机坐标转成 world 坐标 |
| 基座估计 | `estimate_arm_base_pose_from_rgbd_markers()` | 从 RGB-D 彩色标记估计 `arm_base` 位姿 |
| 夹爪位置 | `endpoint_arm_base()` | 从 FK 传感器读取末端在 `arm_base` 下的位置 |
| 实时坐标 | `update_live_coordinates()` | 组合基座估计和夹爪 FK，得到 `gripper_est_world` |
| IK | `solve_position_only_ik()` | 只约束末端位置，枚举姿态候选并选误差小的解 |
| 路径 | `plan_joint_path()` | 生成平滑关节空间点到点轨迹 |
| 执行 | `execute_joint_path()` | 用位置控制执行路径，并记录诊断 trace |

## 3. 坐标系

严格来说，脚本里不止两个坐标系。`world` 和 `arm_base` 是规划主坐标系；相机链路里还会经过像素坐标、深度相机光学坐标和 MuJoCo camera 坐标。

```text
pixel:
  图像像素坐标 [u, v]。
  u 是图像列，v 是图像行。

camera optical:
  深度相机光学坐标系。
  深度图的 depth 值先结合相机内参 K 反投影到这个坐标系。

mujoco camera:
  MuJoCo 相机局部坐标系。
  脚本会把 optical 坐标转成 MuJoCo camera 约定，当前实现里会翻转 Z 轴符号。

world:
  MuJoCo worldbody 坐标系，Z 轴向上。
  eye_side 的外参表示为相机在 world 中的位置和姿态。

arm_base:
  机械臂基座坐标系，IK 和大多数规划目标默认使用这个坐标系。
```

深度相机给出的不是直接的 `world` 或 `arm_base` 点，而是：

```text
pixel [u, v] + depth
    -> camera optical
    -> mujoco camera
    -> world
    -> arm_base
```

其中 `camera -> world` 依赖已知的 `eye_side` 相机外参；`world -> arm_base` 优先依赖 RGB-D 彩色基座标记估计出来的 `arm_base` 位姿。

默认输入目标使用 `arm_base`：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --target 0.28 0.0 0.24 --target-frame arm_base
```

也可以显式输入 world 目标：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --target 0.28 0.0 1.02 --target-frame world
```

当 world 目标需要转到 `arm_base` 时，`target_to_arm_base()` 会优先使用实时 RGB-D 估计到的基座位姿。如果当前没有有效估计，才回退到 MuJoCo 模型里的 `arm_base` ground truth 变换。

## 4. 外部相机 eye_side

默认目标相机是：

```text
DEFAULT_TARGET_CAMERA = "eye_side"
```

`eye_side` 来自 `models/mjcf/scene/qz_lab3.xml`，脚本中记录的文档参数是：

```text
position world = [-0.324, 0.697, 1.02]
fovy = 72.02 deg
x axis = [0.828, -0.561, 0.0]
y axis = [0.394, 0.582, 0.702]
```

当 `--target-camera eye_side` 或默认相机触发时，`main()` 会让 `make_cfg()` 加载：

```text
mjcf/manipulator/robot_airbot_play_eye_side_qz_lab3.xml
```

这个模型变体包含：

- Airbot Play 机械臂。
- qz_lab3 中固定的外部 `eye_side` 深度相机。
- 放在 `arm_base` 附近的 RGB-D 彩色基座标记。
- 可见的 `target` / `target_box` 物块。

注意：当前调试可视化已经不再画橙色的 `eye_side` 相机盒子和视锥线。这个改动只影响 MuJoCo 窗口中的辅助可视化，不影响 `eye_side` 相机本身。深度图、录制、像素反投影和基座估计仍然使用这个固定外部相机。

## 5. 深度像素到 world 坐标

深度像素反投影入口是：

```text
target_world_from_depth_pixel()
depth_pixel_to_world_with_camera_pose()
```

流程如下：

```text
输入 pixel = [u, v]
读取 target camera 的 depth image
读取相机内参 K
用 depthPixelToCamera 得到 optical camera 坐标
把 optical Z 轴转换到 MuJoCo camera 约定
读取 eye_side 的 world 位姿
用 camera rotation + camera position 转到 world
再按需要转成 arm_base 目标
```

命令示例：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-depth-pixel 320 240
```

如果希望到达深度点上方，而不是直接到达表面点，可以加 `arm_base +Z` 偏移：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-depth-pixel 320 240 --depth-target-z-offset 0.08
```

## 6. RGB-D 基座位姿估计

脚本通过四个彩色小球标记估计机械臂基座：

| 标记 | 局部坐标来源 | 颜色 |
| --- | --- | --- |
| `base_depth_marker_o` | `[-0.075, -0.075, 0.115]` | 洋红 |
| `base_depth_marker_x` | `[0.075, -0.075, 0.115]` | 黄色 |
| `base_depth_marker_y` | `[-0.075, 0.075, 0.115]` | 绿色 |
| `base_depth_marker_diag` | `[0.075, 0.075, 0.115]` | 蓝色 |

这些局部坐标都定义在 `arm_base` 下。估计步骤：

1. `marker_color_mask()` 把 RGB 图按色度距离分割出每种颜色。
2. `largest_connected_mask()` 取最大连通区域，并过滤太小的噪声块。
3. `estimate_marker_center_from_rgbd()` 在标记 mask 内取较近深度分位数，避免用到小球背面或混入背景。
4. 深度像素通过 `depth_pixel_to_world_with_camera_pose()` 反投影到 world 表面点。
5. 沿相机射线方向补偿小球半径，得到估计的小球中心。
6. `rigid_transform_local_to_world()` 用局部标记点和 world 标记点做刚体配准。

如果可见标记不少于 3 个，使用 SVD 刚体拟合，方法名是：

```text
rgbd_markers_rigid_fit
```

如果只看到 1 到 2 个标记，则退化为平移估计，姿态固定为单位阵，方法名是：

```text
rgbd_markers_translation_only_fixed_orientation
```

估计结果会写入 `env.live_coordinates`：

```text
arm_base_est_world
arm_base_est_rotation_world
arm_base_est_method
arm_base_est_marker_count
arm_base_est_markers
```

## 7. 夹爪位置估计

夹爪在 `arm_base` 坐标系下的位置来自正运动学传感器：

```text
endpoint_arm_base(env) -> env.sensor_endpoint_posi_local
```

在 `update_live_coordinates()` 里，脚本用 RGB-D 估计出的基座位姿把夹爪 FK 转成 world：

```text
gripper_arm_base = endpoint_arm_base(env)
gripper_world_est = base_world_est + base_rot_est @ gripper_arm_base
```

因此实时输出里的 `gripper_est_world` 不是直接从深度图识别夹爪，而是：

```text
RGB-D 估计的 arm_base world 位姿 + 机械臂 FK 得到的末端 arm_base 坐标
```

如果加上调试参数：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --show-ground-truth-coordinate-check
```

脚本还会额外打印 MuJoCo ground truth 基座和末端位置，用于对比 RGB-D 估计误差。默认不显示 ground truth，避免把仿真真值混进正常流程说明。

## 8. 物块上方目标

物块目标由 `configure_target_block()` 管理。默认可见物块是：

```text
body = target
geom = target_box
default center arm_base = [0.28, 0.0, 0.05]
default half size = [0.05, 0.05, 0.05]
default clearance = 0.10 m
```

如果没有手动指定 `--target-block-pos`，且检测到桌面，脚本会把物块中心的 Z 放在桌面上方：

```text
block_center_z = table_upper_z + half_size_z
```

真正传给 IK 的目标点不是物块中心，而是物块上表面再上抬 clearance：

```text
target_arm_base = block_center_arm_base
target_arm_base.z += half_size.z + target_block_clearance
```

非交互模式下直接到达物块上方：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-block
```

指定物块中心：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-block --target-block-pos 0.32 0.05 0.05
```

指定物块上方高度：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-block --target-block-clearance 0.15
```

交互模式里也可以移动物块：

```text
block 0.32 0.05 0.05
```

这会重新放置可见物块，并把目标更新为物块上方点。

## 9. IK 和轨迹规划

IK 入口是：

```text
solve_position_only_ik()
```

脚本当前只约束末端位置，不固定末端姿态。实现方式是：

1. `orientation_candidates()` 根据目标位置枚举多组末端姿态。
2. 对每组姿态调用 `AirbotPlayIK.properIK()`。
3. 用 MuJoCo FK 复核候选解的末端位置误差。
4. 如果启用桌面检查，则过滤最终姿态或插值路径中碰到桌子的候选。
5. 在剩余候选中按 FK 误差和关节运动代价排序，取最优解。

轨迹入口是：

```text
plan_joint_path()
```

它在当前关节角和目标关节角之间做关节空间插值，并用 smoothstep 曲线让起止更平滑：

```text
blend = t * t * (3 - 2 * t)
```

执行入口是：

```text
execute_joint_path()
```

每个控制步都会把 6 个机械臂关节写入 action，夹爪保持当前开合量。执行过程中可按 `--path-diagnostic-stride` 采样记录：

- 当前时间。
- 当前关节角。
- `endpoint_arm_base`。
- `endpoint_world`。
- 桌面接触数量。
- 可视几何是否进入桌面薄盒。

## 10. 桌面避碰

脚本会尽量读取模型中的真实桌面范围，并转成 `arm_base` 坐标。默认不会把桌子当作虚构平面，而是从 MuJoCo 模型几何体得到桌面实体范围。

对于桌面上方目标，IK 和路径会过滤桌面碰撞。也就是说，候选 IK 解不仅要让末端到达目标点，还要避免最终姿态或插值路径中的机械臂几何体碰到桌面。

如果只想做简单点到点调试，可以禁用桌面：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --no-table
```

## 11. 相机录制和预览

保存目标相机的一帧 RGB-D 预览：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --save-depth-preview reports\eye_side_preview --depth-preview-only
```

交互模式下录制相机帧：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --interactive --render --record-camera-dir reports\eye_side_record
```

录制目录中会包含：

```text
rgb/*.png
depth/*.npy
depth_vis/*.png
camera_poses.jsonl
```

`camera_poses.jsonl` 会记录：

- 相机名和相机 id。
- 相机 world 位姿。
- 相机内参。
- 深度可视化模式。
- `documented_eye_side_camera` 元数据。
- 当前 `target_block` 信息。
- 可选 `live_coordinates` 实时估计结果。

## 12. 常用运行命令

到达默认目标点：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render
```

到达物块上方：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-block
```

用深度像素作为目标：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-depth-pixel 320 240
```

交互式输入：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --interactive --render
```

保存轨迹诊断：

```powershell
.\.venv39\Scripts\python.exe examples\planning\airbot_reach_point.py --render --target-from-block --save-trace reports\reach_block_trace.json
```

只做语法检查：

```powershell
.\.venv39\Scripts\python.exe -m py_compile examples\planning\airbot_reach_point.py
```
