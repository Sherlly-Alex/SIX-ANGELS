# 任务三导航问题交接（2026-08-09）

## 当前封存状态

本提交用于保存导航、抓放和速度调整的当前开发现场，并非三任务满分完成版。

- Git 基线：`e5b8aa4`（此前已验证的 160/160 版本）。
- 当前回归测试：全部通过，详见提交前验证记录。
- 有窗口 GS + YOLO + `task123_full` 连续实测：任务一 40/40、任务二 60/60，均无碰撞。
- 任务三：抓取成功并得到阶段分 40，但运输阶段导航失败，未完成货架放置。

## 首要已知问题：任务三货架末端导航

复现种子：

```bash
MATERIAL_SEED=485841371
```

任务三当前会在安全直退后左转，并直接驶向货架扫描位：

```text
goal=(-1.08, 0.54, 3.14)
```

这已经消除了旧流程“先到 y=0.90，再横移到 y=0.60，最后恢复朝向”的冗余转向；目标 y 也已改为 `0.54`，用于让物体最终落在货架白色长方体左侧并留出物理间隙。

但实测在距扫描位约 `0.05 m` 时，导航状态依次为 `navigating -> replanning -> navigating -> failed`，客户端报告：

```text
task 1 direct shelf-scan navigation stopped safely:
goal=(-1.08, 0.54, 3.14); nav_status=failed;
clearance=0.050m; footprint=transit_carry
```

裁判端当局记录：

```text
Task1: 40/40, collision=0
Task2: 60/60, collision=0
Task3 attempt 1: 40 points (contact=1, lift=1, place=0, return=1, collision=0)
```

因此下一步应集中审查货架末端 5 cm 的到达条件，而不是恢复旧的横移路线。重点检查：

1. `transit_carry` 轮廓在货架前的静态净空膨胀是否过于保守。
2. `NavigationController` 在终点容差内是否仍触发动态重规划/阻塞超时。
3. 扫描位 x 是否可略微外移，同时保持后续浅放动作的可达性。
4. 仅对货架末端建立受约束的直线终端段，避免放宽全局碰撞阈值。

不要简单禁用碰撞检测；目前失败属于安全停车，实测没有碰撞。

## 另一项已修复问题

任务一曾在升降完成后停住。原因是 `SlideLiftController` 使用双臂接触速度判断升降轴稳定，夹持物体时约 `0.024` 的臂部微动超过 `0.01` 阈值。现在只用升降轴速度判稳，同时保留双臂位置误差和日志中的最大关节速度；对应回归测试已加入 `tests/test_task1_pregrasp_executor.py`。

## 环境与启动

仓库：

```bash
cd /workspace/SIX-ANGELS
python3 scripts/check_workspace.py
.venv/bin/python -m pytest -q
```

当前系统 Python 的 pytest 与全局 anyio 插件版本不匹配，直接执行
`python3 -m pytest` 会在加载插件时失败；仓库 `.venv` 已验证可用，结果为
`77 passed, 1 warning`。该 warning 是货架开口朝向缺失时使用默认值，不影响启动。

本机参考 Server 原地运行时，模型资源不在 Git 仓库中，需要显式指定：

```bash
cd /workspace/SIX-ANGELS/examples/material_sorting/reference/server
MATERIAL_ASSETS_DIR=/workspace/material_sorting_task/examples/material_sorting/models \
MATERIAL_SEED=485841371 MATERIAL_USE_GS=1 MATERIAL_ENABLE_RENDER=1 \
MATERIAL_HEADLESS=0 MATERIAL_DEBUG_GRASP=1 \
python3 material_sorting_server.py
```

另开终端启动识别端：

```bash
cd /workspace/SIX-ANGELS/examples/material_sorting
source /opt/ros/humble/setup.bash
export PYTHONPATH=/workspace/SIX-ANGELS/examples/material_sorting:${PYTHONPATH:-}
python3 perception/box_detect.py --backend yolo --no-result-image --detection-log-period 5
```

再开终端启动正式三任务客户端：

```bash
cd /workspace/SIX-ANGELS/examples/material_sorting
source /opt/ros/humble/setup.bash
export PYTHONPATH=/workspace/SIX-ANGELS/examples/material_sorting:${PYTHONPATH:-}
export MATERIAL_EXECUTION_MODE=task123_full
python3 client_task.py
```

有窗口运行要求 `DISPLAY` 可用。当前环境已用 `MATERIAL_HEADLESS=0` 成功显示 Shentoon RobotStudio 窗口。参考 Server 兼容“screeninfo 找不到 primary monitor”的容器显示环境，但不会把有窗口运行改成无头模式。

停止旧进程后再开下一局，避免多个客户端同时发布控制命令。

## 随机场景范围

Server 的离散随机组合共 72 种：

- 桌面三个颜色位置：6 种排列；
- 桌子左右侧：2 种；
- 货架彩色物体层：3 种；
- 白色包装物层：2 种。

已枚举出覆盖全部 72 个组合的种子集合，但由于任务三末端导航失败，本版本没有完成 72 局物理满分验证。修复后必须重新执行随机矩阵；任何非 160 分结果都应保存种子、三任务分项、碰撞状态和客户端最后阶段日志。

## 提交边界

本提交不包含本地 `.orig/.rej` 补丁残留，也不包含运行时生成的裁判 JSON。它们不是可执行源代码。当前差异中的新增导航模块和测试属于本次开发现场的一部分。
