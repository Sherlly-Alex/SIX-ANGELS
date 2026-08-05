# SIX ANGELS Material Sorting Client

DG-202612 文旅机器人搬运赛题的参赛 Client 工作区。本仓库只维护参赛端代码；正式
Server、场景随机化和裁判由赛方镜像提供，不在这里修改。

## 当前状态

当前已经整理出指令解析、感知、导航、几何计算和 ROS 2 Client 入口。`client_task.py`
目前是安全骨架：能够接收并校验三项结构化指令、监听裁判状态和保持底盘停止，但抓取与
放置状态机仍需继续实现。

## 目录

```text
examples/material_sorting/
  client_task.py                 正式 Client 入口
  instruction_parser.py          结构化指令解析与校验
  task_orchestration.py          三任务编排辅助函数
  navigation/                    底盘导航模块
  perception/                    RGB-D / YOLO 感知模块
  desktop_grasp/                 任务 1/3 桌面双臂抓取模块
  mjcf/                          Client 坐标和碰撞计算所需模型
  models/mjcf, models/meshes/    Client 运动学所需资产
  reference/                     赛方示例与本地参考实现，不作为正式入口
scripts/
  run_client.sh                  容器内正式启动脚本
  run_desktop_grasp.sh           桌面抓取联调启动脚本
  setup_env_gpu.sh               ROS 2 / GPU 环境初始化
tests/                            不依赖 ROS 2 的单元测试
docs/                             架构和开发说明
```

## 运行

将仓库挂载到赛方 Client 容器的 `/workspace/baseline/material_sorting_task`，然后执行：

```bash
cd /workspace/baseline/material_sorting_task
bash scripts/run_client.sh
```

默认使用 YOLO：

```bash
MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
```

默认权重为 `examples/material_sorting/perception/checkpoints/best.pt`，可通过
`MATERIAL_YOLO_CHECKPOINT` 覆盖。

没有权重时可先用颜色后端检查通信链路：

```bash
MATERIAL_DETECT_BACKEND=color bash scripts/run_client.sh
```

正式运行必须使用：

```text
ROS_DOMAIN_ID=99
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

桌面抓取联调（仅任务 1 或 3）见 [docs/DESKTOP_GRASP.md](docs/DESKTOP_GRASP.md)。

## 测试

```bash
python3 -m unittest discover -s tests -t .
python3 scripts/check_workspace.py
```

开发约束和后续实现顺序见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。
