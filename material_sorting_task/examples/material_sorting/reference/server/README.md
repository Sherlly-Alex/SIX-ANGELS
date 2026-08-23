# 参考 Server 与裁判实现

本目录保存本地参考 Server、裁判逻辑和裁判配置，用于理解场景接口、结构化指令、随机布局和得分判定。它不是正式 Client 入口，也不应与赛方官方 Server 或正式 `client_task.py` 并行启动。

## 文件职责

| 文件/目录 | 作用 |
| --- | --- |
| `material_sorting_server.py` | 本地参考仿真 Server，读取环境变量、加载场景、发布任务和运行裁判。 |
| `referee.py` | MuJoCo 参考裁判，在每个物理步更新触碰、夹起、放置、返回和碰撞里程碑。 |
| `referee_json/material_referee_config.json` | 时间、尝试次数、计分、区域和碰撞结构配置覆盖。 |

## 参考裁判流程

```text
物理状态 / 接触对 / 目标位置
                  |
              Referee.update()
                  |
      touch -> lift -> place -> return
                  |
        当前尝试结算与最高分保留
                  |
        game_info / task_info / results
```

每个任务默认最多 3 次尝试，任务一每个里程碑 10 分，任务二和任务三按照触碰、夹起、放置、返回分别计分。机器人撞击货架或外围墙体会记录碰撞，并取消该次返回分。参考裁判使用 MuJoCo 状态判定，不能作为正式 Client 的感知输入或 Server 私有真值接口。

## 随机化与渲染

参考 Server 从 `MATERIAL_SEED` 读取整数 seed；未提供时使用系统随机源。`MATERIAL_RANDOMIZE=1` 时生成随机布局，`MATERIAL_RANDOMIZE=0` 时使用固定布局。`MATERIAL_ENABLE_RENDER`、`MATERIAL_USE_GS` 和 `MUJOCO_GL` 控制本地渲染兼容性。正式远程运行应使用 `scripts/competitionctl.sh` 启动赛方镜像，不直接运行本目录脚本。

## 使用边界

- 参考 Server 可用于离线协议、场景随机化和消息格式调试。
- 参考裁判的阈值或分数配置变化，不等于官方比赛规则变化。
- Client 正式路径不得导入 `reference.server` 来取得目标坐标或布局真值。
- 修改参考实现后，应运行参考代码测试，并确认正式 Client 的 import 路径未被污染。

上级参考代码说明见 [`../README.md`](../README.md)，正式运行说明见仓库根目录 README。
