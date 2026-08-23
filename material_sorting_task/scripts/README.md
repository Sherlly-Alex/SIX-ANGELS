# 工具脚本、启动和验收

本目录包含正式启动、部署、测试、回放、模型验证和远程验收脚本。脚本只负责编排，不应在脚本里复制一套机器人控制逻辑。

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `run_client.sh` | 在 Client 容器内启动正式 ROS 2 Client。 |
| `competitionctl.sh` | 预检、启动 Server/Client、查看状态、停止容器和 Legacy 回退。 |
| `run_formal_tests.sh` | 运行 `tests/` 下不依赖 ROS 的正式单元测试。 |
| `check_workspace.py` | 检查必需文件、权重大小和 Python 语法。 |
| `run_desktop_grasp.sh` | 桌面抓取联调。 |
| `replay_scheduler_events.py` | EventLog 回放和训练数据准入。 |
| `validate_remote_run.py` | 校验 Client/Server/Events 的单次远程验收。 |
| `validate_remote_matrix.py` | 校验多 seed 运行矩阵。 |
| `benchmark_scheduler_policy.py` | 调度策略离线基准。 |
| `train_scheduler_policy.py` | 可选离线策略训练。 |
| `validate_scheduler_model.py` | 模型包、schema 和 provenance 校验。 |
| `validate_rl_shadow.py` | RL Shadow 运行门禁。 |
| `deploy_competition_release.sh`、`freeze_competition_release.sh` | 生成和部署不可变发布包。 |

## 远程仿真标准流程

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_random_$(date +%Y%m%d_%H%M%S)

bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" preflight
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" stop

# 终端一：Server；不传 seed 时使用 release env 默认值
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN"

# 终端二：Client
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" v2
```

需要随机场景时显式生成新的 seed，并把同一个 seed 传给 Server；Client 不需要设置 seed：

```bash
export SEED=$(shuf -i 1-999999999 -n 1)
export RUN="v2_random_$(date +%Y%m%d_%H%M%S)_$SEED"
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN" "$SEED"
```

默认正式配置：`task123_full`、`v2/heuristic`、ROS Domain `102`、CycloneDDS、YOLO。每次正式测试都使用全新的 Server/Client 和独立的 `remote_artifacts/$RUN/` 目录。

## 测试层级

```bash
# Python 正式回归
bash scripts/run_formal_tests.sh
python3 scripts/check_workspace.py

# 调度 EventLog 回放
python3 scripts/replay_scheduler_events.py run1.jsonl run2.jsonl \
  --min-decisions 1000 --require-training-ready \
  --dataset scheduler_replay.jsonl --output replay_report.json
```

`dry_run` 只验证阶段状态机，不产生机器人动作，也不代表 Server 得分。`nav_only`、`pregrasp_only`、`contact_only` 和 `lift_only` 会产生真实动作，测试时必须停止其他 Client 和机械臂控制节点。

## 日志与验收

每次运行至少保存：

- `server_$RUN.log`
- `client_$RUN.log`
- `scheduler_$RUN.jsonl`
- 使用的代码 commit、Server/Client 镜像名和随机 seed
- 关键阶段截图和异常截图

完成后运行：

```bash
python3 scripts/validate_remote_run.py \
  --client remote_artifacts/$RUN/client_$RUN.log \
  --server remote_artifacts/$RUN/server_$RUN.log \
  --events remote_artifacts/$RUN/scheduler_$RUN.jsonl \
  --require-candidate-application \
  --reject-duplicate-candidate-applications
```

失败运行不要覆盖或删除。先按任务、阶段和第一条 fatal 日志归因，再修改代码或标定参数。

## `competitionctl.sh` 命令

```text
preflight                  检查镜像、权重、必需文件和 Python 语法
server RUN [SEED]          启动官方 Server，并保存 server_RUN.log
client RUN [v2|legacy]     启动 Client，保存 client_RUN.log 和 EventLog
rollback RUN               用 Legacy 调度器启动一个新的 Client
status                     查看容器和最近的运行目录
stop                       删除当前 Client/Server 容器
freeze [OUTPUT_DIR]        生成带 commit/SHA256 的发布包
```

`server` 和 `client` 必须使用相同的 `RUN`。Server 的 `SEED` 决定随机场景，Client 不读取或
设置场景 seed；需要复现实验时保存该 seed 和 `competition_release.env`。脚本默认使用
`material_sorting:offline-server` 与 `material_sorting:offline-client`，若远程镜像名称不同，
先修改 release env 或通过 `MATERIAL_RELEASE_ENV` 指定配置。

## 单轮和多 seed 验收

单轮验收要求分数、任务终止、运行周期、执行耗时、fatal/safe-hold、候选应用和重复应用都通过：

```bash
python3 scripts/validate_remote_run.py \
  --client remote_artifacts/$RUN/client_$RUN.log \
  --server remote_artifacts/$RUN/server_$RUN.log \
  --events remote_artifacts/$RUN/scheduler_$RUN.jsonl \
  --expected-score 160 \
  --require-candidate-application \
  --min-applied-candidates 1 \
  --min-noncenter-applied 1 \
  --reject-duplicate-candidate-applications \
  --output remote_artifacts/$RUN/validation.json
```

measured-carry A/B 还要加 `--require-measured-carry`，并分别保存 guard 开启和关闭的运行目录。
五 seed 矩阵必须逐 seed 检查 EventLog，不能只统计平均分：

```bash
python3 scripts/validate_remote_matrix.py \
  --root remote_artifacts/v2_matrix \
  --seeds 20260817 20260818 20260819 20260820 20260821 \
  --expected-score 160 \
  --require-events \
  --require-candidate-application \
  --min-applied-candidates-per-seed 1 \
  --min-noncenter-applied-total 1 \
  --reject-duplicate-candidate-applications \
  --output remote_artifacts/v2_matrix/matrix_report.json
```

运行健康故障注入使用独立的 EventLog 和独立容器，不与得分运行混用：

```bash
python3 scripts/validate_runtime_health_run.py \
  --events remote_artifacts/health_run/scheduler_health.jsonl \
  --expect-recovered input_dropout \
  --expect-terminal joint_state_dropout \
  --output remote_artifacts/health_run/health_report.json
```

## 抓取指标

`record_grasp_metrics.py` 和 `plot_grasp_metrics.py` 是被动工具，只订阅反馈和日志，不发布
机器人命令。记录的 `/joint_states` `effort` 是 MuJoCo 执行器广义 effort，不是牛顿单位的
指尖力。建议在 Client 启动前开始记录，运行结束后保存 CSV、PNG、JSON 和摘要，并把图像与
对应 seed、commit 和日志一起归档。

## 发布和回退

```bash
bash scripts/freeze_competition_release.sh release_artifacts
bash scripts/deploy_competition_release.sh \
  /home/abc123/SIX-ANGELS-competition-COMMIT.tar.gz \
  /home/abc123/polaris/workspace/SIX-ANGELS-release-COMMIT
```

部署必须使用新的空目录，不覆盖当前运行版本。调度器现场回退只替换 Client 的调度引擎，
保留 Server 和原日志；完整代码回退应重新部署之前已校验的归档。
