# Guarded 原型部署说明

## 1. 定位

本原型不改变 SIX-ANGELS 的正式控制权：`CompetitionController` 状态机始终负责 Task 1–3 流程，V2 Heuristic 只辅助选择有限宏动作，Guarded 只是在同一安全候选层上显式启用 RL。默认 Client 模式为 `heuristic`。

## 2. 冻结资产

仓库根目录包含：

```text
release_assets/rl_guarded/scheduler_policy.zip
release_assets/rl_guarded/scheduler_policy.zip.metadata.json
release_assets/rl_guarded/scheduler_guarded_approval.json
```

固定哈希：

```text
model    5340c47b1fbcfaf799667e1b36a2474e7809817abca78e38875f690a222fb785
approval 0f92ad4a1a0039c9dbefc54d3710aeba38910b0aaf259a443db7dd9af9a95f0a
```

## 3. 部署

```bash
git clone --branch 'qzhRL版' --single-branch \
  https://github.com/qzhvscode/SIX-ANGELS.git SIX-ANGELS-qzhRL

export PROJECT="$PWD/SIX-ANGELS-qzhRL"
cd "$PROJECT"
python3 material_sorting_task/scripts/check_workspace.py
```

需要预先加载：

```text
material_sorting:offline-server
material_sorting:offline-client
material_sorting:offline-client-rl-shadow-e3f5284
```

配置显示：

```bash
source material_sorting_task/config/competition_release.env
printf 'default=%s\n' "$MATERIAL_DEFAULT_CLIENT_MODE"
printf 'model=%s\n' "$MATERIAL_RL_MODEL_SHA256"
printf 'approval=%s\n' "$MATERIAL_RL_APPROVAL_SHA256"
```

`default` 必须为 `heuristic`。

## 4. 正式 Heuristic 运行

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
xhost +SI:localuser:root

bash material_sorting_task/scripts/competitionctl.sh stop
bash material_sorting_task/scripts/competitionctl.sh preflight heuristic
```

终端一：

```bash
export PROJECT=/path/to/SIX-ANGELS-qzhRL
export RUN=official_heuristic_$(date +%Y%m%d_%H%M%S)
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" server "$RUN"
```

终端二：

```bash
export PROJECT=/path/to/SIX-ANGELS-qzhRL
export RUN=official_heuristic_YYYYMMDD_HHMMSS
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" heuristic
```

省略模式时仍是 Heuristic：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN"
```

## 5. 显式 Guarded 原型

先执行只读预检：

```bash
cd "$PROJECT"
bash material_sorting_task/scripts/competitionctl.sh preflight guarded
```

预检必须同时验证模型、metadata、Approval、Schema 和哈希。通过后才可在终端二显式启动：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "$RUN" guarded
```

没有 `guarded` 参数就不会发生 RL 实际接管。

## 6. 回退

Guarded 出现模型错误、超时、非法动作、Approval 拒绝或其他异常时，停止 Client，保留 Server 与全部日志，然后以新 RUN 回到 Heuristic：

```bash
export ROLLBACK_RUN="${RUN}_heuristic"
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" rollback "$ROLLBACK_RUN"
```

完全绕过 V2 调度辅助：

```bash
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" client "${RUN}_legacy" legacy
```

回退不得覆盖原 RUN，不得删除 Scheduler EventLog，也不得修改 Task 1–3 执行器。

## 7. 运行边界

- 状态机永远是主控制器。
- Heuristic 是正式可用的确定性辅助和 RL 回退。
- Guarded 只用于后续研发样品。
- RL 不输出底盘速度或关节命令。
- Server/裁判结果是最终验收依据。
