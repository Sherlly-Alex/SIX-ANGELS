# RL-2 成功率优先运行说明

RL-2 在独立工作区实验。冻结比赛版的 `competition_release.env`、
`release_assets/rl_guarded/` 模型和 approval **不得替换**，直到 Shadow 10/10
和 Guarded 10/10 全部 160 分并且 `select-candidate` 通过。

## 运行时总开关

```text
MATERIAL_SCHEDULER_RL_ENABLED=0   # 默认；未设置也视为 0
```

| 模式 | 开关 | 行为 |
|---|---|---|
| heuristic / rollback | 0 | V2 Heuristic，不加载模型 |
| shadow | 1 | Heuristic 控制，RL 只记录建议 |
| guarded | 1 | 仅在 mask 与审批链通过后选择安全候选 |

任意模型、SHA、metadata、approval、超时或非法动作失败时，控制器和任务状态机
保持原样，当次或当前进程改用 Heuristic。

```bash
bash material_sorting_task/scripts/competitionctl.sh client RUN heuristic
bash material_sorting_task/scripts/competitionctl.sh client RUN shadow
bash material_sorting_task/scripts/competitionctl.sh client RUN guarded
bash material_sorting_task/scripts/competitionctl.sh rollback RUN
```

## 一键关闭

```bash
bash material_sorting_task/scripts/competitionctl.sh stop
bash material_sorting_task/scripts/competitionctl.sh preflight heuristic
bash material_sorting_task/scripts/competitionctl.sh rollback "$RUN"
export MATERIAL_SCHEDULER_RL_ENABLED=0
```

## 门禁

`rl2ctl.sh` 覆盖仿真采集、官方串行 Shadow/Guarded、数据集覆盖审计、6 候选训练、
500 种子成功率优先盲测和选型。覆盖未通过禁止训练。没有候选通过时：

```text
selected_model=null
promotion_allowed=false
effective_policy=heuristic
```

Guarded 晋级仍要求独立 approval 和 1 场 canary + 9 场矩阵。任一场非 160 分，
RL-2 不晋级，也不修改冻结比赛版本。
