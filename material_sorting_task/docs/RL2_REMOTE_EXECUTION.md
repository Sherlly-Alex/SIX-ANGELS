# RL-2 远程采集、训练、验收与回退命令

本流程使用独立工作区 `/home/abc123/polaris/workspace/SIX-ANGELS-rl2`。
官方 Server 始终单路运行；仿真、训练和盲测在训练容器内并行。数据审计只输出
JSON、CSV 和失败清单，不生成图表。

## 1. 登录与宿主机预检

```bash
ssh -p 8001 abc123@8.130.157.142

export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-rl2
cd "$PROJECT"

python3 material_sorting_task/scripts/check_workspace.py
bash material_sorting_task/scripts/competitionctl.sh preflight heuristic
python3 -S material_sorting_task/scripts/rl2_cli.py collect-official \
  --output-root /tmp/rl2_dry_run \
  --mode heuristic \
  --seeds 1 2 \
  --dry-run
```

最后一条故意使用 `python3 -S`，用于证明宿主机没有 NumPy/Torch 时仍能编排官方
Docker采集。

## 2. 准备隔离训练容器

复用已安装 Gymnasium、Stable-Baselines3 和 sb3-contrib 的旧训练容器，先固化为
独立镜像，再挂载RL-2代码和产物目录：

```bash
export SOURCE_TRAIN_CTN=material_sorting_rl1_bandit_6100213
export TRAIN_IMAGE=material_sorting:rl2-train-20260824
export TRAIN_CTN=material_sorting_rl2_20260824

docker inspect -f '{{.State.Running}}' "$SOURCE_TRAIN_CTN"
docker commit "$SOURCE_TRAIN_CTN" "$TRAIN_IMAGE"
docker rm -f "$TRAIN_CTN" 2>/dev/null || true

docker run -dit --gpus all --network none --ipc host \
  --name "$TRAIN_CTN" \
  --entrypoint bash \
  -v "$PROJECT/material_sorting_task":/workspace/baseline:ro \
  -v "$PROJECT/release_assets":/workspace/release_assets:ro \
  -v "$PROJECT/remote_artifacts":/workspace/out:rw \
  "$TRAIN_IMAGE" -lc 'exec tail -f /dev/null'

docker exec -i "$TRAIN_CTN" bash -lc '
python3 - <<'"'"'PY'"'"'
import torch
import gymnasium
import stable_baselines3
import sb3_contrib
print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("gpu =", torch.cuda.get_device_name(0))
print("gymnasium =", gymnasium.__version__)
print("stable_baselines3 =", stable_baselines3.__version__)
print("sb3_contrib =", sb3_contrib.__version__)
PY
'
```

## 3. 8路并行生成仿真数据

终端一：

```bash
export TRAIN_CTN=material_sorting_rl2_20260824
export SIM_RUN=rl2_data_20260824_r1

docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py generate-sim \
  --output-root /workspace/out/'"$SIM_RUN"'/simulation \
  --workers 8 \
  --episodes-per-worker 334 \
  --seed-start 41000 \
  --profile-config /workspace/baseline/examples/material_sorting/learning/configs/project_simulation_v2.json
'
```

终端二查看状态：

```bash
export TRAIN_CTN=material_sorting_rl2_20260824
export SIM_RUN=rl2_data_20260824_r1

docker exec "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py status \
  --output-root /workspace/out/'"$SIM_RUN"'/simulation
'
```

验收：

```bash
docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py validate-sim \
  --input-root /workspace/out/'"$SIM_RUN"'/simulation \
  --expected-workers 8 \
  --minimum-decisions 24000 \
  --output /workspace/out/'"$SIM_RUN"'/simulation_acceptance.json
'

cat "$PROJECT/remote_artifacts/$SIM_RUN/simulation_acceptance.json"
```

## 4. 官方Server串行Shadow采集

终端一自动依次运行。任一seed不是160分会立即停止：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-rl2
export OFFICIAL_RUN=rl2_official_shadow_20260824_r1
export OFFICIAL_ROOT="$PROJECT/remote_artifacts/$OFFICIAL_RUN"

cd "$PROJECT"
bash material_sorting_task/scripts/competitionctl.sh stop
bash material_sorting_task/scripts/competitionctl.sh preflight shadow

python3 -S material_sorting_task/scripts/rl2_cli.py collect-official \
  --output-root "$OFFICIAL_ROOT" \
  --mode shadow \
  --seeds \
    20260824 20260825 20260826 20260827 20260828 \
    20260829 20260830 20260831 20260832 20260833
```

终端二查看当前Client：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-rl2
export OFFICIAL_ROOT="$PROJECT/remote_artifacts/rl2_official_shadow_20260824_r1"

tail -F "$OFFICIAL_ROOT/current/client.log" | grep --line-buffered -Ei \
'controller=|score=|rl_shadow|fallback|timeout|blocked|safe_hold|collision|executor error'
```

矩阵验收：

```bash
python3 "$PROJECT/material_sorting_task/scripts/validate_remote_matrix.py" \
  --root "$OFFICIAL_ROOT" \
  --seeds \
    20260824 20260825 20260826 20260827 20260828 \
    20260829 20260830 20260831 20260832 20260833 \
  --expected-score 160 \
  --require-events \
  --require-candidate-application \
  --reject-duplicate-candidate-applications \
  --min-applied-candidates-per-seed 1 \
  --max-interval-p99-ms 125 \
  --output "$OFFICIAL_ROOT/official_matrix_acceptance.json"

cat "$OFFICIAL_ROOT/official_matrix_acceptance.json"
```

Shadow验收需要训练容器中的NumPy环境：

```bash
docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/validate_rl_shadow.py \
  /workspace/out/'"$OFFICIAL_RUN"'/v2_multiseed_*/scheduler_v2_multiseed_*.jsonl \
  --min-suggestions 3000 \
  --max-inference-p95-ms 25 \
  --max-fallback-rate 0.01 \
  --expected-model-sha256 364d5cf5e94be08597cd9bde643b1ed132ab347ec520bd8e16d2d24fc68e3322 \
  --output /workspace/out/'"$OFFICIAL_RUN"'/shadow_acceptance.json
'
```

## 5. 合并并审计30,000条数据

```bash
export DATA_RUN=rl2_dataset_20260824_r1

docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py build-dataset \
  --simulation-root /workspace/out/rl2_data_20260824_r1/simulation \
  --official-root /workspace/out/rl2_official_shadow_20260824_r1 \
  --minimum-total 30000 \
  --minimum-simulation 24000 \
  --minimum-official 6000 \
  --output /workspace/out/'"$DATA_RUN"'/scheduler_replay_rl2.jsonl \
  --manifest /workspace/out/'"$DATA_RUN"'/dataset_manifest.json \
  --coverage-json /workspace/out/'"$DATA_RUN"'/coverage_report.json \
  --coverage-csv /workspace/out/'"$DATA_RUN"'/coverage_report.csv \
  --failures /workspace/out/'"$DATA_RUN"'/coverage_failures.txt
'

cat "$PROJECT/remote_artifacts/$DATA_RUN/coverage_report.json"
cat "$PROJECT/remote_artifacts/$DATA_RUN/coverage_failures.txt"
sha256sum "$PROJECT/remote_artifacts/$DATA_RUN/scheduler_replay_rl2.jsonl"
```

覆盖报告不是 `passed=true` 时禁止训练。

## 6. 并行训练6个候选

```bash
export TRAIN_RUN=rl2_train_20260824_r1

docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py train-matrix \
  --dataset /workspace/out/rl2_dataset_20260824_r1/scheduler_replay_rl2.jsonl \
  --dataset-manifest /workspace/out/rl2_dataset_20260824_r1/dataset_manifest.json \
  --output-root /workspace/out/'"$TRAIN_RUN"' \
  --workers 6 \
  --timesteps 150000 \
  --gamma 0.0 \
  --seeds 20260824 20260825 20260826 \
  --reward-configs baseline success_time \
  --code-revision rl2-success-first
'
```

另一个终端：

```bash
watch -n 2 nvidia-smi
docker exec "$TRAIN_CTN" bash -lc \
  'pgrep -af "run_rl1_pipeline|train_scheduler_policy|MaskablePPO" || true'
```

训练验收：

```bash
docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py validate-training \
  --input-root /workspace/out/'"$TRAIN_RUN"' \
  --expected-models 6 \
  --output /workspace/out/'"$TRAIN_RUN"'/training_matrix_acceptance.json
'
```

## 7. 500种子配对盲测与选型

```bash
export BENCH_RUN=rl2_blind_20260824_r1

docker exec -i "$TRAIN_CTN" bash -lc '
python3 /workspace/baseline/scripts/rl2_cli.py benchmark-matrix \
  --models-root /workspace/out/rl2_train_20260824_r1 \
  --baseline-model /workspace/release_assets/rl_guarded/scheduler_policy.zip \
  --seed-start 50000 \
  --episodes 500 \
  --workers 6 \
  --max-inference-p95-ms 25 \
  --output-root /workspace/out/'"$BENCH_RUN"'

python3 /workspace/baseline/scripts/rl2_cli.py select-candidate \
  --benchmark-root /workspace/out/'"$BENCH_RUN"' \
  --success-first \
  --require-no-success-regression \
  --minimum-success-improvement 0.02 \
  --minimum-elapsed-improvement 0.05 \
  --maximum-return-regression 0.02 \
  --maximum-path-regression 0.02 \
  --maximum-recovery-regression 0.02 \
  --output /workspace/out/'"$BENCH_RUN"'/rl2_selection.json
'

cat "$PROJECT/remote_artifacts/$BENCH_RUN/rl2_selection.json"
```

没有候选通过时必须保持：

```text
selected_model=null
promotion_allowed=false
effective_policy=heuristic
```

## 8. 新模型Shadow、Guarded和回退

后续Shadow与Guarded仍使用第4节相同的 `collect-official` 命令，并额外传入：

```bash
--model "$SELECTED_MODEL" \
--model-sha256 "$MODEL_SHA"
```

Guarded还必须传入：

```bash
--approval "$APPROVAL" \
--approval-sha256 "$APPROVAL_SHA"
```

任何时候关闭RL并恢复V2 Heuristic：

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-rl2
export RUN=heuristic_rollback_$(date +%Y%m%d_%H%M%S)

bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" stop
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" preflight heuristic
bash "$PROJECT/material_sorting_task/scripts/competitionctl.sh" rollback "$RUN"
```

`heuristic`/`rollback` 不加载RL模型，也不依赖approval文件。
