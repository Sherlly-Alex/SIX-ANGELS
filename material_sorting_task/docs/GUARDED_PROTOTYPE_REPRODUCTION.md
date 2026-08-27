# Guarded 原型复现与验收

## 1. 证据位置

完整冻结证据位于仓库根目录：

```text
prototype_release/guarded_v1/
```

主要文件：

```text
evidence/training/
evidence/benchmark/approval_benchmark.json
evidence/shadow/five_seed/rl_shadow_acceptance_5seed.json
evidence/shadow/default1000_canary/
evidence/guarded/seed_20260917/
evidence/guarded/seeds_20260918_20260919/
release_assets/rl_guarded/
SHA256SUMS.txt
```

冻结目录还保留远程源码快照 `source/`，用于与分支工作树交叉审计。

## 2. 完整性检查

```bash
export PROJECT=/path/to/SIX-ANGELS-qzhRL
cd "$PROJECT"

sha256sum release_assets/rl_guarded/scheduler_policy.zip
sha256sum release_assets/rl_guarded/scheduler_guarded_approval.json

cd prototype_release/guarded_v1
sha256sum -c SHA256SUMS.txt
```

预期模型与 Approval 哈希分别为：

```text
5340c47b1fbcfaf799667e1b36a2474e7809817abca78e38875f690a222fb785
0f92ad4a1a0039c9dbefc54d3710aeba38910b0aaf259a443db7dd9af9a95f0a
```

## 3. Approval 验证

```bash
cd "$PROJECT"
python3 material_sorting_task/scripts/validate_guarded_release.py \
  --model release_assets/rl_guarded/scheduler_policy.zip \
  --model-sha256 5340c47b1fbcfaf799667e1b36a2474e7809817abca78e38875f690a222fb785 \
  --approval release_assets/rl_guarded/scheduler_guarded_approval.json \
  --approval-sha256 0f92ad4a1a0039c9dbefc54d3710aeba38910b0aaf259a443db7dd9af9a95f0a
```

没有训练依赖的宿主机可使用已验证 Guarded 镜像运行同一命令，或直接执行：

```bash
bash material_sorting_task/scripts/competitionctl.sh preflight guarded
```

## 4. 500 盲种子报告

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("prototype_release/guarded_v1/evidence/benchmark/approval_benchmark.json")
report = json.loads(path.read_text(encoding="utf-8"))
heuristic = report["heuristic"]
rl = report["rl"]
print("passed =", report["passed"])
print("seed_count =", len(report["seeds"]))
print("heuristic_success =", sum(bool(item["success"]) for item in heuristic))
print("rl_success =", sum(bool(item["success"]) for item in rl))
print("rl_inference_p95_ms =", report["rl_inference_p95_ms"])
for item in report["improvements"]:
    print(item["metric"], item["relative_improvement"], item["improved"])
PY
```

冻结结果：Heuristic `241/500`，RL `303/500`，RL 推理 p95 `3.698 ms`；恢复次数改善，耗时和路径没有改善。

## 5. Shadow 五场验收

冻结报告：

```text
prototype_release/guarded_v1/evidence/shadow/five_seed/rl_shadow_acceptance_5seed.json
```

关键结果：五个会话、2132 次建议、运行时 fallback 0、推理 p95 1.900 ms、实际接管 0。Shadow 的实际接管必须为 0。

## 6. Guarded 实际来源验收

对两个最终 Guarded 种子执行仓库内追踪器：

```bash
python3 material_sorting_task/scripts/validate_guarded_lineage.py \
  prototype_release/guarded_v1/evidence/guarded/seeds_20260918_20260919/v2_multiseed_20260918/scheduler_v2_multiseed_20260918.jsonl \
  prototype_release/guarded_v1/evidence/guarded/seeds_20260918_20260919/v2_multiseed_20260919/scheduler_v2_multiseed_20260919.jsonl \
  --minimum-applied 8 \
  --minimum-rl-origin-applied 8 \
  --require-all-applied-from-rl \
  --output guarded_lineage_acceptance.json
```

预期总计 `16/16` 个应用候选可追溯至 RL；Hysteresis 只是保持已经选择的动作，不扩大 RL 权限。

## 7. 单场任务与实时性验收

```bash
export RUN=v2_multiseed_20260918
export DIR=prototype_release/guarded_v1/evidence/guarded/seeds_20260918_20260919/$RUN

python3 material_sorting_task/scripts/validate_remote_run.py \
  --client "$DIR/client_$RUN.log" \
  --server "$DIR/server_$RUN.log" \
  --events "$DIR/scheduler_$RUN.jsonl" \
  --require-candidate-application \
  --reject-duplicate-candidate-applications \
  --min-applied-candidates 1 \
  --max-interval-p99-ms 125 \
  --output remote_acceptance_reproduced.json
```

种子 20260918 和 20260919 均为 160 分，无重复应用、无 blocked、无 safe hold、无执行器错误和无碰撞。

## 8. 结论边界

本证据链只支持以下结论：Guarded 已能在硬约束候选空间中真实选择动作，并在三场金丝雀中保持满分和安全；它是可迭代样品，不是默认正式控制器。正式运行仍以状态机和 Heuristic 为准。
