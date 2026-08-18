# Scheduler EventLog replay gate

This gate closes Phase RL-0 before any MaskablePPO training. It audits the
deterministic scheduler trace and exports only observations that were encoded
by the production allow-list at decision time.

## Safety boundary

- The dataset contains the fixed observation vector, action mask and selected
  finite macro-action slot. It does not contain motor commands.
- `ObservationBuilder` ignores unknown keys. Server-private layout truth,
  referee score internals and semantic-audit fields cannot enter the export.
- A selected action must exist in the same candidate snapshot and be enabled
  by its hard action mask.
- Every production event carries `scheduler-event-v2`, a session-scoped unique
  `event_id`, and the active `task_run_id/attempt_run_id/step_run_id`. Candidate
  evaluation and action selection additionally share one `decision_id`.
- Replay pairs v2 decisions by `decision_id`, not file adjacency. Interleaved
  asynchronous decisions are valid; cross-session or cross-step pairs are
  rejected.
- The observation schema version, SHA256, shape and finiteness are checked for
  every record.
- Logs produced before the observation payload or complete v2 correlation
  chain was introduced remain useful for trace auditing, but are counted as
  `legacy_decisions` and never exported as training records.

## Audit existing logs

```bash
python3 material_sorting_task/scripts/replay_scheduler_events.py \
  run_a/scheduler.jsonl run_b/scheduler.jsonl \
  --min-decisions 1000 \
  --output scheduler_replay_audit.json
```

## Export a training-ready heuristic baseline

```bash
python3 material_sorting_task/scripts/replay_scheduler_events.py \
  run_a/scheduler.jsonl run_b/scheduler.jsonl \
  --min-decisions 1000 \
  --require-training-ready \
  --dataset scheduler_heuristic_baseline.jsonl \
  --output scheduler_replay_training_gate.json
```

The command exits non-zero for malformed JSON, unpaired evaluation/selection
events, a selected masked action, schema mismatch, non-finite observation or an
insufficient number of decisions. Dataset output is a deterministic JSONL
projection of validated fields only.

The exported `scheduler-replay-v2` record preserves source SHA256 plus
`session_id`, task/attempt/step run IDs, `decision_id`, and both source event
IDs. This makes every training row traceable to exactly one production
decision without exposing Server-private state.

## Current evidence (2026-08-18)

Four previously captured official-Server V2 heuristic logs were replayed:

- sessions: 4
- paired decisions: 3703
- invalid or unpaired selections: 0
- heuristic selections: 2297
- hysteresis selections: 667
- no-safe-candidate decisions: 739
- mean heuristic regret caused by bounded hysteresis: 0.01967
- maximum heuristic regret: 0.36777

The historical logs predate exact observation recording, so all 2964 selected
decisions are deliberately classified as legacy and `training_ready_decisions`
is zero. This proves the trace pairing/audit path, but it is not permission to
train. New local/remote runs must pass `--require-training-ready` before Phase
RL-1 begins.

## Replay pretraining environment

`learning.replay_env.ReplayBanditEnv` revalidates every exported record and
exposes its production observation and hard action mask through the interface
required by MaskablePPO. It is a contextual-bandit ranking environment:

- choosing the best valid candidate utility receives reward 1.0;
- other valid candidates receive `1.0 - utility_regret`;
- a masked or empty slot receives -100 and is reported invalid;
- no motor command, referee mutation or Server-private state exists in the
  environment.

```bash
export MATERIAL_SCHEDULER_REPLAY_DATASET=/data/scheduler_heuristic_baseline.jsonl
export MATERIAL_SCHEDULER_REPLAY_CONFIG=/workspace/baseline/examples/material_sorting/learning/configs/replay_training_v1.json
python3 material_sorting_task/scripts/train_scheduler_policy.py \
  --env-factory learning.replay_env:build_replay_env \
  --output /models/scheduler_maskable_ppo.zip \
  --timesteps 100000 \
  --seed 20260818 \
  --code-revision <git-commit> \
  --provenance "$MATERIAL_SCHEDULER_REPLAY_DATASET" \
  --provenance "$MATERIAL_SCHEDULER_REPLAY_CONFIG"
```

This stage pretrains candidate ranking only. Runtime hysteresis, recovery
outcomes, task success and physical safety must still be evaluated by the
paired simulation benchmark, RL Shadow and final official-Server gates.
