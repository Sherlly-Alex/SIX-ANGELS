# RL Shadow offline acceptance

RL Shadow is an audit mode. The deterministic HeuristicPolicy remains the
control authority; the learned policy may only suggest one already-generated,
hard-masked macro-action slot.

## Model-package gate

Offline training writes `<model>.metadata.json` with:

- `scheduler-model-metadata-v1` metadata schema;
- MaskablePPO algorithm identity;
- exact observation schema version and SHA256;
- model-file SHA256;
- canonical training-configuration SHA256;
- name, size and SHA256 of each declared dataset/config provenance file.

Validate it before copying the model into any Client image:

```bash
python3 material_sorting_task/scripts/validate_scheduler_model.py \
  --model scheduler_maskable_ppo.zip \
  --expected-model-sha256 <approved-model-sha256> \
  --expected-provenance-sha256 <approved-replay-dataset-sha256> \
  --output model_package_acceptance.json
```

The production loader independently checks model bytes against the configured
SHA256 and checks that metadata names the same bytes, approved metadata schema,
MaskablePPO algorithm and observation schema.

The Client loads the approved model and performs two synthetic, non-dispatching
inferences before the scheduler loop starts. This absorbs CUDA/model cold-start
latency outside the 25 ms guarded deadline; warm-up failure falls back to the
Heuristic scheduler before any task action is selected.

## Shadow EventLog gate

Each `action_selected` event records the policy suggestion, guard reason,
measured inference latency and SHA256 of the actually loaded model. The main
selection source remains `heuristic` or `hysteresis`.

```bash
python3 material_sorting_task/scripts/validate_rl_shadow.py \
  shadow_seed_1.jsonl shadow_seed_2.jsonl \
  --min-suggestions 1000 \
  --max-inference-p95-ms 25 \
  --max-fallback-rate 0.01 \
  --expected-model-sha256 <approved-model-sha256> \
  --output rl_shadow_acceptance.json
```

The gate fails when:

- the session does not explicitly declare `policy_mode=rl_shadow`;
- an RL suggestion is absent from the candidate snapshot or masked;
- any selection reports RL as the actual control source;
- an accepted suggestion lacks finite inference time or the actual model hash;
- more than one model hash appears, or it differs from the approved hash;
- inference p95 exceeds 25 ms or fallback rate exceeds its limit;
- the underlying replay/schema/training-ready checks fail.

Passing this offline gate permits evaluation to continue. It does not permit
`rl_guarded` control. Simulation, multi-seed Shadow and final official-Server
validation remain separate release gates.

## Paired blind-seed benchmark

The offline simulation factory must be deterministic under `reset(seed=...)`.
Run the approved model and the observation-space deterministic utility baseline
on separate environment instances with the same blind seeds:

```bash
python3 material_sorting_task/scripts/benchmark_scheduler_policy.py \
  --env-factory project_training_env:build \
  --model scheduler_maskable_ppo.zip \
  --model-sha256 <approved-model-sha256> \
  --seed-start 30000 \
  --episodes 100 \
  --max-inference-p95-ms 25 \
  --minimum-relative-improvement 0.02 \
  --output scheduler_blind_benchmark.json
```

The tool rejects overlap with the training seed stored in model metadata. It
requires complete episodes, zero policy/mask/safety failures, no success-count
regression, inference p95 within budget, and at least one paired metric
(`elapsed_s`, `path_length_m`, `recoveries`) whose 95% bootstrap lower bound is
positive and whose relative improvement reaches the configured threshold.
This is an offline pre-release gate, not a substitute for runtime hysteresis,
Shadow or official-Server acceptance.

## Promotion to guarded control

Passing the model, benchmark and Shadow gates separately still does not enable
control. Bind their exact artifacts with `scripts/approve_guarded_policy.py`
and configure the resulting manifest plus its independently approved SHA256 as
described in `docs/GUARDED_POLICY_PROMOTION.md`. Without that manifest,
`MATERIAL_SCHEDULER_POLICY=rl_guarded` fails closed to Heuristic.
