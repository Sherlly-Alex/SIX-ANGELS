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
