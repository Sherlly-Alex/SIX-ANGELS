# Guarded RL policy promotion

`rl_guarded` is fail-closed. A model file and its SHA256 are necessary but no
longer sufficient to give the learned scheduler control authority. Promotion
also requires one immutable approval manifest binding three passed gates:

1. the model package and observation schema integrity chain;
2. at least 100 paired blind-seed benchmark episodes using the 25 ms inference,
   2% improvement and 2,000-sample bootstrap gates;
3. official-Client `rl_shadow` evidence with at least 1,000 accepted
   suggestions, zero RL takeovers, zero mask violations, p95 inference at most
   25 ms and fallback rate at most 1%.

Generate the manifest only after those reports pass:

```bash
python3 material_sorting_task/scripts/approve_guarded_policy.py \
  --model scheduler_maskable_ppo.zip \
  --benchmark scheduler_blind_benchmark.json \
  --shadow rl_shadow_acceptance.json \
  --output scheduler_guarded_approval.json
```

The command prints `approval_sha256=...`. Record that digest independently and
start a guarded Client with all four explicit values:

```bash
export MATERIAL_SCHEDULER_POLICY=rl_guarded
export MATERIAL_SCHEDULER_MODEL=/workspace/artifacts/scheduler_maskable_ppo.zip
export MATERIAL_SCHEDULER_MODEL_SHA256=<approved-model-sha256>
export MATERIAL_RL_GUARDED_APPROVAL=/workspace/artifacts/scheduler_guarded_approval.json
export MATERIAL_RL_GUARDED_APPROVAL_SHA256=<printed-approval-sha256>
```

At startup the Client hashes the manifest and deployed model, checks the
manifest schema, model identity and current observation schema, then constructs
the RL policy. Any missing value, altered byte, schema drift or model mismatch
forces the effective policy back to Heuristic before `scheduler_started` is
written. `rl_shadow` intentionally does not require the promotion manifest,
because it never controls the selected action.

This manifest is an operator-controlled release artifact, not a substitute for
the final official-Server guarded canary. Keep Heuristic as the competition
default until that separate canary is explicitly approved.
