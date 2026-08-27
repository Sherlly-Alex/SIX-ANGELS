# RL-2.1 Contextual-Outcome Training

RL-2.1 fixes an evaluation ceiling found in the first RL-2 paired benchmark.
In that benchmark every learned policy failed on the same 192 of 500 seeds and
no policy rescued a Heuristic failure.  The legacy simulator exposed the same
success probability that it used as outcome truth, so the deterministic
Heuristic already had oracle-like access.

## Safety boundary

- CompetitionController, robot executors, Heuristic policy and formal runtime
  are unchanged.
- The new outcome model exists only in the offline project simulator.
- Counterfactual outcome labels are stored beside replay records for training,
  but are never appended to the observation vector.
- The observation schema hash therefore remains compatible with the existing
  guarded runtime.
- RL remains disabled by default.  Failure to pass any gate leaves the
  effective policy as Heuristic.

## Identifiable simulator

`project_simulation_v3.json` selects `contextual_latent` outcomes.  Public
candidate success probability remains a calibrated estimate.  Private outcome
probability additionally depends on public task, stage, payload and pose
interactions that are not represented by the linear Heuristic utility.  A
stable SHA-256 draw is assigned to every seed, stage and action, preserving
paired potential outcomes across policies.

The simulator emits:

- `candidate_outcome_probabilities` as training-only counterfactual labels;
- `candidate_potential_successes` for audit;
- `oracle_action_index` for offline attribution;
- `avoidable_failure`, `failure_reason` and `oracle_miss` in transitions.

## Required gates

1. `audit-identifiability` must show a positive Oracle success gap and positive
   net rescue count without putting private labels in the observation.
2. Simulation validation must report every contextual decision as outcome
   labelled.
3. Training uses `contextual_success`; the failed `success_time` reward is not
   reused.
4. Paired blind evaluation uses an unseen seed range and requires no success,
   safety, path or recovery regression.
5. A selected model may enter Shadow only.  Guarded and official deployment
   still require the existing full-score gates.

## Seed policy

- `60000..60499`: frozen RL-2 rejection benchmark; never train on it.
- `70000..73999`: RL-2.1 simulation training range.
- `80000..80499`: simulator identifiability audit only.
- `90000..90499`: RL-2.1 paired blind benchmark; never train on it.

