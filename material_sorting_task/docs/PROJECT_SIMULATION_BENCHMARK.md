# Project scheduler simulation benchmark

`learning.simulation_backend` is the ROS-free project backend for Phase RL-1
training experiments and PR-11 paired blind benchmarks. It exercises the real
bounded scheduler contract:

```text
versioned public randomization
  -> CandidateAction center/left/right/replan
  -> PathMetrics + hard safety rejection
  -> deterministic Multi-Critic utility
  -> fixed SchedulingEnv observation/action mask
  -> seeded macro outcome and reward events
```

It models nine navigation decisions across Task 1/2/3 (pick, transport and
return). It does not model robot dynamics, arm contact, referee scoring or
Server-private truth and therefore can never replace Shadow or official-Server
validation.

## Reproducibility and safety boundary

- Configuration is fixed by `learning/configs/project_simulation_v1.json` and
  rejected on unknown keys or schema mismatch.
- Pose/yaw, RGB-D scale, detection noise/dropout, speed/friction, message
  latency, planner failure and dynamic obstacle presence are seeded before an
  episode. Paired environments receive identical samples and success draws.
- Collision, clearance and planner failures become hard action-mask entries;
  the backend rejects any direct masked dispatch as a second boundary.
- Observations contain only the existing public allow-list. There are no motor
  commands, target-body truth, semantic audit results or referee score labels.
- A simulation-only model is research evidence. Model SHA/schema checks do not
  grant runtime approval; fresh production EventLog, RL Shadow, paired gates
  and final official-Server validation remain mandatory.

## Optional offline training

Install Gymnasium, Stable-Baselines3 and sb3-contrib only in an isolated
training environment, then run:

```bash
export MATERIAL_SCHEDULER_SIM_CONFIG=/workspace/baseline/examples/material_sorting/learning/configs/project_simulation_v1.json
python3 material_sorting_task/scripts/train_scheduler_policy.py \
  --env-factory learning.simulation_backend:build_project_sim_env \
  --output /models/scheduler_project_sim.zip \
  --timesteps 100000 \
  --seed 20260818 \
  --code-revision <git-commit> \
  --provenance "$MATERIAL_SCHEDULER_SIM_CONFIG"
```

Run the blind paired gate with non-overlapping seeds:

```bash
python3 material_sorting_task/scripts/benchmark_scheduler_policy.py \
  --env-factory learning.simulation_backend:build_project_sim_env \
  --model /models/scheduler_project_sim.zip \
  --model-sha256 <approved-file-hash> \
  --seed-start 30000 \
  --episodes 100 \
  --output /models/project_sim_blind_report.json
```

A green simulation report permits only the next `rl_shadow` evaluation step.
It does not permit `rl_guarded` or change the default Heuristic policy.
