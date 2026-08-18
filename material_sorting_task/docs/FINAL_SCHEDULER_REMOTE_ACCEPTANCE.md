# Final scheduler remote acceptance

This is the final remote phase after all local commits are complete. Do not mix
fault-injection logs with scoring runs, do not append two Client sessions to the
same JSONL file, and start a fresh Server and Client container for every run.
The official remote host uses ROS domain 102 for this project.

## 1. Package and upload the committed workspace

Local PowerShell, from `D:\discover-last\0813`:

```powershell
git status --short
git archive --format=tar.gz -o D:\discover-last\SIX-ANGELS-scheduler-final.tar.gz HEAD
Get-FileHash D:\discover-last\SIX-ANGELS-scheduler-final.tar.gz -Algorithm SHA256
scp -P 8001 D:\discover-last\SIX-ANGELS-scheduler-final.tar.gz abc123@8.130.157.142:/home/abc123/
ssh -p 8001 abc123@8.130.157.142
```

Remote preflight:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
docker rm -f material_sorting_client material_sorting_server 2>/dev/null || true
mkdir -p "$PROJECT"
tar -xzf /home/abc123/SIX-ANGELS-scheduler-final.tar.gz -C "$PROJECT"
cd "$PROJECT"
python3 material_sorting_task/scripts/check_workspace.py
git_hash_file="$PROJECT/material_sorting_task/examples/material_sorting/learning/promotion.py"
test -f "$git_hash_file" && echo 'latest scheduler workspace OK'
```

## 2. Run A: latest-code Heuristic baseline, carry guard off

Use three remote terminals. The artifact names below are part of the matrix
naming contract.

Terminal 1 — official Server:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_multiseed_20260817
export OUT="$PROJECT/remote_artifacts/$RUN"
mkdir -p "$OUT"
cd "$PROJECT"

docker run --rm -it --gpus all --network host --ipc host \
  --name material_sorting_server \
  -e DISPLAY="$DISPLAY" \
  -e ROS_DOMAIN_ID=102 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e MATERIAL_ENABLE_RENDER=1 \
  -e MATERIAL_USE_GS=1 \
  -e MATERIAL_RANDOMIZE=1 \
  -e MATERIAL_SEED=20260817 \
  -e MATERIAL_ENABLE_SCORE=1 \
  -e MATERIAL_DEBUG_GRASP=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v material_sorting_cache:/opt/torch_ext \
  material_sorting:offline-server 2>&1 | tee "$OUT/server_$RUN.log"
```

Terminal 2 — Client container:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_multiseed_20260817
export OUT="$PROJECT/remote_artifacts/$RUN"

docker rm -f material_sorting_client 2>/dev/null || true
docker run --rm -dit --gpus all --network host --ipc host \
  --name material_sorting_client \
  -e ROS_DOMAIN_ID=102 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PROJECT/material_sorting_task":/workspace/baseline:ro \
  -v "$OUT":/workspace/artifacts:rw \
  material_sorting:offline-client bash -lc 'tail -f /dev/null'

docker exec -i material_sorting_client bash -lc '
cd /workspace/baseline
export ROS_DOMAIN_ID=102
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONDONTWRITEBYTECODE=1
export MATERIAL_EXECUTION_MODE=task123_full
export MATERIAL_SCHEDULER_ENGINE=v2
export MATERIAL_SCHEDULER_POLICY=heuristic
export MATERIAL_SCHEDULER_EVENT_LOG=/workspace/artifacts/scheduler_v2_multiseed_20260817.jsonl
export MATERIAL_MEASURED_CARRY_GUARD=0
export MATERIAL_LOOP_HEALTH_PERIOD_S=5.0
export MATERIAL_DETECT_BACKEND=yolo
export MATERIAL_DETECTION_LOG_PERIOD=0
unset MATERIAL_INPUT_FAULT_DIR
exec bash scripts/run_client.sh
' 2>&1 | tee "$OUT/client_$RUN.log"
```

Terminal 3 — monitor:

```bash
export OUT=/home/abc123/polaris/workspace/SIX-ANGELS-v5/remote_artifacts/v2_multiseed_20260817
tail -F "$OUT/client_v2_multiseed_20260817.log" | grep --line-buffered -Ei \
'controller=|candidate action=|candidate_application|runtime_health=|measured_carried_guard|blocked|safe_hold|timeout|unsafe|collision|failed'
```

After `controller=finished task=3 ... score=160`, validate in Terminal 3:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_multiseed_20260817
export OUT="$PROJECT/remote_artifacts/$RUN"

python3 "$PROJECT/material_sorting_task/scripts/validate_remote_run.py" \
  --client "$OUT/client_$RUN.log" \
  --server "$OUT/server_$RUN.log" \
  --events "$OUT/scheduler_$RUN.jsonl" \
  --max-interval-p99-ms 125 \
  --output "$OUT/acceptance.json"
cat "$OUT/acceptance.json"
```

Required result: `passed: true`, score 160, all fatal counts zero and
`runtime_health.passed: true`.

## 3. Run B: same-seed measured-carry A/B

Remove both containers, restart the same Server command with seed `20260817`,
and use a new artifact directory `v2_measured_carry_20260817`. In the Client
command change only these values:

```bash
export MATERIAL_SCHEDULER_EVENT_LOG=/workspace/artifacts/scheduler_v2_measured_carry_20260817.jsonl
export MATERIAL_MEASURED_CARRY_GUARD=1
```

The host-side Client log must be
`client_v2_measured_carry_20260817.log` and the Server log must be
`server_v2_measured_carry_20260817.log`. Validate with:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export RUN=v2_measured_carry_20260817
export OUT="$PROJECT/remote_artifacts/$RUN"

python3 "$PROJECT/material_sorting_task/scripts/validate_remote_run.py" \
  --client "$OUT/client_$RUN.log" \
  --server "$OUT/server_$RUN.log" \
  --events "$OUT/scheduler_$RUN.jsonl" \
  --require-measured-carry \
  --max-interval-p99-ms 125 \
  --output "$OUT/acceptance.json"
cat "$OUT/acceptance.json"
```

Do not make the carry guard default-on from this single run. It first proves
the same-seed feature gate; broader randomized evidence remains separate.

## 4. Five-seed Heuristic release matrix

Repeat Run A with fresh containers for seeds `20260818` through `20260821`.
For each seed `S`, use exactly:

```text
RUN=v2_multiseed_S
OUT=$PROJECT/remote_artifacts/v2_multiseed_S
MATERIAL_SEED=S
Client log:    client_v2_multiseed_S.log
Server log:    server_v2_multiseed_S.log
Scheduler log: scheduler_v2_multiseed_S.jsonl
```

Then run:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export ROOT="$PROJECT/remote_artifacts"

python3 "$PROJECT/material_sorting_task/scripts/validate_remote_matrix.py" \
  --root "$ROOT" \
  --seeds 20260817 20260818 20260819 20260820 20260821 \
  --require-events \
  --max-interval-p99-ms 125 \
  --output "$ROOT/multiseed_acceptance.json"
cat "$ROOT/multiseed_acceptance.json"
```

Required result: five passed seeds, no failed seeds. `--require-events` must not
be removed for a new release candidate.

## 5. New EventLog replay/data qualification

The five logs above use `scheduler-event-v2` IDs and exact observations. Audit
and export only fully paired records:

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export ROOT="$PROJECT/remote_artifacts"

python3 "$PROJECT/material_sorting_task/scripts/replay_scheduler_events.py" \
  "$ROOT"/v2_multiseed_*/scheduler_v2_multiseed_*.jsonl \
  --min-decisions 1000 \
  --require-training-ready \
  --dataset "$ROOT/scheduler_replay_v2.jsonl" \
  --output "$ROOT/scheduler_replay_acceptance.json"
cat "$ROOT/scheduler_replay_acceptance.json"
```

This completes the Heuristic scheduler release evidence. It does not require or
authorize RL training.

## 6. Conditional RL path — only after a model exists

RL remains optional. Keep `MATERIAL_SCHEDULER_POLICY=heuristic` unless all of
the following later exist and pass: model-package validation, 100-seed paired
simulation benchmark, at least 1,000 official-Client Shadow suggestions, and a
hashed `scheduler-policy-approval-v1` manifest. Follow
`docs/RL_SHADOW_ACCEPTANCE.md` and `docs/GUARDED_POLICY_PROMOTION.md`.

Even a valid manifest permits only a separate official-Server guarded canary;
it does not automatically change the competition default.
