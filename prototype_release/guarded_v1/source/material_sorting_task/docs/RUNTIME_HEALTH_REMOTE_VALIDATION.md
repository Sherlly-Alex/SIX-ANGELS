# Runtime-health official Server validation

This run intentionally drops Client-side odometry and joint-state callbacks.
It is a safety test, not a scoring run. Use ROS domain 102 and the official
Server launch form already validated on the remote host.

## Local upload

Run in local PowerShell after `SIX-ANGELS-runtime-latest.tar.gz` is generated:

```powershell
scp -P 8001 D:\discover-last\SIX-ANGELS-runtime-latest.tar.gz abc123@8.130.157.142:/home/abc123/
ssh -p 8001 abc123@8.130.157.142
```

## Remote preflight

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export OUT="$PROJECT/remote_artifacts/v2_runtime_health_r1"

docker rm -f material_sorting_client material_sorting_server 2>/dev/null || true
mkdir -p "$PROJECT" "$OUT"
tar -xzf /home/abc123/SIX-ANGELS-runtime-latest.tar.gz -C "$PROJECT"

cd "$PROJECT"
python3 material_sorting_task/scripts/check_workspace.py
grep -n 'MATERIAL_INPUT_FAULT_DIR' \
  material_sorting_task/examples/material_sorting/client_task.py
grep -n 'InputDropFaultInjector' \
  material_sorting_task/examples/material_sorting/runtime_health.py
```

## Terminal 1: official Server

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export OUT="$PROJECT/remote_artifacts/v2_runtime_health_r1"
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
  material_sorting:offline-server 2>&1 | tee "$OUT/server_runtime_health_r1.log"
```

## Terminal 2: create the Client container

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export OUT="$PROJECT/remote_artifacts/v2_runtime_health_r1"

docker rm -f material_sorting_client 2>/dev/null || true
docker run --rm -dit --gpus all --network host --ipc host \
  --name material_sorting_client \
  -e ROS_DOMAIN_ID=102 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PROJECT/material_sorting_task":/workspace/baseline:ro \
  -v "$OUT":/workspace/artifacts:rw \
  material_sorting:offline-client bash -lc 'tail -f /dev/null'

docker exec material_sorting_client bash -lc '
test -f /workspace/baseline/examples/material_sorting/runtime_health.py
test -f /workspace/baseline/scripts/validate_runtime_health_run.py
mkdir -p /tmp/material_input_faults
rm -f /tmp/material_input_faults/drop_odometry
rm -f /tmp/material_input_faults/drop_joint_states
echo "runtime-health mount OK"
'
```

## Terminal 3: run the V2 Client

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export OUT="$PROJECT/remote_artifacts/v2_runtime_health_r1"

docker exec -i material_sorting_client bash -lc '
cd /workspace/baseline
export ROS_DOMAIN_ID=102
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONDONTWRITEBYTECODE=1
export MATERIAL_EXECUTION_MODE=task123_full
export MATERIAL_SCHEDULER_ENGINE=v2
export MATERIAL_SCHEDULER_POLICY=heuristic
export MATERIAL_SCHEDULER_EVENT_LOG=/workspace/artifacts/scheduler_runtime_health_r1.jsonl
export MATERIAL_MEASURED_CARRY_GUARD=0
export MATERIAL_ODOM_MAX_AGE_S=0.75
export MATERIAL_JOINT_STATE_MAX_AGE_S=0.75
export MATERIAL_INPUT_STALE_GRACE_S=2.0
export MATERIAL_LOOP_HEALTH_PERIOD_S=2.0
export MATERIAL_INPUT_FAULT_DIR=/tmp/material_input_faults
export MATERIAL_DETECT_BACKEND=yolo
export MATERIAL_DETECTION_LOG_PERIOD=0
exec bash scripts/run_client.sh
' 2>&1 | tee "$OUT/client_runtime_health_r1.log"
```

Do not use `dry_run`, `nav_only`, `pregrasp_only`, or `contact_only` here.

## Terminal 4: monitor

```bash
export OUT=/home/abc123/polaris/workspace/SIX-ANGELS-v5/remote_artifacts/v2_runtime_health_r1

tail -F "$OUT/client_runtime_health_r1.log" | grep --line-buffered -Ei \
'controller=|runtime_health=|TEST-ONLY|runtime input|CONTROL_LOOP_HEALTH|INPUT_STALE|safe_hold|blocked|unsafe|collision|failed'
```

Wait until `controller=executing_stage` appears before injecting faults.

## Terminal 2: inject two recoveries and one terminal dropout

```bash
export PROJECT=/home/abc123/polaris/workspace/SIX-ANGELS-v5
export OUT="$PROJECT/remote_artifacts/v2_runtime_health_r1"

until grep -q 'controller=executing_stage' "$OUT/client_runtime_health_r1.log"; do
  sleep 1
done

echo 'SHORT ODOM DROP START'
docker exec material_sorting_client touch /tmp/material_input_faults/drop_odometry
sleep 1.2
docker exec material_sorting_client rm -f /tmp/material_input_faults/drop_odometry
sleep 3

echo 'SHORT JOINT DROP START'
docker exec material_sorting_client touch /tmp/material_input_faults/drop_joint_states
sleep 1.2
docker exec material_sorting_client rm -f /tmp/material_input_faults/drop_joint_states
sleep 3

echo 'TERMINAL JOINT DROP START'
docker exec material_sorting_client touch /tmp/material_input_faults/drop_joint_states
sleep 3.5
docker exec material_sorting_client rm -f /tmp/material_input_faults/drop_joint_states
sleep 2

python3 "$PROJECT/material_sorting_task/scripts/validate_runtime_health_run.py" \
  --events "$OUT/scheduler_runtime_health_r1.jsonl" \
  --expect-recovered odometry joint_states \
  --expect-terminal joint_states \
  --output "$OUT/runtime_health_acceptance.json"
```

The final command must exit zero and print `"passed": true`. The terminal
joint-state dropout is expected to end in `controller=safe_hold`; that is the
success condition for this deliberate fault run.

## Evidence and cleanup

```bash
export OUT=/home/abc123/polaris/workspace/SIX-ANGELS-v5/remote_artifacts/v2_runtime_health_r1

grep -nEi 'runtime input stale|runtime inputs recovered|freshness grace exhausted|CONTROL_LOOP_HEALTH|controller=safe_hold' \
  "$OUT/client_runtime_health_r1.log"
cat "$OUT/runtime_health_acceptance.json"

docker rm -f material_sorting_client material_sorting_server 2>/dev/null || true
```

After this deliberate-fault run passes, start a fresh Server and Client before
any full-score or measured-carry validation.
