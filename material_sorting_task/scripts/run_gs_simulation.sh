#!/usr/bin/env bash
# qzh backup: three-terminal GS simulation runner.
set -eo pipefail
SCRIPTS="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
TASK="$(cd "$SCRIPTS/.." && pwd)"
PROJECT="$PROJECT"; [[ -n "$PROJECT" ]] || PROJECT="$(cd "$TASK/.." && pwd)"
ROOT="$MATERIAL_SIM_ARTIFACT_ROOT"; [[ -n "$ROOT" ]] || ROOT="$PROJECT/simulation_artifacts"
ROLE="$1"; WATCH="$2"; ID="$MATERIAL_SIM_ID"; RUNS="$MATERIAL_SIM_RUNS"; BASE="$MATERIAL_SIM_SEED_BASE"; RANDOM_SEED="$MATERIAL_SIM_RANDOM_SEED"; LIMIT="$MATERIAL_SIM_TIMEOUT_S"; READY="$MATERIAL_SIM_READY_TIMEOUT_S"
[[ -n "$RUNS" ]] || RUNS=1; [[ -n "$BASE" ]] || BASE=20260827; [[ -n "$RANDOM_SEED" ]] || RANDOM_SEED=0; [[ -n "$LIMIT" ]] || LIMIT=720; [[ -n "$READY" ]] || READY=90
fail(){ echo "error: $*" >&2; exit 2; }
positive(){ [[ "$1" != *[!0-9]* && "$1" != "" && "$1" -gt 0 ]] || fail "$2 must be a positive integer"; }
seed_mode(){ [[ "$RANDOM_SEED" == 0 || "$RANDOM_SEED" == 1 ]] || fail "MATERIAL_SIM_RANDOM_SEED must be 0 or 1"; }
need_id(){ [[ -n "$ID" ]] || fail "run init first"; }
dir(){ printf "%s/%s/run_%03d" "$ROOT" "$ID" "$1"; }
wait_file(){ local f="$1" s="$2" t="$(date +%s)"; while [[ ! -e "$f" ]]; do (( $(date +%s)-t < s )) || return 1; sleep 1; done; }
count(){ [[ -f "$1" ]] && { grep -cE "$2" "$1" || true; } || printf "0\n"; }
score(){ local x; [[ -f "$1" ]] || { echo n/a; return; }; x="$(grep -Eo 'score=[0-9]+' "$1" || true)"; [[ -n "$x" ]] && printf "%s\n" "$x" | tail -1 | cut -d= -f2 || echo n/a; }
layer(){ local x; [[ -f "$1" ]] || { echo n/a; return; }; x="$(grep -Eo 'empty shelf layer L[0-9]+' "$1" || true)"; [[ -n "$x" ]] && printf "%s\n" "$x" | tail -1 | sed 's/.* //' || echo n/a; }
help(){ cat <<'TXT'
Usage:
  MATERIAL_SIM_RUNS=3 bash scripts/run_gs_simulation.sh init
  source simulation_artifacts/<session>/session.env
Terminal 1: bash scripts/run_gs_simulation.sh server
Terminal 2: bash scripts/run_gs_simulation.sh client
Terminal 3: bash scripts/run_gs_simulation.sh report --watch
TXT
}
init(){
  positive "$RUNS" MATERIAL_SIM_RUNS; seed_mode; [[ -n "$ID" ]] || ID="qzh_gs_$(date +%Y%m%d_%H%M%S)"
  local p="$ROOT/$ID"; mkdir -p "$p"
  {
    printf "export MATERIAL_SIM_ID=%q\n" "$ID"; printf "export MATERIAL_SIM_RUNS=%q\n" "$RUNS"
    printf "export MATERIAL_SIM_SEED_BASE=%q\n" "$BASE"; printf "export MATERIAL_SIM_RANDOM_SEED=%q\n" "$RANDOM_SEED"; printf "export MATERIAL_SIM_TIMEOUT_S=%q\n" "$LIMIT"
    printf "export MATERIAL_SIM_READY_TIMEOUT_S=%q\n" "$READY"; printf "export MATERIAL_SIM_ARTIFACT_ROOT=%q\n" "$ROOT"
  } > "$p/session.env"
  echo "Session $ID prepared. Source: $p/session.env"
}
server(){
  need_id; positive "$RUNS" MATERIAL_SIM_RUNS; seed_mode
  local i p seed actual pid start end ready
  for ((i=1;i<=RUNS;i++)); do
    p="$(dir "$i")"; if [[ "$RANDOM_SEED" == 1 ]]; then seed=random; else seed=$((BASE+i-1)); fi; mkdir -p "$p"; rm -f "$p/server.ready" "$p/client.done" "$p/server.failed"
    printf "run=%s\nseed_mode=%s\nseed_request=%s\n" "$i" "$RANDOM_SEED" "$seed" > "$p/run.meta"; start="$(date +%s)"
    (
      source /opt/ros/humble/setup.bash
      export PYTHONPATH="$TASK:$PYTHONPATH"; export ROS_DOMAIN_ID="$ROS_DOMAIN_ID"; [[ -n "$ROS_DOMAIN_ID" ]] || export ROS_DOMAIN_ID=99
      export RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"; [[ -n "$RMW_IMPLEMENTATION" ]] || export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      export DISPLAY="$DISPLAY"; [[ -n "$DISPLAY" ]] || export DISPLAY=:0; export MUJOCO_GL=glfw
      export MATERIAL_ASSETS_DIR="$MATERIAL_SIM_ASSETS_DIR"; [[ -n "$MATERIAL_ASSETS_DIR" ]] || export MATERIAL_ASSETS_DIR=/workspace/material_sorting_task/examples/material_sorting/models
      export MATERIAL_RANDOMIZE=1 MATERIAL_USE_GS=1 MATERIAL_ENABLE_RENDER=1 MATERIAL_HEADLESS=0 MATERIAL_DEBUG_GRASP=1
      if [[ "$RANDOM_SEED" == 1 ]]; then unset MATERIAL_SEED; else export MATERIAL_SEED="$seed"; fi
      cd "$TASK/examples/material_sorting/reference/server"; exec python3 material_sorting_server.py
    ) > "$p/server.log" 2>&1 &
    pid=$!; ready=0; echo "[server] run $i seed_mode=$RANDOM_SEED seed=$seed"
    while kill -0 "$pid" 2>/dev/null; do
      if grep -Fq "[server] referee enabled" "$p/server.log"; then actual="$(grep -Eo "randomized layout seed=[0-9]+" "$p/server.log" | tail -1 | cut -d= -f2)"; [[ -n "$actual" ]] || actual="$seed"; printf "%s\n" "$actual" > "$p/actual_seed"; echo "$(date +%s)" > "$p/server.ready"; ready=1; break; fi
      (( $(date +%s)-start < READY )) || break; sleep 1
    done
    if (( ready == 0 )); then echo "server not ready" > "$p/server.failed"; kill -INT "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; return 1; fi
    wait_file "$p/client.done" "$LIMIT" || echo "client timeout" > "$p/server.failed"
    end="$(date +%s)"; kill -INT "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    echo $((end-start)) > "$p/server_duration_s"; touch "$p/server.done"
  done
}
client(){
  need_id; positive "$RUNS" MATERIAL_SIM_RUNS; seed_mode
  local i p start end code
  for ((i=1;i<=RUNS;i++)); do
    p="$(dir "$i")"; mkdir -p "$p"; wait_file "$p/server.ready" "$READY" || fail "Server run $i not ready"
    start="$(date +%s)"; set +e
    (
      source /opt/ros/humble/setup.bash
      export PYTHONPATH="$TASK:$PYTHONPATH"; export ROS_DOMAIN_ID="$ROS_DOMAIN_ID"; [[ -n "$ROS_DOMAIN_ID" ]] || export ROS_DOMAIN_ID=99
      export RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"; [[ -n "$RMW_IMPLEMENTATION" ]] || export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      export MATERIAL_EXECUTION_MODE="$MATERIAL_EXECUTION_MODE"; [[ -n "$MATERIAL_EXECUTION_MODE" ]] || export MATERIAL_EXECUTION_MODE=task123_full
      export MATERIAL_DETECT_BACKEND="$MATERIAL_DETECT_BACKEND"; [[ -n "$MATERIAL_DETECT_BACKEND" ]] || export MATERIAL_DETECT_BACKEND=yolo
      export MATERIAL_DETECTION_LOG_PERIOD=0 MATERIAL_SCHEDULER_ENGINE=v2 MATERIAL_SCHEDULER_POLICY=heuristic MATERIAL_LOCAL_MAP=0 MATERIAL_LOCAL_MAP_APPLY=0
      cd "$TASK"; exec timeout --signal=INT --kill-after=20s "$LIMIT"s bash scripts/run_client.sh
    ) > "$p/client.log" 2>&1
    code=$?; set -e; end="$(date +%s)"
    echo "$code" > "$p/client.exit_code"; echo $((end-start)) > "$p/client_duration_s"; touch "$p/client.done"
    echo "[client] run $i exit=$code"
  done
}
report(){
  need_id; local s="$ROOT/$ID" out="$ROOT/$ID/SIMULATION_REPORT.md" i p log state actual
  mkdir -p "$s"
  {
    echo "# GS Simulation Report"; echo; echo "- Session: $ID"; echo "- Runs: $RUNS"; echo "- Random seed mode: $RANDOM_SEED"; [[ "$RANDOM_SEED" == 0 ]] && echo "- Fixed seed base: $BASE"; echo "- Generated: $(date -Iseconds)"; echo
    echo "| Run | Seed | State | Score | Time(s) | Grasp entered/settled | Shelf layer/ready | Place entered/verify | Exit |"
    echo "|---|---:|---|---:|---:|---|---|---|---:|"
    for ((i=1;i<=RUNS;i++)); do
      p="$(dir "$i")"; log="$p/client.log"; state=running; [[ -f "$p/client.done" ]] && state=client-finished; [[ -f "$p/server.failed" ]] && state=server-failed; actual="$(cat "$p/actual_seed" 2>/dev/null || echo pending)"
      printf "| %03d | %s | %s | %s | %s | %s/%s | %s/%s | %s/%s | %s |\n" "$i" "$actual" "$state" "$(score "$log")" "$(cat "$p/client_duration_s" 2>/dev/null || echo n/a)" "$(count "$log" 'entering grasp')" "$(count "$log" 'bilateral compliant grasp settled')" "$(layer "$log")" "$(count "$log" 'shelf_state=ready')" "$(count "$log" 'entering place')" "$(count "$log" 'entering verify_place')" "$(cat "$p/client.exit_code" 2>/dev/null || echo running)"
    done
    echo; echo "Raw logs: each run directory contains server.log and client.log."
  } > "$out"
  python3 "$SCRIPTS/export_simulation_xlsx.py" "$out" "$s/SIMULATION_REPORT.xlsx"
  echo "[report] $out and $s/SIMULATION_REPORT.xlsx"
}
watch_report(){
  local complete i
  while true; do
    report; complete=0
    for ((i=1;i<=RUNS;i++)); do if [[ -f "$(dir "$i")/client.done" ]]; then ((complete+=1)); fi; done
    (( complete == RUNS )) && break; sleep 5
  done
}
case "$ROLE" in
  init) init ;;
  server) server ;;
  client) client ;;
  report) [[ "$WATCH" == "--watch" ]] && watch_report || report ;;
  -h|--help|help|"") help ;;
  *) fail "unknown role: $ROLE" ;;
esac
