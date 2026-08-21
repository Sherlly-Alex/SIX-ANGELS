#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT="${PROJECT:-$(cd -- "$TASK_ROOT/.." && pwd)}"
RELEASE_ENV="${MATERIAL_RELEASE_ENV:-$TASK_ROOT/config/competition_release.env}"

if [[ ! -f "$RELEASE_ENV" ]]; then
  echo "release config not found: $RELEASE_ENV" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$RELEASE_ENV"

usage() {
  cat <<'EOF'
Usage:
  competitionctl.sh preflight
  competitionctl.sh server RUN [SEED]
  competitionctl.sh client RUN [v2|legacy]
  competitionctl.sh rollback RUN
  competitionctl.sh status
  competitionctl.sh stop
  competitionctl.sh freeze [OUTPUT_DIR]

PROJECT may override the project directory. The default is the parent of
material_sorting_task. RUN may contain letters, numbers, dots, underscores and dashes.
EOF
}

require_run() {
  local run="${1:-}"
  if [[ -z "$run" || ! "$run" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid or missing RUN: $run" >&2
    exit 2
  fi
}

artifact_dir() {
  printf '%s/remote_artifacts/%s' "$PROJECT" "$1"
}

preflight() {
  command -v docker >/dev/null
  command -v python3 >/dev/null
  [[ -f "$PROJECT/material_sorting_task/scripts/run_client.sh" ]]
  [[ -f "$PROJECT/material_sorting_task/examples/material_sorting/perception/checkpoints/best.pt" ]]
  docker image inspect "$MATERIAL_SERVER_IMAGE" >/dev/null
  docker image inspect "$MATERIAL_CLIENT_IMAGE" >/dev/null
  python3 "$PROJECT/material_sorting_task/scripts/check_workspace.py"
  echo "release=$MATERIAL_RELEASE_ID acceptance_base=$MATERIAL_ACCEPTANCE_BASE_COMMIT"
  echo "project=$PROJECT"
  echo "runtime=$MATERIAL_EXECUTION_MODE scheduler=$MATERIAL_SCHEDULER_ENGINE/$MATERIAL_SCHEDULER_POLICY measured_carry=$MATERIAL_MEASURED_CARRY_GUARD"
  echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
}

run_server() {
  local run="$1"
  local seed="${2:-$MATERIAL_OFFICIAL_SEED}"
  local out
  require_run "$run"
  out="$(artifact_dir "$run")"
  mkdir -p "$out"
  docker rm -f material_sorting_server >/dev/null 2>&1 || true
  cd "$PROJECT"
  docker run --rm -it --gpus all --network host --ipc host \
    --name material_sorting_server \
    -e DISPLAY="${DISPLAY:-}" \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
    -e MUJOCO_GL=glfw \
    -e MATERIAL_ENABLE_RENDER=1 \
    -e MATERIAL_USE_GS=1 \
    -e MATERIAL_RANDOMIZE=1 \
    -e MATERIAL_SEED="$seed" \
    -e MATERIAL_ENABLE_SCORE=1 \
    -e MATERIAL_DEBUG_GRASP=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v material_sorting_cache:/opt/torch_ext \
    "$MATERIAL_SERVER_IMAGE" 2>&1 | tee "$out/server_$run.log"
}

run_client() {
  local run="$1"
  local mode="${2:-v2}"
  local engine policy out
  require_run "$run"
  case "$mode" in
    v2)
      engine="$MATERIAL_SCHEDULER_ENGINE"
      policy="$MATERIAL_SCHEDULER_POLICY"
      ;;
    legacy)
      engine=legacy
      policy=heuristic
      ;;
    *)
      echo "client mode must be v2 or legacy" >&2
      exit 2
      ;;
  esac
  out="$(artifact_dir "$run")"
  mkdir -p "$out"
  docker rm -f material_sorting_client >/dev/null 2>&1 || true
  docker run --rm -dit --gpus all --network host --ipc host \
    --name material_sorting_client \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$PROJECT/material_sorting_task":/workspace/baseline:ro \
    -v "$out":/workspace/artifacts:rw \
    "$MATERIAL_CLIENT_IMAGE" bash -lc 'tail -f /dev/null' >/dev/null

  echo "client mode=$mode engine=$engine policy=$policy; log=$out/client_$run.log"
  docker exec -i \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e MATERIAL_EXECUTION_MODE="$MATERIAL_EXECUTION_MODE" \
    -e MATERIAL_SCHEDULER_ENGINE="$engine" \
    -e MATERIAL_SCHEDULER_POLICY="$policy" \
    -e MATERIAL_SCHEDULER_EVENT_LOG="/workspace/artifacts/scheduler_$run.jsonl" \
    -e MATERIAL_MEASURED_CARRY_GUARD="$MATERIAL_MEASURED_CARRY_GUARD" \
    -e MATERIAL_LOOP_HEALTH_PERIOD_S="$MATERIAL_LOOP_HEALTH_PERIOD_S" \
    -e MATERIAL_DETECT_BACKEND="$MATERIAL_DETECT_BACKEND" \
    -e MATERIAL_DETECTION_LOG_PERIOD="$MATERIAL_DETECTION_LOG_PERIOD" \
    material_sorting_client bash -lc '
      cd /workspace/baseline
      unset MATERIAL_INPUT_FAULT_DIR
      exec bash scripts/run_client.sh
    ' 2>&1 | tee "$out/client_$run.log"
}

status() {
  docker ps -a --filter name=material_sorting_server --filter name=material_sorting_client \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  if [[ -d "$PROJECT/remote_artifacts" ]]; then
    find "$PROJECT/remote_artifacts" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr | head -n 5 | cut -d' ' -f2-
  fi
}

case "${1:-}" in
  preflight) preflight ;;
  server) require_run "${2:-}"; run_server "$2" "${3:-$MATERIAL_OFFICIAL_SEED}" ;;
  client) require_run "${2:-}"; run_client "$2" "${3:-v2}" ;;
  rollback) require_run "${2:-}"; run_client "$2" legacy ;;
  status) status ;;
  stop) docker rm -f material_sorting_client material_sorting_server >/dev/null 2>&1 || true ;;
  freeze) exec "$SCRIPT_DIR/freeze_competition_release.sh" "${2:-$PROJECT/release_artifacts}" ;;
  *) usage; exit 2 ;;
esac
