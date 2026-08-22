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
  competitionctl.sh preflight [guarded|heuristic]
  competitionctl.sh server RUN [SEED]
  competitionctl.sh client RUN [guarded|heuristic|legacy|v2]
  competitionctl.sh rollback RUN
  competitionctl.sh status
  competitionctl.sh stop
  competitionctl.sh freeze [OUTPUT_DIR]

The default client mode is guarded. "v2" is a compatibility alias for the
policy frozen in competition_release.env. "rollback" always starts V2 with
the independently validated Heuristic policy and never requires RL assets.
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

project_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$PROJECT" "$value"
  fi
}

guarded_model_path() {
  project_path "$MATERIAL_RL_MODEL_RELATIVE_PATH"
}

guarded_approval_path() {
  project_path "$MATERIAL_RL_APPROVAL_RELATIVE_PATH"
}

verify_file_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "$path" ]] || { echo "$label not found: $path" >&2; exit 2; }
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256 mismatch: expected=$expected actual=$actual" >&2
    exit 2
  fi
}

verify_guarded_assets() {
  local model approval
  model="$(guarded_model_path)"
  approval="$(guarded_approval_path)"
  verify_file_hash "$model" "$MATERIAL_RL_MODEL_SHA256" "guarded model"
  [[ -f "$model.metadata.json" ]] || {
    echo "guarded model metadata not found: $model.metadata.json" >&2
    exit 2
  }
  verify_file_hash "$approval" "$MATERIAL_RL_APPROVAL_SHA256" "guarded approval"
}

preflight() {
  local mode="${1:-guarded}"
  local model approval
  command -v docker >/dev/null
  command -v python3 >/dev/null
  command -v sha256sum >/dev/null
  [[ -f "$PROJECT/material_sorting_task/scripts/run_client.sh" ]]
  [[ -f "$PROJECT/material_sorting_task/examples/material_sorting/perception/checkpoints/best.pt" ]]
  docker image inspect "$MATERIAL_SERVER_IMAGE" >/dev/null
  case "$mode" in
    guarded)
      [[ "$MATERIAL_SCHEDULER_POLICY" == "rl_guarded" ]] || {
        echo "frozen release policy is not rl_guarded" >&2
        exit 2
      }
      docker image inspect "$MATERIAL_GUARDED_CLIENT_IMAGE" >/dev/null
      verify_guarded_assets
      model="$(guarded_model_path)"
      approval="$(guarded_approval_path)"
      docker run --rm --network none --entrypoint python3 \
        -v "$PROJECT/material_sorting_task:/workspace/baseline:ro" \
        -v "$model:/workspace/rl_release/scheduler_policy.zip:ro" \
        -v "$model.metadata.json:/workspace/rl_release/scheduler_policy.zip.metadata.json:ro" \
        -v "$approval:/workspace/rl_release/scheduler_guarded_approval.json:ro" \
        "$MATERIAL_GUARDED_CLIENT_IMAGE" \
        /workspace/baseline/scripts/validate_guarded_release.py \
        --model /workspace/rl_release/scheduler_policy.zip \
        --model-sha256 "$MATERIAL_RL_MODEL_SHA256" \
        --approval /workspace/rl_release/scheduler_guarded_approval.json \
        --approval-sha256 "$MATERIAL_RL_APPROVAL_SHA256"
      ;;
    heuristic)
      docker image inspect "$MATERIAL_HEURISTIC_CLIENT_IMAGE" >/dev/null
      ;;
    *) echo "preflight mode must be guarded or heuristic" >&2; exit 2 ;;
  esac
  python3 "$PROJECT/material_sorting_task/scripts/check_workspace.py"
  echo "release=$MATERIAL_RELEASE_ID acceptance_base=$MATERIAL_ACCEPTANCE_BASE_COMMIT"
  echo "project=$PROJECT mode=$mode"
  echo "runtime=$MATERIAL_EXECUTION_MODE scheduler=$MATERIAL_SCHEDULER_ENGINE/$MATERIAL_SCHEDULER_POLICY"
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
  local mode="${2:-guarded}"
  local engine policy image out model approval
  local -a run_args exec_env
  require_run "$run"
  case "$mode" in
    guarded)
      engine=v2
      policy=rl_guarded
      image="$MATERIAL_GUARDED_CLIENT_IMAGE"
      verify_guarded_assets
      model="$(guarded_model_path)"
      approval="$(guarded_approval_path)"
      ;;
    heuristic)
      engine=v2
      policy=heuristic
      image="$MATERIAL_HEURISTIC_CLIENT_IMAGE"
      ;;
    legacy)
      engine=legacy
      policy=heuristic
      image="$MATERIAL_HEURISTIC_CLIENT_IMAGE"
      ;;
    v2)
      if [[ "$MATERIAL_SCHEDULER_POLICY" == "rl_guarded" ]]; then
        run_client "$run" guarded
      else
        run_client "$run" heuristic
      fi
      return
      ;;
    *) echo "client mode must be guarded, heuristic, legacy, or v2" >&2; exit 2 ;;
  esac

  out="$(artifact_dir "$run")"
  mkdir -p "$out"
  docker rm -f material_sorting_client >/dev/null 2>&1 || true
  run_args=(
    run --rm -dit --gpus all --network host --ipc host
    --name material_sorting_client
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID"
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"
    -e PYTHONDONTWRITEBYTECODE=1
    -e OMP_NUM_THREADS=1
    -e MKL_NUM_THREADS=1
    -e MKL_DYNAMIC=FALSE
    -e OPENBLAS_NUM_THREADS=1
    -e NUMEXPR_NUM_THREADS=1
    -e OMP_WAIT_POLICY=PASSIVE
    -v "$PROJECT/material_sorting_task:/workspace/baseline:ro"
    -v "$out:/workspace/artifacts:rw"
  )
  if [[ "$mode" == "guarded" ]]; then
    run_args+=(
      -v "$model:/workspace/rl_release/scheduler_policy.zip:ro"
      -v "$model.metadata.json:/workspace/rl_release/scheduler_policy.zip.metadata.json:ro"
      -v "$approval:/workspace/rl_release/scheduler_guarded_approval.json:ro"
    )
  fi
  docker "${run_args[@]}" "$image" bash -lc 'tail -f /dev/null' >/dev/null

  exec_env=(
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID"
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"
    -e PYTHONDONTWRITEBYTECODE=1
    -e OMP_NUM_THREADS=1
    -e MKL_NUM_THREADS=1
    -e MKL_DYNAMIC=FALSE
    -e OPENBLAS_NUM_THREADS=1
    -e NUMEXPR_NUM_THREADS=1
    -e OMP_WAIT_POLICY=PASSIVE
    -e MATERIAL_EXECUTION_MODE="$MATERIAL_EXECUTION_MODE"
    -e MATERIAL_SCHEDULER_ENGINE="$engine"
    -e MATERIAL_SCHEDULER_POLICY="$policy"
    -e MATERIAL_SCHEDULER_EVENT_LOG="/workspace/artifacts/scheduler_$run.jsonl"
    -e MATERIAL_MEASURED_CARRY_GUARD="$MATERIAL_MEASURED_CARRY_GUARD"
    -e MATERIAL_LOOP_HEALTH_PERIOD_S="$MATERIAL_LOOP_HEALTH_PERIOD_S"
    -e MATERIAL_DETECT_BACKEND="$MATERIAL_DETECT_BACKEND"
    -e MATERIAL_DETECTION_LOG_PERIOD="$MATERIAL_DETECTION_LOG_PERIOD"
  )
  if [[ "$mode" == "guarded" ]]; then
    exec_env+=(
      -e MATERIAL_SCHEDULER_MODEL=/workspace/rl_release/scheduler_policy.zip
      -e MATERIAL_SCHEDULER_MODEL_SHA256="$MATERIAL_RL_MODEL_SHA256"
      -e MATERIAL_RL_GUARDED_APPROVAL=/workspace/rl_release/scheduler_guarded_approval.json
      -e MATERIAL_RL_GUARDED_APPROVAL_SHA256="$MATERIAL_RL_APPROVAL_SHA256"
      -e MATERIAL_RL_DEVICE="$MATERIAL_RL_DEVICE"
      -e MATERIAL_RL_TIMEOUT_MS="$MATERIAL_RL_TIMEOUT_MS"
    )
  fi

  echo "client mode=$mode engine=$engine policy=$policy image=$image; log=$out/client_$run.log"
  docker exec -i "${exec_env[@]}" material_sorting_client bash -lc '
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
  preflight) preflight "${2:-guarded}" ;;
  server) require_run "${2:-}"; run_server "$2" "${3:-$MATERIAL_OFFICIAL_SEED}" ;;
  client) require_run "${2:-}"; run_client "$2" "${3:-guarded}" ;;
  rollback) require_run "${2:-}"; run_client "$2" heuristic ;;
  status) status ;;
  stop) docker rm -f material_sorting_client material_sorting_server >/dev/null 2>&1 || true ;;
  freeze) exec "$SCRIPT_DIR/freeze_competition_release.sh" "${2:-$PROJECT/release_artifacts}" ;;
  *) usage; exit 2 ;;
esac
