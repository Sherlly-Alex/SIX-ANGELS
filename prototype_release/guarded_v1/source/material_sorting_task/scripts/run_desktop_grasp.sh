#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$REPO_ROOT/examples/material_sorting"
TASK_ID="${1:-${MATERIAL_TASK_ID:-1}}"
CHECKPOINT="${MATERIAL_YOLO_CHECKPOINT:-$TASK_DIR/perception/checkpoints/best.pt}"

if [[ "$TASK_ID" != "1" && "$TASK_ID" != "3" ]]; then
  echo "desktop grasp supports task 1 or 3, got: $TASK_ID" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing YOLO weight: $CHECKPOINT" >&2
  exit 2
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export PYTHONPATH="$REPO_ROOT:$TASK_DIR:${PYTHONPATH:-}"

cleanup() {
  for pid in "${SEMANTIC_PID:-}" "${PERCEPTION_PID:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

cd "$TASK_DIR"
python3 perception/box_detect.py \
  --backend yolo \
  --checkpoint "$CHECKPOINT" \
  --conf "${MATERIAL_YOLO_CONF:-0.60}" \
  --center-compensation-scale "${MATERIAL_CENTER_COMPENSATION_SCALE:-0.70}" \
  --no-result-image &
PERCEPTION_PID=$!

python3 desktop_grasp/semantic_target_locator.py --task "$TASK_ID" &
SEMANTIC_PID=$!

python3 desktop_grasp/manual_dual_arm_pregrasp.py \
  --target-topic /material/target_world
