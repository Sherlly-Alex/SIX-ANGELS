#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$REPO_ROOT/examples/material_sorting"
BACKEND="${MATERIAL_DETECT_BACKEND:-yolo}"
CHECKPOINT="${MATERIAL_YOLO_CHECKPOINT:-$TASK_DIR/perception/checkpoints/best.pt}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export MATERIAL_USE_LIDAR="${MATERIAL_USE_LIDAR:-0}"

if [[ "$BACKEND" == "yolo" ]] && [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing YOLO weight: $CHECKPOINT" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${PERCEPTION_PID:-}" ]]; then
    kill "$PERCEPTION_PID" 2>/dev/null || true
    wait "$PERCEPTION_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$TASK_DIR"

# Fail before starting the high-rate perception process when a remote upload
# mixed incompatible Client files from different revisions.
echo "checking Client module consistency..."
python3 -c \
  'from client_task import CompetitionClient; from executors.base import TargetObservation'

python3 perception/box_detect.py \
  --backend "$BACKEND" \
  --checkpoint "$CHECKPOINT" \
  --no-result-image &
PERCEPTION_PID=$!

python3 client_task.py
