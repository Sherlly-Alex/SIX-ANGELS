#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$REPO_ROOT/examples/material_sorting"
BACKEND="${MATERIAL_DETECT_BACKEND:-yolo}"
CHECKPOINT="${MATERIAL_YOLO_CHECKPOINT:-$TASK_DIR/perception/checkpoints/best.pt}"
DETECTION_LOG_PERIOD="${MATERIAL_DETECTION_LOG_PERIOD:-5.0}"
ROS_SETUP="${MATERIAL_ROS_SETUP:-/opt/ros/humble/setup.bash}"

# The official image may start a plain shell where ROS 2 has not been sourced.
# Bootstrap it here so this script behaves the same from Docker CMD, SSH and a
# local terminal. Keep an already configured ROS installation untouched.
if ! python3 -c 'import rclpy' >/dev/null 2>&1 && [[ -f "$ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$ROS_SETUP"
  set -u
fi
if ! python3 -c 'import rclpy' >/dev/null 2>&1; then
  echo "ROS 2 Python module rclpy is unavailable; source ROS or set MATERIAL_ROS_SETUP" >&2
  exit 2
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export MATERIAL_USE_LIDAR="${MATERIAL_USE_LIDAR:-0}"
export MATERIAL_EXECUTION_MODE="${MATERIAL_EXECUTION_MODE:-task123_full}"
export PYTHONPATH="$REPO_ROOT:$TASK_DIR:$TASK_DIR/perception:${PYTHONPATH:-}"

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
  'from client_task import CompetitionClient; from control_types import ArmCommand; from desktop_grasp.pregrasp_core import ContactGraspController, OpenPregraspController, SlideLiftController; from executors.base import TargetObservation; from executors.task1 import Task1LiftExecutor; from executors.task1_full import Task1IntegratedExecutor; from executors.task2 import Task2IntegratedExecutor; from shelf.state_tracker import ShelfStateTracker'

python3 perception/box_detect.py \
  --backend "$BACKEND" \
  --checkpoint "$CHECKPOINT" \
  --detection-log-period "$DETECTION_LOG_PERIOD" \
  --no-result-image &
PERCEPTION_PID=$!

python3 client_task.py
