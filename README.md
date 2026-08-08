终端1：cd /home/abc123/polaris/workspace/SIX-ANGELS-v3

export DISPLAY="${DISPLAY:-:0}"
xhost +local:docker

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name material_sorting_server_v3 \
  -e DISPLAY="$DISPLAY" \
  -e ROS_DOMAIN_ID=102 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e MATERIAL_ENABLE_RENDER=1 \
  -e MATERIAL_USE_GS=1 \
  -e MATERIAL_RANDOMIZE=1 \
  -e MATERIAL_SEED=20260808 \
  -e MATERIAL_ENABLE_SCORE=1 \
  -e MATERIAL_DEBUG_GRASP=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v material_sorting_cache:/opt/torch_ext \
  material_sorting:offline-server
  终端2：cd /home/abc123/polaris/workspace/SIX-ANGELS-v3

docker rm -f material_sorting_client_v3 2>/dev/null || true

docker run --rm -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name material_sorting_client_v3 \
  -e ROS_DOMAIN_ID=102 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$PWD/material_sorting_task":/workspace/baseline:ro \
  material_sorting:offline-client
  终端3：cd /home/abc123/polaris/workspace/SIX-ANGELS-v3

rm -f /tmp/client_seed20260808.log

docker exec -it material_sorting_client_v3 bash -lc '
set -e

cd /workspace/baseline

export ROS_DOMAIN_ID=102
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/baseline:${PYTHONPATH:-}
export MATERIAL_YOLO_CHECKPOINT=/workspace/baseline/examples/material_sorting/perception/checkpoints/best.pt

python3 scripts/check_workspace.py
python3 -m unittest discover -s tests -t .
python3 scripts/nav_segment_n10.py --count 10

exec env \
  MATERIAL_EXECUTION_MODE=task123_full \
  MATERIAL_DETECT_BACKEND=yolo \
  MATERIAL_DETECTION_LOG_PERIOD=0 \
  MATERIAL_YOLO_CHECKPOINT=/workspace/baseline/examples/material_sorting/perception/checkpoints/best.pt \
  bash scripts/run_client.sh
' 2>&1 | tee /tmp/client_seed20260808.log
