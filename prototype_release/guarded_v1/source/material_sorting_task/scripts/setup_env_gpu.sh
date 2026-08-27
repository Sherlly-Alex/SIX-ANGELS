#!/usr/bin/env bash
# 赛事远程 GPU/渲染环境配置。
# 用法：source scripts/setup_env_gpu.sh
#
# 该脚本只设置环境变量，不安装 CUDA、PyTorch 或 gaussian_renderer。
# 依赖应由赛事 Docker/远程镜像提供。

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "请使用 source scripts/setup_env_gpu.sh，而不是直接执行脚本。" >&2
    exit 2
fi

_GPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GPU_REPO="$(cd "${_GPU_SCRIPT_DIR}/.." && pwd)"

# 使用 MATERIAL_GPU_* 覆盖默认值。
export MUJOCO_GL="${MATERIAL_GPU_MUJOCO_GL:-egl}"
export MATERIAL_USE_GS="${MATERIAL_GPU_USE_GS:-1}"
export MATERIAL_HEADLESS="${MATERIAL_GPU_HEADLESS:-1}"
export MATERIAL_ENABLE_RENDER="${MATERIAL_GPU_ENABLE_RENDER:-1}"
export MATERIAL_RANDOMIZE="${MATERIAL_GPU_RANDOMIZE:-1}"
export MATERIAL_ENABLE_SCORE="${MATERIAL_GPU_ENABLE_SCORE:-1}"
# 远程默认打开货架层高视觉修正（几何仍是失败回退）。
export MATERIAL_ENABLE_LAYER_REFINE="${MATERIAL_GPU_ENABLE_LAYER_REFINE:-1}"
export MATERIAL_ENABLE_CANNY_REFINE="${MATERIAL_GPU_ENABLE_CANNY_REFINE:-1}"

# 远程验证通常通过 SSH 无显示器运行；需要窗口时可在 source 前设置 MATERIAL_HEADLESS=0。
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export PYTHONPATH="${_GPU_REPO}:${_GPU_REPO}/examples/material_sorting:${PYTHONPATH:-}"

# 远程 GPU 扩展使用独立缓存。
export TORCH_EXTENSIONS_DIR="${MATERIAL_GPU_TORCH_EXTENSIONS_DIR:-/tmp/material_sorting_torch_ext}"
echo "setup_env_gpu: ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} MUJOCO_GL=${MUJOCO_GL} MATERIAL_USE_GS=${MATERIAL_USE_GS} MATERIAL_ENABLE_RENDER=${MATERIAL_ENABLE_RENDER} MATERIAL_HEADLESS=${MATERIAL_HEADLESS} MATERIAL_RANDOMIZE=${MATERIAL_RANDOMIZE} LAYER_REFINE=${MATERIAL_ENABLE_LAYER_REFINE} CANNY_REFINE=${MATERIAL_ENABLE_CANNY_REFINE}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "setup_env_gpu: NVIDIA GPU detected"
else
    echo "setup_env_gpu: WARNING nvidia-smi not found; GPU rendering may not work" >&2
fi

unset _GPU_SCRIPT_DIR _GPU_REPO
