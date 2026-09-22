#!/usr/bin/env bash
# Shared environment for the portable Yangluo Thor deployment scripts.
# This file is sourced; do not enable `set -e` here.

yangluo_tools_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export YANGLUO_REPO_ROOT="$(cd "$yangluo_tools_dir/../.." && pwd)"
export YANGLUO_BUNDLE_ROOT="$(cd "$YANGLUO_REPO_ROOT/.." && pwd)"

export YANGLUO_MODEL_DIR="$YANGLUO_REPO_ROOT/model/yangluo_resnet50_fp16"
export YANGLUO_CORE_DIR="$YANGLUO_REPO_ROOT/yangluo_deploy/generated/build_arm64_core"
export YANGLUO_ROS_WS="$YANGLUO_REPO_ROOT/yangluo_deploy/ros_ws"
export YANGLUO_TRT_BUILDER="$YANGLUO_REPO_ROOT/build_yangluo_trt_builder_static/yangluo_trt_builder"
export YANGLUO_SPCONV_DIR="$YANGLUO_BUNDLE_ROOT/libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda13.0"

export YANGLUO_ROS_PREFIX="${YANGLUO_ROS_PREFIX:-/ota/miniconda3/envs/ros}"
export YANGLUO_TRT_ROOT="${YANGLUO_TRT_ROOT:-/opt/vehicle101}"
export YANGLUO_CUDA_ROOT="${YANGLUO_CUDA_ROOT:-/usr/local/cuda}"

case ":${LD_LIBRARY_PATH:-}:" in
  *:/tmp/yangluo_cuda_driver_stub:*)
    echo "ERROR: decode-only CUDA driver stub is present in LD_LIBRARY_PATH." >&2
    echo "Start a clean Thor shell before running production inference." >&2
    return 41 2>/dev/null || exit 41
    ;;
esac

export PATH="$YANGLUO_ROS_PREFIX/bin:$YANGLUO_CUDA_ROOT/bin:${PATH:-}"
export LD_LIBRARY_PATH="$YANGLUO_ROS_WS/devel/lib:$YANGLUO_CORE_DIR:$YANGLUO_SPCONV_DIR:$YANGLUO_TRT_ROOT/lib/aarch64-linux-gnu:$YANGLUO_TRT_ROOT/lib/aarch64-linux-gnu/nvidia:$YANGLUO_CUDA_ROOT/targets/sbsa-linux/lib:$YANGLUO_ROS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$YANGLUO_ROS_WS/devel:$YANGLUO_ROS_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ROS_PACKAGE_PATH="$YANGLUO_ROS_WS/src${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"

python_site="$YANGLUO_ROS_WS/devel/lib/python3/dist-packages"
if [ -d "$python_site" ]; then
  export PYTHONPATH="$python_site${PYTHONPATH:+:$PYTHONPATH}"
fi

