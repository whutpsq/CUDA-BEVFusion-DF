#!/usr/bin/env bash
set -u

failures=0
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
spconv_root="${YANGLUO_SPCONV_ROOT:-$repo_root/../libraries/3DSparseConvolution/libspconv}"
check_file() {
  if [ -e "$1" ]; then echo "OK   $1"; else echo "MISS $1"; failures=$((failures + 1)); fi
}

echo "arch=$(uname -m)"
echo "os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '\"')"
echo "gpu_devices=$(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ' || true)"
check_file /usr/local/cuda/bin/nvcc
check_file /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so
check_file /opt/vehicle101/include/aarch64-linux-gnu/NvInfer.h
check_file /opt/vehicle101/lib/aarch64-linux-gnu/libnvinfer.so
check_file /opt/vehicle101/lib/aarch64-linux-gnu/libnvonnxparser.so
check_file /ota/miniconda3/envs/ros/bin/roscore
check_file /ota/miniconda3/envs/ros/lib/libopencv_core.so.4.5
check_file /ota/miniconda3/envs/ros/lib/libcv_bridge.so

check_file "$spconv_root/include/spconv/engine.hpp"
check_file "$spconv_root/lib/aarch64_cuda13.0/libspconv.so"
echo "spconv_root=$spconv_root"

if [ "$(uname -m)" != "aarch64" ]; then failures=$((failures + 1)); fi
if [ "$failures" -ne 0 ]; then
  echo "ARM64_ENV_INCOMPLETE failures=$failures"
  exit 1
fi
echo "ARM64_ENV_OK"
