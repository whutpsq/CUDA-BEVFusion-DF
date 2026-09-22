#!/usr/bin/env bash
set -euo pipefail

cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
stub_source="$cuda_root/targets/sbsa-linux/lib/stubs/libcuda.so"
stub_dir="${1:-/tmp/yangluo_cuda_driver_stub}"

if [ ! -f "$stub_source" ]; then
  echo "Missing CUDA driver stub: $stub_source" >&2
  exit 1
fi
mkdir -p "$stub_dir"
ln -sfn "$stub_source" "$stub_dir/libcuda.so.1"

echo "DECODE_ONLY_STUB_OK dir=$stub_dir"
echo "Use this directory first in LD_LIBRARY_PATH only for decode_only in the GPU-less container."
echo "Never put it in the real Thor runtime LD_LIBRARY_PATH."
