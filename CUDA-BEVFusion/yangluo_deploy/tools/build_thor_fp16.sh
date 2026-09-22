#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=thor_runtime_env.sh
source "$script_dir/thor_runtime_env.sh" || exit $?

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: Thor build requires aarch64, got $(uname -m)." >&2
  exit 42
fi
if [ ! -e /dev/nvidiactl ] && [ ! -e /dev/nvidia0 ]; then
  echo "ERROR: no real NVIDIA GPU device is visible." >&2
  exit 43
fi

required=(
  "$YANGLUO_TRT_BUILDER"
  "$YANGLUO_CORE_DIR/libcustom_layernorm.so"
  "$YANGLUO_MODEL_DIR/camera.backbone.onnx"
  "$YANGLUO_MODEL_DIR/camera.vtransform.onnx"
  "$YANGLUO_MODEL_DIR/lidar.backbone.xyz.onnx"
  "$YANGLUO_MODEL_DIR/fuser.onnx"
  "$YANGLUO_MODEL_DIR/head.bbox.onnx"
  "$YANGLUO_SPCONV_DIR/libspconv.so"
)
for path in "${required[@]}"; do
  if [ ! -s "$path" ]; then
    echo "ERROR: missing required deployment file: $path" >&2
    exit 44
  fi
done

echo "THOR_FP16_BUILD_BEGIN root=$YANGLUO_BUNDLE_ROOT"
bash "$YANGLUO_REPO_ROOT/yangluo_deploy/tensorrt/build_fp16_engines.sh" \
  "$YANGLUO_MODEL_DIR" \
  "$YANGLUO_CORE_DIR" \
  "$YANGLUO_TRT_BUILDER"
status=$?
if [ "$status" -ne 0 ]; then
  echo "THOR_FP16_BUILD_FAILED status=$status" >&2
  exit "$status"
fi

echo "THOR_FP16_BUILD_COMPLETE model=$YANGLUO_MODEL_DIR"

