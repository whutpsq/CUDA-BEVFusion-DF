#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 MODEL_DIR CORE_BUILD_DIR TRT_BUILDER" >&2
  echo "Run this script only on the real DRIVE AGX Thor." >&2
  exit 2
fi

model_dir="$(realpath "$1")"
core_build_dir="$(realpath "$2")"
builder="$(realpath "$3")"
output_dir="$model_dir/build"
plugin="$core_build_dir/libcustom_layernorm.so"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "Refusing to build target plans on non-aarch64 host: $(uname -m)" >&2
  exit 1
fi
if [ ! -e /dev/nvidiactl ] && [ ! -e /dev/nvidia0 ]; then
  echo "No NVIDIA GPU device is visible. TensorRT plans must be built on real Thor, not QEMU." >&2
  exit 1
fi
for path in "$builder" "$plugin"; do
  if [ ! -s "$path" ]; then echo "Missing required file: $path" >&2; exit 1; fi
done
mkdir -p "$output_dir"

for name in camera.backbone camera.vtransform fuser; do
  onnx="$model_dir/$name.onnx"
  plan="$output_dir/$name.plan"
  if [ ! -s "$onnx" ]; then echo "Missing ONNX: $onnx" >&2; exit 1; fi
  "$builder" --onnx "$onnx" --engine "$plan" --workspace-mib 2048
done

name=head.bbox
onnx="$model_dir/$name.onnx"
plan="$output_dir/$name.plan"
if [ ! -s "$onnx" ]; then echo "Missing ONNX: $onnx" >&2; exit 1; fi
"$builder" --onnx "$onnx" --engine "$plan" --plugin "$plugin" --workspace-mib 2048

echo "FP16_ENGINE_SET_OK output=$output_dir"
echo "lidar.backbone.xyz.onnx remains an ONNX consumed by libspconv; no TensorRT plan is created for it."

