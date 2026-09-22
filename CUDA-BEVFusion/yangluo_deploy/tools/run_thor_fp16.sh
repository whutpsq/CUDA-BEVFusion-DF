#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=thor_runtime_env.sh
source "$script_dir/thor_runtime_env.sh" || exit $?

check_only=false
if [ "${1:-}" = "--check-only" ]; then
  check_only=true
elif [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--check-only]" >&2
  exit 2
fi

config="$YANGLUO_REPO_ROOT/yangluo_deploy/configs/runtime.yaml"
classes="$YANGLUO_REPO_ROOT/yangluo_deploy/configs/classes.txt"
node="$YANGLUO_ROS_WS/devel/lib/yangluo_bevfusion/yangluo_bevfusion_node"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: production inference requires aarch64, got $(uname -m)." >&2
  exit 51
fi
if [ ! -e /dev/nvidiactl ] && [ ! -e /dev/nvidia0 ]; then
  echo "ERROR: no real NVIDIA GPU device is visible." >&2
  exit 52
fi

required=(
  "$config"
  "$classes"
  "$node"
  "$YANGLUO_CORE_DIR/libbevfusion_core.so"
  "$YANGLUO_CORE_DIR/libcustom_layernorm.so"
  "$YANGLUO_SPCONV_DIR/libspconv.so"
  "$YANGLUO_MODEL_DIR/lidar.backbone.xyz.onnx"
  "$YANGLUO_MODEL_DIR/build/camera.backbone.plan"
  "$YANGLUO_MODEL_DIR/build/camera.vtransform.plan"
  "$YANGLUO_MODEL_DIR/build/fuser.plan"
  "$YANGLUO_MODEL_DIR/build/head.bbox.plan"
)
for path in "${required[@]}"; do
  if [ ! -s "$path" ]; then
    echo "ERROR: missing required runtime file: $path" >&2
    exit 53
  fi
done

unresolved="$(ldd "$node" 2>/dev/null | grep 'not found' || true)"
if [ -n "$unresolved" ]; then
  echo "ERROR: unresolved production-node libraries:" >&2
  echo "$unresolved" >&2
  exit 54
fi

echo "THOR_RUNTIME_PREFLIGHT_OK root=$YANGLUO_BUNDLE_ROOT"
echo "model=$YANGLUO_MODEL_DIR"
echo "output_topic=/perception/bevfusion/objects"

if $check_only; then
  exit 0
fi

if ! rosparam get /rosversion >/dev/null 2>&1; then
  echo "ERROR: ROS master is not reachable at ${ROS_MASTER_URI:-http://localhost:11311}." >&2
  echo "Start the vehicle ROS master, then run this script again." >&2
  exit 55
fi

echo "THOR_ROS_NODE_STARTING"
exec "$node" \
  __name:=yangluo_bevfusion \
  _config:="$config" \
  _classes:="$classes" \
  _decode_only:=false

