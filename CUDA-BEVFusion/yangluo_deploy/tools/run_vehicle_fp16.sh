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

config="$YANGLUO_REPO_ROOT/yangluo_deploy/configs/runtime_vehicle_48ms.yaml"
calibration="$YANGLUO_REPO_ROOT/yangluo_deploy/configs/calibration_vehicle_0_10.json"
classes="$YANGLUO_REPO_ROOT/yangluo_deploy/configs/classes.txt"
node="$YANGLUO_ROS_WS/devel/lib/yangluo_bevfusion/yangluo_bevfusion_node"

# Dedicated vehicle defaults. Override them explicitly when the vehicle network
# configuration changes; do not inherit an isolated bag-test ROS master.
export ROS_MASTER_URI="${YANGLUO_VEHICLE_ROS_MASTER_URI:-http://192.168.10.101:11311}"
export ROS_IP="${YANGLUO_VEHICLE_ROS_IP:-192.168.10.101}"
unset ROS_HOSTNAME

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
  "$calibration"
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
    echo "ERROR: missing required vehicle runtime file: $path" >&2
    exit 53
  fi
done

if ! grep -q '/cam5/compressed' "$config" || ! grep -q '/cam10/compressed' "$config"; then
  echo "ERROR: vehicle config does not contain the verified cam5/cam10 topics." >&2
  exit 56
fi
if ! grep -q 'calibration_file: "calibration_vehicle_0_10.json"' "$config"; then
  echo "ERROR: vehicle config does not select the verified calibration 0/10 mapping." >&2
  exit 58
fi
if ! grep -q 'camera_time_offsets_ms: \[48.0, 48.0\]' "$config"; then
  echo "ERROR: vehicle config does not contain the verified 48 ms camera offsets." >&2
  exit 57
fi

unresolved="$(ldd "$node" 2>/dev/null | grep 'not found' || true)"
if [ -n "$unresolved" ]; then
  echo "ERROR: unresolved production-node libraries:" >&2
  echo "$unresolved" >&2
  exit 54
fi

echo "THOR_VEHICLE_RUNTIME_PREFLIGHT_OK root=$YANGLUO_BUNDLE_ROOT"
echo "config=$config"
echo "cameras=/cam5/compressed,/cam10/compressed"
echo "camera_calibration_ids=0,10"
echo "camera_time_offsets_ms=48.0,48.0"
echo "output_topic=/perception/bevfusion/objects"
echo "ros_master=$ROS_MASTER_URI"

if $check_only; then
  exit 0
fi

if ! rosparam get /rosversion >/dev/null 2>&1; then
  echo "ERROR: vehicle ROS master is not reachable at $ROS_MASTER_URI." >&2
  exit 55
fi

echo "THOR_VEHICLE_ROS_NODE_STARTING"
exec "$node" \
  __name:=yangluo_bevfusion \
  _config:="$config" \
  _classes:="$classes" \
  _decode_only:=false
