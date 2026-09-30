#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

config="${YANGLUO_VEHICLE_CONFIG:-$repo_root/yangluo_deploy/configs/runtime_vehicle_swint18_epoch5_threshold_010.yaml}"
classes="${YANGLUO_CLASSES_FILE:-$repo_root/yangluo_deploy/configs/classes_swint18_epoch5.txt}"
node="$repo_root/yangluo_deploy/ros_ws/devel/lib/yangluo_bevfusion/yangluo_bevfusion_node"
run_dir="${YANGLUO_VEHICLE_RUN_DIR:-/home/nvidia/psq/swint18_vehicle_service}"
pid_file="$run_dir/node.pid"
status_file="$run_dir/status"
log_file="$run_dir/node.log"

export YANGLUO_MODEL_DIR="$repo_root/model/yangluogang_swint18_epoch5_fp16_v1"
export YANGLUO_VEHICLE_CONFIG="$config"
export YANGLUO_CLASSES_FILE="$classes"
export YANGLUO_VEHICLE_ROS_MASTER_URI="${YANGLUO_VEHICLE_ROS_MASTER_URI:-http://192.168.10.101:11311}"
export YANGLUO_VEHICLE_ROS_IP="${YANGLUO_VEHICLE_ROS_IP:-192.168.10.101}"
export ROS_MASTER_URI="$YANGLUO_VEHICLE_ROS_MASTER_URI"
export ROS_IP="$YANGLUO_VEHICLE_ROS_IP"
unset ROS_HOSTNAME

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|log}

  start    preflight and start the Swin18 FP16 vehicle node in background
  stop     stop /yangluo_bevfusion and wait for the local process to exit
  restart  stop and then start the node
  status   show process, ROS node, output topic, config, and recent inference
  log      show the latest 100 log lines
EOF
}

node_pids() {
  pgrep -f "^$node( |$)" 2>/dev/null || true
}

ros_node_reachable() {
  rosnode ping -c 1 /yangluo_bevfusion 2>&1 | grep -q 'xmlrpc reply'
}

start_node() {
  mkdir -p "$run_dir"

  local existing
  existing="$(node_pids | head -n 1)"
  if [ -n "$existing" ]; then
    echo "YANGLUO_VEHICLE_ALREADY_RUNNING pid=$existing"
    status_node
    return 0
  fi

  if ros_node_reachable; then
    echo "ERROR: /yangluo_bevfusion is already registered by another process." >&2
    return 61
  fi

  if ! grep -q '^score_threshold: 0.10$' "$config"; then
    echo "ERROR: expected score_threshold 0.10 in $config" >&2
    return 62
  fi
  if ! grep -q '^undistort_images: false$' "$config"; then
    echo "ERROR: expected raw-image inference in $config" >&2
    return 63
  fi

  if ! bash "$script_dir/run_vehicle_fp16.sh" --check-only; then
    echo "ERROR: vehicle runtime preflight failed." >&2
    return 64
  fi

  printf "%s\n" STARTING > "$status_file"
  : > "$log_file"

  nohup bash "$script_dir/run_vehicle_fp16.sh" >"$log_file" 2>&1 &
  local pid=$!
  printf "%s\n" "$pid" > "$pid_file"
  printf "%s\n" RUNNING > "$status_file"
  echo "YANGLUO_VEHICLE_START_REQUESTED pid=$pid"

  local attempt
  for attempt in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      local result=$?
      printf "%s\n" "$result" > "$status_file"
      echo "ERROR: vehicle node exited during startup with status $result." >&2
      tail -n 80 "$log_file" >&2
      return "$result"
    fi
    if ros_node_reachable; then
      echo "YANGLUO_VEHICLE_STARTED pid=$pid"
      return 0
    fi
    sleep 1
  done

  echo "ERROR: process is running but ROS node did not become ready in 20 seconds." >&2
  return 65
}

stop_node() {
  mkdir -p "$run_dir"
  local pids
  pids="$(node_pids)"

  if ros_node_reachable; then
    rosnode kill /yangluo_bevfusion >/dev/null 2>&1 || true
  fi

  local attempt
  for attempt in $(seq 1 10); do
    pids="$(node_pids)"
    [ -z "$pids" ] && break
    sleep 1
  done

  pids="$(node_pids)"
  if [ -n "$pids" ]; then
    echo "$pids" | xargs -r kill
    for attempt in $(seq 1 5); do
      pids="$(node_pids)"
      [ -z "$pids" ] && break
      sleep 1
    done
  fi

  pids="$(node_pids)"
  if [ -n "$pids" ]; then
    echo "ERROR: vehicle node did not stop: $pids" >&2
    return 66
  fi

  # The process may exit before the ROS master removes its XML-RPC
  # registration.  Wait for that registration to disappear so an immediate
  # restart is not rejected as a duplicate node.
  for attempt in $(seq 1 10); do
    if ! ros_node_reachable; then
      break
    fi
    sleep 1
  done

  if ros_node_reachable; then
    echo "ERROR: ROS master still reports /yangluo_bevfusion after shutdown." >&2
    return 67
  fi

  printf "%s\n" STOPPED > "$status_file"
  : > "$pid_file"
  echo "YANGLUO_VEHICLE_STOPPED"
}

status_node() {
  mkdir -p "$run_dir"
  echo "=== SERVICE ==="
  echo "repo=$repo_root"
  echo "config=$config"
  echo "classes=$classes"
  echo "ros_master=$ROS_MASTER_URI"
  echo "status=$(cat "$status_file" 2>/dev/null || echo UNKNOWN)"

  echo "=== PROCESS ==="
  local pids
  pids="$(node_pids)"
  if [ -n "$pids" ]; then
    ps -fp $pids
  else
    echo NOT_RUNNING
  fi

  echo "=== ROS NODE ==="
  rosnode ping -c 1 /yangluo_bevfusion 2>&1 || true

  echo "=== OUTPUT TOPIC ==="
  rostopic type /perception/bevfusion/objects 2>&1 || true

  echo "=== RECENT INFERENCE ==="
  grep -E     'BEVFusion ready|runner_infer_done|ERROR|Error|Segmentation|Aborted'     "$log_file" 2>/dev/null | tail -n 20 || true
}

show_log() {
  if [ -f "$log_file" ]; then
    tail -n 100 "$log_file"
  else
    echo "NO_LOG_FILE $log_file"
  fi
}

case "${1:-}" in
  start)
    start_node
    ;;
  stop)
    stop_node
    ;;
  restart)
    stop_node && start_node
    ;;
  status)
    status_node
    ;;
  log)
    show_log
    ;;
  *)
    usage
    exit 2
    ;;
esac

