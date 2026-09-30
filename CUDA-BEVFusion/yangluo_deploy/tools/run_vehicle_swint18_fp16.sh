#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

export YANGLUO_MODEL_DIR="$repo_root/model/yangluogang_swint18_epoch5_fp16_v1"
export YANGLUO_VEHICLE_CONFIG="$repo_root/yangluo_deploy/configs/runtime_vehicle_swint18_epoch5.yaml"
export YANGLUO_CLASSES_FILE="$repo_root/yangluo_deploy/configs/classes_swint18_epoch5.txt"

exec bash "$script_dir/run_vehicle_fp16.sh" "$@"
