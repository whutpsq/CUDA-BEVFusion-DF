#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

export YANGLUO_MODEL_DIR="$repo_root/model/yangluogang_swint18_epoch5_fp16_v1"

exec bash "$script_dir/build_thor_fp16.sh" "$@"
