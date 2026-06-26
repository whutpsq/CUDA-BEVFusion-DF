#!/usr/bin/env bash
set -euo pipefail

release_root="${1:-/path/to/695_x86}"
output_root="${2:-.sdk_cache/695_x86}"
with_thirdparty="${WITH_THIRDPARTY:-0}"

extract_first() {
  local pattern="$1"
  local label="$2"
  local pkg
  pkg="$(find "$release_root" -maxdepth 1 -name "$pattern" -print -quit)"
  if [[ -z "$pkg" ]]; then
    echo "Cannot find $label package in $release_root with pattern $pattern" >&2
    exit 1
  fi
  echo "Extracting $label package: $(basename "$pkg")"
  tar -xf "$pkg" -C "$output_root"
}

mkdir -p "$output_root"
extract_first 'senseauto-rscl-*.tar.gz' 'senseauto-rscl'
extract_first 'senseauto-msgs-*.tar.gz' 'senseauto-msgs'

if [[ "$with_thirdparty" == "1" ]]; then
  extract_first 'senseauto-3rdparty-*.tar.gz' 'senseauto-3rdparty'
else
  echo "Skipping senseauto-3rdparty. Re-run with WITH_THIRDPARTY=1 if runtime linker reports missing capnp/kj/protobuf/glog/gflags libraries."
fi

echo "SDK cache prepared at $output_root"
