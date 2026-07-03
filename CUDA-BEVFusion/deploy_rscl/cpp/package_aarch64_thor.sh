#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

build_dir="${BUILD_DIR:-$repo_root/build_aarch64_thor}"
package_dir="${PACKAGE_DIR:-$repo_root/package_aarch64_thor}"
cache_file="$build_dir/CMakeCache.txt"

if [[ ! -f "$cache_file" ]]; then
  echo "Cannot find CMakeCache.txt under $build_dir. Set BUILD_DIR=/path/to/build_aarch64_thor." >&2
  exit 1
fi

cache_value() {
  local key="$1"
  local line
  line="$(grep -E "^${key}:" "$cache_file" | tail -n 1 || true)"
  [[ -n "$line" ]] || return 0
  printf '%s\n' "${line#*=}"
}

copy_file() {
  local src="$1"
  local dst_dir="$2"
  [[ -n "$src" && -e "$src" ]] || return 0
  mkdir -p "$dst_dir"
  cp -aL "$src" "$dst_dir/"
}

copy_matching() {
  local pattern="$1"
  local dst_dir="$2"
  local matched=0
  shopt -s nullglob
  for src in $pattern; do
    copy_file "$src" "$dst_dir"
    matched=1
  done
  shopt -u nullglob
  return 0
}

rm -rf "$package_dir"
mkdir -p "$package_dir/bin" "$package_dir/lib" "$package_dir/deploy_rscl" "$package_dir/model"

copy_file "$build_dir/rscl_bevfusion_bag_runner" "$package_dir/bin"
copy_file "$build_dir/rscl_bevfusion_online_node" "$package_dir/bin"
copy_file "$build_dir/bevfusion" "$package_dir/bin"

copy_file "$build_dir/libbevfusion_core.so" "$package_dir/lib"
copy_file "$build_dir/libcustom_layernorm.so" "$package_dir/lib"
copy_file "$build_dir/librscl_online_backend_ad.so" "$package_dir/lib"

copy_file "$(cache_value BEVFUSION_SPCONV_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value Protobuf_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value PROTOBUF_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value BEVFUSION_ZLIB_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value BEVFUSION_UUID_LIBRARY)" "$package_dir/lib"
# Do not package the CUDA driver stub used for cross-linking. The vehicle must
# load the real NVIDIA driver libcuda.so.1 from its runtime environment.

copy_file "$(cache_value RSCL_FFMPEG_AVCODEC_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value RSCL_FFMPEG_AVUTIL_LIBRARY)" "$package_dir/lib"
copy_file "$(cache_value RSCL_FFMPEG_SWSCALE_LIBRARY)" "$package_dir/lib"

rscl_sdk_root="$(cache_value RSCL_SDK_ROOT)"
rscl_thirdparty_root="$(cache_value RSCL_THIRDPARTY_ROOT)"
rscl_msgs_root="$(cache_value RSCL_MSGS_ROOT)"
if [[ -n "$rscl_sdk_root" ]]; then
  copy_matching "$rscl_sdk_root/lib/libad_*.so*" "$package_dir/lib"
fi
if [[ -n "$rscl_msgs_root" ]]; then
  copy_matching "$rscl_msgs_root/lib/*.so*" "$package_dir/lib"
fi
if [[ -n "$rscl_thirdparty_root" ]]; then
  copy_matching "$rscl_thirdparty_root/3rdparty/protobuf/lib/libprotobuf.so*" "$package_dir/lib"
  copy_matching "$rscl_thirdparty_root/3rdparty/libz/lib/libz.so*" "$package_dir/lib"
fi

cp -a "$repo_root/deploy_rscl/configs" "$package_dir/deploy_rscl/"
if [[ -f "$repo_root/calibration.json" ]]; then
  cp -a "$repo_root/calibration.json" "$package_dir/"
fi

cat > "$package_dir/run_env.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$this_dir/lib:${LD_LIBRARY_PATH:-}"
export RSCL_ONLINE_BACKEND_LIB="$this_dir/lib/librscl_online_backend_ad.so"
EOF
chmod +x "$package_dir/run_env.sh"

cat > "$package_dir/README_DEPLOY.md" <<'EOF'
# CUDA-BEVFusion Thor AArch64 Runtime Bundle

Source `run_env.sh` before running:

```bash
cd /path/to/package_aarch64_thor
source ./run_env.sh

./bin/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /ota/qinghua/mybag.000.rsclbag \
  --decode-images-only \
  --max-frames 5
```

Model paths in the YAML must exist on the vehicle. Either copy the model
directory into this package and edit `cuda_model_root`, or keep the vehicle
absolute model path in the YAML.
EOF

echo "Package created at: $package_dir"
echo "Copy it to the vehicle, then run:"
echo "  cd <vehicle-package-path>"
echo "  source ./run_env.sh"
echo "  ./bin/rscl_bevfusion_bag_runner --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml --bag /ota/qinghua/mybag.000.rsclbag --decode-images-only --max-frames 5"
