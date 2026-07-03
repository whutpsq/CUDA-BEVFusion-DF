#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

ffmpeg_source="${1:-${FFMPEG_SOURCE_DIR:-}}"
prefix="${FFMPEG_INSTALL_PREFIX:-$repo_root/third_party/ffmpeg_aarch64_thor}"
toolchain_root="${THOR_TOOLCHAIN_ROOT:-/usr/local/thor/aarch64--glibc--bleeding-edge-2024.02-1}"
cross_prefix="${THOR_CROSS_PREFIX:-$toolchain_root/bin/aarch64-linux-}"

if [[ -z "$ffmpeg_source" || ! -f "$ffmpeg_source/configure" ]]; then
  echo "Usage: FFMPEG_SOURCE_DIR=/path/to/ffmpeg-source bash deploy_rscl/cpp/build_ffmpeg_aarch64_thor.sh" >&2
  echo "   or: bash deploy_rscl/cpp/build_ffmpeg_aarch64_thor.sh /path/to/ffmpeg-source" >&2
  exit 1
fi

mkdir -p "$prefix"
cd "$ffmpeg_source"

./configure \
  --prefix="$prefix" \
  --target-os=linux \
  --arch=aarch64 \
  --cross-prefix="$cross_prefix" \
  --enable-cross-compile \
  --enable-pic \
  --enable-shared \
  --disable-static \
  --disable-programs \
  --disable-doc \
  --disable-debug \
  --disable-autodetect \
  --disable-everything \
  --disable-avdevice \
  --disable-avformat \
  --disable-postproc \
  --disable-network \
  --disable-zlib \
  --disable-bzlib \
  --disable-lzma \
  --enable-avcodec \
  --enable-avutil \
  --enable-swscale \
  --enable-decoder=h264 \
  --enable-decoder=hevc \
  --enable-parser=h264 \
  --enable-parser=hevc

make -j"$(nproc)"
make install

echo "FFmpeg aarch64 install prefix: $prefix"
echo "Use it with:"
echo "  RSCL_ENABLE_FFMPEG_DECODER=ON RSCL_FFMPEG_ROOT=$prefix bash deploy_rscl/cpp/build_aarch64_thor.sh"
