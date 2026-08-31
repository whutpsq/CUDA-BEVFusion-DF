#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

build_dir="${BUILD_DIR:-$repo_root/build_aarch64_thor}"
cuda_root="${CUDA_HOME:-/usr/local/thor/cuda-12.8}"
tensorrt_root="${TENSORRT_ROOT:-/usr/local/thor/aarch64-linux-gnu}"
toolchain_root="${THOR_TOOLCHAIN_ROOT:-/usr/local/thor/aarch64--glibc--bleeding-edge-2024.02-1}"
spconv_root="${BEVFUSION_SPCONV_ROOT:-$(cd "$repo_root/.." && pwd)/libraries/3DSparseConvolution/libspconv}"
senseauto_root="${SENSEAUTO_INSTALL_ROOT:-/opt/senseauto}"
cuda_archs="${BEVFUSION_CUDA_ARCHS:-101}"
build_bag_runner="${BUILD_RSCL_BAG_RUNNER:-ON}"
build_online_node="${BUILD_RSCL_ONLINE_NODE:-ON}"
enable_ffmpeg="${RSCL_ENABLE_FFMPEG_DECODER:-OFF}"
enable_opencv_undistort="${RSCL_ENABLE_OPENCV_UNDISTORT:-ON}"

first_dir() {
  local pattern="$1"
  local found
  found="$(compgen -G "$pattern" | sort | head -n 1 || true)"
  if [[ -n "$found" && -d "$found" ]]; then
    printf '%s\n' "$found"
  fi
}

first_file() {
  local pattern="$1"
  local found
  found="$(compgen -G "$pattern" | sort | head -n 1 || true)"
  if [[ -n "$found" && -f "$found" ]]; then
    printf '%s\n' "$found"
  fi
}

rscl_root="${RSCL_SDK_ROOT:-$(first_dir "$senseauto_root/senseauto-rscl/*")}"
thirdparty_root="${RSCL_THIRDPARTY_ROOT:-$(first_dir "$senseauto_root/senseauto-3rdparty/*")}"
msgs_root="${RSCL_MSGS_ROOT:-$(first_dir "$senseauto_root/senseauto-msgs/*")}"
protobuf_root="${BEVFUSION_PROTOBUF_ROOT:-}"
if [[ -z "$protobuf_root" && -n "$thirdparty_root" && -d "$thirdparty_root/3rdparty/protobuf" ]]; then
  protobuf_root="$thirdparty_root/3rdparty/protobuf"
fi

senseauto_toolchain="${SENSEAUTO_THOR_TOOLCHAIN_FILE:-$(first_file "$senseauto_root/senseauto-buildtools/*/v0.0.1/toolchains/thor/linux-aarch64-gcc-thor.cmake")}"
if [[ -n "$senseauto_toolchain" ]]; then
  toolchain_file="$senseauto_toolchain"
else
  toolchain_file="${CMAKE_TOOLCHAIN_FILE:-$repo_root/cmake/toolchains/aarch64-thor.cmake}"
fi

cuda_include="$cuda_root/targets/aarch64-linux/include"
cuda_runtime_lib="$cuda_root/targets/aarch64-linux/lib"
cuda_target_stub_lib="$cuda_root/targets/aarch64-linux/lib/stubs"
cuda_thor_stub_lib="$cuda_root/thor/targets/aarch64-linux/lib/stubs"
cuda_stub_lib="$cuda_thor_stub_lib"
if [[ ! -f "$cuda_stub_lib/libcublasLt.so" && -f "$cuda_target_stub_lib/libcublasLt.so" ]]; then
  cuda_stub_lib="$cuda_target_stub_lib"
fi
cudart_library="$(first_file "$cuda_runtime_lib/libcudart.so*")"
cublaslt_library="$(first_file "$cuda_stub_lib/libcublasLt.so")"
cuda_driver_library="$(first_file "$cuda_stub_lib/libcuda.so")"
if [[ -z "$cuda_driver_library" ]]; then
  cuda_driver_library="$(first_file "$cuda_target_stub_lib/libcuda.so")"
fi
if [[ -z "$cuda_driver_library" ]]; then
  cuda_driver_library="$(first_file "$cuda_thor_stub_lib/libcuda.so")"
fi

if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "Cannot find nvcc under $cuda_root/bin. Set CUDA_HOME=/usr/local/thor/cuda-12.8." >&2
  exit 1
fi
if [[ -z "$cudart_library" ]]; then
  echo "Cannot find libcudart under $cuda_runtime_lib." >&2
  exit 1
fi
if [[ -z "$cublaslt_library" ]]; then
  echo "Cannot find libcublasLt stub under $cuda_stub_lib." >&2
  exit 1
fi
if [[ ! -f "$tensorrt_root/include/NvInfer.h" ]]; then
  echo "Cannot find TensorRT headers under $tensorrt_root/include. Set TENSORRT_ROOT=/usr/local/thor/aarch64-linux-gnu." >&2
  exit 1
fi
if [[ ! -f "$spconv_root/lib/aarch64_cuda12.8/libspconv.so" ]]; then
  echo "Cannot find aarch64 CUDA 12.8 spconv at $spconv_root/lib/aarch64_cuda12.8/libspconv.so." >&2
  exit 1
fi
if [[ -z "$rscl_root" || ! -f "$rscl_root/include/ad_rscl/runtime.h" ]]; then
  echo "Cannot find SenseAuto RSCL SDK. Set RSCL_SDK_ROOT to the senseauto-rscl package root." >&2
  exit 1
fi
if [[ -z "$thirdparty_root" || ! -d "$thirdparty_root/3rdparty" ]]; then
  echo "Cannot find SenseAuto thirdparty package. Set RSCL_THIRDPARTY_ROOT to the senseauto-3rdparty package root." >&2
  exit 1
fi
if [[ -z "$protobuf_root" || ! -f "$protobuf_root/include/google/protobuf/message.h" ]]; then
  echo "Cannot find SenseAuto protobuf root. Set BEVFUSION_PROTOBUF_ROOT to .../3rdparty/protobuf." >&2
  exit 1
fi

search_library() {
  local name="$1"
  shift
  local dir
  local found
  for dir in "$@"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    found="$(find "$dir" -maxdepth 6 \( -type f -o -type l \) \( -name "$name" -o -name "$name.*" \) 2>/dev/null | sort | head -n 1 || true)"
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
  return 1
}

toolchain_sysroot="$toolchain_root/aarch64-buildroot-linux-gnu/sysroot"
if [[ -z "$cuda_driver_library" ]]; then
  cuda_driver_library="${BEVFUSION_CUDA_DRIVER_LIBRARY:-$(search_library libcuda.so \
    "$cuda_root" \
    "$cuda_target_stub_lib" \
    "$cuda_thor_stub_lib" \
    "$tensorrt_root" \
    "$toolchain_sysroot/usr/lib" \
    "$toolchain_sysroot/lib" \
    "/usr/local/thor" || true)}"
fi
zlib_library="${BEVFUSION_ZLIB_LIBRARY:-$(search_library libz.so \
  "$thirdparty_root/3rdparty" \
  "$rscl_root" \
  "$toolchain_sysroot/usr/lib" \
  "$toolchain_sysroot/lib" || true)}"
uuid_library="${BEVFUSION_UUID_LIBRARY:-$(search_library libuuid.so \
  "$thirdparty_root/3rdparty" \
  "$rscl_root" \
  "$toolchain_sysroot/usr/lib" \
  "$toolchain_sysroot/lib" || true)}"
if [[ -z "$zlib_library" ]]; then
  echo "Cannot find aarch64 libz.so. Set BEVFUSION_ZLIB_LIBRARY to the target zlib path." >&2
  exit 1
fi
if [[ -z "$cuda_driver_library" ]]; then
  echo "Cannot find aarch64 libcuda.so. Set BEVFUSION_CUDA_DRIVER_LIBRARY to the target CUDA driver stub path." >&2
  exit 1
fi
if [[ -z "$uuid_library" ]]; then
  echo "Cannot find aarch64 libuuid.so. Set BEVFUSION_UUID_LIBRARY to the target uuid path." >&2
  exit 1
fi

ffmpeg_include_dir="${RSCL_FFMPEG_INCLUDE_DIR:-}"
ffmpeg_library_dir="${RSCL_FFMPEG_LIBRARY_DIR:-}"
ffmpeg_root="${RSCL_FFMPEG_ROOT:-}"
if [[ "$enable_ffmpeg" == "ON" ]]; then
  ffmpeg_header="${RSCL_FFMPEG_AVCODEC_HEADER:-}"
  ffmpeg_avcodec="${RSCL_FFMPEG_AVCODEC_LIBRARY:-}"
  ffmpeg_avutil="${RSCL_FFMPEG_AVUTIL_LIBRARY:-}"
  ffmpeg_swscale="${RSCL_FFMPEG_SWSCALE_LIBRARY:-}"

  if [[ -n "$ffmpeg_root" ]]; then
    [[ -z "$ffmpeg_header" && -f "$ffmpeg_root/include/libavcodec/avcodec.h" ]] && ffmpeg_header="$ffmpeg_root/include/libavcodec/avcodec.h"
    [[ -z "$ffmpeg_avcodec" ]] && ffmpeg_avcodec="$(first_file "$ffmpeg_root/lib/libavcodec.so")"
    [[ -z "$ffmpeg_avcodec" ]] && ffmpeg_avcodec="$(first_file "$ffmpeg_root/lib/libavcodec.so.*")"
    [[ -z "$ffmpeg_avutil" ]] && ffmpeg_avutil="$(first_file "$ffmpeg_root/lib/libavutil.so")"
    [[ -z "$ffmpeg_avutil" ]] && ffmpeg_avutil="$(first_file "$ffmpeg_root/lib/libavutil.so.*")"
    [[ -z "$ffmpeg_swscale" ]] && ffmpeg_swscale="$(first_file "$ffmpeg_root/lib/libswscale.so")"
    [[ -z "$ffmpeg_swscale" ]] && ffmpeg_swscale="$(first_file "$ffmpeg_root/lib/libswscale.so.*")"
  fi

  [[ -z "$ffmpeg_header" ]] && ffmpeg_header="$(search_library avcodec.h \
    "$ffmpeg_root" \
    "$thirdparty_root/3rdparty" \
    "$rscl_root" \
    "$toolchain_sysroot/usr/include" \
    "$toolchain_sysroot/usr/local/include" \
    "/usr/local/thor" || true)"
  [[ -z "$ffmpeg_avcodec" ]] && ffmpeg_avcodec="$(search_library libavcodec.so \
    "$ffmpeg_root" \
    "$thirdparty_root/3rdparty" \
    "$rscl_root" \
    "$toolchain_sysroot/usr/lib" \
    "$toolchain_sysroot/usr/local/lib" \
    "/usr/local/thor" || true)"
  [[ -z "$ffmpeg_avutil" ]] && ffmpeg_avutil="$(search_library libavutil.so \
    "$ffmpeg_root" \
    "$thirdparty_root/3rdparty" \
    "$rscl_root" \
    "$toolchain_sysroot/usr/lib" \
    "$toolchain_sysroot/usr/local/lib" \
    "/usr/local/thor" || true)"
  [[ -z "$ffmpeg_swscale" ]] && ffmpeg_swscale="$(search_library libswscale.so \
    "$ffmpeg_root" \
    "$thirdparty_root/3rdparty" \
    "$rscl_root" \
    "$toolchain_sysroot/usr/lib" \
    "$toolchain_sysroot/usr/local/lib" \
    "/usr/local/thor" || true)"
  if [[ -z "$ffmpeg_header" || -z "$ffmpeg_avcodec" || -z "$ffmpeg_avutil" || -z "$ffmpeg_swscale" ]]; then
    echo "Cannot find complete aarch64 FFmpeg headers/libraries." >&2
    echo "  ffmpeg_header=${ffmpeg_header:-<missing>}" >&2
    echo "  ffmpeg_avcodec=${ffmpeg_avcodec:-<missing>}" >&2
    echo "  ffmpeg_avutil=${ffmpeg_avutil:-<missing>}" >&2
    echo "  ffmpeg_swscale=${ffmpeg_swscale:-<missing>}" >&2
    echo "Set RSCL_FFMPEG_INCLUDE_DIR and RSCL_FFMPEG_LIBRARY_DIR, or set:" >&2
    echo "  RSCL_FFMPEG_AVCODEC_LIBRARY=/path/to/libavcodec.so" >&2
    echo "  RSCL_FFMPEG_AVUTIL_LIBRARY=/path/to/libavutil.so" >&2
    echo "  RSCL_FFMPEG_SWSCALE_LIBRARY=/path/to/libswscale.so" >&2
    exit 1
  fi
  if [[ -z "$ffmpeg_include_dir" ]]; then
    ffmpeg_include_dir="$(dirname "$(dirname "$ffmpeg_header")")"
  fi
  if [[ -z "$ffmpeg_library_dir" ]]; then
    ffmpeg_library_dir="$(dirname "$ffmpeg_avcodec")"
  fi
fi

opencv_root="${RSCL_OPENCV_ROOT:-$thirdparty_root/3rdparty/opencv4}"
opencv_include_dir="${RSCL_OPENCV_INCLUDE_DIR:-$opencv_root/include/opencv4}"
opencv_library_dir="${RSCL_OPENCV_LIBRARY_DIR:-$opencv_root/lib}"
opencv_core="${RSCL_OPENCV_CORE_LIBRARY:-}"
opencv_imgproc="${RSCL_OPENCV_IMGPROC_LIBRARY:-}"
opencv_calib3d="${RSCL_OPENCV_CALIB3D_LIBRARY:-}"
if [[ "$enable_opencv_undistort" == "ON" ]]; then
  [[ -z "$opencv_core" ]] && opencv_core="$(first_file "$opencv_library_dir/libopencv_core.so")"
  [[ -z "$opencv_imgproc" ]] && opencv_imgproc="$(first_file "$opencv_library_dir/libopencv_imgproc.so")"
  [[ -z "$opencv_calib3d" ]] && opencv_calib3d="$(first_file "$opencv_library_dir/libopencv_calib3d.so")"
  if [[ ! -f "$opencv_include_dir/opencv2/calib3d.hpp" || -z "$opencv_core" ||
        -z "$opencv_imgproc" || -z "$opencv_calib3d" ]]; then
    echo "Cannot find complete aarch64 OpenCV headers/libraries for camera undistortion." >&2
    echo "  opencv_include_dir=$opencv_include_dir" >&2
    echo "  opencv_core=${opencv_core:-<missing>}" >&2
    echo "  opencv_imgproc=${opencv_imgproc:-<missing>}" >&2
    echo "  opencv_calib3d=${opencv_calib3d:-<missing>}" >&2
    echo "Set RSCL_OPENCV_ROOT to the target OpenCV 4.x root." >&2
    exit 1
  fi
fi

link_dirs=("$cuda_runtime_lib" "$cuda_stub_lib" "$cuda_target_stub_lib" "$cuda_thor_stub_lib" "$(dirname "$zlib_library")" "$(dirname "$uuid_library")" "$(dirname "$cuda_driver_library")")
if [[ "$enable_ffmpeg" == "ON" ]]; then
  link_dirs+=("$ffmpeg_library_dir")
fi
if [[ "$enable_opencv_undistort" == "ON" ]]; then
  link_dirs+=("$opencv_library_dir")
fi
thor_link_flags=""
for link_dir in "${link_dirs[@]}"; do
  thor_link_flags+=" -L$link_dir -Wl,-rpath-link,$link_dir"
done
thor_link_flags="${thor_link_flags# }"

export PATH="$cuda_root/bin:$toolchain_root/bin:$PATH"
export CUDA_HOME="$cuda_root"
export CUDA_Inc="$cuda_include"
export TensorRT_Inc="$tensorrt_root/include"
export TensorRT_Lib="$tensorrt_root/lib/stubs"
export SPCONV_CUDA_VERSION=12.8
export RSCL_SDK_ROOT="$rscl_root"
export RSCL_THIRDPARTY_ROOT="$thirdparty_root"
if [[ -n "$msgs_root" ]]; then
  export RSCL_MSGS_ROOT="$msgs_root"
fi
export BEVFUSION_PROTOBUF_ROOT="$protobuf_root"

echo "repo_root=$repo_root"
echo "build_dir=$build_dir"
echo "toolchain_file=$toolchain_file"
echo "CUDA_HOME=$CUDA_HOME"
echo "TensorRT_Inc=$TensorRT_Inc"
echo "TensorRT_Lib=$TensorRT_Lib"
echo "BEVFUSION_CUDART_LIBRARY=$cudart_library"
echo "BEVFUSION_CUBLASLT_LIBRARY=$cublaslt_library"
echo "BEVFUSION_CUDA_DRIVER_LIBRARY=$cuda_driver_library"
echo "BEVFUSION_ZLIB_LIBRARY=$zlib_library"
echo "BEVFUSION_UUID_LIBRARY=$uuid_library"
if [[ "$enable_ffmpeg" == "ON" ]]; then
  echo "RSCL_FFMPEG_INCLUDE_DIR=$ffmpeg_include_dir"
  echo "RSCL_FFMPEG_LIBRARY_DIR=$ffmpeg_library_dir"
  echo "RSCL_FFMPEG_AVCODEC_LIBRARY=$ffmpeg_avcodec"
  echo "RSCL_FFMPEG_AVUTIL_LIBRARY=$ffmpeg_avutil"
  echo "RSCL_FFMPEG_SWSCALE_LIBRARY=$ffmpeg_swscale"
fi
if [[ "$enable_opencv_undistort" == "ON" ]]; then
  echo "RSCL_OPENCV_INCLUDE_DIR=$opencv_include_dir"
  echo "RSCL_OPENCV_LIBRARY_DIR=$opencv_library_dir"
  echo "RSCL_OPENCV_CORE_LIBRARY=$opencv_core"
  echo "RSCL_OPENCV_IMGPROC_LIBRARY=$opencv_imgproc"
  echo "RSCL_OPENCV_CALIB3D_LIBRARY=$opencv_calib3d"
fi
echo "THOR_LINK_FLAGS=$thor_link_flags"
echo "spconv_root=$spconv_root"
echo "RSCL_SDK_ROOT=$RSCL_SDK_ROOT"
echo "RSCL_THIRDPARTY_ROOT=$RSCL_THIRDPARTY_ROOT"
echo "RSCL_MSGS_ROOT=${RSCL_MSGS_ROOT:-<not found>}"
echo "BEVFUSION_PROTOBUF_ROOT=$BEVFUSION_PROTOBUF_ROOT"
echo "BEVFUSION_CUDA_ARCHS=$cuda_archs"

mkdir -p "$build_dir"
cd "$build_dir"

cmake_args=(
  "-DCMAKE_TOOLCHAIN_FILE=$toolchain_file"
  "-DTHOR_TOOLCHAIN_ROOT=$toolchain_root"
  "-DBEVFUSION_TARGET_ARCH=aarch64"
  "-DBEVFUSION_SPCONV_ARCH=aarch64"
  "-DBEVFUSION_SPCONV_ROOT=$spconv_root"
  "-DBEVFUSION_CUDA_ARCHS=$cuda_archs"
  "-DBEVFUSION_CUDA_INCLUDE_DIRS=$cuda_include"
  "-DBEVFUSION_CUDA_LIBRARY_DIRS=$cuda_runtime_lib;$cuda_stub_lib"
  "-DBEVFUSION_CUDART_LIBRARY=$cudart_library"
  "-DBEVFUSION_CUBLASLT_LIBRARY=$cublaslt_library"
  "-DBEVFUSION_CUDA_DRIVER_LIBRARY=$cuda_driver_library"
  "-DBEVFUSION_ZLIB_LIBRARY=$zlib_library"
  "-DBEVFUSION_UUID_LIBRARY=$uuid_library"
  "-DBEVFUSION_EXTRA_LINK_FLAGS=$thor_link_flags"
  "-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF"
  "-DCMAKE_SHARED_LINKER_FLAGS=$thor_link_flags"
  "-DCMAKE_EXE_LINKER_FLAGS=$thor_link_flags"
  "-DBEVFUSION_PROTOBUF_ROOT=$protobuf_root"
  "-DCUDA_TOOLKIT_ROOT_DIR=$cuda_root"
  # CMake 3.21 FindCUDA still assumes the legacy $CUDA_HOME/include/lib
  # layout. Thor keeps target headers and cudart under
  # targets/aarch64-linux, so seed both FindCUDA's public and internal cache
  # variables explicitly for cross compilation.
  "-DCUDA_NVCC_EXECUTABLE=$cuda_root/bin/nvcc"
  "-DCUDA_TOOLKIT_INCLUDE=$cuda_include"
  "-DCUDA_INCLUDE_DIRS=$cuda_include"
  "-DCUDA_CUDART_LIBRARY=$cudart_library"
  "-DCUDA_CUDA_LIBRARY=$cuda_driver_library"
  "-DCUDA_LIBRARIES=$cudart_library"
  "-DTensorRT_INCLUDE_DIR=$tensorrt_root/include"
  "-DTensorRT_NVINFER_LIBRARY=$tensorrt_root/lib/stubs/libnvinfer.so"
  "-DTensorRT_NVINFER_PLUGIN_LIBRARY=$tensorrt_root/lib/stubs/libnvinfer_plugin.so"
  "-DRSCL_SDK_ROOT=$rscl_root"
  "-DRSCL_THIRDPARTY_ROOT=$thirdparty_root"
  "-DBUILD_RSCL_CPP=ON"
  "-DBUILD_RSCL_BAG_RUNNER=$build_bag_runner"
  "-DBUILD_RSCL_ONLINE_NODE=$build_online_node"
  "-DRSCL_ENABLE_FFMPEG_DECODER=$enable_ffmpeg"
  "-DRSCL_ENABLE_OPENCV_UNDISTORT=$enable_opencv_undistort"
)
if [[ "$enable_ffmpeg" == "ON" ]]; then
  cmake_args+=(
    "-DRSCL_FFMPEG_ROOT=$ffmpeg_root"
    "-DRSCL_FFMPEG_INCLUDE_DIR=$ffmpeg_include_dir"
    "-DRSCL_FFMPEG_LIBRARY_DIR=$ffmpeg_library_dir"
    "-DRSCL_FFMPEG_AVCODEC_INCLUDE_DIR=$ffmpeg_include_dir"
    "-DRSCL_FFMPEG_AVUTIL_INCLUDE_DIR=$ffmpeg_include_dir"
    "-DRSCL_FFMPEG_SWSCALE_INCLUDE_DIR=$ffmpeg_include_dir"
    "-DRSCL_FFMPEG_AVCODEC_LIBRARY=$ffmpeg_avcodec"
    "-DRSCL_FFMPEG_AVUTIL_LIBRARY=$ffmpeg_avutil"
    "-DRSCL_FFMPEG_SWSCALE_LIBRARY=$ffmpeg_swscale"
  )
fi
if [[ "$enable_opencv_undistort" == "ON" ]]; then
  cmake_args+=(
    "-DRSCL_OPENCV_ROOT=$opencv_root"
    "-DRSCL_OPENCV_INCLUDE_DIR=$opencv_include_dir"
    "-DRSCL_OPENCV_LIBRARY_DIR=$opencv_library_dir"
    "-DRSCL_OPENCV_CORE_LIBRARY=$opencv_core"
    "-DRSCL_OPENCV_IMGPROC_LIBRARY=$opencv_imgproc"
    "-DRSCL_OPENCV_CALIB3D_LIBRARY=$opencv_calib3d"
  )
fi
if [[ -n "$msgs_root" ]]; then
  cmake_args+=("-DRSCL_MSGS_ROOT=$msgs_root")
fi
cmake_args+=("$@")
cmake_args+=("$repo_root")

cmake "${cmake_args[@]}"

make -j"$(nproc)"
