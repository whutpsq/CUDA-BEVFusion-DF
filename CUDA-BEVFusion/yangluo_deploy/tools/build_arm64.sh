#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
generated_root="$repo_root/yangluo_deploy/generated"
shadow_root="$generated_root/core_source"
core_build="$generated_root/build_arm64_core"
ros_ws="$repo_root/yangluo_deploy/ros_ws"
ros_prefix="${ROS_PREFIX:-/ota/miniconda3/envs/ros}"
trt_root="${TENSORRT_ROOT:-/opt/vehicle101}"
cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
spconv_root="${YANGLUO_SPCONV_ROOT:-$repo_root/../libraries/3DSparseConvolution/libspconv}"
cuda_archs="${YANGLUO_CUDA_ARCHS:-110}"
build_jobs="${YANGLUO_BUILD_JOBS:-4}"
nvcc_wrapper="$repo_root/yangluo_deploy/tools/nvcc_cxx17_wrapper.sh"

export PATH="$ros_prefix/bin:$cuda_root/bin:$PATH"
export CMAKE_PREFIX_PATH="$ros_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$ros_prefix/lib:$trt_root/lib/aarch64-linux-gnu:$cuda_root/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_Inc="$cuda_root/targets/sbsa-linux/include"
export CUDA_Lib="$cuda_root/targets/sbsa-linux/lib"
export TensorRT_Inc="$trt_root/include/aarch64-linux-gnu"
export TensorRT_Lib="$trt_root/lib/aarch64-linux-gnu"
export SPCONV_CUDA_VERSION="${SPCONV_CUDA_VERSION:-13.0}"

if [ "$(uname -m)" != "aarch64" ]; then echo "Expected aarch64 container" >&2; exit 1; fi
cuda_driver_stub="$cuda_root/targets/sbsa-linux/lib/stubs/libcuda.so"
if [ ! -f "$cuda_driver_stub" ]; then echo "Missing CUDA driver link stub: $cuda_driver_stub" >&2; exit 1; fi
cuda_cublaslt_stub="$cuda_root/targets/sbsa-linux/lib/stubs/libcublasLt.so"
if [ ! -f "$cuda_cublaslt_stub" ]; then echo "Missing cuBLASLt link stub: $cuda_cublaslt_stub" >&2; exit 1; fi
if [ ! -x "$nvcc_wrapper" ]; then echo "NVCC C++17 wrapper is not executable: $nvcc_wrapper" >&2; exit 1; fi
export REAL_NVCC="$cuda_root/bin/nvcc"
if [ ! -f "$spconv_root/lib/aarch64_cuda${SPCONV_CUDA_VERSION}/libspconv.so" ]; then
  echo "Missing target spconv: $spconv_root/lib/aarch64_cuda${SPCONV_CUDA_VERSION}/libspconv.so" >&2
  exit 1
fi

# Keep the passenger-car source tree untouched.  CUDA-BEVFusion checks in
# generated ONNX protobuf files, but those files were produced by protoc
# 3.21.12.  The ROS SDK provides protoc/libprotobuf 3.15.8, so create an
# isolated source shadow and regenerate only its protobuf files.
mkdir -p "$shadow_root/src" "$shadow_root/deploy_rscl" "$generated_root"
cp -a "$repo_root/src/." "$shadow_root/src/"
cp -a "$repo_root/deploy_rscl/." "$shadow_root/deploy_rscl/"
mkdir -p "$shadow_root/yangluo_deploy"
cp -a "$repo_root/yangluo_deploy/cpp" "$shadow_root/yangluo_deploy/"
cp -a "$repo_root/CMakeLists.txt" "$shadow_root/CMakeLists.txt"
ln -sfn "$repo_root/../dependencies" "$generated_root/dependencies"
ln -sfn "$repo_root/../libraries" "$generated_root/libraries"

# CUDA 13's CCCL (Thrust/CUB/libcu++) requires real C++17 syntax.  Apply the
# dialect change only to the generated Yangluo source shadow so the original
# passenger-car build files and their C++14 contract remain untouched.  A
# fresh generated build directory also prevents FindCUDA from reusing a stale
# CUDA_NVCC_EXECUTABLE or previously generated C++14 command lines.
sed -i 's/-std=c++14/-std=c++17/g' "$shadow_root/CMakeLists.txt"
if grep -q -- '-std=c++14' "$shadow_root/CMakeLists.txt"; then
  echo "Failed to convert the Yangluo source shadow to C++17" >&2
  exit 1
fi
rm -rf "$core_build"
echo "CUDA_CXX17_SHADOW_OK source=$shadow_root/CMakeLists.txt"

protoc_bin="$ros_prefix/bin/protoc"
if [ ! -x "$protoc_bin" ]; then echo "Missing matching protoc: $protoc_bin" >&2; exit 1; fi
(
  cd "$shadow_root/src/onnx"
  "$protoc_bin" --cpp_out=. onnx-ml.proto onnx-operators-ml.proto
  mv -f onnx-ml.pb.cc onnx-ml.pb.cpp
  mv -f onnx-operators-ml.pb.cc onnx-operators-ml.pb.cpp
)
echo "PROTOBUF_SHADOW_OK version=$($protoc_bin --version) source=$shadow_root"

cmake -S "$shadow_root" -B "$core_build" \
  -DBUILD_RSCL_CPP=ON \
  -DBUILD_RSCL_BAG_RUNNER=OFF \
  -DBUILD_RSCL_ONLINE_NODE=OFF \
  -DRSCL_ENABLE_FFMPEG_DECODER=OFF \
  -DRSCL_ENABLE_OPENCV_UNDISTORT=OFF \
  -DRSCL_ENABLE_NATIVE_UNDISTORT=ON \
  -DBEVFUSION_TARGET_ARCH=aarch64 \
  -DBEVFUSION_SPCONV_ARCH=aarch64 \
  -DBEVFUSION_SPCONV_ROOT="$spconv_root" \
  -DBEVFUSION_CUDA_ARCHS="$cuda_archs" \
  -DBEVFUSION_CUDA_DRIVER_LIBRARY="$cuda_driver_stub" \
  -DBEVFUSION_CUBLASLT_LIBRARY="$cuda_cublaslt_stub" \
  -DBEVFUSION_EXTRA_LINK_FLAGS="-Wl,--allow-shlib-undefined" \
  -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" \
  -DCUDA_NVCC_EXECUTABLE="$nvcc_wrapper" \
  -DBEVFUSION_PROTOBUF_ROOT="$ros_prefix"
cmake --build "$core_build" --target custom_layernorm bevfusion_core rscl_adapter_cpp -j"$build_jobs"

spconv_library="$spconv_root/lib/aarch64_cuda${SPCONV_CUDA_VERSION}/libspconv.so"
catkin_make -C "$ros_ws" -j"$build_jobs" -l"$build_jobs" \
  -DBEVFUSION_REPO_ROOT="$repo_root" \
  -DBEVFUSION_BUILD_DIR="$core_build" \
  -DBEVFUSION_SPCONV_LIBRARY="$spconv_library" \
  -DBEVFUSION_PROTOBUF_LIBRARY="$ros_prefix/lib/libprotobuf.so" \
  -DTENSORRT_ROOT="$trt_root" \
  -DCUDA_ROOT="$cuda_root" \
  -DCUDA_DRIVER_LIBRARY="$cuda_driver_stub" \
  -DCUBLASLT_LIBRARY="$cuda_cublaslt_stub"

file "$core_build/libbevfusion_core.so" "$core_build/libcustom_layernorm.so" \
  "$ros_ws/devel/lib/yangluo_bevfusion/yangluo_bevfusion_node"
echo "ARM64_BUILD_OK core=$core_build ros_ws=$ros_ws"
