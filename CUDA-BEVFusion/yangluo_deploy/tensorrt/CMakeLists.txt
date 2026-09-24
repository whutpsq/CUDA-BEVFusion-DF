cmake_minimum_required(VERSION 3.10)
project(yangluo_trt_builder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TENSORRT_ROOT "/opt/vehicle101" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS "${TENSORRT_ROOT}/include/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS
    "${CUDA_ROOT}/include"
    "${CUDA_ROOT}/targets/sbsa-linux/include"
  NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_LIBRARY nvinfer
  HINTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
  HINTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)

add_executable(yangluo_trt_builder build_engine.cpp)
target_include_directories(yangluo_trt_builder PRIVATE
  "${TENSORRT_INCLUDE_DIR}"
  "${CUDA_INCLUDE_DIR}"
)
target_link_libraries(yangluo_trt_builder PRIVATE
  "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${NVONNXPARSER_LIBRARY}" dl)
set_target_properties(yangluo_trt_builder PROPERTIES
  BUILD_RPATH "${TENSORRT_ROOT}/lib/aarch64-linux-gnu")
