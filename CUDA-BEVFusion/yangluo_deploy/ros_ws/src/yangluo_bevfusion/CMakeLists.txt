cmake_minimum_required(VERSION 3.10)
project(yangluo_bevfusion)

add_compile_options(-std=c++14 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  sensor_msgs
  std_msgs
  yangluo_bevfusion_msgs
)

set(BEVFUSION_REPO_ROOT "" CACHE PATH "CUDA-BEVFusion source root")
set(BEVFUSION_BUILD_DIR "" CACHE PATH "Directory containing librscl_adapter_cpp.a and libbevfusion_core.so")
set(BEVFUSION_SPCONV_LIBRARY "" CACHE FILEPATH "Target libspconv.so")
set(BEVFUSION_PROTOBUF_LIBRARY "" CACHE FILEPATH "Target libprotobuf.so")
set(TENSORRT_ROOT "/opt/vehicle101" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
set(ROS_OPENCV_ROOT "/ota/miniconda3/envs/ros" CACHE PATH "ROS OpenCV 4.5 prefix")
set(CUDA_DRIVER_LIBRARY "${CUDA_ROOT}/targets/sbsa-linux/lib/stubs/libcuda.so" CACHE FILEPATH
    "CUDA driver stub used only for linking in the GPU-less build container")

foreach(required_var BEVFUSION_REPO_ROOT BEVFUSION_BUILD_DIR BEVFUSION_SPCONV_LIBRARY BEVFUSION_PROTOBUF_LIBRARY)
  if("${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "Set -D${required_var}=...; see yangluo_deploy/ros_ws/README.md")
  endif()
endforeach()

find_library(RSCL_ADAPTER_LIBRARY rscl_adapter_cpp HINTS "${BEVFUSION_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(BEVFUSION_CORE_LIBRARY bevfusion_core HINTS "${BEVFUSION_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_LIBRARY nvinfer HINTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
find_library(CUDART_LIBRARY cudart HINTS "${CUDA_ROOT}/targets/sbsa-linux/lib" NO_DEFAULT_PATH REQUIRED)
find_library(CUBLASLT_LIBRARY cublasLt HINTS "${CUDA_ROOT}/targets/sbsa-linux/lib" NO_DEFAULT_PATH REQUIRED)
find_library(OPENCV_CORE_LIBRARY opencv_core HINTS "${ROS_OPENCV_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(OPENCV_IMGPROC_LIBRARY opencv_imgproc HINTS "${ROS_OPENCV_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(OPENCV_CALIB3D_LIBRARY opencv_calib3d HINTS "${ROS_OPENCV_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
if(NOT EXISTS "${CUDA_DRIVER_LIBRARY}")
  message(FATAL_ERROR "CUDA driver stub not found: ${CUDA_DRIVER_LIBRARY}")
endif()

catkin_package(CATKIN_DEPENDS roscpp sensor_msgs std_msgs yangluo_bevfusion_msgs)

include_directories(
  ${catkin_INCLUDE_DIRS}
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/include"
  "${BEVFUSION_REPO_ROOT}/../dependencies/stb"
  "${BEVFUSION_REPO_ROOT}/src"
  "${TENSORRT_ROOT}/include/aarch64-linux-gnu"
  "${CUDA_ROOT}/targets/sbsa-linux/include"
)

add_executable(yangluo_bevfusion_node src/yangluo_bevfusion_node.cpp)
add_dependencies(yangluo_bevfusion_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(yangluo_bevfusion_node
  -Wl,--whole-archive ${RSCL_ADAPTER_LIBRARY} -Wl,--no-whole-archive
  ${BEVFUSION_CORE_LIBRARY}
  ${BEVFUSION_SPCONV_LIBRARY}
  ${BEVFUSION_PROTOBUF_LIBRARY}
  ${NVINFER_LIBRARY}
  ${NVINFER_PLUGIN_LIBRARY}
  ${CUDART_LIBRARY}
  ${CUBLASLT_LIBRARY}
  ${CUDA_DRIVER_LIBRARY}
  ${OPENCV_CALIB3D_LIBRARY}
  ${OPENCV_IMGPROC_LIBRARY}
  ${OPENCV_CORE_LIBRARY}
  ${catkin_LIBRARIES}
  dl pthread
)
target_link_options(yangluo_bevfusion_node PRIVATE "-Wl,--allow-shlib-undefined")
set_target_properties(yangluo_bevfusion_node PROPERTIES
  BUILD_RPATH "${BEVFUSION_BUILD_DIR};${TENSORRT_ROOT}/lib/aarch64-linux-gnu;${CUDA_ROOT}/targets/sbsa-linux/lib;${ROS_OPENCV_ROOT}/lib"
)

# GPU-free/QEMU validation executable.  Compile only the real configuration,
# JPEG, PointCloud2 and synchronization path; intentionally do not link the
# TensorRT/CUDA/spconv inference stack used by yangluo_bevfusion_node.
add_executable(yangluo_bevfusion_decode_only_node
  src/yangluo_decode_only_node.cpp
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/config.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/json.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/codecs.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/sync.cpp"
)
add_dependencies(yangluo_bevfusion_decode_only_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(yangluo_bevfusion_decode_only_node
  ${catkin_LIBRARIES}
  pthread
)

# Offline ARM64 bag validation avoids roscpp communication threads, which are
# not reliable under x86-to-AArch64 QEMU user-mode emulation.
add_executable(yangluo_bevfusion_bag_validator
  src/yangluo_bag_validator.cpp
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/config.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/json.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/codecs.cpp"
  "${BEVFUSION_REPO_ROOT}/deploy_rscl/cpp/src/sync.cpp"
)
add_dependencies(yangluo_bevfusion_bag_validator ${catkin_EXPORTED_TARGETS})
target_link_libraries(yangluo_bevfusion_bag_validator
  ${catkin_LIBRARIES}
  pthread
)

install(TARGETS yangluo_bevfusion_node yangluo_bevfusion_decode_only_node yangluo_bevfusion_bag_validator
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
