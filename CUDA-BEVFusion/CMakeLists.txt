# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 2.8.12)
project(bevfusion)

set(CMAKE_BUILD_PYTHON "$ENV{USE_Python}")
if("${CMAKE_BUILD_PYTHON}" STREQUAL "")
  set(CMAKE_BUILD_PYTHON "OFF")
endif()
option(BUILD_RSCL_CPP "Build the C++ RSCL adapter core library" ON)
option(BUILD_RSCL_BAG_RUNNER "Build the C++ rsclbag offline runner" ON)
option(BUILD_RSCL_ONLINE_NODE "Build the C++ RSCL online inference node" ON)
option(RSCL_ENABLE_FFMPEG_DECODER "Enable FFmpeg-based H264/H265 camera decoding for the C++ RSCL adapter" ON)
option(RSCL_ENABLE_OPENCV_UNDISTORT "Enable OpenCV camera undistortion for the C++ RSCL adapter" OFF)
set(RSCL_BAG_BACKEND_SOURCE "" CACHE STRING "Optional source file implementing rscl_adapter::create_rscl_bag_reader with the vehicle RSCL SDK")
set(RSCL_ONLINE_BACKEND_SOURCE "" CACHE STRING "Optional source file implementing the dlopen-loaded RSCL online backend")
set(RSCL_RELEASE_ROOT "" CACHE PATH "Optional 695_x86 release directory or extracted SenseAuto package root")
set(SENSEAUTO_INSTALL_ROOT "$ENV{SENSEAUTO_INSTALL_ROOT}" CACHE PATH "Optional SenseAuto installed root, for example /opt/senseauto_active")
set(RSCL_SDK_ROOT "$ENV{RSCL_SDK_ROOT}" CACHE PATH "Optional extracted senseauto-rscl SDK root, ending at opt/senseauto/tmp/senseauto-rscl")
set(RSCL_THIRDPARTY_ROOT "$ENV{RSCL_THIRDPARTY_ROOT}" CACHE PATH "Optional extracted senseauto-3rdparty root, ending at opt/senseauto/tmp/senseauto-3rdparty")
set(RSCL_MSGS_ROOT "$ENV{RSCL_MSGS_ROOT}" CACHE PATH "Optional extracted senseauto-msgs root, ending at opt/senseauto/tmp/senseauto-msgs")
set(RSCL_SDK_INCLUDE_DIRS "" CACHE STRING "Optional RSCL SDK include directories")
set(RSCL_SDK_LIBRARY_DIRS "" CACHE STRING "Optional RSCL SDK library directories")
set(RSCL_SDK_LIBRARIES "" CACHE STRING "Optional RSCL SDK libraries for rscl_bevfusion_bag_runner")
set(RSCL_ONLINE_SDK_LIBRARIES "" CACHE STRING "Optional RSCL SDK libraries for librscl_online_backend_ad.so")
set(RSCL_FFMPEG_ROOT "$ENV{RSCL_FFMPEG_ROOT}" CACHE PATH "Optional FFmpeg root for RSCL camera decoder")
set(RSCL_FFMPEG_INCLUDE_DIR "" CACHE PATH "Optional FFmpeg include directory")
set(RSCL_FFMPEG_LIBRARY_DIR "" CACHE PATH "Optional FFmpeg library directory")
set(RSCL_OPENCV_ROOT "$ENV{RSCL_OPENCV_ROOT}" CACHE PATH "Optional target OpenCV root for camera undistortion")
set(RSCL_OPENCV_INCLUDE_DIR "" CACHE PATH "Optional target OpenCV include directory")
set(RSCL_OPENCV_LIBRARY_DIR "" CACHE PATH "Optional target OpenCV library directory")
set(RSCL_OPENCV_CORE_LIBRARY "" CACHE FILEPATH "Optional target OpenCV core library")
set(RSCL_OPENCV_IMGPROC_LIBRARY "" CACHE FILEPATH "Optional target OpenCV imgproc library")
set(RSCL_OPENCV_CALIB3D_LIBRARY "" CACHE FILEPATH "Optional target OpenCV calib3d library")
set(BEVFUSION_TARGET_ARCH "" CACHE STRING "Target runtime architecture. Defaults to CMAKE_SYSTEM_PROCESSOR, then CMAKE_HOST_SYSTEM_PROCESSOR")
set(BEVFUSION_SPCONV_ROOT "../libraries/3DSparseConvolution/libspconv" CACHE PATH "Path to the libspconv root directory")
set(BEVFUSION_SPCONV_ARCH "" CACHE STRING "Architecture directory prefix used under libspconv/lib, for example x86_64 or aarch64")
set(BEVFUSION_CUDA_ARCHS "86" CACHE STRING "Semicolon-separated CUDA SM architectures, for example 86, 87, or 90")
set(BEVFUSION_CUDA_INCLUDE_DIRS "" CACHE STRING "Optional additional CUDA include directories")
set(BEVFUSION_CUDA_LIBRARY_DIRS "" CACHE STRING "Optional additional CUDA library directories")
set(BEVFUSION_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH "Optional CUDA driver library or stub")
set(BEVFUSION_ZLIB_LIBRARY "" CACHE FILEPATH "Optional zlib library used by protobuf")
set(BEVFUSION_UUID_LIBRARY "" CACHE FILEPATH "Optional uuid library used by RSCL")
set(BEVFUSION_EXTRA_LINK_FLAGS "" CACHE STRING "Optional extra linker flags for all BEVFusion targets")
set(BEVFUSION_PROTOBUF_ROOT "$ENV{BEVFUSION_PROTOBUF_ROOT}" CACHE PATH "Optional protobuf root containing include and lib directories")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
if("${BEVFUSION_TARGET_ARCH}" STREQUAL "")
  if(CMAKE_CROSSCOMPILING AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "")
    set(BEVFUSION_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING "Target runtime architecture. Defaults to CMAKE_SYSTEM_PROCESSOR, then CMAKE_HOST_SYSTEM_PROCESSOR" FORCE)
  else()
    set(BEVFUSION_TARGET_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}" CACHE STRING "Target runtime architecture. Defaults to CMAKE_SYSTEM_PROCESSOR, then CMAKE_HOST_SYSTEM_PROCESSOR" FORCE)
  endif()
endif()
if("${BEVFUSION_SPCONV_ARCH}" STREQUAL "")
  set(BEVFUSION_SPCONV_ARCH "${BEVFUSION_TARGET_ARCH}" CACHE STRING "Architecture directory prefix used under libspconv/lib, for example x86_64 or aarch64" FORCE)
endif()
set(arch ${BEVFUSION_SPCONV_ARCH})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use static CUDA runtime" FORCE)
find_package(CUDA REQUIRED)
if(NOT "${BEVFUSION_PROTOBUF_ROOT}" STREQUAL "")
  if(EXISTS "${BEVFUSION_PROTOBUF_ROOT}/include/google/protobuf/message.h")
    set(Protobuf_INCLUDE_DIR "${BEVFUSION_PROTOBUF_ROOT}/include" CACHE PATH "Protobuf include directory" FORCE)
  else()
    find_path(Protobuf_INCLUDE_DIR google/protobuf/message.h
      HINTS
        "${BEVFUSION_PROTOBUF_ROOT}/include"
      NO_DEFAULT_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )
  endif()
  set(PROTOBUF_INCLUDE_DIR "${Protobuf_INCLUDE_DIR}" CACHE PATH "Protobuf include directory" FORCE)
  if(EXISTS "${BEVFUSION_PROTOBUF_ROOT}/lib/libprotobuf.so")
    set(Protobuf_LIBRARY "${BEVFUSION_PROTOBUF_ROOT}/lib/libprotobuf.so" CACHE FILEPATH "Protobuf library" FORCE)
  else()
    file(GLOB BEVFUSION_PROTOBUF_LIBRARY_CANDIDATES
      "${BEVFUSION_PROTOBUF_ROOT}/lib/libprotobuf.so.*"
      "${BEVFUSION_PROTOBUF_ROOT}/lib64/libprotobuf.so.*"
    )
    list(SORT BEVFUSION_PROTOBUF_LIBRARY_CANDIDATES)
    list(LENGTH BEVFUSION_PROTOBUF_LIBRARY_CANDIDATES BEVFUSION_PROTOBUF_LIBRARY_CANDIDATE_COUNT)
    if(BEVFUSION_PROTOBUF_LIBRARY_CANDIDATE_COUNT GREATER 0)
      list(GET BEVFUSION_PROTOBUF_LIBRARY_CANDIDATES 0 Protobuf_LIBRARY_CANDIDATE)
      set(Protobuf_LIBRARY "${Protobuf_LIBRARY_CANDIDATE}" CACHE FILEPATH "Protobuf library" FORCE)
    else()
      find_library(Protobuf_LIBRARY protobuf
        HINTS
          "${BEVFUSION_PROTOBUF_ROOT}/lib"
          "${BEVFUSION_PROTOBUF_ROOT}/lib64"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
      )
    endif()
  endif()
  set(PROTOBUF_LIBRARY "${Protobuf_LIBRARY}" CACHE FILEPATH "Protobuf library" FORCE)
endif()
find_package(Protobuf REQUIRED)
if(NOT "${BEVFUSION_PROTOBUF_ROOT}" STREQUAL "" AND EXISTS "${BEVFUSION_PROTOBUF_ROOT}/include/google/protobuf/message.h")
  set(Protobuf_INCLUDE_DIR "${BEVFUSION_PROTOBUF_ROOT}/include" CACHE PATH "Protobuf include directory" FORCE)
  set(PROTOBUF_INCLUDE_DIR "${BEVFUSION_PROTOBUF_ROOT}/include" CACHE PATH "Protobuf include directory" FORCE)
  set(Protobuf_INCLUDE_DIRS "${BEVFUSION_PROTOBUF_ROOT}/include")
  set(PROTOBUF_INCLUDE_DIRS "${BEVFUSION_PROTOBUF_ROOT}/include")
endif()
if(NOT "${BEVFUSION_PROTOBUF_ROOT}" STREQUAL "" AND NOT "${Protobuf_LIBRARY}" MATCHES "NOTFOUND")
  set(Protobuf_LIBRARIES "${Protobuf_LIBRARY}")
  set(PROTOBUF_LIBRARIES "${Protobuf_LIBRARY}")
elseif(("${Protobuf_LIBRARIES}" STREQUAL "" OR "${Protobuf_LIBRARIES}" MATCHES "NOTFOUND") AND NOT "${PROTOBUF_LIBRARIES}" MATCHES "NOTFOUND")
  set(Protobuf_LIBRARIES "${PROTOBUF_LIBRARIES}")
endif()
message(STATUS "Protobuf_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIBRARIES = ${Protobuf_LIBRARIES}")

find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS
    $ENV{TensorRT_Inc}
    $ENV{TENSORRT_INCLUDE}
    $ENV{TENSORRT_INCLUDE_DIR}
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/TensorRT/include
)

find_library(TensorRT_NVINFER_LIBRARY nvinfer
  HINTS
    $ENV{TensorRT_Lib}
    $ENV{TENSORRT_LIB}
    $ENV{TENSORRT_LIBRARY_DIR}
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/TensorRT/lib
)

find_library(TensorRT_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS
    $ENV{TensorRT_Lib}
    $ENV{TENSORRT_LIB}
    $ENV{TENSORRT_LIBRARY_DIR}
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/TensorRT/lib
)

if("${TensorRT_INCLUDE_DIR}" STREQUAL "TensorRT_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Cannot find NvInfer.h. Set TensorRT_Inc to the TensorRT include directory, for example: export TensorRT_Inc=/path/to/TensorRT/include")
endif()
if("${TensorRT_NVINFER_LIBRARY}" STREQUAL "TensorRT_NVINFER_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Cannot find libnvinfer.so. Set TensorRT_Lib to the TensorRT lib directory, for example: export TensorRT_Lib=/path/to/TensorRT/lib")
endif()
if("${TensorRT_NVINFER_PLUGIN_LIBRARY}" STREQUAL "TensorRT_NVINFER_PLUGIN_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Cannot find libnvinfer_plugin.so. Set TensorRT_Lib to the TensorRT lib directory, for example: export TensorRT_Lib=/path/to/TensorRT/lib")
endif()

get_filename_component(TensorRT_LIBRARY_DIR ${TensorRT_NVINFER_LIBRARY} PATH)
message(STATUS "TensorRT_INCLUDE_DIR = ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT_LIBRARY_DIR = ${TensorRT_LIBRARY_DIR}")

set(spconv_include ${BEVFUSION_SPCONV_ROOT}/include)
set(spconv_lib ${BEVFUSION_SPCONV_ROOT}/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
find_library(BEVFUSION_SPCONV_LIBRARY spconv
  HINTS
    ${spconv_lib}
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if("${BEVFUSION_SPCONV_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_SPCONV_LIBRARY libspconv.so)
endif()
find_library(BEVFUSION_CUDART_LIBRARY cudart
  HINTS
    $ENV{CUDA_Lib}
    ${BEVFUSION_CUDA_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if("${BEVFUSION_CUDART_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_CUDART_LIBRARY libcudart.so)
endif()
find_library(BEVFUSION_CUBLASLT_LIBRARY cublasLt
  HINTS
    $ENV{CUDA_Lib}
    ${BEVFUSION_CUDA_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if("${BEVFUSION_CUBLASLT_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_CUBLASLT_LIBRARY libcublasLt.so)
endif()
if("${BEVFUSION_CUDA_DRIVER_LIBRARY}" STREQUAL "")
  find_library(BEVFUSION_CUDA_DRIVER_LIBRARY cuda
    HINTS
      $ENV{CUDA_Lib}
      ${BEVFUSION_CUDA_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
endif()
if("${BEVFUSION_CUDA_DRIVER_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_CUDA_DRIVER_LIBRARY "")
endif()
if("${BEVFUSION_ZLIB_LIBRARY}" STREQUAL "")
  find_library(BEVFUSION_ZLIB_LIBRARY z
    HINTS
      ${RSCL_SDK_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
endif()
if("${BEVFUSION_ZLIB_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_ZLIB_LIBRARY "")
endif()
if("${BEVFUSION_UUID_LIBRARY}" STREQUAL "")
  find_library(BEVFUSION_UUID_LIBRARY uuid
    HINTS
      ${RSCL_SDK_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
endif()
if("${BEVFUSION_UUID_LIBRARY}" MATCHES "NOTFOUND")
  set(BEVFUSION_UUID_LIBRARY "")
endif()
message(STATUS "BEVFUSION_TARGET_ARCH = ${BEVFUSION_TARGET_ARCH}")
message(STATUS "BEVFUSION_SPCONV_ARCH = ${BEVFUSION_SPCONV_ARCH}")
message(STATUS "spconv_lib = ${spconv_lib}")
message(STATUS "BEVFUSION_SPCONV_LIBRARY = ${BEVFUSION_SPCONV_LIBRARY}")
message(STATUS "BEVFUSION_CUDART_LIBRARY = ${BEVFUSION_CUDART_LIBRARY}")
message(STATUS "BEVFUSION_CUBLASLT_LIBRARY = ${BEVFUSION_CUBLASLT_LIBRARY}")
message(STATUS "BEVFUSION_CUDA_DRIVER_LIBRARY = ${BEVFUSION_CUDA_DRIVER_LIBRARY}")
message(STATUS "BEVFUSION_ZLIB_LIBRARY = ${BEVFUSION_ZLIB_LIBRARY}")
message(STATUS "BEVFUSION_UUID_LIBRARY = ${BEVFUSION_UUID_LIBRARY}")

set(BEVFUSION_EXTRA_LINK_LIBRARIES "")
if(NOT "${BEVFUSION_CUDA_DRIVER_LIBRARY}" STREQUAL "")
  list(APPEND BEVFUSION_EXTRA_LINK_LIBRARIES ${BEVFUSION_CUDA_DRIVER_LIBRARY})
endif()
if(NOT "${BEVFUSION_ZLIB_LIBRARY}" STREQUAL "")
  list(APPEND BEVFUSION_EXTRA_LINK_LIBRARIES ${BEVFUSION_ZLIB_LIBRARY})
endif()
if(NOT "${BEVFUSION_UUID_LIBRARY}" STREQUAL "")
  list(APPEND BEVFUSION_EXTRA_LINK_LIBRARIES ${BEVFUSION_UUID_LIBRARY})
endif()

#if("$ENV{SPCONV_CUDA_VERSION}" EQUAL "11.4")
#  # For cuda 11.x
#  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
#  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
#else()
#  # For cuda 12.x
#  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
#  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
#endif()

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -std=c++14
)
foreach(cuda_arch ${BEVFUSION_CUDA_ARCHS})
  if(NOT "${cuda_arch}" STREQUAL "")
    # Embed a native cubin for the selected GPU so deployment does not depend
    # on the target having the NVIDIA PTX JIT compiler installed. Keep PTX as
    # a forward-compatible fallback for environments that do provide the JIT.
    list(APPEND CUDA_NVCC_FLAGS
      -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}
      -gencode arch=compute_${cuda_arch},code=compute_${cuda_arch}
    )
  endif()
endforeach()
message(STATUS "BEVFUSION_CUDA_ARCHS = ${BEVFUSION_CUDA_ARCHS}")

include_directories(BEFORE
  ${Protobuf_INCLUDE_DIRS}
)

include_directories(
  $ENV{CUDA_Inc}
  ${BEVFUSION_CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIR}
  $ENV{Python_Inc}
  ../dependencies/stb
  ../dependencies/dlpack/include
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  ${spconv_include}
  deploy_rscl/cpp/include
  src/common
  src/
)

if("${RSCL_SDK_ROOT}" STREQUAL "")
  set(RSCL_SDK_CANDIDATES "")
  if(NOT "$ENV{SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_INSTALL_CANDIDATES
      "$ENV{SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-rscl"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/senseauto-rscl"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-rscl"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-rscl"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-rscl"
    )
    list(APPEND RSCL_SDK_CANDIDATES ${RSCL_ENV_INSTALL_CANDIDATES})
  endif()
  if(NOT "${SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_INSTALL_CANDIDATES
      "${SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-rscl"
      "${SENSEAUTO_INSTALL_ROOT}/senseauto-rscl"
      "${SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-rscl"
      "${SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-rscl"
      "${SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-rscl"
    )
    list(APPEND RSCL_SDK_CANDIDATES ${RSCL_INSTALL_CANDIDATES})
  endif()
  file(GLOB RSCL_OPT_CANDIDATES
    "/opt/senseauto_active/tmp/senseauto-rscl"
    "/opt/senseauto_active/senseauto-rscl"
    "/opt/senseauto_active/*/tmp/senseauto-rscl"
    "/opt/senseauto_active/*/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto_active/*/*/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto/senseauto-rscl"
    "/opt/senseauto/*/tmp/senseauto-rscl"
    "/opt/senseauto/*/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto/*/*/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto_install_path/tmp/senseauto-rscl"
    "/opt/senseauto_install_path/senseauto-rscl"
    "/opt/senseauto_install_path/*/tmp/senseauto-rscl"
    "/opt/senseauto_install_path/*/opt/senseauto/tmp/senseauto-rscl"
    "/opt/senseauto_install_path/*/*/opt/senseauto/tmp/senseauto-rscl"
  )
  list(APPEND RSCL_SDK_CANDIDATES ${RSCL_OPT_CANDIDATES})
  if(NOT "$ENV{RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_RELEASE_CANDIDATES
      "$ENV{RSCL_RELEASE_ROOT}/senseauto-rscl*/opt/senseauto/tmp/senseauto-rscl"
      "$ENV{RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-rscl"
    )
    list(APPEND RSCL_SDK_CANDIDATES ${RSCL_ENV_RELEASE_CANDIDATES})
  endif()
  if(NOT "${RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_RELEASE_CANDIDATES
      "${RSCL_RELEASE_ROOT}/senseauto-rscl*/opt/senseauto/tmp/senseauto-rscl"
      "${RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-rscl"
      "${RSCL_RELEASE_ROOT}/opt/senseauto/tmp/senseauto-rscl"
    )
    list(APPEND RSCL_SDK_CANDIDATES ${RSCL_RELEASE_CANDIDATES})
  endif()
  file(GLOB RSCL_WORKSPACE_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/senseauto-rscl*/opt/senseauto/tmp/senseauto-rscl"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/*/opt/senseauto/tmp/senseauto-rscl"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-rscl*/opt/senseauto/tmp/senseauto-rscl"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/opt/senseauto/tmp/senseauto-rscl"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-rscl*"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/senseauto-rscl*"
  )
  list(APPEND RSCL_SDK_CANDIDATES
    ${RSCL_WORKSPACE_CANDIDATES}
  )

  foreach(candidate ${RSCL_SDK_CANDIDATES})
    if(EXISTS "${candidate}/include/ad_bag/bag_reader.h" AND EXISTS "${candidate}/lib/libad_bag.so")
      set(RSCL_SDK_ROOT "${candidate}" CACHE PATH "Optional extracted senseauto-rscl SDK root, ending at opt/senseauto/tmp/senseauto-rscl" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT "${RSCL_SDK_ROOT}" STREQUAL "")
  if("${RSCL_SDK_INCLUDE_DIRS}" STREQUAL "")
    set(RSCL_SDK_INCLUDE_DIRS "${RSCL_SDK_ROOT}/include" CACHE STRING "Optional RSCL SDK include directories" FORCE)
  endif()
  if("${RSCL_SDK_LIBRARY_DIRS}" STREQUAL "")
    set(RSCL_SDK_LIBRARY_DIRS "${RSCL_SDK_ROOT}/lib" CACHE STRING "Optional RSCL SDK library directories" FORCE)
  endif()
  if("${RSCL_SDK_LIBRARIES}" STREQUAL "")
    set(RSCL_SDK_LIBRARIES "ad_bag;ad_serde;ad_serde_capnp" CACHE STRING "Optional RSCL SDK libraries for rscl_bevfusion_bag_runner" FORCE)
  endif()
  message(STATUS "RSCL_SDK_ROOT = ${RSCL_SDK_ROOT}")
endif()

if("${RSCL_THIRDPARTY_ROOT}" STREQUAL "")
  set(RSCL_THIRDPARTY_CANDIDATES "")
  if(NOT "$ENV{SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_THIRDPARTY_INSTALL_CANDIDATES
      "$ENV{SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-3rdparty"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/senseauto-3rdparty"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-3rdparty"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-3rdparty"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-3rdparty"
    )
    list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_ENV_THIRDPARTY_INSTALL_CANDIDATES})
  endif()
  if(NOT "${SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_THIRDPARTY_INSTALL_CANDIDATES
      "${SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-3rdparty"
      "${SENSEAUTO_INSTALL_ROOT}/senseauto-3rdparty"
      "${SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-3rdparty"
      "${SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-3rdparty"
      "${SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-3rdparty"
    )
    list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_THIRDPARTY_INSTALL_CANDIDATES})
  endif()
  file(GLOB RSCL_THIRDPARTY_OPT_CANDIDATES
    "/opt/senseauto_active/tmp/senseauto-3rdparty"
    "/opt/senseauto_active/senseauto-3rdparty"
    "/opt/senseauto_active/*/tmp/senseauto-3rdparty"
    "/opt/senseauto_active/*/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto_active/*/*/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto/senseauto-3rdparty"
    "/opt/senseauto/*/tmp/senseauto-3rdparty"
    "/opt/senseauto/*/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto/*/*/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto_install_path/tmp/senseauto-3rdparty"
    "/opt/senseauto_install_path/senseauto-3rdparty"
    "/opt/senseauto_install_path/*/tmp/senseauto-3rdparty"
    "/opt/senseauto_install_path/*/opt/senseauto/tmp/senseauto-3rdparty"
    "/opt/senseauto_install_path/*/*/opt/senseauto/tmp/senseauto-3rdparty"
  )
  list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_THIRDPARTY_OPT_CANDIDATES})
  if(NOT "$ENV{RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_THIRDPARTY_CANDIDATES
      "$ENV{RSCL_RELEASE_ROOT}/senseauto-3rdparty*/opt/senseauto/tmp/senseauto-3rdparty"
      "$ENV{RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-3rdparty"
    )
    list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_ENV_THIRDPARTY_CANDIDATES})
  endif()
  if(NOT "${RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_THIRDPARTY_RELEASE_CANDIDATES
      "${RSCL_RELEASE_ROOT}/senseauto-3rdparty*/opt/senseauto/tmp/senseauto-3rdparty"
      "${RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-3rdparty"
    )
    list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_THIRDPARTY_RELEASE_CANDIDATES})
  endif()
  file(GLOB RSCL_THIRDPARTY_WORKSPACE_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/senseauto-3rdparty*/opt/senseauto/tmp/senseauto-3rdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/*/opt/senseauto/tmp/senseauto-3rdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-3rdparty*/opt/senseauto/tmp/senseauto-3rdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/opt/senseauto/tmp/senseauto-3rdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-3rdparty*"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/senseauto-3rdparty*"
  )
  list(APPEND RSCL_THIRDPARTY_CANDIDATES ${RSCL_THIRDPARTY_WORKSPACE_CANDIDATES})
  foreach(candidate ${RSCL_THIRDPARTY_CANDIDATES})
    if(EXISTS "${candidate}/3rdparty/lib")
      set(RSCL_THIRDPARTY_ROOT "${candidate}" CACHE PATH "Optional extracted senseauto-3rdparty root, ending at opt/senseauto/tmp/senseauto-3rdparty" FORCE)
      break()
    endif()
  endforeach()
endif()

if("${RSCL_MSGS_ROOT}" STREQUAL "")
  set(RSCL_MSGS_CANDIDATES "")
  if(NOT "$ENV{SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_MSGS_INSTALL_CANDIDATES
      "$ENV{SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-msgs"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/senseauto-msgs"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-msgs"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-msgs"
      "$ENV{SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-msgs"
    )
    list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_ENV_MSGS_INSTALL_CANDIDATES})
  endif()
  if(NOT "${SENSEAUTO_INSTALL_ROOT}" STREQUAL "")
    file(GLOB RSCL_MSGS_INSTALL_CANDIDATES
      "${SENSEAUTO_INSTALL_ROOT}/tmp/senseauto-msgs"
      "${SENSEAUTO_INSTALL_ROOT}/senseauto-msgs"
      "${SENSEAUTO_INSTALL_ROOT}/*/tmp/senseauto-msgs"
      "${SENSEAUTO_INSTALL_ROOT}/*/opt/senseauto/tmp/senseauto-msgs"
      "${SENSEAUTO_INSTALL_ROOT}/*/*/opt/senseauto/tmp/senseauto-msgs"
    )
    list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_MSGS_INSTALL_CANDIDATES})
  endif()
  file(GLOB RSCL_MSGS_OPT_CANDIDATES
    "/opt/senseauto_active/tmp/senseauto-msgs"
    "/opt/senseauto_active/senseauto-msgs"
    "/opt/senseauto_active/*/tmp/senseauto-msgs"
    "/opt/senseauto_active/*/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto_active/*/*/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto/senseauto-msgs"
    "/opt/senseauto/*/tmp/senseauto-msgs"
    "/opt/senseauto/*/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto/*/*/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto_install_path/tmp/senseauto-msgs"
    "/opt/senseauto_install_path/senseauto-msgs"
    "/opt/senseauto_install_path/*/tmp/senseauto-msgs"
    "/opt/senseauto_install_path/*/opt/senseauto/tmp/senseauto-msgs"
    "/opt/senseauto_install_path/*/*/opt/senseauto/tmp/senseauto-msgs"
  )
  list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_MSGS_OPT_CANDIDATES})
  if(NOT "$ENV{RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_ENV_MSGS_CANDIDATES
      "$ENV{RSCL_RELEASE_ROOT}/senseauto-msgs*/opt/senseauto/tmp/senseauto-msgs"
      "$ENV{RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-msgs"
    )
    list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_ENV_MSGS_CANDIDATES})
  endif()
  if(NOT "${RSCL_RELEASE_ROOT}" STREQUAL "")
    file(GLOB RSCL_MSGS_RELEASE_CANDIDATES
      "${RSCL_RELEASE_ROOT}/senseauto-msgs*/opt/senseauto/tmp/senseauto-msgs"
      "${RSCL_RELEASE_ROOT}/*/opt/senseauto/tmp/senseauto-msgs"
    )
    list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_MSGS_RELEASE_CANDIDATES})
  endif()
  file(GLOB RSCL_MSGS_WORKSPACE_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/senseauto-msgs*/opt/senseauto/tmp/senseauto-msgs"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/695_x86/*/opt/senseauto/tmp/senseauto-msgs"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-msgs*/opt/senseauto/tmp/senseauto-msgs"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/opt/senseauto/tmp/senseauto-msgs"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/senseauto-msgs*"
    "${CMAKE_CURRENT_SOURCE_DIR}/.sdk_cache/${BEVFUSION_TARGET_ARCH}/*/senseauto-msgs*"
  )
  list(APPEND RSCL_MSGS_CANDIDATES ${RSCL_MSGS_WORKSPACE_CANDIDATES})
  foreach(candidate ${RSCL_MSGS_CANDIDATES})
    if(EXISTS "${candidate}/lib")
      set(RSCL_MSGS_ROOT "${candidate}" CACHE PATH "Optional extracted senseauto-msgs root, ending at opt/senseauto/tmp/senseauto-msgs" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT "${RSCL_THIRDPARTY_ROOT}" STREQUAL "")
  list(APPEND RSCL_SDK_LIBRARY_DIRS "${RSCL_THIRDPARTY_ROOT}/3rdparty/lib")
  message(STATUS "RSCL_THIRDPARTY_ROOT = ${RSCL_THIRDPARTY_ROOT}")
endif()

if(NOT "${RSCL_MSGS_ROOT}" STREQUAL "")
  list(APPEND RSCL_SDK_INCLUDE_DIRS "${RSCL_MSGS_ROOT}/include")
  list(APPEND RSCL_SDK_LIBRARY_DIRS "${RSCL_MSGS_ROOT}/lib")
  message(STATUS "RSCL_MSGS_ROOT = ${RSCL_MSGS_ROOT}")
endif()

link_directories(
  $ENV{CUDA_Lib}
  ${BEVFUSION_CUDA_LIBRARY_DIRS}
  ${TensorRT_LIBRARY_DIR}
  ${spconv_lib}
  build
  $ENV{Python_Lib}
)

function(bevfusion_apply_extra_link_flags target_name)
  if(NOT "${BEVFUSION_EXTRA_LINK_FLAGS}" STREQUAL "")
    get_target_property(existing_link_flags ${target_name} LINK_FLAGS)
    if("${existing_link_flags}" STREQUAL "existing_link_flags-NOTFOUND")
      set(existing_link_flags "")
    endif()
    set_target_properties(${target_name} PROPERTIES
      LINK_FLAGS "${existing_link_flags} ${BEVFUSION_EXTRA_LINK_FLAGS}"
    )
  endif()
endfunction()

set(RSCL_SDK_TARGET_INCLUDE_DIRS ${RSCL_SDK_INCLUDE_DIRS})
if(NOT "${RSCL_THIRDPARTY_ROOT}" STREQUAL "")
  list(APPEND RSCL_SDK_TARGET_INCLUDE_DIRS "${RSCL_THIRDPARTY_ROOT}/3rdparty/include")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ad_rscl/runtime.h")
  list(APPEND RSCL_SDK_TARGET_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

set(RSCL_SDK_LINK_LIBRARIES "")
if(NOT "${RSCL_SDK_LIBRARIES}" STREQUAL "")
  foreach(rscl_lib ${RSCL_SDK_LIBRARIES})
    string(REPLACE "-" "_" rscl_lib_var "${rscl_lib}")
    string(TOUPPER "${rscl_lib_var}" rscl_lib_var)
    if(NOT "${RSCL_SDK_ROOT}" STREQUAL "" AND EXISTS "${RSCL_SDK_ROOT}/lib/lib${rscl_lib}.so")
      set(RSCL_${rscl_lib_var}_LIBRARY "${RSCL_SDK_ROOT}/lib/lib${rscl_lib}.so" CACHE FILEPATH "RSCL ${rscl_lib} library" FORCE)
    else()
      find_library(RSCL_${rscl_lib_var}_LIBRARY
        NAMES ${rscl_lib}
        PATHS ${RSCL_SDK_LIBRARY_DIRS}
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
      )
    endif()
    if(NOT "${RSCL_${rscl_lib_var}_LIBRARY}" MATCHES "NOTFOUND")
      list(APPEND RSCL_SDK_LINK_LIBRARIES "${RSCL_${rscl_lib_var}_LIBRARY}")
    else()
      list(APPEND RSCL_SDK_LINK_LIBRARIES "${rscl_lib}")
    endif()
  endforeach()
endif()

if("${RSCL_ONLINE_SDK_LIBRARIES}" STREQUAL "")
  set(RSCL_ONLINE_SDK_LIBRARIES "ad_rscl;ad_serde;ad_serde_capnp" CACHE STRING "Optional RSCL SDK libraries for librscl_online_backend_ad.so" FORCE)
endif()
set(RSCL_ONLINE_SDK_LINK_LIBRARIES "")
if(NOT "${RSCL_ONLINE_SDK_LIBRARIES}" STREQUAL "")
  foreach(rscl_lib ${RSCL_ONLINE_SDK_LIBRARIES})
    string(REPLACE "-" "_" rscl_lib_var "${rscl_lib}")
    string(TOUPPER "${rscl_lib_var}" rscl_lib_var)
    if(NOT "${RSCL_SDK_ROOT}" STREQUAL "" AND EXISTS "${RSCL_SDK_ROOT}/lib/lib${rscl_lib}.so")
      set(RSCL_ONLINE_${rscl_lib_var}_LIBRARY "${RSCL_SDK_ROOT}/lib/lib${rscl_lib}.so" CACHE FILEPATH "RSCL online ${rscl_lib} library" FORCE)
    else()
      find_library(RSCL_ONLINE_${rscl_lib_var}_LIBRARY
        NAMES ${rscl_lib}
        PATHS ${RSCL_SDK_LIBRARY_DIRS}
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
      )
    endif()
    if(NOT "${RSCL_ONLINE_${rscl_lib_var}_LIBRARY}" MATCHES "NOTFOUND")
      list(APPEND RSCL_ONLINE_SDK_LINK_LIBRARIES "${RSCL_ONLINE_${rscl_lib_var}_LIBRARY}")
    else()
      list(APPEND RSCL_ONLINE_SDK_LINK_LIBRARIES "${rscl_lib}")
    endif()
  endforeach()
endif()

set(RSCL_FFMPEG_INCLUDE_DIRS "")
set(RSCL_FFMPEG_LIBRARIES "")
if(RSCL_ENABLE_FFMPEG_DECODER)
  set(RSCL_FFMPEG_HINT_INCLUDE_DIRS "")
  set(RSCL_FFMPEG_HINT_LIBRARY_DIRS "")
  if(NOT "${RSCL_FFMPEG_INCLUDE_DIR}" STREQUAL "")
    list(APPEND RSCL_FFMPEG_HINT_INCLUDE_DIRS "${RSCL_FFMPEG_INCLUDE_DIR}")
  endif()
  if(NOT "${RSCL_FFMPEG_LIBRARY_DIR}" STREQUAL "")
    list(APPEND RSCL_FFMPEG_HINT_LIBRARY_DIRS "${RSCL_FFMPEG_LIBRARY_DIR}")
  endif()
  if(NOT "${RSCL_FFMPEG_ROOT}" STREQUAL "")
    list(APPEND RSCL_FFMPEG_HINT_INCLUDE_DIRS
      "${RSCL_FFMPEG_ROOT}/include"
      "${RSCL_FFMPEG_ROOT}/include/aarch64-linux-gnu"
    )
    list(APPEND RSCL_FFMPEG_HINT_LIBRARY_DIRS
      "${RSCL_FFMPEG_ROOT}/lib"
      "${RSCL_FFMPEG_ROOT}/lib64"
      "${RSCL_FFMPEG_ROOT}/lib/aarch64-linux-gnu"
    )
  endif()
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(RSCL_LIBAVCODEC QUIET libavcodec)
    pkg_check_modules(RSCL_LIBAVUTIL QUIET libavutil)
    pkg_check_modules(RSCL_LIBSWSCALE QUIET libswscale)
  endif()

  if(RSCL_LIBAVCODEC_FOUND AND RSCL_LIBAVUTIL_FOUND AND RSCL_LIBSWSCALE_FOUND)
    set(RSCL_FFMPEG_INCLUDE_DIRS
      ${RSCL_LIBAVCODEC_INCLUDE_DIRS}
      ${RSCL_LIBAVUTIL_INCLUDE_DIRS}
      ${RSCL_LIBSWSCALE_INCLUDE_DIRS}
    )
    set(RSCL_FFMPEG_LIBRARIES
      ${RSCL_LIBAVCODEC_LIBRARIES}
      ${RSCL_LIBAVUTIL_LIBRARIES}
      ${RSCL_LIBSWSCALE_LIBRARIES}
    )
    add_definitions(-DRSCL_HAVE_FFMPEG)
    include_directories(${RSCL_FFMPEG_INCLUDE_DIRS})
    message(STATUS "RSCL FFmpeg decoder enabled")
  else()
    find_path(RSCL_FFMPEG_AVCODEC_INCLUDE_DIR libavcodec/avcodec.h
      HINTS
        ${RSCL_FFMPEG_HINT_INCLUDE_DIRS}
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_path(RSCL_FFMPEG_AVUTIL_INCLUDE_DIR libavutil/imgutils.h
      HINTS
        ${RSCL_FFMPEG_HINT_INCLUDE_DIRS}
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_path(RSCL_FFMPEG_SWSCALE_INCLUDE_DIR libswscale/swscale.h
      HINTS
        ${RSCL_FFMPEG_HINT_INCLUDE_DIRS}
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_library(RSCL_FFMPEG_AVCODEC_LIBRARY avcodec
      HINTS
        ${RSCL_FFMPEG_HINT_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_library(RSCL_FFMPEG_AVUTIL_LIBRARY avutil
      HINTS
        ${RSCL_FFMPEG_HINT_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_library(RSCL_FFMPEG_SWSCALE_LIBRARY swscale
      HINTS
        ${RSCL_FFMPEG_HINT_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
      NO_CMAKE_FIND_ROOT_PATH
    )

    if(RSCL_FFMPEG_AVCODEC_INCLUDE_DIR AND RSCL_FFMPEG_AVUTIL_INCLUDE_DIR AND RSCL_FFMPEG_SWSCALE_INCLUDE_DIR AND
       RSCL_FFMPEG_AVCODEC_LIBRARY AND RSCL_FFMPEG_AVUTIL_LIBRARY AND RSCL_FFMPEG_SWSCALE_LIBRARY)
      set(RSCL_FFMPEG_INCLUDE_DIRS
        ${RSCL_FFMPEG_AVCODEC_INCLUDE_DIR}
        ${RSCL_FFMPEG_AVUTIL_INCLUDE_DIR}
        ${RSCL_FFMPEG_SWSCALE_INCLUDE_DIR}
      )
      set(RSCL_FFMPEG_LIBRARIES
        ${RSCL_FFMPEG_AVCODEC_LIBRARY}
        ${RSCL_FFMPEG_AVUTIL_LIBRARY}
        ${RSCL_FFMPEG_SWSCALE_LIBRARY}
      )
      add_definitions(-DRSCL_HAVE_FFMPEG)
      include_directories(${RSCL_FFMPEG_INCLUDE_DIRS})
      message(STATUS "RSCL FFmpeg decoder enabled via manual detection")
    else()
      message(WARNING
        "RSCL_ENABLE_FFMPEG_DECODER=ON but FFmpeg development headers/libraries were not found. "
        "The C++ RSCL adapter will build without H264/H265 decode support. "
        "Install libavcodec-dev libavutil-dev libswscale-dev, or disable RSCL_ENABLE_FFMPEG_DECODER.")
    endif()
  endif()
endif()

set(RSCL_OPENCV_LIBRARIES "")
if(RSCL_ENABLE_OPENCV_UNDISTORT)
  set(RSCL_OPENCV_HINT_INCLUDE_DIRS "")
  set(RSCL_OPENCV_HINT_LIBRARY_DIRS "")
  if(NOT "${RSCL_OPENCV_INCLUDE_DIR}" STREQUAL "")
    list(APPEND RSCL_OPENCV_HINT_INCLUDE_DIRS "${RSCL_OPENCV_INCLUDE_DIR}")
  endif()
  if(NOT "${RSCL_OPENCV_LIBRARY_DIR}" STREQUAL "")
    list(APPEND RSCL_OPENCV_HINT_LIBRARY_DIRS "${RSCL_OPENCV_LIBRARY_DIR}")
  endif()
  if(NOT "${RSCL_OPENCV_ROOT}" STREQUAL "")
    list(APPEND RSCL_OPENCV_HINT_INCLUDE_DIRS
      "${RSCL_OPENCV_ROOT}/include/opencv4"
      "${RSCL_OPENCV_ROOT}/include"
    )
    list(APPEND RSCL_OPENCV_HINT_LIBRARY_DIRS
      "${RSCL_OPENCV_ROOT}/lib"
      "${RSCL_OPENCV_ROOT}/lib64"
    )
  endif()

  find_path(RSCL_OPENCV_DETECTED_INCLUDE_DIR opencv2/calib3d.hpp
    HINTS ${RSCL_OPENCV_HINT_INCLUDE_DIRS}
    NO_CMAKE_FIND_ROOT_PATH
  )
  if("${RSCL_OPENCV_CORE_LIBRARY}" STREQUAL "")
    find_library(RSCL_OPENCV_CORE_LIBRARY opencv_core
      HINTS ${RSCL_OPENCV_HINT_LIBRARY_DIRS} NO_CMAKE_FIND_ROOT_PATH)
  endif()
  if("${RSCL_OPENCV_IMGPROC_LIBRARY}" STREQUAL "")
    find_library(RSCL_OPENCV_IMGPROC_LIBRARY opencv_imgproc
      HINTS ${RSCL_OPENCV_HINT_LIBRARY_DIRS} NO_CMAKE_FIND_ROOT_PATH)
  endif()
  if("${RSCL_OPENCV_CALIB3D_LIBRARY}" STREQUAL "")
    find_library(RSCL_OPENCV_CALIB3D_LIBRARY opencv_calib3d
      HINTS ${RSCL_OPENCV_HINT_LIBRARY_DIRS} NO_CMAKE_FIND_ROOT_PATH)
  endif()

  if(RSCL_OPENCV_DETECTED_INCLUDE_DIR AND RSCL_OPENCV_CORE_LIBRARY AND
     RSCL_OPENCV_IMGPROC_LIBRARY AND RSCL_OPENCV_CALIB3D_LIBRARY)
    add_definitions(-DRSCL_HAVE_OPENCV_UNDISTORT)
    include_directories(${RSCL_OPENCV_DETECTED_INCLUDE_DIR})
    set(RSCL_OPENCV_LIBRARIES
      ${RSCL_OPENCV_CALIB3D_LIBRARY}
      ${RSCL_OPENCV_IMGPROC_LIBRARY}
      ${RSCL_OPENCV_CORE_LIBRARY}
    )
    message(STATUS "RSCL OpenCV undistortion enabled")
  else()
    message(FATAL_ERROR
      "RSCL_ENABLE_OPENCV_UNDISTORT=ON but target OpenCV core/imgproc/calib3d headers or libraries were not found. "
      "Set RSCL_OPENCV_ROOT, or explicit RSCL_OPENCV_* paths.")
  endif()
endif()

file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)
list(APPEND CORE_FILES
  src/onnx/onnx-ml.pb.cpp
  src/onnx/onnx-operators-ml.pb.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)
bevfusion_apply_extra_link_flags(bevfusion_core)
if(NOT "${Protobuf_INCLUDE_DIRS}" STREQUAL "")
  target_include_directories(bevfusion_core BEFORE PRIVATE ${Protobuf_INCLUDE_DIRS})
endif()

target_link_libraries(bevfusion_core
  ${BEVFUSION_CUDART_LIBRARY}
  ${BEVFUSION_CUBLASLT_LIBRARY}
  ${TensorRT_NVINFER_LIBRARY}
  ${BEVFUSION_SPCONV_LIBRARY}
  ${Protobuf_LIBRARIES}
  ${TensorRT_NVINFER_PLUGIN_LIBRARY}
  ${BEVFUSION_EXTRA_LINK_LIBRARIES}
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/main.cpp
  ${VISUALIZE_FILES}
)
bevfusion_apply_extra_link_flags(bevfusion)

target_link_libraries(bevfusion
  bevfusion_core
  ${TensorRT_NVINFER_LIBRARY}
  ${BEVFUSION_SPCONV_LIBRARY}
  ${Protobuf_LIBRARIES}
  ${TensorRT_NVINFER_PLUGIN_LIBRARY}
  ${BEVFUSION_EXTRA_LINK_LIBRARIES}
)
#######################################################################


########################## custom_layernorm.so ################################
cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)
bevfusion_apply_extra_link_flags(custom_layernorm)

target_link_libraries(custom_layernorm
  ${TensorRT_NVINFER_LIBRARY}
  ${TensorRT_NVINFER_PLUGIN_LIBRARY}
  ${BEVFUSION_EXTRA_LINK_LIBRARIES}
)
#######################################################################


########################## libpybev.so ################################
if("${CMAKE_BUILD_PYTHON}" STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )
  bevfusion_apply_extra_link_flags(pybev)
  target_link_libraries(pybev
    bevfusion_core
    ${TensorRT_NVINFER_LIBRARY}
    ${BEVFUSION_SPCONV_LIBRARY}
    ${Protobuf_LIBRARIES}
    ${TensorRT_NVINFER_PLUGIN_LIBRARY}
    ${BEVFUSION_EXTRA_LINK_LIBRARIES}
    $ENV{Python_Soname}
  )
endif()
#######################################################################


########################## librscl_adapter_cpp.a #######################
if(BUILD_RSCL_CPP)
  file(GLOB_RECURSE RSCL_ADAPTER_CPP_FILES
    deploy_rscl/cpp/src/*.cpp
  )
  list(REMOVE_ITEM RSCL_ADAPTER_CPP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/src/rscl_bag_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/src/rscl_bag_reader_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/src/rscl_online_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/src/rscl_online_node_dlopen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/src/rscl_online_node_stub.cpp
  )

  cuda_add_library(rscl_adapter_cpp STATIC
    ${RSCL_ADAPTER_CPP_FILES}
  )

  target_link_libraries(rscl_adapter_cpp
    bevfusion_core
    ${BEVFUSION_CUDART_LIBRARY}
    ${TensorRT_NVINFER_LIBRARY}
    ${BEVFUSION_SPCONV_LIBRARY}
    ${Protobuf_LIBRARIES}
    ${TensorRT_NVINFER_PLUGIN_LIBRARY}
    ${RSCL_FFMPEG_LIBRARIES}
    ${RSCL_OPENCV_LIBRARIES}
    ${BEVFUSION_EXTRA_LINK_LIBRARIES}
    ${CMAKE_DL_LIBS}
  )

  if(BUILD_RSCL_BAG_RUNNER)
    if("${RSCL_BAG_BACKEND_SOURCE}" STREQUAL "")
      if(NOT "${RSCL_SDK_ROOT}" STREQUAL "" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/vehicle_rscl_bag_reader_ad_bag.cpp")
        set(RSCL_BAG_BACKEND_SOURCE deploy_rscl/cpp/vehicle_rscl_bag_reader_ad_bag.cpp)
      elseif(NOT "${RSCL_SDK_INCLUDE_DIRS}" STREQUAL "" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/vehicle_rscl_bag_reader_ad_bag.cpp")
        set(RSCL_BAG_BACKEND_SOURCE deploy_rscl/cpp/vehicle_rscl_bag_reader_ad_bag.cpp)
      else()
        set(RSCL_BAG_BACKEND_SOURCE deploy_rscl/cpp/src/rscl_bag_reader_stub.cpp)
      endif()
    endif()

    add_executable(rscl_bevfusion_bag_runner
      deploy_rscl/cpp/src/rscl_bag_runner.cpp
      ${RSCL_BAG_BACKEND_SOURCE}
    )
    bevfusion_apply_extra_link_flags(rscl_bevfusion_bag_runner)
    add_dependencies(rscl_bevfusion_bag_runner custom_layernorm)

    if(NOT "${RSCL_SDK_TARGET_INCLUDE_DIRS}" STREQUAL "")
      target_include_directories(rscl_bevfusion_bag_runner PRIVATE
        ${RSCL_SDK_TARGET_INCLUDE_DIRS}
      )
    endif()

    target_link_libraries(rscl_bevfusion_bag_runner
      rscl_adapter_cpp
      bevfusion_core
      ${BEVFUSION_CUDART_LIBRARY}
      ${TensorRT_NVINFER_LIBRARY}
      ${BEVFUSION_SPCONV_LIBRARY}
      ${Protobuf_LIBRARIES}
      ${TensorRT_NVINFER_PLUGIN_LIBRARY}
      ${RSCL_FFMPEG_LIBRARIES}
      ${RSCL_OPENCV_LIBRARIES}
      ${RSCL_SDK_LINK_LIBRARIES}
      ${BEVFUSION_EXTRA_LINK_LIBRARIES}
      ${CMAKE_DL_LIBS}
    )
    if(NOT "${RSCL_SDK_LIBRARY_DIRS}" STREQUAL "")
      set_target_properties(rscl_bevfusion_bag_runner PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${RSCL_SDK_LIBRARY_DIRS}"
        INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR};${RSCL_SDK_LIBRARY_DIRS}"
      )
    else()
      set_target_properties(rscl_bevfusion_bag_runner PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
        INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
      )
    endif()
  endif()

  if(BUILD_RSCL_ONLINE_NODE)
    if(UNIX)
      set(RSCL_ONLINE_NODE_BACKEND_LOADER deploy_rscl/cpp/src/rscl_online_node_dlopen.cpp)
    else()
      set(RSCL_ONLINE_NODE_BACKEND_LOADER deploy_rscl/cpp/src/rscl_online_node_stub.cpp)
    endif()

    add_executable(rscl_bevfusion_online_node
      deploy_rscl/cpp/src/rscl_online_node.cpp
      ${RSCL_ONLINE_NODE_BACKEND_LOADER}
    )
    bevfusion_apply_extra_link_flags(rscl_bevfusion_online_node)
    add_dependencies(rscl_bevfusion_online_node custom_layernorm)

    target_link_libraries(rscl_bevfusion_online_node
      rscl_adapter_cpp
      bevfusion_core
      ${BEVFUSION_CUDART_LIBRARY}
      ${TensorRT_NVINFER_LIBRARY}
      ${BEVFUSION_SPCONV_LIBRARY}
      ${Protobuf_LIBRARIES}
      ${TensorRT_NVINFER_PLUGIN_LIBRARY}
      ${RSCL_FFMPEG_LIBRARIES}
      ${RSCL_OPENCV_LIBRARIES}
      ${BEVFUSION_EXTRA_LINK_LIBRARIES}
      ${CMAKE_DL_LIBS}
    )

    set_target_properties(rscl_bevfusion_online_node PROPERTIES
      BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
      INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    )

    if("${RSCL_ONLINE_BACKEND_SOURCE}" STREQUAL "")
      set(RSCL_HAVE_ONLINE_SDK_HINT OFF)
      if(NOT "${RSCL_SDK_ROOT}" STREQUAL "")
        set(RSCL_HAVE_ONLINE_SDK_HINT ON)
      endif()
      if(NOT "${RSCL_SDK_INCLUDE_DIRS}" STREQUAL "")
        set(RSCL_HAVE_ONLINE_SDK_HINT ON)
      endif()
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ad_rscl/runtime.h")
        set(RSCL_HAVE_ONLINE_SDK_HINT ON)
      endif()
      if(RSCL_HAVE_ONLINE_SDK_HINT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deploy_rscl/cpp/vehicle_rscl_online_node_ad.cpp")
        set(RSCL_ONLINE_BACKEND_SOURCE deploy_rscl/cpp/vehicle_rscl_online_node_ad.cpp)
      endif()
    endif()

    if(NOT "${RSCL_ONLINE_BACKEND_SOURCE}" STREQUAL "")
      add_library(rscl_online_backend_ad SHARED
        ${RSCL_ONLINE_BACKEND_SOURCE}
      )
      bevfusion_apply_extra_link_flags(rscl_online_backend_ad)

      if(NOT "${RSCL_SDK_TARGET_INCLUDE_DIRS}" STREQUAL "")
        target_include_directories(rscl_online_backend_ad PRIVATE
          ${RSCL_SDK_TARGET_INCLUDE_DIRS}
        )
      endif()

      target_link_libraries(rscl_online_backend_ad
        ${RSCL_ONLINE_SDK_LINK_LIBRARIES}
        ${BEVFUSION_EXTRA_LINK_LIBRARIES}
      )

      if(NOT "${RSCL_SDK_LIBRARY_DIRS}" STREQUAL "")
        set_target_properties(rscl_online_backend_ad PROPERTIES
          BUILD_WITH_INSTALL_RPATH TRUE
          BUILD_RPATH "${RSCL_SDK_LIBRARY_DIRS}"
          INSTALL_RPATH "${RSCL_SDK_LIBRARY_DIRS}"
        )
      endif()
    else()
      message(WARNING "RSCL online backend source was not found. rscl_bevfusion_online_node will build, but runtime dlopen will fail until librscl_online_backend_ad.so is provided or RSCL_ONLINE_BACKEND_LIB is set.")
    endif()
  endif()
endif()
#######################################################################
