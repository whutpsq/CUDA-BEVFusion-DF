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
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

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

set(spconv_include ../libraries/3DSparseConvolution/libspconv/include)
set(spconv_lib ../libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
message(STATUS "spconv_lib = ${spconv_lib}")

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
  -gencode arch=compute_86,code=compute_86
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
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
  ${TensorRT_LIBRARY_DIR}
  ${spconv_lib}
  build
  $ENV{Python_Lib}
)

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
    find_library(RSCL_${rscl_lib_var}_LIBRARY
      NAMES ${rscl_lib}
      PATHS ${RSCL_SDK_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    if(NOT "${RSCL_${rscl_lib_var}_LIBRARY}" STREQUAL "RSCL_${rscl_lib_var}_LIBRARY-NOTFOUND")
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
    find_library(RSCL_ONLINE_${rscl_lib_var}_LIBRARY
      NAMES ${rscl_lib}
      PATHS ${RSCL_SDK_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    if(NOT "${RSCL_ONLINE_${rscl_lib_var}_LIBRARY}" STREQUAL "RSCL_ONLINE_${rscl_lib_var}_LIBRARY-NOTFOUND")
      list(APPEND RSCL_ONLINE_SDK_LINK_LIBRARIES "${RSCL_ONLINE_${rscl_lib_var}_LIBRARY}")
    else()
      list(APPEND RSCL_ONLINE_SDK_LINK_LIBRARIES "${rscl_lib}")
    endif()
  endforeach()
endif()

set(RSCL_FFMPEG_INCLUDE_DIRS "")
set(RSCL_FFMPEG_LIBRARIES "")
if(RSCL_ENABLE_FFMPEG_DECODER)
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
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
    )
    find_path(RSCL_FFMPEG_AVUTIL_INCLUDE_DIR libavutil/imgutils.h
      HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
    )
    find_path(RSCL_FFMPEG_SWSCALE_INCLUDE_DIR libswscale/swscale.h
      HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
    )
    find_library(RSCL_FFMPEG_AVCODEC_LIBRARY avcodec
      HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    )
    find_library(RSCL_FFMPEG_AVUTIL_LIBRARY avutil
      HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    )
    find_library(RSCL_FFMPEG_SWSCALE_LIBRARY swscale
      HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
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

file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
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

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


########################## custom_layernorm.so ################################
cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)
#######################################################################


########################## libpybev.so ################################
if("${CMAKE_BUILD_PYTHON}" STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )
  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
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
    libcudart.so
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    ${RSCL_FFMPEG_LIBRARIES}
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
    add_dependencies(rscl_bevfusion_bag_runner custom_layernorm)

    if(NOT "${RSCL_SDK_TARGET_INCLUDE_DIRS}" STREQUAL "")
      target_include_directories(rscl_bevfusion_bag_runner PRIVATE
        ${RSCL_SDK_TARGET_INCLUDE_DIRS}
      )
    endif()

    target_link_libraries(rscl_bevfusion_bag_runner
      rscl_adapter_cpp
      bevfusion_core
      libcudart.so
      libnvinfer.so
      libspconv.so
      ${Protobuf_LIBRARIES}
      libnvinfer_plugin.so
      ${RSCL_FFMPEG_LIBRARIES}
      ${RSCL_SDK_LINK_LIBRARIES}
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
    add_dependencies(rscl_bevfusion_online_node custom_layernorm)

    target_link_libraries(rscl_bevfusion_online_node
      rscl_adapter_cpp
      bevfusion_core
      libcudart.so
      libnvinfer.so
      libspconv.so
      ${Protobuf_LIBRARIES}
      libnvinfer_plugin.so
      ${RSCL_FFMPEG_LIBRARIES}
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

      if(NOT "${RSCL_SDK_TARGET_INCLUDE_DIRS}" STREQUAL "")
        target_include_directories(rscl_online_backend_ad PRIVATE
          ${RSCL_SDK_TARGET_INCLUDE_DIRS}
        )
      endif()

      target_link_libraries(rscl_online_backend_ad
        ${RSCL_ONLINE_SDK_LINK_LIBRARIES}
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
