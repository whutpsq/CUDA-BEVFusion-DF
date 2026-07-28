#!/bin/bash
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


export ConfigurationStatus=Failed
tool_directory="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
repo_directory="$(cd "$tool_directory/.." && pwd)"

first_dir() {
    local path
    for path in "$@"; do
        if [ -n "$path" ] && [ -d "$path" ]; then printf '%s' "$path"; return 0; fi
    done
    return 1
}
first_trtexec_dir() {
    local path
    for path in "$@"; do
        if [ -n "$path" ] && [ -x "$path/trtexec" ]; then printf '%s' "$path"; return 0; fi
    done
    return 1
}

target_arch="$(uname -m 2>/dev/null || true)"
if [ -z "$THOR_STRICT_ARCH" ]; then export THOR_STRICT_ARCH=ON; fi
if [ "$THOR_STRICT_ARCH" = "ON" ] && [ "$target_arch" != "aarch64" ]; then
    echo "Expected Thor/aarch64, but uname -m reports: $target_arch."
    echo "Generate plans on Thor. THOR_STRICT_ARCH=OFF is only for static checks."
    return 1 2>/dev/null || exit 1
fi

if [ -z "$CUDA_HOME" ]; then CUDA_HOME="$(first_dir /usr/local/thor/cuda-12.8 /usr/local/cuda || true)"; fi
export CUDA_HOME
if [ -z "$CUDA_Bin" ]; then export CUDA_Bin="$CUDA_HOME/bin"; fi
if [ -z "$CUDA_Inc" ]; then export CUDA_Inc="$CUDA_HOME/include"; fi
if [ -z "$CUDA_Lib" ]; then
    CUDA_Lib="$(first_dir "$CUDA_HOME/targets/aarch64-linux/lib" "$CUDA_HOME/lib64" || true)"
fi
export CUDA_Bin CUDA_Inc CUDA_Lib

if [ -z "$TENSORRT_ROOT" ]; then export TENSORRT_ROOT=/usr/local/thor/aarch64-linux-gnu; fi
if [ -z "$TensorRT_Bin" ]; then
    path_trtexec="$(command -v trtexec 2>/dev/null || true)"
    path_trtexec_dir=""
    if [ -n "$path_trtexec" ]; then path_trtexec_dir="$(dirname "$path_trtexec")"; fi
    TensorRT_Bin="$(first_trtexec_dir "$TENSORRT_ROOT/bin" /usr/src/tensorrt/bin "$path_trtexec_dir" || true)"
fi
if [ -z "$TensorRT_Inc" ]; then
    TensorRT_Inc="$(first_dir "$TENSORRT_ROOT/include" /usr/include/aarch64-linux-gnu /usr/include || true)"
fi
if [ -z "$TensorRT_Lib" ]; then
    TensorRT_Lib="$(first_dir "$TENSORRT_ROOT/lib" /usr/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/tegra || true)"
fi
export TensorRT_Bin TensorRT_Inc TensorRT_Lib

if [ -z "$CUDNN_Lib" ]; then
    CUDNN_Lib="$(first_dir "$TENSORRT_ROOT/lib" /usr/lib/aarch64-linux-gnu "$CUDA_HOME/targets/aarch64-linux/lib" "$CUDA_HOME/lib64" || true)"
fi
export CUDNN_Lib
if [ -z "$BEVFUSION_SPCONV_ROOT" ]; then
    export BEVFUSION_SPCONV_ROOT="$repo_directory/libraries/3DSparseConvolution/libspconv"
fi
export SPCONV_CUDA_VERSION="$SPCONV_CUDA_VERSION"

if [ -z "$DEBUG_MODEL" ]; then export DEBUG_MODEL=bevfusion_df_detect_0723; fi
if [ -z "$DEBUG_PROFILE" ]; then export DEBUG_PROFILE=bevfusion_df_detect_0723; fi
if [ -z "$DEBUG_PRECISION" ]; then export DEBUG_PRECISION=fp16; fi
if [ -z "$DEBUG_DATA" ]; then export DEBUG_DATA=dump_df/00000; fi
if [ -z "$USE_Python" ]; then export USE_Python=OFF; fi
if [ -z "$BuildDirectory" ]; then export BuildDirectory="$repo_directory/build"; fi

if [ ! -x "$TensorRT_Bin/trtexec" ]; then
    echo "Cannot find executable $TensorRT_Bin/trtexec. Set TensorRT_Bin to the aarch64 trtexec directory."
    return 1 2>/dev/null || exit 1
fi
case "$TensorRT_Lib" in
    *stubs*)
        echo "TensorRT_Lib points to stubs: $TensorRT_Lib"
        echo "Use the real Thor TensorRT runtime library directory."
        return 1 2>/dev/null || exit 1
        ;;
esac
if ! compgen -G "$TensorRT_Lib/libnvinfer.so*" >/dev/null; then
    echo "Cannot find libnvinfer.so under TensorRT_Lib=$TensorRT_Lib."
    return 1 2>/dev/null || exit 1
fi

export PATH="$TensorRT_Bin:$CUDA_Bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH"
export PYTHONPATH="$BuildDirectory:$PYTHONPATH"

if command -v file >/dev/null 2>&1; then
    trtexec_info="$(file "$TensorRT_Bin/trtexec" 2>/dev/null || true)"
    if [ "$THOR_STRICT_ARCH" = "ON" ] &&
       ! printf '%s' "$trtexec_info" | grep -Eiq 'aarch64|ARM aarch64'; then
        echo "trtexec is not an aarch64 executable: $trtexec_info"
        return 1 2>/dev/null || exit 1
    fi
fi
if ! trtexec_version="$("$TensorRT_Bin/trtexec" --version 2>&1)"; then
    echo "Failed to execute $TensorRT_Bin/trtexec --version:"
    echo "$trtexec_version"
    return 1 2>/dev/null || exit 1
fi

if [ "$USE_Python" = "ON" ]; then
    export Python_Inc="$(python3 -c "import sysconfig;print(sysconfig.get_path('include'))")"
    export Python_Lib="$(python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")"
    export Python_Soname="$(python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))")"
fi
if [ ! -f "$TensorRT_Inc/NvInfer.h" ]; then
    echo "Warning: $TensorRT_Inc/NvInfer.h is missing; the custom plugin cannot be compiled."
fi
if [ ! -x "$CUDA_Bin/nvcc" ]; then
    echo "Warning: $CUDA_Bin/nvcc is missing; head.bbox needs a compatible aarch64 plugin."
fi

export CUDASM="$CUDASM"
if [ -z "$CUDASM" ] && command -v python3 >/dev/null 2>&1 && [ -f "$tool_directory/cudasm.sh" ]; then
    echo "Querying the target GPU compute capability."
    . "$tool_directory/cudasm.sh"
    export CUDASM="$cudasm"
fi
if [ -z "$CUDASM" ]; then
    echo "Warning: CUDA SM was not queried. This is fine when using trtexec with a prebuilt aarch64 plugin."
    echo "CUDASM and nvcc are required only when compiling libcustom_layernorm.so on this machine."
fi

if [ -n "$SPCONV_CUDA_VERSION" ]; then
    spconv_library="$BEVFUSION_SPCONV_ROOT/lib/aarch64_cuda$SPCONV_CUDA_VERSION/libspconv.so"
    if [ ! -f "$spconv_library" ]; then
        echo "Warning: $spconv_library does not exist."
        echo "Plan generation can continue, but LiDAR inference needs this aarch64 library."
    fi
else
    echo "Warning: SPCONV_CUDA_VERSION is not configured."
    echo "Set it to an existing aarch64_cuda<version> directory before full inference."
fi

echo "=========================================================="
echo "|| TARGET ARCH: $target_arch"
echo "|| CUDA SM: $CUDASM"
echo "|| MODEL: $DEBUG_MODEL"
echo "|| PROFILE: $DEBUG_PROFILE"
echo "|| PRECISION: $DEBUG_PRECISION"
echo "|| TensorRT Bin: $TensorRT_Bin"
echo "|| TensorRT Lib: $TensorRT_Lib"
echo "|| CUDA: $CUDA_HOME"
echo "|| CUDA Lib: $CUDA_Lib"
echo "|| CUDNN Lib: $CUDNN_Lib"
echo "|| SPCONV CUDA: $SPCONV_CUDA_VERSION"
echo "=========================================================="

export ConfigurationStatus=Success
echo "Thor environment configuration done."
unset -f first_dir first_trtexec_dir
unset tool_directory repo_directory target_arch path_trtexec path_trtexec_dir
unset trtexec_info trtexec_version spconv_library
