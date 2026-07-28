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

set -o pipefail
script_directory="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
repo_directory="$(cd "$script_directory/.." && pwd)"
. "$script_directory/environment.sh"
if [ "$ConfigurationStatus" != "Success" ]; then echo "Thor environment configuration failed."; exit 1; fi

trtexec="$TensorRT_Bin/trtexec"
base="$repo_directory/model/$DEBUG_MODEL"
precision="$DEBUG_PRECISION"
if [ -z "$FORCE_REBUILD" ]; then export FORCE_REBUILD=1; fi
if [ -z "$DEPLOY_VALIDATED" ]; then export DEPLOY_VALIDATED=OFF; fi
if [ ! -d "$base" ]; then echo "Model directory does not exist: $base"; exit 1; fi

trt_version="$("$trtexec" --version 2>&1 | sed -n 's/.*TensorRT v\([^ ]*\).*/\1/p' | head -n 1)"
if [ -z "$trt_version" ]; then trt_version=unknown; fi
trt_version_tag="$(printf '%s' "$trt_version" | tr -c '[:alnum:]._' '_')"
if [ -z "$ENGINE_OUTPUT_DIR" ]; then result_save_directory="$base/build_thor_trt$trt_version_tag"; else result_save_directory="$ENGINE_OUTPUT_DIR"; fi
mkdir -p "$result_save_directory"

plugin_is_external=0
if [ -n "$CUSTOM_LAYERNORM_PLUGIN" ]; then plugin="$CUSTOM_LAYERNORM_PLUGIN"; plugin_is_external=1; else plugin="$result_save_directory/libcustom_layernorm.so"; fi

validate_aarch64_file() {
    local path="$1"
    local description="$2"
    local info
    if ! command -v file >/dev/null 2>&1; then echo "Warning: cannot inspect $description architecture."; return 0; fi
    info="$(file "$path" 2>/dev/null || true)"
    echo "$description: $info"
    if ! printf '%s' "$info" | grep -Eiq 'aarch64|ARM aarch64'; then
        echo "$description is not an aarch64 ELF: $path"
        return 1
    fi
}

ensure_custom_layernorm_plugin() {
    local source="$repo_directory/src/plugins/custom_layernorm.cu"
    local nvcc="$CUDA_Bin/nvcc"
    local tmp_plugin="$plugin.tmp.$$"

    if [ "$plugin_is_external" = "1" ]; then
        if [ ! -f "$plugin" ]; then echo "CUSTOM_LAYERNORM_PLUGIN does not exist: $plugin"; exit 1; fi
        validate_aarch64_file "$plugin" "Custom LayerNorm plugin" || exit 1
        return
    fi
    if [ "$FORCE_REBUILD" != "1" ] && [ -f "$plugin" ]; then
        echo "Reusing plugin because FORCE_REBUILD=$FORCE_REBUILD: $plugin"
        validate_aarch64_file "$plugin" "Custom LayerNorm plugin" || exit 1
        return
    fi
    if [ ! -x "$nvcc" ]; then
        if [ -f "$plugin" ]; then
            echo "nvcc is unavailable; using supplied plugin: $plugin"
            validate_aarch64_file "$plugin" "Custom LayerNorm plugin" || exit 1
            return
        fi
        echo "Cannot build head.bbox.plan because nvcc is missing: $nvcc"
        echo "Install the native toolkit or set CUSTOM_LAYERNORM_PLUGIN."
        exit 1
    fi
    if [ ! -f "$TensorRT_Inc/NvInfer.h" ]; then echo "Missing $TensorRT_Inc/NvInfer.h."; exit 1; fi
    if [ -z "$CUDASM" ]; then echo "CUDASM is empty."; exit 1; fi

    echo "Building native aarch64 Custom LayerNorm plugin for CUDA SM $CUDASM."
    if ! "$nvcc" -std=c++14 -shared -Xcompiler=-fPIC \
        -gencode "arch=compute_$CUDASM,code=sm_$CUDASM" \
        -gencode "arch=compute_$CUDASM,code=compute_$CUDASM" \
        -I"$TensorRT_Inc" -L"$TensorRT_Lib" "$source" \
        -lnvinfer -lnvinfer_plugin -o "$tmp_plugin"; then
        echo "Failed to build $plugin."
        exit 1
    fi
    mv -f "$tmp_plugin" "$plugin"
    validate_aarch64_file "$plugin" "Custom LayerNorm plugin" || exit 1
}

make_io_format_flag() {
    local option="$1"
    local count="$2"
    local formats=""
    local i
    for ((i = 0; i < count; i++)); do
        if [ -n "$formats" ]; then formats="$formats,"; fi
        formats="$formats"fp16:chw
    done
    printf '%s=%s' "$option" "$formats"
}

validate_trt_model() {
    local name="$1"
    local plan="$2"
    local plugin_path="$3"
    local validation_log="$result_save_directory/$name.load.log"
    local command=("$trtexec" "--loadEngine=$plan" --skipInference --verbose)
    if [ -n "$plugin_path" ]; then command+=("--plugins=$plugin_path"); fi
    echo "Load-testing $name.plan on the target TensorRT runtime."
    if ! "${command[@]}" >"$validation_log" 2>&1; then
        echo "Failed to load-test $plan. See $validation_log"
        exit 1
    fi
}

compile_trt_model() {
    local name="$1"
    local precision_mode="$2"
    local inputs="$3"
    local outputs="$4"
    local plugin_path="$5"
    local onnx="$base/$name.onnx"
    local plan="$result_save_directory/$name.plan"
    local tmp_plan="$plan.tmp.$$"
    local build_log="$result_save_directory/$name.log"
    local layer_json="$result_save_directory/$name.json"
    local input_flag="$(make_io_format_flag --inputIOFormats "$inputs")"
    local output_flag="$(make_io_format_flag --outputIOFormats "$outputs")"
    local command

    if [ ! -f "$onnx" ]; then echo "Required ONNX model does not exist: $onnx"; exit 1; fi
    if [ "$FORCE_REBUILD" != "1" ] && [ -f "$plan" ]; then
        echo "Reusing plan because FORCE_REBUILD=$FORCE_REBUILD: $plan"
        validate_trt_model "$name" "$plan" "$plugin_path"
        return
    fi
    command=(
        "$trtexec" "--onnx=$onnx" --fp16 "$input_flag" "$output_flag"
        "--saveEngine=$tmp_plan" --memPoolSize=workspace:2048
        --verbose --dumpLayerInfo --dumpProfile --separateProfileRun
        --profilingVerbosity=detailed "--exportLayerInfo=$layer_json"
    )
    if [ "$precision_mode" = "dynamic" ] && [ "$precision" = "int8" ]; then command+=(--int8); fi
    if [ -n "$plugin_path" ]; then command+=("--plugins=$plugin_path"); fi

    echo "Building $name.plan with native target trtexec."
    if ! "${command[@]}" >"$build_log" 2>&1; then
        echo "Failed to build $name.plan. See $build_log"
        exit 1
    fi
    if [ ! -s "$tmp_plan" ]; then echo "trtexec did not create a non-empty engine: $tmp_plan"; exit 1; fi
    mv -f "$tmp_plan" "$plan"
    validate_trt_model "$name" "$plan" "$plugin_path"
}

deploy_validated_engines() {
    local production_directory="$base/build"
    local name
    if [ "$DEPLOY_VALIDATED" != "ON" ]; then return; fi
    mkdir -p "$production_directory"
    for name in camera.backbone camera.vtransform fuser head.bbox head.map; do
        if [ -f "$result_save_directory/$name.plan" ]; then cp -f "$result_save_directory/$name.plan" "$production_directory/$name.plan"; fi
    done
    if [ -f "$plugin" ]; then
        cp -f "$plugin" "$production_directory/libcustom_layernorm.so"
        mkdir -p "$BuildDirectory"
        cp -f "$plugin" "$BuildDirectory/libcustom_layernorm.so"
    fi
    echo "Deployed validated Thor engines and plugin to: $production_directory"
}

validate_aarch64_file "$trtexec" "TensorRT trtexec" || exit 1
compile_trt_model camera.backbone dynamic 2 2 ""
compile_trt_model fuser dynamic 2 1 ""
compile_trt_model camera.vtransform fp16 1 1 ""
if [ -f "$base/head.bbox.onnx" ]; then
    ensure_custom_layernorm_plugin
    compile_trt_model head.bbox fp16 1 6 "$plugin"
else
    echo "Optional $base/head.bbox.onnx not found. Skipping it."
fi
if [ -f "$base/head.map.onnx" ]; then compile_trt_model head.map fp16 1 1 ""; fi

echo "lidar.backbone.xyz.onnx is a 3DSparseConvolution model; no TensorRT plan is generated for it."
echo "Complete LiDAR inference requires a matching aarch64 libspconv.so."
deploy_validated_engines
echo "All generated TensorRT plans were load-tested successfully."
echo "Validated output directory: $result_save_directory"
if [ "$DEPLOY_VALIDATED" != "ON" ]; then echo "model/build was not modified. Re-run with DEPLOY_VALIDATED=ON to deploy."; fi
