/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_fp16.h>

#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-map.hpp"

namespace bevfusion {
namespace head {
namespace map {

static __global__ void half_to_float_kernel(unsigned int numel, const half* input, float* output) {
  int idx = cuda_linear_index;
  if (idx >= numel) return;
  output[idx] = __half2float(input[idx]);
}

class MapSegImplement : public MapSeg {
 public:
  virtual ~MapSegImplement() {
    if (output_device_) checkRuntime(cudaFree(output_device_));
    if (output_device_float_) checkRuntime(cudaFree(output_device_float_));
    if (output_host_float_) checkRuntime(cudaFreeHost(output_host_float_));
  }

  bool init(const MapParameter& param) {
    param_ = param;
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    output_shape_ = engine_->static_dims(param_.output.c_str());
    output_dtype_ = engine_->dtype(param_.output.c_str());
    Asserts(
        output_dtype_ == TensorRT::DType::HALF || output_dtype_ == TensorRT::DType::FLOAT,
        "Map head output must be float16 or float32.");

    output_numel_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int>());
    const size_t output_bytes = output_numel_ * (output_dtype_ == TensorRT::DType::HALF ? sizeof(half) : sizeof(float));
    checkRuntime(cudaMalloc(&output_device_, output_bytes));
    checkRuntime(cudaMalloc(&output_device_float_, output_numel_ * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host_float_, output_numel_ * sizeof(float)));
    return true;
  }

  virtual void print() override { engine_->print("MapSeg"); }

  virtual MapOutput forward(const nvtype::half* transfusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward(
        std::unordered_map<std::string, const void*>{
            {param_.input, transfusion_feature},
            {param_.output, output_device_},
        },
        _stream);

    if (output_dtype_ == TensorRT::DType::HALF) {
      cuda_linear_launch(half_to_float_kernel, _stream, output_numel_, reinterpret_cast<const half*>(output_device_),
                         output_device_float_);
      checkRuntime(cudaMemcpyAsync(output_host_float_, output_device_float_, output_numel_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, _stream));
    } else {
      checkRuntime(cudaMemcpyAsync(output_host_float_, output_device_, output_numel_ * sizeof(float), cudaMemcpyDeviceToHost,
                                   _stream));
    }
    checkRuntime(cudaStreamSynchronize(_stream));

    MapOutput output;
    output.shape = output_shape_;
    output.data.assign(output_host_float_, output_host_float_ + output_numel_);
    return output;
  }

 private:
  MapParameter param_;
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<int> output_shape_;
  TensorRT::DType output_dtype_ = TensorRT::DType::HALF;
  int output_numel_ = 0;
  void* output_device_ = nullptr;
  float* output_device_float_ = nullptr;
  float* output_host_float_ = nullptr;
};

std::shared_ptr<MapSeg> create_mapseg(const MapParameter& param) {
  if (param.model.empty()) return nullptr;
  std::shared_ptr<MapSegImplement> instance(new MapSegImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace map
};  // namespace head
};  // namespace bevfusion
