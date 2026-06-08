/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include "bevfusion/bevfusion.hpp"
#include "common/tensor.hpp"
#include "common/tensorrt.hpp"
#include "common/timer.hpp"
#include <dlfcn.h>
#include <dlpack/dlpack.h>

using namespace std;
namespace py = pybind11;

nv::Tensor convert_to(py::array& array) {
  vector<size_t> tshape(array.shape(), array.shape() + array.ndim());
  vector<int64_t> shape(tshape.size());
  std::transform(tshape.begin(), tshape.end(), shape.begin(), [](size_t v) { return v; });

  nv::DataType nvdtype = nv::DataType::None;
  if (array.dtype() == py::dtype::of<float>())
    nvdtype = nv::DataType::Float32;
  else if (array.dtype() == py::dtype::of<unsigned char>())
    nvdtype = nv::DataType::UInt8;
  else if (array.dtype() == py::dtype("half"))
    nvdtype = nv::DataType::Float16;
  else if (array.dtype() == py::dtype::of<int>())
    nvdtype = nv::DataType::Int32;
  else {
    Assertf(false, "Unsupported data type: %s", std::string(py::str(array.dtype())).c_str());
  }
  return nv::Tensor::from_data_reference((void*)array.data(), shape, nvdtype, false);
}

static int infer_dlpack_point_count(const DLManagedTensor* tensor, int point_dim = 5) {
  if (tensor == nullptr || tensor->dl_tensor.shape == nullptr || tensor->dl_tensor.ndim <= 0) {
    return 0;
  }

  const auto& shape = tensor->dl_tensor.shape;
  const int ndim = tensor->dl_tensor.ndim;

  // Common point-cloud layouts are [N, C] or [1, N, C]. Fall back to the first
  // dimension when the layout is already flattened.
  if (ndim == 1) {
    if (shape[0] % point_dim != 0) {
      return static_cast<int>(shape[0]);
    }
    return static_cast<int>(shape[0] / point_dim);
  }
  if (ndim == 2) {
    return static_cast<int>(shape[0]);
  }
  if (shape[0] == 1) {
    size_t count = 1;
    for (int i = 1; i < ndim - 1; ++i) {
      count *= static_cast<size_t>(shape[i]);
    }
    return static_cast<int>(count);
  }
  return static_cast<int>(shape[0]);
}

class BEVFusion {
 public:
  std::shared_ptr<bevfusion::Core> core_;
  cudaStream_t stream_ = nullptr;

  static std::shared_ptr<BEVFusion> load_instance(string camera, string vtransform, string lidar, string fuser, string headbbox,
                                                  string precision, string profile, string mapseg, string map_input,
                                                  string map_output) {
    std::shared_ptr<BEVFusion> instance(new BEVFusion());
    if (!instance->load(camera, vtransform, lidar, fuser, headbbox, precision, profile, mapseg, map_input, map_output)) {
      instance.reset();
    }
    return instance;
  }

  virtual ~BEVFusion() {
    if (stream_) checkRuntime(cudaStreamDestroy(stream_));
  }

  bool load(string camera, string vtransform, string lidar, string fuser, string headbbox, string precision, string profile,
            string mapseg, string map_input, string map_output) {
    bool is_bevfusion_df = profile == "bevfusion_df";

    bevfusion::camera::NormalizationParameter normalization;
    normalization.image_width = is_bevfusion_df ? 3840 : 1600;
    normalization.image_height = is_bevfusion_df ? 2160 : 900;
    normalization.output_width = is_bevfusion_df ? 352 : 704;
    normalization.output_height = is_bevfusion_df ? 128 : 256;
    normalization.num_camera = 6;
    normalization.resize_lim = 0.48f;
    normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

    bevfusion::lidar::VoxelizationParameter voxelization;
    voxelization.min_range = is_bevfusion_df ? nvtype::Float3(-51.2f, -51.2f, -5.0f) : nvtype::Float3(-54.0f, -54.0f, -5.0);
    voxelization.max_range = is_bevfusion_df ? nvtype::Float3(+51.2f, +51.2f, +3.0f) : nvtype::Float3(+54.0f, +54.0f, +3.0);
    voxelization.voxel_size = is_bevfusion_df ? nvtype::Float3(0.2f, 0.2f, 0.2f) : nvtype::Float3(0.075f, 0.075f, 0.2f);
    voxelization.grid_size =
        voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
    voxelization.max_points_per_voxel = 10;
    voxelization.max_points = 300000;
    voxelization.max_voxels = 160000;
    voxelization.num_feature = 5;

    bevfusion::lidar::SCNParameter scn;
    scn.voxelization = voxelization;
    scn.model = lidar;
    scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

    if (precision == "int8") {
      scn.precision = bevfusion::lidar::Precision::Int8;
    } else {
      scn.precision = bevfusion::lidar::Precision::Float16;
    }

    bevfusion::camera::GeometryParameter geometry;
    geometry.xbound = is_bevfusion_df ? nvtype::Float3(-51.2f, 51.2f, 0.8f) : nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.ybound = is_bevfusion_df ? nvtype::Float3(-51.2f, 51.2f, 0.8f) : nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
    geometry.image_width = is_bevfusion_df ? 352 : 704;
    geometry.image_height = is_bevfusion_df ? 128 : 256;
    geometry.feat_width = is_bevfusion_df ? 44 : 88;
    geometry.feat_height = is_bevfusion_df ? 16 : 32;
    geometry.num_camera = 6;
    geometry.geometry_dim = is_bevfusion_df ? nvtype::Int3(128, 128, 80) : nvtype::Int3(360, 360, 80);

    bevfusion::head::transbbox::TransBBoxParameter transbbox;
    transbbox.out_size_factor = 8;
    transbbox.pc_range = is_bevfusion_df ? nvtype::Float2{-51.2f, -51.2f} : nvtype::Float2{-54.0f, -54.0f};
    transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
    transbbox.post_center_range_end = {61.2, 61.2, 10.0};
    transbbox.voxel_size = is_bevfusion_df ? nvtype::Float2{0.2f, 0.2f} : nvtype::Float2{0.075f, 0.075f};
    transbbox.model = headbbox;
    transbbox.confidence_threshold = is_bevfusion_df ? 0.2f : 0.0f;
    transbbox.sorted_bboxes = true;

    bevfusion::CoreParameter param;
    param.camera_model = camera;
    param.normalize = normalization;
    param.lidar_scn = scn;
    param.geometry = geometry;
    param.transfusion = fuser;
    param.transbbox = transbbox;
    param.camera_vtransform = vtransform;
    param.mapseg.model = mapseg;
    param.mapseg.input = map_input;
    param.mapseg.output = map_output;
    core_ = bevfusion::create_core(param);
    if (core_ == nullptr) return false;

    checkRuntime(cudaStreamCreate(&stream_));
    return true;
  }

  void print() { core_->print(); }

  void update(py::array camera2lidar, py::array camera_intrinsics, py::array lidar2image, py::array img_aug_matrix) {
    auto t_lidar2image = convert_to(lidar2image);
    auto t_img_aug_matrix = convert_to(img_aug_matrix);
    auto t_camera2lidar = convert_to(camera2lidar);
    auto t_camera_intrinsics = convert_to(camera_intrinsics);
    core_->update(t_camera2lidar.ptr<float>(), t_camera_intrinsics.ptr<float>(), t_lidar2image.ptr<float>(),
                  t_img_aug_matrix.ptr<float>(), stream_);
  }

  py::array forward_with_normalization(py::array images, py::array points) {
    auto t_points = convert_to(points);
    auto t_images = convert_to(images);

    int64_t volumn = std::accumulate(t_images.shape.begin() + 2, t_images.shape.end(), 1, std::multiplies<int64_t>());
    std::vector<unsigned char*> image_pointers(t_images.size(1));
    for (size_t i = 0; i < image_pointers.size(); ++i) image_pointers[i] = t_images.ptr<unsigned char>() + i * volumn;

    auto bboxes =
        core_->forward((const unsigned char**)image_pointers.data(), t_points.ptr<nvtype::half>(), t_points.size(0), stream_);

    nv::Tensor output(std::vector<int>{static_cast<int>(bboxes.size()), 11}, nv::DataType::Float32, false);
    for (size_t i = 0; i < bboxes.size(); ++i) {
      auto& box = bboxes[i];
      float* row = output.ptr<float>() + output.size(1) * i;
      memcpy(row + 0, &box.position, sizeof(box.position));
      memcpy(row + 3, &box.size, sizeof(box.size));
      row[6] = box.z_rotation;
      memcpy(row + 7, &box.velocity, sizeof(box.velocity));
      row[9] = box.id;
      row[10] = box.score;
    }
    return py::array(py::dtype("float32"), output.shape, output.ptr());
  }
  
  py::array forward_without_normalization_dlpack(const py::capsule& images, const py::capsule& points) {

    DLManagedTensor* dl_managed_images = static_cast<DLManagedTensor*>(images.get_pointer());
    DLManagedTensor* dl_managed_points = static_cast<DLManagedTensor*>(points.get_pointer());

    void* memory_ptr_images = dl_managed_images->dl_tensor.data;
    void* memory_ptr_points = dl_managed_points->dl_tensor.data;

    nvtype::half* camera_images = static_cast<nvtype::half*>(memory_ptr_images);
    nvtype::half* lidar_points = static_cast<nvtype::half*>(memory_ptr_points);
    int num_points = infer_dlpack_point_count(dl_managed_points);
    Assertf(num_points > 0, "Failed to infer point count from DLPack input");

    auto bboxes = core_->forward_no_normalize(camera_images, lidar_points, num_points, stream_);
    nv::Tensor output(std::vector<int>{static_cast<int>(bboxes.size()), 11}, nv::DataType::Float32, false);
    for (size_t i = 0; i < bboxes.size(); ++i) {
      auto& box = bboxes[i];
      float* row = output.ptr<float>() + output.size(1) * i;
      memcpy(row + 0, &box.position, sizeof(box.position));
      memcpy(row + 3, &box.size, sizeof(box.size));
      row[6] = box.z_rotation;
      memcpy(row + 7, &box.velocity, sizeof(box.velocity));
      row[9] = box.id;
      row[10] = box.score;
    }
    return py::array(py::dtype("float32"), output.shape, output.ptr());
  }

  py::array forward_without_normalization(py::array images, py::array points) {
    auto t_points = convert_to(points);
    auto t_images = convert_to(images);
    t_images.to_device_();
    t_images = t_images.to_half();

    auto bboxes =
        core_->forward_no_normalize(t_images.ptr<nvtype::half>(), t_points.ptr<nvtype::half>(), t_points.size(0), stream_);

    nv::Tensor output(std::vector<int>{static_cast<int>(bboxes.size()), 11}, nv::DataType::Float32, false);
    for (size_t i = 0; i < bboxes.size(); ++i) {
      auto& box = bboxes[i];
      float* row = output.ptr<float>() + output.size(1) * i;
      memcpy(row + 0, &box.position, sizeof(box.position));
      memcpy(row + 3, &box.size, sizeof(box.size));
      row[6] = box.z_rotation;
      memcpy(row + 7, &box.velocity, sizeof(box.velocity));
      row[9] = box.id;
      row[10] = box.score;
    }
    return py::array(py::dtype("float32"), output.shape, output.ptr());
  }

  py::array forward(py::object images, py::object points, bool with_normalization, bool with_dlpack){
    if(with_normalization){
      return this->forward_with_normalization(images, points);
    }else{
      if(with_dlpack){
        return this->forward_without_normalization_dlpack(images, points);
      }else{
        return this->forward_without_normalization(images, points);
      }
    }
  }

  py::array forward_map_without_normalization(py::array images, py::array points) {
    auto t_points = convert_to(points);
    auto t_images = convert_to(images);
    t_images.to_device_();
    t_images = t_images.to_half();

    auto output = core_->forward_map_no_normalize(t_images.ptr<nvtype::half>(), t_points.ptr<nvtype::half>(), t_points.size(0), stream_);
    std::vector<ssize_t> shape(output.shape.begin(), output.shape.end());
    py::array_t<float> array(shape);
    memcpy(array.mutable_data(), output.data.data(), output.data.size() * sizeof(float));
    return array;
  }
};

PYBIND11_MODULE(libpybev, m) {
  py::class_<BEVFusion, shared_ptr<BEVFusion>>(m, "BEVFusion")
      .def("forward", &BEVFusion::forward, py::arg("images"), py::arg("points"), py::arg("with_normalization")=true, py::arg("with_dlpack")=false)
      .def("forward_map", &BEVFusion::forward_map_without_normalization, py::arg("images"), py::arg("points"))
      .def("print", &BEVFusion::print)
      .def("update", &BEVFusion::update);

  m.def("load_bevfusion", BEVFusion::load_instance, py::arg("camera"), py::arg("vtransform"), py::arg("lidar"),
        py::arg("fuser"), py::arg("headbbox"), py::arg("precision"), py::arg("profile") = "default", py::arg("mapseg") = "",
        py::arg("map_input") = "middle", py::arg("map_output") = "map");
  dlopen("libcustom_layernorm.so", RTLD_NOW);
};
