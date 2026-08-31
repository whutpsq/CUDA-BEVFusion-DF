#include "rscl_adapter/runner.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#endif

#include <iostream>
#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"

namespace rscl_adapter {
namespace {

static bool file_exists(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream f(path.c_str());
  return f.good();
}

static std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  char last = a[a.size() - 1];
  return (last == '/' || last == '\\') ? a + b : a + "/" + b;
}

static std::string default_or_override(const std::string& override_path, const std::string& root, const std::string& suffix) {
  return override_path.empty() ? join_path(root, suffix) : override_path;
}

#if defined(__linux__)
static std::string executable_dir() {
  char path[PATH_MAX] = {0};
  ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (n <= 0) return "";
  path[n] = '\0';
  std::string text(path);
  size_t slash = text.find_last_of('/');
  return slash == std::string::npos ? "" : text.substr(0, slash);
}

static bool load_shared_library(const std::string& path) {
  if (path.empty()) return false;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle) return true;
  return false;
}

static void load_custom_tensorrt_plugins() {
  const std::string exe_dir = executable_dir();
  std::vector<std::string> candidates;
  candidates.push_back("libcustom_layernorm.so");
  candidates.push_back("./libcustom_layernorm.so");
  candidates.push_back("build/libcustom_layernorm.so");
  if (!exe_dir.empty()) candidates.push_back(join_path(exe_dir, "libcustom_layernorm.so"));

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (load_shared_library(candidates[i])) return;
  }

  const char* error = dlerror();
  std::cerr << "Warning: failed to load libcustom_layernorm.so";
  if (error) std::cerr << ": " << error;
  std::cerr << std::endl;
}
#endif

static std::vector<nvtype::half> floats_to_host_half(const std::vector<float>& values) {
  static_assert(sizeof(nvtype::half) == sizeof(__half), "nvtype::half must match CUDA __half storage");
  std::vector<nvtype::half> output(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    __half h = __float2half(values[i]);
    std::memcpy(&output[i], &h, sizeof(h));
  }
  return output;
}

static bevfusion::CoreParameter make_core_parameter(const AdapterConfig& cfg) {
  const bool is_bevfusion_df = cfg.profile == "bevfusion_df";

  const std::string camera = default_or_override(cfg.camera_plan, cfg.cuda_model_root, "build/camera.backbone.plan");
  const std::string vtransform =
      default_or_override(cfg.vtransform_plan, cfg.cuda_model_root, "build/camera.vtransform.plan");
  const std::string lidar = default_or_override(cfg.lidar_onnx, cfg.cuda_model_root, "lidar.backbone.xyz.onnx");
  const std::string fuser = default_or_override(cfg.fuser_plan, cfg.cuda_model_root, "build/fuser.plan");
  const std::string head = default_or_override(cfg.head_plan, cfg.cuda_model_root, "build/head.bbox.plan");
  const std::string map = default_or_override(cfg.map_plan, cfg.cuda_model_root, "build/head.map.plan");

  if (cfg.enable_object_detection && !file_exists(head)) {
    throw std::runtime_error("Object detection is enabled but bbox plan does not exist: " + head);
  }
  if (cfg.enable_map_segmentation && !file_exists(map)) {
    throw std::runtime_error("Map segmentation is enabled but map plan does not exist: " + map);
  }

  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = cfg.image_preprocess_width > 0 ? cfg.image_preprocess_width
                                                             : (is_bevfusion_df ? 1920 : 1600);
  normalization.image_height = cfg.image_preprocess_height > 0 ? cfg.image_preprocess_height
                                                               : (is_bevfusion_df ? 1080 : 900);
  normalization.output_width = cfg.image_width;
  normalization.output_height = cfg.image_height;
  normalization.num_camera = static_cast<int>(cfg.camera_order.size());
  normalization.resize_lim = cfg.image_resize;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;
  normalization.method = bevfusion::camera::NormMethod::mean_std(cfg.image_mean, cfg.image_std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = is_bevfusion_df ? nvtype::Float3(-51.2f, -51.2f, -5.0f) : nvtype::Float3(-54.0f, -54.0f, -5.0f);
  voxelization.max_range = is_bevfusion_df ? nvtype::Float3(+51.2f, +51.2f, +3.0f) : nvtype::Float3(+54.0f, +54.0f, +3.0f);
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
  scn.precision = cfg.precision == "int8" ? bevfusion::lidar::Precision::Int8 : bevfusion::lidar::Precision::Float16;

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = is_bevfusion_df ? nvtype::Float3(-51.2f, 51.2f, 0.8f) : nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = is_bevfusion_df ? nvtype::Float3(-51.2f, 51.2f, 0.8f) : nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0f, 60.0f, 0.5f);
  geometry.image_width = cfg.image_width;
  geometry.image_height = cfg.image_height;
  geometry.feat_width = cfg.image_width / 8;
  geometry.feat_height = cfg.image_height / 8;
  geometry.num_camera = static_cast<int>(cfg.camera_order.size());
  geometry.geometry_dim = is_bevfusion_df ? nvtype::Int3(128, 128, 80) : nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = is_bevfusion_df ? nvtype::Float2{-51.2f, -51.2f} : nvtype::Float2{-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2f, -61.2f, -10.0f};
  transbbox.post_center_range_end = {61.2f, 61.2f, 10.0f};
  transbbox.voxel_size = is_bevfusion_df ? nvtype::Float2{0.2f, 0.2f} : nvtype::Float2{0.075f, 0.075f};
  transbbox.model = cfg.enable_object_detection ? head : "";
  transbbox.confidence_threshold = is_bevfusion_df ? 0.2f : 0.0f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = camera;
  param.camera_vtransform = vtransform;
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = fuser;
  param.transbbox = transbbox;
  param.mapseg.model = cfg.enable_map_segmentation ? map : "";
  param.mapseg.input = cfg.map_input_binding;
  param.mapseg.output = cfg.map_output_binding;
  return param;
}

}  // namespace

struct BevFusionRunner::Impl {
  explicit Impl(const AdapterConfig& config) : cfg(config) {
#if defined(__linux__)
    load_custom_tensorrt_plugins();
#endif
    bevfusion::CoreParameter param = make_core_parameter(cfg);
    core = bevfusion::create_core(param);
    if (!core) throw std::runtime_error("Failed to create CUDA-BEVFusion core");
    checkRuntime(cudaStreamCreate(&stream));
    if (cfg.print_model_info) core->print();
  }

  ~Impl() {
    if (stream) checkRuntime(cudaStreamDestroy(stream));
  }

  AdapterConfig cfg;
  std::shared_ptr<bevfusion::Core> core;
  cudaStream_t stream = nullptr;
};

BevFusionRunner::BevFusionRunner(const AdapterConfig& cfg) : impl_(new Impl(cfg)) {}

BevFusionRunner::~BevFusionRunner() = default;

InferenceOutput BevFusionRunner::infer(const ModelInput& input) {
  if (input.points.empty()) throw std::runtime_error("Cannot run BEVFusion with empty lidar point cloud");

  static std::atomic<bool> logged_first_infer(false);
  const bool log_first_infer = !logged_first_infer.exchange(true);
  const int num_points = static_cast<int>(input.points.size() / 5);
  if (log_first_infer) {
    std::cout << "runner_stage=infer_begin num_points=" << num_points
              << " image_values=" << input.images_chw.size() << std::endl;
  }
  std::vector<float> image_host = input.images_chw;
  std::vector<nvtype::half> point_host = floats_to_host_half(input.points);
  if (log_first_infer) std::cout << "runner_stage=host_conversion_done" << std::endl;
  nv::Tensor images = nv::Tensor::from_data_reference(
      image_host.data(), std::vector<int64_t>{1, input.num_cameras, 3, input.image_height, input.image_width},
      nv::DataType::Float32, false);
  images.to_device_(impl_->stream);
  images = images.to_half(impl_->stream);
  if (log_first_infer) std::cout << "runner_stage=image_upload_done" << std::endl;

  impl_->core->update(input.camera2lidar.data(), input.camera_intrinsics.data(), input.lidar2image.data(),
                      input.img_aug_matrix.data(), impl_->stream);
  if (log_first_infer) std::cout << "runner_stage=core_update_done" << std::endl;

  InferenceOutput output;
  if (impl_->cfg.enable_object_detection) {
    if (log_first_infer) std::cout << "runner_stage=forward_bbox_begin" << std::endl;
    std::vector<bevfusion::head::transbbox::BoundingBox> boxes =
        impl_->core->forward_no_normalize(images.ptr<nvtype::half>(), point_host.data(), num_points, impl_->stream);
    if (log_first_infer) std::cout << "runner_stage=forward_bbox_done boxes=" << boxes.size() << std::endl;
    output.detections.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
      Detection det;
      det.label = boxes[i].id;
      det.score = boxes[i].score;
      det.box[0] = boxes[i].position.x;
      det.box[1] = boxes[i].position.y;
      det.box[2] = boxes[i].position.z;
      det.box[3] = boxes[i].size.w;
      det.box[4] = boxes[i].size.l;
      det.box[5] = boxes[i].size.h;
      det.box[6] = boxes[i].z_rotation;
      det.box[7] = boxes[i].velocity.vx;
      det.box[8] = boxes[i].velocity.vy;
      output.detections.push_back(det);
    }
  }

  if (impl_->cfg.enable_map_segmentation) {
    bevfusion::head::map::MapOutput map =
        impl_->core->forward_map_no_normalize(images.ptr<nvtype::half>(), point_host.data(), num_points, impl_->stream);
    output.has_map = true;
    output.map.shape = map.shape;
    output.map.data.resize(map.data.size());
    for (size_t i = 0; i < map.data.size(); ++i) {
      output.map.data[i] = map.data[i] >= impl_->cfg.map_score_threshold ? 1 : 0;
    }
  }
  if (log_first_infer) std::cout << "runner_stage=stream_sync_begin" << std::endl;
  checkRuntime(cudaStreamSynchronize(impl_->stream));
  if (log_first_infer) std::cout << "runner_stage=stream_sync_done" << std::endl;
  return output;
}

}  // namespace rscl_adapter
