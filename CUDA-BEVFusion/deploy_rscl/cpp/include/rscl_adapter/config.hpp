#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rscl_adapter {

struct AdapterConfig {
  std::string config_path;
  std::string checkpoint_path;
  std::string output_topic = "/perception/bevfusion/objects";
  std::string lidar_topic = "/perception/lidar/preproc_points_cloud";
  std::vector<std::string> camera_topics = {
      "/sensor/camera/center_camera_fov120/encode",
      "/sensor/camera/left_front_camera/encode",
      "/sensor/camera/right_front_camera/encode",
      "/sensor/camera/rear_camera/encode",
      "/sensor/camera/left_rear_camera/encode",
      "/sensor/camera/right_rear_camera/encode",
  };
  std::vector<std::string> camera_order = {"front", "front_left", "front_right", "rear", "rear_left", "rear_right"};
  std::string node_name = "bevfusion_rscl";
  std::string module_name = "bevfusion_rscl";
  std::string device = "cuda:0";
  std::string model = "bevfusion_df";
  std::string precision = "fp16";
  std::string profile = "bevfusion_df";
  std::string cuda_model_root = "model/bevfusion_df";
  std::string cuda_build_dir = "build";
  std::string camera_plan;
  std::string vtransform_plan;
  std::string lidar_onnx;
  std::string fuser_plan;
  std::string head_plan;
  bool enable_object_detection = true;
  bool enable_map_segmentation = false;
  std::string map_plan;
  std::string map_input_binding = "middle";
  std::string map_output_binding = "map";
  float map_score_threshold = 0.5f;
  std::vector<std::string> map_classes = {
      "drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"};
  bool print_model_info = false;
  float score_threshold = 0.2f;
  float sync_tolerance_ms = 50.0f;
  std::vector<float> camera_time_offsets_ms;
  float lidar_time_offset_ms = 0.0f;
  bool sync_debug = false;
  int sync_debug_limit = 100;
  int sync_queue_size = 30;
  int image_height = 128;
  int image_width = 352;
  // Optional training-time raw-resolution normalization applied before
  // image_resize. A zero size keeps the legacy single-stage preprocessing.
  int image_preprocess_height = 0;
  int image_preprocess_width = 0;
  float image_resize = 0.48f;
  float image_mean[3] = {0.485f, 0.456f, 0.406f};
  float image_std[3] = {0.229f, 0.224f, 0.225f};
  bool undistort_images = false;
  int point_dim = 5;
  float point_cloud_range[6] = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
  std::string calibration_file;
  std::string calibration_extrinsic_direction = "lidar2camera";
  bool allow_identity_calibration = false;
  std::string input_message_type = "RawMessage";
  std::string output_message_type = "RawMessage";
  bool publish_empty_frame = true;
  std::string bag_path;
  int max_frames = -1;
  std::string output_file;
};

AdapterConfig load_adapter_config(const std::string& path);

}  // namespace rscl_adapter
