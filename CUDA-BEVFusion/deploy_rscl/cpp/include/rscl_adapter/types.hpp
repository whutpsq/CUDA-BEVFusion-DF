#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rscl_adapter {

struct Image {
  int width = 0;
  int height = 0;
  int channels = 3;
  std::vector<unsigned char> rgb;
};

struct CameraPacket {
  std::string topic;
  int64_t timestamp_us = 0;
  Image image;
};

struct LidarPacket {
  std::string topic;
  int64_t timestamp_us = 0;
  int point_dim = 5;
  std::vector<float> points;
};

struct SyncedFrame {
  int64_t timestamp_us = 0;
  std::vector<Image> cameras;
  std::vector<float> lidar;
  int lidar_point_dim = 5;
};

struct ModelInput {
  int num_cameras = 0;
  int image_height = 0;
  int image_width = 0;
  std::vector<float> images_chw;
  std::vector<float> points;
  std::vector<float> camera2lidar;
  std::vector<float> camera_intrinsics;
  std::vector<float> lidar2image;
  std::vector<float> img_aug_matrix;
};

struct Detection {
  int label = 0;
  float score = 0.0f;
  float box[9] = {0.0f};
};

struct MapMask {
  std::vector<int> shape;
  std::vector<unsigned char> data;
};

struct InferenceOutput {
  std::vector<Detection> detections;
  bool has_map = false;
  MapMask map;
};

}  // namespace rscl_adapter
