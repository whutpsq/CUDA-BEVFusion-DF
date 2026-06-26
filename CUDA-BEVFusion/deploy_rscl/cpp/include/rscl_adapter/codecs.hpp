#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rscl_adapter/config.hpp"
#include "rscl_adapter/types.hpp"

namespace rscl_adapter {

class StatefulCameraDecoder;

CameraPacket decode_camera_packet(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                  const std::string& encoding = "", int width = 0, int height = 0);

LidarPacket decode_lidar_packet(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                int point_dim, int point_step = 0, int width = 0);

class VideoFrameNotReady : public std::runtime_error {
 public:
  explicit VideoFrameNotReady(const std::string& what) : std::runtime_error(what) {}
};

class StatefulCameraDecoder {
 public:
  StatefulCameraDecoder();
  ~StatefulCameraDecoder();
  StatefulCameraDecoder(const StatefulCameraDecoder&) = delete;
  StatefulCameraDecoder& operator=(const StatefulCameraDecoder&) = delete;

  CameraPacket decode(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                      const std::string& encoding = "", int width = 0, int height = 0);

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

std::string encode_detection_message(const InferenceOutput& output, const AdapterConfig& cfg, int64_t timestamp_us);

CameraPacket decode_camera_raw_message(const std::string& topic, const unsigned char* data, size_t size,
                                       StatefulCameraDecoder* decoder = nullptr);

std::vector<unsigned char> extract_camera_raw_payload(const unsigned char* data, size_t size);

LidarPacket decode_lidar_raw_message(const std::string& topic, const unsigned char* data, size_t size, int point_dim);

int64_t decode_raw_message_timestamp_us(const unsigned char* data, size_t size);

}  // namespace rscl_adapter
