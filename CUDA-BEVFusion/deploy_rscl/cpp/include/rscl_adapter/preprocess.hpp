#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rscl_adapter/config.hpp"
#include "rscl_adapter/types.hpp"

namespace rscl_adapter {

struct UndistortCache;

struct Calibration {
  std::vector<float> camera2lidar;
  std::vector<float> camera_intrinsics;
  // OpenCV rational-model coefficients in k1,k2,p1,p2,k3,k4,k5,k6 order.
  std::vector<float> camera_distortions;
  std::vector<unsigned char> camera_has_distortion;
  std::vector<float> lidar2image;
  // Cached remap tables; calibration geometry itself remains immutable.
  std::shared_ptr<UndistortCache> undistort_cache;
};

Calibration load_calibration(const std::string& path, const std::vector<std::string>& camera_order,
                             const std::string& extrinsic_direction, bool allow_identity);

ModelInput build_model_input(const SyncedFrame& frame, const AdapterConfig& cfg, const Calibration& calibration);

}  // namespace rscl_adapter
