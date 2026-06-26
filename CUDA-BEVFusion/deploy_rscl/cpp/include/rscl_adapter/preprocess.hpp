#pragma once

#include <string>
#include <vector>

#include "rscl_adapter/config.hpp"
#include "rscl_adapter/types.hpp"

namespace rscl_adapter {

struct Calibration {
  std::vector<float> camera2lidar;
  std::vector<float> camera_intrinsics;
  std::vector<float> lidar2image;
};

Calibration load_calibration(const std::string& path, const std::vector<std::string>& camera_order,
                             const std::string& extrinsic_direction, bool allow_identity);

ModelInput build_model_input(const SyncedFrame& frame, const AdapterConfig& cfg, const Calibration& calibration);

}  // namespace rscl_adapter
