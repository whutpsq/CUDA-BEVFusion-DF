#pragma once

#include <memory>

#include "rscl_adapter/config.hpp"
#include "rscl_adapter/preprocess.hpp"
#include "rscl_adapter/runner.hpp"
#include "rscl_adapter/sync.hpp"

namespace rscl_adapter {

class BevFusionPipeline {
 public:
  explicit BevFusionPipeline(const AdapterConfig& cfg, bool decode_only = false);
  ~BevFusionPipeline();

  bool add_camera(const CameraPacket& packet, std::string* output_json);
  bool add_lidar(const LidarPacket& packet, std::string* output_json);

  int synced_frames() const { return synced_frames_; }

 private:
  bool process_frame(const SyncedFrame& frame, std::string* output_json);

  AdapterConfig cfg_;
  bool decode_only_ = false;
  FrameSynchronizer sync_;
  Calibration calibration_;
  std::unique_ptr<BevFusionRunner> runner_;
  int synced_frames_ = 0;
};

}  // namespace rscl_adapter
