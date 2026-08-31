#include "rscl_adapter/pipeline.hpp"

#include <atomic>
#include <iostream>
#include <stdexcept>

#include "rscl_adapter/codecs.hpp"

namespace rscl_adapter {

BevFusionPipeline::BevFusionPipeline(const AdapterConfig& cfg, bool decode_only)
    : cfg_(cfg),
      decode_only_(decode_only),
      sync_(cfg.camera_topics, cfg.camera_order, cfg.lidar_topic, cfg.sync_tolerance_ms,
            static_cast<size_t>(cfg.sync_queue_size), cfg.sync_debug, cfg.sync_debug_limit,
            cfg.camera_time_offsets_ms, cfg.lidar_time_offset_ms) {
  if (!decode_only_) {
    calibration_ =
        load_calibration(cfg_.calibration_file, cfg_.camera_order, cfg_.calibration_extrinsic_direction, cfg_.allow_identity_calibration);
    runner_.reset(new BevFusionRunner(cfg_));
  }
}

BevFusionPipeline::~BevFusionPipeline() = default;

bool BevFusionPipeline::add_camera(const CameraPacket& packet, std::string* output_json) {
  SyncedFrame frame;
  if (!sync_.add_camera(packet, &frame)) return false;
  return process_frame(frame, output_json);
}

bool BevFusionPipeline::add_lidar(const LidarPacket& packet, std::string* output_json) {
  SyncedFrame frame;
  if (!sync_.add_lidar(packet, &frame)) return false;
  return process_frame(frame, output_json);
}

bool BevFusionPipeline::process_frame(const SyncedFrame& frame, std::string* output_json) {
  ++synced_frames_;
  if (decode_only_) {
    if (output_json) *output_json = "";
    return true;
  }
  if (!runner_) throw std::runtime_error("BevFusionPipeline runner is not initialized");

  static std::atomic<bool> logged_first_inference(false);
  const bool log_first_inference = !logged_first_inference.exchange(true);
  if (log_first_inference) {
    std::cout << "inference_stage=build_model_input_begin model_coordinate_frame=car_center" << std::endl;
  }
  ModelInput input = build_model_input(frame, cfg_, calibration_);
  if (log_first_inference) {
    std::cout << "inference_stage=build_model_input_done images=" << input.images_chw.size()
              << " points=" << input.points.size() << " cameras=" << input.num_cameras << std::endl;
    std::cout << "inference_stage=runner_infer_begin" << std::endl;
  }
  InferenceOutput output = runner_->infer(input);
  if (log_first_inference) {
    std::cout << "inference_stage=runner_infer_done detections=" << output.detections.size() << std::endl;
  }
  const bool has_objects = !output.detections.empty();
  if (!has_objects && !output.has_map && !cfg_.publish_empty_frame) {
    if (output_json) *output_json = "";
    return true;
  }
  if (output_json) *output_json = encode_detection_message(output, cfg_, frame.timestamp_us);
  return true;
}

}  // namespace rscl_adapter
