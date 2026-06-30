#include "rscl_adapter/online_node.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace rscl_adapter {
namespace {

// Vehicle-side RSCL C++ integration template.
//
// Replace this file with a small backend that binds the vehicle RSCL SDK:
//   - construct an RSCL node from cfg.module_name / cfg.node_name
//   - create subscribers for cfg.camera_topics and cfg.lidar_topic
//   - convert every SDK callback into a BagMessage
//   - publish output JSON to cfg.output_topic as cfg.output_message_type
//
// BagMessage conversion rules are the same as the offline bag backend:
//   message.topic        = callback topic
//   message.timestamp_us = SDK timestamp in microseconds if available
//   message.payload      = RawMessage JSON bytes, encoded H26x bytes wrapped by JSON,
//                          decoded RGB/BGR/NV12 image bytes, or point cloud bytes
//   message.raw_payload  = optional original SDK buffer
//   message.encoding     = "rgb8" / "bgr8" / "nv12" / "h265" / "h264" when known
//   message.image_width/image_height for decoded raw images
//   message.point_step/point_width for point cloud messages
//
// If your RSCL SDK exposes reflected JSON for RawMessage payloads, pass that JSON
// directly in message.payload. The shared codecs already understand the inspected
// RSCL echo-hex byte string format.

class TemplatePublisher : public OnlinePublisher {
 public:
  void publish(const std::string& payload) override {
    (void)payload;
    throw std::runtime_error("TemplatePublisher is not connected to the RSCL C++ SDK.");
  }
};

class TemplateOnlineNode : public OnlineNode {
 public:
  explicit TemplateOnlineNode(const AdapterConfig& cfg) : cfg_(cfg) {
    throw std::runtime_error("TemplateOnlineNode is not connected to the RSCL C++ SDK.");
  }

  bool is_valid() const override { return false; }

  void subscribe(const std::string& topic, const std::string& message_type, MessageCallback callback) override {
    (void)topic;
    (void)message_type;
    (void)callback;
  }

  std::unique_ptr<OnlinePublisher> create_publisher(const std::string& topic,
                                                    const std::string& message_type) override {
    (void)topic;
    (void)message_type;
    return std::unique_ptr<OnlinePublisher>(new TemplatePublisher());
  }

  void spin() override {}

 private:
  AdapterConfig cfg_;
};

}  // namespace

std::unique_ptr<OnlineNode> create_rscl_online_node(const AdapterConfig& cfg) {
  return std::unique_ptr<OnlineNode>(new TemplateOnlineNode(cfg));
}

}  // namespace rscl_adapter
