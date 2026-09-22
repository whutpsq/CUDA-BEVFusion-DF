#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <yangluo_bevfusion_msgs/DetectedObjectArray.h>

#include <rscl_adapter/codecs.hpp>
#include <rscl_adapter/config.hpp>
#include <rscl_adapter/sync.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void startup_stage(const char* stage) {
  std::fprintf(stderr, "YANGLUO_LIGHT_STAGE=%s\n", stage);
  std::fflush(stderr);
}

int64_t stamp_us(const ros::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000LL + static_cast<int64_t>(stamp.nsec) / 1000LL;
}

const sensor_msgs::PointField* find_field(const sensor_msgs::PointCloud2& msg, const std::string& name) {
  for (const auto& field : msg.fields) {
    if (field.name == name) return &field;
  }
  return nullptr;
}

uint32_t swap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24U) | ((value & 0x0000ff00U) << 8U) |
         ((value & 0x00ff0000U) >> 8U) | ((value & 0xff000000U) >> 24U);
}

float read_float32(const uint8_t* ptr, bool big_endian) {
  uint32_t bits = 0;
  std::memcpy(&bits, ptr, sizeof(bits));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (big_endian) bits = swap32(bits);
#else
  if (!big_endian) bits = swap32(bits);
#endif
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

size_t load_class_count(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input) throw std::runtime_error("Cannot open classes file: " + path);
  size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && line[0] != '#') ++count;
  }
  if (count == 0) throw std::runtime_error("Classes file is empty: " + path);
  return count;
}

class DecodeOnlyNode {
 public:
  DecodeOnlyNode() : private_nh_("~") {
    startup_stage("constructor_body");
    std::string config_path;
    std::string classes_path;
    private_nh_.param<std::string>("config", config_path, "");
    private_nh_.param<std::string>("classes", classes_path, "");
    if (config_path.empty() || classes_path.empty()) {
      throw std::runtime_error("Both private parameters ~config and ~classes are required");
    }

    startup_stage("load_config_begin");
    cfg_ = rscl_adapter::load_adapter_config(config_path);
    startup_stage("load_config_done");
    class_count_ = load_class_count(classes_path);
    startup_stage("load_classes_done");
    if (cfg_.camera_topics.size() != 2 || cfg_.camera_order.size() != 2) {
      throw std::runtime_error("Yangluo runtime config must contain exactly two cameras");
    }

    sync_.reset(new rscl_adapter::FrameSynchronizer(
        cfg_.camera_topics, cfg_.camera_order, cfg_.lidar_topic, cfg_.sync_tolerance_ms,
        static_cast<size_t>(cfg_.sync_queue_size), cfg_.sync_debug, cfg_.sync_debug_limit,
        cfg_.camera_time_offsets_ms, cfg_.lidar_time_offset_ms));
    startup_stage("synchronizer_done");

    // Advertise the production output type to validate the ROS message contract.
    // Decode-only mode intentionally publishes no detections.
    publisher_ = nh_.advertise<yangluo_bevfusion_msgs::DetectedObjectArray>(cfg_.output_topic, 5);
    startup_stage("publisher_done");
    for (const auto& topic : cfg_.camera_topics) {
      camera_subscribers_.push_back(nh_.subscribe<sensor_msgs::CompressedImage>(
          topic, 10, [this, topic](const sensor_msgs::CompressedImage::ConstPtr& msg) { camera_callback(msg, topic); }));
    }
    startup_stage("camera_subscribers_done");
    lidar_subscriber_ = nh_.subscribe(cfg_.lidar_topic, 5, &DecodeOnlyNode::lidar_callback, this);
    startup_stage("lidar_subscriber_done");

    ROS_INFO_STREAM("Yangluo lightweight decode-only ready: cameras=" << cfg_.camera_topics[0] << ","
                    << cfg_.camera_topics[1] << " lidar=" << cfg_.lidar_topic << " output=" << cfg_.output_topic
                    << " classes=" << class_count_);
  }

 private:
  void record_sync(bool emitted) {
    if (!emitted) return;
    ++synced_frames_;
    ROS_INFO_STREAM_THROTTLE(2.0, "decode_only synced_frames=" << synced_frames_
                                      << " decoded_cameras=" << decoded_cameras_
                                      << " decoded_lidar=" << decoded_lidar_);
  }

  void camera_callback(const sensor_msgs::CompressedImage::ConstPtr& msg, const std::string& topic) {
    try {
      const int64_t timestamp = msg->header.stamp.isZero() ? stamp_us(ros::Time::now()) : stamp_us(msg->header.stamp);
      rscl_adapter::CameraPacket packet = rscl_adapter::decode_camera_packet(
          topic, timestamp, msg->data.data(), msg->data.size(), msg->format);
      ++decoded_cameras_;
      rscl_adapter::SyncedFrame frame;
      record_sync(sync_->add_camera(packet, &frame));
    } catch (const std::exception& error) {
      ROS_ERROR_STREAM_THROTTLE(1.0, "Camera decode-only callback failed for " << topic << ": " << error.what());
    }
  }

  void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    try {
      const auto* x = find_field(*msg, "x");
      const auto* y = find_field(*msg, "y");
      const auto* z = find_field(*msg, "z");
      const auto* intensity = find_field(*msg, "intensity");
      if (!x || !y || !z || !intensity) throw std::runtime_error("PointCloud2 requires x/y/z/intensity fields");
      for (const auto* field : {x, y, z, intensity}) {
        if (field->datatype != sensor_msgs::PointField::FLOAT32 || field->count != 1) {
          throw std::runtime_error("PointCloud2 x/y/z/intensity fields must be scalar FLOAT32");
        }
        if (field->offset + sizeof(float) > msg->point_step) {
          throw std::runtime_error("PointCloud2 field offset exceeds point_step");
        }
      }
      if (msg->point_step == 0 || msg->row_step < msg->width * msg->point_step ||
          msg->data.size() < static_cast<size_t>(msg->row_step) * msg->height) {
        throw std::runtime_error("Malformed PointCloud2 layout");
      }

      rscl_adapter::LidarPacket packet;
      packet.topic = cfg_.lidar_topic;
      packet.timestamp_us = msg->header.stamp.isZero() ? stamp_us(ros::Time::now()) : stamp_us(msg->header.stamp);
      packet.point_dim = 5;
      packet.points.reserve(static_cast<size_t>(msg->width) * msg->height * 5U);
      for (uint32_t row = 0; row < msg->height; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * msg->row_step;
        for (uint32_t col = 0; col < msg->width; ++col) {
          const uint8_t* record = msg->data.data() + row_offset + static_cast<size_t>(col) * msg->point_step;
          const float values[4] = {
              read_float32(record + x->offset, msg->is_bigendian),
              read_float32(record + y->offset, msg->is_bigendian),
              read_float32(record + z->offset, msg->is_bigendian),
              read_float32(record + intensity->offset, msg->is_bigendian)};
          if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || !std::isfinite(values[2]) ||
              !std::isfinite(values[3])) continue;
          packet.points.insert(packet.points.end(), values, values + 4);
          packet.points.push_back(0.0F);
        }
      }
      ++decoded_lidar_;
      rscl_adapter::SyncedFrame frame;
      record_sync(sync_->add_lidar(packet, &frame));
    } catch (const std::exception& error) {
      ROS_ERROR_STREAM_THROTTLE(1.0, "LiDAR decode-only callback failed: " << error.what());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  rscl_adapter::AdapterConfig cfg_;
  std::unique_ptr<rscl_adapter::FrameSynchronizer> sync_;
  ros::Publisher publisher_;
  std::vector<ros::Subscriber> camera_subscribers_;
  ros::Subscriber lidar_subscriber_;
  size_t class_count_ = 0;
  uint64_t decoded_cameras_ = 0;
  uint64_t decoded_lidar_ = 0;
  uint64_t synced_frames_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  startup_stage("main_enter");
  ros::init(argc, argv, "yangluo_bevfusion_decode_only");
  startup_stage("ros_init_done");
  try {
    startup_stage("node_construct_begin");
    DecodeOnlyNode node;
    startup_stage("node_construct_done");
    // QEMU user-mode emulation in the GPU-less build container crashes inside
    // roscpp's blocking ros::spin() wait path.  Explicit single-threaded
    // polling has the same callback ordering while avoiding that emulation
    // boundary.  The production Thor node continues to use ros::spin().
    startup_stage("spin_loop_begin");
    uint64_t loop_count = 0;
    while (ros::ok()) {
      if (loop_count < 3) startup_stage("spin_once_begin");
      ros::spinOnce();
      if (loop_count < 3) startup_stage("spin_once_done");
      ++loop_count;
      ros::WallDuration(0.01).sleep();
    }
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("Yangluo lightweight decode-only startup failed: " << error.what());
    return 1;
  }
  return 0;
}
