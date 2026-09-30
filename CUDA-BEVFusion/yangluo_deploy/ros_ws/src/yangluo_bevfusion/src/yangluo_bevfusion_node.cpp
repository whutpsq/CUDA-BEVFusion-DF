#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <yangluo_bevfusion_msgs/DetectedObject.h>
#include <yangluo_bevfusion_msgs/DetectedObjectArray.h>

#include <rscl_adapter/codecs.hpp>
#include <rscl_adapter/config.hpp>
#include <rscl_adapter/json.hpp>
#include <rscl_adapter/pipeline.hpp>

#include <algorithm>
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
  std::fprintf(stderr, "YANGLUO_STARTUP_STAGE=%s\n", stage);
  std::fflush(stderr);
}

int64_t stamp_us(const ros::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000LL + static_cast<int64_t>(stamp.nsec) / 1000LL;
}

ros::Time time_from_us(int64_t value) {
  ros::Time stamp;
  stamp.sec = static_cast<uint32_t>(value / 1000000LL);
  stamp.nsec = static_cast<uint32_t>((value % 1000000LL) * 1000LL);
  return stamp;
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

std::vector<std::string> load_classes(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input) throw std::runtime_error("Cannot open classes file: " + path);
  std::vector<std::string> names;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && line[0] != '#') names.push_back(line);
  }
  if (names.empty()) throw std::runtime_error("Classes file is empty: " + path);
  return names;
}

class YangluoNode {
 public:
  YangluoNode() : private_nh_("~") {
    startup_stage("constructor_body");
    std::string config_path;
    std::string classes_path;
    private_nh_.param<std::string>("config", config_path, "");
    private_nh_.param<std::string>("classes", classes_path, "");
    private_nh_.param("decode_only", decode_only_, false);
    if (config_path.empty() || classes_path.empty()) {
      throw std::runtime_error("Both private parameters ~config and ~classes are required");
    }

    startup_stage("load_config_begin");
    cfg_ = rscl_adapter::load_adapter_config(config_path);
    startup_stage("load_config_done");
    class_names_ = load_classes(classes_path);
    startup_stage("load_classes_done");
    if (cfg_.camera_topics.size() != 2 || cfg_.camera_order.size() != 2) {
      throw std::runtime_error("Yangluo runtime config must contain exactly two cameras");
    }
    startup_stage("pipeline_begin");
    pipeline_.reset(new rscl_adapter::BevFusionPipeline(cfg_, decode_only_));
    startup_stage("pipeline_done");

    publisher_ = nh_.advertise<yangluo_bevfusion_msgs::DetectedObjectArray>(cfg_.output_topic, 5);
    startup_stage("publisher_done");
    for (const auto& topic : cfg_.camera_topics) {
      camera_subscribers_.push_back(nh_.subscribe<sensor_msgs::CompressedImage>(
          topic, 10, [this, topic](const sensor_msgs::CompressedImage::ConstPtr& msg) { camera_callback(msg, topic); }));
    }
    lidar_subscriber_ = nh_.subscribe(cfg_.lidar_topic, 5, &YangluoNode::lidar_callback, this);
    startup_stage("subscribers_done");
    ROS_INFO_STREAM("Yangluo BEVFusion ready: cameras=" << cfg_.camera_topics[0] << "," << cfg_.camera_topics[1]
                    << " lidar=" << cfg_.lidar_topic << " output=" << cfg_.output_topic
                    << " decode_only=" << (decode_only_ ? "true" : "false"));
  }

 private:
  void camera_callback(const sensor_msgs::CompressedImage::ConstPtr& msg, const std::string& topic) {
    try {
      const int64_t timestamp = msg->header.stamp.isZero() ? stamp_us(ros::Time::now()) : stamp_us(msg->header.stamp);
      rscl_adapter::CameraPacket packet = rscl_adapter::decode_camera_packet(
          topic, timestamp, msg->data.data(), msg->data.size(), msg->format);
      std::string json;
      if (pipeline_->add_camera(packet, &json)) handle_output(json, timestamp);
    } catch (const std::exception& error) {
      ROS_ERROR_STREAM_THROTTLE(1.0, "Camera callback failed for " << topic << ": " << error.what());
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
          packet.points.push_back(0.0F);  // Must match the training converter; not per-point timestamp.
        }
      }
      std::string json;
      if (pipeline_->add_lidar(packet, &json)) handle_output(json, packet.timestamp_us);
    } catch (const std::exception& error) {
      ROS_ERROR_STREAM_THROTTLE(1.0, "LiDAR callback failed: " << error.what());
    }
  }

  void handle_output(const std::string& json, int64_t fallback_timestamp_us) {
    if (decode_only_) {
      ROS_INFO_STREAM_THROTTLE(2.0, "decode_only synced_frames=" << pipeline_->synced_frames());
      return;
    }
    if (json.empty()) return;
    const rscl_adapter::JsonValue root = rscl_adapter::parse_json(json);
    const auto* objects = root.get("objects");
    const auto* timestamp = root.get("timestamp_us");
    if (!objects || !objects->is_array()) throw std::runtime_error("Inference JSON lacks objects array");

    yangluo_bevfusion_msgs::DetectedObjectArray output;
    output.header.frame_id = "base_link";
    output.header.stamp = time_from_us(timestamp ? static_cast<int64_t>(timestamp->as_number()) : fallback_timestamp_us);
    output.objects.reserve(objects->array.size());
    for (size_t index = 0; index < objects->array.size(); ++index) {
      const auto& source = objects->array[index];
      const auto* label_value = source.get("label");
      const auto* score_value = source.get("score");
      const auto* box = source.get("box");
      if (!label_value || !score_value || !box || !box->is_array() || box->array.size() != 9) continue;
      const int label = static_cast<int>(label_value->as_number(-1));
      if (label < 0 || label >= static_cast<int>(class_names_.size())) {
        ROS_WARN_STREAM_THROTTLE(1.0, "Skipping out-of-range class id " << label);
        continue;
      }

      yangluo_bevfusion_msgs::DetectedObject object;
      object.id = static_cast<uint32_t>(index);  // Detection index only; this node does not track objects.
      object.class_id = label;
      object.class_name = class_names_[label];
      object.score = static_cast<float>(score_value->as_number());
      object.pose.position.x = box->array[0].as_number();
      object.pose.position.y = box->array[1].as_number();
      const double yaw = box->array[6].as_number();
      object.pose.orientation.z = std::sin(yaw * 0.5);
      object.pose.orientation.w = std::cos(yaw * 0.5);
      object.dimensions.x = box->array[3].as_number();
      object.dimensions.y = box->array[4].as_number();
      object.dimensions.z = box->array[5].as_number();
      // CUDA-BEVFusion's TransBBox decoder publishes box[2] as the bottom-face
      // Z (it has already subtracted half the decoded height).  ROS Pose uses
      // the geometric box center so that pose +/- dimensions / 2 produces the
      // correct bottom and top faces for camera projection and downstream use.
      const double bottom_z = box->array[2].as_number();
      object.pose.position.z = bottom_z + object.dimensions.z * 0.5;
      object.velocity.linear.x = box->array[7].as_number();
      object.velocity.linear.y = box->array[8].as_number();

      const double half_x = object.dimensions.x * 0.5;
      const double half_y = object.dimensions.y * 0.5;
      const double local_x[4] = {half_x, -half_x, -half_x, half_x};
      const double local_y[4] = {half_y, half_y, -half_y, -half_y};
      const double cosine = std::cos(yaw);
      const double sine = std::sin(yaw);
      for (size_t corner = 0; corner < 4; ++corner) {
        object.corners[corner].x = object.pose.position.x + cosine * local_x[corner] - sine * local_y[corner];
        object.corners[corner].y = object.pose.position.y + sine * local_x[corner] + cosine * local_y[corner];
        object.corners[corner].z = bottom_z;
      }
      output.objects.push_back(object);
    }
    publisher_.publish(output);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  rscl_adapter::AdapterConfig cfg_;
  bool decode_only_ = false;
  std::vector<std::string> class_names_;
  std::unique_ptr<rscl_adapter::BevFusionPipeline> pipeline_;
  std::vector<ros::Subscriber> camera_subscribers_;
  ros::Subscriber lidar_subscriber_;
  ros::Publisher publisher_;
};

}  // namespace

int main(int argc, char** argv) {
  startup_stage("main_enter");
  ros::init(argc, argv, "yangluo_bevfusion");
  startup_stage("ros_init_done");
  try {
    startup_stage("node_construct_begin");
    YangluoNode node;
    startup_stage("node_construct_done");
    ros::spin();  // Single-threaded by design: pipeline synchronization is stateful.
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("Yangluo BEVFusion startup failed: " << error.what());
    return 1;
  }
  return 0;
}
