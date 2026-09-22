#include <ros/message_traits.h>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <yangluo_bevfusion_msgs/DetectedObjectArray.h>

#include <rscl_adapter/codecs.hpp>
#include <rscl_adapter/config.hpp>
#include <rscl_adapter/sync.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::string bag;
  std::string config;
  std::string classes;
  double duration_sec = 30.0;
  uint64_t max_synced = 10;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " --bag FILE --config FILE --classes FILE [--duration-sec N] [--max-synced N]" << std::endl;
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    const std::string key(argv[index]);
    if ((key == "--bag" || key == "--config" || key == "--classes" || key == "--duration-sec" ||
         key == "--max-synced") && index + 1 >= argc) {
      throw std::runtime_error("Missing value for " + key);
    }
    if (key == "--bag") args.bag = argv[++index];
    else if (key == "--config") args.config = argv[++index];
    else if (key == "--classes") args.classes = argv[++index];
    else if (key == "--duration-sec") args.duration_sec = std::strtod(argv[++index], nullptr);
    else if (key == "--max-synced") args.max_synced = std::strtoull(argv[++index], nullptr, 10);
    else if (key == "--help" || key == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::runtime_error("Unknown argument: " + key);
  }
  if (args.bag.empty() || args.config.empty() || args.classes.empty())
    throw std::runtime_error("--bag, --config and --classes are required");
  if (args.duration_sec <= 0.0 || args.max_synced == 0)
    throw std::runtime_error("Duration and max-synced must be positive");
  return args;
}

int64_t stamp_us(const ros::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000LL + static_cast<int64_t>(stamp.nsec) / 1000LL;
}

std::string normalized_topic(const std::string& topic) {
  return !topic.empty() && topic[0] == '/' ? topic : "/" + topic;
}

const sensor_msgs::PointField* find_field(const sensor_msgs::PointCloud2& msg, const std::string& name) {
  for (const auto& field : msg.fields) if (field.name == name) return &field;
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

rscl_adapter::LidarPacket convert_cloud(const sensor_msgs::PointCloud2& msg, const std::string& topic,
                                        int64_t fallback_stamp_us) {
  const auto* x = find_field(msg, "x");
  const auto* y = find_field(msg, "y");
  const auto* z = find_field(msg, "z");
  const auto* intensity = find_field(msg, "intensity");
  if (!x || !y || !z || !intensity) throw std::runtime_error("PointCloud2 requires x/y/z/intensity fields");
  for (const auto* field : {x, y, z, intensity}) {
    if (field->datatype != sensor_msgs::PointField::FLOAT32 || field->count != 1)
      throw std::runtime_error("PointCloud2 x/y/z/intensity fields must be scalar FLOAT32");
    if (field->offset + sizeof(float) > msg.point_step)
      throw std::runtime_error("PointCloud2 field offset exceeds point_step");
  }
  if (msg.point_step == 0 || msg.row_step < msg.width * msg.point_step ||
      msg.data.size() < static_cast<size_t>(msg.row_step) * msg.height)
    throw std::runtime_error("Malformed PointCloud2 layout");

  rscl_adapter::LidarPacket packet;
  packet.topic = topic;
  packet.timestamp_us = msg.header.stamp.isZero() ? fallback_stamp_us : stamp_us(msg.header.stamp);
  packet.point_dim = 5;
  packet.points.reserve(static_cast<size_t>(msg.width) * msg.height * 5U);
  for (uint32_t row = 0; row < msg.height; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * msg.row_step;
    for (uint32_t col = 0; col < msg.width; ++col) {
      const uint8_t* record = msg.data.data() + row_offset + static_cast<size_t>(col) * msg.point_step;
      const float values[4] = {read_float32(record + x->offset, msg.is_bigendian),
                               read_float32(record + y->offset, msg.is_bigendian),
                               read_float32(record + z->offset, msg.is_bigendian),
                               read_float32(record + intensity->offset, msg.is_bigendian)};
      if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || !std::isfinite(values[2]) ||
          !std::isfinite(values[3])) continue;
      packet.points.insert(packet.points.end(), values, values + 4);
      packet.points.push_back(0.0F);
    }
  }
  return packet;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments args = parse_arguments(argc, argv);
    const rscl_adapter::AdapterConfig cfg = rscl_adapter::load_adapter_config(args.config);
    const size_t class_count = load_class_count(args.classes);
    if (cfg.camera_topics.size() != 2 || cfg.camera_order.size() != 2)
      throw std::runtime_error("Yangluo runtime config must contain exactly two cameras");

    const std::string output_type =
        ros::message_traits::DataType<yangluo_bevfusion_msgs::DetectedObjectArray>::value();
    if (output_type != "yangluo_bevfusion_msgs/DetectedObjectArray")
      throw std::runtime_error("Unexpected output message type: " + output_type);

    rscl_adapter::FrameSynchronizer synchronizer(
        cfg.camera_topics, cfg.camera_order, cfg.lidar_topic, cfg.sync_tolerance_ms,
        static_cast<size_t>(cfg.sync_queue_size), cfg.sync_debug, cfg.sync_debug_limit,
        cfg.camera_time_offsets_ms, cfg.lidar_time_offset_ms);

    rosbag::Bag bag;
    bag.open(args.bag, rosbag::bagmode::Read);
    std::vector<std::string> topics = cfg.camera_topics;
    topics.push_back(cfg.lidar_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    if (view.size() == 0) throw std::runtime_error("Selected topics contain no messages");

    uint64_t camera_counts[2] = {0, 0};
    uint64_t lidar_count = 0, synced_count = 0, camera_errors = 0, lidar_errors = 0;
    ros::Time first_record_time;
    bool have_first_record_time = false;
    std::cout << "OFFLINE_BAG_VALIDATION_BEGIN classes=" << class_count << " output_type=" << output_type
              << " selected_messages=" << view.size() << std::endl;

    for (const rosbag::MessageInstance& message : view) {
      if (!have_first_record_time) { first_record_time = message.getTime(); have_first_record_time = true; }
      if ((message.getTime() - first_record_time).toSec() > args.duration_sec) break;
      const std::string topic = normalized_topic(message.getTopic());
      bool emitted = false;
      rscl_adapter::SyncedFrame frame;
      if (topic == cfg.camera_topics[0] || topic == cfg.camera_topics[1]) {
        const size_t camera_index = topic == cfg.camera_topics[0] ? 0U : 1U;
        try {
          const sensor_msgs::CompressedImage::ConstPtr image = message.instantiate<sensor_msgs::CompressedImage>();
          if (!image) throw std::runtime_error("Message is not sensor_msgs/CompressedImage");
          const int64_t fallback = stamp_us(message.getTime());
          const int64_t timestamp = image->header.stamp.isZero() ? fallback : stamp_us(image->header.stamp);
          rscl_adapter::CameraPacket packet = rscl_adapter::decode_camera_packet(
              topic, timestamp, image->data.data(), image->data.size(), image->format);
          ++camera_counts[camera_index];
          emitted = synchronizer.add_camera(packet, &frame);
        } catch (const std::exception& error) {
          ++camera_errors;
          if (camera_errors <= 3) std::cerr << "CAMERA_ERROR topic=" << topic << " what=" << error.what() << std::endl;
        }
      } else if (topic == cfg.lidar_topic) {
        try {
          const sensor_msgs::PointCloud2::ConstPtr cloud = message.instantiate<sensor_msgs::PointCloud2>();
          if (!cloud) throw std::runtime_error("Message is not sensor_msgs/PointCloud2");
          rscl_adapter::LidarPacket packet = convert_cloud(*cloud, cfg.lidar_topic, stamp_us(message.getTime()));
          ++lidar_count;
          emitted = synchronizer.add_lidar(packet, &frame);
        } catch (const std::exception& error) {
          ++lidar_errors;
          if (lidar_errors <= 3) std::cerr << "LIDAR_ERROR what=" << error.what() << std::endl;
        }
      }
      if (emitted) {
        ++synced_count;
        std::cout << "OFFLINE_SYNCED frame=" << synced_count << " timestamp_us=" << frame.timestamp_us
                  << " cameras=" << frame.cameras.size() << " points="
                  << (frame.lidar_point_dim > 0 ? frame.lidar.size() / frame.lidar_point_dim : 0) << std::endl;
        if (synced_count >= args.max_synced) break;
      }
    }
    bag.close();
    std::cout << "OFFLINE_BAG_VALIDATION_RESULT camera0=" << camera_counts[0] << " camera1=" << camera_counts[1]
              << " lidar=" << lidar_count << " synced=" << synced_count << " camera_errors=" << camera_errors
              << " lidar_errors=" << lidar_errors << std::endl;
    if (camera_counts[0] == 0 || camera_counts[1] == 0 || lidar_count == 0 || synced_count == 0 ||
        camera_errors != 0 || lidar_errors != 0) {
      std::cerr << "OFFLINE_BAG_VALIDATION_FAILED" << std::endl;
      return 2;
    }
    std::cout << "OFFLINE_BAG_VALIDATION_OK" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "OFFLINE_BAG_VALIDATION_FATAL what=" << error.what() << std::endl;
    usage(argv[0]);
    return 1;
  }
}
