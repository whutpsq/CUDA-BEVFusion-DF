#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "rscl_adapter/bag_reader.hpp"
#include "rscl_adapter/codecs.hpp"
#include "rscl_adapter/config.hpp"
#include "rscl_adapter/sync.hpp"

namespace {

struct Args {
  std::string adapter_config = "deploy_rscl/configs/bevfusion_rscl.yaml";
  std::string bag;
  std::string output_dir;
  int max_frames = -1;
  float sync_tolerance_ms = -1.0f;
  int sync_queue_size = -1;
  int sync_debug_limit = -1;
  std::vector<float> camera_time_offsets_ms;
  bool has_camera_time_offsets_ms = false;
  float lidar_time_offset_ms = 0.0f;
  bool has_lidar_time_offset_ms = false;
  bool sync_debug = false;
};

bool arg_eq(const char* lhs, const char* rhs) { return std::strcmp(lhs, rhs) == 0; }

std::vector<float> parse_float_csv(const std::string& text) {
  std::vector<float> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) values.push_back(static_cast<float>(std::atof(item.c_str())));
  }
  return values;
}

void print_usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " --bag FILE --output-dir DIR [options]\n"
            << "  --adapter-config PATH         Adapter YAML/JSON\n"
            << "  --max-frames N                Stop after N synchronized frames\n"
            << "  --sync-tolerance-ms N         Override synchronization tolerance\n"
            << "  --sync-queue-size N           Override synchronization queue size\n"
            << "  --sync-debug                  Print synchronization diagnostics\n"
            << "  --sync-debug-limit N          Limit synchronization diagnostics\n"
            << "  --camera-time-offsets-ms CSV  Per-camera timestamp offsets\n"
            << "  --lidar-time-offset-ms N      LiDAR timestamp offset\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--adapter-config") && i + 1 < argc) {
      args.adapter_config = argv[++i];
    } else if (arg_eq(argv[i], "--bag") && i + 1 < argc) {
      args.bag = argv[++i];
    } else if (arg_eq(argv[i], "--output-dir") && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (arg_eq(argv[i], "--max-frames") && i + 1 < argc) {
      args.max_frames = std::atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--sync-tolerance-ms") && i + 1 < argc) {
      args.sync_tolerance_ms = static_cast<float>(std::atof(argv[++i]));
    } else if (arg_eq(argv[i], "--sync-queue-size") && i + 1 < argc) {
      args.sync_queue_size = std::atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--sync-debug")) {
      args.sync_debug = true;
    } else if (arg_eq(argv[i], "--sync-debug-limit") && i + 1 < argc) {
      args.sync_debug_limit = std::atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--camera-time-offsets-ms") && i + 1 < argc) {
      args.camera_time_offsets_ms = parse_float_csv(argv[++i]);
      args.has_camera_time_offsets_ms = true;
    } else if (arg_eq(argv[i], "--lidar-time-offset-ms") && i + 1 < argc) {
      args.lidar_time_offset_ms = static_cast<float>(std::atof(argv[++i]));
      args.has_lidar_time_offset_ms = true;
    } else if (arg_eq(argv[i], "-h") || arg_eq(argv[i], "--help")) {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("Unknown or incomplete argument: ") + argv[i]);
    }
  }
  if (args.bag.empty()) throw std::runtime_error("--bag is required");
  if (args.output_dir.empty()) throw std::runtime_error("--output-dir is required");
  return args;
}

bool contains_topic(const std::vector<std::string>& topics, const std::string& topic) {
  return std::find(topics.begin(), topics.end(), topic) != topics.end();
}

void ensure_dir(const std::string& path) {
#if defined(_WIN32)
  _mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0755);
#endif
}

std::string sanitize_filename(std::string text) {
  for (size_t i = 0; i < text.size(); ++i) {
    if (!std::isalnum(static_cast<unsigned char>(text[i]))) text[i] = '_';
  }
  return text;
}

void write_float_file(const std::string& path, const std::vector<float>& data) {
  std::ofstream output(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("Failed to open lidar export: " + path);
  if (!data.empty()) {
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(float)));
  }
}

void write_text_file(const std::string& path, const std::string& text) {
  std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
  if (!output) throw std::runtime_error("Failed to open metadata export: " + path);
  output << text;
}

void export_frame(const std::string& root, const rscl_adapter::AdapterConfig& cfg,
                  const rscl_adapter::SyncedFrame& frame, int index) {
  char dirname[128];
  std::snprintf(dirname, sizeof(dirname), "frame_%06d_%lld", index,
                static_cast<long long>(frame.timestamp_us));
  const std::string frame_dir = root + "/" + dirname;
  ensure_dir(frame_dir);

  if (frame.cameras.size() != cfg.camera_order.size()) {
    throw std::runtime_error("Synchronized camera count does not match camera_order");
  }

  std::ostringstream metadata;
  metadata << "{\"timestamp_us\":" << frame.timestamp_us
           << ",\"lidar_file\":\"lidar.bin\",\"lidar_point_dim\":" << frame.lidar_point_dim
           << ",\"cameras\":[";
  for (size_t i = 0; i < frame.cameras.size(); ++i) {
    const rscl_adapter::Image& image = frame.cameras[i];
    if (image.width <= 0 || image.height <= 0 || image.channels != 3 || image.rgb.empty()) {
      throw std::runtime_error("Decoded camera image is empty or is not RGB");
    }
    const std::string filename = sanitize_filename(cfg.camera_order[i]) + ".jpg";
    const std::string path = frame_dir + "/" + filename;
    if (!stbi_write_jpg(path.c_str(), image.width, image.height, 3, image.rgb.data(), 92)) {
      throw std::runtime_error("Failed to write camera JPEG: " + path);
    }
    if (i) metadata << ',';
    metadata << "{\"name\":\"" << cfg.camera_order[i] << "\",\"file\":\"" << filename
             << "\",\"width\":" << image.width << ",\"height\":" << image.height << '}';
  }
  metadata << "]}";

  write_float_file(frame_dir + "/lidar.bin", frame.lidar);
  write_text_file(frame_dir + "/frame.json", metadata.str());
}

rscl_adapter::CameraPacket decode_camera(rscl_adapter::StatefulCameraDecoder* decoder,
                                         const rscl_adapter::BagMessage& msg) {
  try {
    return rscl_adapter::decode_camera_raw_message(msg.topic, msg.payload.data(), msg.payload.size(), decoder);
  } catch (const rscl_adapter::VideoFrameNotReady&) {
    throw;
  } catch (const std::exception&) {
    if (msg.timestamp_us <= 0 || decoder == nullptr) throw;
    const std::vector<unsigned char>& fallback = msg.raw_payload.empty() ? msg.payload : msg.raw_payload;
    return decoder->decode(msg.topic, msg.timestamp_us, fallback.data(), fallback.size(), msg.encoding,
                           msg.image_width, msg.image_height);
  }
}

rscl_adapter::LidarPacket decode_lidar(const rscl_adapter::BagMessage& msg, int point_dim) {
  try {
    return rscl_adapter::decode_lidar_raw_message(msg.topic, msg.payload.data(), msg.payload.size(), point_dim);
  } catch (const std::exception&) {
    if (msg.timestamp_us <= 0) throw;
    const std::vector<unsigned char>& fallback = msg.raw_payload.empty() ? msg.payload : msg.raw_payload;
    return rscl_adapter::decode_lidar_packet(msg.topic, msg.timestamp_us, fallback.data(), fallback.size(),
                                             point_dim, msg.point_step, msg.point_width);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    rscl_adapter::AdapterConfig cfg = rscl_adapter::load_adapter_config(args.adapter_config);
    if (args.sync_tolerance_ms >= 0.0f) cfg.sync_tolerance_ms = args.sync_tolerance_ms;
    if (args.sync_queue_size > 0) cfg.sync_queue_size = args.sync_queue_size;
    if (args.sync_debug) cfg.sync_debug = true;
    if (args.sync_debug_limit >= 0) cfg.sync_debug_limit = args.sync_debug_limit;
    if (args.has_camera_time_offsets_ms) cfg.camera_time_offsets_ms = args.camera_time_offsets_ms;
    if (args.has_lidar_time_offset_ms) cfg.lidar_time_offset_ms = args.lidar_time_offset_ms;

    ensure_dir(args.output_dir);
    std::vector<std::string> included_topics = cfg.camera_topics;
    included_topics.push_back(cfg.lidar_topic);
    std::unique_ptr<rscl_adapter::BagReader> reader =
        rscl_adapter::create_rscl_bag_reader(args.bag, included_topics);
    if (!reader || !reader->is_valid()) throw std::runtime_error("Invalid rsclbag: " + args.bag);

    rscl_adapter::FrameSynchronizer sync(
        cfg.camera_topics, cfg.camera_order, cfg.lidar_topic, cfg.sync_tolerance_ms,
        static_cast<size_t>(cfg.sync_queue_size), cfg.sync_debug, cfg.sync_debug_limit,
        cfg.camera_time_offsets_ms, cfg.lidar_time_offset_ms);
    std::map<std::string, std::unique_ptr<rscl_adapter::StatefulCameraDecoder> > camera_decoders;
    for (size_t i = 0; i < cfg.camera_topics.size(); ++i) {
      camera_decoders[cfg.camera_topics[i]].reset(new rscl_adapter::StatefulCameraDecoder());
    }

    int frames = 0;
    int decode_errors = 0;
    rscl_adapter::BagMessage msg;
    while (reader->read_next(&msg)) {
      try {
        rscl_adapter::SyncedFrame frame;
        bool synced = false;
        if (contains_topic(cfg.camera_topics, msg.topic)) {
          const rscl_adapter::CameraPacket camera = decode_camera(camera_decoders[msg.topic].get(), msg);
          synced = sync.add_camera(camera, &frame);
        } else if (msg.topic == cfg.lidar_topic) {
          const rscl_adapter::LidarPacket lidar = decode_lidar(msg, cfg.point_dim);
          synced = sync.add_lidar(lidar, &frame);
        }
        if (!synced) continue;

        export_frame(args.output_dir, cfg, frame, frames);
        ++frames;
        std::cout << "exported_frames=" << frames << " timestamp_us=" << frame.timestamp_us
                  << " lidar_points="
                  << (frame.lidar_point_dim > 0 ? frame.lidar.size() / static_cast<size_t>(frame.lidar_point_dim) : 0)
                  << " decode_errors=" << decode_errors << std::endl;
        if (args.max_frames >= 0 && frames >= args.max_frames) break;
      } catch (const rscl_adapter::VideoFrameNotReady&) {
        continue;
      } catch (const std::exception& error) {
        ++decode_errors;
        std::cerr << "failed to process topic=" << msg.topic << " error=" << error.what() << std::endl;
      }
    }

    std::cout << "exported_frames=" << frames << " decode_errors=" << decode_errors
              << " output_dir=" << args.output_dir << std::endl;
    return frames > 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
