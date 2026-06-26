#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <algorithm>
#include <map>
#include <fstream>
#include <iostream>
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

#include "rscl_adapter/bag_reader.hpp"
#include "rscl_adapter/codecs.hpp"
#include "rscl_adapter/config.hpp"
#include "rscl_adapter/pipeline.hpp"

namespace {

struct Args {
  std::string adapter_config = "deploy_rscl/configs/bevfusion_rscl.yaml";
  std::string bag;
  std::string output_file;
  std::string dump_camera_debug_dir;
  int max_frames = -1;
  int dump_camera_debug_count = 0;
  float sync_tolerance_ms = -1.0f;
  int sync_queue_size = -1;
  int sync_debug_limit = -1;
  std::vector<float> camera_time_offsets_ms;
  bool has_camera_time_offsets_ms = false;
  float lidar_time_offset_ms = 0.0f;
  bool has_lidar_time_offset_ms = false;
  bool decode_only = false;
  bool decode_images_only = false;
  bool sync_debug = false;
};

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static std::vector<float> parse_float_csv(const std::string& text) {
  std::vector<float> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    values.push_back(static_cast<float>(std::atof(item.c_str())));
  }
  return values;
}

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--adapter-config") && i + 1 < argc) {
      args.adapter_config = argv[++i];
    } else if (arg_eq(argv[i], "--bag") && i + 1 < argc) {
      args.bag = argv[++i];
    } else if (arg_eq(argv[i], "--output-file") && i + 1 < argc) {
      args.output_file = argv[++i];
    } else if (arg_eq(argv[i], "--max-frames") && i + 1 < argc) {
      args.max_frames = std::atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--dump-camera-debug-dir") && i + 1 < argc) {
      args.dump_camera_debug_dir = argv[++i];
    } else if (arg_eq(argv[i], "--dump-camera-debug-count") && i + 1 < argc) {
      args.dump_camera_debug_count = std::atoi(argv[++i]);
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
    } else if (arg_eq(argv[i], "--decode-only")) {
      args.decode_only = true;
    } else if (arg_eq(argv[i], "--decode-images-only")) {
      args.decode_images_only = true;
    } else if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) {
      std::cout
          << "Usage: rscl_bevfusion_bag_runner --adapter-config <yaml> --bag <file.rsclbag> [options]\n"
          << "Options:\n"
          << "  --max-frames N          Stop after N synchronized frames\n"
          << "  --output-file PATH      Write JSONL detections instead of stdout\n"
          << "  --dump-camera-debug-dir DIR    Dump camera JSON/raw/payload samples for debugging\n"
          << "  --dump-camera-debug-count N    Number of camera samples to dump, default 0\n"
          << "  --sync-tolerance-ms N   Override camera/lidar sync tolerance in milliseconds\n"
          << "  --sync-queue-size N     Override per-topic sync queue size\n"
          << "  --sync-debug            Print sync nearest-difference diagnostics\n"
          << "  --sync-debug-limit N    Limit sync diagnostic lines\n"
          << "  --camera-time-offsets-ms CSV  Override per-camera offsets, e.g. -350,0,0,0,0,0\n"
          << "  --lidar-time-offset-ms N      Override lidar timestamp offset in milliseconds\n"
          << "  --decode-only           Decode timestamps/synchronize only, no CUDA inference\n"
          << "  --decode-images-only    Decode image/lidar packets and synchronize only, no CUDA inference\n";
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("Unknown or incomplete argument: ") + argv[i]);
    }
  }
  return args;
}

static bool contains_topic(const std::vector<std::string>& topics, const std::string& topic) {
  for (size_t i = 0; i < topics.size(); ++i) {
    if (topics[i] == topic) return true;
  }
  return false;
}

static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
#if defined(_WIN32)
  _mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0755);
#endif
}

static std::string sanitize_filename(std::string text) {
  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (!std::isalnum(c)) text[i] = '_';
  }
  return text;
}

static void write_binary_file(const std::string& path, const std::vector<unsigned char>& data) {
  std::ofstream out(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("Failed to open debug file: " + path);
  if (!data.empty()) out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

static void write_text_file(const std::string& path, const std::string& text) {
  std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("Failed to open debug file: " + path);
  out << text;
}

static std::string hex_head(const std::vector<unsigned char>& data, size_t max_bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  const size_t n = std::min(max_bytes, data.size());
  out.reserve(n * 3);
  for (size_t i = 0; i < n; ++i) {
    if (i) out.push_back(' ');
    out.push_back(kHex[(data[i] >> 4) & 0x0F]);
    out.push_back(kHex[data[i] & 0x0F]);
  }
  return out;
}

static void dump_camera_debug_sample(const Args& args, const rscl_adapter::BagMessage& msg, int index) {
  if (args.dump_camera_debug_dir.empty() || args.dump_camera_debug_count <= 0) return;
  ensure_dir(args.dump_camera_debug_dir);

  char prefix[64];
  std::snprintf(prefix, sizeof(prefix), "camera_%04d_", index);
  const std::string base = args.dump_camera_debug_dir + "/" + prefix + sanitize_filename(msg.topic);

  write_binary_file(base + ".raw_capnp.bin", msg.raw_payload);
  if (!msg.reflected_json.empty()) {
    write_text_file(base + ".reflected.json", msg.reflected_json);
  } else {
    write_binary_file(base + ".payload.bin", msg.payload);
  }

  try {
    std::vector<unsigned char> camera_payload =
        rscl_adapter::extract_camera_raw_payload(msg.payload.data(), msg.payload.size());
    write_binary_file(base + ".camera_payload.bin", camera_payload);
    std::cerr << "camera_debug index=" << index << " topic=" << msg.topic << " json_len=" << msg.reflected_json.size()
              << " raw_capnp_len=" << msg.raw_payload.size() << " camera_payload_len=" << camera_payload.size()
              << " camera_payload_head=" << hex_head(camera_payload, 32) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "camera_debug index=" << index << " topic=" << msg.topic << " failed_to_extract_payload=" << e.what()
              << " json_len=" << msg.reflected_json.size() << " raw_capnp_len=" << msg.raw_payload.size()
              << " payload_head=" << hex_head(msg.payload, 32) << std::endl;
  }
}

static rscl_adapter::CameraPacket decode_camera_message(rscl_adapter::StatefulCameraDecoder* decoder,
                                                        const rscl_adapter::BagMessage& msg) {
  try {
    return rscl_adapter::decode_camera_raw_message(msg.topic, msg.payload.data(), msg.payload.size(), decoder);
  } catch (const rscl_adapter::VideoFrameNotReady&) {
    throw;
  } catch (const std::exception&) {
    if (msg.timestamp_us <= 0) throw;
    if (decoder == nullptr) throw std::runtime_error("camera decoder is not initialized");
    const std::vector<unsigned char>& fallback = msg.raw_payload.empty() ? msg.payload : msg.raw_payload;
    return decoder->decode(msg.topic, msg.timestamp_us, fallback.data(), fallback.size(), msg.encoding, msg.image_width,
                           msg.image_height);
  }
}

static rscl_adapter::LidarPacket decode_lidar_message(const rscl_adapter::BagMessage& msg, int point_dim) {
  try {
    return rscl_adapter::decode_lidar_raw_message(msg.topic, msg.payload.data(), msg.payload.size(), point_dim);
  } catch (const std::exception&) {
    if (msg.timestamp_us <= 0) throw;
    const std::vector<unsigned char>& fallback = msg.raw_payload.empty() ? msg.payload : msg.raw_payload;
    return rscl_adapter::decode_lidar_packet(msg.topic, msg.timestamp_us, fallback.data(), fallback.size(), point_dim,
                                            msg.point_step, msg.point_width);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    rscl_adapter::AdapterConfig cfg = rscl_adapter::load_adapter_config(args.adapter_config);
    if (!args.bag.empty()) cfg.bag_path = args.bag;
    if (!args.output_file.empty()) cfg.output_file = args.output_file;
    if (args.max_frames >= 0) cfg.max_frames = args.max_frames;
    if (args.sync_tolerance_ms >= 0.0f) cfg.sync_tolerance_ms = args.sync_tolerance_ms;
    if (args.sync_queue_size > 0) cfg.sync_queue_size = args.sync_queue_size;
    if (args.sync_debug) cfg.sync_debug = true;
    if (args.sync_debug_limit >= 0) cfg.sync_debug_limit = args.sync_debug_limit;
    if (args.has_camera_time_offsets_ms) cfg.camera_time_offsets_ms = args.camera_time_offsets_ms;
    if (args.has_lidar_time_offset_ms) cfg.lidar_time_offset_ms = args.lidar_time_offset_ms;
    if (cfg.bag_path.empty()) throw std::runtime_error("Pass --bag or set bag_path in adapter config");

    std::vector<std::string> included_topics = cfg.camera_topics;
    included_topics.push_back(cfg.lidar_topic);
    std::unique_ptr<rscl_adapter::BagReader> reader = rscl_adapter::create_rscl_bag_reader(cfg.bag_path, included_topics);
    if (!reader || !reader->is_valid()) throw std::runtime_error("Invalid rsclbag: " + cfg.bag_path);

    const bool dry_run = args.decode_only || args.decode_images_only;
    rscl_adapter::BevFusionPipeline pipeline(cfg, dry_run);
    std::map<std::string, std::unique_ptr<rscl_adapter::StatefulCameraDecoder> > camera_decoders;
    for (size_t i = 0; i < cfg.camera_topics.size(); ++i) {
      camera_decoders[cfg.camera_topics[i]].reset(new rscl_adapter::StatefulCameraDecoder());
    }

    std::ofstream output_file;
    if (!cfg.output_file.empty()) {
      output_file.open(cfg.output_file.c_str(), std::ios::out | std::ios::trunc);
      if (!output_file) throw std::runtime_error("Failed to open output file: " + cfg.output_file);
    }

    int frames = 0;
    int decode_errors = 0;
    int camera_debug_dumps = 0;
    rscl_adapter::BagMessage msg;
    while (reader->read_next(&msg)) {
      try {
        std::string output_json;
        bool synced = false;
        if (contains_topic(cfg.camera_topics, msg.topic)) {
          if (!args.dump_camera_debug_dir.empty() && camera_debug_dumps < args.dump_camera_debug_count) {
            ++camera_debug_dumps;
            dump_camera_debug_sample(args, msg, camera_debug_dumps);
          }
          rscl_adapter::CameraPacket camera;
          if (args.decode_only) {
            camera.topic = msg.topic;
            camera.timestamp_us =
                msg.timestamp_us > 0 ? msg.timestamp_us
                                     : rscl_adapter::decode_raw_message_timestamp_us(msg.payload.data(), msg.payload.size());
          } else {
            camera = decode_camera_message(camera_decoders[msg.topic].get(), msg);
          }
          synced = pipeline.add_camera(camera, &output_json);
        } else if (msg.topic == cfg.lidar_topic) {
          rscl_adapter::LidarPacket lidar;
          if (args.decode_only) {
            lidar.topic = msg.topic;
            lidar.timestamp_us =
                msg.timestamp_us > 0 ? msg.timestamp_us
                                     : rscl_adapter::decode_raw_message_timestamp_us(msg.payload.data(), msg.payload.size());
            lidar.point_dim = cfg.point_dim;
          } else {
            lidar = decode_lidar_message(msg, cfg.point_dim);
          }
          synced = pipeline.add_lidar(lidar, &output_json);
        }

        if (!synced) continue;
        ++frames;
        if (dry_run) {
          std::cout << "synced_frames=" << frames << " timestamp_us=" << msg.timestamp_us
                    << " decode_errors=" << decode_errors << std::endl;
        } else if (!output_json.empty()) {
          if (output_file) {
            output_file << output_json << "\n";
            output_file.flush();
          } else {
            std::cout << output_json << std::endl;
          }
        }
        if (cfg.max_frames >= 0 && frames >= cfg.max_frames) break;
      } catch (const rscl_adapter::VideoFrameNotReady&) {
        continue;
      } catch (const std::exception& e) {
        ++decode_errors;
        std::cerr << "failed to process topic=" << msg.topic << " error=" << e.what() << std::endl;
      }
    }

    std::cout << "synced_frames=" << frames << " decode_errors=" << decode_errors << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << std::endl;
    return 1;
  }
}
