#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

#include "rscl_adapter/codecs.hpp"
#include "rscl_adapter/config.hpp"
#include "rscl_adapter/online_node.hpp"
#include "rscl_adapter/pipeline.hpp"

namespace {

enum class TimestampSource {
  kAuto,
  kPayload,
  kHeader,
  kReceive,
};

#if defined(__linux__)
static void segfault_handler(int signum) {
  void* trace[64];
  int size = backtrace(trace, 64);
  const char header[] = "fatal: caught SIGSEGV, backtrace:\n";
  (void)write(STDERR_FILENO, header, sizeof(header) - 1);
  backtrace_symbols_fd(trace, size, STDERR_FILENO);
  signal(signum, SIG_DFL);
  raise(signum);
}

static void install_signal_handlers() {
  signal(SIGSEGV, segfault_handler);
}
#else
static void install_signal_handlers() {}
#endif

struct Args {
  std::string adapter_config = "deploy_rscl/configs/bevfusion_rscl.yaml";
  std::string output_file;
  int max_frames = -1;
  float sync_tolerance_ms = -1.0f;
  int sync_queue_size = -1;
  int sync_debug_limit = -1;
  std::vector<float> camera_time_offsets_ms;
  bool has_camera_time_offsets_ms = false;
  float lidar_time_offset_ms = 0.0f;
  bool has_lidar_time_offset_ms = false;
  float print_rate = 1.0f;
  bool decode_only = false;
  bool decode_images_only = false;
  bool sync_debug = false;
  bool message_debug = false;
  int message_debug_limit = 100;
  TimestampSource timestamp_source = TimestampSource::kAuto;
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

static TimestampSource parse_timestamp_source(const std::string& text) {
  if (text == "auto") return TimestampSource::kAuto;
  if (text == "payload") return TimestampSource::kPayload;
  if (text == "header") return TimestampSource::kHeader;
  if (text == "receive" || text == "arrival") return TimestampSource::kReceive;
  throw std::runtime_error("Invalid --timestamp-source: " + text +
                           ". Expected one of: auto, payload, header, receive");
}

static const char* timestamp_source_name(TimestampSource source) {
  switch (source) {
    case TimestampSource::kAuto:
      return "auto";
    case TimestampSource::kPayload:
      return "payload";
    case TimestampSource::kHeader:
      return "header";
    case TimestampSource::kReceive:
      return "receive";
  }
  return "unknown";
}

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--adapter-config") && i + 1 < argc) {
      args.adapter_config = argv[++i];
    } else if (arg_eq(argv[i], "--output-file") && i + 1 < argc) {
      args.output_file = argv[++i];
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
    } else if (arg_eq(argv[i], "--message-debug")) {
      args.message_debug = true;
    } else if (arg_eq(argv[i], "--message-debug-limit") && i + 1 < argc) {
      args.message_debug_limit = std::atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--timestamp-source") && i + 1 < argc) {
      args.timestamp_source = parse_timestamp_source(argv[++i]);
    } else if (arg_eq(argv[i], "--camera-time-offsets-ms") && i + 1 < argc) {
      args.camera_time_offsets_ms = parse_float_csv(argv[++i]);
      args.has_camera_time_offsets_ms = true;
    } else if (arg_eq(argv[i], "--lidar-time-offset-ms") && i + 1 < argc) {
      args.lidar_time_offset_ms = static_cast<float>(std::atof(argv[++i]));
      args.has_lidar_time_offset_ms = true;
    } else if (arg_eq(argv[i], "--print-rate") && i + 1 < argc) {
      args.print_rate = static_cast<float>(std::atof(argv[++i]));
    } else if (arg_eq(argv[i], "--decode-only")) {
      args.decode_only = true;
    } else if (arg_eq(argv[i], "--decode-images-only") || arg_eq(argv[i], "--dry-run")) {
      args.decode_images_only = true;
    } else if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) {
      std::cout
          << "Usage: rscl_bevfusion_online_node --adapter-config <yaml> [options]\n"
          << "Options:\n"
          << "  --max-frames N          Exit after N synchronized frames, default unlimited\n"
          << "  --output-file PATH      Also write JSONL outputs to a file\n"
          << "  --sync-tolerance-ms N   Override camera/lidar sync tolerance in milliseconds\n"
          << "  --sync-queue-size N     Override per-topic sync queue size\n"
          << "  --sync-debug            Print sync nearest-difference diagnostics\n"
          << "  --sync-debug-limit N    Limit sync diagnostic lines\n"
          << "  --message-debug         Print received online message topic, bytes, and timestamp\n"
          << "  --message-debug-limit N Limit received-message diagnostic lines, default 100\n"
          << "  --timestamp-source S    Timestamp used for online sync: auto, payload, header, receive\n"
          << "                          Use receive for rsclbag play when RSCL header stamps differ by topic\n"
          << "  --camera-time-offsets-ms CSV  Override per-camera offsets, e.g. 450,450,0,0,0,450\n"
          << "  --lidar-time-offset-ms N      Override lidar timestamp offset in milliseconds\n"
          << "  --print-rate HZ         Status print frequency. Set 0 to print every synced frame\n"
          << "  --decode-only           Subscribe and synchronize timestamps only, no image/TensorRT decode\n"
          << "  --decode-images-only    Decode image/lidar packets and synchronize only, no CUDA inference\n"
          << "  --dry-run               Alias for --decode-images-only\n";
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

static int64_t try_decode_payload_timestamp_us(const rscl_adapter::BagMessage& msg) {
  try {
    if (!msg.payload.empty()) {
      return rscl_adapter::decode_raw_message_timestamp_us(msg.payload.data(), msg.payload.size());
    }
  } catch (const std::exception&) {
  }
  return 0;
}

static int64_t message_timestamp_us(const rscl_adapter::BagMessage& msg) {
  const int64_t payload_timestamp_us = try_decode_payload_timestamp_us(msg);
  if (payload_timestamp_us > 0) return payload_timestamp_us;
  return msg.timestamp_us;
}

static int64_t select_message_timestamp_us(const rscl_adapter::BagMessage& msg, TimestampSource source) {
  if (source == TimestampSource::kReceive) {
    if (msg.receive_timestamp_us > 0) return msg.receive_timestamp_us;
    return message_timestamp_us(msg);
  }
  if (source == TimestampSource::kHeader) return msg.timestamp_us;
  const int64_t payload_timestamp_us = try_decode_payload_timestamp_us(msg);
  if (source == TimestampSource::kPayload) return payload_timestamp_us;
  if (payload_timestamp_us > 0) return payload_timestamp_us;
  return msg.timestamp_us;
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

class OnlineRunner {
 public:
  OnlineRunner(const rscl_adapter::AdapterConfig& cfg, bool dry_run, const Args& args)
      : cfg_(cfg),
        dry_run_(dry_run),
        message_debug_(args.message_debug),
        message_debug_limit_(args.message_debug_limit),
        timestamp_source_(args.timestamp_source),
        pipeline_(cfg, dry_run),
        print_interval_s_(args.print_rate > 0.0f ? 1.0 / args.print_rate : 0.0) {
    for (size_t i = 0; i < cfg_.camera_topics.size(); ++i) {
      camera_decoders_[cfg_.camera_topics[i]].reset(new rscl_adapter::StatefulCameraDecoder());
    }
    if (!cfg_.output_file.empty()) {
      output_file_.open(cfg_.output_file.c_str(), std::ios::out | std::ios::trunc);
      if (!output_file_) throw std::runtime_error("Failed to open output file: " + cfg_.output_file);
    }
  }

  void set_publisher(std::unique_ptr<rscl_adapter::OnlinePublisher> publisher) { publisher_ = std::move(publisher); }

  void on_message(const rscl_adapter::BagMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
      print_message_debug(msg);
      std::string output_json;
      bool synced = false;
      int64_t selected_timestamp_us = 0;
      if (contains_topic(cfg_.camera_topics, msg.topic)) {
        rscl_adapter::CameraPacket camera;
        if (decode_only_) {
          camera.topic = msg.topic;
          camera.timestamp_us = select_message_timestamp_us(msg, timestamp_source_);
        } else {
          camera = decode_camera_message(camera_decoders_[msg.topic].get(), msg);
          if (timestamp_source_ == TimestampSource::kReceive) {
            camera.timestamp_us = select_message_timestamp_us(msg, timestamp_source_);
          }
        }
        selected_timestamp_us = camera.timestamp_us;
        synced = pipeline_.add_camera(camera, &output_json);
      } else if (msg.topic == cfg_.lidar_topic) {
        rscl_adapter::LidarPacket lidar;
        if (decode_only_) {
          lidar.topic = msg.topic;
          lidar.timestamp_us = select_message_timestamp_us(msg, timestamp_source_);
          lidar.point_dim = cfg_.point_dim;
        } else {
          lidar = decode_lidar_message(msg, cfg_.point_dim);
          if (timestamp_source_ == TimestampSource::kReceive) {
            lidar.timestamp_us = select_message_timestamp_us(msg, timestamp_source_);
          }
        }
        selected_timestamp_us = lidar.timestamp_us;
        synced = pipeline_.add_lidar(lidar, &output_json);
      }

      if (!synced) return;
      ++synced_frames_;
      if (dry_run_) {
        print_status(selected_timestamp_us, -1);
      } else if (!output_json.empty() || cfg_.publish_empty_frame) {
        if (publisher_) publisher_->publish(output_json);
        if (output_file_) {
          output_file_ << output_json << "\n";
          output_file_.flush();
        }
        print_status(selected_timestamp_us, count_objects(output_json));
      }
      if (cfg_.max_frames >= 0 && synced_frames_ >= cfg_.max_frames) {
        std::cerr << "Reached max_frames=" << cfg_.max_frames << ". Exiting." << std::endl;
        std::exit(0);
      }
    } catch (const rscl_adapter::VideoFrameNotReady&) {
      return;
    } catch (const std::exception& e) {
      ++decode_errors_;
      std::cerr << "failed to process topic=" << msg.topic << " error=" << e.what() << std::endl;
    }
  }

  void set_decode_only(bool value) { decode_only_ = value; }

 private:
  void print_message_debug(const rscl_adapter::BagMessage& msg) {
    if (!message_debug_) return;
    if (message_debug_limit_ >= 0 && message_debug_count_ >= message_debug_limit_) return;
    ++message_debug_count_;
    const int64_t payload_timestamp_us = try_decode_payload_timestamp_us(msg);
    const int64_t selected_timestamp_us = select_message_timestamp_us(msg, timestamp_source_);
    std::cout << "message_debug topic=" << msg.topic << " payload_bytes=" << msg.payload.size()
              << " raw_bytes=" << msg.raw_payload.size() << " header_timestamp_us=" << msg.timestamp_us
              << " payload_timestamp_us=" << payload_timestamp_us << " receive_timestamp_us=" << msg.receive_timestamp_us
              << " selected_timestamp_us=" << selected_timestamp_us
              << " timestamp_source=" << timestamp_source_name(timestamp_source_)
              << " type=" << msg.message_type << std::endl;
  }

  static int count_objects(const std::string& output_json) {
    size_t pos = 0;
    int count = 0;
    while ((pos = output_json.find("\"box\"", pos)) != std::string::npos) {
      ++count;
      pos += 5;
    }
    return count;
  }

  void print_status(int64_t timestamp_us, int objects) {
    const double now = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
    if (last_print_time_ > 0.0 && print_interval_s_ > 0.0 && now - last_print_time_ < print_interval_s_) return;
    last_print_time_ = now;
    std::cout << "synced_frames=" << synced_frames_ << " timestamp_us=" << timestamp_us
              << " decode_errors=" << decode_errors_;
    if (objects >= 0) std::cout << " objects=" << objects;
    std::cout << std::endl;
  }

  rscl_adapter::AdapterConfig cfg_;
  bool dry_run_ = false;
  bool decode_only_ = false;
  bool message_debug_ = false;
  int message_debug_limit_ = 100;
  int message_debug_count_ = 0;
  TimestampSource timestamp_source_ = TimestampSource::kAuto;
  rscl_adapter::BevFusionPipeline pipeline_;
  std::map<std::string, std::unique_ptr<rscl_adapter::StatefulCameraDecoder> > camera_decoders_;
  std::unique_ptr<rscl_adapter::OnlinePublisher> publisher_;
  std::ofstream output_file_;
  std::mutex mutex_;
  int synced_frames_ = 0;
  int decode_errors_ = 0;
  double print_interval_s_ = 1.0;
  double last_print_time_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    install_signal_handlers();
    std::cerr << "online_node stage=parse_args" << std::endl;
    Args args = parse_args(argc, argv);
    std::cerr << "online_node stage=load_config path=" << args.adapter_config << std::endl;
    rscl_adapter::AdapterConfig cfg = rscl_adapter::load_adapter_config(args.adapter_config);
    if (!args.output_file.empty()) cfg.output_file = args.output_file;
    if (args.max_frames >= 0) cfg.max_frames = args.max_frames;
    if (args.sync_tolerance_ms >= 0.0f) cfg.sync_tolerance_ms = args.sync_tolerance_ms;
    if (args.sync_queue_size > 0) cfg.sync_queue_size = args.sync_queue_size;
    if (args.sync_debug) cfg.sync_debug = true;
    if (args.sync_debug_limit >= 0) cfg.sync_debug_limit = args.sync_debug_limit;
    if (args.has_camera_time_offsets_ms) cfg.camera_time_offsets_ms = args.camera_time_offsets_ms;
    if (args.has_lidar_time_offset_ms) cfg.lidar_time_offset_ms = args.lidar_time_offset_ms;

    const bool dry_run = args.decode_only || args.decode_images_only;
    std::cerr << "online_node stage=create_pipeline dry_run=" << (dry_run ? "true" : "false") << std::endl;
    OnlineRunner runner(cfg, dry_run, args);
    runner.set_decode_only(args.decode_only);

    std::cerr << "online_node stage=create_rscl_online_node" << std::endl;
    std::unique_ptr<rscl_adapter::OnlineNode> node = rscl_adapter::create_rscl_online_node(cfg);
    if (!node || !node->is_valid()) throw std::runtime_error("Invalid RSCL online node");
    std::cerr << "online_node stage=create_publisher dry_run=" << (dry_run ? "true" : "false") << std::endl;
    if (!dry_run) runner.set_publisher(node->create_publisher(cfg.output_topic, cfg.output_message_type));

    std::cerr << "online_node stage=create_subscribers" << std::endl;
    for (size_t i = 0; i < cfg.camera_topics.size(); ++i) {
      const std::string topic = cfg.camera_topics[i];
      node->subscribe(topic, cfg.input_message_type,
                      [&runner](const rscl_adapter::BagMessage& msg) { runner.on_message(msg); });
    }
    node->subscribe(cfg.lidar_topic, cfg.input_message_type,
                    [&runner](const rscl_adapter::BagMessage& msg) { runner.on_message(msg); });

    std::cout << "RSCL BEVFusion online node started"
              << " node=" << cfg.node_name << " module=" << cfg.module_name << " cameras=" << cfg.camera_topics.size()
              << " lidar=" << cfg.lidar_topic << " output=" << cfg.output_topic
              << " sync_tolerance_ms=" << cfg.sync_tolerance_ms << " dry_run=" << (dry_run ? "true" : "false")
              << " timestamp_source=" << timestamp_source_name(args.timestamp_source)
              << std::endl;
    std::cerr << "online_node stage=spin" << std::endl;
    node->spin();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << std::endl;
    return 1;
  }
}
