#include "rscl_adapter/online_node.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <capnp/serialize.h>
#include <kj/array.h>

#include "ad_msg_idl/ad_sensor/sensor.capnp.h"
#include "ad_rscl/comm/node.h"
#include "ad_rscl/runtime.h"
#include "ad_service_discovery/service_discovery/service_discovery.h"
#include "ad_serde/header.h"
#include "ad_serde/msgmeta.h"
#include "ad_serde/wrapped_type/dynamic_reflection_utils.h"
#include "ad_serde/wrapped_type/raw.h"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace rscl_adapter {
namespace {

using RawMessage = senseAD::serde::RawMessage;
using RawPublisher = senseAD::rscl::comm::Publisher<RawMessage>;
using RawSubscriber = senseAD::rscl::comm::Subscriber<RawMessage>;
using RawReceivedMsg = ReceivedMsg<RawMessage>;
using RawSendMsg = SendMsg<RawMessage>;

static std::string make_process_arg(const std::string& module_name) {
  return module_name.empty() ? std::string("rscl_bevfusion_online_node") : module_name;
}

static void raw_log(const std::string& line) {
#if defined(__linux__)
  (void)write(STDERR_FILENO, line.data(), line.size());
  const char nl = '\n';
  (void)write(STDERR_FILENO, &nl, 1);
#else
  (void)line;
#endif
}

static int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static size_t payload_size_without_rscl_header(const char* data, size_t size, int64_t* stamp_us) {
  if (stamp_us) *stamp_us = 0;
  if (data == nullptr || size < kHeaderSize) return size;

  RsclMsgHeader header;
  if (!ParseHeaderFromBufferTail(data, static_cast<uint64_t>(size), &header) || !header.is_enabled) {
    return size;
  }

  if (stamp_us && header.stamp > 0) *stamp_us = static_cast<int64_t>(header.stamp / 1000ULL);
  size_t trailer_size = kHeaderSize;
  if ((header.feature_flag & RsclMsgHeader::CHECKSOME64) != 0) trailer_size += sizeof(uint64_t);
  return size > trailer_size ? size - trailer_size : size;
}

static bool decode_lidar_capnp_message(const char* data, size_t size, BagMessage* out) {
  if (data == nullptr || size == 0 || out == nullptr) return false;

  try {
    // Copy into aligned Cap'n Proto words because RawMessage exposes bytes as
    // char* and does not guarantee capnp::word alignment.
    const size_t word_count = (size + sizeof(capnp::word) - 1) / sizeof(capnp::word);
    kj::Array<capnp::word> words = kj::heapArray<capnp::word>(word_count);
    std::memset(words.begin(), 0, word_count * sizeof(capnp::word));
    std::memcpy(words.begin(), data, size);

    capnp::FlatArrayMessageReader reader(words.asPtr());
    senseAD::msg::sensor::AdsfiLidarPointCloud::Reader cloud =
        reader.getRoot<senseAD::msg::sensor::AdsfiLidarPointCloud>();
    if (!cloud.hasData()) return false;

    const capnp::Data::Reader point_data = cloud.getData();
    out->payload.assign(point_data.begin(), point_data.end());
    out->raw_payload = out->payload;
    out->point_step = static_cast<int>(cloud.getPointStep());
    out->point_width = static_cast<int>(cloud.getWidth());
    out->message_type = "CAPNP@AdsfiLidarPointCloud_13564378383716798607_2180624248";

    if (cloud.hasHeader()) {
      const senseAD::msg::std_msgs::Header::Reader header = cloud.getHeader();
      if (header.hasTime()) {
        out->timestamp_us = static_cast<int64_t>(header.getTime().getNanoSec() / 1000ULL);
      }
    }

    static std::atomic<bool> logged_success(false);
    if (!logged_success.exchange(true)) {
      raw_log("ad_rscl_backend stage=typed_lidar_decode_enabled point_step=" +
              std::to_string(out->point_step) + " width=" + std::to_string(out->point_width) +
              " data_bytes=" + std::to_string(out->payload.size()));
    }
    return true;
  } catch (const std::exception& e) {
    static std::atomic<bool> logged_failure(false);
    if (!logged_failure.exchange(true)) {
      raw_log("ad_rscl_backend stage=typed_lidar_decode_failed error=" + std::string(e.what()));
    }
    return false;
  }
}

class AdRsclPublisher : public OnlinePublisher {
 public:
  explicit AdRsclPublisher(std::shared_ptr<RawPublisher> publisher) : publisher_(std::move(publisher)) {
    if (!publisher_) throw std::runtime_error("RSCL publisher is null");
  }

  void publish(const std::string& payload) override {
    RawSendMsg msg(payload);
    std::error_code ec = publisher_->Publish(msg);
    if (ec.value() != 0) {
      throw std::runtime_error("Failed to publish RSCL RawMessage, error=" + std::to_string(ec.value()));
    }
  }

 private:
  std::shared_ptr<RawPublisher> publisher_;
};

class AdRsclOnlineNode : public OnlineNode {
 public:
  explicit AdRsclOnlineNode(const AdapterConfig& cfg) : cfg_(cfg) {
    raw_log("ad_rscl_backend stage=init_runtime");
    init_runtime();
    raw_log("ad_rscl_backend stage=create_node name=" + cfg_.node_name);
    node_ = senseAD::rscl::GetCurRuntime()->CreateNode(cfg_.node_name);
    valid_ = static_cast<bool>(node_);
    raw_log(std::string("ad_rscl_backend stage=create_node_done valid=") + (valid_ ? "true" : "false"));
  }

  ~AdRsclOnlineNode() override {
    subscribers_.clear();
    publishers_.clear();
    node_.reset();
    if (runtime_initialized_) {
      senseAD::rscl::GetCurRuntime()->Shutdown();
    }
  }

  bool is_valid() const override { return valid_; }

  void subscribe(const std::string& topic, const std::string& message_type, MessageCallback callback) override {
    if (!node_) throw std::runtime_error("RSCL node is not initialized");
    if (message_type != "RawMessage") {
      throw std::runtime_error("C++ online backend currently supports RawMessage only, requested: " + message_type);
    }

    raw_log("ad_rscl_backend stage=create_subscriber topic=" + topic + " type=" + message_type);
    // RSCL may invoke callbacks for the same topic concurrently. Dynamic
    // reflection expands large LiDAR messages substantially, so allowing many
    // same-topic callbacks to reflect and decode at once can consume gigabytes
    // of memory. Serialize each topic independently; different camera topics
    // and the LiDAR topic can still run in parallel.
    std::shared_ptr<std::mutex> topic_callback_mutex(new std::mutex());
    auto sub = node_->CreateSubscriber<RawMessage>(
        topic, [this, topic, callback, topic_callback_mutex](const std::shared_ptr<RawReceivedMsg>& msg) {
          if (!msg || !msg->IsValid()) return;
          std::lock_guard<std::mutex> topic_callback_lock(*topic_callback_mutex);
          const char* bytes = msg->Bytes();
          const size_t byte_size = msg->ByteSize();
          int64_t header_stamp_us = 0;
          const size_t payload_size = payload_size_without_rscl_header(bytes, byte_size, &header_stamp_us);

          BagMessage out;
          out.topic = topic;
          out.timestamp_us = header_stamp_us;
          out.receive_timestamp_us = now_us();

          // Avoid expanding LiDAR Cap'n Proto data into a ~900 KB JSON string.
          // The typed path extracts timestamp, layout, and point bytes directly.
          if (topic == cfg_.lidar_topic && decode_lidar_capnp_message(bytes, byte_size, &out)) {
            static std::atomic<bool> logged_typed_callback(false);
            const bool log_this_callback = !logged_typed_callback.exchange(true);
            if (log_this_callback) raw_log("ad_rscl_backend stage=typed_lidar_callback_begin");
            callback(out);
            if (log_this_callback) raw_log("ad_rscl_backend stage=typed_lidar_callback_done");
            return;
          }

          std::string reflected_json;
          std::string reflected_type;
          if (bytes && payload_size > 0 && reflect_message_to_json(topic, bytes, payload_size, &reflected_json, &reflected_type)) {
            out.reflected_json = reflected_json;
            out.payload.assign(reflected_json.begin(), reflected_json.end());
            out.message_type = reflected_type;
          } else if (bytes && payload_size > 0) {
            out.payload.assign(reinterpret_cast<const unsigned char*>(bytes),
                               reinterpret_cast<const unsigned char*>(bytes) + payload_size);
            out.message_type = "RawMessage";
          }
          if (bytes && byte_size > 0) {
            out.raw_payload.assign(reinterpret_cast<const unsigned char*>(bytes),
                                   reinterpret_cast<const unsigned char*>(bytes) + byte_size);
          }
          callback(out);
        });

    if (!sub) throw std::runtime_error("Failed to create RSCL subscriber for topic: " + topic);
    subscribers_.push_back(sub);
    raw_log("ad_rscl_backend stage=create_subscriber_done topic=" + topic);
  }

  std::unique_ptr<OnlinePublisher> create_publisher(const std::string& topic,
                                                    const std::string& message_type) override {
    if (!node_) throw std::runtime_error("RSCL node is not initialized");
    if (message_type != "RawMessage") {
      throw std::runtime_error("C++ online backend currently supports RawMessage only, requested: " + message_type);
    }

    raw_log("ad_rscl_backend stage=create_publisher topic=" + topic + " type=" + message_type);
    auto pub = node_->CreatePublisher<RawMessage>(topic);
    if (!pub) throw std::runtime_error("Failed to create RSCL publisher for topic: " + topic);
    publishers_.push_back(pub);
    raw_log("ad_rscl_backend stage=create_publisher_done topic=" + topic);
    return std::unique_ptr<OnlinePublisher>(new AdRsclPublisher(pub));
  }

  void spin() override {
    raw_log("ad_rscl_backend stage=wait_for_shutdown");
    senseAD::rscl::GetCurRuntime()->WaitForShutdown();
  }

 private:
  bool get_topic_msg_meta(const std::string& topic, senseAD::serde::MsgMeta* meta) {
    if (meta == nullptr || !node_) return false;

    {
      std::lock_guard<std::mutex> lock(meta_mutex_);
      std::unordered_map<std::string, senseAD::serde::MsgMeta>::const_iterator it = topic_meta_cache_.find(topic);
      if (it != topic_meta_cache_.end()) {
        *meta = it->second;
        return !meta->IsEmpty() && !meta->msg_descriptor.empty();
      }
    }

    senseAD::serde::MsgMeta discovered;
    bool ok = false;
    if (node_->GetServiceDiscovery() != nullptr) {
      ok = node_->GetServiceDiscovery()->GetMsgMetabyTopic(topic, &discovered);
    }
    if (!ok || discovered.IsEmpty() || discovered.msg_descriptor.empty()) return false;

    {
      std::lock_guard<std::mutex> lock(meta_mutex_);
      topic_meta_cache_[topic] = discovered;
      if (logged_meta_topics_.insert(topic).second) {
        raw_log("ad_rscl_backend stage=discovered_msg_meta topic=" + topic + " type=" + discovered.msg_type +
                " descriptor_bytes=" + std::to_string(discovered.msg_descriptor.size()));
      }
    }
    *meta = discovered;
    return true;
  }

  bool reflect_message_to_json(const std::string& topic, const char* data, size_t size, std::string* json,
                               std::string* msg_type) {
    if (data == nullptr || size == 0 || json == nullptr) return false;

    senseAD::serde::MsgMeta meta;
    if (!get_topic_msg_meta(topic, &meta)) return false;

    try {
      senseAD::base::optional<std::string> reflected =
          senseAD::serde::JsonStringByDynamicReflection(meta.msg_type, meta.msg_descriptor, data, size);
      if (!reflected) return false;
      *json = reflected.value();
      if (msg_type != nullptr) *msg_type = meta.msg_type;
      return true;
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(meta_mutex_);
      if (logged_reflection_errors_.insert(topic).second) {
        raw_log("ad_rscl_backend stage=reflect_message_failed topic=" + topic + " type=" + meta.msg_type +
                " error=" + e.what());
      }
      return false;
    }
  }

  void init_runtime() {
    raw_log("ad_rscl_backend stage=get_runtime");
    if (senseAD::rscl::GetCurRuntime()->OK()) {
      raw_log("ad_rscl_backend stage=runtime_already_ok");
      runtime_initialized_ = false;
      return;
    }

    process_arg_ = make_process_arg(cfg_.module_name);
    argv_storage_.clear();
    argv_storage_.push_back(process_arg_);
    argv_.clear();
    for (size_t i = 0; i < argv_storage_.size(); ++i) {
      argv_.push_back(const_cast<char*>(argv_storage_[i].c_str()));
    }
    int argc = static_cast<int>(argv_.size());
    raw_log("ad_rscl_backend stage=runtime_init argc=" + std::to_string(argc) + " argv0=" + argv_storage_[0]);
    senseAD::rscl::GetCurRuntime()->Init(argc, argv_.data());
    runtime_initialized_ = true;
    raw_log(std::string("ad_rscl_backend stage=runtime_init_done ok=") +
            (senseAD::rscl::GetCurRuntime()->OK() ? "true" : "false"));
  }

  AdapterConfig cfg_;
  std::unique_ptr<senseAD::rscl::comm::Node> node_;
  std::vector<std::shared_ptr<RawSubscriber> > subscribers_;
  std::vector<std::shared_ptr<RawPublisher> > publishers_;
  std::mutex meta_mutex_;
  std::unordered_map<std::string, senseAD::serde::MsgMeta> topic_meta_cache_;
  std::set<std::string> logged_meta_topics_;
  std::set<std::string> logged_reflection_errors_;
  std::vector<std::string> argv_storage_;
  std::vector<char*> argv_;
  std::string process_arg_;
  bool runtime_initialized_ = false;
  bool valid_ = false;
};

}  // namespace

std::unique_ptr<OnlineNode> create_rscl_online_node(const AdapterConfig& cfg) {
  return std::unique_ptr<OnlineNode>(new AdRsclOnlineNode(cfg));
}

// The C factory must not call the externally visible C++ factory by symbol.
// Without RTLD_DEEPBIND that name can interpose with the executable's dlopen
// wrapper and recursively load this backend until stack overflow. Keep this
// raw factory translation-unit local so the call cannot be interposed.
static OnlineNode* create_rscl_online_node_backend_raw(const AdapterConfig& cfg) {
  return new AdRsclOnlineNode(cfg);
}

}  // namespace rscl_adapter

extern "C" rscl_adapter::OnlineNode* rscl_adapter_create_online_node(
    const rscl_adapter::AdapterConfig* cfg) {
  if (cfg == nullptr) return nullptr;
  return rscl_adapter::create_rscl_online_node_backend_raw(*cfg);
}
