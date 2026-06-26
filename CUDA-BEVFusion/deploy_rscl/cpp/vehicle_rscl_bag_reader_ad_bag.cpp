#include "rscl_adapter/bag_reader.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ad_bag/bag_reader.h"
#include "ad_serde/wrapped_type/dynamic_reflection_utils.h"
#include "rscl_adapter/codecs.hpp"
#include "rscl_adapter/json.hpp"

namespace rscl_adapter {
namespace {

static int64_t ns_to_us(uint64_t ns) {
  return static_cast<int64_t>(ns / 1000ULL);
}

static const JsonValue* first_json_value(const JsonValue& value, const char* a, const char* b = nullptr,
                                         const char* c = nullptr, const char* d = nullptr) {
  const char* keys[] = {a, b, c, d};
  for (int i = 0; i < 4; ++i) {
    if (!keys[i]) continue;
    const JsonValue* item = value.get(keys[i]);
    if (item && !item->is_null()) return item;
  }
  return nullptr;
}

static const JsonValue* first_json_value_nested(const JsonValue& value, const char* a, const char* b = nullptr,
                                                const char* c = nullptr, const char* d = nullptr) {
  const JsonValue* item = first_json_value(value, a, b, c, d);
  if (item) return item;
  if (!value.is_object()) return nullptr;
  for (std::map<std::string, JsonValue>::const_iterator it = value.object.begin(); it != value.object.end(); ++it) {
    item = first_json_value_nested(it->second, a, b, c, d);
    if (item) return item;
  }
  return nullptr;
}

static void fill_optional_metadata_from_json(const JsonValue& root, BagMessage* message) {
  if (!root.is_object() || message == nullptr) return;

  const JsonValue* encoding = first_json_value_nested(root, "encoding", "format", "pixelFormat", "pixel_format");
  if (encoding && encoding->is_string()) {
    message->encoding = encoding->string;
  } else {
    const JsonValue* video_format = first_json_value_nested(root, "videoFormat", "codec", "codecName", "frameType");
    if (video_format && video_format->is_string()) message->encoding = video_format->string;
  }

  const JsonValue* width = first_json_value_nested(root, "width", "cols", "imageWidth", "image_width");
  const JsonValue* height = first_json_value_nested(root, "height", "rows", "imageHeight", "image_height");
  if (width && width->is_number()) message->image_width = static_cast<int>(width->as_number());
  if (height && height->is_number()) message->image_height = static_cast<int>(height->as_number());

  const JsonValue* point_step = first_json_value_nested(root, "pointStep", "point_step", "stride");
  const JsonValue* point_width = first_json_value_nested(root, "pointWidth", "point_width");
  if (point_step && point_step->is_number()) message->point_step = static_cast<int>(point_step->as_number());
  if (point_width && point_width->is_number()) message->point_width = static_cast<int>(point_width->as_number());

  const JsonValue* points = first_json_value_nested(root, "points");
  if (message->point_width <= 0 && points && points->is_array()) {
    message->point_width = static_cast<int>(points->array.size());
  }
}

class VehicleRsclBagReader : public BagReader {
 public:
  VehicleRsclBagReader(const std::string& bag_path, const std::vector<std::string>& included_topics)
  {
    senseAD::bag::BagReaderAttribute attr;
    for (size_t i = 0; i < included_topics.size(); ++i) {
      attr.included_channels.insert(included_topics[i]);
    }
    attr.iterator_type = senseAD::bag::BagIteratorType::READ_NEXT_BY_TIMESTAMP_ORDER;
    reader_.reset(new senseAD::bag::BagReader(bag_path, attr));
    valid_ = reader_ && reader_->IsValid();
    if (valid_) {
      const std::set<std::string> channels = reader_->GetChannelList();
      std::cerr << "RSCL bag opened: " << bag_path << " channels=" << channels.size() << std::endl;
      for (std::set<std::string>::const_iterator it = channels.begin(); it != channels.end(); ++it) {
        if (included_topics.empty() || attr.included_channels.count(*it) > 0) {
          std::cerr << "  channel=" << *it << " type=" << reader_->GetMessageType(*it)
                    << " count=" << reader_->GetMessageNumber(*it) << std::endl;
        }
      }
    }
  }

  bool is_valid() const override {
    return valid_;
  }

  bool read_next(BagMessage* message) override {
    if (message == nullptr) throw std::invalid_argument("message must not be null");
    if (!valid_ || !reader_) return false;

    senseAD::bag::ReadedMessage msg;
    while (reader_->ReadNextMessage(&msg)) {
      try {
        BagMessage out;
        out.topic = msg.channel_name;
        out.timestamp_us = ns_to_us(msg.timestamp);
        out.message_type = reader_->GetMessageType(out.topic);
        out.raw_payload.assign(msg.message_buffer.begin(), msg.message_buffer.end());

        std::string json_payload = message_to_json(out.topic, msg);
        out.reflected_json = json_payload;
        out.payload.assign(json_payload.begin(), json_payload.end());
        JsonValue root = parse_json(json_payload);
        try {
          out.timestamp_us =
              decode_raw_message_timestamp_us(reinterpret_cast<const unsigned char*>(json_payload.data()),
                                              json_payload.size());
        } catch (const std::exception&) {
          // Keep the bag timestamp when the payload does not expose a timestamp field.
        }
        fill_optional_metadata_from_json(root, &out);

        if (root.is_object()) {
          const JsonValue* ts = first_json_value(root, "timestamp_us", "timestampUs", "time_us", "timeUs");
          if (!ts) ts = first_json_value(root, "timestamp", "time", "headerTimestamp", "header_time");
          if (!ts) {
            // Keep the bag timestamp as the authoritative timestamp if the message
            // payload itself does not expose one.
          }
        }

        *message = std::move(out);
        return true;
      } catch (const std::exception&) {
        // If dynamic reflection fails for a message, fall back to the raw buffer.
        // This still preserves topic/timestamp for diagnostics and lets the caller
        // attempt alternate decoding paths.
        BagMessage out;
        out.topic = msg.channel_name;
        out.timestamp_us = ns_to_us(msg.timestamp);
        out.message_type = reader_->GetMessageType(out.topic);
        out.raw_payload.assign(msg.message_buffer.begin(), msg.message_buffer.end());
        out.payload.assign(msg.message_buffer.begin(), msg.message_buffer.end());
        *message = std::move(out);
        return true;
      }
    }

    return false;
  }

 private:
  struct TopicDecoder {
    std::string message_type;
    std::string descriptor;
    std::unique_ptr<senseAD::serde::DynamicMessageReader> reader;
  };

  std::string message_to_json(const std::string& topic, const senseAD::bag::ReadedMessage& msg) {
    TopicDecoder& decoder = get_or_create_decoder(topic);
    auto parsed = decoder.reader->ParseMessage(msg.message_buffer.data(), msg.message_buffer.size());
    if (!parsed) {
      throw std::runtime_error("Failed to parse RSCL message buffer for topic: " + topic);
    }

    auto json = senseAD::serde::JsonStringByDynamicReflection(parsed.value());
    if (!json) {
      throw std::runtime_error("Failed to convert RSCL message to JSON for topic: " + topic);
    }

    return json.value();
  }

  TopicDecoder& get_or_create_decoder(const std::string& topic) {
    std::unordered_map<std::string, TopicDecoder>::iterator it = decoders_.find(topic);
    if (it != decoders_.end()) return it->second;

    TopicDecoder decoder;
    decoder.message_type = reader_->GetMessageType(topic);
    decoder.descriptor = reader_->GetDescriptor(topic);
    if (decoder.message_type.empty()) {
      throw std::runtime_error("RSCL bag does not expose a message type for topic: " + topic);
    }
    if (decoder.descriptor.empty()) {
      throw std::runtime_error("RSCL bag does not expose a descriptor for topic: " + topic);
    }
    decoder.reader.reset(new senseAD::serde::DynamicMessageReader(decoder.message_type, decoder.descriptor));
    it = decoders_.insert(std::make_pair(topic, std::move(decoder))).first;
    return it->second;
  }

  std::unique_ptr<senseAD::bag::BagReader> reader_;
  std::unordered_map<std::string, TopicDecoder> decoders_;
  bool valid_ = false;
};

}  // namespace

std::unique_ptr<BagReader> create_rscl_bag_reader(const std::string& bag_path,
                                                  const std::vector<std::string>& included_topics) {
  return std::unique_ptr<BagReader>(new VehicleRsclBagReader(bag_path, included_topics));
}

}  // namespace rscl_adapter
