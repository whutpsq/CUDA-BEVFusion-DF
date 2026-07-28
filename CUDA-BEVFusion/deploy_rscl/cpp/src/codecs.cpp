#include "rscl_adapter/codecs.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "rscl_adapter/json.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#if defined(RSCL_HAVE_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}
#endif

namespace rscl_adapter {
namespace {

static std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

static std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

static std::string base64_encode(const unsigned char* data, size_t size) {
  static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((size + 2) / 3) * 4);
  for (size_t i = 0; i < size; i += 3) {
    unsigned int v = static_cast<unsigned int>(data[i]) << 16;
    if (i + 1 < size) v |= static_cast<unsigned int>(data[i + 1]) << 8;
    if (i + 2 < size) v |= static_cast<unsigned int>(data[i + 2]);
    out.push_back(table[(v >> 18) & 0x3F]);
    out.push_back(table[(v >> 12) & 0x3F]);
    out.push_back(i + 1 < size ? table[(v >> 6) & 0x3F] : '=');
    out.push_back(i + 2 < size ? table[v & 0x3F] : '=');
  }
  return out;
}

static std::vector<unsigned char> base64_decode(const std::string& text) {
  int table[256];
  std::fill(table, table + 256, -1);
  const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (size_t i = 0; i < alphabet.size(); ++i) table[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
  std::vector<unsigned char> out;
  int val = 0;
  int valb = -8;
  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (std::isspace(c)) continue;
    if (c == '=') break;
    int d = table[c];
    if (d < 0) throw std::runtime_error("Invalid base64 payload");
    val = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

static bool hex_value(char c, unsigned int* value) {
  if (value == nullptr) return false;
  if (c >= '0' && c <= '9') {
    *value = static_cast<unsigned int>(c - '0');
    return true;
  }
  if (c >= 'a' && c <= 'f') {
    *value = static_cast<unsigned int>(c - 'a' + 10);
    return true;
  }
  if (c >= 'A' && c <= 'F') {
    *value = static_cast<unsigned int>(c - 'A' + 10);
    return true;
  }
  return false;
}

static bool decode_rscl_echo_hex_string(const std::string& text, std::vector<unsigned char>* out) {
  if (out == nullptr) return false;
  std::vector<unsigned char> decoded;
  size_t i = 0;
  bool saw_token = false;

  while (i < text.size()) {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i >= text.size()) break;

    uint32_t value = 0;
    int digits = 0;
    while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
      unsigned int nibble = 0;
      if (!hex_value(text[i], &nibble)) return false;
      value = (value << 4) | nibble;
      ++digits;
      ++i;
    }

    if (digits == 0) return false;
    decoded.push_back(static_cast<unsigned char>(value & 0xFFu));
    saw_token = true;
  }

  if (!saw_token || decoded.size() < 4) return false;
  out->swap(decoded);
  return true;
}

static std::string video_format_to_encoding(const JsonValue* value) {
  if (value == nullptr || value->is_null()) return "";
  if (value->is_string()) return value->string;
  if (value->is_number()) {
    const int format = static_cast<int>(value->as_number());
    if (format == 6) return "hevc";
    if (format == 5) return "h264";
  }
  return "";
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

static int64_t normalize_timestamp_us(double value) {
  int64_t v = static_cast<int64_t>(value);
  if (v > 10000000000000000LL) return v / 1000;
  if (v > 10000000000000LL) return v;
  if (v > 10000000000LL) return v * 1000;
  return v;
}

static const JsonValue* first_number_value(const JsonValue& value, const char* a, const char* b = nullptr,
                                           const char* c = nullptr, const char* d = nullptr) {
  const JsonValue* item = first_json_value(value, a, b, c, d);
  return item && (item->is_number() || item->is_string()) ? item : nullptr;
}

static bool timestamp_from_object(const JsonValue& value, int64_t* timestamp_us) {
  if (!value.is_object() || timestamp_us == nullptr) return false;

  const JsonValue* ts = first_number_value(value, "timestamp_us", "timestampUs", "time_us", "timeUs");
  if (ts) {
    *timestamp_us = static_cast<int64_t>(ts->as_number());
    return true;
  }

  ts = first_number_value(value, "timestamp_ns", "timestampNs", "time_ns", "timeNs");
  if (!ts) ts = first_number_value(value, "nanoSec");
  if (ts) {
    *timestamp_us = static_cast<int64_t>(ts->as_number()) / 1000;
    return true;
  }

  ts = first_number_value(value, "nsec", "nsecs", "nanosec", "nanosecs");
  const JsonValue* sec = first_number_value(value, "sec", "secs", "second", "seconds");
  if (sec) {
    int64_t usec = static_cast<int64_t>(sec->as_number()) * 1000000LL;
    if (ts) usec += static_cast<int64_t>(ts->as_number()) / 1000;
    *timestamp_us = usec;
    return true;
  }

  ts = first_number_value(value, "timestamp_ms", "timestampMs", "msec", "millisecond");
  if (ts) {
    *timestamp_us = static_cast<int64_t>(ts->as_number() * 1000.0);
    return true;
  }

  return false;
}

static int64_t timestamp_from_json(const JsonValue& value) {
  const JsonValue* ts = first_json_value_nested(value, "timestamp_us", "timestampUs", "time_us", "timeUs");
  int64_t object_timestamp_us = 0;
  if (ts && ts->is_object() && timestamp_from_object(*ts, &object_timestamp_us)) return object_timestamp_us;
  if (ts && !ts->is_object()) return static_cast<int64_t>(ts->as_number());
  ts = first_json_value_nested(value, "timestamp_ns", "timestampNs", "time_ns", "timeNs");
  if (ts && ts->is_object() && timestamp_from_object(*ts, &object_timestamp_us)) return object_timestamp_us;
  if (ts && !ts->is_object()) return static_cast<int64_t>(ts->as_number()) / 1000;
  ts = first_json_value_nested(value, "timestamp_ms", "timestampMs");
  if (ts && ts->is_object() && timestamp_from_object(*ts, &object_timestamp_us)) return object_timestamp_us;
  if (ts && !ts->is_object()) return static_cast<int64_t>(ts->as_number() * 1000.0);
  ts = first_json_value_nested(value, "timestamp", "time", "headerTimestamp", "header_time");
  if (ts && ts->is_object() && timestamp_from_object(*ts, &object_timestamp_us)) return object_timestamp_us;
  if (ts && !ts->is_object()) return normalize_timestamp_us(ts->as_number());

  const JsonValue* header = first_json_value_nested(value, "header");
  if (header && header->is_object()) return timestamp_from_json(*header);
  throw std::runtime_error("RawMessage JSON does not contain a timestamp");
}

static std::vector<unsigned char> bytes_from_json_data(const JsonValue& value) {
  const JsonValue* data = first_json_value_nested(value, "data", "raw", "buffer", "payload");
  if (!data) data = first_json_value_nested(value, "pointsData", "point_data", "points_data", "imageData");
  if (!data) throw std::runtime_error("RawMessage JSON does not contain byte payload data");
  if (data->is_string()) {
    std::vector<unsigned char> raw;
    if (decode_rscl_echo_hex_string(data->string, &raw)) return raw;
    return base64_decode(data->string);
  }
  if (data->is_array()) {
    std::vector<unsigned char> out;
    out.reserve(data->array.size());
    for (size_t i = 0; i < data->array.size(); ++i) {
      double value = data->array[i].as_number(-1.0);
      if (value < 0.0 || value > 255.0) throw std::runtime_error("RawMessage JSON data array contains non-byte value");
      out.push_back(static_cast<unsigned char>(value));
    }
    return out;
  }
  throw std::runtime_error("RawMessage JSON data must be a base64 string or byte array");
}

static void infer_camera_dimensions_from_topic(const std::string& topic, int* width, int* height) {
  if (width == nullptr || height == nullptr) return;
  if (*width > 0 && *height > 0) return;

  const std::string topic_l = lower(topic);
  if (topic_l.find("center_camera_fov120") != std::string::npos) {
    *width = 3840;
    *height = 2160;
    return;
  }

  if (topic_l.find("/sensor/camera/") != std::string::npos && topic_l.find("/encode") != std::string::npos) {
    *width = 1920;
    *height = 1280;
  }
}

static std::vector<float> points_from_json_array(const JsonValue& points, int* point_dim) {
  if (!points.is_array()) throw std::runtime_error("points must be an array");
  std::vector<float> out;
  if (points.array.empty()) {
    *point_dim = 5;
    return out;
  }
  if (points.array[0].is_array()) {
    *point_dim = static_cast<int>(points.array[0].array.size());
    for (size_t i = 0; i < points.array.size(); ++i) {
      for (size_t j = 0; j < points.array[i].array.size(); ++j) {
        out.push_back(static_cast<float>(points.array[i].array[j].as_number()));
      }
    }
  } else {
    *point_dim = 5;
    for (size_t i = 0; i < points.array.size(); ++i) out.push_back(static_cast<float>(points.array[i].as_number()));
  }
  return out;
}

static unsigned char clamp_u8(int v) {
  return static_cast<unsigned char>(std::max(0, std::min(255, v)));
}

static bool looks_like_video_packet(const unsigned char* data, size_t size, const std::string& encoding) {
  const std::string enc = lower(encoding);
  if (enc.find("h264") != std::string::npos || enc.find("h265") != std::string::npos || enc.find("hevc") != std::string::npos ||
      enc.find("video") != std::string::npos) {
    return true;
  }
  return size >= 4 && data[0] == 0 && data[1] == 0 && (data[2] == 1 || (data[2] == 0 && data[3] == 1));
}

static bool likely_encoded_video_packet(const unsigned char* data, size_t size, const std::string& encoding, int width,
                                        int height) {
  if (looks_like_video_packet(data, size, encoding)) return true;

  const std::string enc = lower(encoding);
  if (enc.find("h264") != std::string::npos || enc.find("h265") != std::string::npos || enc.find("hevc") != std::string::npos ||
      enc.find("avc") != std::string::npos || enc.find("video") != std::string::npos) {
    return true;
  }

  if (width > 0 && height > 0) {
    const size_t rgb_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    const size_t nv12_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u / 2u;
    if (size > 0 && size < nv12_size) return true;
    if (size > 0 && size < rgb_size / 2u) return true;
  }
  return false;
}

static uint32_t read_be32(const unsigned char* data) {
  return (static_cast<uint32_t>(data[0]) << 24) | (static_cast<uint32_t>(data[1]) << 16) |
         (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

static uint16_t read_be16(const unsigned char* data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8) | static_cast<uint16_t>(data[1]));
}

static bool looks_like_annexb_h26x(const unsigned char* data, size_t size) {
  return size >= 3 && data[0] == 0 && data[1] == 0 && (data[2] == 1 || (size >= 4 && data[2] == 0 && data[3] == 1));
}

static bool convert_length_prefixed_nals_to_annexb(const unsigned char* data, size_t size, std::vector<unsigned char>* out) {
  if (out == nullptr || size < 4) return false;
  if (looks_like_annexb_h26x(data, size)) return false;

  for (int nal_length_size = 4; nal_length_size >= 2; nal_length_size -= 2) {
    size_t offset = 0;
    bool valid = true;
    std::vector<unsigned char> converted;
    converted.reserve(size + size / 4);

    while (offset + static_cast<size_t>(nal_length_size) <= size) {
      uint32_t nal_size = 0;
      if (nal_length_size == 4) {
        nal_size = read_be32(data + offset);
      } else {
        nal_size = read_be16(data + offset);
      }
      offset += static_cast<size_t>(nal_length_size);
      if (nal_size == 0 || offset + nal_size > size) {
        valid = false;
        break;
      }
      converted.push_back(0);
      converted.push_back(0);
      converted.push_back(0);
      converted.push_back(1);
      converted.insert(converted.end(), data + offset, data + offset + nal_size);
      offset += nal_size;
    }

    if (valid && offset == size && !converted.empty()) {
      out->swap(converted);
      return true;
    }
  }

  return false;
}

static std::string infer_video_codec_name(const unsigned char* data, size_t size, const std::string& encoding) {
  const std::string enc = lower(encoding);
  if (enc.find("h265") != std::string::npos || enc.find("hevc") != std::string::npos || enc.find("265") != std::string::npos) {
    return "hevc";
  }
  if (enc.find("h264") != std::string::npos || enc.find("avc") != std::string::npos || enc.find("264") != std::string::npos) {
    return "h264";
  }
  int start = 0;
  if (size >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
    start = 4;
  } else if (size >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) {
    start = 3;
  }
  if (size > static_cast<size_t>(start)) {
    unsigned char byte = data[start];
    int h265_type = (byte & 0x7E) >> 1;
    int h264_type = byte & 0x1F;
    if (h265_type >= 0 && h265_type < 64 && (h264_type == 0 || h264_type == 2)) return "hevc";
  }
  return "hevc";
}

static void decode_nv12(const unsigned char* data, size_t size, int width, int height, Image* image) {
  const size_t y_size = static_cast<size_t>(width * height);
  if (size < y_size + y_size / 2) throw std::runtime_error("NV12 payload is smaller than width*height*3/2");
  image->width = width;
  image->height = height;
  image->channels = 3;
  image->rgb.resize(y_size * 3);
  const unsigned char* y_plane = data;
  const unsigned char* uv_plane = data + y_size;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int Y = static_cast<int>(y_plane[y * width + x]);
      int uv_index = (y / 2) * width + (x / 2) * 2;
      int U = static_cast<int>(uv_plane[uv_index]) - 128;
      int V = static_cast<int>(uv_plane[uv_index + 1]) - 128;
      int C = Y - 16;
      int R = (298 * C + 409 * V + 128) >> 8;
      int G = (298 * C - 100 * U - 208 * V + 128) >> 8;
      int B = (298 * C + 516 * U + 128) >> 8;
      unsigned char* px = &image->rgb[(static_cast<size_t>(y) * width + x) * 3];
      px[0] = clamp_u8(R);
      px[1] = clamp_u8(G);
      px[2] = clamp_u8(B);
    }
  }
}

}  // namespace

#if defined(RSCL_HAVE_FFMPEG)
struct StatefulCameraDecoder::Impl {
  ~Impl() {
    if (sws_) sws_freeContext(sws_);
    if (frame_) av_frame_free(&frame_);
    if (packet_) av_packet_free(&packet_);
    if (codec_) avcodec_free_context(&codec_);
  }

  void ensure_codec(const std::string& codec_name) {
    if (codec_ != nullptr && codec_name_ == codec_name) return;

    static bool av_log_configured = false;
    if (!av_log_configured) {
      av_log_set_level(AV_LOG_FATAL);
      av_log_configured = true;
    }

    if (codec_) avcodec_free_context(&codec_);
    codec_ = nullptr;
    codec_name_.clear();

    const AVCodec* codec_desc = codec_name == "h264" ? avcodec_find_decoder(AV_CODEC_ID_H264) : avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (codec_desc == nullptr) throw std::runtime_error("FFmpeg decoder not found for codec: " + codec_name);

    codec_ = avcodec_alloc_context3(codec_desc);
    if (codec_ == nullptr) throw std::runtime_error("Failed to allocate FFmpeg codec context");

    codec_->thread_count = 1;
    codec_->thread_type = FF_THREAD_FRAME;
    if (avcodec_open2(codec_, codec_desc, nullptr) < 0) {
      avcodec_free_context(&codec_);
      throw std::runtime_error("Failed to open FFmpeg codec: " + codec_name);
    }

    if (packet_ == nullptr) packet_ = av_packet_alloc();
    if (frame_ == nullptr) frame_ = av_frame_alloc();
    if (packet_ == nullptr || frame_ == nullptr) throw std::runtime_error("Failed to allocate FFmpeg packet/frame");

    codec_name_ = codec_name;
  }

  bool prepare_packet_buffer(const unsigned char* data, size_t size, std::vector<unsigned char>* storage,
                             const unsigned char** packet_data, size_t* packet_size) {
    if (storage == nullptr || packet_data == nullptr || packet_size == nullptr) return false;
    if (looks_like_annexb_h26x(data, size)) {
      *packet_data = data;
      *packet_size = size;
      return false;
    }
    if (convert_length_prefixed_nals_to_annexb(data, size, storage)) {
      *packet_data = storage->data();
      *packet_size = storage->size();
      return true;
    }
    *packet_data = data;
    *packet_size = size;
    return false;
  }

  Image decode_packet(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                      const std::string& encoding) {
    const std::string primary = infer_video_codec_name(data, size, encoding);
    const std::string alternate = primary == "hevc" ? "h264" : "hevc";
    std::string errors;
    try {
      return decode_packet_with_codec(topic, timestamp_us, data, size, encoding, primary);
    } catch (const VideoFrameNotReady&) {
      throw;
    } catch (const std::exception& e) {
      errors += primary + ": " + e.what();
    }

    try {
      return decode_packet_with_codec(topic, timestamp_us, data, size, encoding, alternate);
    } catch (const VideoFrameNotReady&) {
      throw;
    } catch (const std::exception& e) {
      if (!errors.empty()) errors += "; ";
      errors += alternate + ": " + std::string(e.what());
    }

    throw std::runtime_error("Failed to decode camera video packet topic=" + topic + " encoding=" + encoding + " errors=" + errors);
  }

  Image decode_packet_with_codec(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                 const std::string& encoding, const std::string& codec_name) {
    ensure_codec(codec_name);
    av_frame_unref(frame_);
    av_packet_unref(packet_);

    std::vector<unsigned char> annexb;
    const unsigned char* packet_data = data;
    size_t packet_size = size;
    prepare_packet_buffer(data, size, &annexb, &packet_data, &packet_size);

    packet_->data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(packet_data));
    packet_->size = static_cast<int>(packet_size);
    packet_->pts = timestamp_us;
    packet_->dts = timestamp_us;

    int ret = avcodec_send_packet(codec_, packet_);
    if (ret < 0) {
      if (ret == AVERROR_INVALIDDATA) {
        throw VideoFrameNotReady("FFmpeg " + codec_name
                                 + " rejected packet while waiting for a decodable keyframe or codec config");
      }
      throw std::runtime_error("FFmpeg send_packet failed");
    }

    ret = avcodec_receive_frame(codec_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      throw VideoFrameNotReady("FFmpeg " + codec_name + " decoder has not produced a frame yet");
    }
    if (ret < 0) {
      if (ret == AVERROR_INVALIDDATA) {
        throw VideoFrameNotReady("FFmpeg " + codec_name + " decoder has not produced a valid frame yet");
      }
      throw std::runtime_error("FFmpeg receive_frame failed");
    }

    return frame_to_image();
  }

  Image frame_to_image() {
    if (frame_->width <= 0 || frame_->height <= 0) throw std::runtime_error("Decoded video frame has invalid dimensions");

    Image image;
    image.width = frame_->width;
    image.height = frame_->height;
    image.channels = 3;
    image.rgb.resize(static_cast<size_t>(frame_->width * frame_->height * 3));

    if (sws_ == nullptr || sws_width_ != frame_->width || sws_height_ != frame_->height || sws_src_format_ != frame_->format) {
      if (sws_) sws_freeContext(sws_);
      sws_ = sws_getContext(frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format), frame_->width,
                            frame_->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
      if (sws_ == nullptr) throw std::runtime_error("Failed to create FFmpeg swscale context");
      sws_width_ = frame_->width;
      sws_height_ = frame_->height;
      sws_src_format_ = frame_->format;
    }

    uint8_t* dst_data[4] = {image.rgb.data(), nullptr, nullptr, nullptr};
    int dst_linesize[4] = {frame_->width * 3, 0, 0, 0};
    int h = sws_scale(sws_, frame_->data, frame_->linesize, 0, frame_->height, dst_data, dst_linesize);
    if (h != frame_->height) throw std::runtime_error("FFmpeg sws_scale returned incomplete frame");
    return image;
  }

  void reset_codec() {
    if (codec_) avcodec_free_context(&codec_);
    codec_ = nullptr;
    codec_name_.clear();
  }

  AVCodecContext* codec_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* packet_ = nullptr;
  SwsContext* sws_ = nullptr;
  std::string codec_name_;
  int sws_width_ = 0;
  int sws_height_ = 0;
  int sws_src_format_ = -1;
};
#else
struct StatefulCameraDecoder::Impl {};
#endif

StatefulCameraDecoder::StatefulCameraDecoder() : impl_(new Impl()) {}

StatefulCameraDecoder::~StatefulCameraDecoder() { delete impl_; }

CameraPacket StatefulCameraDecoder::decode(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                           const std::string& encoding, int width, int height) {
  if (!likely_encoded_video_packet(data, size, encoding, width, height)) {
    return decode_camera_packet(topic, timestamp_us, data, size, encoding, width, height);
  }

#if defined(RSCL_HAVE_FFMPEG)
  CameraPacket packet;
  packet.topic = topic;
  packet.timestamp_us = timestamp_us;
  packet.image = impl_->decode_packet(topic, timestamp_us, data, size, encoding);
  return packet;
#else
  (void)width;
  (void)height;
  throw std::runtime_error(
      "Camera payload is H264/H265 video but this build was compiled without FFmpeg support. "
      "Install libavcodec/libavutil/libswscale dev packages and rebuild with RSCL_ENABLE_FFMPEG_DECODER=ON.");
#endif
}

CameraPacket decode_camera_packet(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                  const std::string& encoding, int width, int height) {
  CameraPacket packet;
  packet.topic = topic;
  packet.timestamp_us = timestamp_us;

  int decoded_w = 0;
  int decoded_h = 0;
  int decoded_c = 0;
  unsigned char* decoded = stbi_load_from_memory(data, static_cast<int>(size), &decoded_w, &decoded_h, &decoded_c, 3);
  if (decoded) {
    packet.image.width = decoded_w;
    packet.image.height = decoded_h;
    packet.image.channels = 3;
    packet.image.rgb.assign(decoded, decoded + static_cast<size_t>(decoded_w * decoded_h * 3));
    stbi_image_free(decoded);
    return packet;
  }

  const std::string enc = lower(encoding);
  if (likely_encoded_video_packet(data, size, encoding, width, height)) {
    throw std::runtime_error("C++ adapter received H264/H265 packet; feed decoded RGB/BGR/NV12 frames from RSCL/vehicle decoder");
  }

  if (width <= 0 || height <= 0) throw std::runtime_error("Raw camera payload needs width and height");
  packet.image.width = width;
  packet.image.height = height;
  packet.image.channels = 3;
  packet.image.rgb.resize(static_cast<size_t>(width * height * 3));

  if (enc.empty() || enc == "rgb" || enc == "rgb8") {
    if (size < packet.image.rgb.size()) throw std::runtime_error("RGB payload is too small");
    std::memcpy(packet.image.rgb.data(), data, packet.image.rgb.size());
  } else if (enc == "bgr" || enc == "bgr8") {
    if (size < packet.image.rgb.size()) throw std::runtime_error("BGR payload is too small");
    for (int i = 0; i < width * height; ++i) {
      packet.image.rgb[static_cast<size_t>(i) * 3 + 0] = data[static_cast<size_t>(i) * 3 + 2];
      packet.image.rgb[static_cast<size_t>(i) * 3 + 1] = data[static_cast<size_t>(i) * 3 + 1];
      packet.image.rgb[static_cast<size_t>(i) * 3 + 2] = data[static_cast<size_t>(i) * 3 + 0];
    }
  } else if (enc == "rgba" || enc == "rgba8" || enc == "bgra" || enc == "bgra8") {
    if (size < static_cast<size_t>(width * height * 4)) throw std::runtime_error("RGBA/BGRA payload is too small");
    const bool bgra = enc[0] == 'b';
    for (int i = 0; i < width * height; ++i) {
      packet.image.rgb[static_cast<size_t>(i) * 3 + 0] = data[static_cast<size_t>(i) * 4 + (bgra ? 2 : 0)];
      packet.image.rgb[static_cast<size_t>(i) * 3 + 1] = data[static_cast<size_t>(i) * 4 + 1];
      packet.image.rgb[static_cast<size_t>(i) * 3 + 2] = data[static_cast<size_t>(i) * 4 + (bgra ? 0 : 2)];
    }
  } else if (enc == "nv12") {
    decode_nv12(data, size, width, height, &packet.image);
  } else {
    throw std::runtime_error("Unsupported raw camera encoding: " + encoding);
  }
  return packet;
}

CameraPacket decode_camera_raw_message(const std::string& topic, const unsigned char* data, size_t size,
                                       StatefulCameraDecoder* decoder) {
  JsonValue root = parse_json(std::string(static_cast<const char*>(static_cast<const void*>(data)), size));
  if (!root.is_object()) throw std::runtime_error("Camera RawMessage JSON must be an object");
  const int64_t timestamp_us = timestamp_from_json(root);

  std::vector<unsigned char> raw = bytes_from_json_data(root);
  std::string encoding;
  const JsonValue* enc = first_json_value_nested(root, "encoding", "format", "pixelFormat", "pixel_format");
  if (enc) encoding = video_format_to_encoding(enc);
  if (encoding.empty()) {
    const JsonValue* video_format = first_json_value_nested(root, "videoFormat", "codec", "codecName");
    encoding = video_format_to_encoding(video_format);
  }
  const JsonValue* msg_encoding = root.get("message_encoding");
  if (msg_encoding && msg_encoding->is_string() && msg_encoding->string.find("base64_") == 0) encoding.clear();
  int width = 0;
  int height = 0;
  const JsonValue* w = first_json_value_nested(root, "width", "cols", "imageWidth", "image_width");
  const JsonValue* h = first_json_value_nested(root, "height", "rows", "imageHeight", "image_height");
  if (w) width = static_cast<int>(w->as_number());
  if (h) height = static_cast<int>(h->as_number());
  infer_camera_dimensions_from_topic(topic, &width, &height);
  if (decoder != nullptr) {
    return decoder->decode(topic, timestamp_us, raw.data(), raw.size(), encoding, width, height);
  }
  return decode_camera_packet(topic, timestamp_us, raw.data(), raw.size(), encoding, width, height);
}

std::vector<unsigned char> extract_camera_raw_payload(const unsigned char* data, size_t size) {
  JsonValue root = parse_json(std::string(static_cast<const char*>(static_cast<const void*>(data)), size));
  if (!root.is_object()) throw std::runtime_error("Camera RawMessage JSON must be an object");
  return bytes_from_json_data(root);
}

LidarPacket decode_lidar_packet(const std::string& topic, int64_t timestamp_us, const unsigned char* data, size_t size,
                                int point_dim, int point_step, int width) {
  LidarPacket packet;
  packet.topic = topic;
  packet.timestamp_us = timestamp_us;
  packet.point_dim = point_dim;

  size_t usable = size;
  if (width > 0 && point_step > 0) usable = std::min(usable, static_cast<size_t>(width * point_step));
  if (point_step == 16) {
    const size_t n = usable / 16;
    packet.point_dim = 4;
    packet.points.resize(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
      const unsigned char* rec = data + i * 16;
      int16_t xyz[3];
      std::memcpy(xyz, rec, sizeof(xyz));
      packet.points[i * 4 + 0] = static_cast<float>(xyz[0]) * 0.01f;
      packet.points[i * 4 + 1] = static_cast<float>(xyz[1]) * 0.01f;
      packet.points[i * 4 + 2] = static_cast<float>(xyz[2]) * 0.01f;
    }
    return packet;
  }

  int stride = point_dim;
  if (point_step > 0) {
    if (point_step % 4 != 0) throw std::runtime_error("Unsupported lidar point_step");
    stride = std::max(point_step / 4, 1);
  }
  if (usable % 4 != 0) throw std::runtime_error("Lidar payload is not float32 aligned");
  const size_t floats = usable / 4;
  if (floats % static_cast<size_t>(stride) != 0) throw std::runtime_error("Lidar float count does not match point stride");
  const size_t n = floats / static_cast<size_t>(stride);
  packet.points.resize(n * static_cast<size_t>(std::min(stride, point_dim)));
  packet.point_dim = std::min(stride, point_dim);
  for (size_t i = 0; i < n; ++i) {
    for (int c = 0; c < packet.point_dim; ++c) {
      float value = 0.0f;
      std::memcpy(&value, data + (i * static_cast<size_t>(stride) + c) * sizeof(float), sizeof(float));
      packet.points[i * packet.point_dim + c] = value;
    }
  }
  return packet;
}

LidarPacket decode_lidar_raw_message(const std::string& topic, const unsigned char* data, size_t size, int point_dim) {
  JsonValue root = parse_json(std::string(static_cast<const char*>(static_cast<const void*>(data)), size));
  if (!root.is_object()) throw std::runtime_error("Lidar RawMessage JSON must be an object");
  const int64_t timestamp_us = timestamp_from_json(root);

  const JsonValue* points = root.get("points");
  if (points && points->is_array()) {
    LidarPacket packet;
    packet.topic = topic;
    packet.timestamp_us = timestamp_us;
    packet.points = points_from_json_array(*points, &packet.point_dim);
    return packet;
  }

  std::vector<unsigned char> raw = bytes_from_json_data(root);
  int point_step = 0;
  int width = 0;
  const JsonValue* step = first_json_value_nested(root, "pointStep", "point_step", "stride");
  const JsonValue* w = first_json_value_nested(root, "width");
  if (step) point_step = static_cast<int>(step->as_number());
  if (w) width = static_cast<int>(w->as_number());
  return decode_lidar_packet(topic, timestamp_us, raw.data(), raw.size(), point_dim, point_step, width);
}

int64_t decode_raw_message_timestamp_us(const unsigned char* data, size_t size) {
  JsonValue root = parse_json(std::string(static_cast<const char*>(static_cast<const void*>(data)), size));
  if (!root.is_object()) throw std::runtime_error("RawMessage JSON must be an object");
  return timestamp_from_json(root);
}

std::string encode_detection_message(const InferenceOutput& output, const AdapterConfig& cfg, int64_t timestamp_us) {
  std::ostringstream ss;
  ss << "{\"objects\":[";
  bool first = true;
  for (size_t i = 0; i < output.detections.size(); ++i) {
    const Detection& det = output.detections[i];
    if (det.score < cfg.score_threshold) continue;
    if (!first) ss << ",";
    first = false;
    ss << "{\"label\":" << det.label << ",\"score\":" << det.score << ",\"box\":[";
    for (int j = 0; j < 9; ++j) {
      if (j) ss << ",";
      ss << det.box[j];
    }
    ss << "]}";
  }
  ss << "],\"timestamp_us\":" << timestamp_us;
  if (output.has_map) {
    ss << ",\"map\":{\"classes\":[";
    for (size_t i = 0; i < cfg.map_classes.size(); ++i) {
      if (i) ss << ",";
      ss << "\"" << json_escape(cfg.map_classes[i]) << "\"";
    }
    ss << "],\"shape\":[";
    for (size_t i = 0; i < output.map.shape.size(); ++i) {
      if (i) ss << ",";
      ss << output.map.shape[i];
    }
    ss << "],\"threshold\":" << cfg.map_score_threshold << ",\"encoding\":\"base64_uint8_chw\",\"data\":\""
       << base64_encode(output.map.data.data(), output.map.data.size()) << "\"}";
  }
  ss << "}";
  return ss.str();
}

}  // namespace rscl_adapter
