#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rscl_adapter {

struct BagMessage {
  std::string topic;
  int64_t timestamp_us = 0;
  std::vector<unsigned char> payload;
  std::vector<unsigned char> raw_payload;
  std::string message_type;
  std::string reflected_json;

  // Optional metadata for SDKs that expose decoded/typed image and point cloud
  // messages instead of RawMessage JSON bytes.
  std::string encoding;
  int image_width = 0;
  int image_height = 0;
  int point_step = 0;
  int point_width = 0;
};

class BagReader {
 public:
  virtual ~BagReader() = default;
  virtual bool is_valid() const = 0;
  virtual bool read_next(BagMessage* message) = 0;
};

std::unique_ptr<BagReader> create_rscl_bag_reader(const std::string& bag_path,
                                                  const std::vector<std::string>& included_topics);

}  // namespace rscl_adapter
