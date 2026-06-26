#include "rscl_adapter/bag_reader.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace rscl_adapter {
namespace {

// This file is a vehicle-side integration template for the RSCL C++ SDK.
// It is intentionally not compiled by default because the repository does not
// vendor RSCL headers or libraries.
//
// The message layout below is based on the inspected rsclbag sample:
//
// Camera topics:
//   /sensor/camera/*/encode
// Fields on message_obj:
//   data         : bytes, H26x elementary stream
//   dataSize     : int
//   frameStartTime / frameTime / sendTime : ns timestamps
//   header.time.nanoSec                   : ns timestamp
//   videoFormat  : integer enum (observed value 6)
//   width/height : 0 for encoded topics
//
// Lidar topic:
//   /perception/lidar/preproc_points_cloud
// Fields on message_obj:
//   data         : bytes
//   header.time.nanoSec : ns timestamp
//   pointStep    : 16
//   width        : point count
//   rowStep      : byte size
//
// The CUDA-BEVFusion adapter expects:
//   - camera payload: decoded RGB/BGR/NV12 image bytes OR encoded bytes plus a
//     separate decode stage before decode_camera_packet()
//   - lidar payload : raw bytes + point_step + point_width
//
// For your inspected bag, camera topics are encoded H26x packets. The simplest
// vehicle integration is:
//   1. Use RSCL/vehicle video decoder to convert each camera packet to RGB/BGR/NV12
//   2. Fill BagMessage.payload with the decoded bytes
//   3. Fill BagMessage.encoding, image_width, image_height
//
// If your RSCL C++ SDK can only expose the encoded packet here, then this bag
// reader should reject camera packets and the online node should subscribe to a
// decoded image topic instead.

static int64_t ns_to_us(int64_t ns) { return ns / 1000; }

}  // namespace

class VehicleRsclBagReader : public BagReader {
 public:
  VehicleRsclBagReader(const std::string& bag_path, const std::vector<std::string>& included_topics) {
    (void)bag_path;
    (void)included_topics;

    // TODO:
    // 1. Include the RSCL bag reader headers here.
    // 2. Construct the SDK reader.
    // 3. Configure channel filtering with included_topics.
    //
    // Example shape only:
    //   reader_ = rscl::BagReader(bag_path, attrs);
    throw std::runtime_error("VehicleRsclBagReader template is not connected to the RSCL C++ SDK yet.");
  }

  bool is_valid() const override {
    // TODO: return SDK reader validity.
    return false;
  }

  bool read_next(BagMessage* message) override {
    if (message == nullptr) throw std::invalid_argument("message must not be null");

    // TODO:
    // 1. Read one message from the RSCL SDK
    // 2. Detect topic name
    // 3. Access the typed message object / payload
    //
    // Camera mapping from the inspected bag:
    //   message->topic = "/sensor/camera/.../encode";
    //   message->timestamp_us = ns_to_us(obj.header.time.nanoSec);
    //   message->payload = obj.data;
    //
    // If you decode video inside this reader:
    //   message->payload = decoded_rgb_or_nv12_bytes;
    //   message->encoding = "rgb8" / "bgr8" / "nv12";
    //   message->image_width = decoded_width;
    //   message->image_height = decoded_height;
    //
    // Lidar mapping from the inspected bag:
    //   message->topic = "/perception/lidar/preproc_points_cloud";
    //   message->timestamp_us = ns_to_us(obj.header.time.nanoSec);
    //   message->payload = obj.data;
    //   message->point_step = obj.pointStep;   // observed 16
    //   message->point_width = obj.width;      // observed point count
    //
    // Return false on EOF.
    return false;
  }

 private:
  // TODO: replace with the actual RSCL SDK reader type.
  // Example:
  // std::unique_ptr<rscl::BagReader> reader_;
};

std::unique_ptr<BagReader> create_rscl_bag_reader(const std::string& bag_path,
                                                  const std::vector<std::string>& included_topics) {
  return std::unique_ptr<BagReader>(new VehicleRsclBagReader(bag_path, included_topics));
}

}  // namespace rscl_adapter
