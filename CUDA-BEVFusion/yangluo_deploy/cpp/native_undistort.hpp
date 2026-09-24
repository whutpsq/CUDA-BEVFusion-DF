#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "rscl_adapter/preprocess.hpp"

namespace rscl_adapter {

// Yangluo-only OpenCV-free remap cache.  The passenger-car build keeps using
// its existing OpenCV implementation unless RSCL_ENABLE_NATIVE_UNDISTORT is
// explicitly enabled.
struct UndistortCache {
  struct Entry {
    int width = 0;
    int height = 0;
    std::vector<int32_t> x0;
    std::vector<int32_t> y0;
    std::vector<uint16_t> fractions;
  };

  std::mutex mutex;
  std::vector<Entry> entries;
};

Image undistort_image_native(const Image& src, const Calibration& calibration,
                             size_t camera_index);

}  // namespace rscl_adapter
