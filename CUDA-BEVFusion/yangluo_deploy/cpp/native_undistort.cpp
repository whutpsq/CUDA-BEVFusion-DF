#include "native_undistort.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace rscl_adapter {
namespace {

constexpr int kInterpolationBits = 5;
constexpr int kInterpolationSize = 1 << kInterpolationBits;
constexpr int kInterpolationArea = kInterpolationSize * kInterpolationSize;

static int floor_div_32(int value) {
  if (value >= 0) return value >> kInterpolationBits;
  return -static_cast<int>((static_cast<unsigned int>(-value) + kInterpolationSize - 1) >>
                           kInterpolationBits);
}

static bool invert3x3(const double* matrix, double* inverse) {
  const double a = matrix[0], b = matrix[1], c = matrix[2];
  const double d = matrix[3], e = matrix[4], f = matrix[5];
  const double g = matrix[6], h = matrix[7], i = matrix[8];
  const double determinant = a * (e * i - f * h) - b * (d * i - f * g) +
                             c * (d * h - e * g);
  if (!std::isfinite(determinant) || std::fabs(determinant) < 1e-12) return false;
  const double scale = 1.0 / determinant;
  inverse[0] = (e * i - f * h) * scale;
  inverse[1] = (c * h - b * i) * scale;
  inverse[2] = (b * f - c * e) * scale;
  inverse[3] = (f * g - d * i) * scale;
  inverse[4] = (a * i - c * g) * scale;
  inverse[5] = (c * d - a * f) * scale;
  inverse[6] = (d * h - e * g) * scale;
  inverse[7] = (b * g - a * h) * scale;
  inverse[8] = (a * e - b * d) * scale;
  return true;
}

static void create_map(const Image& src, const float* intrinsic, const float* distortion,
                       UndistortCache::Entry* entry) {
  const double camera[9] = {
      intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[4], intrinsic[5],
      intrinsic[6], intrinsic[8], intrinsic[9], intrinsic[10],
  };
  double camera_inverse[9];
  if (!invert3x3(camera, camera_inverse)) {
    throw std::runtime_error("Camera intrinsic matrix is singular during native undistortion");
  }

  const double k1 = distortion[0], k2 = distortion[1];
  const double p1 = distortion[2], p2 = distortion[3];
  const double k3 = distortion[4], k4 = distortion[5];
  const double k5 = distortion[6], k6 = distortion[7];
  const size_t pixel_count = static_cast<size_t>(src.width) * src.height;
  entry->x0.resize(pixel_count);
  entry->y0.resize(pixel_count);
  entry->fractions.resize(pixel_count);

  for (int v = 0; v < src.height; ++v) {
    for (int u = 0; u < src.width; ++u) {
      const double homogeneous_x = camera_inverse[0] * u + camera_inverse[1] * v + camera_inverse[2];
      const double homogeneous_y = camera_inverse[3] * u + camera_inverse[4] * v + camera_inverse[5];
      const double homogeneous_w = camera_inverse[6] * u + camera_inverse[7] * v + camera_inverse[8];
      if (!std::isfinite(homogeneous_w) || std::fabs(homogeneous_w) < 1e-12) {
        throw std::runtime_error("Invalid homogeneous coordinate during native undistortion");
      }
      const double x = homogeneous_x / homogeneous_w;
      const double y = homogeneous_y / homogeneous_w;
      const double r2 = x * x + y * y;
      const double r4 = r2 * r2;
      const double r6 = r4 * r2;
      const double denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
      if (!std::isfinite(denominator) || std::fabs(denominator) < 1e-12) {
        throw std::runtime_error("Invalid rational distortion denominator");
      }
      const double radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / denominator;
      const double distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
      const double distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
      const double projected_x = camera[0] * distorted_x + camera[1] * distorted_y + camera[2];
      const double projected_y = camera[3] * distorted_x + camera[4] * distorted_y + camera[5];
      const double projected_w = camera[6] * distorted_x + camera[7] * distorted_y + camera[8];
      const double source_x = projected_x / projected_w;
      const double source_y = projected_y / projected_w;

      // OpenCV's CV_16SC2 map stores coordinates at INTER_BITS=5 precision.
      // lrint follows the default round-to-nearest-even mode used by cvRound
      // on the target architecture.
      const int quantized_x = static_cast<int>(std::lrint(source_x * kInterpolationSize));
      const int quantized_y = static_cast<int>(std::lrint(source_y * kInterpolationSize));
      const int base_x = floor_div_32(quantized_x);
      const int base_y = floor_div_32(quantized_y);
      const int fraction_x = quantized_x - base_x * kInterpolationSize;
      const int fraction_y = quantized_y - base_y * kInterpolationSize;
      const size_t index = static_cast<size_t>(v) * src.width + u;
      entry->x0[index] = base_x;
      entry->y0[index] = base_y;
      entry->fractions[index] = static_cast<uint16_t>((fraction_y << kInterpolationBits) | fraction_x);
    }
  }
  entry->width = src.width;
  entry->height = src.height;
}

static int sample(const Image& src, int x, int y, int channel) {
  if (x < 0 || y < 0 || x >= src.width || y >= src.height) return 0;
  return src.rgb[(static_cast<size_t>(y) * src.width + x) * 3 + channel];
}

}  // namespace

Image undistort_image_native(const Image& src, const Calibration& calibration,
                             size_t camera_index) {
  if (src.rgb.empty() || src.width <= 0 || src.height <= 0 || src.channels != 3 ||
      src.rgb.size() != static_cast<size_t>(src.width) * src.height * 3) {
    throw std::runtime_error("Invalid RGB camera image for native undistortion");
  }
  const size_t camera_count = calibration.camera_intrinsics.size() / 16;
  if (camera_index >= camera_count || calibration.camera_distortions.size() != camera_count * 8 ||
      calibration.camera_has_distortion.size() != camera_count ||
      !calibration.camera_has_distortion[camera_index]) {
    throw std::runtime_error("undistort_images=true requires 8 distortion coefficients for every camera");
  }
  if (!calibration.undistort_cache) {
    throw std::runtime_error("Native camera undistortion cache was not initialized");
  }

  std::lock_guard<std::mutex> lock(calibration.undistort_cache->mutex);
  if (calibration.undistort_cache->entries.size() != camera_count) {
    calibration.undistort_cache->entries.resize(camera_count);
  }
  UndistortCache::Entry& map = calibration.undistort_cache->entries[camera_index];
  bool map_created = false;
  if (map.width != src.width || map.height != src.height || map.x0.empty() || map.y0.empty() ||
      map.fractions.empty()) {
    create_map(src, calibration.camera_intrinsics.data() + camera_index * 16,
               calibration.camera_distortions.data() + camera_index * 8, &map);
    map_created = true;
  }
  Image output;
  output.width = src.width;
  output.height = src.height;
  output.channels = 3;
  output.rgb.resize(src.rgb.size());
  size_t changed_bytes = 0;
  for (size_t index = 0; index < map.x0.size(); ++index) {
    const int x0 = map.x0[index];
    const int y0 = map.y0[index];
    const int fraction_x = map.fractions[index] & (kInterpolationSize - 1);
    const int fraction_y = map.fractions[index] >> kInterpolationBits;
    const int weight00 = (kInterpolationSize - fraction_x) * (kInterpolationSize - fraction_y);
    const int weight01 = fraction_x * (kInterpolationSize - fraction_y);
    const int weight10 = (kInterpolationSize - fraction_x) * fraction_y;
    const int weight11 = fraction_x * fraction_y;
    for (int channel = 0; channel < 3; ++channel) {
      const int weighted = weight00 * sample(src, x0, y0, channel) +
                           weight01 * sample(src, x0 + 1, y0, channel) +
                           weight10 * sample(src, x0, y0 + 1, channel) +
                           weight11 * sample(src, x0 + 1, y0 + 1, channel);
      const unsigned char value = static_cast<unsigned char>((weighted + kInterpolationArea / 2) /
                                                              kInterpolationArea);
      output.rgb[index * 3 + channel] = value;
      if (map_created && value != src.rgb[index * 3 + channel]) ++changed_bytes;
    }
  }
  if (map_created) {
    std::cerr << "preprocess_stage=native_undistort_applied camera_index=" << camera_index
              << " width=" << src.width << " height=" << src.height
              << " changed_bytes=" << changed_bytes << std::endl;
  }
  return output;
}

}  // namespace rscl_adapter
