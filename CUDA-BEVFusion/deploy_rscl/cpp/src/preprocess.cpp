#include "rscl_adapter/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifdef RSCL_HAVE_OPENCV_UNDISTORT
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "rscl_adapter/json.hpp"

namespace rscl_adapter {

#ifdef RSCL_HAVE_OPENCV_UNDISTORT
struct UndistortCache {
  struct Entry {
    int width = 0;
    int height = 0;
    cv::Mat map1;
    cv::Mat map2;
  };

  std::mutex mutex;
  std::vector<Entry> entries;
};
#else
struct UndistortCache {};
#endif

namespace {

static std::vector<float> eye4() {
  std::vector<float> m(16, 0.0f);
  for (int i = 0; i < 4; ++i) m[i * 4 + i] = 1.0f;
  return m;
}

static std::vector<float> matrix_from_json(const JsonValue* value) {
  if (!value || !value->is_array()) return eye4();
  int rows = static_cast<int>(value->array.size());
  int cols = rows > 0 && value->array[0].is_array() ? static_cast<int>(value->array[0].array.size()) : 0;
  std::vector<float> out = eye4();
  if (rows == 3 && cols == 3) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) out[r * 4 + c] = static_cast<float>(value->array[r].array[c].as_number());
    }
    return out;
  }
  if (rows == 3 && cols == 4) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) out[r * 4 + c] = static_cast<float>(value->array[r].array[c].as_number());
    }
    return out;
  }
  if (rows == 4 && cols == 4) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) out[r * 4 + c] = static_cast<float>(value->array[r].array[c].as_number());
    }
    return out;
  }
  throw std::runtime_error("Expected calibration matrix shape 3x3, 3x4, or 4x4");
}

static std::vector<float> matmul4(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> out(16, 0.0f);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      for (int k = 0; k < 4; ++k) out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    }
  }
  return out;
}

static std::vector<float> inverse4(std::vector<float> m) {
  std::vector<float> inv = eye4();
  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    float best = std::fabs(m[col * 4 + col]);
    for (int row = col + 1; row < 4; ++row) {
      float v = std::fabs(m[row * 4 + col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-8f) throw std::runtime_error("Calibration matrix is singular");
    if (pivot != col) {
      for (int c = 0; c < 4; ++c) {
        std::swap(m[col * 4 + c], m[pivot * 4 + c]);
        std::swap(inv[col * 4 + c], inv[pivot * 4 + c]);
      }
    }
    float div = m[col * 4 + col];
    for (int c = 0; c < 4; ++c) {
      m[col * 4 + c] /= div;
      inv[col * 4 + c] /= div;
    }
    for (int row = 0; row < 4; ++row) {
      if (row == col) continue;
      float scale = m[row * 4 + col];
      for (int c = 0; c < 4; ++c) {
        m[row * 4 + c] -= scale * m[col * 4 + c];
        inv[row * 4 + c] -= scale * inv[col * 4 + c];
      }
    }
  }
  return inv;
}

static void append_matrix(std::vector<float>* dst, const std::vector<float>& mat) {
  dst->insert(dst->end(), mat.begin(), mat.end());
}

static bool append_distortion(std::vector<float>* dst, const JsonValue* value) {
  static const char* const keys[8] = {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"};
  if (!value) {
    dst->insert(dst->end(), 8, 0.0f);
    return false;
  }

  if (value->is_array()) {
    if (value->array.size() != 8) {
      throw std::runtime_error("Camera distortion must contain 8 values in k1,k2,p1,p2,k3,k4,k5,k6 order");
    }
    for (size_t i = 0; i < 8; ++i) {
      if (!value->array[i].is_number() || !std::isfinite(value->array[i].number)) {
        throw std::runtime_error("Camera distortion contains a non-finite value");
      }
      dst->push_back(static_cast<float>(value->array[i].number));
    }
    return true;
  }

  if (!value->is_object()) {
    throw std::runtime_error("Camera distortion must be an object or an array of 8 values");
  }
  const JsonValue* type = value->get("type");
  if (type && type->is_string() && type->string != "pinhole") {
    throw std::runtime_error("Only OpenCV pinhole/rational camera distortion is supported, got: " + type->string);
  }
  for (size_t i = 0; i < 8; ++i) {
    const JsonValue* coefficient = value->get(keys[i]);
    if (!coefficient || !coefficient->is_number() || !std::isfinite(coefficient->number)) {
      throw std::runtime_error(std::string("Missing or invalid camera distortion coefficient: ") + keys[i]);
    }
    dst->push_back(static_cast<float>(coefficient->number));
  }
  return true;
}

static const JsonValue* camera_entry(const JsonValue& root, const std::string& name) {
  const JsonValue* cameras = root.get("cameras");
  if (cameras && cameras->is_object()) return cameras->get(name);
  return root.get(name);
}

#ifdef RSCL_HAVE_OPENCV_UNDISTORT
static Image undistort_image(const Image& src, const Calibration& calibration, size_t camera_index) {
  if (src.rgb.empty() || src.width <= 0 || src.height <= 0 || src.channels != 3 ||
      src.rgb.size() != static_cast<size_t>(src.width) * src.height * 3) {
    throw std::runtime_error("Invalid RGB camera image for undistortion");
  }
  const size_t camera_count = calibration.camera_intrinsics.size() / 16;
  if (camera_index >= camera_count || calibration.camera_distortions.size() != camera_count * 8 ||
      calibration.camera_has_distortion.size() != camera_count ||
      !calibration.camera_has_distortion[camera_index]) {
    throw std::runtime_error("undistort_images=true requires 8 distortion coefficients for every camera");
  }
  if (!calibration.undistort_cache) throw std::runtime_error("Camera undistortion cache was not initialized");

  const float* intrinsic = calibration.camera_intrinsics.data() + camera_index * 16;
  const float* distortion = calibration.camera_distortions.data() + camera_index * 8;
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << intrinsic[0], intrinsic[1], intrinsic[2],
                           intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[8], intrinsic[9], intrinsic[10]);
  cv::Mat distortion_coefficients(1, 8, CV_64F);
  for (int i = 0; i < 8; ++i) distortion_coefficients.at<double>(0, i) = distortion[i];

  cv::Mat map1;
  bool map_created = false;
  cv::Mat map2;
  {
    std::lock_guard<std::mutex> lock(calibration.undistort_cache->mutex);
    if (calibration.undistort_cache->entries.size() != camera_count) {
      calibration.undistort_cache->entries.resize(camera_count);
    }
    UndistortCache::Entry& entry = calibration.undistort_cache->entries[camera_index];
    if (entry.width != src.width || entry.height != src.height || entry.map1.empty() || entry.map2.empty()) {
      cv::initUndistortRectifyMap(camera_matrix, distortion_coefficients, cv::Mat(), camera_matrix,
                                  cv::Size(src.width, src.height), CV_16SC2, entry.map1, entry.map2);
      entry.width = src.width;
      entry.height = src.height;
      map_created = true;
    }
    map1 = entry.map1;
    map2 = entry.map2;
  }

  cv::Mat source(src.height, src.width, CV_8UC3, const_cast<unsigned char*>(src.rgb.data()));
  cv::Mat rectified;
  cv::remap(source, rectified, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
  if (!rectified.isContinuous()) rectified = rectified.clone();
  if (map_created) {
    size_t changed_bytes = 0;
    const size_t byte_count = rectified.total() * rectified.elemSize();
    for (size_t i = 0; i < byte_count; ++i) {
      if (src.rgb[i] != rectified.data[i]) ++changed_bytes;
    }
    std::cerr << "preprocess_stage=undistort_applied camera_index=" << camera_index
              << " width=" << src.width << " height=" << src.height
              << " changed_bytes=" << changed_bytes << std::endl;
  }

  Image out;
  out.width = rectified.cols;
  out.height = rectified.rows;
  out.channels = 3;
  out.rgb.assign(rectified.data, rectified.data + rectified.total() * rectified.elemSize());
  return out;
}
#endif

static Image resize_crop_image(const Image& src, const AdapterConfig& cfg, float* aug16) {
  if (src.rgb.empty() || src.width <= 0 || src.height <= 0) throw std::runtime_error("Invalid camera image");
  const int final_h = cfg.image_height;
  const int final_w = cfg.image_width;
  const int resize_w = std::max(static_cast<int>(src.width * cfg.image_resize), final_w);
  const int resize_h = std::max(static_cast<int>(src.height * cfg.image_resize), final_h);
  const int crop_w = static_cast<int>(std::max(0, resize_w - final_w) / 2);
  const int crop_h = static_cast<int>(std::max(0, resize_h - final_h));

  Image out;
  out.width = final_w;
  out.height = final_h;
  out.channels = 3;
  out.rgb.resize(static_cast<size_t>(final_w * final_h * 3));

  const float scale_x = static_cast<float>(src.width) / static_cast<float>(resize_w);
  const float scale_y = static_cast<float>(src.height) / static_cast<float>(resize_h);
  for (int y = 0; y < final_h; ++y) {
    float sy = (static_cast<float>(y + crop_h) + 0.5f) * scale_y - 0.5f;
    int y0 = std::max(0, std::min(src.height - 1, static_cast<int>(std::floor(sy))));
    int y1 = std::max(0, std::min(src.height - 1, y0 + 1));
    float wy = sy - y0;
    for (int x = 0; x < final_w; ++x) {
      float sx = (static_cast<float>(x + crop_w) + 0.5f) * scale_x - 0.5f;
      int x0 = std::max(0, std::min(src.width - 1, static_cast<int>(std::floor(sx))));
      int x1 = std::max(0, std::min(src.width - 1, x0 + 1));
      float wx = sx - x0;
      for (int c = 0; c < 3; ++c) {
        float p00 = src.rgb[(static_cast<size_t>(y0) * src.width + x0) * 3 + c];
        float p01 = src.rgb[(static_cast<size_t>(y0) * src.width + x1) * 3 + c];
        float p10 = src.rgb[(static_cast<size_t>(y1) * src.width + x0) * 3 + c];
        float p11 = src.rgb[(static_cast<size_t>(y1) * src.width + x1) * 3 + c];
        float value = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01) + wy * ((1.0f - wx) * p10 + wx * p11);
        out.rgb[(static_cast<size_t>(y) * final_w + x) * 3 + c] =
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, value + 0.5f)));
      }
    }
  }

  std::fill(aug16, aug16 + 16, 0.0f);
  for (int i = 0; i < 4; ++i) aug16[i * 4 + i] = 1.0f;
  aug16[0] = cfg.image_resize;
  aug16[5] = cfg.image_resize;
  aug16[3] = -static_cast<float>(crop_w);
  aug16[7] = -static_cast<float>(crop_h);
  return out;
}

}  // namespace

Calibration load_calibration(const std::string& path, const std::vector<std::string>& camera_order,
                             const std::string& extrinsic_direction, bool allow_identity) {
  if (path.empty()) {
    if (!allow_identity) throw std::runtime_error("calibration_file is required for C++ RSCL inference");
    Calibration cal;
    for (size_t i = 0; i < camera_order.size(); ++i) {
      append_matrix(&cal.camera2lidar, eye4());
      append_matrix(&cal.camera_intrinsics, eye4());
      cal.camera_distortions.insert(cal.camera_distortions.end(), 8, 0.0f);
      cal.camera_has_distortion.push_back(0);
      append_matrix(&cal.lidar2image, eye4());
    }
    cal.undistort_cache = std::make_shared<UndistortCache>();
    return cal;
  }

  JsonValue root = parse_json(read_text_file(path));
  std::vector<float> lidar2ego = matrix_from_json(root.get("lidar2ego"));
  std::vector<float> ego2lidar = inverse4(lidar2ego);

  Calibration cal;
  for (size_t i = 0; i < camera_order.size(); ++i) {
    const JsonValue* cam = camera_entry(root, camera_order[i]);
    if (!cam || !cam->is_object()) throw std::runtime_error("Missing calibration camera entry: " + camera_order[i]);

    std::vector<float> intrinsics;
    std::vector<float> camera2lidar;
    std::vector<float> lidar2camera;

    const JsonValue* cameras = root.get("cameras");
    if (cameras && cameras->is_object()) {
      intrinsics = matrix_from_json(cam->get("camera_intrinsics"));
      std::vector<float> camera2ego = matrix_from_json(cam->get("camera2ego"));
      camera2lidar = matmul4(ego2lidar, camera2ego);
      lidar2camera = inverse4(camera2lidar);
    } else {
      intrinsics = matrix_from_json(cam->get("cam_intrinsic"));
      std::vector<float> extrinsic = matrix_from_json(cam->get("extrinsic"));
      if (extrinsic_direction == "lidar2camera") {
        lidar2camera = extrinsic;
        camera2lidar = inverse4(lidar2camera);
      } else if (extrinsic_direction == "camera2lidar") {
        camera2lidar = extrinsic;
        lidar2camera = inverse4(camera2lidar);
      } else {
        throw std::runtime_error("calibration_extrinsic_direction must be lidar2camera or camera2lidar");
      }
    }

    append_matrix(&cal.camera2lidar, camera2lidar);
    append_matrix(&cal.camera_intrinsics, intrinsics);
    const JsonValue* distortion = cam->get("cam_dist");
    if (!distortion) distortion = cam->get("distortion");
    if (!distortion) distortion = cam->get("camera_distortion");
    cal.camera_has_distortion.push_back(append_distortion(&cal.camera_distortions, distortion) ? 1 : 0);
    append_matrix(&cal.lidar2image, matmul4(intrinsics, lidar2camera));
  }
  cal.undistort_cache = std::make_shared<UndistortCache>();
  return cal;
}

ModelInput build_model_input(const SyncedFrame& frame, const AdapterConfig& cfg, const Calibration& calibration) {
#ifndef RSCL_HAVE_OPENCV_UNDISTORT
  if (cfg.undistort_images) {
    throw std::runtime_error("undistort_images=true requires a build with RSCL_ENABLE_OPENCV_UNDISTORT=ON");
  }
#endif
  if (frame.cameras.size() != cfg.camera_order.size()) throw std::runtime_error("Synced frame camera count mismatch");

  ModelInput input;
  input.num_cameras = static_cast<int>(frame.cameras.size());
  input.image_height = cfg.image_height;
  input.image_width = cfg.image_width;
  const size_t image_plane = static_cast<size_t>(cfg.image_height * cfg.image_width);
  input.images_chw.resize(static_cast<size_t>(input.num_cameras) * 3 * image_plane);
  input.img_aug_matrix.resize(static_cast<size_t>(input.num_cameras) * 16);

  for (int cam = 0; cam < input.num_cameras; ++cam) {
    float* aug = input.img_aug_matrix.data() + static_cast<size_t>(cam) * 16;
#ifdef RSCL_HAVE_OPENCV_UNDISTORT
    Image rectified;
    const Image* source = &frame.cameras[cam];
    if (cfg.undistort_images) {
      rectified = undistort_image(frame.cameras[cam], calibration, static_cast<size_t>(cam));
      source = &rectified;
    }
    Image resized = resize_crop_image(*source, cfg, aug);
#else
    Image resized = resize_crop_image(frame.cameras[cam], cfg, aug);
#endif
    for (int y = 0; y < cfg.image_height; ++y) {
      for (int x = 0; x < cfg.image_width; ++x) {
        const unsigned char* px = &resized.rgb[(static_cast<size_t>(y) * cfg.image_width + x) * 3];
        for (int c = 0; c < 3; ++c) {
          float value = static_cast<float>(px[c]) / 255.0f;
          value = (value - cfg.image_mean[c]) / cfg.image_std[c];
          input.images_chw[(static_cast<size_t>(cam) * 3 + c) * image_plane + static_cast<size_t>(y) * cfg.image_width + x] =
              value;
        }
      }
    }
  }

  const int in_dim = frame.lidar_point_dim > 0 ? frame.lidar_point_dim : cfg.point_dim;
  const size_t npoints = frame.lidar.size() / static_cast<size_t>(in_dim);
  input.points.reserve(npoints * 5);
  for (size_t i = 0; i < npoints; ++i) {
    const float* p = frame.lidar.data() + i * in_dim;
    if (p[0] <= cfg.point_cloud_range[0] || p[0] >= cfg.point_cloud_range[3] || p[1] <= cfg.point_cloud_range[1] ||
        p[1] >= cfg.point_cloud_range[4] || p[2] <= cfg.point_cloud_range[2] || p[2] >= cfg.point_cloud_range[5]) {
      continue;
    }
    for (int c = 0; c < 5; ++c) input.points.push_back(c < in_dim ? p[c] : 0.0f);
  }

  input.camera2lidar = calibration.camera2lidar;
  input.camera_intrinsics = calibration.camera_intrinsics;
  input.lidar2image = calibration.lidar2image;
  return input;
}

}  // namespace rscl_adapter
