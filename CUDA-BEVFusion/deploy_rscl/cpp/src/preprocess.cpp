#include "rscl_adapter/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "rscl_adapter/json.hpp"

namespace rscl_adapter {
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

static const JsonValue* camera_entry(const JsonValue& root, const std::string& name) {
  const JsonValue* cameras = root.get("cameras");
  if (cameras && cameras->is_object()) return cameras->get(name);
  return root.get(name);
}

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
      append_matrix(&cal.lidar2image, eye4());
    }
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
    append_matrix(&cal.lidar2image, matmul4(intrinsics, lidar2camera));
  }
  return cal;
}

ModelInput build_model_input(const SyncedFrame& frame, const AdapterConfig& cfg, const Calibration& calibration) {
  if (cfg.undistort_images) {
    throw std::runtime_error("C++ adapter does not undistort images yet; use rectified camera streams or disable undistort_images");
  }
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
    Image resized = resize_crop_image(frame.cameras[cam], cfg, aug);
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
