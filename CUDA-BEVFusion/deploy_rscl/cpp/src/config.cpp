#include "rscl_adapter/config.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "rscl_adapter/json.hpp"

namespace rscl_adapter {
namespace {

static bool is_absolute_path(const std::string& path) {
  if (path.empty()) return false;
  if (path[0] == '/' || path[0] == '\\') return true;
  return path.size() > 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':';
}

static std::string dirname(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) return ".";
  return path.substr(0, pos);
}

static std::string resolve_path(const std::string& value, const std::string& base_dir) {
  if (value.empty() || is_absolute_path(value)) return value;
  return base_dir + "/" + value;
}

static std::string lower_ext(const std::string& path) {
  size_t pos = path.find_last_of('.');
  std::string ext = pos == std::string::npos ? "" : path.substr(pos);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

static const JsonValue* get(const JsonValue& root, const char* key) { return root.get(key); }

static void set_string(const JsonValue& root, const char* key, std::string* dst) {
  const JsonValue* v = get(root, key);
  if (v && !v->is_null() && !v->is_array() && !v->is_object()) *dst = v->as_string();
}

static void set_bool(const JsonValue& root, const char* key, bool* dst) {
  const JsonValue* v = get(root, key);
  if (v && !v->is_null()) *dst = v->as_bool(*dst);
}

static void set_float(const JsonValue& root, const char* key, float* dst) {
  const JsonValue* v = get(root, key);
  if (v && !v->is_null()) *dst = static_cast<float>(v->as_number(*dst));
}

static void set_int(const JsonValue& root, const char* key, int* dst) {
  const JsonValue* v = get(root, key);
  if (v && !v->is_null()) *dst = static_cast<int>(v->as_number(*dst));
}

static void set_string_vector(const JsonValue& root, const char* key, std::vector<std::string>* dst) {
  const JsonValue* v = get(root, key);
  if (!v || !v->is_array()) return;
  dst->clear();
  for (size_t i = 0; i < v->array.size(); ++i) dst->push_back(v->array[i].as_string());
}

static void set_float_vector(const JsonValue& root, const char* key, std::vector<float>* dst) {
  const JsonValue* v = get(root, key);
  if (!v || !v->is_array()) return;
  dst->clear();
  for (size_t i = 0; i < v->array.size(); ++i) dst->push_back(static_cast<float>(v->array[i].as_number()));
}

static void set_float_vector_fixed(const JsonValue& root, const char* key, float* dst, size_t count) {
  const JsonValue* v = get(root, key);
  if (!v || !v->is_array()) return;
  for (size_t i = 0; i < count && i < v->array.size(); ++i) dst[i] = static_cast<float>(v->array[i].as_number(dst[i]));
}

static void set_image_size(const JsonValue& root, AdapterConfig* cfg) {
  const JsonValue* v = get(root, "image_size");
  if (!v || !v->is_array() || v->array.size() < 2) return;
  cfg->image_height = static_cast<int>(v->array[0].as_number(cfg->image_height));
  cfg->image_width = static_cast<int>(v->array[1].as_number(cfg->image_width));
}

static void set_image_preprocess_size(const JsonValue& root, AdapterConfig* cfg) {
  const JsonValue* v = get(root, "image_preprocess_size");
  if (!v || !v->is_array() || v->array.size() < 2) return;
  cfg->image_preprocess_height = static_cast<int>(v->array[0].as_number(cfg->image_preprocess_height));
  cfg->image_preprocess_width = static_cast<int>(v->array[1].as_number(cfg->image_preprocess_width));
}

}  // namespace

AdapterConfig load_adapter_config(const std::string& path) {
  const std::string text = read_text_file(path);
  JsonValue root = lower_ext(path) == ".json" ? parse_json(text) : parse_simple_yaml(text);
  if (!root.is_object()) throw std::runtime_error("Adapter config must be an object: " + path);

  AdapterConfig cfg;
  set_string(root, "config_path", &cfg.config_path);
  set_string(root, "checkpoint_path", &cfg.checkpoint_path);
  set_string(root, "output_topic", &cfg.output_topic);
  set_string(root, "lidar_topic", &cfg.lidar_topic);
  set_string_vector(root, "camera_topics", &cfg.camera_topics);
  set_string_vector(root, "camera_order", &cfg.camera_order);
  set_string(root, "node_name", &cfg.node_name);
  set_string(root, "module_name", &cfg.module_name);
  set_string(root, "device", &cfg.device);
  set_string(root, "model", &cfg.model);
  set_string(root, "precision", &cfg.precision);
  set_string(root, "profile", &cfg.profile);
  set_string(root, "cuda_model_root", &cfg.cuda_model_root);
  set_string(root, "cuda_build_dir", &cfg.cuda_build_dir);
  set_string(root, "camera_plan", &cfg.camera_plan);
  set_string(root, "vtransform_plan", &cfg.vtransform_plan);
  set_string(root, "lidar_onnx", &cfg.lidar_onnx);
  set_string(root, "fuser_plan", &cfg.fuser_plan);
  set_string(root, "head_plan", &cfg.head_plan);
  set_bool(root, "enable_object_detection", &cfg.enable_object_detection);
  set_bool(root, "enable_map_segmentation", &cfg.enable_map_segmentation);
  set_string(root, "map_plan", &cfg.map_plan);
  set_string(root, "map_input_binding", &cfg.map_input_binding);
  set_string(root, "map_output_binding", &cfg.map_output_binding);
  set_float(root, "map_score_threshold", &cfg.map_score_threshold);
  set_string_vector(root, "map_classes", &cfg.map_classes);
  set_bool(root, "print_model_info", &cfg.print_model_info);
  set_float(root, "score_threshold", &cfg.score_threshold);
  set_float(root, "sync_tolerance_ms", &cfg.sync_tolerance_ms);
  set_float_vector(root, "camera_time_offsets_ms", &cfg.camera_time_offsets_ms);
  set_float(root, "lidar_time_offset_ms", &cfg.lidar_time_offset_ms);
  set_bool(root, "sync_debug", &cfg.sync_debug);
  set_int(root, "sync_debug_limit", &cfg.sync_debug_limit);
  set_int(root, "sync_queue_size", &cfg.sync_queue_size);
  set_image_size(root, &cfg);
  set_image_preprocess_size(root, &cfg);
  set_float(root, "image_resize", &cfg.image_resize);
  set_float_vector_fixed(root, "image_mean", cfg.image_mean, 3);
  set_float_vector_fixed(root, "image_std", cfg.image_std, 3);
  set_bool(root, "undistort_images", &cfg.undistort_images);
  set_int(root, "point_dim", &cfg.point_dim);
  set_float_vector_fixed(root, "point_cloud_range", cfg.point_cloud_range, 6);
  set_string(root, "calibration_file", &cfg.calibration_file);
  set_string(root, "calibration_extrinsic_direction", &cfg.calibration_extrinsic_direction);
  set_bool(root, "allow_identity_calibration", &cfg.allow_identity_calibration);
  set_string(root, "input_message_type", &cfg.input_message_type);
  set_string(root, "output_message_type", &cfg.output_message_type);
  set_bool(root, "publish_empty_frame", &cfg.publish_empty_frame);
  set_string(root, "bag_path", &cfg.bag_path);
  set_int(root, "max_frames", &cfg.max_frames);
  set_string(root, "output_file", &cfg.output_file);

  const std::string base = dirname(path);
  cfg.config_path = resolve_path(cfg.config_path, base);
  cfg.checkpoint_path = resolve_path(cfg.checkpoint_path, base);
  cfg.cuda_model_root = resolve_path(cfg.cuda_model_root, base);
  cfg.cuda_build_dir = resolve_path(cfg.cuda_build_dir, base);
  cfg.camera_plan = resolve_path(cfg.camera_plan, base);
  cfg.vtransform_plan = resolve_path(cfg.vtransform_plan, base);
  cfg.lidar_onnx = resolve_path(cfg.lidar_onnx, base);
  cfg.fuser_plan = resolve_path(cfg.fuser_plan, base);
  cfg.head_plan = resolve_path(cfg.head_plan, base);
  cfg.map_plan = resolve_path(cfg.map_plan, base);
  cfg.calibration_file = resolve_path(cfg.calibration_file, base);
  cfg.bag_path = resolve_path(cfg.bag_path, base);
  cfg.output_file = resolve_path(cfg.output_file, base);
  return cfg;
}

}  // namespace rscl_adapter
