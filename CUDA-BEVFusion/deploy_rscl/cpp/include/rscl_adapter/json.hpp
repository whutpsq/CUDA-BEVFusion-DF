#pragma once

#include <map>
#include <string>
#include <vector>

namespace rscl_adapter {

class JsonValue {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;

  bool is_null() const { return type == Type::Null; }
  bool is_bool() const { return type == Type::Bool; }
  bool is_number() const { return type == Type::Number; }
  bool is_string() const { return type == Type::String; }
  bool is_array() const { return type == Type::Array; }
  bool is_object() const { return type == Type::Object; }

  const JsonValue* get(const std::string& key) const;
  std::string as_string(const std::string& fallback = "") const;
  double as_number(double fallback = 0.0) const;
  bool as_bool(bool fallback = false) const;
};

JsonValue parse_json(const std::string& text);
JsonValue parse_simple_yaml(const std::string& text);
std::string read_text_file(const std::string& path);

}  // namespace rscl_adapter
