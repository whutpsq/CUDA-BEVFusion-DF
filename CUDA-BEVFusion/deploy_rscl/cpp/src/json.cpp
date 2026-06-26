#include "rscl_adapter/json.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rscl_adapter {
namespace {

static std::string trim(const std::string& text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return text.substr(begin, end - begin);
}

static std::string unquote(const std::string& text) {
  if (text.size() >= 2 && ((text.front() == '"' && text.back() == '"') || (text.front() == '\'' && text.back() == '\''))) {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) throw std::runtime_error("Unexpected trailing JSON data");
    return value;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }

  bool consume(char c) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    skip_ws();
    if (pos_ >= text_.size()) throw std::runtime_error("Unexpected end of JSON");
    const char c = text_[pos_];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::Type::String;
      v.string = parse_string();
      return v;
    }
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    consume('{');
    if (consume('}')) return v;
    while (true) {
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != '"') throw std::runtime_error("Expected JSON object key");
      std::string key = parse_string();
      if (!consume(':')) throw std::runtime_error("Expected ':' after JSON object key");
      v.object[key] = parse_value();
      if (consume('}')) break;
      if (!consume(',')) throw std::runtime_error("Expected ',' in JSON object");
    }
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    consume('[');
    if (consume(']')) return v;
    while (true) {
      v.array.push_back(parse_value());
      if (consume(']')) break;
      if (!consume(',')) throw std::runtime_error("Expected ',' in JSON array");
    }
    return v;
  }

  std::string parse_string() {
    if (!consume('"')) throw std::runtime_error("Expected JSON string");
    std::string out;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= text_.size()) throw std::runtime_error("Invalid JSON escape");
        char e = text_[pos_++];
        switch (e) {
          case '"':
          case '\\':
          case '/':
            out.push_back(e);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            throw std::runtime_error("Unsupported JSON escape");
        }
      } else {
        out.push_back(c);
      }
    }
    throw std::runtime_error("Unterminated JSON string");
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      v.boolean = true;
      return v;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      v.boolean = false;
      return v;
    }
    throw std::runtime_error("Invalid JSON boolean");
  }

  JsonValue parse_null() {
    if (text_.compare(pos_, 4, "null") != 0) throw std::runtime_error("Invalid JSON null");
    pos_ += 4;
    JsonValue v;
    v.type = JsonValue::Type::Null;
    return v;
  }

  JsonValue parse_number() {
    size_t begin = pos_;
    if (text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    JsonValue v;
    v.type = JsonValue::Type::Number;
    v.number = std::stod(text_.substr(begin, pos_ - begin));
    return v;
  }

  const std::string& text_;
  size_t pos_ = 0;
};

static JsonValue scalar_from_yaml(const std::string& value) {
  const std::string t = trim(value);
  JsonValue v;
  if (t.empty() || t == "null" || t == "~") return v;
  if (t == "true" || t == "True") {
    v.type = JsonValue::Type::Bool;
    v.boolean = true;
    return v;
  }
  if (t == "false" || t == "False") {
    v.type = JsonValue::Type::Bool;
    v.boolean = false;
    return v;
  }
  if (t.size() >= 2 && t.front() == '[' && t.back() == ']') {
    std::string json = t;
    return parse_json(json);
  }
  char* end = nullptr;
  double num = std::strtod(t.c_str(), &end);
  if (end && *end == '\0' && end != t.c_str()) {
    v.type = JsonValue::Type::Number;
    v.number = num;
    return v;
  }
  v.type = JsonValue::Type::String;
  v.string = unquote(t);
  return v;
}

static int indentation(const std::string& line) {
  int n = 0;
  while (n < static_cast<int>(line.size()) && line[n] == ' ') ++n;
  return n;
}

static std::string strip_comment(const std::string& line) {
  bool in_quote = false;
  char quote = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if ((c == '"' || c == '\'') && (i == 0 || line[i - 1] != '\\')) {
      if (!in_quote) {
        in_quote = true;
        quote = c;
      } else if (quote == c) {
        in_quote = false;
      }
    }
    if (!in_quote && c == '#') return line.substr(0, i);
  }
  return line;
}

}  // namespace

const JsonValue* JsonValue::get(const std::string& key) const {
  if (!is_object()) return nullptr;
  std::map<std::string, JsonValue>::const_iterator it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

std::string JsonValue::as_string(const std::string& fallback) const {
  if (is_string()) return string;
  if (is_number()) {
    std::ostringstream ss;
    ss << number;
    return ss.str();
  }
  if (is_bool()) return boolean ? "true" : "false";
  return fallback;
}

double JsonValue::as_number(double fallback) const {
  if (is_number()) return number;
  if (is_string()) {
    char* end = nullptr;
    double value = std::strtod(string.c_str(), &end);
    return end && *end == '\0' ? value : fallback;
  }
  return fallback;
}

bool JsonValue::as_bool(bool fallback) const {
  if (is_bool()) return boolean;
  if (is_string()) {
    if (string == "true" || string == "True") return true;
    if (string == "false" || string == "False") return false;
  }
  return fallback;
}

JsonValue parse_json(const std::string& text) { return JsonParser(text).parse(); }

JsonValue parse_simple_yaml(const std::string& text) {
  JsonValue root;
  root.type = JsonValue::Type::Object;

  std::istringstream in(text);
  std::string line;
  std::string current_key;
  int current_list_indent = -1;
  while (std::getline(in, line)) {
    line = strip_comment(line);
    if (trim(line).empty()) continue;
    const int ind = indentation(line);
    std::string content = trim(line);

    if (!current_key.empty() && ind > current_list_indent && content.size() >= 2 && content[0] == '-' &&
        std::isspace(static_cast<unsigned char>(content[1]))) {
      root.object[current_key].array.push_back(scalar_from_yaml(content.substr(2)));
      continue;
    }

    current_key.clear();
    current_list_indent = -1;
    size_t colon = content.find(':');
    if (colon == std::string::npos) continue;
    std::string key = trim(content.substr(0, colon));
    std::string value = trim(content.substr(colon + 1));
    JsonValue parsed;
    if (value.empty()) {
      parsed.type = JsonValue::Type::Array;
      current_key = key;
      current_list_indent = ind;
    } else {
      parsed = scalar_from_yaml(value);
    }
    root.object[key] = parsed;
  }
  return root;
}

std::string read_text_file(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open file: " + path);
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

}  // namespace rscl_adapter
