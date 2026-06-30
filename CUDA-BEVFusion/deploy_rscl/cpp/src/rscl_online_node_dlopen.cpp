#include "rscl_adapter/online_node.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace rscl_adapter {
namespace {

typedef OnlineNode* (*CreateOnlineNodeFn)(const AdapterConfig*);

static std::string executable_dir() {
  char path[PATH_MAX];
  const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len <= 0) return std::string();
  path[len] = '\0';
  std::string text(path);
  const size_t slash = text.find_last_of('/');
  if (slash == std::string::npos) return std::string();
  return text.substr(0, slash);
}

static std::vector<std::string> backend_candidates() {
  std::vector<std::string> paths;
  const char* env_path = std::getenv("RSCL_ONLINE_BACKEND_LIB");
  if (env_path && env_path[0] != '\0') paths.push_back(env_path);
  const std::string exe_dir = executable_dir();
  if (!exe_dir.empty()) paths.push_back(exe_dir + "/librscl_online_backend_ad.so");
  paths.push_back("./build/librscl_online_backend_ad.so");
  paths.push_back("./build_rscl/librscl_online_backend_ad.so");
  paths.push_back("./librscl_online_backend_ad.so");
  paths.push_back("librscl_online_backend_ad.so");
  return paths;
}

static void* open_backend_library(std::string* loaded_path) {
  int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND)
  // RSCL depends on protobuf 3.14 while BEVFusion may already use system protobuf.
  // Prefer symbols inside the RSCL dependency subtree to avoid cross-version binding.
  const char* deepbind_env = std::getenv("RSCL_ONLINE_BACKEND_DEEPBIND");
  if (deepbind_env == nullptr || std::string(deepbind_env) != "0") {
    flags |= RTLD_DEEPBIND;
  }
#endif

  std::string errors;
  const std::vector<std::string> paths = backend_candidates();
  for (size_t i = 0; i < paths.size(); ++i) {
    dlerror();
    void* handle = dlopen(paths[i].c_str(), flags);
    if (handle) {
      if (loaded_path) *loaded_path = paths[i];
      return handle;
    }
    const char* err = dlerror();
    errors += "\n  ";
    errors += paths[i];
    errors += ": ";
    errors += err ? err : "unknown dlopen error";
  }
  throw std::runtime_error("Failed to load RSCL online backend library. Set RSCL_ONLINE_BACKEND_LIB if needed." +
                           errors);
}

class DlopenOnlineNode : public OnlineNode {
 public:
  explicit DlopenOnlineNode(const AdapterConfig& cfg) {
    handle_ = open_backend_library(&loaded_path_);
    std::cerr << "online_node stage=loaded_rscl_backend path=" << loaded_path_ << std::endl;

    dlerror();
    void* sym = dlsym(handle_, "rscl_adapter_create_online_node");
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
      throw std::runtime_error("RSCL online backend missing symbol rscl_adapter_create_online_node: " +
                               std::string(err ? err : "null symbol"));
    }

    CreateOnlineNodeFn create = reinterpret_cast<CreateOnlineNodeFn>(sym);
    node_.reset(create(&cfg));
    if (!node_) throw std::runtime_error("RSCL online backend returned null node");
  }

  ~DlopenOnlineNode() override {
    node_.reset();
    // Keep the SDK loaded until process exit. Some middleware stacks are not
    // safe to unload after static initializers, protobuf registries, or worker
    // threads have been created.
    handle_ = nullptr;
  }

  bool is_valid() const override { return node_ && node_->is_valid(); }

  void subscribe(const std::string& topic, const std::string& message_type, MessageCallback callback) override {
    node_->subscribe(topic, message_type, callback);
  }

  std::unique_ptr<OnlinePublisher> create_publisher(const std::string& topic,
                                                    const std::string& message_type) override {
    return node_->create_publisher(topic, message_type);
  }

  void spin() override { node_->spin(); }

 private:
  void* handle_ = nullptr;
  std::string loaded_path_;
  std::unique_ptr<OnlineNode> node_;
};

}  // namespace

std::unique_ptr<OnlineNode> create_rscl_online_node(const AdapterConfig& cfg) {
  return std::unique_ptr<OnlineNode>(new DlopenOnlineNode(cfg));
}

}  // namespace rscl_adapter
