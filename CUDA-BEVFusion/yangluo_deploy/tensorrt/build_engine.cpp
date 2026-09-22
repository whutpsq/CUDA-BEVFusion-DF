#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <dlfcn.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kINFO) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <typename T>
struct TrtDelete {
  void operator()(T* value) const { delete value; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

std::vector<char> read_file(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Cannot open file: " + path);
  const std::streamsize size = input.tellg();
  if (size <= 0) throw std::runtime_error("File is empty: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<size_t>(size));
  if (!input.read(bytes.data(), size)) throw std::runtime_error("Cannot read file: " + path);
  return bytes;
}

void write_file(const std::string& path, const void* data, size_t size) {
  std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!output || !output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("Cannot write engine: " + path);
  }
}

void load_plugin(const std::string& path) {
  if (path.empty()) return;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) throw std::runtime_error("Cannot load plugin " + path + ": " + dlerror());
}

int load_test(Logger& logger, const std::string& engine_path, const std::string& plugin_path) {
  load_plugin(plugin_path);
  initLibNvInferPlugins(&logger, "");
  const auto bytes = read_file(engine_path);
  TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (!runtime) throw std::runtime_error("createInferRuntime failed");
  TrtPtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!engine) throw std::runtime_error("TensorRT failed to deserialize: " + engine_path);
  std::cout << "ENGINE_LOAD_OK path=" << engine_path << " io_tensors=" << engine->getNbIOTensors() << '\n';
  return 0;
}

int build(Logger& logger, const std::string& onnx_path, const std::string& engine_path,
          const std::string& plugin_path, size_t workspace_mib) {
  load_plugin(plugin_path);
  initLibNvInferPlugins(&logger, "");
  TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) throw std::runtime_error("createInferBuilder failed");
  const uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !parser || !config) throw std::runtime_error("TensorRT builder object creation failed");
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
    for (int i = 0; i < parser->getNbErrors(); ++i) std::cerr << parser->getError(i)->desc() << '\n';
    throw std::runtime_error("ONNX parse failed: " + onnx_path);
  }
  for (int i = 0; i < network->getNbInputs(); ++i) {
    const nvinfer1::Dims dims = network->getInput(i)->getDimensions();
    for (int axis = 0; axis < dims.nbDims; ++axis) {
      if (dims.d[axis] < 0) {
        throw std::runtime_error("Dynamic ONNX input shapes are unsupported by this deployment builder: " +
                                 std::string(network->getInput(i)->getName()));
      }
    }
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_mib << 20U);
  TrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("TensorRT engine build failed: " + onnx_path);
  write_file(engine_path, plan->data(), plan->size());
  std::cout << "ENGINE_BUILD_OK onnx=" << onnx_path << " plan=" << engine_path
            << " bytes=" << plan->size() << '\n';
  return load_test(logger, engine_path, plugin_path);
}

void usage(const char* argv0) {
  std::cerr << "Usage:\n  " << argv0
            << " --onnx model.onnx --engine model.plan [--plugin lib.so] [--workspace-mib 2048]\n  "
            << argv0 << " --load-engine model.plan [--plugin lib.so]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string onnx;
  std::string engine;
  std::string load_engine;
  std::string plugin;
  size_t workspace_mib = 2048;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if ((arg == "--onnx" || arg == "--engine" || arg == "--load-engine" || arg == "--plugin" ||
         arg == "--workspace-mib") && i + 1 >= argc) {
      usage(argv[0]);
      return 2;
    }
    if (arg == "--onnx") onnx = argv[++i];
    else if (arg == "--engine") engine = argv[++i];
    else if (arg == "--load-engine") load_engine = argv[++i];
    else if (arg == "--plugin") plugin = argv[++i];
    else if (arg == "--workspace-mib") workspace_mib = static_cast<size_t>(std::stoul(argv[++i]));
    else {
      usage(argv[0]);
      return 2;
    }
  }
  try {
    Logger logger;
    if (!load_engine.empty() && onnx.empty() && engine.empty()) return load_test(logger, load_engine, plugin);
    if (!onnx.empty() && !engine.empty() && load_engine.empty()) return build(logger, onnx, engine, plugin, workspace_mib);
    usage(argv[0]);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

