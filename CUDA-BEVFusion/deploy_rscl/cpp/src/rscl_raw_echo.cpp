#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "ad_rscl/comm/node.h"
#include "ad_rscl/runtime.h"
#include "ad_serde/wrapped_type/raw.h"

namespace {

using RawMessage = senseAD::serde::RawMessage;
using RawReceivedMsg = ReceivedMsg<RawMessage>;
using RawSubscriber = senseAD::rscl::comm::Subscriber<RawMessage>;

struct Options {
  std::string topic = "/perception/bevfusion/objects";
  std::string output_file;
  std::size_t count = 0;
};

void print_usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --topic TOPIC         RawMessage topic to subscribe\n"
            << "  --output-file PATH    Save extracted JSON as JSONL\n"
            << "  --count N             Exit after N valid JSON messages (0: keep running)\n"
            << "  -h, --help            Show this help\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--topic" && i + 1 < argc) {
      options.topic = argv[++i];
    } else if (arg == "--output-file" && i + 1 < argc) {
      options.output_file = argv[++i];
    } else if (arg == "--count" && i + 1 < argc) {
      options.count = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }
  return options;
}

// Return exactly the first complete JSON object. This deliberately ignores any
// RSCL header/trailer bytes after the JSON and handles braces inside strings.
bool extract_json_object(const char* data, std::size_t size, std::string* json) {
  if (data == nullptr || json == nullptr) return false;

  std::size_t begin = 0;
  while (begin < size && data[begin] != '{') ++begin;
  if (begin == size) return false;

  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = begin; i < size; ++i) {
    const char ch = data[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }

    if (ch == '"') {
      in_string = true;
    } else if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      if (depth == 0) return false;
      --depth;
      if (depth == 0) {
        json->assign(data + begin, i - begin + 1);
        return true;
      }
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);

    std::ofstream output;
    if (!options.output_file.empty()) {
      output.open(options.output_file.c_str(), std::ios::out | std::ios::trunc);
      if (!output) throw std::runtime_error("Cannot open output file: " + options.output_file);
    }

    // Do not pass this tool's private flags into the RSCL runtime.
    std::string process_name = "rscl_raw_echo";
    char* runtime_argv[] = {const_cast<char*>(process_name.c_str())};
    int runtime_argc = 1;
    senseAD::rscl::GetCurRuntime()->Init(runtime_argc, runtime_argv);
    if (!senseAD::rscl::GetCurRuntime()->OK()) {
      throw std::runtime_error("RSCL runtime initialization failed");
    }

    std::unique_ptr<senseAD::rscl::comm::Node> node =
        senseAD::rscl::GetCurRuntime()->CreateNode("rscl_raw_echo");
    if (!node) throw std::runtime_error("Failed to create RSCL node");

    std::size_t received = 0;
    std::mutex output_mutex;
    std::condition_variable count_reached;
    std::shared_ptr<RawSubscriber> subscriber = node->CreateSubscriber<RawMessage>(
        options.topic,
        [&options, &output, &output_mutex, &received,
         &count_reached](const std::shared_ptr<RawReceivedMsg>& msg) {
          if (!msg || !msg->IsValid()) return;

          std::string json;
          if (!extract_json_object(msg->Bytes(), msg->ByteSize(), &json)) {
            std::cerr << "raw_echo: received bytes=" << msg->ByteSize()
                      << " but no complete JSON object was found\n";
            return;
          }

          std::size_t current = 0;
          {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << json << '\n' << std::flush;
            if (output) output << json << '\n' << std::flush;
            current = ++received;
          }
          if (options.count > 0 && current >= options.count) {
            count_reached.notify_one();
          }
        });
    if (!subscriber) throw std::runtime_error("Failed to subscribe to: " + options.topic);

    std::cerr << "raw_echo: listening topic=" << options.topic << " type=RawMessage";
    if (!options.output_file.empty()) std::cerr << " output=" << options.output_file;
    std::cerr << '\n';

    if (options.count > 0) {
      // Shutdown can wait for subscriber callbacks, so it must not be called
      // from inside the callback itself.  Poll runtime state as well so Ctrl+C
      // can still terminate a count-limited invocation.
      std::unique_lock<std::mutex> lock(output_mutex);
      while (received < options.count && senseAD::rscl::GetCurRuntime()->OK()) {
        count_reached.wait_for(lock, std::chrono::milliseconds(200));
      }
      lock.unlock();
      if (senseAD::rscl::GetCurRuntime()->OK()) senseAD::rscl::GetCurRuntime()->Shutdown();
    } else {
      senseAD::rscl::GetCurRuntime()->WaitForShutdown();
    }
    subscriber.reset();
    node.reset();
    if (senseAD::rscl::GetCurRuntime()->OK()) senseAD::rscl::GetCurRuntime()->Shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rscl_raw_echo: " << error.what() << '\n';
    return 1;
  }
}
