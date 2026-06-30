#pragma once

#include <functional>
#include <memory>
#include <string>

#include "rscl_adapter/bag_reader.hpp"
#include "rscl_adapter/config.hpp"

namespace rscl_adapter {

class OnlinePublisher {
 public:
  virtual ~OnlinePublisher() = default;
  virtual void publish(const std::string& payload) = 0;
};

class OnlineNode {
 public:
  typedef std::function<void(const BagMessage&)> MessageCallback;

  virtual ~OnlineNode() = default;
  virtual bool is_valid() const = 0;
  virtual void subscribe(const std::string& topic, const std::string& message_type, MessageCallback callback) = 0;
  virtual std::unique_ptr<OnlinePublisher> create_publisher(const std::string& topic,
                                                            const std::string& message_type) = 0;
  virtual void spin() = 0;
};

std::unique_ptr<OnlineNode> create_rscl_online_node(const AdapterConfig& cfg);

}  // namespace rscl_adapter
