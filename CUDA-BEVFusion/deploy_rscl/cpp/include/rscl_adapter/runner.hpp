#pragma once

#include <memory>

#include "rscl_adapter/config.hpp"
#include "rscl_adapter/types.hpp"

namespace rscl_adapter {

class BevFusionRunner {
 public:
  explicit BevFusionRunner(const AdapterConfig& cfg);
  ~BevFusionRunner();

  InferenceOutput infer(const ModelInput& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rscl_adapter
